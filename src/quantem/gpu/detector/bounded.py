"""Torch detector reductions over borrowed bounded-read sources."""

import math

import numpy as np
import torch

from quantem.gpu.resident.queries import DetectorQueries


class BoundedDetectorCompute(DetectorQueries):
    """Reduce native regions without materializing a complete measurement.

    ``data`` is a bounded reader from quantem.widget (``read(scan_region=...)``,
    ``valid``, ``_detector_region``), which this package cannot import. When the
    reader views an encoded acquisition, ``native`` is that acquisition's
    ``DetectorSession``: masked sums, the mean pattern and frame reductions run
    on the encoded counts, and only the reduced products cross to the host.
    """

    def __init__(self, data, native):
        self.data = data
        self.scan_shape = tuple(data.shape[:2])
        self.det_shape = tuple(data.shape[2:])
        self.n_frames = math.prod(self.scan_shape)
        self.device = torch.device(data.device)
        self.valid_pixels = None if data.valid is None else data.valid.cpu().numpy()
        self._native = native

    def frame(self, index):
        row, col = divmod(int(index), self.scan_shape[1])
        return self.data.read(scan_region=(row, row + 1, col, col + 1))[0, 0].cpu().numpy()

    def mean_dp(self):
        """Mean pattern: the total divided in float64 and rounded once to float32, like every mean pattern."""
        if self._native is None:
            total = np.zeros(self.det_shape, np.float64)
            for _, _, _, block_t in self._blocks():
                # A float32 block sum of at most 32 integer patterns is exact; the total is not.
                total += block_t.sum((0, 1)).cpu().numpy()
        elif self.n_frames == self._native.num_frames:
            # The session sums integer counts exactly and float intensities (MAPED merges,
            # saved precision) in float64, then divides once, like every mean pattern.
            return self._valid_counts(self._native.mean_dp()).astype(np.float32)
        else:
            return self._valid_counts(
                self._native.reduce_frames(self._owner_indices(range(self.n_frames)), "mean")
            ).astype(np.float32)
        return (total / self.n_frames).astype(np.float32)

    def masked_sum(self, mask):
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if self._native is not None:
            if self.data.valid is not None:
                mask_t = mask_t & self.data.valid
            # No decoded measurement cube is constructed; the native kernels reduce on the device.
            image = self._native.masked_sum(mask_t.cpu().numpy())
            row_start, row_stop, col_start, col_stop = self.data._detector_region
            return torch.as_tensor(image[row_start:row_stop, col_start:col_stop], device=self.device)
        image_t = torch.empty(self.scan_shape, device=self.device)
        for row, col, stop, block_t in self._blocks():
            image_t[row, col:stop] = block_t[..., mask_t].sum(-1)[0]
        return image_t

    def masked_sum_native(self, mask, *, out=None):
        image_t = self.masked_sum(mask)
        if out is not None:
            out.copy_(image_t)
            return out
        return image_t

    def masked_sum_exact_native(self, mask, *, out=None):
        """The reads are float32 intensities, so there is no exact integer image on any output."""
        return self.masked_sum_exact(mask)

    def reduce_frames(self, indices, reduce="mean"):
        if reduce not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown reduction {reduce!r}; use mean, sum or max.")
        indices = list(indices)
        if not indices:
            raise ValueError("Select at least one scan position.")
        if self._native is not None:
            # Exact for integer counts and float32 for float intensities, decided by the session.
            return self._valid_counts(self._native.reduce_frames(self._owner_indices(indices), reduce)).astype(np.float32)
        if reduce == "max":
            maximum_t = None
            for value_t in self._patterns(indices):
                maximum_t = value_t if maximum_t is None else torch.maximum(maximum_t, value_t)
            return maximum_t.cpu().numpy()
        else:
            total = np.zeros(self.det_shape, np.float64)
            for value_t in self._patterns(indices):
                total += value_t.cpu().numpy()
        # Divided in float64 and rounded once, like every mean pattern.
        return (total / len(indices) if reduce == "mean" else total).astype(np.float32)

    def _owner_indices(self, indices):
        """Map this view's row-major scan indices to the acquisition's.

        A view may show one scan region of its acquisition, while the native
        session indexes the acquisition's whole raster.
        """
        rows, cols = np.divmod(np.asarray(indices, dtype=np.int64), self.scan_shape[1])
        row_start, _, col_start, _ = self.data._detector_region
        return (rows + row_start) * self._native.scan_shape[1] + cols + col_start

    def _valid_counts(self, exact):
        """Zero the detector pixels the reader zeroes, so the exact path equals reducing the reads."""
        return exact if self.data.valid is None else exact * self.data.valid.cpu().numpy()

    def _patterns(self, indices):
        """Yield the selected patterns, one bounded read each, as float32 tensors."""
        for index in indices:
            row, col = divmod(int(index), self.scan_shape[1])
            yield self.data.read(scan_region=(row, row + 1, col, col + 1))[0, 0].float()

    def _blocks(self):
        """Yield 32-scan-position float blocks so no reduction holds more than one row segment."""
        for row in range(self.scan_shape[0]):
            for col in range(0, self.scan_shape[1], 32):
                stop = min(col + 32, self.scan_shape[1])
                yield row, col, stop, self.data.read(scan_region=(row, row + 1, col, stop)).float()
