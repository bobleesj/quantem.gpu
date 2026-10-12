"""Detector reductions of a bounded reader, computed by the session of the acquisition it views."""

import math

import numpy as np
import torch

from quantem.gpu.resident.queries import DetectorQueries


class BoundedDetectorCompute(DetectorQueries):
    """One scan region of an acquisition, reduced by the acquisition's own detector session.

    ``data`` is a bounded reader from quantem.widget (``read(scan_region=...)``,
    ``valid``, ``_detector_region``), which this package cannot import, and
    ``native`` the ``DetectorSession`` of the acquisition or GPU tensor it
    views. Encoded counts, float intensities, paired counts and dense tensors
    each reduce on their own backend; this class maps the view's scan region
    onto the acquisition's raster and zeroes the detector pixels the reader
    zeroes. Only a single pattern is read through the reader: a reduction over
    bounded reads took 6 s for the mean of a 512 x 512 scan.
    """

    def __init__(self, data, native):
        self.data = data
        self.scan_shape = tuple(data.shape[:2])
        self.detector_shape = tuple(data.shape[2:])
        self.n_frames = math.prod(self.scan_shape)
        self.device = torch.device(data.device)
        self.valid_pixels = None if data.valid is None else data.valid.cpu().numpy()
        self._native = native

    def frame(self, index):
        row, col = divmod(int(index), self.scan_shape[1])
        return self.data.read(scan_region=(row, row + 1, col, col + 1))[0, 0].cpu().numpy()

    def mean_dp(self):
        """Mean pattern: the total divided in float64 and rounded once to float32, like every mean pattern.

        The session sums integer counts exactly and float intensities to
        float64 accuracy. A view of the whole scan takes the session's mean
        pattern; a scan region reduces its own positions.
        """
        if self.n_frames == self._native.num_frames:
            return self._valid_pattern(self._native.mean_dp())
        return self.reduce_frames(range(self.n_frames), "mean")

    def masked_sum(self, mask):
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if self.data.valid is not None:
            mask_t = mask_t & self.data.valid
        # No decoded measurement cube is constructed; the native kernels reduce on the device.
        image = self._native.masked_sum(mask_t.cpu().numpy())
        row_start, row_stop, col_start, col_stop = self.data._detector_region
        return torch.as_tensor(image[row_start:row_stop, col_start:col_stop], device=self.device)

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
        """Reduce the patterns at this view's row-major scan ``indices`` by ``mean``, ``sum`` or ``max``, as float32.

        A view may show one scan region of its acquisition, while the session
        indexes the acquisition's whole raster: ``(row, col)`` of the view is
        ``(row + row_start, col + col_start)`` of the acquisition.
        """
        if reduce not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown reduction {reduce!r}; use mean, sum or max.")
        indices = np.asarray(list(indices), dtype=np.int64)
        if not indices.size:
            raise ValueError("Select at least one scan position.")
        rows, cols = np.divmod(indices, self.scan_shape[1])
        row_start, _, col_start, _ = self.data._detector_region
        owner_indices = (rows + row_start) * self._native.scan_shape[1] + cols + col_start
        # Exact for integer counts and float32 for float intensities, decided by the session.
        return self._valid_pattern(self._native.reduce_frames(owner_indices, reduce))

    def _valid_pattern(self, pattern):
        """Zero the detector pixels the reader zeroes, so a reduction equals reducing the reads; float32."""
        valid = pattern if self.data.valid is None else pattern * self.data.valid.cpu().numpy()
        return valid.astype(np.float32)
