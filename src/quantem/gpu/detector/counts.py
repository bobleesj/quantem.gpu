"""Exact detector operations over Metal streamed count residents."""

import numpy as np

from quantem.gpu.io.read import _torch_value
from quantem.gpu.resident.queries import DetectorQueries


class CountDetectorCompute(DetectorQueries):
    """Keep counts on the accelerator; return only requested DP/maps to callers.

    Every product is an exact integer reduction of the resident streams; no
    dense expansion or CPU fallback is performed.
    """

    def __init__(self, source):
        self.source = source
        self.scan_shape = tuple(source.shape[:2])
        self.det_shape = tuple(source.shape[2:])
        self.n_frames = self.scan_shape[0] * self.scan_shape[1]
        self.valid_pixels = source.valid_pixels

    def frame(self, index):
        if not 0 <= index < self.n_frames:
            raise IndexError("Scan index is outside the resident counts.")
        row, column = divmod(index, self.scan_shape[1])
        return _copy_output(self.source.extract_diffraction_device(row, column))

    def frame_native(self, index, *, out=None, wait=True):
        """Decode one pattern into independently owned device storage."""
        row, column = divmod(index, self.scan_shape[1])
        # The Metal result moves into an independently owned MPS tensor and its buffer is released.
        result = _torch_value(self.source.extract_diffraction_device(row, column))
        if out is not None:
            if (
                out.shape != result.shape
                or out.dtype != result.dtype
                or out.device != result.device
            ):
                raise ValueError(
                    "out must match the pattern's shape, dtype and device."
                )
            out[...] = result
            return out
        return result

    def masked_sum_exact(self, mask):
        """Exact uint64 image of one binary mask; the source rejects other masks."""
        return _copy_output(self.source.detector_sum_device(mask))

    def masked_sum(self, mask):
        return self.masked_sum_exact(mask).astype(np.float32)

    def weighted_sum_exact(self, weights):
        """Exact uint64 per-scan sums of nonnegative integer pixel weights times counts."""
        return self.source.weighted_code_sums(weights)

    def detector_total(self):
        """Exact uint64 sum of every pattern, decoding each detector pixel once."""
        return _copy_output(self.source.detector_total_device())

    def mean_dp(self):
        return _copy_output(self.source.mean_dp_device())

    def mean_dp_native(self):
        """Return the reduced pattern without a host copy."""
        return _torch_value(self.source.mean_dp_device())

    def center_of_mass(self, mask=None):
        """Return count-weighted detector centres, defined exactly as on CUDA.

        The total and both moments are exact integers over the valid pixels of
        ``mask``; each coordinate is their float64 quotient rounded once to
        float32, and 0 where a pattern holds no counts. The moments weight each
        pixel by its row or column index; the total reads the tile index.
        """
        weights = np.ones(self.det_shape, dtype=np.uint32)
        if mask is not None:
            selected = np.asarray(mask, dtype=bool)
            if selected.shape != self.det_shape:
                raise ValueError(
                    f"Detector mask shape {selected.shape} does not match {self.det_shape}."
                )
            weights *= selected
        rows, cols = np.indices(self.det_shape, dtype=np.uint32)
        total = self.masked_sum_exact(weights)
        row_moment = self.weighted_sum_exact(weights * rows)
        col_moment = self.weighted_sum_exact(weights * cols)
        denominator = np.maximum(total.astype(np.float64), 1.0)
        com_row = (row_moment.astype(np.float64) / denominator).astype(np.float32)
        com_col = (col_moment.astype(np.float64) / denominator).astype(np.float32)
        return com_col, com_row

    def _selected_blocks(self, indices):
        """Yield exact decoded counts for each contiguous run of selected frames.

        Runs are decoded from the resident streams at most 4,096 frames at a
        time; each block arrives as ``(frames, pixels)`` with its multiplicity.
        Shared by the exact sum and maximum so both decode each run once.
        """
        selected = np.asarray(list(indices), dtype=np.int64)
        if not selected.size:
            raise ValueError("Select at least one scan position.")
        if selected.min() < 0 or selected.max() >= self.n_frames:
            raise IndexError("A selected scan index lies outside the resident counts.")
        frames, counts = np.unique(selected, return_counts=True)
        breaks = np.flatnonzero(np.diff(frames) != 1) + 1
        for run, weights in zip(np.split(frames, breaks), np.split(counts, breaks)):
            for low in range(0, run.size, 4096):
                first = int(run[low])
                stop = int(run[min(run.size, low + 4096) - 1]) + 1
                block = _copy_output(
                    self.source.decode_scan_range_device(first, stop)
                ).reshape(stop - first, -1)
                yield block, weights[low:low + 4096]

    def reduce_frames_exact(self, indices):
        """Exact uint64 sum of the selected patterns, repeated indices included.

        Flagged detector pixels are 0, as in every other exact product and on CUDA.
        """
        total = np.zeros(np.prod(self.det_shape), np.uint64)
        for block, weights in self._selected_blocks(indices):
            total += block.sum(axis=0, dtype=np.uint64)
            for row in np.flatnonzero(weights > 1):
                total += block[row].astype(np.uint64) * np.uint64(weights[row] - 1)
        return total.reshape(self.det_shape) * self.valid_pixels

    def reduce_frames_max(self, indices):
        """Exact integer maximum of the selected patterns; flagged detector pixels are 0."""
        result = None
        for block, _ in self._selected_blocks(indices):
            value = block.max(axis=0)
            result = value if result is None else np.maximum(result, value)
        return result.reshape(self.det_shape) * self.valid_pixels

    def reduce_frames(self, indices, reduce="mean"):
        """Exact uint64 ``sum`` and ``max`` of the selected patterns, or their float32 ``mean``."""
        selected = list(indices)
        if reduce not in {"sum", "mean", "max"}:
            raise ValueError("Use reduce='mean', 'sum' or 'max'.")
        if reduce == "max":
            return self.reduce_frames_max(selected).astype(np.uint64)
        total = self.reduce_frames_exact(selected)
        return (total / len(selected)).astype(np.float32) if reduce == "mean" else total


class CountSeriesCompute(DetectorQueries):
    """Answer each detector query on every Metal acquisition of a series.

    The acquisitions stay independent residents. Each query runs on each one
    in turn and the results are stacked along a leading series axis, so every
    value equals the result of preparing that acquisition alone.
    """

    def __init__(self, sources):
        self.acquisitions = [CountDetectorCompute(source) for source in sources]
        first = self.acquisitions[0]
        if any(member.source.shape != first.source.shape for member in self.acquisitions):
            raise ValueError(
                "Linked acquisitions must have the same shape; open differently "
                "shaped acquisitions separately."
            )
        self.scan_shape, self.det_shape = first.scan_shape, first.det_shape
        self.n_frames = first.n_frames
        self.series_shape = (len(self.acquisitions),)
        self.valid_pixels = np.stack([member.valid_pixels for member in self.acquisitions])

    def frame(self, index):
        return np.stack([member.frame(index) for member in self.acquisitions])

    def masked_sum_exact(self, mask):
        return np.stack([member.masked_sum_exact(mask) for member in self.acquisitions])

    def masked_sum(self, mask):
        return self.masked_sum_exact(mask).astype(np.float32)

    def weighted_sum_exact(self, weights):
        return np.stack([member.weighted_sum_exact(weights) for member in self.acquisitions])

    def detector_total(self):
        return np.stack([member.detector_total() for member in self.acquisitions])

    def mean_dp(self):
        return np.stack([member.mean_dp() for member in self.acquisitions])

    def reduce_frames(self, indices, reduce="mean"):
        return np.stack(
            [member.reduce_frames(indices, reduce) for member in self.acquisitions]
        )

    def center_of_mass(self, mask=None):
        raise NotImplementedError(
            "Joint center_of_mass is not implemented; select an acquisition for this calculation."
        )


def _copy_output(output):
    """Copy one Metal result to the host and release its private buffer.

    PyObjC never frees a Metal buffer on collection, so a result that is not
    released here would hold its device memory until the process exits.
    """
    try:
        return output.get()
    finally:
        output.release()
