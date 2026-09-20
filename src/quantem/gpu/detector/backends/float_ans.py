"""Scientific products from exact float ANS residents, not integer counts."""

import numpy as np


def _numpy(value):
    if hasattr(value, "to_numpy"):
        try:
            return value.to_numpy()
        finally:
            value.release()
    if hasattr(value, "get"):
        return value.get()
    return value.detach().cpu().numpy()


class FloatANSDetectorCompute:
    """Expose float measurements and bounded GPU reductions to detector clients."""

    capabilities = ()

    def __init__(self, source):
        self.source = source
        self.device = f"cuda:{source.device}" if source.backend == "cuda" else "mps"
        self.scan_shape = source.shape[:2]
        self.det_shape = source.shape[2:]
        self.n_frames = int(np.prod(self.scan_shape))
        self.valid_pixels = source.valid_pixels

    def frame(self, index):
        if type(index) is not int or not 0 <= index < self.n_frames:
            raise IndexError("Choose a scan index within the acquisition.")
        return _numpy(self.source._tensor(index, index + 1)).reshape(self.det_shape)

    def masked_sum(self, mask):
        return _numpy(self.source.detector_sum_device(mask))

    def mean_dp(self):
        return _numpy(self.source.mean_dp_device())

    def reduce_frames(self, indices, reduce="mean"):
        return _numpy(self.source.reduce_frames_device(indices, reduce))

    def center_of_mass(self, mask=None):
        with self.source.device_context():
            return self._center_of_mass(mask)

    def _center_of_mass(self, mask):
        total, row, column = self.source.products_device(mask, moments=True)
        valid = (total != 0) & (abs(total) < float("inf"))
        denominator = self.source._where(valid, total, 1)
        row = self.source._where(valid, row / denominator, float("nan"))
        column = self.source._where(valid, column / denominator, float("nan"))
        if self.source.backend == "cuda":
            import cupy as cp

            row -= cp.nanmean(row)
            column -= cp.nanmean(column)
        else:
            import torch

            row -= torch.nanmean(row)
            column -= torch.nanmean(column)
        return _numpy(column), _numpy(row)

    def masked_sum_exact(self, mask):
        raise TypeError(
            "Exact integer sums do not apply to float32 measurements; use masked_sum."
        )
