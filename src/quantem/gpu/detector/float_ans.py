"""Scientific products from exact float ANS residents, not integer counts."""

import numpy as np

from quantem.gpu.resident.queries import DetectorQueries


class FloatANSDetectorCompute(DetectorQueries):
    """Expose float measurements and bounded GPU reductions to detector clients."""

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
        values = self.source._tensor(index, index + 1)
        return _host(values, self.source.backend).reshape(self.det_shape)

    def frame_native(self, index, *, out=None, wait=True):
        """Return one corrected pattern without downloading measurements."""
        result = self.source._tensor(index, index + 1).reshape(self.det_shape)
        return _into(out, result, "pattern")

    def masked_sum(self, mask):
        return _host(self.source.detector_sum_device(mask), self.source.backend)

    def masked_sum_native(self, mask, *, out=None, wait=True):
        """Return the virtual-detector image without a host copy."""
        return _into(out, self.source.detector_sum_device(mask), "detector image")

    def mean_dp(self):
        return _host(self.source.mean_dp_device(), self.source.backend)

    def mean_dp_native(self):
        """Return an independent mean pattern on the source device."""
        return self.source.mean_dp_device()

    def reduce_frames(self, indices, reduce="mean"):
        return _host(self.source.reduce_frames_device(indices, reduce), self.source.backend)

    def center_of_mass(self, mask=None):
        """Absolute detector ``(column, row)`` centres, flat float32, as every detector backend defines them.

        ``row = sum(row * I) / sum(I)`` over the (masked) pattern, in detector
        pixels and not mean-subtracted (``dpc.center_of_mass`` does that). A
        pattern whose total is 0, such as an empty frame, gives 0 like the count
        backends, so one empty frame cannot turn the DPC field into NaN. A
        pattern holding inf or NaN measurements has no centre and gives NaN.
        """
        with self.source.device_context():
            total, row, column = self.source.products_device(mask, moments=True)
            empty = total == 0
            denominator = self.source._where(empty, 1, total)
            row = self.source._where(empty, 0, row / denominator)
            column = self.source._where(empty, 0, column / denominator)
            return _host(column, self.source.backend).reshape(-1), _host(row, self.source.backend).reshape(-1)

    def masked_sum_exact(self, mask):
        raise TypeError(
            "Exact integer sums do not apply to float32 measurements; use masked_sum."
        )

    def masked_sum_exact_native(self, mask, *, out=None):
        """Float measurements have no exact integer sum on the device either."""
        return self.masked_sum_exact(mask)


def _host(value, backend: str):
    """Copy a small reduced product to the host: CuPy arrays on CUDA, Torch tensors on MPS."""
    return value.get() if backend == "cuda" else value.detach().cpu().numpy()


def _into(out, result, product: str):
    """Fill a caller buffer, or return ``result`` itself when there is none."""
    if out is None:
        return result
    if out.shape != result.shape or out.dtype != result.dtype or out.device != result.device:
        raise ValueError(f"out must match the {product}'s shape, dtype and device.")
    out[...] = result
    return out
