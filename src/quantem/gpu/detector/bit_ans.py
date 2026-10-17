"""Scientific products from typed bit-lane ANS residents."""

import numpy as np

from quantem.gpu.resident.queries import DetectorQueries


class BitANSDetectorCompute(DetectorQueries):
    """Expose typed measurements and bounded GPU reductions to detector clients."""

    def __init__(self, source):
        self.source = source
        self.device = f"cuda:{source.device}" if source.backend == "cuda" else "mps"
        self.scan_shape = source.shape[:2]
        self.detector_shape = source.shape[2:]
        self.n_frames = int(np.prod(self.scan_shape))
        self.valid_pixels = source.valid_pixels

    def frame(self, index):
        if type(index) is not int or not 0 <= index < self.n_frames:
            raise IndexError("Choose a scan index within the acquisition.")
        if self.source.backend == "mps" and self.source.dtype == np.dtype("float64"):
            values = self.frame_native(index)
            try:
                return values.get()
            finally:
                values.release()
        values = self.source._tensor(index, index + 1)
        return _host(values, self.source.backend).reshape(self.detector_shape)

    def frame_native(self, index, *, out=None, wait=True):
        """Return one corrected pattern without downloading measurements."""
        if self.source.backend == "mps" and self.source.dtype == np.dtype("float64"):
            import torch

            from quantem.gpu.io.read import _native_result, torch_value

            raw = self.source.decode_scan_range_device(index, index + 1)
            # Torch cannot express float64 on Metal. Mask the original bits
            # through an equally wide integer view, then return Metal storage.
            bits_t = torch_value(raw, reinterpret_dtype=np.int64)
            bits_t = torch.where(self.source._array(self.valid_pixels), bits_t, 0)
            if out is not None:
                raise NotImplementedError(
                    "Float64 Metal patterns own their buffer; omit out and "
                    "release the returned Metal array after use."
                )
            return _native_result(bits_t.reshape(self.detector_shape), np.float64)
        result = self.source._tensor(index, index + 1).reshape(self.detector_shape)
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

    def detector_total(self):
        """Return checked exact integer totals, with flagged pixels excluded."""
        if self.source.dtype.kind not in "iu":
            return super().detector_total()
        return _host(self.source.detector_total_device(), self.source.backend)

    def reduce_frames(self, indices, reduce="mean"):
        return _host(
            self.source.reduce_frames_device(indices, reduce), self.source.backend
        )

    def reduce_frames_exact(self, indices):
        """Sum selected integer patterns with checked 64-bit accumulation."""
        if self.source.dtype.kind not in "iu":
            raise TypeError("Exact detector sums require integer measurements.")
        return self.reduce_frames(indices, "sum")

    def reduce_frames_max(self, indices):
        """Return selected integer maxima without a floating conversion."""
        if self.source.dtype.kind not in "iu":
            raise TypeError("Exact detector maxima require integer measurements.")
        return self.reduce_frames(indices, "max")

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
            return _host(column, self.source.backend).reshape(-1), _host(
                row, self.source.backend
            ).reshape(-1)

    def masked_sum_exact(self, mask):
        if self.source.dtype.kind not in "iu":
            return super().masked_sum_exact(mask)
        return self.masked_sum(mask)

    def masked_sum_exact_native(self, mask, *, out=None):
        """Return a checked signed or unsigned 64-bit integer detector image."""
        if self.source.dtype.kind not in "iu":
            return super().masked_sum_exact(mask)
        return self.masked_sum_native(mask, out=out)


def _host(value, backend: str):
    """Copy a small reduced product to the host: CuPy arrays on CUDA, Torch tensors on MPS."""
    return value.get() if backend == "cuda" else value.detach().cpu().numpy()


def _into(out, result, product: str):
    """Fill a caller buffer, or return ``result`` itself when there is none."""
    if out is None:
        return result
    if (
        out.shape != result.shape
        or out.dtype != result.dtype
        or out.device != result.device
    ):
        raise ValueError(f"out must match the {product}'s shape, dtype and device.")
    out[...] = result
    return out
