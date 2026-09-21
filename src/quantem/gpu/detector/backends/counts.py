"""Exact detector operations over new count-ANS and word-packed residents."""

import numpy as np


def is_count_source(data):
    """Recognize package-owned count sources without array coercion or GPU work."""
    from quantem.gpu.io.backends.cuda._ans import (
        CudaANSResidentCounts,
        CudaPackedResidentCounts,
    )
    from quantem.gpu.io.backends.mps._ans import (
        MPSANSResidentCounts,
        MPSPackedResidentCounts,
    )
    from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts

    return isinstance(
        data,
        (
            CudaANSResidentCounts,
            CudaPackedResidentCounts,
            MPSANSResidentCounts,
            MPSPackedResidentCounts,
            MPSStreamedCounts,
        ),
    )


def _copy_output(output):
    """Read only an explicitly requested product and release its private owner."""
    read = getattr(output, "to_numpy", None)
    if read is not None:
        try:
            return read()
        finally:
            output.release()
    return output.get()  # Caller-owned CuPy output; no source buffer is released.


def _native_output(output):
    """Transfer private Metal ownership into a tensor, retaining device storage."""
    from quantem.gpu.io._read import _torch_value

    return _torch_value(output) if hasattr(output, "to_torch") else output


class CountDetectorCompute:
    """Keep counts on the accelerator; return only requested DP/maps to callers.

    Missing science is explicit. No dense expansion, CPU scientific fallback,
    sidecar, proxy, or display-cadence qualification is implied by this adapter.
    """

    capabilities = ()

    def __init__(self, source):
        self.source = source
        self.scan_shape = tuple(source.shape[:2])
        self.det_shape = tuple(source.shape[2:])
        self.n_frames = self.scan_shape[0] * self.scan_shape[1]

    def frame(self, index):
        if not 0 <= index < self.n_frames:
            raise IndexError("Scan index is outside the resident counts.")
        row, column = divmod(index, self.scan_shape[1])
        return _copy_output(self.source.extract_diffraction_device(row, column))

    def frame_native(self, index, *, out=None, wait=True):
        """Decode one pattern into independently owned device storage."""
        row, column = divmod(index, self.scan_shape[1])
        result = _native_output(self.source.extract_diffraction_device(row, column))
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
        mask = np.asarray(mask)
        if mask.shape != self.det_shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                "Exact detector masks must be binary and match the detector shape."
            )
        return _copy_output(self.source.detector_sum_device(mask.astype(np.uint8)))

    def masked_sums_exact(self, masks):
        """Return exact uint64 images for several binary masks in one decode pass.

        MPS ANS sources use their bounded multi-mask Metal reduction. Other
        resident backends retain exact behavior by issuing one mask request at
        a time when they do not provide that optional primitive.
        """
        values = np.asarray(masks)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3 or values.shape[1:] != self.det_shape:
            raise ValueError(
                "Exact detector masks must have shape "
                f"(mask, {self.det_shape[0]}, {self.det_shape[1]})."
            )
        if len(values) < 1 or not np.all((values == 0) | (values == 1)):
            raise ValueError("Exact detector masks must be binary.")
        batch = getattr(self.source, "detector_sums_device", None)
        if batch is not None:
            return _copy_output(batch(values.astype(np.uint8, copy=False)))
        return np.stack([self.masked_sum_exact(mask) for mask in values], axis=0)

    def masked_sum(self, mask):
        return self.masked_sum_exact(mask).astype(np.float32)

    def masked_sums(self, masks):
        return self.masked_sums_exact(masks).astype(np.float32)

    def _unsupported(self, operation):
        return NotImplementedError(
            f"{operation} is not qualified for this new resident profile yet; "
            "no dense expansion or CPU fallback was performed."
        )

    def mean_dp(self):
        operation = getattr(self.source, "mean_dp_device", None)
        if operation is None:
            raise self._unsupported("Mean diffraction")
        # ``mean_dp_device`` returns a privately owned device buffer, exactly like
        # ``detector_sum_device`` and ``extract_diffraction_device`` above. Read it
        # through ``_copy_output`` so the owner is released and the caller receives
        # a NumPy mean pattern instead of the unreleased Metal owner object.
        return _copy_output(operation())

    def mean_dp_native(self):
        """Return the reduced pattern without a host copy."""
        operation = getattr(self.source, "mean_dp_device", None)
        if operation is None:
            raise self._unsupported("Mean diffraction")
        return _native_output(operation())

    def center_of_mass(self, mask=None):
        raise self._unsupported("Center of mass")

    def reduce_frames(self, indices, reduce="mean"):
        selected = list(indices)
        if len(selected) == 1 and reduce in {"sum", "mean", "max"}:
            return self.frame(selected[0]).astype(np.float32)
        raise self._unsupported("Selected-frame reduction")
