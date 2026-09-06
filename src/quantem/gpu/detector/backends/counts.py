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

    return isinstance(
        data,
        (
            CudaANSResidentCounts,
            CudaPackedResidentCounts,
            MPSANSResidentCounts,
            MPSPackedResidentCounts,
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

    def masked_sum_exact(self, mask):
        mask = np.asarray(mask)
        if mask.shape != self.det_shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                "Exact detector masks must be binary and match the detector shape."
            )
        return _copy_output(self.source.detector_sum_device(mask.astype(np.uint8)))

    def masked_sum(self, mask):
        return self.masked_sum_exact(mask).astype(np.float32)

    def _unsupported(self, operation):
        return NotImplementedError(
            f"{operation} is not qualified for this new resident profile yet; "
            "no dense expansion or CPU fallback was performed."
        )

    def mean_dp(self):
        raise self._unsupported("Mean diffraction")

    def center_of_mass(self, mask=None):
        raise self._unsupported("Center of mass")

    def reduce_frames(self, indices, reduce="mean"):
        selected = list(indices)
        if len(selected) == 1 and reduce in {"sum", "mean", "max"}:
            return self.frame(selected[0]).astype(np.float32)
        raise self._unsupported("Selected-frame reduction")
