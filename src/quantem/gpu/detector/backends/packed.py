"""Detector dispatch over the existing lossless-packed resident sources."""

import numpy as np


def is_packed_source(data: object) -> bool:
    """Recognize package-owned sources before any dense-array conversion."""
    from quantem.gpu.io.backends.cuda.packed import CudaCompactH5ResidentSource
    from quantem.gpu.io.backends.mps.packed import MPSCompactV3Resident

    return isinstance(data, (CudaCompactH5ResidentSource, MPSCompactV3Resident))


class PackedDetectorCompute:
    """Adapt resident detector operations without expanding the logical array.

    Only small requested diffraction patterns and scan maps are read back.
    Missing reductions raise explicitly; this adapter never creates a dense
    four-dimensional tensor to satisfy an unsupported operation.
    """

    capabilities: tuple[str, ...] = ()

    def __init__(self, source) -> None:
        self.source = source
        self.index = getattr(source, "index", None) or source.metadata
        self.scan_shape = tuple(self.index.shape[:2])
        self.det_shape = tuple(self.index.shape[2:])
        self.n_frames = int(np.prod(self.scan_shape))
        self.device = "mps" if hasattr(source, "index") else "cuda"

    def _unsupported(self, operation: str) -> NotImplementedError:
        return NotImplementedError(
            f"{operation} is not available for this {self.device} lossless-packed "
            "source. Load the original source with representation='dense' for "
            "this operation, or use a supported resident operation."
        )

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Scan index {index} is outside {self.n_frames} frames.")
        row, column = divmod(index, self.scan_shape[1])
        return self.source.extract_diffraction(row, column).reshape(self.det_shape)

    def masked_sum_exact(self, mask) -> np.ndarray:
        values = np.asarray(mask)
        if values.shape != self.det_shape or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"Lossless-packed detector masks must be binary with shape "
                f"{self.det_shape}. Weighted masks require a supported weighted reducer."
            )
        self.source.update_virtual_detector(values.astype(np.uint8))
        return self.source.virtual_detector_values().reshape(self.scan_shape)

    def masked_sum(self, mask) -> np.ndarray:
        return self.masked_sum_exact(mask).astype(np.float32)

    def mean_dp(self) -> np.ndarray:
        mean = getattr(self.source, "mean_diffraction_pattern", None)
        if mean is None:
            raise self._unsupported("Mean diffraction")
        return mean().mean.reshape(self.det_shape)

    def reduce_frames(self, indices, reduce="mean") -> np.ndarray:
        selected = list(indices)
        if len(selected) == 1 and reduce in {"mean", "sum", "max"}:
            return self.frame(selected[0]).astype(np.float32)
        raise self._unsupported("Selected scan reduction")

    def center_of_mass(self, mask=None) -> tuple[np.ndarray, np.ndarray]:
        prepared = getattr(self.source, "prepared_center_of_mass", None)
        if mask is None and prepared is not None:
            row, column = prepared()
            return column, row
        moments = getattr(self.source, "prepared_dpc_moment_values", None)
        if mask is not None or moments is None:
            raise self._unsupported("Center of mass with the requested mask")
        values = moments()
        if values is None:
            raise self._unsupported("Center of mass without prepared moments")
        # Integer moments remain exact. Convert only the requested small fields,
        # using the same ratio and zero-intensity convention as the dense reference.
        denominator = np.maximum(values.total.astype(np.float64), 1e-10)
        row = (values.detector_row_moment / denominator).astype(np.float32)
        column = (values.detector_column_moment / denominator).astype(np.float32)
        return column, row
