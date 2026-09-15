"""Joint detector queries over independently resident MPS count-ANS sources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


class MPSANSSeriesCompute:
    """Borrow equally shaped MPS ANS sources and query them concurrently.

    Each acquisition keeps its authenticated ANS payload resident in its own
    source owner. Queries use one worker per source, allowing Metal command
    queues to overlap while keeping the returned products in a small host
    array. No dense 4D stack is materialized and source ownership remains with
    the caller.
    """

    capabilities = ()

    def __init__(self, acquisitions):
        from quantem.gpu.io.backends.mps._ans import (
            MPSANSResidentCounts,
            MPSPackedResidentCounts,
        )

        if not acquisitions:
            raise ValueError("Select at least one complete ANS acquisition.")
        self.owners = tuple(acquisitions)
        sources = []
        common_shape = None
        dtypes = []
        for loaded in self.owners:
            source = (
                loaded.data
                if hasattr(loaded, "_fields") and "data" in loaded._fields
                else loaded
            )
            if not isinstance(source, (MPSANSResidentCounts, MPSPackedResidentCounts)):
                raise TypeError(
                    "MPS ANS series requires sources returned by io.load(..., "
                    "backend='mps', representation='encoded')."
                )
            if source.is_released:
                raise ValueError("A selected ANS source was released; load it again.")
            shape = tuple(int(value) for value in source.shape)
            if len(shape) != 4 or any(value < 1 for value in shape):
                raise ValueError(f"Each ANS acquisition must have a complete 4D shape; got {shape}.")
            if common_shape is not None and shape != common_shape:
                raise ValueError(
                    "Linked ANS acquisitions must have the same shape; got "
                    f"{shape} and {common_shape}."
                )
            common_shape = shape
            sources.append(source)
            dtypes.append(np.dtype(source.dtype))
        self.sources = tuple(sources)
        self.series_shape = (len(self.sources),)
        self.scan_shape = common_shape[:2]
        self.det_shape = common_shape[2:]
        self.n_frames = int(np.prod(self.scan_shape))
        self.frame_dtype = np.result_type(*dtypes)
        self.valid_pixels = np.ones((len(self.sources), *self.det_shape), dtype=bool)
        self.backend_metadata = {
            "backend": "mps",
            "device": "mps",
            "query_abi": "mps-ans-series-v1",
            "frame_dtype": self.frame_dtype.name,
            "series_shape": self.series_shape,
            "acquisitions": len(self.sources),
        }
        self.last = {}

    def _parallel(self, operation):
        """Run one source operation per worker and preserve acquisition order."""
        workers = min(len(self.sources), 8)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mps-ans-query") as pool:
            futures = [pool.submit(operation, source) for source in self.sources]
            return [future.result() for future in futures]

    def frame(self, index):
        """Return one exact native diffraction pattern from every acquisition."""
        index = int(index)
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Scan index {index} is outside {self.n_frames} frames.")
        row, column = divmod(index, self.scan_shape[1])
        outputs = self._parallel(
            lambda source: source.extract_diffraction_device(row, column)
        )
        try:
            return np.stack([output.to_numpy() for output in outputs], axis=0)
        finally:
            for output in outputs:
                output.release()

    def masked_sums_exact(self, masks):
        """Return exact ``uint64`` products as ``(mask, acquisition, scan)``."""
        values = np.asarray(masks)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3 or values.shape[1:] != self.det_shape:
            raise ValueError(
                "masks must have shape "
                f"(mask, {self.det_shape[0]}, {self.det_shape[1]})."
            )
        if len(values) < 1 or not np.all((values == 0) | (values == 1)):
            raise ValueError("masks must contain only zero or one.")
        values = values.astype(np.uint8, copy=False)
        outputs = self._parallel(lambda source: source.detector_sums_device(values))
        try:
            per_source = [output.to_numpy() for output in outputs]
            return np.stack(per_source, axis=1).astype(np.uint64, copy=False)
        finally:
            for output in outputs:
                output.release()

    def masked_sum_exact(self, mask):
        """Return one exact ``uint64`` product for every acquisition."""
        return self.masked_sums_exact(np.asarray(mask)[None, ...])[0]

    def masked_sums(self, masks):
        """Return float32 products for several masks."""
        return self.masked_sums_exact(masks).astype(np.float32)

    def masked_sum(self, mask):
        """Return one float32 virtual-detector image per acquisition."""
        return self.masked_sum_exact(mask).astype(np.float32)

    def mean_dp(self):
        raise NotImplementedError(
            "Joint ANS series currently supports point patterns and binary "
            "detector products; select one acquisition for mean_dp."
        )

    def reduce_frames(self, indices, reduce="mean"):
        raise NotImplementedError(
            "Joint ANS scan-ROI reductions are not implemented; select one acquisition."
        )

    def finish(self):
        """Return timing metadata for synchronous MPS operations."""
        return dict(self.last)
