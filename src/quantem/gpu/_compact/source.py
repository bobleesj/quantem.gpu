"""Exact, device-owned queries over the prepared series format."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from . import FORMAT, QUERY_ABI, implementation_id


class CompactSeries:
    """Own encoded counts, exact indexes, and serialized query workspace.

    This private source is returned by ``io.load``. The first supported format
    has a fixed native shape and model layout; it is not an arbitrary codec.
    All query results are complete and synchronized before returning. Default
    native outputs own their buffers; ``out`` borrows caller-owned storage.
    """

    shape = (66, 512, 512, 192, 192)
    dtype = np.dtype("uint16")
    ndim = 5
    scan_shape = (512, 512)
    det_shape = (192, 192)
    series_shape = (66,)
    n_frames = 512 * 512
    capabilities = ()
    storage_format = FORMAT
    query_abi = QUERY_ABI
    decoder_variant = "funnel32-pairs4"

    def __init__(self, device: int) -> None:
        self._implementation_id = implementation_id()
        self.device_id = device
        self.device = f"cuda:{device}"
        self._lock = threading.RLock()
        self.previous = None
        self.output = None
        self._previous_output = None
        self.last = {}

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "Encoded counts cannot expand implicitly. Use detector.prepare(data) "
            "for virtual images and diffraction patterns."
        )

    @property
    def nbytes(self) -> int:
        return self.resident_bytes

    @property
    def valid_pixels(self) -> np.ndarray:
        return self.valid.reshape(self.det_shape).astype(bool)

    @property
    def backend_metadata(self) -> dict:
        from importlib.metadata import version

        return {
            "backend": "cuda",
            "package_version": version("quantem.gpu"),
            "implementation_id": self._implementation_id,
            "format": self.storage_format,
            "query_abi": self.query_abi,
            "decoder": self.decoder_variant,
            "device": self.device,
            "resident_bytes": self.resident_bytes,
        }

    def _mask(self, mask) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.shape != self.det_shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                f"Use a binary detector mask of shape {self.det_shape}; "
                f"got {mask.shape}. Weighted compact detectors are unsupported."
            )
        return np.ascontiguousarray(mask, dtype=np.int8)

    def _destination(self, out, shape, dtype):
        import cupy as cp

        if out is None:
            return cp.empty(shape, dtype)
        if (
            not isinstance(out, cp.ndarray)
            or out.shape != shape
            or out.dtype != np.dtype(dtype)
            or not out.flags.c_contiguous
            or out.device.id != self.device_id
        ):
            raise ValueError(
                f"out must be a contiguous {np.dtype(dtype)}{shape} array "
                f"on {self.device}."
            )
        # Source/index storage may never be used as a query destination.
        begin, end = out.data.ptr, out.data.ptr + out.nbytes
        for owner in (*self.owners.values(), self.coarse_index):
            if begin < owner.data.ptr + owner.nbytes and owner.data.ptr < end:
                raise ValueError(
                    "out overlaps encoded source/index storage; use a separate output array."
                )
        return out

    def masked_sum_native(self, mask, *, out=None):
        """Return exact uint32[acquisition, scan_row, scan_col] counts."""
        import cupy as cp

        current = self._mask(mask)
        with self._lock, cp.cuda.Device(self.device_id):
            destination = self._destination(out, self.shape[:3], np.uint32)
            before = time.perf_counter()
            self.output = destination
            try:
                # Caller-owned results are mutable. Preserve a private baseline
                # so their reuse cannot change the next detector calculation.
                if self._previous_output is None:
                    self._previous_output = cp.empty_like(destination)
                    self.previous = None
                if self.previous is not None:
                    cp.copyto(destination, self._previous_output)
                self.update(current)
                cp.copyto(self._previous_output, destination)
                self.ready.record()
                self.ready.synchronize()
                self.last["gpu_ms"] = cp.cuda.get_elapsed_time(self.start, self.ready)
                self.last["host_ms"] = (time.perf_counter() - before) * 1000
                return destination
            except BaseException:
                self.previous = None
                raise

    def frame_native(self, index: int, *, out=None):
        """Return original uint16[acquisition, detector_row, detector_col]."""
        import cupy as cp

        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or not 0 <= index < self.n_frames
        ):
            raise IndexError(
                f"Use a flat scan position from 0 to {self.n_frames - 1}; got {index}."
            )
        with self._lock, cp.cuda.Device(self.device_id):
            destination = self._destination(out, (66, 192, 192), np.uint16)
            before = time.perf_counter()
            self.start.record()
            self.pattern_module.get_function("all_patterns")(
                (144, 66),
                (256,),
                (
                    self.payload_addresses,
                    self.offset_addresses,
                    self.event_addresses,
                    self.sparse_addresses,
                    self.decoding,
                    self.pattern_ids,
                    self.pattern_cache_map,
                    np.int32(index),
                    destination,
                ),
            )
            self.ready.record()
            self.ready.synchronize()
            self.last = {
                "gpu_ms": cp.cuda.get_elapsed_time(self.start, self.ready),
                "host_ms": (time.perf_counter() - before) * 1000,
            }
            return destination

    def update(self, mask, *, rebase: bool = False):
        """Integrate into the selected workspace with the frozen exact planner."""
        import cupy as cp

        before = time.perf_counter()
        current = np.asarray(mask).reshape(-1).astype(np.int8)
        current[self.hardware] = 0
        seed, difference, tile_sign, coarse, _cost = self.native_planner(
            current,
            None if rebase else self.previous,
        )
        chosen = np.flatnonzero(tile_sign).astype(np.int32)
        selected = np.flatnonzero(difference).astype(np.int32)
        dense = selected[self.cache_map[selected] < 0]
        sparse = selected[self.cache_map[selected] >= 0]
        self.tile_indices[: len(chosen)].set(chosen)
        self.tile_signs[: len(chosen)].set(tile_sign[chosen])
        self.sparse_indices[: len(sparse)].set(self.cache_map[sparse])
        self.sparse_coefficients[: len(sparse)].set(difference[sparse])
        dense_difference = np.zeros(36864, np.int8)
        dense_difference[dense] = difference[dense]
        coarse_chosen = np.flatnonzero(coarse).astype(np.int32)
        self.coarse_selected[: len(coarse_chosen)].set(coarse_chosen)
        self.coarse_signs[: len(coarse_chosen)].set(coarse[coarse_chosen])
        self.start.record()
        count = self.groups.build(dense_difference) if len(dense) else 0
        self.group_ready.record()
        self.tile_module.get_function("seed_tiles")(
            (16, 1056),
            (256,),
            (
                self.field_addresses,
                self.field_widths,
                self.tile_indices,
                self.tile_signs,
                np.int32(len(chosen)),
                np.int32({"zero": 0, "total": 1, "previous": 2}[seed]),
                self.total,
                self.output,
            ),
        )
        if len(coarse_chosen):
            self.coarse_module.get_function("add_coarse")(
                (16, 1056),
                (256,),
                (
                    self.coarse_addresses,
                    self.coarse_widths,
                    self.coarse_selected,
                    self.coarse_signs,
                    np.int32(len(coarse_chosen)),
                    self.output,
                ),
            )
        self.seed_ready.record()
        if count:
            groups = self.groups
            self.module.get_function("integrate_streams")(
                (count, 2, 4),
                (512,),
                (
                    self.payload_addresses,
                    self.offset_addresses,
                    self.decoding,
                    groups.selected,
                    groups.slots,
                    groups.models,
                    self.output,
                    groups.context,
                    groups.signs,
                    groups.lengths,
                    groups.model_counts,
                ),
                shared_mem=20480,
            )
        self.dense_ready.record()
        if len(sparse):
            self.sparse_module.get_function("integrate_sparse")(
                (32, 1056),
                (256,),
                (
                    self.event_addresses,
                    self.sparse_addresses,
                    self.sparse_indices,
                    self.sparse_coefficients,
                    np.int32(len(sparse)),
                    np.int32(len(self.cached)),
                    np.int32(32),
                    self.output,
                ),
            )
        self.ready.record()
        self.ready.synchronize()
        self.previous = current
        self.last = {
            "gpu_ms": cp.cuda.get_elapsed_time(self.start, self.ready),
            "grouping_ms": cp.cuda.get_elapsed_time(self.start, self.group_ready),
            "seed_ms": cp.cuda.get_elapsed_time(self.group_ready, self.seed_ready),
            "kernel_ms": cp.cuda.get_elapsed_time(self.seed_ready, self.dense_ready),
            "dense_ms": cp.cuda.get_elapsed_time(self.start, self.dense_ready),
            "sparse_ms": cp.cuda.get_elapsed_time(self.dense_ready, self.ready),
            "host_ms": (time.perf_counter() - before) * 1000,
            "dense_columns": len(dense),
            "sparse_columns": len(sparse),
            "tiles": len(chosen),
            "coarse_tiles": len(coarse_chosen),
            "groups": count,
            "seed": seed,
            "kernel": self.decoder_variant,
        }
        return self.output

    def mean_dp(self):
        raise NotImplementedError(
            "Compact full-scan mean DP is not implemented; use frame() for point patterns."
        )

    def reduce_frames(self, indices, reduce="mean"):
        raise NotImplementedError(
            "Compact multi-position DP reductions are not implemented; use frame()."
        )

    def center_of_mass(self, mask=None):
        raise NotImplementedError("Compact center of mass is not implemented.")


def build_kernels(source: CompactSeries) -> None:
    """Compile packaged kernels without checkpoint-supplied executable code."""
    import cupy as cp

    kernels = Path(__file__).with_name("kernels")
    for name in (
        "module",
        "sparse_module",
        "tile_module",
        "coarse_module",
        "pattern_module",
    ):
        setattr(
            source,
            name,
            cp.RawModule(
                code=(kernels / f"{name}.cu").read_text(),
                options=("--std=c++17",),
            ),
        )
    kernel = source.module.get_function("integrate_streams")
    kernel.max_dynamic_shared_size_bytes = 20480
    kernel.preferred_shared_memory_carveout = 64
    source.pattern_module.get_function("all_patterns")
