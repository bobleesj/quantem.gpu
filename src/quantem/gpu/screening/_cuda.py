"""Private exact CUDA engine for prepared screening products."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

_CUDA_DECODE_BATCH_FRAMES = 10_000
_CUDA_COMPRESSED_FRAME_BOUND = 1.02
_CUDA_FIXED_RESERVE_BYTES = 256 * 1024**2
_CUDA_WORKING_SET_FRACTION = 0.88
_MASK_GUARD_MARGIN_PX = 0.25
_MASK_GUARD_MAX_BYTES = 1 << 30
_SCREENING_MASK_NAMES = (
    "bright_field",
    "annular_bright_field",
    "annular_dark_field",
    "dark_field",
)


def _exact_cuda_working_set_bytes(memory_plan, chunk_rows: int) -> int:
    """Conservatively bound the exact CUDA screening allocator reserve.

    The decoder owns the decoded result, two uncompressed scratch buffers,
    and one or two compressed upload buffers. The fused screening kernel adds
    seven uint64 statistics per frame. A fixed reserve and twelve-percent
    envelope headroom cover metadata, kernels, allocator size classes, and
    fragmentation; the independent runtime reserve gate remains authoritative.
    """

    scan_rows, scan_columns = memory_plan.scan_shape
    detector_rows, detector_columns = memory_plan.detector_shape
    rows = max(1, min(int(chunk_rows), int(scan_rows)))
    frames = rows * int(scan_columns)
    frame_bytes = (
        int(detector_rows)
        * int(detector_columns)
        * np.dtype(memory_plan.dtype).itemsize
    )
    decoded_bytes = frames * frame_bytes
    decode_batch_frames = min(frames, _CUDA_DECODE_BATCH_FRAMES)
    decode_batch_bytes = decode_batch_frames * frame_bytes
    decode_scratch_bytes = 2 * decode_batch_bytes
    upload_slots = 2 if frames > decode_batch_frames else 1
    compressed_upload_bound = upload_slots * int(
        np.ceil(decode_batch_bytes * _CUDA_COMPRESSED_FRAME_BOUND)
    )
    fused_output_bytes = 7 * frames * np.dtype(np.uint64).itemsize
    detector_state_bytes = (
        int(detector_rows)
        * int(detector_columns)
        * (np.dtype(np.uint8).itemsize + np.dtype(np.uint64).itemsize)
    )
    return int(
        decoded_bytes
        + decode_scratch_bytes
        + compressed_upload_bound
        + fused_output_bytes
        + detector_state_bytes
        + _CUDA_FIXED_RESERVE_BYTES
    )


def _exact_cuda_memory_plan(memory_plan):
    """Tighten an automatic plan for the exact CUDA decoder topology."""

    if memory_plan.chunk_rows_source != "budget":
        return memory_plan
    limit_bytes = int(
        float(memory_plan.memory_budget_gb)
        * (1 << 30)
        * _CUDA_WORKING_SET_FRACTION
    )
    rows = int(memory_plan.chunk_rows)
    while rows > 1 and _exact_cuda_working_set_bytes(memory_plan, rows) > limit_bytes:
        rows -= 1
    if _exact_cuda_working_set_bytes(memory_plan, rows) > limit_bytes:
        raise MemoryError(
            "The exact CUDA screening working set does not fit the requested "
            f"{memory_plan.memory_budget_gb:.3g} GiB process envelope even at "
            "chunk_rows=1. Increase memory_budget_gb."
        )
    if rows == int(memory_plan.chunk_rows):
        return memory_plan
    bytes_per_row = (
        int(memory_plan.scan_shape[1])
        * int(memory_plan.detector_shape[0])
        * int(memory_plan.detector_shape[1])
        * np.dtype(memory_plan.dtype).itemsize
    )
    return replace(
        memory_plan,
        raw_target_gb=float(rows * bytes_per_row) / 1e9,
        chunk_rows=rows,
        chunk_rows_source="budget_cuda_exact",
        chunk_count=int(np.ceil(int(memory_plan.scan_shape[0]) / rows)),
        chunk_resident_gb=float(rows * bytes_per_row) / 1e9,
    )


def _band_masks(
    center: tuple[float, float],
    radius: float,
    detector_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Return canonical BF/ABF/ADF/DF masks for one fitted probe disk."""

    from quantem.gpu.detector import detector_mask

    return {
        "bright_field": detector_mask(center, 0.0, radius, detector_shape),
        "annular_bright_field": detector_mask(
            center,
            0.5 * radius,
            radius,
            detector_shape,
        ),
        "annular_dark_field": detector_mask(
            center,
            radius,
            2.0 * radius,
            detector_shape,
        ),
        "dark_field": detector_mask(
            center,
            radius,
            np.inf,
            detector_shape,
        ),
    }


def _band_bits(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Pack the four canonical masks into detector-shaped uint8 bit flags."""

    shape = np.asarray(masks[_SCREENING_MASK_NAMES[0]], dtype=bool).shape
    bits = np.zeros(shape, dtype=np.uint8)
    for bit, name in enumerate(_SCREENING_MASK_NAMES):
        mask = np.asarray(masks[name], dtype=bool)
        if mask.shape != shape:
            raise ValueError("Screening detector masks have inconsistent shapes")
        bits[mask] |= np.uint8(1 << bit)
    return bits


def _mask_guard_indices(
    center: tuple[float, float],
    radius: float,
    detector_shape: tuple[int, int],
    *,
    margin_px: float = _MASK_GUARD_MARGIN_PX,
) -> np.ndarray:
    """Return detector pixels near every provisional radial mask boundary."""

    detector_row, detector_column = np.indices(detector_shape, dtype=np.float64)
    distance = np.hypot(
        detector_row - float(center[0]),
        detector_column - float(center[1]),
    )
    guard = np.zeros(detector_shape, dtype=bool)
    for multiplier in (0.5, 1.0, 2.0):
        guard |= np.abs(distance - multiplier * float(radius)) <= float(margin_px)
    return np.flatnonzero(guard).astype(np.int32, copy=False)


def _apply_mask_guard_correction(
    exact_flat: np.ndarray,
    guard_batches: list[tuple[int, int, np.ndarray]],
    guard_indices: np.ndarray,
    provisional_bits: np.ndarray,
    authoritative_bits: np.ndarray,
) -> tuple[bool, tuple[int, int, int, int]]:
    """Correct four band sums exactly when every changed pixel was retained."""

    provisional = np.asarray(provisional_bits, dtype=np.uint8).reshape(-1)
    authoritative = np.asarray(authoritative_bits, dtype=np.uint8).reshape(-1)
    indices = np.asarray(guard_indices, dtype=np.int64).reshape(-1)
    expected_start = 0
    for start, stop, counts in guard_batches:
        if start != expected_start or stop <= start:
            raise ValueError("Mask-guard batches must cover frames in order")
        if np.asarray(counts).shape != (indices.size, stop - start):
            raise ValueError("Mask-guard batch shape does not match its frame range")
        expected_start = stop
    if expected_start != exact_flat.shape[1]:
        raise ValueError("Mask-guard batches do not cover every frame")
    covered = np.zeros(provisional.size, dtype=bool)
    covered[indices] = True
    changed = provisional != authoritative
    if np.any(changed & ~covered):
        return False, (0, 0, 0, 0)

    changed_counts: list[int] = []
    provisional_guard = provisional[indices]
    authoritative_guard = authoritative[indices]
    for bit in range(4):
        flag = np.uint8(1 << bit)
        old_member = (provisional_guard & flag) != 0
        new_member = (authoritative_guard & flag) != 0
        add_columns = np.flatnonzero(new_member & ~old_member)
        remove_columns = np.flatnonzero(old_member & ~new_member)
        product = exact_flat[3 + bit]
        for start, stop, counts in guard_batches:
            if add_columns.size:
                product[start:stop] += counts[add_columns].sum(
                    axis=0,
                    dtype=np.uint64,
                )
            if remove_columns.size:
                product[start:stop] -= counts[remove_columns].sum(
                    axis=0,
                    dtype=np.uint64,
                )
        changed_counts.append(int(add_columns.size + remove_columns.size))
    return True, tuple(changed_counts)


def _center_of_mass_from_exact(
    total: np.ndarray,
    detector_row_moment: np.ndarray,
    detector_column_moment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert exact integer moments to absolute float32 CoM coordinates."""

    total = np.asarray(total, dtype=np.uint64)
    row = np.zeros(total.shape, dtype=np.float64)
    column = np.zeros(total.shape, dtype=np.float64)
    np.divide(
        np.asarray(detector_row_moment, dtype=np.uint64),
        total,
        out=row,
        where=total != 0,
    )
    np.divide(
        np.asarray(detector_column_moment, dtype=np.uint64),
        total,
        out=column,
        where=total != 0,
    )
    return row.astype(np.float32), column.astype(np.float32)


def _screening_product_views(
    exact_flat: np.ndarray,
    scan_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Map the fused CUDA statistic order to exact scan-shaped products."""

    exact = np.asarray(exact_flat)
    expected_shape = (7, int(np.prod(scan_shape, dtype=np.int64)))
    if exact.dtype != np.dtype(np.uint64):
        raise TypeError(
            "Fused CUDA screening statistics must use uint64; "
            f"got {exact.dtype}."
        )
    if exact.shape != expected_shape:
        raise ValueError(
            f"Fused CUDA screening statistics have shape {exact.shape}, "
            f"expected {expected_shape}."
        )
    return {
        "total_intensity": exact[0].reshape(scan_shape),
        "detector_row_moment": exact[1].reshape(scan_shape),
        "detector_column_moment": exact[2].reshape(scan_shape),
        "bright_field": exact[3].reshape(scan_shape),
        "annular_bright_field": exact[4].reshape(scan_shape),
        "annular_dark_field": exact[5].reshape(scan_shape),
        "dark_field": exact[6].reshape(scan_shape),
    }


def _host_peak_rss_bytes() -> int | None:
    """Return process high-water RSS on Unix hosts."""

    try:
        import resource
        import sys

        high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return high_water if sys.platform == "darwin" else high_water * 1024
    except (OSError, ValueError):
        return None


def _add_stage_seconds(
    totals: dict[str, float],
    timing: dict[str, Any],
) -> None:
    """Accumulate scalar stage timings without discarding unknown fields."""

    for name, seconds in timing.items():
        if isinstance(seconds, (int, float, np.integer, np.floating)):
            totals[name] = totals.get(name, 0.0) + float(seconds)


def _build_exact_cuda_products(
    master: Path,
    *,
    scan_shape: tuple[int, int],
    chunk_rows: int,
    rotation_steps: int,
    output_dtype,
    memory_plan,
    verbose: bool,
    source_fingerprint_fn: Callable[[Path], dict[str, Any]],
):
    """Build full-scan exact products with bounded, overlapped CUDA batches."""

    import cupy as cp

    from quantem.gpu.detector import auto_probe
    from quantem.gpu.detector.compute.cuda.kernels import (
        _cuda_screening_sums_exact,
    )
    from quantem.gpu.dpc.workflow import find_optimal_rotation
    from quantem.gpu.io import inspect as inspect_source
    from quantem.gpu.io.load import (
        _decompress_prepared,
        _discover_chunk_names,
        _prepare_master_frames,
        _SparseFrameReadSession,
    )

    from .workflow import (
        _CACHE_VERSION,
        ScreeningResult,
        _dpc_phase,
        _require_source_unchanged,
        _screening_output_dtype,
    )

    started = time.perf_counter()
    metadata = inspect_source(str(master)).metadata
    source_fingerprint = source_fingerprint_fn(master)
    detector_shape = tuple(int(v) for v in metadata.get("detector_shape") or ())
    if len(detector_shape) != 2:
        raise ValueError("Could not determine detector_shape from HDF5 metadata")
    source_dtype = np.dtype(metadata.get("dtype") or output_dtype)
    output_dtype = _screening_output_dtype(output_dtype)
    if output_dtype != np.dtype(np.uint16):
        raise ValueError(
            "The private exact CUDA screening candidate is qualified only for "
            f"uint16 decoded counts; got {output_dtype}."
        )
    if not np.can_cast(source_dtype, output_dtype, casting="safe"):
        raise ValueError(
            "Exact CUDA screening may preserve or safely widen source counts, "
            f"but may not narrow {source_dtype} to {output_dtype}."
        )

    num_frames = int(np.prod(scan_shape, dtype=np.int64))
    max_batch_frames = min(
        num_frames,
        max(1, int(chunk_rows) * int(scan_shape[1])),
    )
    detector_sum = np.zeros(detector_shape, dtype=np.uint64)
    # Product order: total, row moment, column moment, BF, ABF, ADF, DF.
    exact_flat = np.empty((7, num_frames), dtype=np.uint64)
    full_frame_count = 0
    provisional_masks = None
    detector_band_bits = None
    detector_band_bits_gpu = None
    provisional_center = None
    provisional_radius = None
    mask_guard_batches: list[tuple[int, int, np.ndarray]] = []
    mask_guard_pinned: list[Any] = []
    mask_guard_indices = None
    mask_guard_slots = None
    mask_guard_bytes = 0
    mask_guard_capture_s = 0.0
    mask_guard_correction_s = 0.0
    mask_guard_changed_pixels = (0, 0, 0, 0)
    mask_change_resolution = "pending"
    fallback_reason = None
    bootstrap_source = "pending"
    sample_positions_used = 0
    prepare_stage_s: dict[str, float] = {}
    prepare_wall_s = 0.0
    decode_wall_s = 0.0
    reduction_wall_s = 0.0
    source_read_bytes = 0
    source_read_spans = 0
    chunk_timings: list[dict[str, float | int]] = []
    peak_gpu_allocated_bytes = 0
    peak_gpu_reserved_bytes = 0
    gpu_free_bytes, gpu_total_bytes = cp.cuda.runtime.memGetInfo()
    gpu_total_used_baseline_bytes = int(gpu_total_bytes - gpu_free_bytes)
    gpu_total_used_peak_bytes = gpu_total_used_baseline_bytes
    memory_limit_bytes = int(float(memory_plan.memory_budget_gb) * (1 << 30))
    chunk_names = _discover_chunk_names(str(master))
    if not chunk_names:
        chunk_names = ["data"]

    _require_source_unchanged(
        master,
        source_fingerprint,
        stage="before the primary CUDA stream",
    )

    def accumulate_prepare_metadata(prepared: dict) -> None:
        nonlocal prepare_wall_s, source_read_bytes, source_read_spans
        timing = prepared.get("prepare_timing_s", {})
        prepare_wall_s += float(timing.get("total", 0.0))
        _add_stage_seconds(prepare_stage_s, timing)
        source_read_bytes += int(prepared.get("total_compressed_bytes", 0))
        source_read_spans += int(prepared.get("read_span_count", 0))

    def update_memory_peaks() -> None:
        nonlocal peak_gpu_allocated_bytes
        nonlocal peak_gpu_reserved_bytes
        nonlocal gpu_total_used_peak_bytes
        pool = cp.get_default_memory_pool()
        allocated = int(pool.used_bytes())
        reserved = int(pool.total_bytes())
        peak_gpu_allocated_bytes = max(peak_gpu_allocated_bytes, allocated)
        peak_gpu_reserved_bytes = max(peak_gpu_reserved_bytes, reserved)
        free_now, total_now = cp.cuda.runtime.memGetInfo()
        gpu_total_used_peak_bytes = max(
            gpu_total_used_peak_bytes,
            int(total_now - free_now),
        )
        if reserved > memory_limit_bytes:
            raise MemoryError(
                "CUDA screening exceeded its process VRAM envelope: "
                f"reserved={reserved}, limit={memory_limit_bytes}. "
                "Reduce chunk_rows or memory_budget_gb."
            )

    def prepare_range(session, frame_range: tuple[int, int]) -> dict:
        start, stop = frame_range
        return _prepare_master_frames(
            str(master),
            chunk_names,
            np.arange(start, stop, dtype=np.int64),
            apply_mask=True,
            read_session=session,
        )

    stream_started = time.perf_counter()
    fallback_s = 0.0
    fallback_prepare_s = 0.0
    fallback_decode_s = 0.0
    fallback_reduce_s = 0.0
    fallback_source_read_bytes = 0
    with _SparseFrameReadSession(
        str(master),
        chunk_names,
        apply_mask=True,
    ) as read_session:
        batch_ranges = read_session.source_aligned_ranges(
            frame_count=num_frames,
            max_batch_frames=max_batch_frames,
        )
        with ThreadPoolExecutor(max_workers=1) as prepare_pool:
            future = prepare_pool.submit(
                prepare_range,
                read_session,
                batch_ranges[0],
            )
            for batch_index, (start, stop) in enumerate(batch_ranges):
                prepared = future.result()
                accumulate_prepare_metadata(prepared)
                if batch_index + 1 < len(batch_ranges):
                    future = prepare_pool.submit(
                        prepare_range,
                        read_session,
                        batch_ranges[batch_index + 1],
                    )

                decode_started = time.perf_counter()
                data = _decompress_prepared(
                    prepared,
                    verbose=False,
                    auto_narrow=False,
                    output_dtype=output_dtype,
                    prune_device_pool=False,
                )
                decode_s = time.perf_counter() - decode_started
                decode_wall_s += decode_s
                flat = data.reshape(-1, *detector_shape)
                if int(flat.shape[0]) != stop - start:
                    raise RuntimeError(
                        f"Prepared range [{start}, {stop}) decoded "
                        f"{int(flat.shape[0])} frames."
                    )

                reduce_started = time.perf_counter()
                chunk_sum_gpu = flat.sum(axis=0, dtype=cp.uint64)
                if detector_band_bits is None:
                    chunk_sum_host = cp.asnumpy(chunk_sum_gpu)
                    first_mean = chunk_sum_host.astype(np.float32) / float(stop - start)
                    first_center, first_radius = auto_probe(first_mean)
                    provisional_center = (
                        float(first_center[0]),
                        float(first_center[1]),
                    )
                    provisional_radius = float(first_radius)
                    provisional_masks = _band_masks(
                        first_center,
                        first_radius,
                        detector_shape,
                    )
                    detector_band_bits = _band_bits(provisional_masks)
                    detector_band_bits_gpu = cp.asarray(detector_band_bits)
                    candidate_guard_indices = _mask_guard_indices(
                        provisional_center,
                        provisional_radius,
                        detector_shape,
                    )
                    candidate_guard_bytes = int(
                        num_frames
                        * candidate_guard_indices.size
                        * output_dtype.itemsize
                    )
                    if candidate_guard_bytes <= _MASK_GUARD_MAX_BYTES:
                        mask_guard_indices = candidate_guard_indices
                        mask_guard_slots = np.full(
                            detector_shape,
                            -1,
                            dtype=np.int32,
                        )
                        mask_guard_slots.reshape(-1)[mask_guard_indices] = np.arange(
                            mask_guard_indices.size,
                            dtype=np.int32,
                        )
                        mask_guard_bytes = candidate_guard_bytes
                    else:
                        fallback_reason = "mask_guard_host_limit"
                    sample_positions_used = stop - start
                    bootstrap_source = (
                        "full_scan" if stop - start == num_frames else "first_batch"
                    )
                else:
                    chunk_sum_host = None
                fused_result = _cuda_screening_sums_exact(
                    data,
                    detector_band_bits_gpu,
                    mask_guard_slots,
                    guard_count=(
                        None
                        if mask_guard_indices is None
                        else int(mask_guard_indices.size)
                    ),
                )
                if fused_result is None:
                    raise RuntimeError(
                        "Exact fused CUDA screening does not support the decoded "
                        f"layout {data.shape} {data.dtype}."
                    )
                if mask_guard_indices is None:
                    fused_gpu = fused_result
                    guard_gpu = None
                else:
                    fused_gpu, guard_gpu = fused_result
                if chunk_sum_host is None:
                    chunk_sum_host = cp.asnumpy(chunk_sum_gpu)
                fused_host = cp.asnumpy(fused_gpu)
                if guard_gpu is not None:
                    guard_started = time.perf_counter()
                    pinned = cp.cuda.alloc_pinned_memory(int(guard_gpu.nbytes))
                    guard_host = np.frombuffer(
                        pinned,
                        dtype=output_dtype,
                        count=int(guard_gpu.size),
                    ).reshape(tuple(int(value) for value in guard_gpu.shape))
                    guard_gpu.get(out=guard_host)
                    mask_guard_pinned.append(pinned)
                    mask_guard_batches.append((start, stop, guard_host))
                    mask_guard_capture_s += time.perf_counter() - guard_started
                reduce_s = time.perf_counter() - reduce_started
                reduction_wall_s += reduce_s
                np.add(detector_sum, chunk_sum_host, out=detector_sum)
                exact_flat[:, start:stop] = fused_host
                full_frame_count += stop - start
                update_memory_peaks()
                chunk_timings.append(
                    {
                        "start": int(start),
                        "stop": int(stop),
                        "decode_s": float(decode_s),
                        "reduce_s": float(reduce_s),
                        "decoded_bytes": int(data.nbytes),
                    }
                )
                del data, flat, chunk_sum_gpu, fused_gpu
                if guard_gpu is not None:
                    del guard_gpu
                del chunk_sum_host, fused_host
                if verbose:
                    print(
                        f"  frames {start}:{stop} decode={decode_s:.3f}s "
                        f"reduce={reduce_s:.3f}s"
                    )

        if full_frame_count != num_frames:
            raise RuntimeError(
                f"CUDA screening visited {full_frame_count} frames, "
                f"expected {num_frames}."
            )
        if provisional_masks is None:
            raise RuntimeError("Screening masks were not initialized")
        if provisional_center is None or provisional_radius is None:
            raise RuntimeError("Screening bootstrap geometry was not initialized")

        mean_dp = detector_sum.astype(np.float32) / float(full_frame_count)
        center, radius = auto_probe(mean_dp)
        authoritative_masks = _band_masks(center, radius, detector_shape)
        masks_identical = all(
            np.array_equal(provisional_masks[name], authoritative_masks[name])
            for name in authoritative_masks
        )
        pass_count = 1
        if not masks_identical:
            authoritative_bits = _band_bits(authoritative_masks)
            corrected_from_guard = False
            if mask_guard_batches and mask_guard_indices is not None:
                correction_started = time.perf_counter()
                corrected_from_guard, mask_guard_changed_pixels = (
                    _apply_mask_guard_correction(
                        exact_flat,
                        mask_guard_batches,
                        mask_guard_indices,
                        detector_band_bits,
                        authoritative_bits,
                    )
                )
                mask_guard_correction_s = time.perf_counter() - correction_started
            if corrected_from_guard:
                mask_change_resolution = "exact_guard_correction"
            else:
                pass_count = 2
                mask_change_resolution = "exact_second_pass"
                if fallback_reason is None:
                    fallback_reason = "mask_change_outside_guard"
                fallback_started = time.perf_counter()
                authoritative_bits_gpu = cp.asarray(authoritative_bits)
                with ThreadPoolExecutor(max_workers=1) as prepare_pool:
                    future = prepare_pool.submit(
                        prepare_range,
                        read_session,
                        batch_ranges[0],
                    )
                    for batch_index, (start, stop) in enumerate(batch_ranges):
                        prepared = future.result()
                        timing = prepared.get("prepare_timing_s", {})
                        fallback_prepare_s += float(timing.get("total", 0.0))
                        fallback_source_read_bytes += int(
                            prepared.get("total_compressed_bytes", 0)
                        )
                        if batch_index + 1 < len(batch_ranges):
                            future = prepare_pool.submit(
                                prepare_range,
                                read_session,
                                batch_ranges[batch_index + 1],
                            )
                        decode_started = time.perf_counter()
                        data = _decompress_prepared(
                            prepared,
                            verbose=False,
                            auto_narrow=False,
                            output_dtype=output_dtype,
                            prune_device_pool=False,
                        )
                        fallback_decode_s += time.perf_counter() - decode_started
                        reduce_started = time.perf_counter()
                        fused_gpu = _cuda_screening_sums_exact(
                            data,
                            authoritative_bits_gpu,
                        )
                        if fused_gpu is None:
                            raise RuntimeError("Authoritative CUDA screening failed")
                        fused_host = cp.asnumpy(fused_gpu)
                        fallback_reduce_s += time.perf_counter() - reduce_started
                        exact_flat[3:7, start:stop] = fused_host[3:7]
                        update_memory_peaks()
                        del data, fused_gpu, fused_host
                fallback_s = time.perf_counter() - fallback_started
        else:
            mask_change_resolution = "identical_bootstrap_mask"

    stream_s = time.perf_counter() - stream_started
    _require_source_unchanged(
        master,
        source_fingerprint,
        stage="after exact CUDA screening",
    )

    exact_products = _screening_product_views(exact_flat, scan_shape)
    com_row, com_col = _center_of_mass_from_exact(
        exact_products["total_intensity"],
        exact_products["detector_row_moment"],
        exact_products["detector_column_moment"],
    )
    com_row -= float(com_row.mean())
    com_col -= float(com_col.mean())
    rotation_started = time.perf_counter()
    _, _, rotation_deg, transposed = find_optimal_rotation(
        com_row,
        com_col,
        rotation_steps=rotation_steps,
    )
    rotation_s = time.perf_counter() - rotation_started
    phase_started = time.perf_counter()
    phase = _dpc_phase(
        com_row,
        com_col,
        float(rotation_deg),
        bool(transposed),
    )
    phase_s = time.perf_counter() - phase_started
    elapsed_s = time.perf_counter() - started
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    timing = {
        "probe_source": "full_scan_exact",
        "bootstrap_source": bootstrap_source,
        "sample_positions_requested": 0,
        "sample_positions_used": int(sample_positions_used),
        "exact_sum_method": (
            "CUDA uint64 detector sums plus one fused uint64 total/moment/"
            "BF/ABF/ADF/DF traversal per batch"
        ),
        "full_frame_count": int(full_frame_count),
        "masks_identical": masks_identical,
        "pass_count": int(pass_count),
        "prepare_wall_accumulated_s": float(prepare_wall_s),
        "prepare_stage_accumulated_s": prepare_stage_s,
        "decode_wall_accumulated_s": float(decode_wall_s),
        "reduce_wall_accumulated_s": float(reduction_wall_s),
        "fallback_s": float(fallback_s),
        "fallback_prepare_s": float(fallback_prepare_s),
        "fallback_decode_s": float(fallback_decode_s),
        "fallback_reduce_s": float(fallback_reduce_s),
        "mask_guard_capture_s": float(mask_guard_capture_s),
        "mask_guard_correction_s": float(mask_guard_correction_s),
        "stream_s": float(stream_s),
        "rotation_s": float(rotation_s),
        "idpc_s": float(phase_s),
        "first_usable_s": float(elapsed_s),
        "exact_complete_s": float(elapsed_s),
        "elapsed_s": float(elapsed_s),
        "batch_count": len(batch_ranges),
        "batch_decode_median_s": float(
            np.median([chunk["decode_s"] for chunk in chunk_timings])
        ),
        "batch_reduce_median_s": float(
            np.median([chunk["reduce_s"] for chunk in chunk_timings])
        ),
    }
    parameters = {
        "source_scan_shape": [int(value) for value in scan_shape],
        "working_scan_shape": [int(value) for value in scan_shape],
        "source_detector_shape": [int(value) for value in detector_shape],
        "working_detector_shape": [int(value) for value in detector_shape],
        "source_dtype": str(source_dtype),
        "working_dtype": str(output_dtype),
        "scan_region": None,
        "detector_region": None,
        "scan_bin": 1,
        "detector_bin": 1,
        "chunk_rows_budget": int(chunk_rows),
        "max_batch_frames": int(max_batch_frames),
        "batch_policy": "source-shard-aligned",
        "batch_ranges": [[int(start), int(stop)] for start, stop in batch_ranges],
        "sample_positions": 0,
        "sample_positions_used": int(sample_positions_used),
        "probe_source": "full_scan_exact",
        "bootstrap_source": bootstrap_source,
        "full_frame_count": int(full_frame_count),
        "masks_identical": masks_identical,
        "pass_count": int(pass_count),
        "mask_change_resolution": mask_change_resolution,
        "fallback_reason": fallback_reason,
        "mask_guard_margin_px": _MASK_GUARD_MARGIN_PX,
        "mask_guard_pixel_count": (
            0 if mask_guard_indices is None else int(mask_guard_indices.size)
        ),
        "mask_guard_changed_pixels_by_band": list(mask_guard_changed_pixels),
        "rotation_steps": int(rotation_steps),
        "bootstrap_center": [
            float(provisional_center[0]),
            float(provisional_center[1]),
        ],
        "bootstrap_radius_px": float(provisional_radius),
        "center": [float(center[0]), float(center[1])],
        "radius_px": float(radius),
        "rotation_deg": float(rotation_deg),
        "transposed": bool(transposed),
        "backend": "cuda",
        "memory_budget_gb": float(memory_plan.memory_budget_gb),
        "memory_budget_source": memory_plan.memory_budget_source,
        "chunk_rows_source": memory_plan.chunk_rows_source,
    }
    memory = {
        **asdict(memory_plan),
        "process_vram_limit_bytes": int(memory_limit_bytes),
        "planned_working_set_bytes": _exact_cuda_working_set_bytes(
            memory_plan,
            chunk_rows,
        ),
        "planned_working_set_fraction": _CUDA_WORKING_SET_FRACTION,
        "mask_guard_host_bytes": int(mask_guard_bytes),
        "mask_guard_host_limit_bytes": _MASK_GUARD_MAX_BYTES,
        "peak_gpu_allocated_bytes": int(peak_gpu_allocated_bytes),
        "peak_gpu_reserved_bytes": int(peak_gpu_reserved_bytes),
        "gpu_total_used_baseline_bytes": int(gpu_total_used_baseline_bytes),
        "gpu_total_used_peak_bytes": int(gpu_total_used_peak_bytes),
        "gpu_total_used_delta_peak_bytes": int(
            max(0, gpu_total_used_peak_bytes - gpu_total_used_baseline_bytes)
        ),
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
    }
    return ScreeningResult(
        mean_dp=np.asarray(mean_dp, dtype=np.float32),
        bright_field=exact_products["bright_field"].astype(np.float32),
        dark_field=exact_products["dark_field"].astype(np.float32),
        dpc_phase=phase,
        com_row=com_row.astype(np.float32, copy=False),
        com_col=com_col.astype(np.float32, copy=False),
        probe_center=(float(center[0]), float(center[1])),
        probe_radius=float(radius),
        rotation_deg=float(rotation_deg),
        transposed=bool(transposed),
        metadata={
            "version": _CACHE_VERSION,
            "source": source_fingerprint,
            "parameters": parameters,
            "timing": timing,
            "memory": memory,
            "io": {
                "primary_compressed_bytes": int(source_read_bytes),
                "primary_read_spans": int(source_read_spans),
                "fallback_compressed_bytes": int(fallback_source_read_bytes),
                "logical_decoded_bytes": int(
                    num_frames * np.prod(detector_shape) * output_dtype.itemsize
                ),
            },
            "exact_accumulation": {
                "dtype": "uint64",
                "published_uint64_products": [
                    "total_intensity",
                    "annular_bright_field",
                    "annular_dark_field",
                ],
                "detector_band_order": ["BF", "ABF", "ADF", "DF"],
                "detector_band_radius_multipliers": [
                    [0.0, 1.0],
                    [0.5, 1.0],
                    [1.0, 2.0],
                    [1.0, None],
                ],
                "mean_dp_divisor": int(full_frame_count),
                "coordinate_order": "row-column",
            },
            "mode": "cached-screening-products",
            "note": (
                "The primary pass used exact uint64 sufficient statistics for "
                "the cached total/BF/ABF/ADF/DF/CoM products. A changed "
                "bootstrap mask is "
                "resolved by validated exact guard-pixel correction or an exact "
                "second source pass. The raw HDF5 remains the reconstruction "
                "evidence source."
            ),
        },
        from_cache=False,
        elapsed_s=elapsed_s,
        total_intensity=exact_products["total_intensity"],
        annular_bright_field=exact_products["annular_bright_field"],
        annular_dark_field=exact_products["annular_dark_field"],
    )
