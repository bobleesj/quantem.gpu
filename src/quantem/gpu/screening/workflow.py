"""Cached BF/DF/CoM/rotation products for large 4D-STEM HDF5 masters."""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._memory import (
    MemoryPlan,
    _memory_plan_for_shapes,
    _memory_plan_with_chunk_rows,
)
from ._mps import (
    _clear_mps_transients,
    _metadata_dtype,
    _mps_chunked_frames_for,
    _mps_mean_dp,
)

_CACHE_VERSION = 3


def _screening_output_dtype(output_dtype) -> np.dtype:
    """Return a NumPy dtype for product caches, rejecting packed raw-stack tokens."""
    if isinstance(output_dtype, str) and output_dtype.lower() in {"u4", "uint4"}:
        raise ValueError(
            "output_dtype='u4' means packed 4-bit raw detector counts and is "
            "not valid for derived screening products. Use output_dtype=np.uint8, "
            "np.uint16, or np.float32."
        )
    return np.dtype(output_dtype)


@dataclass
class ScreeningResult:
    """Small screening products for instant UI and ptychography setup."""

    mean_dp: np.ndarray
    bright_field: np.ndarray
    dark_field: np.ndarray
    dpc_phase: np.ndarray
    com_row: np.ndarray
    com_col: np.ndarray
    probe_center: tuple[float, float]
    probe_radius: float
    rotation_deg: float
    transposed: bool
    metadata: dict[str, Any]
    cache_path: Path | None = None
    from_cache: bool = False
    elapsed_s: float = 0.0


def _cache_path(
    master: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the default cache path for a master HDF5 file."""
    master_path = Path(master).expanduser()
    if cache_dir is None:
        cache_root = master_path.parent / ".quantem_gpu_cache"
    else:
        cache_root = Path(cache_dir).expanduser()
    return cache_root / f"{master_path.stem}.screening-v{_CACHE_VERSION}.npz"


def _source_fingerprint(master: Path) -> dict[str, Any]:
    """Return the complete master-and-shard source identity."""
    from quantem.gpu.io import inspect as inspect_source

    inspection = inspect_source(str(master))
    signature = inspection.source_signature
    if signature:
        return dict(signature)
    stat = master.stat()
    return {
        "master": str(master.resolve()),
        "files": [
            {
                "path": str(master.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        ],
        "datasets": [],
        "expectation": {"frames": None, "basis": None},
    }


def _cache_matches(metadata: dict[str, Any], master: Path) -> bool:
    if int(metadata.get("version", -1)) != _CACHE_VERSION:
        return False
    cached_source = metadata.get("source")
    stat_match = _strong_cached_source_match(cached_source, master)
    if stat_match is not None:
        return stat_match
    return cached_source == _source_fingerprint(master)


def _strong_cached_source_match(
    cached_source: object,
    master: Path,
) -> bool | None:
    """Validate a cached source identity using fail-closed file statistics.

    Complete HDF5 inspections retain size, modification/change times, device,
    and inode for the master and every external shard. When all strong fields
    are present, comparing them is equivalent to rediscovering the same HDF5
    links but avoids reparsing every source header on each cache reopen.

    ``None`` asks the caller to use full HDF5 inspection for an older or
    reduced signature. A malformed complete signature returns ``False``.
    """

    if not isinstance(cached_source, dict):
        return False
    files = cached_source.get("files")
    cached_master = cached_source.get("master")
    if not isinstance(files, list) or not files or not isinstance(cached_master, str):
        return False
    required = {"path", "size", "mtime_ns", "ctime_ns", "device", "inode"}
    if any(not isinstance(item, dict) or not required <= item.keys() for item in files):
        return None

    try:
        master_path = str(master.expanduser().resolve(strict=True))
    except OSError:
        return False
    if cached_master != master_path:
        return False

    observed_paths: set[str] = set()
    for item in files:
        path = item["path"]
        if not isinstance(path, str) or path in observed_paths:
            return False
        observed_paths.add(path)
        try:
            stat = Path(path).stat()
        except OSError:
            return False
        observed = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
        }
        if any(int(item[name]) != value for name, value in observed.items()):
            return False
    return master_path in observed_paths


def _require_source_unchanged(
    master: Path,
    expected: dict[str, Any],
    *,
    stage: str,
) -> None:
    """Fail closed if a source file changed during a screening pass."""
    if _source_fingerprint(master) != expected:
        raise RuntimeError(
            "The 4D-STEM source changed while screening products were being "
            f"computed ({stage}). Wait for acquisition to finish, then retry."
        )


def _metadata_array(metadata: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(metadata, separators=(",", ":")), dtype=np.str_)


def _read_metadata_array(array: np.ndarray) -> dict[str, Any]:
    return json.loads(str(array.reshape(())))


def _save_cache(products: ScreeningResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(products.metadata)
    np.savez(
        path,
        metadata_json=_metadata_array(metadata),
        mean_dp=np.asarray(products.mean_dp, dtype=np.float32),
        bright_field=np.asarray(products.bright_field, dtype=np.float32),
        dark_field=np.asarray(products.dark_field, dtype=np.float32),
        dpc_phase=np.asarray(products.dpc_phase, dtype=np.float32),
        com_row=np.asarray(products.com_row, dtype=np.float32),
        com_col=np.asarray(products.com_col, dtype=np.float32),
    )


def _prepare_cache(path: Path, master: Path) -> ScreeningResult | None:
    if not path.exists():
        return None
    t0 = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        metadata = _read_metadata_array(data["metadata_json"])
        if not _cache_matches(metadata, master):
            return None
        params = metadata.get("parameters", {})
        com_row = np.asarray(data["com_row"], dtype=np.float32)
        com_col = np.asarray(data["com_col"], dtype=np.float32)
        dpc_phase = (
            np.asarray(data["dpc_phase"], dtype=np.float32)
            if "dpc_phase" in data.files
            else _dpc_phase(
                com_row,
                com_col,
                float(params["rotation_deg"]),
                bool(params["transposed"]),
            )
        )
        products = ScreeningResult(
            mean_dp=np.asarray(data["mean_dp"], dtype=np.float32),
            bright_field=np.asarray(data["bright_field"], dtype=np.float32),
            dark_field=np.asarray(data["dark_field"], dtype=np.float32),
            dpc_phase=dpc_phase,
            com_row=com_row,
            com_col=com_col,
            probe_center=tuple(float(v) for v in params["center"]),
            probe_radius=float(params["radius_px"]),
            rotation_deg=float(params["rotation_deg"]),
            transposed=bool(params["transposed"]),
            metadata=metadata,
            cache_path=path,
            from_cache=True,
            elapsed_s=time.perf_counter() - t0,
        )
    return products


def _dpc_phase(
    com_row: np.ndarray,
    com_col: np.ndarray,
    rotation_deg: float,
    transposed: bool,
) -> np.ndarray:
    """Return the float32 iDPC phase for a prepared CoM field."""

    from quantem.gpu.dpc import integrate

    source_row = com_col if transposed else com_row
    source_col = com_row if transposed else com_col
    angle = np.radians(rotation_deg)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    aligned_row = (cosine * source_row - sine * source_col).astype(np.float32)
    aligned_col = (sine * source_row + cosine * source_col).astype(np.float32)
    gradient_row, gradient_col = (
        (aligned_col, aligned_row) if transposed else (aligned_row, aligned_col)
    )
    return integrate(gradient_row, gradient_col)


def _clear_cuda_pools() -> None:
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001, S110 - cleanup must remain best-effort
        pass
    gc.collect()


def _memory_plan(
    master: str | Path,
    *,
    scan_shape: tuple[int, int] | None = None,
    memory_budget_gb: float | None = None,
    output_dtype=np.uint16,
) -> MemoryPlan:
    """Return the streaming memory plan without reading detector frames."""
    from quantem.gpu.io import inspect as inspect_source

    master_path = Path(master).expanduser()
    metadata = inspect_source(str(master_path)).metadata
    if scan_shape is None:
        scan_shape = tuple(int(v) for v in metadata.get("scan_shape") or ())
    if len(scan_shape) != 2:
        raise ValueError("scan_shape=(rows, cols) is required for screening products")
    detector_shape = tuple(int(v) for v in metadata.get("detector_shape") or ())
    if len(detector_shape) != 2:
        raise ValueError("Could not determine detector_shape from HDF5 metadata")
    return _memory_plan_for_shapes(
        scan_shape,
        detector_shape,
        _screening_output_dtype(output_dtype).itemsize,
        memory_budget_gb,
    )


def _build_cuda_products(
    master: Path,
    *,
    scan_shape: tuple[int, int],
    chunk_rows: int,
    sample_positions: int,
    seed: int | None,
    rotation_steps: int,
    output_dtype,
    memory_plan: MemoryPlan,
    verbose: bool,
) -> ScreeningResult:
    exact_dtype = _screening_output_dtype(output_dtype)
    if int(sample_positions) == 0 and exact_dtype == np.dtype(np.uint16):
        from ._cuda import _build_exact_cuda_products

        try:
            return _build_exact_cuda_products(
                master,
                scan_shape=scan_shape,
                chunk_rows=chunk_rows,
                rotation_steps=rotation_steps,
                output_dtype=output_dtype,
                memory_plan=memory_plan,
                verbose=verbose,
                source_fingerprint_fn=_source_fingerprint,
            )
        except BaseException:
            _clear_cuda_pools()
            raise

    import cupy as cp

    from quantem.gpu.detector import auto_probe, detector_mask, mean_dp
    from quantem.gpu.detector.compute.cuda.kernels import (
        cuda_center_of_mass,
        cuda_masked_sum,
    )
    from quantem.gpu.dpc.workflow import find_optimal_rotation
    from quantem.gpu.io import inspect as inspect_source
    from quantem.gpu.io import load

    t0 = time.perf_counter()
    metadata = inspect_source(str(master)).metadata
    source_fingerprint = _source_fingerprint(master)
    detector_shape = tuple(int(v) for v in metadata.get("detector_shape") or ())
    if len(detector_shape) != 2:
        raise ValueError("Could not determine detector_shape from HDF5 metadata")

    bf_map = np.empty(scan_shape, dtype=np.float32)
    df_map = np.empty(scan_shape, dtype=np.float32)
    com_row = np.empty(scan_shape, dtype=np.float32)
    com_col = np.empty(scan_shape, dtype=np.float32)
    chunk_rows = int(max(1, min(chunk_rows, scan_shape[0])))
    full_scan_single_chunk = chunk_rows >= scan_shape[0]
    sample_load_s = 0.0
    sample_product_s = 0.0
    sample_positions_used = 0
    dp = None
    center = None
    radius = None
    bf_mask = None
    df_mask = None
    detector_sum = np.zeros(detector_shape, dtype=np.uint64)
    full_frame_count = 0
    bootstrap_source = "pending"
    if int(sample_positions) > 0 and not full_scan_single_chunk:
        sample_t0 = time.perf_counter()
        sample = load(
            str(master),
            random_positions=int(sample_positions),
            scan_shape=scan_shape,
            seed=seed,
            backend="cuda",
            dtype=output_dtype,
            verbose=False,
        )
        cp.cuda.Stream.null.synchronize()
        sample_load_s = time.perf_counter() - sample_t0

        dp_t0 = time.perf_counter()
        dp = mean_dp(sample.data)
        center, radius = auto_probe(dp)
        bf_mask = detector_mask(center, 0.0, radius, dp.shape)
        df_mask = detector_mask(center, radius, np.inf, dp.shape)
        cp.cuda.Stream.null.synchronize()
        sample_product_s = time.perf_counter() - dp_t0
        sample_positions_used = int(sample_positions)
        bootstrap_source = "random_scan_sample"
        del sample
        _clear_cuda_pools()

    _require_source_unchanged(
        master,
        source_fingerprint,
        stage="before the primary CUDA stream",
    )

    chunk_timings: list[dict[str, float]] = []
    stream_t0 = time.perf_counter()
    for r0 in range(0, scan_shape[0], chunk_rows):
        r1 = min(scan_shape[0], r0 + chunk_rows)
        load_t0 = time.perf_counter()
        if r0 == 0 and r1 == scan_shape[0]:
            result = load(
                str(master),
                scan_shape=scan_shape,
                backend="cuda",
                dtype=output_dtype,
                verbose=False,
            )
        else:
            result = load(
                str(master),
                scan_region=(r0, r1, 0, scan_shape[1]),
                scan_shape=scan_shape,
                backend="cuda",
                dtype=output_dtype,
                verbose=False,
            )
        cp.cuda.Stream.null.synchronize()
        load_s = time.perf_counter() - load_t0

        data = result.data
        if dp is None:
            product_t0 = time.perf_counter()
            dp = mean_dp(data)
            center, radius = auto_probe(dp)
            bf_mask = detector_mask(center, 0.0, radius, dp.shape)
            df_mask = detector_mask(center, radius, np.inf, dp.shape)
            cp.cuda.Stream.null.synchronize()
            sample_product_s = time.perf_counter() - product_t0
            sample_positions_used = int(np.prod(data.shape[:2]))
            bootstrap_source = (
                "full_scan" if full_scan_single_chunk else "first_chunk"
            )
        exact_sum_t0 = time.perf_counter()
        flat = data.reshape(-1, *data.shape[-2:])
        chunk_sum = flat.sum(axis=0, dtype=cp.uint64)
        cp.cuda.Stream.null.synchronize()
        np.add(detector_sum, cp.asnumpy(chunk_sum), out=detector_sum)
        full_frame_count += int(flat.shape[0])
        exact_sum_s = time.perf_counter() - exact_sum_t0
        reduce_t0 = time.perf_counter()
        if dp is None or center is None or radius is None or bf_mask is None or df_mask is None:
            raise RuntimeError("Screening masks were not initialized")
        bf_gpu = cuda_masked_sum(data, bf_mask)
        df_gpu = cuda_masked_sum(data, df_mask)
        row_gpu, col_gpu = cuda_center_of_mass(data, None)
        cp.cuda.Stream.null.synchronize()
        reduce_s = time.perf_counter() - reduce_t0

        bf_map[r0:r1, :] = cp.asnumpy(bf_gpu).reshape(r1 - r0, scan_shape[1])
        df_map[r0:r1, :] = cp.asnumpy(df_gpu).reshape(r1 - r0, scan_shape[1])
        com_row[r0:r1, :] = cp.asnumpy(row_gpu).reshape(r1 - r0, scan_shape[1])
        com_col[r0:r1, :] = cp.asnumpy(col_gpu).reshape(r1 - r0, scan_shape[1])
        chunk_timings.append(
            {
                "load_s": float(load_s),
                "exact_sum_s": float(exact_sum_s),
                "reduce_s": float(reduce_s),
                "resident_gb": float(data.nbytes / 1e9),
            }
        )
        del result, data, flat, chunk_sum, bf_gpu, df_gpu, row_gpu, col_gpu
        _clear_cuda_pools()
        if verbose:
            print(f"  rows {r0}:{r1} load={load_s:.3f}s reduce={reduce_s:.3f}s")

    stream_s = time.perf_counter() - stream_t0
    _require_source_unchanged(
        master,
        source_fingerprint,
        stage="after the primary CUDA stream",
    )

    if full_frame_count != int(np.prod(scan_shape)):
        raise RuntimeError(
            "CUDA screening visited "
            f"{full_frame_count} frames, expected {int(np.prod(scan_shape))}."
        )
    provisional_bf_mask = np.asarray(bf_mask, dtype=bool)
    provisional_df_mask = np.asarray(df_mask, dtype=bool)
    dp = detector_sum.astype(np.float32) / float(full_frame_count)
    center, radius = auto_probe(dp)
    bf_mask = detector_mask(center, 0.0, radius, dp.shape)
    df_mask = detector_mask(center, radius, np.inf, dp.shape)
    masks_identical = bool(
        np.array_equal(provisional_bf_mask, bf_mask)
        and np.array_equal(provisional_df_mask, df_mask)
    )

    fallback_s = 0.0
    fallback_load_s = 0.0
    fallback_reduce_s = 0.0
    pass_count = 1
    if not masks_identical:
        pass_count = 2
        fallback_t0 = time.perf_counter()
        for r0 in range(0, scan_shape[0], chunk_rows):
            r1 = min(scan_shape[0], r0 + chunk_rows)
            load_t0 = time.perf_counter()
            if r0 == 0 and r1 == scan_shape[0]:
                result = load(
                    str(master),
                    scan_shape=scan_shape,
                    backend="cuda",
                    dtype=output_dtype,
                    verbose=False,
                )
            else:
                result = load(
                    str(master),
                    scan_region=(r0, r1, 0, scan_shape[1]),
                    scan_shape=scan_shape,
                    backend="cuda",
                    dtype=output_dtype,
                    verbose=False,
                )
            cp.cuda.Stream.null.synchronize()
            fallback_load_s += time.perf_counter() - load_t0
            reduce_t0 = time.perf_counter()
            bf_gpu = cuda_masked_sum(result.data, bf_mask)
            df_gpu = cuda_masked_sum(result.data, df_mask)
            cp.cuda.Stream.null.synchronize()
            fallback_reduce_s += time.perf_counter() - reduce_t0
            bf_map[r0:r1, :] = cp.asnumpy(bf_gpu).reshape(
                r1 - r0,
                scan_shape[1],
            )
            df_map[r0:r1, :] = cp.asnumpy(df_gpu).reshape(
                r1 - r0,
                scan_shape[1],
            )
            del result, bf_gpu, df_gpu
            _clear_cuda_pools()
        fallback_s = time.perf_counter() - fallback_t0
        _require_source_unchanged(
            master,
            source_fingerprint,
            stage="after the authoritative CUDA BF/DF pass",
        )

    probe_source = "full_scan_exact"
    com_row -= float(com_row.mean())
    com_col -= float(com_col.mean())
    rotation_t0 = time.perf_counter()
    _, _, rotation_deg, transposed = find_optimal_rotation(
        com_row,
        com_col,
        rotation_steps=rotation_steps,
    )
    rotation_s = time.perf_counter() - rotation_t0
    elapsed_s = time.perf_counter() - t0

    timing = {
        "sample_load_s": float(sample_load_s),
        "sample_product_s": float(sample_product_s),
        "probe_source": probe_source,
        "bootstrap_source": bootstrap_source,
        "exact_sum_method": "CUDA uint64 chunk sums with fixed host merge",
        "full_frame_count": int(full_frame_count),
        "masks_identical": masks_identical,
        "pass_count": int(pass_count),
        "fallback_s": float(fallback_s),
        "fallback_load_s": float(fallback_load_s),
        "fallback_reduce_s": float(fallback_reduce_s),
        "stream_s": float(stream_s),
        "rotation_s": float(rotation_s),
        "elapsed_s": float(elapsed_s),
        "chunk_count": len(chunk_timings),
        "chunk_load_median_s": float(np.median([c["load_s"] for c in chunk_timings])),
        "chunk_reduce_median_s": float(np.median([c["reduce_s"] for c in chunk_timings])),
        "chunk_load_min_s": float(np.min([c["load_s"] for c in chunk_timings])),
        "chunk_load_max_s": float(np.max([c["load_s"] for c in chunk_timings])),
        "chunk_reduce_min_s": float(np.min([c["reduce_s"] for c in chunk_timings])),
        "chunk_reduce_max_s": float(np.max([c["reduce_s"] for c in chunk_timings])),
        "chunk_exact_sum_median_s": float(
            np.median([c["exact_sum_s"] for c in chunk_timings])
        ),
    }
    params = {
        "scan_shape": [int(v) for v in scan_shape],
        "detector_shape": [int(v) for v in detector_shape],
        "chunk_rows": int(chunk_rows),
        "sample_positions": int(sample_positions),
        "sample_positions_used": int(sample_positions_used),
        "probe_source": probe_source,
        "bootstrap_source": bootstrap_source,
        "full_frame_count": int(full_frame_count),
        "masks_identical": masks_identical,
        "pass_count": int(pass_count),
        "seed": None if seed is None else int(seed),
        "rotation_steps": int(rotation_steps),
        "center": [float(center[0]), float(center[1])],
        "radius_px": float(radius),
        "rotation_deg": float(rotation_deg),
        "transposed": bool(transposed),
        "backend": "cuda",
        "dtype": str(_screening_output_dtype(output_dtype)),
        "memory_budget_gb": float(memory_plan.memory_budget_gb),
        "memory_budget_source": memory_plan.memory_budget_source,
        "chunk_rows_source": memory_plan.chunk_rows_source,
        "chunk_count": int(memory_plan.chunk_count),
        "chunk_resident_gb": float(memory_plan.chunk_resident_gb),
    }
    return ScreeningResult(
        mean_dp=np.asarray(dp, dtype=np.float32),
        bright_field=bf_map,
        dark_field=df_map,
        dpc_phase=_dpc_phase(
            com_row,
            com_col,
            float(rotation_deg),
            bool(transposed),
        ),
        com_row=com_row.astype(np.float32, copy=False),
        com_col=com_col.astype(np.float32, copy=False),
        probe_center=(float(center[0]), float(center[1])),
        probe_radius=float(radius),
        rotation_deg=float(rotation_deg),
        transposed=bool(transposed),
        metadata={
            "version": _CACHE_VERSION,
            "source": source_fingerprint,
            "parameters": params,
            "timing": timing,
            "memory": asdict(memory_plan),
            "mode": "cached-screening-products",
            "note": "BF/DF/CoM/rotation products are derived from raw HDF5; raw HDF5 remains the ptychography evidence source.",
        },
        from_cache=False,
        elapsed_s=elapsed_s,
    )


def _build_mps_products(
    master: Path,
    *,
    scan_shape: tuple[int, int],
    chunk_rows: int,
    sample_positions: int,
    seed: int | None,
    rotation_steps: int,
    memory_plan: MemoryPlan,
    verbose: bool,
    skip_mps_memory_check: bool | None,
) -> ScreeningResult:
    from quantem.gpu.detector import auto_probe, detector_mask
    from quantem.gpu.dpc.workflow import find_optimal_rotation
    from quantem.gpu.io import inspect as inspect_source
    from quantem.gpu.io import load

    t0 = time.perf_counter()
    metadata = inspect_source(str(master)).metadata
    source_fingerprint = _source_fingerprint(master)
    detector_shape = tuple(int(v) for v in metadata.get("detector_shape") or ())
    if len(detector_shape) != 2:
        raise ValueError("Could not determine detector_shape from HDF5 metadata")

    bf_map = np.empty(scan_shape, dtype=np.float32)
    df_map = np.empty(scan_shape, dtype=np.float32)
    com_row = np.empty(scan_shape, dtype=np.float32)
    com_col = np.empty(scan_shape, dtype=np.float32)
    chunk_rows = int(max(1, min(chunk_rows, scan_shape[0])))
    full_scan_single_chunk = chunk_rows >= scan_shape[0]
    sample_load_s = 0.0
    sample_product_s = 0.0
    sample_positions_used = 0
    dp = None
    center = None
    radius = None
    bf_mask = None
    df_mask = None
    detector_sum = np.zeros(detector_shape, dtype=np.uint64)
    full_frame_count = 0
    bootstrap_source = "pending"

    if int(sample_positions) > 0 and not full_scan_single_chunk:
        sample_t0 = time.perf_counter()
        sample = load(
            str(master),
            random_positions=int(sample_positions),
            scan_shape=scan_shape,
            seed=seed,
            backend="mps",
            verbose=False,
        )
        sample_load_s = time.perf_counter() - sample_t0

        dp_t0 = time.perf_counter()
        sample_frames = _mps_chunked_frames_for(sample.data)
        dp = _mps_mean_dp(sample_frames)
        center, radius = auto_probe(dp)
        bf_mask = detector_mask(center, 0.0, radius, dp.shape)
        df_mask = detector_mask(center, radius, np.inf, dp.shape)
        sample_product_s = time.perf_counter() - dp_t0
        sample_positions_used = int(sample_positions)
        bootstrap_source = "random_scan_sample"
        del sample, sample_frames
        _clear_mps_transients()

    _require_source_unchanged(
        master,
        source_fingerprint,
        stage="before the primary MPS stream",
    )

    chunk_timings: list[dict[str, float]] = []
    stream_t0 = time.perf_counter()
    for r0 in range(0, scan_shape[0], chunk_rows):
        r1 = min(scan_shape[0], r0 + chunk_rows)
        load_t0 = time.perf_counter()
        if r0 == 0 and r1 == scan_shape[0]:
            result = load(
                str(master),
                scan_shape=scan_shape,
                backend="mps",
                verbose=False,
            )
        else:
            result = load(
                str(master),
                scan_region=(r0, r1, 0, scan_shape[1]),
                scan_shape=scan_shape,
                backend="mps",
                verbose=False,
            )
        load_s = time.perf_counter() - load_t0

        data = result.data
        frames = _mps_chunked_frames_for(data)
        if dp is None:
            product_t0 = time.perf_counter()
            dp = _mps_mean_dp(frames)
            center, radius = auto_probe(dp)
            bf_mask = detector_mask(center, 0.0, radius, dp.shape)
            df_mask = detector_mask(center, radius, np.inf, dp.shape)
            sample_product_s = time.perf_counter() - product_t0
            sample_positions_used = int(frames.vi.n)
            bootstrap_source = (
                "full_scan" if full_scan_single_chunk else "first_chunk"
            )
        exact_sum_t0 = time.perf_counter()
        np.add(detector_sum, frames.vi.detector_sum_exact(), out=detector_sum)
        full_frame_count += int(frames.vi.n)
        exact_sum_s = time.perf_counter() - exact_sum_t0
        reduce_t0 = time.perf_counter()
        if dp is None or center is None or radius is None or bf_mask is None or df_mask is None:
            raise RuntimeError("Screening masks were not initialized")
        bf_flat = np.asarray(frames.vi.masked_sum(bf_mask), dtype=np.float32).copy()
        df_flat = np.asarray(frames.vi.masked_sum(df_mask), dtype=np.float32).copy()
        ccol_flat, crow_flat = frames.vi.center_of_mass(None)
        reduce_s = time.perf_counter() - reduce_t0

        bf_map[r0:r1, :] = bf_flat.reshape(r1 - r0, scan_shape[1])
        df_map[r0:r1, :] = df_flat.reshape(r1 - r0, scan_shape[1])
        com_row[r0:r1, :] = np.asarray(crow_flat, dtype=np.float32).reshape(
            r1 - r0, scan_shape[1]
        )
        com_col[r0:r1, :] = np.asarray(ccol_flat, dtype=np.float32).reshape(
            r1 - r0, scan_shape[1]
        )
        chunk_timings.append(
            {
                "load_s": float(load_s),
                "exact_sum_s": float(exact_sum_s),
                "reduce_s": float(reduce_s),
                "resident_gb": float(getattr(data, "nbytes", 0) / 1e9),
            }
        )
        del result, data, frames, bf_flat, df_flat, ccol_flat, crow_flat
        _clear_mps_transients()
        if verbose:
            print(f"  rows {r0}:{r1} load={load_s:.3f}s reduce={reduce_s:.3f}s")

    stream_s = time.perf_counter() - stream_t0
    _require_source_unchanged(
        master,
        source_fingerprint,
        stage="after the primary MPS stream",
    )

    if full_frame_count != int(np.prod(scan_shape)):
        raise RuntimeError(
            "MPS screening visited "
            f"{full_frame_count} frames, expected {int(np.prod(scan_shape))}."
        )
    provisional_bf_mask = np.asarray(bf_mask, dtype=bool)
    provisional_df_mask = np.asarray(df_mask, dtype=bool)
    dp = detector_sum.astype(np.float32) / float(full_frame_count)
    center, radius = auto_probe(dp)
    bf_mask = detector_mask(center, 0.0, radius, dp.shape)
    df_mask = detector_mask(center, radius, np.inf, dp.shape)
    masks_identical = bool(
        np.array_equal(provisional_bf_mask, bf_mask)
        and np.array_equal(provisional_df_mask, df_mask)
    )

    fallback_s = 0.0
    fallback_load_s = 0.0
    fallback_reduce_s = 0.0
    pass_count = 1
    if not masks_identical:
        pass_count = 2
        fallback_t0 = time.perf_counter()
        for r0 in range(0, scan_shape[0], chunk_rows):
            r1 = min(scan_shape[0], r0 + chunk_rows)
            load_t0 = time.perf_counter()
            if r0 == 0 and r1 == scan_shape[0]:
                result = load(
                    str(master),
                    scan_shape=scan_shape,
                    backend="mps",
                    verbose=False,
                )
            else:
                result = load(
                    str(master),
                    scan_region=(r0, r1, 0, scan_shape[1]),
                    scan_shape=scan_shape,
                    backend="mps",
                    verbose=False,
                )
            fallback_load_s += time.perf_counter() - load_t0
            frames = _mps_chunked_frames_for(result.data)
            reduce_t0 = time.perf_counter()
            bf_flat = np.asarray(
                frames.vi.masked_sum(bf_mask),
                dtype=np.float32,
            ).copy()
            df_flat = np.asarray(
                frames.vi.masked_sum(df_mask),
                dtype=np.float32,
            ).copy()
            fallback_reduce_s += time.perf_counter() - reduce_t0
            bf_map[r0:r1, :] = bf_flat.reshape(
                r1 - r0,
                scan_shape[1],
            )
            df_map[r0:r1, :] = df_flat.reshape(
                r1 - r0,
                scan_shape[1],
            )
            del result, frames, bf_flat, df_flat
            _clear_mps_transients()
        fallback_s = time.perf_counter() - fallback_t0
        _require_source_unchanged(
            master,
            source_fingerprint,
            stage="after the authoritative MPS BF/DF pass",
        )

    probe_source = "full_scan_exact"
    com_row -= float(com_row.mean())
    com_col -= float(com_col.mean())
    rotation_t0 = time.perf_counter()
    _, _, rotation_deg, transposed = find_optimal_rotation(
        com_row,
        com_col,
        rotation_steps=rotation_steps,
    )
    rotation_s = time.perf_counter() - rotation_t0
    elapsed_s = time.perf_counter() - t0
    source_dtype = _metadata_dtype(metadata, np.uint16)

    timing = {
        "sample_load_s": float(sample_load_s),
        "sample_product_s": float(sample_product_s),
        "probe_source": probe_source,
        "bootstrap_source": bootstrap_source,
        "exact_sum_method": "Metal uint64 chunk sums with fixed host merge",
        "full_frame_count": int(full_frame_count),
        "masks_identical": masks_identical,
        "pass_count": int(pass_count),
        "fallback_s": float(fallback_s),
        "fallback_load_s": float(fallback_load_s),
        "fallback_reduce_s": float(fallback_reduce_s),
        "stream_s": float(stream_s),
        "rotation_s": float(rotation_s),
        "elapsed_s": float(elapsed_s),
        "chunk_count": len(chunk_timings),
        "chunk_load_median_s": float(np.median([c["load_s"] for c in chunk_timings])),
        "chunk_reduce_median_s": float(np.median([c["reduce_s"] for c in chunk_timings])),
        "chunk_load_min_s": float(np.min([c["load_s"] for c in chunk_timings])),
        "chunk_load_max_s": float(np.max([c["load_s"] for c in chunk_timings])),
        "chunk_reduce_min_s": float(np.min([c["reduce_s"] for c in chunk_timings])),
        "chunk_reduce_max_s": float(np.max([c["reduce_s"] for c in chunk_timings])),
        "chunk_exact_sum_median_s": float(
            np.median([c["exact_sum_s"] for c in chunk_timings])
        ),
    }
    params = {
        "scan_shape": [int(v) for v in scan_shape],
        "detector_shape": [int(v) for v in detector_shape],
        "chunk_rows": int(chunk_rows),
        "sample_positions": int(sample_positions),
        "sample_positions_used": int(sample_positions_used),
        "probe_source": probe_source,
        "bootstrap_source": bootstrap_source,
        "full_frame_count": int(full_frame_count),
        "masks_identical": masks_identical,
        "pass_count": int(pass_count),
        "seed": None if seed is None else int(seed),
        "rotation_steps": int(rotation_steps),
        "center": [float(center[0]), float(center[1])],
        "radius_px": float(radius),
        "rotation_deg": float(rotation_deg),
        "transposed": bool(transposed),
        "backend": "mps",
        "dtype": str(source_dtype),
        "memory_budget_gb": float(memory_plan.memory_budget_gb),
        "memory_budget_source": memory_plan.memory_budget_source,
        "chunk_rows_source": memory_plan.chunk_rows_source,
        "chunk_count": int(memory_plan.chunk_count),
        "chunk_resident_gb": float(memory_plan.chunk_resident_gb),
    }
    return ScreeningResult(
        mean_dp=np.asarray(dp, dtype=np.float32),
        bright_field=bf_map,
        dark_field=df_map,
        dpc_phase=_dpc_phase(
            com_row,
            com_col,
            float(rotation_deg),
            bool(transposed),
        ),
        com_row=com_row.astype(np.float32, copy=False),
        com_col=com_col.astype(np.float32, copy=False),
        probe_center=(float(center[0]), float(center[1])),
        probe_radius=float(radius),
        rotation_deg=float(rotation_deg),
        transposed=bool(transposed),
        metadata={
            "version": _CACHE_VERSION,
            "source": source_fingerprint,
            "parameters": params,
            "timing": timing,
            "memory": asdict(memory_plan),
            "mode": "cached-screening-products",
            "note": "BF/DF/CoM/rotation products are derived from raw HDF5; raw HDF5 remains the ptychography evidence source.",
        },
        from_cache=False,
        elapsed_s=elapsed_s,
    )


def prepare(
    source: str | Path,
    *,
    backend: str = "auto",
    scan_shape: tuple[int, int] | None = None,
    rotation_angle_deg: float | None = None,
    cache: bool = True,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    memory_budget_gb: float | None = None,
    chunk_rows: int | None = None,
    sample_positions: int = 0,
    seed: int | None = 0,
    rotation_steps: int = 90,
    dtype=np.uint16,
    verbose: bool = False,
) -> ScreeningResult:
    """Load or build cached BF/DF/CoM/rotation products for one HDF5 master.

    Cached products are intended for instant UI launch and ptychography
    calibration setup. A cache miss first streams raw HDF5 with GPU
    bitshuffle/LZ4 decode and backend-native reductions. If the bootstrap
    BF/DF masks differ from the exact full-scan masks, only BF/DF are streamed
    a second time; that fallback and its reason are recorded in metadata. CUDA
    uses the optimized RawKernel path, while MPS uses chunk-backed Metal
    reductions. The raw HDF5 master remains the evidence source for stochastic
    ptychography batches.
    """
    master_path = Path(source).expanduser()
    if not master_path.exists():
        raise FileNotFoundError(f"HDF5 master not found: {master_path}")

    cache_path = _cache_path(master_path, cache_dir)
    if cache and not refresh:
        products = _prepare_cache(cache_path, master_path)
        if products is not None:
            return _with_rotation(products, rotation_angle_deg)

    from quantem.gpu.io import inspect as inspect_source

    metadata = inspect_source(str(master_path)).metadata
    if scan_shape is None:
        scan_shape = tuple(int(v) for v in metadata.get("scan_shape") or ())
    if len(scan_shape) != 2:
        raise ValueError("scan_shape=(rows, cols) is required for screening products")
    detector_shape = tuple(int(v) for v in metadata.get("detector_shape") or ())
    if len(detector_shape) != 2:
        raise ValueError("Could not determine detector_shape from HDF5 metadata")
    if int(sample_positions) < 0:
        raise ValueError("sample_positions must be non-negative")

    from quantem.gpu import device

    resolved_backend = device.resolve(backend)
    if resolved_backend not in {"cuda", "mps"}:
        raise RuntimeError(
            "prepare cache generation requires backend='cuda' "
            f"or backend='mps'; backend={resolved_backend!r} was selected. "
            "Existing caches can still be read by CPU-facing callers."
        )
    plan_dtype = (
        _metadata_dtype(metadata, dtype)
        if resolved_backend == "mps"
        else _screening_output_dtype(dtype)
    )
    plan = _memory_plan_for_shapes(
        scan_shape,
        detector_shape,
        plan_dtype.itemsize,
        memory_budget_gb,
    )
    if chunk_rows is not None:
        plan = _memory_plan_with_chunk_rows(plan, int(chunk_rows))
    elif (
        resolved_backend == "cuda"
        and int(sample_positions) == 0
        and plan_dtype == np.dtype(np.uint16)
    ):
        from ._cuda import _exact_cuda_memory_plan

        plan = _exact_cuda_memory_plan(plan)
    if verbose:
        budget = f"{plan.memory_budget_gb:.1f} GB ({plan.memory_budget_source})"
        cuda = (
            ""
            if plan.cuda_free_gb is None
            else f", cuda_free={plan.cuda_free_gb:.1f}/{plan.cuda_total_gb:.1f} GB"
        )
        print(
            "Screening memory plan: "
            f"budget={budget}, chunk_rows={plan.chunk_rows}, "
            f"chunks={plan.chunk_count}, raw_chunk={plan.chunk_resident_gb:.2f} GB"
            f"{cuda}"
        )

    if resolved_backend == "cuda":
        products = _build_cuda_products(
            master_path,
            scan_shape=scan_shape,
            chunk_rows=int(plan.chunk_rows),
            sample_positions=int(sample_positions),
            seed=seed,
            rotation_steps=int(rotation_steps),
            output_dtype=dtype,
            memory_plan=plan,
            verbose=verbose,
        )
    else:
        products = _build_mps_products(
            master_path,
            scan_shape=scan_shape,
            chunk_rows=int(plan.chunk_rows),
            sample_positions=int(sample_positions),
            seed=seed,
            rotation_steps=int(rotation_steps),
            memory_plan=plan,
            verbose=verbose,
            skip_mps_memory_check=False,
        )
    products.cache_path = cache_path if cache else None
    if cache:
        _save_cache(products, cache_path)
    return _with_rotation(products, rotation_angle_deg)


def _with_rotation(
    result: ScreeningResult,
    rotation_angle_deg: float | None,
) -> ScreeningResult:
    """Apply an explicit DPC rotation without touching the raw evidence."""

    if rotation_angle_deg is None:
        return result
    result.rotation_deg = float(rotation_angle_deg)
    result.transposed = False
    result.dpc_phase = _dpc_phase(
        result.com_row,
        result.com_col,
        result.rotation_deg,
        False,
    )
    return result
