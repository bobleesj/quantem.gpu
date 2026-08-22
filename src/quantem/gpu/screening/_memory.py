"""Resource planning for streamed 4D-STEM screening products."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MemoryPlan:
    """Memory plan for streaming BF/DF/CoM/rotation cache generation."""

    memory_budget_gb: float
    memory_budget_source: str
    raw_target_gb: float
    chunk_rows: int
    chunk_rows_source: str
    chunk_count: int
    chunk_resident_gb: float
    scan_shape: tuple[int, int]
    detector_shape: tuple[int, int]
    dtype: str
    cuda_free_gb: float | None = None
    cuda_total_gb: float | None = None


def _cuda_memory_info_gb() -> tuple[float, float] | None:
    """Return free and total CUDA memory in decimal gigabytes."""
    try:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return float(free_bytes) / 1e9, float(total_bytes) / 1e9


def _resolve_memory_budget_gb(
    memory_budget_gb: float | None,
) -> tuple[float, str, float | None, float | None]:
    """Resolve a user or device-derived streaming memory budget."""
    if memory_budget_gb is not None:
        budget = float(memory_budget_gb)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError(
                f"memory_budget_gb must be positive, got {memory_budget_gb!r}"
            )
        return budget, "user", None, None

    info = _cuda_memory_info_gb()
    if info is None:
        return 12.0, "fallback", None, None
    free_gb, total_gb = info
    # This is a working-set budget, not permission to consume all free VRAM.
    budget = max(4.0, free_gb * 0.90)
    return budget, "auto_cuda", free_gb, total_gb


def _memory_plan_for_shapes(
    scan_shape: tuple[int, int],
    detector_shape: tuple[int, int],
    itemsize: int,
    memory_budget_gb: float | None,
) -> MemoryPlan:
    """Choose scan-row chunks and report the effective memory budget."""
    scan_cols = int(scan_shape[1])
    bytes_per_row = (
        scan_cols
        * int(detector_shape[0])
        * int(detector_shape[1])
        * int(itemsize)
    )
    budget_gb, source, free_gb, total_gb = _resolve_memory_budget_gb(
        memory_budget_gb
    )
    budget_bytes = budget_gb * (1 << 30)
    full_raw_bytes = int(scan_shape[0]) * bytes_per_row
    if full_raw_bytes <= budget_bytes * 0.90:
        target_raw_bytes = full_raw_bytes
    else:
        target_raw_bytes = max(512 * 1024**2, budget_bytes * 0.50)
    rows = max(1, int(target_raw_bytes // max(1, bytes_per_row)))
    chunk_rows = max(1, min(int(scan_shape[0]), rows))
    chunk_count = int(math.ceil(int(scan_shape[0]) / chunk_rows))
    chunk_resident_gb = float(chunk_rows * bytes_per_row) / 1e9
    return MemoryPlan(
        memory_budget_gb=float(budget_gb),
        memory_budget_source=source,
        raw_target_gb=float(target_raw_bytes) / 1e9,
        chunk_rows=int(chunk_rows),
        chunk_rows_source="budget",
        chunk_count=chunk_count,
        chunk_resident_gb=chunk_resident_gb,
        scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
        detector_shape=(int(detector_shape[0]), int(detector_shape[1])),
        dtype=str(np.dtype(f"u{itemsize}")),
        cuda_free_gb=free_gb,
        cuda_total_gb=total_gb,
    )


def _memory_plan_with_chunk_rows(
    plan: MemoryPlan,
    chunk_rows: int,
) -> MemoryPlan:
    """Return ``plan`` with an explicit user chunk-row override applied."""
    rows = int(max(1, min(int(chunk_rows), plan.scan_shape[0])))
    itemsize = np.dtype(plan.dtype).itemsize
    bytes_per_row = (
        int(plan.scan_shape[1])
        * int(plan.detector_shape[0])
        * int(plan.detector_shape[1])
        * int(itemsize)
    )
    return MemoryPlan(
        memory_budget_gb=plan.memory_budget_gb,
        memory_budget_source=plan.memory_budget_source,
        raw_target_gb=plan.raw_target_gb,
        chunk_rows=rows,
        chunk_rows_source="user",
        chunk_count=int(math.ceil(int(plan.scan_shape[0]) / rows)),
        chunk_resident_gb=float(rows * bytes_per_row) / 1e9,
        scan_shape=plan.scan_shape,
        detector_shape=plan.detector_shape,
        dtype=plan.dtype,
        cuda_free_gb=plan.cuda_free_gb,
        cuda_total_gb=plan.cuda_total_gb,
    )
