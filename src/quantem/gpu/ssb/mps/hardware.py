"""The Apple GPU side of MPS SSB: the MLX import, the machine's memory and chip, and the measured chunk sizes.

Chunk sizes and pair-packing policies were measured per memory class and chip (M5 Max, 24-128 GB). They change only
how many bright-field planes a Metal dispatch handles, never the reduction boundaries or the arithmetic.
"""

import subprocess
from functools import lru_cache

# the 512 pair-pack policy where no machine-specific measurement applies
_EXACT_ROW_PACK_STORAGE_BF_512 = 300
_EXACT_ROW_STORAGE_CLASSES_BF_512 = (288, 320)


def require_mlx():
    """Return ``mlx.core``, imported on first use so the package imports on machines without MLX."""
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MPS SSB preview requires MLX on Apple Silicon. Install with "
            "`python -m pip install mlx` in the Mac environment."
        ) from exc
    return mx


@lru_cache(maxsize=1)
def default_object_setup_chunk_bf() -> int:
    """BF chunk size for first-use object-mode MPS setup."""
    total = _physical_memory_bytes()
    if total is None:
        return 256
    if total >= 96 * 1024**3:
        return 1024
    if total >= 48 * 1024**3:
        return 512
    return 256


def default_object_redraw_chunk_bf() -> int:
    """BF chunk size for repeated object-mode MPS redraws."""
    return 128


@lru_cache(maxsize=1)
def default_object_redraw_threadgroup(
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """Metal threadgroup size for repeated object-mode MPS redraws."""
    if scan_shape is not None and max(int(scan_shape[0]), int(scan_shape[1])) >= 512:
        return 64
    return 16


def effective_phase_loss_chunk_bf(
    chunk_bf: int,
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """Use a faster full phase/loss chunk unless the caller retuned it."""
    requested = max(1, int(chunk_bf))
    if requested == 16:
        return _default_phase_loss_chunk_bf(scan_shape)
    return requested


@lru_cache(maxsize=1)
def _default_phase_loss_chunk_bf(
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """BF chunk size for full phase/loss reconstruction on MPS."""
    total = _physical_memory_bytes()
    if total is None:
        return 512
    if total >= 96 * 1024**3:
        chunk = 4096
    elif total >= 64 * 1024**3:
        chunk = 1024
    else:
        chunk = 512

    if scan_shape is None:
        return chunk

    ny, nx = (max(1, int(scan_shape[0])), max(1, int(scan_shape[1])))
    if max(ny, nx) <= 256:
        if total >= 96 * 1024**3:
            return 16384
        if total >= 64 * 1024**3:
            return 8192
        return 4096
    if max(ny, nx) >= 1024:
        # Full-BF 1024 phase/loss on MLX/Metal hits a scheduling and
        # allocation cliff at very large chunks. After scalar-loss reduction,
        # 512 BF is the best measured default on a 96 GB-class Apple GPU.
        return min(chunk, 512)
    return chunk


@lru_cache(maxsize=1)
def default_phase_col_k_bf(
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """BF grouping for fused Metal column phase/loss accumulation."""
    if scan_shape == (512, 512):
        # The 512 radix-8 kernel reaches its best occupancy with eight columns
        # per threadgroup. Keeping the whole BF chunk in each group avoids the
        # partial-image traffic and follow-up MLX reduction without reducing
        # the exact BF evidence.
        return 4096
    return 32


@lru_cache(maxsize=2)
def exact_pair_row_policy_512(batch: int = 2) -> tuple[int, tuple[int, ...]]:
    """Return the measured 512 pair-pack policy for this hardware class."""

    total, chip = _apple_hardware_profile()
    if chip == "Apple M5 Max" and int(batch) == 2:
        if total >= 120 * 1024**3:
            # The measured 128 GB M5 Max is fastest when one 2,496-plane allocation
            # can hold the fixture's 2,476 compact planes. Every original
            # 512-logical-BF reduction range remains separate in the column
            # output, so launch consolidation does not alter summation order.
            return 2476, (2496,)
        if total >= 96 * 1024**3:
            # Lower-memory M5 Max systems retain the measured four-pack policy.
            # A 716-plane greedy limit uses one stable 720-plane class.
            return 716, (720,)
    return _EXACT_ROW_PACK_STORAGE_BF_512, _EXACT_ROW_STORAGE_CLASSES_BF_512


def exact_pair_row_allocation_bf_512(
    pack_bf: int,
    storage_classes: tuple[int, ...] | None = None,
) -> int:
    """Return one exact pair pack's row allocation in BF planes.

    The retained storage classes only exist so repeated packs reuse one
    allocation shape. A single logical pair boundary is never split, so a
    pack can legitimately be wider than every retained class on hardware
    whose measured policy was tuned for narrower boundaries. In that case the
    pack still runs at its own width: the row kernel allocates
    ``max(pack, class)`` and only the written prefix is consumed, so falling
    back to the pack width changes allocation size alone, never the BF
    boundaries, the reduction order, or the arithmetic.

    Parameters
    ----------
    pack_bf : int
        BF planes actually written by one packed pair slice.
    storage_classes : tuple of int, optional
        Retained allocation classes; defaults to the measured 512 policy.

    Returns
    -------
    int
        BF planes to allocate for the packed row intermediate.
    """
    pack_bf = int(pack_bf)
    if storage_classes is None:
        _pack_limit, storage_classes = exact_pair_row_policy_512()
    for storage_class in storage_classes:
        if pack_bf <= storage_class:
            return storage_class
    return pack_bf


def use_simd_radix8_col_stage_512() -> bool:
    """Use the exact register transpose on the measured M5 Max target."""

    _total, chip = _apple_hardware_profile()
    return chip == "Apple M5 Max"


@lru_cache(maxsize=1)
def _apple_hardware_profile() -> tuple[int, str]:
    """Return total memory and Apple chip name without repeated subprocesses."""
    total = _physical_memory_bytes()
    if total is None:
        total = 0
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        chip = ""
    return total, chip


def _physical_memory_bytes() -> int | None:
    """Return total unified memory from ``sysctl hw.memsize``, or None when it cannot be read.

    MPS chunk defaults scale with the memory of the Apple GPU class.
    """
    try:
        return int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
