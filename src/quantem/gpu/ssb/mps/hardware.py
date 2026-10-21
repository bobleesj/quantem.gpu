"""The Apple GPU side of MPS SSB: the MLX import, the machine's memory and chip, and the measured chunk sizes.

Chunk sizes and pair-packing policies were measured per memory class and chip (M5 Max, 24-128 GB). They set how many
bright-field planes a Metal dispatch handles. The phase and loss chunk also sets where the float32 bright-field sums
are split (``reconstruct.py`` adds the chunks in order), so two Macs of different memory classes can differ in the
last bits of a phase or loss.
"""

import subprocess
from functools import lru_cache


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

    scan_rows, scan_cols = (max(1, int(scan_shape[0])), max(1, int(scan_shape[1])))
    if max(scan_rows, scan_cols) <= 256:
        if total >= 96 * 1024**3:
            return 16384
        if total >= 64 * 1024**3:
            return 8192
        return 4096
    if max(scan_rows, scan_cols) >= 1024:
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
