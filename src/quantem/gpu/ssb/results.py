"""Backend-neutral SSB fit, source, evaluation, and reconstruction results."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # MPS/WebGPU clients do not install CuPy.
    cp = None


def _is_cupy_array(value: object) -> bool:
    """Return whether *value* is a CuPy array without requiring CUDA."""

    return cp is not None and isinstance(value, cp.ndarray)


# =========================================================================
#  Aberration formatting
# =========================================================================

def _format_aberrations(aberrations: dict) -> str:
    """Format SSB aberration dict as aligned key-value lines."""
    if not aberrations:
        return "  (none)"
    lines = []
    if "C10" in aberrations:
        lines.append(f"  Defocus (C10)  {aberrations['C10']:.1f} nm")
    if "C12" in aberrations:
        lines.append(f"  Astigmatism    {aberrations['C12']:.1f} nm")
    if "phi12" in aberrations:
        lines.append(f"  Astig. angle   {math.degrees(aberrations['phi12']):.1f}°")
    return "\n".join(lines)


@dataclass
class SSBResult:
    """Result from SSB ptychographic reconstruction.

    The primary output is ``object_wave``, the complex transmission function.
    Convenience properties ``phase`` and ``amplitude`` are derived from it.

    Attributes
    ----------
    object_wave : cp.ndarray
        Complex transmission function (scan_row, scan_col).
    aberrations : dict[str, float]
        Aberration coefficients ``{C10, C12, phi12}`` in nm / radians.
    rotation_angle_deg : float
        Rotation angle in degrees.
    loss : float | None
        Variance loss value.
    elapsed : float | None
        Wall-clock time in seconds.
    timings : dict[str, float]
        Backend stage timings in seconds when the operation records them.
    """
    object_wave: object
    backend: Literal["cuda", "mps", "webgpu"]
    aberrations: dict[str, float] = field(default_factory=dict)
    rotation_angle_deg: float = 0.0
    loss: float | None = None
    elapsed: float | None = None
    timings: dict[str, float] = field(default_factory=dict)
    n_trials: int | None = None
    num_bf: int | None = None
    refine_method: str | None = None
    refine_nfev: int | None = None
    refine_elapsed: float | None = None
    voltage_kV: float | None = None
    semiangle_mrad: float | None = None
    scan_sampling_A: float | tuple[float, float] | None = None
    source_path: str | None = None
    bf_center: tuple[float, float] | None = None
    bf_radius: float | None = None
    detected_bf_radius: float | None = None
    # Full Optuna trial history, one entry per evaluated trial, in order.
    # Each entry: ``{"params": {"C10_nm", "C12_nm", "phi12_deg"}, "loss"}``.
    # Used by the Screening dashboard (#26) to plot the loss landscape.
    optuna_trials: list[dict] | None = None

    def __repr__(self) -> str:
        lines = ["SSB Result"]
        lines.append(f"  Shape          {tuple(self.object_wave.shape)}")
        if self.loss is not None:
            lines.append(f"  Loss           {self.loss:.6f}")
        if self.num_bf is not None:
            lines.append(f"  BF pixels      {self.num_bf}")
        if self.n_trials is not None:
            lines.append(f"  Trials         {self.n_trials}")
        lines.append(f"  Rotation       {self.rotation_angle_deg:.1f}°")
        if self.aberrations:
            lines.append(_format_aberrations(self.aberrations))
        if self.elapsed is not None:
            lines.append(f"  Time           {self.elapsed:.2f}s")
        return "\n".join(lines)

    @property
    def phase(self):
        """Phase of the complex transmission function: ``angle(object_wave)``."""
        if _is_cupy_array(self.object_wave):
            return cp.angle(self.object_wave)
        return np.angle(self.object_wave)

    @property
    def amplitude(self):
        """Amplitude of the complex transmission function: ``abs(object_wave)``."""
        if _is_cupy_array(self.object_wave):
            return cp.abs(self.object_wave)
        return np.abs(self.object_wave)
