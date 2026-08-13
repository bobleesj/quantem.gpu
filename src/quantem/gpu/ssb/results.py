"""Backend-neutral SSB fit, source, evaluation, and reconstruction results."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
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
    reused : bool
        Whether the result was restored from an exact saved-result match.
    saved_path : pathlib.Path | None
        Array artifact used for persistence, when ``save_to`` was provided.
    metadata : dict[str, object]
        Complete saved scientific signature, provenance, and result metadata.
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
    reused: bool = False
    saved_path: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)

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
        if self.saved_path is not None:
            lines.append(
                f"  Saved result   {self.saved_path}"
                + (" (reused)" if self.reused else "")
            )
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


@dataclass
class SSBSeriesResult:
    """One independent or fixed-probe SSB reconstruction series."""

    phase: np.ndarray
    bright_field: np.ndarray
    dark_field: np.ndarray
    frames: tuple[int, ...]
    datasets: tuple[str, ...]
    master_names: tuple[str, ...]
    probe_reference_frame: int | None
    probe_reference_dataset: str | None
    records: tuple[dict[str, object], ...]
    source_directory: Path
    results_directory: Path
    requested_backend: str
    trials: int
    refinement: str | None

    @property
    def alignment(self) -> dict[str, float | str]:
        """Validated registration preparation for an SSB phase series."""
        return {
            "normalization": "median_mad",
            "pad_fraction": 3.0 / 32.0,
            "upsample_factor": 50,
            "running_avg_frames": 12.0,
        }

    def show(self, **kwargs: object):
        """Return the native-resolution SSB series in Show3D."""
        from quantem.widget import Show3D

        labels = [
            f"F{frame} | {dataset}"
            for frame, dataset in zip(self.frames, self.datasets, strict=True)
        ]
        panel_title = "Independent SSB fit"
        if self.probe_reference_frame is not None:
            panel_title = (
                f"Fixed probe from F{self.probe_reference_frame} | "
                f"{self.probe_reference_dataset}"
            )
        show_kwargs = {
            "labels": labels,
            "panel_titles": (panel_title, "Bright field", "Dark field"),
            "hidden_panels": ("Bright field", "Dark field"),
            "title": "SSB reconstruction series",
            "display_bin": 1,
            "cmap": ("magma", "gray", "gray"),
            "offline": False,
        }
        show_kwargs.update(kwargs)
        return Show3D(
            self.phase,
            self.bright_field,
            self.dark_field,
            **show_kwargs,
        )

    def metrics(self):
        """Return one readable row per SSB acquisition."""
        import pandas as pd

        return (
            pd.DataFrame(self.records)
            .style.format(
                {
                    "C10 (nm)": "{:.3f}",
                    "C12 (nm)": "{:.3f}",
                    "phi12 (rad)": "{:.5f}",
                    "loss": "{:.7f}",
                }
            )
            .hide(axis="index")
        )

    def metadata(self):
        """Return compact source and reconstruction metadata as a readable table."""
        import pandas as pd

        reused = sum(record["result"] != "computed" for record in self.records)
        fixed_probe = self.probe_reference_frame is not None
        mode = "fixed probe" if fixed_probe else "independent fits"
        rows = [
            ("Source directory", str(self.source_directory)),
            ("Saved results", str(self.results_directory)),
            ("First frame", self.frames[0]),
            ("Last frame", self.frames[-1]),
            ("Frame count", len(self.frames)),
            ("First dataset", self.datasets[0]),
            ("Last dataset", self.datasets[-1]),
            ("First raw master", self.master_names[0]),
            ("Last raw master", self.master_names[-1]),
            ("Probe mode", mode),
        ]
        if fixed_probe:
            rows.extend(
                [
                    ("Probe reference frame", self.probe_reference_frame),
                    ("Probe reference dataset", self.probe_reference_dataset),
                ]
            )
        rows.extend(
            [
                ("Shape", " x ".join(str(value) for value in self.phase.shape)),
                ("Requested backend", self.requested_backend),
                ("Trials", self.trials),
                ("Refinement", self.refinement or "none"),
                ("Results reused", reused),
            ]
        )
        return pd.DataFrame(rows, columns=("Setting", "Value")).style.hide(
            axis="index"
        )
