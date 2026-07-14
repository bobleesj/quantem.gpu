"""
Result dataclasses for all reconstructions.

- :class:`DPCResult` - returned by :func:`quantem.gpu.ssb.dpc`.
- :class:`SSBResult` - returned by :func:`quantem.gpu.ssb` and each
  element of the list returned by :func:`quantem.gpu.ssb_series`.

Both classes expose ``.save(path)`` and ``.show()`` for persistence and
quick visualization. Complex fields (object wave, CoM components) remain
on CPU-side numpy arrays after construction so they survive GPU memory
cleanup.

``DefocusSweepResult`` and ``BFRadiusSweepResult`` are internal-only
containers used by the screening CLI and bf radius sweep diagnostics.
They are not part of the public API.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import cupy as cp
import numpy as np


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
class DPCResult:
    """Result from DPC reconstruction.

    Attributes
    ----------
    phase : cp.ndarray
        Reconstructed phase image (scan_row, scan_col).
    com_k_row_aligned : cp.ndarray
        Aligned center-of-mass in k_row direction.
    com_k_col_aligned : cp.ndarray
        Aligned center-of-mass in k_col direction.
    rotation_angle_deg : float
        Optimal rotation angle in degrees.
    elapsed : float | None
        Wall-clock time in seconds.
    """
    phase: cp.ndarray
    com_k_row_aligned: cp.ndarray
    com_k_col_aligned: cp.ndarray
    rotation_angle_deg: float
    elapsed: float | None = None
    # Curl-vs-angle profile from the rotation auto-fit (always present after
    # preprocess). Used by the screening dashboard to visualize how sharp the
    # rotation minimum was - flat curve means an unreliable fit and the user
    # should consider forcing a known rotation.
    curl_angles_deg: cp.ndarray | None = None
    curl_normal: cp.ndarray | None = None
    curl_transpose: cp.ndarray | None = None
    # When ``rotation_angle_deg`` was forced via the API arg, the auto-fit
    # phase + CoM are also captured here so the dashboard can display both
    # side-by-side. None when no forced rotation was applied.
    autofit_phase: cp.ndarray | None = None
    autofit_com_k_row_aligned: cp.ndarray | None = None
    autofit_com_k_col_aligned: cp.ndarray | None = None
    autofit_rotation_angle_deg: float | None = None

    def __repr__(self) -> str:
        lines = ["DPC Result"]
        lines.append(f"  Phase shape    {tuple(self.phase.shape)}")
        lines.append(f"  Rotation       {self.rotation_angle_deg:.1f}°")
        if self.elapsed is not None:
            lines.append(f"  Time           {self.elapsed:.2f}s")
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
    """
    object_wave: cp.ndarray
    aberrations: dict[str, float] = field(default_factory=dict)
    rotation_angle_deg: float = 0.0
    loss: float | None = None
    elapsed: float | None = None
    n_trials: int | None = None
    num_bf: int | None = None
    refine_method: str | None = None
    refine_nfev: int | None = None
    refine_elapsed: float | None = None
    voltage_kV: float | None = None
    semiangle_mrad: float | None = None
    scan_sampling_A: float | None = None
    source_path: str | None = None
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
        if isinstance(self.object_wave, cp.ndarray):
            return cp.angle(self.object_wave)
        return np.angle(self.object_wave)

    @property
    def amplitude(self):
        """Amplitude of the complex transmission function: ``abs(object_wave)``."""
        if isinstance(self.object_wave, cp.ndarray):
            return cp.abs(self.object_wave)
        return np.abs(self.object_wave)

    def flip_phase(self) -> None:
        """Conjugate the object wave (flip phase sign) in place."""
        if isinstance(self.object_wave, cp.ndarray):
            self.object_wave = cp.conj(self.object_wave)
        else:
            self.object_wave = np.conj(self.object_wave)

    def save_png(self, path: str) -> None:
        """Save phase image as auto-contrasted grayscale PNG with JSON sidecar.

        The JSON sidecar (same stem, ``.json`` extension) records aberrations,
        loss, num_bf, and rotation_angle_deg for reproducibility.

        Parameters
        ----------
        path : str
            Output PNG file path.
        """
        import json
        from pathlib import Path
        from PIL import Image
        obj = self.object_wave
        if hasattr(obj, "get"):
            obj = obj.get()
        phase = np.angle(np.asarray(obj)).astype(np.float32)
        lo, hi = float(phase.min()), float(phase.max())
        if hi - lo > 0:
            scaled = ((phase - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(phase, dtype=np.uint8)
        img = Image.fromarray(scaled, mode="L")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out))
        sidecar = {
            "aberrations": {k: float(v) for k, v in self.aberrations.items()},
            "loss": float(self.loss) if self.loss is not None else None,
            "num_bf": self.num_bf,
            "rotation_angle_deg": self.rotation_angle_deg,
            "phase_min": lo,
            "phase_max": hi,
        }
        json_path = out.with_suffix(".json")
        # Atomic write: tmp + rename. Sidecar lives next to a PNG that's
        # written above; a crash between the two left a half-PNG already,
        # but at least the JSON itself stays atomic.
        tmp = json_path.with_suffix(json_path.suffix + ".tmp")
        tmp.write_text(json.dumps(sidecar, indent=2))
        tmp.replace(json_path)

    def save(self, output_dir, *, source_path=None,
             dpc_result=None, bf_image=None, df_image=None, **kwargs):
        """Save result to disk (result.json + .npy files).

        Parameters
        ----------
        output_dir : str or Path
            Directory for this scan's results.
        source_path : str, optional
            Path to the source H5 file.
        dpc_result : DPCResult, optional
            DPC result to save alongside.
        bf_image, df_image : ndarray, optional
            Virtual images to save.
        """
        raise RuntimeError(
            "SSBResult.save() is a dashboard persistence helper and is not "
            "part of quantem.gpu. Use quantem.gpu result persistence for "
            "screening/dashboard artifacts."
        )

    def save_calibration(
        self,
        path: str | Path,
        *,
        flip_phase: bool = False,
        source_file: str | None = None,
    ) -> Path:
        """Save rotation + aberrations as a calibration JSON.

        Use this after running full optimization on one representative file.
        The saved calibration can be loaded with ``load_calibration()`` and
        applied to all subsequent files without re-running Optuna.

        Parameters
        ----------
        path : str or Path
            Output JSON file path.
        flip_phase : bool
            Whether phase was flipped for this dataset.
        source_file : str, optional
            Which file was used for calibration.

        Returns
        -------
        Path
            The written file path.
        """
        raise RuntimeError(
            "SSBResult.save_calibration() is a live calibration persistence "
            "helper and is not part of quantem.gpu yet."
        )

    def to_dashboard(self, *, label: str, dpc_phase=None,
                     bf=None, df=None) -> dict:
        """Package result for ``Live.on_file_complete()``.

        Parameters
        ----------
        label : str
            Scan label (e.g. file stem).
        dpc_phase : ndarray, optional
            DPC phase image (numpy, 2D).
        bf, df : ndarray, optional
            Virtual images (numpy, 2D).

        Returns
        -------
        dict
            Ready to unpack: ``dash.on_file_complete(idx=i, **result.to_dashboard(...))``.
        """
        phase = self.phase
        images = {'SSB': cp.asnumpy(phase) if isinstance(phase, cp.ndarray) else np.asarray(phase)}
        if dpc_phase is not None:
            images['DPC'] = np.asarray(dpc_phase)
        if bf is not None:
            images['BF'] = np.asarray(bf)
        if df is not None:
            images['DF'] = np.asarray(df)
        return {
            'label': label,
            'images': images,
            'loss': self.loss if self.loss is not None else -1.0,
            'aberrations': self.aberrations,
            'elapsed_s': self.elapsed or 0.0,
        }


@dataclass
class DefocusSweepResult:
    """Result from a defocus (C10) sweep.

    Attributes
    ----------
    c10_values_nm : np.ndarray
        Defocus values swept, in nm.
    losses : np.ndarray
        Variance loss at each defocus value.
    images : np.ndarray
        Phase images at each defocus, shape ``(n_steps, scan_row, scan_col)``.
    best_c10_nm : float
        Defocus with the lowest loss.
    best_loss : float
        Lowest loss value found.
    elapsed : float | None
        Wall-clock time in seconds.
    """
    c10_values_nm: np.ndarray
    losses: np.ndarray
    images: np.ndarray
    best_c10_nm: float
    best_loss: float
    elapsed: float | None = None
    labels: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = ["Defocus Sweep"]
        lines.append(
            f"  Best C10       {self.best_c10_nm:.1f} nm (loss={self.best_loss:.6f})"
        )
        c_min = self.c10_values_nm.min()
        c_max = self.c10_values_nm.max()
        n_steps = len(self.c10_values_nm)
        lines.append(f"  Range          {c_min:.1f} to {c_max:.1f} nm ({n_steps} steps)")
        if self.elapsed is not None:
            lines.append(f"  Time           {self.elapsed:.2f}s")
        return "\n".join(lines)


@dataclass
class BFRadiusSweepResult:
    """Result from a BF radius sweep with full optimization at each radius.

    Each radius gets its own SSB engine with independent aberration optimization
    (Optuna + grid search). Larger radii include higher-angle scattering and can
    resolve finer features, but need more accurate aberration correction.

    Attributes
    ----------
    radii : list[int]
        BF radii swept, in pixels.
    results : dict[int, SSBResult]
        Full SSBResult at each radius (keyed by radius).
    images : np.ndarray
        Phase images at each radius, shape ``(n_radii, scan_row, scan_col)``.
    losses : np.ndarray
        Best loss at each radius.
    best_radius : int
        Radius with the lowest loss.
    best_loss : float
        Lowest loss value found.
    elapsed : float | None
        Total wall-clock time in seconds.
    labels : list[str]
        Display labels for each radius (for ``show_2d``).
    """
    radii: list[int]
    results: dict[int, SSBResult]
    images: np.ndarray
    losses: np.ndarray
    best_radius: int
    best_loss: float
    elapsed: float | None = None
    labels: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = ["BF Radius Sweep"]
        lines.append(f"  Best radius    {self.best_radius} px (loss={self.best_loss:.6f})")
        if self.elapsed is not None:
            lines.append(f"  Time           {self.elapsed:.1f}s")
        for i, r in enumerate(self.radii):
            res = self.results[r]
            a = res.aberrations
            marker = " *" if r == self.best_radius else ""
            lines.append(
                f"  r={r:>3}  C10={a['C10']:.1f} nm  "
                f"C12={a['C12']:.1f} nm  loss={self.losses[i]:.6f}{marker}"
            )
        return "\n".join(lines)
