"""Settle the 180-degree branch of an estimated scan-detector rotation from two independent SSB fits.

The centre-of-mass curl fixes the scan-detector rotation only modulo 180 degrees: the angle and the angle plus 180
degrees reverse every CoM vector and both leave the field curl-free. SSB fits both angles independently, with the same
settings, and reconstructs each fit's phase at the native scan sampling. The two phases are close to negatives of each
other, so one decision chooses the branch:

1. Atomic columns are located on the dark-field image of the same scan. Dark field is incoherent, so the column
   positions are the same on both branches. Atoms push the electron phase forward, so on the right branch the phase
   at the columns is positive. A branch is eligible only when its positive mean column phase is at least
   ``COLUMN_Z_MIN`` standard errors from zero and at least ``COLUMN_EFFECT_MIN`` of its spread (so many columns cannot
   make a negligible shift decisive); the eligible branch with the more positive columns is kept. Negative
   evidence on one independent fit does not establish positive columns on the other fit.
2. Otherwise the phase tail decides: the branch with the larger mean-centred third-moment ratio ``phase_skewness``
   (a few bright columns over a flatter background) is kept.

The fits are the session's default tilt-corrected joint fits: correcting the sample tilt and depth spread makes the
atomic-column contrast clearer, while the two branches still fit equally well.

Both tests read the native phases. What ``RotationBranches.show`` draws of each branch is its fit reconstructed at
``upsampling="auto"`` (``RotationBranches.upsampled_phases``), with the columns at scan pixels times the factor.
"""

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from quantem.gpu.direct_ptycho.results import host_array, phase_skewness

# Dark field from just outside the bright-field disk to the detector edge, in units of the bright-field radius.
DARK_FIELD_INNER_RADIUS = 1.1
# Column detection on the dark-field image: a 1-pixel gaussian against shot noise, a high pass that subtracts a
# gaussian of COLUMN_HIGH_PASS_A (thickness and support-film variations are slower than a column spacing), then the
# local maxima at least COLUMN_SEPARATION_A apart (a lattice-scale window, so noise does not give thousands of maxima
# at a coarse scan step), keeping the COLUMN_TOP_FRACTION with the largest prominence (height above the window's
# minimum). The lengths are converted with the scan step: an 8-pixel high pass and a 17 x 17 window at 0.264 Å.
COLUMN_SMOOTHING_PX = 1.0
COLUMN_HIGH_PASS_A = 2.1
COLUMN_SEPARATION_A = 2.0
COLUMN_TOP_FRACTION = 0.2
# The column test decides only when the mean column phase is COLUMN_Z_MIN standard errors from zero and its effect,
# mean / std, reaches COLUMN_EFFECT_MIN. With tilt-corrected fits a 512 x 512 scan at 0.264 Å with resolved columns
# gave an effect of 0.70 (z 10.8), and scans at 0.5 Å without resolved columns 0.08 and 0.002.
COLUMN_Z_MIN = 3.0
COLUMN_EFFECT_MIN = 0.25

KEPT_COLOR = "#29243e"
OTHER_COLOR = "#cc4b42"


def column_filter_px(scan_sampling_A) -> tuple[tuple[float, float], tuple[int, int]]:
    """The high-pass gaussian sigma and the peak window, (row, col) in scan pixels, for one scan step.

    The window is ``2 r + 1`` pixels with ``r = round(COLUMN_SEPARATION_A / step)``, at least 1: two maxima of one
    window are at least ``r`` pixels apart.
    """
    rows, cols = (float(value) for value in np.broadcast_to(np.asarray(scan_sampling_A, dtype=float), (2,)))
    sigma = (COLUMN_HIGH_PASS_A / rows, COLUMN_HIGH_PASS_A / cols)
    window = tuple(2 * max(1, round(COLUMN_SEPARATION_A / step)) + 1 for step in (rows, cols))
    return sigma, window


def _high_pass(image: np.ndarray, sigma_px) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return image - gaussian_filter(image, sigma_px)


def locate_atomic_columns(dark_field, *, scan_sampling_A) -> np.ndarray:
    """Positions of atomic columns on a dark-field image, as an ``(n, 2)`` array of (row, col) scan pixels.

    The image is smoothed by a 1-pixel gaussian and high-passed by subtracting a gaussian of ``COLUMN_HIGH_PASS_A``
    (2.1 Å). Its local maxima at least ``COLUMN_SEPARATION_A`` (2 Å) apart are ranked by prominence, the height above
    the window's minimum, and the top ``COLUMN_TOP_FRACTION`` (20%) are the columns. A featureless image has none.

    Examples
    --------
    >>> columns = locate_atomic_columns(detector.df(data), scan_sampling_A=0.264)  # doctest: +SKIP
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    image = host_array(dark_field).astype(np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("The dark-field image must be one finite 2D scan image.")
    from scipy.ndimage import minimum_filter

    sigma, window = column_filter_px(scan_sampling_A)
    smoothed = gaussian_filter(image, COLUMN_SMOOTHING_PX)
    filtered = _high_pass(smoothed, sigma)
    prominence = filtered - minimum_filter(filtered, size=window)
    peaks = np.argwhere((filtered == maximum_filter(filtered, size=window)) & (prominence > 0))
    if not len(peaks):
        return peaks.astype(np.int32)
    heights = prominence[peaks[:, 0], peaks[:, 1]]
    return peaks[heights >= np.quantile(heights, 1.0 - COLUMN_TOP_FRACTION)].astype(np.int32)


def column_phase_test(phase, columns, *, scan_sampling_A) -> tuple[np.ndarray, float | None, float | None]:
    """The high-passed phase at atomic columns, its z score and the fraction of positive columns.

    ``p = phase - mean(phase)``, high-passed by subtracting a gaussian of ``COLUMN_HIGH_PASS_A``, is sampled at
    ``columns``; ``z = mean(p) / (std(p) / sqrt(n))``. Fewer than three columns, or no spread among them, give
    ``z = None``.
    """
    values = host_array(phase).astype(np.float64)
    sigma, _ = column_filter_px(scan_sampling_A)
    filtered = _high_pass(values - values.mean(), sigma)
    columns = np.asarray(columns, dtype=np.int64).reshape(-1, 2)
    sampled = filtered[columns[:, 0], columns[:, 1]]
    if sampled.size < 3:
        return sampled, None, None
    spread = float(np.std(sampled))
    if not spread > 0.0:
        return sampled, None, float(np.mean(sampled > 0))
    return sampled, float(np.mean(sampled) / (spread / math.sqrt(sampled.size))), float(np.mean(sampled > 0))


def _signed(value: float | None, digits: int) -> str:
    """A signed number with a typographic minus, as the announcements and figures print it."""
    return "n/a" if value is None else f"{value:+.{digits}f}".replace("-", "−")


@dataclass
class RotationBranch:
    """One independent fit on one 180-degree branch of the scan-detector rotation.

    Attributes
    ----------
    scan_detector_rotation_deg : float
        The rotation of this fit, in [0, 360).
    aberrations : dict[str, float]
        Fitted ``C10_nm``, ``C12_nm`` (nm) and ``phi12_deg`` (degrees).
    loss : float | None
        The fit's loss.
    phase_skewness : float
        Mean-centred third-moment ratio of the native phase (``phase_skewness``): positive for bright columns.
    column_z : float | None
        Mean high-passed phase at the dark-field columns in standard errors (None when no columns were located).
    column_effect : float | None
        Mean high-passed phase at the columns over its standard deviation, ``column_z / sqrt(n)``.
    column_positive_fraction : float | None
        Fraction of the columns with positive high-passed phase.
    phase : numpy.ndarray
        The fit's native-sampling phase in radians, which the tests read (``RotationBranches.upsampled_phases`` is
        what ``show`` draws).
    column_phases : numpy.ndarray | None
        The high-passed phase at each column, in radians.
    sample_tilt_mrad, depth_spread_nm
        The joint fit's tilt and depth spread; None for a standard fit.
    """
    scan_detector_rotation_deg: float
    aberrations: dict[str, float]
    loss: float | None
    phase_skewness: float
    column_z: float | None
    column_positive_fraction: float | None
    phase: np.ndarray = field(repr=False)
    column_phases: np.ndarray | None = field(default=None, repr=False)
    sample_tilt_mrad: tuple[float, float] | None = None
    depth_spread_nm: float | None = None
    column_effect: float | None = None


@dataclass
class RotationBranches:
    """Both 180-degree branches of an estimated rotation, fitted independently, and the branch kept.

    Returned by ``SSB.scan_rotations`` and recorded on the fit as ``result.rotation_branches``.

    Attributes
    ----------
    branches : tuple[RotationBranch, RotationBranch]
        The session's rotation first, then that rotation plus 180 degrees.
    columns : numpy.ndarray | None
        (row, col) scan pixels of the atomic columns located on the dark-field image, or None without one.
    chosen : int
        Index of the branch the evidence selects.
    kept : int
        Index of the branch the session holds; the chosen one unless a given rotation was kept.
    decided_by : {"columns", "phase_tail"}
        ``"columns"`` when the selected branch has decisive positive columns (``z >= COLUMN_Z_MIN`` and
        ``effect >= COLUMN_EFFECT_MIN``), else ``"phase_tail"`` (``phase_skewness``).
    scan_sampling_A : tuple[float, float] | None
        The native scan step (row, col) in Å.
    upsampling : int | None
        Output factor of ``upsampled_phases``, set when they are reconstructed (None before).
    """
    branches: tuple[RotationBranch, RotationBranch]
    columns: np.ndarray | None
    chosen: int
    kept: int
    decided_by: Literal["columns", "phase_tail"]
    scan_sampling_A: tuple[float, float] | None = None
    upsampling: int | None = None
    # (the requested upsampling, both panels) once reconstructed
    _upsampled: tuple[object, tuple[np.ndarray, np.ndarray]] | None = field(default=None, repr=False, compare=False)
    # a weakref.WeakMethod to the fitting session's reconstruction of both panels: it never keeps a session alive
    _panel_source: Callable | None = field(default=None, repr=False, compare=False)

    @property
    def chosen_rotation_deg(self) -> float:
        """The rotation the evidence selects, in degrees."""
        return self.branches[self.chosen].scan_detector_rotation_deg

    @property
    def kept_rotation_deg(self) -> float:
        """The rotation the session holds after the comparison, in degrees."""
        return self.branches[self.kept].scan_detector_rotation_deg

    @property
    def num_columns(self) -> int:
        """Number of atomic columns located on the dark-field image."""
        return 0 if self.columns is None else len(self.columns)

    @property
    def object_sampling_A(self) -> tuple[float, float] | None:
        """Pixel size (row, col) in Å of ``upsampled_phases``: the scan step over ``upsampling``."""
        if self.scan_sampling_A is None or self.upsampling is None:
            return None
        return tuple(float(step) / self.upsampling for step in self.scan_sampling_A)

    def upsampled_phases(self, upsampling: int | Literal["auto"] = "auto") -> tuple[np.ndarray, np.ndarray]:
        """Each branch's fit reconstructed at ``upsampling`` (radians): the panels ``show`` draws.

        The SSB session that fitted or reloaded the branches reconstructs each one at its own rotation, aberrations,
        sample tilt and depth spread with the G(q, k) it already holds: no search and no second G(q, k) build (a
        session that reloaded the fit builds its G(q, k) first, as its first reconstruction would). On
        the 512 x 512 gold scan at 0.5 Å (4x auto, 2048 x 2048 each) both took 0.4 s on an M5 Max, 0.35 GiB above
        the fitted session and below the fit's own peak. The result is kept for the next call. ``"auto"`` is
        ``ssb.reconstruct(upsampling="auto")``'s factor; the column test and the phase tail stay on the native
        ``phase``.
        """
        cached = self._upsampled
        if cached is None or cached[0] != upsampling:
            source = None if self._panel_source is None else self._panel_source()
            phases = None if source is None else source(self, upsampling)
            if phases is None:
                raise ValueError(
                    "The branch panels are reconstructed by the SSB session that fitted or reloaded these branches, and "
                    "it is closed. Show them before ssb.close(), or reload the fit with the same save_dir; "
                    "show(panels=False) draws the evidence alone.")
            factor, phases = phases
            self._upsampled, self.upsampling = (upsampling, tuple(phases)), int(factor)
        return self._upsampled[1]

    def bind_session(self, reconstruct_panels: Callable) -> None:
        """Let a session's bound method reconstruct ``upsampled_phases``, held weakly so a session is never kept alive."""
        self._panel_source = weakref.WeakMethod(reconstruct_panels)

    @property
    def summary(self) -> str:
        """One line: the branch kept and the test that decided it."""
        chosen, other = self.branches[self.chosen], self.branches[1 - self.chosen]
        if self.decided_by == "columns":
            evidence = (f"atomic columns positive (z {_signed(chosen.column_z, 1)}, effect "
                        f"{_signed(chosen.column_effect, 2)}, {chosen.column_positive_fraction:.0%} of "
                        f"{self.num_columns} columns); "
                        f"{other.scan_detector_rotation_deg:.1f}° "
                        + ("negative" if (other.column_z or 0.0) < 0 else f"z {_signed(other.column_z, 1)}"))
            choice = f"{chosen.scan_detector_rotation_deg:.1f}°: {evidence}"
        else:
            reason = (f"positive columns not decisive (z {_signed(chosen.column_z, 1)}, effect "
                      f"{_signed(chosen.column_effect, 2)})" if self.num_columns else
                      "no atomic columns located on a dark-field image")
            choice = (f"{chosen.scan_detector_rotation_deg:.1f}° by the phase tail "
                      f"(S {_signed(chosen.phase_skewness, 2)} over {_signed(other.phase_skewness, 2)}); {reason}")
        if self.kept == self.chosen:
            return f"kept {choice}."
        return f"kept the given {self.kept_rotation_deg:.1f}°; the evidence selects {choice}."

    def __repr__(self) -> str:
        return f"RotationBranches({self.summary})"

    def report(self):
        """Both branches as a table: fitted aberrations, loss, column test and phase tail, and the branch kept."""
        import pandas as pd

        rows = []
        for index, branch in enumerate(self.branches):
            row = {
                "C10 (nm)": branch.aberrations.get("C10_nm"),
                "C12 (nm)": branch.aberrations.get("C12_nm"),
                "phi12 (deg)": branch.aberrations.get("phi12_deg", 0.0),
            }
            if branch.sample_tilt_mrad is not None:
                row.update({"tilt row (mrad)": branch.sample_tilt_mrad[0], "tilt col (mrad)": branch.sample_tilt_mrad[1],
                            "depth spread (nm)": branch.depth_spread_nm})
            row.update({
                "loss": branch.loss,
                "column z": branch.column_z,
                "column effect": branch.column_effect,
                "positive columns (%)": None if branch.column_positive_fraction is None
                else 100.0 * branch.column_positive_fraction,
                "phase tail S": branch.phase_skewness,
                "kept": "yes" if index == self.kept else "",
            })
            rows.append(row)
        index = pd.Index([f"{branch.scan_detector_rotation_deg:.2f}°" for branch in self.branches],
                         name=f"Rotation (decided by {'atomic columns' if self.decided_by == 'columns' else 'phase tail'}"
                              f"{f', {self.num_columns} columns' if self.num_columns else ''})")
        table = pd.DataFrame(rows, index=index)
        return table.round({name: 3 for name in table.columns if name not in ("loss", "kept")}).round({"loss": 7})

    def show(self, *, upsampling: int | Literal["auto"] = "auto", panels: bool = True,
             size: str | int | None = None, zoom: float = 1,
             axsize: tuple[float, float] | None = None,
             cmap: str = "inferno", save: str | Path | None = None):
        """Both rotations' phase images above their full-field phase histograms.

        Top: each branch's fit reconstructed at ``upsampling`` (``upsampled_phases``, ``"auto"`` by default),
        minus its mean, on shared symmetric limits (pooled 0.5-99.5%), pixel size = step / factor, with the dark-field
        columns at scan pixels times the factor. Bottom, from the native phases the decision used: each branch's
        mean-subtracted phase as the fraction of pixels per bin, on identical symmetric bins and a log axis (bins with
        no pixels left out). The kept
        branch is solid, the other dashed; phase in mrad. ``panels=False`` draws the evidence alone (no GPU).
        ``size`` selects each panel's width, medium by default. ``axsize`` is
        an advanced alternative in inches; a phase panel is ``axsize[0]`` square.
        ``zoom`` adds boxed central magnifications below the full fitted fields;
        the full-region evidence and fitted arrays stay unchanged.
        ``cmap`` chooses the phase colormap; ``save`` exports the static figure.
        The phase/evidence layout is fixed rather than a rows/columns gallery.

        Returns
        -------
        matplotlib.figure.Figure
            Closed, so a bare notebook expression renders it once.

        Examples
        --------
        >>> ssb.scan_rotations(report=False, ).show()  # doctest: +SKIP
        >>> ssb.scan_rotations(report=False, ).show(size="large")  # doctest: +SKIP
        """
        import matplotlib.pyplot as plt

        from quantem.gpu.display._layout import panel_inches, zoom_bounds

        zoom_bounds((1, 1), zoom)

        if size is not None and axsize is not None:
            raise ValueError("Use size or axsize, not both.")
        if axsize is None:
            width = panel_inches("medium" if size is None else size)
            axsize = (width, width * 3.4 / 5)
        heights = ([axsize[0]] * (2 if zoom > 1 else 1) if panels else []) + [axsize[1]]
        figure = plt.figure(figsize=(2 * axsize[0], sum(heights) + (0.6 if panels else 0.0)))
        grid = figure.add_gridspec(len(heights), 2, height_ratios=heights)
        if panels:
            axes = [figure.add_subplot(grid[0, index]) for index in (0, 1)]
            self._show_panels(figure, axes, upsampling, cmap)
            if zoom > 1:
                from matplotlib.patches import Rectangle

                zoom_axes = [figure.add_subplot(grid[1, index]) for index in (0, 1)]
                self._show_panels(figure, zoom_axes, upsampling, cmap, zoom=zoom)
                bounds = zoom_bounds(self.branches[0].phase.shape, zoom)
                row, stop_row, col, stop_col = (value * self.upsampling for value in bounds)
                for axis in axes:
                    axis.add_patch(Rectangle((col - .5, row - .5), stop_col - col, stop_row - row,
                                             fill=False, edgecolor="#00D5FF", linewidth=1))
        tail_axis = figure.add_subplot(grid[-1, :])
        order = (self.kept, 1 - self.kept)
        styles = {self.kept: {"color": KEPT_COLOR, "linestyle": "-"}, 1 - self.kept: {"color": OTHER_COLOR, "linestyle": "--"}}

        centred = [branch.phase.astype(np.float64) - branch.phase.astype(np.float64).mean() for branch in self.branches]
        limit = max(float(np.max(np.abs(values))) for values in centred) or 1.0
        edges = np.linspace(-limit, limit, 102)
        centres = 1e3 * (edges[1:] + edges[:-1]) / 2.0
        for index in order:
            fractions = np.histogram(centred[index], bins=edges)[0] / centred[index].size
            branch = self.branches[index]
            tail_axis.plot(centres, np.where(fractions > 0, fractions, np.nan), linewidth=1.4, **styles[index],
                           label=f"{branch.scan_detector_rotation_deg:.1f}° · S = {_signed(branch.phase_skewness, 3)}")
        tail_axis.axvline(0.0, color="0.5", linewidth=0.6)
        tail_axis.set(yscale="log", xlim=(-1e3 * limit, 1e3 * limit), xlabel="phase − mean (mrad)",
                      ylabel="fraction of pixels")
        tail_axis.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0))

        tail_axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        if save is not None:
            figure.savefig(save, dpi="figure", bbox_inches="tight")
        plt.close(figure)
        return figure

    def _show_panels(self, figure, axes, upsampling, cmap="inferno", *, zoom=1) -> None:
        """Draw ``upsampled_phases`` on ``axes`` (branch order), the columns at scan pixels times the factor."""
        from quantem.core.visualization import show_2d

        centred = [phase.astype(np.float64) - phase.astype(np.float64).mean()
                   for phase in self.upsampled_phases(upsampling)]
        factor = self.upsampling
        lower, upper = np.quantile(np.concatenate([values.ravel() for values in centred]), [0.005, 0.995])
        limit = max(abs(float(lower)), abs(float(upper))) or 1.0
        label = f"{factor}× auto" if upsampling == "auto" else f"{factor}×"
        titles = [f"{branch.scan_detector_rotation_deg:.1f}°{' kept' if index == self.kept else ''} · {label} (rad)"
                  for index, branch in enumerate(self.branches)]
        sampling = self.object_sampling_A
        row = col = 0
        displayed = centred
        if zoom > 1:
            from quantem.gpu.display._layout import zoom_bounds

            row, stop_row, col, stop_col = (value * factor for value in zoom_bounds(self.branches[0].phase.shape, zoom))
            displayed = [phase[row:stop_row, col:stop_col] for phase in centred]
            titles = [f"{title}\nCentral zoom {zoom:g}×" for title in titles]
        show_2d(displayed, figax=(figure, np.asarray(axes)), title=titles, cmap=cmap, cbar=True,
                norm={"interval_type": "manual", "vmin": -limit, "vmax": limit}, tight_layout=False, title_fontsize=12,
                scalebar=None if sampling is None else {"sampling": sampling[1], "units": "Å", "fontsize": 11})
        if self.num_columns:
            positions = self.columns.astype(np.float64) * factor - np.array([row, col])
            for axis in axes:
                limits = axis.get_xlim(), axis.get_ylim()
                axis.scatter(positions[:, 1], positions[:, 0], s=14, facecolors="none", edgecolors="#00D5FF",
                             linewidths=0.6)
                axis.set(xlim=limits[0], ylim=limits[1])

    def to_record(self) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        """The JSON part and the arrays of a saved comparison (``persistence.save_result``).

        Auto-upsampled panels are saved only when they exist at save time (a ``scan_region`` fit, whose region session
        closes after the fit); otherwise the session that reloads the record reconstructs them.
        """
        record = {
            "chosen": self.chosen, "kept": self.kept, "decided_by": self.decided_by,
            "scan_sampling_A": self.scan_sampling_A,
            "branches": [{
                "scan_detector_rotation_deg": branch.scan_detector_rotation_deg, "aberrations": branch.aberrations,
                "loss": branch.loss, "phase_skewness": branch.phase_skewness, "column_z": branch.column_z,
                "column_effect": branch.column_effect,
                "column_positive_fraction": branch.column_positive_fraction,
                "sample_tilt_mrad": branch.sample_tilt_mrad, "depth_spread_nm": branch.depth_spread_nm,
            } for branch in self.branches],
        }
        arrays = {"rotation_branch_phases": np.stack([branch.phase for branch in self.branches])}
        if self.columns is not None:
            arrays["rotation_branch_columns"] = self.columns
            arrays["rotation_branch_column_phases"] = np.stack([branch.column_phases for branch in self.branches])
        if self._upsampled is not None and self._upsampled[0] == "auto":
            record["upsampling"] = self.upsampling
            arrays["rotation_branch_upsampled_phases"] = np.stack(self._upsampled[1])
        return record, arrays

    @classmethod
    def from_record(cls, record: dict[str, object], arrays) -> "RotationBranches":
        """Rebuild a saved comparison from ``to_record``'s two parts."""
        columns = np.asarray(arrays["rotation_branch_columns"]) if "rotation_branch_columns" in arrays else None
        branches = []
        for index, saved in enumerate(record["branches"]):
            tilt = saved.get("sample_tilt_mrad")
            branches.append(RotationBranch(
                scan_detector_rotation_deg=float(saved["scan_detector_rotation_deg"]),
                aberrations=dict(saved["aberrations"]), loss=saved["loss"],
                phase_skewness=float(saved["phase_skewness"]), column_z=saved["column_z"],
                column_positive_fraction=saved["column_positive_fraction"],
                phase=np.asarray(arrays["rotation_branch_phases"][index]),
                column_phases=None if columns is None else np.asarray(arrays["rotation_branch_column_phases"][index]),
                sample_tilt_mrad=None if tilt is None else tuple(tilt), depth_spread_nm=saved.get("depth_spread_nm"),
                column_effect=saved.get("column_effect"),
            ))
        sampling = record.get("scan_sampling_A")
        restored = cls(branches=tuple(branches), columns=columns, chosen=int(record["chosen"]), kept=int(record["kept"]),
                       decided_by=record["decided_by"], scan_sampling_A=None if sampling is None else tuple(sampling))
        if "rotation_branch_upsampled_phases" in arrays:
            restored.upsampling = int(record["upsampling"])
            restored._upsampled = ("auto", tuple(np.asarray(arrays["rotation_branch_upsampled_phases"])))
        return restored


def compare_branches(fits, *, dark_field, scan_sampling_A, keep_first: bool) -> RotationBranches:
    """Score two fits (the session's rotation first, then plus 180 degrees) and choose the branch.

    ``fits`` are SSB results whose phase is the native-sampling reconstruction. ``dark_field`` is the same scan's
    dark-field image, or None when the session has none; then, or when neither branch has decisive positive
    columns (``z >= COLUMN_Z_MIN`` and ``effect >= COLUMN_EFFECT_MIN``), the larger ``phase_skewness`` decides.
    A negative column score never certifies the other independent fit. ``keep_first`` keeps the first branch whatever the evidence says
    (a rotation the user gave), and the result still says which branch the evidence selects.
    """
    phases = [host_array(fit.phase).astype(np.float32) for fit in fits]
    columns = None
    if dark_field is not None and np.shape(dark_field) == phases[0].shape:
        columns = locate_atomic_columns(dark_field, scan_sampling_A=scan_sampling_A)
    branches = []
    for fit, phase in zip(fits, phases, strict=True):
        sampled, z, positive = (None, None, None) if columns is None else column_phase_test(
            phase, columns, scan_sampling_A=scan_sampling_A)
        branches.append(RotationBranch(
            scan_detector_rotation_deg=float(fit.scan_detector_rotation_deg), aberrations=dict(fit.aberrations),
            loss=None if fit.loss is None else float(fit.loss), phase_skewness=phase_skewness(phase),
            column_z=z, column_positive_fraction=positive, phase=phase,
            column_phases=None if sampled is None else sampled.astype(np.float32),
            sample_tilt_mrad=fit.sample_tilt_mrad, depth_spread_nm=fit.depth_spread_nm,
            column_effect=None if z is None else z / math.sqrt(sampled.size),
        ))
    positive_branches = [index for index, branch in enumerate(branches)
                         if branch.column_z is not None and branch.column_z >= COLUMN_Z_MIN
                         and branch.column_effect is not None and branch.column_effect >= COLUMN_EFFECT_MIN]
    if positive_branches:
        decided_by = "columns"
        chosen = max(positive_branches, key=lambda index: branches[index].column_z)
    else:
        decided_by = "phase_tail"
        chosen = 0 if branches[0].phase_skewness >= branches[1].phase_skewness else 1
    step = tuple(float(value) for value in np.broadcast_to(np.asarray(scan_sampling_A, dtype=float), (2,)))
    return RotationBranches(branches=tuple(branches), columns=columns, chosen=chosen,
                            kept=0 if keep_first else chosen, decided_by=decided_by, scan_sampling_A=step)
