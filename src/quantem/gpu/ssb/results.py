"""Backend-neutral SSB fit, source, evaluation, and reconstruction results."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.optics.physics import wrap_degrees
from quantem.gpu.ssb.probe import model_probe, probe_grid

# A resolved crystal has sparse positive columns; below this magnitude the phase histogram is too symmetric to tell which
# 180-degree branch the scan-detector rotation is on, so the branch is left as given.
COLUMN_SIGN_MIN = 0.2


@dataclass
class SSBResult(Mapping[str, float]):
    """Result from SSB ptychographic reconstruction.

    The primary output is ``object_wave``, the complex transmission function.
    Convenience properties ``phase`` and ``amplitude`` are derived from it.

    Read coefficients directly with ``aberrations["C10_nm"]``,
    ``aberrations["C12_nm"]`` (nm), and ``aberrations["phi12_deg"]`` (degrees).
    ``dict(aberrations)`` copies the coefficients. Tilt, rotation, phase, and
    trial history remain properties. After ``find_aberrations(tilt=True)``,
    C10 is the defocus at the model's mid-depth. Use reconstruction overrides
    to explore changed coefficients without modifying the fitted result.

    Attributes
    ----------
    object_wave : cp.ndarray
        Complex transmission function (scan_row, scan_col).
    sample_tilt_mrad : tuple[float, float] | None
        Sample tilt (row, col) in mrad, scan frame, from ``find_aberrations(tilt=True)``; None for standard SSB.
    depth_spread_nm : float | None
        Depth over which the tilted columns spread in the thick-sample model (a model parameter, not a measured
        thickness); None for standard SSB.
    tilt_fit_gain : float | None
        Least-squares agreement with the data of the tilt fit relative to standard SSB (> 1: the tilt explains more).
    scan_detector_rotation_deg : float
        Physical scan-to-detector rotation in degrees, in [0, 360). The CoM curl fixes it only modulo 180 degrees;
        ``find_aberrations(check_rotation=True)`` settles the branch from the atom columns (see ``column_sign``).
    detector_transposed : bool
        Whether the detector axes were exchanged before applying the rotation.
    initial_scan_detector_rotation_deg : float | None
        The rotation before the polarity check, in [0, 360); the check changed the branch when it differs from
        ``scan_detector_rotation_deg``. None when no check ran.
    upsampling_factor : int
        Output factor relative to the native scan grid. ``object_sampling_A``
        records output pixel spacing, preserving the physical field of view.
    object_sampling_A : tuple[float, float] | None
        Pixel size of the reconstructed image in Angstrom, (row, col): the scan step divided by ``upsampling_factor``.
    phase_estimator : str
        ``mean_phase`` or ``phase_of_mean`` for phase-only reconstruction;
        ``complex_wave`` identifies a direct complex-object reconstruction.
    amplitude_estimated : bool
        False for phase-only paths, whose complex wave has unit amplitude.
    loss : float | None
        Variance loss value.
    elapsed_s : float | None
        Wall-clock time in seconds.
    timings : dict[str, float]
        Backend stage timings in seconds when the operation records them.
    reused : bool
        Whether the result was restored from an exact saved-result match.
    saved_path : pathlib.Path | None
        Array artifact used for persistence, when ``save_to`` was provided.
    metadata : dict[str, object]
        Complete saved scientific signature, provenance, and result metadata.
    parameter_slices : dict[str, object] or None
        Measured conditional focus-depth and tilt objective slices, added by
        ``ssb.measure_parameters(result)``. Includes units, rotation and timing;
        reloading or showing them does not rerun the objective.
    """
    object_wave: object
    backend: Literal["cuda", "mps"]
    aberrations: dict[str, float] = field(default_factory=dict)
    sample_tilt_mrad: tuple[float, float] | None = None
    depth_spread_nm: float | None = None
    tilt_fit_gain: float | None = None
    scan_detector_rotation_deg: float = 0.0
    detector_transposed: bool = False
    loss: float | None = None
    elapsed_s: float | None = None
    timings: dict[str, float] = field(default_factory=dict)
    num_trials: int | None = None
    num_bright_field_pixels: int | None = None
    refine_method: str | None = None
    refine_nfev: int | None = None
    refine_elapsed_s: float | None = None
    voltage_kV: float | None = None
    semiangle_mrad: float | None = None
    object_sampling_A: tuple[float, float] | None = None
    upsampling_factor: int = 1
    phase_estimator: str = "complex_wave"
    amplitude_estimated: bool = True
    source_path: str | None = None
    bright_field_center_px: tuple[float, float] | None = None
    bright_field_radius_px: float | None = None
    bright_field_edge_radius_px: float | None = None
    # Every evaluated search trial in order, ``{"trial", "number", "search", "stage", "params": {"C10_nm", "C12_nm",
    # "phi12_deg"}, "loss"}`` (a tilt fit adds the objective, band, tilt and depth spread); ``trials`` tabulates them.
    trial_records: list[dict] | None = None
    # Phase skewness (+ = bright atom columns) and the selected 180-degree branch.
    column_sign: float | None = None
    initial_scan_detector_rotation_deg: float | None = None
    initial_column_sign: float | None = None
    rotation_check_phases: object | None = None
    phase_clim: tuple[float, float] | None = None
    fit_scan_region: tuple[int, int, int, int] | None = None
    reused: bool = False
    saved_path: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    parameter_slices: dict[str, object] | None = None

    def __post_init__(self) -> None:
        # one representation of the physical angle, whatever turn count a caller or an engine gives
        self.scan_detector_rotation_deg = wrap_degrees(self.scan_detector_rotation_deg, 360.0)

    def __getitem__(self, coefficient: str) -> float:
        """Return a coefficient: ``C10_nm`` or ``C12_nm`` in nm, ``phi12_deg`` in degrees.

        Examples
        --------
        >>> aberrations = ssb.find_aberrations()
        >>> defocus_nm = aberrations["C10_nm"]
        """
        return self.aberrations[coefficient]

    def __iter__(self) -> Iterator[str]:
        """Iterate over coefficient names only."""
        return iter(self.aberrations)

    def __len__(self) -> int:
        """Return the number of coefficients."""
        return len(self.aberrations)

    def __repr__(self) -> str:
        lines = ["SSB Result"]
        lines.append(f"  Shape          {tuple(self.object_wave.shape)} · {self.upsampling_factor}x output")
        lines.append(f"  Estimator      {self.phase_estimator}")
        if not self.amplitude_estimated:
            lines.append("  Amplitude      not estimated (unit-amplitude phase representation)")
        if self.loss is not None:
            lines.append(f"  Loss           {self.loss:.6f}")
        if self.num_bright_field_pixels is not None:
            lines.append(f"  BF pixels      {self.num_bright_field_pixels}")
        if self.num_trials is not None:
            lines.append(f"  Trials         {self.num_trials}")
        if self.initial_scan_detector_rotation_deg is not None:
            lines.append(f"  Input rotation {self.initial_scan_detector_rotation_deg:.2f}°")
        lines.append(f"  SSB rotation   {self.scan_detector_rotation_deg:.2f}°")
        if self.column_sign is not None:
            changed = self.initial_scan_detector_rotation_deg not in (None, self.scan_detector_rotation_deg)
            lines.append(
                f"  Column sign    {self.column_sign:+.2f}"
                + ("  (SSB selected the other 180° branch)" if changed else "")
            )
        if self.aberrations:
            lines.append(_format_aberrations(self.aberrations))
        if self.sample_tilt_mrad is not None:
            lines.append(
                f"  Sample tilt    ({self.sample_tilt_mrad[0]:+.2f}, {self.sample_tilt_mrad[1]:+.2f}) mrad scan frame, "
                f"depth spread {self.depth_spread_nm:.1f} nm"
            )
        if self.elapsed_s is not None:
            lines.append(f"  Time           {self.elapsed_s:.2f}s")
        if self.saved_path is not None:
            lines.append(
                f"  Saved result   {self.saved_path}"
                + (" (reused)" if self.reused else "")
            )
        return "\n".join(lines)

    @property
    def trials(self):
        """Completed search trials as a table, indexed by stable trial ID.

        Coefficient lengths are nm, angles explicitly name their units, and
        loss is the recorded search objective (lower is better). Different
        objectives must not be ranked together. Refinement is reported separately.
        """
        import pandas as pd

        rows = []
        for index, record in enumerate(self.trial_records or ()):
            if record.get("stage", "search") != "search":
                continue
            rows.append({"trial": record.get("trial", index),
                         **{key: value for key, value in record.items() if key not in {"params", "trial"}},
                         **record["params"]})
        if not rows:
            return pd.DataFrame(index=pd.Index([], name="trial"))
        return pd.DataFrame.from_records(rows).set_index("trial")

    def report(self):
        """One-row table of the fitted parameters; ``pd.concat([a.report(), b.report()])`` compares fits side by side."""
        import pandas as pd

        if self.object_wave.ndim == 3:
            return self.trials
        aberrations = self.aberrations
        row = {
            "C10 (nm)": aberrations.get("C10_nm"),
            "C12 (nm)": aberrations.get("C12_nm"),
            "phi12 (deg)": aberrations.get("phi12_deg", 0.0),
        }
        if self.initial_scan_detector_rotation_deg is not None:
            row["Suggested rotation (deg)"] = self.initial_scan_detector_rotation_deg
        row["SSB-selected rotation (deg)"] = self.scan_detector_rotation_deg
        if self.sample_tilt_mrad is not None:
            row.update({"tilt row (mrad)": self.sample_tilt_mrad[0],
                        "tilt col (mrad)": self.sample_tilt_mrad[1],
                        "depth spread (nm)": self.depth_spread_nm})
        model = "standard SSB" if self.sample_tilt_mrad is None else "tilt-aware SSB"
        return pd.DataFrame([row], index=pd.Index([model], name="model")).round(3)

    @property
    def phase(self):
        """Phase of the complex transmission function: ``angle(object_wave)``."""
        if _is_cupy_array(self.object_wave):
            return cp.angle(self.object_wave)
        return np.angle(self.object_wave)

    @property
    def phase_limits(self) -> tuple[float, float]:
        """Shared linear phase limits in radians, symmetric about zero.

        Examples
        --------
        >>> result.phase_limits
        """
        if self.phase_clim is not None:
            return tuple(self.phase_clim)
        limit = float(abs(self.phase).max())
        if self.rotation_check_phases is not None:
            limit = max(limit, float(np.max(np.abs(self.rotation_check_phases))))
        limit = limit or 1.0
        return (-limit, limit)

    @property
    def amplitude(self):
        """Amplitude of the complex transmission function: ``abs(object_wave)``."""
        if _is_cupy_array(self.object_wave):
            return cp.abs(self.object_wave)
        return np.abs(self.object_wave)

    @property
    def probe_sampling_A(self) -> tuple[float, float]:
        """Real-space probe spacing in Å per pixel, in (row, column) order.

        The model grid resolves the aperture and expands for defocus and
        astigmatism. It is independent of detector binning, scan spacing,
        and reconstruction upsampling.

        Examples
        --------
        >>> row_step_A, column_step_A = result.probe_sampling_A
        """
        return probe_grid(self)[1]

    @property
    def probe_sampling_mrad(self) -> tuple[float, float]:
        """Model-probe Fourier spacing in mrad per pixel, (row, column).

        Examples
        --------
        >>> row_step_mrad, column_step_mrad = result.probe_sampling_mrad
        """
        return probe_grid(self)[2]

    def probe(self, *, space: Literal["real", "fourier"] = "real"):
        """Calculate the normalized complex model probe on the result's GPU.

        Reuses voltage, convergence semiangle and
        the effective aberrations, including reconstruction overrides. Both
        arrays are centered and expressed in scan-axis (row, column) order.
        This is the fitted optical model, not an independently retrieved probe
        or a measured vacuum diffraction pattern. Specimen tilt is not beam
        tilt and does not shift the aperture. For a thick model, the probe is
        evaluated at the model's mid-depth.

        Parameters
        ----------
        space : {"real", "fourier"}, optional
            Real space uses ``probe_sampling_A`` (Å per pixel). Fourier space
            uses ``probe_sampling_mrad`` (mrad per pixel). The model grid is
            chosen automatically; it is not the measured detector grid.

        Returns
        -------
        cupy.ndarray or mlx.core.array
            Centered complex64 wave with unit summed intensity. Use
            ``abs(probe) ** 2`` to display intensity. Probe sampling is
            numerical model sampling, not a claim of measured resolution.

        Examples
        --------
        >>> result = ssb.reconstruct(aberrations)
        >>> probe = result.probe()
        >>> probe_fourier = result.probe(space="fourier")
        """
        if space not in ("real", "fourier"):
            raise ValueError(f"space must be 'real' or 'fourier'; got {space!r}.")
        if self.object_wave.ndim != 2:
            raise ValueError("A trial stack has several probes. Reconstruct one trial's parameters first.")
        return model_probe(self, space=space)

    def show(self, view: Literal["phase", "probe", "rotation", "parameters"] = "phase", *,
             axsize: tuple[float, float] | None = None, compare: Self | None = None,
             histogram: bool = False, show_parameters: bool = True):
        """Show a large phase image or the real-space and Fourier-space probes.

        Phase views reuse symmetric linear limits from their fitted result.
        Probe intensities use their full linear range. Plotting does not change the
        reconstructed wave or model probe.

        Parameters
        ----------
        view : {"phase", "probe", "rotation", "parameters"}, optional
            Show the reconstructed phase alone, or two model-probe intensity
            panels side by side. Fourier intensity shows the aperture;
            aberrations are encoded in the complex Fourier probe's phase.
            ``rotation`` shows the exact phase images used in the polarity
            check, before and after branch selection. Call on the result of
            ``find_aberrations()``, not a subsequent reconstruction.
            It explains the decision and displays only when SSB selected a
            different physical rotation. Agreement returns None with no plot.
        axsize : tuple[float, float] or None, optional
            Width and height of each panel in inches. Defaults to (12, 12) for
            phase and (6, 6) for each probe or comparison panel.
        compare : SSBResult or None, optional
            A second phase reconstruction of the same field of view. Shows
            both full images above matching central crops, with rectangles
            marking the crop locations. The crop is the central quarter of
            each dimension, selected on this result's grid. All four panels
            share this result's phase_limits; no alignment is performed.
        histogram : bool, optional
            Add median-centered phase histograms to the rotation view, with
            the recorded decision scores and shared horizontal and vertical
            limits. The comparison is omitted when the rotation is unchanged.
        show_parameters : bool, optional
            Include saved objective slices beside a single phase image by
            default. This only displays recorded values; it runs no fitting
            or GPU evaluations. Use ``view="parameters"`` for the plots alone.
            Measure them first with ``ssb.measure_parameters(result)``.

        Returns
        -------
        matplotlib.figure.Figure or None
            A calibrated figure, closed so a bare notebook expression renders
            once. Phase is in radians; real-space and Fourier-space probe
            scale bars use Å and mrad, respectively. Each probe has unit summed
            intensity and its own linear display range.
            Returns None for an unchanged rotation in the rotation view.

        Examples
        --------
        >>> result = ssb.reconstruct(aberrations)
        >>> result.show()
        >>> result.show("probe")
        >>> result.show(axsize=(14, 14))  # width and height in inches
        >>> result.show(compare=result_4x, axsize=(7, 7))
        >>> aberrations.show("rotation", histogram=True, axsize=(7, 7))
        >>> ssb.measure_parameters(aberrations)
        >>> aberrations.show("parameters")
        """
        import matplotlib.pyplot as plt
        from quantem.core.visualization import show_2d

        if histogram and view != "rotation":
            raise ValueError("Use aberrations.show('rotation', histogram=True) for the polarity histograms.")
        if compare is not None:
            if view != "phase":
                raise ValueError("Use result.show(compare=other) to compare phase images.")
            return _phase_comparison(self, compare, axsize=(6, 6) if axsize is None else axsize)
        if view == "rotation":
            return _rotation_comparison(self, axsize=(6, 6) if axsize is None else axsize, histogram=histogram)
        if view == "parameters" or (view == "phase" and show_parameters and self.parameter_slices):
            from quantem.gpu.ssb.parameters import plot_parameters

            return plot_parameters(self, include_phase=view == "phase", axsize=axsize)
        if view == "phase":
            column_sampling = float(self.object_sampling_A[1])
            arrays = self.phase
            titles = f"SSB phase · {self.upsampling_factor}× (rad)"
            bars = {"sampling": column_sampling, "units": "Å", "fontsize": 13}
            panel_size = (12, 12)
        elif view == "probe":
            arrays = [abs(self.probe()) ** 2, abs(self.probe(space="fourier")) ** 2]
            titles = ["Real-space model probe · intensity", "Fourier-space model probe · intensity"]
            bars = [
                {"sampling": self.probe_sampling_A[1], "units": "Å", "fontsize": 13},
                {"sampling": self.probe_sampling_mrad[1], "units": "mrad", "fontsize": 13},
            ]
            panel_size = (6, 6)
        else:
            raise ValueError(f"Choose view='phase', 'probe', 'rotation' or 'parameters'; got {view!r}.")
        figure, _ = show_2d(
            arrays, title=titles,
            norm=({"interval_type": "manual", "vmin": self.phase_limits[0],
                   "vmax": self.phase_limits[1]} if view == "phase" else "minmax"),
            cmap="inferno", cbar=view == "phase",
            axsize=panel_size if axsize is None else axsize, title_fontsize=16,
            scalebar=bars,
        )
        plt.close(figure)
        return figure


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
    num_trials: int
    refinement: str | None

    def metadata(self):
        """Return compact source and reconstruction metadata as a table indexed by setting."""
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
                ("Trials", self.num_trials),
                ("Refinement", self.refinement or "none"),
                ("Results reused", reused),
            ]
        )
        return pd.DataFrame(rows, columns=("Setting", "Value")).set_index("Setting")


def column_sign(phase: object) -> float:
    """Skewness of the phase about its median: positive when atom columns are bright.

    The projected potential of atoms is positive, so on the correct scan-detector rotation a resolved crystal's phase
    histogram has a long tail on the positive side (few bright columns over a flatter background). Rotating the scan by
    180 degrees negates the phase and the sign. SSB loses the absolute phase level, hence the median rather than zero.
    """
    # float64 so the third moment of a large image is not rounded away
    values = host_array(phase).astype(np.float64).ravel()
    values = values - np.median(values)
    variance = float(np.mean(values * values))
    if variance <= 0.0:
        return 0.0
    return float(np.mean(values ** 3) / variance ** 1.5)


def draw_column_histogram(phase: object, title: str, *, limit: float | None = None, bins: int = 101) -> float:
    """Draw the phase histogram about its median with its mirror image (dashed); return the axis half-width used.

    Bars beyond the dashed line are the tail: on the right when atom columns are bright (correct rotation), on the left
    when the scan-detector rotation is 180 degrees off. Pass the returned ``limit`` to the next call so two histograms
    share one axis. Displays in a notebook only.
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    values = host_array(phase).astype(np.float64).ravel()
    values = values - np.median(values)
    if limit is None:
        limit = float(np.percentile(np.abs(values), 99.9))
    edges = np.linspace(-limit, limit, bins)
    counts = np.histogram(values, edges)[0] / values.size
    centres = (edges[1:] + edges[:-1]) / 2.0
    figure, axis = plt.subplots(figsize=(6, 2.6))
    axis.bar(centres, counts, width=edges[1] - edges[0], color="#e6873c")
    axis.plot(centres, counts[::-1], "k--", linewidth=1, label="mirrored")
    axis.axvline(0.0, color="0.5", linewidth=0.8)
    axis.set_yscale("log")
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("phase - median (rad)")
    axis.set_ylabel("fraction of pixels")
    axis.set_title(title, fontsize=10)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    figure.tight_layout()
    display(figure)
    plt.close(figure)
    return limit


def host_array(value: object) -> np.ndarray:
    """Return a CuPy or NumPy array as a NumPy array.

    Histograms, figures and saved files are made on the host, while CUDA
    results keep their arrays on the device until one of those needs them.
    """
    return cp.asnumpy(value) if _is_cupy_array(value) else np.asarray(value)


def _is_cupy_array(value: object) -> bool:
    """Return whether *value* is a CuPy array without requiring CUDA."""

    return cp is not None and isinstance(value, cp.ndarray)


def _format_aberrations(aberrations: dict) -> str:
    """Format SSB aberration dict as aligned key-value lines."""
    lines = []
    if "C10_nm" in aberrations:
        lines.append(f"  Defocus (C10)  {aberrations['C10_nm']:.1f} nm")
    if "C12_nm" in aberrations:
        lines.append(f"  Astigmatism    {aberrations['C12_nm']:.1f} nm")
    if "phi12_deg" in aberrations:
        lines.append(f"  Astig. angle   {aberrations['phi12_deg']:.1f}°")
    return "\n".join(lines)


def _rotation_comparison(result, *, axsize: tuple[float, float], histogram: bool):
    """Explain a changed rotation using the recorded phases and their histograms."""
    if result.rotation_check_phases is None:
        raise ValueError("No rotation-check images are saved. Run ssb.find_aberrations(check_rotation=True, force=True).")
    difference = (result.scan_detector_rotation_deg - result.initial_scan_detector_rotation_deg + 180) % 360 - 180
    if np.isclose(difference, 0, rtol=0, atol=1e-8):
        return None

    import matplotlib.pyplot as plt
    from quantem.core.visualization import show_2d

    phases = result.rotation_check_phases
    titles = [f"Suggested rotation · {result.initial_scan_detector_rotation_deg:.2f}°",
              f"SSB-selected rotation · {result.scan_detector_rotation_deg:.2f}°"]
    width, height = axsize
    figure_height = height + (3.2 if histogram else 0) + 1.0
    figure = plt.figure(figsize=(2 * width, figure_height))
    figure.text(
        .04, 1 - .15 / figure_height,
        f"Suggested rotation: {result.initial_scan_detector_rotation_deg:.2f}°. "
        f"SSB selected {result.scan_detector_rotation_deg:.2f}° after testing the opposite branch and refitting.\n"
        "Negative phase asymmetry prompted the check. "
        + ("Positive asymmetry after refitting supports the bright-column assumption."
           if result.column_sign >= COLUMN_SIGN_MIN else
           "The refitted phase does not clearly support the bright-column assumption."),
        va="top", fontsize=12, linespacing=1.5,
    )
    grid = figure.add_gridspec(2 if histogram else 1, 2,
                              height_ratios=[height, 2.5] if histogram else [height],
                              left=.04, right=.97, bottom=.06,
                              top=1 - 1.05 / figure_height, wspace=.16, hspace=.15)
    axes = np.array([figure.add_subplot(grid[0, col]) for col in range(2)])
    sampling = result.object_sampling_A
    show_2d(list(phases), title=titles, figax=(figure, axes),
            norm={"interval_type": "manual", "vmin": result.phase_limits[0],
                  "vmax": result.phase_limits[1]}, cmap="inferno", cbar=True, title_fontsize=15,
            scalebar={"sampling": float(sampling[1]), "units": "Å", "fontsize": 13},
            tight_layout=False)
    if histogram:
        values = phases.astype(np.float64)
        values -= np.median(values, axis=(-2, -1), keepdims=True)
        limit = float(np.max(np.abs(values))) or 1.0
        edges = np.linspace(-limit, limit, 102)
        fractions = [np.histogram(phase, bins=edges)[0] / phase.size for phase in values]
        positive = np.concatenate([counts[counts > 0] for counts in fractions])
        histogram_axes = []
        for col, score in enumerate((result.initial_column_sign, result.column_sign)):
            axis = figure.add_subplot(grid[1, col])
            histogram_axes.append(axis)
            axis.stairs(fractions[col], edges, color="#8C1515", fill=True)
            axis.set(yscale="log", xlabel="Phase − median (rad)", ylabel="Fraction of pixels",
                     xlim=(-limit, limit), ylim=(positive.min() / 2, positive.max() * 2),
                     title=f"{'Before' if col == 0 else 'After'} · phase asymmetry {score:+.3f}")
            axis.axvline(0, color="black", linewidth=.7)
        histogram_axes[1].sharey(histogram_axes[0])
    plt.close(figure)
    return figure


def _phase_comparison(first, second, *, axsize: tuple[float, float]):
    """Show the same central region without notebook-side crop arithmetic."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from quantem.core.visualization import show_2d

    phases = [first.phase, second.phase]
    if any(phase.ndim != 2 for phase in phases):
        raise ValueError("Compare two individual reconstructions, not trial stacks.")
    sampling = [np.asarray(result.object_sampling_A, dtype=float) for result in (first, second)]
    shapes = [np.asarray(phase.shape) for phase in phases]
    if not all(np.all(np.isfinite(step) & (step > 0)) for step in sampling):
        raise ValueError("Both results need positive object_sampling_A in Å per pixel.")
    if not np.allclose(shapes[0] * sampling[0], shapes[1] * sampling[1]):
        raise ValueError("Compare results of the same physical scan field; their shape × sampling differs.")

    crop_shape = np.maximum(1, shapes[0] // 4)
    start = (shapes[0] - crop_shape) // 2
    bounds = np.stack((start, start + crop_shape))
    mapped = bounds * (sampling[0] / sampling[1])
    if not np.allclose(mapped, np.rint(mapped)):
        raise ValueError("Crop edges do not match the second grid. Call show(compare=...) on the coarser result.")
    all_bounds = [bounds, np.rint(mapped).astype(int)]
    crops = [phase[begin[0]:end[0], begin[1]:end[1]]
             for phase, (begin, end) in zip(phases, all_bounds, strict=True)]
    titles = [f"{result.upsampling_factor}× phase · full field (rad)" for result in (first, second)]
    crop_titles = [f"{result.upsampling_factor}× phase · marked crop (rad)" for result in (first, second)]
    if (first.sample_tilt_mrad is None) != (second.sample_tilt_mrad is None):
        labels = ["Tilt fixed at zero" if result.sample_tilt_mrad is None else "Tilt-corrected"
                  for result in (first, second)]
        titles = [f"{label} · {result.upsampling_factor}× phase (rad)"
                  for label, result in zip(labels, (first, second), strict=True)]
        crop_titles = [f"{label} · marked crop (rad)" for label in labels]
    bars = [{"sampling": step[1], "units": "Å", "fontsize": 13} for step in sampling]
    figure, axes = plt.subplots(2, 2, figsize=(2 * axsize[0], 2 * axsize[1]))
    figure.subplots_adjust(left=.015, right=.97, top=.95, bottom=.025, wspace=.18, hspace=.13)
    show_2d(
        [phases, crops], title=[titles, crop_titles], figax=(figure, axes),
        norm={"interval_type": "manual", "vmin": first.phase_limits[0], "vmax": first.phase_limits[1]},
        cmap="inferno", cbar=True, scalebar=[bars, bars], title_fontsize=16,
        tight_layout=False,
    )
    image_axes = [axis for axis in figure.axes if axis.images]
    for axis, (begin, end) in zip(image_axes[:2], all_bounds, strict=True):
        axis.add_patch(Rectangle(
            (begin[1] - .5, begin[0] - .5), end[1] - begin[1], end[0] - begin[0],
            fill=False, edgecolor="#00D5FF", linewidth=2,
        ))
    plt.close(figure)
    return figure
