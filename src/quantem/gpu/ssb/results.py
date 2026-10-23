"""Backend-neutral SSB fit, source, evaluation, and reconstruction results."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.display._layout import panel_inches, resolve_columns
from quantem.gpu.optics.physics import wrap_degrees
from quantem.gpu.ssb.probe import model_probe, probe_grid

if TYPE_CHECKING:
    from quantem.gpu.ssb.branches import RotationBranches

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
    sample_agreement : float | None
        Final thick-sample objective, comparable between starts on the same
        data, region, band and calibration. Not normalized across regions.
    tilt_fit_gain : float | None
        Least-squares agreement with the data of the tilt fit relative to standard SSB (> 1: the tilt explains more).
    scan_detector_rotation_deg : float
        Physical scan-to-detector rotation in degrees, in [0, 360). The CoM curl fixes it only modulo 180 degrees;
        for an estimated rotation ``find_aberrations`` fits both branches and keeps one (``rotation_branches``).
    scan_detector_rotation_flipped : bool
        Whether the detector axes were exchanged before applying the rotation.
    rotation_branches : RotationBranches | None
        Both 180-degree branches, fitted independently, with the atomic-column and phase-tail evidence and the branch
        kept; ``show("rotation")`` draws it. None when no comparison ran (a given or recorded rotation).
    rotation_refined : bool
        Whether ``find_aberrations(refine_rotation=True)`` refined the estimated rotation; ``scan_detector_rotation_deg``
        is then the refined angle.
    rotation_estimate_deg : float | None
        The estimated rotation (on the kept 180-degree branch) the refinement started from; None when not refined.
    rotation_refinement : dict | None
        The refinement's record: ``estimate_deg``, ``refined_deg``, the agreement (``estimate_agreement``,
        ``refined_agreement``) and loss at both, the angles evaluated, ``evaluations`` and ``seconds``.
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
    optimizer : dict[str, object]
        Numerical search/refinement device, precision, and Newton diagnostics.
    objective : str | None
        What ``find_aberrations`` optimized: ``"agreement"`` (the least-squares agreement of the SSB model with the
        data, the default, with or without tilt) or ``"phase_variance"`` (the older SSB search); None for a result
        that was not fitted.
    refinement : str | None
        Local refinement the fit applied after its search: ``"nelder-mead"``, ``"newton"``, or None for none (the
        resolved method, never ``"auto"``); None for a result that was not fitted.
    reused : bool
        Whether the result was restored from an exact saved-result match.
    saved_path : pathlib.Path | None
        Array artifact used for persistence, when ``save_dir`` was provided.
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
    scan_detector_rotation_flipped: bool = False
    loss: float | None = None
    elapsed_s: float | None = None
    timings: dict[str, float] = field(default_factory=dict)
    optimizer: dict[str, object] = field(default_factory=dict)
    num_trials: int | None = None
    num_bright_field_pixels: int | None = None
    refinement: str | None = None
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
    # Both 180-degree branches of an estimated rotation, fitted independently, and the one kept (branches.py).
    rotation_branches: "RotationBranches | None" = None
    phase_clim: tuple[float, float] | None = None
    fit_scan_region: tuple[int, int, int, int] | None = None
    reused: bool = False
    saved_path: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    parameter_slices: dict[str, object] | None = None
    # Small immutable-at-fit acquisition snapshot for portable calibration export; no GPU arrays/session reference.
    calibration_context: dict[str, object] | None = field(default=None, repr=False)
    # Local descriptors for deferred file fingerprinting, excluded from saved result metadata.
    _calibration_source: dict[str, object] | None = field(default=None, repr=False, compare=False)
    sample_agreement: float | None = None
    objective: Literal["agreement", "phase_variance"] | None = None
    # find_aberrations(refine_rotation=True): the estimate it started from and the search (rotation_refinement.py)
    rotation_refined: bool = False
    rotation_estimate_deg: float | None = None
    rotation_refinement: dict[str, object] | None = None

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
        if self.objective is not None:
            lines.append(f"  Objective      {self.objective}")
        if self.num_trials is not None:
            lines.append(f"  Trials         {self.num_trials}")
        lines.append(f"  SSB rotation   {self.scan_detector_rotation_deg:.2f}°")
        if self.rotation_refined:
            lines.append(f"  Refined from   {self.rotation_estimate_deg:.2f}° (estimate)")
        if self.rotation_branches is not None:
            lines.append(f"  180° branch    {self.rotation_branches.summary}")
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
        """One-row table of the fitted parameters and the fit's ``objective``; ``pd.concat([a.report(), b.report()])``
        compares fits side by side."""
        import pandas as pd

        if self.object_wave.ndim == 3:
            return self.trials
        aberrations = self.aberrations
        row = {
            "C10 (nm)": aberrations.get("C10_nm"),
            "C12 (nm)": aberrations.get("C12_nm"),
            "phi12 (deg)": aberrations.get("phi12_deg", 0.0),
        }
        if self.rotation_branches is not None:
            row["Suggested rotation (deg)"] = self.rotation_branches.branches[0].scan_detector_rotation_deg
        if self.rotation_refined:
            row["Estimated rotation (deg)"] = self.rotation_estimate_deg
        row["SSB-selected rotation (deg)"] = self.scan_detector_rotation_deg
        if self.rotation_branches is not None:
            row["180° branch by"] = {"columns": "atomic columns", "phase_tail": "phase tail"}[self.rotation_branches.decided_by]
        if self.sample_tilt_mrad is not None:
            row.update({"tilt row (mrad)": self.sample_tilt_mrad[0],
                        "tilt col (mrad)": self.sample_tilt_mrad[1],
                        "depth spread (nm)": self.depth_spread_nm})
        if self.objective is not None:
            row["objective"] = self.objective
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
        limit = float(abs(self.phase).max()) or 1.0
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

    def save_calibration(self, path: str | Path = "calibration.json") -> Path:
        """Save this fitted model for reuse in ShowPtycho without fitting again.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            JSON destination. Parent directories are created. An existing file
            is replaced atomically. The default is ``calibration.json``.

        Returns
        -------
        pathlib.Path
            Written calibration, including all fitted aberrations, rotation,
            tilt/depth, sampling and portable acquisition identity. ShowPtycho
            checks that identity before restoring it and uses Auto sampling.

        Notes
        -----
        Exports this result's values, not subsequent widget edits. No image,
        model probe or objective is computed. File fingerprinting and JSON IO
        happen here, outside fitting. Requires a calibrated file-backed fit;
        array inputs cannot establish a persistent acquisition identity.
        Save widget edits with the widget's ``save_calibration()`` instead.

        Examples
        --------
        >>> fit = ssb.find_aberrations(search="broad")
        >>> fit.show("convergence")
        >>> fit.save_calibration("ssb-results/calibration.json")
        """
        from quantem.gpu.ssb.calibration import save_calibration

        return save_calibration(self, path)

    def show(self, view: Literal["phase", "probe", "rotation", "parameters", "depth", "tilt", "convergence"] = "phase", *,
             axsize: tuple[float, float] | None = None, compare: Self | Mapping[object, Self] | None = None,
             x: Literal["auto", "time", "evaluations"] = "auto",
             show_parameters: bool = True,
             size: Literal["small", "medium", "large", "xlarge"] | None = None,
             rows: int | None = None, columns: int | None = None,
             cmap: str = "inferno", save: str | Path | None = None):
        """Show phase, probes or fit diagnostics at a named panel size.

        Phase views reuse symmetric linear limits from their fitted result.
        Probe intensities use their full linear range. Plotting does not change the
        reconstructed wave or model probe.

        Parameters
        ----------
        view : {"phase", "probe", "rotation", "parameters", "depth", "tilt", "convergence"}, optional
            Show the reconstructed phase alone, or two model-probe intensity
            panels side by side. Fourier intensity shows the aperture;
            aberrations are encoded in the complex Fourier probe's phase.
            ``rotation`` draws the 180-degree branch evidence of an estimated
            rotation (``rotation_branches.show()``): both branches' fits at
            ``upsampling_factor="auto"`` above their native phase tails and
            phase at the dark-field atomic columns. A phase view coarser than
            ``"auto"`` (a native fit) is shown as reconstructed, and one line
            names the auto factor.
            ``depth`` shows the conditional focus-depth agreement slice;
            ``tilt`` shows its row-column tilt slice. The marker is the
            fitted point. Neither is a spatial parameter map or an uncertainty
            estimate; other parameters are held at their jointly fitted values.
            ``convergence`` plots recorded search and refinement evaluations.
            Different objectives and rotation searches remain separate. A missing
            refinement trace is stated explicitly; no evaluations are rerun.
        axsize : tuple[float, float] or None, optional
            Advanced explicit panel dimensions in inches, instead of size.
        x : {"auto", "time", "evaluations"}, optional
            Convergence axis. Auto uses recorded elapsed time when every fit
            has it, otherwise evaluations. Time starts before the primary fit;
            loading and plotting are excluded. Missing times are not estimated.
        compare : SSBResult or mapping, optional
            For convergence, a fit or labeled mapping of fits from the same
            data, calibration, region and objective. The current fit is included
            once. Different refinement methods are supported.
            For phase, a second phase reconstruction of the same field of view. Shows
            both full images above matching central crops, with rectangles
            marking the crop locations. The crop is the central quarter of
            each dimension, selected on this result's grid. All four panels
            share this result's phase_limits; no alignment is performed.
        show_parameters : bool, optional
            Include saved objective slices beside a single phase image by
            default. This only displays recorded values; it runs no fitting
            or GPU evaluations. Use ``view="parameters"`` for the plots alone.
            Measure them first with ``ssb.measure_parameters(result)``.
        size : {"small", "medium", "large", "xlarge"} or None, optional
            Per-panel width: small 3, medium 4.5, large 6 or xlarge 9 inches
            at the static 100-DPI reference. Medium is the default. Plot aspect
            ratios are preserved. Changes only display size, never sampling
            or fitted values. Use either ``size`` or an explicit ``axsize``.
        rows, columns : int, optional
            Images down or across phase, probe or parameter galleries. Omit
            both to retain the view's existing layout. Rotation evidence and
            ``compare`` keep their fixed scientific comparison layout.
        cmap : str, optional
            Image and landscape colormap, inferno by default.
        save : str or pathlib.Path, optional
            Export the complete static figure. Scientific results are unchanged.

        Returns
        -------
        matplotlib.figure.Figure
            A calibrated figure, closed so a bare notebook expression renders
            once. Phase is in radians; real-space and Fourier-space probe
            scale bars use Å and mrad, respectively. Each probe has unit summed
            intensity and its own linear display range.

        Examples
        --------
        >>> result = ssb.reconstruct(aberrations, upsampling_factor="auto")
        >>> result.show()
        >>> result.show("probe")
        >>> result.show(size="large")
        >>> result.show(compare=result_4x, size="medium")
        >>> result.show("rotation")
        >>> ssb.measure_parameters(aberrations)
        >>> aberrations.show("parameters")
        >>> aberrations.show("depth", size="large")
        >>> aberrations.show("tilt", size="small")
        >>> aberrations.show("convergence")
        >>> fits[200].show("convergence", compare=fits)
        """
        import matplotlib.pyplot as plt
        from quantem.core.visualization import show_2d

        if x != "auto" and view != "convergence":
            raise ValueError("x selects the convergence axis; use show('convergence', x=...).")
        if size is not None and axsize is not None:
            raise ValueError("Use size='small', 'medium', 'large' or 'xlarge', or axsize=(width, height), not both.")
        if (compare is not None or view == "rotation") and (rows is not None or columns is not None):
            raise ValueError("Rotation and compare views have a fixed comparison layout; omit rows and columns.")
        if axsize is None:
            width = panel_inches("medium" if size is None else size)
            if view == "convergence":
                panel_size = (6, 4)
            elif compare is not None or view == "probe":
                panel_size = (6, 6)
            elif view == "rotation":
                panel_size = (5, 3.4)
            elif view in ("parameters", "depth", "tilt") or (show_parameters and self.parameter_slices):
                panel_size = (5.2, 4.5)
            else:
                panel_size = (12, 12)
            axsize = (width, width * panel_size[1] / panel_size[0])
        if compare is not None and view != "convergence":
            if view != "phase" or not isinstance(compare, SSBResult):
                raise ValueError("Use result.show(compare=other) to compare phase images.")
            figure = _phase_comparison(self, compare, axsize=axsize, cmap=cmap)
            if save is not None:
                figure.savefig(save, dpi="figure", bbox_inches="tight")
            return figure
        if view == "rotation":
            if self.rotation_branches is None:
                raise ValueError(
                    "This result has no 180-degree branch comparison: its rotation was given or recorded, or "
                    "check_rotation=False. Run ssb.compare_rotation_branches() to fit and compare both branches.")
            return self.rotation_branches.show(axsize=axsize, cmap=cmap, save=save)
        if view == "convergence":
            from quantem.gpu.ssb.convergence import plot_convergence

            figure = plot_convergence(self, axsize=axsize, rows=rows, columns=columns,
                                      compare=compare, x=x)
            if save is not None:
                figure.savefig(save, dpi="figure", bbox_inches="tight")
            return figure
        if view == "phase" and self.object_wave.ndim == 2:
            _announce_below_auto(self)
        if view in ("parameters", "depth", "tilt") or (view == "phase" and show_parameters and self.parameter_slices):
            from quantem.gpu.ssb.parameters import plot_parameters

            figure = plot_parameters(self, view="parameters" if view == "phase" else view,
                                     include_phase=view == "phase", axsize=axsize,
                                     rows=rows, columns=columns, cmap=cmap)
            if save is not None:
                figure.savefig(save, dpi="figure", bbox_inches="tight")
            return figure
        if view == "phase":
            column_sampling = float(self.object_sampling_A[1])
            arrays = host_array(self.phase)
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
            raise ValueError(f"Choose view='phase', 'probe', 'rotation', 'parameters', 'depth', 'tilt' or 'convergence'; "
                             f"got {view!r}.")
        if rows is not None or columns is not None:
            frames = arrays if isinstance(arrays, list) else ([arrays] if arrays.ndim == 2 else list(arrays))
            count = len(frames)
            columns = resolve_columns(count, rows=rows, columns=columns, default=count)
            arrays = [frames[index:index + columns] for index in range(0, count, columns)]
            if isinstance(titles, list):
                titles = [titles[index:index + columns] for index in range(0, count, columns)]
            if isinstance(bars, list):
                bars = [bars[index:index + columns] for index in range(0, count, columns)]
        figure, _ = show_2d(
            arrays, title=titles,
            norm=({"interval_type": "manual", "vmin": self.phase_limits[0],
                   "vmax": self.phase_limits[1]} if view == "phase" else "minmax"),
            cmap=cmap, cbar=view == "phase",
            axsize=panel_size if axsize is None else axsize, title_fontsize=16,
            scalebar=bars,
        )
        if save is not None:
            figure.savefig(save, dpi="figure", bbox_inches="tight")
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


def phase_skewness(phase: object) -> float:
    """Mean-centred third-moment ratio of the phase, ``mean(p**3) / mean(p**2)**1.5`` with ``p = phase - mean(phase)``.

    The phase-tail test of the 180-degree branch choice (``SSB.compare_rotation_branches``): positive when a few bright
    atomic columns sit over a flatter background. Unlike ``column_sign`` it centres on the mean, not the median.

    Examples
    --------
    >>> phase_skewness(np.array([0.0, 0.0, 0.0, 1.0]))
    1.1547005383792515
    """
    # float64 so the third moment of a large image is not rounded away
    values = host_array(phase).astype(np.float64).ravel()
    values = values - values.mean()
    variance = float(np.mean(values * values))
    if variance <= 0.0:
        return 0.0
    return float(np.mean(values ** 3) / variance ** 1.5)


def column_histogram(
    phase: object, *, limit: float | None = None, bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the median-centered phase distribution used by rotation plots.

    Parameters
    ----------
    phase : array-like
        Finite phase samples in radians. Host-side report reduction, not a
        reconstruction or an accelerator hot-path operation.
    limit : float, optional
        Positive symmetric axis half-width in radians. By default use the
        99.9th percentile of absolute median-centered phase, as in the notebook.
        A constant image uses a one-radian half-width for a nondegenerate axis.
    bins : int, optional
        Number of equal-width bins, by default 100.

    Returns
    -------
    fractions, edges : tuple of numpy.ndarray
        Fractions of all input pixels and bin edges, in NumPy histogram order.
        Samples outside the axis remain in the denominator, not renormalized
        away. Reversing the fractions gives the mirrored *distribution*, not
        an opposite-rotation reconstruction.

    Examples
    --------
    >>> fractions, edges = column_histogram(np.array([0., 0., 0., 1.]))
    """
    values = host_array(phase)
    if np.iscomplexobj(values) or not values.size or not np.isfinite(values).all():
        raise ValueError("Phase must contain finite real samples. Select a complete saved phase image.")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    values = values.astype(np.float64).ravel()
    values = values - np.median(values)
    if limit is None:
        limit = float(np.percentile(np.abs(values), 99.9)) or 1.0
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("limit must be a positive finite phase half-width in radians.")
    edges = np.linspace(-limit, limit, bins + 1)
    return np.histogram(values, edges)[0] / values.size, edges


def host_array(value: object) -> np.ndarray:
    """Return a CuPy or NumPy array as a NumPy array.

    Histograms, figures and saved files are made on the host, while CUDA
    results keep their arrays on the device until one of those needs them.
    """
    return cp.asnumpy(value) if _is_cupy_array(value) else np.asarray(value)


def _is_cupy_array(value: object) -> bool:
    """Return whether *value* is a CuPy array without requiring CUDA."""

    return cp is not None and isinstance(value, cp.ndarray)


def _announce_below_auto(result: SSBResult) -> None:
    """Say in one line when a shown phase is coarser than ``upsampling_factor="auto"`` and how to get the auto one."""
    from quantem.gpu.optics.physics import ssb_upsampling_factor

    if None in (result.voltage_kV, result.semiangle_mrad, result.object_sampling_A):
        return
    try:
        auto = ssb_upsampling_factor(
            voltage_kV=result.voltage_kV, semiangle_mrad=result.semiangle_mrad,
            scan_sampling_A=tuple(float(step) * result.upsampling_factor for step in result.object_sampling_A))
    except ValueError:
        return
    if result.upsampling_factor < auto:
        print(f"SSB: showing this phase at {result.upsampling_factor}x, the grid it was reconstructed on; "
              f"ssb.reconstruct(result, upsampling_factor='auto') shows it at {auto}x.")


def _format_aberrations(aberrations: dict) -> str:
    """Format SSB aberration dict as aligned key-value lines."""
    lines = []
    if "C10_nm" in aberrations:
        lines.append(f"  Defocus (C10)  {aberrations['C10_nm']:.1f} nm")
    if "C12_nm" in aberrations:
        lines.append(f"  Astigmatism    {aberrations['C12_nm']:.1f} nm")
    if "phi12_deg" in aberrations:
        lines.append(f"  Astig. angle   {aberrations['phi12_deg']:.1f}°")
    for name, label in (("C21", "Coma (C21)"), ("C23", "Trefoil (C23)")):
        if f"{name}_nm" in aberrations:
            lines.append(f"  {label:14s} {aberrations[f'{name}_nm']:.1f} nm at "
                         f"{aberrations.get(f'phi{name[1:]}_deg', 0.0):.1f}°")
    if "C30_nm" in aberrations:
        lines.append(f"  Spherical (C30) {aberrations['C30_nm'] / 1000:.2f} µm")
    return "\n".join(lines)


def _phase_comparison(first, second, *, axsize: tuple[float, float], cmap: str = "inferno"):
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
        cmap=cmap, cbar=True, scalebar=[bars, bars], title_fontsize=16,
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
