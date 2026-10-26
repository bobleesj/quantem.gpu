"""Backend-neutral SSB fit, source, evaluation, and reconstruction results."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.display._layout import panel_figure, panel_inches, resolve_columns, scalebar
from quantem.gpu.optics.physics import wrap_degrees
from quantem.gpu.direct_ptycho.probe import model_probe, probe_grid

if TYPE_CHECKING:
    import pandas as pd

    from quantem.gpu.direct_ptycho.branches import RotationBranches

# SSB reconstructs the phase of the bright-field mean wave, ``phase_of_mean`` (``method="ssb"``). The older estimators
# were retired on 2026-10-25, each in favour of the regularized method named here. Saved results and records that name
# one still load as they were saved; asking for one raises an error naming its replacement.
RETIRED_PHASE_ESTIMATORS = {"mean_phase": "phase", "complex_wave": "complex"}


def check_phase_estimator(phase_estimator: str | None) -> None:
    """Accept ``None`` or ``"phase_of_mean"``; raise for any other estimator, naming the method that replaces it.

    Examples
    --------
    >>> check_phase_estimator("phase_of_mean")  # what method="ssb" computes
    """
    if phase_estimator in (None, "phase_of_mean"):
        return
    method = RETIRED_PHASE_ESTIMATORS.get(phase_estimator)
    if method == "phase":
        raise ValueError("phase_estimator='mean_phase' was retired. method='ssb' reconstructs the phase of the mean "
                         "wave; for the regularized phase reconstruction use method='phase'.")
    if method == "complex":
        raise ValueError("phase_estimator='complex_wave' was retired. Use method='complex', which reconstructs "
                         "amplitude and phase together.")
    raise ValueError(f"phase_estimator must be 'phase_of_mean' (what method='ssb' computes) or omitted; "
                     f"got {phase_estimator!r}.")


def retired_estimator_note(phase_estimator: object, record: str) -> str | None:
    """The one-line notice for a saved ``record`` whose image a retired estimator made; None otherwise."""
    method = RETIRED_PHASE_ESTIMATORS.get(phase_estimator) if isinstance(phase_estimator, str) else None
    if method is None:
        return None
    return (f"SSB: {record} was saved with the retired {phase_estimator} estimator and loads as saved; "
            f"run(method='ssb') reconstructs the phase of the mean wave, run(method='{method}') replaces "
            f"{phase_estimator}.")


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
        The fraction of G's power in the fit band (0.2-0.9 1/A) that the fitted model explains, in [0, 1]: the
        agreement sum_q |sum_k G conj(gamma)|^2 / sum_k |gamma|^2 over sum_q sum_k |G|^2, its upper bound. The
        low-order and fifth-order fits report the same quantity; noise and unmodelled structure keep it below 1.
    tilt_fit_gain : float | None
        The fit's agreement over the best standard-SSB agreement (tilt and depth spread 0) on the same data
        (> 1: the tilt explains more); None when ``compare_standard=False``.
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
    reconstruction_method : str
        ``ssb``, ``phase`` or ``complex``. The regularized methods are experimental.
    reconstruction_options : dict[str, object]
        Method, strengths, smoothness prior and detector weighting used by a
        regularized reconstruction; saved and restored with the result.
    reconstruction_diagnostics : dict[str, object]
        Model conventions and optional training prediction error. This is not
        the conventional phase-variance ``loss`` or an independent validation.
    phase_estimator : str | None
        ``phase_of_mean`` for SSB (the phase of the bright-field mean wave);
        a regularized result records its method, ``phase`` or ``complex``.
        None marks a backend fit's working image, which the session replaces
        with its SSB phase. Results saved with the retired ``mean_phase`` or
        ``complex_wave`` estimators load with that name.
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
        What ``find_aberrations`` maximized: ``"agreement"`` (the least-squares agreement of the SSB model with the
        data, with or without tilt) or ``"broad_fifth_agreement"`` (the same through fifth order); None for a result
        that was not fitted. Results saved by earlier versions may say ``"phase_variance"`` (a retired search).
    refinement : str | None
        Local refinement the fit applied after its search: ``"bfgs"`` (the fifth-order exact-gradient refinement),
        ``"newton"`` (the low-order trust-region Newton on the exact derivatives; a locked or bounded standard fit
        also gives ``"bfgs"``, the exact-gradient quasi-Newton), or None for none (the resolved method, never
        ``"auto"``); None for a result that was not fitted. Results saved by earlier versions may say
        ``"nelder-mead"``.
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
    reconstruction_method: str = "ssb"
    reconstruction_options: dict[str, object] = field(default_factory=dict)
    reconstruction_diagnostics: dict[str, object] = field(default_factory=dict)
    phase_estimator: str | None = None
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
    objective: str | None = None
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
        lines = ["SSB Result" if self.reconstruction_method == "ssb" else f"Direct {self.reconstruction_method} result"]
        lines.append(f"  Shape          {tuple(self.object_wave.shape)} · {self.upsampling_factor}x output")
        if self.phase_estimator is not None:
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
        return pd.DataFrame.from_records(deepcopy(rows)).set_index("trial")

    def report(self, *, columns: int = 1, to_dict: bool = False,
               show_convergence: bool = False) -> "pd.DataFrame | dict[str | int, object]":
        """Report fitted optics, geometry, objective and recorded timing.

        Parameters are rows, so ``fit.report()`` is readable directly in a
        notebook. ``pd.concat([a.report(), b.report()], axis=1)`` compares fits
        side by side. No reconstruction or fitting is performed.

        Parameters
        ----------
        columns : int, optional
            Number of parameter/value groups across a single-fit table.
            Use ``columns=2`` or ``columns=3`` for a shorter report. Read down
            each group, then across; every parameter retains its value and
            units. This layout option does not change dictionary exports or
            series trial tables.
        to_dict : bool, optional
            Return an independent dictionary instead of a pandas table.
            Coefficients are nested under ``aberrations`` with their public
            unit-bearing names. Geometry uses the public attribute names;
            ``sample_tilt_mrad`` is a (row, col) pair. For a series result,
            return the trial table as a dictionary indexed by trial ID.
        show_convergence : bool, optional
            Display the saved convergence figure in a notebook, then return
            the table. Uses the same recorded history as
            ``fit.show("convergence")``; does not refit. Cannot be combined
            with ``to_dict=True``. Missing history raises a helpful error.

        Returns
        -------
        pandas.DataFrame or dict
            Parameters with their units, arranged in the requested number of
            groups, or an unrounded dictionary of the same fitted parameters.
            A series returns its trial records. Only coefficients present in
            the result are listed.
            ``depth_spread_nm`` is effective model depth, not a measured
            specimen thickness. This report describes the result, not later
            widget edits, and does not replace ``save_calibration()``.

        Notes
        -----
        ``fit.aberrations`` remains the coefficient dictionary. The report
        adds the objective name, model agreement, variance diagnostic loss,
        search/refinement counts, stopping reason and timings when recorded.
        These objectives have different meanings and scales. A stopping
        reason does not establish physical convergence. Stage timings may
        overlap; they must not be summed to estimate total time. ``elapsed_s``
        describes the recorded operation, not notebook or acquisition IO time.

        Examples
        --------
        >>> fit = ptycho.find_aberrations()  # doctest: +SKIP
        >>> fit.report()  # doctest: +SKIP
        >>> fit.report(columns=2)  # doctest: +SKIP
        >>> parameters = fit.report(to_dict=True)  # doctest: +SKIP
        >>> parameters["aberrations"]["C30_nm"]  # doctest: +SKIP
        >>> fit.report(show_convergence=True)  # doctest: +SKIP
        """
        resolve_columns(1, columns=columns, default=1)
        if show_convergence:
            if to_dict:
                raise ValueError("Use report(to_dict=True) to export data, or report(show_convergence=True) "
                                 "to display the convergence figure and table.")
            from IPython.display import display

            display(self.show("convergence"))
        if self.object_wave.ndim == 3:
            table = self.trials
            return table.to_dict(orient="index") if to_dict else table
        aberrations = {name: self.aberrations[name]
                       for name in ("C10_nm", "C12_nm", "phi12_deg") if name in self.aberrations}
        aberrations.update(self.aberrations)
        record = {
            "aberrations": aberrations,
            "scan_detector_rotation_deg": self.scan_detector_rotation_deg,
        }
        if self.rotation_branches is not None:
            record["suggested_rotation_deg"] = self.rotation_branches.branches[0].scan_detector_rotation_deg
            record["rotation_branch_decided_by"] = self.rotation_branches.decided_by
        if self.rotation_refined:
            record["rotation_estimate_deg"] = self.rotation_estimate_deg
        if self.sample_tilt_mrad is not None:
            record["sample_tilt_mrad"] = tuple(self.sample_tilt_mrad)
        if self.depth_spread_nm is not None:
            record["depth_spread_nm"] = self.depth_spread_nm
        if self.objective is not None:
            record["objective"] = self.objective
        summary_fields = (
            ("backend", "Backend"),
            ("reconstruction_method", "Reconstruction method"),
            ("elapsed_s", "Elapsed (s)"),
            ("sample_agreement", "Model agreement"),
            ("loss", "Variance diagnostic loss"),
            ("num_trials", "Trials"),
            ("num_bright_field_pixels", "Bright-field pixels"),
            ("refinement", "Refinement"),
            ("refine_nfev", "Refinement evaluations"),
            ("refine_elapsed_s", "Refinement (s)"),
        )
        for name, _ in summary_fields:
            value = getattr(self, name)
            if value is not None:
                record[name] = value
        if self.optimizer.get("stop") is not None:
            record["stop_reason"] = self.optimizer["stop"]
        if self.timings:
            record["timings"] = dict(self.timings)
        if None not in (self.voltage_kV, self.semiangle_mrad, self.object_sampling_A):
            from quantem.gpu.optics import ssb_upsampling_requirement

            required = ssb_upsampling_requirement(
                voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
                scan_sampling_A=tuple(float(step) * self.upsampling_factor for step in self.object_sampling_A),
            )
            if self.upsampling_factor < required - 1e-12:
                record["upsampling_factor"] = self.upsampling_factor
                record["nominal_upsampling_requirement"] = required
        if to_dict:
            return record

        import pandas as pd

        row = {f"{name.rsplit('_', 1)[0]} ({name.rsplit('_', 1)[1]})": value
               for name, value in record["aberrations"].items()}
        if self.rotation_branches is not None:
            row["Suggested rotation (deg)"] = record["suggested_rotation_deg"]
        if self.rotation_refined:
            row["Estimated rotation (deg)"] = record["rotation_estimate_deg"]
        row["SSB-selected rotation (deg)"] = record["scan_detector_rotation_deg"]
        if self.rotation_branches is not None:
            row["180° branch by"] = {"columns": "atomic columns", "phase_tail": "phase tail"}[
                record["rotation_branch_decided_by"]]
        if self.sample_tilt_mrad is not None:
            row.update({"tilt row (mrad)": record["sample_tilt_mrad"][0],
                        "tilt col (mrad)": record["sample_tilt_mrad"][1]})
        if self.depth_spread_nm is not None:
            row["depth spread (nm)"] = record["depth_spread_nm"]
        if self.objective is not None:
            row["objective"] = record["objective"]
        row.update({label: record[name] for name, label in summary_fields if name in record})
        if "stop_reason" in record:
            row["Stopping reason"] = record["stop_reason"]
        row.update({f"Timing: {name} (s)": seconds for name, seconds in self.timings.items()})
        if "nominal_upsampling_requirement" in record:
            row["Output sampling (×)"] = record["upsampling_factor"]
            row["Nominal requirement (×)"] = record["nominal_upsampling_requirement"]
        model = "standard SSB" if self.sample_tilt_mrad is None else "tilt-aware SSB"
        table = pd.DataFrame.from_dict(row, orient="index", columns=[model]).rename_axis("Parameter")
        columns = resolve_columns(len(table), columns=columns, default=1)
        if columns == 1:
            return table
        height = (len(table) + columns - 1) // columns
        groups = {}
        for index in range(columns):
            start, stop = len(table) * index // columns, len(table) * (index + 1) // columns
            group = table.iloc[start:stop].rename(columns={model: "Value"}).reset_index()
            groups[f"Column {index + 1}"] = group.reindex(range(height), fill_value="")
        return _grouped_report()(pd.concat(groups, axis=1, names=[model]))

    def _repr_html_(self) -> str:
        """Render the parameter table once, without changing numerical values."""
        table = self.report()
        return table.to_html(float_format=lambda value: f"{value:.6g}", max_rows=None)

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
        >>> result = ssb.run(aberrations)
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
        from quantem.gpu.direct_ptycho.calibration import save_calibration

        return save_calibration(self, path)

    def show(self, view: Literal["phase", "probe", "rotation", "parameters", "depth", "tilt", "convergence"] = "phase", *,
             axsize: tuple[float, float] | None = None,
             compare: Self | Sequence[Self] | Mapping[object, Self] | None = None,
             x: Literal["auto", "time", "evaluations"] = "auto",
             show_parameters: bool = True,
             interactive: bool = False, show_fft: bool = False,
             size: Literal["small", "medium", "large", "xlarge"] | int | None = None,
             rows: int | None = None, columns: int | None = None,
             region: Literal["center", "top-left", "top-right", "bottom-left", "bottom-right"] | None = None,
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
        compare : SSBResult, sequence of SSBResult, or mapping, optional
            For convergence, a fit or labeled mapping of fits from the same
            data, calibration, region and objective. The current fit is included
            once. Different refinement methods are supported.
            For phase, one or more reconstructions of the same field of view. Shows
            all full images above matching zooms, with rectangles marking
            their locations. Each zoom spans one quarter of each dimension,
            selected on this result's grid using ``region``. All panels
            share this result's phase_limits; no alignment is performed.
        show_parameters : bool, optional
            Include saved objective slices beside a single phase image by
            default. This only displays recorded values; it runs no fitting
            or GPU evaluations. Use ``view="parameters"`` for the plots alone.
            Measure them first with ``ssb.measure_parameters(result)``.
        interactive : bool, optional
            Inspect saved phase images with QuantEM Show2D instead of Matplotlib.
            Comparisons retain full fields, marked zooms and optional crop FFTs.
            Pan, zoom and contrast controls do not rerun reconstruction. Use
            ``ptycho.show()`` for the live ShowPtycho aberration editor.
        show_fft : bool, optional
            Add the Fourier magnitude of each displayed zoom (or full image
            without a zoom). Mean subtraction and a Hann window reduce edge
            leakage. Magnitudes share one scale in dB relative to their common
            maximum and one physical frequency field; they are not FRC curves.
        size : {"small", "medium", "large", "xlarge"}, int or None, optional
            Per-panel width: small 3, medium 4.5, large 6 or xlarge 9 inches
            at the static 100-DPI reference, or a positive integer pixel width
            such as ``size=750``. Medium is the default. Plot aspect
            ratios are preserved. Changes only display size, never sampling
            or fitted values. Use either ``size`` or an explicit ``axsize``.
        rows, columns : int, optional
            Images down or across phase, probe or parameter galleries. Omit
            both to retain the view's existing layout. Rotation evidence
            keeps its fixed layout. Phase comparisons keep full fields, zooms
            and FFTs grouped by method.
            In phase comparisons, ``columns`` sets methods per row; ``rows``
            is not used. Interactive comparisons require matching pixel grids.
        region : {"center", "top-left", "top-right", "bottom-left", "bottom-right"} or None, optional
            Location of the marked zoom. None uses the center in comparisons
            and the full field for a single result. All methods show the same
            physical region below their full fields. This is a display
            selection, not a scan crop or refit.
        cmap : str, optional
            Image and landscape colormap, inferno by default.
        save : str or pathlib.Path, optional
            Export the complete static figure. Scientific results are unchanged.

        Returns
        -------
        matplotlib.figure.Figure or ipywidgets.Widget
            A calibrated figure, closed so a bare notebook expression renders
            once. Phase is in radians; real-space and Fourier-space probe
            scale bars use Å and mrad, respectively. Each probe has unit summed
            intensity and its own linear display range.

        Examples
        --------
        >>> result = ssb.run(aberrations, upsampling_factor="auto")
        >>> result.show()
        >>> result.show("probe")
        >>> result.show(size="large")
        >>> result.show(compare=result_4x, size="medium", region="top-left")
        >>> phase_001.show(compare=[phase_003, phase_005], cmap="gray")
        >>> phase_001.show(compare=[phase_003, phase_005], region="bottom-left",
        ...                interactive=True, show_fft=True, columns=3, size="large")
        >>> result.show("rotation")
        >>> ssb.measure_parameters(aberrations)
        >>> aberrations.show("parameters")
        >>> aberrations.show("depth", size="large")
        >>> aberrations.show("tilt", size="small")
        >>> aberrations.show("convergence")
        >>> fits[200].show("convergence", compare=fits)
        """
        import matplotlib.pyplot as plt

        if (region is not None or interactive or show_fft) and view != "phase":
            raise ValueError("interactive, show_fft and region apply to phase images; use show(view='phase', ...).")
        if interactive and (save is not None or axsize is not None):
            raise ValueError("Use size for an interactive view; use interactive=False with axsize or save for a figure.")
        if compare is not None and view not in ("phase", "convergence"):
            raise ValueError("Use compare with phase images or convergence traces.")
        if x != "auto" and view != "convergence":
            raise ValueError("x selects the convergence axis; use show('convergence', x=...).")
        if size is not None and axsize is not None:
            raise ValueError("Use size (a preset or pixel width) or axsize=(width, height), not both.")
        if ((compare is not None and rows is not None)
                or (view == "rotation" and (rows is not None or columns is not None))):
            raise ValueError("This view has a fixed comparison layout; omit rows. Phase comparisons support columns.")
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
        if view == "phase" and (compare is not None or region is not None or show_fft or interactive):
            others = [] if compare is None else ([compare] if isinstance(compare, SSBResult) else compare)
            if (not isinstance(others, Sequence) or (compare is not None and not others)
                    or not all(isinstance(result, SSBResult) for result in others)):
                raise ValueError("Compare phase images with show(compare=other) or show(compare=[second, third]).")
            figure = _phase_comparison(self, *others, axsize=axsize, cmap=cmap,
                                       region=("center" if compare is not None else None) if region is None else region,
                                       show_fft=show_fft, interactive=interactive, size=size or "medium",
                                       columns=columns, rows=rows)
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
            from quantem.gpu.direct_ptycho.convergence import plot_convergence

            figure = plot_convergence(self, axsize=axsize, rows=rows, columns=columns,
                                      compare=compare, x=x)
            if save is not None:
                figure.savefig(save, dpi="figure", bbox_inches="tight")
            return figure
        if view == "phase" and self.object_wave.ndim == 2:
            _announce_below_auto(self)
        if view in ("parameters", "depth", "tilt") or (view == "phase" and show_parameters and self.parameter_slices):
            from quantem.gpu.direct_ptycho.parameters import plot_parameters

            figure = plot_parameters(self, view="parameters" if view == "phase" else view,
                                     include_phase=view == "phase", axsize=axsize,
                                     rows=rows, columns=columns, cmap=cmap)
            if save is not None:
                figure.savefig(save, dpi="figure", bbox_inches="tight")
            return figure
        if view == "phase":
            column_sampling = float(self.object_sampling_A[1])
            arrays = host_array(self.phase)
            titles = _phase_title(self)
            bars = scalebar(column_sampling, "Å", arrays.shape[0], axsize[0])
            panel_size = (12, 12)
        elif view == "probe":
            arrays = [abs(self.probe()) ** 2, abs(self.probe(space="fourier")) ** 2]
            titles = ["Real-space model probe · intensity", "Fourier-space model probe · intensity"]
            bars = [scalebar(self.probe_sampling_A[1], "Å", arrays[0].shape[0], axsize[0]),
                    scalebar(self.probe_sampling_mrad[1], "mrad", arrays[1].shape[0], axsize[0])]
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
        from quantem.core.visualization import show_2d

        figure, _ = show_2d(
            arrays, title=titles,
            norm=({"interval_type": "manual", "vmin": self.phase_limits[0],
                   "vmax": self.phase_limits[1]} if view == "phase" else "minmax"),
            cmap=cmap, cbar=view == "phase",
            axsize=panel_size if axsize is None else axsize,
            title_fontsize=min(16, max(9, axsize[0] * 2.2)),
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
              f"ssb.run(result, upsampling_factor='auto') shows it at {auto}x.")


@cache
def _grouped_report():
    """DataFrame for ``report(columns=n)`` that draws a rule between parameter groups.

    The table stays an ordinary DataFrame for indexing, comparison and export;
    only its notebook HTML adds a left border on each group after the first,
    so neighbouring Parameter/Value pairs are not read across. Pandas is
    imported here because the module keeps it optional at import time.
    """
    import pandas as pd

    class GroupedReport(pd.DataFrame):
        @property
        def _constructor(self):
            return pd.DataFrame

        def _repr_html_(self) -> str:
            groups = self.columns.get_level_values(0)
            rules = [{"selector": f"th.col{index}, td.col{index}", "props": "border-left: 2px solid #9aa0a6;"}
                     for index in range(1, len(groups)) if groups[index] != groups[index - 1]]
            return self.style.set_table_styles(rules).to_html()

    return GroupedReport


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


def _phase_title(result: SSBResult, area: str = "Full field") -> str:
    """Name the measured channel, reconstruction method and recorded strengths."""
    method = result.reconstruction_method
    options = result.reconstruction_options
    if method == "ssb":
        heading = "Conventional SSB"
        detail = "No reconstruction regularization"
    else:
        heading = "Regularized phase-only" if method == "phase" else "Amplitude + phase"
        strength = options.get("phase_regularization")
        phase = "not recorded" if strength is None else f"{strength:g}"
        detail = f"Phase regularization {phase}"
        if method == "complex":
            strength = options.get("amplitude_regularization")
            amplitude = "not recorded" if strength is None else f"{strength:g}"
            detail = f"Regularization: phase {phase}, amplitude {amplitude}"
    return f"{heading}\n{detail}\n{area} · {result.upsampling_factor}× upsampling · phase (rad)"


def _phase_comparison(first, *others, axsize: tuple[float, float], cmap: str = "inferno",
                      region: str | None = "center", show_fft: bool = False,
                      interactive: bool = False, size: str | int = "medium",
                      columns: int | None = None, rows: int | None = None):
    """Show calibrated full fields, matching crops and optionally their spectra."""
    results = (first, *others)
    count = len(results)
    columns = resolve_columns(count, columns=columns, rows=rows, default=count)
    phases = [host_array(result.phase) for result in results]
    if any(phase.ndim != 2 for phase in phases):
        raise ValueError("Compare individual reconstructions, not trial stacks.")
    sampling = [np.asarray(result.object_sampling_A, dtype=float) for result in results]
    shapes = [np.asarray(phase.shape) for phase in phases]
    if not all(np.all(np.isfinite(step) & (step > 0)) for step in sampling):
        raise ValueError("All results need positive object_sampling_A in Å per pixel.")
    if not all(np.allclose(shapes[0] * sampling[0], shape * step)
               for shape, step in zip(shapes[1:], sampling[1:], strict=True)):
        raise ValueError("Compare results of the same physical scan field; their shape × sampling differs.")
    if interactive and not all(np.array_equal(shapes[0], shape) for shape in shapes[1:]):
        raise ValueError("Interactive comparisons need the same pixel grids; use matching upsampling or interactive=False.")

    all_bounds = []
    crops = phases
    if region is not None:
        crop_shape = np.maximum(1, shapes[0] // 4)
        anchors = {
            "center": (.5, .5), "top-left": (0, 0), "top-right": (0, 1),
            "bottom-left": (1, 0), "bottom-right": (1, 1),
        }
        if region not in anchors:
            raise ValueError(f"Choose region from {', '.join(anchors)}; got {region!r}.")
        start = ((shapes[0] - crop_shape) * anchors[region]).astype(int)
        bounds = np.stack((start, start + crop_shape))
        for step in sampling:
            mapped = bounds * (sampling[0] / step)
            if not np.allclose(mapped, np.rint(mapped)):
                raise ValueError("Crop edges do not match a comparison grid. Call show(compare=...) on the coarser result.")
            all_bounds.append(np.rint(mapped).astype(int))
        crops = [phase[begin[0]:end[0], begin[1]:end[1]]
                 for phase, (begin, end) in zip(phases, all_bounds, strict=True)]

    titles = [_phase_title(result) for result in results]
    bars = [scalebar(step[1], "Å", phase.shape[0], axsize[0]) for step, phase in zip(sampling, phases, strict=True)]
    groups = [phases]
    group_titles = [titles]
    group_bars = [bars]
    limits = [first.phase_limits]
    if region is not None:
        groups.append(crops)
        group_titles.append([_phase_title(result, area=f"{region.capitalize()} zoom") for result in results])
        group_bars.append([scalebar(step[1], "Å", crop.shape[0], axsize[0])
                           for step, crop in zip(sampling, crops, strict=True)])
        limits.append(first.phase_limits)
    if len({result.sample_tilt_mrad is None for result in results}) > 1:
        for labels in group_titles:
            for index, result in enumerate(results):
                labels[index] += "\nTilt fixed at zero" if result.sample_tilt_mrad is None else "\nTilt-corrected"
    if show_fft:
        spectra, frequency_sampling = _phase_spectra(crops, sampling)
        groups.append(spectra)
        area = "Crop" if region is not None else "Full-field"
        group_titles.append([
            "\n".join(_phase_title(result).splitlines()[:2])
            + f"\n{area} FFT · shared magnitude (dB)" for result in results
        ])
        group_bars.append([scalebar(frequency_sampling[1], "1/Å", spectrum.shape[0], axsize[0]) for spectrum in spectra])
        limits.append((-60., 0.))
    # Same method and strength read identically; name where each set of optics was fitted.
    if len({result.fit_scan_region for result in results}) > 1:
        for labels in group_titles:
            for index, result in enumerate(results):
                fitted = result.fit_scan_region
                labels[index] += ("\nOptics fitted on the full field" if fitted is None else
                                  f"\nOptics fitted on rows {fitted[0]}:{fitted[1]}, columns {fitted[2]}:{fitted[3]}")

    if interactive:
        from ipywidgets import VBox
        from quantem.widget import Show2D

        viewers = []
        for group, (frames, labels, bar, clim) in enumerate(zip(groups, group_titles, group_bars, limits, strict=True)):
            overlays = None
            if group == 0 and all_bounds:
                overlays = [[{"shape": "rect", "box": [float(value) for value in (begin - .5).tolist() + (end - .5).tolist()],
                              "color": "#00D5FF", "line_width": 2}]
                            for begin, end in all_bounds]
            step = frequency_sampling if show_fft and group == len(groups) - 1 else sampling[0]
            viewers.append(Show2D(
                frames, labels=labels, sampling=tuple(step), units=bar[0]["units"],
                columns=columns, size=size if isinstance(size, str) else "medium",
                panel_width_px=round(panel_inches(size) * 100),
                cmap=cmap, vmin=clim[0], vmax=clim[1],
                auto_contrast=False, link_contrast=True, link_zoom=True, link_pan=True,
                panel_overlays=overlays, show_panel_titles=True, show_scale_bar=True,
                controls_collapsed=True, display_bin=1, verbose=False,
                title="Saved crop FFT · mean removed · Hann window · shared 60 dB range"
                if show_fft and group == len(groups) - 1 else "Phase comparison · shared contrast (rad)",
            ))
        return viewers[0] if len(viewers) == 1 else VBox(viewers)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from textwrap import fill
    from quantem.core.visualization import show_2d

    blocks = (count + columns - 1) // columns
    total_rows = blocks * len(groups)
    heading = "Phase comparison · shared contrast (rad)"
    if show_fft:
        heading += "\nFFT: mean removed · Hann window · shared 60 dB magnitude range"
    heading = "\n".join(fill(line, width=max(20, int(columns * axsize[0] * 7)))
                        for line in heading.splitlines())
    title_lines = max(label.count("\n") + 1 for labels in group_titles for label in labels)
    figure, axes = panel_figure(total_rows, columns, *axsize, title_lines=title_lines,
                                top_in=.35 * len(heading.splitlines()) + .2)
    figure.suptitle(heading, fontsize=16, y=1 - .1 / figure.get_figheight(), va="top")
    for block, offset in enumerate(range(0, count, columns)):
        stop = min(offset + columns, count)
        for group, (frames, labels, bar, clim) in enumerate(zip(groups, group_titles, group_bars, limits, strict=True)):
            row_axes = axes[block * len(groups) + group]
            show_2d(
                frames[offset:stop], title=labels[offset:stop], figax=(figure, row_axes[:stop - offset]),
                norm={"interval_type": "manual", "vmin": clim[0], "vmax": clim[1]},
                cmap=cmap, cbar=True, scalebar=bar[offset:stop],
                title_fontsize=min(14, max(8, axsize[0] * 2.2)), tight_layout=False,
            )
            for axis in row_axes[stop - offset:]:
                axis.set_visible(False)
            if group == 0 and all_bounds:
                for axis, (begin, end) in zip(row_axes, all_bounds[offset:stop]):
                    axis.add_patch(Rectangle(
                        (begin[1] - .5, begin[0] - .5), end[1] - begin[1], end[0] - begin[0],
                        fill=False, edgecolor="#00D5FF", linewidth=2,
                    ))
    plt.close(figure)
    return figure


def _phase_spectra(phases: list[np.ndarray], sampling: list[np.ndarray]):
    """Compare window-normalized FFT magnitudes on the common physical band."""
    common_shape = np.min([phase.shape for phase in phases], axis=0)
    spectra = []
    for phase in phases:
        if min(phase.shape) < 3:
            raise ValueError("An FFT crop needs at least 3 pixels per side; use a larger image or omit region.")
        window = np.outer(np.hanning(phase.shape[0]), np.hanning(phase.shape[1]))
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2((phase - phase.mean()) * window))) / window.sum()
        # The physical crop is identical, so frequency bin spacing is identical.
        # Compare only frequencies sampled by every result, without interpolation.
        start = np.asarray(phase.shape) // 2 - common_shape // 2
        stop = start + common_shape
        spectra.append(spectrum[start[0]:stop[0], start[1]:stop[1]])
    peak = max(float(spectrum.max()) for spectrum in spectra)
    peak = peak if peak > 0 else 1.
    spectra = [20 * np.log10(np.maximum(spectrum / peak, 1e-3)) for spectrum in spectra]
    frequency_sampling = 1 / (np.asarray(phases[0].shape) * sampling[0])
    return spectra, frequency_sampling
