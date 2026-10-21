"""Aberration fits on MPS: one objective for the standard (tilt=False) and the joint (tilt=True) fit.

Both maximize the least-squares agreement of the SSB model with G(q, k) (``thick_sample.thick_fit_batch``, the fused
Metal kernel) with the device-resident search of ``ssb._resident_search``. The standard fit holds the sample tilt and
depth spread at 0, where the model is ordinary SSB exactly. The phase-variance loss remains the reported diagnostic
(``SSBResult.loss``): it is evaluated once, at the fitted aberrations.
"""

import math
import time

import numpy as np

from quantem.gpu.ssb.brightfield import BrightfieldDisk
from quantem.gpu.ssb.mps.hardware import (
    default_object_redraw_chunk_bf,
    effective_phase_loss_chunk_bf,
    require_mlx,
)
from quantem.gpu.ssb.mps.kernels.object_sum import object_fourier_sum_dynamic
from quantem.gpu.ssb.mps.prepared import PreparedMpsSSB
from quantem.gpu.ssb.mps.reconstruct import reconstruct_prepared
from quantem.gpu.ssb.mps.thick_sample import (
    THICK_FIT_MAX_BATCH,
    thick_fit,
    thick_fit_batch,
)
from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.ssb.thick_sample_fit import fit_sample_search

# The standard fit's search box when search_ranges leaves a coefficient out: C10 +-40 nm, C12 up to 10 nm, phi12 over
# its period (engine keys and units).
STANDARD_RANGES = {"C10_A": (-400.0, 400.0), "C12_A": (0.0, 100.0), "phi12_deg": (-90.0, 90.0)}
# where a zero-trial refinement starts when the session has no aberrations
_DEFAULT_START = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}


def standard_bounds(search_ranges: dict | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(lower, upper) of (C10, C12, phi12) in Angstrom, Angstrom and rad from engine search ranges; a value holds."""
    ranges = {**STANDARD_RANGES, **{key: value for key, value in (search_ranges or {}).items() if value is not None}}
    lower, upper = [], []
    for key in ("C10_A", "C12_A", "phi12_deg"):
        value = ranges[key]
        low, high = (value, value) if np.ndim(value) == 0 else (float(value[0]), float(value[1]))
        if not (math.isfinite(float(low)) and math.isfinite(float(high)) and float(low) <= float(high)):
            raise ValueError(f"search_ranges takes a finite (low, high) with low <= high, or one value; got {value!r}.")
        scale = math.pi / 180.0 if key == "phi12_deg" else 1.0
        lower.append(float(low) * scale)
        upper.append(float(high) * scale)
    if lower[1] < 0.0:
        raise ValueError("C12_nm is a magnitude: its search range starts at 0 or above.")
    return tuple(lower), tuple(upper)


def newton_supported(search_ranges: dict | None, refine_lock: list[str] | None) -> bool:
    """Whether Newton refinement can honour these constraints (``fit_standard_resident``); else Nelder-Mead does."""
    lower, upper = standard_bounds(search_ranges)
    return not refine_lock and lower[1] == 0.0 and lower[2] <= -math.pi / 2 + 1e-9 and upper[2] >= math.pi / 2 - 1e-9


def optimize(
    prepared: PreparedMpsSSB,
    selection: BrightfieldDisk,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    aberrations: dict | None = None,
    search_ranges: dict | None = None,
    num_trials: int = 200,
    refine: str | None = "newton",
    refine_lock: list[str] | None = None,
    scan_detector_rotation_deg: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
) -> tuple[SSBResult, np.ndarray]:
    """Standard SSB fit of C10/C12/phi12 on Apple GPU, then the complex object and the phase at the fit.

    ``num_trials`` trials of the resident search maximize the agreement with the sample tilt and depth spread at 0,
    within ``search_ranges`` (engine keys; a value holds that coefficient in every trial); ``refine`` ("newton",
    "nelder-mead" or None) polishes from the best distinct trials, and from the session's ``aberrations`` when
    given. ``refine_lock`` (engine names) holds coefficients at the search's value through Nelder-Mead. Returns the
    result (phase-variance ``loss`` at the fit) and the float32 mean phase the session keeps for its first preview.
    """
    from quantem.gpu.ssb._resident_search import fit_standard_resident
    from quantem.gpu.ssb.mps.resident_fit import evaluate

    started = time.perf_counter()
    lower, upper = standard_bounds(search_ranges)
    start = None
    if aberrations:
        start = [float(aberrations[name]) for name in ("C10", "C12", "phi12")]
    elif int(num_trials) == 0:
        start = [_DEFAULT_START[name] for name in ("C10", "C12", "phi12")]
    if start is not None:
        # the start obeys the search box: a held coefficient is every trial's and the result's value
        start = [min(max(value, low), high) for value, low, high in zip(start, lower, upper, strict=True)]
    lock = tuple(sorted({"C10": 0, "C12": 1, "phi12": 2}[name] for name in (refine_lock or ())))
    fit = fit_standard_resident(
        lambda trial: evaluate(prepared, trial, band_inv_A), backend="mps", lower=lower, upper=upper,
        trials=int(num_trials), seed=int(seed), polish=refine, lock=lock, start=start,
    )
    best = {name: fit[name] for name in ("C10", "C12", "phi12")}
    timings = {"search_seconds": fit["timings"]["coarse_search_s"], "refinement_seconds": fit["timings"]["refinement_s"]}
    final_object_started = time.perf_counter()
    object_wave_mx = object_fourier_sum_dynamic(
        prepared, C10=best["C10"], C12=best["C12"], phi12=best["phi12"], chunk_bf=default_object_redraw_chunk_bf(),
    )
    mx = require_mlx()
    mx.eval(object_wave_mx)
    object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    timings["final_object_seconds"] = time.perf_counter() - final_object_started
    final_loss_started = time.perf_counter()
    _object_wave, loss, phase = reconstruct_prepared(
        prepared, C10=best["C10"], C12=best["C12"], phi12=best["phi12"],
        chunk_bf=effective_phase_loss_chunk_bf(16, prepared.scan_shape), compute_loss=True, compute_object=False,
    )
    timings["final_phase_loss_seconds"] = time.perf_counter() - final_loss_started
    if phase is None or loss is None:
        raise RuntimeError("MPS optimizer did not produce its final phase and loss.")
    records = [{"stage": "search", "number": record["number"], "objective": "negative_thick_agreement",
                "band_inv_A": list(band_inv_A),
                "params": {"C10_A": record["params"]["C10"], "C12_A": record["params"]["C12"],
                           "phi12_deg": math.degrees(record["params"]["phi12"])},
                "loss": record["loss"]} for record in fit["trial_records"]]
    result = SSBResult(
        object_wave=object_wave,
        backend="mps",
        aberrations=dict(best),
        scan_detector_rotation_deg=float(scan_detector_rotation_deg),
        loss=float(loss),
        sample_agreement=float(fit["fit"]),
        elapsed_s=time.perf_counter() - started,
        timings=timings,
        optimizer=dict(fit["optimizer"]),
        num_trials=int(num_trials),
        num_bright_field_pixels=selection.size,
        refinement=refine,
        refine_nfev=int(fit["nfev"]),
        refine_elapsed_s=timings["refinement_seconds"],
        voltage_kV=float(voltage_kV),
        semiangle_mrad=float(semiangle_mrad),
        object_sampling_A=tuple(float(value) for value in np.broadcast_to(scan_sampling_A, (2,))),
        bright_field_center_px=selection.bright_field_center_px,
        bright_field_radius_px=selection.bright_field_radius_px,
        bright_field_edge_radius_px=selection.bright_field_edge_radius_px,
        trial_records=records,
    )
    return result, np.asarray(phase, dtype=np.float32)


# =========================================================================
#  Thick-sample fit: sample tilt and thickness with the aberrations
# =========================================================================

def fit_sample(
    prepared: PreparedMpsSSB,
    *,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    **options,
) -> dict[str, object]:
    """Fit C10, C12, phi12, sample tilt and thickness by maximising ``thick_sample.thick_fit`` (search: ``ssb.thick_sample_fit``,
    shared with CUDA). Trials are evaluated in batches of ``THICK_FIT_MAX_BATCH`` with the fused Metal kernel
    (``thick_fit_batch``, same value as ``thick_fit``). Units: C10, C12, thickness in Angstrom (engine unit); tilt in mrad,
    scan frame (row, col)."""
    if options.get("polish", "newton") == "newton" or options.get("polish_starts", 3) == 0:
        from quantem.gpu.ssb._resident_search import fit_resident
        from quantem.gpu.ssb.mps.resident_fit import evaluate

        if options.get("polish_starts", 3) == 0:
            options = {**options, "polish": None}
        return fit_resident(lambda trial: evaluate(prepared, trial, band_inv_A),
                            backend="mps", band_inv_A=band_inv_A, **options)
    return fit_sample_search(
        lambda c10, c12, phi12, tilt, thickness: thick_fit(prepared, C10=c10, C12=c12, phi12=phi12, sample_tilt_mrad=tilt,
                                                           thickness=thickness, band_inv_A=band_inv_A),
        objective_batch=lambda rows: thick_fit_batch(prepared, rows, band_inv_A),
        batch_size=THICK_FIT_MAX_BATCH,
        batch_polish=True,
        band_inv_A=band_inv_A, **options)


__all__ = ["STANDARD_RANGES", "fit_sample", "newton_supported", "optimize", "standard_bounds"]
