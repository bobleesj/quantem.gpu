"""Aberration fits on MPS: the model agreement, tilt=True and tilt=False.

Both maximize the least-squares agreement of the SSB model with G(q, k) (the fused Metal kernel) with the
device-resident search of ``direct_ptycho._resident_search`` and the trust-region Newton on the exact derivatives: the standard
fit (``optimize``) holds the sample tilt and depth spread at 0, where the model is ordinary SSB exactly, and the joint
fit (``fit_sample``) fits them. The phase variance at the fit is ``SSBResult.loss``, a diagnostic.
"""

import math
import time

import numpy as np

from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk
from quantem.gpu.direct_ptycho.mps.hardware import (
    default_object_redraw_chunk_bf,
    effective_phase_loss_chunk_bf,
    require_mlx,
)
from quantem.gpu.direct_ptycho.mps.kernels.object_sum import object_fourier_sum_dynamic
from quantem.gpu.direct_ptycho.mps.prepared import PreparedMpsSSB
from quantem.gpu.direct_ptycho.mps.reconstruct import reconstruct_prepared
from quantem.gpu.direct_ptycho.results import SSBResult

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
    """Whether the Newton can honour these constraints (``fit_standard_resident``); else the exact-gradient
    quasi-Newton in (C10, C12, phi12) does."""
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
    compute_loss: bool = True,
) -> tuple[SSBResult, np.ndarray | None]:
    """Standard SSB fit of C10/C12/phi12 on Apple GPU, then the complex object and the phase at the fit.

    ``num_trials`` trials of the resident search maximize the agreement with the sample tilt and depth spread at 0,
    within ``search_ranges`` (engine keys; a value holds that coefficient in every trial); ``refine`` ("newton",
    "bfgs" or None) polishes from the best distinct trials, and from the session's ``aberrations`` when
    given. ``refine_lock`` (engine names) holds coefficients at the search's value through the refinement. Returns the
    result (phase-variance ``loss`` at the fit) and the float32 mean phase the session keeps for its first preview.
    ``compute_loss=False`` omits that diagnostic and its mean-phase image; the object and fit are unchanged.
    """
    from quantem.gpu.direct_ptycho._resident_search import fit_standard_resident
    from quantem.gpu.direct_ptycho.mps.resident_fit import evaluate

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
    from quantem.gpu.direct_ptycho.mps.exact_derivatives import LowOrderDerivatives

    fit = fit_standard_resident(
        lambda trial: evaluate(prepared, trial, band_inv_A), backend="mps", lower=lower, upper=upper,
        trials=int(num_trials), seed=int(seed), polish=refine, lock=lock, start=start,
        derivatives=None if refine is None else LowOrderDerivatives(prepared, band_inv_A),
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
    loss, phase = None, None
    if compute_loss:
        _object_wave, loss, phase = reconstruct_prepared(
            prepared, C10=best["C10"], C12=best["C12"], phi12=best["phi12"],
            chunk_bf=effective_phase_loss_chunk_bf(16, prepared.scan_shape), compute_loss=True, compute_object=False,
        )
    timings["final_phase_loss_seconds"] = time.perf_counter() - final_loss_started
    if compute_loss and (phase is None or loss is None):
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
        loss=None if loss is None else float(loss),
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
    return result, None if phase is None else np.asarray(phase, dtype=np.float32)


# =========================================================================
#  Thick-sample fit: sample tilt and thickness with the aberrations
# =========================================================================

def fit_sample(
    prepared: PreparedMpsSSB,
    *,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    **options,
) -> dict[str, object]:
    """Fit C10, C12, phi12, sample tilt and thickness by maximising the joint agreement with the fused Metal kernel:
    the resident search (``_resident_search.fit_resident``) and the trust-region Newton on the exact derivatives
    (``LowOrderDerivatives``). Units: C10, C12, thickness in Angstrom (engine unit); tilt in mrad, scan frame (row,
    col)."""
    from quantem.gpu.direct_ptycho._resident_search import fit_resident
    from quantem.gpu.direct_ptycho.mps.exact_derivatives import LowOrderDerivatives
    from quantem.gpu.direct_ptycho.mps.resident_fit import evaluate

    if options.get("polish_starts", 3) == 0:
        options = {**options, "polish": None}
    return fit_resident(lambda trial: evaluate(prepared, trial, band_inv_A), backend="mps", band_inv_A=band_inv_A,
                        derivatives=LowOrderDerivatives(prepared, band_inv_A), **options)


__all__ = ["STANDARD_RANGES", "fit_sample", "newton_supported", "optimize", "standard_bounds"]
