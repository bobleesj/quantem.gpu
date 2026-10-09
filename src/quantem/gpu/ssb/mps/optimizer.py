"""Exact full-BF aberration optimization for the MPS backend."""

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
from quantem.gpu.ssb.mps.loss import reconstruct_prepared_batch_exact_loss
from quantem.gpu.ssb.mps.prepared import PreparedMpsSSB
from quantem.gpu.ssb.mps.reconstruct import reconstruct_prepared
from quantem.gpu.ssb.mps.thick_sample import (
    THICK_FIT_MAX_BATCH,
    thick_fit,
    thick_fit_batch,
)
from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.ssb.thick_sample_fit import fit_sample_search


def optimize(
    prepared: PreparedMpsSSB,
    selection: BrightfieldDisk,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    aberrations: dict | None = None,
    search_ranges: dict | None = None,
    n_trials: int = 200,
    refine: str | None = "nelder-mead",
    refine_lock: list[str] | None = None,
    rotation_angle_deg: float = 0.0,
    chunk_bf: int = 16,
    optuna_batch_size: int = 2,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[SSBResult, np.ndarray]:
    """Free-fit C10/C12/phi12 on Apple GPU, then reconstruct the best SSB phase.

    Every candidate uses the same full active-BF phase-variance loss as the final reconstruction, on the session's
    prepared evidence (``MpsSSBBackend``), so a fit and the previews before and after it read the same ``G_qk``. Returns
    the result and the final float32 mean phase, which the session keeps for its first preview.
    """
    import optuna

    started = time.perf_counter()
    timings: dict[str, float] = {}
    scan_shape = prepared.scan_shape
    fit_chunk_bf = effective_phase_loss_chunk_bf(max(1, int(chunk_bf)), scan_shape)

    start = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}
    if aberrations:
        start.update({name: float(value) for name, value in aberrations.items() if name in start})
    ranges = _ranges_from_start(start, search_ranges)
    trials: list[dict] = []

    def evaluate(C10: float, C12: float, phi12: float) -> float:
        loss = reconstruct_prepared_batch_exact_loss(
            prepared,
            C10=np.asarray([C10], dtype=np.float32),
            C12=np.asarray([C12], dtype=np.float32),
            phi12=np.asarray([phi12], dtype=np.float32),
            chunk_bf=fit_chunk_bf,
        )[0]
        return float(loss)

    def evaluate_batch(params: list[dict[str, float]]) -> np.ndarray:
        c10 = np.asarray([candidate["C10"] for candidate in params], dtype=np.float32)
        c12 = np.asarray([candidate["C12"] for candidate in params], dtype=np.float32)
        phi = np.asarray([candidate["phi12"] for candidate in params], dtype=np.float32)
        return reconstruct_prepared_batch_exact_loss(
            prepared,
            C10=c10,
            C12=c12,
            phi12=phi,
            chunk_bf=fit_chunk_bf,
        )

    best = dict(start)
    initial_started = time.perf_counter()
    best_loss = evaluate(best["C10"], best["C12"], best["phi12"])
    timings["initial_loss_seconds"] = time.perf_counter() - initial_started
    trials.append({"stage": "initial", "params": dict(best), "loss": best_loss})

    optuna_started = time.perf_counter()
    if n_trials > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
        )

        from tqdm.auto import tqdm

        n_completed = 0
        batch_size = max(1, int(optuna_batch_size))
        progress = tqdm(
            total=int(n_trials),
            desc="SSB optimize",
            disable=not verbose,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}]"
            ),
        )
        while n_completed < int(n_trials):
            current = min(batch_size, int(n_trials) - n_completed)
            trial_records = [study.ask() for _ in range(current)]
            trial_params = []
            for trial in trial_records:
                C10 = _suggest_or_fixed(trial, ranges, "C10_nm", best["C10"])
                C12 = _suggest_or_fixed(trial, ranges, "C12_nm", best["C12"])
                phi12 = math.radians(_suggest_or_fixed(
                    trial, ranges, "phi12_deg", math.degrees(best["phi12"])
                ))
                trial_params.append({"C10": C10, "C12": C12, "phi12": phi12})
            losses = evaluate_batch(trial_params)
            for trial, params, loss in zip(trial_records, trial_params, losses):
                loss_value = float(loss)
                study.tell(trial, loss_value)
                trials.append({"stage": "search", "number": trial.number, "params": dict(params), "loss": loss_value})
            n_completed += current
            progress.update(current)
        progress.close()

        if study.best_trial is not None and float(study.best_value) < best_loss:
            params = study.best_trial.params
            best = {
                "C10": float(params.get("C10_nm", best["C10"])),
                "C12": float(params.get("C12_nm", best["C12"])),
                "phi12": math.radians(float(params.get("phi12_deg", math.degrees(best["phi12"])))),
            }
            best_loss = float(study.best_value)
    timings["optuna_seconds"] = time.perf_counter() - optuna_started

    refine_started = time.perf_counter()
    refine_nfev = 0
    if refine == "nelder-mead":
        lock = set(refine_lock or [])
        refine_cache: dict[tuple[float, float, float, float], float] = {}

        def refine_eval(params: dict[str, float]) -> float:
            loss = _evaluate_exact_float32_cached(
                params,
                lambda current: evaluate(
                    current["C10"], current["C12"], current["phi12"]
                ),
                refine_cache,
            )
            trials.append({"stage": "refinement", "params": dict(params), "loss": loss})
            return loss

        best, best_loss = _nelder_mead_refine(
            best,
            best_loss,
            refine_eval,
            lock=lock,
            fatol=3e-6,
            max_iter=80,
            initial_step_floor={"C12": 2.0, "phi12": 0.04},
            initial_step_decimals=2,
        )
        refine_nfev = len(refine_cache)
    elif refine is not None:
        raise ValueError(f"refine must be 'nelder-mead' or None, got {refine!r}")
    timings["refinement_seconds"] = time.perf_counter() - refine_started

    final_object_started = time.perf_counter()
    object_wave_mx = object_fourier_sum_dynamic(
        prepared,
        C10=best["C10"],
        C12=best["C12"],
        phi12=best["phi12"],
        chunk_bf=default_object_redraw_chunk_bf(),
    )
    mx = require_mlx()
    mx.eval(object_wave_mx)
    object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    timings["final_object_seconds"] = time.perf_counter() - final_object_started
    final_loss_started = time.perf_counter()
    _object_wave, full_loss, phase = reconstruct_prepared(
        prepared,
        C10=best["C10"],
        C12=best["C12"],
        phi12=best["phi12"],
        chunk_bf=fit_chunk_bf,
        compute_loss=True,
        compute_object=False,
    )
    timings["final_phase_loss_seconds"] = time.perf_counter() - final_loss_started
    final_loss = full_loss if full_loss is not None else best_loss
    elapsed = time.perf_counter() - started
    final_loss_value = float(final_loss if final_loss is not None else best_loss)
    if phase is None:
        raise RuntimeError("MPS optimizer did not produce its final exact phase.")
    normalized_trials: list[dict[str, object]] = []
    for trial in trials:
        params = dict(trial["params"])
        normalized_trials.append(
            {
                "stage": trial["stage"],
                "number": trial.get("number"),
                "params": {
                    "C10_nm": float(params["C10"]),
                    "C12_nm": float(params["C12"]),
                    "phi12_deg": math.degrees(float(params["phi12"])),
                },
                "loss": float(trial["loss"]),
            }
        )
    result = SSBResult(
        object_wave=object_wave,
        backend="mps",
        aberrations=dict(best),
        rotation_angle_deg=float(rotation_angle_deg),
        loss=final_loss_value,
        elapsed=elapsed,
        timings=timings,
        n_trials=int(n_trials),
        num_bf=selection.size,
        refine_method=refine,
        refine_nfev=refine_nfev,
        refine_elapsed=timings["refinement_seconds"],
        voltage_kV=float(voltage_kV),
        semiangle_mrad=float(semiangle_mrad),
        scan_sampling_A=scan_sampling_A,
        bf_center=selection.center_row_col,
        bf_radius=selection.radius_px,
        detected_bf_radius=selection.detected_radius_px,
        trial_records=normalized_trials,
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
    return fit_sample_search(
        lambda c10, c12, phi12, tilt, thickness: thick_fit(prepared, C10=c10, C12=c12, phi12=phi12, tilt_mrad=tilt,
                                                           thickness=thickness, band_inv_A=band_inv_A),
        objective_batch=lambda rows: thick_fit_batch(prepared, rows, band_inv_A),
        batch_size=THICK_FIT_MAX_BATCH,
        band_inv_A=band_inv_A, **options)


def _ranges_from_start(
    start: dict[str, float],
    search_ranges: dict | None,
) -> dict[str, tuple[float, float] | float]:
    if search_ranges is not None:
        return dict(search_ranges)
    return {
        "C10_nm": (-400.0, 400.0),
        "C12_nm": (0.0, 100.0),
        "phi12_deg": (-90.0, 90.0),
    }


def _suggest_or_fixed(trial, ranges: dict, key: str, default: float) -> float:
    value = ranges.get(key, default)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if lo == hi:
            return lo
        return float(trial.suggest_float(key, lo, hi))
    return float(value)


def _nelder_mead_refine(
    best: dict[str, float],
    best_loss: float,
    evaluate,
    *,
    lock: set[str],
    xatol: float = 0.1,
    fatol: float = 1e-8,
    max_iter: int = 300,
    initial_step_floor: dict[str, float] | None = None,
    initial_step_decimals: int | None = None,
) -> tuple[dict[str, float], float]:
    """Pure-Python Nelder-Mead matching the CUDA optimizer's simplex policy."""
    keys = [key for key in ("C10", "C12", "phi12") if key not in lock]
    if not keys:
        return dict(best), float(best_loss)
    x0 = np.array([best[key] for key in keys], dtype=np.float64)
    n = int(x0.size)
    simplex = np.empty((n + 1, n), dtype=np.float64)
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = max(abs(x0[i]) * 0.05, 0.00025)
        if initial_step_floor is not None:
            step = max(step, float(initial_step_floor.get(keys[i], 0.0)))
        if initial_step_decimals is not None:
            step = round(step, int(initial_step_decimals))
        simplex[i + 1, i] += step

    def params_from_x(x: np.ndarray) -> dict[str, float]:
        params = dict(best)
        for i, key in enumerate(keys):
            value = float(x[i])
            if key == "C12":
                value = max(0.0, value)
            params[key] = value
        return params

    f_values = np.empty(n + 1, dtype=np.float64)
    f_values[0] = float(best_loss)
    for i in range(1, n + 1):
        params = params_from_x(simplex[i])
        f_values[i] = evaluate(params)

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    for _ in range(max_iter):
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        x_spread = float(np.max(np.abs(simplex[-1] - simplex[0])))
        f_spread = float(abs(f_values[-1] - f_values[0]))
        if x_spread < xatol and f_spread < fatol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = evaluate(params_from_x(x_r))

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = evaluate(params_from_x(x_e))
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        if f_r < f_values[-1]:
            x_c = centroid + rho * (x_r - centroid)
            f_c = evaluate(params_from_x(x_c))
            if f_c <= f_r:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue
        else:
            x_c = centroid - rho * (centroid - simplex[-1])
            f_c = evaluate(params_from_x(x_c))
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue

        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_values[i] = evaluate(params_from_x(simplex[i]))

    best_idx = int(np.argmin(f_values))
    return params_from_x(simplex[best_idx]), float(f_values[best_idx])


def _evaluate_exact_float32_cached(
    params: dict[str, float],
    evaluate,
    cache: dict[tuple[float, float, float, float], float],
) -> float:
    """Evaluate once for each distinct set of float32 MPS kernel inputs."""
    phi12 = np.float32(params["phi12"])
    key = (
        float(np.float32(params["C10"])),
        float(np.float32(params["C12"])),
        float(np.float32(np.cos(np.float32(2.0) * phi12))),
        float(np.float32(np.sin(np.float32(2.0) * phi12))),
    )
    if key not in cache:
        cache[key] = float(evaluate(params))
    return cache[key]


__all__ = ["fit_sample", "optimize"]
