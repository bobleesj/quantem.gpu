"""Broad fifth-order SSB search with an explicit objective and search record.

The candidate generator and float64 refinement solvers run on the CPU. Scientific
objective evaluation and reconstruction run on CUDA. This is a hybrid solver.
"""

import copy
import time

import numpy as np
import optuna

from ._broad_nelder_mead import nelder_mead
from ._broad_newton import BudgetReached, Tracker, newton
from ._tpe_independent import make_independent_sampler
from .persistence import load_result, result_paths


def find_broad(ssb, *, trials, seed, save_dir, force, verbose, primary_options, refinement="newton"):
    """Fit the broad fifth-order model, retaining native reconstruction sampling."""
    if ssb.backend != "cuda":
        raise NotImplementedError(
            "Broad fifth-order search is currently CUDA-only. "
            "Use search='standard' on MPS; no estimator is substituted."
        )
    if trials < 1:
        raise ValueError(
            "Broad search requires trials >= 1; use search='standard' for zero-trial refinement."
        )
    from .cuda.broad_batch import Objective
    from .cuda.broad_objective import public_params, vector

    refinement_cap = 5000 if refinement == "newton" else 3000
    settings = {
        "refinement": refinement,
        "search": "broad",
        "model": "broad_fifth_v1",
        "trials": trials,
        "seed": seed,
        "primary_trials": 200,
        "primary_seed": 42,
        "newton_cap": refinement_cap,
        "fit_band_inv_A": [0.2, 0.9],
        "primary_options": primary_options,
    }
    paths = result_paths(save_dir, "find_aberrations") if save_dir is not None else None
    signature = (
        ssb._result_signature("find_aberrations", settings)
        if paths is not None
        else None
    )
    if paths is not None and not force and ssb.source_kind != "array":
        reused = load_result(paths=paths, signature=signature, backend=ssb.backend)
        if reused is not None:
            ssb._aberration_search = ssb._accept_result(reused)
            return reused
    started = time.perf_counter()
    primary = ssb.find_aberrations(
        search="standard",
        trials=200,
        refinement="newton",
        seed=42,
        compare_standard=False,
        compute_loss=False,
        verbose=verbose,
        **primary_options,
    )
    primary_elapsed_s = time.perf_counter() - started
    objective = Objective(ssb._backend()._get_accelerator())
    bounds = np.array(
        [(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)] + [(-2, 2)] * 22
    )
    norm = float(objective.evaluate(vector(primary, 1), 1)[0])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            "Broad search requires a positive finite primary objective; inspect calibration and signal."
        )
    names = [f"p{i}" for i in range(28)]
    study = optuna.create_study(
        direction="maximize", sampler=make_independent_sampler(seed=seed, bounds=bounds)
    )
    zero = np.zeros(28)
    zero[5] = 1
    study.enqueue_trial(dict(zip(names, zero.tolist())))
    proposals = []

    def coarse(trial):
        point = np.array(
            [
                trial.suggest_float(name, float(lo), float(hi))
                for name, (lo, hi) in zip(names, bounds)
            ]
        )
        proposals.append(point.tolist())
        return float(objective.evaluate(point, 5)[0]) / norm

    coarse_elapsed_s = []

    def record_time(study, trial):
        # Scalar bookkeeping only: the objective has already returned its value.
        coarse_elapsed_s.append(time.perf_counter() - started)

    tpe_started = time.perf_counter()
    study.optimize(coarse, n_trials=trials, show_progress_bar=verbose, callbacks=[record_time])
    tpe_elapsed_s = time.perf_counter() - tpe_started
    start = np.array([study.best_params[name] for name in names])
    cache = {}

    def evaluate(points):
        keys = [point.tobytes() for point in points]
        missing = {key: point for key, point in zip(keys, points) if key not in cache}
        if missing:
            values = objective.evaluate(np.array(list(missing.values())), 5) / norm
            cache.update(zip(missing, values))
        return np.array([cache[key] for key in keys])

    tracker = Tracker(evaluate, start, bounds, refinement_cap)
    refinement_start_s = tracker.started - started
    try:
        tracker(start)
        polish = newton if refinement == "newton" else nelder_mead
        _, stop = polish(tracker, start, bounds)
    except BudgetReached:
        stop = "evaluation cap"
    aberrations, tilt, depth = public_params(objective, tracker.best_z, 5)
    chosen = copy.copy(primary)
    chosen.aberrations = aberrations
    chosen.sample_tilt_mrad = tilt
    chosen.depth_spread_nm = depth
    chosen.objective = "broad_fifth_agreement"
    chosen.sample_agreement = float(tracker.best * norm)
    chosen.tilt_fit_gain = None
    chosen.parameter_slices = None
    chosen.rotation_branches = None
    chosen.trial_records = None
    chosen.optimizer = {
        "search": "broad",
        "model": "broad_fifth_v1",
        "max_order": 5,
        "host_numerical_search": True,
        "sampler": "Optuna TPE",
        "primary_trials": 200,
        "primary_seed": 42,
        "seed": seed,
        "trials": trials,
        "newton_cap": refinement_cap,
        "stop": stop,
        "executed_objectives": len(cache),
        "normalized_agreement": tracker.best,
        "normalization": norm,
        "fit_band_inv_A": [0.2, 0.9],
        "bounds": bounds.tolist(),
        "coordinates": tracker.best_z.tolist(),
        "proposals": proposals,
        "coarse_scores": [t.value for t in study.trials],
        "coarse_elapsed_s": coarse_elapsed_s,
        "refinement_start_s": refinement_start_s,
        "convergence_clock": "fit_start",
        "refinement": refinement,
        "refinement_cap": refinement_cap,
        "refinement_trace": tracker.trace,
        f"{refinement.replace('-', '_')}_trace": tracker.trace,
        "rotation_policy": "primary fit selection; broad search holds that rotation",
    }
    chosen.num_trials = trials
    if refinement != "newton":
        chosen.optimizer.pop("newton_cap")
    chosen.refinement = refinement
    chosen.refine_nfev = tracker.nfev
    chosen.refine_elapsed_s = time.perf_counter() - tracker.started
    reconstruction_started = time.perf_counter()
    result = ssb.reconstruct(
        chosen, upsampling_factor=1, compute_loss=False, force=True
    )
    # reconstruct carries fit provenance only when compatible; assign the new fit explicitly.
    for field in (
        "objective",
        "sample_agreement",
        "optimizer",
        "num_trials",
        "refinement",
        "refine_nfev",
        "refine_elapsed_s",
    ):
        setattr(result, field, getattr(chosen, field))
    result.elapsed_s = time.perf_counter() - started
    result.timings = {
        "broad_search_s": result.elapsed_s,
        "primary_fit_s": primary_elapsed_s,
        "broad_setup_s": tpe_started - started - primary_elapsed_s,
        "tpe_s": tpe_elapsed_s,
        "refinement_s": chosen.refine_elapsed_s,
        "reconstruction_s": time.perf_counter() - reconstruction_started,
    }
    if paths is not None:
        result = ssb._save_result(result, paths=paths, signature=signature)
    ssb._aberration_search = ssb._accept_result(result)
    return result


def parameter_evaluator(ssb, result, band):
    """Vary primary parameters while retaining the fitted broad optical model."""
    from .cuda.broad_batch import Objective

    objective = Objective(ssb._backend()._get_accelerator())
    coordinates = np.asarray(result.optimizer["coordinates"], dtype=float)

    def evaluate(rows):
        rows = np.atleast_2d(rows)
        points = np.tile(coordinates, (len(rows), 1))
        points[:, 0] = rows[:, 0] / 100
        points[:, 1] = rows[:, 1] * np.cos(2 * rows[:, 2]) / 100
        points[:, 2] = rows[:, 1] * np.sin(2 * rows[:, 2]) / 100
        points[:, 3:5] = rows[:, 3:5] / 5
        points[:, 5] = rows[:, 5] / 100
        return objective.evaluate(points, 5, band=band)

    return evaluate
