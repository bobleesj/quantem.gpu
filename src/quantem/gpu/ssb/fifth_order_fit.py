"""Fifth-order SSB fitting with a shared initialization and refinement path.

The candidate generator and float64 refinement solvers run on the CPU. Scientific
objective evaluation and reconstruction run on the selected GPU. This is a hybrid solver.
"""

import copy
import time

import numpy as np
import optuna

from ._broad_nelder_mead import nelder_mead
from ._broad_newton import BudgetReached, Tracker, newton
from ._tpe_independent import make_independent_sampler
from .persistence import load_result, result_paths


def _objective(ssb):
    """Use the same coordinates and model on the session's physical backend."""
    backend = ssb._backend()
    if ssb.backend == "cuda":
        from .cuda.broad_batch import Objective

        return Objective(backend._get_accelerator())
    if ssb.backend == "mps":
        from .mps.fifth_order import Objective

        backend.cache_rotation(np.deg2rad(ssb.scan_detector_rotation_deg))
        return Objective(backend._prepared)
    raise NotImplementedError("Fifth-order fitting requires a CUDA or MPS session.")


def fit_fifth_order(
    ssb, *, search, trials, seed, save_dir, force, verbose,
    primary_options, refinement="newton", compute_loss=False,
    show=True, on_initial_fit=None,
):
    """Fit the fifth-order model; standard skips additional higher-order TPE."""
    if trials < 1:
        raise ValueError(
            "Fifth-order fitting requires trials >= 1; use search='primary' for zero-trial refinement."
        )
    from .fifth_order_parameters import public_params, vector

    primary_trials = trials if search == "standard" else 200
    primary_seed = seed if search == "standard" else 42
    exploration_trials = trials if search == "broad" else 0
    refinement_cap = 5000 if refinement == "newton" else 3000
    settings = {
        "refinement": refinement,
        "search": search,
        "model": "broad_fifth_v1",
        "trials": trials,
        "seed": seed,
        "primary_trials": primary_trials,
        "primary_seed": primary_seed,
        "newton_cap": refinement_cap,
        "fit_band_inv_A": [0.2, 0.9],
        "primary_options": primary_options,
        "compute_loss": bool(compute_loss),
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
            from .fit_display import FitDisplay

            FitDisplay(ssb, enabled=show).update(reused, final=True)
            return reused
    started = time.perf_counter()
    primary = ssb._find_aberrations(
        ("apply" if primary_options["check_rotation"] == "both" else
         "auto" if primary_options["check_rotation"] else "off"),
        num_trials=primary_trials,
        refinement="newton",
        seed=primary_seed,
        compare_standard=False,
        compute_loss=False,
        verbose=verbose,
        refine_rotation=primary_options["refine_rotation"],
    )
    primary_elapsed_s = time.perf_counter() - started
    from .fit_display import FitDisplay

    display = FitDisplay(ssb, enabled=show)
    handoff_started = time.perf_counter()
    if on_initial_fit is not None:
        on_initial_fit(primary)
    display.update(primary, final=False)
    initial_handoff_s = time.perf_counter() - handoff_started
    # Keep plotting/caller work out of the numerical fit and convergence clocks.
    started += initial_handoff_s
    objective = _objective(ssb)
    bounds = np.array(
        [(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)] + [(-2, 2)] * 22
    )
    norm = float(objective.evaluate(vector(primary, 1), 1)[0])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            "Fifth-order fitting requires a positive finite initial objective; inspect calibration and signal."
        )
    proposals = []
    coarse_scores = []
    coarse_elapsed_s = []
    tpe_started = time.perf_counter()
    tpe_elapsed_s = 0.0
    if search == "broad":
        names = [f"p{i}" for i in range(28)]
        study = optuna.create_study(
            direction="maximize", sampler=make_independent_sampler(seed=seed, bounds=bounds)
        )
        zero = np.zeros(28)
        zero[5] = 1
        study.enqueue_trial(dict(zip(names, zero.tolist())))
        def coarse(trial):
            point = np.array(
                [
                    trial.suggest_float(name, float(lo), float(hi))
                    for name, (lo, hi) in zip(names, bounds)
                ]
            )
            proposals.append(point.tolist())
            return float(objective.evaluate(point, 5)[0]) / norm

        def record_time(study, trial):
            # Scalar bookkeeping only: the objective has already returned its value.
            coarse_elapsed_s.append(time.perf_counter() - started)

        tpe_started = time.perf_counter()
        study.optimize(coarse, n_trials=trials, show_progress_bar=verbose, callbacks=[record_time])
        tpe_elapsed_s = time.perf_counter() - tpe_started
        start = np.array([study.best_params[name] for name in names])
        coarse_scores = [trial.value for trial in study.trials]
    else:
        start = vector(primary, 5)
        if not np.isfinite(start).all() or np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
            raise ValueError(
                "The initial fit lies outside the fifth-order model bounds. "
                "Inspect calibration or use search='broad' for an independent exploration."
            )
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
    chosen.rotation_branches = primary.rotation_branches
    chosen.trial_records = primary.trial_records
    chosen.optimizer = {
        "search": search,
        "model": "broad_fifth_v1",
        "max_order": 5,
        "host_numerical_search": True,
        "sampler": "Optuna TPE" if exploration_trials else "none",
        "exploration_trials": exploration_trials,
        "initial_coordinates": start.tolist(),
        "primary_trial_records": primary.trial_records,
        "primary_trials": primary_trials,
        "primary_seed": primary_seed,
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
        "coarse_scores": coarse_scores,
        "coarse_elapsed_s": coarse_elapsed_s,
        "refinement_start_s": refinement_start_s,
        "convergence_clock": "fit_start",
        "refinement": refinement,
        "refinement_cap": refinement_cap,
        "refinement_trace": tracker.trace,
        f"{refinement.replace('-', '_')}_trace": tracker.trace,
        "rotation_policy": "initial fit selection; fifth-order refinement holds that rotation",
    }
    chosen.num_trials = trials
    if refinement != "newton":
        chosen.optimizer.pop("newton_cap")
    chosen.refinement = refinement
    chosen.refine_nfev = tracker.nfev
    chosen.refine_elapsed_s = time.perf_counter() - tracker.started
    reconstruction_started = time.perf_counter()
    result = ssb.run(
        chosen, upsampling_factor=1, compute_loss=compute_loss, force=True
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
        "trial_records",
    ):
        setattr(result, field, getattr(chosen, field))
    result.phase_clim = None
    result.elapsed_s = time.perf_counter() - started
    result.timings = {
        "joint_fit_s": result.elapsed_s,
        "primary_fit_s": primary_elapsed_s,
        "fifth_order_setup_s": tpe_started - started - primary_elapsed_s,
        "initial_handoff_s": initial_handoff_s,
        "tpe_s": tpe_elapsed_s,
        "refinement_s": chosen.refine_elapsed_s,
        "reconstruction_s": time.perf_counter() - reconstruction_started,
    }
    final_handoff_started = time.perf_counter()
    display.update(result, final=True)
    result.timings["final_display_s"] = time.perf_counter() - final_handoff_started
    result.timings["fit_and_display_s"] = (
        result.elapsed_s + initial_handoff_s + result.timings["final_display_s"]
    )
    if paths is not None:
        result = ssb._save_result(result, paths=paths, signature=signature)
    ssb._aberration_search = ssb._accept_result(result)
    return result


def parameter_evaluator(ssb, result, band):
    """Vary primary parameters while retaining the fitted broad optical model."""

    objective = _objective(ssb)
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
