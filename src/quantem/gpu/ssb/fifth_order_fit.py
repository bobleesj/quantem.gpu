"""Fifth-order SSB fitting with a shared initialization and refinement path.

The candidate generator and float64 refinement solvers run on the CPU. Scientific
objective evaluation and reconstruction run on the selected GPU. This is a hybrid solver.
"""

import copy
import math
import time

import numpy as np
import optuna

from ._broad_nelder_mead import nelder_mead
from ._broad_newton import BudgetReached, Tracker, newton
from ._gradient_refinement import Settings as GradientSettings
from ._gradient_refinement import refine as gradient_refine
from ._tpe_independent import make_independent_sampler
from .persistence import load_result, result_paths


def _objective(ssb, gradients=False):
    """Use the same coordinates and model on the session's physical backend.

    ``gradients=True`` returns the same objective with exact gradients (CUDA and MPS).
    """
    backend = ssb._backend()
    if ssb.backend == "cuda":
        if gradients:
            from .cuda.exact_derivatives import FifthOrderGradients

            return FifthOrderGradients(backend._get_accelerator())
        from .cuda.broad_batch import Objective

        return Objective(backend._get_accelerator())
    if ssb.backend == "mps":
        from .mps.fifth_order import Objective

        backend.cache_rotation(np.deg2rad(ssb.scan_detector_rotation_deg))
        if gradients:
            from .mps.exact_derivatives import FifthOrderGradients

            return FifthOrderGradients(backend._prepared)
        return Objective(backend._prepared)
    raise NotImplementedError("Fifth-order fitting requires a CUDA or MPS session.")


def _exact_gradients(ssb) -> bool:
    """Whether the session's backend serves exact fifth-order gradients."""
    return ssb.backend in ("cuda", "mps")


def _refine_with_gradients(objective, start, bounds, norm, cap, *, on_progress=None):
    """``_gradient_refinement.refine`` on the objective's exact gradients, normalized by ``norm``.

    A BF subsample steers only while its agreement depends on the coordinates (that of a single pixel is
    constant): it keeps at least 32 pixels, so a small bright field uses all of them.
    """
    settings = GradientSettings()
    usable = max(1, objective.num_bright_field_pixels // 32)
    settings.hessian_bf_stride = min(settings.hessian_bf_stride, usable)
    settings.coarse_bf_stride = min(settings.coarse_bf_stride, usable)
    settings.max_objectives = min(settings.max_objectives, cap)
    last = {}

    def evaluate(points, kind, stride):
        if kind == "gradient":
            return last["values"], objective.gradients(points, stride) / norm
        values = objective.values(points, stride) / norm
        last["values"] = values
        return values, (None if kind == "value" else objective.gradients(points, stride) / norm)

    start_value = float(objective.values(start, 1)[0]) / norm
    return gradient_refine(evaluate, start, bounds, start_value=start_value, settings=settings,
                           progress=on_progress)


# A broad search refines from the standard start (the initial fit) and from the best BROAD_STARTS distinct TPE
# candidates, and keeps the highest agreement. A later start replaces the kept one only when its float64-summed
# agreement is higher by BASIN_MARGIN (relative), so a tie keeps the standard fit.
BROAD_STARTS = 8
BASIN_MARGIN = 1e-6
# Two candidates are the same start when their aperture phases differ by less than 0.25 rad rms (piston removed),
# each tilt component by less than 0.5 mrad and the depth spread by less than 1 nm.
SAME_START = {"phase_rad": 0.25, "tilt_mrad": 0.5, "depth_nm": 1.0}


def broad_starts(standard, proposals, scores, wavelength, semiangle_rad, count=BROAD_STARTS):
    """[(name, coordinates)]: ``standard`` (skipped when None), then the ``count`` best TPE candidates by coarse
    score that differ from every earlier start (``SAME_START``), named ``"trial <n>"``."""
    from .fifth_order_parameters import aperture_phase, aperture_rms

    starts = [] if standard is None else [("standard", np.asarray(standard, dtype=float))]
    proposals = np.asarray(proposals, dtype=float).reshape(-1, 28)
    scores = np.asarray(scores, dtype=float)
    phases = aperture_phase(proposals, wavelength, semiangle_rad)
    kept = [] if standard is None else [(starts[0][1], aperture_phase(standard, wavelength, semiangle_rad)[0])]
    for index in np.argsort(-scores, kind="stable"):
        if len(starts) - (standard is not None) == count or not np.isfinite(scores[index]):
            break
        z, phase = proposals[index], phases[index]
        if all(aperture_rms(phase, other_phase) >= SAME_START["phase_rad"]
               or np.max(np.abs(z[3:5] - other[3:5])) * 5 >= SAME_START["tilt_mrad"]
               or abs(z[5] - other[5]) * 10 >= SAME_START["depth_nm"] for other, other_phase in kept):
            starts.append((f"trial {index}", z))
            kept.append((z, phase))
    return starts


def _refine_from_starts(objective, starts, bounds, norm, cap, *, on_progress=None):
    """``_refine_with_gradients`` from each of ``starts``; the kept record has the highest agreement (BASIN_MARGIN).

    Returns (kept record, its start name, per-start summaries, one trace over all starts with objective counts and
    seconds running on and the best agreement so far).
    """
    kept, kept_name, summaries, trace = None, None, [], []
    done, best, started = 0, -np.inf, time.perf_counter()
    for name, point in starts:
        offset, offset_s, prior = done, time.perf_counter() - started, best
        progress = None if on_progress is None else (
            lambda count, value, offset=offset, prior=prior: on_progress(offset + count, max(prior, value)))
        record = _refine_with_gradients(objective, point, bounds, norm, cap, on_progress=progress)
        summaries.append({"start": name, "agreement": float(record["agreement"]),
                          "objectives": int(record["objectives"]), "gradients": int(record["gradients"]),
                          "stop": record["stop"], "initial_coordinates": np.asarray(point).tolist()})
        trace += [[offset + count, max(prior, value), offset_s + seconds] for count, value, seconds in record["trace"]]
        done += int(record["objectives"])
        best = max(best, float(record["agreement"]))
        if kept is None or record["agreement"] > kept["agreement"] * (1.0 + BASIN_MARGIN):
            kept, kept_name = record, name
    return kept, kept_name, summaries, trace


HIGHER_ORDER_LIMITS = {2.0: "standard", 20.0: "wide"}


def validate_higher_order_limit(limit) -> float:
    """The box of the 22 higher-order coordinates: 2 (standard) or 20 (wide) rad of phase at the aperture edge."""
    try:
        value = float(limit)
    except (TypeError, ValueError):
        value = math.nan
    if value not in HIGHER_ORDER_LIMITS:
        raise ValueError(
            f"higher_order_limit is 2 (standard) or 20 (wide): the largest phase, in rad at the aperture edge, of "
            f"each higher-order Cartesian component; got {limit!r}.")
    return value


def fifth_order_bounds(higher_order_limit=2.0) -> np.ndarray:
    """Bounds of the 28 coordinates: C10, the C12 pair, tilt, depth spread, then +-limit for the higher orders."""
    limit = validate_higher_order_limit(higher_order_limit)
    return np.array([(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)] + [(-limit, limit)] * 22)


def fit_fifth_order(
    ssb, *, search, trials, seed, save_dir, force, verbose,
    primary_options, refinement="auto", compute_loss=False,
    show=True, on_initial_fit=None, size="medium", columns=3, show_convergence=False,
    higher_order_limit=2.0,
):
    """Fit the fifth-order model; standard skips additional higher-order TPE.

    ``refinement="auto"`` refines with exact gradients (``"bfgs"``) where the backend has them, else with the
    finite-difference Newton (``"newton"``), which stays available as the trajectory reference. With exact gradients a
    broad search refines from the standard start and the best distinct TPE candidates (``broad_starts``) and keeps
    the highest agreement; the finite-difference refinements start from the best TPE trial only.
    ``higher_order_limit`` bounds each higher-order Cartesian coordinate at +-2 (standard) or +-20 (wide) rad of
    phase at the aperture edge.
    """
    higher_order_limit = validate_higher_order_limit(higher_order_limit)
    if refinement == "auto":
        refinement = "bfgs" if _exact_gradients(ssb) else "newton"
    if refinement == "bfgs" and not _exact_gradients(ssb):
        raise ValueError("refinement='bfgs' needs exact fifth-order gradients, which this backend does not have; "
                         "use refinement='newton' or 'nelder-mead'.")
    if trials < 1:
        raise ValueError(
            "Fifth-order fitting requires trials >= 1; use search='primary' for zero-trial refinement."
        )
    from .fifth_order_parameters import public_params, vector
    from .fit_display import FitDisplay

    display = FitDisplay(ssb, enabled=show, size=size, columns=columns,
                         show_convergence=show_convergence)

    primary_trials = trials if search == "standard" else 200
    primary_seed = seed if search == "standard" else 42
    exploration_trials = trials if search == "broad" else 0
    refinement_cap = {"newton": 5000, "bfgs": 400}.get(refinement, 3000)
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
        "higher_order_limit": higher_order_limit,
        "primary_options": primary_options,
        "compute_loss": bool(compute_loss),
    }
    if search == "broad" and refinement == "bfgs":
        settings["broad_starts"] = BROAD_STARTS
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
            display.update(reused, final=True)
            if show_convergence and display.display is not None:
                display.display(reused.show("convergence", size=size))
            return reused
    wall_started = time.perf_counter()
    display.progress(0)
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
    handoff_started = time.perf_counter()
    if on_initial_fit is not None:
        on_initial_fit(primary)
    display.update(primary, final=False)
    initial_handoff_s = time.perf_counter() - handoff_started
    # Keep plotting/caller work out of the numerical fit and convergence clocks.
    started += initial_handoff_s
    objective = _objective(ssb, gradients=True) if refinement == "bfgs" else _objective(ssb)
    bounds = fifth_order_bounds(higher_order_limit)
    norm = float(objective.evaluate(vector(primary, 1), 1)[0])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            "Fifth-order fitting requires a positive finite initial objective; inspect calibration and signal."
        )
    proposals = []
    coarse_scores = []
    coarse_elapsed_s = []
    tpe_started = time.perf_counter()
    setup_elapsed_s = tpe_started - started - primary_elapsed_s
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
            nonlocal started, tpe_started
            # Scalar bookkeeping only: the objective has already returned its value.
            coarse_elapsed_s.append(time.perf_counter() - started)
            if show_convergence:
                before = time.perf_counter()
                display.progress(trial.number + 1, study.best_value, stage="Broad fifth-order search")
                overhead = time.perf_counter() - before
                started += overhead
                tpe_started += overhead

        tpe_started = time.perf_counter()
        study.optimize(coarse, n_trials=trials, show_progress_bar=verbose, callbacks=[record_time])
        tpe_elapsed_s = time.perf_counter() - tpe_started
        coarse_scores = [trial.value for trial in study.trials]
    standard = vector(primary, 5)
    if not np.isfinite(standard).all() or np.any(standard < bounds[:, 0]) or np.any(standard > bounds[:, 1]):
        if search != "broad":
            raise ValueError(
                "The initial fit lies outside the fifth-order model bounds. "
                "Inspect calibration or use search='broad' for an independent exploration."
            )
        standard = None
    if search != "broad":
        starts = [("standard", standard)]
    elif refinement == "bfgs":
        starts = broad_starts(standard, proposals, coarse_scores, float(objective.p.wavelength),
                              float(objective.p.semiangle_rad))
    else:
        # The finite-difference refinements keep their reference trajectory: one start, the best TPE trial.
        starts = [("best trial", np.array([study.best_params[name] for name in names]))]
    start_name, start = starts[0]
    cache = {}

    def evaluate(points):
        keys = [point.tobytes() for point in points]
        missing = {key: point for key, point in zip(keys, points) if key not in cache}
        if missing:
            values = objective.evaluate(np.array(list(missing.values())), 5) / norm
            cache.update(zip(missing, values))
        return np.array([cache[key] for key in keys])

    def refinement_progress(count, best):
        nonlocal started
        before = time.perf_counter()
        prior = max(coarse_scores, default=float("-inf"))
        display.progress(exploration_trials + count, max(prior, best), stage="Joint fifth-order refinement")
        started += time.perf_counter() - before

    gradient_record = None
    start_summaries = None
    if refinement == "bfgs":
        refinement_started = time.perf_counter()
        refinement_start_s = refinement_started - started
        gradient_record, start_name, start_summaries, trace = _refine_from_starts(
            objective, starts, bounds, norm, refinement_cap,
            on_progress=refinement_progress if show_convergence else None)
        start = np.asarray(next(s["initial_coordinates"] for s in start_summaries if s["start"] == start_name))
        best_z = gradient_record["coordinates"]
        stop = gradient_record["stop"]
        nfev = sum(s["objectives"] for s in start_summaries)
        # The reported agreement is the search objective's float32 score, as the other refinements report it.
        best_score = float(objective.evaluate(best_z, 5)[0])
        refine_elapsed_s = time.perf_counter() - refinement_started
    else:
        tracker = Tracker(evaluate, start, bounds, refinement_cap,
                          on_progress=refinement_progress if show_convergence else None)
        refinement_start_s = tracker.started - started
        try:
            tracker(start)
            polish = newton if refinement == "newton" else nelder_mead
            _, stop = polish(tracker, start, bounds)
        except BudgetReached:
            stop = "evaluation cap"
        best_z, best_score, nfev, trace = tracker.best_z, tracker.best * norm, tracker.nfev, tracker.trace
        refine_elapsed_s = time.perf_counter() - tracker.started
    aberrations, tilt, depth = public_params(objective, best_z, 5)
    chosen = copy.copy(primary)
    chosen.aberrations = aberrations
    chosen.sample_tilt_mrad = tilt
    chosen.depth_spread_nm = depth
    chosen.objective = "broad_fifth_agreement"
    chosen.sample_agreement = float(best_score)
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
        "executed_objectives": len(cache) if gradient_record is None else nfev,
        "normalized_agreement": float(best_score / norm),
        "normalization": norm,
        "fit_band_inv_A": [0.2, 0.9],
        "higher_order_limit": higher_order_limit,
        "bounds": bounds.tolist(),
        "coordinates": np.asarray(best_z).tolist(),
        "proposals": proposals,
        "coarse_scores": coarse_scores,
        "coarse_elapsed_s": coarse_elapsed_s,
        "refinement_start_s": refinement_start_s,
        "convergence_clock": "fit_start",
        "refinement": refinement,
        "refinement_cap": refinement_cap,
        "refinement_trace": trace,
        f"{refinement.replace('-', '_')}_trace": trace,
        "rotation_policy": "initial fit selection; fifth-order refinement holds that rotation",
    }
    if gradient_record is not None:
        # Values and gradients of the agreement summed in float64; subsamples steer only. The record is the kept
        # start's; ``starts`` lists every start's agreement (normalized) and cost.
        chosen.optimizer["exact_gradients"] = {
            key: gradient_record[key] for key in (
                "agreement", "objectives", "gradients", "subsampled", "iterations", "hessian_bf_stride",
                "coarse_bf_stride")
        }
        chosen.optimizer["starts"] = start_summaries
        chosen.optimizer["kept_start"] = start_name
    chosen.num_trials = trials
    if refinement != "newton":
        chosen.optimizer.pop("newton_cap")
    chosen.refinement = refinement
    chosen.refine_nfev = nfev
    chosen.refine_elapsed_s = refine_elapsed_s
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
        "fifth_order_setup_s": setup_elapsed_s,
        "initial_handoff_s": initial_handoff_s,
        "tpe_s": tpe_elapsed_s,
        "refinement_s": chosen.refine_elapsed_s,
        "reconstruction_s": time.perf_counter() - reconstruction_started,
    }
    final_handoff_started = time.perf_counter()
    display.progress(exploration_trials + nfev, float(best_score / norm),
                     stage=f"Fit complete · {stop}", final=True)
    display.update(result, final=True)
    result.timings["final_display_s"] = time.perf_counter() - final_handoff_started
    result.timings["fit_and_display_s"] = time.perf_counter() - wall_started
    if show_convergence:
        result.timings["convergence_display_s"] = display.display_s
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
