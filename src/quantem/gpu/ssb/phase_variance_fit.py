"""The phase-variance standard fit (``find_aberrations(tilt=False, objective="phase_variance")``), backend-neutral.

For candidate aberrations the loss is the mean over scan pixels of the variance, over bright-field pixels, of the
corrected phase: the better the correction, the more every bright-field pixel agrees on the object phase. The backend
supplies only that loss (``evaluate_batch``); the search is this module's, so a seeded fit visits the same trials and
the same simplex whichever backend evaluates it, and fits agree as closely as the backends' losses do. It is the
search of ``mps/optimizer.optimize_phase_variance`` (unchanged since before 4fc281f); CUDA runs it from here, and MPS
can (its pair-batched loss as ``evaluate_batch``).

The search: the start (the session's aberrations, or C10 = 0, C12 = 50 Angstrom, phi12 = 0) is evaluated once and kept
unless a trial beats it, then ``num_trials`` Optuna TPE trials (seeded, asked two at a time), then Nelder-Mead from the
best point. Nelder-Mead's first simplex steps at least 2 Angstrom in C12 and 0.04 rad in phi12, rounded to two decimals,
because the loss is nearly flat along phi12 at small C12 and a 5 % step from the best trial can stall there.
"""

import math
import time

import numpy as np

# The search box when search_ranges leaves a coefficient out: C10 +-40 nm, C12 up to 10 nm, phi12 over its period
# (engine keys and units).
DEFAULT_RANGES = {"C10_A": (-400.0, 400.0), "C12_A": (0.0, 100.0), "phi12_deg": (-90.0, 90.0)}


def fit_phase_variance(
    evaluate_batch,
    *,
    aberrations: dict | None = None,
    search_ranges: dict | None = None,
    num_trials: int = 200,
    refine: str | None = "nelder-mead",
    refine_lock: list[str] | None = None,
    seed: int = 42,
    verbose: bool = False,
    batch_size: int = 2,
) -> dict[str, object]:
    """Minimize the phase-variance loss over C10, C12 and phi12 (engine units: Angstrom, Angstrom, rad).

    ``evaluate_batch(C10, C12, phi12)`` takes three float32 arrays and returns the losses. ``search_ranges`` (engine
    keys, see ``DEFAULT_RANGES``) bounds the trials; a single value, or a range of zero width, holds that coefficient in
    every trial and in the result. ``refine_lock`` (engine names) holds coefficients through Nelder-Mead at the
    search's value. Returns the best aberrations, their loss, every evaluated trial (stages ``initial``, ``search``
    and ``refinement``), the number of distinct refinement evaluations and the stage timings.
    """
    import optuna

    def evaluate(C10: float, C12: float, phi12: float) -> float:
        """One candidate's loss through ``evaluate_batch``, so single and batched evaluations see the same kernel."""
        return float(evaluate_batch(np.asarray([C10], dtype=np.float32), np.asarray([C12], dtype=np.float32),
                                    np.asarray([phi12], dtype=np.float32))[0])

    timings: dict[str, float] = {}
    start = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}
    if aberrations:
        start.update({name: float(value) for name, value in aberrations.items() if name in start})
    # a coefficient given as None is searched over its default range, as in the agreement fit's standard_bounds
    ranges = {**DEFAULT_RANGES, **{key: value for key, value in (search_ranges or {}).items() if value is not None}}
    # A pinned coefficient is every trial's and the result's value. It is not put into the start: the start is a
    # candidate the trials must beat, and pinning it there could make it a degenerate point (C10 = C12 = 0, no
    # correction at all) whose loss no corrected trial reaches.
    pinned = _pinned(ranges)
    trials: list[dict] = []
    best = dict(start)
    initial_started = time.perf_counter()
    best_loss = evaluate(best["C10"], best["C12"], best["phi12"])
    timings["initial_loss_seconds"] = time.perf_counter() - initial_started
    trials.append({"stage": "initial", "params": dict(best), "loss": best_loss})

    search_started = time.perf_counter()
    if num_trials > 0:
        from tqdm.auto import tqdm

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
        progress = tqdm(total=int(num_trials), desc="SSB optimize", disable=not verbose,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        completed = 0
        while completed < int(num_trials):
            # the batch size is part of the trial stream: TPE sees a batch's losses only after it asked for all of it
            batch_count = min(max(1, int(batch_size)), int(num_trials) - completed)
            asked = [study.ask() for _ in range(batch_count)]
            candidates = []
            for trial in asked:
                C10 = _suggest_or_fixed(trial, ranges, "C10_A", best["C10"])
                C12 = _suggest_or_fixed(trial, ranges, "C12_A", best["C12"])
                phi12 = math.radians(_suggest_or_fixed(trial, ranges, "phi12_deg", math.degrees(best["phi12"])))
                candidates.append({"C10": C10, "C12": C12, "phi12": phi12})
            losses = evaluate_batch(*(np.asarray([candidate[name] for candidate in candidates], dtype=np.float32)
                                      for name in ("C10", "C12", "phi12")))
            for trial, candidate, loss in zip(asked, candidates, losses, strict=True):
                study.tell(trial, float(loss))
                trials.append({"stage": "search", "number": trial.number, "params": dict(candidate), "loss": float(loss)})
            completed += batch_count
            progress.update(batch_count)
        progress.close()
        # with pins the start, which ignores them, is no candidate: the best trial is the fit
        if study.best_trial is not None and (pinned or float(study.best_value) < best_loss):
            params = study.best_trial.params
            best = {
                "C10": float(params.get("C10_A", pinned.get("C10", best["C10"]))),
                "C12": float(params.get("C12_A", pinned.get("C12", best["C12"]))),
                "phi12": math.radians(float(params.get("phi12_deg", math.degrees(pinned.get("phi12", best["phi12"]))))),
            }
            best_loss = float(study.best_value)
    elif pinned:
        # no search: refine from the start with the pinned coefficients in place
        best = {**best, **pinned}
        best_loss = evaluate(best["C10"], best["C12"], best["phi12"])
    timings["search_seconds"] = time.perf_counter() - search_started

    refine_started = time.perf_counter()
    refine_nfev = 0
    if refine == "nelder-mead":
        cache: dict[tuple[float, float, float, float], float] = {}

        def refine_evaluate(params: dict[str, float]) -> float:
            """A vertex's loss, evaluated once per distinct float32 input and recorded as a refinement trial."""
            loss = _evaluate_float32_cached(params, lambda vertex: evaluate(**vertex), cache)
            trials.append({"stage": "refinement", "params": dict(params), "loss": loss})
            return loss

        best, best_loss = _nelder_mead(best, best_loss, refine_evaluate, lock=set(refine_lock or []), fatol=3e-6,
                                       max_iter=80, initial_step_floor={"C12": 2.0, "phi12": 0.04},
                                       initial_step_decimals=2)
        refine_nfev = len(cache)
    elif refine is not None:
        raise ValueError(f"refine must be 'nelder-mead' or None, got {refine!r}")
    timings["refinement_seconds"] = time.perf_counter() - refine_started
    records = [{"stage": trial["stage"], "number": trial.get("number"),
                "params": {"C10_A": float(trial["params"]["C10"]), "C12_A": float(trial["params"]["C12"]),
                           "phi12_deg": math.degrees(float(trial["params"]["phi12"]))},
                "loss": float(trial["loss"])} for trial in trials]
    return {"aberrations": best, "loss": float(best_loss), "trial_records": records, "refine_nfev": refine_nfev,
            "timings": timings}


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

def _pinned(ranges: dict) -> dict[str, float]:
    """The coefficients ``ranges`` holds at one value, in the engine's keys and units (Angstrom, radians)."""
    pinned = {}
    for key, name in (("C10_A", "C10"), ("C12_A", "C12"), ("phi12_deg", "phi12")):
        value = ranges.get(key)
        if isinstance(value, (tuple, list)) and len(value) == 2 and float(value[0]) == float(value[1]):
            value = value[0]
        if value is not None and not isinstance(value, (tuple, list)):
            pinned[name] = math.radians(float(value)) if key == "phi12_deg" else float(value)
    return pinned


def _suggest_or_fixed(trial, ranges: dict, key: str, default: float) -> float:
    """TPE's suggestion within a ``(low, high)`` range; a single value, or a zero-width range, is returned as is."""
    value = ranges.get(key, default)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        if low == high:
            return low
        return float(trial.suggest_float(key, low, high))
    return float(value)


def _nelder_mead(
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
    """Nelder-Mead (reflection 1, expansion 2, contraction 0.5, shrink 0.5) over the unlocked coefficients.

    The first simplex steps max(5 % of the coefficient, 0.00025), raised to ``initial_step_floor`` and rounded to
    ``initial_step_decimals``. C12 is a magnitude, so a vertex with C12 < 0 is evaluated at 0. Vertices are ordered
    by a stable sort, so tied losses keep the earlier vertex and a flat loss leaves the start in place. Stops when
    both the simplex spread and the loss spread are below ``xatol`` and ``fatol``, or after ``max_iter`` iterations.
    """
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
        """The full coefficient set at a simplex point: locked coefficients from ``best``, C12 clamped at 0."""
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
        f_values[i] = evaluate(params_from_x(simplex[i]))
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        order = np.argsort(f_values, kind="stable")
        simplex = simplex[order]
        f_values = f_values[order]
        if float(np.max(np.abs(simplex[-1] - simplex[0]))) < xatol and float(abs(f_values[-1] - f_values[0])) < fatol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = evaluate(params_from_x(x_r))
        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1], f_values[-1] = x_r, f_r
            continue
        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = evaluate(params_from_x(x_e))
            simplex[-1], f_values[-1] = (x_e, f_e) if f_e < f_r else (x_r, f_r)
            continue
        if f_r < f_values[-1]:
            x_c = centroid + rho * (x_r - centroid)
            f_c = evaluate(params_from_x(x_c))
            if f_c <= f_r:
                simplex[-1], f_values[-1] = x_c, f_c
                continue
        else:
            x_c = centroid - rho * (centroid - simplex[-1])
            f_c = evaluate(params_from_x(x_c))
            if f_c < f_values[-1]:
                simplex[-1], f_values[-1] = x_c, f_c
                continue
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_values[i] = evaluate(params_from_x(simplex[i]))
    best_index = int(np.argmin(f_values))
    return params_from_x(simplex[best_index]), float(f_values[best_index])


def _evaluate_float32_cached(
    params: dict[str, float],
    evaluate,
    cache: dict[tuple[float, float, float, float], float],
) -> float:
    """Evaluate once per distinct float32 key (C10, C12, cos 2 phi12, sin 2 phi12), the inputs MPS's kernel took.

    Nelder-Mead's late vertices often differ by less than a float32 step and would repeat an evaluation. The key is
    the search's, not the backend's: CUDA rounds cos/sin 2 phi12 from float64, so on CUDA it is the same key as on
    MPS (and the same path), not exactly the kernel's input. ``refine_nfev`` counts distinct keys.
    """
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


__all__ = ["DEFAULT_RANGES", "fit_phase_variance"]
