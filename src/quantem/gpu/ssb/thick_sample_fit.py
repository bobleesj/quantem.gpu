"""Backend-neutral search for the thick-sample SSB fit (defocus, astigmatism, sample tilt, depth spread).

Every backend (CUDA engine, MPS, torch reference) supplies the same objective - the least-squares agreement of the
thick-sample SSB model with G(Q, K), larger is better (see ``ThickSample.fit``) - and this module owns the search, so the
backends differ only in how fast they evaluate it.

Search: multivariate TPE over all six parameters jointly, then a polish from the best trials (Nelder-Mead, or Newton steps on
finite-difference models whose evaluations share passes over G). C10 is searched jointly because
the thick model's C10 is the defocus at mid-depth, not the standard-SSB optimum; freeing only the tilt from the standard
optimum misses the answer (simulated BaTiO3 15 nm tilted (3, -4) mrad: joint search (3.0, -4.1); tilt-only (8.6, -6.3)).
300 trials give the same answer as 1500 on the simulation and on the logic device (Nelder-Mead polishes the rest). A
standard-SSB fit (depth spread 0, 200 trials, the production budget) is run on the same objective so the reported gain compares
like with like. Units: C10, C12, depth_spread_A in the engine's C10 unit (Angstrom); phi12 rad; tilt mrad, scan frame (row, col).
"""

import math
import time
import warnings
from collections.abc import Callable

import numpy as np

Objective = Callable[[float, float, float, tuple[float, float], float], float]
# (B, 6) rows of (C10, C12, phi12, tilt_row_mrad, tilt_col_mrad, depth_spread_A) -> (B,) values; lets a backend evaluate a batch
# of trials in one pass over G (CUDA: ThickSample.fit_batch)
BatchObjective = Callable[[np.ndarray], np.ndarray]
PARAMETERS = ("C10", "C12", "phi12", "tilt_row_mrad", "tilt_col_mrad", "depth_spread_A")
# The depth spreads the joint fit searches, in nm (public) and in the engine's Angstrom: SSB.find_aberrations(tilt=True)
# returns a depth_spread_nm inside it, and its parameter slices span it.
DEPTH_SPREAD_RANGE_NM = (2.0, 60.0)
DEPTH_SPREAD_RANGE_A = (20.0, 600.0)
# Optuna's label for depth_spread_A. Multivariate TPE orders its search space by label, so another label would change
# the proposals of every seed; the search keeps it, and every record and result uses depth_spread_A.
_DEPTH_SEARCH_LABEL = "thickness"


def _search_label(name: str) -> str:
    return _DEPTH_SEARCH_LABEL if name == "depth_spread_A" else name


def _make_sampler(**options):
    """Reuse unchanged TPE densities while proposing an independent batch."""
    import optuna
    from optuna.samplers import TPESampler

    # The specialized Gaussian path preserves 4.9.0's sampling equations.
    # Other versions retain their own sampler until their parity is verified.
    cache_gaussians = optuna.__version__ == "4.9.0"
    if cache_gaussians:
        from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions

        from quantem.gpu.ssb._tpe_gaussian import _CachedGaussianMixture

    class BatchTPESampler(TPESampler):
        # Optuna's 4.x density builder is private. Keep this small override
        # isolated and test the entire trial sequence against stock TPE.
        # Sampling still uses Optuna's RNG, candidate ranking and ask/tell.
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._density_cache = {}

        def _build_parzen_estimator(self, study, search_space, trials, handle_below):
            key = (id(study), tuple(search_space.items()), tuple(t.number for t in trials))
            cached = self._density_cache.get(handle_below)
            if cached is None or cached[0] != key:
                cached = (key, super()._build_parzen_estimator(
                    study, search_space, trials, handle_below,
                ))
                if cache_gaussians:
                    mixture = cached[1]._mixture_distribution
                    if all(isinstance(part, _BatchedTruncNormDistributions) for part in mixture.distributions):
                        cached[1]._mixture_distribution = _CachedGaussianMixture(mixture)
                self._density_cache[handle_below] = cached
            return cached[1]

    return BatchTPESampler(**options)


def fit_sample_search(
    objective: Objective,
    *,
    objective_batch: BatchObjective | None = None,
    batch_size: int = 8,
    trials: int = 300,
    tilt_limit_mrad: float = 25.0,
    depth_spread_range_A: tuple[float, float] = DEPTH_SPREAD_RANGE_A,
    c10_range: tuple[float, float] = (-300.0, 300.0),
    c12_max: float = 200.0,
    seed: int = 0,
    verbose: bool = True,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    polish_starts: int = 3,
    batch_polish: bool = False,
    refinement_objective_batch: BatchObjective | None = None,
    polish: str = "nelder-mead",
    compare_standard: bool = True,
    start=None,
    fixed_aberrations: dict[str, float] | None = None,
) -> dict[str, object]:
    """Maximise ``objective(C10, C12, phi12, (tilt_row, tilt_col), depth_spread_A)``; returns the fitted parameters and the gain.

    ``batch_polish`` evaluates independent Nelder-Mead starts together.
    ``refinement_objective_batch`` can preserve a backend's singleton numerical
    policy during refinement while leaving its initial trial batches unchanged.

    ``polish`` refines the best ``polish_starts`` distinct trials: "nelder-mead" (scipy, sequential unless ``batch_polish=True``) or
    "newton" (``newton_polish``: every start's finite-difference design in one batched evaluation, a few steps; needs
    ``objective_batch``). ``compare_standard=False`` skips the standard-SSB search that only feeds the gain, which is then
    None.

    ``start`` is a point (C10, C12, phi12, tilt_row, tilt_col, depth_spread_A) to polish from, the first of the starts: the fit
    of the previous scan of a session. With ``trials=0`` there is no search and the polish from ``start`` is the fit: the
    next scan differs from the last by a drift of focus and tilt, not by a new specimen, so the blind search (200 trials
    and two more starts) buys nothing there.
    """
    import optuna
    from scipy.optimize import minimize

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    half_pi = math.pi / 2.0
    fixed = None if fixed_aberrations is None else np.array(
        [fixed_aberrations[name] for name in PARAMETERS[:3]], dtype=np.float64
    )
    if fixed is not None and polish not in ("newton",):
        raise ValueError("Fixed aberrations use polish='newton'.")

    def sampler(**options):
        # multivariate TPE is flagged experimental; it is what finds the joint (C10, tilt) optimum here. The warning is
        # silenced for this construction only, not for the caller's process.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            return _make_sampler(seed=seed, multivariate=True, **options)

    def suggest(trial, with_sample: bool) -> list[float]:
        row = ([trial.suggest_float(name, value, value) for name, value in zip(PARAMETERS[:3], fixed)]
               if fixed is not None else
               [trial.suggest_float("C10", *c10_range), trial.suggest_float("C12", 0.0, c12_max), trial.suggest_float("phi12", -half_pi, half_pi)])
        if with_sample:
            row += [trial.suggest_float("tilt_row_mrad", -tilt_limit_mrad, tilt_limit_mrad),
                    trial.suggest_float("tilt_col_mrad", -tilt_limit_mrad, tilt_limit_mrad), trial.suggest_float(_DEPTH_SEARCH_LABEL, *depth_spread_range_A)]
        else:
            row += [0.0, 0.0, 0.0]
        return row

    def evaluate(trial, with_sample: bool) -> float:
        row = suggest(trial, with_sample)
        return -objective(row[0], row[1], row[2], (row[3], row[4]), row[5])

    def run(study, n_trials: int, with_sample: bool, progress=None) -> None:
        if objective_batch is None:
            study.optimize(lambda trial: evaluate(trial, with_sample), n_trials=int(n_trials),
                           callbacks=[lambda _study, _trial: progress.update(1)] if progress is not None else None)
            return
        # ask/tell in batches: TPE proposes batch_size trials, the backend evaluates them in one pass
        done = 0
        while done < n_trials:
            batch = [study.ask() for _ in range(min(batch_size, n_trials - done))]
            values = objective_batch(np.array([suggest(trial, with_sample) for trial in batch], dtype=np.float64))
            for trial, value in zip(batch, values, strict=True):
                study.tell(trial, -float(value))
            done += len(batch)
            if progress is not None:
                progress.update(len(batch))

    def wrap_phi(phi: float) -> float:
        # chi depends on 2 (phi - phi12): phi12 is periodic in pi, so it wraps instead of being bounded
        return (phi + half_pi) % math.pi - half_pi

    def single(point) -> float:
        row = np.array([[point[0], abs(point[1]), wrap_phi(point[2]), point[3], point[4], abs(point[5])]], dtype=np.float64)
        if fixed is not None:
            row[0, :3] = fixed
        return float(objective_batch(row)[0]) if objective_batch is not None else objective(row[0, 0], row[0, 1], row[0, 2], (row[0, 3], row[0, 4]), row[0, 5])

    timings = {}
    stage_started = time.perf_counter()
    thin = None
    if compare_standard:
        thin = optuna.create_study(sampler=sampler())
        run(thin, 200, with_sample=False)
    timings["standard_comparison_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    if trials == 0 and start is None:
        raise ValueError("trials=0 polishes from start: give the point to refine from.")
    ranked = []
    if trials > 0:
        study = optuna.create_study(sampler=sampler(n_startup_trials=max(30, trials // 6)))
        progress = None
        if verbose:
            from tqdm.auto import tqdm
            progress = tqdm(total=trials, desc="SSB tilt fit")
        run(study, int(trials), with_sample=True, progress=progress)
        if progress is not None:
            progress.close()
        ranked = sorted(
            (trial for trial in study.get_trials(deepcopy=False) if trial.value is not None),
            key=lambda trial: trial.value,
        )
    timings["coarse_search_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    starts: list[np.ndarray] = []
    if start is not None:
        # C12 >= 0 and phi12 wrapped, as searched: a fit kept from the other 180-degree branch carries -C12
        starts.append(from_newton(to_newton(np.asarray(start, dtype=np.float64))))
        if fixed is not None:
            starts[-1][:3] = fixed
    # Nelder-Mead on the objective scaled to ~1 by the best trial: its function-change test (fatol) is absolute, and at the
    # raw scale (~1e12) it never passes, so the polish ran to maxiter (641 of 704 evaluations on the full logic field).
    best_point = np.array([ranked[0].params[_search_label(name)] for name in PARAMETERS]) if ranked else starts[0]
    best_value = -float(ranked[0].value) if ranked else single(starts[0])
    scale = max(abs(best_value), 1e-30)
    # The polish keeps the search box (unbounded it walked the depth spread below the 20 A floor) except phi12, which wraps.
    bounds = [c10_range, (0.0, c12_max), (None, None), (-tilt_limit_mrad, tilt_limit_mrad), (-tilt_limit_mrad, tilt_limit_mrad),
              depth_spread_range_A]
    # Polish from the best few distinct trials and keep the best: one start is fragile - a float-level difference between
    # backends sent the MPS search on the logic crop to a worse local optimum (fit 4.0e12 vs 5.9e12 at the CUDA answer).
    for trial in ranked:
        if len(starts) >= polish_starts:
            break
        point = np.array([trial.params[_search_label(name)] for name in PARAMETERS])
        if all(np.max(np.abs(point - kept) / np.array([50.0, 20.0, 0.3, 2.0, 2.0, 50.0])) > 1.0 for kept in starts):
            starts.append(point)
    if polish_starts == 0:      # refinement=None: the best trial (or the start) as is
        best = dict(zip(PARAMETERS, (float(value) for value in best_point), strict=True))
        fit = best_value
    else:
        if polish == "newton":
            if objective_batch is None:
                raise ValueError("polish='newton' evaluates its finite-difference designs in batches: it needs objective_batch.")
            lower = np.array([c10_range[0], -c12_max, -c12_max, -tilt_limit_mrad, -tilt_limit_mrad, depth_spread_range_A[0]])
            upper = np.array([c10_range[1], c12_max, c12_max, tilt_limit_mrad, tilt_limit_mrad, depth_spread_range_A[1]])
            polished = newton_polish(starts, lambda rows: np.asarray(objective_batch(rows), dtype=np.float64) / scale,
                                     lower, upper, c12_max, fixed_aberrations=fixed)
        elif batch_polish and objective_batch is not None and len(starts) > 1:
            from quantem.gpu.ssb.batch_refinement import _minimize_batch

            def evaluate_starts(points: np.ndarray) -> np.ndarray:
                rows = np.array([
                    [p[0], abs(p[1]), wrap_phi(p[2]), p[3], p[4], abs(p[5])]
                    for p in points
                ], dtype=np.float64)
                evaluate = refinement_objective_batch or objective_batch
                return -evaluate(rows) / scale

            polished = _minimize_batch(
                evaluate_starts, starts, bounds=bounds,
                options={"xatol": 0.05, "fatol": 1e-6, "maxiter": 400},
            )
        else:
            polished = [minimize(lambda point: -single(point) / scale, kept, method="Nelder-Mead", bounds=bounds,
                                 options={"xatol": 0.05, "fatol": 1e-6, "maxiter": 400}) for kept in starts]
        outcome = min(polished, key=lambda outcome: outcome.fun)
        best = dict(zip(PARAMETERS, (float(value) for value in outcome.x), strict=True))
        fit = float(-outcome.fun) * scale
    timings["refinement_s"] = time.perf_counter() - stage_started
    if not math.isfinite(fit) or fit <= 0:
        raise ValueError(
            "Joint tilt fitting found no positive finite model agreement. "
            "Check scan sampling and signal in the fitting band; use the "
            "phase-variance search (tilt=False, objective='phase_variance') when it has none."
        )
    best["C12"], best["depth_spread_A"], best["phi12"] = abs(best["C12"]), abs(best["depth_spread_A"]), wrap_phi(best["phi12"])
    if fixed is not None:
        best.update(zip(PARAMETERS[:3], map(float, fixed), strict=True))
    records = [{"number": trial.number, "loss": float(trial.value),
                "params": {name: trial.params[_search_label(name)] for name in PARAMETERS if _search_label(name) in trial.params}}
               for trial in ranked if math.isfinite(trial.value)]
    records.sort(key=lambda record: record["number"])
    common = {"band_inv_A": list(band_inv_A), "trials": int(trials), "trial_records": records, "timings": timings}
    if thin is None:
        return {**best, "fit": fit, "standard_fit": None, "gain": None, "standard": None, **common}
    standard_fit = float(-thin.best_value)
    return {**best, "fit": fit, "standard_fit": standard_fit, "gain": fit / standard_fit if standard_fit > 0 else float("nan"),
            "standard": {name: float(value) for name, value in thin.best_params.items()}, **common}


# ---

# Newton polish, in the coordinates z = (C10, C12 cos 2 phi12, C12 sin 2 phi12, tilt_row, tilt_col, depth_spread_A): the
# astigmatism as a vector is smooth through C12 = 0, where (C12, phi12) is singular and phi12 undefined.
# Finite-difference steps (engine units: A, A, A, mrad, mrad, A) and the step below which the polish has converged.
NEWTON_STEP = np.array([2.0, 2.0, 2.0, 0.25, 0.25, 5.0])
NEWTON_TOLERANCE = np.array([0.05, 0.05, 0.05, 0.005, 0.005, 0.5])


def newton_polish(starts, objective_rows, lower, upper, c12_max, max_iterations=30, *, fixed_aberrations=None):
    """Maximise the objective from every start at once by Newton steps on finite-difference models.

    Each iteration evaluates, for every unfinished start, the design around its current point (``_newton_design``) in ONE
    ``objective_rows`` call: 28 points the first time and after a long step (gradient and full Hessian), 13 otherwise
    (gradient and diagonal, the mixed terms kept). The step maximises the quadratic model inside a trust region and the
    search box (``_bounded_step``), in the coordinates z of ``to_newton``. A step that lowers the objective is undone
    and the region shrinks to a quarter of that step; a full-length step that raises it doubles the region. A start
    stops when its step is below ``NEWTON_TOLERANCE`` in every coordinate, or when it heads for the same point as a
    higher start (same maximum, walked once). After two steps only the highest start goes on: by then each has moved up
    to 24 finite-difference steps and shown where it climbs, and walking all three to the end cost twice the evaluations
    to reach, on every acquisition tried, one maximum.

    Why: the Nelder-Mead polish needs 340 to 510 evaluations a start, each a pass over G that cannot start before the
    previous one ends (360 to 620 passes of 6 to 23 ms for three starts on a 512 x 512 scan). Newton converges in a few
    steps near a smooth maximum, and every step's evaluations are independent, so they share passes over G (5 or 6 calls,
    260 to 400 evaluations for the three starts).
    """
    from scipy.optimize import OptimizeResult

    locked = fixed_aberrations is not None
    step = NEWTON_STEP[3:] if locked else NEWTON_STEP
    tolerance = NEWTON_TOLERANCE[3:] if locked else NEWTON_TOLERANCE
    lower, upper = (lower[3:], upper[3:]) if locked else (lower, upper)
    encode = (lambda point: np.asarray(point[3:], dtype=np.float64)) if locked else to_newton
    decode = (lambda z: np.concatenate((fixed_aberrations, z))) if locked else from_newton
    size = len(step)
    active = list(range(len(starts)))
    center = [encode(point) for point in starts]
    curvature = [np.zeros((size, size)) for _ in starts]
    full = [True] * len(starts)
    radius = [8.0] * len(starts)
    previous = [None] * len(starts)         # (center, value, gradient, hessian) of the last accepted point
    evaluations = [0] * len(starts)
    iterations = [0] * len(starts)
    converged = [False] * len(starts)
    heading = [None] * len(starts)          # where each start steps next: its proposal, or its point once converged

    def clip(z):
        z = np.clip(z, lower, upper)        # rounding only: _bounded_step already stays inside the box
        norm = 0.0 if locked else math.hypot(z[1], z[2])
        if not locked and norm > c12_max:
            z[1:3] *= c12_max / norm
        return z

    for iteration in range(max_iterations):
        if not active:
            break
        designs = [_newton_design(center[index], step, full[index]) for index in active]
        rows = np.array([decode(z) for design in designs for z in design])
        values = np.asarray(objective_rows(rows), dtype=np.float64)
        offset = 0
        for index, design in zip(list(active), designs, strict=True):
            design_values = values[offset:offset + len(design)]
            offset += len(design)
            evaluations[index] += len(design)
            iterations[index] += 1
            if previous[index] is not None and design_values[0] < previous[index][1]:
                # the step lowered the objective: go back, and step a quarter as far from the kept model (a quarter of
                # the region alone would repeat a step that was already shorter than that)
                rejected = float(np.linalg.norm((center[index] - previous[index][0]) / step))
                center[index], _, gradient, hessian = previous[index]
                radius[index] = min(radius[index], rejected) / 4.0
            else:
                gradient, hessian = _newton_model(design_values, step, curvature[index])
                moved = math.inf if previous[index] is None else float(np.linalg.norm((center[index] - previous[index][0]) / step))
                if previous[index] is not None and moved >= 0.99 * radius[index]:
                    radius[index] *= 2.0
                curvature[index] = hessian
                # the mixed terms are measured again only after a long step: near the maximum they barely change
                full[index] = previous[index] is not None and moved > 2.0
                previous[index] = (center[index].copy(), float(design_values[0]), gradient, hessian)
            proposal = clip(center[index] + _bounded_step(center[index], gradient, hessian, step, radius[index], lower, upper))
            if np.all(np.abs(proposal - center[index]) < tolerance):
                converged[index] = True
                active.remove(index)
                heading[index] = center[index]
            else:
                center[index] = heading[index] = proposal
        # two starts standing near each other, or heading for the same point, climb the same maximum: the higher goes on
        for index in list(active):
            for other in range(len(starts)):
                if other == index or previous[other] is None or previous[index] is None:
                    continue
                near = (np.all(np.abs(previous[index][0] - previous[other][0]) < 4 * step)
                        or np.all(np.abs(heading[index] - heading[other]) < 2 * step))
                if near and (previous[index][1], -index) < (previous[other][1], -other):
                    active.remove(index)
                    break
        if iteration == 1 and len(active) > 1:
            active = [max(active, key=lambda index: (previous[index][1], -index))]
    results = []
    for index in range(len(starts)):
        z, value = (previous[index][0], previous[index][1]) if previous[index] is not None else (encode(starts[index]), -math.inf)
        results.append(OptimizeResult(x=decode(z), fun=-value, nfev=evaluations[index], nit=iterations[index],
                                      success=converged[index]))
    return results


def to_newton(point) -> np.ndarray:
    """(C10, C12, phi12, tilt_row, tilt_col, depth_spread_A) -> z, astigmatism as the vector C12 (cos 2 phi12, sin 2 phi12)."""
    return np.array([point[0], point[1] * math.cos(2 * point[2]), point[1] * math.sin(2 * point[2]), point[3], point[4], point[5]],
                    dtype=np.float64)


def from_newton(z) -> np.ndarray:
    """z -> the objective's row: C12 = |(a, b)|, phi12 = atan2(b, a) / 2 in (-pi/2, pi/2], depth spread |t|."""
    return np.array([z[0], math.hypot(z[1], z[2]), 0.5 * math.atan2(z[2], z[1]), z[3], z[4], abs(z[5])], dtype=np.float64)


def _newton_design(center, step, hessian: bool) -> np.ndarray:
    """The points whose values give the gradient (and the Hessian) at center by finite differences.

    Always center and center +- h_i e_i (central gradient, diagonal curvature); with hessian also center + h_i e_i + h_j e_j
    for i < j, whose mixed second difference (f_ij - f_i - f_j + f_0) / (h_i h_j) is the off-diagonal curvature.
    6 parameters: 13 points, or 28 with the Hessian.
    """
    size = len(center)
    points = [center]
    for i in range(size):
        for sign in (1.0, -1.0):
            point = center.copy()
            point[i] += sign * step[i]
            points.append(point)
    if hessian:
        for i in range(size):
            for j in range(i + 1, size):
                point = center.copy()
                point[i] += step[i]
                point[j] += step[j]
                points.append(point)
    return np.array(points)


def _newton_model(values, step, curvature):
    """Gradient and curvature at the design's center from its values (``_newton_design`` order).

    g_i = (f(+i) - f(-i)) / 2 h_i, H_ii = (f(+i) - 2 f_0 + f(-i)) / h_i^2, and with 28 values H_ij = (f(+i+j) - f(+i) - f(+j)
    + f_0) / (h_i h_j); with 13 values the off-diagonal terms are kept from ``curvature`` (the last full design), so a
    gradient step costs 13 evaluations instead of 28 once the polish is near its maximum, where the Hessian barely changes.
    """
    size = len(step)
    plus, minus = values[1:2 * size + 1:2], values[2:2 * size + 2:2]
    gradient = (plus - minus) / (2 * step)
    hessian = curvature.copy()
    hessian[np.diag_indices(size)] = (plus - 2 * values[0] + minus) / step ** 2
    if len(values) > 2 * size + 1:
        mixed = iter(values[2 * size + 1:])
        for i in range(size):
            for j in range(i + 1, size):
                hessian[i, j] = hessian[j, i] = (next(mixed) - plus[i] - plus[j] + values[0]) / (step[i] * step[j])
    return gradient, hessian


def _trust_step(gradient, hessian, step, radius) -> np.ndarray:
    """The step s maximising g.s + s.H.s / 2 within |s / h| <= radius (h the finite-difference steps).

    In scaled coordinates y = s / h the model is gy.y + y.Hy.y / 2 with gy = h g and Hy = h H h. Its maximiser inside the
    ball is y = (mu I - Hy)^-1 gy with mu >= 0 the smallest value that makes mu I - Hy positive definite and |y| <= radius
    (mu = 0 is the plain Newton step when Hy is negative definite and the step fits): Levenberg-Marquardt, found by
    bisection on mu over the eigenvalues of Hy.
    """
    scaled_gradient, scaled_hessian = gradient * step, hessian * step[:, None] * step[None, :]
    eigenvalues, vectors = np.linalg.eigh(scaled_hessian)
    along = vectors.T @ scaled_gradient

    def scaled(mu):
        return vectors @ (along / (mu - eigenvalues))

    low = max(0.0, float(eigenvalues.max()))
    if eigenvalues.max() < 0 and np.linalg.norm(scaled(0.0)) <= radius:
        return scaled(0.0) * step
    # |y(mu)| falls from infinity just above the largest eigenvalue to 0: bisect for |y| = radius
    high = low + np.linalg.norm(scaled_gradient) / radius + 1e-30
    low += 1e-12 * (abs(low) + 1.0)
    for _ in range(100):
        middle = 0.5 * (low + high)
        if np.linalg.norm(scaled(middle)) > radius:
            low = middle
        else:
            high = middle
    return scaled(high) * step


def _bounded_step(center, gradient, hessian, step, radius, lower, upper) -> np.ndarray:
    """The trust step from center that stays inside the box [lower, upper].

    A coordinate the step pushes past a bound is held on the bound (displacement p), and the others are solved again on
    that face, where the model's gradient is g_f + H_fp p: clipping the free step instead would leave the other
    coordinates at the maximum of the unbounded model, which is not the maximum on the face (a depth spread on its floor
    left C10 1 A and the tilt 0.02 mrad short). Held coordinates are chosen afresh at every call, so one leaves its
    bound as soon as the model says so.
    """
    free = np.ones(len(center), dtype=bool)
    held = np.zeros(len(center))
    for _ in range(len(center)):
        move = held.copy()
        if free.any():
            on_face = gradient[free] + hessian[np.ix_(free, ~free)] @ held[~free]
            move[free] = _trust_step(on_face, hessian[np.ix_(free, free)], step[free], radius)
        outside = free & ((center + move < lower) | (center + move > upper))
        if not outside.any():
            break
        held[outside] = np.clip(center + move, lower, upper)[outside] - center[outside]
        free &= ~outside
    return move
