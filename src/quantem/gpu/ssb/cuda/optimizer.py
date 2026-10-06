"""Aberration search on the CUDA phase-variance objective: Optuna TPE, Nelder-Mead, and the thick-sample fit.

Optuna's ask/tell API collects a batch of suggestions, which one ``PhaseVarianceObjective.loss_batch`` call evaluates,
and the Nelder-Mead simplex evaluations are grouped the same way.
"""

import math

import cupy as cp
import numpy as np

from quantem.gpu.ssb.cuda.engine import SSBEngine
from quantem.gpu.ssb.cuda.objective import PhaseVarianceObjective
from quantem.gpu.ssb.thick_sample_fit import fit_sample_search


def batch_optimize(
    objective: PhaseVarianceObjective,
    aberrations: dict,
    aberration_defaults: dict,
    n_trials: int = 50,
    batch_size: int = 16,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict, float, list[dict]]:
    """Run Optuna TPE with one batched loss evaluation per ``batch_size`` trials.

    Parameters
    ----------
    objective : PhaseVarianceObjective
        Loss of the rotation-cached engine.
    aberrations : dict
        Parameter specs: key -> (low, high) tuple or fixed float. Keys: "C10_nm", "C12_nm", "phi12_deg".
    aberration_defaults : dict
        Current aberration values {C10, C12, phi12} for parameters ``aberrations`` leaves out.
    n_trials : int
        Total number of Optuna trials.
    batch_size : int
        Number of trials to evaluate per GPU call.
    seed : int
        Random seed for TPE sampler reproducibility.
    verbose : bool
        Show the progress bar.

    Returns
    -------
    best_params : dict
        Best parameters found (Optuna param names).
    best_value : float
        Best variance loss value.
    trials : list[dict]
        Every evaluated trial in order, ``{"number", "params": {...}, "loss"}``; the result keeps it so the loss
        landscape (and degenerate minima with the same loss at very different C10) can be inspected.
    """
    # Optuna and tqdm load only when a fit runs, not with every CUDA SSB session
    import optuna
    from optuna.samplers import TPESampler
    from tqdm.auto import tqdm

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # one output buffer for every batch
    out_buffer = cp.empty(batch_size, dtype=cp.float32)
    pbar = tqdm(total=n_trials, desc="SSB optimize", disable=not verbose,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    trial_history = []
    n_completed = 0
    while n_completed < n_trials:
        current_batch = min(batch_size, n_trials - n_completed)
        trials = [study.ask() for _ in range(current_batch)]
        c10_arr = np.empty(current_batch, dtype=np.float32)
        c12_arr = np.empty(current_batch, dtype=np.float32)
        phi12_arr = np.empty(current_batch, dtype=np.float32)
        for i, trial in enumerate(trials):
            c10_arr[i] = _suggest(trial, "C10_nm", aberrations.get("C10_nm", aberration_defaults.get("C10", 0.0)))
            c12_arr[i] = _suggest(trial, "C12_nm", aberrations.get("C12_nm", aberration_defaults.get("C12", 0.0)))
            phi12_deg = _suggest(trial, "phi12_deg", aberrations.get("phi12_deg", math.degrees(aberration_defaults.get("phi12", 0.0))))
            phi12_arr[i] = math.radians(phi12_deg)
        losses_gpu = objective.loss_batch(c10_arr, c12_arr, phi12_arr, out=out_buffer[:current_batch])
        losses_cpu = cp.asnumpy(losses_gpu[:current_batch])
        for i, trial in enumerate(trials):
            study.tell(trial, float(losses_cpu[i]))
            trial_history.append({
                "number": trial.number,
                "params": {"C10_nm": float(c10_arr[i]), "C12_nm": float(c12_arr[i]),
                           "phi12_deg": math.degrees(float(phi12_arr[i]))},
                "loss": float(losses_cpu[i]),
            })
        n_completed += current_batch
        pbar.update(current_batch)
    pbar.close()
    return study.best_params, study.best_value, trial_history


def _suggest(trial, name: str, spec: tuple[float, float] | float) -> float:
    """Suggest a value in a ``(low, high)`` range, or return a locked value unchanged."""
    if isinstance(spec, tuple):
        return trial.suggest_float(name, spec[0], spec[1])
    return spec


# =========================================================================
#  Batched Nelder-Mead refinement
# =========================================================================

def batch_nelder_mead(
    objective: PhaseVarianceObjective,
    x0: np.ndarray,
    xatol: float = 0.1,
    fatol: float = 1e-8,
    max_iter: int = 300,
) -> tuple[np.ndarray, float, int]:
    """
    Nelder-Mead simplex optimization with batched vertex evaluations.

    Standard Nelder-Mead in 3D has a simplex of 4 vertices. At each iteration
    it evaluates 1-3 candidate points (reflect, expand, contract). This
    implementation batches all vertex evaluations where possible.

    However, Nelder-Mead is inherently sequential - each step depends on
    the previous result. The initial simplex (4 vertices) and a shrink
    (3 vertices) are evaluated in one ``loss_batch`` call each.

    Parameters
    ----------
    objective : PhaseVarianceObjective
        Loss of the rotation-cached engine.
    x0 : np.ndarray
        Starting point [C10, C12, phi12] (3 values).
    xatol, fatol : float
        Convergence tolerances.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    best_x : np.ndarray
        Optimized [C10, C12, phi12].
    best_loss : float
        Loss at best_x.
    n_evals : int
        Total number of variance evaluations.
    """
    n = len(x0)
    assert n == 3, "Expected 3 parameters: C10, C12, phi12"

    # Build initial simplex (n+1 = 4 vertices)
    simplex = np.empty((n + 1, n), dtype=np.float64)
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        # Standard Nelder-Mead initial step: 5% of value or 0.00025
        h = max(abs(x0[i]) * 0.05, 0.00025)
        simplex[i + 1, i] += h

    # Evaluate all 4 vertices in one batch call
    f_values = np.empty(n + 1, dtype=np.float64)
    c10_arr = simplex[:, 0].astype(np.float32)
    c12_arr = simplex[:, 1].astype(np.float32)
    phi12_arr = simplex[:, 2].astype(np.float32)
    losses_gpu = objective.loss_batch(c10_arr, c12_arr, phi12_arr)
    f_values[:] = cp.asnumpy(losses_gpu).astype(np.float64)
    n_evals = n + 1

    # Standard Nelder-Mead coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for iteration in range(max_iter):
        # Sort vertices by function value
        order = np.argsort(f_values, kind="stable")
        simplex = simplex[order]
        f_values = f_values[order]

        # Check convergence
        x_spread = np.max(np.abs(simplex[-1] - simplex[0]))
        f_spread = abs(f_values[-1] - f_values[0])
        if x_spread < xatol and f_spread < fatol:
            break

        # Centroid of all vertices except worst
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = _eval_single(objective, x_r)
        n_evals += 1

        if f_values[0] <= f_r < f_values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = _eval_single(objective, x_e)
            n_evals += 1
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        # Contraction
        if f_r < f_values[-1]:
            # Outside contraction
            x_c = centroid + rho * (x_r - centroid)
            f_c = _eval_single(objective, x_c)
            n_evals += 1
            if f_c <= f_r:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue
        else:
            # Inside contraction
            x_c = centroid - rho * (centroid - simplex[-1])
            f_c = _eval_single(objective, x_c)
            n_evals += 1
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue

        # Shrink: move all vertices toward best - batch evaluate n vertices
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
        losses_gpu = objective.loss_batch(
            simplex[1:, 0].astype(np.float32),
            simplex[1:, 1].astype(np.float32),
            simplex[1:, 2].astype(np.float32),
        )
        f_values[1:] = cp.asnumpy(losses_gpu).astype(np.float64)
        n_evals += 3

    best_idx = np.argsort(f_values, kind="stable")[0]
    return simplex[best_idx], float(f_values[best_idx]), n_evals


def _eval_single(objective: PhaseVarianceObjective, x: np.ndarray) -> float:
    """Loss of one simplex point, the four Nelder-Mead moves' shared call."""
    return float(objective.loss(float(x[0]), float(x[1]), float(x[2])))


# =========================================================================
#  Thick-sample fit: sample tilt and thickness with the aberrations
# =========================================================================

def fit_sample(
    accel: SSBEngine,
    *,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    **options,
) -> dict[str, object]:
    """Fit C10, C12, phi12, sample tilt and thickness by maximising ``ThickSample.fit`` (search: ``ssb.thick_sample_fit``)."""
    return fit_sample_search(lambda c10, c12, phi12, tilt, thickness: accel.thick.fit(c10, c12, phi12, tilt, thickness, band_inv_A),
                             objective_batch=lambda rows: accel.thick.fit_batch(rows, band_inv_A),
                             band_inv_A=band_inv_A, **options)
