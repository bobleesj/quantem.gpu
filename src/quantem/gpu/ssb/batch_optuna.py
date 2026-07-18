"""
Batched Optuna optimization for SSB aberration fitting.

The standard Optuna study.optimize() evaluates one trial at a time. Each call
to variance_loss() internally pads to batch=4, wasting 3/4 of GPU compute.
This module uses Optuna's ask/tell API to collect N trial suggestions, pack
them into arrays, and evaluate them in a single variance_loss_batch() call -
eliminating per-trial GPU sync overhead and fully utilizing batch parallelism.

Typical speedup: 4-8x over sequential trials for variance metric.
"""

import math
import numpy as np
import cupy as cp

from quantem.gpu.ssb.engine import SSBEngine

# =========================================================================
#  Core batch evaluation
# =========================================================================

def _build_param_arrays(
    trials: list,
    aberrations: dict,
    rotation_angle_deg_spec: tuple | float | None,
    defaults: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float | None]]:
    """
    Extract parameter arrays from a list of Optuna trials.

    Returns (c10_arr, c12_arr, phi12_arr, rotation_rads) as numpy float32
    arrays ready for variance_loss_batch.
    """
    n = len(trials)
    c10_arr = np.empty(n, dtype=np.float32)
    c12_arr = np.empty(n, dtype=np.float32)
    phi12_arr = np.empty(n, dtype=np.float32)
    rotation_rads: list[float | None] = [None] * n

    for i, trial in enumerate(trials):
        c10_arr[i] = _suggest(trial, "C10_nm", aberrations.get("C10_nm", defaults.get("C10", 0.0)))
        c12_arr[i] = _suggest(trial, "C12_nm", aberrations.get("C12_nm", defaults.get("C12", 0.0)))
        phi12_deg = _suggest(trial, "phi12_deg", aberrations.get("phi12_deg", math.degrees(defaults.get("phi12", 0.0))))
        phi12_arr[i] = math.radians(phi12_deg)

        if rotation_angle_deg_spec is not None and isinstance(rotation_angle_deg_spec, tuple):
            rot_deg = _suggest(trial, "rotation_angle_deg", rotation_angle_deg_spec)
            rotation_rads[i] = math.radians(rot_deg)

    return c10_arr, c12_arr, phi12_arr, rotation_rads

def _suggest(trial, name: str, spec: tuple[float, float] | float) -> float:
    """Get value from trial: suggest_float if tuple (low, high), else fixed."""
    if isinstance(spec, tuple):
        return trial.suggest_float(name, spec[0], spec[1])
    return spec

def batch_optimize(
    accel: SSBEngine,
    aberrations: dict,
    rotation_angle_rad: float,
    rotation_angle_deg_spec: tuple | float | None,
    aberration_defaults: dict,
    n_trials: int = 50,
    batch_size: int = 16,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict, float]:
    """
    Run Optuna TPE optimization with batched variance_loss evaluation.

    Instead of evaluating one trial at a time (each padded to batch=4),
    this asks for `batch_size` trials at once, packs parameters into arrays,
    and evaluates them in a single variance_loss_batch() call.

    Parameters
    ----------
    accel : SSBEngine
        Initialized and rotation-cached SSB engine.
    aberrations : dict
        Parameter specs: key -> (low, high) tuple or fixed float.
        Keys: "C10_nm", "C12_nm", "phi12_deg".
    rotation_angle_rad : float
        Current rotation angle in radians.
    rotation_angle_deg_spec : tuple or float or None
        If tuple, rotation is optimized within range.
    aberration_defaults : dict
        Current aberration values {C10, C12, phi12} for fallback.
    n_trials : int
        Total number of Optuna trials.
    batch_size : int
        Number of trials to evaluate per GPU call. 16-32 recommended.
    seed : int
        Random seed for TPE sampler reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    best_params : dict
        Best parameters found (Optuna param names).
    best_value : float
        Best variance loss value.
    trials : list[dict]
        Full history of every trial that was evaluated, in the order they
        were tried. Each entry has ``{"params": {...}, "loss": float}``.
        Used by the Screening dashboard (#26) to visualize the loss
        landscape and flag degenerate local minima that give the same
        loss at very different C10 values.
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    has_rotation_opt = (
        rotation_angle_deg_spec is not None
        and isinstance(rotation_angle_deg_spec, tuple)
    )

    # When rotation is being optimized, each trial may need a different
    # rotation cache, so batching across different rotations is not possible
    # with the current engine design. Fall back to sequential in that case.
    if has_rotation_opt:
        return _sequential_optimize(
            accel, aberrations, rotation_angle_rad,
            rotation_angle_deg_spec, aberration_defaults,
            n_trials, seed, verbose,
        )

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Pre-allocate output buffer on GPU to avoid per-batch allocation
    out_buffer = cp.empty(batch_size, dtype=cp.float32)

    from tqdm.auto import tqdm
    pbar = tqdm(total=n_trials, desc="SSB optimize", disable=not verbose,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    n_completed = 0
    while n_completed < n_trials:
        # Ask for a batch of trials
        current_batch = min(batch_size, n_trials - n_completed)
        trials = [study.ask() for _ in range(current_batch)]

        # Build parameter arrays from all trials
        c10_arr, c12_arr, phi12_arr, _ = _build_param_arrays(
            trials, aberrations, None, aberration_defaults,
        )

        # Single GPU call for the entire batch - no per-trial sync
        if current_batch >= 4:
            losses_gpu = accel.variance_loss_batch(
                c10_arr, c12_arr, phi12_arr,
                out=out_buffer[:current_batch] if current_batch <= batch_size else None,
            )
        else:
            pad_size = 4
            c10_padded = np.zeros(pad_size, dtype=np.float32)
            c12_padded = np.zeros(pad_size, dtype=np.float32)
            phi12_padded = np.zeros(pad_size, dtype=np.float32)
            c10_padded[:current_batch] = c10_arr
            c12_padded[:current_batch] = c12_arr
            phi12_padded[:current_batch] = phi12_arr
            for j in range(current_batch, pad_size):
                c10_padded[j] = c10_arr[0]
                c12_padded[j] = c12_arr[0]
                phi12_padded[j] = phi12_arr[0]
            losses_gpu = accel.variance_loss_batch(c10_padded, c12_padded, phi12_padded)
            losses_gpu = losses_gpu[:current_batch]

        losses_cpu = cp.asnumpy(losses_gpu[:current_batch])

        for i, trial in enumerate(trials):
            study.tell(trial, float(losses_cpu[i]))

        n_completed += current_batch
        pbar.update(current_batch)

    pbar.close()

    # Dump every completed trial as a plain dict so the caller can
    # persist it (sidecar JSON) without needing Optuna in downstream
    # code. See #26 for the scatter plot consumer.
    trial_history = [
        {"params": dict(t.params), "loss": float(t.value)}
        for t in study.trials if t.value is not None
    ]
    return study.best_params, study.best_value, trial_history

def _sequential_optimize(
    accel: SSBEngine,
    aberrations: dict,
    rotation_angle_rad: float,
    rotation_angle_deg_spec: tuple | float | None,
    aberration_defaults: dict,
    n_trials: int,
    seed: int,
    verbose: bool,
) -> tuple[dict, float, list[dict]]:
    """
    Fallback: sequential Optuna optimization when rotation is being optimized.

    Rotation optimization requires re-caching per trial, so batching is not
    possible. This uses standard study.optimize() with ask/tell.
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        c10 = _suggest(trial, "C10_nm", aberrations.get("C10_nm", aberration_defaults.get("C10", 0.0)))
        c12 = _suggest(trial, "C12_nm", aberrations.get("C12_nm", aberration_defaults.get("C12", 0.0)))
        phi12_deg = _suggest(trial, "phi12_deg", aberrations.get("phi12_deg", math.degrees(aberration_defaults.get("phi12", 0.0))))
        phi12 = math.radians(phi12_deg)
        if rotation_angle_deg_spec is not None and isinstance(rotation_angle_deg_spec, tuple):
            rot_deg = _suggest(trial, "rotation_angle_deg", rotation_angle_deg_spec)
            accel.cache_rotation(math.radians(rot_deg))
        return accel.variance_loss(c10, c12, phi12)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    trial_history = [
        {"params": dict(t.params), "loss": float(t.value)}
        for t in study.trials if t.value is not None
    ]
    return study.best_params, study.best_value, trial_history

# =========================================================================
#  Batched Nelder-Mead refinement
# =========================================================================

def batch_nelder_mead(
    accel: SSBEngine,
    x0: np.ndarray,
    xatol: float = 0.1,
    fatol: float = 1e-8,
    max_iter: int = 300,
    flat_fatol: float | None = None,
) -> tuple[np.ndarray, float, int]:
    """
    Nelder-Mead simplex optimization with batched vertex evaluations.

    Standard Nelder-Mead in 3D has a simplex of 4 vertices. At each iteration
    it evaluates 1-3 candidate points (reflect, expand, contract). This
    implementation batches all vertex evaluations where possible.

    However, Nelder-Mead is inherently sequential - each step depends on
    the previous result. The main win here is evaluating the initial simplex
    (4 vertices) in one batch call, and avoiding per-evaluation GPU sync by
    keeping results on GPU until the final comparison.

    For SSB with 3 parameters, the simplex has 4 vertices - exactly the
    minimum batch size for variance_loss_batch. So the padding waste is
    eliminated.

    Parameters
    ----------
    accel : SSBEngine
        Initialized and rotation-cached SSB engine.
    x0 : np.ndarray
        Starting point [C10, C12, phi12] (3 values).
    xatol, fatol : float
        Convergence tolerances.
    max_iter : int
        Maximum iterations.
    flat_fatol : float or None
        Optional early-stop threshold for flat objectives. When the initial
        simplex loss spread is already below this value, return the best
        vertex instead of spending serial Nelder-Mead evaluations on numerical
        noise.

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
    losses_gpu = accel.variance_loss_batch(c10_arr, c12_arr, phi12_arr)
    f_values[:] = cp.asnumpy(losses_gpu).astype(np.float64)
    n_evals = n + 1
    if flat_fatol is not None and np.ptp(f_values) <= float(flat_fatol):
        # Treat a numerically flat initial simplex as converged at the
        # incoming solution.  Picking the min vertex inside this band lets
        # sub-ULP objective noise move coefficients without a meaningful loss
        # improvement, which is especially visible in the 1024 sparse path.
        return simplex[0], float(f_values[0]), n_evals

    # Standard Nelder-Mead coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for iteration in range(max_iter):
        # Sort vertices by function value
        order = np.argsort(f_values)
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
        f_r = _eval_single(accel, x_r)
        n_evals += 1

        if f_values[0] <= f_r < f_values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            # Try expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = _eval_single(accel, x_e)
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
            f_c = _eval_single(accel, x_c)
            n_evals += 1
            if f_c <= f_r:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue
        else:
            # Inside contraction
            x_c = centroid - rho * (centroid - simplex[-1])
            f_c = _eval_single(accel, x_c)
            n_evals += 1
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue

        # Shrink: move all vertices toward best - batch evaluate n vertices
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
        # Batch evaluate the n shrunk vertices (3 points, padded to 4)
        c10_pad = np.zeros(4, dtype=np.float32)
        c12_pad = np.zeros(4, dtype=np.float32)
        phi12_pad = np.zeros(4, dtype=np.float32)
        c10_pad[:3] = simplex[1:, 0].astype(np.float32)
        c12_pad[:3] = simplex[1:, 1].astype(np.float32)
        phi12_pad[:3] = simplex[1:, 2].astype(np.float32)
        c10_pad[3] = c10_pad[0]
        c12_pad[3] = c12_pad[0]
        phi12_pad[3] = phi12_pad[0]
        losses_gpu = accel.variance_loss_batch(c10_pad, c12_pad, phi12_pad)
        f_values[1:] = cp.asnumpy(losses_gpu[:3]).astype(np.float64)
        n_evals += 3

    best_idx = np.argmin(f_values)
    return simplex[best_idx], float(f_values[best_idx]), n_evals


def _eval_single(accel: SSBEngine, x: np.ndarray) -> float:
    """Evaluate one optimizer point without duplicating exact fallback work."""
    if getattr(accel, "uses_optimizer_reconstruct_fallback", False):
        return float(accel.variance_loss(float(x[0]), float(x[1]), float(x[2])))
    c10_arr = np.full(4, x[0], dtype=np.float32)
    c12_arr = np.full(4, x[1], dtype=np.float32)
    phi12_arr = np.full(4, x[2], dtype=np.float32)
    result = accel.variance_loss_batch(c10_arr, c12_arr, phi12_arr)
    return float(result[0])
