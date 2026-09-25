"""Backend-neutral search for the thick-sample SSB fit (defocus, astigmatism, sample tilt, thickness).

Every backend (CUDA engine, MPS, torch reference) supplies the same objective - the least-squares agreement of the
thick-sample SSB model with G(Q, K), larger is better (see ``SSBEngine.thick_fit``) - and this module owns the search, so the
backends differ only in how fast they evaluate it.

Search: multivariate TPE over all six parameters jointly, then Nelder-Mead from the best trial. C10 is searched jointly because
the thick model's C10 is the defocus at mid-depth, not the standard-SSB optimum; freeing only the tilt from the standard
optimum misses the answer (simulated BaTiO3 15 nm tilted (3, -4) mrad: joint search (3.0, -4.1); tilt-only (8.6, -6.3)).
300 trials give the same answer as 1500 on the simulation and on the logic device (Nelder-Mead polishes the rest). A
standard-SSB fit (thickness 0, 200 trials, the production budget) is run on the same objective so the reported gain compares
like with like. Units: C10, C12, thickness in the engine's C10 unit (Angstrom); phi12 rad; tilt mrad, scan frame (row, col).
"""
from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np

Objective = Callable[[float, float, float, tuple[float, float], float], float]
# (B, 6) rows of (C10, C12, phi12, tilt_row_mrad, tilt_col_mrad, thickness) -> (B,) values; lets a backend evaluate a batch
# of trials in one pass over G (CUDA: SSBEngine.thick_fit_batch)
BatchObjective = Callable[[np.ndarray], np.ndarray]
PARAMETERS = ("C10", "C12", "phi12", "tilt_row_mrad", "tilt_col_mrad", "thickness")


def fit_sample_search(
    objective: Objective,
    *,
    objective_batch: BatchObjective | None = None,
    batch_size: int = 8,
    trials: int = 300,
    tilt_limit_mrad: float = 25.0,
    thickness_range: tuple[float, float] = (20.0, 600.0),
    c10_range: tuple[float, float] = (-300.0, 300.0),
    c12_max: float = 200.0,
    seed: int = 0,
    verbose: bool = True,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    polish_starts: int = 3,
) -> dict[str, object]:
    """Maximise ``objective(C10, C12, phi12, (tilt_row, tilt_col), thickness)``; returns the fitted parameters and the gain."""
    import optuna
    from scipy.optimize import minimize

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # multivariate TPE is flagged experimental; it is what finds the joint (C10, tilt) optimum here
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    half_pi = math.pi / 2.0

    def standard(trial):
        return -objective(trial.suggest_float("C10", *c10_range), trial.suggest_float("C12", 0.0, c12_max),
                          trial.suggest_float("phi12", -half_pi, half_pi), (0.0, 0.0), 0.0)

    def thick(trial):
        return -objective(trial.suggest_float("C10", *c10_range), trial.suggest_float("C12", 0.0, c12_max),
                          trial.suggest_float("phi12", -half_pi, half_pi),
                          (trial.suggest_float("tilt_row_mrad", -tilt_limit_mrad, tilt_limit_mrad),
                           trial.suggest_float("tilt_col_mrad", -tilt_limit_mrad, tilt_limit_mrad)),
                          trial.suggest_float("thickness", *thickness_range))

    def suggest(trial, with_sample: bool) -> list[float]:
        row = [trial.suggest_float("C10", *c10_range), trial.suggest_float("C12", 0.0, c12_max), trial.suggest_float("phi12", -half_pi, half_pi)]
        if with_sample:
            row += [trial.suggest_float("tilt_row_mrad", -tilt_limit_mrad, tilt_limit_mrad),
                    trial.suggest_float("tilt_col_mrad", -tilt_limit_mrad, tilt_limit_mrad), trial.suggest_float("thickness", *thickness_range)]
        else:
            row += [0.0, 0.0, 0.0]
        return row

    def run(study, n_trials: int, with_sample: bool, progress=None) -> None:
        if objective_batch is None:
            study.optimize(thick if with_sample else standard, n_trials=int(n_trials),
                           callbacks=[lambda s, t: progress.update(1)] if progress is not None else None)
            return
        # ask/tell in batches: TPE proposes batch_size trials, the backend evaluates them in one pass
        done = 0
        while done < n_trials:
            batch = [study.ask() for _ in range(min(batch_size, n_trials - done))]
            values = objective_batch(np.array([suggest(t, with_sample) for t in batch], dtype=np.float64))
            for t, v in zip(batch, values):
                study.tell(t, -float(v))
            done += len(batch)
            if progress is not None:
                progress.update(len(batch))

    def wrap_phi(phi: float) -> float:
        # chi depends on 2 (phi - phi12): phi12 is periodic in pi, so it wraps instead of being bounded
        return (phi + half_pi) % math.pi - half_pi

    def single(x) -> float:
        row = np.array([[x[0], abs(x[1]), wrap_phi(x[2]), x[3], x[4], abs(x[5])]], dtype=np.float64)
        return float(objective_batch(row)[0]) if objective_batch is not None else objective(row[0, 0], row[0, 1], row[0, 2], (row[0, 3], row[0, 4]), row[0, 5])

    thin = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    run(thin, 200, with_sample=False)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=max(30, trials // 6)))
    progress = None
    if verbose:
        from tqdm.auto import tqdm
        progress = tqdm(total=trials, desc="SSB tilt fit")
    run(study, int(trials), with_sample=True, progress=progress)
    if progress is not None:
        progress.close()
    # Nelder-Mead on the objective scaled to ~1 by the best trial: its function-change test (fatol) is absolute, and at the
    # raw scale (~1e12) it never passes, so the polish ran to maxiter (641 of 704 evaluations on the full logic field).
    scale = max(abs(float(study.best_value)), 1e-30)
    # The polish keeps the search box (unbounded it walked thickness below the 20 A floor) except phi12, which wraps.
    bounds = [c10_range, (0.0, c12_max), (None, None), (-tilt_limit_mrad, tilt_limit_mrad), (-tilt_limit_mrad, tilt_limit_mrad),
              thickness_range]
    # Polish from the best few distinct trials and keep the best: one start is fragile - a float-level difference between
    # backends sent the MPS search on the logic crop to a worse local optimum (fit 4.0e12 vs 5.9e12 at the CUDA answer).
    ranked = sorted((t for t in study.trials if t.value is not None), key=lambda t: t.value)
    starts: list[np.ndarray] = []
    for t in ranked:
        x = np.array([t.params[name] for name in PARAMETERS])
        if all(np.max(np.abs(x - y) / np.array([50.0, 20.0, 0.3, 2.0, 2.0, 50.0])) > 1.0 for y in starts):
            starts.append(x)
        if len(starts) == polish_starts:
            break
    if polish_starts == 0:      # refinement=None: the best trial as is
        best = {name: float(study.best_params[name]) for name in PARAMETERS}
        fit = float(-study.best_value)
    else:
        polished = [minimize(lambda x: -single(x) / scale, x0, method="Nelder-Mead", bounds=bounds,
                             options={"xatol": 0.05, "fatol": 1e-6, "maxiter": 400}) for x0 in starts]
        polish = min(polished, key=lambda r: r.fun)
        best = dict(zip(PARAMETERS, (float(v) for v in polish.x)))
        fit = float(-polish.fun) * scale
    best["C12"], best["thickness"], best["phi12"] = abs(best["C12"]), abs(best["thickness"]), wrap_phi(best["phi12"])
    standard_fit = float(-thin.best_value)
    return {**best, "fit": fit, "standard_fit": standard_fit, "gain": fit / standard_fit if standard_fit > 0 else float("nan"),
            "standard": {k: float(v) for k, v in thin.best_params.items()}, "band_inv_A": list(band_inv_A), "trials": int(trials)}
