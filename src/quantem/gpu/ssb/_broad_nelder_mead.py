"""Bounded simplex refinement of the broad CUDA agreement objective."""

import numpy as np
from scipy.optimize import minimize

from ._broad_newton import Tracker


def nelder_mead(
    track: Tracker, start: np.ndarray, bounds: np.ndarray,
) -> tuple[int, str]:
    """Refine the broad winner using the established research simplex policy.

    The host controls a float64 simplex; every scientific objective evaluation
    is dispatched through ``track`` to the selected CUDA objective. Nonzero
    initial steps let initially absent higher-order terms participate.
    """
    steps = np.array([.02, .02, .02, .05, .05, .05] + [.1] * (len(start) - 6))
    simplex = np.tile(start, (len(start) + 1, 1))
    for index, step in enumerate(steps):
        direction = step if start[index] + step <= bounds[index, 1] else -step
        simplex[index + 1, index] = np.clip(
            start[index] + direction, bounds[index, 0], bounds[index, 1]
        )
    outcome = minimize(
        lambda point: -float(track(point)[0]), start,
        method="Nelder-Mead", bounds=bounds,
        options={
            "initial_simplex": simplex, "adaptive": True,
            "xatol": .001, "fatol": 1e-7,
            "maxfev": track.cap - track.nfev, "maxiter": track.cap,
        },
    )
    return int(outcome.nit), str(outcome.message)
