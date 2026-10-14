"""The Newton polish of the thick-sample SSB fit (``ssb.thick_sample_fit``) on analytic objectives: no GPU needed.

The polish lands on a known maximum within its tolerance, in a few batched evaluations, including where the astigmatism
vanishes (C12 = 0, where (C12, phi12) is singular) and where the maximum lies on the thickness floor. With no trials the
search polishes from a given start alone.
"""

import math

import numpy as np
import pytest

from quantem.gpu.ssb import thick_sample_fit

# a maximum in engine units (C10, C12, phi12, tilt_row, tilt_col, thickness) and a curvature with the scales of the real
# objective: about 1e-4 relative drop for 1 A of defocus or astigmatism, 0.01 mrad of tilt, 3 A of thickness, coupled
MAXIMUM = np.array([-86.5, 17.0, 0.4, 0.72, 2.99, 135.0])
SCALES = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 3.0])


def _objective(maximum, seed=0):
    """exp(-(z - z*).A.(z - z*) / 2) in the Newton coordinates, A = S^-1 R R^T S^-1 1e-4: smooth, not quadratic, coupled."""
    rng = np.random.default_rng(seed)
    mixing = np.eye(6) + 0.3 * rng.normal(size=(6, 6))
    curvature = 1e-4 * (mixing @ mixing.T) / np.outer(SCALES, SCALES)
    center = thick_sample_fit.to_newton(maximum)
    calls = []

    def rows_to_values(rows):
        calls.append(len(rows))
        values = []
        for row in np.atleast_2d(rows):
            offset = thick_sample_fit.to_newton(row) - center
            values.append(math.exp(-0.5 * offset @ curvature @ offset))
        return np.array(values)
    return rows_to_values, calls


def _box():
    lower = np.array([-300.0, -200.0, -200.0, -25.0, -25.0, 20.0])
    upper = np.array([300.0, 200.0, 200.0, 25.0, 25.0, 600.0])
    return lower, upper


def _polish(maximum, starts):
    objective, calls = _objective(maximum)
    lower, upper = _box()
    results = thick_sample_fit.newton_polish([np.asarray(start, dtype=float) for start in starts], objective, lower, upper, 200.0)
    kept = min(results, key=lambda result: result.fun)
    return kept, calls


def _distance(found, maximum) -> np.ndarray:
    return np.abs(thick_sample_fit.to_newton(found) - thick_sample_fit.to_newton(maximum))


# ---


def test_newton_polish_reaches_the_maximum_from_three_starts_in_few_batched_passes():
    starts = [MAXIMUM + offset for offset in ([3.0, 2.0, 0.1, 0.2, -0.1, 10.0], [-4.0, -3.0, -0.2, -0.1, 0.15, -15.0],
                                              [1.0, 4.0, 0.3, 0.05, 0.05, 20.0])]
    kept, calls = _polish(MAXIMUM, starts)
    assert np.all(_distance(kept.x, MAXIMUM) < 5 * thick_sample_fit.NEWTON_TOLERANCE)
    assert -kept.fun == pytest.approx(1.0, abs=1e-9)
    # every iteration is one objective call for all starts together; Nelder-Mead needed 350 to 610 calls a start
    assert len(calls) <= 12
    assert sum(calls) <= 3 * 28 + 11 * 3 * 13


def test_newton_polish_is_smooth_where_the_astigmatism_vanishes():
    maximum = MAXIMUM.copy()
    maximum[1] = 0.0
    kept, _ = _polish(maximum, [maximum + np.array([2.0, 3.0, 1.0, 0.1, -0.1, 8.0])])
    assert np.all(_distance(kept.x, maximum) < 5 * thick_sample_fit.NEWTON_TOLERANCE)


def test_newton_polish_finds_the_maximum_on_the_thickness_floor():
    """With the free maximum below the 20 A floor, the polish ends at the maximum ON the floor: the other coordinates move
    to where the objective peaks at thickness 20 A (z_f = c_f - A_ff^-1 A_fp (20 - c_p) for a Gaussian of curvature A), not
    where it peaks without the bound."""
    maximum = MAXIMUM.copy()
    maximum[5] = 5.0
    kept, _ = _polish(maximum, [MAXIMUM])
    rng = np.random.default_rng(0)
    mixing = np.eye(6) + 0.3 * rng.normal(size=(6, 6))
    curvature = 1e-4 * (mixing @ mixing.T) / np.outer(SCALES, SCALES)
    on_floor = thick_sample_fit.to_newton(maximum)
    on_floor[:5] -= np.linalg.solve(curvature[:5, :5], curvature[:5, 5]) * (20.0 - 5.0)
    on_floor[5] = 20.0
    assert kept.x[5] == pytest.approx(20.0, abs=1e-9)
    assert np.all(np.abs(thick_sample_fit.to_newton(kept.x) - on_floor) < 5 * thick_sample_fit.NEWTON_TOLERANCE)


def test_search_with_no_trials_polishes_from_the_given_start(monkeypatch):
    """trials=0 with a start (the previous scan's fit) creates no study and lands on the maximum from the start alone."""
    import optuna

    from quantem.gpu.ssb.thick_sample_fit import fit_sample_search

    def no_study(*args, **kwargs):
        raise AssertionError("trials=0 must not search")
    monkeypatch.setattr(optuna, "create_study", no_study)
    objective, calls = _objective(MAXIMUM)
    start = MAXIMUM + np.array([4.0, 3.0, 0.2, 0.5, -0.4, 20.0])
    start[1] = -start[1]          # a fit kept from the other 180 degree branch carries -C12: the same aberration at phi12 + 90 deg
    start[2] -= math.pi / 2
    fit = fit_sample_search(lambda *row: None, objective_batch=objective, trials=0, start=start, polish="newton",
                            compare_standard=False, verbose=False)
    found = np.array([fit[name] for name in thick_sample_fit.PARAMETERS])
    assert np.all(_distance(found, MAXIMUM) < 5 * thick_sample_fit.NEWTON_TOLERANCE)
    assert fit["gain"] is None and fit["trials"] == 0
    assert sum(calls) < 200       # fewer evaluations than the search it replaces has trials

