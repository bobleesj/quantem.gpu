"""Batching preserves the independent refinement searches and their outcomes."""

import numpy as np
import pytest
from scipy.optimize import minimize, rosen

from quantem.gpu.ssb.batch_refinement import _minimize_batch


def test_refinement_preserves_each_scipy_trajectory():
    starts = [np.array([-1.2, 1.0]), np.array([0.1, 0.6]), np.array([1.4, 1.1])]
    bounds = [(-2.0, 2.0)] * 2
    options = {"maxiter": 400, "xatol": 0.05, "fatol": 1e-6, "return_all": True}
    expected = [minimize(rosen, start, method="Nelder-Mead", bounds=bounds,
                         options=options) for start in starts]
    actual = _minimize_batch(
        lambda points: np.array([rosen(point) for point in points]),
        starts, bounds=bounds, options=options,
    )
    for serial, batched in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(batched.allvecs, serial.allvecs)
        np.testing.assert_array_equal(batched.x, serial.x)
        assert batched.fun == serial.fun
        assert batched.nfev == serial.nfev
        assert batched.success == serial.success


def test_joint_search_preserves_trials_parameters_and_diagnostic_gain():
    from quantem.gpu.ssb.thick_sample_fit import fit_sample_search

    target = np.array([80.0, 20.0, 0.2, 3.0, -4.0, 150.0])
    widths = np.array([100.0, 50.0, 0.5, 10.0, 10.0, 200.0])

    def objective(rows):
        return 2.0 - np.sum(((rows - target) / widths) ** 2, axis=1)

    def scalar(c10, c12, phi12, tilt, thickness):
        return float(objective(np.array([[c10, c12, phi12, *tilt, thickness]]))[0])

    settings = dict(objective_batch=objective, trials=24, seed=42, verbose=False)
    expected = fit_sample_search(scalar, **settings)
    actual = fit_sample_search(scalar, batch_polish=True, **settings)
    actual_timings = actual.pop("timings")
    expected_timings = expected.pop("timings")
    assert actual_timings.keys() == expected_timings.keys()
    assert all(np.isfinite(value) and value >= 0 for value in actual_timings.values())
    assert actual == expected  # Every scientific field and trial remains exact.


def test_objective_failure_releases_refinement_workers():
    def failed_objective(points):
        raise RuntimeError("Objective evaluation failed")

    with pytest.raises(RuntimeError, match="Objective evaluation failed"):
        _minimize_batch(
            failed_objective, [np.ones(2), np.zeros(2)],
            bounds=[(-2.0, 2.0)] * 2, options={"maxiter": 10},
        )
