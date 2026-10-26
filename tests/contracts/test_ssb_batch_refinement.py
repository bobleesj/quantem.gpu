"""Batching preserves the independent refinement searches and their outcomes."""

import numpy as np
import pytest
from scipy.optimize import minimize, rosen

from quantem.gpu.direct_ptycho.batch_refinement import _minimize_batch


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


def test_objective_failure_releases_refinement_workers():
    def failed_objective(points):
        raise RuntimeError("Objective evaluation failed")

    with pytest.raises(RuntimeError, match="Objective evaluation failed"):
        _minimize_batch(
            failed_objective, [np.ones(2), np.zeros(2)],
            bounds=[(-2.0, 2.0)] * 2, options={"maxiter": 10},
        )
