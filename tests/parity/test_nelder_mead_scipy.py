"""The tilt fit's Nelder-Mead polish against ``scipy.optimize.minimize``: same points, same result, bit for bit."""

import numpy as np
import pytest

from quantem.gpu.ssb.thick_sample_fit import _nelder_mead

LOWER = np.array([-300.0, 0.0, -np.inf, -25.0, -25.0, 20.0])
UPPER = np.array([300.0, 200.0, np.inf, 25.0, 25.0, 600.0])
OPTIONS = {"xatol": 0.05, "fatol": 1e-6, "maxiter": 400}


INTERIOR = np.array([50.0, 60.0, 0.3, 5.0, -5.0, 300.0])


def valley(point):
    """A curved valley whose minimum lies inside the bounds: reflections, expansions and both contractions, unclipped."""
    offset = (point - INTERIOR) / np.array([50.0, 20.0, 0.3, 2.0, 2.0, 50.0])
    return float(np.sum(100.0 * (offset[1:] - offset[:-1] ** 2) ** 2 + (1.0 - offset[:-1]) ** 2))


def beyond_the_bounds(point):
    """A bowl centred outside the box, so the polish runs along the bounds and every step is clipped."""
    return float(np.sum((point - np.array([400.0, -50.0, 3.0, 40.0, -40.0, 5.0])) ** 2))


def rough(point):
    """A slope under a fast ripple: contractions keep failing, so the simplex shrinks again and again."""
    return float(np.sum(np.abs(point - INTERIOR)) + 30.0 * np.sin(37.0 * np.sum(point)))


@pytest.mark.parametrize("function", [valley, beyond_the_bounds, rough])
@pytest.mark.parametrize("start", [
    [120.0, 35.0, 0.4, 3.0, -4.0, 150.0],
    [-280.0, 0.0, -1.2, 24.9, -24.9, 599.0],      # zero coordinate; first steps cross the upper bounds
    [0.5, 199.0, 0.0, 0.0, 0.0, 20.0],
])
def test_polish_takes_the_same_steps_as_scipy(function, start) -> None:
    minimize = pytest.importorskip("scipy.optimize").minimize
    start = np.array(start, dtype=np.float64)
    reference_points = []

    def recorded(point):
        reference_points.append(point.copy())
        return function(point)

    reference = minimize(recorded, start, method="Nelder-Mead", bounds=list(zip(LOWER, UPPER, strict=True)), options=OPTIONS)
    polish = _nelder_mead(start, LOWER, UPPER, **OPTIONS)
    points = []
    state = next(polish)
    while type(state) is np.ndarray:
        points.extend(state.copy())
        state = polish.send([function(point) for point in state])

    np.testing.assert_array_equal(np.array(points), np.array(reference_points))
    np.testing.assert_array_equal(state[0], reference.x)
    assert state[1] == reference.fun
