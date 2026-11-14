"""CUDA exact derivatives and fits against the float64 torch reference (``tests/hardware/exact_derivative_checks``).

The weak-phase simulation of ``thin_object`` is tiled to a 128 x 128 scan, which CUDA reconstructs without padding.
"""

import numpy as np
import pytest

from tests.hardware import exact_derivative_checks as checks
from tests.hardware.thin_object import TRUTH, session, simulate

pytestmark = pytest.mark.slow

cp = pytest.importorskip("cupy")


@pytest.fixture(scope="module")
def simulation():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    return simulate(TRUTH, tile=2)


@pytest.fixture(scope="module")
def direct(simulation):
    data, _, detector_mrad = simulation
    with session(data, detector_mrad, "cuda") as opened:
        yield opened


@pytest.mark.parametrize("stride", [1, 4])
def test_fifth_order_gradient_matches_the_reference(direct, stride):
    checks.check_fifth_order_gradient(direct, stride)


def test_gradient_pass_needs_its_value_pass(direct):
    objective = checks.fifth_objective(direct)
    z = np.zeros(28)
    z[5] = 1.0
    objective.values(z, 1)
    with pytest.raises(RuntimeError, match="same points"):
        objective.gradients(z, 4)


def test_ring_halves_partition_the_band(direct):
    checks.check_band_halves(direct)


def test_low_order_gradient_and_hessian_match_the_reference(direct):
    checks.check_low_order_derivatives(direct)


def test_low_order_derivatives_of_each_position_do_not_depend_on_its_launch_group(direct, monkeypatch):
    """Positions evaluated one launch at a time (a small device's partial-sum limit) give the bits they give together."""
    import math

    from quantem.gpu.direct_ptycho.cuda import exact_derivatives

    low = checks.low_order(direct)
    phi = math.radians(checks.TRUTH["phi12_deg"] + 7) * 2
    base = np.array([10 * checks.TRUTH["C10_nm"] + 8, 25 * math.cos(phi), 25 * math.sin(phi), 1.5, -2.0, 150.0])
    points = base + np.linspace(-1, 1, 5)[:, None] * np.array([4.0, 3.0, -2.0, 0.5, 0.3, 20.0])
    low.values(points)
    together = low.derivatives(range(5))
    monkeypatch.setattr(exact_derivatives, "DERIVATIVE_PARTIAL_BYTES", 1)
    one_by_one = low.derivatives(range(5))
    for (value, gradient, hessian), (value_one, gradient_one, hessian_one) in zip(together, one_by_one, strict=True):
        assert value == value_one
        np.testing.assert_array_equal(gradient_one, gradient)
        np.testing.assert_array_equal(hessian_one, hessian)


def test_opposite_branch_point_has_the_same_agreement(simulation):
    checks.check_opposite_branch(simulation, "cuda")


def test_exact_fit_reaches_the_float64_optimum(simulation):
    checks.check_fit(simulation, "cuda")


def test_default_wide_fit_is_checked_on_two_band_halves(simulation):
    checks.check_wide_fit(simulation, "cuda")
