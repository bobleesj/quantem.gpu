"""MPS exact derivatives and fits against the float64 torch reference (``tests/hardware/exact_derivative_checks``).

The 64 x 64 weak-phase simulation of ``thin_object``.
"""

import numpy as np
import pytest

from tests.hardware import exact_derivative_checks as checks
from tests.hardware.thin_object import TRUTH, backend_or_skip, session, simulate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def simulation():
    if backend_or_skip() != "mps":
        pytest.skip("MPS only")
    return simulate(TRUTH)


@pytest.fixture(scope="module")
def ssb(simulation):
    data, _, detector_mrad = simulation
    with session(data, detector_mrad, "mps") as opened:
        yield opened


@pytest.mark.parametrize("stride", [1, 4])
def test_fifth_order_gradient_matches_the_reference(ssb, stride):
    checks.check_fifth_order_gradient(ssb, stride)


def test_gradient_pass_needs_its_value_pass(ssb):
    objective = checks.fifth_objective(ssb)
    z = np.zeros(28)
    z[5] = 1.0
    objective.values(z, 1)
    with pytest.raises(RuntimeError, match="same points"):
        objective.gradients(z, 4)


def test_ring_halves_partition_the_band(ssb):
    checks.check_band_halves(ssb)


def test_low_order_gradient_and_hessian_match_the_reference(ssb):
    checks.check_low_order_derivatives(ssb)


def test_opposite_branch_point_has_the_same_agreement(simulation):
    checks.check_opposite_branch(simulation, "mps")


def test_exact_fit_reaches_the_float64_optimum(simulation):
    checks.check_fit(simulation, "mps")


def test_default_wide_fit_is_checked_on_two_band_halves(simulation):
    checks.check_wide_fit(simulation, "mps")
