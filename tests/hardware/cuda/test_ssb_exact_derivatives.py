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
def ssb(simulation):
    data, _, detector_mrad = simulation
    with session(data, detector_mrad, "cuda") as opened:
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
    checks.check_opposite_branch(simulation, "cuda")


def test_exact_fit_reaches_the_float64_optimum(simulation):
    checks.check_fit(simulation, "cuda")


def test_default_wide_fit_is_checked_on_two_band_halves(simulation):
    checks.check_wide_fit(simulation, "cuda")
