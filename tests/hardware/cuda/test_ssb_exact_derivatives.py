"""Exact derivatives of the CUDA SSB agreement against central differences, and the fits that use them.

The data are the weak-phase simulation of ``thin_object`` tiled to a 128 x 128 scan. Points away from the optimum keep
the gradients far above the float32 rounding of the objective, so central differences of the production objective
must reproduce them; the Hessian is compared with central differences of the exact gradients.
"""

import math

import numpy as np
import pytest

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
def engine(simulation):
    data, _, detector_mrad = simulation
    with session(data, detector_mrad, "cuda") as ssb:
        yield ssb._backend()._get_accelerator()


def _relative_error(exact, reference):
    return float(np.abs(exact - reference).max() / np.abs(reference).max())


@pytest.mark.parametrize("stride", [1, 4])
def test_fifth_order_gradient_matches_central_differences(engine, stride):
    from quantem.gpu.ssb.cuda.exact_derivatives import FifthOrderGradients

    objective = FifthOrderGradients(engine)
    z = np.random.default_rng(23).uniform(-0.6, 0.6, 28)
    z[5] = 1.2
    values, gradient = objective.values_and_gradients(z, stride)
    step = 2e-3
    rows = np.repeat(z[None], 56, axis=0)
    rows[0::2] += step * np.eye(28)
    rows[1::2] -= step * np.eye(28)
    shifted = objective.values(rows, stride)
    central = (shifted[0::2] - shifted[1::2]) / (2 * step)
    assert _relative_error(gradient[0], central) < 2e-2
    assert values[0] > 0
    # the value pass is the production objective's kernels, summed in float64
    if stride == 1:
        assert values[0] == pytest.approx(float(objective.evaluate(z, 5)[0]), rel=1e-6)


def test_gradient_pass_needs_its_value_pass(engine):
    from quantem.gpu.ssb.cuda.exact_derivatives import FifthOrderGradients

    objective = FifthOrderGradients(engine)
    z = np.zeros(28)
    z[5] = 1.0
    objective.values(z, 1)
    with pytest.raises(RuntimeError, match="same points"):
        objective.gradients(z, 4)


def test_low_order_gradient_and_hessian_match_central_differences(engine):
    from quantem.gpu.ssb.cuda.exact_derivatives import LowOrderDerivatives

    low = LowOrderDerivatives(engine)
    phi = math.radians(TRUTH["phi12_deg"] + 7) * 2
    z = np.array([10 * TRUTH["C10_nm"] + 8, 25 * math.cos(phi), 25 * math.sin(phi), 1.5, -2.0, 150.0])
    low.values(z[None])
    value, gradient, hessian = low.derivatives([0])[0]
    steps = np.array([0.5, 0.5, 0.5, 0.05, 0.05, 1.0])
    rows = np.repeat(z[None], 12, axis=0)
    rows[0::2] += np.diag(steps)
    rows[1::2] -= np.diag(steps)
    shifted = low.values(rows)
    central = (shifted[0::2] - shifted[1::2]) / (2 * steps)
    assert _relative_error(gradient, central) < 1e-2
    gradients = np.array([row[1] for row in low.derivatives(list(range(12)))])
    differences = (gradients[0::2] - gradients[1::2]) / (2 * steps[:, None])
    assert _relative_error(hessian, 0.5 * (differences + differences.T)) < 1e-2
    np.testing.assert_allclose(hessian, hessian.T)
    # the values are ThickSample's resident objective
    trial = np.array([[z[0], math.hypot(z[1], z[2]), 0.5 * math.atan2(z[2], z[1]), z[3], z[4], z[5]]])
    assert value == pytest.approx(float(engine.thick.fit_batch(trial)[0]), rel=1e-5)


def test_opposite_branch_point_has_the_same_agreement(simulation):
    """The 180-degree map (even chi terms and tilt negated) keeps the joint agreement."""
    data, _, detector_mrad = simulation
    point = np.array([[-48.0, 21.0, math.radians(30.0), 1.2, -0.8, 140.0]])
    mapped = point.copy()
    mapped[0, 0] = -point[0, 0]
    mapped[0, 2] = point[0, 2] + math.pi / 2
    mapped[0, 3:5] = -point[0, 3:5]
    with session(data, detector_mrad, "cuda") as ssb:
        first = float(ssb._backend().sample_agreement(point, (0.2, 0.9))[0])
        ssb.set_rotation(180.0)
        second = float(ssb._backend().sample_agreement(mapped, (0.2, 0.9))[0])
    assert second == pytest.approx(first, rel=1e-5)


def test_exact_fit_reaches_the_finite_difference_agreement(simulation):
    """The default fit (exact Newton, exact-gradient fifth order, mapped branch) against the earlier algorithms."""
    import quantem.gpu.ssb._exact_newton as exact_newton
    import quantem.gpu.ssb.workflow as workflow

    data, _, detector_mrad = simulation
    with session(data, detector_mrad, "cuda") as ssb:
        exact = ssb.find_aberrations(show=False, verbose=False, check_rotation="both")
    previous = exact_newton.EXACT_NEWTON, workflow.MAP_OPPOSITE_BRANCH
    exact_newton.EXACT_NEWTON, workflow.MAP_OPPOSITE_BRANCH = False, False
    try:
        with session(data, detector_mrad, "cuda") as ssb:
            reference = ssb.find_aberrations(show=False, verbose=False, check_rotation="both", refinement="newton")
    finally:
        exact_newton.EXACT_NEWTON, workflow.MAP_OPPOSITE_BRANCH = previous
    assert exact.refinement == "bfgs" and reference.refinement == "newton"
    assert exact.scan_detector_rotation_deg == reference.scan_detector_rotation_deg
    assert exact.rotation_branches.kept == reference.rotation_branches.kept
    assert exact.sample_agreement >= reference.sample_agreement * (1 - 1e-6)
    assert exact.refine_nfev < reference.refine_nfev
    # C10 changes sign on the opposite branch
    mirrored = 1.0 if exact.scan_detector_rotation_deg == 0.0 else -1.0
    assert abs(exact.aberrations["C10_nm"] - mirrored * TRUTH["C10_nm"]) < 0.5
