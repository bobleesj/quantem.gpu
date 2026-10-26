"""Checks of a backend's exact-derivative fit against the plain-torch float64 reference (``ssb.torch_agreement``).

Used by ``tests/hardware/{cuda,mps}/test_ssb_exact_derivatives.py`` on the weak-phase simulation of ``thin_object``.
The reference reads the session's own float32 G, k and band (``TorchAgreement.from_session``) and computes in float64
on the CPU, so a difference is the kernels' float32 arithmetic or a wrong derivative, never different inputs.
"""

import math

import numpy as np
import pytest

from tests.hardware.thin_object import TRUTH, session

BOUNDS = np.array([(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)] + [(-2, 2)] * 22, dtype=float)
LOW_SCALE = np.array([100.0, 100.0, 100.0, 5.0, 5.0, 100.0])


def reference(ssb, chunk=256):
    from quantem.gpu.ssb.torch_agreement import TorchAgreement

    return TorchAgreement.from_session(ssb, chunk=chunk)


def fifth_objective(ssb):
    from quantem.gpu.ssb.fifth_order_fit import _objective

    return _objective(ssb, gradients=True)


def low_order(ssb):
    backend = ssb._backend()
    if ssb.backend == "cuda":
        from quantem.gpu.ssb.cuda.exact_derivatives import LowOrderDerivatives

        return LowOrderDerivatives(backend._get_accelerator())
    from quantem.gpu.ssb.mps.exact_derivatives import LowOrderDerivatives

    backend.cache_rotation(math.radians(ssb.scan_detector_rotation_deg))
    return LowOrderDerivatives(backend._prepared)


def relative(test, truth):
    return float(np.abs(np.asarray(test) - np.asarray(truth)).max() / np.abs(truth).max())


def away_from_the_optimum():
    z = np.random.default_rng(23).uniform(-0.6, 0.6, 28)
    z[5] = 1.2
    return z


def check_fifth_order_gradient(ssb, stride):
    """Value and exact gradient of the kernels against float64 autograd, and the gradient against central
    differences of the production float32 objective."""
    objective = fifth_objective(ssb)
    z = away_from_the_optimum()
    values, gradient = objective.values_and_gradients(z, stride)
    truth_value, truth_gradient = reference(ssb).subsample(stride).value_and_gradient(z)
    assert values[0] == pytest.approx(truth_value, rel=1e-5)
    assert relative(gradient[0], truth_gradient) < 2e-3
    step = 2e-3
    rows = np.repeat(z[None], 56, axis=0)
    rows[0::2] += step * np.eye(28)
    rows[1::2] -= step * np.eye(28)
    shifted = objective.values(rows, stride)
    assert relative(gradient[0], (shifted[0::2] - shifted[1::2]) / (2 * step)) < 2e-2
    if stride == 1:
        assert values[0] == pytest.approx(float(objective.evaluate(z, 5)[0]), rel=1e-6)


def check_band_halves(ssb):
    """The two ring halves of the half-band check partition the band: their values and gradients add up to the full
    objective's, and unit weights change nothing."""
    from quantem.gpu.ssb.fifth_order_fit import ring_parity

    objective = fifth_objective(ssb)
    z = away_from_the_optimum()
    full_values, full_gradient = objective.values_and_gradients(z, 1)
    parity = ring_parity(*objective.band_frequencies())
    assert 0 < parity.sum() < parity.size
    halves = [objective.with_band_weights(parity == ring).values_and_gradients(z, 1) for ring in (0, 1)]
    assert halves[0][0][0] + halves[1][0][0] == pytest.approx(full_values[0], rel=1e-9)
    assert relative(halves[0][1][0] + halves[1][1][0], full_gradient[0]) < 1e-5
    unit_values, unit_gradient = objective.with_band_weights(np.ones(parity.size)).values_and_gradients(z, 1)
    np.testing.assert_array_equal(unit_values, full_values)
    np.testing.assert_array_equal(unit_gradient, full_gradient)


def check_low_order_derivatives(ssb):
    """Value, exact gradient and exact Hessian of the low-order kernel against float64 autograd."""
    from quantem.gpu.ssb.torch_agreement import fifth_from_low

    low = low_order(ssb)
    phi = math.radians(TRUTH["phi12_deg"] + 7) * 2
    z6 = np.array([10 * TRUTH["C10_nm"] + 8, 25 * math.cos(phi), 25 * math.sin(phi), 1.5, -2.0, 150.0])
    low.values(z6[None])
    value, gradient, hessian = low.derivatives([0])[0]
    truth = reference(ssb)
    z = fifth_from_low(z6).numpy()
    truth_value, truth_gradient = truth.value_and_gradient(z)
    truth_hessian = truth.hessian(z, range(6)) / np.outer(LOW_SCALE, LOW_SCALE)
    assert value == pytest.approx(truth_value, rel=1e-5)
    assert relative(gradient, truth_gradient[:6] / LOW_SCALE) < 2e-3
    assert relative(hessian, truth_hessian) < 2e-3
    np.testing.assert_allclose(hessian, hessian.T)


def check_opposite_branch(simulation, backend):
    """The 180-degree map (C10, C12 pair and tilt negated) keeps the joint agreement."""
    data, _, detector_mrad = simulation
    point = np.array([[-48.0, 21.0, math.radians(30.0), 1.2, -0.8, 140.0]])
    mapped = point.copy()
    mapped[0, 0] = -point[0, 0]
    mapped[0, 2] = point[0, 2] + math.pi / 2
    mapped[0, 3:5] = -point[0, 3:5]
    with session(data, detector_mrad, backend) as ssb:
        first = float(ssb._backend().sample_agreement(point, (0.2, 0.9))[0])
        ssb.set_rotation(180.0)
        second = float(ssb._backend().sample_agreement(mapped, (0.2, 0.9))[0])
    assert second == pytest.approx(first, rel=1e-5)


def float64_optimum(truth, start, bounds=BOUNDS):
    """The float64 maximum near ``start`` inside ``bounds`` (equal bounds hold a coordinate): the exact-gradient
    search on the reference's own values and gradients."""
    from quantem.gpu.ssb._gradient_refinement import Settings, refine

    norm = truth.agreement(start)

    def evaluate(points, kind, stride):
        values, gradients = [], []
        for z in np.atleast_2d(points):
            value, gradient = truth.value_and_gradient(z)
            values.append(value / norm)
            gradients.append(gradient / norm)
        return np.array(values), (None if kind == "value" else np.array(gradients))

    start = np.clip(start, bounds[:, 0], bounds[:, 1])
    return refine(evaluate, start, bounds, settings=Settings(hessian_bf_stride=1, coarse_bf_stride=1,
                                                              initial_radius=0.1))["coordinates"]


def check_fit(simulation, backend):
    """The default fit (exact low-order Newton, exact-gradient fifth order, mapped branch) reaches the float64 optimum
    of the reference: at fifth order, and over the six low-order coordinates with the higher orders held at 0."""
    data, _, detector_mrad = simulation
    with session(data, detector_mrad, backend) as ssb:
        exact = ssb.find_aberrations(show=False, verbose=False, check_rotation="both", higher_order_limit=2)
        truth = reference(ssb)
        fitted = np.asarray(exact.optimizer["coordinates"])
        assert truth.agreement(fitted) >= truth.agreement(float64_optimum(truth, fitted)) * (1 - 2e-7)
        # the low-order fit is the standard start of the fifth-order refinement
        low = np.asarray(exact.optimizer["starts"][0]["initial_coordinates"])
        assert exact.optimizer["starts"][0]["start"] == "standard" and not np.any(low[6:])
        low_bounds = BOUNDS.copy()
        low_bounds[6:] = 0.0
        low_optimum = float64_optimum(truth, low, low_bounds)
        assert truth.agreement(low) >= truth.agreement(low_optimum) * (1 - 2e-7)
    assert exact.refinement == "bfgs"
    # C10 changes sign on the opposite branch
    mirrored = 1.0 if exact.scan_detector_rotation_deg == 0.0 else -1.0
    assert abs(exact.aberrations["C10_nm"] - mirrored * TRUTH["C10_nm"]) < 0.5


def check_wide_fit(simulation, backend):
    """The default fit is wide (+-20) and checked on two ring halves of the band; on the weak-phase simulation the
    halves agree, so +-20 is shown, and each half is the masked refit from the same low-order start."""
    from quantem.gpu.ssb.fifth_order_fit import _refine_with_gradients, fifth_order_bounds, ring_parity

    data, _, detector_mrad = simulation
    with session(data, detector_mrad, backend) as ssb:
        fit = ssb.find_aberrations(show=False, verbose=False)
        check = fit.optimizer["higher_order_consistency"]
        assert check["trusted"] and check["shown_limit"] == 20.0 and fit.optimizer["higher_order_limit"] == 20.0
        assert check["half_band_rms_rad"] < 0.1
        assert [half["ring_parity"] for half in check["halves"]] == [0, 1]
        assert check["checked_coordinates"] == fit.optimizer["coordinates"]
        assert fit.timings["higher_order_consistency_s"] > 0 and "higher_order_fallback_s" not in fit.timings
        objective = fifth_objective(ssb)
        parity = ring_parity(*objective.band_frequencies())
        assert check["halves"][1]["frequencies"] == int(parity.sum())
        half = objective.with_band_weights(parity == 1)
        start = np.asarray(check["start_coordinates"])
        refit = _refine_with_gradients(half, start, fifth_order_bounds(20), float(half.values(start, 1)[0]), 400)
        np.testing.assert_allclose(refit["coordinates"], check["halves"][1]["coordinates"], rtol=0, atol=1e-9)
