"""Exact-derivative SSB refinements: host controllers, the gradient's coordinate map and the 180-degree branch map.

CPU only: the trust-region controllers run on analytic objectives, the coordinate map against a float64 evaluation of
the shared chi polynomial, and the branch map on a stub session. The CUDA derivative kernels are checked against
central differences in ``tests/hardware/cuda/test_ssb_exact_derivatives.py``.
"""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu.ssb import _exact_newton, fifth_order_fit
from quantem.gpu.ssb._gradient_refinement import Settings, bounded_step, projected_gradient, refine
from quantem.gpu.ssb.fifth_order_parameters import Coordinates
from quantem.gpu.ssb.higher_order import ORDERS
from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.ssb.workflow import SSB

FIFTH_BOUNDS = np.array([(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)] + [(-2, 2)] * 22, dtype=float)


def _chi(row, col, wavelength, packed):
    """``higher_order.CHI_SOURCE`` in float64."""
    x, y = row * wavelength, col * wavelength
    r2 = x * x + y * y
    total = 0.0
    for i, (n, m) in enumerate(ORDERS):
        power = (x + 1j * y) ** m
        radial = r2 ** ((n + 1 - m) // 2)
        total += radial * (power.real * packed[2 * i] + power.imag * packed[2 * i + 1])
    return 2 * math.pi / wavelength * total


def _coordinates():
    coordinates = Coordinates()
    coordinates.p = SimpleNamespace(wavelength=0.019687489006849, semiangle_rad=0.03)
    return coordinates


def _packed(coordinates, z):
    """``Coordinates.encode``'s packed coefficients of one z, in float64 (without the float32 rounding)."""
    packed = np.zeros(28)
    packed[0], packed[2], packed[3] = z[0] * 50, z[1] * 50, z[2] * 50
    j = 6
    for i, (n, m) in enumerate(ORDERS):
        if n < 2:
            continue
        scale = coordinates.p.wavelength / (2 * np.pi * coordinates.p.semiangle_rad ** (n + 1))
        packed[2 * i] = z[j] * scale
        j += 1
        if m:
            packed[2 * i + 1] = z[j] * scale
            j += 1
    return packed


def test_packed_encoding_matches_the_shared_coordinates():
    coordinates = _coordinates()
    z = np.random.default_rng(3).uniform(-1, 1, 28)
    _, packed = coordinates.encode(z[None], 5)
    np.testing.assert_allclose(packed[0].ravel(), _packed(coordinates, z), rtol=1e-6, atol=1e-12)


def test_monomial_sums_map_to_the_phase_gradient_in_every_coordinate():
    """One pair's T_a = (k - q)^a - k^a maps to d phi1 / dz, phi1 = chi(k - q) - chi(k), for the optical z."""
    coordinates = _coordinates()
    rng = np.random.default_rng(11)
    wavelength = coordinates.p.wavelength
    for _ in range(3):
        k = rng.uniform(-1.4, 1.4, 2)
        q = rng.uniform(-0.9, 0.9, 2)
        raw = np.zeros(28)
        for d in range(2, 7):
            for c in range(d + 1):
                a = d - c
                raw[d * (d + 1) // 2 + c] = (k[0] - q[0]) ** a * (k[1] - q[1]) ** c - k[0] ** a * k[1] ** c
        gradient = coordinates.gradient(raw)[0]

        def phase(z):
            packed = _packed(coordinates, z)
            return _chi(k[0] - q[0], k[1] - q[1], wavelength, packed) - _chi(k[0], k[1], wavelength, packed)

        z = rng.uniform(-1, 1, 28)
        # phi1 is linear in z, so a unit difference is exact
        expected = np.array([phase(z + np.eye(28)[j]) - phase(z) for j in range(28)])
        optical = [0, 1, 2] + list(range(6, 28))
        np.testing.assert_allclose(gradient[optical], expected[optical], rtol=1e-9,
                                   atol=1e-9 * np.abs(expected).max())
        np.testing.assert_array_equal(gradient[3:6], 0.0)


def test_tilt_and_depth_sums_scale_to_their_coordinates():
    coordinates = _coordinates()
    raw = np.zeros(28)
    raw[:3] = [2.0, 3.0, 5.0]   # d/d depth spread (A), d/d tilt row and column (rad)
    gradient = coordinates.gradient(raw)[0]
    np.testing.assert_allclose(gradient[3:6], [3.0 * 5e-3, 5.0 * 5e-3, 2.0 * 100])


def _box_quadratic(seed=0, n=28):
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((n, n))
    curvature = basis @ basis.T / n + np.eye(n)
    target = rng.uniform(-3, 3, n)        # partly outside the +-2 box: those coordinates end on a bound
    target[5] = 1.0
    return curvature, target


def _quadratic_evaluator(curvature, target, calls):
    def evaluate(points, kind, stride):
        calls.append((kind, stride, len(np.atleast_2d(points))))
        points = np.atleast_2d(points)
        difference = points - target
        values = 1e4 - 0.5 * np.einsum("bi,ij,bj->b", difference, curvature, difference)
        gradients = -difference @ curvature
        # a BF subsample is a scaled copy that only steers
        scale = 1.0 if stride == 1 else 0.7
        return values * scale / 1e4, (None if kind == "value" else gradients * scale / 1e4)
    return evaluate


def test_gradient_refinement_reaches_the_box_constrained_maximum():
    curvature, target = _box_quadratic()
    calls = []
    start = np.zeros(28)
    start[5] = 1.0
    result = refine(_quadratic_evaluator(curvature, target, calls), start, FIFTH_BOUNDS)
    z = result["coordinates"]
    assert np.all(z >= FIFTH_BOUNDS[:, 0]) and np.all(z <= FIFTH_BOUNDS[:, 1])
    gradient = -(z - target) @ curvature
    assert np.abs(projected_gradient(gradient, z, FIFTH_BOUNDS)).max() < 1e-2
    assert result["stop"] == "converged"
    assert result["objectives"] < 60
    # the start Hessian and the coarse stage ran on BF subsamples
    assert {stride for _, stride, _ in calls} == {64, 8, 1}
    assert calls[0] == ("value_and_gradient", 64, 29)


def test_gradient_refinement_respects_the_objective_cap():
    """A quartic term keeps the quasi-Newton model inexact, so the search needs more full objectives than allowed."""
    curvature, target = _box_quadratic(1)

    def evaluate(points, kind, stride):
        difference = np.atleast_2d(points) - target
        values = 1e4 - 0.5 * np.einsum("bi,ij,bj->b", difference, curvature, difference) - np.sum(difference ** 4, 1)
        gradients = -difference @ curvature - 4 * difference ** 3
        return values / 1e4, (None if kind == "value" else gradients / 1e4)

    start = np.zeros(28)
    start[5] = 1.0
    unlimited = refine(evaluate, start, FIFTH_BOUNDS)
    assert unlimited["objectives"] > 3 and unlimited["stop"] == "converged"
    capped = refine(evaluate, start, FIFTH_BOUNDS, settings=Settings(max_objectives=3))
    assert capped["objectives"] == 3
    assert capped["stop"] == "evaluation cap"


def test_bounded_step_holds_coordinates_that_would_leave_the_box():
    curvature, _ = _box_quadratic(2, n=6)
    z = np.array([2.0, 0.0, -2.0, 0.5, 0.0, 1.0])
    lower, upper = -2 * np.ones(6), 2 * np.ones(6)
    gradient = np.array([1.0, 0.5, -1.0, 4.0, -0.2, 0.1])
    step = bounded_step(gradient, -curvature, 10.0, z, lower, upper, np.ones(6))
    assert step[0] == 0 and step[2] == 0
    assert np.all(z + step >= lower - 1e-12) and np.all(z + step <= upper + 1e-12)


class _LowOrderQuadratic:
    """A concave agreement in Newton coordinates with its exact derivatives, staged as the CUDA objective."""

    def __init__(self, maximum, scale):
        self.maximum, self.scale = np.asarray(maximum, float), np.asarray(scale, float)
        self.latest = None
        self.derivative_calls = 0

    def value(self, z):
        return 5e13 * (1 - np.sum(((np.asarray(z) - self.maximum) / self.scale) ** 2))

    def values(self, z_rows, all_derivatives):
        self.latest = np.atleast_2d(z_rows)
        return np.array([self.value(z) for z in self.latest])

    def derivatives(self, indices):
        self.derivative_calls += len(indices)
        out = []
        for index in indices:
            z = self.latest[index]
            gradient = -2 * 5e13 * (z - self.maximum) / self.scale ** 2
            out.append((self.value(z), gradient, np.diag(-2 * 5e13 / self.scale ** 2)))
        return out


def test_exact_newton_polish_converges_on_the_maximum_from_the_best_trials():
    maximum_point = np.array([60.0, 9.0, 0.4, 3.0, -2.0, 120.0])
    objective = _LowOrderQuadratic(_exact_newton.encode(maximum_point), [80, 40, 40, 6, 6, 200])
    rng = np.random.default_rng(7)
    lower = [-300.0, 0.0, -math.pi / 2, -25.0, -25.0, 20.0]
    upper = [300.0, 200.0, math.pi / 2, 25.0, 25.0, 600.0]
    points = rng.uniform(lower, upper, (200, 6))
    scores = np.array([objective.value(_exact_newton.encode(p)) for p in points])
    point, agreement, starts = _exact_newton.fit(objective, points, scores, lower, upper)
    np.testing.assert_allclose(_exact_newton.encode(point), objective.maximum,
                               atol=float(_exact_newton.EXACT_TOLERANCES.max()))
    assert agreement == pytest.approx(5e13, rel=1e-9)
    assert -math.pi / 2 <= point[2] < math.pi / 2 and point[1] >= 0
    assert starts and starts[0]["converged"]
    # one derivative evaluation per accepted center, not a stencil
    assert objective.derivative_calls <= sum(s["evaluations"] for s in starts)


def test_other_basins_keep_the_first_polish_unless_clearly_higher():
    objective = _LowOrderQuadratic(np.zeros(6) + [0, 5, 5, 0, 0, 100], [80, 40, 40, 6, 6, 200])
    first = (np.array([0.0, 7.0, math.pi / 8, 0.0, 0.0, 100.0]), 5e13, [])
    points = np.array([[0.0, 7.0, math.pi / 8, 0.0, 0.0, 100.0], [0.0, 7.0, -math.pi / 8, 10.0, 10.0, 300.0]])
    scores = np.array([objective.value(_exact_newton.encode(p)) for p in points])
    kept = _exact_newton.other_basins(objective, points, scores, [-300.0, 0, -math.pi / 2, -25, -25, 20],
                                      [300.0, 200, math.pi / 2, 25, 25, 600], first)
    assert kept[1] == first[1] and np.array_equal(kept[0], first[0])


def test_mapped_branch_is_the_exact_180_degree_image():
    first = SSBResult(
        object_wave=np.ones((4, 4), np.complex64), backend="cuda",
        aberrations={"C10_nm": 6.25, "C12_nm": 1.59, "phi12_deg": -84.9}, sample_tilt_mrad=(-0.07, -3.28),
        depth_spread_nm=14.4, scan_detector_rotation_deg=345.8, sample_agreement=3.5e14,
        optimizer={"search": "resident multivariate TPE"}, trial_records=[{"number": 0}],
    )
    agreement = Mock(return_value=np.array([3.5e14]))
    phase = Mock(return_value=(np.zeros((4, 4), np.float32), None))
    session = SimpleNamespace(
        _backend=lambda: SimpleNamespace(sample_agreement=agreement), _phase=phase,
        _unit_wave=lambda image: np.exp(1j * image), scan_detector_rotation_deg=165.8,
    )
    mapped = SSB._mapped_branch(session, first, compute_loss=False)
    assert mapped.aberrations == {"C10_nm": -6.25, "C12_nm": 1.59, "phi12_deg": pytest.approx(5.1)}
    assert mapped.sample_tilt_mrad == (0.07, 3.28)
    assert mapped.depth_spread_nm == 14.4
    assert mapped.scan_detector_rotation_deg == 165.8
    assert mapped.trial_records == [] and mapped.sample_agreement == 3.5e14
    assert mapped.optimizer["mapped_from_rotation_deg"] == 345.8
    row = agreement.call_args.args[0][0]
    np.testing.assert_allclose(row, [-62.5, 15.9, math.radians(5.1), 0.07, 3.28, 144.0])
    assert first.aberrations["C10_nm"] == 6.25 and first.sample_tilt_mrad == (-0.07, -3.28)


@pytest.mark.parametrize("backend, expected", [("cuda", "bfgs"), ("mps", "bfgs"), ("webgpu", "newton")])
def test_auto_refinement_uses_exact_gradients_where_the_backend_has_them(monkeypatch, backend, expected):
    import sys

    start = np.zeros(28)
    start[5] = 1.0
    seen = {}
    gradient_objective = SimpleNamespace(
        evaluate=lambda points, order: np.ones(len(np.atleast_2d(points))),
        values=lambda points, stride=1: np.ones(len(np.atleast_2d(points))),
        gradients=lambda points, stride=1: np.zeros((len(np.atleast_2d(points)), 28)),
        num_bright_field_pixels=9000,
    )

    def objective(ssb, gradients=False):
        seen["gradients"] = gradients
        return gradient_objective

    monkeypatch.setattr(fifth_order_fit, "_objective", objective)
    monkeypatch.setitem(sys.modules, "quantem.gpu.ssb.fifth_order_parameters", SimpleNamespace(
        vector=lambda fit, order: start.copy(), public_params=lambda *args: ({}, (0.0, 0.0), 10.0)))
    monkeypatch.setattr(fifth_order_fit, "newton", lambda tracker, initial, bounds: (0, "small bounded Newton step"))
    session = SimpleNamespace(
        backend=backend, _find_aberrations=Mock(return_value=SimpleNamespace(rotation_branches=None,
                                                                              trial_records=[])),
        run=lambda result, **kwargs: result, _accept_result=lambda result: result,
    )
    fit = fifth_order_fit.fit_fifth_order(
        session, search="standard", trials=200, seed=42, save_dir=None, force=False, verbose=False,
        primary_options={"check_rotation": False, "refine_rotation": False}, show=False)
    assert fit.refinement == expected
    assert seen["gradients"] is (expected == "bfgs")
    if expected == "bfgs":
        assert fit.optimizer["refinement_cap"] == 400
        assert fit.optimizer["exact_gradients"]["objectives"] >= 1
        assert fit.optimizer["stop"] == "converged"


def test_bfgs_on_a_backend_without_gradients_fails_before_computation():
    session = SimpleNamespace(backend="webgpu", _find_aberrations=Mock())
    with pytest.raises(ValueError, match="exact fifth-order gradients"):
        fifth_order_fit.fit_fifth_order(
            session, search="standard", trials=200, seed=42, save_dir=None, force=False, verbose=False,
            primary_options={"check_rotation": False, "refine_rotation": False}, refinement="bfgs", show=False)
    session._find_aberrations.assert_not_called()
