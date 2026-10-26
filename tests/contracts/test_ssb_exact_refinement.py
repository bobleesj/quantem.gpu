"""Exact-derivative SSB refinements: host controllers, the gradient's coordinate map and the 180-degree branch map.

CPU only: the trust-region controllers run on analytic objectives, the coordinate map against a float64 evaluation of
the shared chi polynomial, and the branch map on a stub session. The CUDA derivative kernels are checked against
central differences in ``tests/hardware/cuda/test_ssb_exact_derivatives.py``.
"""

import copy
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu.direct_ptycho import _exact_newton, fifth_order_fit, fifth_order_parameters
from quantem.gpu.direct_ptycho._gradient_refinement import Settings, bounded_step, projected_gradient, refine
from quantem.gpu.direct_ptycho.fifth_order_parameters import Coordinates, aperture_phase, aperture_rms, packed_coefficients
from quantem.gpu.direct_ptycho.higher_order import ORDERS
from quantem.gpu.direct_ptycho.results import SSBResult
from quantem.gpu.direct_ptycho.workflow import SSB

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


def test_aperture_phase_is_the_shared_chi_over_the_aperture():
    coordinates = _coordinates()
    wavelength, semiangle = coordinates.p.wavelength, coordinates.p.semiangle_rad
    z = np.random.default_rng(5).uniform(-1, 1, (2, 28))
    np.testing.assert_allclose(packed_coefficients(z, wavelength, semiangle).reshape(2, 28)[1], _packed(coordinates, z[1]),
                               rtol=1e-14)
    chi = aperture_phase(z, wavelength, semiangle, points=8)
    axis = ((np.arange(8) + 0.5) / 8 * 2 - 1) * semiangle
    angles = [(a, b) for a in axis for b in axis if a * a + b * b <= semiangle ** 2]
    expected = [_chi(a / wavelength, b / wavelength, wavelength, _packed(coordinates, z[0])) for a, b in angles]
    assert chi.shape == (2, 52)
    np.testing.assert_allclose(chi[0], expected, rtol=1e-12, atol=1e-12)
    # C10 = 1 nm: chi = pi C10 theta^2 / wl, a paraboloid whose rms over a disk is its edge value / sqrt(12)
    defocus = np.zeros(28)
    defocus[0] = 0.1
    edge = math.pi * 10 * semiangle ** 2 / wavelength
    chi = aperture_phase(defocus, wavelength, semiangle)
    assert chi.shape == (1, 7232)
    assert aperture_rms(chi, np.zeros_like(chi))[0] == pytest.approx(edge / math.sqrt(12), rel=2e-3)
    # a higher-order coordinate is its phase at the aperture edge: C30 = z (theta / alpha)^4
    spherical = np.zeros(28)
    spherical[10] = 1.0
    chi = aperture_phase(spherical, wavelength, semiangle, points=4)[0]
    unit = (np.arange(4) + 0.5) / 4 * 2 - 1
    radius2 = np.add.outer(unit ** 2, unit ** 2).ravel()
    np.testing.assert_allclose(np.sort(chi), np.sort(radius2[radius2 <= 1] ** 2), rtol=1e-12)
    assert aperture_rms(chi + 3.0, chi) == pytest.approx(0.0, abs=1e-12)


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


def test_gradient_refinement_holds_coordinates_with_equal_bounds():
    """The stream's refit: C10 and the C12 pair free, the other 25 coordinates held where they start."""
    curvature, target = _box_quadratic(2)
    start = np.zeros(28)
    start[5] = 1.0
    start[10] = 0.7
    bounds = np.column_stack([start, start])
    bounds[:3] = FIFTH_BOUNDS[:3]
    seen = []

    def evaluate(points, kind, stride):
        seen.append(np.atleast_2d(points).shape[1])
        return _quadratic_evaluator(curvature, target, [])(points, kind, stride)

    result = refine(evaluate, start, bounds)
    z = result["coordinates"]
    np.testing.assert_array_equal(z[3:], start[3:])
    assert set(seen) == {28}
    # the maximum over the free coordinates with the others held: grad_free = 0 inside the box
    gradient = -(z - target) @ curvature
    assert np.abs(projected_gradient(gradient[:3], z[:3], bounds[:3])).max() < 1e-2
    assert result["stop"] == "converged"
    with pytest.raises(ValueError, match="inside the higher-order bounds"):
        refine(evaluate, start, np.column_stack([start, start]))


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


@pytest.mark.parametrize("held, fixed", [
    ("C10", {0: 50.0}), ("C12 = 0", {1: 0.0}), ("C10, C12 and phi12", {0: 50.0, 1: 12.0, 2: 0.3}),
    ("tilt and depth spread (standard fit)", {3: 0.0, 4: 0.0, 5: 0.0}),
])
def test_exact_newton_holds_coordinates_with_equal_bounds(held, fixed):
    """Fixed aberrations and a standard fit's tilt and depth spread: the Newton climbs the other coordinates only."""
    maximum_point = np.array([60.0, 9.0, 0.4, 3.0, -2.0, 120.0])
    objective = _LowOrderQuadratic(_exact_newton.encode(maximum_point), [80, 40, 40, 6, 6, 200])
    lower = [-300.0, 0.0, -math.pi / 2, -25.0, -25.0, 0.0]
    upper = [300.0, 200.0, math.pi / 2, 25.0, 25.0, 600.0]
    for index, value in fixed.items():
        lower[index] = upper[index] = value
    if 1 in fixed and fixed[1] == 0.0:
        lower[1] = upper[1] = 0.0
    rng = np.random.default_rng(3)
    points = rng.uniform([-300.0, 0.0, -math.pi / 2, -25.0, -25.0, 20.0], [300.0, 200.0, math.pi / 2, 25.0, 25.0, 600.0],
                         (100, 6))
    for index, value in fixed.items():
        points[:, index] = value
    scores = np.array([objective.value(_exact_newton.encode(p)) for p in points])
    # a start that differs in the held coordinates is moved onto them
    start = np.array([0.0, 30.0, -0.5, 1.0, 1.0, 50.0])
    point, _, starts = _exact_newton.fit(objective, points, scores, lower, upper, start=start)
    z = _exact_newton.encode(point)
    held_z = _exact_newton.encode(np.where(np.array(lower) == np.array(upper), lower, point))
    for index in fixed:
        if index in (1, 2):
            np.testing.assert_allclose(z[1:3], held_z[1:3], atol=1e-12)
        else:
            assert point[index] == fixed[index]
    free = [index for index in range(6) if index not in fixed and not (index in (1, 2) and (1 in fixed))]
    np.testing.assert_allclose(z[free], objective.maximum[free], atol=float(_exact_newton.EXACT_TOLERANCES.max()))
    assert any(s["converged"] for s in starts)


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


def _gradient_session(monkeypatch, backend, start, seen=None):
    """A session stub whose fifth-order objective is constant, with exact (zero) gradients."""
    gradient_objective = SimpleNamespace(
        evaluate=lambda points, order: np.ones(len(np.atleast_2d(points))),
        values=lambda points, stride=1: np.ones(len(np.atleast_2d(points))),
        gradients=lambda points, stride=1: np.zeros((len(np.atleast_2d(points)), 28)),
        num_bright_field_pixels=9000,
    )

    def objective(ssb, gradients=False):
        if seen is not None:
            seen.append(gradients)
        return gradient_objective

    monkeypatch.setattr(fifth_order_fit, "_objective", objective)
    monkeypatch.setattr(fifth_order_parameters, "vector", lambda fit, order: start.copy())
    monkeypatch.setattr(fifth_order_parameters, "public_params", lambda *args: ({}, (0.0, 0.0), 10.0))
    return SimpleNamespace(
        backend=backend, _find_aberrations=Mock(return_value=SimpleNamespace(rotation_branches=None,
                                                                              trial_records=[])),
        run=lambda result, **kwargs: result, _accept_result=lambda result: result,
    )


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_the_fifth_order_fit_refines_on_exact_gradients(monkeypatch, backend):
    start = np.zeros(28)
    start[5] = 1.0
    seen = []
    session = _gradient_session(monkeypatch, backend, start, seen)
    fit = fifth_order_fit.fit_fifth_order(
        session, search="standard", trials=200, seed=42, save_dir=None, force=False, verbose=False,
        primary_options={"check_rotation": False, "refine_rotation": False}, show=False, higher_order_limit=2)
    assert fit.refinement == "bfgs" and seen == [True]
    assert fit.optimizer["refinement_cap"] == 400
    assert fit.optimizer["exact_gradients"]["objectives"] >= 1
    assert fit.optimizer["stop"] == "converged"


def test_a_backend_without_exact_gradients_has_no_fifth_order_fit():
    session = SimpleNamespace(backend="webgpu", _find_aberrations=Mock())
    with pytest.raises(NotImplementedError, match="CUDA or MPS"):
        fifth_order_fit.fit_fifth_order(
            session, search="standard", trials=200, seed=42, save_dir=None, force=False, verbose=False,
            primary_options={"check_rotation": False, "refine_rotation": False}, show=False)
    session._find_aberrations.assert_not_called()


@pytest.mark.parametrize("limit", [2, 20, 2.0, 20.0])
def test_higher_order_limit_sets_the_box_of_the_22_higher_orders(limit):
    bounds = fifth_order_fit.fifth_order_bounds(limit)
    np.testing.assert_array_equal(bounds[6:], np.tile([-float(limit), float(limit)], (22, 1)))
    np.testing.assert_array_equal(bounds[:6], [(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)])


@pytest.mark.parametrize("limit", [10, 0, -2, "wide", None, float("nan")])
def test_other_higher_order_limits_are_refused(limit):
    with pytest.raises(ValueError, match="2 \\(standard\\) or 20 \\(wide\\)"):
        fifth_order_fit.validate_higher_order_limit(limit)


def test_higher_order_limit_is_recorded_and_checked_before_computation(monkeypatch):
    import inspect

    start = np.zeros(28)
    start[5] = 1.0
    session = _gradient_session(monkeypatch, "cuda", start)
    seen = {}

    def refine(objective, initial, bounds, norm, cap, *, on_progress=None):
        seen["bounds"] = bounds
        return {"coordinates": initial, "agreement": 1.0, "objectives": 1, "gradients": 1, "subsampled": 0,
                "iterations": 1, "stop": "converged", "trace": [[1, 1.0, 0.0]], "hessian_bf_stride": 64,
                "coarse_bf_stride": 8}

    monkeypatch.setattr(fifth_order_fit, "_refine_with_gradients", refine)
    fit = fifth_order_fit.fit_fifth_order(
        session, search="standard", trials=200, seed=42, save_dir=None, force=False, verbose=False,
        primary_options={"check_rotation": False, "refine_rotation": False}, show=False, higher_order_limit=2)
    assert fit.optimizer["higher_order_limit"] == 2.0
    assert seen["bounds"][6:, 1].tolist() == [2.0] * 22
    assert fit.optimizer["bounds"][27] == [-2.0, 2.0]
    assert "higher_order_consistency" not in fit.optimizer
    with pytest.raises(ValueError, match="higher_order_limit"):
        fifth_order_fit.fit_fifth_order(
            SimpleNamespace(backend="cuda", _find_aberrations=Mock()), search="standard", trials=200, seed=42,
            save_dir=None, force=False, verbose=False, show=False,
            primary_options={"check_rotation": False, "refine_rotation": False}, higher_order_limit=5)
    call = inspect.unwrap(SSB.find_aberrations)
    with pytest.raises(ValueError, match="search='primary'"):
        call(SimpleNamespace(), search="primary", higher_order_limit=20)
    with pytest.raises(ValueError, match="search='primary'"):
        call(SimpleNamespace(), search="primary", higher_order_limit=2)
    with pytest.raises(ValueError, match="2 \\(standard\\)"):
        call(SimpleNamespace(), higher_order_limit=4)


def _broad_candidates():
    """A standard start and TPE proposals: the best, a copy within 0.25 rad, one differing only in tilt, a copy of the
    standard start, a non-finite score and a distinct worse one."""
    standard = np.zeros(28)
    standard[[0, 5]] = 0.5, 1.2
    best = standard.copy()
    best[6:] = 0.8
    near = best.copy()
    near[6] += 0.01
    tilted = best.copy()
    tilted[3] += 0.2
    worse = standard.copy()
    worse[10] = -1.5
    proposals = [worse, near, best, tilted, standard + 1e-4, standard]
    scores = [0.5, 1.9, 2.0, 1.8, 1.7, float("nan")]
    return standard, np.array(proposals), np.array(scores)


def test_broad_starts_put_the_standard_start_first_then_the_best_distinct_candidates():
    coordinates = _coordinates()
    standard, proposals, scores = _broad_candidates()
    starts = fifth_order_fit.broad_starts(standard, proposals, scores, coordinates.p.wavelength,
                                          coordinates.p.semiangle_rad)
    assert [name for name, _ in starts] == ["standard", "trial 2", "trial 3", "trial 0"]
    np.testing.assert_array_equal(starts[0][1], standard)
    np.testing.assert_array_equal(starts[2][1], proposals[3])
    limited = fifth_order_fit.broad_starts(standard, proposals, scores, coordinates.p.wavelength,
                                           coordinates.p.semiangle_rad, count=1)
    assert [name for name, _ in limited] == ["standard", "trial 2"]
    alone = fifth_order_fit.broad_starts(None, proposals, scores, coordinates.p.wavelength,
                                         coordinates.p.semiangle_rad)
    assert [name for name, _ in alone] == ["trial 2", "trial 3", "trial 4", "trial 0"]


def test_refinement_from_starts_keeps_the_first_unless_clearly_higher(monkeypatch):
    agreements = {0: 1.5, 1: 1.5 * (1 + 5e-7), 2: 1.6, 3: 1.55}

    def refine_one(objective, start, bounds, norm, cap, *, on_progress=None):
        key = int(start[0])
        if on_progress is not None:
            on_progress(1, agreements[key] - 0.1)
            on_progress(2, agreements[key])
        return {"agreement": agreements[key], "coordinates": start + 0.5, "objectives": 2 + key, "gradients": 1,
                "stop": "converged", "trace": [[1, agreements[key] - 0.1, 0.0], [2 + key, agreements[key], 0.0]]}

    monkeypatch.setattr(fifth_order_fit, "_refine_with_gradients", refine_one)
    starts = [(name, np.full(28, float(i))) for i, name in enumerate(["standard", "trial 7", "trial 3", "trial 9"])]
    seen = []
    kept, name, summaries, trace = fifth_order_fit._refine_from_starts(
        None, starts, FIFTH_BOUNDS, 1.0, 400, on_progress=lambda count, best: seen.append((count, best)))
    assert name == "trial 3" and kept["agreement"] == 1.6
    assert [s["agreement"] for s in summaries] == [agreements[i] for i in range(4)]
    assert [s["objectives"] for s in summaries] == [2, 3, 4, 5]
    trace = np.array(trace)
    assert trace[:, 0].tolist() == [1, 2, 3, 5, 6, 9, 10, 14]
    assert np.all(np.diff(trace[:, 1]) >= 0) and np.all(np.diff(trace[:, 2]) >= 0)
    assert seen[-1] == (11, 1.6)
    # a gain inside BASIN_MARGIN keeps the earlier start
    kept, name, _, _ = fifth_order_fit._refine_from_starts(None, starts[:2], FIFTH_BOUNDS, 1.0, 400)
    assert name == "standard"


class _TwoBumps:
    """Exact-gradient objective 1 + sum_i h_i exp(-|z - c_i|^2 / w) with the standard start at c_0."""

    def __init__(self, centers, heights, width=0.5):
        self.centers, self.heights, self.width = np.asarray(centers), np.asarray(heights), width
        self.p = SimpleNamespace(wavelength=0.019687489006849, semiangle_rad=0.03)
        self.num_bright_field_pixels = 9000

    def _terms(self, points):
        points = np.atleast_2d(points)
        offsets = points[:, None, :] - self.centers[None]
        return offsets, self.heights * np.exp(-np.sum(offsets ** 2, axis=2) / self.width)

    def evaluate(self, points, order):
        points = np.atleast_2d(points)
        return np.ones(len(points)) if points.shape[1] < 28 else self.values(points)

    def values(self, points, stride=1):
        return 1 + self._terms(points)[1].sum(axis=1)

    def gradients(self, points, stride=1):
        offsets, terms = self._terms(points)
        return -2 / self.width * np.sum(terms[:, :, None] * offsets, axis=1)


def _broad_session(monkeypatch, objective, standard):
    monkeypatch.setattr(fifth_order_fit, "_objective", lambda ssb, gradients=False: objective)
    monkeypatch.setattr(fifth_order_parameters, "vector", lambda fit, order: standard.copy())
    monkeypatch.setattr(fifth_order_parameters, "public_params", lambda *args: ({}, (0.0, 0.0), 10.0))
    return SimpleNamespace(
        backend="cuda", _find_aberrations=Mock(return_value=SimpleNamespace(rotation_branches=None,
                                                                             trial_records=[])),
        run=lambda result, **kwargs: result, _accept_result=lambda result: result,
    )


@pytest.mark.parametrize("tpe_height, kept", [(0.5, "standard"), (1.2, "trial 0")])
def test_broad_search_never_ends_below_its_standard_start(monkeypatch, tpe_height, kept):
    """The TPE study's first trial (zero optics, unit depth) sits on a second bump; the standard start on the first."""
    standard = np.zeros(28)
    standard[[0, 3, 5]] = 1.0, 2.0, 2.0
    tpe_peak = np.zeros(28)
    tpe_peak[5] = 1.0
    objective = _TwoBumps([standard, tpe_peak], [0.8, tpe_height])
    session = _broad_session(monkeypatch, objective, standard)
    options = dict(trials=12, save_dir=None, force=False, verbose=False, show=False, higher_order_limit=2,
                   primary_options={"check_rotation": False, "refine_rotation": False})
    broad = fifth_order_fit.fit_fifth_order(session, search="broad", seed=17, **options)
    alone = fifth_order_fit.fit_fifth_order(session, search="standard", seed=42, **options)
    starts = broad.optimizer["starts"]
    assert starts[0]["start"] == "standard" and 2 <= len(starts) <= 1 + fifth_order_fit.BROAD_STARTS
    assert starts[0]["agreement"] == alone.optimizer["exact_gradients"]["agreement"]
    assert broad.optimizer["kept_start"] == kept
    assert broad.optimizer["exact_gradients"]["agreement"] == max(s["agreement"] for s in starts)
    assert broad.optimizer["exact_gradients"]["agreement"] >= alone.optimizer["exact_gradients"]["agreement"]
    assert broad.optimizer["executed_objectives"] == sum(s["objectives"] for s in starts)
    kept_start = next(s for s in starts if s["start"] == kept)
    assert broad.optimizer["initial_coordinates"] == kept_start["initial_coordinates"]
    if kept == "standard":
        assert broad.optimizer["coordinates"] == alone.optimizer["coordinates"]
    assert alone.optimizer["kept_start"] == "standard" and len(alone.optimizer["starts"]) == 1


def test_ring_parity_splits_the_band_into_alternate_rings():
    q_row = np.array([0.21, 0.26, -0.33, 0.0, 0.47, 0.03], dtype=np.float32)
    q_col = np.array([0.0, 0.0, 0.0, -0.38, 0.0, 0.52], dtype=np.float32)
    assert fifth_order_fit.ring_parity(q_row, q_col).tolist() == [0, 1, 0, 1, 1, 0]
    assert fifth_order_fit.ring_parity(-q_row, -q_col).tolist() == [0, 1, 0, 1, 1, 0]


class _RingBumps:
    """Exact-gradient objective sum_q w_q (1 + exp(-|z - c(q)|^2 / width)) over six band frequencies whose optimum
    c(q) is ``even`` or ``odd`` by ring parity: the half-band fits find ``even`` and ``odd``."""

    def __init__(self, even, odd, width=40.0):
        self.q_row = np.array([0.21, 0.26, 0.33, 0.38, 0.47, 0.52], dtype=np.float32)
        self.q_col = np.zeros(6, dtype=np.float32)
        parity = fifth_order_fit.ring_parity(self.q_row, self.q_col)
        self.centers = np.where(parity[:, None] == 0, np.asarray(even)[None], np.asarray(odd)[None])
        self.weights = np.ones(6)
        self.width = width
        self.p = SimpleNamespace(wavelength=0.019687489006849, semiangle_rad=0.03)
        self.num_bright_field_pixels = 9000

    def band_frequencies(self):
        return self.q_row, self.q_col

    def with_band_weights(self, factor):
        weighted = copy.copy(self)
        weighted.weights = self.weights * np.asarray(factor, dtype=float)
        return weighted

    def _terms(self, points):
        offsets = np.atleast_2d(points)[:, None, :] - self.centers[None]
        return offsets, self.weights * np.exp(-np.sum(offsets ** 2, axis=2) / self.width)

    def evaluate(self, points, order):
        points = np.atleast_2d(points)
        return np.full(len(points), self.weights.sum()) if points.shape[1] < 28 else self.values(points)

    def values(self, points, stride=1):
        return self.weights.sum() + self._terms(points)[1].sum(axis=1)

    def gradients(self, points, stride=1):
        offsets, terms = self._terms(points)
        return -2 / self.width * np.sum(terms[:, :, None] * offsets, axis=1)


@pytest.mark.parametrize("difference, trusted", [(0.0, True), (5.0, False)])
def test_wide_fit_is_checked_on_two_band_halves(monkeypatch, capsys, difference, trusted):
    """Halves that agree keep the +-20 fit; halves whose C30 differs by 5 rad at the aperture edge (1.49 rad rms over
    the aperture) return the +-2 fit with the checked +-20 fit in the record."""
    standard = np.zeros(28)
    standard[[0, 3, 5]] = 1.0, 2.0, 2.0
    even = standard.copy()
    even[[6, 10]] = 1.0, 3.0
    odd = even.copy()
    odd[10] += difference
    session = _broad_session(monkeypatch, _RingBumps(even, odd), standard)
    fit = fifth_order_fit.fit_fifth_order(
        session, search="standard", trials=200, seed=42, save_dir=None, force=False, verbose=True, show=False,
        primary_options={"check_rotation": False, "refine_rotation": False})
    check = fit.optimizer["higher_order_consistency"]
    assert check["limit"] == 20.0 and check["ring_width_inv_A"] == 0.05 and check["max_half_band_rms_rad"] == 0.5
    assert [half["frequencies"] for half in check["halves"]] == [3, 3]
    np.testing.assert_allclose(check["halves"][0]["coordinates"], even, atol=1e-3)
    np.testing.assert_allclose(check["halves"][1]["coordinates"], odd, atol=1e-3)
    assert check["start_coordinates"] == standard.tolist()
    assert check["checked_stop"] == "converged" and check["checked_objectives"] >= 1
    rms_of_c30 = difference * math.sqrt(1 / 5 - 1 / 9)
    assert check["half_band_rms_rad"] == pytest.approx(rms_of_c30, rel=2e-3, abs=2e-3)
    assert check["trusted"] is trusted
    assert check["shown_limit"] == (20.0 if trusted else 2.0)
    assert fit.optimizer["higher_order_limit"] == check["shown_limit"]
    assert fit.optimizer["bounds"][27] == [-check["shown_limit"], check["shown_limit"]]
    assert "higher_order_consistency_s" in fit.timings
    printed = capsys.readouterr().out
    if trusted:
        assert fit.optimizer["coordinates"] == check["checked_coordinates"]
        assert "higher_order_fallback_s" not in fit.timings and "not trusted" not in printed
    else:
        assert max(abs(v) for v in fit.optimizer["coordinates"][6:]) <= 2.0
        assert check["checked_coordinates"][10] > 2.0
        assert fit.timings["higher_order_fallback_s"] > 0
        assert "Higher orders not trusted" in printed and "1.49 rad" in printed
