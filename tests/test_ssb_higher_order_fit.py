"""Host side of find_aberrations(higher_order=...): term names, coordinates and the Newton refinement."""

import math

import numpy as np
import pytest

from quantem.gpu.ssb.higher_order_fit import (
    COHERENCE_MIN,
    assess,
    edge_phase,
    from_coordinates,
    kernel_rows,
    refine_higher_order,
    standard_errors,
    to_coordinates,
    validate_terms,
)


def test_terms_are_ordered_and_checked():
    assert validate_terms(None) == ()
    assert validate_terms("C30") == ("C30",)
    assert validate_terms(["C30", "C21", "C23"]) == ("C21", "C23", "C30")
    for bad in (["C32"], ["C21", "C21"], ["C21_nm"]):
        with pytest.raises(ValueError, match="'C21' \\(coma\\)"):
            validate_terms(bad)


def test_coordinates_round_trip_through_the_azimuth_period():
    terms = ("C21", "C23", "C30")
    higher = {"C21": (400.0, math.radians(170.0)), "C23": (250.0, math.radians(-50.0)), "C30": (-3.0e4, 0.0)}
    z = to_coordinates(-50.0, 20.0, math.radians(30.0), (3.0, -4.0), 150.0, higher, terms)
    assert z.shape == (11,)
    back = from_coordinates(z, terms)
    assert back["C10"] == pytest.approx(-50.0)
    assert back["C12"] == pytest.approx(20.0) and math.degrees(back["phi12"]) == pytest.approx(30.0)
    assert (back["tilt_row_mrad"], back["tilt_col_mrad"], back["depth_spread_A"]) == pytest.approx((3.0, -4.0, 150.0))
    assert back["C21"] == pytest.approx(400.0) and math.degrees(back["phi21"]) == pytest.approx(170.0)
    # trefoil's azimuth has a period of 120 degrees: -50 is its own representative in (-60, 60]
    assert back["C23"] == pytest.approx(250.0) and math.degrees(back["phi23"]) == pytest.approx(-50.0)
    assert back["C30"] == pytest.approx(-3.0e4)          # a round term keeps its sign


def test_kernel_rows_fill_the_krivanek_slots():
    # C30 alone: slots C10, C12, C21, C23, C30 (the polynomial up to the highest fitted slot), coma and trefoil zero
    z = to_coordinates(-50.0, 20.0, 0.0, (1.0, 2.0), 30.0, {"C30": (1.0e4, 0.0)}, ("C30",))
    rows = kernel_rows(z, ("C30",))
    assert rows.shape == (1, 3 + 2 * 5)
    np.testing.assert_allclose(rows[0, :3], (30.0, 1.0, 2.0))
    np.testing.assert_allclose(rows[0, 3:], (-50.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e4, 0.0))


@pytest.mark.parametrize("tilt", [True, False])
def test_newton_reaches_a_quadratic_maximum(tilt):
    """A smooth agreement with a known maximum: the polish reaches it, and tilt=False holds the sample."""
    terms = ("C21", "C23", "C30")
    target = to_coordinates(-40.0, 12.0, 0.4, (2.0, -3.0), 120.0,
                            {"C21": (300.0, 1.0), "C23": (200.0, -0.3), "C30": (2.0e4, 0.0)}, terms)
    widths = np.array([5.0, 3.0, 3.0, 1.0, 1.0, 15.0, 300.0, 300.0, 120.0, 120.0, 1.0e4])
    rng = np.random.default_rng(4)
    mixing = np.eye(11) + 0.2 * rng.standard_normal((11, 11)) * (rng.random((11, 11)) < 0.2)

    def agreement(rows):
        # the rows' coordinates: z back from the kernel layout (C10, C12 a/b, tilt, depth, C21 a/b, C23 a/b, C30)
        z = np.column_stack([rows[:, 3], rows[:, 5], rows[:, 6], rows[:, 1], rows[:, 2], rows[:, 0],
                             rows[:, 7], rows[:, 8], rows[:, 9], rows[:, 10], rows[:, 11]])
        u = ((z - target) / widths) @ mixing.T
        return 1e12 * np.exp(-0.5 * np.sum(u * u, axis=1))

    start = target.copy()
    start[6:] = 0.0
    start[0] += 8.0
    if not tilt:
        start[3:6] = (0.5, 0.5, 100.0)
    fit = refine_higher_order(agreement, start, terms, tilt=tilt)
    assert fit["converged"]
    assert fit["agreement"] > fit["start_agreement"]
    if tilt:
        np.testing.assert_allclose((fit["z"] - target) / widths, 0.0, atol=0.02)
    else:
        np.testing.assert_array_equal(fit["z"][3:6], start[3:6])
        assert fit["agreement"] < 1e12


def _gaussian_maximum():
    """An agreement A0 exp(-|u|^2 / 2), u = M (z - target) / widths, whose curvature at the maximum is known."""
    terms = ("C21", "C23", "C30")
    target = to_coordinates(-40.0, 12.0, 0.4, (2.0, -3.0), 120.0,
                            {"C21": (300.0, 1.0), "C23": (200.0, -0.3), "C30": (2.0e4, 0.0)}, terms)
    # widths several times the polish's finite-difference steps, as on simulated crystals (1 % of the agreement is
    # lost 23-38 nm off in coma), so its second differences measure the curvature
    widths = np.array([20.0, 10.0, 10.0, 2.0, 2.0, 50.0, 1500.0, 1500.0, 700.0, 700.0, 4.0e4])
    rng = np.random.default_rng(4)
    mixing = np.eye(11) + 0.2 * rng.standard_normal((11, 11)) * (rng.random((11, 11)) < 0.2)
    peak = 1e12

    def agreement(rows):
        z = np.column_stack([rows[:, 3], rows[:, 5], rows[:, 6], rows[:, 1], rows[:, 2], rows[:, 0],
                             rows[:, 7], rows[:, 8], rows[:, 9], rows[:, 10], rows[:, 11]])
        u = ((z - target) / widths) @ mixing.T
        return peak * np.exp(-0.5 * np.sum(u * u, axis=1))

    precision = (mixing / widths[None]).T @ (mixing / widths[None])        # -Hessian / peak at the maximum
    start = target + 0.3 * widths * rng.standard_normal(11)
    return terms, agreement, start, peak, precision


def test_standard_errors_follow_the_curvature_at_the_maximum():
    """cov = 2 sigma^2 (-Hessian of A)^-1: compare with the exact curvature of a Gaussian maximum."""
    terms, agreement, start, peak, precision = _gaussian_maximum()
    fit = refine_higher_order(agreement, start, terms, tilt=True)
    variance = 50.0
    covariance = 2.0 * variance * np.linalg.inv(peak * precision)
    errors = standard_errors(fit, terms, variance)
    expected = {"C21": math.sqrt(covariance[6, 6] + covariance[7, 7]),
                "C23": math.sqrt(covariance[8, 8] + covariance[9, 9]), "C30": math.sqrt(covariance[10, 10])}
    for name in terms:
        assert errors[name] == pytest.approx(expected[name], rel=0.05), name
    assert standard_errors(fit, terms, 0.0) == {name: 0.0 for name in terms}


def test_guard_needs_coherence_and_precision():
    terms, agreement, start, _, _ = _gaussian_maximum()
    fit = refine_higher_order(agreement, start, terms, tilt=True)
    wavelength, semiangle = 0.019687, 0.03
    coherent = {"power": fit["agreement"] / 0.97, "band_weight": 1.0, "noise_variance": 0.0, "num_bf": 1000}
    good = assess(fit, terms, coherent, counted=False, wavelength_A=wavelength, semiangle_rad=semiangle)
    assert good["trusted"] and good["coherence"] == pytest.approx(0.97) and good["noise"] == "not estimated"
    dynamical = {**coherent, "power": fit["agreement"] / 0.6}
    bad = assess(fit, terms, dynamical, counted=False, wavelength_A=wavelength, semiangle_rad=semiangle)
    assert not bad["trusted"] and str(COHERENCE_MIN) in bad["reasons"][0]
    # counted electrons: the Poisson floors come off both sums, and enough noise makes the terms imprecise
    floor = 1e9
    noisy = {"power": fit["agreement"] / 0.97 + floor * 1000, "band_weight": 1.0, "noise_variance": floor, "num_bf": 1000}
    counted = assess({**fit, "agreement": fit["agreement"] + floor}, terms, noisy, counted=True,
                     wavelength_A=wavelength, semiangle_rad=semiangle)
    assert counted["coherence"] == pytest.approx(0.97)
    assert not counted["trusted"] and "aperture edge" in counted["reasons"][0]
    assert counted["edge_phase_error_rad"]["C21"] == pytest.approx(
        edge_phase("C21", counted["standard_error_nm"]["C21"] * 10.0, wavelength, semiangle))
