"""Scan-frame and detector-frame tilts, and rotations known modulo 180 degrees, follow one convention."""

import math

import numpy as np
import pytest

from quantem.gpu.geometry import (
    mean_rotation_deg,
    opposite_rotation_deg,
    propagator_to_sample_tilt,
    rotation_gap_deg,
    sample_to_propagator_tilt,
)


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("rotation_deg", [0.0, 12.3, 158.9, 270.0, 338.9])
def test_tilt_round_trip_and_the_centre_of_mass_convention(rotation_deg, flipped):
    """The detector-frame tilt is the scan-frame vector the DPC alignment would rotate back onto the scan axes."""
    from quantem.gpu.dpc.workflow import _rotate_vector

    sample = (-10.3, 4.7)
    propagator = sample_to_propagator_tilt(
        sample, scan_detector_rotation_deg=rotation_deg, scan_detector_rotation_flipped=flipped)
    back = propagator_to_sample_tilt(
        propagator, scan_detector_rotation_deg=rotation_deg, scan_detector_rotation_flipped=flipped)
    np.testing.assert_allclose(back, sample, rtol=0, atol=1e-12)
    assert math.hypot(*propagator) == pytest.approx(math.hypot(*sample), rel=1e-12)
    # dpc.run / SSB: scan = R(theta) . D . detector, D swapping the detector axes when flipped
    source = np.array(propagator[::-1] if flipped else propagator, dtype=np.float64)
    aligned = _rotate_vector(source[:1], source[1:], math.radians(rotation_deg))
    np.testing.assert_allclose([aligned[0][0], aligned[1][0]], sample, rtol=0, atol=1e-5)


def test_unflipped_tilt_matches_the_published_formula():
    """(t_r cos + t_c sin, -t_r sin + t_c cos): the expression ShowPtycho and myptycho used, now in one place."""
    theta = math.radians(-8.6)
    row, col = -10.3, 4.7
    expected = (row * math.cos(theta) + col * math.sin(theta), -row * math.sin(theta) + col * math.cos(theta))
    assert sample_to_propagator_tilt((row, col), scan_detector_rotation_deg=-8.6) == pytest.approx(expected, abs=1e-12)
    flipped = sample_to_propagator_tilt((row, col), scan_detector_rotation_deg=-8.6, scan_detector_rotation_flipped=True)
    assert flipped == pytest.approx(expected[::-1], abs=1e-12)
    with pytest.raises(ValueError, match="sample_tilt_mrad"):
        sample_to_propagator_tilt(1.0, scan_detector_rotation_deg=0.0)


def test_opposite_branch_is_exact_and_involutive():
    for angle in (0.0, 0.1, 158.88, 179.99999999999997, 180.0, 359.9):
        other = opposite_rotation_deg(angle)
        assert 0.0 <= other < 360.0
        assert opposite_rotation_deg(other) == pytest.approx(angle, abs=1e-12)
        if angle >= 180.0:
            assert opposite_rotation_deg(other) == angle
        assert rotation_gap_deg(angle, other) == pytest.approx(180.0, abs=1e-12)
        assert rotation_gap_deg(angle, other, period_deg=180.0) == pytest.approx(0.0, abs=1e-12)


def test_mean_and_gap_respect_the_period():
    assert rotation_gap_deg(179.0, 1.0, period_deg=180.0) == pytest.approx(2.0)
    assert rotation_gap_deg(179.0, 1.0) == pytest.approx(178.0)
    assert mean_rotation_deg([179.0, 1.0, 0.5], period_deg=180.0) == pytest.approx(1 / 6, abs=1e-3)
    assert mean_rotation_deg([359.0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    assert mean_rotation_deg([158.0, 160.0]) == pytest.approx(159.0)
    assert mean_rotation_deg([-20.0]) == pytest.approx(340.0)
    with pytest.raises(ValueError, match="at least one"):
        mean_rotation_deg([])
    with pytest.raises(ValueError, match="no mean direction"):
        mean_rotation_deg([0.0, 90.0], period_deg=180.0)
