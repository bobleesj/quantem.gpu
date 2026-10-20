"""The bright-field disk fit takes an acquisition or its mean pattern, and refuses a pattern without a disk."""

import numpy as np
import pytest

from quantem.gpu import detector


def test_fit_probe_takes_an_acquisition_or_its_mean_pattern():
    rows, cols = np.indices((24, 20))
    disk = np.hypot(rows - 11.5, cols - 8.0) <= 5
    counts = (np.random.default_rng(4).poisson(2, (3, 4, 24, 20)) + 40 * disk).astype(np.uint16)
    assert detector.fit_probe(counts) == detector.fit_probe(detector.mean(counts))


def test_fit_probe_refuses_a_pattern_without_a_disk():
    """A flat pattern has no disk; returning the detector midpoint placed every virtual detector there."""
    with pytest.raises(ValueError, match="No bright-field disk"):
        detector.fit_probe(np.full((16, 16), 3.0, np.float32))


def test_detector_sampling_is_the_semiangle_over_the_disk_edge_radius():
    """The aperture edge sits at the convergence semiangle: sampling = semiangle / half-plateau radius, both axes."""
    import math

    rows, cols = np.mgrid[:64, :64]
    pattern = np.where((rows - 31.5) ** 2 + (cols - 30.0) ** 2 < 12.0 ** 2, 100.0, 1.0)
    radius = detector.disk_edge_radius(pattern)
    assert radius == pytest.approx(math.sqrt(np.count_nonzero(pattern > 50.0) / math.pi), rel=1e-12)
    assert radius == pytest.approx(12.0, rel=0.02)
    assert detector.detector_sampling_mrad(pattern, 30) == (30 / radius, 30 / radius)
    for semiangle_mrad in (0, -30, float("nan")):
        with pytest.raises(ValueError, match="semiangle_mrad"):
            detector.detector_sampling_mrad(pattern, semiangle_mrad)
    with pytest.raises(ValueError, match="no bright-field disk"):
        detector.detector_sampling_mrad(np.ones((8, 8)), 30)


def _blurred_disk(shape=(64, 64), center=(31.25, 29.5), radius=14.0, edge_px=1.5):
    """A mean pattern whose disk edge is blurred symmetrically: the half-plateau contour stays at ``radius``."""
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    distance = np.hypot(rows - center[0], cols - center[1])
    return 2.0 + 300.0 / (1.0 + np.exp((distance - radius) / edge_px))


def test_bright_field_disk_fit_finds_the_centre_and_the_edge_of_a_blurred_disk():
    """The edge sits at the half-plateau radius even when blurred, so the semiangle maps to it."""
    pattern = _blurred_disk()
    fit = detector.fit_bright_field_disk(pattern, semiangle_mrad=30)
    assert fit.bright_field_center_px == pytest.approx((31.25, 29.5), abs=0.05)
    assert fit.bright_field_edge_radius_px == pytest.approx(14.0, rel=0.02)
    assert fit.bright_field_edge_radius_px == detector.disk_edge_radius(pattern)
    assert fit.detector_sampling_mrad == (30 / fit.bright_field_edge_radius_px,) * 2
    assert fit.detector_sampling_mrad == detector.detector_sampling_mrad(pattern, 30)
    assert fit.method == "half-plateau disk edge"
    # SSB's centre rule: the float64 centroid of the pixels above mean + std
    rows, cols = np.nonzero(pattern > pattern.mean() + pattern.std())
    assert fit.bright_field_center_px == (float(rows.mean()), float(cols.mean()))


def test_bright_field_disk_fit_takes_an_acquisition_and_needs_no_semiangle():
    rows, cols = np.indices((24, 20))
    disk = np.hypot(rows - 11.5, cols - 8.0) <= 5
    counts = (np.random.default_rng(4).poisson(2, (3, 4, 24, 20)) + 40 * disk).astype(np.uint16)
    fit = detector.fit_bright_field_disk(counts)
    same = detector.fit_bright_field_disk(detector.mean(counts))
    assert (fit.bright_field_center_px, fit.bright_field_edge_radius_px) == (same.bright_field_center_px, same.bright_field_edge_radius_px)
    assert fit.detector_sampling_mrad is None and fit.semiangle_mrad is None
    for semiangle_mrad in (0, -30, float("nan")):
        with pytest.raises(ValueError, match="semiangle_mrad"):
            detector.fit_bright_field_disk(counts, semiangle_mrad)
    with pytest.raises(ValueError, match="no bright-field disk"):
        detector.fit_bright_field_disk(np.ones((8, 8)))


def test_bright_field_disk_fit_shows_the_mean_pattern_with_its_edge_in_mrad():
    pytest.importorskip("matplotlib")
    from matplotlib.patches import Ellipse

    fit = detector.fit_bright_field_disk(_blurred_disk(), semiangle_mrad=30)
    figure = fit.show()
    axis = figure.axes[0]
    assert axis.get_xlabel() == "detector column (mrad)" and axis.get_ylabel() == "detector row (mrad)"
    (edge,) = [patch for patch in axis.patches if isinstance(patch, Ellipse)]
    # the circle is the semiangle: its radius in mrad is the semiangle on both axes
    assert (edge.width / 2, edge.height / 2) == pytest.approx((30.0, 30.0), rel=1e-12)
    assert edge.center == pytest.approx((0.0, 0.0))
    uncalibrated = detector.fit_bright_field_disk(_blurred_disk()).show()
    assert uncalibrated.axes[0].get_xlabel() == "detector column (px)"
