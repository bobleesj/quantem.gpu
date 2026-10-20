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
