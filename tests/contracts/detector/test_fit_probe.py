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
