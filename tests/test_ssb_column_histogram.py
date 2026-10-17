"""Notebook histogram semantics remain exact in non-Matplotlib consumers."""

import numpy as np
import pytest

from quantem.gpu.ssb.results import column_histogram, column_sign


def test_histogram_matches_original_notebook_and_keeps_outside_pixels():
    phase = np.random.default_rng(32).exponential(size=(64, 64)).astype(np.float32)
    original = phase.copy()
    values = phase.astype(np.float64).ravel()
    values -= np.median(values)
    limit = np.percentile(np.abs(values), 99.9)
    edges = np.linspace(-limit, limit, 101)
    expected = np.histogram(values, edges)[0] / values.size
    fractions, actual_edges = column_histogram(phase)
    np.testing.assert_array_equal(actual_edges, edges)
    np.testing.assert_array_equal(fractions, expected)
    assert fractions.sum() < 1
    np.testing.assert_array_equal(phase, original)


def test_phase_gauge_and_polarity():
    phase = np.array([[0., 0., 0., 0.], [0., 0., 1., 4.]])
    fractions, edges = column_histogram(phase, limit=5, bins=11)
    shifted, _ = column_histogram(phase + 8, limit=5, bins=11)
    reversed_phase, _ = column_histogram(-phase, limit=5, bins=11)
    np.testing.assert_array_equal(shifted, fractions)
    np.testing.assert_array_equal(reversed_phase, fractions[::-1])
    assert column_sign(-phase) == -column_sign(phase)
    assert edges.size == 12


def test_constant_and_single_pixel_are_non_degenerate():
    for phase in (np.ones((3, 3)), np.array([7.])):
        counts, edges = column_histogram(phase)
        assert counts.sum() == 1
        assert np.all(np.diff(edges) > 0)
        assert column_sign(phase) == 0


@pytest.mark.parametrize("phase", [[], [np.nan], [np.inf], [1j]])
def test_invalid_phase_fails_closed(phase):
    with pytest.raises(ValueError, match="finite real"):
        column_histogram(phase)


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": -1}, {"limit": np.inf},
                                    {"bins": 0}, {"bins": 1.5}, {"bins": True}])
def test_invalid_axis_fails_closed(kwargs):
    with pytest.raises(ValueError):
        column_histogram([0., 1.], **kwargs)
