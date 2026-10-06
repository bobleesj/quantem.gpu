"""Host mean patterns divide the exact total in float64 and round once; integer frame sums stay exact.

Rounding the total to float32 first loses counts above 2^24: on these patterns
``float32(total) / n`` differs from the correctly rounded mean in about four
of every ten pixels. ``reduce_frames`` returns integer counts' ``sum`` and
``max`` as exact uint64, like every other backend.
"""

import numpy as np
import pytest

from quantem.gpu import detector


def _counts():
    """324 bright uint16 frames: every detector total lies between 2^24 and 2^25."""
    return np.random.default_rng(81).integers(52_000, 65_536, (18, 18, 6, 7)).astype(np.uint16)


@pytest.mark.parametrize("runner", ["numpy", "torch-cpu"])
def test_host_mean_pattern_and_frame_reductions(runner):
    counts = _counts()
    data = counts if runner == "numpy" else pytest.importorskip("torch").from_numpy(counts)
    session = detector.prepare(data)
    total = counts.sum(axis=(0, 1), dtype=np.uint64)
    expected = (total / 324).astype(np.float32)
    assert int(total.min()) > 2**24
    assert np.any(total.astype(np.float32) / np.float32(324) != expected)
    np.testing.assert_array_equal(session.mean_dp(), expected)
    indices = np.arange(10, 310)
    selected = counts.reshape(-1, 6, 7)[indices]
    np.testing.assert_array_equal(
        session.reduce_frames(indices, "mean"), (selected.sum(axis=0, dtype=np.uint64) / 300).astype(np.float32)
    )
    for mode, expected_counts in (("sum", selected.sum(axis=0, dtype=np.uint64)), ("max", selected.max(axis=0))):
        result = session.reduce_frames(indices, mode)
        assert result.dtype == np.uint64
        np.testing.assert_array_equal(result, expected_counts)
