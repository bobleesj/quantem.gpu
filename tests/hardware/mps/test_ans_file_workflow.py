"""Small physical file-to-GPU-to-detector workflow, not a speed benchmark."""

import numpy as np
import pytest

from quantem.gpu import detector, io


def test_public_ans_series_keeps_sources_independent_and_exact(tmp_path):
    """A compatible ANS list loads without a dense stack and preserves axes."""
    pytest.importorskip("Metal")
    counts = [
        np.arange(8 * 8 * 2 * 3, dtype=np.uint16).reshape(8, 8, 2, 3),
        np.arange(8 * 8 * 2 * 3, dtype=np.uint16).reshape(8, 8, 2, 3) + 100,
        np.arange(8 * 8 * 2 * 3, dtype=np.uint16).reshape(8, 8, 2, 3) + 200,
    ]
    paths = []
    for index, values in enumerate(counts):
        path = tmp_path / f"series-{index}.qem"
        io.save(path, values, format="quantem", compression="ans", backend="cpu")
        paths.append(path)

    loaded = io.load(
        paths,
        backend="mps",
        representation="encoded",
        stack=False,
        verbose=False,
    )
    try:
        session = detector.prepare(loaded)
        assert session.series_shape == (3,)
        np.testing.assert_array_equal(
            session.frame(17), np.stack([values.reshape(64, 2, 3)[17] for values in counts])
        )
        mask = np.ones((2, 3), dtype=np.uint8)
        expected = np.stack(
            [values.sum(axis=(2, 3), dtype=np.uint64) for values in counts], axis=0
        )
        np.testing.assert_array_equal(session.masked_sum_exact(mask), expected)
    finally:
        for item in loaded:
            item.close()
