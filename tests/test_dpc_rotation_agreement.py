from __future__ import annotations

import numpy as np
import pytest


def test_find_optimal_rotation_matches_batch_reference() -> None:
    """The curl-score search should match the old full-stack search."""
    from quantem.gpu.dpc import (
        _curl_batch,
        _rotate_vector_batch,
        find_optimal_rotation,
    )

    rng = np.random.default_rng(41)
    com_row = rng.normal(size=(32, 40)).astype(np.float32)
    com_col = rng.normal(size=(32, 40)).astype(np.float32)
    rotation_steps = 91
    angles = np.linspace(0, np.pi, rotation_steps, dtype=np.float32)

    r, c = _rotate_vector_batch(com_row, com_col, angles)
    rt, ct = _rotate_vector_batch(com_col, com_row, angles)
    scores = np.concatenate([_curl_batch(r, c), _curl_batch(rt, ct)])
    idx = int(scores.argmin())
    use_transpose = idx >= rotation_steps
    ai = idx % rotation_steps
    expected_angle = float(angles[ai]) * 180.0 / np.pi
    expected_row = rt[ai] if use_transpose else r[ai]
    expected_col = ct[ai] if use_transpose else c[ai]

    got_row, got_col, got_angle, got_transpose = find_optimal_rotation(
        com_row,
        com_col,
        rotation_steps=rotation_steps,
    )

    assert got_transpose == use_transpose
    assert got_angle == pytest.approx(expected_angle, abs=1e-6)
    np.testing.assert_allclose(got_row, expected_row, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_col, expected_col, rtol=1e-6, atol=1e-6)
