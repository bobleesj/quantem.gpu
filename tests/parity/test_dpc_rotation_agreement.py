import numpy as np
import pytest


def _rotate_vector_batch(v_row, v_col, angles_rad):
    """Rotate one vector field by every angle, materializing the full stack."""
    c = np.cos(angles_rad)[:, None, None]
    s = np.sin(angles_rad)[:, None, None]
    return c * v_row - s * v_col, s * v_row + c * v_col


def _curl_batch(v_row, v_col):
    """Mean-squared central-difference curl of each field in a stack."""
    dv_row_dcol = 0.5 * (v_row[:, 1:-1, 2:] - v_row[:, 1:-1, :-2])
    dv_col_drow = 0.5 * (v_col[:, 2:, 1:-1] - v_col[:, :-2, 1:-1])
    curl = dv_col_drow - dv_row_dcol
    return (curl ** 2).mean(axis=(1, 2))


def test_find_optimal_rotation_matches_batch_reference() -> None:
    """The curl-score search should match the old full-stack search."""
    from quantem.gpu.dpc.workflow import find_optimal_rotation

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
