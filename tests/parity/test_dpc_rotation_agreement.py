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
    detector_transposed = idx >= rotation_steps
    ai = idx % rotation_steps
    expected_angle = float(angles[ai]) * 180.0 / np.pi
    expected_row = rt[ai] if detector_transposed else r[ai]
    expected_col = ct[ai] if detector_transposed else c[ai]

    got_row, got_col, got_angle, got_transpose = find_optimal_rotation(
        com_row,
        com_col,
        rotation_steps=rotation_steps,
    )

    assert got_transpose == detector_transposed
    assert got_angle == pytest.approx(expected_angle, abs=1e-6)
    np.testing.assert_allclose(got_row, expected_row, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_col, expected_col, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("exchange_detector_axes", [False, True])
def test_run_hands_its_fitted_geometry_back_without_losing_the_transposition(exchange_detector_axes) -> None:
    """A fit reports its rotation modulo 180 with the transposition; giving both back reproduces the phase.

    The forced rotation used to take an angle alone, so a fit made on transposed detector axes could not be repeated.
    """
    from quantem.gpu import dpc

    # one bright pixel per pattern, displaced by the gradient of a smooth potential: a curl-free centre of mass
    rows, cols = np.mgrid[:24, :28]
    potential = 40 * np.exp(-((rows - 9) ** 2 + (cols - 15) ** 2) / 60)
    shift_row, shift_col = (np.rint(gradient).astype(int) for gradient in np.gradient(potential))
    data = np.zeros((24, 28, 16, 16), np.uint16)
    data[rows, cols, 8 + shift_row, 8 + shift_col] = 100
    if exchange_detector_axes:
        data = np.ascontiguousarray(data.transpose(0, 1, 3, 2))

    fitted = dpc.run(data, rotation_steps=30)
    assert fitted.detector_transposed == exchange_detector_axes
    assert fitted.scan_detector_rotation_deg is None
    # The estimate lies in [0, 180]. This unrotated field fits the end point of the search grid, which is pi in
    # float32 and reads 180.000005 degrees, so the bound allows for that rounding; 0 and 180 are the same estimate.
    assert 0.0 <= fitted.scan_detector_rotation_mod180_deg <= 180.0 + 1e-5
    # the fit is returned as it was applied, not wrapped: its phase belongs to this angle and not to the one at 0
    assert fitted.scan_detector_rotation_mod180_deg == float(np.float32(np.pi)) * 180.0 / np.pi
    given = dpc.run(data, scan_detector_rotation_deg=fitted.scan_detector_rotation_mod180_deg,
                    detector_transposed=fitted.detector_transposed)
    assert given.scan_detector_rotation_mod180_deg is None
    assert given.scan_detector_rotation_deg == fitted.scan_detector_rotation_mod180_deg % 360.0
    assert given.detector_transposed == fitted.detector_transposed
    np.testing.assert_allclose(given.phase, fitted.phase, atol=1e-4 * np.abs(fitted.phase).max())
