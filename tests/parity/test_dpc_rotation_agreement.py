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


@pytest.mark.parametrize("seed", [41, 7, 2026])
def test_find_optimal_rotation_is_the_minimum_of_the_full_stack_search(seed) -> None:
    """The closed-form angle is the minimizer of the curl score a fine full-stack search approaches.

    Before 2026-10-20 the angle was the best of 180 grid angles 1.0056 degrees apart, so an estimate could sit half a
    step from the minimum (a 512 x 512 scan read 158.88 degrees where 1800 steps gave 159.39).
    """
    from quantem.gpu.dpc.workflow import find_optimal_rotation

    rng = np.random.default_rng(seed)
    com_row = rng.normal(size=(32, 40)).astype(np.float32)
    com_col = rng.normal(size=(32, 40)).astype(np.float32)
    rotation_steps = 3601  # 0.05-degree grid
    angles = np.linspace(0, np.pi, rotation_steps)

    r, c = _rotate_vector_batch(com_row.astype(np.float64), com_col.astype(np.float64), angles)
    rt, ct = _rotate_vector_batch(com_col.astype(np.float64), com_row.astype(np.float64), angles)
    scores = np.concatenate([_curl_batch(r, c), _curl_batch(rt, ct)])
    idx = int(scores.argmin())
    grid_flipped = idx >= rotation_steps
    grid_angle = float(angles[idx % rotation_steps]) * 180.0 / np.pi

    got_row, got_col, got_angle, got_flipped = find_optimal_rotation(com_row, com_col)

    assert got_flipped == grid_flipped
    assert 0.0 <= got_angle < 180.0
    gap = abs(got_angle - grid_angle) % 180.0
    assert min(gap, 180.0 - gap) <= 0.05
    exact = np.radians(got_angle)
    source = (com_col, com_row) if got_flipped else (com_row, com_col)
    expected_row, expected_col = _rotate_vector_batch(*(np.asarray(v, np.float64) for v in source), np.array([exact]))
    # the exact angle scores no worse than any grid angle
    assert float(_curl_batch(expected_row, expected_col)[0]) <= float(scores.min()) * (1 + 1e-9)
    np.testing.assert_allclose(got_row, expected_row[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_col, expected_col[0], rtol=1e-5, atol=1e-5)
    # a given flip keeps that convention and fits the angle alone
    _, _, kept_angle, kept = find_optimal_rotation(com_row, com_col, scan_detector_rotation_flipped=not got_flipped)
    assert kept is (not got_flipped)
    other = (rt, ct) if not got_flipped else (r, c)
    kept_grid = float(angles[int(_curl_batch(*other).argmin())]) * 180.0 / np.pi
    gap = abs(kept_angle - kept_grid) % 180.0
    assert min(gap, 180.0 - gap) <= 0.05


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

    fitted = dpc.run(data)
    assert fitted.scan_detector_rotation_flipped == exchange_detector_axes
    assert fitted.scan_detector_rotation_deg is None
    # the exact estimate lies in [0, 180); this unrotated field fits 0, which modulo 180 may read just below 180
    angle = fitted.scan_detector_rotation_mod180_deg
    assert 0.0 <= angle < 180.0
    assert min(angle, 180.0 - angle) < 2.0
    given = dpc.run(data, scan_detector_rotation_deg=fitted.scan_detector_rotation_mod180_deg,
                    scan_detector_rotation_flipped=fitted.scan_detector_rotation_flipped)
    assert given.scan_detector_rotation_mod180_deg is None
    assert given.scan_detector_rotation_deg == fitted.scan_detector_rotation_mod180_deg % 360.0
    assert given.scan_detector_rotation_flipped == fitted.scan_detector_rotation_flipped
    np.testing.assert_allclose(given.phase, fitted.phase, atol=1e-4 * np.abs(fitted.phase).max())


def test_a_flipped_detector_integrates_to_the_same_phase() -> None:
    """With the detector axes swapped, dpc.run fits the flip and returns the phase of the unswapped scan.

    Before 2026-10-20 the aligned field, already the scan-frame gradient, was swapped back before integration: the
    flipped scan's phase correlated 0.10 with the unflipped one.
    """
    from quantem.gpu import dpc

    rows, cols = np.mgrid[:32, :40]
    # unequal widths and two sites: a swapped gradient of this potential is not curl-free, so the flip is identifiable
    potential = (30 * np.exp(-((rows - 10) ** 2 / 30 + (cols - 14) ** 2 / 90))
                 + 18 * np.exp(-((rows - 22) ** 2 / 60 + (cols - 30) ** 2 / 20)))
    gradient_row, gradient_col = np.gradient(potential)
    detector_row, detector_col = np.mgrid[:25, :25]
    data = np.exp(-((detector_row - 12 - gradient_row[..., None, None]) ** 2
                    + (detector_col - 12 - gradient_col[..., None, None]) ** 2) / 4.0).astype(np.float32)
    plain = dpc.run(data)
    flipped = dpc.run(np.ascontiguousarray(data.transpose(0, 1, 3, 2)))
    assert (plain.scan_detector_rotation_flipped, flipped.scan_detector_rotation_flipped) == (False, True)
    assert flipped.scan_detector_rotation_mod180_deg == pytest.approx(plain.scan_detector_rotation_mod180_deg, abs=1e-6)
    np.testing.assert_allclose(flipped.phase, plain.phase, atol=1e-4 * float(np.abs(plain.phase).max()))
    reference = potential - potential.mean()
    phase = plain.phase - plain.phase.mean()
    assert abs(float((phase * reference).sum() / np.sqrt((phase ** 2).sum() * (reference ** 2).sum()))) > 0.99
