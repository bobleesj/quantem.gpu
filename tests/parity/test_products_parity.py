import numpy as np
import pytest


def _synthetic_4dstem() -> np.ndarray:
    rng = np.random.default_rng(7)
    data = rng.integers(0, 200, size=(4, 5, 8, 8), dtype=np.uint16)
    rows = np.arange(8, dtype=np.float32)[:, None]
    cols = np.arange(8, dtype=np.float32)[None, :]
    disk = ((rows - 3.5) ** 2 + (cols - 3.5) ** 2 <= 2.5**2).astype(np.uint16)
    data += 25 * disk[None, None, :, :]
    return data


def test_center_of_mass_numpy_fallback_applies_mask() -> None:
    from quantem.gpu.dpc import center_of_mass

    data = _synthetic_4dstem()
    mask = np.zeros(data.shape[-2:], dtype=bool)
    mask[2:6, 1:4] = True

    got_row, got_col = center_of_mass(data, mask=mask)
    full_row, full_col = center_of_mass(data)

    rows = np.arange(data.shape[-2], dtype=np.float64)[:, None]
    cols = np.arange(data.shape[-1], dtype=np.float64)[None, :]
    masked = data.astype(np.float64) * mask
    denom = np.maximum(masked.sum(axis=(2, 3)), 1e-10)
    exp_row = ((masked * rows).sum(axis=(2, 3)) / denom).astype(np.float32)
    exp_col = ((masked * cols).sum(axis=(2, 3)) / denom).astype(np.float32)
    exp_row = exp_row - float(exp_row.mean())
    exp_col = exp_col - float(exp_col.mean())

    np.testing.assert_allclose(got_row, exp_row, atol=1e-5)
    np.testing.assert_allclose(got_col, exp_col, atol=1e-5)
    assert not np.allclose(got_row, full_row)
    assert not np.allclose(got_col, full_col)


def test_cupy_detector_products_match_manual_sum() -> None:
    cp = pytest.importorskip("cupy")
    from quantem.gpu import detector

    data_np = _synthetic_4dstem()
    data = cp.asarray(data_np)

    mean = detector.mean(data)
    np.testing.assert_allclose(
        mean,
        data_np.reshape(-1, 8, 8).mean(axis=0).astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    mask = detector.detector_mask((3.5, 3.5), 0.0, 2.5, (8, 8))
    out = detector.masked_sum(data, mask)
    rows = np.arange(8, dtype=np.float32)[:, None]
    cols = np.arange(8, dtype=np.float32)[None, :]
    mask = (rows - 3.5) ** 2 + (cols - 3.5) ** 2 <= 2.5**2
    expected = (data_np * mask).sum(axis=(2, 3)).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_auto_probe_on_gaussian_dp() -> None:
    cp = pytest.importorskip("cupy")
    from quantem.gpu.detector import fit_probe

    det = 64
    y, x = cp.mgrid[:det, :det]
    center = det / 2 - 0.5
    r2 = (y - center) ** 2 + (x - center) ** 2
    dp = cp.exp(-r2 / (2 * 8.0**2)).astype(cp.float32)

    (row, col), radius = fit_probe(cp.asnumpy(dp))

    assert radius > 0
    assert abs(row - center) < 3
    assert abs(col - center) < 3
