"""Floating-point detector data must keep its bright-field disk.

The array fallback once summed every input with a uint64 accumulator. For
simulated intensities or normalised float32 data every value below 1 truncated
to 0, the mean diffraction pattern came back empty, and the SSB session stopped
with "No bright-field pixels found". Integer counts still accumulate in uint64.
"""

import numpy as np
import pytest

from quantem.gpu.detector import masked_sum, mean as detector_mean
from quantem.gpu.ssb.backends.mps.engine import _resolve_bf_selection

SCALE = 1000.0


def _bright_disk_counts() -> np.ndarray:
    """Return uint16 counts: a bright disk of radius 6 on a dim background."""
    rng = np.random.default_rng(0)
    det_rows, det_cols = np.mgrid[:32, :32]
    disk = (det_rows - 16) ** 2 + (det_cols - 15) ** 2 <= 6**2
    base = np.where(disk, 800, 5).astype(np.uint16)
    counts = base[None, None] + rng.integers(0, 50, (8, 8, 32, 32), dtype=np.uint16)
    return counts


def test_float_mean_dp_matches_scaled_counts():
    counts = _bright_disk_counts()
    intensities = (counts / SCALE).astype(np.float32)
    assert intensities.max() < 1.0
    np.testing.assert_allclose(
        detector_mean(intensities) * SCALE, detector_mean(counts), rtol=1e-6
    )


def test_float_bright_field_mask_matches_counts():
    counts = _bright_disk_counts()
    intensities = (counts / SCALE).astype(np.float32)
    counts_disk = _resolve_bf_selection(counts, threshold=0.5, bf_radius=None)
    float_disk = _resolve_bf_selection(intensities, threshold=0.5, bf_radius=None)
    assert float_disk.size > 0
    np.testing.assert_array_equal(float_disk.rows, counts_disk.rows)
    np.testing.assert_array_equal(float_disk.cols, counts_disk.cols)


def test_float_masked_sum_matches_scaled_counts():
    counts = _bright_disk_counts()
    intensities = (counts / SCALE).astype(np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:22, 10:22] = True
    np.testing.assert_allclose(
        masked_sum(intensities, mask) * SCALE, masked_sum(counts, mask), rtol=1e-6
    )


def test_integer_mean_dp_stays_exact():
    counts = _bright_disk_counts()
    expected = counts.sum(axis=(0, 1), dtype=np.uint64) / (8 * 8)
    np.testing.assert_array_equal(detector_mean(counts), expected.astype(np.float32))


def test_reuse_fitted_disk_for_bright_and_dark_field(monkeypatch):
    """Fit once, then make several detector images with the same geometry."""
    from quantem.gpu import detector

    data = _bright_disk_counts()
    mean_dp = detector.mean(data)
    center, radius = detector.fit_probe(mean_dp)
    assert center == (16.0, 15.0)
    assert radius == np.sqrt(113 / np.pi)

    def already_computed(_data):
        raise AssertionError("Supplied geometry must reuse the existing fit.")

    monkeypatch.setattr(detector.workflow, "mean", already_computed)
    rows, columns = np.indices(mean_dp.shape)
    distance = np.hypot(rows - center[0], columns - center[1])
    bright = detector.bf(data, center=center, radius=radius)
    dark = detector.df(data, center=center, radius=radius)
    np.testing.assert_array_equal(
        bright, data[..., distance <= radius].sum(axis=-1, dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        dark, data[..., distance >= radius].sum(axis=-1, dtype=np.uint64)
    )



def test_cuda_float_mean_dp_matches_scaled_counts():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable")
    from quantem.gpu.detector.backends.cuda.probe import mean_dp as cuda_mean_dp

    counts = _bright_disk_counts()
    intensities = (counts / SCALE).astype(np.float32)
    expected = detector_mean(counts)
    np.testing.assert_allclose(
        detector_mean(cp.asarray(intensities)) * SCALE, expected, rtol=1e-6
    )
    np.testing.assert_allclose(
        cuda_mean_dp(cp.asarray(intensities)).get() * SCALE, expected, rtol=1e-6
    )
