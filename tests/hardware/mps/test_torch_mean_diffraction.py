"""Selected mean diffraction through the public Torch MPS detector API."""

import numpy as np
import pytest

from quantem.gpu import detector


def test_uint16_mean_diffraction_preserves_counts_before_output_conversion():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("A Torch MPS device is required for mean diffraction.")
    counts = np.random.default_rng(2).integers(
        0, 65536, size=(4, 4, 8, 8), dtype=np.uint16
    )
    counts[0, 0] = 65535
    expected = counts.sum(axis=(0, 1), dtype=np.uint64).astype(np.float32) / 16
    result = detector.mean_dp(torch.as_tensor(counts, device="mps"))
    np.testing.assert_array_equal(result, expected)


def test_float32_mean_diffraction_keeps_fractional_intensities():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("A Torch MPS device is required for mean diffraction.")
    values = np.arange(4 * 4 * 8 * 8, dtype=np.float32).reshape(4, 4, 8, 8) / 8
    result = detector.mean_dp(torch.as_tensor(values, device="mps"))
    np.testing.assert_array_equal(result, values.mean(axis=(0, 1)))
