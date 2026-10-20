"""Rectangular selections preserve counts and match an explicitly padded fit."""

import numpy as np
import pytest
import torch

from quantem.gpu import SSB


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")
@pytest.mark.parametrize("shape", [(37, 53), (143, 170)])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_rectangular_scan_matches_explicit_mean_padding(shape, dtype):
    rng = np.random.default_rng(37)
    counts = rng.integers(0, 65536, (*shape, 12, 12), dtype=np.uint16)
    counts = counts.astype(dtype)
    if dtype == np.float32:
        counts /= 1024
    size = 128 if max(shape) <= 128 else 256
    mean = counts.mean(axis=(0, 1), dtype=np.float64).astype(counts.dtype)
    expected = np.broadcast_to(mean, (size, size, 12, 12)).copy()
    row, col = ((size - n) // 2 for n in shape)
    expected[row:row + shape[0], col:col + shape[1]] = counts
    optics = {"C10_nm": 8.0, "C12_nm": 2.0, "phi12_deg": 20.0}
    options = dict(backend="mps", voltage_kV=200, semiangle_mrad=20,
                   scan_sampling_A=(.3, .3), detector_sampling_mrad=(5, 5),
                   bright_field_center_px=(5.5, 5.5), bright_field_radius_px=4,
                   scan_detector_rotation_deg=0.0, aberrations=optics)
    with SSB(torch.as_tensor(counts, device="mps"), **options) as regional:
        phase, _ = regional.preview(optics, compute_loss=False)
        assert regional.scan_shape == (size, size)
    with SSB(expected, **options) as reference:
        baseline, _ = reference.preview(optics, compute_loss=False)
    np.testing.assert_array_equal(phase, baseline)
