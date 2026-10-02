"""Explicit wave averaging preserves calibration and native diagnostic loss."""

import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session


def _direct_wave_average(ssb, factor, tilt, depth):
    """Independent Torch correction and per-detector inverse transforms."""
    import torch
    from quantem.gpu.ssb.torch_ssb import TorchSSB

    reference = TorchSSB.from_ssb(ssb)
    rows, cols = reference.ny * factor, reference.nx * factor
    q_row = torch.fft.fftfreq(rows, d=0.3 / factor, device=reference.device)
    q_col = torch.fft.fftfreq(cols, d=0.3 / factor, device=reference.device)
    wave_sum_t = torch.zeros((rows, cols), dtype=torch.complex128, device=reference.device)
    for start in range(0, reference.num_bf, 4):
        stop = min(start + 4, reference.num_bf)
        native_t = torch.fft.ifft2(reference._full_plane(reference.G[start:stop])).real
        sparse_t = torch.zeros((stop - start, rows, cols), device=reference.device)
        sparse_t[:, ::factor, ::factor] = native_t
        spectrum_t = torch.fft.fft2(sparse_t)
        gamma_t = reference._gamma(
            q_row[None, :, None], q_col[None, None, :],
            reference.kx[start:stop, None, None], reference.ky[start:stop, None, None],
            -80.0, 40.0, -0.7, tilt, depth * 10,
        )
        magnitude_squared_t = gamma_t.abs().square()
        unit_t = gamma_t * torch.where(
            magnitude_squared_t > 1e-16,
            torch.rsqrt(magnitude_squared_t.clamp_min(1e-16)),
            torch.full_like(magnitude_squared_t, 1e8),
        )
        corrected_t = spectrum_t * torch.conj(unit_t)
        corrected_t[:, 0, 0] = reference.dc_value
        wave_sum_t += torch.fft.ifft2(corrected_t).sum(0, dtype=torch.complex128)
    return torch.angle(wave_sum_t).float().cpu().numpy()


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
@pytest.mark.parametrize("depth", [0.0, 18.0])
def test_explicit_estimator_matches_direct_waves_and_preserves_native_loss(factor, depth):
    """Change output sampling with a fixed estimator and native calibration."""
    ssb, _ = _synthetic_session()
    aberrations = {"C10": -8.0, "C12": 4.0, "phi12": -0.7}
    tilt = (-9.0, 5.0)
    try:
        native, native_loss = ssb.preview(aberrations, tilt_mrad=tilt, depth_spread_nm=depth)
        legacy, _ = ssb.preview(
            aberrations, tilt_mrad=tilt, depth_spread_nm=depth, upsampling_factor=factor,
        )
        candidate, loss = ssb.preview(
            aberrations, tilt_mrad=tilt, depth_spread_nm=depth,
            upsampling_factor=factor, phase_estimator="phase_of_mean",
        )
        expected = _direct_wave_average(ssb, factor, tilt, depth)
        np.testing.assert_allclose(candidate, expected, atol=1e-6, rtol=1e-5)
        assert loss == native_loss
        assert np.isfinite(candidate).all()
        restored, restored_loss = ssb.preview(
            aberrations, tilt_mrad=tilt, depth_spread_nm=depth,
        )
        np.testing.assert_array_equal(restored, native)
        assert restored_loss == native_loss
        repeated, _ = ssb.preview(
            aberrations, tilt_mrad=tilt, depth_spread_nm=depth, upsampling_factor=factor,
        )
        np.testing.assert_array_equal(repeated, legacy)
    finally:
        ssb.close()


@pytest.mark.parametrize("depth", [0.0, 18.0])
def test_wave_average_chunking_and_inactive_higher_order_angles(depth):
    """Preview the same correction after changing memory batching and packed coefficients."""
    import cupy as cp

    ssb, engine = _synthetic_session()
    aberrations = {"C10": -8.0, "C12": 4.0, "phi12": -0.7}
    tilt = (-9.0, 5.0)
    try:
        candidate, loss = ssb.preview(
            aberrations, tilt_mrad=tilt, depth_spread_nm=depth,
            upsampling_factor=4, phase_estimator="phase_of_mean", compute_loss=False,
        )
        assert loss is None
        batched, _ = engine.reconstruct_thick(
            -80.0, 40.0, -0.7, tilt, depth * 10, upsampling_factor=4,
            phase_estimator="phase_of_mean", compute_loss=False,
            chunk_bytes=2 * 512 * 512 * 24,
        )
        np.testing.assert_allclose(cp.asnumpy(batched), candidate, atol=1e-6, rtol=0)
        magnitudes = np.zeros(14, np.float32)
        magnitudes[:2] = [-8, 4]
        angles = np.zeros(14, np.float32)
        angles[1:3] = [-0.7, 1.2]
        packed, _ = ssb.preview(
            aberrations, tilt_mrad=tilt, depth_spread_nm=depth,
            upsampling_factor=4, phase_estimator="phase_of_mean", compute_loss=False,
            higher_order_magnitudes=magnitudes, higher_order_angles=angles,
        )
        np.testing.assert_allclose(packed, candidate, atol=1e-6, rtol=0)
    finally:
        ssb.close()
