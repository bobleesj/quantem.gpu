"""GPU upsampling preserves the model, native search loss, and source state."""
import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


@pytest.mark.parametrize('factor', [2, 3, 4, 8])
def test_upsampling_matches_zero_inserted_scan(factor):
    import cupy as cp
    import torch
    from tests.parity.torch_ssb import TorchSSB

    ssb, _ = _synthetic_session()
    try:
        coefs = {'C10': 30.0, 'C12': 5.0, 'phi12': 0.3}
        native, native_loss = ssb.preview(coefs, phase_estimator="mean_phase")
        phase, loss = ssb.preview(coefs, phase_estimator="mean_phase", upsampling_factor=factor)
        assert phase.shape == (128 * factor, 128 * factor)
        assert loss == native_loss
        # Independent Fourier construction: insert zero scan samples in real
        # space, then FFT. This is alias unfolding, not image interpolation.
        base = TorchSSB.from_ssb(ssb)
        full = base._full_plane(base.G)
        images = torch.fft.ifft2(full).real
        sparse = torch.zeros((base.num_bf, 128 * factor, 128 * factor), device=images.device)
        sparse[:, ::factor, ::factor] = images
        q = torch.fft.fftfreq(128 * factor, d=0.3 / factor, device=images.device)
        check = TorchSSB(torch.fft.fft2(sparse), kx=base.kx, ky=base.ky,
                         qx=q, qy=q, nx=128 * factor, wavelength=base.wavelength,
                         semiangle_rad=base.semiangle, ang_y_rad=base.ang_y,
                         ang_x_rad=base.ang_x, dc_value=base.dc_value)
        expected, _ = check.reconstruct(300.0, 50.0, 0.3, compute_loss=False)
        np.testing.assert_allclose(phase, expected.cpu().numpy(), atol=2e-5)
        again, again_loss = ssb.preview(coefs, phase_estimator="mean_phase")
        np.testing.assert_array_equal(again, native)
        assert again_loss == native_loss
    finally:
        ssb.close()
        cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_tilt_sampling_reuses_depth_kernel_and_native_loss(factor):
    import cupy as cp

    ssb, engine = _synthetic_session()
    coefs = {"C10": -8.0, "C12": 4.0, "phi12": -0.7}
    tilt, depth = (-9.0, 5.0), 18.0
    try:
        native, native_loss = ssb.preview(coefs, phase_estimator="mean_phase", tilt_mrad=tilt, depth_spread_nm=depth)
        actual, loss = ssb.preview(
            coefs, phase_estimator="mean_phase", tilt_mrad=tilt, depth_spread_nm=depth, upsampling_factor=factor,
        )
        expected, _ = engine.thick.reconstruct(
            -80.0, 40.0, -0.7, tilt, depth * 10, compute_loss=False,
            upsampling_factor=factor,
        )
        np.testing.assert_array_equal(actual, cp.asnumpy(expected))
        assert loss == native_loss
        restored, restored_loss = ssb.preview(coefs, phase_estimator="mean_phase", tilt_mrad=tilt, depth_spread_nm=depth)
        np.testing.assert_array_equal(restored, native)
        assert restored_loss == native_loss
        # UI can retain an angle after its higher-order magnitude is cleared.
        magnitudes = np.zeros(14, np.float32)
        magnitudes[:2] = [-8, 4]
        angles = np.zeros(14, np.float32)
        angles[1] = -0.7
        angles[2] = 1.2
        inactive, _ = ssb.preview(
            coefs, phase_estimator="mean_phase", tilt_mrad=tilt, depth_spread_nm=depth, upsampling_factor=factor,
            higher_order_magnitudes=magnitudes, higher_order_angles=angles,
        )
        np.testing.assert_allclose(inactive, actual, atol=2e-6)
    finally:
        ssb.close()


@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("depth", [0.0, 150.0])
def test_wave_sum_matches_independent_torch_alias_construction(factor, depth):
    """Alias reads and bounded BF reduction retain the float64 wave average."""
    import cupy as cp
    import torch
    from tests.parity.torch_ssb import TorchSSB

    ssb, engine = _synthetic_session()
    try:
        oracle = TorchSSB.from_ssb(ssb)
        q = torch.fft.fftfreq(128 * factor, d=0.3 / factor, device=oracle.device)
        spectrum = torch.zeros((q.numel(), q.numel()), device=q.device, dtype=torch.complex128)
        for start in range(0, oracle.num_bf, 17):
            stop = min(start + 17, oracle.num_bf)
            images = torch.fft.ifft2(oracle._full_plane(oracle.G[start:stop])).real
            sparse = torch.zeros((stop - start, q.numel(), q.numel()), device=q.device)
            sparse[:, ::factor, ::factor] = images
            aliases = torch.fft.fft2(sparse)
            gamma = oracle._gamma(q[None, :, None], q[None, None, :],
                                  oracle.kx[start:stop, None, None], oracle.ky[start:stop, None, None],
                                  -80.0, 40.0, -0.7, (-9.0, 5.0), depth)
            mag2 = gamma.real**2 + gamma.imag**2
            unit = gamma * torch.where(mag2 > 1e-16, torch.rsqrt(mag2.clamp_min(1e-16)), 1e8)
            corrected = aliases * unit.conj()
            corrected[:, 0, 0] = oracle.dc_value
            spectrum += corrected.sum(dim=0, dtype=torch.complex128)
        expected = torch.angle(torch.fft.ifft2(spectrum / oracle.num_bf)).float()
        actual, _ = engine.thick.reconstruct(-80, 40, -0.7, (-9, 5), depth,
                    compute_loss=False, phase_estimator="phase_of_mean", upsampling_factor=factor,
                    chunk_bytes=2 * q.numel()**2 * 16)
        np.testing.assert_allclose(cp.asnumpy(actual), expected.cpu().numpy(), atol=2e-5)
    finally:
        ssb.close()
