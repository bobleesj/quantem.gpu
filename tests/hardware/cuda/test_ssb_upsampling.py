"""GPU upsampling preserves the model, native search loss, and source state."""
import math
import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def _wave_sum_oracle(ssb, factor, coefficients, tilt, depth):
    """The phase of the BF-averaged wave from an independent Torch construction of the zero-inserted scan.

    Inserting zero scan samples in real space, then an FFT, unfolds the measured aliases; it is not image interpolation.
    """
    import torch
    from tests.parity.torch_ssb import TorchSSB

    oracle = TorchSSB.from_ssb(ssb)
    q = torch.fft.fftfreq(128 * factor, d=0.3 / factor, device=oracle.device)
    spectrum = torch.zeros((q.numel(), q.numel()), device=q.device, dtype=torch.complex128)
    for start in range(0, oracle.num_bright_field_pixels, 17):
        stop = min(start + 17, oracle.num_bright_field_pixels)
        images = torch.fft.ifft2(oracle._full_plane(oracle.G[start:stop])).real
        sparse = torch.zeros((stop - start, q.numel(), q.numel()), device=q.device)
        sparse[:, ::factor, ::factor] = images
        aliases = torch.fft.fft2(sparse)
        gamma = oracle._gamma(q[None, :, None], q[None, None, :],
                              oracle.kx[start:stop, None, None], oracle.ky[start:stop, None, None],
                              *coefficients, tilt, depth)
        mag2 = gamma.real**2 + gamma.imag**2
        unit = gamma * torch.where(mag2 > 1e-16, torch.rsqrt(mag2.clamp_min(1e-16)), 1e8)
        corrected = aliases * unit.conj()
        corrected[:, 0, 0] = oracle.dc_value
        spectrum += corrected.sum(dim=0, dtype=torch.complex128)
    return torch.angle(torch.fft.ifft2(spectrum / oracle.num_bright_field_pixels)).float().cpu().numpy()


@pytest.mark.parametrize('factor', [2, 3, 4, 8])
def test_upsampling_matches_zero_inserted_scan(factor):
    import cupy as cp

    ssb, _ = _synthetic_session()
    try:
        coefs = {'C10_nm': 30.0, 'C12_nm': 5.0, 'phi12_deg': math.degrees(0.3)}
        native, native_loss = ssb.preview(coefs)
        phase, loss = ssb.preview(coefs, upsampling=factor)
        assert phase.shape == (128 * factor, 128 * factor)
        assert loss == native_loss
        expected = _wave_sum_oracle(ssb, factor, (300.0, 50.0, 0.3), (0.0, 0.0), 0.0)
        np.testing.assert_allclose(phase, expected, atol=2e-5)
        again, again_loss = ssb.preview(coefs)
        np.testing.assert_array_equal(again, native)
        assert again_loss == native_loss
    finally:
        ssb.close()
        cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_tilt_sampling_reuses_depth_kernel_and_native_loss(factor):
    import cupy as cp

    ssb, engine = _synthetic_session()
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": math.degrees(-0.7)}
    tilt, depth = (-9.0, 5.0), 18.0
    try:
        native, native_loss = ssb.preview(coefs, sample_tilt_mrad=tilt, depth_spread_nm=depth)
        actual, loss = ssb.preview(
            coefs, sample_tilt_mrad=tilt, depth_spread_nm=depth, upsampling=factor,
        )
        expected, _ = engine.thick.reconstruct(
            -80.0, 40.0, -0.7, tilt, depth * 10, compute_loss=False,
            upsampling_factor=factor,
        )
        np.testing.assert_array_equal(actual, cp.asnumpy(expected))
        assert loss == native_loss
        restored, restored_loss = ssb.preview(coefs, sample_tilt_mrad=tilt, depth_spread_nm=depth)
        np.testing.assert_array_equal(restored, native)
        assert restored_loss == native_loss
        # UI can retain an angle after its higher-order magnitude is cleared.
        inactive, _ = ssb.preview(
            coefs | {"phi21_deg": math.degrees(1.2)}, sample_tilt_mrad=tilt,
            depth_spread_nm=depth, upsampling=factor,
        )
        np.testing.assert_allclose(inactive, actual, atol=2e-6)
    finally:
        ssb.close()


@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("depth", [0.0, 150.0])
def test_wave_sum_matches_independent_torch_alias_construction(factor, depth):
    """Alias reads and bounded BF reduction retain the float64 wave average."""
    import cupy as cp

    ssb, engine = _synthetic_session()
    try:
        expected = _wave_sum_oracle(ssb, factor, (-80.0, 40.0, -0.7), (-9.0, 5.0), depth)
        size = 128 * factor
        actual, _ = engine.thick.reconstruct(-80, 40, -0.7, (-9, 5), depth,
                    compute_loss=False, upsampling_factor=factor, chunk_bytes=2 * size**2 * 16)
        np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=2e-5)
    finally:
        ssb.close()
