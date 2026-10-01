"""GPU upsampling preserves the model, native search loss, and source state."""
import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session


@pytest.mark.parametrize('factor', [2, 4, 8])
def test_upsampling_matches_zero_inserted_scan(factor):
    import cupy as cp
    import torch
    from quantem.gpu.ssb.torch_ssb import TorchSSB

    ssb, _ = _synthetic_session()
    try:
        coefs = {'C10': 30.0, 'C12': 5.0, 'phi12': 0.3}
        native, native_loss = ssb.preview(coefs)
        phase, loss = ssb.preview(coefs, upsampling_factor=factor)
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
        again, again_loss = ssb.preview(coefs)
        np.testing.assert_array_equal(again, native)
        assert again_loss == native_loss
        with pytest.raises(ValueError, match='standard'):
            ssb.preview(coefs, upsampling_factor=factor, depth_spread_nm=10)
    finally:
        ssb.close()
        cp.get_default_memory_pool().free_all_blocks()
