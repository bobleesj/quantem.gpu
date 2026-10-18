"""A scan mean cannot become a spatial signal when the output grid changes."""

import platform

import numpy as np
import pytest


def _session(constant=True, source_path=None):
    from quantem.gpu import SSB

    backend = "mps" if platform.system() == "Darwin" else "cuda"
    if backend == "mps":
        pytest.importorskip("mlx.core")
    else:
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("This workflow requires a physical CUDA GPU")
    row, col = np.meshgrid(np.arange(16) - 7.5, np.arange(16) - 7.5, indexing="ij")
    disk = (50 + 1000 * np.exp(-(row**2 + col**2) / 7)) * (1 + .03 * row + .02 * col)
    counts = np.broadcast_to(disk.astype(np.uint16), (128, 128, 16, 16)).copy()
    if not constant:
        counts += np.random.default_rng(9).poisson(10, counts.shape).astype(np.uint16)
    if source_path is not None:
        np.save(source_path, counts)
    return SSB(
        counts, backend=backend, voltage_kV=300, semiangle_mrad=30, source_path=source_path,
        scan_sampling_A=(.5, .5), detector_sampling_mrad=(10., 10.),
    )


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("estimator", ["mean_phase", "phase_of_mean"])
@pytest.mark.parametrize("depth", [0., 18.])
def test_constant_scan_remains_flat_inside_alias_support(factor, estimator, depth):
    """Test coarse scan steps whose DC replicas lie inside aperture support."""
    with _session() as ssb:
        phase, loss = ssb.preview(
            {"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.},
            sample_tilt_mrad=(-9., 5.), depth_spread_nm=depth,
            upsampling_factor=factor, phase_estimator=estimator, compute_loss=False,
        )
        assert loss is None
        assert np.isfinite(phase).all()
        np.testing.assert_allclose(phase, 0, rtol=0, atol=1e-7)


def test_shared_default_retains_native_diagnostic_and_saved_policy(tmp_path):
    """An explicit legacy comparison does not change the fit diagnostic."""
    coefs = {"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.}
    sample = dict(sample_tilt_mrad=(-9., 5.), depth_spread_nm=18.)
    with _session(constant=False, source_path=tmp_path / "counts.npy") as ssb:
        wave_phase, native_loss = ssb.preview(coefs, **sample)
        explicit, explicit_loss = ssb.preview(coefs, **sample, phase_estimator="phase_of_mean")
        np.testing.assert_array_equal(wave_phase, explicit)
        assert native_loss == explicit_loss
        _, legacy_loss = ssb.preview(coefs, **sample, phase_estimator="mean_phase")
        assert legacy_loss == native_loss
        saved = ssb.reconstruct(aberrations=coefs, upsampling_factor="auto", save_to=tmp_path)
        assert saved.phase_estimator == "phase_of_mean"
        assert saved.metadata["signature"]["reconstruction"]["upsampling_dc_policy"] == "single_origin"


def test_higher_order_default_uses_shared_estimator():
    from quantem.gpu.ssb.workflow import _default_phase_estimator

    assert _default_phase_estimator({"C10_nm": 2.}) == "phase_of_mean"
    base = {"C10_nm": 0., "C12_nm": 0., "phi12_deg": 0.}
    assert _default_phase_estimator(base | {"C30_nm": 1.}) == "phase_of_mean"
    assert _default_phase_estimator(base | {"C30_nm": 0., "phi21_deg": 20.}) == "phase_of_mean"


def _mean_free_reference(ssb, factor, depth, estimator):
    """Subtract the scan mean before zero insertion on the physical GPU."""
    if ssb.backend == "mps":
        from quantem.gpu.ssb.mps.prepared import expand_hermitian_mx
        from quantem.gpu.ssb.mps.thick_sample import _compiled_terms, _params

        prepared = ssb._backend()._prepared
        mx = prepared.mx
        native = prepared.scan_shape[0]
        n = native * factor
        q = mx.array((np.fft.fftfreq(n) * n * float(prepared.q_row[1])).astype(np.float32))
        params = _params(prepared, -80., 40., -0.7, (-9., 5.), depth * 10)
        total = mx.zeros((n, n), dtype=mx.complex64 if estimator == "phase_of_mean" else mx.float32)
        for start in range(0, prepared.g_qk.shape[0], 7):
            stop = min(start + 7, prepared.g_qk.shape[0])
            images = mx.real(mx.fft.ifft2(expand_hermitian_mx(mx, prepared.g_qk[start:stop], native)))
            images = images - mx.mean(images, axis=(1, 2), keepdims=True)
            sparse = mx.zeros((stop - start, n, n))
            sparse[:, ::factor, ::factor] = images
            aliases = mx.fft.fft2(sparse)
            real, imag, weight2 = _compiled_terms()(
                mx.real(aliases), mx.imag(aliases), q[None, :, None], q[None, None, :],
                prepared.kx[start:stop, None, None], prepared.ky[start:stop, None, None], params,
            )
            corrected = (real + 1j * imag) / mx.maximum(mx.sqrt(weight2), 1e-8)
            corrected[:, 0, 0] = prepared.dc_value
            if estimator == "phase_of_mean":
                total = total + mx.sum(corrected, axis=0)
            else:
                wave = mx.fft.ifft2(corrected)
                total = total + mx.sum(mx.arctan2(mx.imag(wave), mx.real(wave)), axis=0)
            mx.eval(total)
        if estimator == "phase_of_mean":
            total[0, 0] = total[0, 0] + (prepared.num_bright_field_pixels - prepared.g_qk.shape[0]) * prepared.dc_value
            wave = mx.fft.ifft2(total)
            total = mx.arctan2(mx.imag(wave), mx.real(wave))
        else:
            total = total / prepared.num_bright_field_pixels
        mx.eval(total)
        return np.asarray(total)

    import torch
    from tests.parity.torch_ssb import TorchSSB

    reference = TorchSSB.from_ssb(ssb)
    n = reference.ny * factor
    q = torch.fft.fftfreq(n, d=0.5 / factor, device=reference.device)
    total = torch.zeros((n, n), device=reference.device,
                        dtype=torch.complex128 if estimator == "phase_of_mean" else torch.float32)
    for start in range(0, reference.num_bright_field_pixels, 7):
        stop = min(start + 7, reference.num_bright_field_pixels)
        images = torch.fft.ifft2(reference._full_plane(reference.G[start:stop])).real
        images = images - images.mean(dim=(1, 2), keepdim=True)
        sparse = torch.zeros((stop - start, n, n), device=reference.device)
        sparse[:, ::factor, ::factor] = images
        aliases = torch.fft.fft2(sparse)
        gamma = reference._gamma(
            q[None, :, None], q[None, None, :], reference.kx[start:stop, None, None],
            reference.ky[start:stop, None, None], -80., 40., -0.7, (-9., 5.), depth * 10,
        )
        mag2 = gamma.abs().square()
        corrected = aliases * (gamma / torch.sqrt(mag2).clamp_min(1e-8)).conj()
        corrected[:, 0, 0] = reference.dc_value
        if estimator == "phase_of_mean":
            total += corrected.sum(0, dtype=torch.complex128)
        else:
            total += torch.angle(torch.fft.ifft2(corrected)).sum(0)
    if estimator == "phase_of_mean":
        total = torch.angle(torch.fft.ifft2(total)).float()
    else:
        total /= reference.num_bright_field_pixels
    return total.cpu().numpy()


@pytest.mark.parametrize("estimator", ["mean_phase", "phase_of_mean"])
@pytest.mark.parametrize("depth", [0., 18.])
def test_nonconstant_scan_preserves_mean_free_fourier_signal(estimator, depth):
    """The DC correction must retain measured contrast at other frequencies."""
    coefs = {"C10_nm": -8., "C12_nm": 4., "phi12_deg": np.degrees(-0.7)}
    with _session(constant=False) as ssb:
        actual, _ = ssb.preview(
            coefs, sample_tilt_mrad=(-9., 5.), depth_spread_nm=depth,
            upsampling_factor=4, phase_estimator=estimator, compute_loss=False,
        )
        expected = _mean_free_reference(ssb, 4, depth, estimator)
        assert np.std(actual) > 1e-5
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
