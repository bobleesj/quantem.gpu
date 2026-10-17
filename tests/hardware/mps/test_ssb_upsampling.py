"""Fine-grid MPS reconstruction retains measured aliases and native loss."""

import math

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps


def _disk_session(n=128, source_path=None, detector_sampling=(10.0, 10.0)):
    from quantem.gpu import SSB

    rng = np.random.default_rng(9)
    counts = rng.poisson(30.0, size=(n, n, 16, 16)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(16) - 7.5, np.arange(16) - 7.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 3] += 150
    if source_path is not None:
        np.save(source_path, counts)
    return SSB(
        counts,
        backend="mps",
        voltage_kV=300,
        semiangle_mrad=30,
        scan_sampling_A=(0.3, 0.3),
        detector_sampling_mrad=detector_sampling,
        source_path=None if source_path is None else str(source_path),
    )


def _zero_inserted_reference(prepared, factor, coefs, tilt, depth, estimator):
    """Independent GPU scan construction, followed by the existing model graph."""
    from quantem.gpu.ssb.mps.prepared import expand_hermitian_mx
    from quantem.gpu.ssb.mps.thick_sample import _compiled_terms, _params

    mx = prepared.mx
    native = prepared.scan_shape[0]
    n = native * factor
    q = mx.array((np.fft.fftfreq(n) * n * float(prepared.q_row[1])).astype(np.float32))
    params = _params(prepared, *coefs, tilt, depth)
    total = mx.zeros(
        (n, n), dtype=mx.complex64 if estimator == "phase_of_mean" else mx.float32
    )
    for start in range(0, prepared.g_qk.shape[0], 7):
        stop = min(start + 7, prepared.g_qk.shape[0])
        source = expand_hermitian_mx(mx, prepared.g_qk[start:stop], native)
        images = mx.real(mx.fft.ifft2(source))
        sparse = mx.zeros((stop - start, n, n))
        sparse[:, ::factor, ::factor] = images
        aliases = mx.fft.fft2(sparse)
        real, imag, weight2 = _compiled_terms()(
            mx.real(aliases),
            mx.imag(aliases),
            q[None, :, None],
            q[None, None, :],
            prepared.kx[start:stop, None, None],
            prepared.ky[start:stop, None, None],
            params,
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
        total[0, 0] = (
            total[0, 0]
            + (prepared.num_bright_field_pixels - prepared.g_qk.shape[0])
            * prepared.dc_value
        )
        wave = mx.fft.ifft2(total / prepared.num_bright_field_pixels)
        total = mx.arctan2(mx.imag(wave), mx.real(wave))
    else:
        total = total / prepared.num_bright_field_pixels
    mx.eval(total)
    return np.asarray(total)


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("estimator", ["mean_phase", "phase_of_mean"])
@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_alias_sampling_preserves_gpu_model_and_native_loss(
    factor, estimator, depth_nm
):
    _require_mps()
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": math.degrees(-0.7)}
    sample = {"sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": depth_nm}
    with _disk_session() as ssb:
        native, native_loss = ssb.preview(coefs, **sample)
        prepared = ssb._backend()._prepared
        expected = _zero_inserted_reference(
            prepared, factor, (-80.0, 40.0, -0.7), (-9, 5), depth_nm * 10, estimator
        )
        phase, loss = ssb.preview(
            coefs, **sample, upsampling_factor=factor, phase_estimator=estimator
        )
        assert phase.shape == (128 * factor, 128 * factor)
        np.testing.assert_allclose(phase, expected, rtol=0, atol=2e-5)
        assert loss == native_loss
        restored, restored_loss = ssb.preview(coefs, **sample)
        np.testing.assert_array_equal(restored, native)
        assert restored_loss == native_loss


# The unchanged 1024 kernel has a 1.4e-6 rad difference on this fixture;
# retain the existing thick-model 5e-6 float32 bound for that path.
@pytest.mark.parametrize(
    "n,atol", [(128, 5e-8), (256, 5e-8), (512, 5e-8), (1024, 5e-6)]
)
def test_fused_depth_model_at_all_native_sizes(n, atol):
    _require_mps()
    from quantem.gpu.ssb.mps.thick_sample import reconstruct_thick_reference

    with _disk_session(n) as ssb:
        coefs = {"C10_nm": 3.0, "C12_nm": 0.5, "phi12_deg": math.degrees(0.3)}
        phase, loss = ssb.preview(
            coefs, sample_tilt_mrad=(5.0, -3.0), depth_spread_nm=15.0,
            phase_estimator="mean_phase",  # Preserve the fused mean-phase oracle.
        )
        expected, expected_loss = reconstruct_thick_reference(
            ssb._backend()._prepared,
            C10=30.0,
            C12=5.0,
            phi12=0.3,
            sample_tilt_mrad=(5.0, -3.0),
            thickness=150.0,
        )
        np.testing.assert_allclose(phase, expected, rtol=0, atol=atol)
        assert abs(loss - expected_loss) <= 1e-6 * abs(expected_loss)


def test_automatic_sampling_rotation_and_saved_result(tmp_path):
    """Replay fine sampling and both rotations without refitting."""
    _require_mps()
    from quantem.gpu.optics.physics import ssb_upsampling_factor

    coefs = {"C10_nm": 3.0, "C12_nm": 0.5, "phi12_deg": 15.0}
    with _disk_session(source_path=tmp_path / "source.npy") as ssb:
        fine = ssb.reconstruct(aberrations=coefs, upsampling_factor="auto")
        factor = ssb_upsampling_factor(
            voltage_kV=300,
            semiangle_mrad=30,
            scan_sampling_A=(0.3, 0.3),
        )
        assert fine.upsampling_factor == factor
        assert fine.phase.shape == (128 * factor, 128 * factor)
        np.testing.assert_allclose(fine.object_sampling_A, (0.3 / factor,) * 2)
        assert fine.amplitude_estimated is False
        original = fine.phase.copy()
        ssb.set_rotation(180.0)
        opposite = ssb.reconstruct(aberrations=coefs, upsampling_factor="auto")
        assert opposite.scan_detector_rotation_deg == 180.0
        assert not np.array_equal(original, opposite.phase)
        ssb.set_rotation(0.0)
        restored = ssb.reconstruct(
            aberrations=coefs, upsampling_factor="auto", save_to=tmp_path / "fine"
        )
        np.testing.assert_array_equal(restored.phase, original)
        assert (tmp_path / "fine" / "ssb-reconstruct.npz").is_file()


@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_coherent_half_spectrum_retains_full_complex_result(monkeypatch, depth_nm):
    """Reduce paired frequencies only when the complete soft aperture fits."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling

    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    sample = {"sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": depth_nm}
    with _disk_session() as ssb:
        ssb.set_rotation(165.8)
        half, _ = ssb.preview(coefs, **sample, upsampling_factor=4, compute_loss=False)
        prepared = ssb._backend()._prepared
        assert upsampling._coherent_columns(prepared, 512) == 257
        # Native Nyquist frequencies intersect the aperture and are unpaired.
        assert upsampling._coherent_columns(prepared, 128) == 128
        monkeypatch.setattr(upsampling, "_coherent_columns", lambda prepared, n: n)
        full, _ = ssb.preview(coefs, **sample, upsampling_factor=4, compute_loss=False)
        np.testing.assert_allclose(half, full, rtol=0, atol=2e-5)


@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
@pytest.mark.parametrize("higher_order", [False, True])
def test_coherent_reduction_keeps_partial_bf_groups(monkeypatch, depth_nm, higher_order):
    """Every BF contribution survives grouping, including an incomplete tail."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling

    from .test_ssb_thick_sample_mps import _poisson_disk_session

    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    if higher_order:
        coefs.update(C21_nm=300.0, phi21_deg=23.0)
    sample = {"sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": depth_nm}
    with _poisson_disk_session() as ssb:
        ssb.set_rotation(165.8)
        actual, loss = ssb.preview(coefs, **sample, upsampling_factor=3)
        stored = int(ssb._backend()._prepared.g_qk.shape[0])
        chunk = upsampling._WORK_BYTES // ((128 * 3) ** 2 * 8 * 3)
        assert stored > chunk
        assert stored % chunk != 0

        def per_detector(prepared, qr, qc, params, dc, probe, coefficients,
                         higher, n, chunk, columns):
            mx = prepared.mx
            total = mx.zeros((n, n), dtype=mx.complex64)
            # A final single-BF tile exercises constant-address MLX inputs.
            for start in range(0, stored, stored - 1):
                stop = min(start + stored - 1, stored)
                corrected = upsampling._alias_correction_kernel()(
                    inputs=[prepared.g_qk[start:stop], qr, qc,
                            prepared.kx[start:stop], prepared.ky[start:stop],
                            params, dc, probe[start:stop], coefficients],
                    template=[("N", n), ("SOURCE_N", 128),
                              ("G_COLS", prepared.g_qk.shape[-1]),
                              ("K", stop - start), ("HIGHER", higher)],
                    grid=(n, n, stop - start), threadgroup=(32, 8, 1),
                    output_shapes=[(stop - start, n, n)],
                    output_dtypes=[mx.complex64],
                )[0]
                total = total + mx.sum(corrected, axis=0)
                mx.eval(total)
            return total[:, :columns]

        with monkeypatch.context() as ungrouped:
            ungrouped.setattr(upsampling, "_coherent_spectrum", per_detector)
            expected, expected_loss = ssb.preview(coefs, **sample, upsampling_factor=3)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
        assert loss == expected_loss
        repeated, repeated_loss = ssb.preview(coefs, **sample, upsampling_factor=3)
        np.testing.assert_array_equal(repeated, actual)
        assert repeated_loss == loss


@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_coherent_reconstruction_preserves_anisotropic_aperture(depth_nm):
    """A rotated detector with unequal pixel angles retains its soft edge."""
    _require_mps()
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": math.degrees(-0.7)}
    with _disk_session(detector_sampling=(10.0, 7.0)) as ssb:
        ssb.set_rotation(165.8)
        phase, _ = ssb.preview(
            coefs, sample_tilt_mrad=(-9.0, 5.0), depth_spread_nm=depth_nm,
            upsampling_factor=3, compute_loss=False,
        )
        expected = _zero_inserted_reference(
            ssb._backend()._prepared, 3, (-80.0, 40.0, -0.7),
            (-9.0, 5.0), depth_nm * 10, "phase_of_mean",
        )
        np.testing.assert_allclose(phase, expected, rtol=0, atol=2e-5)
