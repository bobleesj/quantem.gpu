"""Coherent redraw keeps all detector waves across bounded Metal launches."""

import numpy as np
import pytest

from tests.hardware.test_ssb_scan_mean import _session

pytest.importorskip("mlx.core", reason="Physical Apple Metal backend required")


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_coherent_sum_matches_per_detector_correction(monkeypatch, factor, anisotropic):
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.direct_ptycho.mps import upsampling as up
    from quantem.gpu.direct_ptycho.mps.thick_sample import _params

    with _session(constant=False) as ssb:
        ssb.preview({"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.}, compute_loss=False)
        prepared = ssb._backend()._prepared
        if anisotropic:
            prepared.ang_y_rad *= 0.5
            prepared.ang_x_rad *= 1.5
        native = prepared.scan_shape[0]
        n = native * factor
        # Exercise multiple launches and an incomplete final launch.
        monkeypatch.setattr(up, "_WORK_BYTES", n * n * 8 * 6)
        chunk = 2
        stored = prepared.g_qk.shape[0]
        assert stored > 12 and stored % 12 != 0
        frequency = np.fft.fftfreq(n) * n
        qr = mx.array((frequency * float(prepared.q_row[1])).astype(np.float32))
        qc = mx.array((frequency * float(prepared.q_col[1])).astype(np.float32))
        sample = {"C10": -80., "C12": 40., "phi12": -0.7,
                  "sample_tilt_mrad": (-9., 5.), "thickness": 180.}
        params = _params(prepared, **sample)
        coefficients = mx.zeros((14, 2))
        dc = mx.array([prepared.dc_value], dtype=mx.complex64)
        probe = up._alias_probe_kernel()(
            inputs=[prepared.kx, prepared.ky, params, coefficients],
            template=[("K", stored), ("HIGHER", False)],
            grid=(stored, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(stored, 5)], output_dtypes=[mx.float32],
        )[0]
        total = mx.zeros((n, n), dtype=mx.complex64)
        for start in range(0, stored, chunk):
            stop = min(start + chunk, stored)
            corrected = up._alias_correction_kernel()(
                inputs=[prepared.g_qk[start:stop], qr, qc,
                        prepared.kx[start:stop], prepared.ky[start:stop],
                        params, dc, probe[start:stop], coefficients],
                template=[("N", n), ("SOURCE_N", native),
                          ("G_COLS", prepared.g_qk.shape[-1]),
                          ("K", stop - start), ("HIGHER", False)],
                grid=(n, n, stop - start), threadgroup=(32, 8, 1),
                output_shapes=[(stop - start, n, n)], output_dtypes=[mx.complex64],
            )[0]
            total = total + mx.sum(corrected, axis=0)
            mx.eval(total)
        total[0, 0] += (prepared.num_bright_field_pixels - stored) * dc[0]
        wave = mx.fft.ifft2(total / prepared.num_bright_field_pixels)
        expected = mx.arctan2(mx.imag(wave), mx.real(wave))
        mx.eval(expected)
        actual = up._reconstruct_upsampled(prepared, **sample, upsampling_factor=factor)
        # Existing MPS coherent-object tolerance; no reduced support or precision.
        np.testing.assert_allclose(actual, np.asarray(expected), rtol=0, atol=1e-7)


@pytest.mark.parametrize("native", [128, 256, 512, 1024])
def test_full_phase_matches_dense_fourier_support(monkeypatch, native):
    """Fine-grid tilt previews retain the full phase when zero support is omitted."""
    from quantem.gpu.direct_ptycho.mps import upsampling as up
    from .test_ssb_upsampling import _disk_session

    coefs = {"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.}
    # Odd and even aberration orders exercise the complete corrected spectrum.
    if native == 256:
        coefs.update(C21_nm=3., phi21_deg=25., C30_nm=20.)
    sample = {"sample_tilt_mrad": (-9., 5.), "depth_spread_nm": 18.}
    with _disk_session(native, detector_sampling=(8., 12.)) as ssb:
        compact, loss = ssb.preview(
            coefs, **sample, upsampling_factor=2, compute_loss=False,
        )
        monkeypatch.setattr(
            up, "_coherent_support",
            lambda prepared, n, columns: (np.arange(n, dtype=np.uint32),
                                          np.arange(columns, dtype=np.uint32)),
        )
        # Rebuild the independent dense-support oracle rather than reusing
        # the compact geometry retained by the previous slider evaluation.
        ssb._backend()._prepared.upsampled_geometry = None
        dense, dense_loss = ssb.preview(
            coefs, **sample, upsampling_factor=2, compute_loss=False,
        )
        np.testing.assert_array_equal(compact, dense)
        assert loss is dense_loss is None
