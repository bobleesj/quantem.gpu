from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import pytest

STEPH_MASTER_CANDIDATES = (
    Path("/home/owner/data/steph/251115_ncem_arina_steph/lamella_2_005_master.h5"),
    Path("/home/owner/ssd/data/steph/20251115_ncem_arina/lamella_2_005_master.h5"),
)


def _cupy():
    return pytest.importorskip("cupy")


def _steph_master() -> Path:
    for path in STEPH_MASTER_CANDIDATES:
        if path.exists():
            return path
    pytest.skip("Steph real-data master is not available on this host.")


def _clean_gpu() -> None:
    cp = _cupy()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _geometry(cp, dx, dy, wavelength, semiangle_rad, ang_y_rad, ang_x_rad):
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    r = cp.sqrt(r2)
    alpha = r * np.float32(wavelength)
    alpha2 = alpha * alpha
    inv_r2 = cp.where(r2 > np.float32(1e-30), np.float32(1.0) / r2, np.float32(0.0))
    cos2 = (dx2 - dy2) * inv_r2
    sin2 = np.float32(2.0) * dx * dy * inv_r2
    denom_num2 = (dx * np.float32(ang_y_rad)) ** 2 + (dy * np.float32(ang_x_rad)) ** 2
    inv_r = cp.where(r > np.float32(1e-15), np.float32(1.0) / r, np.float32(0.0))
    denom = cp.sqrt(denom_num2) * inv_r
    edge = cp.where(
        denom > np.float32(1e-15),
        (np.float32(semiangle_rad) - alpha) / denom + np.float32(0.5),
        np.float32(1.0),
    )
    aperture = cp.clip(edge, np.float32(0.0), np.float32(1.0))
    return alpha2, cos2, sin2, aperture


def _reference_phase_loss(accel, c10: float, c12: float, phi12: float):
    cp = _cupy()
    c = accel._cache
    qx = c["qx_1d"][None, :, None]
    qy = c["qy_1d"][None, None, :]
    kx = c["kx_bf"][:, None, None]
    ky = c["ky_bf"][:, None, None]
    cos2phi12 = np.float32(math.cos(2.0 * phi12))
    sin2phi12 = np.float32(math.sin(2.0 * phi12))
    alpha_k2 = c["alpha_k2_1d"]
    cos2_k = c["cos2phi_k_1d"]
    sin2_k = c["sin2phi_k_1d"]
    aperture_k = c["aperture_k_1d"]
    chi_k = np.float32(accel._factor) * alpha_k2 * (
        np.float32(c12) * (cos2_k * cos2phi12 + sin2_k * sin2phi12)
        + np.float32(c10)
    )
    pk = aperture_k * (cp.cos(chi_k) - 1j * cp.sin(chi_k))
    alpha_m2, cos2_m, sin2_m, ap_m = _geometry(
        cp,
        qx - kx,
        qy - ky,
        c["wavelength"],
        c["semiangle_rad"],
        c["ang_y_rad"],
        c["ang_x_rad"],
    )
    alpha_p2, cos2_p, sin2_p, ap_p = _geometry(
        cp,
        qx + kx,
        qy + ky,
        c["wavelength"],
        c["semiangle_rad"],
        c["ang_y_rad"],
        c["ang_x_rad"],
    )
    chi_m = np.float32(accel._factor) * alpha_m2 * (
        np.float32(c12) * (cos2_m * cos2phi12 + sin2_m * sin2phi12)
        + np.float32(c10)
    )
    chi_p = np.float32(accel._factor) * alpha_p2 * (
        np.float32(c12) * (cos2_p * cos2phi12 + sin2_p * sin2phi12)
        + np.float32(c10)
    )
    pm = ap_m * (cp.cos(chi_m) - 1j * cp.sin(chi_m))
    pp = ap_p * (cp.cos(chi_p) - 1j * cp.sin(chi_p))
    pk3 = pk[:, None, None]
    gamma = pm * cp.conj(pk3) - cp.conj(pp) * pk3
    gamma_mag_sq = gamma.real * gamma.real + gamma.imag * gamma.imag
    gamma = gamma * cp.where(
        gamma_mag_sq > np.float32(1e-16),
        np.float32(1.0) / cp.sqrt(gamma_mag_sq),
        np.float32(1e8),
    )
    corrected = accel.G_qk * cp.conj(gamma)
    corrected[:, 0, 0] = cp.complex64(accel._dc_value_host)
    obj = cp.fft.ifft2(corrected)
    angles = cp.angle(obj)
    phase = angles.mean(axis=0).astype(cp.float32)
    loss = cp.mean((angles * angles).mean(axis=0) - phase * phase)
    return phase, float(loss)


def _reference_phase_loss_chunked(
    accel,
    c10: float,
    c12: float,
    phi12: float,
    *,
    chunk_bf: int = 512,
):
    cp = _cupy()
    c = accel._cache
    qx = c["qx_1d"][None, :, None]
    qy = c["qy_1d"][None, None, :]
    cos2phi12 = np.float32(math.cos(2.0 * phi12))
    sin2phi12 = np.float32(math.sin(2.0 * phi12))
    phase_sum = cp.zeros((int(c["ny"]), int(c["nx"])), dtype=cp.float32)
    phase_sumsq = cp.zeros_like(phase_sum)
    num_bf = int(c["num_bf"])

    for start in range(0, num_bf, chunk_bf):
        end = min(num_bf, start + chunk_bf)
        kx = c["kx_bf"][start:end, None, None]
        ky = c["ky_bf"][start:end, None, None]
        alpha_k2 = c["alpha_k2_1d"][start:end]
        cos2_k = c["cos2phi_k_1d"][start:end]
        sin2_k = c["sin2phi_k_1d"][start:end]
        aperture_k = c["aperture_k_1d"][start:end]
        chi_k = np.float32(accel._factor) * alpha_k2 * (
            np.float32(c12) * (cos2_k * cos2phi12 + sin2_k * sin2phi12)
            + np.float32(c10)
        )
        pk = aperture_k * (cp.cos(chi_k) - 1j * cp.sin(chi_k))
        alpha_m2, cos2_m, sin2_m, ap_m = _geometry(
            cp,
            qx - kx,
            qy - ky,
            c["wavelength"],
            c["semiangle_rad"],
            c["ang_y_rad"],
            c["ang_x_rad"],
        )
        alpha_p2, cos2_p, sin2_p, ap_p = _geometry(
            cp,
            qx + kx,
            qy + ky,
            c["wavelength"],
            c["semiangle_rad"],
            c["ang_y_rad"],
            c["ang_x_rad"],
        )
        chi_m = np.float32(accel._factor) * alpha_m2 * (
            np.float32(c12) * (cos2_m * cos2phi12 + sin2_m * sin2phi12)
            + np.float32(c10)
        )
        chi_p = np.float32(accel._factor) * alpha_p2 * (
            np.float32(c12) * (cos2_p * cos2phi12 + sin2_p * sin2phi12)
            + np.float32(c10)
        )
        pm = ap_m * (cp.cos(chi_m) - 1j * cp.sin(chi_m))
        pp = ap_p * (cp.cos(chi_p) - 1j * cp.sin(chi_p))
        pk3 = pk[:, None, None]
        gamma = pm * cp.conj(pk3) - cp.conj(pp) * pk3
        gamma_mag_sq = gamma.real * gamma.real + gamma.imag * gamma.imag
        gamma = gamma * cp.where(
            gamma_mag_sq > np.float32(1e-16),
            np.float32(1.0) / cp.sqrt(gamma_mag_sq),
            np.float32(1e8),
        )
        corrected = accel.G_qk[start:end] * cp.conj(gamma)
        corrected[:, 0, 0] = cp.complex64(accel._dc_value_host)
        angles = cp.angle(cp.fft.ifft2(corrected))
        phase_sum += angles.sum(axis=0)
        phase_sumsq += (angles * angles).sum(axis=0)

    phase = phase_sum / float(num_bf)
    loss = cp.mean(phase_sumsq / float(num_bf) - phase * phase)
    return phase.astype(cp.float32, copy=False), float(loss)


def _make_engine(size: int = 128, num_bf: int = 7):
    cp = _cupy()
    from quantem.gpu.ssb.engine import SSBEngine

    rng = np.random.default_rng(1234)
    real = rng.standard_normal((num_bf, size, size), dtype=np.float32)
    imag = rng.standard_normal((num_bf, size, size), dtype=np.float32)
    g_qk = cp.asarray(real + 1j * imag, dtype=cp.complex64)
    row_pattern = np.asarray([13, 14, 15, 16, 17, 16, 15], dtype=np.int32)
    col_pattern = np.asarray([14, 15, 16, 17, 16, 15, 14], dtype=np.int32)
    bf_inds_row = cp.asarray(np.resize(row_pattern, num_bf), dtype=cp.int32)
    bf_inds_col = cp.asarray(np.resize(col_pattern, num_bf), dtype=cp.int32)
    q = cp.fft.fftfreq(size, d=0.5).astype(cp.float32)
    q_row, q_col = cp.meshgrid(q, q, indexing="ij")
    engine = SSBEngine(
        G_qk=g_qk,
        bf_inds_row=bf_inds_row,
        bf_inds_col=bf_inds_col,
        bf_center=(15.5, 15.5),
        dc_value=complex(g_qk[:, 0, 0].mean().get()),
        gpts=(32, 32),
        sampling=(1.0, 1.0),
        q_row=q_row,
        q_col=q_col,
        wavelength=0.0197,
        semiangle_cutoff=21.4,
        angular_sampling=(1.0, 1.0),
    )
    engine.cache_rotation(0.0)
    return engine


def test_cuda_128_rejects_mismatched_bf_count() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.engine import SSBEngine

    q = cp.fft.fftfreq(128, d=0.5).astype(cp.float32)
    q_row, q_col = cp.meshgrid(q, q, indexing="ij")
    with pytest.raises(ValueError, match="G_qk first dimension"):
        SSBEngine(
            G_qk=cp.zeros((8, 128, 128), dtype=cp.complex64),
            bf_inds_row=cp.arange(7, dtype=cp.int32),
            bf_inds_col=cp.arange(7, dtype=cp.int32),
            bf_center=(3.0, 3.0),
            dc_value=0.0,
            gpts=(16, 16),
            sampling=(1.0, 1.0),
            q_row=q_row,
            q_col=q_col,
            wavelength=0.0197,
            semiangle_cutoff=21.4,
            angular_sampling=(1.0, 1.0),
        )


def test_cuda_128_engine_matches_explicit_cupy_reference() -> None:
    cp = _cupy()
    engine = _make_engine()

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    phase, loss = engine.reconstruct_with_loss(c10, c12, phi12)
    ref_phase, ref_loss = _reference_phase_loss(engine, c10, c12, phi12)

    cp.testing.assert_allclose(phase, ref_phase, rtol=2e-4, atol=2e-4)
    assert loss == pytest.approx(ref_loss, rel=2e-6, abs=2e-6)


def test_cuda_1024_engine_matches_explicit_cupy_reference() -> None:
    cp = _cupy()
    engine = _make_engine(size=1024, num_bf=3)

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    phase, loss = engine.reconstruct_with_loss(c10, c12, phi12)
    ref_phase, ref_loss = _reference_phase_loss_chunked(
        engine, c10, c12, phi12, chunk_bf=1
    )

    # A tiny number of pixels can cross the atan2 branch cut differently
    # between the fixed-size CUDA IFFT and cuFFT, shifting the arithmetic mean
    # by 2π / num_bf.  The scalar objective and essentially all pixels should
    # still match the explicit reference.
    phase_abs_err = cp.abs(phase - ref_phase)
    assert float(cp.percentile(phase_abs_err, 99.9)) < 3e-4
    assert loss == pytest.approx(ref_loss, rel=1e-4, abs=1e-4)

    phase_sum_only = engine._reconstruct_chunked(c10, c12, phi12)
    sum_only_abs_err = cp.abs(phase_sum_only - ref_phase)
    assert float(cp.percentile(sum_only_abs_err, 99.9)) < 3e-4


@pytest.mark.parametrize("size,num_bf", [(128, 7), (256, 5), (512, 5), (1024, 3)])
def test_cuda_fourier_sum_object_matches_chunked_ifft(size: int, num_bf: int) -> None:
    cp = _cupy()
    engine = _make_engine(size=size, num_bf=num_bf)

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    old_obj = engine._run_correction_pipeline_chunked(c10, c12, phi12, chunk_bf=1)
    new_obj = engine._reconstruct_object_fourier_sum(c10, c12, phi12)

    abs_err = cp.abs(old_obj - new_obj)
    rel_err = abs_err / cp.maximum(cp.abs(old_obj), cp.float32(1e-6))
    assert float(cp.percentile(abs_err, 99.9)) < 5e-9
    assert float(cp.percentile(rel_err, 99.9)) < 1e-4


def test_cuda_128_variance_loss_batch_matches_reference() -> None:
    cp = _cupy()
    engine = _make_engine()
    c10 = np.asarray([-120.0, -80.0, 20.0, 100.0], dtype=np.float32)
    c12 = np.asarray([55.0, 30.0, 40.0, 10.0], dtype=np.float32)
    phi = np.radians(np.asarray([17.0, -5.0, 11.0, 43.0], dtype=np.float32))

    got = engine.variance_loss_batch(c10, c12, phi)
    expected = []
    for a, b, c in zip(c10, c12, phi):
        _phase, loss = _reference_phase_loss(engine, float(a), float(b), float(c))
        expected.append(loss)

    cp.testing.assert_allclose(got, cp.asarray(expected, dtype=cp.float32), rtol=2e-6, atol=2e-6)


def test_cuda_256_variance_kernel_matches_staged_cupy_reference() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.engine import _pk_kernel

    engine = _make_engine(size=256, num_bf=37)
    c = engine._cache
    batch = 4
    num_bf = int(c["num_bf"])
    n = int(c["ny"])
    c10 = np.asarray([-120.0, -80.0, 20.0, 100.0], dtype=np.float32)
    c12 = np.asarray([55.0, 30.0, 40.0, 10.0], dtype=np.float32)
    phi = np.radians(np.asarray([17.0, -5.0, 11.0, 43.0], dtype=np.float32))
    cos2phi = np.cos(2.0 * phi).astype(np.float32)
    sin2phi = np.sin(2.0 * phi).astype(np.float32)
    c10_gpu = cp.asarray(c10)
    c12_gpu = cp.asarray(c12)
    cos_gpu = cp.asarray(cos2phi)
    sin_gpu = cp.asarray(sin2phi)
    pk = cp.empty((batch, num_bf), dtype=cp.complex64)
    _pk_kernel(
        c["alpha_k2_1d"][None, :],
        c["cos2phi_k_1d"][None, :],
        c["sin2phi_k_1d"][None, :],
        c["aperture_k_1d"][None, :],
        c10_gpu[:, None],
        c12_gpu[:, None],
        cos_gpu[:, None],
        sin_gpu[:, None],
        cp.float32(engine._factor),
        pk,
    )
    staged = cp.empty((batch, num_bf, n, n), dtype=cp.complex64)
    engine._custom_fft._rows_fused_pk_batch_quad_transpose(
        (1, engine._custom_fft._rows_fused_pk_grid_y_quad, num_bf),
        engine._custom_fft._rows_fused_pk_block_quad,
        (
            c["kx_bf"],
            c["ky_bf"],
            c["qx_1d"],
            c["qy_1d"],
            np.float32(c["wavelength"]),
            np.float32(c["semiangle_rad"]),
            np.float32(c["ang_y_rad"]),
            np.float32(c["ang_x_rad"]),
            c10_gpu,
            c12_gpu,
            cos_gpu,
            sin_gpu,
            np.float32(engine._factor),
            pk,
            engine.G_qk,
            staged,
            np.float32(engine._dc_value_host.real),
            np.float32(engine._dc_value_host.imag),
            np.int32(num_bf),
            np.int32(batch),
        ),
    )
    sum_buf = cp.zeros((batch, n, n), dtype=cp.float32)
    sumsq_buf = cp.zeros_like(sum_buf)
    groups = (num_bf + engine._custom_fft._colvar_group - 1) // engine._custom_fft._colvar_group
    engine._custom_fft._rows_var_batch(
        (1, engine._custom_fft._rows_var_grid_y, groups * batch),
        engine._custom_fft._rows_var_block,
        (
            staged,
            sum_buf,
            sumsq_buf,
            np.int32(num_bf),
            np.int32(batch),
            np.float32(1.0 / (n * n)),
            np.int32(0),
        ),
    )
    got = cp.mean(sumsq_buf / np.float32(num_bf) - (sum_buf / np.float32(num_bf)) ** 2, axis=(1, 2))
    expected = cp.var(cp.angle(cp.fft.ifft(staged, axis=-1)), axis=1).mean(axis=(1, 2))

    cp.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)


def test_cuda_128_real_steph_crop_matches_explicit_cupy_reference() -> None:
    cp = _cupy()
    from quantem.gpu.io.hdf5 import load
    from quantem.gpu.ssb import SSB

    _clean_gpu()
    path = _steph_master()
    loaded = load(
        path,
        scan_region=(64, 192, 64, 192),
        backend="cuda",
        verbose=False,
    )
    ssb = SSB(
        loaded.data,
        scan_shape=(128, 128),
        voltage_kV=300,
        semiangle=21.9,
        scan_sampling=0.5,
        rotation_angle_deg=0.0,
    )
    accel = ssb._get_accelerator()
    accel.cache_rotation(0.0)
    assert accel._custom_fft._size == 128

    args = (-134.94, 37.08, math.radians(-4.73))
    phase, loss = accel.reconstruct_with_loss(*args)
    ref_phase, ref_loss = _reference_phase_loss_chunked(accel, *args)

    assert loss == pytest.approx(ref_loss, rel=2e-6, abs=2e-6)
    cp.testing.assert_allclose(phase, ref_phase, rtol=2e-4, atol=2e-4)
    del ssb, loaded, phase, ref_phase
    _clean_gpu()


def test_ssb_roi96_auto_pads_to_128_not_256() -> None:
    cp = _cupy()
    from quantem.gpu.ssb import SSB

    rng = np.random.default_rng(5)
    data = rng.poisson(4.0, size=(96, 96, 16, 16)).astype(np.uint16)
    yy, xx = np.ogrid[:16, :16]
    bf = (yy - 8) ** 2 + (xx - 8) ** 2 <= 4 ** 2
    data[..., bf] += 100

    ssb = SSB(
        cp.asarray(data),
        voltage_kV=300,
        semiangle=21.4,
        scan_sampling=0.5,
        det_sampling=1.0,
        bf_radius=3,
    )

    assert ssb._scan_shape == (128, 128)
    accel = ssb._get_accelerator()
    accel.cache_rotation(0.0)
    assert accel._custom_fft._size == 128
