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

FLOAT32_PARITY_RTOL = 1e-5
FLOAT32_PARITY_ATOL = 1e-6


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


def _make_engine(
    size: int = 128,
    num_bf: int = 7,
    g_qk=None,
    bf_center: tuple[float, float] = (15.5, 15.5),
):
    cp = _cupy()
    from quantem.gpu.ssb.engine import SSBEngine

    if g_qk is None:
        rng = np.random.default_rng(1234)
        real = rng.standard_normal((num_bf, size, size), dtype=np.float32)
        imag = rng.standard_normal((num_bf, size, size), dtype=np.float32)
        g_qk = cp.asarray(real + 1j * imag, dtype=cp.complex64)
    else:
        g_qk = cp.asarray(g_qk, dtype=cp.complex64)
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
        bf_center=bf_center,
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


def _expand_hermitian_cp(half_gqk):
    cp = _cupy()
    num_bf, n, stored_cols = half_gqk.shape
    if stored_cols != n // 2 + 1:
        raise ValueError("half_gqk must have shape (bf, n, n//2 + 1)")
    full = cp.empty((num_bf, n, n), dtype=half_gqk.dtype)
    full[:, :, :stored_cols] = half_gqk
    mirror_rows = cp.asarray((-np.arange(n)) % n, dtype=cp.int32)
    mirror_cols = cp.asarray(np.arange(n - stored_cols, 0, -1), dtype=cp.int32)
    full[:, :, stored_cols:] = cp.conj(half_gqk[:, mirror_rows][:, :, mirror_cols])
    return full


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
    assert loss == pytest.approx(
        ref_loss,
        rel=FLOAT32_PARITY_RTOL,
        abs=FLOAT32_PARITY_ATOL,
    )


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


def test_cuda_512_subpixel_bf_dual_path_matches_explicit_reference() -> None:
    cp = _cupy()
    engine = _make_engine(size=512, num_bf=6, bf_center=(15.3, 15.7))

    cache = engine._cache
    assert int(cache["pair_a"].shape[0]) == 0
    assert int(cache["dual_pair_a"].shape[0]) == 3
    assert int(cache["dual_tail"].shape[0]) == 0

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    phase, loss = engine.reconstruct_with_loss(c10, c12, phi12)
    ref_phase, ref_loss = _reference_phase_loss_chunked(
        engine, c10, c12, phi12, chunk_bf=2
    )

    phase_abs_err = cp.abs(phase - ref_phase)
    assert float(cp.percentile(phase_abs_err, 99.9)) < 3e-4
    assert loss == pytest.approx(ref_loss, rel=1e-4, abs=1e-4)


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


@pytest.mark.parametrize("size,num_bf", [(128, 7), (256, 5), (512, 5), (1024, 3)])
def test_cuda_fourier_sum_object_accepts_hermitian_gqk(size: int, num_bf: int) -> None:
    cp = _cupy()
    rng = np.random.default_rng(4321)
    real_stack = cp.asarray(
        rng.standard_normal((num_bf, size, size), dtype=np.float32),
        dtype=cp.float32,
    )
    full_gqk = cp.fft.fft2(real_stack).astype(cp.complex64, copy=False)
    herm_gqk = cp.ascontiguousarray(full_gqk[:, :, : size // 2 + 1])
    sym_full_gqk = _expand_hermitian_cp(herm_gqk)
    full_engine = _make_engine(size=size, num_bf=num_bf, g_qk=sym_full_gqk)
    herm_engine = _make_engine(size=size, num_bf=num_bf, g_qk=herm_gqk)

    assert herm_engine.G_qk.shape == (num_bf, size, size // 2 + 1)
    assert herm_engine.G_qk.nbytes < full_engine.G_qk.nbytes

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    full_obj = full_engine._reconstruct_object_fourier_sum(c10, c12, phi12)
    herm_obj = herm_engine._reconstruct_object_fourier_sum(c10, c12, phi12)

    abs_err = cp.abs(full_obj - herm_obj)
    rel_err = abs_err / cp.maximum(cp.abs(full_obj), cp.float32(1e-6))
    assert float(cp.percentile(abs_err, 99.9)) < 5e-9
    assert float(cp.percentile(rel_err, 99.9)) < 1e-4

    full_phase = full_engine.reconstruct(c10, c12, phi12)
    herm_phase = herm_engine.reconstruct(c10, c12, phi12)
    phase_abs_err = cp.abs(full_phase - herm_phase)
    assert float(cp.percentile(phase_abs_err, 99.9)) < 3e-4


def test_extract_gqk_hermitian_storage_keeps_nonredundant_columns() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.reconstruction import SSB

    data = cp.arange(8 * 8 * 6 * 6, dtype=cp.uint16).reshape(8, 8, 6, 6)
    bf_rows = cp.asarray([2, 2, 3, 3], dtype=cp.int32)
    bf_cols = cp.asarray([2, 3, 2, 3], dtype=cp.int32)

    full_gqk, full_dc = SSB._extract_gqk(
        data,
        bf_rows,
        bf_cols,
        (8, 8),
        (6, 6),
        gqk_storage="full",
    )
    herm_gqk, herm_dc = SSB._extract_gqk(
        data,
        bf_rows,
        bf_cols,
        (8, 8),
        (6, 6),
        gqk_storage="herm",
    )

    assert full_gqk.shape == (4, 8, 8)
    assert herm_gqk.shape == (4, 8, 5)
    assert herm_gqk.nbytes < full_gqk.nbytes
    cp.testing.assert_allclose(herm_gqk, full_gqk[:, :, :5])
    assert herm_dc == pytest.approx(full_dc)


@pytest.mark.parametrize("size,num_bf", [(128, 7), (256, 5), (512, 5), (1024, 3)])
def test_cuda_phase_loss_accepts_hermitian_gqk(size: int, num_bf: int) -> None:
    cp = _cupy()
    rng = np.random.default_rng(5678)
    real_stack = cp.asarray(
        rng.standard_normal((num_bf, size, size), dtype=np.float32),
        dtype=cp.float32,
    )
    full_gqk = cp.fft.fft2(real_stack).astype(cp.complex64, copy=False)
    herm_gqk = cp.ascontiguousarray(full_gqk[:, :, : size // 2 + 1])
    sym_full_gqk = _expand_hermitian_cp(herm_gqk)
    full_engine = _make_engine(size=size, num_bf=num_bf, g_qk=sym_full_gqk)
    herm_engine = _make_engine(size=size, num_bf=num_bf, g_qk=herm_gqk)

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    full_phase, full_loss = full_engine.reconstruct_with_loss(c10, c12, phi12)
    herm_phase, herm_loss = herm_engine.reconstruct_with_loss(c10, c12, phi12)

    assert herm_engine.G_qk.shape == (num_bf, size, size // 2 + 1)
    phase_abs_err = cp.abs(full_phase - herm_phase)
    assert float(cp.percentile(phase_abs_err, 99.9)) < 3e-4
    assert herm_loss == pytest.approx(full_loss, rel=1e-4, abs=1e-4)

    c10_batch = np.asarray([-120.0, -80.0, 20.0, 100.0], dtype=np.float32)
    c12_batch = np.asarray([55.0, 30.0, 40.0, 10.0], dtype=np.float32)
    phi_batch = np.radians(np.asarray([17.0, -5.0, 11.0, 43.0], dtype=np.float32))
    cp.testing.assert_allclose(
        herm_engine.variance_loss_batch(c10_batch, c12_batch, phi_batch),
        full_engine.variance_loss_batch(c10_batch, c12_batch, phi_batch),
        rtol=FLOAT32_PARITY_RTOL,
        atol=FLOAT32_PARITY_ATOL,
    )


def test_ssb_default_hermitian_result_matches_full_storage_end_to_end() -> None:
    cp = _cupy()
    from quantem.gpu.ssb import SSB

    rng = np.random.default_rng(123)
    data = rng.poisson(4.0, size=(128, 128, 16, 16)).astype(np.uint16)
    yy, xx = np.ogrid[:16, :16]
    bf = (yy - 8) ** 2 + (xx - 8) ** 2 <= 4 ** 2
    data[..., bf] += 80
    kwargs = dict(
        voltage_kV=300,
        semiangle=21.4,
        scan_sampling=0.5,
        det_sampling=1.0,
        bf_radius=3,
        aberrations={"C10": -120.0, "C12": 55.0, "phi12": math.radians(17.0)},
    )

    herm = SSB(cp.asarray(data), **kwargs)
    full = SSB(cp.asarray(data), gqk_storage="full", **kwargs)

    assert herm.gqk_storage == "herm"
    assert herm.G_qk.shape[2] == 65
    assert full.G_qk.shape[2] == 128
    assert herm.G_qk.nbytes < full.G_qk.nbytes
    assert herm.G_qk.nbytes == len(herm.bf_inds_row) * 128 * 65 * 8
    assert full.G_qk.nbytes == len(full.bf_inds_row) * 128 * 128 * 8

    herm_result = herm.result()
    full_result = full.result()
    abs_err = cp.abs(herm_result.object_wave - full_result.object_wave)
    rel_err = abs_err / cp.maximum(cp.abs(full_result.object_wave), cp.float32(1e-6))
    assert float(cp.percentile(abs_err, 99.9)) < 1e-7
    assert float(cp.percentile(rel_err, 99.9)) < 1e-4
    assert herm_result.loss is not None
    assert full_result.loss is not None
    assert herm_result.loss == pytest.approx(full_result.loss, rel=1e-4, abs=1e-4)


def test_ssb_hermitian_storage_preserves_half_plane_for_phase_reconstruction() -> None:
    cp = _cupy()
    from quantem.gpu.ssb import SSB

    rng = np.random.default_rng(124)
    data = rng.poisson(4.0, size=(128, 128, 16, 16)).astype(np.uint16)
    yy, xx = np.ogrid[:16, :16]
    bf = (yy - 8) ** 2 + (xx - 8) ** 2 <= 4 ** 2
    data[..., bf] += 80
    kwargs = dict(
        voltage_kV=300,
        semiangle=21.4,
        scan_sampling=0.5,
        det_sampling=1.0,
        bf_radius=3,
    )

    herm = SSB(cp.asarray(data), **kwargs)
    full = SSB(cp.asarray(data), gqk_storage="full", **kwargs)

    herm_phase = herm._reconstruct(C10=-120.0, C12=55.0, phi12=math.radians(17.0))
    full_phase = full._reconstruct(C10=-120.0, C12=55.0, phi12=math.radians(17.0))

    assert herm.gqk_storage == "herm"
    assert herm.G_qk.shape[2] == 65
    assert herm.G_qk.nbytes < full.G_qk.nbytes
    phase_abs_err = cp.abs(herm_phase - full_phase)
    assert float(cp.percentile(phase_abs_err, 99.9)) < 3e-4


def test_ssb_default_hermitian_optimize_keeps_half_plane() -> None:
    cp = _cupy()
    pytest.importorskip("optuna")
    from quantem.gpu.ssb import SSB

    rng = np.random.default_rng(125)
    data = rng.poisson(4.0, size=(128, 128, 16, 16)).astype(np.uint16)
    yy, xx = np.ogrid[:16, :16]
    bf = (yy - 8) ** 2 + (xx - 8) ** 2 <= 4 ** 2
    data[..., bf] += 80

    ssb = SSB(
        cp.asarray(data),
        voltage_kV=300,
        semiangle=21.4,
        scan_sampling=0.5,
        det_sampling=1.0,
        bf_radius=3,
    )
    before_nbytes = ssb.G_qk.nbytes
    ssb.optimize(n_trials=2, verbose=False)

    assert ssb.gqk_storage == "herm"
    assert ssb.G_qk.shape[2] == 65
    assert ssb.G_qk.nbytes == before_nbytes


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

    cp.testing.assert_allclose(
        got,
        cp.asarray(expected, dtype=cp.float32),
        rtol=FLOAT32_PARITY_RTOL,
        atol=FLOAT32_PARITY_ATOL,
    )


def test_cuda_1024_variance_batch_uses_full_staging_buffers() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.engine import _pk_kernel, _variance_from_sums_batch_kernel
    from quantem.gpu.ssb.fft_common import CustomFFTBase

    engine = _make_engine(size=1024, num_bf=4)
    c = engine._cache
    batch = 4
    num_bf = int(c["num_bf"])
    n = int(c["ny"])
    c10 = np.asarray([-120.0, -80.0, 20.0, 100.0], dtype=np.float32)
    c12 = np.asarray([55.0, 30.0, 40.0, 10.0], dtype=np.float32)
    phi = np.radians(np.asarray([17.0, -5.0, 11.0, 43.0], dtype=np.float32))

    got = engine.variance_loss_batch(c10, c12, phi)
    assert any(key[3:] == (1024, 1024) for key in engine._streaming_cache)
    assert not any(key[3:] == (128, 1024) for key in engine._streaming_cache)

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
    data = cp.empty((batch, num_bf, n, n), dtype=cp.complex64)
    sum_buf = cp.zeros((batch, n, n), dtype=cp.float32)
    sumsq_buf = cp.zeros((batch, n, n), dtype=cp.float32)
    CustomFFTBase.ifft2_inplace_batch_fused_pk_variance(
        engine._custom_fft,
        data,
        engine.G_qk,
        c,
        pk,
        c10_gpu,
        c12_gpu,
        cos_gpu,
        sin_gpu,
        engine._factor,
        engine._dc_value_host,
        sum_buf,
        sumsq_buf,
        stream_bf=0,
    )
    variance = cp.empty((batch, n, n), dtype=cp.float32)
    _variance_from_sums_batch_kernel(
        ((batch * n * n + 255) // 256,),
        (256,),
        (
            sum_buf,
            sumsq_buf,
            variance,
            np.int32(num_bf),
            np.int32(n),
            np.int32(n),
            np.int32(batch),
        ),
    )
    expected = cp.mean(variance, axis=(1, 2))
    cp.testing.assert_allclose(
        got,
        expected,
        rtol=FLOAT32_PARITY_RTOL,
        atol=FLOAT32_PARITY_ATOL,
    )


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
            np.int32(engine.G_qk.shape[2]),
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

    cp.testing.assert_allclose(
        got,
        expected,
        rtol=FLOAT32_PARITY_RTOL,
        atol=FLOAT32_PARITY_ATOL,
    )


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
        gqk_storage="full",
    )
    accel = ssb._get_accelerator()
    accel.cache_rotation(0.0)
    assert accel._custom_fft._size == 128

    args = (-134.94, 37.08, math.radians(-4.73))
    phase, loss = accel.reconstruct_with_loss(*args)
    ref_phase, ref_loss = _reference_phase_loss_chunked(accel, *args)

    assert loss == pytest.approx(
        ref_loss,
        rel=FLOAT32_PARITY_RTOL,
        abs=FLOAT32_PARITY_ATOL,
    )
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
