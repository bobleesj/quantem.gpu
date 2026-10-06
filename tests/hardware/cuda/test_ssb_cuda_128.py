import gc
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tests.parity.ssb_precision import (
    LOSS_ATOL,
    LOSS_RTOL,
    PHASE_ATOL,
    PHASE_RTOL,
    PRECISION,
)

REALDATA_MASTER_ENV = "QUANTEM_GPU_SSB_MASTER"


def _cupy():
    return pytest.importorskip("cupy")


def _realdata_master() -> Path:
    raw_path = os.environ.get(REALDATA_MASTER_ENV)
    if not raw_path:
        pytest.skip(f"{REALDATA_MASTER_ENV} is not set.")
    path = Path(raw_path).expanduser()
    if not path.exists():
        pytest.skip(f"{REALDATA_MASTER_ENV} does not point to an existing file.")
    return path


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
    g_qk = accel.G_qk
    if int(g_qk.shape[2]) != int(c["nx"]):
        g_qk = _expand_hermitian_cp(g_qk)
    corrected = g_qk * cp.conj(gamma)
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
        g_qk = accel.G_qk[start:end]
        if int(g_qk.shape[2]) != int(c["nx"]):
            g_qk = _expand_hermitian_cp(g_qk)
        corrected = g_qk * cp.conj(gamma)
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
    from quantem.gpu.ssb.cuda.engine import SSBEngine

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


def test_ssb_engine_restores_subset_and_writes_exact_source(tmp_path):
    """CUDA internals own temporary BF subsets and exact source writing."""

    engine = _make_engine(size=128, num_bf=7)
    assert engine.scan_shape == (128, 128)
    assert engine.detector_shape == (32, 32)

    state = engine.export_state()
    assert state.scan_shape == (128, 128)
    assert state.brightfield.detector_shape == (32, 32)
    # the radius the calibration implies, semiangle / det_sampling = 21.4 mrad / 1 mrad per pixel (42.8 before
    # 2026-10-05, when the automatic calibration used twice the semiangle)
    assert state.brightfield.detected_radius_px == pytest.approx(21.4)

    subset = engine.prepare_bf_subset(3)
    with subset:
        assert engine.num_bf == 3
    assert engine.num_bf == 7
    subset.close()

    cp = _cupy()
    data = cp.arange(128 * 128 * 32 * 32, dtype=cp.uint8).reshape(
        128, 128, 32, 32
    )
    source_path = engine.write_exact_bf_source(
        data,
        tmp_path / "exact_bf_columns",
    )
    from quantem.gpu.formats.qem.reference import load_array

    counts, _ = load_array(source_path)
    columns = counts.reshape(128 * 128, 7).T
    assert source_path.suffix == ".qem"
    rows = cp.asnumpy(engine.bf_inds_row)
    cols = cp.asnumpy(engine.bf_inds_col)
    expected = cp.asnumpy(data.reshape(-1, 32, 32)[:, rows, cols].T)
    np.testing.assert_array_equal(columns, expected)
    assert engine.export_state().bf_source_path == source_path


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
    from quantem.gpu.ssb.cuda.engine import SSBEngine

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


def test_cuda_engine_uses_shared_float32_precision_contract() -> None:
    """CUDA advertises the shared numeric storage contract."""
    engine = _make_engine()

    assert engine.precision == PRECISION


def test_cuda_128_engine_matches_explicit_cupy_reference() -> None:
    cp = _cupy()
    engine = _make_engine()

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    phase, loss = engine.reconstruct_with_loss(c10, c12, phi12)
    ref_phase, ref_loss = _reference_phase_loss(engine, c10, c12, phi12)

    cp.testing.assert_allclose(
        phase,
        ref_phase,
        rtol=PHASE_RTOL,
        atol=PHASE_ATOL,
    )
    assert loss == pytest.approx(
        ref_loss,
        rel=LOSS_RTOL,
        abs=LOSS_ATOL,
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

    phase_sum_only = engine._fused_chunked_core(c10, c12, phi12, compute_loss=False)
    sum_only_abs_err = cp.abs(phase_sum_only - ref_phase)
    assert float(cp.percentile(sum_only_abs_err, 99.9)) < 3e-4


def test_cuda_512_subpixel_bf_center_matches_explicit_reference() -> None:
    cp = _cupy()
    engine = _make_engine(size=512, num_bf=6, bf_center=(15.3, 15.7))

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    phase, loss = engine.reconstruct_with_loss(c10, c12, phi12)
    ref_phase, ref_loss = _reference_phase_loss_chunked(
        engine, c10, c12, phi12, chunk_bf=2
    )

    phase_abs_err = cp.abs(phase - ref_phase)
    assert float(cp.percentile(phase_abs_err, 99.9)) < 3e-4
    assert loss == pytest.approx(ref_loss, rel=1e-4, abs=1e-4)


def test_cuda_512_chunked_odd_bf_count_matches_explicit_reference() -> None:
    engine = _make_engine(size=512, num_bf=5, bf_center=(15.3, 15.7))

    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    _phase, loss = engine._fused_chunked_core(c10, c12, phi12, compute_loss=True)
    _ref_phase, ref_loss = _reference_phase_loss_chunked(
        engine, c10, c12, phi12, chunk_bf=2
    )

    # With only five BF pixels, arithmetic mean phase is dominated by atan2
    # branch-cut choices, so only the loss is compared.
    assert loss == pytest.approx(ref_loss, rel=1e-4, abs=1e-4)


def test_cuda_optimizer_chunk_reduction_is_repeatable() -> None:
    """Fixed-order optimizer feedback returns identical loss bits."""
    cp = _cupy()
    engine = _make_engine(size=128, num_bf=17)
    args = (-120.0, 55.0, math.radians(17.0))

    phases = []
    losses = []
    for _ in range(4):
        phase, loss = engine._fused_chunked_core(*args, compute_loss=True, chunk_bf=5)
        phases.append(cp.asnumpy(phase))
        losses.append(loss)

    for phase in phases[1:]:
        np.testing.assert_array_equal(phase, phases[0])
    assert all(loss == losses[0] for loss in losses[1:])


@pytest.mark.slow
def test_cuda_512_chunked_preview_is_repeatable() -> None:
    """The chunked path of scans over 6 GB of corrected planes gives the same phase and loss bits on every call.

    Before 2026-10-05 its previews summed the 512 column groups with atomic adds: the phase changed in its last bits
    from call to call, and a preview's loss could differ from the fit's at the same aberrations.
    """
    cp = _cupy()
    engine = _make_engine(size=512, num_bf=256, bf_center=(15.3, 15.7))
    args = (-120.0, 55.0, math.radians(17.0))

    phase_only = [cp.asnumpy(engine._fused_chunked_core(*args)) for _ in range(5)]
    with_loss = [engine._fused_chunked_core(*args, compute_loss=True) for _ in range(5)]

    for phase in phase_only[1:]:
        np.testing.assert_array_equal(phase, phase_only[0])
    for phase, loss in with_loss:
        np.testing.assert_array_equal(cp.asnumpy(phase), phase_only[0])
        assert loss == with_loss[0][1]


@pytest.mark.slow
def test_result_loss_is_the_exact_objective_with_or_without_a_fit() -> None:
    """A reconstruction's loss is the exact phase-variance objective at its aberrations, whether or not a fit ran.

    Before 2026-10-05 a session that had not fit reported the batched optimizer evaluator, which on 256 and 1024 scans
    reduces only a subset of the scan rows: 0.0578 here against the exact 0.1427, which the same reconstruction reported
    once any fit had switched the objective over.
    """
    cp = _cupy()
    from quantem.gpu import SSB
    from quantem.gpu.ssb.units import aberrations_to_engine

    counts = np.random.RandomState(3).poisson(5, (256, 256, 16, 16)).astype(np.uint16)
    aberrations = {"C10": -8.0, "C12": 3.0, "phi12": 0.4}
    with SSB(counts, backend="cuda", voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5,
             det_sampling=2.0, bf_center=(7.5, 7.5), bf_radius=5) as session:
        before = session.reconstruct(aberrations=aberrations, phase_estimator="complex_wave").loss
        session.find_aberrations(trials=4, refinement=None, check_rotation=False, verbose=False)
        after = session.reconstruct(aberrations=aberrations, phase_estimator="complex_wave", force=True).loss
        engine = session._prepare_cuda()._get_accelerator()
        engine_aberrations = aberrations_to_engine(aberrations)
        _, exact = engine.reconstruct_with_loss(
            engine_aberrations["C10"], engine_aberrations["C12"], engine_aberrations["phi12"],
        )
        cp.cuda.Device().synchronize()
    assert before == after == exact


def test_cuda_512_chunked_calibration_matches_unfused_gpu_reference() -> None:
    """Full-BF phase and loss retain scan order across a partial final chunk."""
    cp = _cupy()
    from quantem.gpu import SSB

    # Measured intensities are real and nonnegative. Give every BF pixel its
    # own position; duplicated coordinates invalidate the conjugate-pair path.
    counts = cp.random.RandomState(7).poisson(5, (512, 512, 16, 16)).astype(cp.uint16)
    with SSB(
        counts, backend="cuda", voltage_kV=300, semiangle_mrad=5,
        scan_sampling_A=0.5, det_sampling=1,
        bf_center=(7.5, 7.5), bf_radius=5,
    ) as ssb:
        backend = ssb._prepare_cuda()
        engine = backend._get_accelerator()
        engine.cache_rotation(backend._rotation_angle_rad)
        assert engine.num_bf == 80  # 64-BF chunk, then a 16-BF tail.
        args = (-120.0, 55.0, math.radians(17.0))
        expected_phase, expected_loss = _reference_phase_loss_chunked(
            engine, *args, chunk_bf=32,
        )
        first_phase = first_loss = None
        for _ in range(3):
            phase, loss = engine._fused_chunked_core(
                *args, compute_loss=True,
            )
            cp.testing.assert_allclose(
                phase, expected_phase, rtol=PHASE_RTOL, atol=PHASE_ATOL,
            )
            assert loss == pytest.approx(expected_loss, rel=LOSS_RTOL, abs=LOSS_ATOL)
            if first_phase is None:
                first_phase, first_loss = phase.copy(), loss
            else:
                cp.testing.assert_array_equal(phase, first_phase)
                assert loss == first_loss


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
    from quantem.gpu.ssb.cuda.brightfield import bright_field_spectra

    data = cp.arange(8 * 8 * 6 * 6, dtype=cp.uint16).reshape(8, 8, 6, 6)
    bf_rows = cp.asarray([2, 2, 3, 3], dtype=cp.int32)
    bf_cols = cp.asarray([2, 3, 2, 3], dtype=cp.int32)

    herm_gqk, herm_dc = bright_field_spectra(
        data,
        bf_rows,
        bf_cols,
        (8, 8),
        (6, 6),
    )
    full_gqk = _expand_hermitian_cp(herm_gqk)

    assert full_gqk.shape == (4, 8, 8)
    assert herm_gqk.shape == (4, 8, 5)
    assert herm_gqk.nbytes < full_gqk.nbytes
    cp.testing.assert_allclose(herm_gqk, full_gqk[:, :, :5])
    assert herm_dc == pytest.approx(complex(full_gqk[:, 0, 0].mean().get()))


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
        herm_engine.objective.loss_batch(c10_batch, c12_batch, phi_batch),
        full_engine.objective.loss_batch(c10_batch, c12_batch, phi_batch),
        rtol=LOSS_RTOL,
        atol=LOSS_ATOL,
    )


def test_ssb_default_hermitian_result_matches_full_storage_end_to_end() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.cuda.backend import CudaSSBBackend

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

    herm = CudaSSBBackend(cp.asarray(data), **kwargs)
    assert herm.G_qk.shape[2] == 65
    assert herm.G_qk.nbytes == len(herm.bf_inds_row) * 128 * 65 * 8

    herm_result = herm.result()
    from quantem.gpu.ssb.contract import SSBProtocol

    assert isinstance(herm, SSBProtocol)
    assert herm_result.backend == "cuda"
    assert herm_result.num_bf == len(herm.bf_inds_row)
    assert herm_result.aberrations == herm.aberrations
    accel = herm._get_accelerator()
    full_engine = _make_engine(
        size=128,
        num_bf=len(herm.bf_inds_row),
        g_qk=_expand_hermitian_cp(herm.G_qk),
        bf_center=herm.bf_center,
    )
    full_engine.bf_inds_row = herm.bf_inds_row
    full_engine.bf_inds_col = herm.bf_inds_col
    full_engine.q_row = herm.q_row
    full_engine.q_col = herm.q_col
    full_engine.gpts = herm.gpts
    full_engine.sampling = herm.sampling
    full_engine.wavelength = herm.wavelength
    full_engine.semiangle_cutoff = herm.semiangle_cutoff
    full_engine.angular_sampling = herm.angular_sampling
    full_engine._factor = accel._factor
    full_engine.cache_rotation(herm._rotation_angle_rad, force=True)
    args = (
        herm.aberrations["C10"],
        herm.aberrations["C12"],
        herm.aberrations["phi12"],
    )
    full_obj = full_engine.reconstruct_object(*args)
    _full_phase, full_loss = full_engine.reconstruct_with_loss(*args)
    abs_err = cp.abs(herm_result.object_wave - full_obj)
    rel_err = abs_err / cp.maximum(cp.abs(full_obj), cp.float32(1e-6))
    assert float(cp.percentile(abs_err, 99.9)) < 1e-7
    assert float(cp.percentile(rel_err, 99.9)) < 1e-4
    assert herm_result.loss is not None
    assert herm_result.loss == pytest.approx(full_loss, rel=1e-4, abs=1e-4)


def test_ssb_hermitian_storage_preserves_half_plane_for_phase_reconstruction() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.cuda.backend import CudaSSBBackend

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

    herm = CudaSSBBackend(cp.asarray(data), **kwargs)
    accel = herm._get_accelerator()
    herm_phase = accel.reconstruct(-120.0, 55.0, math.radians(17.0))
    full_engine = _make_engine(
        size=128,
        num_bf=len(herm.bf_inds_row),
        g_qk=_expand_hermitian_cp(herm.G_qk),
        bf_center=herm.bf_center,
    )
    full_engine.bf_inds_row = herm.bf_inds_row
    full_engine.bf_inds_col = herm.bf_inds_col
    full_engine.q_row = herm.q_row
    full_engine.q_col = herm.q_col
    full_engine.gpts = herm.gpts
    full_engine.sampling = herm.sampling
    full_engine.wavelength = herm.wavelength
    full_engine.semiangle_cutoff = herm.semiangle_cutoff
    full_engine.angular_sampling = herm.angular_sampling
    full_engine._factor = accel._factor
    full_engine.cache_rotation(herm._rotation_angle_rad, force=True)
    full_phase = full_engine.reconstruct(-120.0, 55.0, math.radians(17.0))

    assert herm.G_qk.shape[2] == 65
    phase_abs_err = cp.abs(herm_phase - full_phase)
    assert float(cp.percentile(phase_abs_err, 99.9)) < 3e-4


def test_ssb_default_hermitian_optimize_keeps_half_plane() -> None:
    cp = _cupy()
    pytest.importorskip("optuna")
    from quantem.gpu.ssb.cuda.backend import CudaSSBBackend

    rng = np.random.default_rng(125)
    data = rng.poisson(4.0, size=(128, 128, 16, 16)).astype(np.uint16)
    yy, xx = np.ogrid[:16, :16]
    bf = (yy - 8) ** 2 + (xx - 8) ** 2 <= 4 ** 2
    data[..., bf] += 80

    ssb = CudaSSBBackend(
        cp.asarray(data),
        voltage_kV=300,
        semiangle=21.4,
        scan_sampling=0.5,
        det_sampling=1.0,
        bf_radius=3,
    )
    before_nbytes = ssb.G_qk.nbytes
    ssb.optimize(n_trials=2, verbose=False)

    assert ssb.G_qk.shape[2] == 65
    assert ssb.G_qk.nbytes == before_nbytes


def test_cuda_128_variance_loss_batch_matches_reference() -> None:
    cp = _cupy()
    engine = _make_engine()
    c10 = np.asarray([-120.0, -80.0, 20.0, 100.0], dtype=np.float32)
    c12 = np.asarray([55.0, 30.0, 40.0, 10.0], dtype=np.float32)
    phi = np.radians(np.asarray([17.0, -5.0, 11.0, 43.0], dtype=np.float32))

    got = engine.objective.loss_batch(c10, c12, phi)
    expected = []
    for a, b, c in zip(c10, c12, phi):
        _phase, loss = _reference_phase_loss(engine, float(a), float(b), float(c))
        expected.append(loss)

    cp.testing.assert_allclose(
        got,
        cp.asarray(expected, dtype=cp.float32),
        rtol=LOSS_RTOL,
        atol=LOSS_ATOL,
    )


def test_cuda_128_realdata_crop_matches_explicit_cupy_reference() -> None:
    cp = _cupy()
    from quantem.gpu.io import load
    from quantem.gpu.ssb.cuda.backend import CudaSSBBackend

    _clean_gpu()
    path = _realdata_master()
    loaded = load(
        path,
        scan_region=(64, 192, 64, 192),
        backend="cuda",
        verbose=False,
    )
    ssb = CudaSSBBackend(
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

    assert loss == pytest.approx(
        ref_loss,
        rel=LOSS_RTOL,
        abs=LOSS_ATOL,
    )
    cp.testing.assert_allclose(phase, ref_phase, rtol=2e-4, atol=2e-4)
    del ssb, loaded, phase, ref_phase
    _clean_gpu()


def test_ssb_roi96_auto_pads_to_128_not_256() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.cuda.backend import CudaSSBBackend

    rng = np.random.default_rng(5)
    data = rng.poisson(4.0, size=(96, 96, 16, 16)).astype(np.uint16)
    yy, xx = np.ogrid[:16, :16]
    bf = (yy - 8) ** 2 + (xx - 8) ** 2 <= 4 ** 2
    data[..., bf] += 100

    ssb = CudaSSBBackend(
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


@pytest.mark.parametrize(("size", "num_bf"), [(128, 40), (256, 20), (512, 8), (1024, 3)])
@pytest.mark.parametrize("compute_loss", [True, False])
def test_cuda_chunked_phase_is_in_row_col_order(size: int, num_bf: int, compute_loss: bool) -> None:
    """The large-scan chunked core returns the phase in (row, col) order for every kernel size.

    The 512 kernels accumulate column-major; before 2026-09-26 their phase came back transposed (99th-percentile error
    2.3 rad against the reference, 1.5e-5 rad after a transpose), so 512 x 512 previews looked rotated next to results.
    """
    cp = _cupy()
    engine = _make_engine(size=size, num_bf=num_bf)
    c10, c12, phi12 = -120.0, 55.0, math.radians(17.0)
    ref_phase, _ = _reference_phase_loss_chunked(engine, c10, c12, phi12, chunk_bf=1)
    output = engine._fused_chunked_core(c10, c12, phi12, compute_loss=compute_loss)
    phase = output[0] if compute_loss else output
    # a few pixels can cross the atan2 branch cut differently; the 99th percentile is the orientation-sensitive check
    assert float(cp.percentile(cp.abs(phase - ref_phase), 99)) < 3e-4
