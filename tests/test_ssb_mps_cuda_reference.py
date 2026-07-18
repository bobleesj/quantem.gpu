from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_mps_sparse_row_indices_match_cuda_supported_sizes() -> None:
    """Pin MPS sparse objective row masks to the CUDA optimizer kernels."""
    from quantem.gpu.ssb.mps import _cuda_sparse_row_indices

    rows_128 = _cuda_sparse_row_indices((128, 128))
    assert rows_128.shape == (128,)
    assert rows_128[:12].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert rows_128[-8:].tolist() == [120, 121, 122, 123, 124, 125, 126, 127]

    rows_256 = _cuda_sparse_row_indices((256, 256))
    assert rows_256.shape == (128,)
    assert rows_256[:12].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
    assert rows_256[-4:].tolist() == [248, 249, 250, 251]

    rows_512 = _cuda_sparse_row_indices((512, 512))
    assert rows_512.shape == (128,)
    assert rows_512[:10].tolist() == [0, 1, 8, 9, 16, 17, 24, 25, 32, 33]
    assert rows_512[-2:].tolist() == [504, 505]

    rows_1024 = _cuda_sparse_row_indices((1024, 1024))
    assert rows_1024.shape == (128,)
    assert rows_1024[:6].tolist() == [0, 8, 16, 24, 32, 40]
    assert rows_1024[-3:].tolist() == [1000, 1008, 1016]

    with pytest.raises(ValueError, match="128x128, 256x256, 512x512, or 1024x1024"):
        _cuda_sparse_row_indices((64, 64))


def test_mps_phase_loss_default_chunk_is_size_aware(monkeypatch) -> None:
    """MPS exact phase/loss default must avoid oversized 1024 Metal outputs."""
    monkeypatch.delenv("QUANTEM_MPS_SSB_PHASE_CHUNK_BF", raising=False)

    from quantem.gpu.ssb.mps import (
        _default_object_redraw_threadgroup,
        _default_phase_col_k_bf,
        _default_phase_loss_chunk_bf,
    )

    assert _default_phase_loss_chunk_bf((1024, 1024)) <= 512
    assert _default_phase_loss_chunk_bf((512, 512)) >= 512
    assert _default_phase_col_k_bf((512, 512)) == 32
    assert _default_object_redraw_threadgroup((512, 512)) == 64
    assert _default_object_redraw_threadgroup((1024, 1024)) == 64


def test_mps_hermitian_expand_matches_fft2_reference() -> None:
    """MPS half-plane storage must expand to the exact full FFT grid."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import _expand_hermitian_mx, _fft2_hermitian

    rng = np.random.default_rng(17)
    stack = rng.standard_normal((3, 16, 16)).astype(np.float32)
    half = _fft2_hermitian(mx, mx.array(stack))
    full = _expand_hermitian_mx(mx, half, 16)
    mx.eval(full)

    assert tuple(int(v) for v in half.shape) == (3, 16, 9)
    assert tuple(int(v) for v in full.shape) == (3, 16, 16)
    np.testing.assert_allclose(
        np.asarray(full),
        np.fft.fft2(stack).astype(np.complex64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mps_array_frame_columns_match_direct_detector_indexing() -> None:
    """Optimized detector-column gather must preserve exact BF evidence."""
    from quantem.gpu.ssb.mps import _ArrayFrames

    data = np.arange(4 * 5 * 6 * 7, dtype=np.uint16).reshape(4, 5, 6, 7)
    frames = _ArrayFrames(data)
    rows = np.array([0, 2, 5, 3], dtype=np.intp)
    cols = np.array([1, 6, 0, 4], dtype=np.intp)

    result = frames.columns(rows, cols)
    reference = data.reshape(-1, 6, 7)[:, rows, cols].T

    assert result.shape == (4, 20)
    np.testing.assert_array_equal(result, reference)


def test_mps_object_fourier_sum_matches_looped_reference() -> None:
    """MPS object Fourier sum must match the looped corrected-object path."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _ifft2_chunked,
        _object_fourier_sum_dynamic,
        _reconstruct_prepared,
    )

    n = 16
    num_bf = 5
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.linspace(-0.2, 0.2, num_bf, dtype=np.float32)
    ky_np = np.linspace(0.15, -0.15, num_bf, dtype=np.float32)
    rng = np.random.default_rng(22)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )
    c10 = mx.array([-120.0], dtype=mx.float32)
    c12 = mx.array([55.0], dtype=mx.float32)
    phi12 = 0.3
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=mx.array([np.cos(2.0 * phi12)], dtype=mx.float32),
        sin2phi12=mx.array([np.sin(2.0 * phi12)], dtype=mx.float32),
    )[0]
    reference = mx.fft.ifft2(mx.sum(corrected, axis=0) / num_bf)
    obj_chunk_reference = mx.fft.ifft2(corrected)
    obj_chunk_chunked = _ifft2_chunked(mx, corrected)
    result = _object_fourier_sum_dynamic(
        prepared,
        C10=-120.0,
        C12=55.0,
        phi12=phi12,
        chunk_bf=3,
    )
    mx.eval(reference, obj_chunk_reference, obj_chunk_chunked, result)

    np.testing.assert_allclose(
        np.asarray(result),
        np.asarray(reference),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(obj_chunk_chunked),
        np.asarray(obj_chunk_reference),
        rtol=1e-5,
        atol=1e-5,
    )

    phase_reference = mx.arctan2(
        mx.imag(obj_chunk_reference),
        mx.real(obj_chunk_reference),
    )
    phase_sum = mx.sum(phase_reference, axis=0)
    phase_sumsq = mx.sum(phase_reference * phase_reference, axis=0)
    mean_phase_reference = phase_sum / num_bf
    loss_reference = mx.mean(
        phase_sumsq / num_bf - mean_phase_reference * mean_phase_reference
    )
    prepared_object, prepared_loss, prepared_phase = _reconstruct_prepared(
        prepared,
        C10=-120.0,
        C12=55.0,
        phi12=phi12,
        chunk_bf=3,
        compute_loss=True,
        compute_object=True,
    )
    mx.eval(mean_phase_reference, loss_reference)
    np.testing.assert_allclose(
        prepared_object,
        np.asarray(mx.sum(obj_chunk_reference, axis=0) / num_bf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        prepared_phase,
        np.asarray(mean_phase_reference),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        prepared_loss,
        float(np.asarray(loss_reference)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mps_phase_cols512_matches_mlx_reference() -> None:
    """Fused 512-column phase/loss accumulator must match MLX IFFT reference."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import (
        _phase_cols512_from_row_ifft,
        _phase_sums_from_complex,
    )

    rng = np.random.default_rng(31)
    row_ifft_np = (
        rng.standard_normal((5, 512, 512))
        + 1j * rng.standard_normal((5, 512, 512))
    ).astype(np.complex64)
    row_ifft = mx.array(row_ifft_np)
    obj_reference = mx.fft.ifft(row_ifft, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, obj_reference)
    got_sum, got_sumsq = _phase_cols512_from_row_ifft(mx, row_ifft, k_bf=32)
    mx.eval(ref_sum, ref_sumsq, got_sum, got_sumsq)

    np.testing.assert_allclose(
        np.asarray(got_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(got_sumsq),
        np.asarray(ref_sumsq),
        rtol=1e-5,
        atol=1e-3,
    )


def test_mps_phase_cols512_reduced_modes_match_mlx_reference() -> None:
    """Reduced 512-column phase/loss modes must preserve phase and scalar loss."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import (
        _phase_cols512_scalar_loss_from_row_ifft,
        _phase_cols512_sum_from_row_ifft,
        _phase_sums_from_complex,
    )

    rng = np.random.default_rng(37)
    row_ifft_np = (
        rng.standard_normal((7, 512, 512))
        + 1j * rng.standard_normal((7, 512, 512))
    ).astype(np.complex64)
    row_ifft = mx.array(row_ifft_np)
    obj_reference = mx.fft.ifft(row_ifft, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, obj_reference)
    ref_sumsq_scalar = mx.sum(ref_sumsq)
    sum_only = _phase_cols512_sum_from_row_ifft(mx, row_ifft, k_bf=4)
    scalar_sum, scalar_sumsq = _phase_cols512_scalar_loss_from_row_ifft(
        mx,
        row_ifft,
        k_bf=4,
    )
    mx.eval(ref_sum, ref_sumsq_scalar, sum_only, scalar_sum, scalar_sumsq)

    np.testing.assert_allclose(
        np.asarray(sum_only),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sumsq),
        np.asarray(ref_sumsq_scalar),
        rtol=1e-5,
        atol=1e-1,
    )


@pytest.mark.parametrize("n", [128, 256, 1024])
def test_mps_phase_cols_small_reduced_modes_match_mlx_reference(n: int) -> None:
    """Reduced 128/256/1024-column phase/loss modes must match MLX IFFT reference."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import (
        _phase_cols_small_scalar_loss_from_row_ifft,
        _phase_cols_small_sum_from_row_ifft,
        _phase_sums_from_complex,
    )

    rng = np.random.default_rng(37 + n)
    row_ifft_np = (
        rng.standard_normal((7, n, n))
        + 1j * rng.standard_normal((7, n, n))
    ).astype(np.complex64)
    row_ifft = mx.array(row_ifft_np)
    obj_reference = mx.fft.ifft(row_ifft, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, obj_reference)
    ref_sumsq_scalar = mx.sum(ref_sumsq)
    sum_only = _phase_cols_small_sum_from_row_ifft(mx, row_ifft, k_bf=4)
    scalar_sum, scalar_sumsq = _phase_cols_small_scalar_loss_from_row_ifft(
        mx,
        row_ifft,
        k_bf=4,
    )
    mx.eval(ref_sum, ref_sumsq_scalar, sum_only, scalar_sum, scalar_sumsq)

    np.testing.assert_allclose(
        np.asarray(sum_only),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sumsq),
        np.asarray(ref_sumsq_scalar),
        rtol=1e-5,
        atol=1e-1,
    )


def test_mps_row_ifft512_dynamic_matches_mlx_reference() -> None:
    """Fused 512 correction + row-IFFT must match the MLX reference path."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _phase_cols512_from_row_ifft,
        _phase_sums_from_complex,
        _row_ifft512_from_dynamic_geometry,
    )

    n = 512
    num_bf = 3
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.array([-0.17, 0.03, 0.21], dtype=np.float32)
    ky_np = np.array([0.11, -0.19, 0.07], dtype=np.float32)
    rng = np.random.default_rng(47)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )
    c10 = mx.array([0.0], dtype=mx.float32)
    c12 = mx.array([0.0], dtype=mx.float32)
    cos2 = mx.array([1.0], dtype=mx.float32)
    sin2 = mx.array([0.0], dtype=mx.float32)
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    )[0]
    row_reference = mx.fft.ifft(corrected, axis=-1)
    row_fused = _row_ifft512_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    ) / n
    ref_obj = mx.fft.ifft(row_reference, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, ref_obj)
    got_sum, got_sumsq = _phase_cols512_from_row_ifft(mx, row_fused, k_bf=32)
    mx.eval(row_reference, row_fused, ref_sum, ref_sumsq, got_sum, got_sumsq)

    np.testing.assert_allclose(
        np.asarray(row_fused),
        np.asarray(row_reference),
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(got_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(got_sumsq),
        np.asarray(ref_sumsq),
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.parametrize("n", [128, 256, 1024])
def test_mps_row_ifft_small_dynamic_matches_mlx_reference(n: int) -> None:
    """Fused 128/256/1024 correction + row-IFFT must match the MLX reference."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _phase_cols_small_scalar_loss_from_row_ifft,
        _phase_sums_from_complex,
        _row_ifft_small_from_dynamic_geometry,
    )

    num_bf = 3
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.array([-0.17, 0.03, 0.21], dtype=np.float32)
    ky_np = np.array([0.11, -0.19, 0.07], dtype=np.float32)
    rng = np.random.default_rng(47 + n)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )
    c10 = mx.array([0.0], dtype=mx.float32)
    c12 = mx.array([0.0], dtype=mx.float32)
    cos2 = mx.array([1.0], dtype=mx.float32)
    sin2 = mx.array([0.0], dtype=mx.float32)
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    )[0]
    row_reference = mx.fft.ifft(corrected, axis=-1)
    row_fused = _row_ifft_small_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    ) / n
    ref_obj = mx.fft.ifft(row_reference, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, ref_obj)
    got_sum, got_sumsq_scalar = _phase_cols_small_scalar_loss_from_row_ifft(
        mx,
        row_fused,
        k_bf=32,
    )
    mx.eval(row_reference, row_fused, ref_sum, ref_sumsq, got_sum, got_sumsq_scalar)

    np.testing.assert_allclose(
        np.asarray(row_fused),
        np.asarray(row_reference),
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(got_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(got_sumsq_scalar),
        np.asarray(mx.sum(ref_sumsq)),
        rtol=1e-5,
        atol=1e-1,
    )


def test_mps_reconstruct_prepared_uses_scalar_loss_for_small_fused_path() -> None:
    """Top-level fused MPS loss accumulation must not broadcast scalar loss tiles."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.mps import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _phase_sums_from_complex,
        _reconstruct_prepared,
    )

    n = 128
    num_bf = 5
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.array([-0.17, 0.03, 0.21, -0.09, 0.13], dtype=np.float32)
    ky_np = np.array([0.11, -0.19, 0.07, 0.18, -0.05], dtype=np.float32)
    rng = np.random.default_rng(53)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )

    c10 = 0.0
    c12 = 0.0
    phi12 = 0.0
    _, got_loss, got_phase = _reconstruct_prepared(
        prepared,
        C10=c10,
        C12=c12,
        phi12=phi12,
        chunk_bf=4,
        compute_loss=True,
        compute_object=False,
    )
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=mx.array([c10], dtype=mx.float32),
        c12=mx.array([c12], dtype=mx.float32),
        cos2phi12=mx.array([1.0], dtype=mx.float32),
        sin2phi12=mx.array([0.0], dtype=mx.float32),
    )[0]
    ref_obj = mx.fft.ifft(mx.fft.ifft(corrected, axis=-1), axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, ref_obj)
    ref_phase = ref_sum / num_bf
    ref_loss = mx.mean(ref_sumsq / num_bf - ref_phase * ref_phase)
    mx.eval(ref_phase, ref_loss)

    np.testing.assert_allclose(
        got_phase,
        np.asarray(ref_phase),
        rtol=1e-5,
        atol=1e-3,
    )
    assert got_loss is not None
    np.testing.assert_allclose(got_loss, float(np.asarray(ref_loss)), atol=1e-4)


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_SSB_MASTER")
    or not os.environ.get("QUANTEM_GPU_SSB_REFERENCE_NPZ"),
    reason=(
        "set QUANTEM_GPU_SSB_MASTER and QUANTEM_GPU_SSB_REFERENCE_NPZ "
        "for real-data CUDA/MPS SSB reference parity"
    ),
)
def test_mps_ssb_fixed_aberration_matches_cuda_reference() -> None:
    """Compare MPS SSB fixed-aberration output against a CUDA reference artifact."""
    from quantem.gpu.io import load
    from quantem.gpu.ssb.mps import ssb_preview

    master = Path(os.environ["QUANTEM_GPU_SSB_MASTER"])
    reference_path = Path(os.environ["QUANTEM_GPU_SSB_REFERENCE_NPZ"])
    if not master.exists():
        pytest.skip(f"master not available: {master}")
    if not reference_path.exists():
        pytest.skip(f"reference not available: {reference_path}")

    reference = np.load(reference_path)
    reference_meta = json.loads(str(reference["meta"]))
    result = ssb_preview(
        load(str(master), backend="mps", verbose=False).data,
        voltage_kV=300,
        semiangle_mrad=21.4,
        scan_sampling_A=1.0,
        C10=0.0,
        C12=50.0,
        phi12=0.0,
        bf_radius=5,
        chunk_bf=16,
        compute_loss=True,
        verbose=False,
    )

    assert result.num_bf == int(reference_meta["num_bf"])
    assert np.allclose(result.bf_center, reference_meta["bf_center"], atol=1e-4)
    if reference_meta.get("loss_full") is not None:
        assert abs(float(result.loss) - float(reference_meta["loss_full"])) < 0.01

    phase_diff = np.abs(reference["phase"] - result.phase)
    phase_corr = np.corrcoef(reference["phase"].reshape(-1), result.phase.reshape(-1))[
        0, 1
    ]
    # MPS returns the same mean-of-per-BF-phase image as the CUDA fixed preview.
    assert float(phase_corr) > 0.999
    assert float(np.mean(phase_diff)) < 0.01
    assert float(np.quantile(phase_diff.reshape(-1), 0.99)) < 0.03


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_SSB_MASTER")
    or not os.environ.get("QUANTEM_GPU_SSB_SPARSE_REFERENCE_NPZ"),
    reason=(
        "set QUANTEM_GPU_SSB_MASTER and QUANTEM_GPU_SSB_SPARSE_REFERENCE_NPZ "
        "for real-data CUDA/MPS sparse optimizer parity"
    ),
)
def test_mps_ssb_sparse_optimizer_loss_matches_cuda_reference() -> None:
    """Compare MPS optimizer objective against pinned CUDA sparse losses."""
    from quantem.gpu.io import load
    from quantem.gpu.ssb.mps import (
        _as_chunked_frames,
        _as_sampling,
        _bf_pixels,
        _prepare_selection,
        _reconstruct_prepared_batch_cuda_sparse,
        _scan_shape,
    )

    master = Path(os.environ["QUANTEM_GPU_SSB_MASTER"])
    reference_path = Path(os.environ["QUANTEM_GPU_SSB_SPARSE_REFERENCE_NPZ"])
    if not master.exists():
        pytest.skip(f"master not available: {master}")
    if not reference_path.exists():
        pytest.skip(f"reference not available: {reference_path}")

    reference = np.load(reference_path)
    reference_meta = json.loads(str(reference["meta"]))
    assert reference_meta["objective"] == "cuda_sparse_variance_loss_batch"

    frames = _as_chunked_frames(load(str(master), backend="mps", verbose=False).data)
    scan_shape = _scan_shape(frames)
    if "scan_shape" in reference_meta:
        assert tuple(reference_meta["scan_shape"]) == scan_shape
    det_shape = tuple(int(x) for x in frames.shape[-2:])
    threshold = float(reference_meta.get("bf_intensity_threshold", 0.5))
    bf_radius = reference_meta.get("bf_radius_arg", 5)
    center_override = reference_meta.get("bf_center")
    if center_override is not None:
        center_override = (float(center_override[0]), float(center_override[1]))
    bf_row, bf_col, center, _radius, detected_radius = _bf_pixels(
        frames,
        threshold,
        bf_radius,
        center_override=center_override,
    )
    if "num_bf" in reference_meta:
        assert int(bf_row.size) == int(reference_meta["num_bf"])
    semiangle_mrad = float(reference_meta.get("semiangle_mrad", 21.4))
    det_px = (2.0 * semiangle_mrad) / detected_radius
    prepared = _prepare_selection(
        frames,
        scan_shape=scan_shape,
        det_shape=det_shape,
        bf_row=bf_row,
        bf_col=bf_col,
        center=center,
        voltage_kV=float(reference_meta.get("voltage_kV", 300)),
        semiangle_mrad=semiangle_mrad,
        scan_sampling=_as_sampling(float(reference_meta.get("scan_sampling_A", 0.405))),
        det_sampling=(det_px, det_px),
        rotation_angle_deg=0.0,
        chunk_bf=16,
    )
    assert prepared.g_qk.shape[-1] == scan_shape[1] // 2 + 1

    params = reference["params"].astype(np.float32)
    losses = _reconstruct_prepared_batch_cuda_sparse(
        prepared,
        C10=params[:, 0],
        C12=params[:, 1],
        phi12=params[:, 2],
        chunk_bf=16,
    )
    assert np.allclose(losses, reference["losses"], atol=1e-4)
