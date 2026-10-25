"""Explicit wave averaging preserves calibration and native diagnostic loss."""

import math
import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("depth", [0., 18.])
def test_paired_frequencies_match_full_plane_and_keep_native_diagnostics(depth, monkeypatch):
    """Retain the complete BF sum when a finer grid permits frequency pairing."""
    ssb, engine = _synthetic_session()
    options = dict(sample_tilt_mrad=(-9., 5.), depth_spread_nm=depth,
                   upsampling_factor=4)
    aberrations = {"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.}
    try:
        assert engine.thick._wave_columns(128, 128) == 128
        # The native Nyquist edge crosses the aperture support. At 4x it does not.
        assert engine.thick._wave_columns(512, 512) == 257
        paired, paired_loss = ssb.preview(aberrations, **options)
        monkeypatch.setattr(engine.thick, "_wave_columns", lambda rows, cols: cols)
        full, full_loss = ssb.preview(aberrations, **options)
        np.testing.assert_allclose(paired, full, atol=1e-6, rtol=1e-5)
        assert paired_loss == full_loss
    finally:
        ssb.close()


def _direct_wave_average(ssb, factor, tilt, depth):
    """Independent Torch correction and per-detector inverse transforms."""
    import torch
    from tests.parity.torch_ssb import TorchSSB

    reference = TorchSSB.from_ssb(ssb)
    rows, cols = reference.ny * factor, reference.nx * factor
    q_row = torch.fft.fftfreq(rows, d=0.3 / factor, device=reference.device)
    q_col = torch.fft.fftfreq(cols, d=0.3 / factor, device=reference.device)
    wave_sum_t = torch.zeros((rows, cols), dtype=torch.complex128, device=reference.device)
    for start in range(0, reference.num_bright_field_pixels, 4):
        stop = min(start + 4, reference.num_bright_field_pixels)
        native_t = torch.fft.ifft2(reference._full_plane(reference.G[start:stop])).real
        sparse_t = torch.zeros((stop - start, rows, cols), device=reference.device)
        sparse_t[:, ::factor, ::factor] = native_t
        spectrum_t = torch.fft.fft2(sparse_t)
        gamma_t = reference._gamma(
            q_row[None, :, None], q_col[None, None, :],
            reference.kx[start:stop, None, None], reference.ky[start:stop, None, None],
            -80.0, 40.0, -0.7, tilt, depth * 10,
        )
        magnitude_squared_t = gamma_t.abs().square()
        unit_t = gamma_t * torch.where(
            magnitude_squared_t > 1e-16,
            torch.rsqrt(magnitude_squared_t.clamp_min(1e-16)),
            torch.full_like(magnitude_squared_t, 1e8),
        )
        corrected_t = spectrum_t * torch.conj(unit_t)
        corrected_t[:, 0, 0] = reference.dc_value
        wave_sum_t += torch.fft.ifft2(corrected_t).sum(0, dtype=torch.complex128)
    return torch.angle(wave_sum_t).float().cpu().numpy()


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
@pytest.mark.parametrize("depth", [0.0, 18.0])
def test_explicit_estimator_matches_direct_waves_and_preserves_native_loss(factor, depth):
    """Change output sampling with a fixed estimator and native calibration."""
    ssb, _ = _synthetic_session()
    aberrations = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": math.degrees(-0.7)}
    tilt = (-9.0, 5.0)
    try:
        native, native_loss = ssb.preview(aberrations, sample_tilt_mrad=tilt, depth_spread_nm=depth)
        legacy, _ = ssb.preview(
            aberrations, sample_tilt_mrad=tilt, depth_spread_nm=depth, upsampling_factor=factor,
        )
        candidate, loss = ssb.preview(
            aberrations, sample_tilt_mrad=tilt, depth_spread_nm=depth,
            upsampling_factor=factor,
        )
        expected = _direct_wave_average(ssb, factor, tilt, depth)
        np.testing.assert_allclose(candidate, expected, atol=1e-6, rtol=1e-5)
        assert loss == native_loss
        assert np.isfinite(candidate).all()
        restored, restored_loss = ssb.preview(
            aberrations, sample_tilt_mrad=tilt, depth_spread_nm=depth,
        )
        np.testing.assert_array_equal(restored, native)
        assert restored_loss == native_loss
        repeated, _ = ssb.preview(
            aberrations, sample_tilt_mrad=tilt, depth_spread_nm=depth, upsampling_factor=factor,
        )
        np.testing.assert_array_equal(repeated, legacy)
    finally:
        ssb.close()


@pytest.mark.parametrize("depth", [0.0, 18.0])
def test_wave_average_chunking_and_inactive_higher_order_angles(depth):
    """Preview the same correction after changing memory batching and packed coefficients."""
    import cupy as cp

    ssb, engine = _synthetic_session()
    aberrations = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": math.degrees(-0.7)}
    tilt = (-9.0, 5.0)
    try:
        candidate, loss = ssb.preview(
            aberrations, sample_tilt_mrad=tilt, depth_spread_nm=depth,
            upsampling_factor=4, phase_estimator="phase_of_mean", compute_loss=False,
        )
        assert loss is None
        batched, _ = engine.thick.reconstruct(
            -80.0, 40.0, -0.7, tilt, depth * 10, upsampling_factor=4,
            phase_estimator="phase_of_mean", compute_loss=False,
            chunk_bytes=2 * 512 * 512 * 24,
        )
        np.testing.assert_allclose(cp.asnumpy(batched), candidate, atol=1e-6, rtol=0)
        # an azimuth left on a higher order whose magnitude is zero selects the packed coefficients
        packed, _ = ssb.preview(
            aberrations | {"phi21_deg": math.degrees(1.2)}, sample_tilt_mrad=tilt, depth_spread_nm=depth,
            upsampling_factor=4, phase_estimator="phase_of_mean", compute_loss=False,
        )
        np.testing.assert_allclose(packed, candidate, atol=1e-6, rtol=0)
    finally:
        ssb.close()


@pytest.mark.parametrize("depth", [0.0, 18.0])
@pytest.mark.parametrize("higher", [False, True])
def test_group_support_keeps_full_bf_phase_after_rotation(depth, higher, monkeypatch):
    """Compare tight support with unpruned GPU sums through sampling and rotation changes."""
    import cupy as cp

    ssb, engine = _synthetic_session()
    bounds = engine.thick._wave_group_bounds
    aberrations = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    if higher:
        aberrations.update(C21_nm=1.0, phi21_deg=23.0, C30_nm=10.0)

    def unpruned(group_size):
        shape = bounds(group_size).shape
        limits = cp.array([-cp.inf, cp.inf, -cp.inf, cp.inf], dtype=cp.float32)
        return cp.broadcast_to(limits, shape).copy()

    try:
        for rotation in (16.3, 196.3, 0.0):
            ssb.set_rotation(rotation)
            for factor in (1, 4):
                options = dict(sample_tilt_mrad=(-9.0, 5.0), depth_spread_nm=depth,
                               upsampling_factor=factor, compute_loss=False)
                candidate, _ = ssb.preview(aberrations, **options)
                with monkeypatch.context() as patch:
                    patch.setattr(engine.thick, "_wave_group_bounds", unpruned)
                    expected, _ = ssb.preview(aberrations, **options)
                np.testing.assert_array_equal(candidate, expected)
    finally:
        ssb.close()


def test_higher_order_coefficients_are_independent_between_streams():
    """Concurrent viewers retain their own fifth-order optical coefficients."""
    from concurrent.futures import ThreadPoolExecutor

    import cupy as cp

    sessions = [_synthetic_session()[0] for _ in range(2)]
    optics = [
        {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": 15.0, "C21_nm": 50.0, "C50_nm": 800000.0},
        {"C10_nm": 7.0, "C12_nm": 2.0, "phi12_deg": -23.0, "C21_nm": -30.0, "C50_nm": -400000.0},
    ]

    def reconstruct(index):
        with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True):
            phase, _ = sessions[index].preview(
                optics[index], sample_tilt_mrad=(3.0, -4.0),
                depth_spread_nm=18.0, upsampling_factor=2,
                phase_estimator="phase_of_mean", compute_loss=False,
            )
            return np.asarray(phase).copy()

    try:
        references = [reconstruct(i) for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(3):
                for actual, expected in zip(pool.map(reconstruct, range(2)), references):
                    np.testing.assert_array_equal(actual, expected)
    finally:
        for session in sessions:
            session.close()


def test_native_512_joint_optics_keep_phase_with_bounded_workspace():
    """Changing defocus keeps full fifth-order phase independent of workspace size."""
    from quantem.gpu import SSB

    from tests.hardware.test_ssb_coherent_higher_order import ANGLES, COEFS, NAMES

    rng = np.random.default_rng(81)
    counts = rng.integers(10, 80, size=(512, 512, 16, 16), dtype=np.uint16)
    row, col = np.meshgrid(np.arange(16) - 7.5, np.arange(16) - 7.5, indexing="ij")
    counts[:, :, np.hypot(row, col) < 6] += 100
    with SSB(counts, backend="cuda", voltage_kV=300, semiangle_mrad=30,
             scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(5.0, 5.0),
             scan_detector_rotation_deg=17) as ssb:
        options = {
            "phase_estimator": "phase_of_mean", "upsampling_factor": 1,
            "sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": 18.0,
            "compute_loss": False,
        }
        before, _ = ssb.preview(COEFS, **options)
        assert ssb.num_bright_field_pixels > 64
        for defocus in (-8.0, 3.0, -8.0):
            optics = COEFS | {"C10_nm": defocus}
            full, _ = ssb.preview(optics, **options)
            engine = ssb._backend()._get_accelerator()
            magnitudes = np.asarray([optics[f"{name}_nm"] for name in NAMES], dtype=np.float32) * 10
            bounded, _ = engine.thick.reconstruct(
                defocus * 10, optics["C12_nm"] * 10, math.radians(optics["phi12_deg"]),
                (-9.0, 5.0), 180.0, chunk_bytes=512 * 512 * 16,
                phase_estimator="phase_of_mean", compute_loss=False,
                higher_order_magnitudes=magnitudes,
                higher_order_angles=np.asarray(ANGLES, dtype=np.float32),
            )
            np.testing.assert_allclose(full, bounded.get(), atol=1e-6, rtol=1e-5)
            if defocus == -8.0:
                np.testing.assert_array_equal(full, before)
            else:
                assert np.std(full - before) > 1e-6
