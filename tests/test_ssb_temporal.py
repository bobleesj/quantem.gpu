from __future__ import annotations

import math

import numpy as np
import pytest


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # noqa: BLE001 - runtime probes are backend-specific
        pytest.skip(f"CUDA unavailable: {exc}")
    return cp


def _object_wave(cp, shape=(32, 32)):
    rng = cp.random.default_rng(123)
    amp = 1.0 + 0.05 * rng.standard_normal(shape, dtype=cp.float32)
    phase = 0.2 * rng.standard_normal(shape, dtype=cp.float32)
    return (amp * cp.exp(1j * phase)).astype(cp.complex64)


def test_join_object_waves_undoes_known_fourier_shift() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.temporal import fourier_shift_2d, join_object_waves

    base = _object_wave(cp)
    shift = (2.25, -1.5)
    shifted = fourier_shift_2d(base, shift)

    result = join_object_waves(
        cp.stack([base, shifted], axis=0),
        shifts=[(0.0, 0.0), (-shift[0], -shift[1])],
        phase_reference="none",
    )

    max_abs = float(cp.max(cp.abs(result.object_wave - base)).get())
    assert max_abs < 2e-5
    assert result.join_mode == "drift_corrected_complex_mean"
    assert result.num_frames == 2


def test_join_object_waves_aligns_global_phase_offsets() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.temporal import join_object_waves

    base = _object_wave(cp)
    offset = cp.exp(1j * cp.float32(0.7)).astype(cp.complex64)
    stack = cp.stack([base, base * offset], axis=0)

    aligned = join_object_waves(stack, phase_reference="first")
    literal = join_object_waves(stack, phase_reference="none")

    aligned_err = float(cp.max(cp.abs(aligned.object_wave - base)).get())
    literal_err = float(cp.mean(cp.abs(literal.object_wave - base)).get())
    assert aligned_err < 2e-6
    assert literal_err > 0.1


def test_ssb_time_average_reconstructs_frames_with_shared_calibration() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.temporal import fourier_shift_2d, ssb_time_average

    base = _object_wave(cp)
    shift = (1.5, -0.75)
    shifted = fourier_shift_2d(base, shift)

    class FakeSSB:
        def __init__(self, wave):
            self.wave = wave
            self.aberrations = {"C10": 1.0, "C12": 2.0, "phi12": 0.0}
            self._rotation_angle_rad = 0.0
            self.voltage_kV = 300
            self.semiangle_mrad = 21.4
            self.scan_sampling = (0.5, 0.5)
            self.calls = []

        def _reconstruct_object(self, C10, C12, phi12, rotation_angle_rad):
            self.calls.append((C10, C12, phi12, rotation_angle_rad))
            return self.wave

    frames = [FakeSSB(base), FakeSSB(shifted)]
    result = ssb_time_average(
        frames,
        shifts=[(0.0, 0.0), (-shift[0], -shift[1])],
        aberrations={"C10": 10.0, "C12": 4.0, "phi12": math.radians(12.0)},
        rotation_angle_deg=5.0,
        phase_reference="none",
    )

    max_abs = float(cp.max(cp.abs(result.object_wave - base)).get())
    assert max_abs < 2e-5
    assert result.join_mode == "ssb_time_average"
    assert result.voltage_kV == 300
    assert result.semiangle_mrad == 21.4
    assert result.scan_sampling_A == 0.5
    np.testing.assert_allclose(
        np.asarray(frames[0].calls[0], dtype=float),
        np.array([10.0, 4.0, math.radians(12.0), math.radians(5.0)]),
    )


def test_ssb_time_series_preserves_per_frame_objects_and_can_average() -> None:
    cp = _cupy()
    from quantem.gpu.ssb.temporal import fourier_shift_2d, ssb_time_average, ssb_time_series

    base = _object_wave(cp)
    changed = (base * cp.exp(1j * cp.float32(0.03))).astype(cp.complex64)
    shifted_changed = fourier_shift_2d(changed, (1.0, -0.5))

    class FakeSSB:
        def __init__(self, wave):
            self.wave = wave
            self.aberrations = {"C10": 1.0, "C12": 2.0, "phi12": 0.0}
            self._rotation_angle_rad = 0.0
            self.voltage_kV = 300
            self.semiangle_mrad = 21.4
            self.scan_sampling = (0.5, 0.5)

        def _reconstruct_object(self, C10, C12, phi12, rotation_angle_rad):
            return self.wave

    frames = [FakeSSB(base), FakeSSB(shifted_changed)]
    shifts = [(0.0, 0.0), (-1.0, 0.5)]

    series = ssb_time_series(frames, shifts=shifts, phase_reference="none")
    joined = ssb_time_average(frames, shifts=shifts, phase_reference="none")
    series_average = series.time_average()

    assert series.object_waves.shape == (2, *base.shape)
    assert series.mode == "shared_calibration_time_series"
    assert series_average.join_mode == "shared_calibration_time_series_average"

    frame_delta = float(cp.mean(cp.abs(series.object_waves[1] - series.object_waves[0])).get())
    assert frame_delta > 0.01

    avg_err = float(cp.max(cp.abs(series_average.object_wave - joined.object_wave)).get())
    assert avg_err < 2e-6
