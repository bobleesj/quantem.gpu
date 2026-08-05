"""Diffraction stack registration: shift recovery, gating, and batch equivalence."""

import numpy as np
import pytest

from quantem.gpu import diffraction


def _ring_pattern(shape=(96, 96), center=(48.0, 48.0), radii=(12.0, 22.0)):
    """Synthetic SAED-like frame: a bright center plus a couple of sharp rings."""

    rows = np.arange(shape[0], dtype=np.float64)[:, None] - center[0]
    cols = np.arange(shape[1], dtype=np.float64)[None, :] - center[1]
    radius = np.hypot(rows, cols)

    frame = 50.0 * np.exp(-(radius**2) / (2.0 * 3.0**2))
    for ring in radii:
        frame += 20.0 * np.exp(-((radius - ring) ** 2) / (2.0 * 1.5**2))
    return frame


def _shifted_stack(shifts, noise=0.0, seed=0):
    """Stack whose frame i is the base pattern moved by shifts[i], via Fourier shift."""

    base = _ring_pattern()
    n_rows, n_cols = base.shape
    f_row = np.fft.fftfreq(n_rows)[:, None]
    f_col = np.fft.fftfreq(n_cols)[None, :]
    spectrum = np.fft.fft2(base)

    frames = []
    for d_row, d_col in shifts:
        ramp = np.exp(-2j * np.pi * (f_row * d_row + f_col * d_col))
        frames.append(np.fft.ifft2(spectrum * ramp).real)

    stack = np.stack(frames)
    if noise:
        stack = stack + np.random.default_rng(seed).normal(0.0, noise, stack.shape)
    return stack


def test_recovers_known_subpixel_shifts():
    applied = [(0.0, 0.0), (2.0, -3.0), (-1.5, 2.5), (3.25, 0.75)]
    result = diffraction.align_frames(_shifted_stack(applied))

    # measured shift is what undoes the applied drift, so it carries the opposite sign
    expected = -np.array(applied, dtype=np.float64)
    assert np.allclose(result.shifts, expected, atol=0.15)
    assert result.used.all()


def test_alignment_brings_frames_onto_the_reference():
    stack = _shifted_stack([(0.0, 0.0), (2.0, -3.0), (-1.5, 2.5)])
    result = diffraction.align_frames(stack)

    # compare interiors: bilinear resampling zero-fills the edge the shift vacates
    reference = result.aligned[0][8:-8, 8:-8]
    for frame in result.aligned[1:]:
        interior = frame[8:-8, 8:-8]
        assert np.corrcoef(interior.ravel(), reference.ravel())[0, 1] > 0.99


def test_drift_beyond_max_shift_is_flagged_and_left_unmoved():
    stack = _shifted_stack([(0.0, 0.0), (20.0, 0.0)])
    result = diffraction.align_frames(stack, max_shift=8.0)

    assert bool(result.used[0]) and not bool(result.used[1])
    assert np.allclose(result.aligned[1], stack[1].astype(np.float32), atol=1e-4)


def test_pure_noise_fails_the_quality_gate():
    rng = np.random.default_rng(1)
    stack = np.stack([_ring_pattern(), rng.normal(0.0, 1.0, (96, 96))])
    result = diffraction.align_frames(stack)

    assert not bool(result.used[1])


def test_batched_result_matches_frame_by_frame():
    stack = _shifted_stack([(0.0, 0.0), (2.0, -3.0), (-1.5, 2.5)], noise=0.5)
    batched = diffraction.align_frames(stack)

    # each frame registered on its own against the same reference must agree
    for i in range(1, len(stack)):
        single = diffraction.align_frames(stack[[0, i]])
        assert np.allclose(batched.shifts[i], single.shifts[1], atol=1e-6)


def test_rejects_non_stack_input():
    with pytest.raises(ValueError, match="3D stack"):
        diffraction.align_frames(_ring_pattern())
