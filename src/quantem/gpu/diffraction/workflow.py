"""Diffraction pattern stack registration - backend-agnostic.

A SAED or powder stack drifts between exposures, so every measurement taken
downstream (center, ring radii, d-spacings) is only as good as the registration
underneath it. Registration here is subpixel phase correlation - the same
Fourier cross-correlation the parallax bright-field alignment uses, applied to
whole detector frames instead of virtual images.

The work is embarrassingly parallel over frames, so the whole stack registers in
one batched FFT rather than a Python loop over frames. The array module follows
the input: NumPy stays on the host, CuPy stays resident on the device, so
callers never branch on hardware themselves.

Usage::

    from quantem.gpu import diffraction
    result = diffraction.align_frames(stack)
    result.aligned              # registered stack, float32
    result.shifts               # (n_frames, 2) row/col shift applied per frame
    result.used                 # frames that passed the correlation quality gate
"""
from __future__ import annotations

import time

import numpy as np

from .results import AlignmentResult

# Frequency taper on the normalized cross-power spectrum. Phase correlation
# whitens every frequency equally, which lets detector noise at the Nyquist
# edge outvote the real peak; this keeps the low-frequency content that carries
# the drift.
_LOWPASS_SIGMA = 0.15

# Correlation peak must clear the sidelobe field by this margin, expressed as
# psr / (psr + 10) so the gate is a 0-1 quality rather than a raw ratio.
_MIN_PEAK_QUALITY = 0.2

# Sidelobe statistics ignore this many pixels around the peak.
_PSR_EXCLUDE = 5.0

# scipy.ndimage truncates its Gaussian kernel at 4 sigma.
_GAUSSIAN_TRUNCATE = 4.0


# --- array module ---


def _array_module(frames):
    """NumPy for host arrays, CuPy for device-resident ones."""

    if type(frames).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


# --- separable Gaussian (scipy.ndimage.gaussian_filter defaults) ---


def _gaussian_weights(sigma, xp):
    radius = int(_GAUSSIAN_TRUNCATE * sigma + 0.5)
    offsets = xp.arange(-radius, radius + 1, dtype=xp.float64)
    weights = xp.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def _reflect_index(n, radius, xp):
    """Index map for scipy's 'reflect' edge mode, where the edge sample repeats."""

    idx = xp.arange(-radius, n + radius)
    period = 2 * n
    idx = xp.mod(idx, period)
    return xp.where(idx >= n, period - 1 - idx, idx)


def _blur_axis(stack, weights, axis, xp):
    """Convolve one axis, reflecting at the edge like scipy's 'reflect' mode.

    Reflect-padding by the kernel radius makes a cyclic FFT convolution agree
    with the spatial one: wraparound only touches the pad, which is cropped.
    That matters because the kernel runs tens of taps wide at these sigmas, so
    a tap-at-a-time pass over the whole stack costs far more than the transform.
    """

    n = stack.shape[axis]
    radius = (weights.size - 1) // 2
    padded = xp.moveaxis(xp.take(stack, _reflect_index(n, radius, xp), axis=axis), axis, -1)

    length = padded.shape[-1]
    kernel = xp.zeros(length, dtype=xp.float64)
    kernel[(xp.arange(weights.size) - radius) % length] = weights

    blurred = xp.fft.irfft(
        xp.fft.rfft(padded, axis=-1) * xp.fft.rfft(kernel), n=length, axis=-1
    )
    return xp.moveaxis(blurred[..., radius : radius + n], -1, axis)


def _gaussian_blur(stack, sigma, xp):
    weights = _gaussian_weights(sigma, xp)
    return _blur_axis(_blur_axis(stack, weights, -2, xp), weights, -1, xp)


def _bandpass(stack, xp):
    """High-pass every frame: subtract a wide Gaussian, then zero the mean."""

    work = stack - stack.min(axis=(-2, -1), keepdims=True)
    sigma = max(5.0, 0.02 * min(stack.shape[-2:]))
    work = work - _gaussian_blur(work, sigma, xp)
    return work - work.mean(axis=(-2, -1), keepdims=True)


# --- windowing ---


def _tukey(n, alpha, xp):
    """Tapered cosine window, matching scipy.signal.windows.tukey."""

    x = xp.linspace(0.0, 1.0, n)
    window = xp.ones(n, dtype=xp.float64)
    taper = alpha / 2.0

    rising = x < taper
    falling = x >= 1.0 - taper
    window[rising] = 0.5 * (1.0 + xp.cos(2.0 * np.pi / alpha * (x[rising] - taper)))
    window[falling] = 0.5 * (
         1.0 + xp.cos(2.0 * np.pi / alpha * (x[falling] - 1.0 + taper))
    )
    return window


# --- subpixel peak location ---


def _parabolic_offset(values, peaks, n, xp):
    """Three-point parabola vertex around each frame's peak, in samples."""

    rows = xp.arange(values.shape[0])
    mid = values[rows, peaks]
    lo = values[rows, (peaks - 1) % n]
    hi = values[rows, (peaks + 1) % n]

    denom = 2.0 * mid - lo - hi
    delta = xp.where(denom > 0, 0.5 * (hi - lo) / xp.where(denom > 0, denom, 1.0), 0.0)
    return xp.where(xp.abs(delta) <= 1.0, delta, 0.0)


def _peak_to_sidelobe(corr, p_row, p_col, xp):
    """Peak height over the spread of the correlation field outside the peak."""

    n_frames, n_rows, n_cols = corr.shape
    row_dist = xp.abs(xp.arange(n_rows)[None, :] - p_row[:, None])
    col_dist = xp.abs(xp.arange(n_cols)[None, :] - p_col[:, None])
    row_dist = xp.minimum(row_dist, n_rows - row_dist)
    col_dist = xp.minimum(col_dist, n_cols - col_dist)

    outside = (row_dist[:, :, None] > _PSR_EXCLUDE) | (col_dist[:, None, :] > _PSR_EXCLUDE)
    count = outside.sum(axis=(1, 2)).astype(xp.float64)
    total = (corr * outside).sum(axis=(1, 2))
    total_sq = (corr**2 * outside).sum(axis=(1, 2))

    mean = total / count
    spread = xp.sqrt(xp.maximum(total_sq / count - mean**2, 0.0))
    peak = corr[xp.arange(n_frames), p_row, p_col]
    return xp.where(spread > 0, (peak - mean) / xp.where(spread > 0, spread, 1.0), 0.0)


def _phase_shift(ref, moving, xp):
    """Subpixel shift of every frame in ``moving`` relative to ``ref``."""

    n_rows, n_cols = ref.shape
    window = _tukey(n_rows, 0.2, xp)[:, None] * _tukey(n_cols, 0.2, xp)[None, :]

    cross = xp.fft.fft2(ref * window)[None] * xp.conj(
        xp.fft.fft2(moving * window, axes=(-2, -1))
    )
    cross = cross / xp.maximum(xp.abs(cross), 1e-12)

    f_row = xp.fft.fftfreq(n_rows)[:, None]
    f_col = xp.fft.fftfreq(n_cols)[None, :]
    cross = cross * xp.exp(-(f_row**2 + f_col**2) / (2.0 * _LOWPASS_SIGMA**2))
    corr = xp.fft.ifft2(cross, axes=(-2, -1)).real

    n_frames = corr.shape[0]
    peak = xp.argmax(corr.reshape(n_frames, -1), axis=1)
    p_row, p_col = peak // n_cols, peak % n_cols

    frames = xp.arange(n_frames)
    d_row = _parabolic_offset(corr[frames, :, p_col], p_row, n_rows, xp)
    d_col = _parabolic_offset(corr[frames, p_row, :], p_col, n_cols, xp)

    shifts = xp.stack([p_row + d_row, p_col + d_col], axis=1)
    return shifts, _peak_to_sidelobe(corr, p_row, p_col, xp)


# --- resampling ---


def _wrap_signed(shift, n, xp):
    """Fold a periodic correlation coordinate onto the signed shift it means."""

    shift = xp.mod(shift, n)
    return xp.where(shift > n / 2.0, shift - n, shift)


def _shift_bilinear(frames, shifts, xp):
    """Bilinear resample matching ndimage.shift(order=1, mode='constant')."""

    n_frames, n_rows, n_cols = frames.shape
    rows = xp.arange(n_rows, dtype=xp.float64)[None, :, None] - shifts[:, 0][:, None, None]
    cols = xp.arange(n_cols, dtype=xp.float64)[None, None, :] - shifts[:, 1][:, None, None]

    r0, c0 = xp.floor(rows), xp.floor(cols)
    w_row, w_col = rows - r0, cols - c0
    r0, c0 = r0.astype(xp.int64), c0.astype(xp.int64)

    out = xp.zeros((n_frames, n_rows, n_cols), dtype=xp.float64)
    index = xp.arange(n_frames)[:, None, None]
    corners = (
        (0, 0, (1.0 - w_row) * (1.0 - w_col)),
        (0, 1, (1.0 - w_row) * w_col),
        (1, 0, w_row * (1.0 - w_col)),
        (1, 1, w_row * w_col),
    )
    for d_row, d_col, weight in corners:
        rr, cc = r0 + d_row, c0 + d_col
        inside = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
        sample = frames[index, xp.clip(rr, 0, n_rows - 1), xp.clip(cc, 0, n_cols - 1)]
        out += xp.where(inside, sample * weight, 0.0)

    # 'constant' mode fills rather than blends once the source coordinate leaves
    # the input extent, so an edge pixel does not half-fade into the fill value
    covered = (rows >= 0) & (rows <= n_rows - 1) & (cols >= 0) & (cols <= n_cols - 1)
    return xp.where(covered, out, 0.0)


# --- public workflow ---


def align_frames(frames, reference=None, *, max_shift=8.0) -> AlignmentResult:
    """Register a diffraction stack by subpixel phase correlation.

    ``frames`` is a ``(n_frames, n_rows, n_cols)`` stack. Frames whose
    correlation peak is weak, or whose measured drift exceeds ``max_shift``
    pixels, are passed through unshifted and flagged in ``result.used`` rather
    than moved on an unreliable estimate.
    """

    t0 = time.perf_counter()
    xp = _array_module(frames)
    frames = xp.asarray(frames, dtype=xp.float64)
    if frames.ndim != 3:
        raise ValueError(f"frames must be a 3D stack; got shape {frames.shape}.")

    n_frames, n_rows, n_cols = frames.shape
    ref = frames[0] if reference is None else xp.asarray(reference, dtype=xp.float64)
    if ref.shape != (n_rows, n_cols):
        raise ValueError(
            f"reference shape {ref.shape} does not match frame shape {(n_rows, n_cols)}."
        )

    ref = _bandpass(ref[None], xp)[0]
    shifts, psr = _phase_shift(ref, _bandpass(frames, xp), xp)
    shifts = xp.stack(
        [_wrap_signed(shifts[:, 0], n_rows, xp), _wrap_signed(shifts[:, 1], n_cols, xp)],
        axis=1,
    )

    quality = xp.where(psr > 0, psr / (psr + 10.0), 0.0)
    used = (xp.hypot(shifts[:, 0], shifts[:, 1]) <= max_shift) & (
        quality >= _MIN_PEAK_QUALITY
    )
    aligned = xp.where(used[:, None, None], _shift_bilinear(frames, shifts, xp), frames)

    return AlignmentResult(
        aligned=aligned.astype(xp.float32),
        shifts=shifts.astype(xp.float32),
        used=used,
        elapsed=time.perf_counter() - t0,
    )
