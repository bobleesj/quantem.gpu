"""Differential Phase Contrast (CoM / DPC / iDPC) for 4D-STEM — backend-agnostic.

CoM, DPC, and iDPC are DERIVED scalar fields, not raw 4D data, so they live here
(viewed with ``Show2D``), separate from the raw ``Show4DSTEM`` viewer.

The only expensive step is the per-scan-position center of mass over the full
detector - one pass over the (no-bin) 4D block:
  - MPS (MacBook): raw-Metal ``com_u8``/``com_u16`` kernels over chunked
    buffers, int64 accumulate, no float cast.
  - CUDA: CuPy-backed GPU CoM for resident arrays.
  - CPU/Torch: reference fallback with the same formula.
Everything after CoM (rotation alignment, Fourier integration) is small-field
math on the ``(scan_row, scan_col)`` CoM, ported 1:1 from quantem.live's
``engine.dpc`` so results match the dashboard.

Usage::

    from quantem.gpu import dpc, io
    from quantem.widget import Show2D
    result = dpc.run(io.load("scan_master.h5"))
    Show2D(result.phase)                      # the iDPC phase image
    Show2D(result.com_col)                    # raw DPC field (col)
"""
from __future__ import annotations

import time

import numpy as np

from .results import DPCResult


# --- small-field math (ported 1:1 from quantem.live.engine.dpc, cp -> np) ---


def _freq_grid_2d(shape):
    f_row = np.fft.fftfreq(shape[0]).astype(np.float32)
    f_col = np.fft.fftfreq(shape[1]).astype(np.float32)
    return np.meshgrid(f_row, f_col, indexing="ij")


def _rotate_vector_batch(v_row, v_col, angles_rad):
    c = np.cos(angles_rad)[:, None, None]
    s = np.sin(angles_rad)[:, None, None]
    return c * v_row - s * v_col, s * v_row + c * v_col


def _curl_batch(v_row, v_col):
    # curl = d(v_col)/d_row - d(v_row)/d_col, central differences, mean-squared
    dv_row_dcol = 0.5 * (v_row[:, 1:-1, 2:] - v_row[:, 1:-1, :-2])
    dv_col_drow = 0.5 * (v_col[:, 2:, 1:-1] - v_col[:, :-2, 1:-1])
    curl = dv_col_drow - dv_row_dcol
    return (curl ** 2).mean(axis=(1, 2))


def _rotation_curl_scores(v_row, v_col, angles_rad):
    """Curl score for many rotation angles without materializing rotated maps."""
    # For rotated vector field:
    #   r' = cos(a) r - sin(a) c
    #   c' = sin(a) r + cos(a) c
    # curl(r', c') = cos(a) * curl(r, c) + sin(a) * div(r, c).
    # Therefore mean(curl^2) for every angle can be evaluated from three scalar
    # moments of curl/divergence, instead of allocating one full map per angle.
    curl = (
        0.5 * (v_col[2:, 1:-1] - v_col[:-2, 1:-1])
        - 0.5 * (v_row[1:-1, 2:] - v_row[1:-1, :-2])
    ).astype(np.float64, copy=False)
    divergence = (
        0.5 * (v_row[2:, 1:-1] - v_row[:-2, 1:-1])
        + 0.5 * (v_col[1:-1, 2:] - v_col[1:-1, :-2])
    ).astype(np.float64, copy=False)
    cc = float(np.mean(curl * curl))
    dd = float(np.mean(divergence * divergence))
    cd = float(np.mean(curl * divergence))
    cos_a = np.cos(angles_rad, dtype=np.float64)
    sin_a = np.sin(angles_rad, dtype=np.float64)
    return cos_a * cos_a * cc + sin_a * sin_a * dd + 2.0 * cos_a * sin_a * cd


def _rotate_vector(v_row, v_col, angle_rad):
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return (
        (c * v_row - s * v_col).astype(np.float32, copy=False),
        (s * v_row + c * v_col).astype(np.float32, copy=False),
    )


def find_optimal_rotation(com_row, com_col, rotation_steps=180):
    """Rotation (deg) that minimizes the curl of the CoM field; tests transpose too."""
    angles = np.linspace(0, np.pi, rotation_steps, dtype=np.float32)
    if min(com_row.shape) < 3:
        r, c = _rotate_vector(com_row, com_col, angles[0])
        return r, c, 0.0, False
    curls = _rotation_curl_scores(com_row, com_col, angles)
    curls_t = _rotation_curl_scores(com_col, com_row, angles)
    stacked = np.concatenate([curls, curls_t])
    idx = int(stacked.argmin())
    use_transpose = idx >= rotation_steps
    ai = idx % rotation_steps
    angle_deg = float(angles[ai]) * 180.0 / np.pi
    if use_transpose:
        rt, ct = _rotate_vector(com_col, com_row, angles[ai])
        return rt, ct, angle_deg, True
    r, c = _rotate_vector(com_row, com_col, angles[ai])
    return r, c, angle_deg, False


def _integrate_gradient(grad_row, grad_col):
    """Fourier-integrate a gradient field with the shared Poisson solver."""
    grad_row = grad_row.astype(np.float32)
    grad_col = grad_col.astype(np.float32)
    g0 = np.fft.fft2(grad_row)
    g1 = np.fft.fft2(grad_col)
    k0, k1 = _freq_grid_2d(grad_row.shape)
    k2 = k0 ** 2 + k1 ** 2
    k2[0, 0] = 1.0
    phase_fft = (-1j * 0.25) * (k0 * g0 + k1 * g1) / k2
    phase_fft[0, 0] = 0
    return np.real(np.fft.ifft2(phase_fft)).astype(np.float32)


def integrate(com_row, com_col) -> np.ndarray:
    """Integrate row/column DPC gradients into a float32 phase image."""

    phase = _integrate_gradient(com_row, com_col)
    return (-(phase - phase.mean())).astype(np.float32, copy=False)


# --- CoM dispatch (the only step that touches the 4D block) ---


def center_of_mass(data, scan_shape=None, mask=None):
    """Per-scan-position CoM ``(com_row, com_col)``, mean-subtracted, ``(scan,scan)``.

    MPS chunked input -> raw-Metal ``com_u16`` (no-bin, no float cast). Array input
    (numpy/cupy/torch) -> chunked numpy/cupy CoM. Same formula either way.
    """
    from quantem.gpu.detector import prepare

    session = prepare(data)
    com_row, com_col = session.center_of_mass(mask)
    sr, sc = session.scan_shape if scan_shape is None else scan_shape
    if int(sr) * int(sc) != int(com_row.size):
        raise ValueError(
            f"scan_shape={(sr, sc)} does not match {com_row.size} detector frames."
        )
    com_row = np.asarray(com_row, dtype=np.float32) - float(np.mean(com_row))
    com_col = np.asarray(com_col, dtype=np.float32) - float(np.mean(com_col))
    return com_row.reshape(sr, sc), com_col.reshape(sr, sc)


def run(data, scan_shape=None, *, rotation_angle_deg=None, rotation_steps=180,
        mask=None, verbose=False) -> DPCResult:
    """Center-of-mass -> optimal scan/detector rotation -> iDPC phase.

    ``data`` is ``load(...)`` output (MPS chunks, cupy, or numpy). The CoM is the
    one pass over the 4D block; rotation + integration are small-field. View the
    result with ``Show2D`` (``result.phase`` for iDPC, ``result.com_col`` for the
    raw DPC field).
    """
    t0 = time.perf_counter()
    com_row, com_col = center_of_mass(data, scan_shape=scan_shape, mask=mask)
    if rotation_angle_deg is None:
        cr, cc, angle, transp = find_optimal_rotation(com_row, com_col, rotation_steps)
    else:
        a = np.radians(rotation_angle_deg)
        cr = np.cos(a) * com_row - np.sin(a) * com_col
        cc = np.sin(a) * com_row + np.cos(a) * com_col
        angle, transp = float(rotation_angle_deg), False
    # Match engine.dpc.DPC.reconstruct: when transpose was selected the alignment
    # swapped (k_col, k_row), so swap back before integrating; zero-mean the phase;
    # and negate it (STEM convention - atoms appear dark).
    grad_row, grad_col = (cc, cr) if transp else (cr, cc)
    phase = integrate(grad_row, grad_col)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"DPC: rotation {angle:.1f} deg (transpose={transp}), "
              f"{com_row.shape[0]}x{com_row.shape[1]} in {elapsed:.2f}s")
    return DPCResult(phase=phase, com_row=com_row, com_col=com_col,
                     com_row_aligned=cr.astype(np.float32), com_col_aligned=cc.astype(np.float32),
                     rotation_deg=angle, use_transpose=transp, elapsed=elapsed)
