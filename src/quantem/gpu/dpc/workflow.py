"""Differential phase contrast (CoM, DPC and iDPC) for 4D-STEM.

CoM, DPC and iDPC are derived scalar fields, not raw 4D data, so they live here
(viewed with ``Show2D``), separate from the raw ``Show4DSTEM`` viewer.

The only step that reads the 4D acquisition is the per-position centre of
mass, which the detector session computes where the data resides: exact
integer detector moments of encoded counts on CUDA and MPS, which give the same
float32 values, or the array formula for NumPy, CuPy and Torch arrays.
Everything after it (the rotation fit and the Fourier integration) is
small-field NumPy math on the ``(scan_row, scan_col)`` centre of mass. This
module is the canonical owner of that math across GPU workflows and dashboards.

Usage::

    from quantem.gpu import dpc, io
    from quantem.widget import Show2D
    result = dpc.run(io.load("scan_master.h5"))
    Show2D(result.phase)                      # the iDPC phase image
    Show2D(result.com_col)                    # raw DPC field (col)
"""
import time

import numpy as np

from quantem.gpu import detector
from quantem.gpu.dpc.results import DPCResult


def center_of_mass(data, scan_shape=None, mask=None):
    """Mean-subtracted detector centre of mass of every scan position.

    Parameters
    ----------
    data
        ``io.load`` output or a 4D array ordered as scan row, scan column,
        detector row, detector column.
    scan_shape : tuple of int, optional
        ``(rows, cols)`` to reshape the result to; defaults to the data's
        scan shape.
    mask : numpy.ndarray, optional
        Boolean detector mask in ``(row, col)`` order; only its pixels count.

    Returns
    -------
    tuple of numpy.ndarray
        float32 ``(com_row, com_col)`` in detector pixels with ``scan_shape``,
        each minus its mean over the scan.

    Examples
    --------
    >>> com_row, com_col = dpc.center_of_mass(io.load("scan_master.h5"))
    """
    session = detector.prepare(data)
    com_row, com_col = session.center_of_mass(mask)
    scan_rows, scan_cols = session.scan_shape if scan_shape is None else scan_shape
    if int(scan_rows) * int(scan_cols) != int(com_row.size):
        raise ValueError(
            f"scan_shape={(scan_rows, scan_cols)} does not match {com_row.size} detector frames."
        )
    com_row = np.asarray(com_row, dtype=np.float32) - float(np.mean(com_row))
    com_col = np.asarray(com_col, dtype=np.float32) - float(np.mean(com_col))
    return com_row.reshape(scan_rows, scan_cols), com_col.reshape(scan_rows, scan_cols)


def run(data, scan_shape=None, *, rotation_angle_deg=None, rotation_steps=180,
        mask=None, verbose=False) -> DPCResult:
    """Center-of-mass -> optimal scan/detector rotation -> iDPC phase.

    ``data`` is ``io.load`` output or a 4D array. The centre of mass is the one
    pass over the 4D block; rotation and integration are small-field. View the
    result with ``Show2D`` (``result.phase`` for iDPC, ``result.com_col`` for the
    raw DPC field).
    """
    started = time.perf_counter()
    com_row, com_col = center_of_mass(data, scan_shape=scan_shape, mask=mask)
    if rotation_angle_deg is None:
        aligned_row, aligned_col, angle, transposed = find_optimal_rotation(
            com_row, com_col, rotation_steps
        )
    else:
        radians = np.radians(rotation_angle_deg)
        aligned_row = np.cos(radians) * com_row - np.sin(radians) * com_col
        aligned_col = np.sin(radians) * com_row + np.cos(radians) * com_col
        angle, transposed = float(rotation_angle_deg), False
    # A transposed fit swapped the detector axes before aligning them; swap
    # them back so the gradients integrate along scan rows and columns.
    gradient_row, gradient_col = (
        (aligned_col, aligned_row) if transposed else (aligned_row, aligned_col)
    )
    phase = integrate(gradient_row, gradient_col)
    elapsed = time.perf_counter() - started
    if verbose:
        print(f"DPC: rotation {angle:.1f} deg (transpose={transposed}), "
              f"{com_row.shape[0]}x{com_row.shape[1]} in {elapsed:.2f}s")
    # A forced angle rotates in float64; every DPC product is float32.
    return DPCResult(phase=phase, com_row=com_row, com_col=com_col,
                     com_row_aligned=aligned_row.astype(np.float32),
                     com_col_aligned=aligned_col.astype(np.float32),
                     rotation_deg=angle, use_transpose=transposed, elapsed=elapsed)


def integrate(com_row, com_col) -> np.ndarray:
    """Integrate row/column DPC gradients into a float32 phase image.

    Solves the Poisson equation in Fourier space,
    ``phase(k) = -0.25 i (k_row G_row + k_col G_col) / |k|^2`` with the
    ``k = 0`` term set to 0, then zero-means and negates the phase (STEM
    convention: atoms appear dark).
    """
    spectrum_row = np.fft.fft2(com_row.astype(np.float32))
    spectrum_col = np.fft.fft2(com_col.astype(np.float32))
    k_row, k_col = np.meshgrid(
        np.fft.fftfreq(com_row.shape[0]).astype(np.float32),
        np.fft.fftfreq(com_row.shape[1]).astype(np.float32),
        indexing="ij",
    )
    k_squared = k_row ** 2 + k_col ** 2
    k_squared[0, 0] = 1.0
    phase_fft = (-1j * 0.25) * (k_row * spectrum_row + k_col * spectrum_col) / k_squared
    phase_fft[0, 0] = 0
    phase = np.real(np.fft.ifft2(phase_fft)).astype(np.float32)
    return -(phase - phase.mean())


def find_optimal_rotation(com_row, com_col, rotation_steps=180):
    """Find the scan-detector rotation that minimizes the curl of the CoM field.

    A pure phase object has a curl-free centre-of-mass field once the scan
    and detector axes agree. Tries ``rotation_steps`` angles in ``[0, 180]``
    degrees, with the detector axes as given and transposed.

    Returns
    -------
    tuple
        The rotated ``(row, col)`` field as float32, the angle in degrees and
        whether the detector axes were transposed.
    """
    angles = np.linspace(0, np.pi, rotation_steps, dtype=np.float32)
    if min(com_row.shape) < 3:
        rotated_row, rotated_col = _rotate_vector(com_row, com_col, angles[0])
        return rotated_row, rotated_col, 0.0, False
    scores = np.concatenate([
        _rotation_curl_scores(com_row, com_col, angles),
        _rotation_curl_scores(com_col, com_row, angles),
    ])
    best = int(scores.argmin())
    transposed = best >= rotation_steps
    angle = angles[best % rotation_steps]
    source_row, source_col = (com_col, com_row) if transposed else (com_row, com_col)
    rotated_row, rotated_col = _rotate_vector(source_row, source_col, angle)
    return rotated_row, rotated_col, float(angle) * 180.0 / np.pi, transposed


def _rotation_curl_scores(v_row, v_col, angles_rad):
    """Mean squared curl of the rotated field for many angles without rotated copies.

    For the field rotated by ``a``, ``curl = cos(a) curl(r, c) + sin(a) div(r, c)``,
    so the mean squared curl at every angle follows from three scalar moments of
    the unrotated curl and divergence instead of one full map per angle.
    """
    curl = (
        0.5 * (v_col[2:, 1:-1] - v_col[:-2, 1:-1])
        - 0.5 * (v_row[1:-1, 2:] - v_row[1:-1, :-2])
    ).astype(np.float64, copy=False)
    divergence = (
        0.5 * (v_row[2:, 1:-1] - v_row[:-2, 1:-1])
        + 0.5 * (v_col[1:-1, 2:] - v_col[1:-1, :-2])
    ).astype(np.float64, copy=False)
    curl_power = float(np.mean(curl * curl))
    divergence_power = float(np.mean(divergence * divergence))
    cross_power = float(np.mean(curl * divergence))
    cosine = np.cos(angles_rad, dtype=np.float64)
    sine = np.sin(angles_rad, dtype=np.float64)
    return (
        cosine * cosine * curl_power
        + sine * sine * divergence_power
        + 2.0 * cosine * sine * cross_power
    )


def _rotate_vector(v_row, v_col, angle_rad):
    """Rotate a ``(row, col)`` vector field by ``angle_rad`` into float32."""
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return (
        (cosine * v_row - sine * v_col).astype(np.float32, copy=False),
        (sine * v_row + cosine * v_col).astype(np.float32, copy=False),
    )
