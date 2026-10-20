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
import math
import time

import numpy as np

from quantem.gpu import detector
from quantem.gpu.dpc.results import DPCResult
from quantem.gpu.optics.physics import wrap_degrees


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


def run(data, scan_shape=None, *, scan_detector_rotation_deg=None, scan_detector_rotation_flipped=False,
        mask=None, verbose=False) -> DPCResult:
    """Center-of-mass -> optimal scan/detector rotation -> iDPC phase.

    ``data`` is ``io.load`` output or a 4D array. The centre of mass is the one
    pass over the 4D block; rotation and integration are small-field. View the
    result with ``Show2D`` (``result.phase`` for iDPC, ``result.com_col`` for the
    raw DPC field).

    Without ``scan_detector_rotation_deg`` the rotation and whether it is
    flipped (detector row and column axes swapped) are fitted from the curl, at
    the exact minimum of the mean squared curl (``find_optimal_rotation``); the
    result then holds ``scan_detector_rotation_mod180_deg``. With it, the given
    physical rotation and ``scan_detector_rotation_flipped`` are applied and
    nothing is fitted.
    """
    started = time.perf_counter()
    com_row, com_col = center_of_mass(data, scan_shape=scan_shape, mask=mask)
    if scan_detector_rotation_deg is None:
        aligned_row, aligned_col, angle, transposed = find_optimal_rotation(com_row, com_col)
    else:
        source_row, source_col = (com_col, com_row) if scan_detector_rotation_flipped else (com_row, com_col)
        radians = np.radians(scan_detector_rotation_deg)
        aligned_row = np.cos(radians) * source_row - np.sin(radians) * source_col
        aligned_col = np.sin(radians) * source_row + np.cos(radians) * source_col
        angle, transposed = float(scan_detector_rotation_deg), bool(scan_detector_rotation_flipped)
    # A transposed fit swapped the detector axes before aligning them; swap
    # them back so the gradients integrate along scan rows and columns.
    gradient_row, gradient_col = (
        (aligned_col, aligned_row) if transposed else (aligned_row, aligned_col)
    )
    phase = integrate(gradient_row, gradient_col)
    elapsed_s = time.perf_counter() - started
    if verbose:
        print(f"DPC: rotation {angle:.1f}° (rotation flipped: {'yes' if transposed else 'no'}), "
              f"{com_row.shape[0]}x{com_row.shape[1]} in {elapsed_s:.2f}s")
    # A forced angle rotates in float64; every DPC product is float32.
    return DPCResult(phase=phase, com_row=com_row, com_col=com_col,
                     com_row_aligned=aligned_row.astype(np.float32),
                     com_col_aligned=aligned_col.astype(np.float32),
                     scan_detector_rotation_mod180_deg=angle if scan_detector_rotation_deg is None else None,
                     scan_detector_rotation_deg=None if scan_detector_rotation_deg is None else wrap_degrees(angle, 360.0),
                     scan_detector_rotation_flipped=transposed, elapsed_s=elapsed_s)


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


def find_optimal_rotation(com_row, com_col, *, scan_detector_rotation_flipped=None):
    """Find the scan-detector rotation that minimizes the curl of the CoM field, exactly.

    A pure phase object has a curl-free centre-of-mass field once the scan and detector axes agree. Rotating the
    field by ``a`` mixes its curl and divergence, ``curl_a = cos(a) curl + sin(a) div``, so the mean squared curl is
    ``S(a) = M + R cos(2a - psi)`` with ``M``, ``R`` and ``psi`` set by three moments of the unrotated curl and
    divergence. Its minimum, ``a = (psi + pi) / 2`` modulo 180 degrees, is computed in closed form, with no angle
    grid: the estimate is the objective's own minimizer rather than the nearest of a few hundred trial angles.

    Parameters
    ----------
    com_row, com_col : numpy.ndarray
        Mean-subtracted centre of mass in detector pixels, ``(scan_row, scan_col)``.
    scan_detector_rotation_flipped : bool, optional
        None (default) also tries the detector row and column axes swapped and keeps the lower curl. A bool keeps
        that convention and fits the angle alone.

    Returns
    -------
    tuple
        The rotated ``(row, col)`` field as float32, the rotation in degrees known modulo 180
        (``scan_detector_rotation_mod180_deg``, in [0, 180)) and whether the rotation is flipped, the detector row
        and column axes swapped (``scan_detector_rotation_flipped``).
    """
    if min(com_row.shape) < 3:
        rotated_row, rotated_col = _rotate_vector(com_row, com_col, 0.0)
        return rotated_row, rotated_col, 0.0, bool(scan_detector_rotation_flipped)
    candidates = (False, True) if scan_detector_rotation_flipped is None else (bool(scan_detector_rotation_flipped),)
    best = None
    for flipped in candidates:
        source = (com_col, com_row) if flipped else (com_row, com_col)
        angle_rad, score = _curl_minimum(*source)
        # the axes as given win a tie, as the first candidate
        if best is None or score < best[2]:
            best = (flipped, angle_rad, score)
    flipped, angle_rad, _ = best
    source_row, source_col = (com_col, com_row) if flipped else (com_row, com_col)
    rotated_row, rotated_col = _rotate_vector(source_row, source_col, angle_rad)
    angle_deg = math.degrees(angle_rad) % 180.0
    return rotated_row, rotated_col, 0.0 if angle_deg == 180.0 else angle_deg, flipped


def _curl_moments(v_row, v_col) -> tuple[float, float, float]:
    """Mean squared curl, mean squared divergence and their mean product of a ``(row, col)`` field (central differences)."""
    curl = (
        0.5 * (v_col[2:, 1:-1] - v_col[:-2, 1:-1])
        - 0.5 * (v_row[1:-1, 2:] - v_row[1:-1, :-2])
    ).astype(np.float64, copy=False)
    divergence = (
        0.5 * (v_row[2:, 1:-1] - v_row[:-2, 1:-1])
        + 0.5 * (v_col[1:-1, 2:] - v_col[1:-1, :-2])
    ).astype(np.float64, copy=False)
    return float(np.mean(curl * curl)), float(np.mean(divergence * divergence)), float(np.mean(curl * divergence))


def _curl_minimum(v_row, v_col) -> tuple[float, float]:
    """The rotation in radians, in [0, pi), that minimizes the mean squared curl of the rotated field, and that minimum.

    ``S(a) = P_c cos^2 a + P_d sin^2 a + 2 P_x sin a cos a = M + A cos 2a + B sin 2a`` with ``M = (P_c + P_d) / 2``,
    ``A = (P_c - P_d) / 2`` and ``B = P_x``; its minimum ``M - hypot(A, B)`` lies at ``2a = atan2(B, A) + pi``.
    A field with no preferred angle (``A = B = 0``) reports 0.
    """
    curl_power, divergence_power, cross_power = _curl_moments(v_row, v_col)
    half_difference = 0.5 * (curl_power - divergence_power)
    amplitude = math.hypot(half_difference, cross_power)
    minimum = 0.5 * (curl_power + divergence_power) - amplitude
    if amplitude == 0.0:
        return 0.0, minimum
    return (0.5 * (math.atan2(cross_power, half_difference) + math.pi)) % math.pi, minimum


def _rotate_vector(v_row, v_col, angle_rad):
    """Rotate a ``(row, col)`` vector field by ``angle_rad`` into float32."""
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return (
        (cosine * v_row - sine * v_col).astype(np.float32, copy=False),
        (sine * v_row + cosine * v_col).astype(np.float32, copy=False),
    )
