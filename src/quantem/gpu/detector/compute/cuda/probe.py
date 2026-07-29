"""CUDA probe detection used by exact SSB and parallax setup."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cupy as cp


def detect_bf_radius(
    mean_dp: cp.ndarray,
    threshold_ratio: float = 0.1
) -> tuple[tuple[int, int], int]:
    """
    Detect BF disk center and radius from mean diffraction pattern.

    Runs entirely on GPU. Uses intensity thresholding for center-of-mass
    and radial profile analysis for the half-max radius.

    Parameters
    ----------
    mean_dp : cp.ndarray
        Mean diffraction pattern with shape (k_row, k_col).
    threshold_ratio : float
        Fraction of max intensity for thresholding (default: 0.1).

    Returns
    -------
    tuple[tuple[int, int], int]
        ((row_center, col_center), radius) - center coordinates and
        radius in pixels.

    Raises
    ------
    ValueError
        If the diffraction pattern is empty, all-zero, or contains
        only NaN/Inf values.
    """
    import cupy as cp
    if mean_dp.ndim != 2:
        raise ValueError(
            f"Expected 2D diffraction pattern, got {mean_dp.ndim}D "
            f"with shape {mean_dp.shape}"
        )
    n_k_row, n_k_col = mean_dp.shape
    if n_k_row == 0 or n_k_col == 0:
        raise ValueError(
            f"Diffraction pattern has zero-size dimension: shape {mean_dp.shape}"
        )
    dp = mean_dp.astype(cp.float32)
    dp_max = float(cp.nanmax(dp))
    if not np.isfinite(dp_max) or dp_max <= 0:
        raise ValueError(
            "Diffraction pattern has no positive finite values - "
            "cannot detect BF disk. Check that your data is loaded correctly."
        )
    # Threshold to find BF disk
    threshold = threshold_ratio * dp_max
    mask = dp > threshold
    if not bool(cp.any(mask)):
        raise ValueError(
            f"No pixels above threshold ({threshold_ratio:.0%} of max intensity). "
            f"The diffraction pattern may be too noisy or empty."
        )
    # Center of mass on GPU
    mask_f = mask.astype(cp.float32)
    total = float(mask_f.sum())
    row_coords = cp.arange(n_k_row, dtype=cp.float32).reshape(-1, 1)
    col_coords = cp.arange(n_k_col, dtype=cp.float32).reshape(1, -1)
    row_center_f = float((row_coords * mask_f).sum() / total)
    col_center_f = float((col_coords * mask_f).sum() / total)
    if not (np.isfinite(row_center_f) and np.isfinite(col_center_f)):
        raise ValueError(
            "Center-of-mass calculation returned NaN - "
            "diffraction pattern may be degenerate."
        )
    row_center = max(0, min(int(round(row_center_f)), n_k_row - 1))
    col_center = max(0, min(int(round(col_center_f)), n_k_col - 1))
    # Radial profile on GPU
    dr = cp.arange(n_k_row, dtype=cp.float32) - row_center
    dc = cp.arange(n_k_col, dtype=cp.float32) - col_center
    DR, DC = cp.meshgrid(dr, dc, indexing='ij')
    R = cp.sqrt(DR**2 + DC**2)
    # Integer-binned radial profile (vectorized, no Python loop)
    max_r = min(row_center, col_center, n_k_row - row_center, n_k_col - col_center)
    if max_r < 2:
        return (row_center, col_center), max(1, min(n_k_row, n_k_col) // 4)
    R_int = cp.rint(R).astype(cp.int32).ravel()
    dp_flat = dp.ravel()
    profile = cp.zeros(max_r, dtype=cp.float32)
    counts = cp.zeros(max_r, dtype=cp.float32)
    valid = R_int < max_r
    cp.add.at(profile, R_int[valid], dp_flat[valid])
    cp.add.at(counts, R_int[valid], cp.ones_like(dp_flat[valid]))
    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]
    # Gaussian smooth the profile on GPU (1D convolution)
    if int(profile.size) > 5:
        sigma = 2.0
        ksize = int(6 * sigma + 1) | 1  # ensure odd
        x = cp.arange(ksize, dtype=cp.float32) - ksize // 2
        kernel = cp.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        # Pad and convolve
        padded = cp.pad(profile, ksize // 2, mode='edge')
        profile_smooth = cp.convolve(padded, kernel, mode='valid')[:profile.size]
        center_intensity = float(profile_smooth[:5].mean())
        half_max = center_intensity * 0.5
        below_half = cp.where(profile_smooth < half_max)[0]
        if below_half.size > 0:
            radius = int(below_half[0])
        else:
            radius = int(profile.size) // 2
    else:
        radius = min(n_k_row, n_k_col) // 4
    radius = max(1, radius)
    return (row_center, col_center), radius


def mean_dp(data: cp.ndarray) -> cp.ndarray:
    """
    Compute mean diffraction pattern on GPU.

    Uses integer reduction (``uint64`` accumulator) so there is no
    intermediate float32 copy of the full 4D array. For 512x512 x 192x192
    this saves ~38 GB of transient VRAM compared with
    ``data.astype(float32).mean(axis=0)``.

    Parameters
    ----------
    data : cp.ndarray
        3D ``(N, det_row, det_col)`` or 4D ``(scan_row, scan_col, det_row, det_col)``.

    Returns
    -------
    cp.ndarray
        2D array (det_row, det_col), float32.
    """
    import cupy as cp
    if data.ndim == 3:
        n = data.shape[0]
        return data.sum(axis=0, dtype=cp.uint64).astype(cp.float32) / n
    scan_row, scan_col = data.shape[0], data.shape[1]
    n = scan_row * scan_col
    return data.reshape(n, *data.shape[2:]).sum(axis=0, dtype=cp.uint64).astype(cp.float32) / n

