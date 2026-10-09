"""CUDA probe detection used by exact SSB and parallax setup."""

import numpy as np

from quantem.gpu.device.cuda_runtime import cp


def detect_bf_radius(
    mean_dp,
    threshold_ratio: float = 0.1
) -> tuple[tuple[int, int], int]:
    """
    Detect BF disk center and radius from mean diffraction pattern.

    Runs entirely on GPU. Uses intensity thresholding for center-of-mass
    and radial profile analysis for the half-max radius.

    Parameters
    ----------
    mean_dp : cupy.ndarray
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
    # Threshold, centroid and radial profile run in float32 whatever the pattern's dtype.
    dp = mean_dp.astype(cp.float32)
    dp_max = float(cp.nanmax(dp))
    if not np.isfinite(dp_max) or dp_max <= 0:
        raise ValueError(
            "Diffraction pattern has no positive finite values - "
            "cannot detect BF disk. Check that your data is loaded correctly."
        )
    threshold = threshold_ratio * dp_max
    mask = dp > threshold
    if not bool(cp.any(mask)):
        raise ValueError(
            f"No pixels above threshold ({threshold_ratio:.0%} of max intensity). "
            f"The diffraction pattern may be too noisy or empty."
        )
    # The disk center is the unweighted centroid of the thresholded pixels.
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
    row_center = max(0, min(round(row_center_f), n_k_row - 1))
    col_center = max(0, min(round(col_center_f), n_k_col - 1))
    # The radius is where the azimuthal mean profile falls to half its central value.
    row_offsets = cp.arange(n_k_row, dtype=cp.float32) - row_center
    col_offsets = cp.arange(n_k_col, dtype=cp.float32) - col_center
    row_grid, col_grid = cp.meshgrid(row_offsets, col_offsets, indexing='ij')
    distance = cp.sqrt(row_grid**2 + col_grid**2)
    max_radius = min(row_center, col_center, n_k_row - row_center, n_k_col - col_center)
    if max_radius < 2:
        return (row_center, col_center), max(1, min(n_k_row, n_k_col) // 4)
    radius_bin = cp.rint(distance).astype(cp.int32).ravel()
    dp_flat = dp.ravel()
    profile = cp.zeros(max_radius, dtype=cp.float32)
    counts = cp.zeros(max_radius, dtype=cp.float32)
    valid = radius_bin < max_radius
    cp.add.at(profile, radius_bin[valid], dp_flat[valid])
    cp.add.at(counts, radius_bin[valid], cp.ones_like(dp_flat[valid]))
    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]
    # Smoothing keeps detector noise from triggering the half-maximum crossing early.
    if profile.size > 5:
        sigma = 2.0
        kernel_size = 13  # 6 sigma + 1, odd so the kernel has a center tap
        taps = cp.arange(kernel_size, dtype=cp.float32) - kernel_size // 2
        kernel = cp.exp(-0.5 * (taps / sigma) ** 2)
        kernel /= kernel.sum()
        padded = cp.pad(profile, kernel_size // 2, mode='edge')
        profile_smooth = cp.convolve(padded, kernel, mode='valid')[:profile.size]
        center_intensity = float(profile_smooth[:5].mean())
        half_max = center_intensity * 0.5
        below_half = cp.where(profile_smooth < half_max)[0]
        radius = int(below_half[0]) if below_half.size > 0 else profile.size // 2
    else:
        radius = min(n_k_row, n_k_col) // 4
    radius = max(1, radius)
    return (row_center, col_center), radius


def mean_dp(data):
    """
    Compute mean diffraction pattern on GPU.

    Uses integer reduction (``uint64`` accumulator) for integer counts so
    the sum is exact; floating-point input (simulated or normalised
    intensities) accumulates in float64, because a uint64 accumulator
    truncates every sub-unity value to 0. Either way there is no
    intermediate float32 copy of the full 4D array. For 512x512 x 192x192
    this saves ~38 GB of transient VRAM compared with
    ``data.astype(float32).mean(axis=0)``. The total is divided in float64
    and rounded once to float32, the rule of every mean pattern: rounding the
    total to float32 first would lose counts above 2^24.

    Parameters
    ----------
    data : cupy.ndarray
        3D ``(N, det_row, det_col)`` or 4D ``(scan_row, scan_col, det_row, det_col)``.

    Returns
    -------
    cupy.ndarray
        2D array (det_row, det_col), float32.
    """
    accumulator = cp.uint64 if data.dtype.kind in "ui" else cp.float64
    frames = data.reshape(-1, *data.shape[-2:])
    return (frames.sum(axis=0, dtype=accumulator) / frames.shape[0]).astype(cp.float32)
