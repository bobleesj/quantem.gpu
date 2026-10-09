"""DFT matrix-multiply upsampling cross-correlation for subpixel image alignment.

Ports quantem's PyTorch implementation (Guizar-Sicairos 2008) to CuPy.
This gives much better subpixel accuracy than parabolic-only refinement.
"""

import math

import cupy as cp


def cross_correlation_shift_batch_cp(
    ref: cp.ndarray,
    stack: cp.ndarray,
    upsample_factor: int = 4,
) -> cp.ndarray:
    """Measure shifts of all images in a stack relative to a reference.

    Ports cross_correlation_shift_torch, batched over the stack: Fourier
    cross-correlation, a parabolic half-pixel estimate, then DFT upsampling
    around the peak.

    Parameters
    ----------
    ref : cp.ndarray
        (H, W) reference image.
    stack : cp.ndarray
        (N, H, W) image stack.
    upsample_factor : int
        Subpixel precision = 1/upsample_factor.

    Returns
    -------
    cp.ndarray, shape (N, 2), float64
        Shifts [dx, dy] for each image.
    """
    N, M_h, N_w = stack.shape

    # Step 1: Batched FFT cross-correlation
    ref_fft = cp.fft.fft2(ref)  # (H, W)
    stack_fft = cp.fft.fft2(stack, axes=(1, 2))  # (N, H, W)
    cc = ref_fft[None, :, :] * cp.conj(stack_fft)  # (N, H, W)
    cc_real = cp.fft.ifft2(cc, axes=(1, 2)).real  # (N, H, W)

    # Step 2: Batched argmax
    flat_idx = cp.argmax(cc_real.reshape(N, -1), axis=1)  # (N,)
    x0 = flat_idx // N_w
    y0 = flat_idx % N_w

    # Step 3: Batched parabolic refinement
    idx = cp.arange(N)
    xm = (x0 - 1) % M_h
    xp = (x0 + 1) % M_h
    ym = (y0 - 1) % N_w
    yp = (y0 + 1) % N_w

    vx0 = cc_real[idx, xm, y0]
    vx1 = cc_real[idx, x0, y0]
    vx2 = cc_real[idx, xp, y0]
    vy0 = cc_real[idx, x0, ym]
    vy1 = vx1
    vy2 = cc_real[idx, x0, yp]

    denom_x = 4.0 * vx1 - 2.0 * vx2 - 2.0 * vx0
    denom_y = 4.0 * vy1 - 2.0 * vy2 - 2.0 * vy0
    dx = cp.where(denom_x != 0, (vx2 - vx0) / denom_x, 0.0)
    dy = cp.where(denom_y != 0, (vy2 - vy0) / denom_y, 0.0)

    # Round to half-pixel
    x0f = cp.round((x0.astype(cp.float64) + dx) * 2.0) / 2.0
    y0f = cp.round((y0.astype(cp.float64) + dy) * 2.0) / 2.0

    xy_shift = cp.stack([x0f, y0f], axis=1)  # (N, 2)

    # Step 4: Batched DFT upsample refinement
    if upsample_factor > 2:
        xy_shift = _upsampled_correlation_batch_cp(cc, upsample_factor, xy_shift)

    # Step 5: Wrap to [-M/2, M/2)
    xy_shift[:, 0] = ((xy_shift[:, 0] + M_h / 2) % M_h) - M_h / 2
    xy_shift[:, 1] = ((xy_shift[:, 1] + N_w / 2) % N_w) - N_w / 2

    return xy_shift


def _upsampled_correlation_batch_cp(
    cc_batch: cp.ndarray,
    upsample_factor: int,
    xy_shift: cp.ndarray,
) -> cp.ndarray:
    """Batched DFT upsample refinement for N cross-correlations.

    Parameters
    ----------
    cc_batch : cp.ndarray
        (N, M, N_w) complex cross-correlation arrays.
    upsample_factor : int
        Must be > 2.
    xy_shift : cp.ndarray
        (N, 2) initial peak estimates (half-pixel precision).

    Returns
    -------
    cp.ndarray, shape (N, 2)
        Refined shifts.
    """
    N_img, M, N_w = cc_batch.shape
    pixel_radius = 1.5
    num_row = math.ceil(pixel_radius * upsample_factor)
    num_col = num_row

    # Round shifts to nearest 1/upsample_factor
    xy_shift = cp.round(xy_shift * float(upsample_factor)) / float(upsample_factor)
    global_shift = float(math.floor(math.ceil(upsample_factor * 1.5) / 2.0))
    upsample_center = global_shift - upsample_factor * xy_shift  # (N, 2)

    # Shared frequency vectors
    col_freq = cp.fft.ifftshift(cp.arange(N_w, dtype=cp.float64)) - math.floor(N_w / 2)
    row_freq = cp.fft.ifftshift(cp.arange(M, dtype=cp.float64)) - math.floor(M / 2)

    # Per-image coordinates: (N, numRow) and (N, numCol)
    base_row = cp.arange(num_row, dtype=cp.float64)  # (numRow,)
    base_col = cp.arange(num_col, dtype=cp.float64)  # (numCol,)
    row_coords = base_row[None, :] - upsample_center[:, 0:1]  # (N, numRow)
    col_coords = base_col[None, :] - upsample_center[:, 1:2]  # (N, numCol)

    # Batched DFT kernels
    factor_row = -2j * math.pi / (M * float(upsample_factor))
    factor_col = -2j * math.pi / (N_w * float(upsample_factor))

    row_kern = cp.exp(factor_row * row_coords[:, :, None] * row_freq[None, None, :])  # (N, numRow, M)
    col_kern = cp.exp(factor_col * col_freq[None, :, None] * col_coords[:, None, :])  # (N, N_w, numCol)

    # Cast to match cc dtype for matmul
    cc_conj = cp.conj(cc_batch)  # (N, M, N_w)
    row_kern = row_kern.astype(cc_conj.dtype)
    col_kern = col_kern.astype(cc_conj.dtype)

    # Batched matmul: (N, numRow, M) @ (N, M, N_w) @ (N, N_w, numCol) → (N, numRow, numCol)
    temp = cp.matmul(cc_conj, col_kern)  # (N, M, numCol)
    upsampled = cp.matmul(row_kern, temp)  # (N, numRow, numCol)
    upsampled = cp.conj(upsampled).real  # (N, numRow, numCol)

    # Batched argmax on the small patches
    flat_idx = cp.argmax(upsampled.reshape(N_img, -1), axis=1)  # (N,)
    r = flat_idx // num_col
    c = flat_idx % num_col
    xy_sub = cp.stack([r.astype(cp.float64), c.astype(cp.float64)], axis=1)  # (N, 2)

    # Batched parabolic refinement on 3x3 patch
    # Only valid if peak is not on the edge
    dx = cp.zeros(N_img, dtype=cp.float64)
    dy = cp.zeros(N_img, dtype=cp.float64)
    valid = (r >= 1) & (r < num_row - 1) & (c >= 1) & (c < num_col - 1)
    if cp.any(valid):
        idx_v = cp.where(valid)[0]
        rv = r[idx_v]
        cv = c[idx_v]
        v_center = upsampled[idx_v, rv, cv]
        v_rm = upsampled[idx_v, rv - 1, cv]
        v_rp = upsampled[idx_v, rv + 1, cv]
        v_cm = upsampled[idx_v, rv, cv - 1]
        v_cp_val = upsampled[idx_v, rv, cv + 1]
        denom_r = 4.0 * v_center - 2.0 * v_rp - 2.0 * v_rm
        denom_c = 4.0 * v_center - 2.0 * v_cp_val - 2.0 * v_cm
        dx[idx_v] = cp.where(denom_r != 0, (v_rp - v_rm) / denom_r, 0.0)
        dy[idx_v] = cp.where(denom_c != 0, (v_cp_val - v_cm) / denom_c, 0.0)

    xy_sub = xy_sub - global_shift
    return xy_shift + (xy_sub + cp.stack([dx, dy], axis=1)) / float(upsample_factor)
