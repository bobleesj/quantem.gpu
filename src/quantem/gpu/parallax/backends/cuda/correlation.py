"""DFT matrix-multiply upsampling cross-correlation for subpixel image alignment.

Ports quantem's PyTorch implementation (Guizar-Sicairos 2008) to CuPy.
This gives much better subpixel accuracy than parabolic-only refinement.
"""

import math

import cupy as cp
import numpy as np


def dft_upsample_cp(
    imageCorr: cp.ndarray,
    upsampleFactor: int,
    xyShift: cp.ndarray,
) -> cp.ndarray:
    """Matrix-multiply DFT upsampling around a peak.

    Ports dftUpsample_torch exactly to CuPy.

    Parameters
    ----------
    imageCorr:
        (M, N) complex array in FT-domain (cross-correlation)
    upsampleFactor:
        Integer upsampling factor > 2
    xyShift:
        2-element array [x0, y0] giving the center of the upsampled patch

    Returns
    -------
    Real-valued upsampled correlation patch, shape (numRow, numCol)
    """
    M, N = imageCorr.shape
    pixelRadius = 1.5
    numRow = int(math.ceil(pixelRadius * upsampleFactor))
    numCol = numRow

    col_freq = cp.fft.ifftshift(cp.arange(N, dtype=np.float64)) - math.floor(N / 2)
    row_freq = cp.fft.ifftshift(cp.arange(M, dtype=np.float64)) - math.floor(M / 2)

    col_coords = cp.arange(numCol, dtype=np.float64) - float(xyShift[1])
    row_coords = cp.arange(numRow, dtype=np.float64) - float(xyShift[0])

    factor_col = -2j * math.pi / (N * float(upsampleFactor))
    colKern = cp.exp(factor_col * cp.outer(col_freq, col_coords)).astype(imageCorr.dtype)

    factor_row = -2j * math.pi / (M * float(upsampleFactor))
    rowKern = cp.exp(factor_row * cp.outer(row_coords, row_freq)).astype(imageCorr.dtype)

    imageUpsample = rowKern @ imageCorr @ colKern

    return imageUpsample.real


def _upsampled_correlation_cp(
    imageCorr: cp.ndarray,
    upsampleFactor: int,
    xyShift: cp.ndarray,
) -> cp.ndarray:
    """Refine a correlation peak via DFT upsampling.

    Ports upsampled_correlation_torch exactly to CuPy.

    Parameters
    ----------
    imageCorr:
        Complex FT-domain cross-correlation (G1 * conj(G2)), shape (M, N)
    upsampleFactor:
        Integer upsampling factor > 2
    xyShift:
        2-element float64 array (x, y) with half-pixel precision initial estimate

    Returns
    -------
    Refined 2-element float64 array (x, y)
    """
    assert upsampleFactor > 2

    xyShift = cp.round(xyShift * float(upsampleFactor)) / float(upsampleFactor)
    globalShift = math.floor(math.ceil(upsampleFactor * 1.5) / 2.0)

    upsampleCenter = globalShift - (upsampleFactor * xyShift)

    conj_input = imageCorr.conj()
    im_up = dft_upsample_cp(conj_input, upsampleFactor, upsampleCenter)
    imageCorrUpsample = im_up.conj()

    flat_idx = int(cp.argmax(imageCorrUpsample.real))
    r = flat_idx // imageCorrUpsample.shape[1]
    c = flat_idx % imageCorrUpsample.shape[1]
    xySubShift = cp.array([r, c], dtype=np.float64)

    dx = 0.0
    dy = 0.0
    try:
        patch = imageCorrUpsample.real[r - 1 : r + 2, c - 1 : c + 2]
        if patch.shape == (3, 3):
            icc = patch
            denom_x = float(4.0 * icc[1, 1] - 2.0 * icc[2, 1] - 2.0 * icc[0, 1])
            denom_y = float(4.0 * icc[1, 1] - 2.0 * icc[1, 2] - 2.0 * icc[1, 0])
            if denom_x != 0.0:
                dx = float((icc[2, 1] - icc[0, 1]) / denom_x)
            if denom_y != 0.0:
                dy = float((icc[1, 2] - icc[1, 0]) / denom_y)
    except Exception:
        pass

    xySubShift = xySubShift - float(globalShift)

    xyShift = xyShift + (xySubShift + cp.array([dx, dy], dtype=np.float64)) / float(upsampleFactor)

    return xyShift


def _align_images_fourier_cp(
    G1: cp.ndarray,
    G2: cp.ndarray,
    upsample_factor: int,
) -> cp.ndarray:
    """Align images via DFT upsampling of cross-correlation.

    Ports align_images_fourier_torch exactly to CuPy.

    Parameters
    ----------
    G1, G2:
        Complex FT arrays of same shape
    upsample_factor:
        Integer >= 1; if > 2, DFT upsampling is applied

    Returns
    -------
    2-element float64 array (x_shift, y_shift)
    """
    cc = G1 * G2.conj()
    cc_real = cp.fft.ifft2(cc).real

    flat_idx = int(cp.argmax(cc_real))
    M, N = cc_real.shape
    x0 = flat_idx // N
    y0 = flat_idx % N

    x_inds = [((x0 + dx) % M) for dx in (-1, 0, 1)]
    y_inds = [((y0 + dy) % N) for dy in (-1, 0, 1)]

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]

    denom_x = float(4.0 * vx[1] - 2.0 * vx[2] - 2.0 * vx[0])
    denom_y = float(4.0 * vy[1] - 2.0 * vy[2] - 2.0 * vy[0])
    dx = float((vx[2] - vx[0]) / denom_x) if denom_x != 0.0 else 0.0
    dy = float((vy[2] - vy[0]) / denom_y) if denom_y != 0.0 else 0.0

    x0 = float(cp.round(cp.array((x0 + dx) * 2.0)) / 2.0)
    y0 = float(cp.round(cp.array((y0 + dy) * 2.0)) / 2.0)

    xy_shift = cp.array([x0, y0], dtype=np.float64)

    if upsample_factor > 2:
        xy_shift = _upsampled_correlation_cp(cc, upsample_factor, xy_shift)

    return xy_shift


def cross_correlation_shift_batch_cp(
    ref: cp.ndarray,
    stack: cp.ndarray,
    upsample_factor: int = 4,
) -> cp.ndarray:
    """Measure shifts of all images in a stack relative to a reference.

    Fully batched GPU implementation — same algorithm as the sequential
    cross_correlation_shift_cp but 100-1000x faster for large stacks.

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
    x0 = (flat_idx // N_w).astype(cp.int64)
    y0 = (flat_idx % N_w).astype(cp.int64)

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
    num_row = int(math.ceil(pixel_radius * upsample_factor))
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

    # row_kern: (N, numRow, M) = exp(factor_row * row_coords[:,:,None] * row_freq[None,None,:])
    row_kern = cp.exp(factor_row * row_coords[:, :, None] * row_freq[None, None, :])
    # col_kern: (N, N_w, numCol) = exp(factor_col * col_freq[None,:,None] * col_coords[:,None,:])
    col_kern = cp.exp(factor_col * col_freq[None, :, None] * col_coords[:, None, :])

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
    xy_shift = xy_shift + (xy_sub + cp.stack([dx, dy], axis=1)) / float(upsample_factor)

    return xy_shift


def cross_correlation_shift_cp(
    im_ref: cp.ndarray,
    im: cp.ndarray,
    upsample_factor: int = 2,
) -> cp.ndarray:
    """Measure the shift between two images using Fourier cross-correlation.

    Ports cross_correlation_shift_torch exactly to CuPy.

    Parameters
    ----------
    im_ref:
        Reference image (real 2D array)
    im:
        Shifted image (real 2D array, same shape as im_ref)
    upsample_factor:
        Subpixel precision = 1/upsample_factor. If > 2, DFT upsampling is used.

    Returns
    -------
    2-element float64 array [dx, dy] — signed shift in pixels (row, col)
    """
    G1 = cp.fft.fft2(im_ref)
    G2 = cp.fft.fft2(im)

    xy_shift = _align_images_fourier_cp(G1, G2, upsample_factor)

    M, N = im_ref.shape
    dx = ((xy_shift[0] + M / 2) % M) - M / 2
    dy = ((xy_shift[1] + N / 2) % N) - N / 2

    return cp.array([dx, dy], dtype=np.float64)
