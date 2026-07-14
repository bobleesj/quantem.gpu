"""Multiscale VBF alignment utilities (CuPy port of quantem's PyTorch implementation).

Ports _bin_mask_and_stack_centered, _make_periodic_pairs, _synchronize_shifts,
_compute_reference_shifts, _compute_pairwise_shifts, _fourier_shift_stack, and
align_vbf_stack_multiscale from quantem.diffractive_imaging.direct_ptycho_utils.
All computation stays on GPU via CuPy; only small deduplication ops use NumPy CPU side.
"""

import math

import cupy as cp
import numpy as np

from quantem.gpu.image.dft_upsample import cross_correlation_shift_cp, cross_correlation_shift_batch_cp


# =========================================================================
#  Pair construction
# =========================================================================

def make_periodic_pairs_cp(
    bf_mask: cp.ndarray,
    connectivity: int = 4,
    max_pairs: int | None = None,
) -> np.ndarray:
    """Build neighbor pairs from corner-centered mask with periodic wrapping.

    Parameters
    ----------
    bf_mask : cp.ndarray (bool)
        (Q, R) mask of valid positions (corner-centered grid).
    connectivity : int
        4 (axis-aligned) or 8 (includes diagonals).
    max_pairs : int, optional
        If given, randomly subsample to at most this many pairs.

    Returns
    -------
    pairs : np.ndarray, shape (M, 2), dtype int64
        Indices (in flattened valid-position order) of neighbor pairs.
    """
    Q, R = bf_mask.shape
    inds_i, inds_j = cp.where(bf_mask)
    N = int(inds_i.size)

    linear = -cp.ones((Q, R), dtype=cp.int64)
    linear[inds_i, inds_j] = cp.arange(N, dtype=cp.int64)

    if connectivity == 4:
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif connectivity == 8:
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    pair_list = []
    src_base = cp.arange(N, dtype=cp.int64)
    for di, dj in offsets:
        ni = (inds_i + di) % Q
        nj = (inds_j + dj) % R
        neighbor_idx = linear[ni, nj]
        valid = neighbor_idx >= 0
        src = src_base[valid]
        dst = neighbor_idx[valid]
        pair_list.append(cp.stack([src, dst], axis=1))

    pairs_gpu = cp.concatenate(pair_list, axis=0)
    # Sort each row so (i,j) with i<=j, then deduplicate on CPU (stable)
    pairs_gpu = cp.sort(pairs_gpu, axis=1)
    pairs_np = cp.asnumpy(pairs_gpu)
    pairs_np = np.unique(pairs_np, axis=0)

    if max_pairs is not None and len(pairs_np) > max_pairs:
        rng = np.random.default_rng()
        idx = rng.choice(len(pairs_np), max_pairs, replace=False)
        pairs_np = pairs_np[idx]

    return pairs_np.astype(np.int64)


# =========================================================================
#  Shift synchronization (graph Laplacian)
# =========================================================================

def synchronize_shifts_cp(
    num_nodes: int,
    rel_shifts: list,
) -> cp.ndarray:
    """Solve for absolute shifts given pairwise relative shifts.

    Builds a graph Laplacian system A·t = b and solves with gauge fix at node 0.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (vBF images).
    rel_shifts : list of (i, j, shift_array)
        Pairwise shift measurements; shift_array is shape (2,) with δ = t_j - t_i.

    Returns
    -------
    cp.ndarray, shape (N, 2), float32
        Absolute shift for each node.
    """
    N = num_nodes
    A = cp.zeros((N, N), dtype=cp.float64)
    b = cp.zeros((N, 2), dtype=cp.float64)
    for i, j, s in rel_shifts:
        s = cp.asarray(s, dtype=cp.float64)
        A[i, i] += 1
        A[j, j] += 1
        A[i, j] -= 1
        A[j, i] -= 1
        b[i] -= s
        b[j] += s
    # Gauge fix: anchor node 0
    A[0, :] = 0
    A[:, 0] = 0
    A[0, 0] = 1
    b[0] = 0
    t = cp.linalg.solve(A, b)
    return t.astype(cp.float32)


# =========================================================================
#  Reference and pairwise shift computation
# =========================================================================

def compute_reference_shifts_cp(
    vbf_stack: cp.ndarray,
    reference: cp.ndarray,
    upsample_factor: int = 4,
) -> cp.ndarray:
    """Align each image in the stack to a reference image.

    Uses fully batched GPU cross-correlation for maximum speed.

    Parameters
    ----------
    vbf_stack : cp.ndarray
        (N, H, W) stack of virtual BF images.
    reference : cp.ndarray
        (H, W) reference image.
    upsample_factor : int
        Upsampling factor for subpixel accuracy.

    Returns
    -------
    cp.ndarray, shape (N, 2), float32
        Measured shift for each image.
    """
    return cross_correlation_shift_batch_cp(reference, vbf_stack, upsample_factor).astype(cp.float32)


def compute_pairwise_shifts_cp(
    vbf_stack: cp.ndarray,
    pairs: np.ndarray,
    upsample_factor: int = 4,
) -> list:
    """Compute relative shifts between pairs of virtual BF images.

    Parameters
    ----------
    vbf_stack : cp.ndarray
        (N, H, W) stack of virtual BF images.
    pairs : array-like, shape (M, 2)
        Pairs of image indices to correlate.
    upsample_factor : int
        Upsampling factor for subpixel accuracy.

    Returns
    -------
    list of (i, j, shift_cp)
        Relative shift δ_ij = t_j - t_i for each pair.
    """
    rel_shifts = []
    for pair in pairs:
        i, j = int(pair[0]), int(pair[1])
        s_ij = cross_correlation_shift_cp(vbf_stack[i], vbf_stack[j], upsample_factor)
        rel_shifts.append((i, j, s_ij))
    return rel_shifts


# =========================================================================
#  Multiscale alignment
# =========================================================================

def align_vbf_stack_multiscale_cp(
    vbf_stack: cp.ndarray,
    bf_mask: cp.ndarray,
    inds_i: cp.ndarray,
    inds_j: cp.ndarray,
    bin_factors: tuple,
    pair_connectivity: int = 4,
    upsample_factor: int = 4,
    reference: cp.ndarray | None = None,
    initial_shifts: cp.ndarray | None = None,
    running_average: bool = False,
    basis: cp.ndarray | None = None,
    verbose: bool = True,
) -> tuple:
    """Align virtual BF stack using multi-scale coarse-to-fine approach.

    Parameters
    ----------
    vbf_stack : cp.ndarray
        (N, H, W) stack of virtual BF images.
    bf_mask : cp.ndarray (bool)
        (Q, R) corner-centered mask of valid BF positions.
    inds_i, inds_j : cp.ndarray
        Corner-centered coordinates for each vBF.
    bin_factors : tuple of int
        Sequence of binning factors from coarse to fine (e.g., (7, 5, 3, 1)).
    pair_connectivity : int
        Number of neighbors for pairwise alignment (4 or 8). Ignored if reference provided.
    upsample_factor : int
        Upsampling factor for subpixel accuracy.
    reference : cp.ndarray, optional
        (H, W) reference image. If None, uses pairwise graph synchronization.
    initial_shifts : cp.ndarray, optional
        (N, 2) shifts already applied. New shifts are accumulated onto these.
    running_average : bool
        If True in reference mode, updates reference as running average after each scale.
    basis : cp.ndarray, optional
        Basis for regularizing shifts via lstsq projection.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    global_shifts : cp.ndarray, shape (N, 2), float32
    aligned_stack : cp.ndarray, shape (N, H, W)
    """
    N, H, W = vbf_stack.shape

    if initial_shifts is None:
        global_shifts = cp.zeros((N, 2), dtype=cp.float32)
    else:
        global_shifts = cp.asarray(initial_shifts, dtype=cp.float32).copy()

    current_reference = reference.copy() if reference is not None else None

    # FFT the stack once upfront. Intermediate shifts are applied as phase ramps.
    # Only convert to real space for the binning/correlation steps.
    stack_fft = cp.fft.fft2(vbf_stack.astype(cp.float32), axes=(1, 2))
    f_row = cp.fft.fftfreq(H, d=1.0).astype(cp.float32).reshape(1, -1, 1)
    f_col = cp.fft.fftfreq(W, d=1.0).astype(cp.float32).reshape(1, 1, -1)

    for iteration_idx, bin_factor in enumerate(bin_factors):
        iteration = iteration_idx + 1

        # For binning/correlation we need real-space images. But only for the
        # BINNED subset — so bin in Fourier domain then IFFT only the bins.
        bf_mask_b, inds_ib, inds_jb, mapping = _bin_mapping_only(
            bf_mask, inds_i, inds_j, bin_factor
        )
        Nb = int(inds_ib.shape[0])

        # Sum FFT coefficients within each bin, then IFFT only Nb images.
        # cp.add.at doesn't support complex — split into real/imag parts.
        binned_real = cp.zeros((Nb, H, W), dtype=cp.float32)
        binned_imag = cp.zeros((Nb, H, W), dtype=cp.float32)
        cp.add.at(binned_real, mapping, stack_fft.real)
        cp.add.at(binned_imag, mapping, stack_fft.imag)
        vbf_binned_fft = binned_real + 1j * binned_imag
        del binned_real, binned_imag
        vbf_binned = cp.fft.ifft2(vbf_binned_fft, axes=(1, 2)).real
        del vbf_binned_fft

        if current_reference is not None:
            shifts = compute_reference_shifts_cp(vbf_binned, current_reference, upsample_factor)
        else:
            pairs = make_periodic_pairs_cp(bf_mask_b, connectivity=pair_connectivity)
            rel_shifts = compute_pairwise_shifts_cp(vbf_binned, pairs, upsample_factor)
            shifts = synchronize_shifts_cp(Nb, rel_shifts)

        incremental_shifts = shifts[mapping]

        if basis is not None:
            global_shifts_new = global_shifts + incremental_shifts
            basis_np = cp.asnumpy(basis)
            gsnew_np = cp.asnumpy(global_shifts_new.astype(cp.float64))
            coeffs_np = np.linalg.lstsq(basis_np, gsnew_np, rcond=None)[0]
            projected_np = basis_np @ coeffs_np
            projected = cp.asarray(projected_np, dtype=cp.float32)
            incremental_shifts = projected - global_shifts
            global_shifts = projected
        else:
            global_shifts = global_shifts + incremental_shifts

        # Apply incremental shifts as phase ramps in Fourier domain (no FFT+IFFT)
        drow = incremental_shifts[:, 0].reshape(-1, 1, 1)
        dcol = incremental_shifts[:, 1].reshape(-1, 1, 1)
        phase = cp.exp(-2j * cp.pi * (f_row * drow + f_col * dcol))
        stack_fft *= phase

        # Update reference
        if current_reference is not None:
            # Mean of shifted stack in real space = IFFT of mean of FFTs
            mean_fft = stack_fft.mean(axis=0)
            new_mean = cp.fft.ifft2(mean_fft).real
            if running_average:
                alpha = iteration / (iteration + 1)
                current_reference = current_reference * alpha + new_mean * (1 - alpha)
            else:
                current_reference = new_mean

    # Final IFFT
    vbf_stack = cp.fft.ifft2(stack_fft, axes=(1, 2)).real

    return global_shifts, vbf_stack


def _bin_mapping_only(bf_mask, inds_i, inds_j, bin_factor):
    """Compute binning mapping without touching the VBF stack.

    Returns (bf_mask_b, inds_ib, inds_jb, mapping) — no vbf_binned.
    """
    Q, R = bf_mask.shape
    N_orig = inds_i.size

    if bin_factor == 1:
        return bf_mask.copy(), inds_i.copy(), inds_j.copy(), cp.arange(N_orig, dtype=cp.int64)

    center_i = (inds_i + Q // 2) % Q
    center_j = (inds_j + R // 2) % R
    Qb = math.ceil(Q / bin_factor)
    Rb = math.ceil(R / bin_factor)
    offset = bin_factor // 2
    qb_center = ((center_i + offset) // bin_factor) % Qb
    rb_center = ((center_j + offset) // bin_factor) % Rb
    qb = (qb_center - Qb // 2) % Qb
    rb = (rb_center - Rb // 2) % Rb
    coords = (qb * Rb + rb).astype(cp.int64)
    coords_np = cp.asnumpy(coords)
    unique_coords_np, inverse_np = np.unique(coords_np, return_inverse=True)
    mapping = cp.asarray(inverse_np, dtype=cp.int64)
    unique_coords = cp.asarray(unique_coords_np, dtype=cp.int64)
    inds_ib = (unique_coords // Rb).astype(cp.int64)
    inds_jb = (unique_coords % Rb).astype(cp.int64)
    bf_mask_b = cp.zeros((Qb, Rb), dtype=cp.bool_)
    bf_mask_b[inds_ib, inds_jb] = True
    return bf_mask_b, inds_ib, inds_jb, mapping
