"""Coarse-to-fine alignment of the bright-field image stack (CuPy port of QuantEM's PyTorch implementation).

Ports the reference-image mode of ``align_vbf_stack_multiscale`` and
``_bin_mask_and_stack_centered`` from
``quantem.diffractive_imaging.direct_ptycho_utils``. At each detector binning,
coarse to fine, the binned images are correlated with the reference, the
shifts are applied to the stack's spectra as Fourier phase ramps, and the
reference becomes the mean of the shifted stack. Binning first keeps large
shifts from locking onto a wrong correlation peak.
"""

import math

import cupy as cp
import numpy as np

from quantem.gpu.parallax.cuda.correlation import cross_correlation_shift_batch_cp


def align_vbf_stack_multiscale_cp(
    vbf_stack: cp.ndarray,
    bf_mask: cp.ndarray,
    inds_row: cp.ndarray,
    inds_col: cp.ndarray,
    bin_factors: tuple,
    reference: cp.ndarray,
    upsample_factor: int = 4,
) -> cp.ndarray:
    """Measure the shift that aligns each virtual bright-field image to a reference, coarse to fine.

    Parameters
    ----------
    vbf_stack : cp.ndarray
        (N, H, W) float32 stack of virtual BF images; left unchanged.
    bf_mask : cp.ndarray (bool)
        (Q, R) mask of the BF pixels on the detector.
    inds_row, inds_col : cp.ndarray
        Detector (row, col) of each image of the stack.
    bin_factors : tuple of int
        Detector binning factors from coarse to fine (e.g., (3, 2, 1)).
    reference : cp.ndarray
        (H, W) reference image for the first binning.
    upsample_factor : int
        Upsampling factor for subpixel accuracy.

    Returns
    -------
    cp.ndarray, shape (N, 2), float32
        The (row, col) shift of every image in scan pixels: shifting image
        ``n`` by ``shifts[n]`` (the phase ramp ``exp(-2 pi i (f_row * row +
        f_col * col))``) aligns it with the reference. The caller applies the
        shifts once, to the unshifted images.
    """
    N, H, W = vbf_stack.shape
    global_shifts = cp.zeros((N, 2), dtype=cp.float32)
    current_reference = reference
    # The shifts found at one binning move these spectra by phase ramps before
    # the next binning, so the stack is Fourier transformed only once.
    stack_fft = cp.fft.fft2(vbf_stack, axes=(1, 2))
    # float32 frequencies keep the phase ramps complex64, the precision of the spectra.
    f_row = cp.fft.fftfreq(H, d=1.0).astype(cp.float32).reshape(1, -1, 1)
    f_col = cp.fft.fftfreq(W, d=1.0).astype(cp.float32).reshape(1, 1, -1)
    for level, bin_factor in enumerate(bin_factors):
        _, _, _, mapping = _bin_mapping_only(bf_mask, inds_row, inds_col, bin_factor)
        # Binning is linear, so each binned image is the inverse transform of
        # its members' summed spectra. Adding the k-th member of every bin in
        # one gather per k fixes the order of every float32 sum, so identical
        # runs give identical shifts (cp.add.at's atomic additions did not).
        binned_fft = stack_fft
        if bin_factor > 1:
            bin_of_image = cp.asnumpy(mapping)
            order = np.argsort(bin_of_image, kind="stable")
            counts = np.bincount(bin_of_image)
            starts = np.cumsum(counts) - counts
            binned_fft = stack_fft[cp.asarray(order[starts])]
            for member in range(1, int(counts.max())):
                bins = np.nonzero(counts > member)[0]
                binned_fft[cp.asarray(bins)] += stack_fft[cp.asarray(order[starts[bins] + member])]
        vbf_binned = cp.fft.ifft2(binned_fft, axes=(1, 2)).real
        # The shifts accumulate and return in float32.
        shifts = cross_correlation_shift_batch_cp(
            current_reference, vbf_binned, upsample_factor
        ).astype(cp.float32)
        incremental_shifts = shifts[mapping]
        global_shifts = global_shifts + incremental_shifts
        if level + 1 < len(bin_factors):
            drow = incremental_shifts[:, 0].reshape(-1, 1, 1)
            dcol = incremental_shifts[:, 1].reshape(-1, 1, 1)
            stack_fft *= cp.exp(-2j * cp.pi * (f_row * drow + f_col * dcol))
            # Mean of the shifted stack in real space = inverse transform of the mean spectrum.
            current_reference = cp.fft.ifft2(stack_fft.mean(axis=0)).real
    return global_shifts


def _bin_mapping_only(bf_mask, inds_i, inds_j, bin_factor):
    """Compute binning mapping without touching the VBF stack.

    Returns (bf_mask_b, inds_ib, inds_jb, mapping), no vbf_binned.
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
    inds_ib = unique_coords // Rb
    inds_jb = unique_coords % Rb
    bf_mask_b = cp.zeros((Qb, Rb), dtype=cp.bool_)
    bf_mask_b[inds_ib, inds_jb] = True
    return bf_mask_b, inds_ib, inds_jb, mapping
