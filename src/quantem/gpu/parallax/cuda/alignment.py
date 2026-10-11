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

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.parallax.cuda.correlation import cross_correlation_shift_batch_cp


def align_vbf_stack_multiscale_cp(
    vbf_stack,
    inds_row,
    inds_col,
    detector_shape: tuple[int, int],
    bin_factors: tuple,
    reference,
    upsample_factor: int = 4,
):
    """Measure the shift that aligns each virtual bright-field image to a reference, coarse to fine.

    Parameters
    ----------
    vbf_stack : cp.ndarray
        (N, H, W) float32 stack of virtual BF images; left unchanged.
    inds_row, inds_col : cp.ndarray
        Detector (row, col) offset of each image's pixel from the disk
        centre, wrapped by the detector size (QuantEM's corner-centred
        indices), so the detector bins are centred on the disk.
    detector_shape : tuple of int
        (Q, R) detector size the offsets wrap by.
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
        mapping = bin_mapping(inds_row, inds_col, detector_shape, bin_factor)
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


def bin_mapping(inds_row, inds_col, detector_shape, bin_factor):
    """Detector bin of every bright-field pixel when the detector is binned by ``bin_factor`` around the disk.

    The pixel-to-bin map of QuantEM's ``_bin_mask_and_stack_centered``: the
    corner-centred indices are moved by half the detector, which puts the
    disk in the middle, so no bin straddles the wrap and the bin at the disk
    centre is whole. Bins are numbered in order of their position.
    """
    if bin_factor == 1:
        return cp.arange(inds_row.size, dtype=cp.int64)
    rows, cols = detector_shape
    binned_rows, binned_cols = math.ceil(rows / bin_factor), math.ceil(cols / bin_factor)
    offset = bin_factor // 2
    bin_row = (((inds_row + rows // 2) % rows + offset) // bin_factor % binned_rows - binned_rows // 2) % binned_rows
    bin_col = (((inds_col + cols // 2) % cols + offset) // bin_factor % binned_cols - binned_cols // 2) % binned_cols
    bins = cp.asnumpy((bin_row * binned_cols + bin_col).astype(cp.int64))
    return cp.asarray(np.unique(bins, return_inverse=True)[1], dtype=cp.int64)
