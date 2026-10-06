"""Parallax reconstruction on CUDA from a loaded 4D-STEM acquisition.

Each bright-field (BF) detector pixel records the scan image seen through a
slightly tilted beam. Defocus and astigmatism shift these images relative to
each other in proportion to the tilt, so measuring and undoing the shifts and
summing the images gives a phase-contrast image, and the shift pattern itself
measures the aberrations.

The reconstruction pipeline:
1. Read the scan image of every detector pixel inside the BF disk
2. Measure each image's shift by coarse-to-fine cross-correlation
3. Shift each image once by its measured shift and sum, on the upsampled grid
4. Optionally fit aberration coefficients from the shift pattern

References
----------
- Ophus et al., "Four-Dimensional Scanning Transmission Electron Microscopy (4D-STEM)"
- Yang et al., "Simultaneous atomic-resolution electron ptychography and Z-contrast imaging"
"""

import time

import cupy as cp
import numpy as np

from quantem.gpu import detector
from quantem.gpu.detector.cuda.probe import detect_bf_radius
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.optics.cuda import fit_aberrations_svd_polar
from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.parallax.cuda.alignment import align_vbf_stack_multiscale_cp
from quantem.gpu.parallax.results import ParallaxResult

# GPU bytes one batch holds: decoded detector values while reading the
# bright-field images, tiled spectra while summing them.
BATCH_BYTES = 256 * 1024**2


def run_cuda(
    data: Dataset4dstemGPU,
    *,
    center: tuple[int, int] | None,
    bf_radius: int | None,
    sampling_radius: int | None,
    voltage_kV: float,
    scan_sampling: float | None,
    upsampling_factor: int,
    fit_aberrations: bool,
    verbose: bool,
) -> ParallaxResult:
    """Reconstruct one CUDA-resident acquisition; see :func:`quantem.gpu.parallax.run`."""
    started = time.perf_counter()
    scan_rows, scan_cols, det_rows, det_cols = data.shape
    if center is None or bf_radius is None:
        detected_center, detected_radius = detect_bf_radius(cp.asarray(detector.mean(data)))
        if center is None:
            center = detected_center
        if bf_radius is None:
            bf_radius = detected_radius
        if verbose:
            print(f"[parallax] Auto-detected center={center}, bf_radius={bf_radius}")
    radius = int(bf_radius) if sampling_radius is None else sampling_radius
    rows = cp.arange(det_rows, dtype=cp.float32)[:, None]
    cols = cp.arange(det_cols, dtype=cp.float32)[None, :]
    bf_mask = (rows - float(center[0])) ** 2 + (cols - float(center[1])) ** 2 <= float(radius) ** 2

    # Read the scan image of every BF pixel, in row-major detector order. The
    # acquisition stays encoded: bands of scan rows cropped to the disk's
    # bounding box keep the decoded values under BATCH_BYTES, and uint8/uint16
    # counts convert to float32 exactly.
    pixel_rows, pixel_cols = cp.nonzero(bf_mask)
    row0, row1 = int(pixel_rows.min()), int(pixel_rows.max()) + 1
    col0, col1 = int(pixel_cols.min()), int(pixel_cols.max()) + 1
    box_pixels = (pixel_rows - row0) * (col1 - col0) + (pixel_cols - col0)
    n_pixels = len(box_pixels)
    stack = cp.empty((n_pixels, scan_rows, scan_cols), cp.float32)
    band_rows = max(1, BATCH_BYTES // (scan_cols * (row1 - row0) * (col1 - col0) * data.dtype.itemsize))
    for first in range(0, scan_rows, band_rows):
        stop = min(first + band_rows, scan_rows)
        band = cp.from_dlpack(
            data.read(
                scan_region=(first, stop, 0, scan_cols),
                detector_region=(row0, row1, col0, col1),
            )
        )
        stack[:, first:stop] = (
            band.reshape((stop - first) * scan_cols, -1)[:, box_pixels]
            .T.reshape(n_pixels, stop - first, scan_cols)
        )
    if verbose:
        print(f"[parallax] {n_pixels:,} BF pixels within {radius} px of {center}")

    # Measure each image's (row, col) shift against the mean image. Coarse-to-fine
    # detector binning, (3, 2, 1) as in QuantEM's align_vbf_stack_multiscale,
    # keeps large shifts from locking onto a wrong correlation peak.
    inds_row, inds_col = cp.where(bf_mask)
    global_shifts = align_vbf_stack_multiscale_cp(
        vbf_stack=stack,
        bf_mask=bf_mask,
        inds_row=inds_row,
        inds_col=inds_col,
        bin_factors=(3, 2, 1),
        reference=stack.mean(axis=0),
        upsample_factor=4,
    )
    shifts = [(float(row), float(col)) for row, col in cp.asnumpy(global_shifts)]
    if verbose:
        max_shift = float(cp.max(cp.sqrt(global_shifts[:, 0]**2 + global_shifts[:, 1]**2)))
        print(f"[parallax] Max shift: {max_shift:.2f} pixels")

    # Shift every image once by its measured shift and sum, in frequency space
    # on the output grid, as QuantEM's DirectPtychography.reconstruct does.
    # Tiling an image's spectrum upsampling_factor times per axis puts its scan
    # samples on the finer grid with zeros between them; the phase ramp then
    # moves the samples to their measured sub-pixel positions, so the images
    # interleave and the sum resolves detail finer than the scan step. The
    # ramp factorizes into a row part and a column part, in cycles per scan
    # pixel on the output grid.
    out_rows, out_cols = upsampling_factor * scan_rows, upsampling_factor * scan_cols
    shifts_f64 = global_shifts.astype(cp.float64)
    ramp_row = cp.exp(
        -2j * cp.pi * shifts_f64[:, 0:1] * cp.fft.fftfreq(out_rows, d=1.0 / upsampling_factor)[None, :]
    ).astype(cp.complex64)
    ramp_col = cp.exp(
        -2j * cp.pi * shifts_f64[:, 1:2] * cp.fft.fftfreq(out_cols, d=1.0 / upsampling_factor)[None, :]
    ).astype(cp.complex64)
    spectrum = cp.zeros((out_rows, out_cols), cp.complex64)
    batch = max(1, BATCH_BYTES // (out_rows * out_cols * 8))
    for first in range(0, n_pixels, batch):
        stop = min(first + batch, n_pixels)
        tiled = cp.tile(
            cp.fft.fft2(stack[first:stop], axes=(1, 2)), (1, upsampling_factor, upsampling_factor)
        )
        tiled *= ramp_row[first:stop, :, None]
        tiled *= ramp_col[first:stop, None, :]
        spectrum += tiled.sum(axis=0)
    image = cp.fft.ifft2(spectrum).real
    # The density is the same deposition of images of ones, whose spectra are
    # nonzero only at the multiples of the scan size: the number of samples
    # that reach each output pixel (n_pixels everywhere when not upsampled).
    density_spectrum = cp.zeros((out_rows, out_cols), cp.complex128)
    density_spectrum[::scan_rows, ::scan_cols] = scan_rows * scan_cols * (
        ramp_row[:, ::scan_rows].astype(cp.complex128).T @ ramp_col[:, ::scan_cols].astype(cp.complex128)
    )
    density = cp.fft.ifft2(density_spectrum).real.astype(cp.float32)

    # Defocus, astigmatism and scan rotation from the shift pattern, as in
    # QuantEM's fit_aberrations_from_shifts (SVD polar decomposition). The scan
    # sampling converts pixel shifts to Angstrom and sets the detector sampling.
    aberrations = None
    if fit_aberrations:
        delta_k_A = 1.0 / (max(det_rows, det_cols) * scan_sampling)
        aberrations = fit_aberrations_svd_polar(
            np.array(shifts, dtype=np.float64) * scan_sampling,
            cp.asnumpy(bf_mask),
            wavelength_A_from_kV(voltage_kV),
            (det_rows, det_cols),
            (1.0 / (det_rows * delta_k_A), 1.0 / (det_cols * delta_k_A)),
        )
    return ParallaxResult(
        image=image,
        density=density,
        shifts=shifts,
        aberrations=aberrations,
        elapsed=time.perf_counter() - started,
    )
