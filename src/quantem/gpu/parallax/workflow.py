"""Parallax reconstruction of a loaded 4D-STEM acquisition.

Each bright-field (BF) detector pixel records the scan image seen through a slightly tilted beam. Defocus and
astigmatism shift these images relative to each other in proportion to the tilt, so measuring and undoing the shifts
and summing the images gives a phase-contrast image, and the shift pattern itself measures the aberrations.

1. Read the scan image of every detector pixel inside the BF disk
2. Measure each image's shift by coarse-to-fine cross-correlation
3. Shift each image once by its measured shift and sum, on the upsampled grid
4. Optionally fit defocus, astigmatism and scan rotation from the shift pattern

References
----------
- Ophus et al., "Four-Dimensional Scanning Transmission Electron Microscopy (4D-STEM)"
- Yang et al., "Simultaneous atomic-resolution electron ptychography and Z-contrast imaging"
"""

import math
import time

import numpy as np
import torch

from quantem.gpu import detector
from quantem.gpu.detector.cuda.probe import detect_bf_radius
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.optics.physics import wrap_degrees
from quantem.gpu.parallax.cuda.alignment import align_vbf_stack_multiscale_cp
from quantem.gpu.parallax.results import ParallaxResult
from quantem.gpu.ssb.units import sampling_pair

# GPU bytes one batch holds: decoded detector values while reading the
# bright-field images, tiled spectra while summing them.
BATCH_BYTES = 256 * 1024**2


def run(
    data: Dataset4dstemGPU,
    *,
    bright_field_center_px: tuple[int, int] | None = None,
    bright_field_radius_px: float | None = None,
    upsampling_factor: int = 2,
    fit_aberrations: bool = False,
    scan_sampling_A: float | tuple[float, float] | None = None,
    semiangle_mrad: float | None = None,
    detector_sampling_mrad: tuple[float, float] | None = None,
    verbose: bool = False,
) -> ParallaxResult:
    """Reconstruct a parallax image from the bright-field disk of a 4D-STEM acquisition.

    Every detector pixel inside the bright-field disk sees the specimen through
    a slightly tilted beam, so defocus and astigmatism shift its scan image in
    proportion to that tilt. The shifts are measured by cross-correlation and
    undone, and the images are summed into one phase-contrast image. With
    ``fit_aberrations=True`` the shift pattern also gives defocus, two-fold
    astigmatism and the scan-detector rotation, in the units and convention
    ``SSB`` takes.

    The acquisition stays encoded on the GPU: only the bright-field pixels
    are read, in bounded bands of scan rows, and every count is used exactly.
    Parallax runs on CUDA only.

    Parameters
    ----------
    data
        The acquisition returned by ``quantem.gpu.io.load(path,
        backend="cuda")``, ordered ``(scan_row, scan_col, detector_row,
        detector_col)``.
    bright_field_center_px
        Bright-field disk center ``(row, col)`` in detector pixels. Detected
        from the mean diffraction pattern when omitted.
    bright_field_radius_px
        Radius in detector pixels of the bright-field pixels to use. The disk
        radius detected from the mean diffraction pattern when omitted; a
        smaller value bounds memory, since the image stack holds one float32
        scan image per pixel.
    upsampling_factor
        Each output image axis is this many times the scan axis. Every
        image's scan samples are placed at their measured sub-pixel shifts
        on the finer grid, so where the shifts differ (defocus) the images
        interleave and resolve detail finer than the scan step.
    fit_aberrations
        Fit ``C10_nm``, ``C12_nm``, ``phi12_deg`` and the scan-detector rotation from
        the shifts. Needs ``scan_sampling_A`` and the detector calibration,
        ``semiangle_mrad`` or ``detector_sampling_mrad``.
    scan_sampling_A
        Scan step in Angstrom per pixel, ``(row, col)``, or one number for both axes; converts the measured shifts
        to length.
    semiangle_mrad
        Probe convergence semiangle in mrad. Calibrates the detector as ``SSB``
        does: ``semiangle_mrad`` over the radius at which the bright-field disk
        of the mean pattern falls to half its plateau.
    detector_sampling_mrad
        Detector sampling in mrad per pixel, ``(row, col)`` or one number for both axes;
        replaces the calibration from ``semiangle_mrad``.
    verbose
        Print the detected disk and the alignment progress.

    Returns
    -------
    ParallaxResult
        The image (the sum of the bright-field images, each shifted once by
        its measured shift), its density map (the number of samples reaching
        each output pixel; ``image / density`` is the mean), the measured
        ``(row, col)`` shift of every bright-field pixel in scan pixels, and
        the fitted aberrations and rotation when requested.

    Raises
    ------
    TypeError
        If ``data`` is not a loaded acquisition.
    NotImplementedError
        If the acquisition is on an Apple GPU (MPS); parallax has no MPS
        implementation.
    ValueError
        If the acquisition is not on a CUDA GPU, ``fit_aberrations`` is
        requested without the scan or detector calibration, or
        ``scan_sampling_A`` or ``detector_sampling_mrad`` is neither one
        number nor a ``(row, col)`` pair.

    Examples
    --------
    >>> from quantem.gpu import SSB, io, parallax
    >>> data = io.load("scan_master.h5", backend="cuda")
    >>> result = parallax.run(data, fit_aberrations=True, scan_sampling_A=(0.5, 0.5), semiangle_mrad=30)
    >>> ssb = SSB(data, voltage_kV=300, semiangle_mrad=30, scan_sampling_A=(0.5, 0.5),
    ...           aberrations=result.aberrations, scan_detector_rotation_deg=result.scan_detector_rotation_deg)
    """
    if not isinstance(data, Dataset4dstemGPU):
        raise TypeError(
            "parallax.run takes the acquisition returned by "
            f"quantem.gpu.io.load(path, backend='cuda'); got {type(data).__name__}."
        )
    device = data.device
    if device is not None and device.type == "mps":
        raise NotImplementedError(
            "Parallax has no Apple GPU (MPS) implementation; it runs on CUDA only. "
            "Load the acquisition with io.load(path, backend='cuda') on a CUDA machine."
        )
    if device is None or device.type != "cuda":
        raise ValueError(
            f"Parallax runs on a CUDA GPU, but this acquisition is on {device}. "
            "Load it with io.load(path, backend='cuda')."
        )
    if fit_aberrations and (scan_sampling_A is None or (semiangle_mrad is None and detector_sampling_mrad is None)):
        raise ValueError(
            "fit_aberrations=True needs scan_sampling_A (Angstrom per scan pixel) and the detector "
            "calibration: semiangle_mrad, or detector_sampling_mrad in mrad per detector pixel."
        )
    # A sampling is a (row, col) pair, as SSB reads it: one number was broadcast to both axes here,
    # while SSB refuses it, so the same call meant two things in the two workflows.
    scan_sampling_A = None if scan_sampling_A is None else sampling_pair("scan_sampling_A", scan_sampling_A)
    detector_sampling_mrad = None if detector_sampling_mrad is None else sampling_pair("detector_sampling_mrad", detector_sampling_mrad)
    # CuPy allocates and launches on its current device and torch exports DLPack from its own: both are the
    # acquisition's GPU for the whole reconstruction, which need not be the current one
    with torch.cuda.device(device), cp.cuda.Device(device.index):
        started = time.perf_counter()
        det_rows, det_cols = data.shape[2:]
        mean_pattern = None
        center = bright_field_center_px
        if center is None or bright_field_radius_px is None:
            mean_pattern = detector.mean(data)
            detected_center, detected_radius = detect_bf_radius(cp.asarray(mean_pattern))
            center = detected_center if center is None else center
            bright_field_radius_px = detected_radius if bright_field_radius_px is None else bright_field_radius_px
            if verbose:
                print(f"[parallax] Auto-detected bright_field_center_px={center}, bright_field_radius_px={bright_field_radius_px}")
        # used as given: truncating a fractional radius selected other pixels than the caller asked for
        radius = bright_field_radius_px
        rows = cp.arange(det_rows, dtype=cp.float32)[:, None]
        cols = cp.arange(det_cols, dtype=cp.float32)[None, :]
        bf_mask = (rows - float(center[0])) ** 2 + (cols - float(center[1])) ** 2 <= float(radius) ** 2
        pixel_rows, pixel_cols = cp.nonzero(bf_mask)
        if verbose:
            print(f"[parallax] {len(pixel_rows):,} BF pixels within {radius} px of {center}")
        image, density, global_shifts = _shifted_sum(data, center, pixel_rows, pixel_cols, upsampling_factor, verbose)
        # The bright-field stack and its spectra were many times the image; return
        # their blocks to the GPU for the next consumer instead of keeping them cached.
        # CuPy's pool on this GPU and Torch's cache are emptied in place: device.release_cached_memory
        # also collects garbage, about 80 ms where this whole function takes 60 ms on a 256 x 256 scan.
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

        aberrations = scan_detector_rotation_deg = None
        if fit_aberrations:
            # SSB's detector calibration: the disk edge of the full mean pattern is the convergence semiangle.
            if detector_sampling_mrad is None:
                mean_pattern = detector.mean(data) if mean_pattern is None else mean_pattern
                detector_sampling_mrad = detector.detector_sampling_mrad(mean_pattern, semiangle_mrad)
            # The shifts are measured against the mean image, which is seen at the mean beam angle of the
            # selected pixels, so each pixel's beam angle is taken from their centroid.
            pixels = np.stack([cp.asnumpy(pixel_rows), cp.asnumpy(pixel_cols)], axis=1).astype(np.float64)
            # a step per axis: row shifts in row steps, column shifts in column steps
            angles_rad = (pixels - pixels.mean(axis=0)) * np.asarray(detector_sampling_mrad, dtype=np.float64) * 1e-3
            aberrations, signed_rotation_deg = fit_aberrations_from_shifts(
                cp.asnumpy(global_shifts).astype(np.float64) * np.asarray(scan_sampling_A, dtype=np.float64), angles_rad
            )
            # The fit keeps its angle within 90 degrees of zero; a result states the physical rotation in
            # [0, 360), the one range every result and record uses, so -20 reads 340.
            scan_detector_rotation_deg = wrap_degrees(signed_rotation_deg, 360.0)
        return ParallaxResult(
            image=image,
            density=density,
            shifts=[(float(row), float(col)) for row, col in cp.asnumpy(global_shifts)],
            aberrations=aberrations,
            scan_detector_rotation_deg=scan_detector_rotation_deg,
            elapsed_s=time.perf_counter() - started,
        )


def fit_aberrations_from_shifts(shifts_A, angles_rad) -> tuple[dict[str, float], float]:
    """Fit defocus, two-fold astigmatism and the scan-detector rotation to parallax shifts.

    With SSB's aberration function ``chi = (pi / lambda) alpha^T A alpha``,
    ``A = [[C10 + C12 cos 2 phi12, C12 sin 2 phi12], [C12 sin 2 phi12, C10 - C12 cos 2 phi12]]``,
    the bright-field image at beam angle ``alpha`` (radians, detector frame)
    sees the specimen ``A alpha`` away from the probe position, so the shift
    that aligns it, in the scan frame, is ``alpha @ M`` with ``M = A R``: the
    symmetric aberration matrix, then the scan-detector rotation ``R``. A
    least-squares fit of ``M`` and its polar decomposition ``M = R P`` (``P =
    R^T A R``, the aberrations in the scan frame) separate the two, as
    QuantEM's ``fit_aberrations_from_shifts`` does. The half turn ``(R, P) ->
    (-R, -P)`` fits equally well; the rotation is kept within 90 degrees of
    zero and the aberrations take its sign, the 180-degree ambiguity that
    ``SSB.find_aberrations(check_rotation=True)`` settles.

    Parameters
    ----------
    shifts_A
        ``(n, 2)`` measured ``(row, col)`` shift of each bright-field image in
        Angstrom.
    angles_rad
        ``(n, 2)`` ``(row, col)`` beam angle of each image's detector pixel in
        radians, relative to the beam angle of the reference image.

    Returns
    -------
    tuple
        ``{"C10_nm", "C12_nm", "phi12_deg"}`` in nm, nm and degrees (scan frame), and the
        rotation in degrees: what ``SSB(aberrations=..., scan_detector_rotation_deg=...)``
        takes.
    """
    matrix = np.linalg.lstsq(angles_rad, shifts_A, rcond=None)[0]
    left, singular, right = np.linalg.svd(matrix)
    rotation = left @ right
    aberration = right.T @ np.diag(singular) @ right
    rotation_rad = -math.atan2(rotation[1, 0], rotation[0, 0])
    if 2 * abs((rotation_rad + math.pi) % (2 * math.pi) - math.pi) > math.pi:
        rotation_rad = rotation_rad % (2 * math.pi) - math.pi
        aberration = -aberration
    a, b, c = aberration[0, 0], (aberration[1, 0] + aberration[0, 1]) / 2, aberration[1, 1]
    # A is in Angstrom like the shifts; SSB takes C10 and C12 in nm
    aberrations = {
        "C10_nm": float(a + c) / 20,
        "C12_nm": math.hypot((a - c) / 2, b) / 10,
        "phi12_deg": math.degrees(math.atan2(b, (a - c) / 2) / 2),
    }
    return aberrations, math.degrees(rotation_rad)


def _shifted_sum(data, center, pixel_rows, pixel_cols, upsampling_factor, verbose):
    """Read, align and sum the bright-field images: ``(image, density, shifts)``.

    A function of its own so that the image stack (one float32 scan image per
    bright-field pixel, gigabytes for a full disk) and its spectra are freed
    when it returns, before ``run`` hands their memory back to the GPU.
    """
    scan_rows, scan_cols, det_rows, det_cols = data.shape
    # Read the scan image of every BF pixel, in row-major detector order. The
    # acquisition stays encoded: bands of scan rows cropped to the disk's
    # bounding box keep the decoded values under BATCH_BYTES, and uint8/uint16
    # counts convert to float32 exactly.
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

    # Measure each image's (row, col) shift against the mean image. Coarse-to-fine
    # detector binning, (3, 2, 1) as in QuantEM's align_vbf_stack_multiscale,
    # keeps large shifts from locking onto a wrong correlation peak; the bins
    # are centred on the disk, as QuantEM's corner-centred mask places them.
    global_shifts = align_vbf_stack_multiscale_cp(
        vbf_stack=stack,
        inds_row=(pixel_rows - round(center[0])) % det_rows,
        inds_col=(pixel_cols - round(center[1])) % det_cols,
        detector_shape=(det_rows, det_cols),
        bin_factors=(3, 2, 1),
        reference=stack.mean(axis=0),
        upsample_factor=4,
    )
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
    # The ramps are evaluated in float64 and rounded once to complex64.
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
    # The density is the same deposition of images of ones, whose spectra are
    # nonzero only at the multiples of the scan size: the number of samples
    # that reach each output pixel (n_pixels everywhere when not upsampled).
    density_spectrum = cp.zeros((out_rows, out_cols), cp.complex128)
    density_spectrum[::scan_rows, ::scan_cols] = scan_rows * scan_cols * (
        ramp_row[:, ::scan_rows].astype(cp.complex128).T @ ramp_col[:, ::scan_cols].astype(cp.complex128)
    )
    return cp.fft.ifft2(spectrum).real, cp.fft.ifft2(density_spectrum).real.astype(cp.float32), global_shifts
