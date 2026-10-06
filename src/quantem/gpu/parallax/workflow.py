"""Parallax reconstruction of a loaded 4D-STEM acquisition."""

from quantem.gpu.device import release_cached_memory
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.parallax.results import ParallaxResult


def run(
    data: Dataset4dstemGPU,
    *,
    center: tuple[int, int] | None = None,
    bf_radius: int | None = None,
    sampling_radius: int | None = None,
    voltage_kV: float = 300,
    scan_sampling: float | None = None,
    upsampling_factor: int = 2,
    fit_aberrations: bool = False,
    verbose: bool = False,
) -> ParallaxResult:
    """Reconstruct a parallax image from the bright-field disk of a 4D-STEM acquisition.

    Every detector pixel inside the bright-field disk sees the specimen through
    a slightly tilted beam, so defocus and astigmatism shift its scan image in
    proportion to that tilt. The shifts are measured by cross-correlation and
    undone, and the images are summed into one phase-contrast image. With
    ``fit_aberrations=True`` the shift pattern also gives defocus, two-fold
    astigmatism and the scan-detector rotation.

    The acquisition stays encoded on the GPU: only the bright-field pixels
    are read, in bounded bands of scan rows, and every count is used exactly.
    Parallax runs on CUDA only.

    Parameters
    ----------
    data
        The acquisition returned by ``quantem.gpu.io.load(path,
        backend="cuda")``, ordered ``(scan_row, scan_col, detector_row,
        detector_col)``.
    center
        Bright-field disk center ``(row, col)`` in detector pixels. Detected
        from the mean diffraction pattern when omitted.
    bf_radius
        Bright-field disk radius in detector pixels. Detected from the mean
        diffraction pattern when omitted.
    sampling_radius
        Use only the pixels within this radius instead of ``bf_radius``, for
        example to bound memory: the image stack holds one float32 scan image
        per pixel.
    voltage_kV
        Accelerating voltage in kV, used for the aberration fit.
    scan_sampling
        Scan step in Angstrom per pixel. Required with ``fit_aberrations``.
    upsampling_factor
        Each output image axis is this many times the scan axis. Every
        image's scan samples are placed at their measured sub-pixel shifts
        on the finer grid, so where the shifts differ (defocus) the images
        interleave and resolve detail finer than the scan step.
    fit_aberrations
        Fit ``C10``, ``C12``, ``phi12`` and ``rotation_angle`` from the shifts.
    verbose
        Print the detected disk and the alignment progress.

    Returns
    -------
    ParallaxResult
        The image (the sum of the bright-field images, each shifted once by
        its measured shift), its density map (the number of samples reaching
        each output pixel; ``image / density`` is the mean), the measured
        ``(row, col)`` shift of every bright-field pixel in scan pixels, and
        the fitted aberrations when requested.

    Raises
    ------
    TypeError
        If ``data`` is not a loaded acquisition.
    NotImplementedError
        If the acquisition is on an Apple GPU (MPS); parallax has no MPS
        implementation.
    ValueError
        If the acquisition is not on a CUDA GPU, or ``fit_aberrations`` is
        requested without ``scan_sampling``.

    Examples
    --------
    >>> from quantem.gpu import io, parallax
    >>> data = io.load("scan_master.h5", backend="cuda")
    >>> result = parallax.run(data, scan_sampling=0.5, fit_aberrations=True)
    >>> result.to_ssb_aberrations()
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
    if fit_aberrations and scan_sampling is None:
        raise ValueError(
            "fit_aberrations=True needs scan_sampling, the scan step in Angstrom per pixel."
        )
    from quantem.gpu.parallax.cuda.reconstruction import run_cuda

    result = run_cuda(
        data,
        center=center,
        bf_radius=bf_radius,
        sampling_radius=sampling_radius,
        voltage_kV=voltage_kV,
        scan_sampling=scan_sampling,
        upsampling_factor=upsampling_factor,
        fit_aberrations=fit_aberrations,
        verbose=verbose,
    )
    # The bright-field stack and its spectra are many times the image; return
    # their blocks to the GPU for the next consumer instead of keeping them cached.
    release_cached_memory()
    return result
