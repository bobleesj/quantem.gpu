"""Virtual detectors (bright / annular-dark / dark field) for 4D-STEM.

Place a virtual detector on 4D-STEM data and get its image, with collection
angles in **mrad**::

    from quantem.gpu import detector, io
    from quantem.widget import Show2D
    data = io.load("master.h5")
    Show2D(detector.bf(data))                        # bright field (the bright disk)
    Show2D(detector.adf(data))                       # annular dark field (auto band)
    Show2D(detector.adf(data, inner_angle_mrad=50, outer_angle_mrad=180))
    Show2D(detector.df(data))                        # outside the bright disk

``bf`` / ``adf`` / ``df`` build a boolean detector mask with
:func:`detector_mask` and reduce it with the prepared session's masked sum, the
same reduction Show4DSTEM and live Browse use, so a viewer ROI and ``adf`` are
pixel-identical. The probe (disk center and size) is fitted from the mean
diffraction pattern; encoded acquisitions keep their first fit, mutable arrays
are refitted on every call, and an explicit ``bright_field_center_px`` or
radius affects only that call. No detector binning is applied on any backend.
"""

import math
from weakref import WeakKeyDictionary

import numpy as np

from quantem.gpu.detector.session import prepare
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.representation import DataRepresentation

# Automatic disk geometry per encoded source: (logical shape, center, radius).
# Encoded measurements cannot change, so one fit serves the source's lifetime.
_PROBE_FITS = WeakKeyDictionary()


def mean(data) -> np.ndarray:
    """Average scan positions into one mean diffraction pattern.

    Parameters
    ----------
    data
        One supported 4D acquisition or array, ordered as scan row, scan
        column, detector row, detector column. Uses the source's detector
        backend without constructing a dense copy of the acquisition.

    Returns
    -------
    numpy.ndarray
        Mean intensity with shape ``(detector_row, detector_column)``.
        Only this reduced pattern is transferred to the host.

    Examples
    --------
    >>> mean_dp = detector.mean(data)
    >>> bright_field_center_px, bright_field_edge_radius_px = detector.fit_probe(mean_dp)
    """
    return prepare(data).mean_dp()


def masked_sum(data, det_mask) -> np.ndarray:
    """Sum a detector mask at every scan position, as a float32 scan image.

    The one-call form of ``prepare(data).masked_sum(det_mask)`` for code that
    needs the shared widget/live masked-sum path once.
    """
    return prepare(data).masked_sum(det_mask)


def fit_probe(data) -> tuple[tuple[float, float], float]:
    """Estimate the bright-field disk center and radius.

    Parameters
    ----------
    data
        A 2D mean diffraction pattern, typically from :func:`mean`, or an
        acquisition that :func:`prepare` accepts, whose mean pattern is used.

    Returns
    -------
    bright_field_center_px : tuple[float, float]
        Disk center in ``(row, col)`` detector pixels.
    bright_field_edge_radius_px : float
        Equivalent-area disk radius in detector pixels.

    Raises
    ------
    ValueError
        No pixel lies above the threshold (a flat or empty pattern): there is
        no disk to fit, so pass the center and the radius explicitly.

    Notes
    -----
    Uses a threshold of ``mean + std`` (population standard deviation), the
    unweighted centroid of the selected pixels, and ``sqrt(area / pi)``.
    This estimates disk geometry, not complex probe phase or aberrations.

    Examples
    --------
    >>> center, radius = detector.fit_probe(data)
    >>> bright = detector.bf(data, bright_field_center_px=center, bright_field_radius_px=radius)
    """
    pattern = np.asarray(mean(data) if np.ndim(data) > 2 else data, dtype=np.float32)
    disk = pattern > float(pattern.mean()) + float(pattern.std())
    area = int(disk.sum())
    if area == 0:
        raise ValueError(
            "No bright-field disk found: no detector pixel exceeds the pattern's mean plus "
            "one standard deviation (a flat or empty pattern). Pass the center and the radius explicitly."
        )
    rows = np.arange(pattern.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(pattern.shape[1], dtype=np.float32)[None, :]
    center_row = float((rows * disk).sum() / area)
    center_col = float((cols * disk).sum() / area)
    return (center_row, center_col), float(np.sqrt(area / np.pi))


def disk_edge_radius(mean_pattern: np.ndarray) -> float:
    """Return the radius in detector pixels at which the bright-field disk falls to half its plateau intensity.

    The disk is the image of the probe-forming aperture, so this radius corresponds to the convergence semiangle and
    calibrates the detector: ``detector_sampling_mrad = semiangle_mrad / disk_edge_radius(mean_pattern)`` mrad per pixel. A
    symmetric blur of the aperture edge (detector point spread, partial coherence) leaves the half-intensity contour
    at the geometric edge, while a higher threshold moves it inward. The plateau is the median of the pixels above
    mean + std (the disk ``detector.fit_probe`` finds), and the radius is that of a disk with the area of the pixels
    above half of it, which resolves the edge below one pixel. On abTEM 4D-STEM with a known detector sampling this is
    within 0.4 % of the truth; on a 300 kV, 30 mrad Arina acquisition it gives 0.552 mrad per pixel, the measured
    calibration of that camera length (docs/maintainer/2026-10-05-ssb-detector-sampling.md).

    Parameters
    ----------
    mean_pattern : numpy.ndarray
        Mean diffraction pattern of the full detector, (row, col).

    Returns
    -------
    float
        Equivalent-area radius of the half-plateau disk, in detector pixels.

    Raises
    ------
    ValueError
        When no pixel stands above mean + std, so the pattern has no disk to calibrate on.

    Examples
    --------
    >>> radius_px = detector.disk_edge_radius(detector.mean(data))
    """
    pattern = np.asarray(mean_pattern, dtype=np.float64)
    disk = pattern > pattern.mean() + pattern.std()
    if not disk.any():
        raise ValueError(
            "The mean diffraction pattern has no bright-field disk to calibrate the detector on; "
            "pass detector_sampling_mrad (mrad per detector pixel)."
        )
    half_plateau = 0.5 * float(np.median(pattern[disk]))
    return math.sqrt(np.count_nonzero(pattern > half_plateau) / math.pi)


def detector_sampling_mrad(mean_pattern: np.ndarray, semiangle_mrad: float) -> tuple[float, float]:
    """Detector sampling in mrad per pixel, (row, col), from the bright-field disk and the convergence semiangle.

    The bright-field disk is the image of the probe-forming aperture, so its edge sits at the convergence semiangle:
    ``semiangle_mrad / disk_edge_radius(mean_pattern)`` mrad per pixel on both axes. This is the rule SSB, parallax and
    screening apply when no detector sampling is recorded or given; call it rather than another radius rule, which
    can differ by several percent.

    Parameters
    ----------
    mean_pattern : numpy.ndarray
        Mean diffraction pattern of the full detector, (row, col), for example ``detector.mean(data)``.
    semiangle_mrad : float
        Probe convergence semiangle in mrad; finite and positive.

    Returns
    -------
    tuple[float, float]
        Detector sampling in mrad per pixel, (row, col); equal axes.

    Examples
    --------
    >>> detector.detector_sampling_mrad(detector.mean(data), semiangle_mrad=30)
    (0.552..., 0.552...)
    """
    semiangle = float(semiangle_mrad)
    if not (math.isfinite(semiangle) and semiangle > 0):
        raise ValueError(f"semiangle_mrad must be finite and positive; got {semiangle_mrad!r}.")
    sampling = semiangle / disk_edge_radius(mean_pattern)
    return (sampling, sampling)


def detector_mask(center_px, inner_radius_px, outer_radius_px, detector_shape, *, dtype=np.float32) -> np.ndarray:
    """Boolean ``(det_row, det_col)`` mask of the pixels between two radii of ``center_px``.

    A pixel is selected when its distance from ``center_px`` (row, col) lies in
    ``[inner_radius_px, outer_radius_px]`` detector pixels. Every virtual detector (``bf``,
    ``adf``, ``df`` and the Show4DSTEM circle/annulus ROIs) builds
    its mask here, so a viewer ROI and ``adf`` are pixel-identical.

    ``dtype=np.float64`` uses inclusive float64 Euclidean distances for native
    series interactions. The default float32 calculation is unchanged.

    Examples
    --------
    >>> mask = detector_mask((95.5, 95.5), 40, 80, (192, 192), dtype=np.float64)
    """
    center_row, center_col = center_px
    precision = np.dtype(dtype)
    if precision not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Use float32 or float64 detector geometry; got {dtype!r}.")
    rows = np.arange(detector_shape[0], dtype=precision)[:, None]
    cols = np.arange(detector_shape[1], dtype=precision)[None, :]
    if precision == np.dtype(np.float64):
        distance = np.hypot(rows - center_row, cols - center_col)
    else:
        distance = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    return (distance >= inner_radius_px) & (distance <= outer_radius_px)


def bf(data, bright_field_center_px=None, bright_field_radius_px=None) -> np.ndarray:
    """Bright-field image of ``data``: the bright disk (the unscattered probe).

    The probe is fitted unless ``bright_field_center_px`` (row, col) and
    ``bright_field_radius_px`` (detector pixels) are given. Encoded
    acquisitions reuse their first disk fit across BF/ADF/DF calls; mutable
    arrays are fitted on each call. Overrides affect only this call; omitted
    geometry continues to use the automatic fit.

    Examples
    --------
    >>> bright = bf(data)
    >>> smaller_disk = bf(data, bright_field_radius_px=30)
    """
    center, radius = _probe(data, bright_field_center_px, bright_field_radius_px)
    return _detector_image(data, center, 0.0, radius)


def adf(data, *, inner_angle_mrad: float | None = None, outer_angle_mrad: float | None = None,
        inner_radius_px: float | None = None, outer_radius_px: float | None = None,
        bright_field_center_px=None, bright_field_edge_radius_px=None) -> np.ndarray:
    """Annular-dark-field image of ``data``, collected between an inner and an outer bound.

    Give each bound as a collection angle (``inner_angle_mrad``,
    ``outer_angle_mrad``; needs ``semiangle_mrad`` in the metadata) or as a
    radius in detector pixels (``inner_radius_px``, ``outer_radius_px``), not
    both. Omit a bound for the automatic band: inner = the bright-disk edge,
    outer = twice that. The probe is fitted unless ``bright_field_center_px``
    (row, col) and ``bright_field_edge_radius_px`` (detector pixels) are given;
    explicit geometry affects only this call and does not replace the stored fit.

    Examples
    --------
    >>> annular = adf(data)
    >>> wider_ring = adf(data, inner_radius_px=60, outer_radius_px=85)
    """
    center, radius = _probe(data, bright_field_center_px, bright_field_edge_radius_px)
    inner_px = _bound_px(data, inner_angle_mrad, inner_radius_px, radius, radius)
    outer_px = _bound_px(data, outer_angle_mrad, outer_radius_px, radius, 2.0 * radius)
    return _detector_image(data, center, inner_px, outer_px)


def df(data, *, inner_angle_mrad: float | None = None, inner_radius_px: float | None = None,
       bright_field_center_px=None, bright_field_edge_radius_px=None) -> np.ndarray:
    """Dark-field image of ``data``: everything collected beyond the inner bound.

    Give the bound as ``inner_angle_mrad`` (needs ``semiangle_mrad`` in the
    metadata) or ``inner_radius_px``, not both. Omit it for everything outside
    the bright disk. The probe is fitted unless ``bright_field_center_px``
    (row, col) and ``bright_field_edge_radius_px`` (detector pixels) are given;
    explicit geometry affects only this call and does not replace the stored fit.

    Examples
    --------
    >>> dark = df(data)
    >>> outer_signal = df(data, inner_radius_px=60)
    """
    center, radius = _probe(data, bright_field_center_px, bright_field_edge_radius_px)
    inner_px = _bound_px(data, inner_angle_mrad, inner_radius_px, radius, radius)
    return _detector_image(data, center, inner_px, np.inf)


# ---


def _probe(data, center=None, radius=None):
    """Fill in the disk geometry the caller left out; encoded data fits it only once.

    Mutable arrays are fitted on every call because their measurements can
    change; an encoded source keeps its first fit for as long as it lives.
    """
    if center is not None and radius is not None:
        return (float(center[0]), float(center[1])), float(radius)
    source = (
        data.data
        if isinstance(data, Dataset4dstemGPU)
        and data.representation is DataRepresentation.ENCODED
        and not data.data.is_released
        else None
    )
    cached = None if source is None else _PROBE_FITS.get(source)
    if cached is None or cached[0] != data.shape:
        fitted_center, fitted_radius = fit_probe(data)
        if source is not None:
            _PROBE_FITS[source] = (data.shape, fitted_center, fitted_radius)
    else:
        _, fitted_center, fitted_radius = cached
    center = (float(center[0]), float(center[1])) if center is not None else fitted_center
    radius = float(radius) if radius is not None else fitted_radius
    return center, radius


def _bound_px(data, angle_mrad: float | None, radius_px: float | None, edge_radius_px: float, default_px: float) -> float:
    """One collection bound in detector pixels, from an angle, a radius, or the automatic value.

    The bright disk edge spans ``semiangle_mrad``, so an angle in mrad maps to
    ``mrad / semiangle_mrad * edge_radius_px`` pixels; a radius in pixels is
    used as given (calibration-free, exact). A bound stated both ways is
    refused: the two need not agree and one of them would be ignored.
    """
    if angle_mrad is not None and radius_px is not None:
        raise ValueError("Give a collection bound as an angle in mrad or as a radius in pixels, not both.")
    if radius_px is not None:
        return float(radius_px)
    if angle_mrad is None:
        return default_px
    # Loaded data keeps calibration in metadata; quantem.core datasets, which this
    # package cannot import, carry either a metadata dict or a semiangle_mrad attribute.
    metadata = (data.metadata or {}) if isinstance(data, Dataset4dstemGPU) else getattr(data, "metadata", None)
    if isinstance(metadata, dict):
        semiangle_mrad = metadata.get("semiangle_mrad")
    else:
        semiangle_mrad = getattr(data, "semiangle_mrad", None)
    if not semiangle_mrad:
        raise ValueError(
            "A collection angle in mrad needs the convergence semi-angle, which is "
            "unknown for this data. Store semiangle_mrad in metadata or pass detector "
            "pixels instead: adf(data, inner_radius_px=..., outer_radius_px=...).")
    return float(angle_mrad) / float(semiangle_mrad) * edge_radius_px


def _detector_image(data, center, inner_px: float, outer_px: float) -> np.ndarray:
    """Scan image of the annulus between ``inner_px`` and ``outer_px`` detector pixels around ``center``."""
    session = prepare(data)
    return session.masked_sum(detector_mask(center, inner_px, outer_px, session.detector_shape))
