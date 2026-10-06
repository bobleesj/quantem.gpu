"""Virtual detectors (bright / annular-dark / dark field) for 4D-STEM.

Place a virtual detector on 4D-STEM data and get its image, with collection
angles in **mrad**::

    from quantem.gpu import detector, io
    from quantem.widget import Show2D
    data = io.load("master.h5")
    Show2D(detector.bf(data))                        # bright field (the bright disk)
    Show2D(detector.adf(data))                       # annular dark field (auto band)
    Show2D(detector.adf(data, inner=50, outer=180))  # collection angles in mrad
    Show2D(detector.df(data))                        # outside the bright disk

``bf`` / ``adf`` / ``df`` build a boolean detector mask with
:func:`detector_mask` and reduce it with the prepared session's masked sum, the
same reduction Show4DSTEM and live Browse use, so a viewer ROI and ``adf`` are
pixel-identical. The probe (disk center and size) is fitted from the mean
diffraction pattern; encoded acquisitions keep their first fit, mutable arrays
are refitted on every call, and explicit ``center``/``radius`` affect only that
call. No detector binning is applied on any backend.

:func:`virtual` is mode-based (DP/BF/ABF/ADF/HAADF/DF, bands measured in the
fitted disk radius) and is mainly the reference path the parity tests pin.
"""

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
    >>> center, radius = detector.fit_probe(mean_dp)
    """
    return prepare(data).mean_dp()


def masked_sum(data, det_mask) -> np.ndarray:
    """Sum a detector mask at every scan position, as a float32 scan image.

    The one-call form of ``prepare(data).masked_sum(det_mask)`` for code that
    needs the shared widget/live masked-sum path once.
    """
    return prepare(data).masked_sum(det_mask)


def fit_probe(mean_dp: np.ndarray) -> tuple[tuple[float, float], float]:
    """Estimate the bright-field disk center and radius.

    Parameters
    ----------
    mean_dp : numpy.ndarray
        A 2D mean diffraction pattern, typically from :func:`mean`.

    Returns
    -------
    center : tuple[float, float]
        Disk center in ``(row, column)`` detector pixels.
    radius : float
        Equivalent-area disk radius in detector pixels.

    Notes
    -----
    Uses a threshold of ``mean + std`` (population standard deviation), the
    unweighted centroid of the selected pixels, and ``sqrt(area / pi)``.
    This estimates disk geometry, not complex probe phase or aberrations.
    If no pixels exceed the threshold, returns the detector midpoint and
    one quarter of its smaller dimension as the radius.

    Examples
    --------
    >>> center, radius = detector.fit_probe(detector.mean(data))
    >>> bright = detector.bf(data, center=center, radius=radius)
    """
    pattern = np.asarray(mean_dp, dtype=np.float32)
    disk = pattern > float(pattern.mean()) + float(pattern.std())
    area = int(disk.sum())
    if area == 0:
        detector_rows, detector_cols = pattern.shape
        return (detector_rows / 2.0, detector_cols / 2.0), min(detector_rows, detector_cols) * 0.25
    rows = np.arange(pattern.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(pattern.shape[1], dtype=np.float32)[None, :]
    center_row = float((rows * disk).sum() / area)
    center_col = float((cols * disk).sum() / area)
    return (center_row, center_col), float(np.sqrt(area / np.pi))


def detector_mask(center, lo_px, hi_px, det_shape, *, dtype=np.float32) -> np.ndarray:
    """Boolean ``(det_row, det_col)`` mask of the pixels between two radii of ``center``.

    A pixel is selected when its distance from ``center`` (row, col) lies in
    ``[lo_px, hi_px]`` detector pixels. Every virtual detector (``bf``,
    ``adf``, ``df``, ``virtual`` and the Show4DSTEM circle/annulus ROIs) builds
    its mask here, so a viewer ROI and ``adf`` are pixel-identical.

    ``dtype=np.float64`` uses inclusive float64 Euclidean distances for native
    series interactions. The default float32 calculation is unchanged.

    Examples
    --------
    >>> mask = detector_mask((95.5, 95.5), 40, 80, (192, 192), dtype=np.float64)
    """
    center_row, center_col = center
    precision = np.dtype(dtype)
    if precision not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Use float32 or float64 detector geometry; got {dtype!r}.")
    rows = np.arange(det_shape[0], dtype=precision)[:, None]
    cols = np.arange(det_shape[1], dtype=precision)[None, :]
    if precision == np.dtype(np.float64):
        distance = np.hypot(rows - center_row, cols - center_col)
    else:
        distance = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    return (distance >= lo_px) & (distance <= hi_px)


def bf(data, center=None, radius=None) -> np.ndarray:
    """Bright-field image of ``data``: the bright disk (the unscattered probe).

    The probe is fitted unless ``center``/``radius`` (detector pixels) are
    given. Encoded acquisitions reuse their first disk fit across BF/ADF/DF
    calls; mutable arrays are fitted on each call. Overrides affect only this
    call; omitted geometry continues to use the automatic fit.

    Examples
    --------
    >>> bright = bf(data)
    >>> smaller_disk = bf(data, radius=30)
    """
    center, radius = _probe(data, center, radius)
    return _detector_image(data, center, 0.0, radius)


def adf(data, inner: float | None = None, outer: float | None = None,
        unit: str = "mrad", center=None, radius=None) -> np.ndarray:
    """Annular-dark-field image of ``data``, collected between ``inner`` and ``outer``.

    ``unit='mrad'`` (default, needs ``semiangle_mrad`` in the metadata) or
    ``unit='px'`` (raw detector pixels). Omit either for the automatic band:
    ``inner`` = the bright-disk edge, ``outer`` = twice that. The probe is
    fitted unless ``center``/``radius`` (detector pixels) are given; explicit
    geometry affects only this call and does not replace the stored fit.

    Examples
    --------
    >>> annular = adf(data)
    >>> wider_ring = adf(data, inner=60, outer=85, unit="px")
    """
    center, radius = _probe(data, center, radius)
    inner_px = radius if inner is None else _to_px(data, inner, unit, radius)
    outer_px = 2.0 * radius if outer is None else _to_px(data, outer, unit, radius)
    return _detector_image(data, center, inner_px, outer_px)


def df(data, inner: float | None = None, unit: str = "mrad",
       center=None, radius=None) -> np.ndarray:
    """Dark-field image of ``data``: everything collected beyond ``inner``.

    ``unit='mrad'`` (default, needs ``semiangle_mrad`` in the metadata) or
    ``unit='px'``. Omit ``inner`` for everything outside the bright disk. The
    probe is fitted unless ``center``/``radius`` (detector pixels) are given;
    explicit geometry affects only this call and does not replace the stored fit.

    Examples
    --------
    >>> dark = df(data)
    >>> outer_signal = df(data, inner=60, unit="px")
    """
    center, radius = _probe(data, center, radius)
    inner_px = radius if inner is None else _to_px(data, inner, unit, radius)
    return _detector_image(data, center, inner_px, np.inf)


def virtual(data, mode="BF", *, center=None, bf_radius=None, inner=None, outer=None):
    """Virtual image for ``mode`` with automatic probe fitting.

    ``mode`` is case-insensitive (DP/BF/ABF/ADF/HAADF/DF/annular). ``center`` and
    ``bf_radius`` override the fitted probe; ``inner``/``outer`` (BF-radius
    units) define a custom band when ``mode="annular"``. Returns a 2D float array
    (detector-space for DP, scan-space otherwise) for ``Show2D``.
    """
    mean_dp = mean(data)
    mode = str(mode).strip().upper()
    if mode == "DP":
        return mean_dp
    if center is None or bf_radius is None:
        fitted_center, fitted_radius = fit_probe(mean_dp)
        center = center if center is not None else fitted_center
        bf_radius = bf_radius if bf_radius is not None else fitted_radius
    # Bands are measured in bright-disk radii, then built by the one geometry primitive.
    radius = float(max(1.0, bf_radius))
    bands = {
        "BF": (0.0, radius),
        "ABF": (0.5 * radius, radius),
        "ADF": (radius, 2.0 * radius),
        "HAADF": (2.0 * radius, 4.0 * radius),
        "DF": (radius, np.inf),
    }
    if mode == "ANNULAR":
        inner_px = (inner if inner is not None else 0.0) * radius
        outer_px = (outer if outer is not None else np.inf) * radius
    else:
        inner_px, outer_px = bands[mode]
    return masked_sum(data, detector_mask(center, inner_px, outer_px, mean_dp.shape))


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
        fitted_center, fitted_radius = fit_probe(mean(data))
        if source is not None:
            _PROBE_FITS[source] = (data.shape, fitted_center, fitted_radius)
    else:
        _, fitted_center, fitted_radius = cached
    center = (float(center[0]), float(center[1])) if center is not None else fitted_center
    radius = float(radius) if radius is not None else fitted_radius
    return center, radius


def _to_px(data, value: float, unit: str, radius: float) -> float:
    """Convert a collection angle to detector pixels.

    The bright disk radius spans ``semiangle_mrad``, so an angle in mrad maps
    to ``mrad / semiangle_mrad * radius`` pixels; ``unit='px'`` is already
    pixels (calibration-free, exact).
    """
    unit = str(unit).lower()
    if unit in ("px", "pixel", "pixels"):
        return float(value)
    if unit != "mrad":
        raise ValueError(f"unit must be 'mrad' or 'px', got {unit!r}")
    # Loaded data keeps calibration in metadata; quantem.core datasets, which this
    # package cannot import, carry either a metadata dict or a semiangle_mrad attribute.
    metadata = (data.metadata or {}) if isinstance(data, Dataset4dstemGPU) else getattr(data, "metadata", None)
    if isinstance(metadata, dict):
        semiangle_mrad = metadata.get("semiangle_mrad")
    else:
        semiangle_mrad = getattr(data, "semiangle_mrad", None)
    if not semiangle_mrad:
        raise ValueError(
            "inner / outer are collection angles in mrad, but the convergence "
            "semi-angle is unknown for this data. Store semiangle_mrad in metadata "
            "or pass detector pixels instead: adf(data, inner=..., outer=..., unit='px').")
    return float(value) / float(semiangle_mrad) * radius


def _detector_image(data, center, inner_px: float, outer_px: float) -> np.ndarray:
    """Scan image of the annulus between ``inner_px`` and ``outer_px`` detector pixels around ``center``."""
    session = prepare(data)
    return session.masked_sum(detector_mask(center, inner_px, outer_px, session.detector_shape))
