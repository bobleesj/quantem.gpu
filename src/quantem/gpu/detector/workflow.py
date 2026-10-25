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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from weakref import WeakKeyDictionary

import numpy as np

from quantem.gpu.detector.session import prepare
from quantem.gpu.display._layout import panel_inches, resolve_columns, scalebar
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.representation import DataRepresentation

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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


def show(
    data,
    *,
    size: Literal["small", "medium", "large", "xlarge"] | int = "medium",
    columns: int | None = None,
    rows: int | None = None,
    cmap: str = "inferno",
    norm: str | None = "power_sqrt",
    diffraction_norm: str | None = "log_auto",
    save: str | Path | None = None,
) -> "Figure":
    """Display BF, DF, ADF and the mean diffraction pattern in a static grid.

    Uses the existing detector reductions and QuantEM's static ``show_2d``.
    The probe is fitted automatically. BF collects the bright disk, DF collects
    outside it, and ADF collects from the disk edge to twice its radius.
    Use :func:`bf`, :func:`df`, :func:`adf` or :func:`mean` directly to retain
    arrays or choose collection angles. Display options never bin the data.
    Scan panels carry a bar in Å or nm from ``scan_sampling_A`` and the mean
    diffraction one in mrad from ``detector_sampling_mrad``; a panel whose
    sampling is not set has no bar.

    Parameters
    ----------
    data
        One loaded acquisition or array supported by :func:`prepare`, ordered
        as scan row, scan column, detector row, detector column.
    size : {"small", "medium", "large", "xlarge"} or int, optional
        Per-panel size, medium by default: 3, 4.5, 6 or 9 inches, or a panel
        width in pixels at 100 dpi. The presets follow the Show2D/Show3D
        proportions (300, 450, 600 or 900 pixels at 100 dpi).
        Notebook scaling and the space for titles can affect visible width;
        source array resolution is unchanged.
    columns : int, optional
        Number of image panels per row, two by default. Rows are calculated
        automatically. Use 1 for a column or 4 for a row; fewer images do not
        create empty columns.
    rows : int, optional
        Desired rows, with columns inferred when supplied alone. If both are
        supplied, the layout must hold all four images. Use ``rows=1`` for a row.
    cmap : str, optional
        Matplotlib colormap shared by all four images.
    norm : str or None, optional
        BF, DF and ADF normalization passed to ``show_2d``. The default
        square-root contrast does not alter detector intensities.
    diffraction_norm : str or None, optional
        Mean diffraction normalization. Logarithmic contrast is the default
        so weaker scattering remains visible; use ``"linear_auto"`` for linear
        contrast. Contrast is calculated independently for each panel.
    save : str or pathlib.Path or None, optional
        Save the displayed comparison, for example ``"detector-images.png"``.
        The saved colors are a visualization, not raw detector intensities.

    Returns
    -------
    matplotlib.figure.Figure
        Static comparison, closed so a bare notebook expression renders once;
        still available for later saving.

    Examples
    --------
    >>> detector.show(data)  # doctest: +SKIP
    >>> detector.show(data, size="large", columns=2, cmap="gray",
    ...               save="detector-images.png")  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from quantem.core.visualization import show_2d

    width = panel_inches(size)
    columns = resolve_columns(4, columns=columns, rows=rows)
    images = [bf(data), df(data), adf(data), mean(data)]
    titles = ["Bright field (BF)", "Dark field (DF)",
              "Annular dark field (ADF)", "Mean diffraction"]
    norms = [norm, norm, norm, diffraction_norm]
    scan_sampling = data.calibration.scan_sampling_A if hasattr(data, "calibration") else None
    detector_sampling = data.calibration.detector_sampling_mrad if hasattr(data, "calibration") else None
    samplings = [(scan_sampling, "Å")] * 3 + [(detector_sampling, "mrad")]
    bars = [None if sampling is None else _scalebar(image, sampling, units, width)
            for image, (sampling, units) in zip(images, samplings)]
    figure, _ = show_2d(
        [images[i:i + columns] for i in range(0, len(images), columns)],
        title=[titles[i:i + columns] for i in range(0, len(images), columns)],
        cmap=cmap,
        norm=[norms[i:i + columns] for i in range(0, len(images), columns)],
        scalebar=[bars[i:i + columns] for i in range(0, len(images), columns)],
        axsize=(width, width),
    )
    _outline_scalebars(figure.axes)
    if save is not None:  # after the outlines, with show_2d's own save settings
        figure.savefig(save, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(figure)
    return figure


def _scalebar(image, sampling, units, width) -> dict:
    """Scale bar of a round length, 1, 2 or 5 times a power of ten, nearest 20 % of the field.

    The field is ``columns * sampling`` along the bar; ``sampling`` is per pixel
    (Å for scan images, mrad for diffraction), its column entry for a pair.
    """
    sampling = float(sampling[1] if isinstance(sampling, (tuple, list)) else sampling)
    target = .2 * image.shape[1] * sampling
    decade = 10 ** math.floor(math.log10(target))
    length = min((step * decade for step in (1, 2, 5, 10)), key=lambda value: abs(value - target))
    return scalebar(sampling, units, image.shape[0], width) | {"length": length, "bold": True}


def _outline_scalebars(axes) -> None:
    """Stroke each white bar and label in black, a tenth of the label size wide.

    ``show_2d`` draws a bare white bar, which vanishes on the bright parts of
    BF images and the diffraction disk; the stroke keeps it readable there and
    grows with the label, which ``scalebar`` sizes from the panel width.
    """
    from matplotlib import patheffects
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    for axis in axes:
        for bar in axis.artists:
            if isinstance(bar, AnchoredSizeBar):
                label = bar.txt_label._text
                stroke = [patheffects.withStroke(linewidth=.1 * label.get_fontsize(), foreground="black")]
                for artist in (*bar.size_bar.get_children(), label):
                    artist.set_path_effects(stroke)


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
    return _edge_radius(pattern, _bright_disk(pattern))


def _bright_disk(pattern: np.ndarray) -> np.ndarray:
    """The pixels of a float64 mean pattern above mean + std: the bright-field disk every fit starts from."""
    disk = pattern > pattern.mean() + pattern.std()
    if not disk.any():
        raise ValueError(
            "The mean diffraction pattern has no bright-field disk to calibrate the detector on; "
            "pass detector_sampling_mrad (mrad per detector pixel)."
        )
    return disk


def _edge_radius(pattern: np.ndarray, disk: np.ndarray) -> float:
    """Equivalent-area radius of the pixels above half the disk's median plateau (``disk_edge_radius``)."""
    half_plateau = 0.5 * float(np.median(pattern[disk]))
    return math.sqrt(np.count_nonzero(pattern > half_plateau) / math.pi)


BRIGHT_FIELD_DISK_METHOD = "half-plateau disk edge"


@dataclass(frozen=True, eq=False)
class BrightFieldDiskFit:
    """The bright-field disk of a mean diffraction pattern, and the detector sampling its edge implies.

    The disk is the image of the probe-forming aperture, so its edge sits at the convergence semiangle and
    ``detector_sampling_mrad = semiangle_mrad / bright_field_edge_radius_px`` on both axes. Returned by
    :func:`fit_bright_field_disk` and ``data.calibration.fit_detector_sampling()``.

    Attributes
    ----------
    bright_field_center_px : tuple[float, float]
        Disk centre in detector pixels, (row, col): the centroid of the pixels above mean + std.
    bright_field_edge_radius_px : float
        Radius in detector pixels at which the disk falls to half its plateau (:func:`disk_edge_radius`).
    detector_sampling_mrad : tuple[float, float] or None
        Detector sampling in mrad per pixel, (row, col); None when no semiangle was given.
    semiangle_mrad : float or None
        The convergence semiangle the sampling was computed with.
    method : str
        The radius rule, ``"half-plateau disk edge"``.
    mean_pattern : numpy.ndarray
        The float64 mean pattern the disk was fitted on, (row, col).
    """

    bright_field_center_px: tuple[float, float]
    bright_field_edge_radius_px: float
    detector_sampling_mrad: tuple[float, float] | None
    semiangle_mrad: float | None
    method: str
    mean_pattern: np.ndarray = field(repr=False)

    def show(
        self, *, size: Literal["small", "medium", "large", "xlarge"] | None = None,
        cmap: str = "gray", save: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """The mean pattern on a log scale with the fitted disk edge and centre drawn on it.

        Axes are in mrad from the disk centre when the detector sampling is known, else in detector pixels. Plotting
        only: the log scale is a display choice.

        Parameters
        ----------
        size : {"small", "medium", "large", "xlarge"}, optional
            Named panel size; medium by default. Does not change the pattern
            or fitted calibration. This single-panel view has no rows/columns.
        cmap : str, optional
            Colormap for the mean pattern, gray by default.
        save : str or pathlib.Path, optional
            Export the complete static figure, including the fitted disk.
        figsize : tuple[float, float], optional
            Advanced figure dimensions in inches, retained for compatibility.
            Use either size or figsize, not both.

        Returns
        -------
        matplotlib.figure.Figure
            Closed, so a bare notebook expression renders it once.

        Examples
        --------
        >>> data.calibration.fit_detector_sampling().show()  # doctest: +SKIP
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from quantem.gpu.display._layout import panel_inches

        if size is not None and figsize is not None:
            raise ValueError("Use size='medium' or figsize=(width, height), not both.")
        if figsize is None:
            width = panel_inches("medium" if size is None else size)
            figsize = (width, width * 4.6 / 5)

        rows, cols = self.mean_pattern.shape
        center_row, center_col = self.bright_field_center_px
        if self.detector_sampling_mrad is None:
            step_row = step_col = 1.0
            origin_row, origin_col, unit = 0.0, 0.0, "px"
        else:
            step_row, step_col = self.detector_sampling_mrad
            origin_row, origin_col, unit = center_row, center_col, "mrad"
        extent = (-(origin_col + 0.5) * step_col, (cols - 0.5 - origin_col) * step_col,
                  (rows - 0.5 - origin_row) * step_row, -(origin_row + 0.5) * step_row)
        positive = self.mean_pattern[self.mean_pattern > 0]
        floor = float(positive.min()) if positive.size else 1.0
        figure, axis = plt.subplots(figsize=figsize)
        image = axis.imshow(np.log10(np.maximum(self.mean_pattern, floor)), cmap=cmap, extent=extent,
                            interpolation="nearest")
        centre = ((center_col - origin_col) * step_col, (center_row - origin_row) * step_row)
        radius = self.bright_field_edge_radius_px
        axis.add_patch(Ellipse(centre, 2 * radius * step_col, 2 * radius * step_row, fill=False, color="#00e5ff",
                               linewidth=1.4))
        axis.plot(*centre, marker="+", color="#00e5ff", markersize=10)
        label = f"edge radius {radius:.2f} px"
        if self.detector_sampling_mrad is not None:
            label = (f"edge {self.semiangle_mrad:g} mrad = {radius:.2f} px\n"
                     f"detector step {self.detector_sampling_mrad[0]:.4f} mrad per pixel")
        axis.text(0.03, 0.97, label, transform=axis.transAxes, va="top", ha="left", color="white", fontsize=9)
        axis.set(xlabel=f"detector column ({unit})", ylabel=f"detector row ({unit})",
                 title=f"Bright-field disk\nCentre ({center_row:.2f}, {center_col:.2f}) px")
        figure.colorbar(image, ax=axis, label="log10 mean counts", shrink=0.85)
        figure.tight_layout()
        if save is not None:
            figure.savefig(save, dpi="figure", bbox_inches="tight")
        plt.close(figure)
        return figure


def fit_bright_field_disk(data, semiangle_mrad: float | None = None) -> BrightFieldDiskFit:
    """Fit the bright-field disk of a mean pattern: its centre, its edge radius and the detector sampling they imply.

    The one rule every tool uses for the disk (SSB, viewer detector presets, ``data.calibration.fit_detector_sampling``).
    The centre is the centroid of the pixels above mean + std, the edge radius the half-plateau radius of
    :func:`disk_edge_radius`, and with a semiangle the detector sampling is ``semiangle_mrad / edge radius`` on both
    axes. Computed in float64, as SSB selects its bright-field pixels.

    Parameters
    ----------
    data
        A 2D mean diffraction pattern, typically from :func:`mean`, or an acquisition that :func:`prepare` accepts,
        whose mean pattern is computed on its own backend without a dense copy.
    semiangle_mrad : float, optional
        Probe convergence semiangle in mrad, finite and positive; without it the sampling is None.

    Returns
    -------
    BrightFieldDiskFit
        Centre, edge radius, detector sampling and the mean pattern, with ``show()``.

    Raises
    ------
    ValueError
        When no pixel stands above mean + std (no disk), or the semiangle is not finite and positive.

    Examples
    --------
    >>> fit = detector.fit_bright_field_disk(detector.mean(data), semiangle_mrad=30)
    >>> fit.detector_sampling_mrad
    (0.556..., 0.556...)
    """
    if semiangle_mrad is not None:
        semiangle = float(semiangle_mrad)
        if not (math.isfinite(semiangle) and semiangle > 0):
            raise ValueError(f"semiangle_mrad must be finite and positive; got {semiangle_mrad!r}.")
    pattern = np.asarray(mean(data) if np.ndim(data) > 2 else data, dtype=np.float64)
    disk = _bright_disk(pattern)
    rows, cols = np.nonzero(disk)
    radius = _edge_radius(pattern, disk)
    sampling = None if semiangle_mrad is None else (float(semiangle_mrad) / radius,) * 2
    return BrightFieldDiskFit(
        bright_field_center_px=(float(rows.mean()), float(cols.mean())), bright_field_edge_radius_px=radius,
        detector_sampling_mrad=sampling, semiangle_mrad=None if semiangle_mrad is None else float(semiangle_mrad),
        method=BRIGHT_FIELD_DISK_METHOD, mean_pattern=pattern,
    )


def detector_sampling_mrad(mean_pattern: np.ndarray, semiangle_mrad: float) -> tuple[float, float]:
    """Detector sampling in mrad per pixel, (row, col), from the bright-field disk and the convergence semiangle.

    The bright-field disk is the image of the probe-forming aperture, so its edge sits at the convergence semiangle:
    ``semiangle_mrad / disk_edge_radius(mean_pattern)`` mrad per pixel on both axes (:func:`fit_bright_field_disk`).
    This is the rule SSB, parallax and screening apply when no detector sampling is recorded or given; call it rather
    than another radius rule, which can differ by several percent.

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
    if semiangle_mrad is None:
        raise ValueError("semiangle_mrad must be finite and positive; got None.")
    return fit_bright_field_disk(mean_pattern, semiangle_mrad).detector_sampling_mrad


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
