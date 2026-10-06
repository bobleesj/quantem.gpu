"""The bins and crop one request asks of a resident acquisition."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BrowsePlan:
    """One request's view of a resident acquisition: detector bin, scan bin and scan crop.

    Every plan reads the same encoded resident, so changing bin or crop never
    reloads or evicts it. Bins add exact integer counts. ``detector_bin``
    divides the detector. ``scan_bin`` keeps the partial bins at the bottom
    and right edges of the crop, where the missing positions count as zero, as
    the dense loader's scan binning did. ``scan_region`` is the half-open
    ``(row_start, row_stop, col_start, col_stop)`` crop, or None for the
    complete scan. All shapes are ``(row, col)``.

    Examples
    --------
    >>> plan = BrowsePlan((256, 256), (192, 192), detector_bin=2, scan_bin=2)
    >>> plan.scan_shape, plan.detector_shape
    ((128, 128), (96, 96))
    """

    source_scan_shape: tuple[int, int]
    source_detector_shape: tuple[int, int]
    detector_bin: int = 1
    scan_bin: int = 1
    scan_region: tuple[int, int, int, int] | None = None

    @property
    def region(self) -> tuple[int, int, int, int]:
        """Half-open source crop, explicit even for the complete scan."""
        rows, cols = self.source_scan_shape
        return self.scan_region or (0, rows, 0, cols)

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Scan shape of the binned, cropped view."""
        row_start, row_stop, col_start, col_stop = self.region
        return (
            math.ceil((row_stop - row_start) / self.scan_bin),
            math.ceil((col_stop - col_start) / self.scan_bin),
        )

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector shape of the binned view."""
        rows, cols = self.source_detector_shape
        return rows // self.detector_bin, cols // self.detector_bin

    def positions(self, row: int, col: int, rows: int = 1, cols: int = 1) -> np.ndarray:
        """Flat source scan indices binned into view rows ``[row, row + rows)`` and columns ``[col, col + cols)``."""
        row_start, row_stop, col_start, col_stop = self.region
        source_rows = np.arange(
            row_start + row * self.scan_bin,
            min(row_start + (row + rows) * self.scan_bin, row_stop),
        )
        source_cols = np.arange(
            col_start + col * self.scan_bin,
            min(col_start + (col + cols) * self.scan_bin, col_stop),
        )
        return (source_rows[:, None] * self.source_scan_shape[1] + source_cols).ravel()

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop one image over the complete source scan to the plan's region."""
        row_start, row_stop, col_start, col_stop = self.region
        return np.asarray(image).reshape(self.source_scan_shape)[
            row_start:row_stop, col_start:col_stop
        ]

    def scan_image(self, values: np.ndarray) -> np.ndarray:
        """Crop and scan-bin one exact count image over the complete source scan."""
        image = self.crop(np.asarray(values, dtype=np.uint64))
        if self.scan_bin == 1:
            return image
        rows, cols = self.scan_shape
        padded = np.zeros((rows * self.scan_bin, cols * self.scan_bin), np.uint64)
        padded[: image.shape[0], : image.shape[1]] = image
        return padded.reshape(rows, self.scan_bin, cols, self.scan_bin).sum(
            axis=(1, 3), dtype=np.uint64
        )

    def diffraction(self, pattern: np.ndarray) -> np.ndarray:
        """Detector-bin one native diffraction pattern by adding exact counts."""
        pattern = np.asarray(pattern)
        if self.detector_bin == 1:
            return pattern
        rows, cols = self.detector_shape
        return pattern.reshape(rows, self.detector_bin, cols, self.detector_bin).sum(
            axis=(1, 3), dtype=np.uint64
        )

    def source_mask(self, mask: np.ndarray) -> np.ndarray:
        """Expand a mask over the view's detector to the native pixels each binned pixel adds."""
        mask = np.asarray(mask, dtype=bool)
        return mask.repeat(self.detector_bin, axis=0).repeat(self.detector_bin, axis=1)
