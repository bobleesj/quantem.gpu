"""Typed bright-field detector selection for SSB reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BrightfieldDisk:
    """Validated detector coordinates and geometry for one BF selection.

    Coordinates and centers always follow the public ``(row, col)`` convention.
    The coordinate arrays are copied, normalized to ``int32``, and made
    read-only so every SSB stage consumes the same immutable evidence.
    """

    rows: np.ndarray
    cols: np.ndarray
    center_row_col: tuple[float, float]
    radius_px: float
    detected_radius_px: float
    detector_shape: tuple[int, int]

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int32).reshape(-1).copy()
        cols = np.asarray(self.cols, dtype=np.int32).reshape(-1).copy()
        if rows.size == 0 or rows.shape != cols.shape:
            raise ValueError(
                "BF rows and columns must be non-empty matching vectors."
            )

        detector_shape = tuple(int(value) for value in self.detector_shape)
        if len(detector_shape) != 2 or min(detector_shape) < 1:
            raise ValueError(
                "detector_shape must contain two positive (row, col) sizes."
            )
        if (
            int(rows.min()) < 0
            or int(cols.min()) < 0
            or int(rows.max()) >= detector_shape[0]
            or int(cols.max()) >= detector_shape[1]
        ):
            raise ValueError(
                f"BF coordinates fall outside detector shape {detector_shape}."
            )
        linear = rows.astype(np.int64) * detector_shape[1] + cols
        if np.unique(linear).size != rows.size:
            raise ValueError("BF coordinates must not contain duplicates.")

        center = tuple(float(value) for value in self.center_row_col)
        if len(center) != 2 or not all(math.isfinite(value) for value in center):
            raise ValueError("center_row_col must contain two finite values.")
        radius = float(self.radius_px)
        detected_radius = float(self.detected_radius_px)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius_px must be a positive finite value.")
        if not math.isfinite(detected_radius) or detected_radius <= 0:
            raise ValueError(
                "detected_radius_px must be a positive finite value."
            )

        rows.flags.writeable = False
        cols.flags.writeable = False
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "center_row_col", center)
        object.__setattr__(self, "radius_px", radius)
        object.__setattr__(self, "detected_radius_px", detected_radius)
        object.__setattr__(self, "detector_shape", detector_shape)

    @property
    def size(self) -> int:
        """Number of selected BF detector pixels."""

        return int(self.rows.size)


__all__ = ["BrightfieldDisk"]
