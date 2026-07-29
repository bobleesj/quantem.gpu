"""Header-only inspection for 4D-STEM sources."""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

from .load import (
    get_metadata,
    inspect_master_readiness,
    read_pixel_mask,
)


@dataclass(frozen=True)
class Inspection:
    """Header and small calibration state for one 4D-STEM source."""

    ready: bool
    reason: str
    action: str
    metadata: dict[str, Any]
    pixel_mask: np.ndarray | None
    source_kind: str
    actual_frames: int | None
    expected_frames: int | None
    scan_shape: tuple[int, int] | None
    detector_shape: tuple[int, int] | None
    dtype: str | None
    source_signature: dict[str, Any]


def inspect(
    filepath: str | PathLike[str],
    *,
    scan_shape: tuple[int, int] | None = None,
) -> Inspection:
    """Inspect whether a master is complete and internally consistent.

    Parameters
    ----------
    filepath
        HDF5 master path.
    scan_shape
        Optional expected ``(scan_row, scan_col)`` shape.

    Returns
    -------
    Inspection
        Readiness, metadata, and the small detector pixel mask. Detector frames
        are not loaded.
    """
    readiness = inspect_master_readiness(filepath, scan_shape=scan_shape)
    try:
        metadata = get_metadata(str(filepath))
    except (OSError, KeyError, TypeError, ValueError):
        metadata = {}
    metadata.setdefault("scan_shape", scan_shape)
    metadata["detector_shape"] = readiness.detector_shape
    metadata["dtype"] = (
        np.dtype(readiness.dtype).name
        if readiness.dtype is not None
        else metadata.get("dtype")
    )
    metadata["n_frames"] = readiness.actual_frames
    try:
        pixel_mask = read_pixel_mask(filepath)
    except (OSError, KeyError, TypeError, ValueError):
        pixel_mask = None
    return Inspection(
        ready=readiness.ready,
        reason=readiness.reason,
        action=readiness.action,
        metadata=metadata,
        pixel_mask=pixel_mask,
        source_kind=readiness.source_kind,
        actual_frames=readiness.actual_frames,
        expected_frames=readiness.expected_frames,
        scan_shape=(
            tuple(int(value) for value in scan_shape)
            if scan_shape is not None
            else (
                tuple(int(value) for value in metadata["scan_shape"])
                if metadata.get("scan_shape") is not None
                else None
            )
        ),
        detector_shape=readiness.detector_shape,
        dtype=(
            np.dtype(readiness.dtype).name
            if readiness.dtype is not None
            else None
        ),
        source_signature=readiness.source_signature,
    )
