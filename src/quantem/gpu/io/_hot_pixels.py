"""Public hot-pixel correction policy and provenance."""

from __future__ import annotations

import numpy as np


def normalize_hot_pixel_correction(value: str) -> str:
    """Validate the public hot-pixel correction name."""
    method = str(value).strip().lower()
    if method not in {"median", "zero", "none"}:
        raise ValueError(
            "hot_pixel_correction must be 'median', 'zero', or 'none'."
        )
    return method


def hot_pixel_record(pixel_mask, method: str, *, backend: str) -> dict:
    """Build JSON-safe provenance for one detector correction."""
    mask = (
        np.zeros((0, 0), dtype=np.uint8)
        if pixel_mask is None
        else np.asarray(pixel_mask)
    )
    coordinates = np.argwhere(mask != 0)
    applied = method != "none" and len(coordinates) > 0
    return {
        "version": 1,
        "method": method,
        "kernel_size": 3 if method == "median" else None,
        "source": "stored_pixel_mask",
        "pixel_count": int(len(coordinates)),
        "coordinates_row_column": coordinates.astype(int).tolist(),
        "source_flags": [int(mask[tuple(point)]) for point in coordinates],
        "applied": applied,
        "stage": "gpu_load_before_resident_encoding" if applied else "none",
        "backend": backend,
    }


def correction_is_applied(metadata: dict) -> bool:
    """Return whether resident counts already include pixel correction."""
    record = metadata.get("hot_pixel_correction")
    return isinstance(record, dict) and record.get("applied") is True
