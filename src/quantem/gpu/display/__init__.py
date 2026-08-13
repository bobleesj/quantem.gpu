"""Shared GPU display resources for QuantEM native and Python clients.

The Metal source is the canonical Apple-GPU implementation of image scaling,
LUT colormapping, range reduction, and histogram binning. Native Swift clients
consume the same file through the repository's ``QuantEMMetalDisplay`` Swift
package.
"""

from __future__ import annotations

import json
from importlib.resources import files

import numpy as np


_RESOURCE_ROOT = (
    files(__package__)
    / "swift"
    / "Sources"
    / "QuantEMMetalDisplay"
    / "Resources"
)


def metal_source() -> str:
    """Return the canonical QuantEM display Metal source."""
    return (_RESOURCE_ROOT / "display.metal").read_text(encoding="utf-8")


def colormap_names() -> tuple[str, ...]:
    """Return the LUT colormaps shared with native QuantEM clients."""
    points = json.loads(
        (_RESOURCE_ROOT / "colormaps.json").read_text(encoding="utf-8")
    )
    return tuple(points)


def colormap_lut(name: str) -> np.ndarray:
    """Return one 256-entry float32 RGBA lookup table.

    Parameters
    ----------
    name
        Colormap name returned by :func:`colormap_names`.

    Returns
    -------
    numpy.ndarray
        Array shaped ``(256, 4)`` with values in ``[0, 1]``.
    """
    points_by_name = json.loads(
        (_RESOURCE_ROOT / "colormaps.json").read_text(encoding="utf-8")
    )
    if name not in points_by_name:
        choices = ", ".join(points_by_name)
        raise ValueError(f"Unknown colormap {name!r}. Choose one of: {choices}.")
    points = np.asarray(points_by_name[name], dtype=np.float32)
    positions = np.linspace(0, len(points) - 1, 256, dtype=np.float32)
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, len(points) - 1)
    fraction = (positions - lower).reshape(-1, 1)
    rgb = points[lower] + fraction * (points[upper] - points[lower])
    rgba = np.empty((256, 4), dtype=np.float32)
    rgba[:, :3] = np.floor(rgb + np.float32(0.5)) / np.float32(255)
    rgba[:, 3] = 1
    return rgba


__all__ = ["colormap_lut", "colormap_names", "metal_source"]
