"""Shared scientific image-display resources for every GPU client.

The bundled colormap control points are the single source used by Python,
CUDA, Metal/Swift, and WebGPU. Native Swift clients consume the Metal source
through the repository's ``MetalDisplayKernels`` package; browser clients
bundle the TypeScript/WGSL sources from :mod:`quantem.gpu.display.webgpu`.
"""

import json
from importlib.resources import files

import numpy as np

_RESOURCE_ROOT = (
    files("quantem.gpu")
    / "swift"
    / "Sources"
    / "MetalDisplayKernels"
    / "Resources"
)
_COLORMAP_POINTS = json.loads(
    (_RESOURCE_ROOT / "colormaps.json").read_text(encoding="utf-8")
)


def metal_source() -> str:
    """Return the canonical shader source for native Metal display clients.

    Returns
    -------
    str
        Metal source containing the display, range, and histogram entry points.

    Examples
    --------
    >>> "metal_display_fragment" in metal_source()
    True
    """
    return (_RESOURCE_ROOT / "display.metal").read_text(encoding="utf-8")


def colormap_names() -> tuple[str, ...]:
    """Return the LUT colormaps available to every display backend.

    Returns
    -------
    tuple[str, ...]
        Colormap names in their bundled display order.

    Examples
    --------
    >>> "viridis" in colormap_names()
    True
    """
    return tuple(_COLORMAP_POINTS)


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

    Examples
    --------
    >>> lut = colormap_lut("viridis")
    >>> lut.shape
    (256, 4)
    """
    if name not in _COLORMAP_POINTS:
        choices = ", ".join(_COLORMAP_POINTS)
        raise ValueError(f"Unknown colormap {name!r}. Choose one of: {choices}.")
    points = np.asarray(_COLORMAP_POINTS[name], dtype=np.float32)
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
