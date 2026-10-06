"""Backend-neutral numerical reference for scientific image display.

The reference fixes the arithmetic shared by CUDA, Metal, and WebGPU display
paths: finite float32 values are optionally mapped with signed ``log1p``,
normalized between transformed display limits, assigned to 256 histogram bins,
and mapped through a 256-entry RGB lookup table.
"""

from typing import Literal

import numpy as np

DisplayScale = Literal["linear", "log"]


def dequantize_uint8(
    values: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Restore uint8 display samples to float32 physical values.

    A collapsed or reversed range is treated as a constant at ``low``. This is
    the encoding used by QuantEM standalone widget exports.
    """
    encoded = np.asarray(values, dtype=np.uint8)
    low32 = np.float32(low)
    if not np.isfinite(low32):
        low32 = np.float32(0)
    high32 = np.float32(high)
    if not np.isfinite(high32):
        high32 = low32
    scale = (
        (high32 - low32) / np.float32(255)
        if high32 > low32
        else np.float32(0)
    )
    return (encoded.astype(np.float32) * scale + low32).astype(
        np.float32,
        copy=False,
    )


def transform(values: np.ndarray, scale: DisplayScale = "linear") -> np.ndarray:
    """Return float32 display values under linear or signed-log scaling."""
    result = np.asarray(values, dtype=np.float32)
    if scale == "linear":
        return result
    if scale != "log":
        raise ValueError("scale must be 'linear' or 'log'")
    return np.copysign(np.log1p(np.abs(result)), result).astype(
        np.float32,
        copy=False,
    )


def normalize(
    values: np.ndarray,
    low: float,
    high: float,
    scale: DisplayScale = "linear",
) -> np.ndarray:
    """Normalize values to ``[0, 1]`` using the shared display arithmetic."""
    low32 = np.float32(low)
    high32 = np.maximum(low32, np.float32(high))
    transformed = transform(np.asarray(values, dtype=np.float32), scale)
    transformed_low = transform(np.asarray(low32), scale)
    transformed_high = transform(np.asarray(high32), scale)
    if not transformed_high > transformed_low:
        return np.full(transformed.shape, 0.5, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        span = transformed_high - transformed_low
    if np.isfinite(span):
        normalized = (transformed - transformed_low) / span
    else:
        negative_magnitude = -transformed_low
        if negative_magnitude <= transformed_high:
            ratio = negative_magnitude / transformed_high
            center = ratio / (np.float32(1) + ratio)
        else:
            ratio = transformed_high / negative_magnitude
            center = np.float32(1) / (np.float32(1) + ratio)
        normalized = np.where(
            transformed >= 0,
            center + (np.float32(1) - center) * transformed / transformed_high,
            center * (np.float32(1) - transformed / transformed_low),
        )
    normalized = np.clip(
        normalized,
        np.float32(0),
        np.float32(1),
    )
    return np.nan_to_num(
        normalized,
        copy=False,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def histogram(
    values: np.ndarray,
    low: float,
    high: float,
    scale: DisplayScale = "linear",
) -> np.ndarray:
    """Return exact raw counts for finite values in 256 display bins."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    normalized = normalize(flat[np.isfinite(flat)], low, high, scale)
    indices = np.minimum(
        np.floor(normalized * np.float32(256)).astype(np.intp),
        255,
    )
    return np.bincount(indices, minlength=256).astype(np.uint32)


def colorize(
    values: np.ndarray,
    lut: np.ndarray,
    low: float,
    high: float,
    scale: DisplayScale = "linear",
) -> np.ndarray:
    """Map values to exact uint8 RGBA through a 256-entry LUT."""
    colors = np.asarray(lut)
    if colors.shape not in {(256, 3), (256, 4)}:
        raise ValueError("lut must have shape (256, 3) or (256, 4)")
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.floor(colors * np.float32(255) + np.float32(0.5)).astype(
            np.uint8
        )
    else:
        colors = colors.astype(np.uint8, copy=False)
    indices = np.minimum(
        np.floor(normalize(values, low, high, scale) * np.float32(255)).astype(
            np.intp
        ),
        255,
    )
    rgba = np.empty((*indices.shape, 4), dtype=np.uint8)
    rgba[..., :3] = colors[indices, :3]
    rgba[..., 3] = colors[indices, 3] if colors.shape[1] == 4 else 255
    return rgba


__all__ = [
    "DisplayScale",
    "colorize",
    "dequantize_uint8",
    "histogram",
    "normalize",
    "transform",
]
