"""CPU float reference for the shared scientific image-display arithmetic.

Test-only: the parity tests compare WebGPU, Metal, and the bundled goldens
against these functions. The reference fixes the arithmetic shared by CUDA, Metal, and WebGPU display
paths: finite float32 values are optionally mapped with signed ``log1p``,
normalized between transformed display limits, assigned to 256 histogram bins,
and mapped through a 256-entry RGB lookup table.
"""

import math
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



def normalize_rotation_degrees(rotation_degrees: float) -> float:
    """Return a validated finite in-plane rotation angle in degrees."""
    if isinstance(rotation_degrees, (bool, np.bool_)):
        raise ValueError("rotation_degrees must be a finite number, not bool")
    try:
        angle = float(rotation_degrees)
    except (TypeError, ValueError) as exc:
        raise ValueError("rotation_degrees must be a finite number") from exc
    if not math.isfinite(angle):
        raise ValueError(
            f"rotation_degrees must be finite, got {rotation_degrees!r}"
        )
    return angle


def rotate_stack_inplane(
    data: np.ndarray,
    rotation_degrees: float,
) -> np.ndarray:
    """Rotate an ``(N, H, W)`` stack with fixed-shape bilinear interpolation.

    Coordinates follow ``(row, column)``. Values outside the inverse-mapped
    source image use its nearest edge, matching SciPy ``mode="nearest"``.
    Nonzero rotations return contiguous float32 data; an exact full-turn is an
    identity and returns the original array.
    """
    angle = normalize_rotation_degrees(rotation_degrees)
    if math.isclose(angle % 360.0, 0.0, abs_tol=1e-12):
        return data
    source = np.asarray(data)
    if source.ndim != 3:
        raise ValueError(
            "rotate_stack_inplane expects data shaped (frames, rows, columns); "
            f"got {source.shape}"
        )
    src = np.ascontiguousarray(source, dtype=np.float32)
    frames, rows, columns = src.shape
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    grid_row, grid_column = np.indices((rows, columns), dtype=np.float32)
    center_row = (rows - 1) / 2.0
    center_column = (columns - 1) / 2.0
    column_offset = grid_column - center_column
    row_offset = grid_row - center_row
    source_column = cosine * column_offset - sine * row_offset + center_column
    source_row = sine * column_offset + cosine * row_offset + center_row
    np.clip(source_column, 0.0, float(columns - 1), out=source_column)
    np.clip(source_row, 0.0, float(rows - 1), out=source_row)
    column0 = np.floor(source_column).astype(np.intp, copy=False)
    row0 = np.floor(source_row).astype(np.intp, copy=False)
    column1 = np.minimum(column0 + 1, columns - 1)
    row1 = np.minimum(row0 + 1, rows - 1)
    column_fraction = source_column - column0
    row_fraction = source_row - row0
    weight00 = ((1 - column_fraction) * (1 - row_fraction)).astype(np.float32).ravel()
    weight01 = (column_fraction * (1 - row_fraction)).astype(np.float32).ravel()
    weight10 = ((1 - column_fraction) * row_fraction).astype(np.float32).ravel()
    weight11 = (column_fraction * row_fraction).astype(np.float32).ravel()
    index00 = (row0 * columns + column0).ravel()
    index01 = (row0 * columns + column1).ravel()
    index10 = (row1 * columns + column0).ravel()
    index11 = (row1 * columns + column1).ravel()
    output = np.empty_like(src, dtype=np.float32)
    src_flat = src.reshape(frames, rows * columns)
    out_flat = output.reshape(frames, rows * columns)
    for frame_index in range(frames):
        frame = src_flat[frame_index]
        destination = out_flat[frame_index]
        np.multiply(frame[index00], weight00, out=destination)
        destination += frame[index01] * weight01
        destination += frame[index10] * weight10
        destination += frame[index11] * weight11
    return output
