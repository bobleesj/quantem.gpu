"""Reference geometry shared by scientific image viewers."""

import math

import numpy as np


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
    yy, xx = np.indices((rows, columns), dtype=np.float32)
    center_row = (rows - 1) / 2.0
    center_column = (columns - 1) / 2.0
    x = xx - center_column
    y = yy - center_row
    source_x = cosine * x - sine * y + center_column
    source_y = sine * x + cosine * y + center_row
    np.clip(source_x, 0.0, float(columns - 1), out=source_x)
    np.clip(source_y, 0.0, float(rows - 1), out=source_y)
    column0 = np.floor(source_x).astype(np.intp, copy=False)
    row0 = np.floor(source_y).astype(np.intp, copy=False)
    column1 = np.minimum(column0 + 1, columns - 1)
    row1 = np.minimum(row0 + 1, rows - 1)
    column_fraction = source_x - column0
    row_fraction = source_y - row0
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


__all__ = ["normalize_rotation_degrees", "rotate_stack_inplane"]
