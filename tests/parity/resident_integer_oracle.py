"""Independent decoded-array oracle for resident integer contract fixtures.

No production geometry, decoder, detector, or lifecycle helper is imported.
Literal fixture expectations are frozen separately from this calculation.
"""

import json
from pathlib import Path

import numpy as np

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "resident_integer_products_v1.json"
)


def _fixture() -> dict:
    """Read immutable vectors, not regenerated backend output."""
    return json.loads(FIXTURE_PATH.read_text())


def _working_source(case: dict) -> np.ndarray:
    """Apply the declared source exclusions before scientific selection."""
    frames = np.asarray(case["raw_frames_u16"], dtype=np.uint16).copy()
    frames[:, case["excluded_detector_flat_indices"]] = 0
    return frames.reshape(case["shape"])


def _sum_mask(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum Python integers to avoid sharing an accumulator with a backend."""
    selected = np.flatnonzero(mask)
    frames = source.reshape(-1, mask.size)
    result = [sum(int(frame[pixel]) for pixel in selected) for frame in frames]
    return np.asarray(result, dtype=np.uint64).reshape(source.shape[:2])


def _half_open_mask(shape: tuple[int, int], geometry: dict) -> np.ndarray:
    """Rasterize the named binary32 profile without production mask helpers."""
    center_row, center_column = map(
        np.float32, geometry["center_row_column"]
    )
    inner = np.float32(geometry["inner_radius_px"])
    outer = np.float32(geometry["outer_radius_px"])
    mask = np.zeros(shape, dtype=np.uint8)
    for row in range(shape[0]):
        for column in range(shape[1]):
            row_offset = np.float32(row) - center_row
            column_offset = np.float32(column) - center_column
            distance = np.float32(
                np.float32(row_offset * row_offset)
                + np.float32(column_offset * column_offset)
            )
            mask[row, column] = inner * inner <= distance < outer * outer
    return mask
