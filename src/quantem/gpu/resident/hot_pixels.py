"""Public hot-pixel correction policy and provenance.

A detector flags pixels whose stored values are not measurements (dead,
hot, or a sentinel such as 65535). ``"median"`` replaces each one by the
median of the valid pixels around it, so the fill carries no bias: the
median of an even number of neighbors rounds half to even (never down), and a
flagged pixel with no valid pixel in its 3 x 3 window takes the median of the
smallest larger window (5 x 5, 7 x 7, ...) that holds one, never 0. The CPU
reference here and the Metal and CUDA kernels share one neighbor plan and the
same integer arithmetic, so every backend stores the same counts bit for bit.
"""

from dataclasses import dataclass

import numpy as np

from quantem.gpu.formats.qem.metadata import HOT_PIXEL_SCHEMA


def normalize_hot_pixel_correction(value: str) -> str:
    """Validate the public hot-pixel correction name."""
    method = str(value).strip().lower()
    if method not in {"median", "zero", "none"}:
        raise ValueError(
            "hot_pixel_correction must be 'median', 'zero', or 'none'."
        )
    return method


@dataclass(frozen=True)
class MedianPlan:
    """Which valid pixels fill each flagged pixel, as flat detector indices.

    The neighbors of flagged pixel ``bad[i]`` are
    ``neighbors[offsets[i]:offsets[i + 1]]``: the valid pixels of the smallest
    square window around it, of side ``windows[i]``, that holds any.
    """

    bad: np.ndarray
    offsets: np.ndarray
    neighbors: np.ndarray
    windows: np.ndarray

    @property
    def largest_window_px(self) -> int:
        return int(self.windows.max()) if self.windows.size else 3


def median_plan(pixel_mask) -> MedianPlan:
    """Plan the median fill of every flagged pixel of a detector mask (nonzero = flagged)."""
    flagged = np.asarray(pixel_mask) != 0
    valid = ~flagged
    if flagged.any() and not valid.any():
        raise ValueError("Every detector pixel is flagged; no valid neighbor exists for a median fill.")
    columns = flagged.shape[1] if flagged.ndim == 2 else 0
    bad, offsets, neighbors, windows = [], [0], [], []
    for row, column in np.argwhere(flagged):
        radius = 1
        while True:
            top, left = max(0, row - radius), max(0, column - radius)
            window = valid[top: row + radius + 1, left: column + radius + 1]
            if window.any():
                near_rows, near_columns = np.nonzero(window)
                neighbors.extend(((near_rows + top) * columns + near_columns + left).tolist())
                break
            radius += 1
        bad.append(row * columns + column)
        offsets.append(len(neighbors))
        windows.append(2 * radius + 1)
    return MedianPlan(
        bad=np.asarray(bad, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int32),
        neighbors=np.asarray(neighbors, dtype=np.int32),
        windows=np.asarray(windows, dtype=np.int32),
    )


def fill_flagged(frames: np.ndarray, plan: MedianPlan, method: str) -> np.ndarray:
    """Correct the flagged pixels of a ``(frame, detector_row, detector_col)`` block of counts in place.

    The CPU reference of the Metal and CUDA kernels. ``"zero"`` writes 0;
    ``"median"`` writes the median of each pixel's planned neighbors, the mean
    of the two middle values rounded half to even when their number is even.
    Returns ``frames``.
    """
    if method == "none" or not plan.bad.size:
        return frames
    flat = frames.reshape(len(frames), -1)
    if not np.shares_memory(flat, frames):
        raise ValueError("fill_flagged corrects contiguous frames in place.")
    if method == "zero":
        flat[:, plan.bad] = 0
        return frames
    for slot, pixel in enumerate(plan.bad):
        near = flat[:, plan.neighbors[plan.offsets[slot]: plan.offsets[slot + 1]]].astype(np.uint64)
        near.sort(axis=1)
        count = near.shape[1]
        total = near[:, (count - 1) // 2] + near[:, count // 2]
        half = total >> np.uint64(1)
        # an odd total is a .5 median: round it to the even neighbor
        flat[:, pixel] = half + (total & half & np.uint64(1))
    return frames


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
        "schema": HOT_PIXEL_SCHEMA,
        "method": method,
        # the side of the largest window a median fill needed: 3 unless a flagged cluster hid every 3 x 3 neighbor
        "largest_window_px": median_plan(mask).largest_window_px if method == "median" and applied else None,
        "source": "stored_pixel_mask",
        "pixel_count": len(coordinates),
        "coordinates_row_col": coordinates.tolist(),
        "source_flags": [int(mask[tuple(point)]) for point in coordinates],
        "applied": applied,
        "stage": "gpu_load_before_resident_encoding" if applied else "none",
        "backend": backend,
    }


def hot_pixel_outcome(correction: dict) -> str:
    """What a load did with the flagged pixels, and the setting that does the other: the end of its one load line."""
    count = int(correction["pixel_count"])
    flagged = f"{count} bad pixel{'s' if count != 1 else ''}"
    if count == 0:
        return "no bad pixels flagged"
    if correction["method"] == "median":
        window = correction.get("largest_window_px") or 3
        wider = f", windows up to {window} x {window} where a cluster hides every 3 x 3 neighbor" if window > 3 else ""
        return f'{flagged} replaced by neighbor median{wider} (hot_pixel_correction="none" keeps the recorded values)'
    if correction["method"] == "zero":
        return f'{flagged} set to zero (hot_pixel_correction="none" keeps the recorded values)'
    return f'{flagged} left as recorded (hot_pixel_correction="median" replaces them with the neighbor median)'
