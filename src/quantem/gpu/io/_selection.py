"""Exact scan indexing and explicit scientist-requested selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import cupy as cp

ScanOrder = Literal["row-major", "serpentine"]


def _apply_scan_shape(
    data: cp.ndarray,
    explicit: tuple[int, int] | None,
    meta: dict,
    scan_order: str = "row-major",
) -> cp.ndarray:
    """Reshape 3D ``(N, det_r, det_c)`` → 4D ``(scan_r, scan_c, det_r, det_c)``.

    Uses ``explicit`` when the caller passed ``scan_shape=``, else
    ``meta["scan_shape"]`` (auto-derived from ``ntrigger``). No-op when
    no shape is available. ``scan_order="serpentine"`` reverses odd scan rows
    after unflattening so downstream code sees normal ``(row, col)`` order.
    """
    order = _normalize_scan_order(scan_order)
    shape = explicit if explicit is not None else meta.get("scan_shape")
    if shape is None:
        return data
    scan_r, scan_c = shape
    if data.ndim == 3:
        if scan_r * scan_c != data.shape[0]:
            raise ValueError(
                f"scan_shape {shape} incompatible with frame count {data.shape[0]}"
            )
        dr, dc = data.shape[-2:]
        data = data.reshape(scan_r, scan_c, dr, dc)
    elif data.ndim == 4:
        if tuple(int(v) for v in data.shape[:2]) != (int(scan_r), int(scan_c)):
            return data
    else:
        return data
    return _apply_scan_order(data, order)


def _normalize_scan_order(scan_order: str | None) -> ScanOrder:
    """Normalize accepted flattened scan order names."""
    key = "row-major" if scan_order is None else str(scan_order).lower()
    key = key.replace("_", "-").replace(" ", "-")
    aliases: dict[str, ScanOrder] = {
        "row-major": "row-major",
        "raster": "row-major",
        "serpentine": "serpentine",
        "snake": "serpentine",
        "boustrophedon": "serpentine",
    }
    if key not in aliases:
        raise ValueError(
            "scan_order must be 'row-major' or 'serpentine' "
            f"(got {scan_order!r})"
        )
    return aliases[key]


def _apply_scan_order(data: cp.ndarray, scan_order: ScanOrder) -> cp.ndarray:
    """Apply scan-order correction in-place on an already 4D scan array."""
    if scan_order == "row-major" or data.ndim != 4:
        return data
    # Reverse one scan row at a time to avoid materializing a full second
    # 4D array for no-bin 512/1024 acquisitions.
    for row in range(1, int(data.shape[0]), 2):
        data[row] = data[row, ::-1].copy()
    return data


def _normalize_scan_region(
    scan_region,
    scan_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Validate a scan-space region as ``(row_start, row_stop, col_start, col_stop)``.

    The public form is intentionally simple:
    ``(row_start, row_stop, col_start, col_stop)``.
    """
    if not isinstance(scan_region, (tuple, list)) or len(scan_region) != 4:
        raise TypeError(
            "scan_region must be (row_start, row_stop, col_start, col_stop)"
        )
    try:
        row_start, row_stop, col_start, col_stop = (int(v) for v in scan_region)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "scan_region must be (row_start, row_stop, col_start, col_stop)"
        ) from exc

    scan_r, scan_c = (int(v) for v in scan_shape)
    if not (0 <= row_start < row_stop <= scan_r):
        raise ValueError(
            f"scan row region [{row_start}, {row_stop}) is outside scan height {scan_r}"
        )
    if not (0 <= col_start < col_stop <= scan_c):
        raise ValueError(
            f"scan column region [{col_start}, {col_stop}) is outside scan width {scan_c}"
        )
    return row_start, row_stop, col_start, col_stop


def _looks_like_single_scan_region(scan_region) -> bool:
    """Return True when scan_region is one ``(row0, row1, col0, col1)`` tuple."""
    if not isinstance(scan_region, (tuple, list)) or len(scan_region) != 4:
        return False
    return not any(isinstance(item, (tuple, list, np.ndarray)) for item in scan_region)


def _scan_regions_by_file(scan_region, n_files: int) -> tuple[list, str]:
    """Normalize a shared or per-file scan-region argument for a master series."""
    if _looks_like_single_scan_region(scan_region):
        return [scan_region] * int(n_files), "shared"
    if not isinstance(scan_region, (tuple, list)) or len(scan_region) != int(n_files):
        raise TypeError(
            "For load([masters], scan_region=...), pass either one "
            "(row_start, row_stop, col_start, col_stop) region or one such "
            "region per master."
        )
    regions = list(scan_region)
    for index, region in enumerate(regions):
        if not _looks_like_single_scan_region(region):
            raise TypeError(
                "Each per-master scan_region entry must be "
                "(row_start, row_stop, col_start, col_stop); "
                f"entry {index} is invalid."
            )
    return regions, "per_file"


def _scan_shifts_by_file(scan_shift_row_col, n_files: int) -> np.ndarray:
    """Normalize one or per-file scan shifts as ``(row_shift, col_shift)``."""
    shifts = np.asarray(scan_shift_row_col, dtype=np.float32)
    if shifts.shape == (2,):
        return np.tile(shifts.reshape(1, 2), (int(n_files), 1))
    if shifts.shape == (int(n_files), 2):
        return shifts.astype(np.float32, copy=False)
    raise TypeError(
        "scan_shift_row_col must be one (row_shift, col_shift) pair or an "
        f"(n_files, 2) array; got shape {shifts.shape} for {n_files} files."
    )


def _scan_region_dict(
    region: tuple[int, int, int, int],
) -> dict[str, int | list[int]]:
    """Return JSON-friendly scan-region metadata."""
    row_start, row_stop, col_start, col_stop = (int(value) for value in region)
    return {
        "row_start": row_start,
        "row_stop": row_stop,
        "col_start": col_start,
        "col_stop": col_stop,
        "shape": [row_stop - row_start, col_stop - col_start],
    }


def _normalize_detector_region(
    detector_region,
    detector_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Validate a detector-space region as ``(row_start, row_stop, col_start, col_stop)``."""
    if not isinstance(detector_region, (tuple, list)) or len(detector_region) != 4:
        raise TypeError(
            "detector_region must be "
            "(row_start, row_stop, col_start, col_stop)"
        )
    try:
        row_start, row_stop, col_start, col_stop = (
            int(value) for value in detector_region
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "detector_region must be "
            "(row_start, row_stop, col_start, col_stop)"
        ) from exc

    detector_rows, detector_cols = (int(value) for value in detector_shape)
    if not (0 <= row_start < row_stop <= detector_rows):
        raise ValueError(
            "detector row region "
            f"[{row_start}, {row_stop}) is outside detector height {detector_rows}"
        )
    if not (0 <= col_start < col_stop <= detector_cols):
        raise ValueError(
            "detector column region "
            f"[{col_start}, {col_stop}) is outside detector width {detector_cols}"
        )
    return row_start, row_stop, col_start, col_stop


def _detector_region_dict(
    region: tuple[int, int, int, int],
) -> dict[str, int | list[int]]:
    """Return JSON-friendly detector-region metadata."""
    row_start, row_stop, col_start, col_stop = (int(value) for value in region)
    return {
        "row_start": row_start,
        "row_stop": row_stop,
        "col_start": col_start,
        "col_stop": col_stop,
        "shape": [row_stop - row_start, col_stop - col_start],
    }


def _scan_region_frame_indices(
    scan_region: tuple[int, int, int, int],
    scan_shape: tuple[int, int],
    scan_order: str = "row-major",
) -> np.ndarray:
    """Map a rectangular scan-space ROI to flattened detector frame indices."""
    row_start, row_stop, col_start, col_stop = _normalize_scan_region(
        scan_region, scan_shape
    )
    order = _normalize_scan_order(scan_order)
    scan_c = int(scan_shape[1])
    rows = np.arange(row_start, row_stop, dtype=np.int64)
    cols = np.arange(col_start, col_stop, dtype=np.int64)
    if order == "row-major":
        return (rows[:, None] * scan_c + cols[None, :]).reshape(-1)

    frame_indices = np.empty((len(rows), len(cols)), dtype=np.int64)
    for out_row, row in enumerate(rows):
        physical_cols = cols if int(row) % 2 == 0 else (scan_c - 1 - cols)
        frame_indices[out_row] = int(row) * scan_c + physical_cols
    return frame_indices.reshape(-1)


def _scan_positions_to_frame_indices(
    rows: np.ndarray,
    cols: np.ndarray,
    scan_shape: tuple[int, int],
    scan_order: str = "row-major",
) -> np.ndarray:
    """Map logical scan ``(row, col)`` positions to flattened HDF5 frames."""
    order = _normalize_scan_order(scan_order)
    scan_r, scan_c = (int(v) for v in scan_shape)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows.shape != cols.shape:
        raise ValueError("scan position rows and columns must have matching shape")
    if rows.size == 0:
        raise ValueError("scan_indices must contain at least one scan position")
    if np.any(rows < 0) or np.any(rows >= scan_r):
        bad = rows[(rows < 0) | (rows >= scan_r)][0]
        raise ValueError(f"scan row {int(bad)} is outside scan height {scan_r}")
    if np.any(cols < 0) or np.any(cols >= scan_c):
        bad = cols[(cols < 0) | (cols >= scan_c)][0]
        raise ValueError(f"scan column {int(bad)} is outside scan width {scan_c}")

    physical_cols = cols.copy()
    if order == "serpentine":
        odd = rows % 2 == 1
        physical_cols[odd] = scan_c - 1 - physical_cols[odd]
    return rows * scan_c + physical_cols


def _frame_indices_to_scan_positions(
    frame_indices: np.ndarray,
    scan_shape: tuple[int, int],
    scan_order: str = "row-major",
) -> np.ndarray:
    """Map flattened HDF5 frame indices back to logical scan ``(row, col)``."""
    order = _normalize_scan_order(scan_order)
    scan_r, scan_c = (int(v) for v in scan_shape)
    total = scan_r * scan_c
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if frame_indices.size == 0:
        raise ValueError("scan_indices must contain at least one scan position")
    if np.any(frame_indices < 0) or np.any(frame_indices >= total):
        bad = frame_indices[(frame_indices < 0) | (frame_indices >= total)][0]
        raise ValueError(
            f"scan frame index {int(bad)} is outside flattened scan size {total}"
        )

    rows = frame_indices // scan_c
    physical_cols = frame_indices % scan_c
    cols = physical_cols.copy()
    if order == "serpentine":
        odd = rows % 2 == 1
        cols[odd] = scan_c - 1 - cols[odd]
    return np.stack([rows, cols], axis=1).astype(np.int64, copy=False)


def _normalize_scan_indices(
    scan_indices,
    scan_shape: tuple[int, int],
    scan_order: str = "row-major",
    index_mode: str = "scan",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate stochastic scan positions and return HDF5 frames + row/col.

    ``scan_indices`` accepts either a flat vector of logical row-major scan
    indices or an ``(N, 2)`` array of logical ``(row, col)`` positions. Flat
    indices default to logical scan coordinates, matching PyTorch-style
    samplers; pass ``index_mode="hdf5"`` only when the caller already has
    physical flattened detector-frame indices from the file.
    """
    mode = str(index_mode).lower().replace("_", "-")
    if mode not in {"scan", "hdf5"}:
        raise ValueError("index_mode must be 'scan' or 'hdf5'")

    arr = np.asarray(scan_indices)
    if arr.ndim == 1:
        flat = arr.astype(np.int64, copy=False).reshape(-1)
        if mode == "hdf5":
            positions = _frame_indices_to_scan_positions(
                flat,
                scan_shape,
                scan_order,
            )
            return flat.copy(), positions

        scan_r, scan_c = (int(v) for v in scan_shape)
        total = scan_r * scan_c
        if flat.size == 0:
            raise ValueError("scan_indices must contain at least one scan position")
        if np.any(flat < 0) or np.any(flat >= total):
            bad = flat[(flat < 0) | (flat >= total)][0]
            raise ValueError(
                f"scan index {int(bad)} is outside flattened scan size {total}"
            )
        rows = flat // scan_c
        cols = flat % scan_c
        frame_indices = _scan_positions_to_frame_indices(
            rows,
            cols,
            scan_shape,
            scan_order,
        )
        positions = np.stack([rows, cols], axis=1).astype(np.int64, copy=False)
        return frame_indices, positions

    if arr.ndim == 2 and arr.shape[1] == 2:
        if mode == "hdf5":
            raise ValueError(
                "index_mode='hdf5' expects a flat vector of HDF5 frame indices, "
                "not an (N, 2) row/column array"
            )
        rows = arr[:, 0].astype(np.int64, copy=False)
        cols = arr[:, 1].astype(np.int64, copy=False)
        frame_indices = _scan_positions_to_frame_indices(
            rows,
            cols,
            scan_shape,
            scan_order,
        )
        positions = np.stack([rows, cols], axis=1).astype(np.int64, copy=False)
        return frame_indices, positions

    raise TypeError(
        "scan_indices must be a flat vector of scan indices or an "
        "(N, 2) array of (row, col) scan positions"
    )


def _normalize_scan_indices_by_file(
    scan_indices,
    n_files: int,
    scan_shape: tuple[int, int],
    scan_order: str = "row-major",
    index_mode: str = "scan",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Normalize common or per-file stochastic scan indices for file lists."""
    n_files = int(n_files)
    arr = np.asarray(scan_indices)

    # Common positions for every file: flat (N,) or row/col (N, 2).
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[-1] == 2):
        frames, positions = _normalize_scan_indices(
            arr,
            scan_shape,
            scan_order,
            index_mode,
        )
        return [frames.copy() for _ in range(n_files)], [
            positions.copy() for _ in range(n_files)
        ]

    # Per-file flat logical scan indices: (n_files, n_positions).
    if arr.ndim == 2 and arr.shape[0] == n_files:
        frame_lists: list[np.ndarray] = []
        position_lists: list[np.ndarray] = []
        for i in range(n_files):
            frames, positions = _normalize_scan_indices(
                arr[i],
                scan_shape,
                scan_order,
                index_mode,
            )
            frame_lists.append(frames)
            position_lists.append(positions)
        return frame_lists, position_lists

    # Per-file row/column scan positions: (n_files, n_positions, 2).
    if arr.ndim == 3 and arr.shape[0] == n_files and arr.shape[-1] == 2:
        frame_lists = []
        position_lists = []
        for i in range(n_files):
            frames, positions = _normalize_scan_indices(
                arr[i],
                scan_shape,
                scan_order,
                index_mode,
            )
            frame_lists.append(frames)
            position_lists.append(positions)
        return frame_lists, position_lists

    raise TypeError(
        "For multiple files, scan_indices must be common positions shaped "
        "(N,), (N, 2), or per-file positions shaped (n_files, N) / "
        "(n_files, N, 2)"
    )


def _normalize_random_position_count(n: int) -> int:
    """Validate a requested stochastic scan-position count."""
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("random_positions must be a positive integer count")
    try:
        count = int(n)
    except (TypeError, ValueError) as exc:
        raise TypeError("random_positions must be a positive integer count") from exc
    if count <= 0:
        raise ValueError("random_positions must be a positive integer count")
    return count


def random_scan_indices(
    n: int,
    scan_shape: tuple[int, int],
    *,
    n_files: int | None = None,
    seed: int | np.random.Generator | None = None,
    replace: bool = False,
    same_for_all_files: bool = False,
    return_positions: bool = False,
) -> np.ndarray:
    """Sample logical row-major scan indices for stochastic HDF5 minibatches.

    Parameters
    ----------
    n
        Number of scan positions to sample per file.
    scan_shape
        Full scan shape as ``(rows, cols)``.
    n_files
        When provided, return independent per-file samples shaped
        ``(n_files, n)``. If ``same_for_all_files=True``, return one common
        ``(n,)`` sample that can be reused for every file.
    seed
        Optional reproducibility seed, or an existing NumPy ``Generator``.
    replace
        Sample with replacement. Defaults to ``False`` for ptychography-style
        minibatches that should not duplicate positions in one file unless
        explicitly requested.
    same_for_all_files
        Use one common random sample for every file instead of independent
        per-file positions.
    return_positions
        Return logical ``(row, col)`` positions instead of flat scan indices.

    Returns
    -------
    np.ndarray
        ``(n,)`` / ``(n, 2)`` for a single/common sample, or
        ``(n_files, n)`` / ``(n_files, n, 2)`` for independent per-file samples.
    """
    count = _normalize_random_position_count(n)
    scan_r, scan_c = (int(v) for v in scan_shape)
    if scan_r <= 0 or scan_c <= 0:
        raise ValueError("scan_shape must contain positive row/column sizes")
    total = scan_r * scan_c
    if not replace and count > total:
        raise ValueError(
            f"Cannot sample {count} random positions without replacement from "
            f"scan_shape={tuple(scan_shape)} ({total} positions)."
        )
    if n_files is not None:
        n_files = int(n_files)
        if n_files <= 0:
            raise ValueError("n_files must be positive when provided")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def _one() -> np.ndarray:
        return rng.choice(total, size=count, replace=replace).astype(np.int64, copy=False)

    if n_files is None or same_for_all_files:
        indices = _one()
    else:
        indices = np.vstack([_one() for _ in range(n_files)])

    if not return_positions:
        return indices

    rows = indices // scan_c
    cols = indices % scan_c
    return np.stack([rows, cols], axis=-1).astype(np.int64, copy=False)


def _drift_scan_positions(
    scan_positions,
    drift,
    *,
    scan_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply one dense drift field to each frame's shared scan positions.

    The field is sampled exactly at each selected integer scan position. Raw
    diffraction patterns are not resampled, so fractional offsets remain
    available to the ptychography forward model.
    """
    positions = np.asarray(scan_positions, dtype=np.float32)
    raw_drift = drift
    if hasattr(raw_drift, "detach"):
        raw_drift = raw_drift.detach().cpu().numpy()
    fields = np.asarray(raw_drift, dtype=np.float32)
    shared_positions = positions.ndim == 2 and positions.shape[-1] == 2
    if shared_positions:
        positions = positions[None, ...]
    elif positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("scan_positions must have shape (N, 2) or (n_files, N, 2)")
    if fields.ndim == 3 and fields.shape[-1] == 2:
        fields = fields[None, ...]
    elif fields.ndim == 4 and fields.shape[-1] == 2:
        pass
    elif fields.ndim == 4 and fields.shape[1] == 2:
        fields = np.moveaxis(fields, 1, -1)
    else:
        raise ValueError(
            "drift must have shape (frames, rows, cols, 2) or "
            "(frames, 2, rows, cols)"
        )
    if shared_positions and fields.shape[0] > 1:
        positions = np.broadcast_to(positions, (fields.shape[0], *positions.shape[1:]))
    if fields.shape[0] != positions.shape[0]:
        raise ValueError("drift must contain one field per source frame")
    field_shape = tuple(int(v) for v in fields.shape[1:3])
    if scan_shape is not None and tuple(int(v) for v in scan_shape) != field_shape:
        raise ValueError(
            f"drift scan shape {field_shape} does not match "
            f"scan_shape={tuple(scan_shape)}"
        )
    positions_int = np.rint(positions).astype(np.int64)
    if not np.array_equal(positions, positions_int):
        raise ValueError("scan_positions must be integer logical scan positions")
    if np.any(positions_int[..., 0] < 0) or np.any(positions_int[..., 0] >= field_shape[0]):
        raise ValueError("scan_positions row is outside the drift field")
    if np.any(positions_int[..., 1] < 0) or np.any(positions_int[..., 1] >= field_shape[1]):
        raise ValueError("scan_positions column is outside the drift field")
    if not np.isfinite(positions).all() or not np.isfinite(fields).all():
        raise ValueError("scan_positions and drift must contain finite values")
    frame_ids = np.arange(positions.shape[0])[:, None]
    offsets = fields[frame_ids, positions_int[..., 0], positions_int[..., 1]]
    return np.ascontiguousarray(positions + offsets, dtype=np.float32)
