"""
GPU-accelerated HDF5 loading for 4D-STEM diffraction data.

This module provides high-performance bitshuffle+LZ4 decompression
using CUDA kernels, achieving 4-8x speedup over CPU.

Examples
--------
>>> from quantem.gpu.io import load
>>> data = load('/path/to/file.h5').data
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self

# cupy is the CUDA toolkit, absent on a Mac / plain laptop. Guard it so this
# module imports anywhere and the view/screen path (backend='cpu'/'mps') works
# without CUDA. Every `cp.<x>` use below sits inside a function that only runs
# on the cuda backend, so on an NVIDIA box `cp` is the real module (zero
# overhead) and on a non-CUDA box those functions are never reached. The
# `from __future__ import annotations` line keeps `cp.ndarray` annotations as
# strings so they never evaluate at import.
try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only on non-CUDA hosts
    cp = None
import h5py
import hdf5plugin  # noqa: F401 - registers bitshuffle filter
import numpy as np
from numba import njit, prange

from quantem.gpu.io.uint4 import is_packed_uint4, pack_uint4_cupy

from .constants import BLOCK_SIZE

ScanOrder = Literal["row-major", "serpentine"]
_MASTER_FRAME_SOURCE_CACHE: dict[tuple[str, tuple[str, ...], bool], tuple[list[dict[str, Any]], Any]] = {}
_FRAME_SOURCE_CACHE_ENV = "QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR"
_TRUST_FRAME_SOURCE_CACHE_ENV = "QUANTEM_GPU_HDF5_TRUST_FRAME_SOURCE_CACHE"
_PREPARED_SCAN_CROP_CACHE_ENV = "QUANTEM_GPU_HDF5_PREPARED_SCAN_CROP_CACHE_DIR"
_SCAN_CROP_MAX_GAP_ENV = "QUANTEM_GPU_HDF5_SCAN_CROP_MAX_GAP_BYTES"
_PREPARED_SCAN_CROP_CACHE_VERSION = 1


# Lazy bitshuffle+LZ4 kernel proxies. The kernels compile only on first CALL
# (inside the cuda decompress path), so importing this module never touches
# cupy. Each proxy resolves + caches its real kernel on first use, so the
# per-launch cost after that is one list read.
def _lazy_kernel(_name):
    _cache = []

    def _call(*args, **kwargs):
        if not _cache:
            from .backends.cuda import decoder as _bs
            _cache.append(getattr(_bs, _name))
        return _cache[0](*args, **kwargs)

    return _call


_h5lz4dc_kernel = _lazy_kernel("h5lz4dc_kernel")
_bitshuffle_kernel = _lazy_kernel("bitshuffle_kernel")
_bitshuffle_kernel_u16 = _lazy_kernel("bitshuffle_kernel_u16")
_bitshuffle_tail_kernel_u16 = _lazy_kernel("bitshuffle_tail_kernel_u16")
_bitshuffle_tail_kernel_u32 = _lazy_kernel("bitshuffle_tail_kernel_u32")
_clip_u16_to_u8_kernel = _lazy_kernel("clip_u16_to_u8_kernel")
_clip_u32_to_u8_kernel = _lazy_kernel("clip_u32_to_u8_kernel")
_clip_u16_to_u8_count_kernel = _lazy_kernel("clip_u16_to_u8_count_kernel")
_clip_u32_to_u8_count_kernel = _lazy_kernel("clip_u32_to_u8_count_kernel")
_RESAMPLE_SCAN_CROP_KERNELS: dict[str, Any] = {}

__version__ = "0.0.3"
__all__ = ["load"]


@dataclass(frozen=True)
class MasterReadiness:
    """Header-only readiness report for one 4D-STEM master.

    Parameters
    ----------
    ready
        Whether every selected detector source is readable, internally
        consistent, and contains the expected number of stored frames.
    reason
        Concise description of the observed state.
    action
        Corrective next step when ``ready`` is ``False``.
    source_kind
        ``"inline"`` or ``"external"`` according to the selected
        ``entry/data`` source layout, or ``"unavailable"`` when inspection
        could not identify a data source.
    actual_frames
        Total stored frame count across the selected datasets, when known.
    expected_frames
        Frame count derived from an explicit ``scan_shape`` or discoverable
        master metadata, when available.
    detector_shape
        Common detector shape ``(row, col)``, when known.
    dtype
        Common NumPy dtype string, when known.
    source_signature
        JSON-serializable file-stat and dataset-header fingerprint. Callers can
        compare this dictionary across polls without reading detector pixels.
    """

    ready: bool
    reason: str
    action: str
    source_kind: str
    actual_frames: int | None
    expected_frames: int | None
    detector_shape: tuple[int, int] | None
    dtype: str | None
    source_signature: dict[str, Any]


def _clip_to_uint8_count(src, dst):
    """Clip unsigned CuPy array ``src`` to uint8 ``dst`` and count saturations.

    This keeps exact saturation accounting available without the old generic
    CuPy ``minimum(...).astype(uint8)`` plus ``>255`` two-pass cost.
    """
    n = int(src.size)
    if n == 0:
        return cp.zeros((), dtype=cp.uint64)

    src_dtype = np.dtype(src.dtype)
    if src_dtype not in (np.dtype(np.uint16), np.dtype(np.uint32)):
        return None

    threads = 256
    # Keep the count reduction tiny while still exposing enough parallelism for
    # the 100+ million-pixel batches used by no-bin Arina masters.
    blocks = max(1, min(4096, (n + threads - 1) // threads))
    block_counts = cp.empty(blocks, dtype=cp.uint64)
    kernel = (
        _clip_u16_to_u8_count_kernel
        if src_dtype == np.dtype(np.uint16)
        else _clip_u32_to_u8_count_kernel
    )
    kernel(
        (blocks,),
        (threads,),
        (
            src.reshape(-1),
            dst.reshape(-1),
            np.uint64(n),
            block_counts,
        ),
        shared_mem=threads * np.dtype(np.uint64).itemsize,
    )
    return block_counts.sum(dtype=cp.uint64)


def _clip_to_uint8(src, dst) -> bool:
    """Clip unsigned CuPy array ``src`` to uint8 ``dst`` without counting."""
    n = int(src.size)
    if n == 0:
        return True

    src_dtype = np.dtype(src.dtype)
    if src_dtype not in (np.dtype(np.uint16), np.dtype(np.uint32)):
        return False

    threads = 256
    blocks = max(1, min(4096, (n + threads - 1) // threads))
    kernel = (
        _clip_u16_to_u8_kernel
        if src_dtype == np.dtype(np.uint16)
        else _clip_u32_to_u8_kernel
    )
    kernel(
        (blocks,),
        (threads,),
        (
            src.reshape(-1),
            dst.reshape(-1),
            np.uint64(n),
        ),
    )
    return True


def read_pixel_mask(filepath):
    """Return the Arina pixel_mask array from a master HDF5.

    The Arina detector writes a 2-D `pixel_mask` dataset under
    `entry/instrument/detector/detectorSpecific/` enumerating hardware
    dead pixels (>0 = bad). This is the ONLY sanctioned reader - other
    modules must go through here instead of opening h5py directly, so
    the Arina schema stays in one place.

    Parameters
    ----------
    filepath : str or Path
        Path to an Arina master HDF5 file.

    Returns
    -------
    np.ndarray or None
        Raw (H, W) mask array as stored in the HDF5, or None if the
        file is missing/unreadable or has no `pixel_mask` dataset.
    """
    from pathlib import Path
    try:
        with h5py.File(str(Path(filepath)), "r") as f:
            key = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if key not in f:
                return None
            return f[key][:]
    except (OSError, KeyError):
        return None


# =========================================================================
#  CPU helper (Numba JIT)
# =========================================================================

class LoadResult(NamedTuple):
    """Loaded detector data and acquisition metadata.

    Attributes
    ----------
    data
        Backend-native detector data by default. CUDA returns a CuPy array;
        MPS may return a chunk-backed Metal frame source. With
        ``output="torch"``, this is a Torch tensor. Shape is normally
        ``(scan_row, scan_col, detector_row, detector_col)`` when the scan
        shape is known, otherwise ``(frame, detector_row, detector_col)``.
    metadata
        Acquisition and detector metadata from the HDF5 source. The mapping
        includes these normalized fields:

        **Derived, named fields** (always present; value is ``None`` when
        the source field is missing):

        - ``scan_shape`` : ``(H, W)`` or ``None``
            Auto-derived from ``ntrigger`` assuming a square scan.
        - ``n_frames`` : ``int`` or ``None``
            Total frame count.
        - ``dwell_time_us`` : ``float`` or ``None``
            Per-frame dwell in microseconds.
        - ``detector_shape`` : ``(H, W)`` or ``None``
            Detector pixel count.
        - ``detector_name`` : ``str`` or ``None``
            Human-readable detector description.
        - ``saturation`` : ``int`` or ``None``
            ADU ceiling before the detector saturates.

        **Raw HDF5 scalars**: every scalar dataset in the file keyed by its
        full HDF5 path (e.g. ``metadata["entry/instrument/detector/count_time"]``),
        as an escape hatch for fields not in the derived layer.

        .. note::

            Scope-side parameters (``voltage_kV``, ``semiangle``,
            ``scan_sampling``, ``camera_length``, ``rotation``) are NOT in
            the h5 master - pass them to ``ssb()`` explicitly.

    Examples
    --------
    ```python
    data, meta = load("scan_master.h5")
    data.shape             # (512, 512, 192, 192)
    meta["scan_shape"]     # (512, 512)
    meta["dwell_time_us"]  # 99.6
    meta["detector_name"]  # detector model string
    ```
    """

    data: cp.ndarray
    metadata: dict


def _to_torch_data(data):
    """Convert one backend-native load payload to a Torch tensor."""
    import torch

    if isinstance(data, torch.Tensor):
        return data
    chunks = getattr(data, "chunks", None)
    if chunks is not None:
        target = getattr(data, "device", None) or "cpu"
        frames = torch.cat(
            [torch.as_tensor(np.asarray(chunk), device=target) for chunk in chunks],
            dim=0,
        )
        scan_shape = getattr(data, "scan_shape", None)
        if scan_shape is not None:
            frames = frames.reshape(*scan_shape, *frames.shape[1:])
        return frames
    if hasattr(data, "__dlpack__"):
        return torch.from_dlpack(data)
    return torch.as_tensor(np.asarray(data))


def _convert_load_output(result, output: str):
    """Convert a load result or result list according to ``output``."""
    if output == "native":
        return result
    if output != "torch":
        raise ValueError("output must be 'native' or 'torch'")
    if isinstance(result, list):
        return [
            LoadResult(_to_torch_data(item.data), item.metadata)
            for item in result
        ]
    return LoadResult(_to_torch_data(result.data), result.metadata)


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


def _slice_detector_region(data, region: tuple[int, int, int, int], *, compact: bool):
    """Slice detector rows/columns, optionally returning compact storage."""
    row_start, row_stop, col_start, col_stop = (int(value) for value in region)
    sliced = data[..., row_start:row_stop, col_start:col_stop]
    if not compact:
        return sliced
    if cp is not None and isinstance(sliced, cp.ndarray):
        return cp.ascontiguousarray(sliced)
    return np.ascontiguousarray(sliced)


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


def _resample_scan_crop_kernel(dtype: np.dtype):
    """Return a CUDA kernel for bilinear scan-space resampling."""
    if cp is None:  # pragma: no cover - CUDA-only helper
        raise RuntimeError("scan-crop resampling requires CuPy/CUDA")
    dtype = np.dtype(dtype)
    ctype_by_dtype = {
        np.dtype(np.uint8): "unsigned char",
        np.dtype(np.uint16): "unsigned short",
        np.dtype(np.uint32): "unsigned int",
        np.dtype(np.float32): "float",
    }
    ctype = ctype_by_dtype.get(dtype)
    if ctype is None:
        raise TypeError(
            "scan-crop resampling supports uint8, uint16, uint32, "
            f"and float32 decoded crops; got {dtype}."
        )
    key = dtype.str
    kernel = _RESAMPLE_SCAN_CROP_KERNELS.get(key)
    if kernel is not None:
        return kernel
    name = f"quantem_resample_scan_crop_{dtype.name.replace('float', 'f')}"
    code = f"""
    extern "C" __global__
    void {name}(
        const {ctype}* __restrict__ src,
        float* __restrict__ dst,
        const long long n,
        const int src_rows,
        const int src_cols,
        const int det_rows,
        const int det_cols,
        const int out_rows,
        const int out_cols,
        const long long src_stride_row,
        const long long src_stride_col,
        const long long src_stride_det_row,
        const long long src_stride_det_col,
        const float source_row_start,
        const float source_col_start,
        const float target_row_start,
        const float target_col_start,
        const float shift_row,
        const float shift_col
    ) {{
        long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
        if (index >= n) {{
            return;
        }}

        int det_col = (int)(index % det_cols);
        long long tmp = index / det_cols;
        int det_row = (int)(tmp % det_rows);
        tmp /= det_rows;
        int out_col = (int)(tmp % out_cols);
        int out_row = (int)(tmp / out_cols);

        float src_row = target_row_start + (float)out_row + shift_row - source_row_start;
        float src_col = target_col_start + (float)out_col + shift_col - source_col_start;
        float max_row = src_rows > 1 ? (float)src_rows - 1.001f : 0.0f;
        float max_col = src_cols > 1 ? (float)src_cols - 1.001f : 0.0f;
        src_row = fminf(fmaxf(src_row, 0.0f), max_row);
        src_col = fminf(fmaxf(src_col, 0.0f), max_col);

        int r0 = (int)floorf(src_row);
        int c0 = (int)floorf(src_col);
        int r1 = r0 + 1 < src_rows ? r0 + 1 : src_rows - 1;
        int c1 = c0 + 1 < src_cols ? c0 + 1 : src_cols - 1;
        float wr = src_row - (float)r0;
        float wc = src_col - (float)c0;

        long long base00 = (long long)r0 * src_stride_row
            + (long long)c0 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;
        long long base01 = (long long)r0 * src_stride_row
            + (long long)c1 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;
        long long base10 = (long long)r1 * src_stride_row
            + (long long)c0 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;
        long long base11 = (long long)r1 * src_stride_row
            + (long long)c1 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;

        float p00 = (float)src[base00];
        float p01 = (float)src[base01];
        float p10 = (float)src[base10];
        float p11 = (float)src[base11];
        dst[index] =
            (1.0f - wr) * ((1.0f - wc) * p00 + wc * p01)
            + wr * ((1.0f - wc) * p10 + wc * p11);
    }}
    """
    kernel = cp.RawKernel(code, name)
    _RESAMPLE_SCAN_CROP_KERNELS[key] = kernel
    return kernel


def _resample_scan_crop_cuda(
    data: cp.ndarray,
    *,
    source_scan_region: tuple[int, int, int, int],
    target_scan_region: tuple[int, int, int, int],
    scan_shift_row_col,
    scan_resample_dtype: type | np.dtype = np.float32,
) -> cp.ndarray:
    """Resample a decoded scan crop into one target specimen-coordinate crop."""
    if cp is None:  # pragma: no cover - CUDA-only helper
        raise RuntimeError("scan-crop resampling requires CuPy/CUDA")
    if not isinstance(data, cp.ndarray):
        raise TypeError("scan-crop resampling requires a CuPy input array")
    if data.ndim != 4:
        raise ValueError(
            "scan-crop resampling expects decoded data with shape "
            "(scan_row, scan_col, detector_row, detector_col)"
        )
    output_dtype = np.dtype(scan_resample_dtype)
    if output_dtype != np.dtype(np.float32):
        raise TypeError(
            "scan_resample_dtype currently supports only np.float32 because "
            "subpixel scan resampling creates fractional detector counts."
        )
    row_start, row_stop, col_start, col_stop = (
        int(value) for value in target_scan_region
    )
    out_rows = int(row_stop - row_start)
    out_cols = int(col_stop - col_start)
    src_rows, src_cols, det_rows, det_cols = (int(value) for value in data.shape)
    itemsize = int(data.dtype.itemsize)
    src_stride_row, src_stride_col, src_stride_det_row, src_stride_det_col = (
        int(stride // itemsize) for stride in data.strides
    )
    shifts = np.asarray(scan_shift_row_col, dtype=np.float32)
    if shifts.shape != (2,):
        raise TypeError("one decoded crop expects one (row_shift, col_shift) pair")
    out = cp.empty((out_rows, out_cols, det_rows, det_cols), dtype=cp.float32)
    n = int(out.size)
    if n == 0:
        return out
    kernel = _resample_scan_crop_kernel(np.dtype(data.dtype))
    threads = 256
    blocks = (n + threads - 1) // threads
    source_row_start, _source_row_stop, source_col_start, _source_col_stop = (
        int(value) for value in source_scan_region
    )
    kernel(
        (blocks,),
        (threads,),
        (
            data,
            out,
            np.int64(n),
            np.int32(src_rows),
            np.int32(src_cols),
            np.int32(det_rows),
            np.int32(det_cols),
            np.int32(out_rows),
            np.int32(out_cols),
            np.int64(src_stride_row),
            np.int64(src_stride_col),
            np.int64(src_stride_det_row),
            np.int64(src_stride_det_col),
            np.float32(source_row_start),
            np.float32(source_col_start),
            np.float32(row_start),
            np.float32(col_start),
            np.float32(shifts[0]),
            np.float32(shifts[1]),
        ),
    )
    return out


def resample_scan_crop(
    data: cp.ndarray,
    *,
    source_scan_region: tuple[int, int, int, int],
    target_scan_region: tuple[int, int, int, int],
    scan_shift_row_col,
    output_dtype: type | np.dtype = np.float32,
) -> cp.ndarray:
    """Resample a decoded CUDA scan crop into specimen coordinates.

    Parameters
    ----------
    data
        Decoded CuPy array with shape
        ``(scan_row, scan_col, detector_row, detector_col)``. This function
        does not read HDF5; use :func:`load` for loading.
    source_scan_region
        Source scan bounds of ``data`` in full-frame scan coordinates, as
        ``(row_start, row_stop, col_start, col_stop)``.
    target_scan_region
        Requested output specimen-coordinate bounds, also in ``(row, col)``
        order.
    scan_shift_row_col
        Per-frame scan shift ``(row_shift, col_shift)`` that maps specimen
        coordinates into the source frame.
    output_dtype
        Output dtype. Only ``np.float32`` is currently supported because
        subpixel alignment creates fractional detector counts.

    Returns
    -------
    cupy.ndarray
        Resampled float32 array with shape
        ``(target_rows, aligned_cols, detector_row, detector_col)``.
    """

    return _resample_scan_crop_cuda(
        data,
        source_scan_region=source_scan_region,
        target_scan_region=target_scan_region,
        scan_shift_row_col=scan_shift_row_col,
        scan_resample_dtype=output_dtype,
    )


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


def get_metadata(filepath: str) -> dict:
    """Read all scalar metadata from an HDF5 master file.

    Returns a flat dict that mixes two layers:

    **Derived, named fields** (always present as keys; value is ``None`` when
    the source field is missing from the file):

    - ``scan_shape`` : tuple[int, int] or None
        Scan grid as ``(height, width)``. Derived from ``ntrigger`` assuming
        a square scan. If ``ntrigger`` is not a perfect square, this is
        ``None`` and the caller must pass ``scan_shape=`` to ``load()``
        explicitly.
    - ``n_frames`` : int or None
        Total frame count (``ntrigger``).
    - ``dwell_time_us`` : float or None
        Per-frame dwell in microseconds (``frame_time * 1e6``).
    - ``detector_shape`` : tuple[int, int] or None
        Detector pixel count as ``(height, width)``.
    - ``detector_name`` : str or None
        Human-readable detector description, e.g. ``"Dectris ARINA Si"``.
    - ``saturation`` : int or None
        ADU ceiling before the detector saturates.

    **Raw HDF5 scalars** (schema-agnostic): every scalar dataset in the file
    keyed by its full HDF5 path, e.g.
    ``metadata["entry/instrument/detector/frame_time"]``. Arrays of more
    than 100 elements are skipped. This is the escape hatch when you need a
    field the derived layer does not cover.

    .. note::

        Scope-side parameters (``voltage_kV``, ``semiangle``,
        ``scan_sampling``, ``camera_length``, ``rotation``) are NOT in the
        h5 master - they must be passed to ``ssb()`` explicitly or loaded
        from a site config. If a field is in this dict, it came from the
        file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 master file.

    Returns
    -------
    dict
        Mixed dict of derived named fields and raw h5-path scalars.

    Examples
    --------
    ```python
    m = get_metadata("scan_master.h5")
    m["scan_shape"]       # (512, 512)
    m["dwell_time_us"]    # 49.8
    m["detector_name"]    # detector model string
    # any raw HDF5 scalar is also available by its full path:
    m["entry/instrument/detector/count_time"]   # 9.95e-05
    ```
    """
    metadata: dict = {}
    with h5py.File(filepath, "r") as f:
        def _visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.size > 100:
                return  # skip large arrays (flatfield, pixel_mask, etc.)
            if "data_" in name:
                return  # skip data chunk links
            try:
                val = obj[()]
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                metadata[name] = val
            except (TypeError, ValueError, OSError, UnicodeDecodeError):
                return  # Skip non-scalar/non-readable datasets
        f.visititems(_visit)

        def _copy_attrs(attrs):
            for key, val in attrs.items():
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                metadata.setdefault(key, val)

        _copy_attrs(f.attrs)
        data_group = f.get("entry/data")
        if data_group is not None:
            _copy_attrs(data_group.attrs)

        data_ds = f.get("entry/data/data")
        if data_ds is None and data_group is not None:
            for key in sorted(data_group.keys()):
                if key.startswith("data_"):
                    try:
                        data_ds = data_group[key]
                    except (OSError, KeyError):
                        data_ds = None
                    break
        if data_ds is not None:
            if "scan_shape" in data_ds.attrs:
                metadata.setdefault("scan_shape", tuple(int(x) for x in data_ds.attrs["scan_shape"]))
            if "det_shape" in data_ds.attrs:
                metadata.setdefault("detector_shape", tuple(int(x) for x in data_ds.attrs["det_shape"]))
            if data_ds.ndim >= 3:
                metadata.setdefault("n_frames", int(np.prod(data_ds.shape[:-2])))
    _derive_fields(metadata)
    return metadata


def _derive_fields(metadata: dict) -> None:
    """Promote raw h5-path scalars into named fields on the metadata dict.

    Every derived field is set unconditionally - missing sources land as
    ``None`` so the key is always present and code can do ``meta["scan_shape"]``
    without defensive ``.get()`` calls.
    """
    import math

    ntrigger = metadata.get("entry/instrument/detector/detectorSpecific/ntrigger")
    n_frames = int(ntrigger) if ntrigger is not None else metadata.get("n_frames")
    n_frames = int(n_frames) if n_frames is not None else None

    scan_shape = metadata.get("scan_shape")
    if scan_shape is not None:
        scan_shape = tuple(int(x) for x in scan_shape)
    elif n_frames is not None:
        side = math.isqrt(n_frames)
        scan_shape = (side, side) if side * side == n_frames else None

    frame_time = metadata.get("entry/instrument/detector/frame_time")
    dwell_time_us = float(frame_time) * 1e6 if frame_time is not None else None

    y_pix = metadata.get("entry/instrument/detector/detectorSpecific/y_pixels_in_detector")
    x_pix = metadata.get("entry/instrument/detector/detectorSpecific/x_pixels_in_detector")
    detector_shape = metadata.get("detector_shape")
    if detector_shape is not None:
        detector_shape = tuple(int(x) for x in detector_shape)
    elif y_pix is not None and x_pix is not None:
        detector_shape = (int(y_pix), int(x_pix))

    detector_name = metadata.get("entry/instrument/detector/description")
    saturation_raw = metadata.get("entry/instrument/detector/saturation_value")
    saturation = int(saturation_raw) if saturation_raw is not None else None

    metadata["scan_shape"] = scan_shape
    metadata["n_frames"] = n_frames
    metadata["dwell_time_us"] = dwell_time_us
    metadata["detector_shape"] = detector_shape
    metadata["detector_name"] = detector_name
    metadata["saturation"] = saturation


# =============================================================================
# Velox EMD metadata (#178)
# =============================================================================
#
# Velox (.emd) files contain a per-frame JSON blob at
# ``Data/Image/<hash>/Metadata`` holding microscope-side parameters the
# Arina master never captures: StemMagnification, FullScanFieldOfView,
# AccelerationVoltage, ConvergenceSemiAngle. When a collaborator exports
# the same scan through Velox and drops the EMD next to the master, we
# surface those fields in config.json so the screener can auto-derive
# scan_step_A and show magnification in the list view without anyone
# hand-typing them.

def read_emd_metadata(emd_path) -> dict:
    """Extract scope-side fields from a Velox EMD file.

    Reads the first image's ``Data/Image/<hash>/Metadata`` JSON (Velox
    stores metadata as a uint8 byte vector per frame). Returns a dict
    with whichever of the following keys were found; missing keys are
    omitted so callers can merge via ``dict.update`` without clobbering:

    - ``stem_magnification``    : float, e.g. 5_100_000 for 5.1 Mx
    - ``field_of_view_nm``      : float, FullScanFieldOfView.x in nm
    - ``voltage_kV``            : float, AccelerationVoltage / 1000
    - ``semi_angle_mrad``       : float, probe semiangle when exposed

    Returns ``{}`` on any parse failure so callers can always `.update()`
    the result into an existing config dict without guarding. The EMD
    format version varies across microscope builds, so missing-field
    handling is the common path, not the edge case.
    """
    import json as _json
    from pathlib import Path as _Path
    path = _Path(emd_path)
    if not path.is_file():
        return {}
    try:
        with h5py.File(path, "r") as f:
            if "Data/Image" not in f:
                return {}
            image_group = f["Data/Image"]
            first_hash = next(iter(image_group.keys()), None)
            if first_hash is None:
                return {}
            meta_ds = image_group[first_hash].get("Metadata")
            if meta_ds is None:
                return {}
            # Velox stores metadata as a (nbytes, nframes) uint8 JSON buffer.
            # Frame 0 is sufficient; per-frame blobs are near-identical.
            raw = meta_ds[:, 0] if meta_ds.ndim == 2 else meta_ds[()]
            raw_bytes = bytes(np.asarray(raw).tolist()).rstrip(b"\x00")
            doc = _json.loads(raw_bytes)
    except (OSError, ValueError, KeyError, _json.JSONDecodeError):
        return {}

    out: dict = {}
    optics = doc.get("Optics") or {}
    custom = doc.get("CustomProperties") or {}

    # Velox wraps most scalars as {"type": "double", "value": "5100000"};
    # AccelerationVoltage is historically a bare string. Accept both.
    def _as_float(v):
        if isinstance(v, dict):
            v = v.get("value")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    mag = _as_float(custom.get("StemMagnification"))
    if mag is not None:
        out["stem_magnification"] = mag

    fov = optics.get("FullScanFieldOfView")
    if isinstance(fov, dict):
        fov_x = _as_float(fov.get("x"))
        if fov_x is not None:
            # Velox reports FOV in metres; screener works in nm.
            out["field_of_view_nm"] = fov_x * 1e9

    voltage = _as_float(optics.get("AccelerationVoltage"))
    if voltage is not None:
        out["voltage_kV"] = voltage / 1000.0

    semi = _as_float(optics.get("ConvergenceSemiAngle") or optics.get("SemiConvergenceAngle"))
    if semi is not None:
        # Velox stores the convergence angle in radians.
        out["semi_angle_mrad"] = semi * 1000.0
    return out


def find_emd_sibling(master_path) -> Path | None:
    """Locate a Velox EMD next to an Arina master file.

    Arina writes ``<stem>_master.h5`` alongside data chunk files; when
    the operator also exports the scan to Velox, the EMD usually lands
    in the same folder. Strategy:

    1. Prefer a file named ``<stem>.emd`` (strict match).
    2. Fall back to any ``*.emd`` in the same directory - Dectris
       operators often batch-rename after the fact.

    Returns ``None`` when no EMD sibling is found.
    """
    from pathlib import Path as _Path
    master = _Path(master_path)
    folder = master.parent
    stem = master.stem
    stem = stem.removesuffix("_master")
    candidates = list(folder.glob(f"{stem}.emd")) + list(folder.glob(f"{stem}*.emd"))
    if candidates:
        return candidates[0]
    others = list(folder.glob("*.emd"))
    return others[0] if len(others) == 1 else None


# =============================================================================
# GPU CLASSES AND FUNCTIONS
# =============================================================================
#
# CUDA kernels are resolved lazily from ``io.backends.cuda.decoder``.
#

# The CUDA source lives in ``io.backends.cuda.decoder``.


class GPUDecompressor:
    """GPU-accelerated decompressor for bitshuffle+LZ4 HDF5 datasets.

    Uses pinned memory and CUDA kernels for maximum throughput.
    CUDA kernels are compiled at module import time.
    """

    def __init__(
        self,
        max_compressed_bytes: int = 1024 * 1024 * 1024,
        max_frames: int = 100000,
        max_frame_bytes: int = 192 * 192 * 4,
        n_blocks_per_frame: int = 18,
    ):
        """Initialize the decompressor with pre-allocated buffers.

        Parameters
        ----------
        max_compressed_bytes : int, optional
            Maximum size of compressed data, by default 1GB.
        max_frames : int, optional
            Maximum number of frames to support, by default 100000.
        max_frame_bytes : int, optional
            Maximum bytes per frame, by default 147456 (192x192 uint32).
        n_blocks_per_frame : int, optional
            LZ4 blocks per frame, by default 18 for 192x192 uint32.
        """
        self.max_compressed_bytes = max_compressed_bytes
        self.max_frames = max_frames
        self.max_frame_bytes = max_frame_bytes
        self.n_blocks_per_frame = n_blocks_per_frame
        self._h5lz4dc = _h5lz4dc_kernel
        self._shuf = _bitshuffle_kernel
        # Pinned memory for fast CPU->GPU transfers
        self._pinned_mem = cp.cuda.alloc_pinned_memory(max_compressed_bytes)
        self._pinned_buffer = np.frombuffer(
            self._pinned_mem, dtype=np.uint8, count=max_compressed_bytes
        )
        # Pre-allocated metadata arrays
        self._chunk_sizes = np.zeros(max_frames, dtype=np.uint32)
        # uint64: absolute byte offsets into the compressed read buffer
        # can exceed 4 GB on dense 4D-STEM scans (Arina gold etc.)
        self._chunk_offsets = np.zeros(max_frames, dtype=np.uint64)
        self._block_counts = np.zeros(max_frames, dtype=np.uint32)
        self._block_starts_flat = np.zeros(max_frames * n_blocks_per_frame, dtype=np.uint32)
        self._block_offsets = np.zeros(max_frames + 1, dtype=np.uint32)
        # Pre-allocate all GPU buffers for fast first load()
        self._concat_gpu = cp.empty(max_compressed_bytes, dtype=cp.uint8)
        total_output_bytes = max_frames * max_frame_bytes
        self._lz4_output = cp.empty(total_output_bytes, dtype=cp.uint8)
        self._shuffled_output = cp.empty(total_output_bytes, dtype=cp.uint8)

    def load(
        self,
        filepath: str,
        dataset_path: str = "entry/data/data",
    ) -> cp.ndarray:
        """Load and decompress a bitshuffle+LZ4 HDF5 dataset to GPU.

        Parameters
        ----------
        filepath : str
            Path to the HDF5 file.
        dataset_path : str, optional
            Path to the dataset within the HDF5 file, by default "entry/data/data".

        Returns
        -------
        cp.ndarray
            CuPy array on GPU with shape (n_frames, height, width).
        """
        with h5py.File(filepath, "r") as f:
            ds = f[dataset_path]
            n_frames = ds.shape[0]
            frame_shape = ds.shape[1:]
            dtype = ds.dtype
            frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)

            # Reallocate output GPU buffers if dataset exceeds pre-allocated size
            total_needed = n_frames * frame_bytes
            if total_needed > len(self._lz4_output):
                self._lz4_output = cp.empty(total_needed, dtype=cp.uint8)
                self._shuffled_output = cp.empty(total_needed, dtype=cp.uint8)
            # Reallocate metadata arrays if frame count exceeds capacity
            if n_frames > self.max_frames:
                self.max_frames = n_frames
                self._chunk_sizes = np.zeros(n_frames, dtype=np.uint32)
                self._chunk_offsets = np.zeros(n_frames, dtype=np.uint64)
                self._block_counts = np.zeros(n_frames, dtype=np.uint32)
                self._block_starts_flat = np.zeros(
                    n_frames * self.n_blocks_per_frame, dtype=np.uint32
                )
                self._block_offsets = np.zeros(n_frames + 1, dtype=np.uint32)
            # Read chunks into pinned memory, reallocating if compressed data exceeds buffer
            offset = 0
            for i in range(n_frames):
                _, raw = ds.id.read_direct_chunk((i, 0, 0))
                chunk_len = len(raw)
                # Grow pinned buffer if needed
                if offset + chunk_len > self.max_compressed_bytes:
                    new_size = max(
                        self.max_compressed_bytes * 2,
                        offset + chunk_len + 256 * 1024 * 1024,
                    )
                    new_pinned_mem = cp.cuda.alloc_pinned_memory(new_size)
                    new_pinned_buffer = np.frombuffer(
                        new_pinned_mem, dtype=np.uint8, count=new_size
                    )
                    new_pinned_buffer[:offset] = self._pinned_buffer[:offset]
                    self._pinned_mem = new_pinned_mem
                    self._pinned_buffer = new_pinned_buffer
                    self.max_compressed_bytes = new_size
                    self._concat_gpu = cp.empty(new_size, dtype=cp.uint8)
                self._chunk_offsets[i] = offset
                self._chunk_sizes[i] = chunk_len
                self._pinned_buffer[offset : offset + chunk_len] = np.frombuffer(
                    raw, dtype=np.uint8
                )
                offset += chunk_len
            total_compressed = offset
        # Parse headers
        _parse_headers(
            self._pinned_buffer,
            self._chunk_sizes,
            self._chunk_offsets,
            self._block_starts_flat,
            self._block_counts,
            n_frames,
            self.n_blocks_per_frame,
        )
        # Compute block offsets
        self._block_offsets[1 : n_frames + 1] = np.cumsum(self._block_counts[:n_frames])
        total_blocks = int(self._block_offsets[n_frames])
        # Transfer to GPU
        self._concat_gpu[:total_compressed].set(self._pinned_buffer[:total_compressed])
        chunk_offsets_gpu = cp.asarray(self._chunk_offsets[:n_frames])
        block_starts_gpu = cp.asarray(self._block_starts_flat[:total_blocks])
        block_counts_gpu = cp.asarray(self._block_counts[:n_frames])
        block_offsets_gpu = cp.asarray(self._block_offsets[: n_frames + 1])
        # LZ4 decompress
        max_blocks = int(self._block_counts[:n_frames].max())
        max_batch = 10000
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            self._h5lz4dc(
                ((max_blocks + 1) // 2, 1, batch_n),
                (32, 2, 1),
                (
                    self._concat_gpu,
                    chunk_offsets_gpu[start:],
                    block_starts_gpu,
                    block_counts_gpu[start:],
                    block_offsets_gpu[start:],
                    np.uint32(BLOCK_SIZE),
                    np.uint32(frame_bytes),
                    self._lz4_output[byte_offset:],
                ),
            )
        # Bitshuffle - use different kernel based on element size
        n_full_8kb = frame_bytes // BLOCK_SIZE
        tail_bytes = frame_bytes % BLOCK_SIZE
        elem_size = np.dtype(dtype).itemsize

        if elem_size == 2:
            # uint16: use optimized shared memory kernel
            for start in range(0, n_frames, max_batch):
                end = min(start + max_batch, n_frames)
                batch_n = end - start
                byte_offset = start * frame_bytes
                if n_full_8kb:
                    _bitshuffle_kernel_u16(
                        (n_full_8kb, 1, batch_n),
                        (256, 1, 1),
                        (
                            self._lz4_output[byte_offset:],
                            self._shuffled_output[byte_offset:].view(cp.uint16),
                            np.uint32(frame_bytes),
                        ),
                    )
                if tail_bytes:
                    tail_elems = tail_bytes // elem_size
                    if tail_bytes % elem_size or tail_elems % 8:
                        raise ValueError(
                            "GPU bitshuffle/LZ4 load supports partial final "
                            "blocks only when the partial detector frame "
                            f"contains a multiple of 8 elements; got {frame_shape}."
                        )
                    _bitshuffle_tail_kernel_u16(
                        ((tail_elems + 255) // 256, 1, batch_n),
                        (256, 1, 1),
                        (
                            self._lz4_output[byte_offset:],
                            self._shuffled_output[byte_offset:].view(cp.uint16),
                            np.uint32(frame_bytes),
                        ),
                    )
        else:
            # uint32: use optimized ballot-based kernel
            frame_u32s = frame_bytes // 4
            for start in range(0, n_frames, max_batch):
                end = min(start + max_batch, n_frames)
                batch_n = end - start
                byte_offset = start * frame_bytes
                if n_full_8kb:
                    self._shuf(
                        (n_full_8kb, 2, batch_n),
                        (32, 32, 1),
                        (
                            self._lz4_output[byte_offset:].view(cp.uint32),
                            self._shuffled_output[byte_offset:].view(cp.uint32),
                            np.uint32(frame_u32s),
                        ),
                    )
                if tail_bytes:
                    tail_elems = tail_bytes // elem_size
                    if tail_bytes % elem_size or tail_elems % 8:
                        raise ValueError(
                            "GPU bitshuffle/LZ4 load supports partial final "
                            "blocks only when the partial detector frame "
                            f"contains a multiple of 8 elements; got {frame_shape}."
                        )
                    _bitshuffle_tail_kernel_u32(
                        ((tail_elems + 255) // 256, 1, batch_n),
                        (256, 1, 1),
                        (
                            self._lz4_output[byte_offset:],
                            self._shuffled_output[byte_offset:].view(cp.uint32),
                            np.uint32(frame_bytes),
                        ),
                    )
        cp.cuda.Device().synchronize()
        total_bytes = n_frames * frame_bytes
        # Return an independent copy - the view into _shuffled_output would
        # keep the entire oversized pre-allocated buffer alive, preventing
        # the caller from releasing the raw block via `del data`.
        return self._shuffled_output[:total_bytes].view(dtype).reshape(
            (n_frames,) + frame_shape
        ).copy()


@njit(cache=True, parallel=True)
def _parse_headers(
    pinned_buffer,
    chunk_sizes,
    chunk_offsets,
    block_starts_out,
    block_counts_out,
    n_frames,
    n_blocks_per_frame,
):
    """Parse bitshuffle+LZ4 chunk headers in parallel."""
    for i in prange(n_frames):
        offset = chunk_offsets[i]
        chunk = pinned_buffer[offset : offset + chunk_sizes[i]]

        # Parse header (first 12 bytes)
        uncomp_size = (
            int(chunk[0]) << 56
            | int(chunk[1]) << 48
            | int(chunk[2]) << 40
            | int(chunk[3]) << 32
            | int(chunk[4]) << 24
            | int(chunk[5]) << 16
            | int(chunk[6]) << 8
            | int(chunk[7])
        )
        block_size = (
            int(chunk[8]) << 24
            | int(chunk[9]) << 16
            | int(chunk[10]) << 8
            | int(chunk[11])
        )
        n_blocks = (uncomp_size + block_size - 1) // block_size
        block_counts_out[i] = n_blocks
        pos = 12
        base_idx = i * n_blocks_per_frame
        for b in range(n_blocks):
            block_starts_out[base_idx + b] = pos
            comp_size = (
                int(chunk[pos]) << 24
                | int(chunk[pos + 1]) << 16
                | int(chunk[pos + 2]) << 8
                | int(chunk[pos + 3])
            )
            pos += 4 + comp_size


_parse_headers_bulk = _parse_headers  # Same function, works with uint64 offsets

# Lazy-initialized decompressor (not at import time to save GPU memory)
_default_decompressor = None


_LIBC = None
_POSIX_FADV_SEQUENTIAL = 2
_POSIX_FADV_WILLNEED = 3


def _get_libc():
    """Lazy-load libc for posix_fadvise. None on non-Linux platforms."""
    global _LIBC
    if _LIBC is None:
        import ctypes
        import ctypes.util
        lib_name = ctypes.util.find_library("c")
        if lib_name is None:
            _LIBC = False
        else:
            try:
                libc = ctypes.CDLL(lib_name, use_errno=True)
            except OSError:
                _LIBC = False
            else:
                if not hasattr(libc, "posix_fadvise"):
                    _LIBC = False
                else:
                    _LIBC = libc
    return _LIBC if _LIBC is not False else None


# Persistent pinned (page-locked) host memory pool. The compressed read_buffer
# is allocated from it so the subsequent H2D upload can run asynchronously and
# the page-lock cost is amortized. Freed blocks are reused across loads, unlike
# a fresh cp.cuda.alloc_pinned_memory per call (which page-locks from scratch).
# Guarded: the pinned-memory pool is a CUDA resource. On a non-CUDA box `cp` is
# None and there is no pinned pool to set up; the cuda decompress path (the only
# user) never runs there.
if cp is not None:
    _PINNED_POOL = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(_PINNED_POOL.malloc)
else:
    _PINNED_POOL = None

# Fast pinned host buffers for the compressed read_buffer. An anonymous mmap
# provides a page-aligned, lazily faulted region that cudaHostRegister can pin
# without cudaHostAlloc's allocator-side initialization. Registered buffers are
# kept for the process and reused from a free list, so compatible later batches
# and loads reuse the page lock.
# Redundant nearby sizes are unregistered on release so ascending file sizes do
# not leave one multi-gigabyte mapping behind for every master.
_PINNED_BUFS: list[dict] = []
_PINNED_BUFS_LOCK = threading.Lock()
_PINNED_LARGE_BUFFER_THRESHOLD = 64 * 1024 * 1024
_PINNED_LARGE_BUFFER_GRANULARITY = 4 * 1024 * 1024


def _pinned_registration_size(nbytes: int) -> int:
    """Return a bounded-capacity registration size for a requested buffer.

    Large sequential HDF5 batches from one source can differ by a fraction of
    a percent in compressed size. Registering their exact byte counts can make
    a later, slightly larger batch pay for a third page-locked buffer even
    though the pipeline has only two staging slots. Round large registrations
    to 4 MiB so nearby batches reuse those two slots. The extra pinned memory is
    bounded below 4 MiB per slot; small sparse selections keep exact sizing.
    """

    nbytes = int(nbytes)
    if nbytes <= 0:
        raise ValueError("Pinned buffer size must be positive")
    if nbytes < _PINNED_LARGE_BUFFER_THRESHOLD:
        return nbytes
    granularity = _PINNED_LARGE_BUFFER_GRANULARITY
    return ((nbytes + granularity - 1) // granularity) * granularity


def _alloc_pinned_fast(nbytes: int) -> np.ndarray:
    """Return a page-locked uint8 host buffer of length >= nbytes, view[:nbytes].

    Reuses a registered buffer from the free list when one fits (size within
    1.5x, so a 1024-scan buffer is not wasted on a 512-scan load); otherwise
    mmaps a page-aligned anonymous region and cudaHostRegisters it once. The
    page lock permits asynchronous downstream H2D transfer. Reusing a
    compatible registered region amortizes that one-time registration cost.
    """
    with _PINNED_BUFS_LOCK:
        for entry in _PINNED_BUFS:
            if entry["free"] and nbytes <= entry["size"] <= int(nbytes * 1.5):
                entry["free"] = False
                return entry["arr"][:nbytes]
    import ctypes
    import mmap
    try:
        from cuda.bindings import runtime as cudart
    except ModuleNotFoundError:
        return np.empty(nbytes, dtype=np.uint8)
    registration_size = _pinned_registration_size(nbytes)
    region = mmap.mmap(-1, registration_size)  # anonymous → page-aligned base
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    err = cudart.cudaHostRegister(addr, registration_size, 0)
    if int(err[0]) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {int(err[0])}")
    arr = np.frombuffer(region, dtype=np.uint8)
    with _PINNED_BUFS_LOCK:
        _PINNED_BUFS.append(
            {
                "region": region,
                "addr": addr,
                "arr": arr,
                "size": registration_size,
                "free": False,
            }
        )
    return arr[:nbytes]


def _release_pinned(view: np.ndarray, *, prune: bool = True) -> None:
    """Mark a buffer from :func:`_alloc_pinned_fast` reusable.

    Keeps one reusable buffer per non-overlapping size class. A larger free
    buffer supersedes a smaller one when it is no more than 1.5x larger, which
    is the same fit rule used by :func:`_alloc_pinned_fast`. ``view`` is the
    sliced array; its ``.base`` is the full registered array we cached.

    Group-pipelined loads pass ``prune=False`` while disk preparation and GPU
    decode overlap. They prune once after the pipeline drains, so the next
    group can reuse every in-flight staging slot instead of repeatedly
    unregistering and registering a nearby-sized buffer.
    """
    base = view.base if view.base is not None else view
    with _PINNED_BUFS_LOCK:
        released = None
        for entry in _PINNED_BUFS:
            if entry["arr"] is base:
                entry["free"] = True
                released = entry
                break
        if released is None:
            return

        if not prune:
            return

        _prune_pinned_free_locked()


def _prune_pinned_free(*, retain_per_size_class: int = 1) -> None:
    """Discard redundant idle staging buffers after a pipeline drains."""
    with _PINNED_BUFS_LOCK:
        _prune_pinned_free_locked(retain_per_size_class=retain_per_size_class)


def _prune_pinned_free_locked(*, retain_per_size_class: int = 1) -> None:
    """Prune redundant free buffers while ``_PINNED_BUFS_LOCK`` is held."""
    retain_per_size_class = max(1, int(retain_per_size_class))
    retained: list[dict] = []
    redundant: list[dict] = []
    for entry in sorted(
        (item for item in _PINNED_BUFS if item["free"]),
        key=lambda item: item["size"],
        reverse=True,
    ):
        covering = sum(
            keeper["size"] <= int(entry["size"] * 1.5)
            for keeper in retained
        )
        if covering >= retain_per_size_class:
            redundant.append(entry)
        else:
            retained.append(entry)
    for entry in redundant:
        if _unregister_pinned_entry(entry):
            _PINNED_BUFS.remove(entry)
            entry.clear()


def _unregister_pinned_entry(entry: dict) -> bool:
    """Unregister one idle mmap-backed staging buffer."""
    try:
        from cuda.bindings import runtime as cudart
    except ModuleNotFoundError:
        return False
    error = cudart.cudaHostUnregister(entry["addr"])
    return int(error[0]) == 0


def _prepare_master(
    filepath: str,
    chunk_names: list[str],
    apply_mask: bool = True,
) -> dict:
    """CPU-only phase: read compressed bytes from disk + parse headers.

    Returns a dict with everything needed for GPU decompression. This runs
    entirely on CPU threads - no GPU memory or kernels used. Call
    _decompress_prepared() to finish on GPU.

    Each read worker issues posix_fadvise(SEQUENTIAL|WILLNEED) on its
    fd before pulling bytes, so the kernel kicks off readahead before
    the first os.readv lands. On Linux/NVMe Gen4 this trims ~300 ms
    off the cold disk_read phase on a 12 GB / 105-file Arina master
    and ~300 ms off warm (page cache served at higher effective BW
    once SEQUENTIAL doubles the readahead window).

    Read and chunk_iter still share one pool: a 12 GB read is fast
    enough that h5py.File reopen + chunk_iter on the SAME thread (~30
    ms per file) hides behind the per-file disk wait without needing
    a second pool.

    Typical time on 1024 scan Arina master (12 GB / 105 chunk files):
    ~1.1 s warm / ~2.2 s cold on NVMe Gen4 (NVMe-bound).
    """
    import ctypes
    import os
    from concurrent.futures import ThreadPoolExecutor

    master_dir = os.path.dirname(os.path.abspath(filepath))
    with h5py.File(filepath, "r") as f:
        data_group = f["entry/data"]
        data_paths = []
        for chunk_name in chunk_names:
            link = data_group.get(chunk_name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                data_paths.append(os.path.join(master_dir, link.filename))
            else:
                data_paths.append(data_group[chunk_name].file.filename)
        pixel_mask = None
        if apply_mask:
            mask_path = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if mask_path in f:
                pixel_mask = f[mask_path][:]

    file_sizes = [os.path.getsize(p) for p in data_paths]
    total_compressed_est = sum(file_sizes)
    # Page-locked host buffer (parallel cudaHostRegister, reused from free list)
    # → full-PCIe H2D downstream without the ~2 s serial cudaHostAlloc lock.
    # Buffered readv (not O_DIRECT) on purpose: it populates the page cache so
    # the second load of the same data is served from RAM (warm 4.6 s vs 13 s
    # cold). O_DIRECT was tried 2026-05-24 — it bypasses the cache (warm
    # collapsed to 10 s) and in the per-master pipeline it did not reach the
    # concurrency that made it fast in a synthetic all-files bench; net loss.
    read_buffer = _alloc_pinned_fast(total_compressed_est)
    try:
        file_offsets = np.cumsum([0] + file_sizes[:-1]).tolist()

        libc = _get_libc()

        def read_and_index(args):
            data_path, buf_offset, file_size = args
            fd = os.open(data_path, os.O_RDONLY)
            try:
                if libc is not None:
                    # SEQUENTIAL doubles the kernel readahead window;
                    # WILLNEED kicks off immediate prefetch.
                    libc.posix_fadvise(
                        fd,
                        ctypes.c_long(0),
                        ctypes.c_long(0),
                        _POSIX_FADV_SEQUENTIAL,
                    )
                    libc.posix_fadvise(
                        fd,
                        ctypes.c_long(0),
                        ctypes.c_long(0),
                        _POSIX_FADV_WILLNEED,
                    )
                mv = memoryview(read_buffer)[buf_offset : buf_offset + file_size]
                remaining = file_size
                view_off = 0
                while remaining > 0:
                    got = os.readv(fd, [mv[view_off : view_off + remaining]])
                    if got == 0:
                        break
                    view_off += got
                    remaining -= got
            finally:
                os.close(fd)
            with h5py.File(data_path, "r") as df:
                ds = df["entry/data/data"]
                n_frames = ds.shape[0]
                frame_shape = ds.shape[1:]
                dtype = ds.dtype
                chunk_infos = []
                ds.id.chunk_iter(
                    lambda info: chunk_infos.append((info.byte_offset, info.size))
                )
            return {
                "n_frames": n_frames,
                "frame_shape": frame_shape,
                "dtype": dtype,
                "chunk_infos": chunk_infos,
            }

        # 12 threads saturates the NVMe random-file bandwidth on Arina-style
        # masters; past 16 hurts on host queue depth, 8 underfeeds it.
        with ThreadPoolExecutor(max_workers=12) as pool:
            file_infos = list(
                pool.map(
                    read_and_index,
                    zip(data_paths, file_offsets, file_sizes),
                )
            )

        total_frames = sum(fi["n_frames"] for fi in file_infos)
        frame_shape = file_infos[0]["frame_shape"]
        dtype = file_infos[0]["dtype"]
        frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
        n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

        # uint64 chunk_offsets: the absolute byte offset of each frame's
        # compressed chunk inside `read_buffer` can exceed 4 GB on dense
        # multi-file scans, so uint32 silently wraps and the GPU kernel
        # reads garbage. (chunk_sizes stays uint32 - per-frame compressed
        # payload is only ~23 KB.)
        chunk_offsets_arr = np.empty(total_frames, dtype=np.uint64)
        chunk_sizes_arr = np.empty(total_frames, dtype=np.uint32)
        frame_idx = 0
        for fi, buf_offset in zip(file_infos, file_offsets):
            for byte_offset, size in fi["chunk_infos"]:
                chunk_offsets_arr[frame_idx] = buf_offset + byte_offset
                chunk_sizes_arr[frame_idx] = size
                frame_idx += 1

        block_starts_flat = np.zeros(
            total_frames * n_blocks_per_frame,
            dtype=np.uint32,
        )
        block_counts = np.zeros(total_frames, dtype=np.uint32)
        block_offsets_arr = np.zeros(total_frames + 1, dtype=np.uint32)
        _parse_headers_bulk(
            read_buffer,
            chunk_sizes_arr,
            chunk_offsets_arr,
            block_starts_flat,
            block_counts,
            total_frames,
            n_blocks_per_frame,
        )
        block_offsets_arr[1 : total_frames + 1] = np.cumsum(
            block_counts[:total_frames]
        )
        total_blocks = int(block_offsets_arr[total_frames])
        total_used = int(max(chunk_offsets_arr + chunk_sizes_arr))

        return {
            "read_buffer": read_buffer[:total_used],
            "chunk_offsets": chunk_offsets_arr,
            "block_starts": block_starts_flat[:total_blocks],
            "block_counts": block_counts,
            "block_offsets": block_offsets_arr,
            "total_frames": total_frames,
            "frame_shape": frame_shape,
            "frame_bytes": frame_bytes,
            "dtype": dtype,
            "pixel_mask": pixel_mask,
            "n_chunk_files": len(chunk_names),
        }
    except BaseException:
        _release_pinned(read_buffer)
        raise


def _file_stat_signature(path: str) -> dict[str, Any]:
    """Return the small file signature needed to validate an IO metadata cache."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _frame_source_cache_path(
    filepath: str,
    chunk_names: list[str],
    *,
    apply_mask: bool,
) -> str | None:
    """Return the optional private disk-cache path for source chunk metadata."""
    cache_dir = os.environ.get(_FRAME_SOURCE_CACHE_ENV)
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    payload = repr(
        (
            os.path.abspath(filepath),
            tuple(str(name) for name in chunk_names),
            bool(apply_mask),
        )
    ).encode("utf-8")
    key = hashlib.sha256(payload).hexdigest()
    return os.path.join(cache_dir, f"{key}.pkl")


def _master_frame_source_refs(
    filepath: str,
    chunk_names: list[str],
    *,
    apply_mask: bool,
) -> tuple[list[tuple[str, str]], Any, dict[str, Any]]:
    """Read master links and mask without iterating detector-frame chunks."""
    master_dir = os.path.dirname(os.path.abspath(filepath))
    data_refs: list[tuple[str, str]] = []
    pixel_mask = None
    with h5py.File(filepath, "r") as f:
        data_group = f["entry/data"]
        for chunk_name in chunk_names:
            link = data_group.get(chunk_name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                data_path = os.path.join(master_dir, link.filename)
                dataset_path = link.path or "entry/data/data"
            else:
                ds = data_group[chunk_name]
                data_path = ds.file.filename
                dataset_path = ds.name
            data_refs.append((os.path.abspath(data_path), str(dataset_path)))
        if apply_mask:
            mask_path = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if mask_path in f:
                pixel_mask = f[mask_path][:]
    signature = {
        "master": _file_stat_signature(filepath),
        "sources": [
            {
                **_file_stat_signature(data_path),
                "dataset_path": dataset_path,
            }
            for data_path, dataset_path in data_refs
        ],
        "chunk_names": [str(name) for name in chunk_names],
        "apply_mask": bool(apply_mask),
    }
    return data_refs, pixel_mask, signature


def _load_frame_source_disk_cache(
    cache_path: str,
    *,
    signature: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], Any] | None:
    """Load cached frame-source metadata when every file signature still matches."""
    try:
        with open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a corrupt optional cache must not block loading
        return None
    if signature is not None and cached.get("signature") != signature:
        return None
    source_infos = []
    for info in cached.get("source_infos", []):
        item = dict(info)
        item["dtype"] = np.dtype(item["dtype"])
        item["frame_shape"] = tuple(int(v) for v in item["frame_shape"])
        chunk_infos = np.asarray(item["chunk_infos"], dtype=np.uint64)
        if chunk_infos.ndim != 2 or chunk_infos.shape[1:] != (2,):
            return None
        item["chunk_infos"] = chunk_infos
        source_infos.append(item)
    if not source_infos:
        return None
    return source_infos, cached.get("pixel_mask")


def _write_frame_source_disk_cache(
    cache_path: str,
    *,
    signature: dict[str, Any],
    source_infos: list[dict[str, Any]],
    pixel_mask: Any,
) -> None:
    """Write cached frame-source metadata atomically for future worker processes."""
    serial_infos = []
    for info in source_infos:
        item = dict(info)
        item["dtype"] = np.dtype(item["dtype"]).str
        item["frame_shape"] = [int(v) for v in item["frame_shape"]]
        item["chunk_infos"] = np.asarray(item["chunk_infos"], dtype=np.uint64)
        serial_infos.append(item)
    payload = {
        "signature": signature,
        "source_infos": serial_infos,
        "pixel_mask": pixel_mask,
    }
    cache_dir = os.path.dirname(cache_path)
    fd, tmp_path = tempfile.mkstemp(prefix=".frame_source_", suffix=".tmp", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    except Exception:  # noqa: BLE001 - cache persistence is an optional optimization
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _scan_crop_max_gap_bytes() -> int:
    """Return the compressed-span merge gap for private scan-crop profiling."""
    raw = os.environ.get(_SCAN_CROP_MAX_GAP_ENV)
    if raw is None or str(raw).strip() == "":
        return 4096
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_SCAN_CROP_MAX_GAP_ENV} must be a non-negative integer byte count"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{_SCAN_CROP_MAX_GAP_ENV} must be a non-negative integer byte count"
        )
    return value


def _prepared_scan_crop_cache_dir(
    filepath: str,
    scan_region: tuple[int, int, int, int],
    *,
    scan_shape: tuple[int, int],
    scan_order: str,
    apply_mask: bool,
) -> str | None:
    """Return the optional private cache directory for one prepared scan crop."""
    cache_root = os.environ.get(_PREPARED_SCAN_CROP_CACHE_ENV)
    if not cache_root:
        return None
    os.makedirs(cache_root, exist_ok=True)
    payload = repr(
        (
            _PREPARED_SCAN_CROP_CACHE_VERSION,
            os.path.abspath(filepath),
            tuple(int(v) for v in scan_region),
            tuple(int(v) for v in scan_shape),
            str(scan_order),
            bool(apply_mask),
        )
    ).encode("utf-8")
    key = hashlib.sha256(payload).hexdigest()
    return os.path.join(cache_root, key)


def _load_prepared_scan_crop_disk_cache(
    cache_dir: str,
    *,
    signature: dict[str, Any],
) -> dict | None:
    """Load a prepared compressed scan crop from RAM/disk cache."""
    meta_path = os.path.join(cache_dir, "meta.pkl")
    read_path = os.path.join(cache_dir, "read_buffer.u8")
    try:
        with open(meta_path, "rb") as handle:
            cached = pickle.load(handle)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a corrupt optional cache must not block loading
        return None
    if cached.get("version") != _PREPARED_SCAN_CROP_CACHE_VERSION:
        return None
    if cached.get("signature") != signature:
        return None
    prepared = dict(cached.get("prepared") or {})
    read_size = int(prepared.get("read_buffer_nbytes", 0))
    if read_size <= 0:
        return None
    try:
        read_buffer = np.memmap(read_path, mode="r", dtype=np.uint8, shape=(read_size,))
    except OSError:
        return None
    prepared["read_buffer"] = read_buffer
    prepared["chunk_offsets"] = np.asarray(prepared["chunk_offsets"], dtype=np.uint64)
    prepared["block_starts"] = np.asarray(prepared["block_starts"], dtype=np.uint32)
    prepared["block_counts"] = np.asarray(prepared["block_counts"], dtype=np.uint32)
    prepared["block_offsets"] = np.asarray(prepared["block_offsets"], dtype=np.uint32)
    prepared["selected_frame_indices"] = np.asarray(
        prepared["selected_frame_indices"],
        dtype=np.int64,
    )
    prepared["frame_shape"] = tuple(int(v) for v in prepared["frame_shape"])
    prepared["dtype"] = np.dtype(prepared["dtype"])
    prepared["pixel_mask"] = cached.get("pixel_mask")
    prepared["read_gap_bytes"] = int(prepared.get("read_gap_bytes", 0))
    source_prepare_timing_s = dict(prepared.get("prepare_timing_s") or {})
    prepared["prepare_timing_s"] = {}
    prepared["prepared_cache"] = {
        "status": "hit",
        "cache_dir": cache_dir,
        "read_buffer_nbytes": int(read_size),
        "source_prepare_timing_s": source_prepare_timing_s,
    }
    return prepared


def _write_prepared_scan_crop_disk_cache(
    cache_dir: str,
    *,
    signature: dict[str, Any],
    prepared: dict,
) -> None:
    """Write a prepared compressed scan crop for reuse by fresh workers."""
    os.makedirs(cache_dir, exist_ok=True)
    read_buffer = np.asarray(prepared["read_buffer"], dtype=np.uint8)
    read_size = int(read_buffer.size)
    read_path = os.path.join(cache_dir, "read_buffer.u8")
    tmp_read_path = os.path.join(cache_dir, f"read_buffer.u8.{os.getpid()}.tmp")
    tmp_meta_path = os.path.join(cache_dir, f"meta.pkl.{os.getpid()}.tmp")
    meta_path = os.path.join(cache_dir, "meta.pkl")
    serial_prepared = {
        "read_buffer_nbytes": read_size,
        "chunk_offsets": np.asarray(prepared["chunk_offsets"], dtype=np.uint64),
        "block_starts": np.asarray(prepared["block_starts"], dtype=np.uint32),
        "block_counts": np.asarray(prepared["block_counts"], dtype=np.uint32),
        "block_offsets": np.asarray(prepared["block_offsets"], dtype=np.uint32),
        "total_frames": int(prepared["total_frames"]),
        "frame_shape": [int(v) for v in prepared["frame_shape"]],
        "frame_bytes": int(prepared["frame_bytes"]),
        "dtype": np.dtype(prepared["dtype"]).str,
        "n_chunk_files": int(prepared.get("n_chunk_files", 0)),
        "selected_frame_indices": np.asarray(
            prepared.get("selected_frame_indices", []),
            dtype=np.int64,
        ),
        "total_compressed_bytes": int(prepared.get("total_compressed_bytes", read_size)),
        "read_span_count": int(prepared.get("read_span_count", 0)),
        "read_gap_bytes": int(prepared.get("read_gap_bytes", 0)),
        "prepare_timing_s": dict(prepared.get("prepare_timing_s") or {}),
    }
    payload = {
        "version": _PREPARED_SCAN_CROP_CACHE_VERSION,
        "signature": signature,
        "prepared": serial_prepared,
        "pixel_mask": prepared.get("pixel_mask"),
    }
    try:
        read_buffer.tofile(tmp_read_path)
        with open(tmp_meta_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_read_path, read_path)
        os.replace(tmp_meta_path, meta_path)
    except Exception:  # noqa: BLE001 - cache persistence is an optional optimization
        for path in (tmp_read_path, tmp_meta_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _get_master_frame_sources(
    filepath: str,
    chunk_names: list[str],
    *,
    apply_mask: bool,
) -> tuple[list[dict[str, Any]], Any]:
    """Return cached detector-frame chunk offsets for one master."""
    abs_path = os.path.abspath(filepath)
    key = (abs_path, tuple(chunk_names), bool(apply_mask))
    cached = _MASTER_FRAME_SOURCE_CACHE.get(key)
    if cached is not None:
        return cached

    disk_cache_path = _frame_source_cache_path(
        filepath,
        chunk_names,
        apply_mask=apply_mask,
    )
    if (
        disk_cache_path is not None
        and os.environ.get(_TRUST_FRAME_SOURCE_CACHE_ENV, "").lower()
        in {"1", "true", "yes", "on"}
    ):
        cached = _load_frame_source_disk_cache(
            disk_cache_path,
            signature=None,
        )
        if cached is not None:
            _MASTER_FRAME_SOURCE_CACHE[key] = cached
            return cached

    data_refs, pixel_mask, signature = _master_frame_source_refs(
        filepath,
        chunk_names,
        apply_mask=apply_mask,
    )
    if disk_cache_path is not None:
        cached = _load_frame_source_disk_cache(
            disk_cache_path,
            signature=signature,
        )
        if cached is not None:
            _MASTER_FRAME_SOURCE_CACHE[key] = cached
            return cached

    source_infos: list[dict[str, Any]] = []
    for data_path, dataset_path in data_refs:
        with h5py.File(data_path, "r") as df:
            ds = df[dataset_path]
            if ds.ndim != 3:
                raise ValueError(
                    "load(..., scan_region=...) currently supports flattened "
                    f"3D detector chunks; got {ds.shape} in {data_path}"
                )
            if ds.chunks is None or int(ds.chunks[0]) != 1:
                raise ValueError(
                    "load(..., scan_region=...) requires one detector frame per "
                    f"HDF5 chunk; got chunks={ds.chunks} in {data_path}"
                )
            chunk_infos = []

            def collect_chunk(info, output=chunk_infos):
                output.append((info.byte_offset, info.size))

            ds.id.chunk_iter(collect_chunk)
            source_infos.append(
                {
                    "path": data_path,
                    "dataset_path": dataset_path,
                    "n_frames": int(ds.shape[0]),
                    "frame_shape": tuple(int(v) for v in ds.shape[1:]),
                    "dtype": ds.dtype,
                    "chunk_infos": np.asarray(chunk_infos, dtype=np.uint64),
                }
            )
    cached = (source_infos, pixel_mask)
    _MASTER_FRAME_SOURCE_CACHE[key] = cached
    if disk_cache_path is not None:
        _write_frame_source_disk_cache(
            disk_cache_path,
            signature=signature,
            source_infos=source_infos,
            pixel_mask=pixel_mask,
        )
    return cached


class _SparseFrameReadSession:
    """Own reusable host resources for selected reads from one HDF5 master.

    The session is intentionally private.  Public callers continue to use
    :func:`load`; streamed screening uses this object only to avoid reopening
    immutable source shards and recreating a thread pool for every batch.
    """

    def __init__(
        self,
        filepath: str,
        chunk_names: list[str],
        *,
        apply_mask: bool,
    ) -> None:
        import time
        from concurrent.futures import ThreadPoolExecutor

        self.filepath = os.path.abspath(filepath)
        self.chunk_names = tuple(chunk_names)
        self.apply_mask = bool(apply_mask)
        self.closed = False

        started = time.perf_counter()
        metadata_started = time.perf_counter()
        self.meta = get_metadata(filepath)
        metadata_seconds = time.perf_counter() - metadata_started
        source_started = time.perf_counter()
        self.source_infos, self.pixel_mask = _get_master_frame_sources(
            filepath,
            chunk_names,
            apply_mask=apply_mask,
        )
        source_seconds = time.perf_counter() - source_started
        if not self.source_infos:
            raise ValueError(f"No detector data chunks found in {filepath}")
        chunk_index_started = time.perf_counter()
        self.chunk_index_arrays = tuple(
            np.asarray(info["chunk_infos"], dtype=np.uint64)
            for info in self.source_infos
        )
        chunk_index_seconds = time.perf_counter() - chunk_index_started
        self.source_starts = np.cumsum(
            [0] + [info["n_frames"] for info in self.source_infos],
            dtype=np.int64,
        )

        file_open_started = time.perf_counter()
        self.fds: dict[int, int] = {}
        try:
            for source_index, info in enumerate(self.source_infos):
                self.fds[source_index] = os.open(info["path"], os.O_RDONLY)
        except Exception:
            for descriptor in self.fds.values():
                os.close(descriptor)
            raise
        file_open_seconds = time.perf_counter() - file_open_started

        pool_started = time.perf_counter()
        self.thread_pool = (
            ThreadPoolExecutor(max_workers=min(12, len(self.source_infos)))
            if len(self.source_infos) > 1
            else None
        )
        pool_seconds = time.perf_counter() - pool_started
        self._initial_timing = {
            "metadata_discovery": float(metadata_seconds),
            "source_index": float(source_seconds),
            "chunk_index_array_build": float(chunk_index_seconds),
            "persistent_file_open": float(file_open_seconds),
            "persistent_thread_pool_create": float(pool_seconds),
            "persistent_session_total": float(time.perf_counter() - started),
        }

    def take_initial_timing(self) -> dict[str, float]:
        """Return one-time initialization costs on the first prepared batch."""

        timing = dict(self._initial_timing)
        self._initial_timing = {name: 0.0 for name in timing}
        return timing

    def source_aligned_ranges(
        self,
        *,
        frame_count: int,
        max_batch_frames: int,
    ) -> tuple[tuple[int, int], ...]:
        """Group complete source shards into bounded sequential batches."""

        return _source_aligned_frame_ranges(
            self.source_starts,
            frame_count=frame_count,
            max_batch_frames=max_batch_frames,
        )

    def close(self) -> None:
        """Close the persistent descriptors and preparation worker pool."""

        if self.closed:
            return
        self.closed = True
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
        for descriptor in self.fds.values():
            os.close(descriptor)
        self.fds.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _source_aligned_frame_ranges(
    source_starts,
    *,
    frame_count: int,
    max_batch_frames: int,
) -> tuple[tuple[int, int], ...]:
    """Return complete, ordered, bounded ranges aligned to source shards.

    Whole adjacent source shards are combined while they fit.  A source shard
    is split only when that shard alone exceeds ``max_batch_frames``.
    """

    starts = np.asarray(source_starts, dtype=np.int64).reshape(-1)
    frame_count = int(frame_count)
    max_batch_frames = int(max_batch_frames)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if max_batch_frames < 1:
        raise ValueError("max_batch_frames must be positive")
    if starts.size < 2 or int(starts[0]) != 0:
        raise ValueError("source_starts must begin at zero and contain a stop")
    if np.any(starts[1:] < starts[:-1]):
        raise ValueError("source_starts must be monotonically non-decreasing")
    if int(starts[-1]) < frame_count:
        raise ValueError(
            f"Source shards expose {int(starts[-1])} frames, "
            f"but {frame_count} are required"
        )

    ranges: list[tuple[int, int]] = []
    pending_start: int | None = None
    pending_stop: int | None = None
    for raw_start, raw_stop in pairwise(starts):
        source_start = min(int(raw_start), frame_count)
        source_stop = min(int(raw_stop), frame_count)
        if source_start >= source_stop:
            continue
        source_size = source_stop - source_start
        if source_size > max_batch_frames:
            if pending_start is not None and pending_stop is not None:
                ranges.append((pending_start, pending_stop))
                pending_start = None
                pending_stop = None
            for split_start in range(source_start, source_stop, max_batch_frames):
                ranges.append(
                    (split_start, min(split_start + max_batch_frames, source_stop))
                )
            continue
        if (
            pending_start is not None
            and pending_stop is not None
            and source_stop - pending_start > max_batch_frames
        ):
            ranges.append((pending_start, pending_stop))
            pending_start = None
            pending_stop = None
        if pending_start is None:
            pending_start = source_start
        pending_stop = source_stop

    if pending_start is not None and pending_stop is not None:
        ranges.append((pending_start, pending_stop))
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != frame_count:
        raise RuntimeError("Source-aligned range planning did not cover the scan")
    if any(stop - start > max_batch_frames for start, stop in ranges):
        raise RuntimeError("Source-aligned range exceeded max_batch_frames")
    return tuple(ranges)


def _contiguous_frame_read_plan(
    source_infos: list[dict[str, Any]],
    source_starts,
    selected,
    *,
    max_gap_bytes: int,
    chunk_index_arrays: Sequence[np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, list[tuple[int, int, int]]],
    int,
] | None:
    """Plan monotonic contiguous frame reads without per-frame Python work.

    The screening path requests source-aligned consecutive scan positions.  A
    vectorized planner is substantially cheaper for that common case, while
    arbitrary order, duplicates, and non-monotonic physical chunk layouts
    deliberately fall back to the general selector-preserving planner.
    """

    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        return None
    first = int(selected[0])
    if int(selected[-1]) - first + 1 != int(selected.size):
        return None
    if selected.size > 1 and not np.all(selected[1:] == selected[:-1] + 1):
        return None

    starts = np.asarray(source_starts, dtype=np.int64).reshape(-1)
    if starts.size != len(source_infos) + 1:
        return None
    if chunk_index_arrays is not None and len(chunk_index_arrays) != len(source_infos):
        return None

    chunk_offsets = np.empty(selected.size, dtype=np.uint64)
    chunk_sizes = np.empty(selected.size, dtype=np.uint32)
    read_plan: dict[int, list[tuple[int, int, int]]] = {}
    cursor = 0
    stop = first + int(selected.size)

    for source_index, info in enumerate(source_infos):
        source_start = int(starts[source_index])
        source_stop = int(starts[source_index + 1])
        selected_start = max(first, source_start)
        selected_stop = min(stop, source_stop)
        if selected_start >= selected_stop:
            continue

        local_start = selected_start - source_start
        local_stop = selected_stop - source_start
        all_chunks = (
            np.asarray(chunk_index_arrays[source_index], dtype=np.uint64)
            if chunk_index_arrays is not None
            else np.asarray(info["chunk_infos"], dtype=np.uint64)
        )
        if (
            all_chunks.ndim != 2
            or all_chunks.shape[1:] != (2,)
            or local_stop > int(all_chunks.shape[0])
        ):
            return None
        chunks = all_chunks[local_start:local_stop]
        offsets = chunks[:, 0]
        sizes_u64 = chunks[:, 1]
        if np.any(sizes_u64 > np.iinfo(np.uint32).max):
            raise ValueError("Compressed HDF5 chunk exceeds uint32 size range")
        if offsets.size > 1 and np.any(offsets[1:] < offsets[:-1]):
            return None

        ends = offsets + sizes_u64
        if np.any(ends < offsets):
            raise ValueError("Compressed HDF5 chunk byte range overflowed uint64")
        running_ends = np.maximum.accumulate(ends)
        split_points = np.flatnonzero(
            offsets[1:] > running_ends[:-1] + np.uint64(max_gap_bytes)
        ) + 1
        group_starts = np.concatenate((np.array([0]), split_points))
        group_stops = np.concatenate((split_points, np.array([offsets.size])))

        destination_start = cursor
        source_plan: list[tuple[int, int, int]] = []
        for group_start, group_stop in zip(group_starts, group_stops, strict=True):
            group_start = int(group_start)
            group_stop = int(group_stop)
            span_start = int(offsets[group_start])
            span_stop = int(running_ends[group_stop - 1])
            span_nbytes = span_stop - span_start
            source_plan.append((span_start, cursor, span_nbytes))
            cursor += span_nbytes
        read_plan[source_index] = source_plan

        order_start = selected_start - first
        order_stop = selected_stop - first
        group_destinations = np.empty(offsets.size, dtype=np.uint64)
        group_cursors = np.cumsum(
            np.asarray([0] + [item[2] for item in source_plan[:-1]], dtype=np.uint64)
        ) + np.uint64(destination_start)
        for group_number, (group_start, group_stop) in enumerate(
            zip(group_starts, group_stops, strict=True)
        ):
            group_start = int(group_start)
            group_stop = int(group_stop)
            group_destinations[group_start:group_stop] = (
                group_cursors[group_number]
                + offsets[group_start:group_stop]
                - offsets[group_start]
            )
        chunk_offsets[order_start:order_stop] = group_destinations
        chunk_sizes[order_start:order_stop] = sizes_u64.astype(np.uint32, copy=False)

    if sum(len(plan) for plan in read_plan.values()) == 0:
        return None
    return chunk_offsets, chunk_sizes, read_plan, int(cursor)


def _prepare_master_frames(
    filepath: str,
    chunk_names: list[str],
    frame_indices: np.ndarray,
    apply_mask: bool = True,
    read_session: _SparseFrameReadSession | None = None,
) -> dict:
    """Read selected compressed detector frames and index them for GPU decode.

    This is the sparse counterpart to :func:`_prepare_master`: instead of
    bulk-reading every external ``data_######`` file, it pulls only the HDF5
    chunks for the requested flattened scan-frame indices. The returned dict is
    intentionally compatible with :func:`_decompress_prepared`.
    """
    import bisect
    import ctypes
    import time
    from concurrent.futures import ThreadPoolExecutor

    t_total = time.perf_counter()
    selected = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("frame_indices must contain at least one frame")
    if np.any(selected < 0):
        raise ValueError("frame_indices must be non-negative")

    t_sources = time.perf_counter()
    if read_session is None:
        source_infos, pixel_mask = _get_master_frame_sources(
            filepath,
            chunk_names,
            apply_mask=apply_mask,
        )
        source_seconds = time.perf_counter() - t_sources
        session_timing = {
            "metadata_discovery": 0.0,
            "persistent_file_open": 0.0,
            "persistent_thread_pool_create": 0.0,
            "persistent_session_total": 0.0,
        }
    else:
        if read_session.closed:
            raise RuntimeError("Sparse frame read session is already closed")
        if read_session.filepath != os.path.abspath(filepath):
            raise ValueError("Sparse frame read session belongs to another master")
        if read_session.chunk_names != tuple(chunk_names):
            raise ValueError("Sparse frame read session chunk names changed")
        if read_session.apply_mask != bool(apply_mask):
            raise ValueError("Sparse frame read session mask policy changed")
        source_infos = read_session.source_infos
        pixel_mask = read_session.pixel_mask
        session_timing = read_session.take_initial_timing()
        source_seconds = float(session_timing.pop("source_index"))

    if not source_infos:
        raise ValueError(f"No detector data chunks found in {filepath}")
    frame_shape = source_infos[0]["frame_shape"]
    dtype = source_infos[0]["dtype"]
    for info in source_infos[1:]:
        if info["frame_shape"] != frame_shape or np.dtype(info["dtype"]) != np.dtype(dtype):
            raise ValueError("Detector chunk files have inconsistent shape or dtype")

    source_starts = (
        read_session.source_starts
        if read_session is not None
        else np.cumsum([0] + [info["n_frames"] for info in source_infos])
    )
    total_available = int(source_starts[-1])
    if int(selected.max()) >= total_available:
        raise ValueError(
            f"Requested frame {int(selected.max())}, but only {total_available} frames are available"
        )

    t_plan = time.perf_counter()
    max_gap_bytes = _scan_crop_max_gap_bytes()
    fast_plan = _contiguous_frame_read_plan(
        source_infos,
        source_starts,
        selected,
        max_gap_bytes=max_gap_bytes,
        chunk_index_arrays=(
            read_session.chunk_index_arrays if read_session is not None else None
        ),
    )
    if fast_plan is not None:
        (
            chunk_offsets_arr,
            chunk_sizes_arr,
            read_plan_by_source,
            cursor,
        ) = fast_plan
    else:
        chunk_offsets_arr = np.empty(selected.size, dtype=np.uint64)
        chunk_sizes_arr = np.empty(selected.size, dtype=np.uint32)
        entries_by_source: dict[int, list[tuple[int, int, int]]] = {}
        for order_pos, global_idx in enumerate(selected):
            source_idx = bisect.bisect_right(source_starts, int(global_idx)) - 1
            local_idx = int(global_idx) - int(source_starts[source_idx])
            chunk_infos = source_infos[source_idx]["chunk_infos"]
            if local_idx >= len(chunk_infos):
                raise ValueError(
                    f"Requested local frame {local_idx}, but only "
                    f"{len(chunk_infos)} HDF5 chunks were indexed"
                )
            byte_offset, chunk_size = chunk_infos[local_idx]
            entries_by_source.setdefault(source_idx, []).append(
                (order_pos, int(byte_offset), int(chunk_size))
            )

        read_plan_by_source = {}
        cursor = 0

        def append_span(
            source_index: int,
            start: int,
            stop: int,
            span: list[tuple[int, int, int]],
            destination: int,
        ) -> int:
            span_nbytes = int(stop - start)
            read_plan_by_source.setdefault(source_index, []).append(
                (int(start), int(destination), span_nbytes)
            )
            for order_pos, byte_offset, chunk_size in span:
                chunk_offsets_arr[order_pos] = destination + int(byte_offset - start)
                chunk_sizes_arr[order_pos] = int(chunk_size)
            return destination + span_nbytes

        for source_idx, entries in entries_by_source.items():
            entries = sorted(entries, key=lambda item: item[1])
            span_start = entries[0][1]
            span_end = entries[0][1] + entries[0][2]
            span_entries = [entries[0]]

            for entry in entries[1:]:
                _, byte_offset, chunk_size = entry
                next_end = int(byte_offset + chunk_size)
                if int(byte_offset) <= span_end + max_gap_bytes:
                    span_end = max(span_end, next_end)
                    span_entries.append(entry)
                else:
                    cursor = append_span(
                        source_idx, span_start, span_end, span_entries, cursor
                    )
                    span_start = int(byte_offset)
                    span_end = next_end
                    span_entries = [entry]
            cursor = append_span(source_idx, span_start, span_end, span_entries, cursor)
    plan_seconds = time.perf_counter() - t_plan

    total_compressed = int(cursor)
    t_alloc = time.perf_counter()
    read_buffer = _alloc_pinned_fast(total_compressed)
    alloc_seconds = time.perf_counter() - t_alloc
    libc = _get_libc()

    def read_exact_at(fd: int, dst_offset: int, nbytes: int, file_offset: int) -> None:
        mv = memoryview(read_buffer)[dst_offset:dst_offset + nbytes]
        remaining = int(nbytes)
        view_offset = 0
        while remaining > 0:
            if hasattr(os, "preadv"):
                got = os.preadv(
                    fd,
                    [mv[view_offset:view_offset + remaining]],
                    int(file_offset + view_offset),
                )
            else:
                block = os.pread(fd, remaining, int(file_offset + view_offset))
                got = len(block)
                mv[view_offset:view_offset + got] = block
            if got == 0:
                raise OSError("short read while loading selected HDF5 chunks")
            remaining -= int(got)
            view_offset += int(got)

    def read_source(
        item: tuple[int, list[tuple[int, int, int]]],
    ) -> tuple[float, float, float]:
        source_idx, reads = item
        opened_here = read_session is None
        fd_started = time.perf_counter()
        fd = (
            os.open(source_infos[source_idx]["path"], os.O_RDONLY)
            if opened_here
            else read_session.fds[source_idx]
        )
        file_seconds = time.perf_counter() - fd_started
        advice_seconds = 0.0
        pread_seconds = 0.0
        try:
            if libc is not None:
                advice_started = time.perf_counter()
                for file_offset, _, nbytes in reads:
                    libc.posix_fadvise(
                        fd,
                        ctypes.c_long(file_offset),
                        ctypes.c_long(nbytes),
                        _POSIX_FADV_WILLNEED,
                    )
                advice_seconds = time.perf_counter() - advice_started
            pread_started = time.perf_counter()
            for file_offset, dst_offset, nbytes in reads:
                read_exact_at(fd, dst_offset, nbytes, file_offset)
            pread_seconds = time.perf_counter() - pread_started
        finally:
            if opened_here:
                close_started = time.perf_counter()
                os.close(fd)
                file_seconds += time.perf_counter() - close_started
        return file_seconds, advice_seconds, pread_seconds

    worker_count = min(12, max(1, len(read_plan_by_source)))
    t_read = time.perf_counter()
    pool_create_seconds = 0.0
    if worker_count > 1:
        if read_session is not None and read_session.thread_pool is not None:
            read_metrics = list(
                read_session.thread_pool.map(
                    read_source,
                    read_plan_by_source.items(),
                )
            )
        else:
            pool_started = time.perf_counter()
            pool = ThreadPoolExecutor(max_workers=worker_count)
            pool_create_seconds = time.perf_counter() - pool_started
            try:
                read_metrics = list(pool.map(read_source, read_plan_by_source.items()))
            finally:
                pool.shutdown(wait=True)
    else:
        read_metrics = [read_source(item) for item in read_plan_by_source.items()]
    read_seconds = time.perf_counter() - t_read
    file_open_close_seconds = sum(metric[0] for metric in read_metrics)
    fadvise_seconds = sum(metric[1] for metric in read_metrics)
    pread_seconds = sum(metric[2] for metric in read_metrics)

    frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
    n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_starts_flat = np.zeros(selected.size * n_blocks_per_frame, dtype=np.uint32)
    block_counts = np.zeros(selected.size, dtype=np.uint32)
    block_offsets_arr = np.zeros(selected.size + 1, dtype=np.uint32)
    t_headers = time.perf_counter()
    _parse_headers_bulk(
        read_buffer,
        chunk_sizes_arr,
        chunk_offsets_arr,
        block_starts_flat,
        block_counts,
        int(selected.size),
        n_blocks_per_frame,
    )
    block_offsets_arr[1:selected.size + 1] = np.cumsum(block_counts[:selected.size])
    total_blocks = int(block_offsets_arr[selected.size])
    header_seconds = time.perf_counter() - t_headers
    total_seconds = time.perf_counter() - t_total

    return {
        "read_buffer": read_buffer[:total_compressed],
        "chunk_offsets": chunk_offsets_arr,
        "block_starts": block_starts_flat[:total_blocks],
        "block_counts": block_counts,
        "block_offsets": block_offsets_arr,
        "total_frames": int(selected.size),
        "frame_shape": frame_shape,
        "frame_bytes": frame_bytes,
        "dtype": dtype,
        "pixel_mask": pixel_mask,
        "n_chunk_files": len(source_infos),
        "selected_frame_indices": selected,
        "total_compressed_bytes": int(total_compressed),
        "read_span_count": int(
            sum(len(reads) for reads in read_plan_by_source.values())
        ),
        "read_gap_bytes": int(max_gap_bytes),
        "prepare_timing_s": {
            "source_index": float(source_seconds),
            "read_plan": float(plan_seconds),
            "pinned_alloc": float(alloc_seconds),
            "compressed_read": float(read_seconds),
            "compressed_pread_cpu": float(pread_seconds),
            "posix_fadvise_cpu": float(fadvise_seconds),
            "file_open_close_cpu": float(file_open_close_seconds),
            "thread_pool_create": float(pool_create_seconds),
            "header_parse": float(header_seconds),
            "total": float(total_seconds),
            **session_timing,
        },
    }


def _default_decoded_output_dtype(
    source_dtype: type | np.dtype,
    *,
    auto_narrow: bool,
    detector_bin: int,
) -> tuple[np.dtype, bool]:
    """Return the lossless default dtype for one decoded detector frame."""
    if int(detector_bin) > 1:
        # A detector output pixel sums detector_bin**2 source values. Keeping
        # the source uint16 dtype can wrap even though every individual count
        # is valid, so exact binned loads widen before the first write.
        return np.dtype(np.uint32), False
    narrow_mode = bool(
        auto_narrow
        and np.dtype(source_dtype) == np.dtype(np.uint32)
    )
    return (
        np.dtype(np.uint16) if narrow_mode else np.dtype(source_dtype),
        narrow_mode,
    )


def _decompress_prepared_impl(
    prepared: dict,
    verbose: bool = False,
    auto_narrow: bool = True,
    batch_bytes_target: int = 1 << 28,  # 256 MB per scratch buffer
    det_bin: int = 1,
    streaming_bin: bool = False,
    output_dtype: type | np.dtype | None = None,
    streaming_upload: bool | None = None,
    prune_device_pool: bool = True,
) -> cp.ndarray:
    """GPU phase: transfer compressed bytes and decompress on GPU.

    Chunked implementation: processes frames in ~256 MB batches using two
    small per-batch scratch buffers (reused across iterations) and writes
    directly into a pre-allocated final result buffer. Peak transient
    VRAM is ``compressed + 2*batch + final`` instead of the old
    ``compressed + 2*full_uncompressed + final``, which halves or
    better the loader's peak memory on large scans.

    Optional ``auto_narrow`` (default True) casts uint32 detector data
    down to uint16 on the fly when every observed value fits. Arina
    writes uint32 when its auto-config predicts counts might exceed
    65535, but actual counts in 4D-STEM rarely exceed a few thousand,
    so ``auto_narrow`` halves the final buffer for free on the common
    case. If any batch has a real value >= 65536 it raises
    ``ValueError`` so the caller can retry with ``auto_narrow=False``.

    The pixel mask (Arina's dead-pixel map) is applied PER BATCH inside
    the loop rather than at the end, because dead pixels contain
    0xFFFFFFFF sentinels that would otherwise trip the narrow check.

    Parameters
    ----------
    prepared
        Output of :func:`_prepare_master`.
    verbose
        Print timing + final dtype decision.
    auto_narrow
        If True and source is uint32 and all values fit uint16, return
        a uint16 array. Default True - Arina's uint32 is almost always
        over-allocated in practice, so narrowing is a free memory win.
    batch_bytes_target
        Target size of each per-batch scratch buffer. Default 256 MiB,
        empirically the sweet spot on Blackwell (L2 ≈ 96 MB): speed is
        flat from 128 MB to 4 GB (within ±3%), 256 MB gives the best
        mix of low peak VRAM and low kernel-launch overhead. Smaller
        = lower peak, more launches. Larger = fewer launches, higher
        peak.
    """
    import time
    t0 = time.perf_counter()

    read_buffer = prepared["read_buffer"]
    chunk_offsets_arr = prepared["chunk_offsets"]
    block_starts_flat = prepared["block_starts"]
    block_counts = prepared["block_counts"]
    block_offsets_arr = prepared["block_offsets"]
    total_frames = prepared["total_frames"]
    frame_shape = prepared["frame_shape"]
    frame_bytes = prepared["frame_bytes"]
    source_dtype = prepared["dtype"]
    pixel_mask = prepared["pixel_mask"]

    source_itemsize = int(np.dtype(source_dtype).itemsize)

    # Auto-pick streaming vs full upload: streaming only helps when the
    # compressed file is big enough that holding it all on GPU costs more
    # than per-batch refill overhead saves. Measured 2026-05-14:
    #   * 12 GiB compressed (1024² Arina): streaming -0.11s, fits 24 GB ✓
    #   * 3 GiB compressed (512² Arina): streaming +0.02s (marginal slower)
    # Threshold 6 GiB picks streaming only when the file actually needs it.
    _streaming_auto = streaming_upload is None

    # --- Decide final dtype -------------------------------------------------
    uint4_mode = _is_uint4_load_dtype(output_dtype)
    if uint4_mode and int(det_bin) > 1:
        raise ValueError(
            "dtype='u4' means packed 4-bit detector counts (0..15) and cannot "
            "be combined with det_bin>1 because binned detector pixels can "
            "exceed 15. Use dtype='uint8' or dtype='uint16' for binned loads."
        )
    if output_dtype is not None:
        # Honor the project's browse vocabulary: "u8"/"uint8" mean 8-BIT unsigned
        # (the screening dtype the public load(dtype="u8") advertises). numpy's
        # np.dtype("u8") is uint64 (8 BYTES) — passing the browse token straight
        # to np.dtype would silently 8× the output and OOM. Map it explicitly.
        if uint4_mode or isinstance(output_dtype, str) and output_dtype in ("u8", "uint8"):
            final_dtype = np.dtype(np.uint8)
        else:
            final_dtype = np.dtype(output_dtype)
        narrow_mode = False
    else:
        final_dtype, narrow_mode = _default_decoded_output_dtype(
            source_dtype,
            auto_narrow=auto_narrow,
            detector_bin=det_bin if streaming_bin else 1,
        )

    # --- Pick batch size ----------------------------------------------------
    # Cap at max_batch=10000 to match the kernel's old launch characteristics
    # and stay well under any plausible single-batch scratch allocation.
    max_batch_from_target = max(1, int(batch_bytes_target // frame_bytes))
    max_batch = min(10000, max_batch_from_target, total_frames)

    if _streaming_auto:
        # Stream + async-overlap the H2D with the kernels whenever there is
        # more than one batch (a single-batch file cannot overlap). Async
        # upload of batch N+1 hides behind batch N's LZ4/bitshuffle work, so
        # the per-file H2D (~116 ms at 512²) no longer runs serially before
        # the kernels. Single-batch files fall back to one full upload. The
        # streaming slicer also requires chunk offsets to be monotonic in output
        # order; serpentine crops can request physical chunks in reverse order
        # within odd scan rows, so those safely use one full compressed upload.
        offsets_monotonic = bool(
            total_frames <= 1
            or np.all(chunk_offsets_arr[1:] >= chunk_offsets_arr[:-1])
        )
        streaming_upload = (
            offsets_monotonic
            and (total_frames + max_batch - 1) // max_batch > 1
        )
    elif streaming_upload:
        offsets_monotonic = bool(
            total_frames <= 1
            or np.all(chunk_offsets_arr[1:] >= chunk_offsets_arr[:-1])
        )
        if not offsets_monotonic:
            streaming_upload = False

    # --- Upload compressed + metadata to GPU -------------------------------
    # streaming_upload=True (default): per-batch compressed slice (~120 MB)
    # instead of full file (~12 GB for 1024² Arina). Bit-equivalent output,
    # peak VRAM drops by ~sizeof(read_buffer), wall time same-or-faster
    # because per-batch H2D overlaps with prior batch's kernel work.
    block_starts_gpu = cp.asarray(block_starts_flat)
    block_counts_gpu = cp.asarray(block_counts)
    block_offsets_gpu = cp.asarray(block_offsets_arr)
    if streaming_upload:
        # Precompute every batch's compressed slice + rebased chunk offsets
        # once. block_starts are chunk-relative so they need no rebasing.
        batch_slices = []
        all_rebased = np.empty(total_frames, dtype=np.uint64)
        max_batch_compressed = 0
        for _s in range(0, total_frames, max_batch):
            _e = min(_s + max_batch, total_frames)
            _bs = int(chunk_offsets_arr[_s])
            _be = len(read_buffer) if _e == total_frames else int(chunk_offsets_arr[_e])
            all_rebased[_s:_e] = chunk_offsets_arr[_s:_e] - np.uint64(_bs)
            max_batch_compressed = max(max_batch_compressed, _be - _bs)
            batch_slices.append((_s, _e, _bs, _be))
        all_rebased_gpu = cp.asarray(all_rebased)
        # Double-buffered async H2D: upload batch N+1 into the spare buffer on
        # a copy stream while batch N's kernels run on the main stream. The
        # read_buffer is page-locked so .set(stream=) is a true async DMA;
        # events keep the copy stream from overwriting a buffer whose LZ4
        # kernel has not finished reading it. This overlaps the ~116 ms/file
        # H2D with the ~95 ms/file kernels instead of running them serially.
        comp_bufs = [cp.empty(max_batch_compressed, dtype=cp.uint8) for _ in range(2)]
        copy_stream = cp.cuda.Stream(non_blocking=True)
        copy_done = [cp.cuda.Event() for _ in range(2)]
        kernel_done = [cp.cuda.Event() for _ in range(2)]
        main_stream = cp.cuda.get_current_stream()
        def _upload_batch(batch_idx, slot):
            _s, _e, _bs, _be = batch_slices[batch_idx]
            comp_bufs[slot][: _be - _bs].set(read_buffer[_bs:_be], stream=copy_stream)
            copy_stream.record(copy_done[slot])
        _upload_batch(0, 0)
        compressed_gpu = None
        chunk_offsets_gpu = None
    else:
        compressed_gpu = cp.empty(len(read_buffer), dtype=cp.uint8)
        compressed_gpu.set(read_buffer)
        chunk_offsets_gpu = cp.asarray(chunk_offsets_arr)
    # --- Per-batch scratch buffers (reused across iterations) --------------
    batch_scratch_bytes = max_batch * frame_bytes
    lz4_scratch = cp.empty(batch_scratch_bytes, dtype=cp.uint8)
    shuf_scratch = cp.empty(batch_scratch_bytes, dtype=cp.uint8)

    # --- Pre-allocate final result (possibly narrowed) ---------------------
    # Streaming bin: if det_bin > 1, allocate the final BINNED buffer
    # directly. Per-batch loop bins each batch and writes to the final
    # buffer so the full unbinned (det_bin² × bigger) tensor never lives
    # in VRAM. At 512²×192² with det_bin=2 this is 19.3 GB → 4.83 GB
    # final buffer.
    if streaming_bin and det_bin > 1:
        if frame_shape[-2] % det_bin != 0 or frame_shape[-1] % det_bin != 0:
            raise ValueError(
                f"Detector dims {frame_shape[-2:]} not divisible by det_bin={det_bin}"
            )
        binned_shape = (frame_shape[-2] // det_bin, frame_shape[-1] // det_bin)
        result = cp.empty((total_frames,) + binned_shape, dtype=final_dtype)
    else:
        result = cp.empty((total_frames,) + frame_shape, dtype=final_dtype)

    max_blocks_val = int(block_counts.max())
    n_full_8kb = frame_bytes // BLOCK_SIZE
    tail_bytes = frame_bytes % BLOCK_SIZE

    # Precompute pixel-mask indices on GPU once (shared across batches).
    if pixel_mask is not None:
        _bad_row_np, _bad_col_np = np.where(pixel_mask > 0)
        _has_mask = len(_bad_row_np) > 0
        if _has_mask:
            _bad_row = cp.asarray(_bad_row_np)
            _bad_col = cp.asarray(_bad_col_np)
    else:
        _has_mask = False

    # uint8 saturates counts above 255. The binned path still counts exact
    # saturation. The no-bin hot path uses a faster fused clip/cast kernel;
    # load(dtype="u8") tells the caller that values >255 saturate.
    _clip_warn = final_dtype == np.uint8 and not uint4_mode
    _clipped = cp.zeros((), dtype=cp.uint64) if _clip_warn else None

    n_batches = (total_frames + max_batch - 1) // max_batch
    for batch_idx, start in enumerate(range(0, total_frames, max_batch)):
        end = min(start + max_batch, total_frames)
        batch_n = end - start
        # Streaming upload (double-buffered async): this batch's bytes are
        # already in flight on the copy stream; wait for them, then prefetch
        # the next batch into the spare buffer so its H2D overlaps this
        # batch's kernels. block_starts are chunk-relative (no rebasing);
        # chunk_offsets were rebased per-batch into all_rebased_gpu.
        if streaming_upload:
            slot = batch_idx % 2
            main_stream.wait_event(copy_done[slot])
            if batch_idx + 1 < n_batches:
                next_slot = (batch_idx + 1) % 2
                if batch_idx >= 1:
                    copy_stream.wait_event(kernel_done[next_slot])
                _upload_batch(batch_idx + 1, next_slot)
            cur_compressed = comp_bufs[slot]
            batch_chunk_offsets_gpu = all_rebased_gpu[start:]
        else:
            cur_compressed = compressed_gpu
            batch_chunk_offsets_gpu = chunk_offsets_gpu[start:]

        # 1. LZ4 decompress this batch into lz4_scratch (from offset 0).
        _h5lz4dc_kernel(
            ((max_blocks_val + 1) // 2, 1, batch_n),
            (32, 2, 1),
            (
                cur_compressed,
                batch_chunk_offsets_gpu,
                block_starts_gpu,
                block_counts_gpu[start:],
                block_offsets_gpu[start:],
                np.uint32(BLOCK_SIZE),
                np.uint32(frame_bytes),
                lz4_scratch,
            ),
        )
        # LZ4 is the only consumer of the compressed buffer; once it has run
        # the copy stream may refill this slot for batch_idx+2.
        if streaming_upload:
            main_stream.record(kernel_done[slot])

        # 2. Bitshuffle this batch into shuf_scratch. Pass the full
        #    scratch buffers - the kernel uses batch_n to bound work,
        #    and slicing a uint8 buffer then .view()ing into a wider
        #    dtype can leave CuPy confused about strides.
        if source_itemsize == 2:
            if n_full_8kb:
                _bitshuffle_kernel_u16(
                    (n_full_8kb, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        shuf_scratch.view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // source_itemsize
                if tail_bytes % source_itemsize or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u16(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        shuf_scratch.view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
        else:
            frame_u32s = frame_bytes // 4
            if n_full_8kb:
                _bitshuffle_kernel(
                    (n_full_8kb, 2, batch_n),
                    (32, 32, 1),
                    (
                        lz4_scratch.view(cp.uint32),
                        shuf_scratch.view(cp.uint32),
                        np.uint32(frame_u32s),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // source_itemsize
                if tail_bytes % source_itemsize or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u32(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        shuf_scratch.view(cp.uint32),
                        np.uint32(frame_bytes),
                    ),
                )

        # 3. View the batch prefix of shuf_scratch as source dtype +
        #    batch shape. View the full uint8 scratch first THEN slice
        #    (doing it the other way can silently reinterpret strides).
        n_src_per_frame = frame_bytes // source_itemsize
        batch_view = (
            shuf_scratch.view(source_dtype)[: batch_n * n_src_per_frame]
            .reshape((batch_n,) + frame_shape)
        )

        # 4. Apply pixel_mask to this batch first. Arina writes sentinels
        #    (0xFFFFFFFF for uint32, 0xFFFF for uint16) at dead pixels
        #    and the current narrow check needs to see them as 0 so the
        #    sentinels don't trip it. Same end result as zeroing at the
        #    very end; just moves the mask write inside the batch loop.
        if _has_mask:
            batch_view[:, _bad_row, _bad_col] = 0

        # 5. If narrowing: verify the (masked) batch values fit uint16/uint4.
        if narrow_mode:
            batch_max = int(batch_view.max().get())
            if batch_max >= 65536:
                # Rollback: clean up and raise so the caller can retry.
                del lz4_scratch, shuf_scratch, result
                del compressed_gpu, chunk_offsets_gpu, block_starts_gpu
                del block_counts_gpu, block_offsets_gpu
                cp.get_default_memory_pool().free_all_blocks()
                raise ValueError(
                    f"auto_narrow=True but batch frames [{start}, {end}) "
                    f"have max value {batch_max} >= 65536; uint16 cannot "
                    f"represent this data. Retry with auto_narrow=False "
                    f"(requires ~2× more final-buffer VRAM)."
                )
        if uint4_mode:
            batch_max = int(batch_view.max().get())
            if batch_max > 15:
                del lz4_scratch, shuf_scratch, result
                del compressed_gpu, chunk_offsets_gpu, block_starts_gpu
                del block_counts_gpu, block_offsets_gpu
                cp.get_default_memory_pool().free_all_blocks()
                raise ValueError(
                    "dtype='u4' requires every corrected detector count to fit "
                    f"in 0..15; batch frames [{start}, {end}) have max value "
                    f"{batch_max}. Use dtype='uint8' or dtype='uint16' for "
                    "this dataset."
                )

        # 6. Bin batch in-place (streaming) and/or copy with dtype cast
        #    into the final buffer.
        if streaming_bin and det_bin > 1:
            new_dr = frame_shape[-2] // det_bin
            new_dc = frame_shape[-1] // det_bin
            # Reshape (B, det_r, det_c) → (B, new_dr, det_bin, new_dc, det_bin)
            # then sum over binning axes. Integer accumulation, bit-exact
            # to bin_4dstem on the same data.
            binned_batch = batch_view.reshape(
                batch_n, new_dr, det_bin, new_dc, det_bin
            ).sum(axis=(2, 4))
            if final_dtype == source_dtype:
                result[start:end] = binned_batch
            elif final_dtype == np.uint8 and not uint4_mode:
                # browse uint8: clip@255 per batch into the uint8 output, so the
                # full uint16 block is never materialized (peak = uint8 out + one
                # batch + scratch). clip keeps it linear -> virtual-image sums correct.
                _clipped += (binned_batch > 255).sum(dtype=cp.uint64)
                result[start:end] = cp.minimum(binned_batch, 255).astype(cp.uint8)
            else:
                result[start:end] = binned_batch.astype(final_dtype)
            del binned_batch
        elif final_dtype == source_dtype:
            result[start:end] = batch_view
        elif final_dtype == np.uint8 and uint4_mode:
            result[start:end] = batch_view.astype(cp.uint8)
        elif final_dtype == np.uint8:
            if not _clip_to_uint8(batch_view, result[start:end]):
                _clipped += (batch_view > 255).sum(dtype=cp.uint64)
                result[start:end] = cp.minimum(batch_view, 255).astype(cp.uint8)
        else:
            result[start:end] = batch_view.astype(final_dtype)

    cp.cuda.Device().synchronize()
    # For paths where we kept an exact saturation count, warn if pixels clipped.
    if _clip_warn:
        n_clipped = int(_clipped)
        if n_clipped:
            pct = 100.0 * n_clipped / result.size
            print(
                f"  Warning: dtype='u8' saturated {n_clipped:,} pixels "
                f"({pct:.4f}%) above 255 to 255. Pass dtype='u16' or 'auto' "
                f"to keep full counts."
            )

    if uint4_mode:
        result = pack_uint4_cupy(result)

    # --- Release scratches + compressed; keep only result ------------------
    del lz4_scratch, shuf_scratch
    del compressed_gpu, chunk_offsets_gpu, block_starts_gpu
    del block_counts_gpu, block_offsets_gpu
    if prune_device_pool:
        cp.get_default_memory_pool().free_all_blocks()
    # pixel_mask already applied per-batch above; no final touch needed.

    t_total = time.perf_counter() - t0
    if verbose:
        total_output = total_frames * frame_bytes
        throughput = total_output / t_total / 1e9
        final_label = (
            "packed uint4"
            if uint4_mode
            else "uint32 → uint16 (auto_narrow)"
            if narrow_mode
            else str(np.dtype(source_dtype))
        )
        print(
            f"  Decompressed {total_frames} frames as {final_label} "
            f"in {t_total:.2f}s ({throughput:.1f} GB/s)"
        )
    return result


def _decompress_prepared(
    prepared: dict,
    verbose: bool = False,
    auto_narrow: bool = True,
    batch_bytes_target: int = 1 << 28,
    det_bin: int = 1,
    streaming_bin: bool = False,
    output_dtype: type | np.dtype | None = None,
    streaming_upload: bool | None = None,
    prune_pinned: bool = True,
    prune_device_pool: bool = True,
) -> cp.ndarray:
    """Consume one prepared master and always release its host staging buffer."""
    read_buffer = prepared["read_buffer"]
    try:
        return _decompress_prepared_impl(
            prepared,
            verbose=verbose,
            auto_narrow=auto_narrow,
            batch_bytes_target=batch_bytes_target,
            det_bin=det_bin,
            streaming_bin=streaming_bin,
            output_dtype=output_dtype,
            streaming_upload=streaming_upload,
            prune_device_pool=prune_device_pool,
        )
    except BaseException:
        # A failed kernel or allocation may follow an asynchronous H2D from the
        # page-locked buffer. Finish any queued access before making it reusable.
        if cp is not None:
            try:
                cp.cuda.Device().synchronize()
            except Exception:  # noqa: BLE001, S110 - preserve the original failure
                pass
        raise
    finally:
        if prune_pinned:
            _release_pinned(read_buffer)
        else:
            _release_pinned(read_buffer, prune=False)


def _discover_chunk_names(filepath: str) -> list[str]:
    """Get chunk dataset names (data_000001, etc.) from a master file."""
    with h5py.File(filepath, "r") as f:
        data_group = f.get("entry/data")
        if data_group is None:
            return []
        return sorted([
            name for name in data_group
            if re.match(r"data_\d{6}", name)
        ])


def _absolute_source_path(path: str | os.PathLike[str]) -> str:
    """Absolute source spelling without resolving a watched symlink."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _file_source_signature(path: str) -> dict[str, Any]:
    """JSON-serializable identity and stability fields for one source file."""
    absolute = _absolute_source_path(path)
    try:
        stat = os.stat(absolute)
    except FileNotFoundError:
        return {"path": absolute, "missing": True}
    except OSError as exc:
        return {"path": absolute, "unreadable": True, "error": str(exc)}
    signature: dict[str, Any] = {
        "path": absolute,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }
    if os.path.islink(absolute):
        try:
            link_stat = os.lstat(absolute)
            signature["symlink_target"] = os.readlink(absolute)
            signature["symlink_mtime_ns"] = int(link_stat.st_mtime_ns)
            signature["symlink_ctime_ns"] = int(link_stat.st_ctime_ns)
        except OSError:
            signature["symlink_unreadable"] = True
    return signature


def _master_source_signature(
    master_path: str,
    source_paths: set[str],
    datasets: list[dict[str, Any]],
    *,
    expected_frames: int | None,
    expected_basis: str | None,
) -> dict[str, Any]:
    """Build a deterministic master/chunk fingerprint for poll comparison."""
    return {
        "master": master_path,
        "files": [
            _file_source_signature(path)
            for path in sorted({_absolute_source_path(path) for path in source_paths})
        ],
        "datasets": [dict(dataset) for dataset in datasets],
        "expectation": {
            "frames": expected_frames,
            "basis": expected_basis,
        },
    }


def _normalise_readiness_scan_shape(
    scan_shape: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Validate an explicit readiness frame-count contract."""
    if scan_shape is None:
        return None
    try:
        values = tuple(scan_shape)
    except TypeError as exc:
        raise ValueError(
            "scan_shape must be two positive integers (scan_row, scan_col), "
            "for example scan_shape=(512, 512)."
        ) from exc
    if len(values) != 2:
        raise ValueError(
            "scan_shape must contain exactly two positive integers "
            f"(scan_row, scan_col); got {scan_shape!r}."
        )
    normalized: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(  # noqa: TRY004 - preserve the public validation contract
                "scan_shape values must be positive integers, not booleans; "
                f"got {scan_shape!r}."
            )
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "scan_shape values must be positive integers; "
                f"got {scan_shape!r}."
            ) from exc
        if integer < 1 or integer != value:
            raise ValueError(
                "scan_shape values must be positive integers; "
                f"got {scan_shape!r}."
            )
        normalized.append(integer)
    return normalized[0], normalized[1]


def _scalar_int(handle: h5py.File, path: str) -> int | None:
    """Read one optional scalar integer without following detector data."""
    dataset = handle.get(path)
    if dataset is None:
        return None
    try:
        value = np.asarray(dataset[()])
        if value.size != 1:
            return None
        return int(value.reshape(()).item())
    except (OSError, TypeError, ValueError, OverflowError):
        return None


def _attribute_scan_shape(attributes: Any) -> tuple[int, int] | None:
    """Return a valid positive ``scan_shape`` HDF5 attribute, when present."""
    if "scan_shape" not in attributes:
        return None
    try:
        values = tuple(int(value) for value in attributes["scan_shape"])
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 2 or any(value < 1 for value in values):
        return None
    return values[0], values[1]


def inspect_master_readiness(
    filepath: str | os.PathLike[str],
    *,
    scan_shape: tuple[int, int] | None = None,
) -> MasterReadiness:
    """Inspect whether a 4D-STEM master is complete enough to browse.

    The inspection reads HDF5 headers only. It supports a self-contained
    ``entry/data/data`` dataset and the usual ``data_NNNNNN`` entries, including
    external links to sibling files. For the selected source it totals stored
    frames, checks detector shape and dtype consistency, and compares the total
    with ``scan_shape`` or the master's ``ntrigger``/``nimages`` metadata.

    The returned ``source_signature`` captures the master, every selected source
    file, and every selected dataset header. Compare it across separate polls to
    establish a caller-defined stability probation; this function deliberately
    does not sleep or perform a second poll itself.

    Parameters
    ----------
    filepath
        Master HDF5 path.
    scan_shape
        Optional explicit ``(scan_row, scan_col)`` contract. When supplied, it
        takes precedence over the master's expected-frame metadata.

    Returns
    -------
    MasterReadiness
        Structured readiness state, corrective action, and source signature.

    Raises
    ------
    ValueError
        If ``scan_shape`` is not exactly two positive integers.
    """
    explicit_scan_shape = _normalise_readiness_scan_shape(scan_shape)
    master_path = _absolute_source_path(filepath)
    source_paths = {master_path}
    datasets: list[dict[str, Any]] = []
    initial_files = {master_path: _file_source_signature(master_path)}
    expected_frames = (
        int(explicit_scan_shape[0] * explicit_scan_shape[1])
        if explicit_scan_shape is not None
        else None
    )
    expected_basis = (
        f"explicit scan_shape={explicit_scan_shape}"
        if explicit_scan_shape is not None
        else None
    )
    source_kind = "unavailable"
    actual_frames: int | None = None
    detector_shape: tuple[int, int] | None = None
    common_dtype: str | None = None

    def result(ready: bool, reason: str, action: str) -> MasterReadiness:
        return MasterReadiness(
            ready=bool(ready),
            reason=str(reason),
            action=str(action),
            source_kind=source_kind,
            actual_frames=actual_frames,
            expected_frames=expected_frames,
            detector_shape=detector_shape,
            dtype=common_dtype,
            source_signature=_master_source_signature(
                master_path,
                source_paths,
                datasets,
                expected_frames=expected_frames,
                expected_basis=expected_basis,
            ),
        )

    master_stat = initial_files[master_path]
    if master_stat.get("missing", False):
        return result(
            False,
            f"master file is missing: {master_path}",
            "Wait for the master file to be atomically renamed into place, "
            "then poll again.",
        )
    if master_stat.get("unreadable", False):
        return result(
            False,
            f"master file cannot be inspected: {master_path} "
            f"({master_stat.get('error', 'unknown filesystem error')})",
            "Fix file permissions or storage availability, then poll again.",
        )
    if int(master_stat.get("size", 0)) <= 0:
        return result(
            False,
            f"master file is empty: {master_path}",
            "Wait for the master HDF5 header to finish writing, then poll again.",
        )

    try:
        with h5py.File(master_path, "r") as master:
            data_group = master.get("entry/data")
            if data_group is None:
                return result(
                    False,
                    "master is missing the entry/data group",
                    "Wait for acquisition to finish, or recopy a complete master file.",
                )

            chunk_names = sorted(
                name
                for name in data_group
                if re.fullmatch(r"data_\d{6}", name)
            )
            source_names = chunk_names or (["data"] if "data" in data_group else [])
            if not source_names:
                return result(
                    False,
                    "master has no entry/data/data dataset or data_NNNNNN sources",
                    "Wait for detector data links to finish writing, or recopy "
                    "the complete acquisition group.",
                )

            links = [data_group.get(name, getlink=True) for name in source_names]
            source_kind = (
                "external"
                if any(isinstance(link, h5py.ExternalLink) for link in links)
                else "inline"
            )
            ntrigger = _scalar_int(
                master,
                "entry/instrument/detector/detectorSpecific/ntrigger",
            )
            nimages = _scalar_int(
                master,
                "entry/instrument/detector/detectorSpecific/nimages",
            )
            if explicit_scan_shape is None and ntrigger is not None:
                if ntrigger < 1:
                    return result(
                        False,
                        f"master ntrigger metadata is not positive: {ntrigger}",
                        "Wait for acquisition metadata to finish writing, or repair "
                        "ntrigger before loading.",
                    )
                images_per_trigger = int(1 if nimages is None else nimages)
                if images_per_trigger < 1:
                    return result(
                        False,
                        "master nimages metadata is not positive: "
                        f"{images_per_trigger}",
                        "Wait for acquisition metadata to finish writing, or repair "
                        "nimages before loading.",
                    )
                expected_frames = int(ntrigger * images_per_trigger)
                expected_basis = (
                    f"master metadata ntrigger={ntrigger}, nimages={images_per_trigger}"
                )

            observed_frames = 0
            detector_shapes: set[tuple[int, int]] = set()
            dtypes: set[str] = set()
            metadata_scan_shapes: set[tuple[int, int]] = set()
            for attributes in (master.attrs, data_group.attrs):
                metadata_shape = _attribute_scan_shape(attributes)
                if metadata_shape is not None:
                    metadata_scan_shapes.add(metadata_shape)

            for name, link in zip(source_names, links, strict=True):
                if isinstance(link, h5py.ExternalLink):
                    link_filename = os.fsdecode(os.fspath(link.filename))
                    source_path = (
                        _absolute_source_path(link_filename)
                        if os.path.isabs(link_filename)
                        else _absolute_source_path(
                            os.path.join(os.path.dirname(master_path), link_filename)
                        )
                    )
                    dataset_path = str(link.path)
                    source_paths.add(source_path)
                    initial_files[source_path] = _file_source_signature(source_path)
                    record: dict[str, Any] = {
                        "name": str(name),
                        "kind": "external",
                        "file": source_path,
                        "dataset": dataset_path,
                    }
                    datasets.append(record)
                    source_stat = initial_files[source_path]
                    if source_stat.get("missing", False):
                        return result(
                            False,
                            f"linked detector file is missing: {source_path}",
                            "Finish copying or writing the linked detector file "
                            "next to the master, then poll again.",
                        )
                    if source_stat.get("unreadable", False):
                        return result(
                            False,
                            f"linked detector file cannot be inspected: "
                            f"{source_path} "
                            f"({source_stat.get('error', 'unknown filesystem error')})",
                            "Fix file permissions or storage availability, then "
                            "poll again.",
                        )
                    if int(source_stat.get("size", 0)) <= 0:
                        return result(
                            False,
                            f"linked detector file is empty: {source_path}",
                            "Wait for the linked detector HDF5 file to finish "
                            "writing, then poll again.",
                        )
                    try:
                        source_handle = h5py.File(source_path, "r")
                    except OSError as exc:
                        return result(
                            False,
                            "linked detector file is not readable HDF5: "
                            f"{source_path} ({exc})",
                            "Wait for the detector writer to close or flush the "
                            "file, then poll again; recopy it if the error persists.",
                        )
                    with source_handle:
                        dataset = source_handle.get(dataset_path)
                        if dataset is None:
                            return result(
                                False,
                                "linked detector dataset is missing: "
                                f"{source_path}:{dataset_path}",
                                "Repair the external link or recopy the acquisition "
                                "so it targets entry/data/data.",
                            )
                        shape = tuple(int(value) for value in dataset.shape)
                        dtype_str = np.dtype(dataset.dtype).str
                        metadata_shape = _attribute_scan_shape(dataset.attrs)
                else:
                    source_path = master_path
                    dataset_path = f"{data_group.name}/{name}"
                    record = {
                        "name": str(name),
                        "kind": "inline",
                        "file": source_path,
                        "dataset": dataset_path,
                    }
                    datasets.append(record)
                    try:
                        dataset = data_group[name]
                        shape = tuple(int(value) for value in dataset.shape)
                        dtype_str = np.dtype(dataset.dtype).str
                        metadata_shape = _attribute_scan_shape(dataset.attrs)
                    except (KeyError, OSError, TypeError, ValueError) as exc:
                        return result(
                            False,
                            "inline detector dataset is not readable: "
                            f"{dataset_path} ({exc})",
                            "Wait for the master dataset header to finish writing, "
                            "then poll again; recopy it if the error persists.",
                        )

                if len(shape) < 3:
                    record.update({"shape": list(shape), "dtype": dtype_str})
                    return result(
                        False,
                        f"detector dataset {dataset_path} has shape {shape}; "
                        "expected at least (frame, det_row, det_col)",
                        "Repair or reacquire the dataset with explicit frame and "
                        "two detector dimensions.",
                    )
                frames = int(np.prod(shape[:-2], dtype=np.int64))
                current_detector_shape = (int(shape[-2]), int(shape[-1]))
                record.update(
                    {
                        "shape": list(shape),
                        "dtype": dtype_str,
                        "frames": frames,
                        "detector_shape": list(current_detector_shape),
                    }
                )
                if metadata_shape is not None:
                    metadata_scan_shapes.add(metadata_shape)
                    record["scan_shape"] = list(metadata_shape)
                observed_frames += frames
                detector_shapes.add(current_detector_shape)
                dtypes.add(dtype_str)

            actual_frames = int(observed_frames)
            if len(detector_shapes) != 1:
                observed = ", ".join(str(shape) for shape in sorted(detector_shapes))
                return result(
                    False,
                    f"detector sources have inconsistent detector shapes: {observed}",
                    "Use a narrower master pattern or repair the acquisition so "
                    "every detector chunk has the same (row, col) shape.",
                )
            detector_shape = next(iter(detector_shapes))
            if len(dtypes) != 1:
                observed = ", ".join(sorted(dtypes))
                return result(
                    False,
                    f"detector sources have inconsistent dtypes: {observed}",
                    "Repair or reacquire the acquisition so every detector chunk "
                    "uses the same stored dtype.",
                )
            common_dtype = next(iter(dtypes))
            if len(metadata_scan_shapes) > 1:
                observed = ", ".join(
                    str(shape) for shape in sorted(metadata_scan_shapes)
                )
                return result(
                    False,
                    "detector sources have inconsistent scan_shape metadata: "
                    f"{observed}",
                    "Repair the conflicting scan_shape attributes, or pass the "
                    "correct explicit scan_shape after confirming the acquisition "
                    "layout.",
                )
            if (
                expected_frames is None
                and len(metadata_scan_shapes) == 1
            ):
                metadata_scan_shape = next(iter(metadata_scan_shapes))
                expected_frames = int(metadata_scan_shape[0] * metadata_scan_shape[1])
                expected_basis = f"HDF5 scan_shape={metadata_scan_shape}"
            if actual_frames < 1:
                return result(
                    False,
                    "detector sources contain zero stored frames",
                    "Wait for detector frames to be written, then poll again.",
                )
            if expected_frames is not None and actual_frames != expected_frames:
                if actual_frames < expected_frames:
                    action = (
                        "Wait for the remaining detector frames or chunks to finish writing, "
                        "then poll again."
                    )
                else:
                    action = (
                        "Pass the correct explicit scan_shape or repair the master frame-count "
                        "metadata before loading."
                    )
                return result(
                    False,
                    f"stored frame count is {actual_frames}; expected "
                    f"{expected_frames} from {expected_basis}",
                    action,
                )
    except OSError as exc:
        return result(
            False,
            f"master file is not readable HDF5: {master_path} ({exc})",
            "Wait for the master writer to close or flush the file, then poll "
            "again; recopy it if the error persists.",
        )
    except (KeyError, TypeError, ValueError) as exc:
        return result(
            False,
            f"master HDF5 headers are incomplete or invalid: {master_path} ({exc})",
            "Wait for acquisition to finish, then poll again; recopy or repair "
            "the group if the error persists.",
        )

    final_signature = _master_source_signature(
        master_path,
        source_paths,
        datasets,
        expected_frames=expected_frames,
        expected_basis=expected_basis,
    )
    final_files = {item["path"]: item for item in final_signature["files"]}
    changed = [
        path
        for path, before in initial_files.items()
        if final_files.get(path, {"path": path, "missing": True}) != before
    ]
    if changed:
        names = ", ".join(os.path.basename(path) for path in changed)
        return MasterReadiness(
            ready=False,
            reason=f"source files changed during readiness inspection: {names}",
            action=(
                "Wait for acquisition or copy writes to finish, then compare a "
                "fresh readiness signature on the next poll."
            ),
            source_kind=source_kind,
            actual_frames=actual_frames,
            expected_frames=expected_frames,
            detector_shape=detector_shape,
            dtype=common_dtype,
            source_signature=final_signature,
        )
    return MasterReadiness(
        ready=True,
        reason=(
            "master and detector sources are complete, readable, and internally "
            "consistent"
        ),
        action="Ready to open with Show4DSTEM.",
        source_kind=source_kind,
        actual_frames=actual_frames,
        expected_frames=expected_frames,
        detector_shape=detector_shape,
        dtype=common_dtype,
        source_signature=final_signature,
    )


def is_master_ready(
    filepath: str | os.PathLike[str],
    *,
    scan_shape: tuple[int, int] | None = None,
) -> bool:
    """Return whether a master passes header-only completeness inspection.

    This compatibility wrapper delegates to :func:`inspect_master_readiness`.
    Call that function when a folder watcher needs the frame counts, corrective
    reason, or a signature to compare across separate stability polls.

    Parameters
    ----------
    filepath
        Master HDF5 path.
    scan_shape
        Optional explicit ``(scan_row, scan_col)`` expected frame count.

    Returns
    -------
    bool
        ``True`` only for complete, readable, internally consistent sources.
    """
    return inspect_master_readiness(filepath, scan_shape=scan_shape).ready


def _load_master_pipelined(
    filepath: str,
    chunk_names: list[str],
    *,
    apply_mask: bool = True,
    auto_narrow: bool = True,
    det_bin: int = 1,
    streaming_bin: bool = False,
    output_dtype: type | np.dtype | None = None,
    n_groups: int = 9,
):
    """Single-master load with disk‖GPU overlap across chunk-file groups.

    Splits the master's chunk files into ``n_groups`` contiguous groups. A
    producer thread reads + header-parses group N+1 from disk
    (``_prepare_master``, pure CPU/disk, GIL released during I/O) while the main
    thread runs the GPU decompress on group N. Wall drops from disk + gpu
    (serial) toward read(G0) + max(rest_disk, all_gpu).

    Nine groups keeps the fresh no-bin browse path under a second on 512² Arina
    masters by starting the first GPU decode earlier while the remaining chunk
    files continue reading in the producer. No concat: the per-group outputs are
    written into one preallocated output
    array at the right frame offset (each contiguous chunk group → contiguous
    frame range), so peak VRAM is output + one group's transient, not 2× output.
    Returns (data, pixel_mask).
    """
    import queue
    import threading

    n = len(chunk_names)
    n_groups = max(1, min(n_groups, n))
    bounds = [round(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [chunk_names[bounds[i]:bounds[i + 1]] for i in range(n_groups)
              if bounds[i + 1] > bounds[i]]

    q: queue.Queue = queue.Queue(maxsize=1)  # 1 in-flight group while GPU works
    _SENT = object()

    def producer():
        for g in groups:
            try:
                q.put((_prepare_master(filepath, g, apply_mask), None))
            except (FileNotFoundError, OSError, ValueError) as e:
                q.put((None, e))
                return
        q.put(_SENT)

    threading.Thread(target=producer, daemon=True).start()

    out = None
    pixel_mask = None
    cursor = 0
    try:
        while True:
            item = q.get()
            if item is _SENT:
                break
            prepared, err = item
            if err is not None:
                raise err
            d = _decompress_prepared(
                prepared,
                verbose=False,
                auto_narrow=auto_narrow,
                det_bin=det_bin,
                streaming_bin=streaming_bin,
                output_dtype=output_dtype,
                prune_pinned=False,
            )
            g_frames = d.shape[0]
            if out is None:
                # First group reveals the post-bin detector shape; total frames
                # comes from the master's ntrigger (one header read, not 105 chunk
                # opens). Preallocate the full flat output once, then write each
                # group into its frame slice (no concat).
                det_shape = d.shape[1:]
                total = _master_total_frames(filepath, chunk_names)
                out = cp.empty((total, *det_shape), dtype=d.dtype)
                pixel_mask = prepared.get("pixel_mask")
            out[cursor:cursor + g_frames] = d
            cursor += g_frames
            del d
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        # The nine-group loader reaches four overlapping host/GPU staging
        # slots. Retain those slots across repeated loads; collapsing to one
        # makes every reopen page-register roughly three 360 MB buffers again.
        _prune_pinned_free(retain_per_size_class=4)
    if out is None:
        raise FileNotFoundError(f"{filepath}: no data decompressed")
    if cursor < out.shape[0]:
        out = out[:cursor]
    return out, pixel_mask


def _master_total_frames(filepath: str, chunk_names: list[str]) -> int:
    """Total frame count for the master — from ntrigger (one header read)."""
    with h5py.File(filepath, "r") as f:
        nt = f.get("entry/instrument/detector/detectorSpecific/ntrigger")
        if nt is not None:
            return int(nt[()])
        # Fallback: sum chunk-dataset shapes (rare — no ntrigger field).
        import os
        master_dir = os.path.dirname(os.path.abspath(filepath))
        dg = f["entry/data"]
        total = 0
        for cn in chunk_names:
            link = dg.get(cn, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                with h5py.File(os.path.join(master_dir, link.filename), "r") as df:
                    total += df["entry/data/data"].shape[0]
            else:
                total += dg[cn].shape[0]
        return total


def _load_master_optimized(
    filepath: str,
    chunk_names: list[str],
    apply_mask: bool = True,
    verbose: bool = False,
    auto_narrow: bool = True,
    det_bin: int = 1,
    streaming_bin: bool = False,
    output_dtype: type | np.dtype | None = None,
):
    """Bulk-read loader for Dectris master files.

    Two-phase pipeline: _prepare_master() reads compressed bytes from disk
    and parses headers on CPU, _decompress_prepared() transfers and
    decompresses on GPU. Combined here for the single-file load() path.

    Typical speedup: 5s → <0.5s for 262K frames.

    Returns
    -------
    data : cp.ndarray
    pixel_mask : np.ndarray | None
        The Arina pixel_mask read by `_prepare_master` (already used to
        zero dead pixels in-place). Surfaced so callers can thread it to
        downstream consumers without re-opening the HDF5.
    """
    import time
    t0 = time.perf_counter()
    # Enough chunk files to overlap disk read with GPU decompress (group
    # pipeline). Below 8, the single bulk read is already fast and the per-group
    # split isn't worth it.
    if len(chunk_names) >= 8 and not _is_uint4_load_dtype(output_dtype):
        data, pixel_mask = _load_master_pipelined(
            filepath, chunk_names, apply_mask=apply_mask,
            auto_narrow=auto_narrow, det_bin=det_bin,
            streaming_bin=streaming_bin, output_dtype=output_dtype)
        if verbose:
            t_total = time.perf_counter() - t0
            size_gb = data.size * data.dtype.itemsize / 1e9
            print(f"  Loaded {data.shape[0]} frames ({size_gb:.1f} GB) "
                  f"in {t_total:.2f}s (disk‖gpu group pipeline)")
        return data, pixel_mask
    prepared = _prepare_master(filepath, chunk_names, apply_mask)
    result = _decompress_prepared(
        prepared, verbose=False, auto_narrow=auto_narrow,
        det_bin=det_bin, streaming_bin=streaming_bin,
        output_dtype=output_dtype,
    )
    t_total = time.perf_counter() - t0
    if verbose:
        total_output = result.nbytes if is_packed_uint4(result) else (
            result.size * result.dtype.itemsize
        )
        throughput = total_output / t_total / 1e9
        size_gb = total_output / 1e9
        narrowed = (
            not is_packed_uint4(result)
            and result.dtype != prepared["dtype"]
            and np.dtype(prepared["dtype"]) == np.dtype(np.uint32)
        )
        narrow_note = "  (uint32 → uint16 auto-narrowed)" if narrowed else ""
        print(
            f"  Loaded {prepared['total_frames']} frames ({size_gb:.1f} GB) "
            f"in {t_total:.2f}s ({throughput:.1f} GB/s){narrow_note}"
        )
    return result, prepared.get("pixel_mask")


def _load_sharded(
    filepaths: list[str],
    devices: list[int] | str,
    *,
    dataset_path=None, apply_mask=True, scan_shape=None,
    scan_order="row-major", det_bin=1, verbose=True, auto_narrow=True,
    output_dtype=None,
) -> LoadResult:
    """Sharded multi-GPU load — files split across GPUs, each kept on its card.

    Files are assigned to devices in disk-interleaved order: when files are
    spread across physical disks, each GPU gets a balanced disk stream instead
    of both GPUs waiting on the same drive. One thread per device loads its
    subset (each thread pinned to its GPU via ``cp.cuda.Device``) and stacks them
    into one per-device array. No cross-GPU gather, no host bounce — the only way
    a stack exceeding one card's VRAM fits, and faster than gather.

    Returns ``LoadResult`` whose ``.data`` is ``{device: stacked_array}`` (each
    array resident on that device) and ``.metadata["device_map"] = {file_idx:
    device}`` plus ``["shard_order"] = {device: [file_idx, ...]}``.
    """
    import concurrent.futures
    import time

    if devices == "all":
        devices = list(range(cp.cuda.runtime.getDeviceCount()))
    devices = [int(d) for d in devices]
    n_files = len(filepaths)
    assign = _assign_indices_to_devices(filepaths, devices)
    if verbose:
        bin_str = f", det_bin={det_bin}" if det_bin > 1 else ""
        print(f"Loading {n_files} files sharded across GPUs {devices}{bin_str}")

    shards: dict[int, cp.ndarray] = {}
    shard_order: dict[int, list[int]] = {}
    meta_box: dict[int, dict] = {}
    skipped: list[int] = []

    def worker(dev: int):
        idxs = assign[dev]
        if not idxs:
            return
        with cp.cuda.Device(dev):
            stacked = None
            order = []
            for idx in idxs:
                try:
                    r = load(filepaths[idx], dataset_path=dataset_path,
                             apply_mask=apply_mask, scan_shape=scan_shape,
                             scan_order=scan_order,
                             det_bin=det_bin, verbose=False,
                             auto_narrow=auto_narrow, dtype=output_dtype)
                except (FileNotFoundError, OSError, ValueError) as e:
                    if verbose:
                        print(f"  gpu{dev} [{idx+1}/{n_files}] SKIPPED: {e}")
                    skipped.append(idx)
                    continue
                d = r.data
                meta_box.setdefault(dev, r.metadata)
                # Pre-allocate the per-device stack on the first file, then copy
                # each file into its slot and free the temp — peak is stack +
                # one transient file, not stack + all files (which cp.stack does).
                if stacked is None:
                    stacked = cp.empty((len(idxs), *d.shape), dtype=d.dtype)
                    anchor = d.shape
                if d.shape != anchor:
                    if verbose:
                        print(f"  gpu{dev} [{idx+1}/{n_files}] SKIPPED: shape mismatch")
                    del d, r
                    cp.get_default_memory_pool().free_all_blocks()
                    skipped.append(idx)
                    continue
                stacked[len(order)] = d
                order.append(idx)
                del d, r
                cp.get_default_memory_pool().free_all_blocks()
            if order:
                shards[dev] = stacked[:len(order)] if len(order) < len(idxs) else stacked
                shard_order[dev] = order

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, devices))

    if not shards:
        raise FileNotFoundError(f"All {n_files} files failed to load")

    device_map = {idx: dev for dev, idxs in shard_order.items() for idx in idxs}
    meta = dict(next(iter(meta_box.values())))
    meta["device_map"] = device_map
    meta["shard_order"] = shard_order
    meta["sharded"] = True
    if verbose:
        dt = time.perf_counter() - t0
        total_gib = sum(s.nbytes for s in shards.values()) / (1 << 30)
        per = " ".join(f"gpu{d}:{shards[d].shape[0]}f/{shards[d].nbytes/(1<<30):.0f}GiB"
                       for d in sorted(shards))
        skip = f" (skipped {len(skipped)})" if skipped else ""
        print(f"  Done: {len(device_map)} files{skip} sharded [{per}] "
              f"total {total_gib:.1f} GiB in {dt:.2f}s")
    return LoadResult(shards, meta)


def _load_view(
    filepath,
    backend: str,
    *,
    dataset_path=None,
    apply_mask: bool = True,
    scan_shape=None,
    scan_order: str = "row-major",
    det_bin: int = 1,
    verbose: bool = True,
    auto_narrow: bool = True,
    output_dtype=None,
    row_prefix: bool = False,
    precompute_detector_sum: bool = False,
    skip_mps_memory_check: bool | None = None,
):
    """View/screen load path for the non-cuda backends (cpu, mps).

    Decompresses to a numpy array via the chosen backend's ``load_master``,
    then runs the SAME post-processing the cuda path applies — pixel mask,
    auto_narrow (uint32→uint16), output_dtype cast, scan-shape unflatten — so
    the returned LoadResult is shape/metadata-identical to a cuda load, just
    numpy instead of cupy. The MPS no-bin path is the exception: it returns a
    zero-copy ``MPSChunked4DSTEM`` object because a full 512x512x192x192 stack
    cannot be one Metal buffer on 24 GB Apple Silicon.
    """
    import time

    if backend == "cpu":
        from .backends.cpu import reference as _be
    elif backend == "mps":
        from .backends.mps import decoder as _be
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"_load_view does not handle backend={backend!r}")

    def _one(path):
        meta = get_metadata(str(path))
        mps_chunk_output_dtype = None
        if _is_uint4_load_dtype(output_dtype):
            raise NotImplementedError(
                "dtype='u4' means packed 4-bit detector counts (0..15). "
                f"Packed uint4 load is currently implemented for CUDA; "
                f"backend={backend!r} was selected."
            )
        if backend == "mps" and output_dtype is not None:
            if isinstance(output_dtype, str) and output_dtype in ("u8", "uint8"):
                mps_chunk_output_dtype = np.dtype(np.uint8)
            else:
                mps_chunk_output_dtype = np.dtype(output_dtype)
        if (
            backend == "mps"
            and dataset_path is None
            and (
                output_dtype is None
                or mps_chunk_output_dtype
                in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32))
            )
        ):
            if _normalize_scan_order(scan_order) != "row-major":
                raise ValueError(
                    "scan_order='serpentine' is not supported for full no-bin "
                    "MPS zero-copy chunked loads yet. Use scan_region=... or "
                    "det_bin>1, or load on CUDA."
                )
            data = _be.load_mps_4dstem(
                str(path),
                scan_shape=scan_shape,
                apply_mask=apply_mask,
                verbose=verbose,
                row_prefix=row_prefix,
                det_bin=det_bin,
                output_dtype=mps_chunk_output_dtype,
                precompute_detector_sum=precompute_detector_sum,
                skip_mps_memory_check=skip_mps_memory_check,
            )
            meta.update(data.metadata)
            meta["scan_order"] = _normalize_scan_order(scan_order)
            if apply_mask:
                mask = read_pixel_mask(str(path))
                if mask is not None:
                    meta["pixel_mask"] = mask
            return data, meta

        # Read the dead-pixel mask up front and hand it to the backend so dead
        # pixels are zeroed BEFORE binning (the 65535-sentinel-into-a-bin bug),
        # matching the cuda path. Mask is detector-pixel-resolution, so it only
        # records into meta at det_bin == 1.
        mask = read_pixel_mask(str(path)) if apply_mask else None
        backend_load_kwargs = {}
        if backend == "mps" and mps_chunk_output_dtype == np.dtype(np.uint8):
            backend_load_kwargs["output_dtype"] = mps_chunk_output_dtype
        data = _be.load_master(
            str(path),
            det_bin=det_bin,
            pixel_mask=mask,
            verbose=verbose,
            **backend_load_kwargs,
        )
        if mask is not None and det_bin == 1:
            meta["pixel_mask"] = mask
        if auto_narrow and data.dtype == np.uint32 and int(data.max()) < 65536:
            data = data.astype(np.uint16)
        if output_dtype is not None:
            if _is_uint4_load_dtype(output_dtype):
                raise NotImplementedError(
                    "dtype='u4' means packed 4-bit detector counts (0..15). "
                    f"Packed uint4 load is currently implemented for CUDA; "
                    f"backend={backend!r} was selected."
                )
            out_dtype = np.dtype(output_dtype)
            if out_dtype == np.dtype(np.uint8) and data.dtype != np.uint8:
                data = np.minimum(data, 255).astype(np.uint8)
            else:
                data = data.astype(out_dtype)
        data = _apply_scan_shape(data, scan_shape, meta, scan_order)
        meta["scan_order"] = _normalize_scan_order(scan_order)
        if backend == "mps":
            # Torch MPS tensor is the first-class GPU citizen on Apple, the peer
            # of cupy on cuda. Show4DSTEM consumes it directly and runs BF/DF +
            # virtual images on-GPU from here — no numpy hop (numpy BF/DF is
            # ~2000ms = 0 fps; torch MPS GEMM is ~38ms). The decode already ran
            # on the GPU; this is the one unavoidable H2D of the result
            # (torch uses a private Metal heap, ~0.17s for a 4.8 GB binned load).
            import torch
            data = torch.from_numpy(np.ascontiguousarray(data)).to("mps")
        return data, meta

    paths = list(filepath) if isinstance(filepath, (list, tuple)) else None
    if backend == "mps" and det_bin == 1 and paths is not None:
        raise ValueError(
            "MPS no-bin load returns zero-copy chunks and currently supports "
            "one master at a time. Pass one path, or use det_bin>1 for the "
            "stacked view path."
        )
    if row_prefix:
        if backend != "mps":
            raise ValueError("row_prefix=True is only supported with backend='mps'.")
        if det_bin != 1:
            raise ValueError("row_prefix=True is an exact no-bin MPS layout; use det_bin=1.")
        if dataset_path is not None or output_dtype is not None:
            raise ValueError(
                "row_prefix=True is only supported for full master-file MPS loads."
            )
    t0 = time.perf_counter()
    if paths is None:
        data, meta = _one(filepath)
        if verbose and not (
            backend == "mps" and getattr(data, "chunks", None) is not None
        ):
            nbytes = data.nbytes if hasattr(data, "nbytes") else (
                data.element_size() * data.nelement())  # torch tensor
            print(f"  Loaded {tuple(data.shape)} ({nbytes / 1e9:.1f} GB) in "
                  f"{time.perf_counter() - t0:.2f}s ({backend} backend)")
        return LoadResult(data, meta)
    # Multi-file: stack with a leading file axis (matches cuda multi-file).
    first, meta = _one(paths[0])
    out = np.empty((len(paths), *first.shape), dtype=first.dtype)
    out[0] = first
    for i, path in enumerate(paths[1:], start=1):
        arr, _ = _one(path)
        out[i] = arr
    meta["n_files"] = len(paths)
    if verbose:
        gb = out.nbytes / 1e9
        print(f"  Loaded {len(paths)} files {out.shape} ({gb:.1f} GB) in "
              f"{time.perf_counter() - t0:.2f}s ({backend} backend)")
    return LoadResult(out, meta)


def _browse_dtype_advise_and_cast(data, dtype, verbose):
    """Recommend / apply the smallest lossless integer dtype for BROWSING.

    Browsing is a visual call, and Arina counts are usually low, so uint8 (half
    the memory) is often lossless. This inspects the real count range and:
      - always prints a recommendation (verbose),
      - ``dtype='u8'``: clip at 255 + cast to uint8 (linear, so virtual-image
        sums stay correct), printing how many pixels clipped,
      - ``dtype='auto'``: pick uint8 only if it is lossless (max <= 255), else
        keep the native dtype,
      - ``dtype='u16'`` / ``None``: keep native (None still prints advice).
    Raw uint16 stays the source for reconstruction; uint8 is screening-only.
    """
    sel = (dtype or "").lower()
    if not verbose and sel in {"u16", "uint16", "native", "full", "exact"}:
        # The caller explicitly requested count-preserving data. Sampling four
        # million pixels cannot change that decision and can trigger needless
        # peer-device synchronization for multi-GPU folder browsing.
        return data
    if data.dtype != np.uint8 and data.dtype.kind == "u":
        try:
            # Estimate the count range from a strided ~4M-element SAMPLE, not a
            # full-block reduction: cupy max()/mean() run at ~3 GB/s on this card
            # (sm_120), so a full pass over 19 GB adds ~15 s. The sample is plenty
            # for a recommendation; the dtype='u8' decode-direct path counts the
            # real clips exactly anyway.
            flat = data.reshape(-1)
            step = max(1, int(flat.size) // 4_000_000)
            sample = flat[::step]
            mx = int(sample.max())
            pct255 = float((sample > 255).mean()) * 100.0
        except (RuntimeError, MemoryError, ValueError):
            return data
        want_u8 = sel in ("u8", "uint8") or (sel == "auto" and mx <= 255)
        if want_u8 and data.dtype.itemsize > 1:
            # Clipping at 255 keeps the browse view linear; only the >255 tail is lost.
            data = (data if mx <= 255 else data.clip(0, 255)).astype(np.uint8)
            if verbose:
                if pct255 == 0.0:
                    print(f"  Loaded this in uint8 to save you memory - your brightest pixel is "
                          f"only {mx} counts, so nothing was lost and you're using "
                          f"{data.nbytes/1e9:.1f} GB instead of {data.nbytes*2/1e9:.1f} GB.")
                else:
                    print(f"  Loaded this in uint8 for browsing - I clipped {pct255:.2f}% of pixels "
                          f"at 255 (your saturated bright spot, where counts reach {mx}). That's fine "
                          f"for looking at the data; reconstruction always uses the raw uint16.")
        elif verbose:
            if mx <= 255:
                print(f"  Heads up: your brightest pixel is only {mx} counts (well under 255), so you "
                      f"could browse this in uint8 with zero loss and use half the memory "
                      f"({data.nbytes/2/1e9:.1f} GB instead of {data.nbytes/1e9:.1f} GB). "
                      f"Just pass dtype='u8' when you load.")
            else:
                print(f"  Heads up: this dataset has some bright pixels (counts up to {mx}; {pct255:.2f}% "
                      f"sit above 255). uint16 keeps everything exact. If you want to browse lighter, "
                      f"dtype='u8' halves the memory and clips only that {pct255:.2f}% - fine for screening, "
                      f"and reconstruction still uses the raw data.")
    return data


def _is_uint8_browse_dtype(dtype: str | None) -> bool:
    """Return True when the public browse dtype requests 8-bit unsigned data."""
    return isinstance(dtype, str) and dtype.lower() in ("u8", "uint8")


def _is_uint32_load_dtype(dtype: str | None) -> bool:
    """Return True when the public dtype requests native 4-byte unsigned data."""
    return isinstance(dtype, str) and dtype.lower() in ("u32", "uint32")


def _is_uint4_load_dtype(dtype: str | None) -> bool:
    """Return True when the public dtype requests packed 4-bit unsigned data."""
    return isinstance(dtype, str) and dtype.lower() in ("u4", "uint4")


def _load_scan_crop_impl(
    filepath: str,
    scan_region: tuple[int, int, int, int] | list[int],
    *,
    backend: str = "cuda",
    scan_shape: tuple[int, int] | None = None,
    scan_order: str = "row-major",
    det_bin: int = 1,
    apply_mask: bool = True,
    verbose: bool = True,
    auto_narrow: bool = True,
    output_dtype: type | np.dtype | None = None,
    target_scan_region: tuple[int, int, int, int] | list[int] | None = None,
    scan_shift_row_col=None,
    scan_resample_dtype: type | np.dtype = np.float32,
    detector_region: tuple[int, int, int, int] | list[int] | None = None,
) -> LoadResult:
    """Load only a rectangular scan region from a raw HDF5 master.

    Parameters
    ----------
    filepath
        Dectris/Arina master HDF5 file.
    scan_region
        ``(row_start, row_stop, col_start, col_stop)`` in the full scan grid.
    scan_shape
        Full acquisition scan shape. When omitted, it is derived from metadata.
    scan_order
        Flattened scan ordering. ``"row-major"`` keeps the normal raster order;
        ``"serpentine"`` maps odd rows from right to left while returning the
        crop in normal ``(row, col)`` order.
    det_bin
        Optional detector binning factor. The scan region is never binned.
    detector_region
        Optional detector crop as ``(row_start, row_stop, col_start, col_stop)``
        in the decoded detector grid after ``det_bin``.

    Returns
    -------
    LoadResult
        ``data`` is a backend array with shape
        ``(region_rows, region_cols, det_rows, det_cols)``. Metadata keeps the
        full acquisition grid in ``full_scan_shape`` and the loaded patch in
        ``scan_region``.
    """
    import time

    from .backends import resolve_backend

    resolved_backend = resolve_backend(backend)
    if resolved_backend == "cuda" and cp is None:
        raise RuntimeError("load(..., scan_region=...) requires CuPy/CUDA")
    if resolved_backend not in {"cuda", "mps"}:
        raise RuntimeError(
            "load(..., scan_region=...) supports accelerated crop-first IO on CUDA and "
            f"MPS; backend={resolved_backend!r} was selected."
        )
    if target_scan_region is not None and resolved_backend != "cuda":
        raise RuntimeError(
            "target_scan_region= is currently implemented for backend='cuda' "
            "only; MPS needs a separate Metal sampler."
        )
    if (target_scan_region is None) != (scan_shift_row_col is None):
        raise ValueError(
            "target_scan_region= and scan_shift_row_col= must be passed together."
        )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    t0 = time.perf_counter()
    meta = get_metadata(filepath)
    full_scan_shape = scan_shape if scan_shape is not None else meta.get("scan_shape")
    if full_scan_shape is None:
        raise ValueError(
            "scan_shape is required because the HDF5 metadata did not expose a "
            "square scan grid"
        )
    full_scan_shape = tuple(int(v) for v in full_scan_shape)
    order = _normalize_scan_order(scan_order)
    row_start, row_stop, col_start, col_stop = _normalize_scan_region(
        scan_region, full_scan_shape
    )
    patch_h = int(row_stop - row_start)
    patch_w = int(col_stop - col_start)
    frame_indices = _scan_region_frame_indices(
        (row_start, row_stop, col_start, col_stop),
        full_scan_shape,
        order,
    )

    chunk_names = _discover_chunk_names(filepath)
    if not chunk_names:
        with h5py.File(filepath, "r") as f:
            data_group = f.get("entry/data")
            if data_group is not None and "data" in data_group:
                chunk_names = ["data"]
            else:
                raise ValueError(
                    f"{filepath} has no entry/data/data or data_###### detector chunks"
                )

    t_prepare = time.perf_counter()
    prepared = _prepare_master_frames(
        filepath,
        chunk_names,
        frame_indices,
        apply_mask=apply_mask,
    )
    prepare_seconds = time.perf_counter() - t_prepare
    pixel_mask = prepared.get("pixel_mask")
    t_decode = time.perf_counter()
    if resolved_backend == "cuda":
        data = _decompress_prepared(
            prepared,
            verbose=False,
            auto_narrow=auto_narrow,
            det_bin=det_bin,
            streaming_bin=(int(det_bin) > 1),
            output_dtype=output_dtype,
        )
    else:
        from quantem.gpu.io.backends.mps.decoder import load_prepared_frames

        data = load_prepared_frames(
            prepared,
            det_bin=det_bin,
            pixel_mask=pixel_mask if apply_mask else None,
            verbose=False,
            output_dtype=output_dtype,
        )
    decode_seconds = time.perf_counter() - t_decode
    det_r, det_c = (int(data.shape[-2]), int(data.shape[-1]))
    data = data.reshape(patch_h, patch_w, det_r, det_c)
    decoded_detector_shape = (det_r, det_c)
    output_detector_region = (0, det_r, 0, det_c)
    if detector_region is not None:
        output_detector_region = _normalize_detector_region(
            detector_region,
            decoded_detector_shape,
        )
        data = _slice_detector_region(
            data,
            output_detector_region,
            compact=target_scan_region is None,
        )
    source_scan_region = (row_start, row_stop, col_start, col_stop)
    output_scan_region = source_scan_region
    if target_scan_region is not None:
        target_region = _normalize_scan_region(target_scan_region, full_scan_shape)
        t_resample = time.perf_counter()
        data = _resample_scan_crop_cuda(
            data,
            source_scan_region=source_scan_region,
            target_scan_region=target_region,
            scan_shift_row_col=np.asarray(scan_shift_row_col, dtype=np.float32),
            scan_resample_dtype=scan_resample_dtype,
        )
        resample_seconds = time.perf_counter() - t_resample
        output_scan_region = target_region
        row_start, row_stop, col_start, col_stop = output_scan_region
        patch_h = int(row_stop - row_start)
        patch_w = int(col_stop - col_start)
    else:
        resample_seconds = 0.0

    if pixel_mask is not None:
        meta["pixel_mask"] = pixel_mask
    meta["full_scan_shape"] = full_scan_shape
    meta["full_n_frames"] = int(full_scan_shape[0] * full_scan_shape[1])
    meta["scan_shape"] = (patch_h, patch_w)
    meta["scan_order"] = order
    meta["n_frames"] = int(patch_h * patch_w)
    meta["scan_region"] = _scan_region_dict(output_scan_region)
    meta["decoded_detector_shape"] = decoded_detector_shape
    meta["output_detector_shape"] = tuple(int(value) for value in data.shape[-2:])
    meta["detector_region"] = _detector_region_dict(output_detector_region)
    if target_scan_region is not None:
        meta["source_scan_region"] = _scan_region_dict(source_scan_region)
        meta["scan_resampling"] = {
            "mode": "bilinear",
            "target_scan_region": _scan_region_dict(output_scan_region),
            "source_scan_region": _scan_region_dict(source_scan_region),
            "scan_shift_row_col": [
                float(np.asarray(scan_shift_row_col, dtype=np.float32)[0]),
                float(np.asarray(scan_shift_row_col, dtype=np.float32)[1]),
            ],
            "scan_resample_dtype": str(np.dtype(scan_resample_dtype)),
        }
    meta["det_bin"] = int(det_bin)
    meta["backend"] = resolved_backend
    meta["prepare_seconds"] = float(prepare_seconds)
    meta["decode_seconds"] = float(decode_seconds)
    meta["resample_seconds"] = float(resample_seconds)
    meta["load_seconds"] = float(time.perf_counter() - t0)
    meta["total_compressed_bytes"] = int(prepared.get("total_compressed_bytes", 0))
    meta["read_span_count"] = int(prepared.get("read_span_count", 0))
    meta["read_gap_bytes"] = int(prepared.get("read_gap_bytes", 0))
    if "prepare_timing_s" in prepared:
        meta["prepare_timing_s"] = dict(prepared["prepare_timing_s"])
    if verbose:
        size_gb = data.nbytes / 1e9
        print(
            f"  {os.path.basename(filepath)} region "
            f"[{row_start}:{row_stop}, {col_start}:{col_stop}] "
            f"-> {tuple(data.shape)} ({size_gb:.2f} GB) "
            f"in {time.perf_counter() - t0:.2f}s"
        )
    return LoadResult(data, meta)


def _prepare_scan_crop_one(
    filepath: str,
    scan_region: tuple[int, int, int, int] | list[int],
    *,
    scan_shape: tuple[int, int] | None = None,
    scan_order: str = "row-major",
    apply_mask: bool = True,
) -> dict:
    """Prepare one rectangular crop for later serial GPU decode."""
    import time

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    t0 = time.perf_counter()
    t_meta = time.perf_counter()
    meta = get_metadata(filepath)
    meta_seconds = time.perf_counter() - t_meta
    full_scan_shape = scan_shape if scan_shape is not None else meta.get("scan_shape")
    if full_scan_shape is None:
        raise ValueError(
            "scan_shape is required because the HDF5 metadata did not expose a "
            "square scan grid"
        )
    full_scan_shape = tuple(int(v) for v in full_scan_shape)
    order = _normalize_scan_order(scan_order)
    row_start, row_stop, col_start, col_stop = _normalize_scan_region(
        scan_region, full_scan_shape
    )
    patch_h = int(row_stop - row_start)
    patch_w = int(col_stop - col_start)
    frame_indices = _scan_region_frame_indices(
        (row_start, row_stop, col_start, col_stop),
        full_scan_shape,
        order,
    )
    t_discover = time.perf_counter()
    chunk_names = _discover_chunk_names(filepath)
    discover_seconds = time.perf_counter() - t_discover
    if not chunk_names:
        with h5py.File(filepath, "r") as f:
            data_group = f.get("entry/data")
            if data_group is not None and "data" in data_group:
                chunk_names = ["data"]
            else:
                raise ValueError(
                    f"{filepath} has no entry/data/data or data_###### detector chunks"
                )

    prepared_cache_dir = _prepared_scan_crop_cache_dir(
        filepath,
        (row_start, row_stop, col_start, col_stop),
        scan_shape=full_scan_shape,
        scan_order=order,
        apply_mask=apply_mask,
    )
    prepared_cache_status = "disabled"
    prepared_signature = None
    if prepared_cache_dir is not None:
        prepared_cache_status = "miss"
        prepared_signature = _master_frame_source_refs(
            filepath,
            chunk_names,
            apply_mask=apply_mask,
        )[2]
        prepared = _load_prepared_scan_crop_disk_cache(
            prepared_cache_dir,
            signature=prepared_signature,
        )
        if prepared is not None:
            frame_prepare_seconds = time.perf_counter() - t0
            prepared["prepare_timing_s"] = {
                "prepared_cache_hit": float(frame_prepare_seconds),
                "total": float(frame_prepare_seconds),
            }
            return {
                "filepath": filepath,
                "meta": meta,
                "prepared": prepared,
                "full_scan_shape": full_scan_shape,
                "scan_order": order,
                "scan_region_tuple": (row_start, row_stop, col_start, col_stop),
                "patch_shape": (patch_h, patch_w),
                "prepare_seconds": time.perf_counter() - t0,
                "prepare_outer_timing_s": {
                    "metadata": float(meta_seconds),
                    "chunk_discovery": float(discover_seconds),
                    "frame_prepare": float(frame_prepare_seconds),
                },
                "prepared_cache": {
                    "status": "hit",
                    "cache_dir": prepared_cache_dir,
                },
            }

    t_frames = time.perf_counter()
    prepared = _prepare_master_frames(
        filepath,
        chunk_names,
        frame_indices,
        apply_mask=apply_mask,
    )
    frame_prepare_seconds = time.perf_counter() - t_frames
    if prepared_cache_dir is not None and prepared_signature is not None:
        _write_prepared_scan_crop_disk_cache(
            prepared_cache_dir,
            signature=prepared_signature,
            prepared=prepared,
        )
    return {
        "filepath": filepath,
        "meta": meta,
        "prepared": prepared,
        "full_scan_shape": full_scan_shape,
        "scan_order": order,
        "scan_region_tuple": (row_start, row_stop, col_start, col_stop),
        "patch_shape": (patch_h, patch_w),
        "prepare_seconds": time.perf_counter() - t0,
        "prepare_outer_timing_s": {
            "metadata": float(meta_seconds),
            "chunk_discovery": float(discover_seconds),
            "frame_prepare": float(frame_prepare_seconds),
        },
        "prepared_cache": {
            "status": prepared_cache_status,
            "cache_dir": prepared_cache_dir,
        },
    }


def _decode_scan_crop_prepared(
    prepared_item: dict,
    *,
    backend: str,
    det_bin: int,
    apply_mask: bool,
    verbose: bool,
    auto_narrow: bool,
    output_dtype: type | np.dtype | None,
    target_scan_region: tuple[int, int, int, int] | list[int] | None = None,
    scan_shift_row_col=None,
    scan_resample_dtype: type | np.dtype = np.float32,
    detector_region: tuple[int, int, int, int] | list[int] | None = None,
) -> LoadResult:
    """Decode one prepared rectangular crop on the selected backend."""
    import time

    t0 = time.perf_counter()
    filepath = prepared_item["filepath"]
    meta = dict(prepared_item["meta"])
    prepared = prepared_item["prepared"]
    pixel_mask = prepared.get("pixel_mask")
    if backend == "cuda":
        data = _decompress_prepared(
            prepared,
            verbose=False,
            auto_narrow=auto_narrow,
            det_bin=det_bin,
            streaming_bin=(int(det_bin) > 1),
            output_dtype=output_dtype,
        )
    else:
        from quantem.gpu.io.backends.mps.decoder import load_prepared_frames

        data = load_prepared_frames(
            prepared,
            det_bin=det_bin,
            pixel_mask=pixel_mask if apply_mask else None,
            verbose=False,
            output_dtype=output_dtype,
        )
    det_r, det_c = (int(data.shape[-2]), int(data.shape[-1]))
    patch_h, patch_w = (int(v) for v in prepared_item["patch_shape"])
    data = data.reshape(patch_h, patch_w, det_r, det_c)
    decoded_detector_shape = (det_r, det_c)
    output_detector_region = (0, det_r, 0, det_c)
    if detector_region is not None:
        output_detector_region = _normalize_detector_region(
            detector_region,
            decoded_detector_shape,
        )
        data = _slice_detector_region(
            data,
            output_detector_region,
            compact=target_scan_region is None,
        )

    row_start, row_stop, col_start, col_stop = (
        int(v) for v in prepared_item["scan_region_tuple"]
    )
    source_scan_region = (row_start, row_stop, col_start, col_stop)
    output_scan_region = source_scan_region
    if target_scan_region is not None:
        if backend != "cuda":
            raise RuntimeError(
                "target_scan_region= is currently implemented for backend='cuda' "
                "only; MPS needs a separate Metal sampler."
            )
        target_region = _normalize_scan_region(
            target_scan_region,
            tuple(int(v) for v in prepared_item["full_scan_shape"]),
        )
        t_resample = time.perf_counter()
        data = _resample_scan_crop_cuda(
            data,
            source_scan_region=source_scan_region,
            target_scan_region=target_region,
            scan_shift_row_col=np.asarray(scan_shift_row_col, dtype=np.float32),
            scan_resample_dtype=scan_resample_dtype,
        )
        resample_seconds = time.perf_counter() - t_resample
        output_scan_region = target_region
        row_start, row_stop, col_start, col_stop = output_scan_region
        patch_h = int(row_stop - row_start)
        patch_w = int(col_stop - col_start)
    else:
        resample_seconds = 0.0
    if pixel_mask is not None:
        meta["pixel_mask"] = pixel_mask
    meta["full_scan_shape"] = tuple(int(v) for v in prepared_item["full_scan_shape"])
    meta["full_n_frames"] = int(meta["full_scan_shape"][0] * meta["full_scan_shape"][1])
    meta["scan_shape"] = (patch_h, patch_w)
    meta["scan_order"] = prepared_item["scan_order"]
    meta["n_frames"] = int(patch_h * patch_w)
    meta["scan_region"] = _scan_region_dict(output_scan_region)
    meta["decoded_detector_shape"] = decoded_detector_shape
    meta["output_detector_shape"] = tuple(int(value) for value in data.shape[-2:])
    meta["detector_region"] = _detector_region_dict(output_detector_region)
    if target_scan_region is not None:
        shift = np.asarray(scan_shift_row_col, dtype=np.float32)
        meta["source_scan_region"] = _scan_region_dict(source_scan_region)
        meta["scan_resampling"] = {
            "mode": "bilinear",
            "target_scan_region": _scan_region_dict(output_scan_region),
            "source_scan_region": _scan_region_dict(source_scan_region),
            "scan_shift_row_col": [float(shift[0]), float(shift[1])],
            "scan_resample_dtype": str(np.dtype(scan_resample_dtype)),
        }
    meta["det_bin"] = int(det_bin)
    meta["backend"] = backend
    meta["prepare_seconds"] = float(prepared_item["prepare_seconds"])
    if "prepare_outer_timing_s" in prepared_item:
        meta["prepare_outer_timing_s"] = dict(prepared_item["prepare_outer_timing_s"])
    if "prepared_cache" in prepared_item:
        meta["prepared_cache"] = dict(prepared_item["prepared_cache"])
    elif "prepared_cache" in prepared:
        meta["prepared_cache"] = dict(prepared["prepared_cache"])
    meta["decode_seconds"] = float(time.perf_counter() - t0 - resample_seconds)
    meta["resample_seconds"] = float(resample_seconds)
    meta["load_seconds"] = float(
        meta["prepare_seconds"] + meta["decode_seconds"] + meta["resample_seconds"]
    )
    meta["total_compressed_bytes"] = int(prepared.get("total_compressed_bytes", 0))
    meta["read_span_count"] = int(prepared.get("read_span_count", 0))
    meta["read_gap_bytes"] = int(prepared.get("read_gap_bytes", 0))
    if "prepare_timing_s" in prepared:
        meta["prepare_timing_s"] = dict(prepared["prepare_timing_s"])
    if verbose:
        size_gb = data.nbytes / 1e9
        print(
            f"  {os.path.basename(filepath)} region "
            f"[{row_start}:{row_stop}, {col_start}:{col_stop}] "
            f"-> {tuple(data.shape)} ({size_gb:.2f} GB) "
            f"in {float(meta['load_seconds']):.2f}s"
        )
    return LoadResult(data, meta)


def _load_scan_crop_series_impl(
    filepath: list[str] | tuple[str, ...],
    scan_region,
    *,
    backend: str = "cuda",
    scan_shape: tuple[int, int] | None = None,
    scan_order: str = "row-major",
    det_bin: int = 1,
    apply_mask: bool = True,
    verbose: bool = True,
    auto_narrow: bool = True,
    output_dtype: type | np.dtype | None = None,
    stack: bool = True,
    prep_workers: int | None = None,
    target_scan_region: tuple[int, int, int, int] | list[int] | None = None,
    scan_shift_row_col=None,
    scan_resample_dtype: type | np.dtype = np.float32,
    detector_region: tuple[int, int, int, int] | list[int] | None = None,
) -> LoadResult:
    """Load rectangular scan regions from many HDF5 masters.

    ``scan_region`` may be one shared ``(row_start, row_stop, col_start,
    col_stop)`` tuple or a list with one such tuple per master. The per-master
    form is useful for drift-aware time series loading because each frame can
    read only its own source crop before later resampling into a common
    specimen-coordinate ROI.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .backends import resolve_backend

    paths = list(filepath)
    if not paths:
        raise ValueError("filepath list must contain at least one HDF5 master")
    resolved_backend = resolve_backend(backend)
    if target_scan_region is not None and resolved_backend != "cuda":
        raise RuntimeError(
            "target_scan_region= is currently implemented for backend='cuda' "
            "only; MPS needs a separate Metal sampler."
        )
    if (target_scan_region is None) != (scan_shift_row_col is None):
        raise ValueError(
            "target_scan_region= and scan_shift_row_col= must be passed together."
        )
    t0 = time.perf_counter()
    scan_regions, scan_region_mode = _scan_regions_by_file(scan_region, len(paths))
    scan_shifts = (
        _scan_shifts_by_file(scan_shift_row_col, len(paths))
        if target_scan_region is not None
        else None
    )
    worker_count = _normalize_prep_workers(prep_workers, len(paths))
    t_prepare0 = time.perf_counter()
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            prepared_items = list(
                pool.map(
                    lambda item: _prepare_scan_crop_one(
                        os.fspath(item[0]),
                        item[1],
                        scan_shape=scan_shape,
                        scan_order=scan_order,
                        apply_mask=apply_mask,
                    ),
                    zip(paths, scan_regions),
                )
            )
    else:
        prepared_items = [
            _prepare_scan_crop_one(
                os.fspath(path),
                scan_regions[index],
                scan_shape=scan_shape,
                scan_order=scan_order,
                apply_mask=apply_mask,
            )
            for index, path in enumerate(paths)
        ]
    prepare_wall_seconds = time.perf_counter() - t_prepare0
    per_file_meta = []
    arrays = []
    out = None
    t_decode0 = time.perf_counter()
    for index, prepared_item in enumerate(prepared_items):
        result = _decode_scan_crop_prepared(
            prepared_item,
            backend=resolved_backend,
            det_bin=det_bin,
            apply_mask=apply_mask,
            verbose=verbose and len(paths) == 1,
            auto_narrow=auto_narrow,
            output_dtype=output_dtype,
            target_scan_region=target_scan_region,
            scan_shift_row_col=(
                None if scan_shifts is None else scan_shifts[int(index)]
            ),
            scan_resample_dtype=scan_resample_dtype,
            detector_region=detector_region,
        )
        per_file_meta.append(result.metadata)
        decoded = result.data
        if stack:
            if out is None:
                if cp is not None and isinstance(decoded, cp.ndarray):
                    out = cp.empty((len(paths), *decoded.shape), dtype=decoded.dtype)
                else:
                    out = np.empty((len(paths), *decoded.shape), dtype=decoded.dtype)
            if tuple(decoded.shape) != tuple(out.shape[1:]):
                raise ValueError(
                    "Per-file scan-region loads have different output shapes; "
                    "pass stack=False for variable-length batches."
                )
            out[index] = decoded
        else:
            arrays.append(decoded)
        del decoded, result
        if resolved_backend == "cuda" and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    decode_wall_seconds = time.perf_counter() - t_decode0

    data = out if stack else arrays
    first = per_file_meta[0]
    prepare_timing_s: dict[str, float] = {}
    for item in per_file_meta:
        for key, value in (item.get("prepare_timing_s") or {}).items():
            prepare_timing_s[key] = prepare_timing_s.get(key, 0.0) + float(value)
    meta = {
        "backend": resolved_backend,
        "det_bin": int(det_bin),
        "full_scan_shape": first.get("full_scan_shape"),
        "scan_shape": first.get("scan_shape"),
        "scan_order": first.get("scan_order"),
        "scan_region": first.get("scan_region"),
        "source_scan_region": first.get("source_scan_region"),
        "scan_resampling": first.get("scan_resampling"),
        "scan_region_mode": scan_region_mode,
        "scan_regions": [item.get("scan_region") for item in per_file_meta],
        "source_scan_regions": [
            item.get("source_scan_region", item.get("scan_region"))
            for item in per_file_meta
        ],
        "scan_resamplings": [item.get("scan_resampling") for item in per_file_meta],
        "decoded_detector_shape": first.get("decoded_detector_shape"),
        "output_detector_shape": first.get("output_detector_shape"),
        "detector_region": first.get("detector_region"),
        "detector_regions": [item.get("detector_region") for item in per_file_meta],
        "n_files": len(paths),
        "file_paths": [os.fspath(path) for path in paths],
        "file_names": [os.path.basename(os.fspath(path)) for path in paths],
        "n_frames": int(sum(int(item.get("n_frames", 0)) for item in per_file_meta)),
        "prepare_seconds_per_file": [
            float(item.get("prepare_seconds", 0.0)) for item in per_file_meta
        ],
        "decode_seconds_per_file": [
            float(item.get("decode_seconds", 0.0)) for item in per_file_meta
        ],
        "resample_seconds_per_file": [
            float(item.get("resample_seconds", 0.0)) for item in per_file_meta
        ],
        "load_seconds_per_file": [
            float(item.get("load_seconds", 0.0)) for item in per_file_meta
        ],
        "prepare_seconds": float(prepare_wall_seconds),
        "decode_seconds": float(decode_wall_seconds),
        "load_seconds": float(time.perf_counter() - t0),
        "total_compressed_bytes": int(
            sum(int(item.get("total_compressed_bytes", 0)) for item in per_file_meta)
        ),
        "read_span_count": int(
            sum(int(item.get("read_span_count", 0)) for item in per_file_meta)
        ),
        "read_gap_bytes": (
            int(per_file_meta[0].get("read_gap_bytes", 0))
            if per_file_meta
            else 0
        ),
        "prepare_timing_s": prepare_timing_s,
        "prep_workers": int(worker_count),
        "prepared_cache": {
            "hit_count": int(
                sum(
                    1
                    for item in per_file_meta
                    if (item.get("prepared_cache") or {}).get("status") == "hit"
                )
            ),
            "miss_count": int(
                sum(
                    1
                    for item in per_file_meta
                    if (item.get("prepared_cache") or {}).get("status") == "miss"
                )
            ),
            "disabled_count": int(
                sum(
                    1
                    for item in per_file_meta
                    if (item.get("prepared_cache") or {}).get("status") == "disabled"
                )
            ),
        },
        "per_file_metadata": per_file_meta,
    }
    if verbose and len(paths) > 1:
        size_gb = (
            sum(arr.nbytes for arr in data) / 1e9
            if isinstance(data, list)
            else data.nbytes / 1e9
        )
        if scan_region_mode == "shared":
            region = meta["scan_region"]
            region_text = (
                f"[{region['row_start']}:{region['row_stop']}, "
                f"{region['col_start']}:{region['col_stop']}]"
            )
        else:
            region_text = "per-file regions"
        print(
            f"  {len(paths)} masters scan_region {region_text} "
            f"-> {size_gb:.2f} GB in {time.perf_counter() - t0:.2f}s"
        )
    return LoadResult(data, meta)


def _take_requested_scan_order(data, inverse: np.ndarray):
    """Restore stochastic request order after sorted unique HDF5 reads."""
    inverse = np.asarray(inverse, dtype=np.int64)
    if inverse.size == int(data.shape[0]) and np.array_equal(
        inverse,
        np.arange(inverse.size, dtype=np.int64),
    ):
        return data
    if cp is not None and isinstance(data, cp.ndarray):
        return data[cp.asarray(inverse)]
    return data[inverse]


def _normalize_prep_workers(prep_workers: int | None, n_files: int) -> int:
    """Choose a bounded worker count for CPU/HDF5 sparse preparation."""
    n_files = int(n_files)
    if n_files <= 1:
        return 1
    if prep_workers is None:
        return 1
    try:
        workers = int(prep_workers)
    except (TypeError, ValueError) as exc:
        raise TypeError("prep_workers must be a positive integer or None") from exc
    if workers <= 0:
        raise ValueError("prep_workers must be a positive integer or None")
    return max(1, min(workers, n_files))


def _prepare_scan_indices_one(
    filepath: str,
    frame_indices: np.ndarray,
    scan_positions: np.ndarray,
    *,
    apply_mask: bool,
) -> dict:
    """Prepare one stochastic sparse HDF5 batch without GPU decompression."""
    import time

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    t0 = time.perf_counter()
    meta = get_metadata(filepath)
    chunk_names = _discover_chunk_names(filepath)
    if not chunk_names:
        with h5py.File(filepath, "r") as f:
            data_group = f.get("entry/data")
            if data_group is not None and "data" in data_group:
                chunk_names = ["data"]
            else:
                raise ValueError(
                    f"{filepath} has no entry/data/data or data_###### detector chunks"
                )

    unique_frame_indices, inverse = np.unique(frame_indices, return_inverse=True)
    prepared = _prepare_master_frames(
        filepath,
        chunk_names,
        unique_frame_indices,
        apply_mask=apply_mask,
    )
    return {
        "filepath": filepath,
        "meta": meta,
        "prepared": prepared,
        "frame_indices": frame_indices,
        "scan_positions": scan_positions,
        "unique_frame_indices": unique_frame_indices,
        "inverse": inverse,
        "prepare_seconds": time.perf_counter() - t0,
    }


def _decode_scan_indices_prepared(
    prepared_item: dict,
    *,
    backend: str,
    full_scan_shape: tuple[int, int],
    scan_order: ScanOrder,
    det_bin: int,
    apply_mask: bool,
    verbose: bool,
    auto_narrow: bool,
    output_dtype: type | np.dtype | None,
) -> LoadResult:
    """GPU-decompress one prepared stochastic sparse HDF5 batch."""
    import time

    t0 = time.perf_counter()
    filepath = prepared_item["filepath"]
    meta = dict(prepared_item["meta"])
    prepared = prepared_item["prepared"]
    frame_indices = prepared_item["frame_indices"]
    scan_positions = prepared_item["scan_positions"]
    unique_frame_indices = prepared_item["unique_frame_indices"]
    inverse = prepared_item["inverse"]
    pixel_mask = prepared.get("pixel_mask")
    if backend == "cuda":
        data = _decompress_prepared(
            prepared,
            verbose=False,
            auto_narrow=auto_narrow,
            det_bin=det_bin,
            streaming_bin=(int(det_bin) > 1),
            output_dtype=output_dtype,
        )
    else:
        from quantem.gpu.io.backends.mps.decoder import load_prepared_frames

        data = load_prepared_frames(
            prepared,
            det_bin=det_bin,
            pixel_mask=pixel_mask if apply_mask else None,
            verbose=False,
            output_dtype=output_dtype,
        )

    data = _take_requested_scan_order(data, inverse)
    logical_scan_indices = (
        scan_positions[:, 0] * int(full_scan_shape[1]) + scan_positions[:, 1]
    ).astype(np.int64, copy=False)

    if pixel_mask is not None:
        meta["pixel_mask"] = pixel_mask
    meta["full_scan_shape"] = tuple(int(v) for v in full_scan_shape)
    meta["scan_shape"] = None
    meta["scan_order"] = scan_order
    meta["n_frames"] = int(frame_indices.size)
    meta["scan_indices"] = logical_scan_indices.tolist()
    meta["scan_positions"] = scan_positions.astype(np.int64, copy=False).tolist()
    meta["hdf5_frame_indices"] = frame_indices.astype(np.int64, copy=False).tolist()
    meta["unique_hdf5_frame_indices"] = unique_frame_indices.astype(
        np.int64,
        copy=False,
    ).tolist()
    meta["unique_frame_count"] = int(unique_frame_indices.size)
    meta["duplicate_frame_count"] = int(frame_indices.size - unique_frame_indices.size)
    meta["read_order"] = "sorted_unique_hdf5_frame_indices"
    meta["det_bin"] = int(det_bin)
    meta["backend"] = backend
    meta["prepare_seconds"] = float(prepared_item["prepare_seconds"])
    meta["decode_seconds"] = float(time.perf_counter() - t0)
    meta["total_compressed_bytes"] = int(prepared.get("total_compressed_bytes", 0))
    meta["read_span_count"] = int(prepared.get("read_span_count", 0))
    meta["read_gap_bytes"] = int(prepared.get("read_gap_bytes", 0))
    if "prepare_timing_s" in prepared:
        meta["prepare_timing_s"] = dict(prepared["prepare_timing_s"])
    if verbose:
        size_gb = data.nbytes / 1e9
        print(
            f"  {os.path.basename(filepath)} scan_indices "
            f"{frame_indices.size} requested / {unique_frame_indices.size} unique "
            f"-> {tuple(data.shape)} ({size_gb:.2f} GB) "
            f"in {time.perf_counter() - t0:.2f}s"
        )
    return LoadResult(data, meta)


def _load_scan_indices_one(
    filepath: str,
    frame_indices: np.ndarray,
    scan_positions: np.ndarray,
    *,
    backend: str,
    full_scan_shape: tuple[int, int],
    scan_order: ScanOrder,
    det_bin: int,
    apply_mask: bool,
    verbose: bool,
    auto_narrow: bool,
    output_dtype: type | np.dtype | None,
) -> LoadResult:
    """Load one stochastic scan-index batch using sorted unique HDF5 chunks."""
    prepared_item = _prepare_scan_indices_one(
        filepath,
        frame_indices,
        scan_positions,
        apply_mask=apply_mask,
    )
    return _decode_scan_indices_prepared(
        prepared_item,
        backend=backend,
        full_scan_shape=full_scan_shape,
        scan_order=scan_order,
        det_bin=det_bin,
        apply_mask=apply_mask,
        verbose=verbose,
        auto_narrow=auto_narrow,
        output_dtype=output_dtype,
    )


def load_scan_indices(
    filepath: str | list[str] | tuple[str, ...],
    scan_indices,
    *,
    backend: str = "cuda",
    scan_shape: tuple[int, int] | None = None,
    scan_order: str = "row-major",
    index_mode: str = "scan",
    det_bin: int = 1,
    apply_mask: bool = True,
    verbose: bool = True,
    auto_narrow: bool = True,
    output_dtype: type | np.dtype | None = None,
    stack: bool = True,
    prep_workers: int | None = None,
) -> LoadResult:
    """Load stochastic scan positions from one or many raw HDF5 masters.

    This is the ptychography/DataLoader-style sparse IO path. The requested
    stochastic order is the returned order. Internally, each file's requested
    scan positions are converted to flattened HDF5 frame indices, sorted and
    de-duplicated for compressed-chunk reads, GPU-decompressed through the
    CUDA/MPS bitshuffle+LZ4 path, then gathered back to the requested order.

    Parameters
    ----------
    filepath
        One master HDF5 path or a list of masters.
    scan_indices
        For one file, either ``(N,)`` logical row-major scan indices or
        ``(N, 2)`` logical ``(row, col)`` positions. For multiple files, pass a
        common ``(N,)`` / ``(N, 2)`` batch for every file, or per-file arrays
        shaped ``(n_files, N)`` / ``(n_files, N, 2)``.
    index_mode
        ``"scan"`` means flat ``scan_indices`` are logical row-major scan
        positions. ``"hdf5"`` means flat ``scan_indices`` are already physical
        flattened HDF5 detector-frame indices. Row/column arrays are always
        logical scan positions.
    stack
        For multiple files, stack into ``(n_files, N, det_r, det_c)`` when
        shapes match. Use ``stack=False`` to return a list of per-file arrays.
    prep_workers
        Number of worker threads used to read and index compressed HDF5 chunks
        before GPU decompression. Defaults to 1 because many chunked HDF5
        layouts are limited by scattered payload reads and become slower with
        too many concurrent readers. Set this explicitly after benchmarking the
        local storage path.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .backends import resolve_backend

    resolved_backend = resolve_backend(backend)
    if resolved_backend == "cuda" and cp is None:
        raise RuntimeError("load_scan_indices requires CuPy/CUDA")
    if resolved_backend not in {"cuda", "mps"}:
        raise RuntimeError(
            "load_scan_indices supports GPU-decompressed sparse IO on CUDA and "
            f"MPS; backend={resolved_backend!r} was selected."
        )

    paths = list(filepath) if isinstance(filepath, (list, tuple)) else [filepath]
    if not paths:
        raise ValueError("filepath list must contain at least one HDF5 master")
    order = _normalize_scan_order(scan_order)
    if scan_shape is None:
        first_meta = get_metadata(os.fspath(paths[0]))
        scan_shape = first_meta.get("scan_shape")
    if scan_shape is None:
        raise ValueError(
            "scan_shape is required because the HDF5 metadata did not expose a "
            "square scan grid"
        )
    full_scan_shape = tuple(int(v) for v in scan_shape)

    frame_lists, position_lists = _normalize_scan_indices_by_file(
        scan_indices,
        len(paths),
        full_scan_shape,
        order,
        index_mode,
    )

    t0 = time.perf_counter()
    worker_count = _normalize_prep_workers(prep_workers, len(paths))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            prepared_items = list(
                pool.map(
                    lambda args: _prepare_scan_indices_one(*args, apply_mask=apply_mask),
                    [
                        (os.fspath(path), frame_lists[i], position_lists[i])
                        for i, path in enumerate(paths)
                    ],
                )
            )
    else:
        prepared_items = [
            _prepare_scan_indices_one(
                os.fspath(path),
                frame_lists[i],
                position_lists[i],
                apply_mask=apply_mask,
            )
            for i, path in enumerate(paths)
        ]

    per_file_meta = []
    arrays = []
    out = None
    for i, prepared_item in enumerate(prepared_items):
        result = _decode_scan_indices_prepared(
            prepared_item,
            backend=resolved_backend,
            full_scan_shape=full_scan_shape,
            scan_order=order,
            det_bin=det_bin,
            apply_mask=apply_mask,
            verbose=verbose and len(paths) == 1,
            auto_narrow=auto_narrow,
            output_dtype=output_dtype,
        )
        per_file_meta.append(result.metadata)
        decoded = result.data
        if stack and len(paths) > 1:
            if out is None:
                if cp is not None and isinstance(decoded, cp.ndarray):
                    out = cp.empty((len(paths), *decoded.shape), dtype=decoded.dtype)
                else:
                    out = np.empty((len(paths), *decoded.shape), dtype=decoded.dtype)
            if tuple(decoded.shape) != tuple(out.shape[1:]):
                raise ValueError(
                    "Per-file scan-index loads have different output shapes; "
                    "pass stack=False for variable-length batches."
                )
            out[i] = decoded
        else:
            arrays.append(decoded)
        del decoded, result
        if resolved_backend == "cuda" and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    data = out if stack and len(paths) > 1 else arrays[0] if len(paths) == 1 else arrays
    if len(paths) == 1:
        meta = dict(per_file_meta[0])
        meta["n_files"] = 1
        meta["file_paths"] = [os.fspath(paths[0])]
        meta["file_names"] = [os.path.basename(os.fspath(paths[0]))]
        meta["prep_workers"] = int(worker_count)
        return LoadResult(data, meta)

    meta = {
        "backend": resolved_backend,
        "det_bin": int(det_bin),
        "full_scan_shape": full_scan_shape,
        "scan_shape": None,
        "scan_order": order,
        "n_files": len(paths),
        "file_paths": [os.fspath(p) for p in paths],
        "file_names": [os.path.basename(os.fspath(p)) for p in paths],
        "n_frames": int(sum(len(x) for x in frame_lists)),
        "prep_workers": int(worker_count),
        "positions_per_file": [len(x) for x in frame_lists],
        "unique_frame_count_per_file": [
            int(m["unique_frame_count"]) for m in per_file_meta
        ],
        "duplicate_frame_count_per_file": [
            int(m["duplicate_frame_count"]) for m in per_file_meta
        ],
        "prepare_seconds_per_file": [
            float(m["prepare_seconds"]) for m in per_file_meta
        ],
        "decode_seconds_per_file": [
            float(m["decode_seconds"]) for m in per_file_meta
        ],
        "scan_indices_per_file": [m["scan_indices"] for m in per_file_meta],
        "scan_positions_per_file": [m["scan_positions"] for m in per_file_meta],
        "hdf5_frame_indices_per_file": [
            m["hdf5_frame_indices"] for m in per_file_meta
        ],
        "read_order": "sorted_unique_hdf5_frame_indices_per_file",
        "per_file_metadata": per_file_meta,
    }
    if verbose and len(paths) > 1:
        size_gb = (
            sum(arr.nbytes for arr in data) / 1e9
            if isinstance(data, list)
            else data.nbytes / 1e9
        )
        print(
            f"  {len(paths)} masters scan_indices "
            f"{meta['n_frames']} requested -> {size_gb:.2f} GB "
            f"in {time.perf_counter() - t0:.2f}s"
        )
    return LoadResult(data, meta)


def _load(filepath, *args, dtype: str | None = None, gpus=None, stack: bool = True,
         max_concurrent=None, scan_region=None, scan_indices=None,
         random_positions: int | None = None, seed=None,
         replace: bool = False, same_random_positions: bool = False,
         prep_workers: int | None = None, **kwargs):
    """Load 4D-STEM data — one master, or many.

    * ``load(master)`` → one ``LoadResult``.
    * ``load(master, scan_region=(r0, r1, c0, c1))`` → one cropped
      ``LoadResult`` without loading the full scan first.
    * ``load([masters], scan_region=(r0, r1, c0, c1), stack=True)`` → the same
      cropped scan region from each master stacked into one 5D result.
    * ``load([masters], scan_region=[region0, region1, ...], stack=False)`` →
      one cropped scan region per master, useful for drift-aware time series
      loading without a batch-union rectangle.
    * ``load([masters], scan_region=[source0, source1, ...],
      target_scan_region=(r0, r1, c0, c1), scan_shift_row_col=shifts)`` →
      read the per-master source crops and return one shared specimen-coordinate
      specimen-coordinate stack. Subpixel resampling is CUDA-only and returns
      float32 counts.
    * ``load(master, scan_region=(...), detector_region=(dr0, dr1, dc0, dc1))``
      → crop scan and detector evidence in one load call.
    * ``load(master, scan_indices=positions)`` → one stochastic sparse
      ``LoadResult`` in the caller-provided scan-position order.
    * ``load(master, random_positions=1000, seed=42)`` → one stochastic sparse
      ``LoadResult`` after sampling logical scan positions for the caller.
    * ``load([masters])`` → the masters **stacked** into one 5D dataset (the
      series/viewer case).
    * ``load([masters], gpus=[0, 1])`` (or ``stack=False``) → a **list** of separate
      ``LoadResult``, **read in parallel across disks** and **placed across GPUs** —
      the joint-reconstruction path (``gpus``: ``None`` current device / ``int``
      all-that-GPU / ``list`` per-master round-robin). Decode is serial (concurrent
      in-process CUDA decode corrupts the device); reads overlap across disks so
      bandwidth adds.

    Also recommends / applies the smallest lossless browse dtype: ``dtype=None``
    prints the recommendation; ``dtype='u8'`` clips@255 + casts; ``dtype='auto'``
    picks uint8 only if lossless; ``dtype='uint32'`` / ``'u32'`` keeps native
    four-byte unsigned detector counts when the file stores them. ``dtype='u4'``
    requests CUDA packed 4-bit output (two 0..15 counts per byte) and raises if
    any corrected detector count exceeds 15.
    """
    is_seq = isinstance(filepath, (list, tuple))
    verbose = kwargs.get("verbose", True)
    if "region" in kwargs:
        raise TypeError(
            "Use scan_region=(row_start, row_stop, col_start, col_stop), "
            "not region=."
        )
    requested_u8 = _is_uint8_browse_dtype(dtype)
    if requested_u8 and kwargs.get("output_dtype") is None:
        # Route the public browse token through the low-level direct-output
        # path for every loader shape: single master, stacked list,
        # gpus=/stack=False placement, and devices= sharded placement.
        # Without this, list loads can silently materialize uint16 first and
        # only cast later, defeating the U8 memory/speed contract.
        kwargs["output_dtype"] = np.uint8
    requested_u4 = _is_uint4_load_dtype(dtype)
    if requested_u4:
        existing = kwargs.get("output_dtype")
        if existing is not None and not _is_uint4_load_dtype(existing):
            raise ValueError(
                "Pass either dtype='u4' or a different output_dtype=, not both. "
                "dtype='u4' means packed 4-bit counts."
            )
        kwargs["output_dtype"] = "uint4"
    requested_u32 = _is_uint32_load_dtype(dtype)
    if requested_u32 and kwargs.get("output_dtype") is None:
        kwargs["output_dtype"] = np.uint32
    if sum(x is not None for x in (scan_region, scan_indices, random_positions)) > 1:
        raise ValueError(
            "Pass only one of scan_region=, scan_indices=, or random_positions=."
        )
    random_sample_meta = None
    if random_positions is not None:
        if args:
            raise TypeError(
                "load(..., random_positions=...) does not accept positional "
                "dataset_path arguments; sparse scan loading is supported for "
                "4D-STEM master files."
            )
        if str(kwargs.get("index_mode", "scan")).lower().replace("_", "-") != "scan":
            raise ValueError(
                "random_positions generates logical scan indices; do not pass "
                "index_mode='hdf5'. Use scan_indices=... for physical HDF5 frame indices."
            )
        random_scan_shape = kwargs.get("scan_shape")
        if random_scan_shape is None:
            first_path = filepath[0] if isinstance(filepath, (list, tuple)) else filepath
            random_scan_shape = get_metadata(os.fspath(first_path)).get("scan_shape")
        if random_scan_shape is None:
            raise ValueError(
                "scan_shape is required for random_positions because the HDF5 "
                "metadata did not expose a square scan grid"
            )
        random_scan_shape = tuple(int(v) for v in random_scan_shape)
        n_files = len(filepath) if isinstance(filepath, (list, tuple)) else None
        scan_indices = random_scan_indices(
            random_positions,
            random_scan_shape,
            n_files=n_files,
            seed=seed,
            replace=replace,
            same_for_all_files=same_random_positions,
        )
        random_sample_meta = {
            "mode": "random_positions",
            "positions_per_file": int(_normalize_random_position_count(random_positions)),
            "scan_shape": [int(v) for v in random_scan_shape],
            "seed": int(seed) if isinstance(seed, (int, np.integer)) else None,
            "replace": bool(replace),
            "same_random_positions": bool(same_random_positions),
            "n_files": int(n_files) if n_files is not None else 1,
            "index_space": "logical_row_major_scan",
        }
    if scan_indices is not None:
        if random_sample_meta is None and seed is not None:
            raise ValueError("seed= only applies with random_positions=.")
        if args:
            raise TypeError(
                "load(..., scan_indices=...) does not accept positional "
                "dataset_path arguments; sparse scan loading is supported for "
                "4D-STEM master files."
            )
        if gpus is not None:
            raise ValueError(
                "load(..., scan_indices=...) does not accept gpus=. Use "
                "CUDA_VISIBLE_DEVICES for the current single-GPU sparse path."
            )
        backend = kwargs.pop("backend", "auto")
        from .backends import resolve_backend

        resolved_backend = resolve_backend(backend)
        if resolved_backend not in {"cuda", "mps"}:
            raise RuntimeError(
                "load(..., scan_indices=...) supports GPU-decompressed sparse "
                f"IO on CUDA and MPS; backend={resolved_backend!r} was selected."
            )
        allowed = {
            "scan_shape",
            "scan_order",
            "index_mode",
            "det_bin",
            "apply_mask",
            "verbose",
            "auto_narrow",
            "output_dtype",
        }
        extra = sorted(set(kwargs) - allowed)
        if extra:
            raise TypeError(
                "load(..., scan_indices=...) does not accept "
                + ", ".join(f"{name}=" for name in extra)
            )
        result = load_scan_indices(
            filepath,
            scan_indices,
            backend=resolved_backend,
            scan_shape=kwargs.pop("scan_shape", None),
            scan_order=kwargs.pop("scan_order", "row-major"),
            index_mode=kwargs.pop("index_mode", "scan"),
            det_bin=kwargs.pop("det_bin", 1),
            apply_mask=kwargs.pop("apply_mask", True),
            verbose=kwargs.pop("verbose", True),
            auto_narrow=kwargs.pop("auto_narrow", True),
            output_dtype=kwargs.pop("output_dtype", None),
            stack=stack,
            prep_workers=prep_workers,
        )
        if random_sample_meta is not None:
            result.metadata["sample"] = random_sample_meta
        return result
    if scan_region is not None:
        if args:
            raise TypeError(
                "load(..., scan_region=...) does not accept positional "
                "dataset_path arguments; crop-first loading is supported for "
                "4D-STEM master files."
            )
        if gpus is not None:
            raise ValueError(
                "load(..., scan_region=...) does not accept gpus=. Use "
                "CUDA_VISIBLE_DEVICES for the current crop-first path."
            )
        backend = kwargs.pop("backend", "auto")
        from .backends import resolve_backend

        resolved_backend = resolve_backend(backend)
        if resolved_backend not in {"cuda", "mps"}:
            raise RuntimeError(
                "load(..., scan_region=...) supports accelerated crop-first IO "
                f"on CUDA and MPS; backend={resolved_backend!r} was selected."
            )
        allowed = {
            "scan_shape",
            "scan_order",
            "det_bin",
            "apply_mask",
            "verbose",
            "auto_narrow",
            "output_dtype",
            "target_scan_region",
            "scan_shift_row_col",
            "scan_resample_dtype",
            "detector_region",
        }
        extra = sorted(set(kwargs) - allowed)
        if extra:
            raise TypeError(
                "load(..., scan_region=...) does not accept "
                + ", ".join(f"{name}=" for name in extra)
            )
        region_scan_shape = kwargs.pop("scan_shape", None)
        region_scan_order = kwargs.pop("scan_order", "row-major")
        region_det_bin = kwargs.pop("det_bin", 1)
        region_apply_mask = kwargs.pop("apply_mask", True)
        region_verbose = kwargs.pop("verbose", True)
        region_auto_narrow = kwargs.pop("auto_narrow", True)
        region_output_dtype = kwargs.pop("output_dtype", None)
        region_target_scan_region = kwargs.pop("target_scan_region", None)
        region_scan_shift_row_col = kwargs.pop("scan_shift_row_col", None)
        region_scan_resample_dtype = kwargs.pop("scan_resample_dtype", np.float32)
        region_detector_region = kwargs.pop("detector_region", None)
        if region_detector_region is not None and _is_uint4_load_dtype(
            region_output_dtype
        ):
            raise ValueError(
                "detector_region= is not supported with dtype='u4' because "
                "uint4 output packs detector columns into bytes. Use uint8/uint16 "
                "for detector-region loading."
            )
        if (region_target_scan_region is None) != (
            region_scan_shift_row_col is None
        ):
            raise ValueError(
                "target_scan_region= and scan_shift_row_col= must be passed together."
            )
        if region_target_scan_region is not None and resolved_backend != "cuda":
            raise RuntimeError(
                "target_scan_region= is currently implemented for backend='cuda' "
                "only; MPS needs a separate Metal sampler."
            )
        if is_seq:
            return _load_scan_crop_series_impl(
                list(filepath),
                scan_region,
                backend=resolved_backend,
                scan_shape=region_scan_shape,
                scan_order=region_scan_order,
                det_bin=region_det_bin,
                apply_mask=region_apply_mask,
                verbose=region_verbose,
                auto_narrow=region_auto_narrow,
                output_dtype=region_output_dtype,
                stack=stack,
                prep_workers=prep_workers,
                target_scan_region=region_target_scan_region,
                scan_shift_row_col=region_scan_shift_row_col,
                scan_resample_dtype=region_scan_resample_dtype,
                detector_region=region_detector_region,
            )
        if not stack:
            raise ValueError(
                "load(..., scan_region=...) with one master returns one cropped "
                "LoadResult; stack=False is only meaningful for a list of masters."
            )
        if region_scan_shape is not None:
            full_scan_shape = tuple(int(v) for v in region_scan_shape)
            normalized_region = _normalize_scan_region(scan_region, full_scan_shape)
            if (
                region_target_scan_region is None
                and region_detector_region is None
                and normalized_region == (
                    0,
                    int(full_scan_shape[0]),
                    0,
                    int(full_scan_shape[1]),
                )
            ):
                return _load_impl(
                    filepath,
                    scan_shape=full_scan_shape,
                    scan_order=region_scan_order,
                    det_bin=region_det_bin,
                    apply_mask=region_apply_mask,
                    verbose=region_verbose,
                    auto_narrow=region_auto_narrow,
                    output_dtype=region_output_dtype,
                    backend=resolved_backend,
                )
        return _load_scan_crop_impl(
            filepath,
            scan_region,
            backend=resolved_backend,
            scan_shape=region_scan_shape,
            scan_order=region_scan_order,
            det_bin=region_det_bin,
            apply_mask=region_apply_mask,
            verbose=region_verbose,
            auto_narrow=region_auto_narrow,
            output_dtype=region_output_dtype,
            target_scan_region=region_target_scan_region,
            scan_shift_row_col=region_scan_shift_row_col,
            scan_resample_dtype=region_scan_resample_dtype,
            detector_region=region_detector_region,
        )
    if is_seq and (gpus is not None or not stack):
        # N separate GPU-placed datasets (parallel read, serial decode).
        result = _load_many_parallel(list(filepath), gpus=gpus, max_concurrent=max_concurrent,
                                     verbose=kwargs.pop("verbose", False), **kwargs)
        if requested_u8 and verbose:
            print("  Loaded in uint8 for browsing - values >255 saturate to 255. "
                  "Reconstruction uses raw uint16.")
        return result
    if requested_u8:
        # decode-DIRECT to uint8 whenever the backend supports output_dtype:
        # the batched decoder clips@255 into a uint8 output, so the full uint16
        # block is never materialized (peak ~ uint8 out + one batch + scratch,
        # not uint16+uint8). The browse path is explicit and count-clipping is
        # user-visible.
        result = _load_impl(filepath, *args, **kwargs)
        d = getattr(result, "data", None)
        if verbose and d is not None and hasattr(d, "nbytes"):
            print(f"  Loaded in uint8 for browsing - using {d.nbytes/1e9:.1f} GB, half of uint16 "
                  f"(decoded straight to uint8, values >255 saturate to 255, so peak memory stayed low). "
                  f"Reconstruction uses raw uint16.")
        return result
    result = _load_impl(filepath, *args, **kwargs)
    data = getattr(result, "data", None)
    if (data is not None and hasattr(data, "max") and hasattr(data, "dtype")
            and getattr(data, "ndim", 0) >= 3):
        new = _browse_dtype_advise_and_cast(data, dtype, verbose)
        if new is not data:
            result = LoadResult(new, result.metadata)
    return result


def load(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    dtype: str | type | np.dtype | None = None,
    backend: str = "auto",
    dataset_path: str | None = None,
    scan_shape: tuple[int, int] | None = None,
    scan_order: ScanOrder = "row-major",
    scan_region: (
        tuple[int, int, int, int]
        | Sequence[tuple[int, int, int, int]]
        | None
    ) = None,
    detector_region: tuple[int, int, int, int] | None = None,
    target_scan_region: tuple[int, int, int, int] | None = None,
    scan_shift_row_col: (
        tuple[float, float]
        | Sequence[tuple[float, float]]
        | np.ndarray
        | None
    ) = None,
    scan_resample_dtype: type | np.dtype = np.float32,
    scan_indices: Sequence[int] | Sequence[tuple[int, int]] | np.ndarray | None = None,
    index_mode: str = "scan",
    random_positions: int | None = None,
    seed: int | None = None,
    replace: bool = False,
    same_random_positions: bool = False,
    drift: Sequence | np.ndarray | None = None,
    det_bin: int = 1,
    apply_mask: bool = True,
    auto_narrow: bool = True,
    output: str = "native",
    stack: bool = True,
    device: int | str | None = None,
    devices: list[int] | str | None = None,
    verbose: bool = True,
) -> LoadResult | list[LoadResult]:
    """Load one or more 4D-STEM sources through an accelerated backend.

    All spatial arguments use ``(row, col)`` order. ``dtype`` is the only
    output-precision control. ``"u8"`` requests a saturating browse output;
    it is lossless only when a complete source audit proves every corrected
    count is at most 255. Use ``"u16"`` or the native dtype for exact
    raw-count workflows, widening detector sums when required.
    ``backend="auto"`` selects CUDA or MPS and never selects CPU silently.
    ``output="native"`` preserves the backend-native payload; use
    ``output="torch"`` when the consumer expects a Torch tensor.

    Parameters
    ----------
    source
        One master/data HDF5 path, a folder, or a list of master paths.
    dtype
        Requested output dtype, such as ``"u8"``, ``"u16"``, ``"u32"``,
        ``"f32"``, ``"u4"``, ``"native"``, or ``"auto"``. Explicit
        ``"u8"`` saturates values above 255; ``"auto"`` is an advisory
        compact-dtype choice and is not a complete-source losslessness audit.
    backend
        ``"auto"``, ``"cuda"``, ``"mps"``, or explicit reference ``"cpu"``.
    output
        ``"native"`` (default) returns the backend-native array or chunked
        source. ``"torch"`` converts the loaded result to a Torch tensor,
        assembling chunked MPS data when necessary.
    scan_shape
        Optional full scan shape as ``(row, col)``.
    scan_region, detector_region
        Optional bounds as ``(row_start, row_stop, col_start, col_stop)``.
    target_scan_region, scan_shift_row_col
        Shared target crop and per-source row/column shifts for drift-aware
        multi-file loading.
    drift
        Optional dense per-source additive drift fields for a sparse
        ptychography batch, shaped ``(n_files, scan_rows, scan_cols, 2)`` or
        ``(n_files, 2, scan_rows, scan_cols)``. The loader keeps raw
        diffraction patterns unchanged and records corrected floating-point
        probe positions in ``result.metadata["drift_batch"]``. Use with
        ``random_positions=`` or ``scan_indices=``; this is intentionally
        separate from scan-space resampling.
    devices
        CUDA devices for a multi-GPU load. With ``stack=True`` the result may
        be sharded by device; with ``stack=False`` each source is returned
        separately.

    Returns
    -------
    LoadResult or list[LoadResult]
        Data stays backend-resident. MPS list/folder loads return a common
        multi-frame detector object in ``LoadResult.data`` while background
        decoding fills its dataset slots.
    """
    if output not in {"native", "torch"}:
        raise ValueError("output must be 'native' or 'torch'")

    token = dtype.lower() if isinstance(dtype, str) else None
    output_dtype = None
    dtype_aliases = {
        "u16": np.dtype(np.uint16),
        "uint16": np.dtype(np.uint16),
        "f32": np.dtype(np.float32),
        "float32": np.dtype(np.float32),
    }
    if token in dtype_aliases:
        output_dtype = dtype_aliases[token]
    elif dtype is not None and token not in {
        "auto",
        "native",
        "u8",
        "uint8",
        "u32",
        "uint32",
        "u4",
        "uint4",
    }:
        output_dtype = np.dtype(dtype)

    if detector_region is not None and scan_region is None:
        raise ValueError("detector_region= requires scan_region=.")
    if drift is not None and scan_region is not None:
        raise ValueError(
            "drift= belongs to sparse ptychography batches; use "
            "scan_shift_row_col= with scan_region= for scan-space resampling."
        )
    generated_sample = None
    drift_batch = None
    if drift is not None:
        if scan_indices is None and random_positions is None:
            raise ValueError(
                "drift= requires random_positions= or scan_indices=."
            )
        n_files = len(source) if isinstance(source, (list, tuple)) else 1
        if (
            n_files > 1
            and not same_random_positions
            and random_positions is not None
        ):
            raise ValueError(
                "drift= with multiple sources requires "
                "same_random_positions=True."
            )
        resolved_scan_shape = scan_shape
        if resolved_scan_shape is None:
            first_source = source[0] if isinstance(source, (list, tuple)) else source
            resolved_scan_shape = get_metadata(os.fspath(first_source)).get(
                "scan_shape"
            )
        if resolved_scan_shape is None:
            raise ValueError(
                "scan_shape= is required for drift= when source metadata "
                "does not expose a square scan grid."
            )
        resolved_scan_shape = tuple(int(v) for v in resolved_scan_shape)
        raw_drift = drift
        if hasattr(raw_drift, "detach"):
            raw_drift = raw_drift.detach().cpu().numpy()
        drift_array = np.asarray(raw_drift, dtype=np.float32)
        if drift_array.ndim == 3 and drift_array.shape[-1] == 2:
            drift_array = drift_array[None, ...]
        elif drift_array.ndim == 4 and drift_array.shape[1] == 2:
            drift_array = np.moveaxis(drift_array, 1, -1)
        if drift_array.ndim != 4 or drift_array.shape[-1] != 2:
            raise ValueError(
                "drift must have shape (frames, rows, cols, 2) or "
                "(frames, 2, rows, cols)"
            )
        if drift_array.shape[0] != n_files:
            raise ValueError("drift must contain one field per source frame")
        if tuple(drift_array.shape[1:3]) != resolved_scan_shape:
            raise ValueError(
                f"drift scan shape {tuple(drift_array.shape[1:3])} "
                f"does not match scan_shape={resolved_scan_shape}"
            )
        if not np.isfinite(drift_array).all():
            raise ValueError("drift must contain finite values")
        if random_positions is not None:
            scan_indices = random_scan_indices(
                random_positions,
                resolved_scan_shape,
                seed=seed,
                replace=replace,
                same_for_all_files=True,
            )
            generated_sample = {
                "mode": "random_positions",
                "positions_per_file": int(
                    _normalize_random_position_count(random_positions)
                ),
                "scan_shape": [int(v) for v in resolved_scan_shape],
                "seed": int(seed) if isinstance(seed, (int, np.integer)) else None,
                "replace": bool(replace),
                "same_random_positions": True,
                "n_files": int(n_files),
                "index_space": "logical_row_major_scan",
                "scan_indices": np.asarray(scan_indices).tolist(),
            }
            random_positions = None
        _, position_lists = _normalize_scan_indices_by_file(
            scan_indices,
            n_files,
            resolved_scan_shape,
            scan_order,
            index_mode,
        )
        nominal_positions = np.stack(position_lists, axis=0)
        corrected_positions = _drift_scan_positions(
            nominal_positions,
            drift_array,
            scan_shape=resolved_scan_shape,
        )
        drift_batch = {
            "field_shape": list(drift_array.shape),
            "nominal_positions": nominal_positions.tolist(),
            "corrected_positions": corrected_positions.tolist(),
            "position_space": "logical_scan_pixels",
            "mapping": "nominal_plus_dense_drift_field",
            "patterns_resampled": False,
        }
    if (
        target_scan_region is not None or scan_shift_row_col is not None
    ) and scan_region is None:
        raise ValueError(
            "target_scan_region= and scan_shift_row_col= require scan_region=."
        )

    common = {
        "backend": backend,
        "scan_shape": scan_shape,
        "scan_order": scan_order,
        "det_bin": det_bin,
        "apply_mask": apply_mask,
        "auto_narrow": auto_narrow,
        "verbose": verbose,
    }
    if output_dtype is not None:
        common["output_dtype"] = output_dtype
    if dataset_path is not None:
        if scan_region is not None or scan_indices is not None or random_positions is not None:
            raise ValueError(
                "dataset_path= is only supported for complete-source loads."
            )
        common["dataset_path"] = dataset_path

    if scan_region is not None:
        common.update(
            detector_region=detector_region,
            target_scan_region=target_scan_region,
            scan_shift_row_col=scan_shift_row_col,
            scan_resample_dtype=scan_resample_dtype,
        )
    elif scan_indices is not None or random_positions is not None:
        common["index_mode"] = index_mode
    else:
        common.update(device=device, devices=devices)

    result = _load(
        source,
        dtype=dtype if isinstance(dtype, str) else None,
        gpus=devices if not stack else None,
        stack=stack,
        scan_region=scan_region,
        scan_indices=scan_indices,
        random_positions=random_positions,
        seed=None if generated_sample is not None else seed,
        replace=replace,
        same_random_positions=same_random_positions,
        prep_workers=None,
        **common,
    )
    if drift_batch is not None:
        result.metadata["drift_batch"] = drift_batch
        if generated_sample is not None:
            result.metadata["sample"] = generated_sample
    return _convert_load_output(result, output)


def disk_of(path) -> str:
    """The physical disk a path lives on, e.g. ``"nvme1n1"``.

    Two paths with the same ``disk_of`` share one drive (loading them in parallel
    gives no gain — the disk is the floor); different ``disk_of`` = independent
    disks whose read bandwidth adds during internal parallel loading. Resolves the file's
    ``st_dev`` to its block device via ``/sys/dev/block`` and strips the partition.
    Returns ``"?"`` if the path is unreadable.
    """
    import os as _os
    try:
        st = _os.stat(str(path)).st_dev
        real = _os.path.realpath(f"/sys/dev/block/{_os.major(st)}:{_os.minor(st)}")
        parent = _os.path.basename(_os.path.dirname(real))  # partition -> whole disk
        return parent if parent and parent != "block" else _os.path.basename(real)
    except OSError:
        return "?"


def _disk_interleaved_indices(paths) -> list[int]:
    """Return indices ordered to touch different physical disks early.

    This is intentionally small and deterministic so tests can validate the
    scheduling policy without real disks or CUDA. With paths grouped by disk,
    ``[d0a, d0b, d1a, d1b]`` becomes ``[d0a, d1a, d0b, d1b]``; with one disk,
    the original order is preserved.
    """
    buckets: dict[str, list[int]] = {}
    disk_order: list[str] = []
    for idx, path in enumerate(paths):
        disk = disk_of(path)
        if disk not in buckets:
            disk_order.append(disk)
            buckets[disk] = []
        buckets[disk].append(idx)

    queues = [list(buckets[disk]) for disk in disk_order]
    order: list[int] = []
    while any(queues):
        for queue_ in queues:
            if queue_:
                order.append(queue_.pop(0))
    return order


def _assign_indices_to_devices(filepaths, devices) -> dict[int, list[int]]:
    """Assign file indices to devices using disk-interleaved round robin."""
    devices = [int(device) for device in devices]
    if not devices:
        raise ValueError("devices must contain at least one CUDA device")

    assign: dict[int, list[int]] = {device: [] for device in devices}
    for offset, idx in enumerate(_disk_interleaved_indices(filepaths)):
        assign[devices[offset % len(devices)]].append(idx)
    return assign


def _load_many_parallel(masters, *, gpus=None, max_concurrent=None, verbose=False, **load_kwargs):
    """Load many masters with concurrent READS + SERIAL GPU decode, placing each
    master on a chosen GPU. The data-feeding path for joint reconstruction.

    Reached via ``load([masters], devices=..., stack=False)``. See :func:`load`.

    A producer pool reads + header-parses masters concurrently (host/IO only, the
    GIL is released during ``os.readv``); a single consumer decodes them one at a
    time on the GPU. This gives two speedups, both safe:

    * **cross-disk** — readers on different physical disks overlap, bandwidth ADDS.
    * **same-disk** — master N+1 is read *while* master N decodes (read-ahead), so
      the decode hides under the next read.

    GPU decode is SERIAL on purpose: concurrent in-process CUDA decode shares the
    CuPy kernel/plan caches + pinned pool and raises ``cudaErrorIllegalAddress``.
    Serial decode is safe on ANY GPU (it is just a normal single decode) and loses
    no throughput, because one disk feeds slower than one GPU decodes.

    Parameters
    ----------
    masters : list[str]
        Master ``.h5`` paths.
    gpus : None | int | list[int]
        Which GPU each master decodes onto. ``None`` = the current device; an ``int``
        = all masters to that GPU; a ``list`` = per-master (round-robin if shorter),
        e.g. ``gpus=[0, 1]`` places alternating masters on GPU 0 and GPU 1 — the
        placement a multi-GPU joint solver wants. (Placement is serial, never
        concurrent, so it is safe even on a GPU that is also serving the dashboard.)
    max_concurrent : int, optional
        Parallel READERS. Default = max(2, #distinct disks) so there is always a
        read in flight to overlap the decode.

    Returns
    -------
    list[LoadResult]  — one per input master, in input order; each ``.data`` lives
    on its assigned GPU.
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext

    masters = list(masters)
    n = len(masters)
    if gpus is None:
        dev = [None] * n
    elif isinstance(gpus, int):
        dev = [int(gpus)] * n
    else:
        gl = [int(g) for g in gpus]
        if not gl:
            raise ValueError("gpus must contain at least one CUDA device")
        dev = [gl[i % len(gl)] for i in range(n)]

    import cupy as cp

    disks = [disk_of(m) for m in masters]
    n_disks = len({d for d in disks if d != "?"})
    n_read = int(max_concurrent) if max_concurrent else max(1, n_disks)
    n_read = max(n_read, 2) if n > 1 else 1  # >=2 so a read is always queued ahead

    # Disk-interleaved read order: hit different disks first for peak parallel BW.
    order = _disk_interleaved_indices(masters)

    # Producer: concurrent read+prepare (host) into a bounded queue. The pinned-
    # buffer pool is lock-guarded, so concurrent reads are thread-safe; the queue
    # bound caps in-flight host buffers.
    q: queue.Queue = queue.Queue(maxsize=n_read + 1)
    _SENT = object()
    cancelled = threading.Event()

    def _put(item) -> bool:
        while not cancelled.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _release_queued(item) -> None:
        if item is _SENT:
            return
        kind, _index, payload = item
        if kind == "prepared":
            _release_pinned(payload["read_buffer"])

    def _producer():
        def _prep(i):
            try:
                prepared = _prepare_master(
                    masters[i],
                    _discover_chunk_names(masters[i]),
                    True,
                )
            except Exception as exc:  # noqa: BLE001 - transport worker failures
                if _put(("error", i, exc)):
                    cancelled.set()
                return
            if not _put(("prepared", i, prepared)):
                _release_pinned(prepared["read_buffer"])

        try:
            with ThreadPoolExecutor(max_workers=n_read) as pool:
                list(pool.map(_prep, order))
        finally:
            if not cancelled.is_set():
                _put(_SENT)

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    # Consumer: serial decode, each master onto its assigned GPU.
    decode_kw = {k: load_kwargs[k] for k in ("output_dtype", "det_bin", "auto_narrow")
                 if k in load_kwargs}
    scan_order = load_kwargs.get("scan_order", "row-major")
    if decode_kw.get("det_bin", 1) > 1:
        decode_kw["streaming_bin"] = True
    results: list = [None] * n
    try:
        while True:
            item = q.get()
            if item is _SENT:
                break
            kind, i, payload = item
            if kind == "error":
                raise payload
            prepared = payload
            d = dev[i]
            with (cp.cuda.Device(d) if d is not None else nullcontext()):
                data = _decompress_prepared(prepared, **decode_kw)
                meta = get_metadata(masters[i])
                nf = int(data.shape[0])
                side = int(nf ** 0.5)
                if data.ndim == 3 and side * side == nf:
                    meta.setdefault("scan_shape", (side, side))
                data = _apply_scan_shape(
                    data,
                    load_kwargs.get("scan_shape"),
                    meta,
                    scan_order,
                )
                meta["scan_order"] = _normalize_scan_order(scan_order)
            results[i] = LoadResult(data, meta)
    except BaseException:
        cancelled.set()
        while producer_thread.is_alive() or not q.empty():
            try:
                pending = q.get(timeout=0.1)
            except queue.Empty:
                continue
            _release_queued(pending)
        producer_thread.join()
        raise
    producer_thread.join()
    return results


def _load_impl(
    filepath: str | list[str],
    dataset_path: str | None = None,
    apply_mask: bool = True,
    scan_shape: tuple[int, int] | None = None,
    scan_order: str = "row-major",
    det_bin: int = 1,
    verbose: bool = True,
    auto_narrow: bool = True,
    output_dtype: type | np.dtype | None = None,
    device: int | str | None = None,
    devices: list[int] | str | None = None,
    backend: str = "auto",
    row_prefix: bool = False,
    precompute_detector_sum: bool = False,
    skip_mps_memory_check: bool | None = None,
) -> LoadResult:
    """Load bitshuffle+LZ4 compressed HDF5 data directly to GPU.

    Automatically detects file format:
    - Master files (*_master.h5): Auto-discovers data chunks (data_000001, etc.)
    - Single data files: Uses entry/data/data or specified dataset_path

    When a list of file paths is provided, loads each file sequentially and
    stacks the results into a single array with an extra leading dimension.
    The scan dimension is unflattened automatically from metadata so the
    result is ready for ``Show4DSTEM`` (e.g. 5 files → ``(5, 256, 256, 96, 96)``).

    Parameters
    ----------
    filepath : str or list[str]
        Path to the HDF5 file, or a list of paths to load and stack.
    dataset_path : str, optional
        Path to dataset within HDF5 file. If None, auto-detects.
    apply_mask : bool, optional
        Apply pixel mask to zero out bad pixels (master files only), by default True.
    scan_shape : tuple[int, int], optional
        Scan grid shape ``(scan_row, scan_col)``. By default this is
        **auto-derived** from the h5 ``ntrigger`` field assuming a square
        scan, so users rarely need to pass it. Pass explicitly for
        non-square scans, or to override the derived value.
        When provided (or derived), the scan dimension is unflattened:
        ``(N, det_r, det_c)`` → ``(scan_r, scan_c, det_r, det_c)``; for
        multi-file loads: ``(n_files, scan_r, scan_c, det_r, det_c)``.
    scan_order : {"row-major", "serpentine"}, optional
        Ordering of flattened scan frames before unflattening. Serpentine
        acquisitions store odd scan rows right-to-left; the loader corrects
        them so returned arrays are always indexed as normal ``(row, col)``.
    det_bin : int, optional
        Detector binning factor (default 1 = no binning). Applied immediately
        after loading each file, before copying into the output array. Reduces
        VRAM by ``det_bin**2`` (e.g. ``det_bin=2`` quarters detector pixels).
    verbose : bool, optional
        Print progress information (default True).
    auto_narrow : bool, optional
        For master files with uint32 data, cast the final array down to
        uint16 when every observed value fits (< 65536). Arina's uint32
        output is almost always over-allocated in 4D-STEM (actual counts
        rarely exceed a few thousand), so this halves the returned
        array's memory for free. Raises ``ValueError`` if the data
        genuinely contains a value >= 65536 - caller should retry with
        ``auto_narrow=False`` in that case. Default True.
    output_dtype : dtype, optional
        Cast the returned GPU array during load. This is useful for corrected
        4D-STEM archives saved as ``float32``: callers can request
        ``output_dtype=np.float16`` and/or ``det_bin=2`` to work with a much
        smaller GPU array while keeping the on-disk archive high precision.
    skip_mps_memory_check : bool, optional
        Override the Apple Silicon MPS memory guard. By default, MPS loads use
        HDF5 metadata to estimate the unified-memory footprint before allocating
        Metal buffers and refuse no-bin/large loads that can freeze a laptop.
        Prefer ``det_bin=2`` or ``det_bin=4`` for browsing; set this only when
        you intentionally want to force the risky allocation.
    device : int or str, optional
        Pin every allocation of a single-target load to this GPU
        (``device=1`` or ``"cuda:1"``). Default None = current device.
    devices : list[int], optional
        **Sharded multi-GPU load** (lists of files only). Split the files
        across these GPUs in disk-interleaved order; each card decompresses +
        keeps its own subset, with NO gather to one card. This is how a stack
        larger than a single card fits - e.g. ``load(six_512_masters,
        devices=[0, 1])`` holds 108 GiB (6 x 512 x 512 x 192 x 192 no-bin)
        across two 96 GB cards. The result's ``.data`` is a ``{device: array}``
        dict (not one array), and ``metadata["device_map"]`` records which file
        landed on which GPU. Sharding is primarily for capacity, and it also
        unlocks load-speed wins when the masters are spread across independent
        disks because GPU workers no longer all hammer the same drive first.

    Returns
    -------
    LoadResult
        Named tuple with ``data`` (cupy.ndarray) and ``metadata`` (dict).
        See :class:`LoadResult` for the full metadata field list, including
        the derived fields (``scan_shape``, ``n_frames``, ``dwell_time_us``,
        ``detector_shape``, ``detector_name``, ``saturation``). Can be
        unpacked: ``data, meta = load(path)``.

    Examples
    --------
    >>> from quantem.gpu.io import load
    >>> # scan_shape auto-derived from h5 metadata - no need to type it
    >>> data, meta = load('scan_master.h5')
    >>> data.shape
    (512, 512, 192, 192)
    >>> meta['dwell_time_us']
    99.6

    >>> # Multiple files, scan shape still auto-derived
    >>> data, meta = load(masters[:5])
    >>> data.shape
    (5, 256, 256, 192, 192)

    >>> # Override for non-square scans
    >>> data, meta = load('rectangular_master.h5', scan_shape=(128, 256))

    >>> # Load a float32 corrected archive as a smaller working array
    >>> data, meta = load('corrected_master.h5', det_bin=2, output_dtype=np.float16)

    Performance
    -----------
    Two regimes, because the page cache changes everything:

    - **Cold** (first load of a dataset, bytes not in RAM): disk-bound. The
      compressed bytes must come off the NVMe, and that read dominates - the
      GPU decompress runs hidden in its shadow. Wall time scales with the
      *compressed* size, NOT ``det_bin`` (binning happens after decompress, so
      the same bytes are read either way; ``det_bin`` only shrinks the output
      array / VRAM).
    - **Warm** (data already in RAM cache from a prior load): the read is
      served from RAM at ~5x the NVMe rate, exposing the GPU phase. ~2-3x
      faster than cold. Requires the working set to fit free RAM, else the
      cache churns and warm degrades toward cold.

    Measured 2026-05-24 on real Arina data (512²/1024² scan, 192² detector,
    one RTX PRO 6000, data on a WD_BLACK SN850X). COLD = cache evicted,
    WARM = min of 3 with cache hot:

    ====================================  ========  ========  ==========
    case                                  COLD (s)  WARM (s)  peak VRAM
    ====================================  ========  ========  ==========
    single 512²  det_bin=1 (18 GiB out)     0.75      0.35     24.8 GiB
    single 512²  det_bin=2                   0.75      0.36      6.9 GiB
    single 1024² det_bin=2                   2.54      1.15     24.9 GiB
    single 1024² det_bin=4                   2.52      1.12      6.7 GiB
    6x  512²     det_bin=2 -> 1 GPU          3.87      1.69     32.3 GiB
    6x  512²     det_bin=4 -> 1 GPU          3.73      1.63      8.5 GiB
    10x 512²     det_bin=4 -> 1 GPU          6.25      2.63     13.0 GiB
    16x 512²     det_bin=4 -> 1 GPU         10.05      4.16     19.8 GiB
    ====================================  ========  ========  ==========

    Notes:
    - ``det_bin`` barely changes load time (compare 6x det_bin=2 vs 4) - it
      trades VRAM, not wall time. Use it to fit memory, not to go faster.
    - The compressed read is page-locked (parallel ``cudaHostRegister``) and
      its H2D overlaps the LZ4+bitshuffle kernels on a copy stream, so the GPU
      phase is largely hidden. The remaining cold cost is pure NVMe read.
    - Multi-GPU ``devices=[0, 1]`` shards files across cards for *capacity*
      (a stack larger than one card), not speed - cold load is disk-bound and
      both cards share the one NVMe.

    Sharded multi-GPU (``devices=[0, 1]``, 2x 96 GB cards, same setup):

    ====================================  ========  ========  ==========
    case                                  COLD (s)  WARM (s)  total VRAM
    ====================================  ========  ========  ==========
    6x  512² no-bin     (108 GiB)            4.07      2.02     108 GiB
    8x  512² no-bin     (144 GiB)            OOM       -        ceiling
    6x  512² det_bin=2  (27 GiB)             4.21      2.04      27 GiB
    16x 512² det_bin=2  (72 GiB)            10.03      5.10      72 GiB
    19x 512² det_bin=2  (86 GiB)            12.50      6.14      86 GiB
    16x 512² det_bin=4  (18 GiB)            10.43      5.24      18 GiB
    ====================================  ========  ========  ==========

    Capacity rule (per-tilt 512²x192² uint16): no-bin = 18 GiB, det_bin=2 =
    4.5 GiB, det_bin=4 = 1.1 GiB. The usable VRAM is NOT the full 190 GiB -
    each file's decompress transient (compressed + scratch + the growing
    output stack) stacks ~20 GiB on top per card, so:

    - **no-bin caps at ~6 files** on 2 cards (8x = 144 GiB OOMs on the
      transient, not the final size).
    - **det_bin=2 fits ~30 files**, **det_bin=4 fits ~150** (tiny per-file
      transient at bin=4). E.g. a 70-tilt det_bin=4 series = 70 x 1.1 ~= 79 GiB,
      fits two cards (or even one) with room to spare.
    """
    import os
    import re
    import time
    scan_order = _normalize_scan_order(scan_order)

    # Resolve the decompress backend ("auto" → cuda on an NVIDIA box, else mps
    # on Apple Silicon, else cpu). The cuda path below is unchanged. cpu/mps are
    # view/screen-only and land in later steps; until then they raise clearly so
    # a non-cuda box gets an honest error instead of a cupy ImportError crash.
    from .backends import resolve_backend
    backend = resolve_backend(backend)
    packed_uint4_output = _is_uint4_load_dtype(output_dtype)
    if packed_uint4_output:
        if int(det_bin) > 1:
            raise ValueError(
                "dtype='u4' means packed 4-bit detector counts (0..15) and "
                "cannot be combined with det_bin>1 because binned detector "
                "pixels can exceed 15. Use dtype='uint8' or dtype='uint16' "
                "for binned loads."
            )
        if backend != "cuda":
            raise NotImplementedError(
                "dtype='u4' packed HDF5 output is implemented for CUDA. "
                f"backend={backend!r} was selected."
            )
        if devices is not None:
            raise NotImplementedError(
                "dtype='u4' packed output is not enabled for devices= sharded "
                "stacked loads yet. Use gpus=... with stack=False to load "
                "per-master packed results, or use dtype='uint8'."
            )
        if isinstance(filepath, (list, tuple)):
            raise NotImplementedError(
                "dtype='u4' packed output is currently one master per "
                "LoadResult. Use gpus=... with stack=False for a list of "
                "packed results, or use dtype='uint8' for stacked browsing."
            )
    if row_prefix and backend != "mps":
        raise ValueError("row_prefix=True is only supported with backend='mps'.")
    if backend != "cuda":
        # These features are intrinsically CUDA (multi-GPU sharding, GPU
        # pinning, the torch-from-dlpack 5D dataset wrap). Name the fix.
        if device is not None or devices is not None:
            raise ValueError(
                f"device=/devices= multi-GPU requires backend='cuda'; got {backend!r}."
            )
        # MPS multi-dataset: dataset 0 is decoded synchronously and the
        # remaining datasets fill one common detector-input object in the
        # background. The scheduling owner stays attached to ``data``; widget
        # code sees only the common GPU-frame protocol.
        if backend == "mps" and (
            isinstance(filepath, (list, tuple))
            or (isinstance(filepath, (str, os.PathLike))
                and os.path.isdir(os.path.expanduser(str(filepath))))
        ):
            if scan_order != "row-major":
                raise ValueError(
                    "scan_order='serpentine' is not supported for MPS lazy "
                    "multi-dataset loads yet. Load one CUDA dataset, use "
                    "scan_region=..., or use det_bin>1."
                )
            from quantem.gpu.io.backends.mps.series import load_mps_datasets

            owner = load_mps_datasets(
                filepath,
                det_bin=det_bin,
                scan_shape=scan_shape,
                output_dtype=output_dtype,
                verbose=verbose,
                skip_mps_memory_check=skip_mps_memory_check,
            )
            data = owner.data
            data._io_owner = owner
            owner.start()
            metadata = dict(getattr(data, "metadata", {}) or {})
            metadata.update(
                backend="mps",
                file_names=list(owner.names),
                n_files=len(owner.names),
                scan_shape=tuple(int(value) for value in data.shape[1:3]),
                detector_shape=tuple(int(value) for value in data.shape[-2:]),
            )
            return LoadResult(data, metadata)
        return _load_view(
            filepath, backend, dataset_path=dataset_path, apply_mask=apply_mask,
            scan_shape=scan_shape, scan_order=scan_order, det_bin=det_bin, verbose=verbose,
            auto_narrow=auto_narrow, output_dtype=output_dtype,
            row_prefix=row_prefix,
            precompute_detector_sum=precompute_detector_sum,
            skip_mps_memory_check=skip_mps_memory_check,
        )

    # Pin all cupy allocations to `device` for this call. Recurse without
    # device so the wrap is paid once even on multi-file loads.
    if device is not None:
        device_idx = int(device.split(":")[1]) if isinstance(device, str) else int(device)
        with cp.cuda.Device(device_idx):
            return _load_impl(
                filepath, dataset_path=dataset_path, apply_mask=apply_mask,
                scan_shape=scan_shape, scan_order=scan_order, det_bin=det_bin, verbose=verbose,
                auto_narrow=auto_narrow, output_dtype=output_dtype,
                skip_mps_memory_check=skip_mps_memory_check,
            )

    # Sharded multi-GPU: split files across `devices`, each device loads + keeps
    # its own subset (NO gather to a single card). The only way a stack larger
    # than one GPU fits (e.g. 6× 512² no-bin = 108 GiB across 2× 96 GB), and
    # avoids the host-bounce penalty that made gather-mode slower than serial.
    # Returns LoadResult.data = {device: stacked_array_on_that_device}.
    if devices is not None and isinstance(filepath, (list, tuple)):
        return _load_sharded(
            list(filepath), devices, dataset_path=dataset_path,
            apply_mask=apply_mask, scan_shape=scan_shape, scan_order=scan_order,
            det_bin=det_bin,
            verbose=verbose, auto_narrow=auto_narrow, output_dtype=output_dtype,
        )

    # Multi-file: load first to get shape, pre-allocate, copy in-place.
    # Mixed scan shapes: explicit `scan_shape` wins; else first successful
    # file anchors the stack and any later file with a different shape is
    # skipped (same skipped-list pattern as missing data files).
    if isinstance(filepath, (list, tuple)):
        if len(filepath) == 0:
            raise ValueError("Empty file list")
        import queue
        import threading
        n_files = len(filepath)
        if verbose:
            bin_str = f", det_bin={det_bin}" if det_bin > 1 else ""
            print(f"Loading {n_files} files{bin_str} (disk‖gpu pipeline)")

        # Disk‖GPU pipeline: a producer thread reads + header-parses each
        # master's compressed bytes into host memory (_prepare_master, pure
        # CPU/disk, releases the GIL during I/O) while the main thread runs the
        # GPU decompress (_decompress_prepared) on the previous file. Wall time
        # drops from sum(disk+gpu) to ~max(total_disk, total_gpu). A bounded
        # queue (2 slots) caps host memory to ~2 files' compressed bytes.
        # Files that aren't resolvable chunked masters fall back to a full
        # serial load() in the consumer.
        prep_q: queue.Queue = queue.Queue(maxsize=2)
        _SENTINEL = object()

        def producer():
            for i, fp in enumerate(filepath):
                try:
                    chunk_names = _discover_chunk_names(fp)
                    if not chunk_names:
                        prep_q.put((i, fp, None, None))  # fallback to serial
                        continue
                    prepared = _prepare_master(fp, chunk_names, apply_mask)
                    prep_q.put((i, fp, prepared, None))
                except (FileNotFoundError, OSError, ValueError) as e:
                    prep_q.put((i, fp, None, e))
            prep_q.put(_SENTINEL)

        threading.Thread(target=producer, daemon=True).start()

        meta = None
        out = None
        n_loaded = 0
        skipped = []
        effective_shape = scan_shape
        t_multi_start = time.perf_counter()
        while True:
            item = prep_q.get()
            if item is _SENTINEL:
                break
            i, fp, prepared, err = item
            if err is not None:
                if verbose:
                    print(f"  [{i+1}/{n_files}] SKIPPED: {err}")
                skipped.append(i)
                continue
            try:
                if prepared is None:
                    # Fallback: not a chunked master — full serial load.
                    r = load(fp, dataset_path=dataset_path, apply_mask=apply_mask,
                             scan_shape=effective_shape, scan_order=scan_order, det_bin=det_bin,
                             verbose=False, auto_narrow=auto_narrow,
                             dtype=output_dtype)
                    fmeta, d = r.metadata, r.data
                else:
                    d = _decompress_prepared(
                        prepared, verbose=False, auto_narrow=auto_narrow,
                        det_bin=det_bin, streaming_bin=(det_bin > 1),
                        output_dtype=output_dtype)
                    fmeta = get_metadata(fp)
                    if prepared.get("pixel_mask") is not None:
                        fmeta["pixel_mask"] = prepared["pixel_mask"]
                    d = _apply_scan_shape(d, effective_shape, fmeta, scan_order)
                    fmeta["scan_order"] = scan_order
            except (FileNotFoundError, OSError, ValueError) as e:
                if verbose:
                    print(f"  [{i+1}/{n_files}] SKIPPED: {e}")
                skipped.append(i)
                continue
            if out is None:
                meta = fmeta
                out = cp.empty((n_files, *d.shape), dtype=d.dtype)
                if effective_shape is None:
                    effective_shape = meta.get("scan_shape")
            if d.shape != out.shape[1:]:
                if verbose:
                    print(f"  [{i+1}/{n_files}] SKIPPED: shape {tuple(d.shape)} "
                          f"differs from anchor {tuple(out.shape[1:])}")
                del d
                cp.get_default_memory_pool().free_all_blocks()
                skipped.append(i)
                continue
            out[n_loaded] = d
            del d
            cp.get_default_memory_pool().free_all_blocks()
            n_loaded += 1
        if out is None:
            raise FileNotFoundError(
                f"All {n_files} files failed to load (missing data files)"
            )
        if n_loaded < n_files:
            out = out[:n_loaded]
        # Record the per-dataset names (loaded order, skips dropped) so the viewer
        # can label the dataset slider with each source file instead of an index.
        skipped_set = set(skipped)
        loaded_names = [
            os.path.basename(str(filepath[i]))[:-len("_master.h5")]
            if str(filepath[i]).endswith("_master.h5") else os.path.basename(str(filepath[i]))
            for i in range(n_files) if i not in skipped_set
        ]
        meta["file_names"] = loaded_names
        meta["n_files"] = n_loaded
        if verbose:
            t_multi = time.perf_counter() - t_multi_start
            size_gb = out.nbytes / 1e9 if out is not None else 0
            skip_msg = f" (skipped {len(skipped)})" if skipped else ""
            print(f"  Done: {n_loaded} files{skip_msg} → {tuple(out.shape)} ({size_gb:.1f} GB) in {t_multi:.2f}s")
        return LoadResult(out, meta)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    t0 = time.perf_counter()
    global _default_decompressor

    with h5py.File(filepath, "r") as f:
        data_group = f.get("entry/data")
        if data_group is not None:
            # Check for Dectris-style chunked data (data_000001, etc.)
            chunk_names = sorted([
                name for name in data_group
                if re.match(r"data_\d{6}", name)
            ])
            # Two layouts ride the chunk-name prefix:
            #   1. classic Dectris master with external _data_NNNNNN.h5 siblings
            #   2. self-contained master where the data lives inline (rare;
            #      seen in some Velox-exported sets) or where the sibling
            #      files are missing entirely (common for half-copied
            #      datasets, e.g. sample_master.h5)
            # If chunk_names exist AND every external link resolves, take the
            # bulk-loader path. Otherwise drop to the inline `entry/data/data`
            # path and raise a clean error if neither layout has any data.
            chunks_resolvable = False
            if chunk_names:
                master_dir = os.path.dirname(os.path.abspath(filepath))
                for cn in chunk_names:
                    link = data_group.get(cn, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        chunk_path = os.path.join(master_dir, link.filename)
                        if not os.path.exists(chunk_path):
                            break
                    # Internal/soft links resolve in the same file, so they're
                    # always present by definition.
                else:
                    # Every external link points at an existing file.
                    chunks_resolvable = True
            if chunk_names and chunks_resolvable:
                # Master file with external data links (single or multi-chunk).
                # When det_bin > 1, request streaming bin so the full
                # unbinned tensor never lives in VRAM (4× memory savings
                # at det_bin=2).
                data, pixel_mask = _load_master_optimized(
                    filepath, chunk_names, apply_mask=apply_mask,
                    verbose=verbose, auto_narrow=auto_narrow,
                    det_bin=det_bin, streaming_bin=(det_bin > 1),
                    output_dtype=output_dtype,
                )
                meta = get_metadata(filepath)
                if pixel_mask is not None:
                    meta["pixel_mask"] = pixel_mask
                data = _apply_scan_shape(data, scan_shape, meta, scan_order)
                meta["scan_order"] = scan_order
                return LoadResult(data, meta)
            if "data" in data_group:
                # Self-contained master OR a master whose sibling chunk
                # files weren't found - try inline entry/data/data.
                dataset_path = "entry/data/data"
            elif chunk_names and not chunks_resolvable:
                # Master listed external chunks but none of the sibling files
                # exist on disk and there's no inline fallback. This is the
                # half-copied-dataset case; raise a clean FileNotFoundError so
                # callers (e.g. browse router) return 4xx instead of 500.
                raise FileNotFoundError(
                    f"{os.path.basename(filepath)}: external _data_NNNNNN.h5 "
                    "siblings missing and no inline entry/data/data dataset. "
                    "The master.h5 is incomplete; copy the sibling files."
                )

    # Use default path if not set
    if dataset_path is None:
        dataset_path = "entry/data/data"

    # Get dataset info for decompressor initialization
    with h5py.File(filepath, "r") as f:
        if dataset_path not in f:
            raise ValueError(f"Dataset '{dataset_path}' not found in {filepath}")
        ds = f[dataset_path]
        shape = ds.shape
        dtype = ds.dtype

        # Check if data has any filters (bitshuffle, gzip, etc.)
        dcpl = ds.id.get_create_plist()
        n_filters = dcpl.get_nfilters()

        # Calculate frame shape based on dimensionality
        # 4D/5D data (from save()): frame_shape is last 2 dims
        # 3D data (raw Dectris): frame_shape is last 2 dims
        if len(shape) == 5:
            frame_shape = shape[3:]
        elif len(shape) == 4:
            frame_shape = shape[2:]
        else:  # 3D
            frame_shape = shape[1:]

        frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
        n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Get pixel mask if available
        pixel_mask = None
        if apply_mask:
            mask_path = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if mask_path in f:
                pixel_mask = f[mask_path][:]

        # For uncompressed data (no filters), just read with h5py and transfer to GPU
        if n_filters == 0:
            raw_data = ds[:]
            data = cp.asarray(raw_data)
            if output_dtype is not None:
                data = (
                    pack_uint4_cupy(data)
                    if _is_uint4_load_dtype(output_dtype)
                    else data.astype(output_dtype)
                )
            t1 = time.perf_counter()
            if verbose:
                size_gb = data.nbytes / 1e9
                print(f"  {os.path.basename(filepath)}  {tuple(data.shape)}  {size_gb:.1f} GB  {t1-t0:.2f}s")
            meta = get_metadata(filepath)
            if pixel_mask is not None:
                meta["pixel_mask"] = pixel_mask
            data = _apply_scan_shape(data, scan_shape, meta, scan_order)
            meta["scan_order"] = scan_order
            return LoadResult(data, meta)

    # For 4D/5D compressed data, use the dedicated loader
    if len(shape) >= 4:
        data = _load_gpu_decompressed(filepath, dataset_path, shape, dtype, verbose)
        if output_dtype is not None:
            data = (
                pack_uint4_cupy(data)
                if _is_uint4_load_dtype(output_dtype)
                else data.astype(output_dtype)
            )
        t1 = time.perf_counter()
        if verbose:
            size_gb = data.nbytes / 1e9
            print(f"  {os.path.basename(filepath)}  {tuple(data.shape)}  {size_gb:.1f} GB  {t1-t0:.2f}s")
        meta = get_metadata(filepath)
        if pixel_mask is not None:
            meta["pixel_mask"] = pixel_mask
        data = _apply_scan_shape(data, scan_shape, meta, scan_order)
        meta["scan_order"] = scan_order
        return LoadResult(data, meta)

    # For 3D data, use cached GPUDecompressor
    if (
        _default_decompressor is None
        or frame_bytes > _default_decompressor.max_frame_bytes
        or n_blocks_per_frame > _default_decompressor.n_blocks_per_frame
    ):
        _default_decompressor = GPUDecompressor(
            max_compressed_bytes=1024 * 1024 * 1024,
            max_frames=70000,
            max_frame_bytes=frame_bytes,
            n_blocks_per_frame=n_blocks_per_frame,
        )

    data = _default_decompressor.load(filepath, dataset_path)
    if output_dtype is not None and not _is_uint4_load_dtype(output_dtype):
        data = data.astype(output_dtype)

    # Free decompressor buffers - they hold ~12 GB of GPU memory
    # and are only needed during decompression.
    _default_decompressor = None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Apply pixel mask if present
    if pixel_mask is not None:
        bad_row, bad_col = np.where(pixel_mask > 0)
        if len(bad_row) > 0:
            data[:, bad_row, bad_col] = 0

    t1 = time.perf_counter()
    if verbose:
        size_gb = data.nbytes / 1e9
        throughput = data.nbytes / (t1 - t0) / 1e9 if (t1 - t0) > 0 else 0
        print(f"  {os.path.basename(filepath)}  {tuple(data.shape)}  {size_gb:.1f} GB  {t1-t0:.2f}s  {throughput:.1f} GB/s")

    # Bin detector if requested
    if det_bin > 1:
        data = bin(data, factor=det_bin)
    if _is_uint4_load_dtype(output_dtype):
        data = pack_uint4_cupy(data)

    # Unflatten scan dimension - uses explicit scan_shape if passed,
    # else auto-derives from metadata (ntrigger for square scans).
    meta = get_metadata(filepath)
    if pixel_mask is not None:
        meta["pixel_mask"] = pixel_mask
    data = _apply_scan_shape(data, scan_shape, meta, scan_order)
    meta["scan_order"] = scan_order

    return LoadResult(data, meta)


def _load_gpu_decompressed(
    filepath: str,
    dataset_path: str,
    shape: tuple,
    dtype: np.dtype,
    verbose: bool = False,
) -> cp.ndarray:
    """Load HDF5 dataset using GPU decompression.

    Works with files saved using our GPU bitshuffle format.
    Uses the same kernel interface as GPUDecompressor.
    """
    import time

    t0 = time.perf_counter()

    # Calculate frame layout
    if len(shape) == 5:
        n_frames = shape[0] * shape[1] * shape[2]
        frame_shape = shape[3:]
    elif len(shape) == 4:
        n_frames = shape[0] * shape[1]
        frame_shape = shape[2:]
    else:  # 3D
        n_frames = shape[0]
        frame_shape = shape[1:]

    frame_bytes = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize
    n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

    if verbose:
        print(f"  Loading {os.path.basename(filepath)}: {n_frames} frames, {shape}")

    # Pre-allocate metadata arrays
    # uint64 chunk_offsets: see _prepare_master for rationale
    chunk_offsets = np.zeros(n_frames, dtype=np.uint64)
    block_counts = np.zeros(n_frames, dtype=np.uint32)
    block_starts_flat = np.zeros(n_frames * n_blocks_per_frame, dtype=np.uint32)
    block_offsets = np.zeros(n_frames + 1, dtype=np.uint32)

    # First pass: read all chunks and calculate total size
    raw_chunks = []
    with h5py.File(filepath, "r") as f:
        ds = f[dataset_path]

        for frame_idx in range(n_frames):
            # Calculate chunk index based on shape
            if len(shape) == 5:
                idx0 = frame_idx // (shape[1] * shape[2])
                rem = frame_idx % (shape[1] * shape[2])
                idx1 = rem // shape[2]
                idx2 = rem % shape[2]
                chunk_idx = (idx0, idx1, idx2, 0, 0)
            elif len(shape) == 4:
                idx0 = frame_idx // shape[1]
                idx1 = frame_idx % shape[1]
                chunk_idx = (idx0, idx1, 0, 0)
            else:
                chunk_idx = (frame_idx, 0, 0)

            # Read raw chunk
            _, raw = ds.id.read_direct_chunk(chunk_idx)
            raw_chunks.append(raw)

    # Calculate total compressed size and build metadata
    total_compressed = sum(len(r) for r in raw_chunks)

    # Use pinned memory for fast CPU→GPU transfer (26 GB/s vs 11 GB/s)
    # Allocation is slow (~0.5s) but transfer is 2.5x faster
    pinned_mem = cp.cuda.alloc_pinned_memory(total_compressed)
    buffer = np.frombuffer(pinned_mem, dtype=np.uint8, count=total_compressed)

    # Copy chunks to buffer and parse headers
    offset = 0
    for frame_idx, raw in enumerate(raw_chunks):
        chunk_len = len(raw)
        chunk_offsets[frame_idx] = offset
        buffer[offset:offset + chunk_len] = np.frombuffer(raw, dtype=np.uint8)

        # Parse bitshuffle header and block sizes
        # Header: 8 bytes uncompressed size + 4 bytes block size = 12 bytes
        pos = 12
        block_counts[frame_idx] = n_blocks_per_frame
        block_base = frame_idx * n_blocks_per_frame

        for b in range(n_blocks_per_frame):
            # Each LZ4 block has 4-byte big-endian size prefix
            block_starts_flat[block_base + b] = pos
            lz4_size = int.from_bytes(raw[pos:pos+4], 'big')
            pos += 4 + lz4_size

        offset += chunk_len

    # Free raw_chunks to save memory
    del raw_chunks

    # Compute cumulative block offsets
    block_offsets[1:n_frames + 1] = np.cumsum(block_counts[:n_frames])
    total_blocks = int(block_offsets[n_frames])

    t_read = time.perf_counter() - t0

    # Transfer to GPU using pinned memory for maximum throughput
    t0 = time.perf_counter()
    compressed_gpu = cp.empty(total_compressed, dtype=cp.uint8)
    compressed_gpu.set(buffer)

    chunk_offsets_gpu = cp.asarray(chunk_offsets)
    block_starts_gpu = cp.asarray(block_starts_flat[:total_blocks])
    block_counts_gpu = cp.asarray(block_counts)
    block_offsets_gpu = cp.asarray(block_offsets)

    # Allocate output buffers
    total_output_bytes = n_frames * frame_bytes
    lz4_output = cp.empty(total_output_bytes, dtype=cp.uint8)

    # LZ4 decompress with batching
    max_blocks = int(block_counts.max())
    max_batch = 10000

    for start in range(0, n_frames, max_batch):
        end = min(start + max_batch, n_frames)
        batch_n = end - start
        byte_offset = start * frame_bytes

        _h5lz4dc_kernel(
            ((max_blocks + 1) // 2, 1, batch_n),
            (32, 2, 1),
            (
                compressed_gpu,
                chunk_offsets_gpu[start:],
                block_starts_gpu,
                block_counts_gpu[start:],
                block_offsets_gpu[start:],
                np.uint32(BLOCK_SIZE),
                np.uint32(frame_bytes),
                lz4_output[byte_offset:],
            ),
        )

    # Inverse bitshuffle - use different kernel based on element size
    n_full_8kb = frame_bytes // BLOCK_SIZE
    tail_bytes = frame_bytes % BLOCK_SIZE
    elem_size = np.dtype(dtype).itemsize
    result_flat = cp.empty(total_output_bytes, dtype=cp.uint8)

    if elem_size == 2:
        # uint16: use optimized shared memory kernel
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            if n_full_8kb:
                _bitshuffle_kernel_u16(
                    (n_full_8kb, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_output[byte_offset:],
                        result_flat[byte_offset:].view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // elem_size
                if tail_bytes % elem_size or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u16(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_output[byte_offset:],
                        result_flat[byte_offset:].view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
    else:
        # uint32: use optimized ballot-based kernel
        frame_u32s = frame_bytes // 4
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            if n_full_8kb:
                _bitshuffle_kernel(
                    (n_full_8kb, 2, batch_n),
                    (32, 32, 1),
                    (
                        lz4_output[byte_offset:].view(cp.uint32),
                        result_flat[byte_offset:].view(cp.uint32),
                        np.uint32(frame_u32s),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // elem_size
                if tail_bytes % elem_size or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u32(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_output[byte_offset:],
                        result_flat[byte_offset:].view(cp.uint32),
                        np.uint32(frame_bytes),
                    ),
                )

    cp.cuda.Device().synchronize()

    # Reshape to original shape
    result = result_flat.view(dtype).reshape(shape)

    t_decomp = time.perf_counter() - t0

    if verbose:
        print(f"  Read: {t_read:.2f}s, decompress: {t_decomp:.2f}s")

    return result



def bin(
    data,
    factor: int = 2,
    axes: str = "detector",
    dtype=None,
    reduction: str = "sum",
    edge: str = "crop",
):
    """Apply spatial binning on GPU: CuPy or Torch (same type out).

    Pass ``cupy.ndarray`` and get ``cupy.ndarray`` back. Pass ``torch.Tensor``
    and get ``torch.Tensor`` back. NumPy is not accepted.

    Spatial sizes that are not multiples of ``factor`` are cropped to the
    largest multiple (trailing rows/cols dropped). Callers do not need to
    pre-slice.

    Parameters
    ----------
    data : cupy.ndarray or torch.Tensor
        One of:

        - 4D: ``(scan_row, scan_col, k_row, k_col)`` for full 4D-STEM.
        - 3D: ``(n_frames, k_row, k_col)`` for flattened scan / time series.
        - 2D: ``(k_row, k_col)`` for a single diffraction pattern or image.
    factor : int
        Binning factor (2 for 2x2, 4 for 4x4, etc.). Default 2.
    axes : str
        Which axes to bin:

        - ``"detector"`` or ``"k"``: bin ``k_row`` and ``k_col`` (last two dims
          on 2D/3D stacks of STEM frames).
        - ``"scan"`` or ``"r"``: bin ``scan_row`` and ``scan_col``.
        - ``"all"``: bin all four dimensions (4D data only).
    dtype :
        Output dtype in the input library. Default: float32 for mean; integer
        sum uses uint32 (CuPy) or int64 (Torch); otherwise float32.
    reduction : str
        ``"sum"`` (default) or ``"mean"``.
    edge : str
        ``"crop"`` (default) drops incomplete trailing bins for compatibility.
        ``"partial"`` keeps them by zero-padding before an exact sum. Partial
        mean bins are intentionally unsupported because zero padding changes
        their denominator.

    Returns
    -------
    cupy.ndarray or torch.Tensor
        Binned array, same library as ``data``.

    Examples
    --------
    >>> from quantem.gpu.io.load import bin
    >>> stack = bin(stack, factor=4, axes="detector", reduction="sum")  # (N,H,W)
    >>> binned = bin(cupy_4d, factor=2, axes="detector")
    """
    try:
        import torch
    except ImportError:
        torch = None

    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
    if edge not in ("crop", "partial"):
        raise ValueError(f"edge must be 'crop' or 'partial', got {edge!r}")
    if edge == "partial" and reduction == "mean":
        raise ValueError("edge='partial' supports exact sum reduction only")

    is_torch = torch is not None and isinstance(data, torch.Tensor)
    is_cupy = cp is not None and isinstance(data, cp.ndarray)
    if not is_torch and not is_cupy:
        kind = type(data).__name__
        raise TypeError(
            f"bin expects cupy.ndarray or torch.Tensor (GPU), got {kind}. "
            "NumPy is not supported - convert with "
            "torch.as_tensor(..., device=...) or cupy.asarray(...)."
        )

    if factor == 1:
        return data

    axes = axes.lower()
    if axes in ("detector", "diffraction", "q", "k"):
        axes = "detector"
    elif axes in ("scan", "real", "r"):
        axes = "scan"
    elif axes == "all":
        axes = "all"
    else:
        raise ValueError(
            f"axes must be 'detector', 'scan', or 'all', got {axes!r}"
        )

    if dtype is None:
        if reduction == "mean":
            dtype = torch.float32 if is_torch else cp.float32
        elif is_torch:
            dtype = (
                torch.int64
                if data.dtype
                in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
                else torch.float32
            )
        else:
            dtype = (
                cp.uint32
                if cp.issubdtype(data.dtype, cp.integer)
                else cp.float32
            )

    def _reduce(arr, dims):
        if is_torch:
            if reduction == "mean":
                out = arr.mean(dim=dims)
            else:
                out = arr.sum(dim=dims)
            return out.to(dtype=dtype) if dtype is not None else out
        if reduction == "mean":
            return arr.mean(axis=dims, dtype=dtype)
        return arr.sum(axis=dims, dtype=dtype)

    def _fit(size):
        return (size // factor) * factor

    if is_torch and not data.is_contiguous():
        data = data.contiguous()

    if data.ndim == 2:
        if axes == "scan":
            raise ValueError("Cannot bin scan axes on 2D data")
        h, w = data.shape
        h2, w2 = _fit(h), _fit(w)
        if h2 == 0 or w2 == 0:
            raise ValueError(
                f"Dimensions ({h}, {w}) too small for factor {factor}"
            )
        data = data[:h2, :w2]
        reshaped = data.reshape(h2 // factor, factor, w2 // factor, factor)
        return _reduce(reshaped, (1, 3))

    if data.ndim == 3:
        if axes == "scan":
            raise ValueError(
                "Cannot bin scan axes on 3D data. Reshape to 4D first: "
                "data.reshape(Ry, Rx, k_row, k_col)"
            )
        n, h, w = data.shape
        h2, w2 = _fit(h), _fit(w)
        if h2 == 0 or w2 == 0:
            raise ValueError(
                f"Dimensions ({h}, {w}) too small for factor {factor}"
            )
        data = data[:, :h2, :w2]
        reshaped = data.reshape(n, h2 // factor, factor, w2 // factor, factor)
        return _reduce(reshaped, (2, 4))

    if data.ndim == 4:
        sr, sc, kr, kc = data.shape

        def _padded(shape):
            if is_torch:
                out = torch.zeros(shape, dtype=data.dtype, device=data.device)
            else:
                out = cp.zeros(shape, dtype=data.dtype)
            out[:sr, :sc, :kr, :kc] = data
            return out

        if axes == "detector":
            if edge == "partial":
                kr2 = ((kr + factor - 1) // factor) * factor
                kc2 = ((kc + factor - 1) // factor) * factor
                data = _padded((sr, sc, kr2, kc2))
            else:
                kr2, kc2 = _fit(kr), _fit(kc)
            if kr2 == 0 or kc2 == 0:
                raise ValueError(
                    f"Detector dims ({kr}, {kc}) too small for factor {factor}"
                )
            data = data[:, :, :kr2, :kc2]
            reshaped = data.reshape(
                sr, sc, kr2 // factor, factor, kc2 // factor, factor
            )
            return _reduce(reshaped, (3, 5))

        if axes == "scan":
            if edge == "partial":
                sr2 = ((sr + factor - 1) // factor) * factor
                sc2 = ((sc + factor - 1) // factor) * factor
                data = _padded((sr2, sc2, kr, kc))
            else:
                sr2, sc2 = _fit(sr), _fit(sc)
            if sr2 == 0 or sc2 == 0:
                raise ValueError(
                    f"Scan dims ({sr}, {sc}) too small for factor {factor}"
                )
            data = data[:sr2, :sc2]
            reshaped = data.reshape(
                sr2 // factor, factor, sc2 // factor, factor, kr, kc
            )
            return _reduce(reshaped, (1, 3))

        if edge == "partial":
            sr2 = ((sr + factor - 1) // factor) * factor
            sc2 = ((sc + factor - 1) // factor) * factor
            kr2 = ((kr + factor - 1) // factor) * factor
            kc2 = ((kc + factor - 1) // factor) * factor
            data = _padded((sr2, sc2, kr2, kc2))
        else:
            sr2, sc2, kr2, kc2 = _fit(sr), _fit(sc), _fit(kr), _fit(kc)
        if min(sr2, sc2, kr2, kc2) == 0:
            raise ValueError(
                f"Shape {(sr, sc, kr, kc)} too small for factor {factor}"
            )
        data = data[:sr2, :sc2, :kr2, :kc2]
        reshaped = data.reshape(
            sr2 // factor,
            factor,
            sc2 // factor,
            factor,
            kr2 // factor,
            factor,
            kc2 // factor,
            factor,
        )
        return _reduce(reshaped, (1, 3, 5, 7))

    raise ValueError(
        f"Expected 2D, 3D, or 4D array, got {data.ndim}D. "
        "For multi-file data, use load(..., det_bin=2) instead."
    )



def _read_frame_count(filepath: str) -> int | None:
    """Read total frame count from a master HDF5 without decompressing.

    Only reads the first data file's nimages field from the master,
    which is the total frame count for virtual-dataset masters. Falls
    back to summing per-chunk shapes if nimages is unavailable.
    """
    try:
        with h5py.File(filepath, "r") as f:
            # Arina: ntrigger * nimages = total frames
            det = "entry/instrument/detector/detectorSpecific/"
            nimages = int(f[det + "nimages"][()]) if det + "nimages" in f else 1
            ntrigger = int(f[det + "ntrigger"][()]) if det + "ntrigger" in f else None
            if ntrigger is not None:
                return nimages * ntrigger
            # Fallback: sum shapes from data chunks
            import os
            chunk_names = _discover_chunk_names(filepath)
            if not chunk_names:
                return None
            master_dir = os.path.dirname(os.path.abspath(filepath))
            total = 0
            data_group = f["entry/data"]
            for name in chunk_names:
                link = data_group.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    data_path = os.path.join(master_dir, link.filename)
                else:
                    data_path = data_group[name].file.filename
                with h5py.File(data_path, "r") as df:
                    total += df["entry/data/data"].shape[0]
            return total
    except (OSError, KeyError, ValueError, TypeError):
        return None


def discover_masters(
    folder: str,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    scan_shape: tuple[int, int] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Find all master HDF5 files in a folder, sorted by path.

    Parameters
    ----------
    folder : str
        Root directory to search.
    pattern : str
        Glob pattern for matching filenames (default ``*_master.h5``).
    recursive : bool
        Search subdirectories recursively (default True).
    scan_shape : tuple[int, int], optional
        If set, only return files whose frame count matches
        ``scan_shape[0] * scan_shape[1]``. Reads HDF5 headers only,
        no decompression. Useful when a folder contains mixed scan sizes.
    verbose : bool
        Print indexed file list (default True).

    Returns
    -------
    list[str]
        Sorted list of absolute file paths.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    ValueError
        If no files match the pattern.
    """
    import pathlib
    root = pathlib.Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    glob_method = root.rglob if recursive else root.glob
    paths = sorted(str(p) for p in glob_method(pattern))
    if not paths:
        raise ValueError(
            f"No files matching '{pattern}' found in {folder}"
        )
    if scan_shape is not None:
        expected_frames = scan_shape[0] * scan_shape[1]
        filtered = []
        skipped = 0
        for p in paths:
            n = _read_frame_count(p)
            if n == expected_frames:
                filtered.append(p)
            else:
                skipped += 1
        paths = filtered
        if verbose and skipped > 0:
            print(f"  Filtered: {len(paths)} files matching {scan_shape[0]}x{scan_shape[1]} "
                  f"(skipped {skipped})")
    if verbose:
        for i, p in enumerate(paths):
            print(f"  [{i:>2}] {p.split('/')[-1]}")
        print(f"\nFound {len(paths)} files in {root.name}/")
    return paths
