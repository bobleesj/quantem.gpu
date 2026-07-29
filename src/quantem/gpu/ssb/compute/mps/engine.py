"""Exact Apple GPU SSB compute engine using MLX.

The input path is the chunk-backed MPS loader: each BF detector pixel is
streamed from the resident Metal chunks, transformed with MLX FFT on Apple GPU,
corrected, and accumulated without materializing the full 4D stack or using
Torch. Interactive reconstruction and Optuna/Nelder-Mead fitting share the
same exact reconstruction/loss core.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from quantem.gpu.detector import auto_probe, mean_dp
from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.optics.physics import electron_wavelength_angstrom
from quantem.gpu.ssb.bf_selector import BrightfieldDisk


_EXACT_ROW_PACK_STORAGE_BF_512 = 300
_EXACT_ROW_STORAGE_CLASSES_BF_512 = (288, 320)
_EXACT_SCALAR_ROW_PACK_DEPTH_512 = 5


@dataclass
class _PreparedMpsSSB:
    """Device-resident BF FFT stack and geometry for repeated SSB loss calls."""

    mx: object
    g_qk: object
    qx: object
    qy: object
    q_row: object
    q_col: object
    kx: object
    ky: object
    kx_np: np.ndarray
    ky_np: np.ndarray
    dc_value: complex
    scan_shape: tuple[int, int]
    wavelength: float
    semiangle_rad: float
    ang_y_rad: float
    ang_x_rad: float
    factor: float
    dc_mask: object
    num_bf: int
    alpha_k2: object | None
    cos2_k: object | None
    sin2_k: object | None
    aperture_k: object | None
    alpha_m2: object | None
    cos2_m: object | None
    sin2_m: object | None
    ap_m: object | None
    alpha_p2: object | None
    cos2_p: object | None
    sin2_p: object | None
    ap_p: object | None
    alpha_k2_1d: object | None = None
    cos2_k_1d: object | None = None
    sin2_k_1d: object | None = None
    aperture_k_1d: object | None = None
    bf_storage_indices_np: np.ndarray | None = None


def _bf_storage_chunks(
    prepared: _PreparedMpsSSB,
    logical_chunk_bf: int,
):
    """Yield packed storage slices without moving logical reduction boundaries."""
    logical_chunk_bf = max(1, int(logical_chunk_bf))
    indices = prepared.bf_storage_indices_np
    if indices is None:
        for start in range(0, prepared.num_bf, logical_chunk_bf):
            yield start, min(start + logical_chunk_bf, prepared.num_bf)
        return
    indices = np.asarray(indices, dtype=np.intp)
    for logical_start in range(0, prepared.num_bf, logical_chunk_bf):
        logical_stop = min(logical_start + logical_chunk_bf, prepared.num_bf)
        storage_start = int(np.searchsorted(indices, logical_start, side="left"))
        storage_stop = int(np.searchsorted(indices, logical_stop, side="left"))
        if storage_start != storage_stop:
            yield storage_start, storage_stop


def _bf_storage_chunk_packs(
    prepared: _PreparedMpsSSB,
    logical_chunk_bf: int,
    max_storage_bf: int,
) -> list[list[tuple[int, int]]]:
    """Pack adjacent sparse boundaries without exceeding row scratch limits."""
    boundaries = list(_bf_storage_chunks(prepared, logical_chunk_bf))
    if not boundaries:
        return []
    max_storage_bf = max(1, int(max_storage_bf))
    packs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_size = 0
    for start, stop in boundaries:
        boundary_size = int(stop) - int(start)
        if current and current_size + boundary_size > max_storage_bf:
            packs.append(current)
            current = []
            current_size = 0
        current.append((int(start), int(stop)))
        current_size += boundary_size
    if current:
        packs.append(current)
    return packs


def _exact_pair_row_storage_bf_512(chunk_bf: int) -> int:
    """Return the smallest retained allocation class for one exact pair pack."""
    chunk_bf = int(chunk_bf)
    for storage_class in _EXACT_ROW_STORAGE_CLASSES_BF_512:
        if chunk_bf <= storage_class:
            return storage_class
    raise ValueError(
        f"Exact 512 pair pack of {chunk_bf} BF planes exceeds "
        f"the {_EXACT_ROW_STORAGE_CLASSES_BF_512[-1]}-plane storage class."
    )


class _ArrayFrames:
    """Flat detector-column view over a 4D crop-first array.

    Metal reconstruction and fitting stream one detector pixel over all scan
    positions.  Full no-bin MPS loads provide that through ``ChunkedFrames``;
    crop-first MPS loads return a Metal-backed ndarray-like object instead.
    This adapter gives both inputs the same ``column(row, col)`` contract.
    """

    _is_gpu_frames = True

    def __init__(self, data):
        arr = np.asarray(data)
        if arr.ndim == 4:
            self.scan_shape = (int(arr.shape[0]), int(arr.shape[1]))
            self.det_shape = (int(arr.shape[2]), int(arr.shape[3]))
            self._flat = arr.reshape(-1, *self.det_shape)
        elif arr.ndim == 3:
            self.scan_shape = None
            self.det_shape = (int(arr.shape[1]), int(arr.shape[2]))
            self._flat = arr
        else:
            raise TypeError(
                "MPS SSB preview expects 3D/4D detector data or chunk-backed "
                f"MPS data, got shape {arr.shape}."
            )
        self.shape = tuple(int(x) for x in self._flat.shape)
        self.ndim = 3
        self.dtype = self._flat.dtype
        self.detector_sum = None

    def __array__(self, dtype=None):
        arr = np.asarray(self._flat)
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def reshape(self, *shape, **kwargs):
        return self._flat.reshape(*shape, **kwargs)

    def column(self, row: int, col: int) -> np.ndarray:
        return np.asarray(self._flat[:, int(row), int(col)])

    def columns(self, rows, cols) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.intp).reshape(-1)
        cols = np.asarray(cols, dtype=np.intp).reshape(-1)
        if rows.shape != cols.shape:
            raise ValueError("rows and cols must have matching shapes.")
        flat_idx = rows * int(self.det_shape[1]) + cols
        flat = np.asarray(self._flat).reshape(int(self._flat.shape[0]), -1)
        return np.take(flat, flat_idx, axis=1).T


class _BfColumnDetectorView:
    """Minimal detector geometry used by the shared Metal compute backend."""

    def __init__(self, detector_shape: tuple[int, int]) -> None:
        self.det = tuple(int(value) for value in detector_shape)


class MpsBfColumnFrames:
    """Exact disk-backed BF columns for MPS SSB fitting and reconstruction.

    The stored values remain integer-exact. File reads and unified-memory copies
    are host I/O, while detector sums, FFTs, objectives, refinement, and final
    reconstruction use MPS/Metal. Only requested detector columns are copied into
    MLX storage; the full detector stack is never materialized.
    """

    _is_gpu_frames = True

    def __init__(
        self,
        path: str | Path,
        *,
        selection: BrightfieldDisk,
        scan_shape: tuple[int, int],
        dtype: np.dtype | str,
        max_value: int | None = None,
        detector_sum: np.ndarray | None = None,
        dc_value: complex | None = None,
        verbose: bool = False,
    ) -> None:
        self.source_path = Path(path).expanduser().resolve()
        self.scan_shape = tuple(int(value) for value in scan_shape)
        self.selection = selection
        self.det_shape = selection.detector_shape
        self._n = int(np.prod(self.scan_shape))
        self._np_dtype = np.dtype(dtype)
        if self._np_dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
            raise ValueError(
                "MPS BF-column input must use exact uint8 or uint16 values; "
                f"got {self._np_dtype}."
            )
        self.dtype = self._np_dtype
        self.shape = (self._n, *self.det_shape)
        self.ndim = 3
        self.det_bin = 1
        self.vi = _BfColumnDetectorView(self.det_shape)
        # detector.mean_dp recognizes GPU-frame objects through the common
        # chunk-backed dispatch. No detector chunks are present by design.
        self.chunks = ()
        expected_bytes = selection.size * self._n * self._np_dtype.itemsize
        actual_bytes = self.source_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"BF-column file has {actual_bytes} bytes; expected "
                f"{expected_bytes} for {selection.size} BF x "
                f"{self.scan_shape} {self._np_dtype}."
            )
        self._columns = np.memmap(
            self.source_path,
            dtype=self._np_dtype,
            mode="r",
            shape=(selection.size, self._n),
        )
        self._lookup = np.full(self.det_shape, -1, dtype=np.int32)
        self._lookup[selection.rows, selection.cols] = np.arange(
            selection.size, dtype=np.int32,
        )
        self.max_value = (
            int(max_value)
            if max_value is not None
            else int(np.iinfo(self._np_dtype).max)
        )
        if self.max_value < 0 or self.max_value > int(np.iinfo(self._np_dtype).max):
            raise ValueError(
                f"max_value={self.max_value} is invalid for {self._np_dtype}."
            )
        self.dc_value = (
            None
            if dc_value is None
            else complex(np.complex64(dc_value))
        )
        sum_t0 = time.perf_counter()
        self._detector_sum = None
        if detector_sum is not None:
            exact_sum = np.asarray(detector_sum)
            if exact_sum.shape != self.det_shape:
                raise ValueError(
                    "detector_sum shape does not match BF detector geometry: "
                    f"{exact_sum.shape} versus {self.det_shape}."
                )
            if not np.issubdtype(exact_sum.dtype, np.integer):
                raise TypeError(
                    "detector_sum must contain exact integer counts; got "
                    f"{exact_sum.dtype}."
                )
            self._detector_sum = exact_sum.copy()
        elif self.dc_value is None:
            self._detector_sum = self._detector_sum_mps()
        self.load_seconds = time.perf_counter() - sum_t0
        self.gather_seconds = 0.0
        self.gather_calls = 0
        self.gather_bytes = 0
        if verbose:
            gib = actual_bytes / 1024**3
            rate = gib / max(self.load_seconds, 1e-9)
            print(
                f"Loaded exact MPS BF columns in {self.load_seconds:.2f}s "
                f"({selection.size} BF, {gib:.2f} GiB, {rate:.2f} GiB/s)"
            )

    @property
    def nbytes(self) -> int:
        return int(self._columns.nbytes)

    @property
    def detector_sum(self) -> np.ndarray:
        """Return exact detector sums, computing them only when requested."""

        if self._detector_sum is None:
            self._detector_sum = self._detector_sum_mps()
        return self._detector_sum

    def _detector_sum_mps(self) -> np.ndarray:
        """Reduce BF columns on Metal without a CPU scientific fallback."""
        mx = _require_mlx()
        sums = np.zeros(self.selection.size, dtype=np.uint64)
        # Every float32 partial remains an exactly representable integer. The
        # small returned partial vectors are accumulated as uint64 metadata.
        safe_scan = max(1, int((2**24 - 1) // max(1, self.max_value)))
        safe_scan = min(self._n, safe_scan)
        for bf_start in range(0, self.selection.size, 64):
            bf_stop = min(bf_start + 64, self.selection.size)
            partial = np.zeros(bf_stop - bf_start, dtype=np.uint64)
            for scan_start in range(0, self._n, safe_scan):
                scan_stop = min(scan_start + safe_scan, self._n)
                block = mx.array(
                    np.asarray(
                        self._columns[
                            bf_start:bf_stop,
                            scan_start:scan_stop,
                        ]
                    ),
                    dtype=mx.float32,
                )
                reduced = mx.sum(block, axis=1)
                mx.eval(reduced)
                partial += np.rint(np.asarray(reduced)).astype(np.uint64)
                del block, reduced
            sums[bf_start:bf_stop] = partial
        mx.clear_cache()
        detector_sum = np.zeros(self.det_shape, dtype=np.uint64)
        detector_sum[self.selection.rows, self.selection.cols] = sums
        return detector_sum

    def _indices(self, rows, cols) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        cols = np.asarray(cols, dtype=np.int32).reshape(-1)
        if rows.shape != cols.shape:
            raise ValueError("rows and cols must have matching shapes.")
        indices = self._lookup[rows, cols]
        if bool(np.any(indices < 0)):
            missing = np.flatnonzero(indices < 0)
            first = int(missing[0])
            raise ValueError(
                "Requested detector coordinate is absent from the exact BF "
                f"source: (row, col)=({int(rows[first])}, {int(cols[first])})."
            )
        return indices.astype(np.intp, copy=False)

    def columns(self, rows, cols) -> np.ndarray:
        """Return requested exact BF columns as ``(BF, scan)`` integers."""
        return np.asarray(self._columns[self._indices(rows, cols)])

    def columns_float32(self, rows, cols) -> np.ndarray:
        """Return requested BF columns as MPS objective float32 inputs."""
        return self.columns(rows, cols).astype(np.float32, copy=False)

    def columns_float32_into(
        self,
        rows,
        cols,
        out: np.ndarray,
    ) -> np.ndarray:
        """Copy requested exact columns directly into MLX unified storage."""
        t0 = time.perf_counter()
        indices = self._indices(rows, cols)
        expected_shape = (int(indices.size), self._n)
        if tuple(int(value) for value in out.shape) != expected_shape:
            raise ValueError(
                f"Output shape {out.shape} does not match {expected_shape}."
            )
        for start in range(0, int(indices.size), 32):
            stop = min(start + 32, int(indices.size))
            out[start:stop] = self._columns[indices[start:stop]]
        self.gather_seconds += time.perf_counter() - t0
        self.gather_calls += 1
        self.gather_bytes += int(indices.size) * self._n * self._np_dtype.itemsize
        return out


def load_bf_columns_mps(
    calibration: str | Path,
    *,
    verbose: bool = False,
) -> MpsBfColumnFrames:
    """Load a ShowPtycho exact BF-column companion for MPS SSB.

    Parameters
    ----------
    calibration : str or Path
        A ShowPtycho folder or its ``snapshots/cal.json`` file.
    verbose : bool, default False
        Print the exact BF-only load time and bandwidth.
    """
    source = Path(calibration).expanduser().resolve()
    if source.is_dir():
        candidates = [source / "snapshots" / "cal.json", source / "cal.json"]
        cal_path = next((path for path in candidates if path.is_file()), None)
        if cal_path is None:
            raise FileNotFoundError(
                f"No ShowPtycho calibration found under {source}."
            )
    else:
        cal_path = source
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ShowPtycho calibration must be a JSON object: {cal_path}")
    required_fields = (
        "bf_column_companion_path",
        "bf_column_encoding",
        "bf_rows",
        "bf_cols",
        "bf_center",
        "detector_shape",
        "scan_region",
    )
    missing_fields = [name for name in required_fields if name not in payload]
    if missing_fields:
        raise ValueError(
            f"Calibration is missing required BF-column fields "
            f"{missing_fields}: {cal_path}"
        )
    relative = payload["bf_column_companion_path"]
    if not relative:
        raise ValueError(f"BF-column companion path is empty: {cal_path}")
    relative_path = Path(str(relative))
    path_candidates = (
        [relative_path]
        if relative_path.is_absolute()
        else [cal_path.parent / relative_path, cal_path.parent.parent / relative_path]
    )
    bf_path = next((path.resolve() for path in path_candidates if path.is_file()), None)
    if bf_path is None:
        raise FileNotFoundError(
            "Exact BF-column companion was not found. Checked: "
            + ", ".join(str(path) for path in path_candidates)
        )
    encoding = str(payload["bf_column_encoding"])
    if encoding in {"u8", "uint8"}:
        dtype = np.uint8
    elif encoding in {"u16", "uint16"}:
        dtype = np.uint16
    else:
        raise ValueError(
            f"Unsupported BF-column encoding {encoding!r}: {cal_path}"
        )
    scan_region = payload["scan_region"]
    if not isinstance(scan_region, dict) or "shape" not in scan_region:
        raise ValueError(
            f"Calibration scan_region must contain a 2D shape: {cal_path}"
        )
    scan_shape = scan_region["shape"]
    if not isinstance(scan_shape, list) or len(scan_shape) != 2:
        raise ValueError(f"Calibration has no valid 2D scan shape: {cal_path}")
    detector_shape = payload["detector_shape"]
    if not isinstance(detector_shape, list) or len(detector_shape) != 2:
        raise ValueError(
            f"Calibration has no valid 2D detector shape: {cal_path}"
        )
    bf_center = payload["bf_center"]
    if not isinstance(bf_center, list) or len(bf_center) != 2:
        raise ValueError(
            f"Calibration bf_center must be [row, col]: {cal_path}"
        )
    bf_rows = payload["bf_rows"]
    bf_cols = payload["bf_cols"]
    if (
        not isinstance(bf_rows, list)
        or not isinstance(bf_cols, list)
        or not bf_rows
        or len(bf_rows) != len(bf_cols)
    ):
        raise ValueError(
            f"Calibration BF rows and columns must be non-empty matching "
            f"lists: {cal_path}"
        )

    max_value = None
    manifest_candidates = [
        cal_path.parent / "manifest.json",
        cal_path.parent.parent / "manifest.json",
    ]
    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bf_meta = (manifest.get("source") or {}).get("bf_columns") or {}
        if bf_meta.get("max_value") is not None:
            max_value = int(bf_meta["max_value"])
            break
    rows_array = np.asarray(bf_rows, dtype=np.int32)
    cols_array = np.asarray(bf_cols, dtype=np.int32)
    center_row_col = tuple(float(value) for value in bf_center)
    distance_sq = (rows_array.astype(np.float32) - center_row_col[0]) ** 2
    distance_sq += (cols_array.astype(np.float32) - center_row_col[1]) ** 2
    coordinate_radius = float(np.sqrt(distance_sq).max()) + 1e-3
    stored_radius = payload.get("bf_radius_px")
    radius = coordinate_radius if stored_radius is None else float(stored_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            f"Calibration bf_radius_px must be positive and finite: {cal_path}"
        )
    if float(np.sqrt(distance_sq).max()) > radius + 1e-3:
        raise ValueError(
            "Calibration BF coordinates extend beyond bf_radius_px: "
            f"{cal_path}"
        )
    selection = BrightfieldDisk(
        rows=rows_array,
        cols=cols_array,
        center_row_col=center_row_col,
        radius_px=radius,
        detected_radius_px=radius,
        detector_shape=tuple(int(value) for value in detector_shape),
    )
    stored_dc = payload.get("dc_value")
    dc_value = None
    if stored_dc is not None:
        if not isinstance(stored_dc, list) or len(stored_dc) != 2:
            raise ValueError(
                f"Calibration dc_value must be [real, imag]: {cal_path}"
            )
        dc_parts = np.asarray(stored_dc, dtype=np.float64)
        if not bool(np.all(np.isfinite(dc_parts))):
            raise ValueError(
                f"Calibration dc_value must be finite: {cal_path}"
            )
        dc_value = complex(float(dc_parts[0]), float(dc_parts[1]))
    return MpsBfColumnFrames(
        bf_path,
        selection=selection,
        scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
        dtype=dtype,
        max_value=max_value,
        dc_value=dc_value,
        verbose=verbose,
    )


def _require_mlx():
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MPS SSB preview requires MLX on Apple Silicon. Install with "
            "`python -m pip install mlx` in the Mac environment."
        ) from exc
    return mx


def _as_chunked_frames(data):
    from quantem.gpu.detector.compute.mps.kernels import ChunkedFrames
    from quantem.gpu.io.backends.mps import MPSChunked4DSTEM
    from quantem.gpu.io.load import LoadResult

    if isinstance(data, (MpsBfColumnFrames, ChunkedFrames, _ArrayFrames)):
        return data
    if isinstance(data, LoadResult):
        data = data.data
    if isinstance(data, MPSChunked4DSTEM):
        return ChunkedFrames(data)
    if isinstance(data, np.ndarray) and data.ndim in (3, 4):
        return _ArrayFrames(data)
    raise TypeError(
        "MPS SSB preview expects chunk-backed MPS data from "
        "`quantem.gpu.io.load.load(..., backend='mps')` or a crop-first "
        "3D/4D MPS/NumPy array."
    )


def _selected_columns_stack(
    frames,
    rows: np.ndarray,
    cols: np.ndarray,
    scan_shape: tuple[int, int],
) -> np.ndarray:
    """Return selected detector columns as ``(num_bf, scan_y, scan_x)``."""
    from quantem.gpu.detector.compute.mps.kernels import ChunkedFrames

    if isinstance(frames, (MpsBfColumnFrames, ChunkedFrames)):
        flat = frames.columns_float32(rows, cols)
    elif isinstance(frames, _ArrayFrames):
        flat = frames.columns(rows, cols)
    else:
        raise TypeError(
            "Unsupported MPS frame source; expected exact BF columns, "
            "ChunkedFrames, or a 3D/4D NumPy array."
        )
    return np.asarray(flat).reshape(int(rows.size), *scan_shape)


@lru_cache(maxsize=1)
def _default_object_setup_chunk_bf() -> int:
    """BF chunk size for first-use object-mode MPS setup."""
    override = os.environ.get("QUANTEM_MPS_SSB_OBJECT_CHUNK_BF")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    try:
        total = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            ).stdout.strip()
        )
    except Exception:
        return 256
    if total >= 96 * 1024**3:
        return 1024
    if total >= 48 * 1024**3:
        return 512
    return 256


@lru_cache(maxsize=1)
def _default_object_redraw_chunk_bf() -> int:
    """BF chunk size for repeated object-mode MPS redraws."""
    override = os.environ.get("QUANTEM_MPS_SSB_OBJECT_REDRAW_CHUNK_BF")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return 128


@lru_cache(maxsize=1)
def _default_object_redraw_threadgroup(
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """Metal threadgroup size for repeated object-mode MPS redraws."""
    override = os.environ.get("QUANTEM_MPS_SSB_OBJECT_THREADGROUP")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if scan_shape is not None and max(int(scan_shape[0]), int(scan_shape[1])) >= 512:
        return 64
    return 16


@lru_cache(maxsize=1)
def _default_phase_loss_chunk_bf(
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """BF chunk size for full phase/loss reconstruction on MPS."""
    override = os.environ.get("QUANTEM_MPS_SSB_PHASE_CHUNK_BF")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    try:
        total = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            ).stdout.strip()
        )
    except Exception:
        return 512
    if total >= 96 * 1024**3:
        chunk = 4096
    elif total >= 64 * 1024**3:
        chunk = 1024
    else:
        chunk = 512

    if scan_shape is None:
        return chunk

    ny, nx = (max(1, int(scan_shape[0])), max(1, int(scan_shape[1])))
    if max(ny, nx) <= 256:
        if total >= 96 * 1024**3:
            return 16384
        if total >= 64 * 1024**3:
            return 8192
        return 4096
    if max(ny, nx) >= 1024:
        # Full-BF 1024 phase/loss on MLX/Metal hits a scheduling and
        # allocation cliff at very large chunks. After scalar-loss reduction,
        # 512 BF is the best measured default on a 96 GB-class Apple GPU.
        return min(chunk, 512)
    return chunk


def _effective_phase_loss_chunk_bf(
    chunk_bf: int,
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """Use a faster full phase/loss chunk unless the caller retuned it."""
    requested = max(1, int(chunk_bf))
    if requested == 16:
        return _default_phase_loss_chunk_bf(scan_shape)
    return requested


@lru_cache(maxsize=1)
def _default_phase_col_k_bf(
    scan_shape: tuple[int, int] | None = None,
) -> int:
    """BF grouping for fused Metal column phase/loss accumulation."""
    override = os.environ.get("QUANTEM_MPS_SSB_PHASE_COL_K_BF")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if scan_shape == (512, 512):
        # The 512 radix-8 kernel reaches its best occupancy with eight columns
        # per threadgroup. Keeping the whole BF chunk in each group avoids the
        # partial-image traffic and follow-up MLX reduction without reducing
        # the exact BF evidence.
        return 4096
    return 32


def _scan_shape(frames) -> tuple[int, int]:
    from quantem.gpu.detector.compute.mps.kernels import ChunkedFrames

    if isinstance(frames, (MpsBfColumnFrames, _ArrayFrames)):
        shape = frames.scan_shape
    elif isinstance(frames, ChunkedFrames):
        shape = frames.metadata.get("scan_shape")
    else:
        raise TypeError(f"Unsupported MPS frame source: {type(frames).__name__}.")
    if shape is not None:
        return int(shape[0]), int(shape[1])
    n = int(frames.shape[0])
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError("scan_shape is required for non-square frame counts.")
    return side, side


def _spatial_frequencies(shape: tuple[int, int], sampling: tuple[float, float]):
    return (
        np.fft.fftfreq(shape[0], sampling[0]).astype(np.float32),
        np.fft.fftfreq(shape[1], sampling[1]).astype(np.float32),
    )


def _compute_geometry(mx, dx, dy, wavelength, semiangle_rad, ang_y_rad, ang_x_rad):
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    r = mx.sqrt(r2)
    alpha = r * wavelength
    alpha2 = alpha * alpha
    inv_r2 = mx.where(r2 > 1e-30, 1.0 / r2, 0.0)
    cos2phi = (dx2 - dy2) * inv_r2
    sin2phi = 2.0 * dx * dy * inv_r2
    denom_num2 = (dx * ang_y_rad) ** 2 + (dy * ang_x_rad) ** 2
    inv_r = mx.where(r > 1e-15, 1.0 / r, 0.0)
    denom = mx.sqrt(denom_num2) * inv_r
    edge = mx.where(denom > 1e-15, (semiangle_rad - alpha) / denom + 0.5, 1.0)
    aperture = mx.clip(edge, 0.0, 1.0)
    return alpha2, cos2phi, sin2phi, aperture


def _exp_neg_i(mx, chi):
    return mx.cos(chi) - (1j * mx.sin(chi))


def _as_sampling(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    return float(value[0]), float(value[1])


def _resolve_bf_selection(
    data,
    threshold: float,
    bf_radius: float | None,
    center_override: tuple[float, float] | None = None,
    *,
    mean_diffraction: np.ndarray | None = None,
) -> BrightfieldDisk:
    """Return one validated BF selection for exact or raw detector input."""

    if isinstance(data, MpsBfColumnFrames):
        if center_override is not None or bf_radius is not None:
            raise ValueError(
                "MpsBfColumnFrames owns its exact BF selection; do not pass "
                "bf_center or bf_radius overrides."
            )
        return data.selection

    dp = mean_dp(data) if mean_diffraction is None else mean_diffraction
    if bf_radius is None:
        detected_center, detected_radius = auto_probe(dp)
    else:
        detected_center, detected_radius = _detect_bf_radius_numpy(dp)
    mask = dp > float(dp.max()) * float(threshold)
    rr, cc = np.nonzero(mask)
    if rr.size == 0:
        raise ValueError(
            f"No bright-field pixels found with threshold={threshold:.2f}."
        )
    if center_override is not None:
        center = (
            float(center_override[0]),
            float(center_override[1]),
        )
    elif bf_radius is None:
        center = (float(detected_center[0]), float(detected_center[1]))
    else:
        weights = dp[rr, cc].astype(np.float32, copy=False)
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            center = (
                float((rr.astype(np.float32) * weights).sum() / weight_sum),
                float((cc.astype(np.float32) * weights).sum() / weight_sum),
            )
        else:
            center = (float(rr.mean()), float(cc.mean()))
    selected_radius = float(detected_radius if bf_radius is None else bf_radius)
    dist2 = (rr.astype(np.float32) - center[0]) ** 2 + (
        cc.astype(np.float32) - center[1]
    ) ** 2
    keep = dist2 <= selected_radius**2
    rr = rr[keep]
    cc = cc[keep]
    if rr.size == 0:
        raise ValueError(
            f"No BF pixels selected with threshold={threshold} and radius={bf_radius}."
        )
    return BrightfieldDisk(
        rows=rr.astype(np.int32),
        cols=cc.astype(np.int32),
        center_row_col=center,
        radius_px=selected_radius,
        detected_radius_px=float(detected_radius),
        detector_shape=tuple(int(value) for value in dp.shape),
    )


def _detect_bf_radius_numpy(
    mean_dp_array: np.ndarray,
    threshold_ratio: float = 0.1,
) -> tuple[tuple[int, int], int]:
    """NumPy mirror of :func:`quantem.gpu.detector.compute.cuda.probe.detect_bf_radius`."""
    dp = np.asarray(mean_dp_array, dtype=np.float32)
    if dp.ndim != 2:
        raise ValueError(f"Expected 2D diffraction pattern, got shape {dp.shape}.")
    n_k_row, n_k_col = dp.shape
    dp_max = float(np.nanmax(dp))
    if not np.isfinite(dp_max) or dp_max <= 0:
        raise ValueError("Diffraction pattern has no positive finite values.")
    mask = dp > threshold_ratio * dp_max
    if not bool(mask.any()):
        raise ValueError(f"No pixels above threshold ({threshold_ratio:.0%} of max).")
    mask_f = mask.astype(np.float32)
    total = float(mask_f.sum())
    row_coords = np.arange(n_k_row, dtype=np.float32).reshape(-1, 1)
    col_coords = np.arange(n_k_col, dtype=np.float32).reshape(1, -1)
    row_center = max(0, min(int(round(float((row_coords * mask_f).sum() / total))), n_k_row - 1))
    col_center = max(0, min(int(round(float((col_coords * mask_f).sum() / total))), n_k_col - 1))
    dr = np.arange(n_k_row, dtype=np.float32) - row_center
    dc = np.arange(n_k_col, dtype=np.float32) - col_center
    rr, cc = np.meshgrid(dr, dc, indexing="ij")
    radii = np.rint(np.sqrt(rr * rr + cc * cc)).astype(np.int32).reshape(-1)
    max_r = min(row_center, col_center, n_k_row - row_center, n_k_col - col_center)
    if max_r < 2:
        return (row_center, col_center), max(1, min(n_k_row, n_k_col) // 4)
    valid = radii < max_r
    profile = np.bincount(radii[valid], weights=dp.reshape(-1)[valid], minlength=max_r).astype(np.float32)
    counts = np.bincount(radii[valid], minlength=max_r).astype(np.float32)
    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]
    if profile.size > 5:
        sigma = 2.0
        ksize = int(6 * sigma + 1) | 1
        x = np.arange(ksize, dtype=np.float32) - ksize // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        kernel /= kernel.sum()
        padded = np.pad(profile, ksize // 2, mode="edge")
        profile_smooth = np.convolve(padded, kernel, mode="valid")[:profile.size]
        half_max = float(profile_smooth[:5].mean()) * 0.5
        below_half = np.flatnonzero(profile_smooth < half_max)
        radius = int(below_half[0]) if below_half.size else int(profile.size) // 2
    else:
        radius = min(n_k_row, n_k_col) // 4
    return (row_center, col_center), max(1, int(radius))


def _ranges_from_start(
    start: dict[str, float],
    search_ranges: dict | None,
) -> dict[str, tuple[float, float] | float]:
    if search_ranges is not None:
        return dict(search_ranges)
    return {
        "C10_nm": (-400.0, 400.0),
        "C12_nm": (0.0, 100.0),
        "phi12_deg": (-90.0, 90.0),
    }


def _suggest_or_fixed(trial, ranges: dict, key: str, default: float) -> float:
    value = ranges.get(key, default)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if lo == hi:
            return lo
        return float(trial.suggest_float(key, lo, hi))
    return float(value)


def _fft2_hermitian(mx, real_stack):
    """Return the nonredundant FFT half-plane for a real BF stack."""
    rfft2 = getattr(mx.fft, "rfft2", None)
    if rfft2 is not None:
        return rfft2(real_stack)
    full = mx.fft.fft2(real_stack)
    return full[:, :, : int(real_stack.shape[-1]) // 2 + 1]


def _expand_hermitian_mx(mx, g_qk, full_cols: int):
    """Expand an MLX Hermitian half-plane stack to a full Fourier grid."""
    shape = tuple(int(v) for v in g_qk.shape)
    if len(shape) < 3:
        raise ValueError(f"Expected at least 3D G_qk, got shape {shape}.")
    full_cols = int(full_cols)
    if shape[-1] == full_cols:
        return g_qk
    expected_cols = full_cols // 2 + 1
    if shape[-1] != expected_cols:
        raise ValueError(
            f"Expected Hermitian G_qk with {expected_cols} columns or full "
            f"{full_cols} columns, got shape {shape}."
        )
    n_rows = int(shape[-2])
    mirror_rows = mx.array(((-np.arange(n_rows)) % n_rows).astype(np.int32))
    mirror_cols = mx.array(
        np.arange(full_cols - expected_cols, 0, -1, dtype=np.int32)
    )
    mirrored_rows = mx.take(g_qk, mirror_rows, axis=-2)
    mirrored = mx.take(mirrored_rows, mirror_cols, axis=-1)
    return mx.concatenate([g_qk, mx.conjugate(mirrored)], axis=-1)


def _ifft2_chunked(mx, fourier_stack):
    """Run a chunked 2D inverse FFT with the faster MLX row-column schedule."""
    row_ifft = mx.fft.ifft(fourier_stack, axis=-1)
    return mx.fft.ifft(row_ifft, axis=-2)


@lru_cache(maxsize=16)
def _phase_sums_kernel(batch: int, chunk: int, ny: int, nx: int):
    mx = _require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        uint total = BATCH * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / PLANE;
        uint pixel = elem - batch * PLANE;
        size_t base = ((size_t)batch * (size_t)CHUNK * (size_t)PLANE) + (size_t)pixel;
        float s = 0.0f;
        float sq = 0.0f;
        for (uint bf = 0; bf < CHUNK; ++bf) {{
            auto z = obj[base + (size_t)bf * (size_t)PLANE];
            float a = metal::atan2(z.imag, z.real);
            s += a;
            sq += a * a;
        }}
        sum_out[elem] = s;
        sumsq_out[elem] = sq;
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_sums_n{int(batch)}_b{int(chunk)}_{int(ny)}_{int(nx)}",
        input_names=["obj"],
        output_names=["sum_out", "sumsq_out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _phase_sums_from_complex(mx, obj_chunk):
    """Metal fused atan2/sum/sumsq over BF pixels for a chunked object stack."""
    shape = tuple(int(x) for x in obj_chunk.shape)
    if len(shape) == 3:
        chunk, ny, nx = shape
        obj = obj_chunk[None, :, :, :]
        squeeze = True
        batch = 1
    elif len(shape) == 4:
        batch, chunk, ny, nx = shape
        obj = obj_chunk
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D object chunk, got shape {shape}.")
    kernel = _phase_sums_kernel(int(batch), int(chunk), int(ny), int(nx))
    outputs = kernel(
        inputs=[obj],
        template=[],
        grid=(int(batch) * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(int(batch), int(ny), int(nx)), (int(batch), int(ny), int(nx))],
        output_dtypes=[mx.float32, mx.float32],
    )
    if squeeze:
        return outputs[0][0], outputs[1][0]
    return outputs[0], outputs[1]


@lru_cache(maxsize=16)
def _phase_sum_kernel(batch: int, chunk: int, ny: int, nx: int):
    mx = _require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        uint total = BATCH * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / PLANE;
        uint pixel = elem - batch * PLANE;
        size_t base = ((size_t)batch * (size_t)CHUNK * (size_t)PLANE) + (size_t)pixel;
        float s = 0.0f;
        for (uint bf = 0; bf < CHUNK; ++bf) {{
            auto z = obj[base + (size_t)bf * (size_t)PLANE];
            s += metal::atan2(z.imag, z.real);
        }}
        sum_out[elem] = s;
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_sum_n{int(batch)}_b{int(chunk)}_{int(ny)}_{int(nx)}",
        input_names=["obj"],
        output_names=["sum_out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _phase_sum_from_complex(mx, obj_chunk):
    """Metal fused atan2/sum over BF pixels for a chunked object stack."""
    shape = tuple(int(x) for x in obj_chunk.shape)
    if len(shape) == 3:
        chunk, ny, nx = shape
        obj = obj_chunk[None, :, :, :]
        squeeze = True
        batch = 1
    elif len(shape) == 4:
        batch, chunk, ny, nx = shape
        obj = obj_chunk
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D object chunk, got shape {shape}.")
    kernel = _phase_sum_kernel(int(batch), int(chunk), int(ny), int(nx))
    output = kernel(
        inputs=[obj],
        template=[],
        grid=(int(batch) * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(int(batch), int(ny), int(nx))],
        output_dtypes=[mx.float32],
    )[0]
    if squeeze:
        return output[0]
    return output


def _twiddle_512(mx):
    """Return IDFT twiddles for 512-point custom MPS FFT kernels."""
    return _twiddle_n(mx, 512)


def _twiddle_n(mx, n: int):
    """Return IDFT twiddles for custom power-of-two MPS FFT kernels."""
    n = int(n)
    return mx.array(
        np.exp(2j * np.pi * np.arange(n, dtype=np.float32) / n).astype(
            np.complex64
        )
    )


@lru_cache(maxsize=1)
def _twiddle_512_metal_header() -> str:
    """Return the exact complex64 twiddle bits as a Metal constant table."""
    values = np.exp(
        2j * np.pi * np.arange(512, dtype=np.float32) / 512
    ).astype(np.complex64)
    bits = values.view(np.uint32).reshape(512, 2)
    entries = ",".join(
        f"uint2(0x{int(real):08x}u,0x{int(imag):08x}u)"
        for real, imag in bits
    )
    return f"constant uint2 SSB_TWIDDLE_512[512] = {{{entries}}};"


def _small_fft_macros(n: int) -> tuple[str, str, int, bool]:
    """Return Metal digit-reversal macros for fused radix-4 IFFTs."""
    from .kernels import get_fft_config

    config = get_fft_config(n)
    if config.specialized:
        raise ValueError(
            f"MPS {config.size}x{config.size} uses its specialized FFT path."
        )
    return (
        config.digit_reverse_define,
        config.digit_reverse_undef,
        config.radix4_max,
        config.has_final_radix2,
    )


@lru_cache(maxsize=32)
def _phase_cols_small_reduced_kernel(
    n: int,
    num_bf: int,
    k_bf: int,
    compute_loss: bool,
    cols_per_group: int = 4,
    batch: int = 1,
):
    mx = _require_mlx()
    n = int(n)
    t = n // 4
    half = n // 2
    cols_per_group = max(1, int(cols_per_group))
    define_rev, undef_rev, radix4_max, has_final = _small_fft_macros(n)
    loss_decl = (
        "float sq0 = 0.0f; float sq1 = 0.0f; "
        "float sq2 = 0.0f; float sq3 = 0.0f;"
        if compute_loss else ""
    )
    loss_accum = (
        "sq0 += p0 * p0; sq1 += p1 * p1; "
        "sq2 += p2 * p2; sq3 += p3 * p3;"
        if compute_loss else ""
    )
    loss_output = (
        f"sumsq_tile[(((size_t)candidate * (size_t)GROUPS + (size_t)group) * {n}u + (size_t)col) * {t}u + tid] = "
        "sq0 + sq1 + sq2 + sq3;"
        if compute_loss else ""
    )
    final_stage = ""
    if has_final:
        final_stage = f"""
            uint j0 = tid;
            uint j1 = tid + {t}u;
            float2 a0 = srow[j0];
            float2 b0 = CMUL(TW(j0), srow[j0 + {half}u]);
            float2 a1 = srow[j1];
            float2 b1 = CMUL(TW(j1), srow[j1 + {half}u]);
            srow[j0] = CADD(a0, b0);
            srow[j0 + {half}u] = CSUB(a0, b0);
            srow[j1] = CADD(a1, b1);
            srow[j1 + {half}u] = CSUB(a1, b1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    output_names = ["sum_out", "sumsq_tile"] if compute_loss else ["sum_out"]
    name_suffix = "scalar" if compute_loss else "sum"
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        {define_rev}

        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint BATCH = {int(batch)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint N = {n}u;
        constexpr uint T = {t}u;
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        uint candidate = z / GROUPS;
        uint group = z - candidate * GROUPS;
        if (tid >= T || local_col >= {cols_per_group}u || col >= N || group >= GROUPS || candidate >= BATCH) {{
            return;
        }}

        threadgroup float2 shared_cols[{cols_per_group}][{n}];
        threadgroup float2* srow = &shared_cols[local_col][0];

        uint pos0 = tid;
        uint pos1 = tid + T;
        uint pos2 = tid + 2u * T;
        uint pos3 = tid + 3u * T;
        uint rev0 = DIGITREVN(pos0);
        uint rev1 = DIGITREVN(pos1);
        uint rev2 = DIGITREVN(pos2);
        uint rev3 = DIGITREVN(pos3);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        {loss_decl}

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            size_t base = (((size_t)candidate * (size_t)NUM_BF + (size_t)bf)
                * (size_t)N * (size_t)N) + (size_t)col;
            auto z0 = row_ifft[base + (size_t)pos0 * (size_t)N];
            auto z1 = row_ifft[base + (size_t)pos1 * (size_t)N];
            auto z2 = row_ifft[base + (size_t)pos2 * (size_t)N];
            auto z3 = row_ifft[base + (size_t)pos3 * (size_t)N];
            srow[rev0] = float2(z0.real, z0.imag);
            srow[rev1] = float2(z1.real, z1.imag);
            srow[rev2] = float2(z2.real, z2.imag);
            srow[rev3] = float2(z3.real, z3.imag);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint m = 4u; m <= {radix4_max}u; m <<= 2) {{
                uint quarter = m >> 2;
                uint j = tid % quarter;
                uint k = tid / quarter;
                uint idx0 = k * m + j;
                uint idx1 = idx0 + quarter;
                uint idx2 = idx1 + quarter;
                uint idx3 = idx2 + quarter;
                uint tw = j * (N / m);
                float2 x0 = srow[idx0];
                float2 x1 = CMUL(TW(tw), srow[idx1]);
                float2 x2 = CMUL(TW(tw * 2u), srow[idx2]);
                float2 x3 = CMUL(TW(tw * 3u), srow[idx3]);
                float2 t0 = CADD(x0, x2);
                float2 t1 = CSUB(x0, x2);
                float2 t2 = CADD(x1, x3);
                float2 t3 = CSUB(x1, x3);
                float2 it3 = CMULI(t3);
                srow[idx0] = CADD(t0, t2);
                srow[idx1] = CADD(t1, it3);
                srow[idx2] = CSUB(t0, t2);
                srow[idx3] = CSUB(t1, it3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            {final_stage}

            float2 o0 = srow[pos0];
            float2 o1 = srow[pos1];
            float2 o2 = srow[pos2];
            float2 o3 = srow[pos3];
            float p0 = metal::atan2(o0.y, o0.x);
            float p1 = metal::atan2(o1.y, o1.x);
            float p2 = metal::atan2(o2.y, o2.x);
            float p3 = metal::atan2(o3.y, o3.x);
            sum0 += p0;
            sum1 += p1;
            sum2 += p2;
            sum3 += p3;
            {loss_accum}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        size_t out_base = ((size_t)candidate * (size_t)GROUPS + (size_t)group)
            * (size_t)N * (size_t)N;
        sum_out[out_base + (size_t)pos0 * (size_t)N + (size_t)col] = sum0;
        sum_out[out_base + (size_t)pos1 * (size_t)N + (size_t)col] = sum1;
        sum_out[out_base + (size_t)pos2 * (size_t)N + (size_t)col] = sum2;
        sum_out[out_base + (size_t)pos3 * (size_t)N + (size_t)col] = sum3;
        {loss_output}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        {undef_rev}
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_phase_cols{n}_{name_suffix}_n{int(num_bf)}_"
            f"k{int(k_bf)}_c{int(cols_per_group)}_b{int(batch)}"
        ),
        input_names=["row_ifft", "twiddle"],
        output_names=output_names,
        source=source,
        compile_options={"math_mode": "fast"},
    )


@lru_cache(maxsize=16)
def _phase_cols_small_scalar_pair_kernel(
    n: int,
    num_bf: int,
    k_bf: int,
    cols_per_group: int = 4,
):
    """Share a column threadgroup while retaining scalar candidate FFTs."""
    mx = _require_mlx()
    n = int(n)
    t = n // 4
    half = n // 2
    cols_per_group = max(1, int(cols_per_group))
    define_rev, undef_rev, radix4_max, has_final = _small_fft_macros(n)
    final_stage = ""
    if has_final:
        final_stage = f"""
            for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                threadgroup float2* frow =
                    &shared_cols[local_col][candidate][0];
                uint j0 = tid;
                uint j1 = tid + {t}u;
                float2 a0 = frow[j0];
                float2 b0 = CMUL(TW(j0), frow[j0 + {half}u]);
                float2 a1 = frow[j1];
                float2 b1 = CMUL(TW(j1), frow[j1 + {half}u]);
                frow[j0] = CADD(a0, b0);
                frow[j0 + {half}u] = CSUB(a0, b0);
                frow[j1] = CADD(a1, b1);
                frow[j1 + {half}u] = CSUB(a1, b1);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        {define_rev}

        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint N = {n}u;
        constexpr uint T = {t}u;
        constexpr uint PLANE = N * N;
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        if (tid >= T || local_col >= {cols_per_group}u || col >= N || group >= GROUPS) {{
            return;
        }}

        threadgroup float2 shared_cols[{cols_per_group}][2][{n}];
        uint pos0 = tid;
        uint pos1 = tid + T;
        uint pos2 = tid + 2u * T;
        uint pos3 = tid + 3u * T;
        uint rev0 = DIGITREVN(pos0);
        uint rev1 = DIGITREVN(pos1);
        uint rev2 = DIGITREVN(pos2);
        uint rev3 = DIGITREVN(pos3);
        float sum0[2] = {{0.0f, 0.0f}};
        float sum1[2] = {{0.0f, 0.0f}};
        float sum2[2] = {{0.0f, 0.0f}};
        float sum3[2] = {{0.0f, 0.0f}};
        float sq0[2] = {{0.0f, 0.0f}};
        float sq1[2] = {{0.0f, 0.0f}};
        float sq2[2] = {{0.0f, 0.0f}};
        float sq3[2] = {{0.0f, 0.0f}};

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                threadgroup float2* srow =
                    &shared_cols[local_col][candidate][0];
                size_t base = (((size_t)candidate * (size_t)NUM_BF + bf)
                    * (size_t)PLANE) + (size_t)col;
                auto z0 = row_ifft[base + (size_t)pos0 * N];
                auto z1 = row_ifft[base + (size_t)pos1 * N];
                auto z2 = row_ifft[base + (size_t)pos2 * N];
                auto z3 = row_ifft[base + (size_t)pos3 * N];
                srow[rev0] = float2(z0.real, z0.imag);
                srow[rev1] = float2(z1.real, z1.imag);
                srow[rev2] = float2(z2.real, z2.imag);
                srow[rev3] = float2(z3.real, z3.imag);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint m = 4u; m <= {radix4_max}u; m <<= 2) {{
                for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                    threadgroup float2* srow =
                        &shared_cols[local_col][candidate][0];
                    uint quarter = m >> 2;
                    uint j = tid % quarter;
                    uint k = tid / quarter;
                    uint idx0 = k * m + j;
                    uint idx1 = idx0 + quarter;
                    uint idx2 = idx1 + quarter;
                    uint idx3 = idx2 + quarter;
                    uint tw = j * (N / m);
                    float2 x0 = srow[idx0];
                    float2 x1 = CMUL(TW(tw), srow[idx1]);
                    float2 x2 = CMUL(TW(tw * 2u), srow[idx2]);
                    float2 x3 = CMUL(TW(tw * 3u), srow[idx3]);
                    float2 t0 = CADD(x0, x2);
                    float2 t1 = CSUB(x0, x2);
                    float2 t2 = CADD(x1, x3);
                    float2 t3 = CSUB(x1, x3);
                    float2 it3 = CMULI(t3);
                    srow[idx0] = CADD(t0, t2);
                    srow[idx1] = CADD(t1, it3);
                    srow[idx2] = CSUB(t0, t2);
                    srow[idx3] = CSUB(t1, it3);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            {final_stage}

            for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                threadgroup float2* srow =
                    &shared_cols[local_col][candidate][0];
                float2 o0 = srow[pos0];
                float2 o1 = srow[pos1];
                float2 o2 = srow[pos2];
                float2 o3 = srow[pos3];
                float p0 = metal::atan2(o0.y, o0.x);
                float p1 = metal::atan2(o1.y, o1.x);
                float p2 = metal::atan2(o2.y, o2.x);
                float p3 = metal::atan2(o3.y, o3.x);
                sum0[candidate] += p0; sum1[candidate] += p1;
                sum2[candidate] += p2; sum3[candidate] += p3;
                sq0[candidate] += p0 * p0; sq1[candidate] += p1 * p1;
                sq2[candidate] += p2 * p2; sq3[candidate] += p3 * p3;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        for (uint candidate = 0u; candidate < 2u; ++candidate) {{
            size_t out_base = ((size_t)candidate * GROUPS + group) * PLANE;
            sum_out[out_base + (size_t)pos0 * N + col] = sum0[candidate];
            sum_out[out_base + (size_t)pos1 * N + col] = sum1[candidate];
            sum_out[out_base + (size_t)pos2 * N + col] = sum2[candidate];
            sum_out[out_base + (size_t)pos3 * N + col] = sum3[candidate];
            size_t sq_base = (((size_t)candidate * GROUPS + group) * N + col)
                * T + tid;
            sumsq_tile[sq_base] = sq0[candidate] + sq1[candidate]
                + sq2[candidate] + sq3[candidate];
        }}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        {undef_rev}
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_phase_cols{n}_scalar_pair_n{int(num_bf)}_"
            f"k{int(k_bf)}_c{int(cols_per_group)}"
        ),
        input_names=["row_ifft", "twiddle"],
        output_names=["sum_out", "sumsq_tile"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _phase_cols_small_sum_from_row_ifft(mx, row_ifft, *, k_bf: int = 32):
    """Fuse 128/256/1024-column IFFT and phase accumulation without loss work."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if (
        len(shape) != 3
        or shape[-2] != shape[-1]
        or shape[-1] not in (128, 256, 1024)
    ):
        raise ValueError(
            "Expected row-IFFT chunk shape "
            f"(BF, 128/256/1024, 128/256/1024), got {shape}."
        )
    num_bf = int(shape[0])
    n = int(shape[-1])
    t = n // 4
    k_bf = max(1, int(k_bf))
    cols_per_group = 8 if n <= 256 else 4
    groups = (num_bf + k_bf - 1) // k_bf
    kernel = _phase_cols_small_reduced_kernel(
        n,
        num_bf,
        k_bf,
        False,
        cols_per_group,
    )
    partial_sum = kernel(
        inputs=[row_ifft, _twiddle_n(mx, n)],
        template=[],
        grid=(t, n, groups),
        threadgroup=(t, cols_per_group, 1),
        output_shapes=[(groups, n, n)],
        output_dtypes=[mx.float32],
    )[0]
    if groups == 1:
        return partial_sum[0]
    return mx.sum(partial_sum, axis=0)


def _phase_cols_small_scalar_loss_from_row_ifft(mx, row_ifft, *, k_bf: int = 32):
    """Fuse 128/256/1024-column IFFT, phase sum, and scalar phase-squared loss."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if (
        len(shape) != 3
        or shape[-2] != shape[-1]
        or shape[-1] not in (128, 256, 1024)
    ):
        raise ValueError(
            "Expected row-IFFT chunk shape "
            f"(BF, 128/256/1024, 128/256/1024), got {shape}."
        )
    num_bf = int(shape[0])
    n = int(shape[-1])
    t = n // 4
    k_bf = max(1, int(k_bf))
    cols_per_group = 4
    groups = (num_bf + k_bf - 1) // k_bf
    kernel = _phase_cols_small_reduced_kernel(
        n,
        num_bf,
        k_bf,
        True,
        cols_per_group,
    )
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, _twiddle_n(mx, n)],
        template=[],
        grid=(t, n, groups),
        threadgroup=(t, cols_per_group, 1),
        output_shapes=[(groups, n, n), (groups, n, t)],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sum = partial_sum[0] if groups == 1 else mx.sum(partial_sum, axis=0)
    return phase_sum, mx.sum(partial_sumsq_tile)


def _phase_cols_small_scalar_loss_batch_from_row_ifft(
    mx,
    row_ifft,
    *,
    k_bf: int = 32,
    packed_pair: bool = False,
):
    """Run exact small-scan column FFT and reductions for candidate batches."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if (
        len(shape) != 4
        or shape[-2] != shape[-1]
        or shape[-1] not in (128, 256)
    ):
        raise ValueError(
            "Expected batched row-IFFT shape (batch, BF, 128/256, 128/256), "
            f"got {shape}."
        )
    batch, num_bf, n, _ = shape
    t = n // 4
    k_bf = max(1, int(k_bf))
    cols_per_group = 4
    groups = (num_bf + k_bf - 1) // k_bf
    if packed_pair:
        if batch != 2:
            raise ValueError("Packed small MPS columns require two candidates.")
        kernel = _phase_cols_small_scalar_pair_kernel(
            n,
            num_bf,
            k_bf,
            cols_per_group,
        )
        grid_z = groups
    else:
        kernel = _phase_cols_small_reduced_kernel(
            n,
            num_bf,
            k_bf,
            True,
            cols_per_group,
            batch,
        )
        grid_z = batch * groups
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, _twiddle_n(mx, n)],
        template=[],
        grid=(t, n, grid_z),
        threadgroup=(t, cols_per_group, 1),
        output_shapes=[
            (batch, groups, n, n),
            (batch, groups, n, t),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    # Keep each candidate's reduction shape identical to the scalar exact path.
    # A batched multi-axis MLX reduction changes the float32 association by an
    # ULP for some 256x256 inputs even though the Metal column outputs match.
    phase_sum = mx.stack(
        [
            partial_sum[candidate, 0]
            if groups == 1
            else mx.sum(partial_sum[candidate], axis=0)
            for candidate in range(batch)
        ]
    )
    phase_sumsq = mx.stack(
        [mx.sum(partial_sumsq_tile[candidate]) for candidate in range(batch)]
    )
    return phase_sum, phase_sumsq


@lru_cache(maxsize=16)
def _phase_cols512_kernel(num_bf: int, k_bf: int):
    mx = _require_mlx()
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        #define BITREV4_8(x) ((((x) & 0x03u) << 6) | (((x) & 0x0Cu) << 2) | (((x) & 0x30u) >> 2) | (((x) & 0xC0u) >> 6))
        #define DIGITREV512(x) ((((x) & 1u) << 8) | BITREV4_8((x) >> 1))

        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        if (tid >= 128u || local_col >= 4u || col >= 512u || group >= GROUPS) {{
            return;
        }}

        threadgroup float2 shared_cols[4][512];
        threadgroup float2* srow = &shared_cols[local_col][0];

        uint pos0 = tid;
        uint pos1 = tid + 128u;
        uint pos2 = tid + 256u;
        uint pos3 = tid + 384u;
        uint rev0 = DIGITREV512(pos0);
        uint rev1 = DIGITREV512(pos1);
        uint rev2 = DIGITREV512(pos2);
        uint rev3 = DIGITREV512(pos3);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sq0 = 0.0f;
        float sq1 = 0.0f;
        float sq2 = 0.0f;
        float sq3 = 0.0f;

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            size_t base = ((size_t)bf * 512u * 512u) + (size_t)col;
            auto z0 = row_ifft[base + (size_t)pos0 * 512u];
            auto z1 = row_ifft[base + (size_t)pos1 * 512u];
            auto z2 = row_ifft[base + (size_t)pos2 * 512u];
            auto z3 = row_ifft[base + (size_t)pos3 * 512u];
            srow[rev0] = float2(z0.real, z0.imag);
            srow[rev1] = float2(z1.real, z1.imag);
            srow[rev2] = float2(z2.real, z2.imag);
            srow[rev3] = float2(z3.real, z3.imag);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint m = 4u; m <= 256u; m <<= 2) {{
                uint quarter = m >> 2;
                uint j = tid % quarter;
                uint k = tid / quarter;
                uint idx0 = k * m + j;
                uint idx1 = idx0 + quarter;
                uint idx2 = idx1 + quarter;
                uint idx3 = idx2 + quarter;
                uint tw = j * (512u / m);
                float2 x0 = srow[idx0];
                float2 x1 = CMUL(TW(tw), srow[idx1]);
                float2 x2 = CMUL(TW(tw * 2u), srow[idx2]);
                float2 x3 = CMUL(TW(tw * 3u), srow[idx3]);
                float2 t0 = CADD(x0, x2);
                float2 t1 = CSUB(x0, x2);
                float2 t2 = CADD(x1, x3);
                float2 t3 = CSUB(x1, x3);
                float2 it3 = CMULI(t3);
                srow[idx0] = CADD(t0, t2);
                srow[idx1] = CADD(t1, it3);
                srow[idx2] = CSUB(t0, t2);
                srow[idx3] = CSUB(t1, it3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            uint j0 = tid;
            uint j1 = tid + 128u;
            float2 a0 = srow[j0];
            float2 b0 = CMUL(TW(j0), srow[j0 + 256u]);
            float2 a1 = srow[j1];
            float2 b1 = CMUL(TW(j1), srow[j1 + 256u]);
            srow[j0] = CADD(a0, b0);
            srow[j0 + 256u] = CSUB(a0, b0);
            srow[j1] = CADD(a1, b1);
            srow[j1 + 256u] = CSUB(a1, b1);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float2 o0 = srow[pos0];
            float2 o1 = srow[pos1];
            float2 o2 = srow[pos2];
            float2 o3 = srow[pos3];
            float p0 = metal::atan2(o0.y, o0.x);
            float p1 = metal::atan2(o1.y, o1.x);
            float p2 = metal::atan2(o2.y, o2.x);
            float p3 = metal::atan2(o3.y, o3.x);
            sum0 += p0;
            sum1 += p1;
            sum2 += p2;
            sum3 += p3;
            sq0 += p0 * p0;
            sq1 += p1 * p1;
            sq2 += p2 * p2;
            sq3 += p3 * p3;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        size_t out_base = (size_t)group * 512u * 512u;
        sum_out[out_base + (size_t)pos0 * 512u + (size_t)col] = sum0;
        sum_out[out_base + (size_t)pos1 * 512u + (size_t)col] = sum1;
        sum_out[out_base + (size_t)pos2 * 512u + (size_t)col] = sum2;
        sum_out[out_base + (size_t)pos3 * 512u + (size_t)col] = sum3;
        sumsq_out[out_base + (size_t)pos0 * 512u + (size_t)col] = sq0;
        sumsq_out[out_base + (size_t)pos1 * 512u + (size_t)col] = sq1;
        sumsq_out[out_base + (size_t)pos2 * 512u + (size_t)col] = sq2;
        sumsq_out[out_base + (size_t)pos3 * 512u + (size_t)col] = sq3;

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef BITREV4_8
        #undef DIGITREV512
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_cols512_n{int(num_bf)}_k{int(k_bf)}",
        input_names=["row_ifft", "twiddle"],
        output_names=["sum_out", "sumsq_out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _phase_cols512_from_row_ifft(mx, row_ifft, *, k_bf: int = 32):
    """Fuse 512-column IFFT and phase/loss accumulation for one BF chunk."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if len(shape) != 3 or shape[-2:] != (512, 512):
        raise ValueError(f"Expected row-IFFT chunk shape (BF, 512, 512), got {shape}.")
    num_bf = int(shape[0])
    k_bf = max(1, int(k_bf))
    groups = (num_bf + k_bf - 1) // k_bf
    kernel = _phase_cols512_kernel(num_bf, k_bf)
    partial_sum, partial_sumsq = kernel(
        inputs=[row_ifft, _twiddle_512(mx)],
        template=[],
        grid=(128, 512, groups),
        threadgroup=(128, 4, 1),
        output_shapes=[(groups, 512, 512), (groups, 512, 512)],
        output_dtypes=[mx.float32, mx.float32],
    )
    if groups == 1:
        return partial_sum[0], partial_sumsq[0]
    return mx.sum(partial_sum, axis=0), mx.sum(partial_sumsq, axis=0)


@lru_cache(maxsize=32)
def _phase_cols512_reduced_kernel(num_bf: int, k_bf: int, compute_loss: bool):
    mx = _require_mlx()
    loss_decl = (
        "float sq0 = 0.0f; float sq1 = 0.0f; "
        "float sq2 = 0.0f; float sq3 = 0.0f;"
        if compute_loss else ""
    )
    loss_accum = (
        "sq0 += p0 * p0; sq1 += p1 * p1; "
        "sq2 += p2 * p2; sq3 += p3 * p3;"
        if compute_loss else ""
    )
    loss_output = (
        "sumsq_tile[((size_t)group * 512u + (size_t)col) * 128u + tid] = "
        "sq0 + sq1 + sq2 + sq3;"
        if compute_loss else ""
    )
    output_names = ["sum_out", "sumsq_tile"] if compute_loss else ["sum_out"]
    name_suffix = "scalar" if compute_loss else "sum"
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        #define BITREV4_8(x) ((((x) & 0x03u) << 6) | (((x) & 0x0Cu) << 2) | (((x) & 0x30u) >> 2) | (((x) & 0xC0u) >> 6))
        #define DIGITREV512(x) ((((x) & 1u) << 8) | BITREV4_8((x) >> 1))

        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        if (tid >= 128u || local_col >= 4u || col >= 512u || group >= GROUPS) {{
            return;
        }}

        threadgroup float2 shared_cols[4][512];
        threadgroup float2* srow = &shared_cols[local_col][0];

        uint pos0 = tid;
        uint pos1 = tid + 128u;
        uint pos2 = tid + 256u;
        uint pos3 = tid + 384u;
        uint rev0 = DIGITREV512(pos0);
        uint rev1 = DIGITREV512(pos1);
        uint rev2 = DIGITREV512(pos2);
        uint rev3 = DIGITREV512(pos3);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        {loss_decl}

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            size_t base = ((size_t)bf * 512u * 512u) + (size_t)col;
            auto z0 = row_ifft[base + (size_t)pos0 * 512u];
            auto z1 = row_ifft[base + (size_t)pos1 * 512u];
            auto z2 = row_ifft[base + (size_t)pos2 * 512u];
            auto z3 = row_ifft[base + (size_t)pos3 * 512u];
            srow[rev0] = float2(z0.real, z0.imag);
            srow[rev1] = float2(z1.real, z1.imag);
            srow[rev2] = float2(z2.real, z2.imag);
            srow[rev3] = float2(z3.real, z3.imag);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint m = 4u; m <= 256u; m <<= 2) {{
                uint quarter = m >> 2;
                uint j = tid % quarter;
                uint k = tid / quarter;
                uint idx0 = k * m + j;
                uint idx1 = idx0 + quarter;
                uint idx2 = idx1 + quarter;
                uint idx3 = idx2 + quarter;
                uint tw = j * (512u / m);
                float2 x0 = srow[idx0];
                float2 x1 = CMUL(TW(tw), srow[idx1]);
                float2 x2 = CMUL(TW(tw * 2u), srow[idx2]);
                float2 x3 = CMUL(TW(tw * 3u), srow[idx3]);
                float2 t0 = CADD(x0, x2);
                float2 t1 = CSUB(x0, x2);
                float2 t2 = CADD(x1, x3);
                float2 t3 = CSUB(x1, x3);
                float2 it3 = CMULI(t3);
                srow[idx0] = CADD(t0, t2);
                srow[idx1] = CADD(t1, it3);
                srow[idx2] = CSUB(t0, t2);
                srow[idx3] = CSUB(t1, it3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            uint j0 = tid;
            uint j1 = tid + 128u;
            float2 a0 = srow[j0];
            float2 b0 = CMUL(TW(j0), srow[j0 + 256u]);
            float2 a1 = srow[j1];
            float2 b1 = CMUL(TW(j1), srow[j1 + 256u]);
            srow[j0] = CADD(a0, b0);
            srow[j0 + 256u] = CSUB(a0, b0);
            srow[j1] = CADD(a1, b1);
            srow[j1 + 256u] = CSUB(a1, b1);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float2 o0 = srow[pos0];
            float2 o1 = srow[pos1];
            float2 o2 = srow[pos2];
            float2 o3 = srow[pos3];
            float p0 = metal::atan2(o0.y, o0.x);
            float p1 = metal::atan2(o1.y, o1.x);
            float p2 = metal::atan2(o2.y, o2.x);
            float p3 = metal::atan2(o3.y, o3.x);
            sum0 += p0;
            sum1 += p1;
            sum2 += p2;
            sum3 += p3;
            {loss_accum}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        size_t out_base = (size_t)group * 512u * 512u;
        sum_out[out_base + (size_t)pos0 * 512u + (size_t)col] = sum0;
        sum_out[out_base + (size_t)pos1 * 512u + (size_t)col] = sum1;
        sum_out[out_base + (size_t)pos2 * 512u + (size_t)col] = sum2;
        sum_out[out_base + (size_t)pos3 * 512u + (size_t)col] = sum3;
        {loss_output}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef BITREV4_8
        #undef DIGITREV512
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_cols512_{name_suffix}_n{int(num_bf)}_k{int(k_bf)}",
        input_names=["row_ifft", "twiddle"],
        output_names=output_names,
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _phase_cols512_sum_from_row_ifft(
    mx,
    row_ifft,
    *,
    k_bf: int = 32,
    active_bf=None,
):
    """Fuse masked radix-8 column IFFT and phase accumulation."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if len(shape) != 3 or shape[-2:] != (512, 512):
        raise ValueError(f"Expected row-IFFT chunk shape (BF, 512, 512), got {shape}.")
    batch_sum, _batch_sumsq = _phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        row_ifft[None, ...],
        k_bf=k_bf,
        active_bf=active_bf,
    )
    return batch_sum[0]


def _phase_cols512_scalar_loss_from_row_ifft(mx, row_ifft, *, k_bf: int = 32):
    """Fuse 512-column IFFT, phase sum, and scalar phase-squared loss."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if len(shape) != 3 or shape[-2:] != (512, 512):
        raise ValueError(f"Expected row-IFFT chunk shape (BF, 512, 512), got {shape}.")
    num_bf = int(shape[0])
    k_bf = max(1, int(k_bf))
    groups = (num_bf + k_bf - 1) // k_bf
    kernel = _phase_cols512_reduced_kernel(num_bf, k_bf, True)
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, _twiddle_512(mx)],
        template=[],
        grid=(128, 512, groups),
        threadgroup=(128, 4, 1),
        output_shapes=[(groups, 512, 512), (groups, 512, 128)],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sum = partial_sum[0] if groups == 1 else mx.sum(partial_sum, axis=0)
    return phase_sum, mx.sum(partial_sumsq_tile)


@lru_cache(maxsize=32)
def _phase_cols512_reduced_batch_kernel(
    batch: int,
    num_bf: int,
    k_bf: int,
):
    mx = _require_mlx()
    row_ifft_load = """
            size_t base = (
                ((size_t)batch * (size_t)NUM_BF + (size_t)bf) * 512u * 512u
                + (size_t)col
            );
            auto z0 = row_ifft[base + (size_t)pos0 * 512u];
            auto z1 = row_ifft[base + (size_t)pos1 * 512u];
            auto z2 = row_ifft[base + (size_t)pos2 * 512u];
            auto z3 = row_ifft[base + (size_t)pos3 * 512u];
        """
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        #define BITREV4_8(x) ((((x) & 0x03u) << 6) | (((x) & 0x0Cu) << 2) | (((x) & 0x30u) >> 2) | (((x) & 0xC0u) >> 6))
        #define DIGITREV512(x) ((((x) & 1u) << 8) | BITREV4_8((x) >> 1))

        constexpr uint BATCH = {int(batch)};
        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        uint batch = z / GROUPS;
        uint group = z - batch * GROUPS;
        if (tid >= 128u || local_col >= 4u || col >= 512u || batch >= BATCH || group >= GROUPS) {{
            return;
        }}

        threadgroup float2 shared_cols[4][512];
        threadgroup float2* srow = &shared_cols[local_col][0];

        uint pos0 = tid;
        uint pos1 = tid + 128u;
        uint pos2 = tid + 256u;
        uint pos3 = tid + 384u;
        uint rev0 = DIGITREV512(pos0);
        uint rev1 = DIGITREV512(pos1);
        uint rev2 = DIGITREV512(pos2);
        uint rev3 = DIGITREV512(pos3);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sq0 = 0.0f;
        float sq1 = 0.0f;
        float sq2 = 0.0f;
        float sq3 = 0.0f;

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            {row_ifft_load}
            srow[rev0] = float2(z0.real, z0.imag);
            srow[rev1] = float2(z1.real, z1.imag);
            srow[rev2] = float2(z2.real, z2.imag);
            srow[rev3] = float2(z3.real, z3.imag);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint m = 4u; m <= 256u; m <<= 2) {{
                uint quarter = m >> 2;
                uint j = tid % quarter;
                uint k = tid / quarter;
                uint idx0 = k * m + j;
                uint idx1 = idx0 + quarter;
                uint idx2 = idx1 + quarter;
                uint idx3 = idx2 + quarter;
                uint tw = j * (512u / m);
                float2 x0 = srow[idx0];
                float2 x1 = CMUL(TW(tw), srow[idx1]);
                float2 x2 = CMUL(TW(tw * 2u), srow[idx2]);
                float2 x3 = CMUL(TW(tw * 3u), srow[idx3]);
                float2 t0 = CADD(x0, x2);
                float2 t1 = CSUB(x0, x2);
                float2 t2 = CADD(x1, x3);
                float2 t3 = CSUB(x1, x3);
                float2 it3 = CMULI(t3);
                srow[idx0] = CADD(t0, t2);
                srow[idx1] = CADD(t1, it3);
                srow[idx2] = CSUB(t0, t2);
                srow[idx3] = CSUB(t1, it3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            uint j0 = tid;
            uint j1 = tid + 128u;
            float2 a0 = srow[j0];
            float2 b0 = CMUL(TW(j0), srow[j0 + 256u]);
            float2 a1 = srow[j1];
            float2 b1 = CMUL(TW(j1), srow[j1 + 256u]);
            srow[j0] = CADD(a0, b0);
            srow[j0 + 256u] = CSUB(a0, b0);
            srow[j1] = CADD(a1, b1);
            srow[j1 + 256u] = CSUB(a1, b1);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float2 o0 = srow[pos0];
            float2 o1 = srow[pos1];
            float2 o2 = srow[pos2];
            float2 o3 = srow[pos3];
            float p0 = metal::atan2(o0.y, o0.x);
            float p1 = metal::atan2(o1.y, o1.x);
            float p2 = metal::atan2(o2.y, o2.x);
            float p3 = metal::atan2(o3.y, o3.x);
            sum0 += p0;
            sum1 += p1;
            sum2 += p2;
            sum3 += p3;
            sq0 += p0 * p0;
            sq1 += p1 * p1;
            sq2 += p2 * p2;
            sq3 += p3 * p3;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        size_t out_base = (
            ((size_t)batch * (size_t)GROUPS + (size_t)group) * 512u * 512u
        );
        sum_out[out_base + (size_t)pos0 * 512u + (size_t)col] = sum0;
        sum_out[out_base + (size_t)pos1 * 512u + (size_t)col] = sum1;
        sum_out[out_base + (size_t)pos2 * 512u + (size_t)col] = sum2;
        sum_out[out_base + (size_t)pos3 * 512u + (size_t)col] = sum3;
        sumsq_tile[
            (((size_t)batch * (size_t)GROUPS + (size_t)group) * 512u + (size_t)col)
            * 128u + tid
        ] = sq0 + sq1 + sq2 + sq3;

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef BITREV4_8
        #undef DIGITREV512
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_phase_cols512_batch_scalar_n{int(batch)}_"
            f"bf{int(num_bf)}_k{int(k_bf)}"
        ),
        input_names=["row_ifft", "twiddle"],
        output_names=["sum_out", "sumsq_tile"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


@lru_cache(maxsize=32)
def _phase_cols512_radix8_batch_kernel(
    batch: int,
    num_bf: int,
    k_bf: int,
    tiled_input: bool,
    storage_num_bf: int | None = None,
    bf_offset: int = 0,
    bf_ranges: tuple[tuple[int, int], ...] | None = None,
):
    """Build the 64-thread radix-8 exact phase/loss column kernel."""
    mx = _require_mlx()
    storage_num_bf = int(num_bf if storage_num_bf is None else storage_num_bf)
    bf_offset = int(bf_offset)
    range_setup = """
        uint cand = z / GROUPS;
        uint group = z - cand * GROUPS;
        if (tid >= 64u || local_col >= 8u || col >= 512u || cand >= BATCH || group >= GROUPS) {
            return;
        }
        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
    """
    range_name = ""
    if bf_ranges is not None:
        bf_ranges = tuple((int(start), int(stop)) for start, stop in bf_ranges)
        invalid_bounds = any(
            start < 0 or stop <= start or stop > storage_num_bf
            for start, stop in bf_ranges
        )
        overlapping = any(
            start < previous_stop
            for (_previous_start, previous_stop), (start, _stop) in zip(
                bf_ranges, bf_ranges[1:]
            )
        )
        if not bf_ranges or invalid_bounds or overlapping:
            raise ValueError(
                "Packed column BF ranges must be non-empty, ordered, "
                f"non-overlapping, and within [0, {storage_num_bf})."
            )
        start_expr = f"{bf_ranges[-1][0]}u"
        stop_expr = f"{bf_ranges[-1][1]}u"
        for index in range(len(bf_ranges) - 2, -1, -1):
            start_expr = (
                f"group == {index}u ? {bf_ranges[index][0]}u : ({start_expr})"
            )
            stop_expr = (
                f"group == {index}u ? {bf_ranges[index][1]}u : ({stop_expr})"
            )
        range_setup = f"""
        uint cand = z / GROUPS;
        uint group = z - cand * GROUPS;
        if (tid >= 64u || local_col >= 8u || col >= 512u || cand >= BATCH || group >= GROUPS) {{
            return;
        }}
        uint bf_start = {start_expr};
        uint bf_end = {stop_expr};
        """
        range_name = "_r" + "_".join(
            f"{start}x{stop}" for start, stop in bf_ranges
        )
    groups_define = (
        str(len(bf_ranges))
        if bf_ranges is not None
        else "(NUM_BF + K_BF - 1u) / K_BF"
    )
    row_ifft_load = """
            size_t base = (
                ((size_t)cand * (size_t)STORAGE_NUM_BF
                    + (size_t)BF_OFFSET + (size_t)bf)
                * 512u * 512u
                + (size_t)col
            );
            float2 r0=float2(row_ifft[base+(size_t)src0*512u].real,row_ifft[base+(size_t)src0*512u].imag);
            float2 r1=float2(row_ifft[base+(size_t)src1*512u].real,row_ifft[base+(size_t)src1*512u].imag);
            float2 r2=float2(row_ifft[base+(size_t)src2*512u].real,row_ifft[base+(size_t)src2*512u].imag);
            float2 r3=float2(row_ifft[base+(size_t)src3*512u].real,row_ifft[base+(size_t)src3*512u].imag);
            float2 r4=float2(row_ifft[base+(size_t)src4*512u].real,row_ifft[base+(size_t)src4*512u].imag);
            float2 r5=float2(row_ifft[base+(size_t)src5*512u].real,row_ifft[base+(size_t)src5*512u].imag);
            float2 r6=float2(row_ifft[base+(size_t)src6*512u].real,row_ifft[base+(size_t)src6*512u].imag);
            float2 r7=float2(row_ifft[base+(size_t)src7*512u].real,row_ifft[base+(size_t)src7*512u].imag);
    """
    if tiled_input:
        row_ifft_load = """
            size_t plane = (
                ((size_t)cand * (size_t)STORAGE_NUM_BF
                    + (size_t)BF_OFFSET + (size_t)bf)
                * 512u * 512u
            );
            size_t tile = plane + (size_t)(col >> 3) * 4096u + (col & 7u);
            size_t i0=tile+(size_t)src0*8u;
            size_t i1=tile+(size_t)src1*8u;
            size_t i2=tile+(size_t)src2*8u;
            size_t i3=tile+(size_t)src3*8u;
            size_t i4=tile+(size_t)src4*8u;
            size_t i5=tile+(size_t)src5*8u;
            size_t i6=tile+(size_t)src6*8u;
            size_t i7=tile+(size_t)src7*8u;
            float2 r0=float2(row_ifft[i0].real,row_ifft[i0].imag);
            float2 r1=float2(row_ifft[i1].real,row_ifft[i1].imag);
            float2 r2=float2(row_ifft[i2].real,row_ifft[i2].imag);
            float2 r3=float2(row_ifft[i3].real,row_ifft[i3].imag);
            float2 r4=float2(row_ifft[i4].real,row_ifft[i4].imag);
            float2 r5=float2(row_ifft[i5].real,row_ifft[i5].imag);
            float2 r6=float2(row_ifft[i6].real,row_ifft[i6].imag);
            float2 r7=float2(row_ifft[i7].real,row_ifft[i7].imag);
        """
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) as_type<float2>(SSB_TWIDDLE_512[(i) & 511u])
        #define OCTREV512(n) ((((n) & 7u) << 6) | (((n) & 56u)) | ((n) >> 6))
        #define W8_1(a) float2(0.70710678118654752f * ((a).x - (a).y), 0.70710678118654752f * ((a).x + (a).y))
        #define W8_3(a) float2(0.70710678118654752f * (-(a).x - (a).y), 0.70710678118654752f * ((a).x - (a).y))
        #define RADIX8(x0,x1,x2,x3,x4,x5,x6,x7) {{ \
            float2 a0=(x0), a1=(x4), a2=(x2), a3=(x6); \
            float2 a4=(x1), a5=(x5), a6=(x3), a7=(x7); \
            float2 t0=CADD(a0,a1), t1=CSUB(a0,a1); \
            float2 t2=CADD(a2,a3), t3=CSUB(a2,a3); \
            float2 t4=CADD(a4,a5), t5=CSUB(a4,a5); \
            float2 t6=CADD(a6,a7), t7=CSUB(a6,a7); \
            float2 u0=CADD(t0,t2), u2=CSUB(t0,t2); \
            float2 it3=CMULI(t3), u1=CADD(t1,it3), u3=CSUB(t1,it3); \
            float2 u4=CADD(t4,t6), u6=CSUB(t4,t6); \
            float2 it7=CMULI(t7), u5=CADD(t5,it7), u7=CSUB(t5,it7); \
            float2 w1u5=W8_1(u5), w3u7=W8_3(u7), iu6=CMULI(u6); \
            (x0)=CADD(u0,u4); (x4)=CSUB(u0,u4); \
            (x1)=CADD(u1,w1u5); (x5)=CSUB(u1,w1u5); \
            (x2)=CADD(u2,iu6); (x6)=CSUB(u2,iu6); \
            (x3)=CADD(u3,w3u7); (x7)=CSUB(u3,w3u7); \
        }}

        constexpr uint BATCH = {int(batch)}u;
        constexpr uint NUM_BF = {int(num_bf)}u;
        constexpr uint STORAGE_NUM_BF = {storage_num_bf}u;
        constexpr uint BF_OFFSET = {bf_offset}u;
        constexpr uint K_BF = {int(k_bf)}u;
        constexpr uint GROUPS = {groups_define};
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        {range_setup}

        threadgroup float2 shared_cols[8][512];
        threadgroup float2* s = &shared_cols[local_col][0];
        float sum0=0.0f, sum1=0.0f, sum2=0.0f, sum3=0.0f;
        float sum4=0.0f, sum5=0.0f, sum6=0.0f, sum7=0.0f;
        float sq0=0.0f, sq1=0.0f, sq2=0.0f, sq3=0.0f;
        float sq4=0.0f, sq5=0.0f, sq6=0.0f, sq7=0.0f;
        uint logical = tid * 8u;
        uint src0=OCTREV512(logical), src1=OCTREV512(logical+1u);
        uint src2=OCTREV512(logical+2u), src3=OCTREV512(logical+3u);
        uint src4=OCTREV512(logical+4u), src5=OCTREV512(logical+5u);
        uint src6=OCTREV512(logical+6u), src7=OCTREV512(logical+7u);
        uint s2 = tid & 7u;
        uint base2 = (tid >> 3) * 64u + s2;

        for (uint bf=bf_start; bf<bf_end; ++bf) {{
            if (active_bf[BF_OFFSET + bf] == 0u) {{
                continue;
            }}
            {row_ifft_load}
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            s[logical]=r0; s[logical+1u]=r1; s[logical+2u]=r2; s[logical+3u]=r3;
            s[logical+4u]=r4; s[logical+5u]=r5; s[logical+6u]=r6; s[logical+7u]=r7;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            r0=s[base2]; r1=CMUL(TW(s2*8u),s[base2+8u]);
            r2=CMUL(TW(s2*16u),s[base2+16u]); r3=CMUL(TW(s2*24u),s[base2+24u]);
            r4=CMUL(TW(s2*32u),s[base2+32u]); r5=CMUL(TW(s2*40u),s[base2+40u]);
            r6=CMUL(TW(s2*48u),s[base2+48u]); r7=CMUL(TW(s2*56u),s[base2+56u]);
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            s[base2]=r0; s[base2+8u]=r1; s[base2+16u]=r2; s[base2+24u]=r3;
            s[base2+32u]=r4; s[base2+40u]=r5; s[base2+48u]=r6; s[base2+56u]=r7;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            r0=s[tid]; r1=CMUL(TW(tid),s[tid+64u]);
            r2=CMUL(TW(tid*2u),s[tid+128u]); r3=CMUL(TW(tid*3u),s[tid+192u]);
            r4=CMUL(TW(tid*4u),s[tid+256u]); r5=CMUL(TW(tid*5u),s[tid+320u]);
            r6=CMUL(TW(tid*6u),s[tid+384u]); r7=CMUL(TW(tid*7u),s[tid+448u]);
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            float p0=metal::atan2(r0.y,r0.x), p1=metal::atan2(r1.y,r1.x);
            float p2=metal::atan2(r2.y,r2.x), p3=metal::atan2(r3.y,r3.x);
            float p4=metal::atan2(r4.y,r4.x), p5=metal::atan2(r5.y,r5.x);
            float p6=metal::atan2(r6.y,r6.x), p7=metal::atan2(r7.y,r7.x);
            sum0+=p0; sum1+=p1; sum2+=p2; sum3+=p3;
            sum4+=p4; sum5+=p5; sum6+=p6; sum7+=p7;
            sq0+=p0*p0; sq1+=p1*p1; sq2+=p2*p2; sq3+=p3*p3;
            sq4+=p4*p4; sq5+=p5*p5; sq6+=p6*p6; sq7+=p7*p7;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        size_t out_base=((size_t)cand*(size_t)GROUPS+(size_t)group)*512u*512u+(size_t)col;
        sum_out[out_base+(size_t)tid*512u]=sum0;
        sum_out[out_base+(size_t)(tid+64u)*512u]=sum1;
        sum_out[out_base+(size_t)(tid+128u)*512u]=sum2;
        sum_out[out_base+(size_t)(tid+192u)*512u]=sum3;
        sum_out[out_base+(size_t)(tid+256u)*512u]=sum4;
        sum_out[out_base+(size_t)(tid+320u)*512u]=sum5;
        sum_out[out_base+(size_t)(tid+384u)*512u]=sum6;
        sum_out[out_base+(size_t)(tid+448u)*512u]=sum7;
        sumsq_tile[(((size_t)cand*(size_t)GROUPS+(size_t)group)*512u+(size_t)col)*64u+tid]
            =sq0+sq1+sq2+sq3+sq4+sq5+sq6+sq7;

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef OCTREV512
        #undef W8_1
        #undef W8_3
        #undef RADIX8
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_phase_cols512_radix8_batch_n{batch}_bf{num_bf}_"
            f"k{k_bf}_t{int(tiled_input)}_s{storage_num_bf}_o{bf_offset}{range_name}"
        ),
        input_names=["row_ifft", "active_bf", "twiddle"],
        output_names=["sum_out", "sumsq_tile"],
        source=source,
        header=_twiddle_512_metal_header(),
        compile_options={"math_mode": "fast"},
    )


def _phase_cols512_scalar_loss_batch_from_row_ifft(
    mx,
    row_ifft,
    *,
    k_bf: int = 32,
    active_bf=None,
    tiled_input: bool = False,
    bf_start: int = 0,
    bf_stop: int | None = None,
):
    """Fuse 512-column IFFT and scalar loss for candidate-batched row IFFT."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if len(shape) != 4 or shape[-2:] != (512, 512):
        raise ValueError(
            "Expected row-IFFT chunk shape (batch, BF, 512, 512), "
            f"got {shape}."
        )
    batch, storage_num_bf = int(shape[0]), int(shape[1])
    bf_start = max(0, int(bf_start))
    bf_stop = storage_num_bf if bf_stop is None else int(bf_stop)
    if bf_stop <= bf_start or bf_stop > storage_num_bf:
        raise ValueError(
            f"Invalid BF subrange [{bf_start}, {bf_stop}) for "
            f"storage length {storage_num_bf}."
        )
    num_bf = bf_stop - bf_start
    k_bf = max(1, int(k_bf))
    groups = (num_bf + k_bf - 1) // k_bf
    if active_bf is None:
        active_bf = mx.ones((storage_num_bf,), dtype=mx.uint8)
    kernel = _phase_cols512_radix8_batch_kernel(
        batch,
        num_bf,
        k_bf,
        tiled_input,
        storage_num_bf,
        bf_start,
    )
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, active_bf, _twiddle_512(mx)],
        template=[],
        grid=(64, 512, batch * groups),
        threadgroup=(64, 8, 1),
        output_shapes=[(batch, groups, 512, 512), (batch, groups, 512, 64)],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sum = partial_sum[:, 0, :, :] if groups == 1 else mx.sum(partial_sum, axis=1)
    phase_sumsq = mx.sum(mx.sum(mx.sum(partial_sumsq_tile, axis=3), axis=2), axis=1)
    return phase_sum, phase_sumsq


def _phase_cols512_pack_loss_batch_from_row_ifft(
    mx,
    row_ifft,
    *,
    bf_ranges: tuple[tuple[int, int], ...],
    active_bf,
):
    """Evaluate original BF boundaries together while keeping separate sums."""
    shape = tuple(int(x) for x in row_ifft.shape)
    if len(shape) != 4 or shape[-2:] != (512, 512):
        raise ValueError(
            "Expected row-IFFT pack shape (batch, BF, 512, 512), "
            f"got {shape}."
        )
    batch, storage_num_bf = int(shape[0]), int(shape[1])
    ranges = tuple((int(start), int(stop)) for start, stop in bf_ranges)
    kernel = _phase_cols512_radix8_batch_kernel(
        batch,
        storage_num_bf,
        4096,
        True,
        storage_num_bf,
        0,
        ranges,
    )
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, active_bf, _twiddle_512(mx)],
        template=[],
        grid=(64, 512, batch * len(ranges)),
        threadgroup=(64, 8, 1),
        output_shapes=[
            (batch, len(ranges), 512, 512),
            (batch, len(ranges), 512, 64),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sums = [partial_sum[:, index] for index in range(len(ranges))]
    phase_sumsqs = [
        mx.sum(
            mx.sum(
                mx.sum(partial_sumsq_tile[:, index : index + 1], axis=3),
                axis=2,
            ),
            axis=1,
        )
        for index in range(len(ranges))
    ]
    return phase_sums, phase_sumsqs


@lru_cache(maxsize=32)
def _row_ifft512_dynamic_kernel(
    batch: int,
    chunk: int,
    gqk_cols: int,
    tiled_output: bool = False,
    storage_chunk: int | None = None,
):
    mx = _require_mlx()
    storage_chunk = int(chunk if storage_chunk is None else storage_chunk)
    storage_define = ""
    storage_name = ""
    row_ifft_store = """
        size_t base = (
            ((size_t)output_batch * (size_t)CHUNK + (size_t)bf) * (size_t)PLANE
            + (size_t)row * 512u
        );
        row_ifft[base + tid].real = srow[tid].x;
        row_ifft[base + tid].imag = srow[tid].y;
        row_ifft[base + tid + 64u].real = srow[tid + 64u].x;
        row_ifft[base + tid + 64u].imag = srow[tid + 64u].y;
        row_ifft[base + tid + 128u].real = srow[tid + 128u].x;
        row_ifft[base + tid + 128u].imag = srow[tid + 128u].y;
        row_ifft[base + tid + 192u].real = srow[tid + 192u].x;
        row_ifft[base + tid + 192u].imag = srow[tid + 192u].y;
        row_ifft[base + tid + 256u].real = srow[tid + 256u].x;
        row_ifft[base + tid + 256u].imag = srow[tid + 256u].y;
        row_ifft[base + tid + 320u].real = srow[tid + 320u].x;
        row_ifft[base + tid + 320u].imag = srow[tid + 320u].y;
        row_ifft[base + tid + 384u].real = srow[tid + 384u].x;
        row_ifft[base + tid + 384u].imag = srow[tid + 384u].y;
        row_ifft[base + tid + 448u].real = srow[tid + 448u].x;
        row_ifft[base + tid + 448u].imag = srow[tid + 448u].y;
        """
    if tiled_output:
        row_ifft_store = """
        size_t base = (
            ((size_t)output_batch * (size_t)CHUNK + (size_t)bf)
            * (size_t)PLANE
            + (size_t)(tid >> 3) * 4096u
            + (size_t)row * 8u
            + (tid & 7u)
        );
        row_ifft[base].real = r0.x;
        row_ifft[base].imag = r0.y;
        row_ifft[base + 32768u].real = r1.x;
        row_ifft[base + 32768u].imag = r1.y;
        row_ifft[base + 65536u].real = r2.x;
        row_ifft[base + 65536u].imag = r2.y;
        row_ifft[base + 98304u].real = r3.x;
        row_ifft[base + 98304u].imag = r3.y;
        row_ifft[base + 131072u].real = r4.x;
        row_ifft[base + 131072u].imag = r4.y;
        row_ifft[base + 163840u].real = r5.x;
        row_ifft[base + 163840u].imag = r5.y;
        row_ifft[base + 196608u].real = r6.x;
        row_ifft[base + 196608u].imag = r6.y;
        row_ifft[base + 229376u].real = r7.x;
        row_ifft[base + 229376u].imag = r7.y;
        """
    if storage_chunk != int(chunk):
        row_ifft_store = row_ifft_store.replace("CHUNK", "STORAGE_CHUNK")
        storage_define = f"constexpr uint STORAGE_CHUNK = {storage_chunk}u;"
        storage_name = f"_s{storage_chunk}"
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        #define OCTREV512(n) ((((n) & 7u) << 6) | (((n) & 56u)) | ((n) >> 6))
        #define W8_1(a) float2(0.70710678118654752f * ((a).x - (a).y), 0.70710678118654752f * ((a).x + (a).y))
        #define W8_3(a) float2(0.70710678118654752f * (-(a).x - (a).y), 0.70710678118654752f * ((a).x - (a).y))
        #define RADIX8(x0,x1,x2,x3,x4,x5,x6,x7) {{ \
            float2 a0=(x0), a1=(x4), a2=(x2), a3=(x6); \
            float2 a4=(x1), a5=(x5), a6=(x3), a7=(x7); \
            float2 t0=CADD(a0,a1), t1=CSUB(a0,a1); \
            float2 t2=CADD(a2,a3), t3=CSUB(a2,a3); \
            float2 t4=CADD(a4,a5), t5=CSUB(a4,a5); \
            float2 t6=CADD(a6,a7), t7=CSUB(a6,a7); \
            float2 u0=CADD(t0,t2), u2=CSUB(t0,t2); \
            float2 it3=CMULI(t3), u1=CADD(t1,it3), u3=CSUB(t1,it3); \
            float2 u4=CADD(t4,t6), u6=CSUB(t4,t6); \
            float2 it7=CMULI(t7), u5=CADD(t5,it7), u7=CSUB(t5,it7); \
            float2 w1u5=W8_1(u5), w3u7=W8_3(u7), iu6=CMULI(u6); \
            (x0)=CADD(u0,u4); (x4)=CSUB(u0,u4); \
            (x1)=CADD(u1,w1u5); (x5)=CSUB(u1,w1u5); \
            (x2)=CADD(u2,iu6); (x6)=CSUB(u2,iu6); \
            (x3)=CADD(u3,w3u7); (x7)=CSUB(u3,w3u7); \
        }}

        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        {storage_define}
        constexpr uint NX = 512u;
        constexpr uint PLANE = 512u * 512u;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = 512u * GQK_COLS;
        constexpr uint FUSED_CANDIDATES = BATCH == 4u ? 4u : (BATCH == 2u ? 2u : 1u);
        constexpr uint ROWS_PER_GROUP = BATCH == 2u ? 4u : (BATCH >= 4u ? 2u : 4u);
        constexpr bool FUSE_CANDIDATES = FUSED_CANDIDATES > 1u;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        uint batch = FUSE_CANDIDATES ? 0u : z / CHUNK;
        uint bf = FUSE_CANDIDATES ? z : z - batch * CHUNK;
        if (tid >= 64u || local_row >= ROWS_PER_GROUP || row >= 512u || batch >= BATCH || bf >= CHUNK) {{
            return;
        }}

        threadgroup float2 shared_rows[ROWS_PER_GROUP][FUSED_CANDIDATES][512];

        float factor = scalars[0];
        float dc_r = scalars[1];
        float dc_i = scalars[2];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];

        uint candidate_begin = FUSE_CANDIDATES ? 0u : batch;
        uint candidate_end = FUSE_CANDIDATES ? FUSED_CANDIDATES : batch + 1u;
        auto first_pk = pk[(size_t)candidate_begin * (size_t)CHUNK + bf];
        if (first_pk.real == 0.0f && first_pk.imag == 0.0f) {{
            return;
        }}
        for (uint lane = 0u; lane < 8u; ++lane) {{
            uint col = tid + lane * 64u;
            if (row == 0u && col == 0u) {{
                for (uint candidate=candidate_begin; candidate<candidate_end; ++candidate) {{
                    uint slot = FUSE_CANDIDATES ? candidate : 0u;
                    shared_rows[local_row][slot][OCTREV512(col)] = float2(dc_r, dc_i);
                }}
            }} else {{
                float qyv = q_col[col];

                float dx = qxv - kxv;
                float dy = qyv - kyv;
                float dx2 = dx * dx;
                float dy2 = dy * dy;
                float r2 = dx2 + dy2;
                float r = metal::sqrt(r2);
                float alpha = r * wavelength;
                float alpha2_m = alpha * alpha;
                float inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
                float cos2_m = (dx2 - dy2) * inv_r2;
                float sin2_m = 2.0f * dx * dy * inv_r2;
                float denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
                float inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
                float denom = metal::sqrt(denom_num2) * inv_r;
                float edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
                float ap_m = metal::clamp(edge, 0.0f, 1.0f);

                dx = qxv + kxv;
                dy = qyv + kyv;
                dx2 = dx * dx;
                dy2 = dy * dy;
                r2 = dx2 + dy2;
                r = metal::sqrt(r2);
                alpha = r * wavelength;
                float alpha2_p = alpha * alpha;
                inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
                float cos2_p = (dx2 - dy2) * inv_r2;
                float sin2_p = 2.0f * dx * dy * inv_r2;
                denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
                inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
                denom = metal::sqrt(denom_num2) * inv_r;
                edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
                float ap_p = metal::clamp(edge, 0.0f, 1.0f);

                if (ap_m == 0.0f && ap_p == 0.0f) {{
                    for (uint candidate=candidate_begin; candidate<candidate_end; ++candidate) {{
                        uint slot = FUSE_CANDIDATES ? candidate : 0u;
                        shared_rows[local_row][slot][OCTREV512(col)] = float2(0.0f);
                    }}
                }} else {{
                size_t g_idx;
                bool mirror = false;
                if (GQK_COLS == NX) {{
                    g_idx = (size_t)bf * (size_t)PLANE + (size_t)row * (size_t)NX + (size_t)col;
                }} else if (col <= NX / 2u) {{
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)row * (size_t)GQK_COLS
                        + (size_t)col;
                }} else {{
                    uint mirror_row = row == 0u ? 0u : 512u - row;
                    uint mirror_col = NX - col;
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)mirror_row * (size_t)GQK_COLS
                        + (size_t)mirror_col;
                    mirror = true;
                }}
                auto gz = g[g_idx];
                float gr = gz.real;
                float gi = mirror ? -gz.imag : gz.imag;

                for (uint candidate=candidate_begin; candidate<candidate_end; ++candidate) {{
                    float c10v = c10[candidate];
                    float c12v = c12[candidate];
                    float cos2v = cos2phi12[candidate];
                    float sin2v = sin2phi12[candidate];
                    auto pkz = pk[(size_t)candidate * (size_t)CHUNK + bf];
                    float pkr = pkz.real;
                    float pki = pkz.imag;
                    float chi_m = factor * alpha2_m * (c12v * (cos2_m * cos2v + sin2_m * sin2v) + c10v);
                    float cos_chi_m;
                    float sin_chi_m = metal::fast::sincos(chi_m, cos_chi_m);
                    float pmr = ap_m * cos_chi_m;
                    float pmi = -ap_m * sin_chi_m;

                    float chi_p = factor * alpha2_p * (c12v * (cos2_p * cos2v + sin2_p * sin2v) + c10v);
                    float cos_chi_p;
                    float sin_chi_p = metal::fast::sincos(chi_p, cos_chi_p);
                    float ppr = ap_p * cos_chi_p;
                    float ppi = -ap_p * sin_chi_p;

                    float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
                    float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
                    float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
                    float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
                    float conj_gamma_r = gamma_r * inv_mag;
                    float conj_gamma_i = -gamma_i * inv_mag;

                    float2 corrected = float2(
                        gr * conj_gamma_r - gi * conj_gamma_i,
                        gr * conj_gamma_i + gi * conj_gamma_r
                    );
                    uint slot = FUSE_CANDIDATES ? candidate : 0u;
                    shared_rows[local_row][slot][OCTREV512(col)] = corrected;
                }}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint output_batch=candidate_begin; output_batch<candidate_end; ++output_batch) {{
        uint slot = FUSE_CANDIDATES ? output_batch : 0u;
            threadgroup float2* srow = &shared_rows[local_row][slot][0];
        uint logical=tid*8u;
        float2 r0=srow[logical], r1=srow[logical+1u];
        float2 r2=srow[logical+2u], r3=srow[logical+3u];
        float2 r4=srow[logical+4u], r5=srow[logical+5u];
        float2 r6=srow[logical+6u], r7=srow[logical+7u];
        RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
        srow[logical]=r0; srow[logical+1u]=r1; srow[logical+2u]=r2; srow[logical+3u]=r3;
        srow[logical+4u]=r4; srow[logical+5u]=r5; srow[logical+6u]=r6; srow[logical+7u]=r7;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint s2=tid&7u;
        uint base2=(tid>>3)*64u+s2;
        r0=srow[base2]; r1=CMUL(TW(s2*8u),srow[base2+8u]);
        r2=CMUL(TW(s2*16u),srow[base2+16u]); r3=CMUL(TW(s2*24u),srow[base2+24u]);
        r4=CMUL(TW(s2*32u),srow[base2+32u]); r5=CMUL(TW(s2*40u),srow[base2+40u]);
        r6=CMUL(TW(s2*48u),srow[base2+48u]); r7=CMUL(TW(s2*56u),srow[base2+56u]);
        RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
        srow[base2]=r0; srow[base2+8u]=r1; srow[base2+16u]=r2; srow[base2+24u]=r3;
        srow[base2+32u]=r4; srow[base2+40u]=r5; srow[base2+48u]=r6; srow[base2+56u]=r7;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        r0=srow[tid]; r1=CMUL(TW(tid),srow[tid+64u]);
        r2=CMUL(TW(tid*2u),srow[tid+128u]); r3=CMUL(TW(tid*3u),srow[tid+192u]);
        r4=CMUL(TW(tid*4u),srow[tid+256u]); r5=CMUL(TW(tid*5u),srow[tid+320u]);
        r6=CMUL(TW(tid*6u),srow[tid+384u]); r7=CMUL(TW(tid*7u),srow[tid+448u]);
        RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
        srow[tid]=r0; srow[tid+64u]=r1; srow[tid+128u]=r2; srow[tid+192u]=r3;
        srow[tid+256u]=r4; srow[tid+320u]=r5; srow[tid+384u]=r6; srow[tid+448u]=r7;

        {row_ifft_store}
        }}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef OCTREV512
        #undef W8_1
        #undef W8_3
        #undef RADIX8
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_row_ifft512_dyn_n{int(batch)}_b{int(chunk)}_"
            f"g{int(gqk_cols)}_t{int(tiled_output)}{storage_name}"
        ),
        input_names=[
            "g",
            "q_row",
            "q_col",
            "kx",
            "ky",
            "pk",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
            "twiddle",
        ],
        output_names=["row_ifft"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _row_ifft512_from_dynamic_geometry(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
    return_active: bool = False,
):
    """Fused dynamic correction + 512 row IFFT for exact MPS phase/loss."""
    mx = prepared.mx
    if prepared.scan_shape != (512, 512):
        raise ValueError("Fused MPS row IFFT currently supports only 512x512.")
    if int(c10.shape[0]) != 1:
        raise ValueError("Fused MPS row IFFT currently supports one candidate.")
    chunk = int(stop) - int(start)
    kernel = _row_ifft512_dynamic_kernel(
        1,
        chunk,
        int(prepared.g_qk.shape[-1]),
        False,
    )
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    pk = _pk_batch_from_prepared(
        prepared,
        start=start,
        stop=stop,
        c10=c10,
        c12=c12,
        cos2phi12=cos2phi12,
        sin2phi12=sin2phi12,
    )
    row_ifft_batch = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            _twiddle_512(mx),
        ],
        template=[],
        grid=(64, 512, chunk),
        threadgroup=(64, 4, 1),
        output_shapes=[(1, chunk, 512, 512)],
        output_dtypes=[mx.complex64],
    )[0]
    row_ifft = mx.reshape(row_ifft_batch, (chunk, 512, 512))
    if return_active:
        active_bf = (mx.abs(pk[0]) > 0.0).astype(mx.uint8)
        return row_ifft, active_bf
    return row_ifft


def _row_ifft512_batch_from_dynamic_geometry(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
    return_active: bool = False,
    pk_override=None,
    storage_bf: int | None = None,
):
    """Fused dynamic correction + 512 row IFFT for batched exact MPS loss."""
    mx = prepared.mx
    if prepared.scan_shape != (512, 512):
        raise ValueError("Batched fused MPS row IFFT currently supports only 512x512.")
    batch = int(c10.shape[0])
    chunk = int(stop) - int(start)
    storage_bf = chunk if storage_bf is None else max(chunk, int(storage_bf))
    kernel = _row_ifft512_dynamic_kernel(
        batch,
        chunk,
        int(prepared.g_qk.shape[-1]),
        batch <= 2,
        storage_bf,
    )
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    pk = pk_override
    if pk is None:
        pk = _pk_batch_from_prepared(
            prepared,
            start=start,
            stop=stop,
            c10=c10,
            c12=c12,
            cos2phi12=cos2phi12,
            sin2phi12=sin2phi12,
        )
    row_ifft = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            _twiddle_512(mx),
        ],
        template=[],
        grid=(64, 512, chunk if batch in (2, 4) else batch * chunk),
        threadgroup=(64, 4 if batch == 2 else (2 if batch >= 4 else 4), 1),
        output_shapes=[(batch, storage_bf, 512, 512)],
        output_dtypes=[mx.complex64],
    )[0]
    if return_active:
        active_bf = (mx.abs(pk[0]) > 0.0).astype(mx.uint8)
        return row_ifft, active_bf
    return row_ifft


@lru_cache(maxsize=48)
def _row_ifft_small_dynamic_kernel(
    n: int,
    chunk: int,
    gqk_cols: int,
    batch: int = 1,
    rows_per_group: int = 4,
):
    mx = _require_mlx()
    n = int(n)
    t = n // 4
    half = n // 2
    define_rev, undef_rev, radix4_max, has_final = _small_fft_macros(n)
    final_stage = ""
    if has_final:
        final_stage = f"""
        for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
            threadgroup float2* frow =
                &shared_rows[local_row][candidate][0];
            uint j0 = tid;
            uint j1 = tid + {t}u;
            float2 a0 = frow[j0];
            float2 b0 = CMUL(TW(j0), frow[j0 + {half}u]);
            float2 a1 = frow[j1];
            float2 b1 = CMUL(TW(j1), frow[j1 + {half}u]);
            frow[j0] = CADD(a0, b0);
            frow[j0 + {half}u] = CSUB(a0, b0);
            frow[j1] = CADD(a1, b1);
            frow[j1 + {half}u] = CSUB(a1, b1);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        {define_rev}

        constexpr uint CHUNK = {int(chunk)};
        constexpr uint BATCH = {int(batch)};
        constexpr uint ROWS_PER_GROUP = {int(rows_per_group)};
        constexpr uint N = {n}u;
        constexpr uint T = {t}u;
        constexpr uint PLANE = N * N;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = N * GQK_COLS;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint bf = thread_position_in_grid.z;
        if (tid >= T || local_row >= ROWS_PER_GROUP || row >= N || bf >= CHUNK) {{
            return;
        }}

        threadgroup float2 shared_rows[ROWS_PER_GROUP][BATCH][{n}];

        float factor = scalars[0];
        float dc_r = scalars[1];
        float dc_i = scalars[2];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];

        for (uint lane = 0u; lane < 4u; ++lane) {{
            uint col = tid + lane * T;
            if (row == 0u && col == 0u) {{
                for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
                    shared_rows[local_row][candidate][DIGITREVN(col)] =
                        float2(dc_r, dc_i);
                }}
            }} else {{
                float qyv = q_col[col];

                float dx = qxv - kxv;
                float dy = qyv - kyv;
                float dx2 = dx * dx;
                float dy2 = dy * dy;
                float r2 = dx2 + dy2;
                float r = metal::sqrt(r2);
                float alpha = r * wavelength;
                float alpha2_m = alpha * alpha;
                float inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
                float cos2_m = (dx2 - dy2) * inv_r2;
                float sin2_m = 2.0f * dx * dy * inv_r2;
                float denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
                float inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
                float denom = metal::sqrt(denom_num2) * inv_r;
                float edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
                float ap_m = metal::clamp(edge, 0.0f, 1.0f);

                dx = qxv + kxv;
                dy = qyv + kyv;
                dx2 = dx * dx;
                dy2 = dy * dy;
                r2 = dx2 + dy2;
                r = metal::sqrt(r2);
                alpha = r * wavelength;
                float alpha2_p = alpha * alpha;
                inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
                float cos2_p = (dx2 - dy2) * inv_r2;
                float sin2_p = 2.0f * dx * dy * inv_r2;
                denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
                inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
                denom = metal::sqrt(denom_num2) * inv_r;
                edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
                float ap_p = metal::clamp(edge, 0.0f, 1.0f);

                size_t g_idx;
                bool mirror = false;
                if (GQK_COLS == N) {{
                    g_idx = (size_t)bf * (size_t)PLANE + (size_t)row * (size_t)N + (size_t)col;
                }} else if (col <= N / 2u) {{
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)row * (size_t)GQK_COLS
                        + (size_t)col;
                }} else {{
                    uint mirror_row = row == 0u ? 0u : N - row;
                    uint mirror_col = N - col;
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)mirror_row * (size_t)GQK_COLS
                        + (size_t)mirror_col;
                    mirror = true;
                }}
                auto gz = g[g_idx];
                float gr = gz.real;
                float gi = mirror ? -gz.imag : gz.imag;
                for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
                    float c10v = c10[candidate];
                    float c12v = c12[candidate];
                    float cos2v = cos2phi12[candidate];
                    float sin2v = sin2phi12[candidate];
                    auto pkz = pk[(size_t)candidate * (size_t)CHUNK + bf];
                    float pkr = pkz.real;
                    float pki = pkz.imag;
                    float chi_m = factor * alpha2_m * (c12v * (cos2_m * cos2v + sin2_m * sin2v) + c10v);
                    float cos_chi_m;
                    float sin_chi_m = metal::fast::sincos(chi_m, cos_chi_m);
                    float pmr = ap_m * cos_chi_m;
                    float pmi = -ap_m * sin_chi_m;

                    float chi_p = factor * alpha2_p * (c12v * (cos2_p * cos2v + sin2_p * sin2v) + c10v);
                    float cos_chi_p;
                    float sin_chi_p = metal::fast::sincos(chi_p, cos_chi_p);
                    float ppr = ap_p * cos_chi_p;
                    float ppi = -ap_p * sin_chi_p;

                    float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
                    float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
                    float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
                    float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
                    float conj_gamma_r = gamma_r * inv_mag;
                    float conj_gamma_i = -gamma_i * inv_mag;
                    shared_rows[local_row][candidate][DIGITREVN(col)] = float2(
                        gr * conj_gamma_r - gi * conj_gamma_i,
                        gr * conj_gamma_i + gi * conj_gamma_r
                    );
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint m = 4u; m <= {radix4_max}u; m <<= 2) {{
            for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
                threadgroup float2* srow =
                    &shared_rows[local_row][candidate][0];
                uint quarter = m >> 2;
                uint j = tid % quarter;
                uint k = tid / quarter;
                uint idx0 = k * m + j;
                uint idx1 = idx0 + quarter;
                uint idx2 = idx1 + quarter;
                uint idx3 = idx2 + quarter;
                uint tw = j * (N / m);
                float2 x0 = srow[idx0];
                float2 x1 = CMUL(TW(tw), srow[idx1]);
                float2 x2 = CMUL(TW(tw * 2u), srow[idx2]);
                float2 x3 = CMUL(TW(tw * 3u), srow[idx3]);
                float2 t0 = CADD(x0, x2);
                float2 t1 = CSUB(x0, x2);
                float2 t2 = CADD(x1, x3);
                float2 t3 = CSUB(x1, x3);
                float2 it3 = CMULI(t3);
                srow[idx0] = CADD(t0, t2);
                srow[idx1] = CADD(t1, it3);
                srow[idx2] = CSUB(t0, t2);
                srow[idx3] = CSUB(t1, it3);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        {final_stage}

        for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
        threadgroup float2* srow = &shared_rows[local_row][candidate][0];
        size_t base = ((size_t)candidate * (size_t)CHUNK + (size_t)bf)
            * (size_t)PLANE + (size_t)row * (size_t)N;
        row_ifft[base + tid].real = srow[tid].x;
        row_ifft[base + tid].imag = srow[tid].y;
        row_ifft[base + tid + T].real = srow[tid + T].x;
        row_ifft[base + tid + T].imag = srow[tid + T].y;
        row_ifft[base + tid + 2u * T].real = srow[tid + 2u * T].x;
        row_ifft[base + tid + 2u * T].imag = srow[tid + 2u * T].y;
        row_ifft[base + tid + 3u * T].real = srow[tid + 3u * T].x;
        row_ifft[base + tid + 3u * T].imag = srow[tid + 3u * T].y;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        {undef_rev}
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_row_ifft{n}_dyn_n{int(batch)}_b{int(chunk)}_"
            f"g{int(gqk_cols)}_r{int(rows_per_group)}_sb1"
        ),
        input_names=[
            "g",
            "q_row",
            "q_col",
            "kx",
            "ky",
            "pk",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
            "twiddle",
        ],
        output_names=["row_ifft"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _row_ifft_small_from_dynamic_geometry(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Fused dynamic correction + 128/256/1024 row IFFT for exact MPS phase/loss."""
    mx = prepared.mx
    if prepared.scan_shape not in ((128, 128), (256, 256), (1024, 1024)):
        raise ValueError(
            "Fused MPS row IFFT supports only 128x128, 256x256, or 1024x1024."
        )
    if int(c10.shape[0]) != 1:
        raise ValueError("Fused MPS small row IFFT currently supports one candidate.")
    n = int(prepared.scan_shape[0])
    chunk = int(stop) - int(start)
    kernel = _row_ifft_small_dynamic_kernel(
        n,
        chunk,
        int(prepared.g_qk.shape[-1]),
    )
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    pk = _pk_batch_from_prepared(
        prepared,
        start=start,
        stop=stop,
        c10=c10,
        c12=c12,
        cos2phi12=cos2phi12,
        sin2phi12=sin2phi12,
    )[0]
    t = n // 4
    return kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            _twiddle_n(mx, n),
        ],
        template=[],
        grid=(t, n, chunk),
        threadgroup=(t, 4, 1),
        output_shapes=[(chunk, n, n)],
        output_dtypes=[mx.complex64],
    )[0]


def _row_ifft_small_batch_from_dynamic_geometry(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
    pk_override=None,
    rows_per_group: int = 4,
):
    """Share small-scan geometry and G reads across an exact candidate pair."""
    mx = prepared.mx
    if prepared.scan_shape not in ((128, 128), (256, 256)):
        raise ValueError("Batched fused MPS row IFFT supports 128x128 or 256x256.")
    batch = int(c10.shape[0])
    if batch not in (1, 2):
        raise ValueError("Batched fused small MPS row IFFT supports at most a pair.")
    n = int(prepared.scan_shape[0])
    chunk = int(stop) - int(start)
    kernel = _row_ifft_small_dynamic_kernel(
        n,
        chunk,
        int(prepared.g_qk.shape[-1]),
        batch,
        rows_per_group,
    )
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    pk = pk_override
    if pk is None:
        pk = _pk_batch_from_prepared(
            prepared,
            start=start,
            stop=stop,
            c10=c10,
            c12=c12,
            cos2phi12=cos2phi12,
            sin2phi12=sin2phi12,
        )
    t = n // 4
    return kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            _twiddle_n(mx, n),
        ],
        template=[],
        grid=(t, n, chunk),
        threadgroup=(t, rows_per_group, 1),
        output_shapes=[(batch, chunk, n, n)],
        output_dtypes=[mx.complex64],
    )[0]
@lru_cache(maxsize=16)
def _corrected_kernel(batch: int, chunk: int, ny: int, nx: int, gqk_cols: int):
    mx = _require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = NY * GQK_COLS;
        uint total = BATCH * CHUNK * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / (CHUNK * PLANE);
        uint rem = elem - batch * CHUNK * PLANE;
        uint bf = rem / PLANE;
        uint pixel = rem - bf * PLANE;
        uint geom_idx = bf * PLANE + pixel;
        uint row = pixel / NX;
        uint col = pixel - row * NX;

        if (pixel == 0) {{
            corrected[elem].real = scalars[1];
            corrected[elem].imag = scalars[2];
            return;
        }}

        float c10v = c10[batch];
        float c12v = c12[batch];
        float cos2v = cos2phi12[batch];
        float sin2v = sin2phi12[batch];
        float factor = scalars[0];

        float cos_term_k = cos2_k[bf] * cos2v + sin2_k[bf] * sin2v;
        float chi_k = factor * alpha_k2[bf] * (c12v * cos_term_k + c10v);
        float pk_amp = aperture_k[bf];
        float cos_chi_k;
        float sin_chi_k = metal::fast::sincos(chi_k, cos_chi_k);
        float pkr = pk_amp * cos_chi_k;
        float pki = -pk_amp * sin_chi_k;

        float cos_term_m = cos2_m[geom_idx] * cos2v + sin2_m[geom_idx] * sin2v;
        float chi_m = factor * alpha_m2[geom_idx] * (c12v * cos_term_m + c10v);
        float pm_amp = ap_m[geom_idx];
        float cos_chi_m;
        float sin_chi_m = metal::fast::sincos(chi_m, cos_chi_m);
        float pmr = pm_amp * cos_chi_m;
        float pmi = -pm_amp * sin_chi_m;

        float cos_term_p = cos2_p[geom_idx] * cos2v + sin2_p[geom_idx] * sin2v;
        float chi_p = factor * alpha_p2[geom_idx] * (c12v * cos_term_p + c10v);
        float pp_amp = ap_p[geom_idx];
        float cos_chi_p;
        float sin_chi_p = metal::fast::sincos(chi_p, cos_chi_p);
        float ppr = pp_amp * cos_chi_p;
        float ppi = -pp_amp * sin_chi_p;

        float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
        float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
        float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
        float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
        float conj_gamma_r = gamma_r * inv_mag;
        float conj_gamma_i = -gamma_i * inv_mag;

        size_t g_idx;
        bool mirror = false;
        if (GQK_COLS == NX) {{
            g_idx = (size_t)bf * (size_t)PLANE + (size_t)pixel;
        }} else if (col <= NX / 2) {{
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)row * (size_t)GQK_COLS
                + (size_t)col;
        }} else {{
            uint mirror_row = row == 0 ? 0 : NY - row;
            uint mirror_col = NX - col;
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)mirror_row * (size_t)GQK_COLS
                + (size_t)mirror_col;
            mirror = true;
        }}
        auto gz = g[g_idx];
        if (mirror) {{
            gz.imag = -gz.imag;
        }}
        corrected[elem].real = gz.real * conj_gamma_r - gz.imag * conj_gamma_i;
        corrected[elem].imag = gz.real * conj_gamma_i + gz.imag * conj_gamma_r;
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_corrected_fast_sincos_n{int(batch)}_b{int(chunk)}_"
            f"{int(ny)}_{int(nx)}_g{int(gqk_cols)}"
        ),
        input_names=[
            "g",
            "alpha_k2",
            "cos2_k",
            "sin2_k",
            "aperture_k",
            "alpha_m2",
            "cos2_m",
            "sin2_m",
            "ap_m",
            "alpha_p2",
            "cos2_p",
            "sin2_p",
            "ap_p",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
        ],
        output_names=["corrected"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _corrected_from_cached_geometry(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Fused Metal correction for cached-geometry MPS sparse objectives."""
    mx = prepared.mx
    batch = int(c10.shape[0])
    chunk = int(stop) - int(start)
    ny, nx = prepared.scan_shape
    gqk_cols = int(prepared.g_qk.shape[-1])
    kernel = _corrected_kernel(batch, chunk, int(ny), int(nx), gqk_cols)
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
        ],
        dtype=mx.float32,
    )
    outputs = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.alpha_k2[start:stop],
            prepared.cos2_k[start:stop],
            prepared.sin2_k[start:stop],
            prepared.aperture_k[start:stop],
            prepared.alpha_m2[start:stop],
            prepared.cos2_m[start:stop],
            prepared.sin2_m[start:stop],
            prepared.ap_m[start:stop],
            prepared.alpha_p2[start:stop],
            prepared.cos2_p[start:stop],
            prepared.sin2_p[start:stop],
            prepared.ap_p[start:stop],
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
        ],
        template=[],
        grid=(batch * chunk * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, chunk, int(ny), int(nx))],
        output_dtypes=[mx.complex64],
    )
    return outputs[0]


@lru_cache(maxsize=16)
def _corrected_dynamic_kernel(batch: int, chunk: int, ny: int, nx: int, gqk_cols: int):
    mx = _require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = NY * GQK_COLS;
        uint total = BATCH * CHUNK * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / (CHUNK * PLANE);
        uint rem = elem - batch * CHUNK * PLANE;
        uint bf = rem / PLANE;
        uint pixel = rem - bf * PLANE;
        uint row = pixel / NX;
        uint col = pixel - row * NX;

        if (pixel == 0) {{
            corrected[elem].real = scalars[1];
            corrected[elem].imag = scalars[2];
            return;
        }}

        float factor = scalars[0];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float c10v = c10[batch];
        float c12v = c12[batch];
        float cos2v = cos2phi12[batch];
        float sin2v = sin2phi12[batch];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];
        float qyv = q_col[col];
        auto pkz = pk[batch * CHUNK + bf];
        float pkr = pkz.real;
        float pki = pkz.imag;

        float dx = qxv - kxv;
        float dy = qyv - kyv;
        float dx2 = dx * dx;
        float dy2 = dy * dy;
        float r2 = dx2 + dy2;
        float r = metal::sqrt(r2);
        float alpha = r * wavelength;
        float alpha2_m = alpha * alpha;
        float inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
        float cos2_m = (dx2 - dy2) * inv_r2;
        float sin2_m = 2.0f * dx * dy * inv_r2;
        float denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
        float inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
        float denom = metal::sqrt(denom_num2) * inv_r;
        float edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
        float ap_m = metal::clamp(edge, 0.0f, 1.0f);

        dx = qxv + kxv;
        dy = qyv + kyv;
        dx2 = dx * dx;
        dy2 = dy * dy;
        r2 = dx2 + dy2;
        r = metal::sqrt(r2);
        alpha = r * wavelength;
        float alpha2_p = alpha * alpha;
        inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
        float cos2_p = (dx2 - dy2) * inv_r2;
        float sin2_p = 2.0f * dx * dy * inv_r2;
        denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
        inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
        denom = metal::sqrt(denom_num2) * inv_r;
        edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
        float ap_p = metal::clamp(edge, 0.0f, 1.0f);

        float chi_m = factor * alpha2_m * (c12v * (cos2_m * cos2v + sin2_m * sin2v) + c10v);
        float cos_chi_m;
        float sin_chi_m = metal::fast::sincos(chi_m, cos_chi_m);
        float pmr = ap_m * cos_chi_m;
        float pmi = -ap_m * sin_chi_m;

        float chi_p = factor * alpha2_p * (c12v * (cos2_p * cos2v + sin2_p * sin2v) + c10v);
        float cos_chi_p;
        float sin_chi_p = metal::fast::sincos(chi_p, cos_chi_p);
        float ppr = ap_p * cos_chi_p;
        float ppi = -ap_p * sin_chi_p;

        float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
        float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
        float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
        float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
        float conj_gamma_r = gamma_r * inv_mag;
        float conj_gamma_i = -gamma_i * inv_mag;

        size_t g_idx;
        bool mirror = false;
        if (GQK_COLS == NX) {{
            g_idx = (size_t)bf * (size_t)PLANE + (size_t)pixel;
        }} else if (col <= NX / 2) {{
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)row * (size_t)GQK_COLS
                + (size_t)col;
        }} else {{
            uint mirror_row = row == 0 ? 0 : NY - row;
            uint mirror_col = NX - col;
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)mirror_row * (size_t)GQK_COLS
                + (size_t)mirror_col;
            mirror = true;
        }}
        auto gz = g[g_idx];
        if (mirror) {{
            gz.imag = -gz.imag;
        }}
        corrected[elem].real = gz.real * conj_gamma_r - gz.imag * conj_gamma_i;
        corrected[elem].imag = gz.real * conj_gamma_i + gz.imag * conj_gamma_r;
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_corrected_dyn_pk_fast_sincos_n{int(batch)}_b{int(chunk)}_"
            f"{int(ny)}_{int(nx)}_g{int(gqk_cols)}"
        ),
        input_names=[
            "g",
            "q_row",
            "q_col",
            "kx",
            "ky",
            "pk",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
        ],
        output_names=["corrected"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _corrected_from_dynamic_geometry(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Fused Metal correction for large-BF MPS paths without cached geometry."""
    mx = prepared.mx
    batch = int(c10.shape[0])
    chunk = int(stop) - int(start)
    ny, nx = prepared.scan_shape
    gqk_cols = int(prepared.g_qk.shape[-1])
    kernel = _corrected_dynamic_kernel(batch, chunk, int(ny), int(nx), gqk_cols)
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    outputs = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            _pk_batch_from_prepared(
                prepared,
                start=start,
                stop=stop,
                c10=c10,
                c12=c12,
                cos2phi12=cos2phi12,
                sin2phi12=sin2phi12,
            ),
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
        ],
        template=[],
        grid=(batch * chunk * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, chunk, int(ny), int(nx))],
        output_dtypes=[mx.complex64],
    )
    return outputs[0]


@lru_cache(maxsize=16)
def _object_fourier_sum_dynamic_kernel(
    num_bf: int,
    logical_num_bf: int,
    chunk_bf: int,
    ny: int,
    nx: int,
    gqk_cols: int,
    sparse_storage: bool,
):
    mx = _require_mlx()
    groups = (int(num_bf) + int(chunk_bf) - 1) // int(chunk_bf)
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint LOGICAL_NUM_BF = {int(logical_num_bf)};
        constexpr uint CHUNK = {int(chunk_bf)};
        constexpr uint GROUPS = {int(groups)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = NY * GQK_COLS;
        constexpr bool SPARSE_STORAGE = {str(bool(sparse_storage)).lower()};
        uint total = GROUPS * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint group = elem / PLANE;
        uint pixel = elem - group * PLANE;
        uint row = pixel / NX;
        uint col = pixel - row * NX;

        float c10v = params[0];
        float c12v = params[1];
        float cos2v = params[2];
        float sin2v = params[3];
        float factor = params[4];
        float dc_r = params[5];
        float dc_i = params[6];
        float wavelength = params[7];
        float semiangle = params[8];
        float ang_y = params[9];
        float ang_x = params[10];
        float qxv = q_row[row];
        float qyv = q_col[col];
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        uint group_start = group * CHUNK;

        if (pixel == 0) {{
            if (LOGICAL_NUM_BF == NUM_BF) {{
                uint remaining = NUM_BF > group_start ? NUM_BF - group_start : 0;
                uint valid = remaining < CHUNK ? remaining : CHUNK;
                partial[elem].real = dc_r * float(valid);
                partial[elem].imag = dc_i * float(valid);
            }} else {{
                partial[elem].real = group == 0u ? dc_r * float(LOGICAL_NUM_BF) : 0.0f;
                partial[elem].imag = group == 0u ? dc_i * float(LOGICAL_NUM_BF) : 0.0f;
            }}
            return;
        }}

        for (uint local = 0; local < CHUNK; ++local) {{
            uint bf = group_start + local;
            if (bf >= NUM_BF) {{
                continue;
            }}

            uint stored_bf = bf;
            if (SPARSE_STORAGE) {{
                int mapped_bf = storage_map[bf];
                if (mapped_bf < 0) {{
                    continue;
                }}
                stored_bf = uint(mapped_bf);
            }}

            float kxv = kx[stored_bf];
            float kyv = ky[stored_bf];

            float dx = qxv - kxv;
            float dy = qyv - kyv;
            float dx2 = dx * dx;
            float dy2 = dy * dy;
            float r2 = dx2 + dy2;
            float r = metal::sqrt(r2);
            float alpha = r * wavelength;
            float alpha2_m = alpha * alpha;
            float inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
            float cos2_m = (dx2 - dy2) * inv_r2;
            float sin2_m = 2.0f * dx * dy * inv_r2;
            float denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
            float inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
            float denom = metal::sqrt(denom_num2) * inv_r;
            float edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
            float ap_m = metal::clamp(edge, 0.0f, 1.0f);

            dx = qxv + kxv;
            dy = qyv + kyv;
            dx2 = dx * dx;
            dy2 = dy * dy;
            r2 = dx2 + dy2;
            r = metal::sqrt(r2);
            alpha = r * wavelength;
            float alpha2_p = alpha * alpha;
            inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
            float cos2_p = (dx2 - dy2) * inv_r2;
            float sin2_p = 2.0f * dx * dy * inv_r2;
            denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
            inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
            denom = metal::sqrt(denom_num2) * inv_r;
            edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
            float ap_p = metal::clamp(edge, 0.0f, 1.0f);

            if (ap_m <= 0.0f && ap_p <= 0.0f) {{
                continue;
            }}

            auto pkz = pk[stored_bf];
            float pkr = pkz.real;
            float pki = pkz.imag;

            float chi_m = factor * alpha2_m * (c12v * (cos2_m * cos2v + sin2_m * sin2v) + c10v);
            float cos_chi_m;
            float sin_chi_m = metal::fast::sincos(chi_m, cos_chi_m);
            float pmr = ap_m * cos_chi_m;
            float pmi = -ap_m * sin_chi_m;
            float chi_p = factor * alpha2_p * (c12v * (cos2_p * cos2v + sin2_p * sin2v) + c10v);
            float cos_chi_p;
            float sin_chi_p = metal::fast::sincos(chi_p, cos_chi_p);
            float ppr = ap_p * cos_chi_p;
            float ppi = -ap_p * sin_chi_p;

            float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
            float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
            float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
            float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
            float conj_gamma_r = gamma_r * inv_mag;
            float conj_gamma_i = -gamma_i * inv_mag;

            size_t g_idx;
            bool mirror = false;
            if (GQK_COLS == NX) {{
                g_idx = (size_t)stored_bf * (size_t)PLANE + (size_t)pixel;
            }} else if (col <= NX / 2) {{
                g_idx = (size_t)stored_bf * (size_t)GQK_PLANE
                    + (size_t)row * (size_t)GQK_COLS
                    + (size_t)col;
            }} else {{
                uint mirror_row = row == 0 ? 0 : NY - row;
                uint mirror_col = NX - col;
                g_idx = (size_t)stored_bf * (size_t)GQK_PLANE
                    + (size_t)mirror_row * (size_t)GQK_COLS
                    + (size_t)mirror_col;
                mirror = true;
            }}
            auto gz = g[g_idx];
            if (mirror) {{
                gz.imag = -gz.imag;
            }}
            sum_r += gz.real * conj_gamma_r - gz.imag * conj_gamma_i;
            sum_i += gz.real * conj_gamma_i + gz.imag * conj_gamma_r;
        }}
        partial[elem].real = sum_r;
        partial[elem].imag = sum_i;
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_object_fourier_sum_dyn_fast_sincos_b{int(chunk_bf)}_n{int(num_bf)}_"
            f"logical{int(logical_num_bf)}_"
            f"sparse{int(bool(sparse_storage))}_"
            f"{int(ny)}_{int(nx)}_g{int(gqk_cols)}"
        ),
        input_names=[
            "g", "q_row", "q_col", "kx", "ky", "pk", "params", "storage_map"
        ],
        output_names=["partial"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _pk_from_prepared(
    prepared: _PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
):
    """Probe term ``p(k)`` for each selected BF pixel."""
    mx = prepared.mx
    alpha_k2 = prepared.alpha_k2_1d
    cos2_k = prepared.cos2_k_1d
    sin2_k = prepared.sin2_k_1d
    aperture_k = prepared.aperture_k_1d
    if alpha_k2 is None or cos2_k is None or sin2_k is None or aperture_k is None:
        alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
            mx,
            prepared.kx,
            prepared.ky,
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
        )
    chi_k = prepared.factor * alpha_k2 * (
        float(C12)
        * (
            cos2_k * math.cos(2.0 * float(phi12))
            + sin2_k * math.sin(2.0 * float(phi12))
        )
        + float(C10)
    )
    pk = aperture_k * _exp_neg_i(mx, chi_k)
    mx.eval(pk)
    return pk


def _pk_batch_from_prepared(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Batched probe terms ``p(k)`` for a BF slice."""
    mx = prepared.mx
    alpha_k2 = prepared.alpha_k2_1d
    cos2_k = prepared.cos2_k_1d
    sin2_k = prepared.sin2_k_1d
    aperture_k = prepared.aperture_k_1d
    if alpha_k2 is None or cos2_k is None or sin2_k is None or aperture_k is None:
        alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
            mx,
            prepared.kx,
            prepared.ky,
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
        )
    sl = slice(int(start), int(stop))
    alpha = alpha_k2[sl][None, :]
    cos2 = cos2_k[sl][None, :]
    sin2 = sin2_k[sl][None, :]
    aperture = aperture_k[sl][None, :]
    c10 = c10[:, None]
    c12 = c12[:, None]
    cos2phi12 = cos2phi12[:, None]
    sin2phi12 = sin2phi12[:, None]
    chi_k = prepared.factor * alpha * (
        c12 * (cos2 * cos2phi12 + sin2 * sin2phi12) + c10
    )
    pk = aperture * _exp_neg_i(mx, chi_k)
    mx.eval(pk)
    return pk


def _object_fourier_sum_dynamic(
    prepared: _PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    chunk_bf: int,
    threadgroup_size: int | None = None,
):
    """Exact object wave using BF-summed Fourier-domain correction on MPS."""
    mx = prepared.mx
    ny, nx = prepared.scan_shape
    chunk_bf = max(1, int(chunk_bf))
    if threadgroup_size is None:
        threadgroup_size = _default_object_redraw_threadgroup(prepared.scan_shape)
    threadgroup_size = max(1, int(threadgroup_size))
    sparse_storage = prepared.bf_storage_indices_np is not None
    kernel_num_bf = prepared.num_bf if sparse_storage else int(prepared.g_qk.shape[0])
    groups = (kernel_num_bf + chunk_bf - 1) // chunk_bf
    kernel = _object_fourier_sum_dynamic_kernel(
        kernel_num_bf,
        int(prepared.num_bf),
        chunk_bf,
        int(ny),
        int(nx),
        int(prepared.g_qk.shape[-1]),
        sparse_storage,
    )
    params = mx.array(
        [
            float(C10),
            float(C12),
            math.cos(2.0 * float(phi12)),
            math.sin(2.0 * float(phi12)),
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    pk = _pk_from_prepared(prepared, C10=C10, C12=C12, phi12=phi12)
    if sparse_storage:
        storage_map_np = np.full(prepared.num_bf, -1, dtype=np.int32)
        storage_map_np[prepared.bf_storage_indices_np] = np.arange(
            int(prepared.g_qk.shape[0]), dtype=np.int32
        )
        storage_map = mx.array(storage_map_np)
    else:
        storage_map = mx.zeros((1,), dtype=mx.int32)
    partial = kernel(
        inputs=[
            prepared.g_qk,
            prepared.q_row,
            prepared.q_col,
            prepared.kx,
            prepared.ky,
            pk,
            params,
            storage_map,
        ],
        template=[],
        grid=(groups * int(ny) * int(nx), 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(groups, int(ny), int(nx))],
        output_dtypes=[mx.complex64],
    )[0]
    fourier_sum = mx.sum(partial, axis=0) / prepared.num_bf
    object_wave = mx.fft.ifft2(fourier_sum)
    mx.eval(object_wave)
    return object_wave

def _prepare_selection(
    frames,
    *,
    scan_shape: tuple[int, int],
    selection: BrightfieldDisk,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling: tuple[float, float],
    det_sampling: tuple[float, float],
    rotation_angle_deg: float,
    chunk_bf: int,
    compact_inactive: bool = False,
    dc_value_override: complex | None = None,
) -> _PreparedMpsSSB:
    """Precompute BF-column FFTs and static geometry for MPS SSB fitting."""
    if scan_shape[0] != scan_shape[1]:
        raise ValueError(
            "MPS SSB requires a square scan grid; "
            f"got {scan_shape[0]}x{scan_shape[1]}."
        )
    from .kernels import MPS_FFT_CONFIGS, get_fft_config

    if scan_shape[0] in MPS_FFT_CONFIGS:
        get_fft_config(scan_shape[0])
    mx = _require_mlx()
    from quantem.gpu.detector.compute.mps.kernels import ChunkedFrames

    bf_row = selection.rows
    bf_col = selection.cols
    center = selection.center_row_col
    det_shape = selection.detector_shape
    wavelength = float(electron_wavelength_angstrom(float(voltage_kV) * 1e3))
    reciprocal_sampling = (
        det_sampling[0] * 1e-3 / wavelength,
        det_sampling[1] * 1e-3 / wavelength,
    )
    sampling = (
        1.0 / (reciprocal_sampling[0] * det_shape[0]),
        1.0 / (reciprocal_sampling[1] * det_shape[1]),
    )
    q_row_np, q_col_np = _spatial_frequencies(scan_shape, scan_sampling)

    recip_y = 1.0 / (sampling[0] * det_shape[0])
    recip_x = 1.0 / (sampling[1] * det_shape[1])
    kx_np = (bf_row.astype(np.float32) - center[0]) * recip_y
    ky_np = (bf_col.astype(np.float32) - center[1]) * recip_x
    if rotation_angle_deg:
        angle = math.radians(-float(rotation_angle_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        kx_np, ky_np = kx_np * cos_a + ky_np * sin_a, -kx_np * sin_a + ky_np * cos_a
    kx_np = np.asarray(kx_np, dtype=np.float32)
    ky_np = np.asarray(ky_np, dtype=np.float32)

    q_row_mx = mx.array(q_row_np, dtype=mx.float32)
    q_col_mx = mx.array(q_col_np, dtype=mx.float32)
    kx_mx = mx.array(kx_np, dtype=mx.float32)
    ky_mx = mx.array(ky_np, dtype=mx.float32)
    qx = q_row_mx[None, :, None]
    qy = q_col_mx[None, None, :]
    semiangle_rad = float(semiangle_mrad) * 1e-3
    ang_y_rad = float(det_sampling[0]) * 1e-3
    ang_x_rad = float(det_sampling[1]) * 1e-3
    alpha_k2_1d, cos2_k_1d, sin2_k_1d, aperture_k_1d = _compute_geometry(
        mx,
        kx_mx,
        ky_mx,
        wavelength,
        semiangle_rad,
        ang_y_rad,
        ang_x_rad,
    )
    mx.eval(alpha_k2_1d, cos2_k_1d, sin2_k_1d, aperture_k_1d)
    bf_storage_indices_np = None
    active_mask_np = None
    if compact_inactive:
        active_mask_np = np.asarray(aperture_k_1d) > 0.0
        bf_storage_indices_np = np.flatnonzero(active_mask_np).astype(np.int32)
        if bf_storage_indices_np.size == 0:
            raise ValueError(
                "Automatic MPS SSB probe aperture contains no active BF pixels. "
                "Check the BF center, detector sampling, and semiangle."
            )

    g_chunks = []
    dc_chunks = []
    chunk_bf = max(1, int(chunk_bf))
    for start in range(0, int(bf_row.size), chunk_bf):
        stop = min(start + chunk_bf, int(bf_row.size))
        rows = bf_row[start:stop]
        cols = bf_col[start:stop]
        if compact_inactive and dc_value_override is not None:
            chunk_active = active_mask_np[start:stop]
            rows = rows[chunk_active]
            cols = cols[chunk_active]
            if rows.size == 0:
                continue
        stack_mx = None
        direct_mlx_output = isinstance(
            frames,
            (MpsBfColumnFrames, ChunkedFrames),
        ) and (
            not compact_inactive or dc_value_override is not None
        )
        if direct_mlx_output:
            stack_mx = mx.empty(
                (int(rows.size), *scan_shape),
                dtype=mx.float32,
            )
            mx.eval(stack_mx)
            stack_np = np.asarray(stack_mx)
            frames.columns_float32_into(
                rows,
                cols,
                out=stack_np.reshape(int(rows.size), -1),
            )
        else:
            stack_np = _selected_columns_stack(
                frames,
                rows,
                cols,
                scan_shape,
            ).astype(np.float32, copy=False)
        if compact_inactive:
            if dc_value_override is None:
                dc_chunks.append(
                    stack_np.reshape(stop - start, -1).sum(
                        axis=1,
                        dtype=np.float32,
                    )
                )
                stack_np = stack_np[active_mask_np[start:stop]]
                if stack_np.shape[0] == 0:
                    continue
        g_chunk = _fft2_hermitian(
            mx,
            stack_mx if stack_mx is not None else mx.array(stack_np),
        )
        mx.eval(g_chunk)
        g_chunks.append(g_chunk)
    g_qk = g_chunks[0] if len(g_chunks) == 1 else mx.concatenate(g_chunks, axis=0)
    mx.eval(g_qk)
    # MLX's RFFT result is physically column-major. Materialize it once in
    # BF-major order so every repeated exact row kernel reads contiguous
    # evidence instead of paying an implicit slice-layout conversion.
    g_qk = mx.contiguous(g_qk)
    mx.eval(g_qk)
    if compact_inactive:
        if dc_value_override is None:
            dc_values = np.concatenate(dc_chunks).astype(np.complex64)
            dc_value = complex(dc_values.mean())
        else:
            dc_value = complex(np.complex64(dc_value_override))
        storage_index = mx.array(bf_storage_indices_np)
        kx_np = kx_np[bf_storage_indices_np]
        ky_np = ky_np[bf_storage_indices_np]
        kx_mx = kx_mx[storage_index]
        ky_mx = ky_mx[storage_index]
        alpha_k2_1d = alpha_k2_1d[storage_index]
        cos2_k_1d = cos2_k_1d[storage_index]
        sin2_k_1d = sin2_k_1d[storage_index]
        aperture_k_1d = aperture_k_1d[storage_index]
        mx.eval(
            kx_mx,
            ky_mx,
            alpha_k2_1d,
            cos2_k_1d,
            sin2_k_1d,
            aperture_k_1d,
        )
    else:
        dc_value = complex(np.asarray(g_qk[:, 0, 0]).mean())
    alpha_k2 = cos2_k = sin2_k = aperture_k = None
    alpha_m2 = cos2_m = sin2_m = ap_m = None
    alpha_p2 = cos2_p = sin2_p = ap_p = None
    # Cache static geometry when the selected BF set is small enough.  For the
    # held-out dataset bf_radius=5 path this is ~650 MB of float32 geometry and removes
    # repeated sqrt/aperture work from every optimizer batch.
    geometry_values = int(kx_np.size) * int(scan_shape[0]) * int(scan_shape[1])
    if geometry_values <= 32_000_000:
        kx = mx.array(kx_np, dtype=mx.float32)[:, None, None]
        ky = mx.array(ky_np, dtype=mx.float32)[:, None, None]
        alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
            mx, kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        alpha_m2, cos2_m, sin2_m, ap_m = _compute_geometry(
            mx, qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        alpha_p2, cos2_p, sin2_p, ap_p = _compute_geometry(
            mx, qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        mx.eval(
            alpha_k2, cos2_k, sin2_k, aperture_k,
            alpha_m2, cos2_m, sin2_m, ap_m,
            alpha_p2, cos2_p, sin2_p, ap_p,
        )

    dc_mask_np = np.zeros(scan_shape, dtype=bool)
    dc_mask_np[0, 0] = True
    return _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=qx,
        qy=qy,
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=kx_mx,
        ky=ky_mx,
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=dc_value,
        scan_shape=scan_shape,
        wavelength=wavelength,
        semiangle_rad=semiangle_rad,
        ang_y_rad=ang_y_rad,
        ang_x_rad=ang_x_rad,
        factor=math.pi / wavelength,
        dc_mask=mx.array(dc_mask_np),
        num_bf=selection.size,
        alpha_k2=alpha_k2,
        cos2_k=cos2_k,
        sin2_k=sin2_k,
        aperture_k=aperture_k,
        alpha_m2=alpha_m2,
        cos2_m=cos2_m,
        sin2_m=sin2_m,
        ap_m=ap_m,
        alpha_p2=alpha_p2,
        cos2_p=cos2_p,
        sin2_p=sin2_p,
        ap_p=ap_p,
        alpha_k2_1d=alpha_k2_1d,
        cos2_k_1d=cos2_k_1d,
        sin2_k_1d=sin2_k_1d,
        aperture_k_1d=aperture_k_1d,
        bf_storage_indices_np=bf_storage_indices_np,
    )


def _reconstruct_prepared(
    prepared: _PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    chunk_bf: int,
    compute_loss: bool,
    compute_object: bool,
    return_phase: bool = True,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None]:
    """Run SSB correction from a prepared BF FFT stack."""
    mx = prepared.mx
    accumulator = (
        mx.zeros(prepared.scan_shape, dtype=mx.complex64)
        if compute_object else None
    )
    # CUDA's fixed SSB output is the mean of per-BF phase images, not the
    # phase of the averaged complex object wave. Keep that contract for reference agreement.
    phase_sum = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    uses_scalar_512_loss = prepared.scan_shape == (512, 512)
    uses_scalar_dynamic_loss = (
        prepared.scan_shape in ((128, 128), (256, 256), (1024, 1024))
        and prepared.alpha_k2 is None
    )
    use_scalar_loss = (
        compute_loss
        and not compute_object
        and (uses_scalar_512_loss or uses_scalar_dynamic_loss)
    )
    if use_scalar_loss:
        phase_sumsq = mx.array(0.0, dtype=mx.float32)
    elif compute_loss:
        phase_sumsq = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    else:
        phase_sumsq = None
    c10_values = mx.array([float(C10)], dtype=mx.float32)
    c12_values = mx.array([float(C12)], dtype=mx.float32)
    cos2phi12_values = mx.array(
        [math.cos(2.0 * float(phi12))],
        dtype=mx.float32,
    )
    sin2phi12_values = mx.array(
        [math.sin(2.0 * float(phi12))],
        dtype=mx.float32,
    )
    chunk_bf = max(1, int(chunk_bf))
    phase_col_k_bf = _default_phase_col_k_bf(prepared.scan_shape)

    for start, stop in _bf_storage_chunks(prepared, chunk_bf):
        use_fused_row = (
            prepared.scan_shape in ((128, 128), (256, 256), (512, 512), (1024, 1024))
            and not compute_object
            and prepared.alpha_k2 is None
        )
        if use_fused_row:
            if prepared.scan_shape == (512, 512):
                row_ifft, active_bf = _row_ifft512_from_dynamic_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                    return_active=True,
                )
                if compute_loss:
                    batch_sum, batch_sumsq = (
                        _phase_cols512_scalar_loss_batch_from_row_ifft(
                            mx,
                            row_ifft[None, ...],
                            k_bf=phase_col_k_bf,
                            active_bf=active_bf,
                        )
                    )
                    chunk_sum = batch_sum[0]
                    chunk_sumsq = batch_sumsq[0]
                else:
                    chunk_sum = _phase_cols512_sum_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                        active_bf=active_bf,
                    )
                    chunk_sumsq = None
            else:
                row_ifft = _row_ifft_small_from_dynamic_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                )
                if compute_loss:
                    chunk_sum, chunk_sumsq = _phase_cols_small_scalar_loss_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                else:
                    chunk_sum = _phase_cols_small_sum_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                    chunk_sumsq = None
        else:
            if prepared.alpha_k2 is not None:
                corrected = _corrected_from_cached_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                )[0]
            else:
                corrected = _corrected_from_dynamic_geometry(
                    prepared,
                    start=start,
                    stop=stop,
                    c10=c10_values,
                    c12=c12_values,
                    cos2phi12=cos2phi12_values,
                    sin2phi12=sin2phi12_values,
                )[0]
            if prepared.scan_shape == (512, 512) and not compute_object:
                row_ifft = mx.fft.ifft(corrected, axis=-1)
                if compute_loss:
                    chunk_sum, chunk_sumsq = _phase_cols512_scalar_loss_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                else:
                    chunk_sum = _phase_cols512_sum_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                    )
                    chunk_sumsq = None
            else:
                obj_chunk = _ifft2_chunked(mx, corrected)
                if compute_object:
                    accumulator = accumulator + mx.sum(obj_chunk, axis=0)
                if compute_loss:
                    chunk_sum, chunk_sumsq = _phase_sums_from_complex(mx, obj_chunk)
                else:
                    chunk_sum = _phase_sum_from_complex(mx, obj_chunk)
                    chunk_sumsq = None
        if compute_loss:
            phase_sum = phase_sum + chunk_sum
            phase_sumsq = phase_sumsq + chunk_sumsq
        else:
            phase_sum = phase_sum + chunk_sum
        mx.eval(
            *[
                arr for arr in (accumulator, phase_sum, phase_sumsq)
                if arr is not None
            ]
        )

    object_wave = None
    if compute_object:
        object_wave_mx = accumulator / prepared.num_bf
        mx.eval(object_wave_mx)
        object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    mean_phase_mx = phase_sum / prepared.num_bf
    loss = None
    if compute_loss:
        if use_scalar_loss:
            mean_sq = mx.mean(mean_phase_mx * mean_phase_mx)
            norm = float(
                prepared.num_bf * prepared.scan_shape[0] * prepared.scan_shape[1]
            )
            loss = float(np.asarray(phase_sumsq / norm - mean_sq))
        else:
            var_per_pixel = phase_sumsq / prepared.num_bf - mean_phase_mx * mean_phase_mx
            loss = float(np.asarray(mx.mean(var_per_pixel)))
    mean_phase = None
    if return_phase:
        mx.eval(mean_phase_mx)
        mean_phase = np.asarray(mean_phase_mx).astype(np.float32, copy=False)
    return object_wave, loss, mean_phase


def _effective_exact_batch_chunk_bf(
    chunk_bf: int,
    scan_shape: tuple[int, int],
    batch: int,
) -> int:
    """Choose an MPS exact-loss chunk that leaves room for candidate batches."""
    requested = max(1, int(chunk_bf))
    batch = max(1, int(batch))
    if batch <= 1:
        return requested
    if tuple(scan_shape) == (512, 512):
        if batch >= 8:
            return min(requested, 256)
        if batch >= 4:
            return min(requested, 512)
        return min(requested, 1024)
    return requested


_small_exact_stream_state = threading.local()


@lru_cache(maxsize=1)
def _small_exact_executor() -> ThreadPoolExecutor:
    """Persistent workers whose MLX streams are created in their own threads."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="mps-ssb-exact")


def _small_exact_loss_worker(
    prepared: _PreparedMpsSSB,
    c10: float,
    c12: float,
    phi12: float,
    chunk_bf: int,
) -> float:
    """Submit one unchanged exact loss path on a worker-local MLX stream."""
    mx = prepared.mx
    stream = getattr(_small_exact_stream_state, "stream", None)
    if stream is None:
        stream = mx.new_stream(mx.gpu)
        _small_exact_stream_state.stream = stream
    with mx.stream(stream):
        _object_wave, loss, _phase = _reconstruct_prepared(
            prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=chunk_bf,
            compute_loss=True,
            compute_object=False,
            return_phase=False,
        )
    if loss is None:
        raise RuntimeError("Exact MPS SSB objective did not return a loss.")
    return float(loss)


def _reconstruct_prepared_small_batch_exact_loss_fused(
    prepared: _PreparedMpsSSB,
    *,
    c10_np: np.ndarray,
    c12_np: np.ndarray,
    phi_np: np.ndarray,
    chunk_bf: int,
    packed_columns: bool = False,
) -> np.ndarray:
    """Evaluate a small-scan exact pair while sharing row-stage inputs."""
    mx = prepared.mx
    batch = int(c10_np.size)
    c10_values = mx.array(c10_np, dtype=mx.float32)
    c12_values = mx.array(c12_np, dtype=mx.float32)
    # Match the scalar exact path's Python-float trig followed by float32 cast.
    cos2phi12_values = mx.array(
        [math.cos(2.0 * float(phi)) for phi in phi_np],
        dtype=mx.float32,
    )
    sin2phi12_values = mx.array(
        [math.sin(2.0 * float(phi)) for phi in phi_np],
        dtype=mx.float32,
    )
    phase_sum = mx.zeros((batch, *prepared.scan_shape), dtype=mx.float32)
    phase_sumsq = mx.zeros((batch,), dtype=mx.float32)
    phase_col_k_bf = _default_phase_col_k_bf(prepared.scan_shape)
    storage_num_bf = int(prepared.g_qk.shape[0])
    pk_all = _pk_batch_from_prepared(
        prepared,
        start=0,
        stop=storage_num_bf,
        c10=c10_values,
        c12=c12_values,
        cos2phi12=cos2phi12_values,
        sin2phi12=sin2phi12_values,
    )

    storage_chunks = list(_bf_storage_chunks(prepared, chunk_bf))
    for chunk_index, (start, stop) in enumerate(storage_chunks):
        row_ifft = _row_ifft_small_batch_from_dynamic_geometry(
            prepared,
            start=start,
            stop=stop,
            c10=c10_values,
            c12=c12_values,
            cos2phi12=cos2phi12_values,
            sin2phi12=sin2phi12_values,
            pk_override=pk_all[:, start:stop],
            rows_per_group=1,
        )
        chunk_sum, chunk_sumsq = (
            _phase_cols_small_scalar_loss_batch_from_row_ifft(
                mx,
                row_ifft,
                k_bf=phase_col_k_bf,
                packed_pair=packed_columns,
            )
        )
        phase_sum = phase_sum + chunk_sum
        phase_sumsq = phase_sumsq + chunk_sumsq
        # Submit six sparse BF chunks per evaluation barrier. This keeps the
        # exact sequential float32 additions while bounding live row storage.
        if chunk_index % 6 == 5 or chunk_index + 1 == len(storage_chunks):
            mx.eval(phase_sum, phase_sumsq)

    mean_phase = phase_sum / prepared.num_bf
    mean_sq = mx.stack(
        [
            mx.mean(mean_phase[candidate] * mean_phase[candidate])
            for candidate in range(batch)
        ]
    )
    norm = float(prepared.num_bf * prepared.scan_shape[0] * prepared.scan_shape[1])
    losses = phase_sumsq / norm - mean_sq
    mx.eval(losses)
    return np.asarray(losses).astype(np.float32, copy=False)


def _reconstruct_prepared_batch_exact_loss(
    prepared: _PreparedMpsSSB,
    *,
    C10: np.ndarray,
    C12: np.ndarray,
    phi12: np.ndarray,
    chunk_bf: int,
) -> np.ndarray:
    """Evaluate the exact full-BF phase-variance loss for candidate batches."""
    c10_np = np.asarray(C10, dtype=np.float32).reshape(-1)
    c12_np = np.asarray(C12, dtype=np.float32).reshape(-1)
    phi_np = np.asarray(phi12, dtype=np.float32).reshape(-1)
    if c10_np.size == 0:
        return np.empty((0,), dtype=np.float32)
    if c12_np.size != c10_np.size or phi_np.size != c10_np.size:
        raise ValueError("C10, C12, and phi12 must have matching lengths.")

    batch = int(c10_np.size)
    if (
        prepared.scan_shape in ((128, 128), (256, 256))
        and prepared.alpha_k2 is None
        and batch == 2
    ):
        return _reconstruct_prepared_small_batch_exact_loss_fused(
            prepared,
            c10_np=c10_np,
            c12_np=c12_np,
            phi_np=phi_np,
            chunk_bf=chunk_bf,
            packed_columns=prepared.scan_shape == (256, 256),
        )
    if (
        prepared.scan_shape in ((128, 128), (256, 256))
        and prepared.alpha_k2 is None
        and batch > 1
    ):
        executor = _small_exact_executor()
        futures = [
            executor.submit(
                _small_exact_loss_worker,
                prepared,
                c10,
                c12,
                phi,
                chunk_bf,
            )
            for c10, c12, phi in zip(c10_np, c12_np, phi_np)
        ]
        return np.asarray(
            [future.result() for future in futures],
            dtype=np.float32,
        )

    if prepared.scan_shape != (512, 512) or prepared.alpha_k2 is not None:
        losses = []
        for c10, c12, phi in zip(c10_np, c12_np, phi_np):
            _object_wave, loss, _phase = _reconstruct_prepared(
                prepared,
                C10=float(c10),
                C12=float(c12),
                phi12=float(phi),
                chunk_bf=chunk_bf,
                compute_loss=True,
                compute_object=False,
                return_phase=False,
            )
            if loss is None:
                raise RuntimeError("Exact MPS SSB objective did not return a loss.")
            losses.append(float(loss))
        return np.asarray(losses, dtype=np.float32)

    # The 512 kernels fuse two candidates while sharing the large G_qk read.
    # Wider fused batches increase threadgroup storage and register pressure;
    # the real 2,464-plane benchmark made batch 4 about 2.8x slower per
    # candidate than executing two fused pairs. Keep the public batch API, but
    # tile it through the measured pair topology.
    if batch > 2:
        return np.concatenate(
            [
                _reconstruct_prepared_batch_exact_loss(
                    prepared,
                    C10=c10_np[start : start + 2],
                    C12=c12_np[start : start + 2],
                    phi12=phi_np[start : start + 2],
                    chunk_bf=chunk_bf,
                )
                for start in range(0, batch, 2)
            ]
        )

    mx = prepared.mx
    phase_sum = mx.zeros((batch, *prepared.scan_shape), dtype=mx.float32)
    phase_sumsq = mx.zeros((batch,), dtype=mx.float32)
    c10_values = mx.array(c10_np, dtype=mx.float32)
    c12_values = mx.array(c12_np, dtype=mx.float32)
    cos2phi12_values = mx.array(np.cos(2.0 * phi_np).astype(np.float32))
    sin2phi12_values = mx.array(np.sin(2.0 * phi_np).astype(np.float32))
    chunk_bf = _effective_exact_batch_chunk_bf(chunk_bf, prepared.scan_shape, batch)
    phase_col_k_bf = _default_phase_col_k_bf(prepared.scan_shape)
    storage_num_bf = int(prepared.g_qk.shape[0])
    pk_all = _pk_batch_from_prepared(
        prepared,
        start=0,
        stop=storage_num_bf,
        c10=c10_values,
        c12=c12_values,
        cos2phi12=cos2phi12_values,
        sin2phi12=sin2phi12_values,
    )
    active_bf_all = (mx.abs(pk_all[0]) > 0.0).astype(mx.uint8)
    mx.eval(active_bf_all)

    storage_packs = _bf_storage_chunk_packs(
        prepared,
        chunk_bf,
        max_storage_bf=_EXACT_ROW_PACK_STORAGE_BF_512,
    )

    for pack_index, pack in enumerate(storage_packs):
        start = pack[0][0]
        stop = pack[-1][1]
        row_ifft = _row_ifft512_batch_from_dynamic_geometry(
            prepared,
            start=start,
            stop=stop,
            c10=c10_values,
            c12=c12_values,
            cos2phi12=cos2phi12_values,
            sin2phi12=sin2phi12_values,
            pk_override=pk_all[:, start:stop],
            # Reuse a few bounded allocation sizes instead of retaining all
            # ten real sparse-pack shapes. Only the written prefix is consumed.
            storage_bf=(
                _exact_pair_row_storage_bf_512(stop - start)
                if batch == 2
                else None
            ),
        )
        relative_ranges = tuple(
            (boundary_start - start, boundary_stop - start)
            for boundary_start, boundary_stop in pack
        )
        if batch == 2 and len(relative_ranges) > 1:
            chunk_sums, chunk_sumsqs = (
                _phase_cols512_pack_loss_batch_from_row_ifft(
                    mx,
                    row_ifft,
                    bf_ranges=relative_ranges,
                    active_bf=active_bf_all[start:stop],
                )
            )
        else:
            chunk_sums = []
            chunk_sumsqs = []
            for boundary_start, boundary_stop in pack:
                chunk_sum, chunk_sumsq = (
                    _phase_cols512_scalar_loss_batch_from_row_ifft(
                        mx,
                        row_ifft,
                        k_bf=phase_col_k_bf,
                        active_bf=active_bf_all[start:stop],
                        tiled_input=batch <= 2,
                        bf_start=boundary_start - start,
                        bf_stop=boundary_stop - start,
                    )
                )
                chunk_sums.append(chunk_sum)
                chunk_sumsqs.append(chunk_sumsq)
        for chunk_sum, chunk_sumsq in zip(chunk_sums, chunk_sumsqs):
            phase_sum = phase_sum + chunk_sum
            phase_sumsq = phase_sumsq + chunk_sumsq
        # Scalar calls benefit from two in-flight packs without increasing the
        # paired optimizer's working set or changing any reduction boundary.
        if (
            batch != 1
            or (pack_index + 1) % _EXACT_SCALAR_ROW_PACK_DEPTH_512 == 0
            or pack_index + 1 == len(storage_packs)
        ):
            mx.eval(phase_sum, phase_sumsq)

    mean_phase = phase_sum / prepared.num_bf
    mean_sq = mx.mean(mean_phase * mean_phase, axis=(1, 2))
    norm = float(prepared.num_bf * prepared.scan_shape[0] * prepared.scan_shape[1])
    losses = phase_sumsq / norm - mean_sq
    mx.eval(losses)
    return np.asarray(losses).astype(np.float32, copy=False)


def _nelder_mead_refine(
    best: dict[str, float],
    best_loss: float,
    evaluate,
    *,
    lock: set[str],
    xatol: float = 0.1,
    fatol: float = 1e-8,
    max_iter: int = 300,
    initial_step_floor: dict[str, float] | None = None,
    initial_step_decimals: int | None = None,
) -> tuple[dict[str, float], float]:
    """Pure-Python Nelder-Mead matching the CUDA optimizer's simplex policy."""
    keys = [key for key in ("C10", "C12", "phi12") if key not in lock]
    if not keys:
        return dict(best), float(best_loss)
    x0 = np.array([best[key] for key in keys], dtype=np.float64)
    n = int(x0.size)
    simplex = np.empty((n + 1, n), dtype=np.float64)
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = max(abs(x0[i]) * 0.05, 0.00025)
        if initial_step_floor is not None:
            step = max(step, float(initial_step_floor.get(keys[i], 0.0)))
        if initial_step_decimals is not None:
            step = round(step, int(initial_step_decimals))
        simplex[i + 1, i] += step

    def params_from_x(x: np.ndarray) -> dict[str, float]:
        params = dict(best)
        for i, key in enumerate(keys):
            value = float(x[i])
            if key == "C12":
                value = max(0.0, value)
            params[key] = value
        return params

    f_values = np.empty(n + 1, dtype=np.float64)
    f_values[0] = float(best_loss)
    for i in range(1, n + 1):
        params = params_from_x(simplex[i])
        f_values[i] = evaluate(params)

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    for _ in range(max_iter):
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        x_spread = float(np.max(np.abs(simplex[-1] - simplex[0])))
        f_spread = float(abs(f_values[-1] - f_values[0]))
        if x_spread < xatol and f_spread < fatol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = evaluate(params_from_x(x_r))

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = evaluate(params_from_x(x_e))
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        if f_r < f_values[-1]:
            x_c = centroid + rho * (x_r - centroid)
            f_c = evaluate(params_from_x(x_c))
            if f_c <= f_r:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue
        else:
            x_c = centroid - rho * (centroid - simplex[-1])
            f_c = evaluate(params_from_x(x_c))
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue

        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_values[i] = evaluate(params_from_x(simplex[i]))

    best_idx = int(np.argmin(f_values))
    return params_from_x(simplex[best_idx]), float(f_values[best_idx])


def _evaluate_exact_float32_cached(
    params: dict[str, float],
    evaluate,
    cache: dict[tuple[float, float, float, float], float],
) -> float:
    """Evaluate once for each distinct set of float32 MPS kernel inputs."""
    phi12 = np.float32(params["phi12"])
    key = (
        float(np.float32(params["C10"])),
        float(np.float32(params["C12"])),
        float(np.float32(np.cos(np.float32(2.0) * phi12))),
        float(np.float32(np.sin(np.float32(2.0) * phi12))),
    )
    if key not in cache:
        cache[key] = float(evaluate(params))
    return cache[key]


def reconstruct(
    data,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    det_sampling: float | tuple[float, float] | None = None,
    C10: float = 0.0,
    C12: float = 0.0,
    phi12: float = 0.0,
    rotation_angle_deg: float = 0.0,
    bf_intensity_threshold: float = 0.0,
    bf_center: tuple[float, float] | None = None,
    bf_radius: float | None = None,
    chunk_bf: int = 16,
    verbose: bool = False,
    compute_loss: bool = False,
) -> SSBResult:
    """Reconstruct the shared complex SSB result on Metal."""

    t0 = time.perf_counter()
    frames = _as_chunked_frames(data)
    scan_shape = _scan_shape(frames)
    scan_sampling = _as_sampling(scan_sampling_A)
    selection = _resolve_bf_selection(
        frames,
        bf_intensity_threshold,
        bf_radius,
        center_override=bf_center,
    )
    if det_sampling is None:
        det_px = 2.0 * float(semiangle_mrad) / selection.detected_radius_px
        det_sampling = (det_px, det_px)
    else:
        det_sampling = _as_sampling(det_sampling)
    requested_chunk_bf = max(1, int(chunk_bf))
    prepared = _prepare_selection(
        frames,
        scan_shape=scan_shape,
        selection=selection,
        voltage_kV=voltage_kV,
        semiangle_mrad=semiangle_mrad,
        scan_sampling=scan_sampling,
        det_sampling=det_sampling,
        rotation_angle_deg=rotation_angle_deg,
        chunk_bf=max(requested_chunk_bf, _default_object_setup_chunk_bf()),
    )
    object_wave_mx = _object_fourier_sum_dynamic(
        prepared,
        C10=C10,
        C12=C12,
        phi12=phi12,
        chunk_bf=max(requested_chunk_bf, _default_object_redraw_chunk_bf()),
    )
    mx = _require_mlx()
    mx.eval(object_wave_mx)
    object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    loss = None
    if compute_loss:
        _unused_object, loss, _unused_mean_phase = _reconstruct_prepared(
            prepared,
            C10=C10,
            C12=C12,
            phi12=phi12,
            chunk_bf=_effective_phase_loss_chunk_bf(requested_chunk_bf, scan_shape),
            compute_loss=True,
            compute_object=False,
        )
    if verbose:
        print(f"MPS SSB BF {prepared.num_bf}/{prepared.num_bf}")
    return SSBResult(
        object_wave=object_wave,
        backend="mps",
        aberrations={"C10": float(C10), "C12": float(C12), "phi12": float(phi12)},
        rotation_angle_deg=float(rotation_angle_deg),
        loss=None if loss is None else float(loss),
        elapsed=time.perf_counter() - t0,
        num_bf=selection.size,
        voltage_kV=float(voltage_kV),
        semiangle_mrad=float(semiangle_mrad),
        scan_sampling_A=scan_sampling_A,
        bf_center=selection.center_row_col,
        bf_radius=selection.radius_px,
        detected_bf_radius=selection.detected_radius_px,
    )


__all__ = [
    "BrightfieldDisk",
    "MpsBfColumnFrames",
    "load_bf_columns_mps",
    "reconstruct",
]
