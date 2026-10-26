"""Detector-column sources that MPS SSB reads: NumPy arrays, MPS tensors, decoded frames and exact BF-column exports.

MPS SSB streams one bright-field detector pixel over all scan positions at a time, so every source answers "these
(row, col) columns as float32". An export from ShowPtycho or ``DirectPtycho.export_brightfield`` keeps exactly those columns as
integers on disk with a declaration that names them; ``find_bf_columns`` looks for one beside the acquisition before
``DirectPtycho.open`` falls back to decoding the detector. Those columns also answer "as stored" (``columns_into``), so
G(q, k) preparation reads the integers straight into GPU memory and converts them there.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from quantem.gpu.device.metal_runtime import (
    SharedArray,
    complete_command,
    metal_device,
    metal_module,
    metal_queue,
    numpy_view,
    tensor_buffer,
)
from quantem.gpu.resident.mps.frames import ChunkedFrames
from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk
from quantem.gpu.direct_ptycho.calibration import CALIBRATION_SCHEMA
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx

# The layout of a ShowPtycho folder's snapshots/cal.json as quantem.widget writes it; another value is refused.
# 2: keys follow the shared vocabulary (bright_field_rows, bright_field_cols, bright_field_center_px,
#    bright_field_radius_px, detector_sampling_mrad), scan_region is [row_start, row_stop, col_start, col_stop]
#    and scan_shape is [rows, cols]; earlier exports wrote bf_* keys and a scan_region object.
# 3: the display-only phase negation is display_negate_phase (flip_phase before), and scan_detector_rotation_flipped
#    is part of the layout.
BROWSER_CALIBRATION_SCHEMA = "quantem.ssb-browser-calibration/3"

# Threads that read exact BF columns from their file. Each runs positioned reads (``os.preadv``, which releases the
# GIL) of whole columns into the destination. On an M5 Max four read 2.4 GB at the SSD's ~13 GB/s from disk and in
# 0.08 s from the page cache; eight were no faster.
COLUMN_READ_THREADS = 4
# Bytes one positioned read asks for: neighbouring columns are contiguous in the file and read together.
COLUMN_READ_BYTES = 8 * 1024**2


def find_bf_columns(
    source: str,
    calibration: str | None,
    *,
    verbose: bool = False,
):
    """Return the exact BF columns (``MpsBfColumnFrames``) exported for ``source``, or None when no location declares any.

    Locations are tried in ``bf_column_locations`` order; a location without a declaration is skipped, while a
    declaration that does not validate raises.
    """
    for candidate in bf_column_locations(source, calibration):
        try:
            return load_bf_columns_mps(candidate, verbose=verbose)
        except _BfColumnCompanionNotDeclared:
            continue
    return None


def bf_column_locations(
    source: str,
    calibration: str | None,
) -> tuple[Path, ...]:
    """Return the places exact BF columns may sit for one source, in source-authoritative order.

    An exported session keeps its BF columns beside the calibration snapshot,
    the acquisition folder, or the calibration file; each location is tried
    once, in that order.
    """
    source_path = Path(source).expanduser().resolve()
    candidates: list[Path] = []
    calibration_path = (
        Path(calibration).expanduser().resolve()
        if calibration is not None
        else None
    )
    if calibration_path is not None:
        exact_export_calibration = calibration_path.parent / "snapshots" / "cal.json"
        if exact_export_calibration.is_file():
            candidates.append(exact_export_calibration)
    if source_path.is_dir():
        candidates.append(source_path)
    else:
        source_parent = source_path.parent
        if source_parent.name == "source":
            candidates.append(source_parent.parent)
        candidates.append(source_parent)
    if calibration_path is not None:
        candidates.append(calibration_path)
    return tuple(dict.fromkeys(candidates))


def load_bf_columns_mps(
    calibration: str | Path,
    *,
    verbose: bool = False,
):
    """Load a ShowPtycho exact BF-column companion for MPS SSB as ``MpsBfColumnFrames``.

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
            raise _BfColumnCompanionNotDeclared(
                f"No ShowPtycho calibration found under {source}."
            )
    else:
        cal_path = source
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ShowPtycho calibration must be a JSON object: {cal_path}")
    if payload.get("schema") == CALIBRATION_SCHEMA and not any(
        key in payload for key in ("bf_column_companion_path", "bf_column_encoding")
    ):
        # A fitted Python calibration carries optics, not detector-column data.
        # Let open() read the acquisition instead of treating the fit as counts.
        raise _BfColumnCompanionNotDeclared(
            f"No exact BF-column companion is declared: {cal_path}"
        )
    if payload.get("schema") != BROWSER_CALIBRATION_SCHEMA:
        # the keys of earlier exports name other quantities (bf_center, det_sampling_mrad_px); none is reinterpreted
        raise ValueError(
            f"{cal_path} is not a {BROWSER_CALIBRATION_SCHEMA} calibration (schema {payload.get('schema')!r}); "
            "export the ShowPtycho folder again."
        )
    required_fields = (
        "bright_field_rows",
        "bright_field_cols",
        "bright_field_center_px",
        "detector_shape",
        "scan_region",
        "scan_shape",
    )
    missing_fields = [name for name in required_fields if name not in payload]
    if missing_fields:
        raise ValueError(
            f"Calibration is missing required BF-column fields "
            f"{missing_fields}: {cal_path}"
        )
    companion = _resolve_bf_column_companion(cal_path, payload)
    scan_shape = payload["scan_shape"]
    if not isinstance(scan_shape, list) or len(scan_shape) != 2:
        raise ValueError(f"Calibration scan_shape must be [rows, cols]: {cal_path}")
    scan_region = payload["scan_region"]
    # the columns hold the scan positions of this crop of the acquisition, so the crop and the shape must agree
    if (not isinstance(scan_region, list) or len(scan_region) != 4
            or [scan_region[1] - scan_region[0], scan_region[3] - scan_region[2]] != scan_shape):
        raise ValueError(
            f"Calibration scan_region must be [row_start, row_stop, col_start, col_stop] spanning scan_shape "
            f"{scan_shape}, got {scan_region!r}: {cal_path}"
        )
    detector_shape = payload["detector_shape"]
    if not isinstance(detector_shape, list) or len(detector_shape) != 2:
        raise ValueError(
            f"Calibration has no valid 2D detector shape: {cal_path}"
        )
    bright_field_center_px = payload["bright_field_center_px"]
    if not isinstance(bright_field_center_px, list) or len(bright_field_center_px) != 2:
        raise ValueError(
            f"Calibration bright_field_center_px must be [row, col]: {cal_path}"
        )
    bright_field_rows = payload["bright_field_rows"]
    bright_field_cols = payload["bright_field_cols"]
    if (
        not isinstance(bright_field_rows, list)
        or not isinstance(bright_field_cols, list)
        or not bright_field_rows
        or len(bright_field_rows) != len(bright_field_cols)
    ):
        raise ValueError(
            f"Calibration BF rows and columns must be non-empty matching "
            f"lists: {cal_path}"
        )

    rows_array = np.asarray(bright_field_rows, dtype=np.int32)
    cols_array = np.asarray(bright_field_cols, dtype=np.int32)
    bright_field_center_px = tuple(float(value) for value in bright_field_center_px)
    distance_sq = (rows_array.astype(np.float32) - bright_field_center_px[0]) ** 2
    distance_sq += (cols_array.astype(np.float32) - bright_field_center_px[1]) ** 2
    coordinate_radius = float(np.sqrt(distance_sq).max()) + 1e-3
    stored_radius = payload.get("bright_field_radius_px")
    radius = coordinate_radius if stored_radius is None else float(stored_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            f"Calibration bright_field_radius_px must be positive and finite: {cal_path}"
        )
    if float(np.sqrt(distance_sq).max()) > radius + 1e-3:
        raise ValueError(
            "Calibration BF coordinates extend beyond bright_field_radius_px: "
            f"{cal_path}"
        )
    selection = BrightfieldDisk(
        rows=rows_array,
        cols=cols_array,
        bright_field_center_px=bright_field_center_px,
        bright_field_radius_px=radius,
        bright_field_edge_radius_px=radius,
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
    # The export's detector calibration: the stored disk radius is the selection's, not the disk edge it was measured on.
    stored_sampling = payload.get("detector_sampling_mrad")
    detector_sampling_mrad = None
    if stored_sampling is not None:
        sampling = np.asarray(stored_sampling, dtype=np.float64)
        if sampling.shape != (2,) or not bool(np.all(np.isfinite(sampling) & (sampling > 0))):
            raise ValueError(
                f"Calibration detector_sampling_mrad must be two positive (row, col) values: {cal_path}"
            )
        detector_sampling_mrad = (float(sampling[0]), float(sampling[1]))
    return MpsBfColumnFrames(
        companion.path,
        selection=selection,
        scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
        dtype=companion.dtype,
        detector_bin=companion.detector_bin,
        source_provenance=companion.provenance,
        max_value=companion.max_value,
        dc_value=dc_value,
        detector_sampling_mrad=detector_sampling_mrad,
        metadata={name: payload[name] for name in ("voltage_kV", "semiangle_mrad", "scan_sampling_A") if name in payload},
        verbose=verbose,
    )


class MpsBfColumnFrames:
    """Exact disk-backed BF columns for MPS SSB fitting and reconstruction.

    The stored values remain integer-exact. Requested columns are read from the
    file in parallel, as stored, into unified memory (``columns_into``); the
    float32 conversion, detector sums, FFTs, objectives, refinement and final
    reconstruction run on MPS/Metal. The full detector stack is never
    materialized.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        selection: BrightfieldDisk,
        scan_shape: tuple[int, int],
        dtype: np.dtype | str,
        detector_bin: int = 1,
        source_provenance: dict[str, object] | None = None,
        max_value: int | None = None,
        detector_sum: np.ndarray | None = None,
        dc_value: complex | None = None,
        detector_sampling_mrad: tuple[float, float] | None = None,
        metadata: dict[str, object] | None = None,
        verbose: bool = False,
    ) -> None:
        self.source_path = Path(path).expanduser().resolve()
        # (row, col) mrad per detector pixel the columns were exported with; DirectPtycho uses it when given none
        self.detector_sampling_mrad = detector_sampling_mrad
        self.metadata = dict(metadata or {})
        self.scan_shape = tuple(int(value) for value in scan_shape)
        self.selection = selection
        self.detector_shape = selection.detector_shape
        self._n = int(np.prod(self.scan_shape))
        self._np_dtype = np.dtype(dtype)
        if self._np_dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
            raise ValueError(
                "MPS BF-column input must use exact uint8 or uint16 values; "
                f"got {self._np_dtype}."
            )
        self.dtype = self._np_dtype
        self.shape = (self._n, *self.detector_shape)
        self.ndim = 3
        self.det_bin = int(detector_bin)
        if self.det_bin < 1:
            raise ValueError("detector_bin must be a positive integer.")
        self.source_provenance = dict(source_provenance or {})
        self.vi = _BfColumnDetectorView(self.detector_shape)
        # detector.mean recognizes GPU-frame objects through the common
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
        self._lookup = np.full(self.detector_shape, -1, dtype=np.int32)
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
        started = time.perf_counter()
        self._detector_sum = None
        if detector_sum is not None:
            exact_sum = np.asarray(detector_sum)
            if exact_sum.shape != self.detector_shape:
                raise ValueError(
                    "detector_sum shape does not match BF detector geometry: "
                    f"{exact_sum.shape} versus {self.detector_shape}."
                )
            if not np.issubdtype(exact_sum.dtype, np.integer):
                raise TypeError(
                    "detector_sum must contain exact integer counts; got "
                    f"{exact_sum.dtype}."
                )
            self._detector_sum = exact_sum.copy()
        elif self.dc_value is None:
            self._detector_sum = self._detector_sum_mps()
        self.load_seconds = time.perf_counter() - started
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

    def transpose_detector(self):
        """Return a detector-axis view, sharing the unchanged exact columns."""
        view = copy(self)
        view.selection = replace(
            self.selection, rows=self.selection.cols, cols=self.selection.rows,
            bright_field_center_px=self.selection.bright_field_center_px[::-1],
            detector_shape=self.detector_shape[::-1],
        )
        view.detector_shape = self.detector_shape[::-1]
        view.shape = (self._n, *view.detector_shape)
        view.vi = _BfColumnDetectorView(view.detector_shape)
        view._lookup = self._lookup.T
        view._detector_sum = None if self._detector_sum is None else self._detector_sum.T
        if self.detector_sampling_mrad is not None:
            view.detector_sampling_mrad = self.detector_sampling_mrad[::-1]
        return view

    @property
    def detector_sum(self) -> np.ndarray:
        """Return exact detector sums, computing them only when requested."""

        if self._detector_sum is None:
            self._detector_sum = self._detector_sum_mps()
        return self._detector_sum

    def _detector_sum_mps(self) -> np.ndarray:
        """Exact uint64 sum of every stored column, read as stored and summed in uint64 on Metal."""
        mx = require_mlx()
        sums = np.empty(self.selection.size, dtype=np.uint64)
        stored = mlx_count_type(mx, self._np_dtype)
        # 64 MiB of counts per block, so the uint64 copy the sum reads stays at most 512 MiB
        block = max(1, (64 * 1024**2) // (self._n * self._np_dtype.itemsize))
        indices = np.arange(self.selection.size, dtype=np.intp)
        for start in range(0, self.selection.size, block):
            stop = min(start + block, self.selection.size)
            counts = mx.empty((stop - start, self._n), dtype=stored)
            mx.eval(counts)
            _read_file_rows(self.source_path, indices[start:stop], np.asarray(counts))
            total = mx.sum(counts.astype(mx.uint64), axis=1)
            mx.eval(total)
            sums[start:stop] = np.asarray(total)
            del counts, total
        mx.clear_cache()
        detector_sum = np.zeros(self.detector_shape, dtype=np.uint64)
        detector_sum[self.selection.rows, self.selection.cols] = sums
        return detector_sum

    def _indices(self, rows, cols) -> np.ndarray:
        """Map detector ``(row, col)`` coordinates to stored column indices.

        A coordinate the export did not keep raises instead of reading an
        unrelated column.
        """
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
        out = np.empty((np.size(rows), self._n), dtype=self._np_dtype)
        return self.columns_into(rows, cols, out)

    def columns_into(self, rows, cols, out: np.ndarray) -> np.ndarray:
        """Read the requested exact columns, as stored, into ``out``: a C-contiguous ``(BF, scan)`` array of the file's dtype.

        ``prepare_selection`` passes the host view of an MLX array and converts the counts to float32 on the GPU, so
        the host writes one byte (uint8) or two (uint16) per count instead of four and converts nothing.
        """
        indices = self._indices(rows, cols)
        expected_shape = (int(indices.size), self._n)
        if tuple(int(value) for value in out.shape) != expected_shape:
            raise ValueError(
                f"Output shape {out.shape} does not match {expected_shape}."
            )
        if out.dtype != self._np_dtype or not out.flags.c_contiguous:
            raise ValueError(
                f"Output must be a C-contiguous {self._np_dtype} array, got a {out.dtype} array."
            )
        _read_file_rows(self.source_path, indices, out)
        return out

    def columns_float32_into(
        self,
        rows,
        cols,
        out: np.ndarray,
    ) -> np.ndarray:
        """Copy the requested exact columns as ``(BF, scan)`` float32 into ``out``, the column contract of every MPS source.

        G(q, k) preparation reads ``columns_into`` instead; this answers the few closed-aperture columns the direct
        methods sum and a re-export.
        """
        counts = self.columns(rows, cols)
        if tuple(int(value) for value in out.shape) != counts.shape:
            raise ValueError(
                f"Output shape {out.shape} does not match {counts.shape}."
            )
        out[...] = counts
        return out


class EncodedBrightFieldCrop:
    """The bright-field crop of an encoded acquisition, left encoded: what ``crop_bright_field`` decodes, never decoded whole.

    It stands in a Direct Ptycho session where the decoded crop stood, with the same 4D shape (scan rows, scan columns, crop
    rows, crop columns), dtype and counts (invalid detector pixels read 0). MPS SSB reads its bright-field columns from
    the encoded source a few scan rows at a time, only while it builds G(q, k) (``EncodedBrightFieldFrames``), so a
    512 x 512 scan holds no 6 GB decoded copy. Slicing the scan axes (``crop[r0:r1, c0:c1]``) or swapping the
    detector axes returns another view of the same source. The source must stay open while a session reads it.
    """

    ndim = 4

    def __init__(self, source, detector_region, *, scan_region=None, flipped=False):
        import torch

        self.source = source
        self.detector_region = tuple(int(value) for value in detector_region)
        rows, cols = (int(size) for size in source.shape[:2])
        self.scan_region = (0, rows, 0, cols) if scan_region is None else tuple(int(value) for value in scan_region)
        self.flipped = bool(flipped)
        row0, row1, col0, col1 = self.detector_region
        valid = getattr(source.data, "valid_pixels", None)
        # read() returns stored values at invalid pixels; the decoded crop held 0 there, and so does every column here
        self.valid = (np.ones((row1 - row0, col1 - col0), bool) if valid is None
                      else np.asarray(valid, bool)[row0:row1, col0:col1].copy())
        # the dtype the decoded crop tensor had, so a session records the same source
        self.dtype = getattr(torch, np.dtype(source.dtype).name)
        self.device = torch.device("mps")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        row0, row1, col0, col1 = self.detector_region
        detector = (row1 - row0, col1 - col0)
        return (self.scan_region[1] - self.scan_region[0], self.scan_region[3] - self.scan_region[2],
                *(detector[::-1] if self.flipped else detector))

    @property
    def nbytes(self) -> int:
        """Bytes the crop would take decoded (it is not)."""
        return int(np.prod(self.shape)) * np.dtype(self.source.dtype).itemsize

    def __getitem__(self, key):
        """The scan positions ``crop[row_start:row_stop, col_start:col_stop]``, still encoded."""
        if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(part, slice) for part in key):
            raise TypeError("An encoded bright-field crop selects scan positions only: crop[row_start:row_stop, col_start:col_stop].")
        rows, cols = self.shape[:2]
        (row_start, row_stop, row_step), (col_start, col_stop, col_step) = key[0].indices(rows), key[1].indices(cols)
        if row_step != 1 or col_step != 1 or row_start >= row_stop or col_start >= col_stop:
            raise ValueError("Select a nonempty, contiguous scan region.")
        first_row, _, first_col, _ = self.scan_region
        region = (first_row + row_start, first_row + row_stop, first_col + col_start, first_col + col_stop)
        return EncodedBrightFieldCrop(self.source, self.detector_region, scan_region=region, flipped=self.flipped)

    def swapaxes(self, first: int, second: int):
        """The crop with its detector axes swapped (``swapaxes(-1, -2)``), as a flipped scan-detector convention reads it."""
        if {int(first) % 4, int(second) % 4} != {2, 3}:
            raise ValueError("An encoded bright-field crop swaps its two detector axes only.")
        return EncodedBrightFieldCrop(self.source, self.detector_region, scan_region=self.scan_region,
                                      flipped=not self.flipped)

    def read_rows(self, row_start: int, row_stop: int, detector_region: tuple[int, int, int, int]):
        """Counts of scan rows ``[row_start, row_stop)`` of the crop over ``detector_region`` (crop pixels, the
        source's orientation), a ``(rows, cols, region_rows, region_cols)`` Torch MPS tensor with stored values at
        invalid pixels."""
        if getattr(self.source.data, "is_released", False):
            raise RuntimeError(
                "This SSB session reads its bright-field columns from the dataset it was given, which is closed. "
                "Keep the dataset open while the session is in use."
            )
        first_row, _, first_col, last_col = self.scan_region
        row0, _, col0, _ = self.detector_region
        region_row0, region_row1, region_col0, region_col1 = detector_region
        return self.source.read(
            scan_region=(first_row + row_start, first_row + row_stop, first_col, last_col),
            detector_region=(row0 + region_row0, row0 + region_row1, col0 + region_col0, col0 + region_col1),
        )

    def detector_total(self) -> np.ndarray:
        """Exact uint64 sum of the crop's counts over its scan positions, invalid pixels 0, session orientation."""
        from quantem.gpu import detector

        row0, row1, col0, col1 = self.detector_region
        session = detector.prepare(self.source)
        try:
            rows, cols = (int(size) for size in self.source.shape[:2])
            if self.scan_region == (0, rows, 0, cols):
                total = session.detector_total()
            else:
                first_row, last_row, first_col, last_col = self.scan_region
                scans = np.arange(first_row, last_row)[:, None] * cols + np.arange(first_col, last_col)[None, :]
                total = session.reduce_frames_exact(scans.reshape(-1))
        finally:
            session.close()
        total = np.asarray(total)[row0:row1, col0:col1].astype(np.uint64)
        total[~self.valid] = 0
        return total.T.copy() if self.flipped else total

    def scan_sums(self) -> np.ndarray:
        """Exact uint64 sum of every pattern over the whole crop (invalid pixels 0), ``(rows, cols)``."""
        from quantem.gpu import detector

        row0, row1, col0, col1 = self.detector_region
        mask = np.zeros(self.source.shape[2:], bool)
        mask[row0:row1, col0:col1] = self.valid
        session = detector.prepare(self.source)
        try:
            sums = np.asarray(session.masked_sum_exact(mask)).astype(np.uint64, copy=False)
        finally:
            session.close()
        first_row, last_row, first_col, last_col = self.scan_region
        return sums[first_row:last_row, first_col:last_col]


class EncodedBrightFieldFrames:
    """MPS SSB's column source over an ``EncodedBrightFieldCrop``: ``(positions, crop_row, crop_col)`` frames.

    ``scan_row_blocks`` decodes a few scan rows of the requested pixels at a time, which is all the G(q, k) build
    reads. A rectangular or too large scan is centred on the next supported square grid with its mean pattern, as
    ``pad_scan`` pads decoded counts; the padded positions are never stored.
    """

    det_bin = 1
    ndim = 3
    # detector.mean recognizes chunk-backed MPS sources; the mean pattern divides the exact detector_sum
    chunks = ()
    # Decoded counts are staged, converted to float32 and transformed while
    # G(q, k) remains resident. Bound these overlapping temporaries; even-row
    # blocks below preserve the real FFT's row pairing and exact result.
    block_bytes = 64 * 1024**2

    def __init__(self, crop: EncodedBrightFieldCrop) -> None:
        from quantem.gpu.direct_ptycho.mps.kernels import MPS_FFT_CONFIGS

        self.crop = crop
        rows, cols, *detector_shape = crop.shape
        self.detector_shape = tuple(int(size) for size in detector_shape)
        self.dtype = np.dtype(crop.source.dtype)
        self._measured_total = crop.detector_total()
        self.offset = (0, 0)
        self._pattern = None
        if rows == cols and rows <= max(MPS_FFT_CONFIGS):
            self.scan_shape = (rows, cols)
        else:
            target = next((size for size in sorted(MPS_FFT_CONFIGS) if size >= max(rows, cols)), None)
            if target is None:
                raise ValueError(
                    f"MPS SSB supports scans up to 1024x1024; got {rows}x{cols}. "
                    "Select a region with at most 1024 positions per side."
                )
            self.scan_shape = (target, target)
            self.offset = ((target - rows) // 2, (target - cols) // 2)
            # pad_scan's integer mean: the exact total divided in float64, truncated to the counts' dtype
            self._pattern = (self._measured_total.astype(np.float64) / (rows * cols)).astype(self.dtype)
            print(
                f"MPS SSB: padded the {rows}x{cols} scan to {target}x{target} with its "
                "mean pattern for the Metal FFT. Every measured position is retained; "
                "the phase covers the padded grid."
            )
        self.measured_shape = (rows, cols)
        self.metadata = {"scan_shape": self.scan_shape}
        self.shape = (int(np.prod(self.scan_shape)), *self.detector_shape)

    @property
    def nbytes(self) -> int:
        return self.crop.nbytes

    @property
    def detector_sum(self) -> np.ndarray:
        """Exact uint64 detector total over the (padded) scan grid."""
        if self._pattern is None:
            return self._measured_total
        padded = int(np.prod(self.scan_shape)) - int(np.prod(self.measured_shape))
        return self._measured_total + np.uint64(padded) * self._pattern.astype(np.uint64)

    def rows_per_block(self, num_pixels: int, detector_pixels: int) -> int:
        """Scan rows per decoded block: an even count, so every block starts on an even row (see ``prepare_selection``)."""
        row_bytes = self.scan_shape[1] * max(int(num_pixels), int(detector_pixels)) * self.dtype.itemsize
        return max(2, 2 * (int(self.block_bytes) // max(1, row_bytes) // 2))

    def scan_row_blocks(self, rows, cols):
        """Yield ``(row_start, row_stop, counts)`` over the scan grid: the counts of session-orientation pixels
        ``(rows, cols)`` as a contiguous Torch MPS ``(block_rows x scan_cols, pixel)`` tensor, scan-major, holding the
        source dtype's bits in a signed view (Torch lacks unsigned kernels), invalid pixels 0, padding filled.

        Each block decodes its scan rows over the requested pixels' bounding detector rectangle only and is freed
        before the next.
        """
        import torch

        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        source_rows, source_cols = (cols, rows) if self.crop.flipped else (rows, cols)
        region = (int(source_rows.min()), int(source_rows.max()) + 1, int(source_cols.min()), int(source_cols.max()) + 1)
        region_width = region[3] - region[2]
        flat = (source_rows - region[0]) * region_width + (source_cols - region[2])
        index = torch.as_tensor(flat, dtype=torch.int64, device="mps")
        invalid = np.flatnonzero(~self.crop.valid[source_rows, source_cols])
        invalid = None if invalid.size == 0 else torch.as_tensor(invalid, dtype=torch.int64, device="mps")
        signed = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[self.dtype.itemsize]
        pattern = None
        if self._pattern is not None:
            source_pattern = self._pattern.T if self.crop.flipped else self._pattern
            pattern = torch.as_tensor(np.ascontiguousarray(source_pattern[source_rows, source_cols]).view(
                {1: np.uint8, 2: np.int16, 4: np.int32}[self.dtype.itemsize]), device="mps")
        measured_rows, measured_cols = self.measured_shape
        top, left = self.offset
        grid_cols = self.scan_shape[1]
        step = self.rows_per_block(rows.size, (region[1] - region[0]) * region_width)
        for row_start in range(0, self.scan_shape[0], step):
            row_stop = min(self.scan_shape[0], row_start + step)
            first, last = max(row_start, top), min(row_stop, top + measured_rows)
            measured = None
            if first < last:
                counts = self.crop.read_rows(first - top, last - top, region)
                counts = counts.reshape(-1, counts.shape[-2] * counts.shape[-1]).view(signed)
                measured = counts.index_select(1, index)
                del counts
                if invalid is not None:
                    measured[:, invalid] = 0
            if pattern is None:
                yield row_start, row_stop, measured
                continue
            block = pattern.expand(row_stop - row_start, grid_cols, rows.size).clone()
            if measured is not None:
                block[first - row_start:last - row_start, left:left + measured_cols] = measured.reshape(
                    last - first, measured_cols, rows.size)
            del measured
            yield row_start, row_stop, block.reshape(-1, rows.size)

    def columns_float32_into(self, rows, cols, out: np.ndarray) -> np.ndarray:
        """Copy the requested columns as ``(BF, scan)`` float32 into ``out``, the column contract of every MPS source."""
        out = np.asarray(out)
        for row_start, row_stop, block in self.scan_row_blocks(rows, cols):
            start, stop = row_start * self.scan_shape[1], row_stop * self.scan_shape[1]
            out[:, start:stop] = block.cpu().numpy().view(self.dtype).T
        return out


def staged_tensor(mx, tensor, dtype):
    """``tensor`` (Torch MPS) as an MLX array of ``dtype`` with the same bytes, copied on the GPU.

    The MLX storage is a whole number of pages, which Metal needs to wrap it without a copy.
    """
    count = int(tensor.numel())
    itemsize = np.dtype(str(dtype).removeprefix("mlx.core.")).itemsize
    page = 16384 // itemsize
    staged = mx.zeros((-(-count // page) * page,), dtype=dtype)
    mx.eval(staged)
    copy_tensor_into(tensor, np.asarray(staged))
    return staged[:count].reshape(tuple(int(size) for size in tensor.shape))


def copy_tensor_into(tensor, out: np.ndarray) -> np.ndarray:
    """Copy a Torch MPS tensor's bytes into the start of ``out``, the host view of an MLX array, on the GPU."""
    view = _mps_tensor_view(tensor)
    if view.nbytes > out.nbytes:
        raise ValueError(f"The tensor has {view.nbytes} bytes, more than the output's {out.nbytes}.")
    target = metal_device().newBufferWithBytesNoCopy_length_options_deallocator_(
        out, out.nbytes, metal_module().MTLResourceStorageModeShared, None,
    )
    if target is None:
        raise RuntimeError("Metal could not wrap the MLX output; use a contiguous MLX array.")
    try:
        command = metal_queue().commandBuffer()
        encoder = command.blitCommandEncoder()
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(view._mtl, 0, target, 0, view.nbytes)
        encoder.endEncoding()
        complete_command(command, "SSB MPS column transfer")
    finally:
        target.release()
    return out


def as_chunked_frames(data):
    """Return the column source MPS SSB reads for ``data``: BF-column frames as given, arrays and tensors wrapped.

    Integer MPS tensors are read in place through their Metal storage (``TensorChunkedFrames``); float tensors are
    gathered with Torch (``MpsTensorFrames``); NumPy arrays through ``ArrayFrames``; an encoded bright-field crop
    through ``EncodedBrightFieldFrames``.
    """
    if isinstance(data, (MpsBfColumnFrames, ChunkedFrames, ArrayFrames, MpsTensorFrames, EncodedBrightFieldFrames)):
        return data
    if isinstance(data, EncodedBrightFieldCrop):
        return EncodedBrightFieldFrames(data)
    if isinstance(data, np.ndarray) and data.ndim in (3, 4):
        return ArrayFrames(data)
    import torch

    if torch.is_tensor(data) and data.device.type == "mps" and data.ndim in (3, 4):
        if data.dtype not in (torch.uint8, torch.uint16, torch.uint32):
            return MpsTensorFrames(data)
        return TensorChunkedFrames(data)
    raise TypeError(
        "MPS SSB reads a 3D/4D NumPy array or MPS tensor, or exact BF columns; "
        "pass a loaded dataset to quantem.gpu.SSB, which reads its bright-field crop."
    )


class TensorChunkedFrames(ChunkedFrames):
    """``ChunkedFrames`` over the Metal storage of an integer Torch MPS tensor, read without a copy.

    Keeps the tensor alive, and marks its columns as array input: preparation lays them out scan-major on the GPU so
    their FFT layout and rounding match a NumPy array of the same counts.
    """

    def __init__(self, tensor) -> None:
        view = _mps_tensor_view(tensor)
        super().__init__([view.reshape(-1, *tensor.shape[-2:])])
        self.tensor = tensor
        if tensor.ndim == 4:
            self.metadata["scan_shape"] = tuple(tensor.shape[:2])


class ArrayFrames:
    """Flat detector-column view over a 4D crop-first array.

    Metal reconstruction and fitting stream one detector pixel over all scan
    positions.  Integer MPS tensors provide that through ``ChunkedFrames``;
    this adapter gives NumPy arrays the same ``column(row, col)`` contract.
    """

    def __init__(self, data):
        array = np.asarray(data)
        if array.ndim == 4:
            self.scan_shape = (int(array.shape[0]), int(array.shape[1]))
            self.detector_shape = (int(array.shape[2]), int(array.shape[3]))
            self._flat = array.reshape(-1, *self.detector_shape)
        elif array.ndim == 3:
            self.scan_shape = None
            self.detector_shape = (int(array.shape[1]), int(array.shape[2]))
            self._flat = array
        else:
            raise TypeError(
                "MPS SSB preview expects 3D/4D detector data or chunk-backed "
                f"MPS data, got shape {array.shape}."
            )
        self.shape = tuple(int(size) for size in self._flat.shape)
        self.ndim = 3
        self.dtype = self._flat.dtype
        self.detector_sum = None

    def __array__(self, dtype=None):
        array = np.asarray(self._flat)
        return array.astype(dtype, copy=False) if dtype is not None else array

    def reshape(self, *shape, **kwargs):
        return self._flat.reshape(*shape, **kwargs)

    def column(self, row: int, col: int) -> np.ndarray:
        return np.asarray(self._flat[:, int(row), int(col)])

    def columns(self, rows, cols) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.intp).reshape(-1)
        cols = np.asarray(cols, dtype=np.intp).reshape(-1)
        if rows.shape != cols.shape:
            raise ValueError("rows and cols must have matching shapes.")
        flat_indices = rows * int(self.detector_shape[1]) + cols
        flat = np.asarray(self._flat).reshape(int(self._flat.shape[0]), -1)
        return np.take(flat, flat_indices, axis=1).T

    def columns_float32_into(self, rows, cols, out: np.ndarray) -> np.ndarray:
        """Copy the requested columns as ``(BF, scan)`` float32 into ``out``, the column contract of every MPS source.

        ``DirectPtycho.export_brightfield`` streams any session's columns through this call; uint8 and uint16 counts are exact
        in float32.
        """
        out[...] = self.columns(rows, cols)
        return out


class MpsTensorFrames:
    """Gather tensor detector columns on MPS into MLX-owned shared storage."""

    def __init__(self, data):
        self.tensor = data
        self.scan_shape = tuple(data.shape[:2]) if data.ndim == 4 else None
        self.detector_shape = tuple(data.shape[-2:])
        self.shape = (int(data.numel() // np.prod(self.detector_shape)), *self.detector_shape)
        self.ndim = 3
        self.dtype = np.dtype(str(data.dtype).removeprefix("torch."))
        self._flat = data.reshape(self.shape[0], -1)

    def columns_float32_into(self, rows, cols, out):
        """Gather with Torch MPS, then copy directly into the MLX Metal buffer."""
        import torch

        indices = torch.as_tensor(
            np.asarray(rows) * self.detector_shape[1] + np.asarray(cols),
            dtype=torch.int64, device="mps",
        )
        values = self._flat.index_select(1, indices).T.to(torch.float32).contiguous()
        view = _mps_tensor_view(values)
        buffer = view._mtl
        target = metal_device().newBufferWithBytesNoCopy_length_options_deallocator_(
            out, out.nbytes, metal_module().MTLResourceStorageModeShared, None,
        )
        if target is None:
            raise RuntimeError("Metal could not wrap the MLX output; use contiguous float32 output.")
        try:
            command = metal_queue().commandBuffer()
            encoder = command.blitCommandEncoder()
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                buffer, 0, target, 0, out.nbytes,
            )
            encoder.endEncoding()
            complete_command(command, "SSB MPS column transfer")
        finally:
            target.release()
        return out


def frames_scan_shape(frames) -> tuple[int, int]:
    """Return the ``(row, col)`` scan shape of an MPS frame source.

    A source that does not record one (3D frames) must hold a square scan, the
    only shape its frame count alone determines.
    """
    if isinstance(frames, (MpsBfColumnFrames, ArrayFrames, MpsTensorFrames, EncodedBrightFieldFrames)):
        shape = frames.scan_shape
    elif isinstance(frames, ChunkedFrames):
        shape = frames.metadata.get("scan_shape")
    else:
        raise TypeError(f"Unsupported MPS frame source: {type(frames).__name__}.")
    if shape is not None:
        return int(shape[0]), int(shape[1])
    frame_count = int(frames.shape[0])
    side = round(frame_count ** 0.5)
    if side * side != frame_count:
        raise ValueError("scan_shape is required for non-square frame counts.")
    return side, side


def mlx_count_type(mx, dtype: np.dtype):
    """The MLX dtype holding the same unsigned counts as ``dtype``."""
    return {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[np.dtype(dtype).itemsize]


def _read_file_rows(path: Path, indices: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Read rows ``indices`` of the headerless row-major file ``path`` into the C-contiguous ``out``, one row each.

    A row is ``out.shape[1]`` items of ``out.dtype``. Runs of consecutive rows are read together, at most
    ``COLUMN_READ_BYTES`` per ``os.preadv``, on ``COLUMN_READ_THREADS`` threads, straight into ``out``.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if out.shape[0] != indices.size or not out.flags.c_contiguous:
        raise ValueError(f"Expected a C-contiguous output of {indices.size} rows, got shape {out.shape}.")
    if indices.size == 0:
        return out
    row_bytes = int(out.shape[1]) * out.itemsize
    rows_per_read = max(1, COLUMN_READ_BYTES // max(1, row_bytes))
    breaks = (np.flatnonzero(np.diff(indices) != 1) + 1).tolist()
    reads = [
        (start, min(start + rows_per_read, stop))
        for run_start, stop in zip([0, *breaks], [*breaks, int(indices.size)])
        for start in range(run_start, stop, rows_per_read)
    ]
    rows = out.reshape(int(out.shape[0]), -1)
    descriptor = os.open(path, os.O_RDONLY)

    def read(span: tuple[int, int]) -> None:
        start, stop = span
        target = memoryview(rows[start:stop]).cast("B")
        offset = int(indices[start]) * row_bytes
        done = 0
        while done < target.nbytes:
            count = os.preadv(descriptor, [target[done:]], offset + done)
            if count <= 0:
                raise OSError(f"{path} ended {target.nbytes - done} bytes before row {int(indices[start])}'s data.")
            done += count

    try:
        if len(reads) == 1:
            read(reads[0])
        else:
            with ThreadPoolExecutor(min(COLUMN_READ_THREADS, len(reads))) as pool:
                for _ in pool.map(read, reads):
                    pass
    finally:
        os.close(descriptor)
    return out


class _BfColumnDetectorView:
    """Minimal detector geometry used by the shared Metal compute backend."""

    def __init__(self, detector_shape: tuple[int, int]) -> None:
        self.det = tuple(int(value) for value in detector_shape)


def _mps_tensor_view(data):
    """Borrow contiguous Torch MPS storage without copying detector values."""
    import torch

    data = data.contiguous()
    if data.storage_offset():
        data = data.clone()
    torch.mps.synchronize()
    buffer = tensor_buffer(data)
    view = numpy_view(buffer, str(data.dtype).removeprefix("torch."), data.numel())
    view = view.reshape(tuple(data.shape)).view(SharedArray)
    view._mtl = buffer
    # The tensor owns this buffer. Never call release() on the borrowed buffer.
    view._owner = data
    return view


@dataclass(frozen=True)
class _BfColumnCompanion:
    """Validated exact BF-column source declared by an export."""

    path: Path
    dtype: np.dtype
    max_value: int | None
    detector_bin: int
    provenance: dict[str, object]


class _BfColumnCompanionNotDeclared(FileNotFoundError):
    """No exact BF-column declaration exists at this candidate location."""


def _bf_column_dtype(encoding: object, *, source: Path) -> np.dtype:
    """Return the exact integer dtype declared by a BF-column source."""
    token = str(encoding).lower()
    if token in {"u8", "uint8"}:
        return np.dtype(np.uint8)
    if token in {"u16", "uint16"}:
        return np.dtype(np.uint16)
    raise ValueError(f"Unsupported BF-column encoding {encoding!r}: {source}")


def _calibration_companion(
    cal_path: Path,
    payload: dict[str, object],
) -> _BfColumnCompanion:
    """Resolve the declaration a ShowPtycho folder export writes into its calibration (path and encoding)."""
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
    dtype = _bf_column_dtype(payload["bf_column_encoding"], source=cal_path)
    max_value = None
    for manifest_path in (
        cal_path.parent / "manifest.json",
        cal_path.parent.parent / "manifest.json",
    ):
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bf_meta = (manifest.get("source") or {}).get("bf_columns") or {}
        if bf_meta.get("max_value") is not None:
            max_value = int(bf_meta["max_value"])
            break
    detector_bin = int(payload.get("detector_bin", 1) or 1)
    return _BfColumnCompanion(
        path=bf_path,
        dtype=dtype,
        max_value=max_value,
        detector_bin=detector_bin,
        provenance={
            "declaration": "calibration",
            "calibration_path": str(cal_path),
            "detector_bin": detector_bin,
            "detector_bin_source": (
                "calibration" if "detector_bin" in payload else "default"
            ),
        },
    )


def _resolve_bf_column_companion(
    cal_path: Path,
    payload: dict[str, object],
) -> _BfColumnCompanion:
    """Resolve the BF-column source a calibration declares (``bf_column_companion_path`` and ``bf_column_encoding``)."""
    in_calibration = (
        "bf_column_companion_path" in payload
        or "bf_column_encoding" in payload
    )
    if not in_calibration:
        raise _BfColumnCompanionNotDeclared(
            f"No exact BF-column companion is declared: {cal_path}"
        )
    missing = [
        name
        for name in ("bf_column_companion_path", "bf_column_encoding")
        if name not in payload
    ]
    if missing:
        raise ValueError(
            "Calibration has an incomplete BF-column declaration "
            f"{missing}: {cal_path}"
        )
    return _calibration_companion(cal_path, payload)
