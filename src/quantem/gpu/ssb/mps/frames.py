"""Detector-column sources that MPS SSB reads: NumPy arrays, MPS tensors, decoded frames and exact BF-column exports.

MPS SSB streams one bright-field detector pixel over all scan positions at a time, so every source answers "these
(row, col) columns as float32". An export from ShowPtycho or ``SSB.export_brightfield`` keeps exactly those columns as
integers on disk with a declaration that names them; ``find_bf_columns`` looks for one beside the acquisition before
``SSB.open`` falls back to decoding the detector.
"""

import json
import time
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
from quantem.gpu.ssb.brightfield import BrightfieldDisk
from quantem.gpu.ssb.mps.hardware import require_mlx

# The layout of a ShowPtycho folder's snapshots/cal.json as quantem.widget writes it; another value is refused.
# 2: keys follow the shared vocabulary (bright_field_rows, bright_field_cols, bright_field_center_px,
#    bright_field_radius_px, detector_sampling_mrad), scan_region is [row_start, row_stop, col_start, col_stop]
#    and scan_shape is [rows, cols]; earlier exports wrote bf_* keys and a scan_region object.
BROWSER_CALIBRATION_SCHEMA = "quantem.ssb-browser-calibration/2"


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

    The stored values remain integer-exact. File reads and unified-memory copies
    are host I/O, while detector sums, FFTs, objectives, refinement, and final
    reconstruction use MPS/Metal. Only requested detector columns are copied into
    MLX storage; the full detector stack is never materialized.
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
        # (row, col) mrad per detector pixel the columns were exported with; SSB uses it when given none
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
        """Reduce BF columns on Metal without a CPU scientific fallback."""
        mx = require_mlx()
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
        return np.asarray(self._columns[self._indices(rows, cols)])

    def columns_float32_into(
        self,
        rows,
        cols,
        out: np.ndarray,
    ) -> np.ndarray:
        """Copy requested exact columns directly into MLX unified storage."""
        indices = self._indices(rows, cols)
        expected_shape = (int(indices.size), self._n)
        if tuple(int(value) for value in out.shape) != expected_shape:
            raise ValueError(
                f"Output shape {out.shape} does not match {expected_shape}."
            )
        for start in range(0, int(indices.size), 32):
            stop = min(start + 32, int(indices.size))
            out[start:stop] = self._columns[indices[start:stop]]
        return out


def as_chunked_frames(data):
    """Return the column source MPS SSB reads for ``data``: BF-column frames as given, arrays and tensors wrapped.

    Integer MPS tensors are read in place through their Metal storage (``TensorChunkedFrames``); float tensors are
    gathered with Torch (``MpsTensorFrames``); NumPy arrays through ``ArrayFrames``.
    """
    if isinstance(data, (MpsBfColumnFrames, ChunkedFrames, ArrayFrames, MpsTensorFrames)):
        return data
    if isinstance(data, np.ndarray) and data.ndim in (3, 4):
        return ArrayFrames(data)
    import torch

    if torch.is_tensor(data) and data.device.type == "mps" and data.ndim in (3, 4):
        if data.dtype not in (torch.uint8, torch.uint16, torch.uint32):
            return MpsTensorFrames(data)
        return TensorChunkedFrames(data)
    raise TypeError(
        "MPS SSB reads a 3D/4D NumPy array or MPS tensor, or exact BF columns; "
        "pass a loaded dataset to quantem.gpu.SSB, which decodes its bright-field crop."
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

        ``SSB.export_brightfield`` streams any session's columns through this call; uint8 and uint16 counts are exact
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
    if isinstance(frames, (MpsBfColumnFrames, ArrayFrames, MpsTensorFrames)):
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
