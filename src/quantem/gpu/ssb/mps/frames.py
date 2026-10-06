"""Detector-column sources that MPS SSB reads: NumPy arrays, MPS tensors, decoded frames and exact BF-column exports.

MPS SSB streams one bright-field detector pixel over all scan positions at a time, so every source answers "these
(row, col) columns as float32". An export from ShowPtycho or ``SSB.export_brightfield`` keeps exactly those columns as
integers on disk with a declaration that names them; ``find_bf_columns`` looks for one beside the acquisition before
``SSB.open`` falls back to decoding the detector.
"""

import json
import time
from dataclasses import dataclass
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
    required_fields = (
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
    companion = _resolve_bf_column_companion(cal_path, payload)
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
    # The export's detector calibration: the stored disk radius is the selection's, not the disk edge it was measured on.
    stored_sampling = payload.get("det_sampling_mrad_px")
    det_sampling = None
    if stored_sampling is not None:
        sampling = np.asarray(stored_sampling, dtype=np.float64)
        if sampling.shape != (2,) or not bool(np.all(np.isfinite(sampling) & (sampling > 0))):
            raise ValueError(
                f"Calibration det_sampling_mrad_px must be two positive (row, col) values: {cal_path}"
            )
        det_sampling = (float(sampling[0]), float(sampling[1]))
    return MpsBfColumnFrames(
        companion.path,
        selection=selection,
        scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
        dtype=companion.dtype,
        detector_bin=companion.detector_bin,
        source_provenance=companion.provenance,
        max_value=companion.max_value,
        dc_value=dc_value,
        det_sampling=det_sampling,
        verbose=verbose,
    )


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
        detector_bin: int = 1,
        source_provenance: dict[str, object] | None = None,
        max_value: int | None = None,
        detector_sum: np.ndarray | None = None,
        dc_value: complex | None = None,
        det_sampling: tuple[float, float] | None = None,
        verbose: bool = False,
    ) -> None:
        self.source_path = Path(path).expanduser().resolve()
        # (row, col) mrad per detector pixel the columns were exported with; SSB uses it when given none
        self.det_sampling = det_sampling
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
        self.det_bin = int(detector_bin)
        if self.det_bin < 1:
            raise ValueError("detector_bin must be a positive integer.")
        self.source_provenance = dict(source_provenance or {})
        self.vi = _BfColumnDetectorView(self.det_shape)
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
        self.det_shape = tuple(data.shape[-2:])
        self.shape = (int(data.numel() // np.prod(self.det_shape)), *self.det_shape)
        self.ndim = 3
        self.dtype = np.dtype(str(data.dtype).removeprefix("torch."))
        self._flat = data.reshape(self.shape[0], -1)

    def columns_float32_into(self, rows, cols, out):
        """Gather with Torch MPS, then copy directly into the MLX Metal buffer."""
        import torch

        indices = torch.as_tensor(
            np.asarray(rows) * self.det_shape[1] + np.asarray(cols),
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
    if isinstance(frames, (MpsBfColumnFrames, ArrayFrames, MpsTensorFrames)):
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


def _linked_manifest(
    cal_path: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Return the linked manifest and its BF-column metadata."""
    for candidate in (
        cal_path.parent / "manifest.json",
        cal_path.parent.parent / "manifest.json",
    ):
        if not candidate.is_file():
            continue
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
        bf_meta = (manifest.get("source") or {}).get("bf_columns")
        if not bf_meta:
            continue
        calibration = manifest.get("calibration")
        if not isinstance(calibration, str) or not calibration:
            raise ValueError(
                f"BF-column manifest must identify its calibration file: {candidate}"
            )
        linked = (candidate.parent / calibration).resolve()
        if linked != cal_path:
            raise ValueError(
                f"BF-column manifest {candidate} links {linked}, not {cal_path}."
            )
        return candidate.resolve(), manifest, bf_meta
    raise ValueError(
        "Calibration declares an exact BF-column companion, but no linked "
        f"manifest with source.bf_columns was found for {cal_path}."
    )


def _validate_storage(
    manifest_path: Path,
    bf_meta: dict[str, object],
) -> tuple[Path, np.dtype]:
    """Validate the companion path and integer encoding."""
    if bf_meta.get("kind") != "bf_columns":
        raise ValueError(
            f"BF-column manifest kind must be 'bf_columns': {manifest_path}"
        )
    if bf_meta.get("order") != "bf,scan":
        raise ValueError(
            f"BF-column manifest order must be 'bf,scan': {manifest_path}"
        )
    relative = bf_meta.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"BF-column manifest path is missing: {manifest_path}")
    export_root = manifest_path.parent.resolve()
    bf_path = (export_root / relative).resolve()
    if not bf_path.is_relative_to(export_root):
        raise ValueError(
            f"BF-column path must stay inside the export folder: {manifest_path}"
        )
    if not bf_path.is_file():
        raise FileNotFoundError(f"Exact BF-column companion was not found: {bf_path}")

    dtype = _bf_column_dtype(bf_meta.get("encoding"), source=manifest_path)
    declared_dtype = bf_meta.get("dtype")
    if declared_dtype is not None and np.dtype(str(declared_dtype)) != dtype:
        raise ValueError(
            f"BF-column dtype {declared_dtype!r} disagrees with encoding "
            f"{bf_meta.get('encoding')!r}: {manifest_path}"
        )
    expected_suffix = ".u8" if dtype == np.dtype(np.uint8) else ".u16"
    if bf_path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"BF-column filename suffix must be {expected_suffix} for {dtype}: {bf_path}"
        )
    return bf_path, dtype


def _validate_geometry(
    cal_path: Path,
    manifest_path: Path,
    payload: dict[str, object],
    bf_meta: dict[str, object],
) -> tuple[list[int], list[int], int]:
    """Validate scan, BF, and detector coordinate grids."""
    scan_region = payload.get("scan_region")
    scan_shape = scan_region.get("shape") if isinstance(scan_region, dict) else None
    if not isinstance(scan_shape, list) or len(scan_shape) != 2:
        raise ValueError(f"Calibration has no valid 2D scan shape: {cal_path}")
    scan_shape = [int(value) for value in scan_shape]
    if bf_meta.get("scan_shape") != scan_shape:
        raise ValueError(
            f"BF-column scan shape {bf_meta.get('scan_shape')} does not match "
            f"calibration {scan_shape}: {manifest_path}"
        )

    rows = payload.get("bf_rows")
    cols = payload.get("bf_cols")
    if not isinstance(rows, list) or not isinstance(cols, list) or len(rows) != len(cols):
        raise ValueError(f"Calibration BF coordinates are invalid: {cal_path}")
    expected_shape = [len(rows), int(np.prod(scan_shape))]
    if bf_meta.get("shape") != expected_shape:
        raise ValueError(
            f"BF-column shape {bf_meta.get('shape')} does not match "
            f"{expected_shape}: {manifest_path}"
        )

    working_shape = payload.get("detector_shape")
    column_shape = bf_meta.get("detector_shape")
    if (
        not isinstance(working_shape, list)
        or len(working_shape) != 2
        or not isinstance(column_shape, list)
        or len(column_shape) != 2
    ):
        raise ValueError(
            f"BF-column detector shapes must be two-dimensional: {manifest_path}"
        )
    working_shape = [int(value) for value in working_shape]
    column_shape = [int(value) for value in column_shape]
    if any(value <= 0 for value in working_shape + column_shape):
        raise ValueError(f"BF-column detector shapes must be positive: {manifest_path}")
    if column_shape != working_shape:
        raise ValueError(
            "BF-column coordinates use detector shape "
            f"{column_shape}, but calibration uses {working_shape}. Exact "
            "columns cannot infer detector binning; export values on the "
            f"calibration grid and declare detector_bin explicitly: {manifest_path}"
        )

    declared_bin = bf_meta.get("detector_bin", payload.get("detector_bin"))
    detector_bin = 1 if declared_bin is None else int(declared_bin)
    if detector_bin < 1:
        raise ValueError(f"detector_bin must be a positive integer: {manifest_path}")
    return scan_shape, expected_shape, detector_bin


def _validate_byte_count(
    manifest_path: Path,
    bf_path: Path,
    bf_meta: dict[str, object],
    dtype: np.dtype,
    expected_shape: list[int],
) -> int:
    """Validate payload size and return the expected byte count."""
    expected_bytes = int(np.prod(expected_shape)) * dtype.itemsize
    declared_bytes = bf_meta.get("bytes")
    actual_bytes = bf_path.stat().st_size
    if declared_bytes != expected_bytes or actual_bytes != expected_bytes:
        raise ValueError(
            "BF-column byte count mismatch: "
            f"declared={declared_bytes}, expected={expected_bytes}, "
            f"actual={actual_bytes}: {manifest_path}"
        )
    if bf_meta.get("bytes_per_bf") not in {
        None,
        expected_shape[1] * dtype.itemsize,
    }:
        raise ValueError(f"BF-column bytes_per_bf is inconsistent: {manifest_path}")
    if bf_meta.get("bits_per_value") not in {None, dtype.itemsize * 8}:
        raise ValueError(f"BF-column bits_per_value is inconsistent: {manifest_path}")
    return expected_bytes


def _manifest_companion(
    cal_path: Path,
    payload: dict[str, object],
) -> _BfColumnCompanion:
    """Resolve and validate the current manifest-declared source."""
    manifest_path, _manifest, bf_meta = _linked_manifest(cal_path)
    bf_path, dtype = _validate_storage(manifest_path, bf_meta)
    _scan_shape, expected_shape, detector_bin = _validate_geometry(
        cal_path,
        manifest_path,
        payload,
        bf_meta,
    )
    expected_bytes = _validate_byte_count(
        manifest_path,
        bf_path,
        bf_meta,
        dtype,
        expected_shape,
    )
    max_value = bf_meta.get("max_value")
    if max_value is not None:
        max_value = int(max_value)
        if max_value < 0 or max_value > int(np.iinfo(dtype).max):
            raise ValueError(f"BF-column max_value is invalid for {dtype}: {manifest_path}")
    declared_bin = bf_meta.get("detector_bin", payload.get("detector_bin"))
    return _BfColumnCompanion(
        path=bf_path,
        dtype=dtype,
        max_value=max_value,
        detector_bin=detector_bin,
        provenance={
            "declaration": "manifest",
            "manifest_path": str(manifest_path),
            "calibration_path": str(cal_path),
            "order": "bf,scan",
            "shape": expected_shape,
            "bytes": expected_bytes,
            "detector_shape": [int(value) for value in payload["detector_shape"]],
            "detector_bin": detector_bin,
            "detector_bin_source": (
                "declared" if declared_bin is not None else "default"
            ),
        },
    )


def _resolve_bf_column_companion(
    cal_path: Path,
    payload: dict[str, object],
) -> _BfColumnCompanion:
    """Resolve one declared BF-column source without a silent fallback."""
    in_calibration = (
        "bf_column_companion_path" in payload
        or "bf_column_encoding" in payload
    )
    if in_calibration:
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

    declared = payload.get("bf_column_companion")
    transport = payload.get("source_transport")
    if declared is True or transport == "bf_columns":
        if declared is False:
            raise ValueError(
                f"Calibration disables BF columns but requests their transport: {cal_path}"
            )
        if declared not in {None, True}:
            raise ValueError(f"bf_column_companion must be true or absent: {cal_path}")
        return _manifest_companion(cal_path, payload)
    if declared not in {None, False}:
        raise ValueError(f"bf_column_companion must be boolean: {cal_path}")
    raise _BfColumnCompanionNotDeclared(
        f"No exact BF-column companion is declared: {cal_path}"
    )
