"""Validated QH5-indexed Metal loading for prepared 4D-STEM sources.

This module consumes the package-owned ``QH5IDX01`` index and value-range
audit written by :mod:`Native4DSTEMIO`.  It never infers that a uint16 source
fits in eight bits: the low-byte path is enabled only when the requested
master, every shard, every index, the bad-pixel mask, and the source-bound
audit still agree.  Callers retain the complete decoder as their fallback.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_INDEX_MAGIC = b"QH5IDX01"
_AUDIT_SCHEMAS = {
    "quantem.gpu.value-range-audit/v1",
    "live4dstem.value-range-audit/v1",
}
_SOURCE_HASH_SCHEMA = "quantem.gpu.source-hashes/v1"
_SHA256_HEX_LENGTH = 64
_pipeline_cache: tuple[Any, Any, Any, Any, Any] | None = None


class QH5PreparedSourceError(ValueError):
    """A prepared source is stale, incomplete, or scientifically incompatible."""


@dataclass(frozen=True)
class QH5IndexChunk:
    """One contiguous compressed range described by a QH5 index."""

    start_frame: int
    n_frames: int
    range_start: int
    range_end: int
    metadata_offset_words: int
    metadata_words: int


@dataclass(frozen=True)
class QH5Index:
    """Validated metadata and block words for one HDF5 shard."""

    path: Path
    source_path: Path
    source_bytes: int
    source_mtime_ns: int
    detector_shape: tuple[int, int]
    n_frames: int
    source_dtype: str
    block_elements: int
    blocks_per_frame: int
    chunks: tuple[QH5IndexChunk, ...]
    words: np.ndarray


@dataclass(frozen=True)
class QH5PreparedSource:
    """Source-identity-bound inputs for the audited detector-bin-4 path."""

    directory: Path
    master_path: Path
    data_files: tuple[Path, ...]
    indexes: tuple[QH5Index, ...]
    scan_shape: tuple[int, int]
    detector_shape: tuple[int, int]
    source_dtype: str
    source_identity_sha256: str
    bad_pixel_indices: tuple[int, ...]
    audited_maximum: int
    audited_pixels_above_255: int

    @property
    def total_frames(self) -> int:
        return sum(index.n_frames for index in self.indexes)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QH5PreparedSourceError(
            f"Could not read prepared metadata {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QH5PreparedSourceError(f"Prepared metadata {path} must contain an object")
    return value


def _exact_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve(strict=True)


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != _SHA256_HEX_LENGTH or any(
        ch not in "0123456789abcdef" for ch in text
    ):
        raise QH5PreparedSourceError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _read_index(path: Path, source_path: Path) -> QH5Index:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise QH5PreparedSourceError(f"Could not read QH5 index {path}: {exc}") from exc
    if len(raw) < 16 or raw[:8] != _INDEX_MAGIC:
        raise QH5PreparedSourceError(f"{path} is not a QH5IDX01 index")
    json_bytes, word_count = struct.unpack_from("<II", raw, 8)
    json_stop = 16 + int(json_bytes)
    word_start = (json_stop + 3) & ~3
    word_bytes = int(word_count) * 4
    if json_stop > len(raw) or word_start + word_bytes != len(raw):
        raise QH5PreparedSourceError(f"{path} has truncated or trailing QH5 index data")
    try:
        metadata = json.loads(raw[16:json_stop])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QH5PreparedSourceError(f"{path} has invalid QH5 JSON metadata") from exc
    if not isinstance(metadata, dict):
        raise QH5PreparedSourceError(f"{path} QH5 metadata must contain an object")

    stat = source_path.stat()
    indexed_source = _exact_path(metadata.get("sourcePath", ""))
    if indexed_source != source_path:
        raise QH5PreparedSourceError(
            f"{path.name} indexes {indexed_source}, not {source_path}"
        )
    source_bytes = int(metadata.get("sourceBytes", -1))
    source_mtime_ns = int(metadata.get("sourceMtimeNs", -1))
    if source_bytes != stat.st_size or source_mtime_ns != stat.st_mtime_ns:
        raise QH5PreparedSourceError(f"{path.name} is stale for {source_path.name}")

    detector_shape = (int(metadata.get("detRows", 0)), int(metadata.get("detCols", 0)))
    n_frames = int(metadata.get("nFrames", 0))
    source_dtype = str(metadata.get("srcDtype", ""))
    block_elements = int(metadata.get("blockElems", 0))
    blocks_per_frame = int(metadata.get("nBlocksPerFrame", 0))
    if min(*detector_shape, n_frames, block_elements, blocks_per_frame) <= 0:
        raise QH5PreparedSourceError(f"{path.name} has invalid QH5 dimensions")

    chunks: list[QH5IndexChunk] = []
    next_frame = 0
    for raw_chunk in metadata.get("chunks", []):
        chunk = QH5IndexChunk(
            start_frame=int(raw_chunk.get("startFrame", -1)),
            n_frames=int(raw_chunk.get("nFrames", 0)),
            range_start=int(raw_chunk.get("rangeStart", -1)),
            range_end=int(raw_chunk.get("rangeEnd", -1)),
            metadata_offset_words=int(raw_chunk.get("metaOffsetWords", -1)),
            metadata_words=int(raw_chunk.get("metaWords", 0)),
        )
        expected_words = chunk.n_frames * blocks_per_frame * 2
        if (
            chunk.start_frame != next_frame
            or chunk.n_frames <= 0
            or chunk.range_start < 0
            or chunk.range_end <= chunk.range_start
            or chunk.range_end > source_bytes
            or chunk.metadata_offset_words < 0
            or chunk.metadata_words != expected_words
            or chunk.metadata_offset_words + chunk.metadata_words > word_count
        ):
            raise QH5PreparedSourceError(f"{path.name} has an invalid compressed range")
        chunks.append(chunk)
        next_frame += chunk.n_frames
    if not chunks or next_frame != n_frames:
        raise QH5PreparedSourceError(f"{path.name} does not cover all indexed frames")

    words = np.frombuffer(raw, dtype="<u4", count=word_count, offset=word_start).copy()
    for chunk in chunks:
        first_word = chunk.metadata_offset_words
        last_word = first_word + chunk.metadata_words
        pairs = words[first_word:last_word].reshape(-1, 2)
        starts = pairs[:, 0]
        lengths = pairs[:, 1]
        range_bytes = chunk.range_end - chunk.range_start
        if (
            np.any(lengths == 0)
            or np.any(lengths > range_bytes)
            or np.any(starts > range_bytes - lengths)
        ):
            raise QH5PreparedSourceError(
                f"{path.name} has a compressed block outside its indexed range"
            )
    return QH5Index(
        path=path,
        source_path=source_path,
        source_bytes=source_bytes,
        source_mtime_ns=source_mtime_ns,
        detector_shape=detector_shape,
        n_frames=n_frames,
        source_dtype=source_dtype,
        block_elements=block_elements,
        blocks_per_frame=blocks_per_frame,
        chunks=tuple(chunks),
        words=words,
    )


def _validate_source_snapshots(
    source_hashes: dict[str, Any],
    expected_paths: tuple[Path, ...],
    source_identity_sha256: str,
) -> None:
    if source_hashes.get("schema") != _SOURCE_HASH_SCHEMA:
        raise QH5PreparedSourceError(
            "Prepared source-hashes.json has an unsupported schema"
        )
    if (
        _require_sha256(source_hashes.get("aggregateHash"), "aggregateHash")
        != source_identity_sha256
    ):
        raise QH5PreparedSourceError(
            "Prepared source hashes disagree with the dataset identity"
        )
    snapshots = source_hashes.get("snapshots")
    if not isinstance(snapshots, list):
        raise QH5PreparedSourceError("Prepared source hashes do not contain snapshots")
    by_path: dict[Path, dict[str, Any]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        try:
            path = _exact_path(snapshot.get("path", ""))
        except (FileNotFoundError, OSError):
            continue
        by_path[path] = snapshot
    if set(by_path) != set(expected_paths):
        raise QH5PreparedSourceError(
            "Prepared source snapshots do not match the requested dataset"
        )
    for path in expected_paths:
        stat = path.stat()
        snapshot = by_path[path]
        if (
            int(snapshot.get("bytes", -1)) != stat.st_size
            or int(snapshot.get("modificationNanoseconds", -1)) != stat.st_mtime_ns
        ):
            raise QH5PreparedSourceError(
                f"Prepared source snapshot is stale for {path.name}"
            )


def open_prepared_source(
    directory: str | os.PathLike[str],
    *,
    master_path: str | os.PathLike[str] | None = None,
) -> QH5PreparedSource:
    """Validate one indexed source without reading its multi-gigabyte payload."""
    root = _exact_path(directory)
    dataset = _read_json(root / "dataset.json")
    audit = _read_json(root / "value-range-audit.json")
    source_hashes = _read_json(root / "source-hashes.json")

    prepared_master = _exact_path(dataset.get("masterPath", ""))
    if master_path is not None and _exact_path(master_path) != prepared_master:
        raise QH5PreparedSourceError(
            f"Prepared dataset is for {prepared_master}, not {_exact_path(master_path)}"
        )
    data_files = tuple(_exact_path(path) for path in dataset.get("dataFiles", []))
    index_files = tuple(_exact_path(path) for path in dataset.get("indexFiles", []))
    if not data_files or len(data_files) != len(index_files):
        raise QH5PreparedSourceError(
            "Prepared dataset has an incomplete HDF5 index set"
        )

    source_identity = _require_sha256(
        dataset.get("sourceIdentitySHA256"), "sourceIdentitySHA256"
    )
    if (
        _require_sha256(audit.get("sourceIdentitySHA256"), "audit source identity")
        != source_identity
    ):
        raise QH5PreparedSourceError("Value-range audit is for a different source")
    if audit.get("schema") not in _AUDIT_SCHEMAS:
        raise QH5PreparedSourceError("Value-range audit has an unsupported schema")

    source_dtype = str(dataset.get("sourceDtype", ""))
    bad_pixels = tuple(
        sorted(int(value) for value in dataset.get("badPixelIndices", []))
    )
    if len(set(bad_pixels)) != len(bad_pixels):
        raise QH5PreparedSourceError("Prepared bad-pixel indices contain duplicates")
    if str(audit.get("sourceDtype", "")) != source_dtype:
        raise QH5PreparedSourceError("Value-range audit has a different source dtype")
    if (
        tuple(sorted(int(value) for value in audit.get("badPixelIndices", [])))
        != bad_pixels
    ):
        raise QH5PreparedSourceError("Value-range audit has a different bad-pixel mask")
    maximum = int(audit.get("maximum", -1))
    above_255 = int(audit.get("pixelsAbove255", -1))
    if maximum < 0 or maximum > 255 or above_255 != 0:
        raise QH5PreparedSourceError(
            "Value-range audit does not prove an exact low-byte decode"
        )

    scan_shape = (int(dataset.get("scanRows", 0)), int(dataset.get("scanCols", 0)))
    detector_shape = (
        int(dataset.get("detectorRows", 0)),
        int(dataset.get("detectorCols", 0)),
    )
    detector_pixels = detector_shape[0] * detector_shape[1]
    if min(*scan_shape, *detector_shape) <= 0:
        raise QH5PreparedSourceError(
            "Prepared dataset has invalid scan or detector dimensions"
        )
    if any(index < 0 or index >= detector_pixels for index in bad_pixels):
        raise QH5PreparedSourceError("Prepared bad-pixel index is outside the detector")

    _validate_source_snapshots(
        source_hashes,
        (prepared_master, *data_files),
        source_identity,
    )
    indexes = tuple(
        _read_index(path, source) for path, source in zip(index_files, data_files)
    )
    if any(
        index.detector_shape != detector_shape
        or index.source_dtype != source_dtype
        or index.block_elements != 4096
        for index in indexes
    ):
        raise QH5PreparedSourceError(
            "QH5 index geometry disagrees with the prepared dataset"
        )
    total_frames = sum(index.n_frames for index in indexes)
    if total_frames != scan_shape[0] * scan_shape[1]:
        raise QH5PreparedSourceError("QH5 indexes do not cover the full scan")

    return QH5PreparedSource(
        directory=root,
        master_path=prepared_master,
        data_files=data_files,
        indexes=indexes,
        scan_shape=scan_shape,
        detector_shape=detector_shape,
        source_dtype=source_dtype,
        source_identity_sha256=source_identity,
        bad_pixel_indices=bad_pixels,
        audited_maximum=maximum,
        audited_pixels_above_255=above_255,
    )


class _MappedMetalFile:
    """Private file mapping kept alive for one Metal command sequence."""

    def __init__(self, path: Path, device: Any, storage_mode_shared: int):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            self.mapping = mmap.mmap(
                descriptor,
                0,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(descriptor)
        self.buffer = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            self.mapping,
            len(self.mapping),
            storage_mode_shared,
            None,
        )
        if self.buffer is None:
            self.mapping.close()
            raise MemoryError(f"Metal could not map {path.name}")

    def close(self) -> None:
        buffer, self.buffer = self.buffer, None
        if buffer is not None:
            buffer.release()
        mapping, self.mapping = self.mapping, None
        if mapping is not None:
            mapping.close()


class _SequentialReadAhead:
    """Populate source pages sequentially while Metal consumes prior pages."""

    def __init__(
        self,
        paths: tuple[Path, ...],
        *,
        lead_bytes: int = 64 * 1024 * 1024,
        buffer_bytes: int = 4 * 1024 * 1024,
    ):
        self.paths = paths
        self.lead_bytes = max(1, int(lead_bytes))
        self.buffer_bytes = max(4096, int(buffer_bytes))
        self.bytes_read = 0
        self.lead_ready = threading.Event()
        self.done = threading.Event()
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="quantem-gpu-qh5-readahead",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        buffer = bytearray(self.buffer_bytes)
        view = memoryview(buffer)
        try:
            for path in self.paths:
                if self.stop_requested.is_set():
                    break
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    while not self.stop_requested.is_set():
                        count = os.readv(descriptor, [view])
                        if count <= 0:
                            break
                        self.bytes_read += count
                        if self.bytes_read >= self.lead_bytes:
                            self.lead_ready.set()
                finally:
                    os.close(descriptor)
        finally:
            self.lead_ready.set()
            self.done.set()

    def wait_for_lead(self, timeout: float = 0.25) -> None:
        self.lead_ready.wait(timeout=max(0.0, float(timeout)))

    def stop(self) -> None:
        self.stop_requested.set()
        self.thread.join(timeout=0.1)


def _pipelines(device: Any) -> tuple[Any, Any, Any, Any]:
    global _pipeline_cache
    device_key = int(device.__c_void_p__().value)
    if _pipeline_cache is not None and _pipeline_cache[0] == device_key:
        return (
            _pipeline_cache[1],
            _pipeline_cache[2],
            _pipeline_cache[3],
            _pipeline_cache[4],
        )
    import Metal

    source = (
        Path(__file__).parents[3]
        / "swift/Sources/Metal4DSTEMKernels/Resources/qh5idx.metal"
    ).read_text()
    options = Metal.MTLCompileOptions.alloc().init()
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None or error:
        raise RuntimeError(f"Metal QH5 kernel compile error: {error}")
    decode = library.newFunctionWithName_("h5lz4dc_u16_audited_low8_scalar_qh5idx")
    detector_bin_row4 = library.newFunctionWithName_(
        "h5lz4dc_bin_u16_audited_low8_scalar_u16_frame_major_qh5idx"
    )
    detector_bin_row8 = library.newFunctionWithName_(
        "h5lz4dc_bin_u16_audited_low8_scalar_u16_frame_major_row8_qh5idx"
    )
    packed_full_precision = library.newFunctionWithName_(
        "h5lz4dc_unshuffle_u16_single_block_packed_h5"
    )
    decode_pipeline, decode_error = device.newComputePipelineStateWithFunction_error_(
        decode, None
    )
    bin_row4_pipeline, bin_row4_error = (
        device.newComputePipelineStateWithFunction_error_(detector_bin_row4, None)
    )
    bin_row8_pipeline, bin_row8_error = (
        device.newComputePipelineStateWithFunction_error_(detector_bin_row8, None)
    )
    packed_pipeline, packed_error = (
        device.newComputePipelineStateWithFunction_error_(packed_full_precision, None)
    )
    if decode_pipeline is None or decode_error:
        raise RuntimeError(f"Metal QH5 decode pipeline error: {decode_error}")
    if bin_row4_pipeline is None or bin_row4_error:
        raise RuntimeError(
            f"Metal QH5 row-4 detector-bin pipeline error: {bin_row4_error}"
        )
    if bin_row8_pipeline is None or bin_row8_error:
        raise RuntimeError(
            f"Metal QH5 row-8 detector-bin pipeline error: {bin_row8_error}"
        )
    if packed_pipeline is None or packed_error:
        raise RuntimeError(
            f"Metal packed-HDF5 uint16 pipeline error: {packed_error}"
        )
    _pipeline_cache = (
        device_key,
        decode_pipeline,
        bin_row4_pipeline,
        bin_row8_pipeline,
        packed_pipeline,
    )
    return decode_pipeline, bin_row4_pipeline, bin_row8_pipeline, packed_pipeline


def _packed_u16_pipeline(device: Any) -> Any:
    """Return the shared full-precision packed-HDF5 Metal pipeline."""
    return _pipelines(device)[3]


def _can_use_row8_bin(detector_shape: tuple[int, int]) -> bool:
    """Return whether adjacent detector-bin-4 pixels share aligned words."""
    _, detector_columns = detector_shape
    output_columns = detector_columns // 4
    return detector_columns % 32 == 0 and output_columns % 2 == 0


def load_audited_bin4(
    prepared: QH5PreparedSource,
    *,
    batch_scan_rows: int = 8,
    scalar_threads: int = 128,
    sequential_readahead: bool = True,
    profile_stages: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a complete audited uint16 source as exact detector-bin-4 uint16.

    This backend-specific function is intentionally not re-exported from
    :mod:`quantem.gpu.io`. The normal scientist-facing owner remains
    :func:`quantem.gpu.io.load`; this is its prepared-source implementation.
    """
    import Metal

    from . import decoder

    started = time.perf_counter()
    if prepared.source_dtype != "uint16":
        raise QH5PreparedSourceError("Audited detector-bin-4 requires a uint16 source")
    detector_rows, detector_columns = prepared.detector_shape
    if detector_rows % 4 or detector_columns % 4:
        raise QH5PreparedSourceError(
            "Audited detector-bin-4 requires complete 4x4 bins"
        )
    output_rows, output_columns = detector_rows // 4, detector_columns // 4
    output_pixels = output_rows * output_columns
    if output_pixels % 2:
        raise QH5PreparedSourceError("Audited frame-major output requires pixel pairs")
    if prepared.audited_maximum * 16 > np.iinfo(np.uint16).max:
        raise QH5PreparedSourceError("Audited detector-bin-4 sums do not fit uint16")
    batch_scan_rows = int(batch_scan_rows)
    if batch_scan_rows < 1:
        raise ValueError("batch_scan_rows must be at least 1")
    scalar_threads = max(32, min(1024, int(scalar_threads) // 32 * 32))
    device = decoder._device
    decode_pipeline, bin_row4_pipeline, bin_row8_pipeline, _ = _pipelines(device)
    use_row8_bin = _can_use_row8_bin(prepared.detector_shape)
    bin_pipeline = bin_row8_pipeline if use_row8_bin else bin_row4_pipeline
    pipeline_ready = time.perf_counter()
    queue = device.newCommandQueue()
    if queue is None:
        raise RuntimeError("Metal could not create a QH5 command queue")

    total_frames = prepared.total_frames
    frame_elements = detector_rows * detector_columns
    output_bytes = total_frames * output_pixels * np.dtype(np.uint16).itemsize
    output_buffer = decoder._metal_buffer_alloc(output_bytes)
    output_view = decoder._numpy_view(
        output_buffer, np.uint16, total_frames * output_pixels
    )
    scratch_frames = min(total_frames, batch_scan_rows * prepared.scan_shape[1])
    scratch_buffer = device.newBufferWithLength_options_(
        scratch_frames * frame_elements,
        Metal.MTLResourceStorageModePrivate,
    )
    bad_mask = decoder._metal_buffer_alloc(frame_elements)
    bad_mask_view = decoder._numpy_view(bad_mask, np.uint8, frame_elements)
    bad_mask_view[:] = 0
    if prepared.bad_pixel_indices:
        bad_mask_view[np.asarray(prepared.bad_pixel_indices, dtype=np.int64)] = 1
    count_audit = decoder._metal_buffer_alloc(total_frames * 4)
    count_audit_view = decoder._numpy_view(count_audit, np.uint32, total_frames)
    count_audit_view[:] = 0
    if scratch_buffer is None:
        decoder._release_metal_buffer(output_buffer)
        decoder._release_metal_buffer(bad_mask)
        decoder._release_metal_buffer(count_audit)
        raise MemoryError("Metal could not allocate audited low-byte scratch")
    allocated = time.perf_counter()

    metadata_buffers: list[Any] = []
    mapped_files: list[_MappedMetalFile] = []
    read_ahead: _SequentialReadAhead | None = None
    try:
        for index in prepared.indexes:
            metadata = decoder._metal_buffer_alloc(index.words.nbytes)
            decoder._numpy_view(metadata, np.uint32, index.words.size)[:] = index.words
            metadata_buffers.append(metadata)
        metadata_ready = time.perf_counter()
        if sequential_readahead:
            read_ahead = _SequentialReadAhead(prepared.data_files)
            read_ahead.wait_for_lead()
        read_ahead_ready = time.perf_counter()
        for path in prepared.data_files:
            mapped_files.append(
                _MappedMetalFile(path, device, Metal.MTLResourceStorageModeShared)
            )
        mapped = time.perf_counter()

        shard_starts: list[int] = []
        next_start = 0
        for index in prepared.indexes:
            shard_starts.append(next_start)
            next_start += index.n_frames

        commands: list[Any] = []
        decode_commands: list[Any] = []
        detector_bin_commands: list[Any] = []
        submitted_started = time.perf_counter()
        scan_rows, scan_columns = prepared.scan_shape
        direct_parameters = struct.pack(
            "<4I",
            detector_rows,
            detector_columns,
            output_columns,
            total_frames,
        )
        groups_per_frame = (output_pixels // 2 + 127) // 128
        for batch_row_start in range(0, scan_rows, batch_scan_rows):
            batch_row_stop = min(scan_rows, batch_row_start + batch_scan_rows)
            batch_start = batch_row_start * scan_columns
            batch_stop = batch_row_stop * scan_columns
            decode_command = queue.commandBuffer()
            decode_encoder = decode_command.computeCommandEncoder()
            detector_bin_command = (
                queue.commandBuffer() if profile_stages else decode_command
            )
            detector_bin_encoder = (
                detector_bin_command.computeCommandEncoder()
                if profile_stages
                else decode_encoder
            )
            decoded_frames = 0
            for shard_index, index in enumerate(prepared.indexes):
                shard_start = shard_starts[shard_index]
                for chunk in index.chunks:
                    chunk_start = shard_start + chunk.start_frame
                    chunk_stop = chunk_start + chunk.n_frames
                    run_start = max(batch_start, chunk_start)
                    run_stop = min(batch_stop, chunk_stop)
                    if run_start >= run_stop:
                        continue
                    frame_count = run_stop - run_start
                    decoded_frames += frame_count
                    scratch_offset = (run_start - batch_start) * frame_elements
                    metadata_frame_offset = run_start - chunk_start

                    decode_encoder.setComputePipelineState_(decode_pipeline)
                    decode_encoder.setBuffer_offset_atIndex_(
                        mapped_files[shard_index].buffer, 0, 0
                    )
                    decode_encoder.setBuffer_offset_atIndex_(
                        metadata_buffers[shard_index],
                        chunk.metadata_offset_words * 4,
                        1,
                    )
                    decode_encoder.setBytes_length_atIndex_(
                        struct.pack("<Q", chunk.range_start), 8, 2
                    )
                    decode_encoder.setBytes_length_atIndex_(
                        struct.pack("<I", index.blocks_per_frame), 4, 3
                    )
                    decode_encoder.setBytes_length_atIndex_(
                        struct.pack("<I", frame_elements), 4, 4
                    )
                    decode_encoder.setBuffer_offset_atIndex_(
                        scratch_buffer, scratch_offset, 5
                    )
                    decode_encoder.setBytes_length_atIndex_(
                        struct.pack("<I", metadata_frame_offset), 4, 6
                    )
                    decode_encoder.dispatchThreads_threadsPerThreadgroup_(
                        Metal.MTLSizeMake(frame_count * index.blocks_per_frame, 1, 1),
                        Metal.MTLSizeMake(scalar_threads, 1, 1),
                    )
                    if not profile_stages:
                        decode_encoder.memoryBarrierWithScope_(
                            Metal.MTLBarrierScopeBuffers
                        )

                    detector_bin_encoder.setComputePipelineState_(bin_pipeline)
                    detector_bin_encoder.setBuffer_offset_atIndex_(
                        scratch_buffer, scratch_offset, 0
                    )
                    detector_bin_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
                    detector_bin_encoder.setBuffer_offset_atIndex_(bad_mask, 0, 2)
                    detector_bin_encoder.setBuffer_offset_atIndex_(count_audit, 0, 3)
                    detector_bin_encoder.setBytes_length_atIndex_(
                        struct.pack("<I", run_start), 4, 4
                    )
                    detector_bin_encoder.setBytes_length_atIndex_(
                        struct.pack("<I", frame_elements), 4, 5
                    )
                    detector_bin_encoder.setBytes_length_atIndex_(
                        direct_parameters, 16, 6
                    )
                    detector_bin_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        Metal.MTLSizeMake(frame_count * groups_per_frame, 1, 1),
                        Metal.MTLSizeMake(128, 1, 1),
                    )
            if decoded_frames != batch_stop - batch_start:
                decode_encoder.endEncoding()
                if profile_stages:
                    detector_bin_encoder.endEncoding()
                raise QH5PreparedSourceError(
                    f"QH5 indexes cover {decoded_frames} of "
                    f"{batch_stop - batch_start} requested frames"
                )
            decode_encoder.endEncoding()
            if profile_stages:
                detector_bin_encoder.endEncoding()
            decode_command.commit()
            decode_commands.append(decode_command)
            commands.append(decode_command)
            if profile_stages:
                detector_bin_command.commit()
                detector_bin_commands.append(detector_bin_command)
                commands.append(detector_bin_command)

        encoded = time.perf_counter()
        if commands:
            commands[-1].waitUntilCompleted()
        completed = time.perf_counter()
        for command in commands:
            if command.error() is not None:
                raise RuntimeError(f"Metal QH5 decode failed: {command.error()}")
        runtime_maximum = int(count_audit_view.max(initial=0))
        if runtime_maximum != prepared.audited_maximum:
            raise QH5PreparedSourceError(
                "Runtime maximum disagrees with the source-bound value-range audit: "
                f"{runtime_maximum} != {prepared.audited_maximum}"
            )
        gpu_seconds = sum(
            max(0.0, float(command.GPUEndTime()) - float(command.GPUStartTime()))
            for command in commands
        )
        decode_gpu_seconds = (
            sum(
                max(0.0, float(command.GPUEndTime()) - float(command.GPUStartTime()))
                for command in decode_commands
            )
            if profile_stages
            else None
        )
        detector_bin_gpu_seconds = (
            sum(
                max(0.0, float(command.GPUEndTime()) - float(command.GPUStartTime()))
                for command in detector_bin_commands
            )
            if profile_stages
            else None
        )
        output = output_view.reshape(total_frames, output_rows, output_columns).view(
            decoder._MtlArray
        )
        output._mtl = output_buffer
        output_buffer = None
        timing = {
            "pipeline_seconds": pipeline_ready - started,
            "allocation_seconds": allocated - pipeline_ready,
            "index_upload_seconds": metadata_ready - allocated,
            "readahead_lead_seconds": read_ahead_ready - metadata_ready,
            "map_seconds": mapped - read_ahead_ready,
            "encode_and_commit_seconds": encoded - submitted_started,
            "queue_wait_seconds": completed - encoded,
            "submit_and_wait_seconds": completed - submitted_started,
            "gpu_seconds": gpu_seconds,
            "decode_gpu_seconds": decode_gpu_seconds,
            "detector_bin_gpu_seconds": detector_bin_gpu_seconds,
            "wall_seconds": completed - started,
        }
        metadata = {
            "load_path": "qh5-audited-low8-frame-major-detector-bin4",
            "source_identity_sha256": prepared.source_identity_sha256,
            "source_audit_schema": "quantem.gpu.value-range-audit/v1",
            "source_maximum": prepared.audited_maximum,
            "pixels_above_255": prepared.audited_pixels_above_255,
            "bad_pixel_indices": prepared.bad_pixel_indices,
            "scan_region": (0, scan_rows, 0, scan_columns),
            "scan_bin": 1,
            "det_bin": 4,
            "source_detector_shape": prepared.detector_shape,
            "detector_shape": (output_rows, output_columns),
            "source_dtype": prepared.source_dtype,
            "dtype": "uint16",
            "batch_scan_rows": batch_scan_rows,
            "scalar_threads": scalar_threads,
            "sequential_readahead": bool(sequential_readahead),
            "profile_stages": bool(profile_stages),
            "detector_bin_kernel": "row8" if use_row8_bin else "row4",
            "readahead_bytes": int(read_ahead.bytes_read) if read_ahead else 0,
            "timing_s": timing,
        }
        return output, metadata
    finally:
        if read_ahead is not None:
            read_ahead.stop()
        for mapped_file in mapped_files:
            mapped_file.close()
        for buffer in metadata_buffers:
            decoder._release_metal_buffer(buffer)
        decoder._release_metal_buffer(scratch_buffer)
        decoder._release_metal_buffer(bad_mask)
        decoder._release_metal_buffer(count_audit)
        decoder._release_metal_buffer(output_buffer)
