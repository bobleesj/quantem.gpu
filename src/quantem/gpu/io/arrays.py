"""Byte-bounded ingestion into the shared integer and float ANS residents.

Arrays already in memory, and HDF5 layouts outside the direct chunk decoder, both
reach the residents through ``load_array_resident``.
"""

import bisect
import hashlib
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_device_index
from quantem.gpu.device.metal_runtime import shared_array
from quantem.gpu.formats.hdf5.frames import detector_sources
from quantem.gpu.formats.qem import metadata as qem_metadata
from quantem.gpu.formats.qem.snapshot import FLOAT_CODEC, INTEGER_CODEC
from quantem.gpu.io.dataset import Dataset4dstemGPU, resident_metadata
from quantem.gpu.resident import float_ans
from quantem.gpu.resident.cuda.counts import Chunk, StreamedCounts, retain
from quantem.gpu.resident.cuda.hot_pixels import CUDAHotPixelCorrector
from quantem.gpu.resident.hot_pixels import hot_pixel_record
from quantem.gpu.resident.mps import precision as metal_precision
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.hot_pixels import MPSHotPixelCorrector
from quantem.gpu.resident.mps.spatial import build_index

MAX_INGEST_BYTES = 32 << 20


def read_frame_block(data, shape: tuple[int, ...], first: int, stop: int) -> np.ndarray:
    """Copy file-backed rows without reshaping a noncontiguous acquisition."""
    if data.ndim == 3:
        return np.ascontiguousarray(data[first:stop])
    block = np.empty((stop - first, *shape[2:]), data.dtype)
    cursor = first
    while cursor < stop:
        row, column = divmod(cursor, shape[1])
        count = min(shape[1] - column, stop - cursor)
        block[cursor - first : cursor - first + count] = data[
            row, column : column + count
        ]
        cursor += count
    return block


def load_array_resident(
    shape: tuple[int, ...],
    dtype: np.dtype | str,
    read_frames: Callable[[int, int], np.ndarray],
    metadata: dict,
    *,
    backend: str,
    device: int | str | None = None,
    pixel_mask: np.ndarray | None = None,
    hot_pixel_correction: str = "median",
    auto_narrow: bool = False,
    verbose: bool = False,
    float_audit_blocks: Callable | None = None,
) -> Dataset4dstemGPU:
    """Encode bounded input windows; never allocate a dense acquisition."""
    started = time.perf_counter()
    shape, dtype = tuple(shape), np.dtype(dtype)
    source_dtype = dtype
    peak_bytes = 0
    exact_float_narrowing = dtype == np.dtype("float64")
    if exact_float_narrowing:
        if not auto_narrow:
            raise NotImplementedError(
                "float64 ANS ingestion requires auto_narrow=True and a complete "
                "bitwise float32 round-trip audit; no rounding is permitted."
            )
        # Include both casts and the comparison mask in the scratch budget.
        audit_frame_bytes = math.prod(shape[2:]) * 24
        if audit_frame_bytes > MAX_INGEST_BYTES:
            raise MemoryError("One audited detector frame exceeds the ingestion limit.")
        audit_scans = min(512, MAX_INGEST_BYTES // audit_frame_bytes)
        blocks = (
            float_audit_blocks()
            if float_audit_blocks is not None
            else (
                read_frames(first, min(first + audit_scans, math.prod(shape[:2])))
                for first in range(0, math.prod(shape[:2]), audit_scans)
            )
        )
        verified_values = 0
        for audit_block in blocks:
            block = np.ascontiguousarray(audit_block)
            if block.dtype != dtype or block.size * 24 > MAX_INGEST_BYTES:
                raise ValueError(
                    "Float audit block exceeds its dtype or scratch contract."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                restored = block.astype(np.float32).astype(np.float64)
            if not np.array_equal(block.view(np.uint64), restored.view(np.uint64)):
                raise NotImplementedError(
                    "These float64 measurements are not exactly representable as "
                    "float32. Keep the original; no rounding or GPU upload occurred."
                )
            peak_bytes = max(peak_bytes, block.size * 24)
            verified_values += block.size
            del audit_block, block, restored
        if verified_values != math.prod(shape):
            raise ValueError(
                "Float audit did not cover every measurement; reopen the acquisition."
            )
        dtype = np.dtype("float32")
        metadata = dict(
            metadata,
            exact_float_narrowing={
                "method": "float64-float32-float64-bitwise",
                "verified_values": math.prod(shape),
            },
        )
    if dtype.kind in "iu" and dtype.itemsize > 2:
        if not auto_narrow:
            raise NotImplementedError(
                f"{dtype} ingestion requires an exact count-range audit. "
                "Use auto_narrow=True; values outside uint16 remain unsupported."
            )
        frame_bytes = math.prod(shape[2:]) * dtype.itemsize
        if frame_bytes > MAX_INGEST_BYTES:
            raise MemoryError("One detector frame exceeds the 32 MiB ingestion limit.")
        audit_scans = min(512, MAX_INGEST_BYTES // frame_bytes)
        minimum, maximum = np.iinfo(dtype).max, np.iinfo(dtype).min
        for first in range(0, math.prod(shape[:2]), audit_scans):
            block = read_frames(first, min(first + audit_scans, math.prod(shape[:2])))
            minimum = min(minimum, int(block.min()))
            maximum = max(maximum, int(block.max()))
            peak_bytes = max(peak_bytes, block.nbytes)
            if minimum < 0 or maximum > 65535:
                raise NotImplementedError(
                    f"{dtype} counts span [{minimum}, {maximum}] and cannot be stored "
                    "exactly as uint16. Keep the original; no clipping was performed."
                )
            del block
        dtype = np.dtype("uint8" if maximum <= 255 else "uint16")
        metadata = dict(metadata, count_range=dict(minimum=minimum, maximum=maximum))
    floating = dtype == np.dtype("float32")
    if dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")):
        raise NotImplementedError(
            f"Exact ANS ingestion supports uint8, uint16 and float32; got {dtype}. "
            "Keep the original file; no precision conversion was performed."
        )
    if floating and pixel_mask is not None and np.any(pixel_mask):
        raise NotImplementedError(
            "Float ANS cannot yet retain detector masks; keep the original file."
        )
    frame_bytes = math.prod(shape[2:]) * source_dtype.itemsize
    if frame_bytes > MAX_INGEST_BYTES:
        raise MemoryError("One detector frame exceeds the 32 MiB ingestion limit.")
    staging_bytes = frame_bytes
    if source_dtype != dtype:
        staging_bytes += math.prod(shape[2:]) * dtype.itemsize
    if exact_float_narrowing:
        staging_bytes = math.prod(shape[2:]) * 24
    if staging_bytes > MAX_INGEST_BYTES:
        raise MemoryError(
            "One detector frame and its exact conversion exceed the 32 MiB ingestion limit."
        )
    block_scans = min(512, MAX_INGEST_BYTES // staging_bytes)
    correction = hot_pixel_record(pixel_mask, hot_pixel_correction, backend=backend)
    valid = np.ones(shape[2:], bool)
    if pixel_mask is not None and not correction["applied"]:
        valid &= np.asarray(pixel_mask) == 0
    context = nullcontext()
    if backend == "cuda":
        import cupy as cp

        selected = cuda_device_index(device)
        context = cp.cuda.Device(selected)
    elif backend != "mps":
        raise ValueError("Encoded ingestion requires backend='cuda' or 'mps'.")
    elif device not in (None, "mps"):
        raise ValueError("Metal uses device='mps'; omit numeric device selection.")

    source = corrector = None
    logical = hashlib.sha256()
    with context:
        try:
            if floating:
                scientific = qem_metadata.acquisition_metadata(shape, metadata)
                description = dict(metadata.get("qem_empad") or {})
                description.setdefault("format_identifier", "float32")
                description.setdefault("format_name", "Float32")
                description.setdefault(
                    "microscope_metadata",
                    {
                        str(key): str(value)
                        for key, value in scientific["source_metadata"].items()
                    },
                )
                header = dict(
                    container="quantem.qem",
                    container_version=1,
                    version=1,
                    profile=FLOAT_CODEC,
                    codec=FLOAT_CODEC,
                    shape=list(shape),
                    dtype=dtype.name,
                    metadata=dict(metadata),
                    scientific_metadata=scientific,
                    empad=description,
                )
                source = float_ans.FloatANSResident(header, backend, device)
                resident = source._lanes
            elif backend == "cuda":
                source = resident = StreamedCounts(shape, dtype, valid)
                corrector = CUDAHotPixelCorrector(pixel_mask, hot_pixel_correction)
            else:
                source = resident = MPSStreamedCounts(shape, dtype, valid)
                corrector = MPSHotPixelCorrector(pixel_mask, hot_pixel_correction)

            for first in range(0, math.prod(shape[:2]), block_scans):
                stop = min(first + block_scans, math.prod(shape[:2]))
                block = read_frames(first, stop)
                if (
                    block.shape != (stop - first, *shape[2:])
                    or block.dtype != source_dtype
                ):
                    raise ValueError(
                        "The source changed shape or dtype while loading; reopen it."
                    )
                block = np.ascontiguousarray(block)
                peak_bytes = max(peak_bytes, block.nbytes)
                if source_dtype != dtype:
                    # Recheck each window in case the input changes after the audit.
                    narrowed = block.astype(dtype)
                    if exact_float_narrowing:
                        restored = narrowed.astype(np.float64)
                        if not np.array_equal(
                            block.view(np.uint64), restored.view(np.uint64)
                        ):
                            raise ValueError(
                                "Float values changed after the exactness audit; reopen the acquisition."
                            )
                        peak_bytes = max(peak_bytes, block.size * 24)
                        del restored
                    elif block.min() < 0 or block.max() > np.iinfo(dtype).max:
                        raise ValueError(
                            "Counts changed after the range audit; reopen the acquisition."
                        )
                    peak_bytes = max(peak_bytes, block.nbytes + narrowed.nbytes)
                    block = narrowed
                    del narrowed
                if floating:
                    logical.update(memoryview(block).cast("B"))
                    # Reinterpret IEEE bits; do not convert float measurements to counts.
                    block = block.view(np.uint16).reshape(
                        len(block), shape[2], shape[3] * 2
                    )
                if backend == "cuda":
                    raw = cp.asarray(block)
                    if corrector is not None:
                        corrector.apply(raw)
                    resident.append(raw)
                    if floating:
                        # The count index means nothing for float bits; retain
                        # copies the codes alone so its bytes are not kept.
                        chunk = resident.chunks[-1]
                        resident.chunks[-1] = Chunk(
                            chunk.first, chunk.scans, retain(chunk.arrays[:3])
                        )
                    del raw
                else:
                    raw = metal_precision.upload(block)
                    try:
                        if corrector is not None:
                            corrector.apply(raw)
                        resident.append(shared_array(raw._mtl, raw.dtype, raw.shape))
                        if not floating:
                            resident.spatial_chunks.append(
                                build_index(resident, raw._mtl, len(block))
                            )
                    finally:
                        raw.release()
                del block
            if floating:
                source.header["logical_sha256"] = logical.hexdigest()
            record = dict(metadata)
            record.update(
                resident_metadata(shape, dtype, source.nbytes, backend=backend),
                device=f"cuda:{selected}" if backend == "cuda" else "mps",
                source_dtype=source_dtype.name,
                source_logical_tensor_bytes=math.prod(shape) * source_dtype.itemsize,
                resident_profile=FLOAT_CODEC if floating else INTEGER_CODEC,
                file_counts_exact=not correction["applied"],
                lossless_exact=not correction["applied"],
                working_counts_exact=True,
                hot_pixel_correction=correction,
                load_timings=dict(
                    resident.load_metrics,
                    peak_ingest_bytes=peak_bytes,
                    resident_ready_seconds=time.perf_counter() - started,
                ),
            )
            if verbose:
                print(
                    f"Loaded {source_dtype.name} measurements as {dtype.name} ANS on {backend}; "
                    "no binning or cropping."
                )
            return Dataset4dstemGPU(source, record)
        except BaseException:
            if source is not None:
                source.release()
            raise
        finally:
            if corrector is not None:
                corrector.close()


def _read_four_dimensional_frames(data, result, first, stop):
    """Read one flattened interval as one union of at most three scan slabs."""
    columns = data.shape[1]
    source_space = data.id.get_space()
    memory_space = h5py.h5s.create_simple(result.shape)
    try:
        source_space.select_none()
        cursor = first
        while cursor < stop:
            row, column = divmod(cursor, columns)
            rows = (stop - cursor) // columns if column == 0 else 0
            count = columns if rows else min(columns - column, stop - cursor)
            rows = max(1, rows)
            source_space.select_hyperslab(
                (row, column, 0, 0),
                (rows, count, *data.shape[2:]),
                op=h5py.h5s.SELECT_OR,
            )
            cursor += rows * count
        data.id.read(memory_space, source_space, result)
    finally:
        source_space.close()
        memory_space.close()


def load_hdf5_array_resident(
    path,
    *,
    dataset_path,
    backend,
    device,
    verbose,
    hot_pixel_correction,
    info,
    auto_narrow=True,
):
    """Use storage-library reads only where the direct chunk reader cannot apply."""
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    selected = dataset_path or info.metadata.get("dataset_path")
    with h5py.File(path, "r") as handle:
        if selected is not None:
            datasets = [handle[selected]]
        else:
            # Opened through the master, so external links resolve as h5py does.
            datasets = [handle[f"entry/data/{source.name}"] for source in detector_sources(handle)]
        if not datasets:
            return None
        direct = all(
            data.ndim == 3
            and data.chunks is not None
            and data.chunks[0] == 1
            # 32008 is the registered HDF5 filter id of bitshuffle+LZ4.
            and 32008
            in {
                data.id.get_create_plist().get_filter(index)[0]
                for index in range(data.id.get_create_plist().get_nfilters())
            }
            for data in datasets
        )
        # Keep the existing accelerated bitshuffle/LZ4 reader for its native layout.
        # It corrects flagged uint32 pixels before verifying that every count fits uint16.
        if (
            dataset_path is None
            and direct
            and np.dtype(info.dtype)
            in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"))
        ):
            return None
        if backend == "cuda" and np.dtype(info.dtype) == np.dtype("uint32"):
            return None  # Existing count-range verification owns exact narrowing.
        starts = [0]
        for data in datasets:
            if (
                data.ndim not in (3, 4)
                or data.shape[-2:] != shape[2:]
                or np.dtype(data.dtype) != np.dtype(info.dtype)
            ):
                raise ValueError(
                    "HDF5 detector datasets disagree in geometry or dtype; select one calibrated acquisition."
                )
            starts.append(starts[-1] + math.prod(data.shape[:-2]))
        if starts[-1] != math.prod(shape[:2]):
            raise ValueError(
                "HDF5 frame count disagrees with scan geometry; restore the complete acquisition."
            )
        signatures = {
            Path(data.file.filename): (
                Path(data.file.filename).stat().st_size,
                Path(data.file.filename).stat().st_mtime_ns,
            )
            for data in datasets
        }
        status = Path(path).stat()
        signatures[Path(path)] = (status.st_size, status.st_mtime_ns)

        def read_frames(first, stop):
            result = np.empty((stop - first, *shape[2:]), np.dtype(info.dtype))
            cursor = first
            while cursor < stop:
                index = bisect.bisect_right(starts, cursor) - 1
                end = min(stop, starts[index + 1])
                data = datasets[index]
                local = cursor - starts[index]
                if data.ndim == 3:
                    data.read_direct(
                        result,
                        source_sel=np.s_[local : local + end - cursor],
                        dest_sel=np.s_[cursor - first : end - first],
                    )
                else:
                    _read_four_dimensional_frames(
                        data,
                        result[cursor - first : end - first],
                        local,
                        local + end - cursor,
                    )
                cursor = end
            return result

        # A full-scan storage chunk must be audited once, not once per scan row.
        # Oversized or contiguous storage keeps the existing bounded frame path.
        chunk_audit = np.dtype(info.dtype) == np.dtype("float64") and all(
            data.chunks is not None
            and math.prod(data.chunks) * 24 <= MAX_INGEST_BYTES
            for data in datasets
        )

        def audit_blocks():
            for data in datasets:
                for selection in data.iter_chunks():
                    yield data[selection]

        loaded = load_array_resident(
            shape,
            info.dtype,
            read_frames,
            info.metadata,
            backend=backend,
            device=device,
            pixel_mask=info.pixel_mask,
            hot_pixel_correction=hot_pixel_correction,
            verbose=verbose,
            auto_narrow=auto_narrow,
            float_audit_blocks=audit_blocks if chunk_audit else None,
        )
        try:
            for source, signature in signatures.items():
                status = source.stat()
                if (status.st_size, status.st_mtime_ns) != signature:
                    raise ValueError(
                        "HDF5 source changed during loading; reopen the acquisition."
                    )
        except BaseException:
            loaded.close()
            raise
        loaded.metadata["load_timings"]["storage_reader"] = "bounded-hdf5"
        return loaded
