"""Byte-bounded ingestion into the shared integer and float ANS residents."""

from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Callable
import hashlib
import math
import time

import numpy as np

from .models import FourDSTEMData

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
) -> FourDSTEMData:
    """Encode bounded input windows; never allocate a dense acquisition."""
    from ._hot_pixels import hot_pixel_record
    from ._float_ans import FloatANSResident, PROFILE
    from . import _qem_metadata

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
        for block in blocks:
            block = np.ascontiguousarray(block)
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
            del block, restored
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
            minimum, maximum = min(minimum, int(block.min())), max(
                maximum, int(block.max())
            )
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

        selected = (
            cp.cuda.Device().id
            if device is None
            else int(str(device).removeprefix("cuda:"))
        )
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
                scientific = _qem_metadata.acquisition_metadata(shape, metadata)
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
                    profile=PROFILE,
                    codec=PROFILE,
                    shape=list(shape),
                    dtype=dtype.name,
                    metadata=dict(metadata),
                    scientific_metadata=scientific,
                    empad=description,
                )
                source = FloatANSResident(header, backend, device)
                resident = source._lanes
            elif backend == "cuda":
                from quantem.gpu._compact.streamed import StreamedCounts
                from .backends.cuda.hot_pixels import CUDAHotPixelCorrector

                source = resident = StreamedCounts(shape, dtype, valid)
                corrector = CUDAHotPixelCorrector(pixel_mask, hot_pixel_correction)
            else:
                from .backends.mps._streamed import MPSStreamedCounts
                from .backends.mps.hot_pixels import MPSHotPixelCorrector

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
                        from quantem.gpu._compact.streamed import Chunk

                        chunk = resident.chunks[-1]
                        resident.chunks[-1] = Chunk(
                            chunk.first, chunk.scans, chunk.arrays[:3]
                        )
                    del raw
                else:
                    from .backends.mps.precision import upload
                    from .backends.mps._spatial import build_index

                    raw = upload(block)
                    try:
                        if corrector is not None:
                            corrector.apply(raw)
                        resident.append(raw)
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
                backend=backend,
                device=f"cuda:{selected}" if backend == "cuda" else "mps",
                representation="encoded",
                residency="device",
                source_shape=shape,
                working_shape=shape,
                scan_shape=shape[:2],
                detector_shape=shape[2:],
                dtype=dtype.name,
                source_dtype=source_dtype.name,
                working_dtype=dtype.name,
                n_frames=math.prod(shape[:2]),
                physical_resident_bytes=source.nbytes,
                source_logical_tensor_bytes=math.prod(shape) * source_dtype.itemsize,
                working_logical_tensor_bytes=math.prod(shape) * dtype.itemsize,
                resident_profile=(
                    PROFILE if floating else "runtime-column-rans-spatial-v2"
                ),
                file_counts_exact=not correction["applied"],
                lossless_exact=not correction["applied"],
                working_counts_exact=True,
                hot_pixel_correction=correction,
                scan_bin=1,
                detector_bin=1,
                crop=None,
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
            return FourDSTEMData(source, record)
        except BaseException:
            if source is not None:
                source.release()
            raise
        finally:
            if corrector is not None:
                corrector.close()
