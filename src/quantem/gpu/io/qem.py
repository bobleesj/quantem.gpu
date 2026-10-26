"""Save encoded GPU residents as checksummed ``.qem`` copies and reopen them without decoding.

A ``.qem`` copy holds the exact encoded arrays of a resident (integer counts,
scaled uint16 codes or float32 bit lanes) with its metadata, so reopening
uploads the arrays as they are instead of decoding and encoding again.
Every 64 MiB body block is checked against the header before it reaches a
decoder. The envelope format itself lives in ``formats.qem.snapshot``.
"""

import base64
import bisect
import copy
import hashlib
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_device_index, start_after_queued_work
from quantem.gpu.device.metal_runtime import (
    allocate_shared,
    buffer_view,
    read_exact,
    release_buffer,
    upload_shared,
)
from quantem.gpu.device.select import resolve_backend
from quantem.gpu.formats.qem import metadata as qem_metadata
from quantem.gpu.formats.qem.reference import load_array
from quantem.gpu.formats.qem.snapshot import (
    ARRAY_DTYPES,
    BIT_CODEC,
    BLOCK_BYTES,
    FLOAT_CODEC,
    INTEGER_CODEC,
    SCALED_CODEC,
    decode_valid,
    encode_valid,
    read_header_with_digest,
    write_envelope,
)
from quantem.gpu.io.dataset import Dataset4dstemGPU, resident_metadata
from quantem.gpu.io.hdf5.cpu import record_dense_representation
from quantem.gpu.resident.bit_ans import BitANSResident
from quantem.gpu.resident.cuda.counts import Chunk, StreamedCounts
from quantem.gpu.resident.float_ans import FloatANSResident
from quantem.gpu.resident.mps.counts import MPSStreamedCounts, _Chunk
from quantem.gpu.resident.mps.precision import PrecisionSource, _ANSPart


_COPY_STREAMS = r"""
extern "C" __global__ void copy_streams(const unsigned char* source, const unsigned int* source_offsets,
                                        const unsigned int* offsets, const unsigned int* streams,
                                        unsigned char* payload, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int stream = streams[i], first = source_offsets[stream], stop = source_offsets[stream + 1];
    unsigned char* out = payload + offsets[stream];
    for (unsigned int j = first; j < stop; ++j) out[j - first] = source[j];
}
"""


def _copy_streams(grid, block, arguments):
    """Copy kept count streams (one detector pixel over 512 scans each) into a compact payload, a thread each."""
    import cupy as cp

    cp.RawKernel(_COPY_STREAMS, "copy_streams")(grid, block, arguments)


def save_streamed(path, source, metadata: dict | None = None) -> None:
    """Atomically save exact encoded arrays as ``.qem``; never decode or overwrite."""
    metal = isinstance(source, MPSStreamedCounts)
    if not metal and type(source) is not StreamedCounts:
        raise TypeError("Snapshot saving requires native CUDA or Metal streamed ANS counts.")
    if metal and len(source.spatial_chunks) != len(source.chunks):
        raise ValueError("This Metal resident has no spatial indexes; load native DM4 or a saved camera snapshot.")
    if source.is_released or source.ready_scans != math.prod(source.shape[:2]):
        raise ValueError("Save a complete, live CUDA source; reload the acquisition first.")
    path = Path(path)
    if path.suffix.lower() != ".qem":
        raise ValueError(
            f"Saved copies use the .qem extension; got {path.name!r}. "
            "Choose a name ending in .qem."
        )
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new destination.")
    metadata = {} if metadata is None else metadata
    table, spans, cursor = [], [], 0
    for number, chunk in enumerate(source.chunks):
        if metal:
            arrays = [np.frombuffer(buffer_view(buffer), np.dtype(dtype))
                      for buffer, dtype in zip((*chunk.buffers, *source.spatial_chunks[number]), ARRAY_DTYPES)]
        else:
            arrays = chunk.arrays
        entries = []
        for array in arrays:
            # Arrays start on 8-byte boundaries so readers can view them in place.
            cursor += -cursor % 8
            entries.append(dict(offset=cursor, count=int(array.size)))
            spans.append((cursor, array))
            cursor += array.nbytes
        table.append(dict(first=chunk.first, scans=chunk.scans, arrays=entries))
    header = dict(
        version=1, profile=INTEGER_CODEC, interval=512,
        shape=list(source.shape), dtype=source.dtype.name,
        valid=encode_valid(source.valid_pixels), chunks=table, metadata=qem_metadata.retained_metadata(metadata),
        container=qem_metadata.CONTAINER,
        container_version=qem_metadata.CONTAINER_VERSION,
        codec=INTEGER_CODEC,
        scientific_metadata=qem_metadata.acquisition_metadata(source.shape, metadata),
    )
    write_envelope(path, header, cursor, spans if metal else _staged(spans))


def save_bits(path: str | Path, source: BitANSResident, metadata: dict | None = None) -> None:
    """Save encoded measurement bit lanes as ``.qem`` without a decoded cube or the original file."""
    source._check()
    source._lanes.check_readable(0, math.prod(source.shape[:2]))
    path = Path(path)
    if path.suffix.lower() != ".qem" or path.exists():
        raise ValueError("Choose a non-existing .qem destination.")
    source.synchronize()
    header = copy.deepcopy(source.header)
    if metadata is not None:
        header["metadata"] = qem_metadata.retained_metadata(metadata)
        header["scientific_metadata"] = qem_metadata.acquisition_metadata(
            source.shape, metadata
        )
    # Build the authenticated directory from resident bytes, including newly
    # ingested sources. Only encoded blocks visit host storage for file writing.
    header["chunks"], spans, cursor = [], [], 0
    metal = source.backend == "mps"
    for chunk in source._lanes.chunks:
        arrays = [buffer_view(buffer) for buffer in chunk.buffers] if metal else chunk.arrays
        entry = {"first": chunk.first, "scans": chunk.scans}
        for name, array in zip(("payload", "offset", "model"), arrays):
            entry[name + "_offset"] = cursor
            entry[name + "_bytes"] = array.nbytes
            spans.append((cursor, array))
            cursor += array.nbytes
        header["chunks"].append(entry)
    write_envelope(path, header, cursor, spans if metal else _staged(spans))


def load_streamed(path, *, backend, representation, scan_shape, device, verbose):
    """Reopen a saved ``.qem`` copy as the resident it was saved from, without encoding.

    ``backend="cpu"`` decodes the CPU reference array instead.
    """
    if backend == "cpu":
        if representation not in (None, "dense"):
            raise ValueError("CPU QEM reference loading returns dense data; use representation='dense'.")
        data, metadata = load_array(path)
        if scan_shape is not None and tuple(scan_shape) != data.shape[:2]:
            raise ValueError("scan_shape disagrees with the saved QEM file.")
        return record_dense_representation(Dataset4dstemGPU(data, metadata))
    selected_backend = resolve_backend(backend)
    if selected_backend not in ("cuda", "mps") or representation not in (None, "encoded"):
        raise NotImplementedError("ANS snapshots reopen as encoded counts; choose backend='cuda' or 'mps'.")
    started = time.perf_counter()
    # The authenticated header binds the payload checksum table as well.
    header, start, header_digest = read_header_with_digest(path)
    shape = tuple(header["shape"])
    if scan_shape is not None and tuple(scan_shape) != shape[:2]:
        raise ValueError("scan_shape disagrees with the saved acquisition; omit scan_shape.")
    if header["codec"] in (FLOAT_CODEC, BIT_CODEC):
        return _load_bits(path, header, start, selected_backend, device)
    if header["codec"] == SCALED_CODEC:
        if selected_backend != "mps":
            raise NotImplementedError(
                "Scaled uint16 .qem results reopen on Apple GPUs (backend='mps'), or "
                "as a dense CPU reference with backend='cpu'."
            )
        return _load_scaled_mps(path, header, start, verbose)
    if selected_backend == "mps":
        dataset = _load_counts_mps(path, header, start, verbose)
    else:
        dataset = _load_counts_cuda(path, header, start, device, verbose, started)
    try:
        with open(path, "rb") as handle:
            if handle.read(56)[24:].hex() != header_digest:
                raise ValueError("QEM changed while loading; reopen the completed file.")
    except Exception:
        dataset.close()
        raise
    dataset.metadata["qem_header_sha256"] = header_digest
    return dataset


def mps_counts_dataset(resident: MPSStreamedCounts, metadata: dict, started: float, verbose: bool) -> Dataset4dstemGPU:
    """Wrap an exact native-count Metal resident with the common loaded-data metadata.

    Saved ``.qem`` copies and native DigitalMicrograph cameras both end here,
    so the two report identical fields. ``source_dtype``, ``file_counts_exact``
    and ``working_counts_exact`` are the caller's: a ``.qem`` header states them
    for the original acquisition, and a camera reader for the file it read.
    """
    shape = resident.shape
    record = dict(metadata)
    record.update(
        resident_metadata(shape, resident.dtype, resident.nbytes, backend="mps"),
        device="mps",
        source_logical_tensor_bytes=math.prod(shape) * resident.dtype.itemsize,
        lossless_exact=True,
        resident_profile=INTEGER_CODEC,
        load_timings=dict(
            resident.load_metrics, resident_ready_seconds=time.perf_counter() - started
        ),
    )
    if verbose:
        print(
            f"Loaded exact native camera ANS on Metal in {time.perf_counter() - started:.3f} s."
        )
    return Dataset4dstemGPU(resident, record)


def _staged(spans):
    """Yield ``(offset, bytes)`` of CUDA arrays for ``write_envelope``, at most 64 MiB a piece.

    ``array.get()`` allocates new pageable host memory for every array: the
    whole body in fresh pages on each save, which stalls for seconds on a host
    short of free memory. The pieces alternate between two pinned buffers
    instead, the two the writer has in use at once, and copy at the pinned
    transfer rate. Both are 64 MiB whatever the arrays, the size the loader
    stages through, so every save and load reuses the same two blocks of
    CuPy's pinned pool, which keeps one free list per size and shrinks only
    when ``device.release_cached_memory`` empties it.
    """
    import cupy as cp

    buffers = [np.frombuffer(cp.cuda.alloc_pinned_memory(BLOCK_BYTES), np.uint8, count=BLOCK_BYTES) for _ in range(2)]
    number = 0
    for offset, array in spans:
        encoded = array.ravel().view(cp.uint8)
        for first in range(0, encoded.size, BLOCK_BYTES):
            piece = buffers[number % 2][: min(BLOCK_BYTES, encoded.size - first)]
            encoded[first : first + piece.size].get(out=piece)
            yield offset + first, piece
            number += 1


def _cached_block_order(path: str | Path, start: int, size: int) -> list[int]:
    """Read cached blocks before an uncached prefix can evict them on Linux.

    This is a best-effort page-cache snapshot, not a read or a reservation of
    the file contents. Checksums and destination offsets are independent of
    the resulting order. Other platforms retain sequential scheduling.
    """
    import ctypes
    import sys

    order = list(range(math.ceil(size / BLOCK_BYTES)))
    if sys.platform != "linux" or len(order) < 2:
        return order
    try:
        library = ctypes.CDLL(None, use_errno=True)
        library.mmap.restype = ctypes.c_void_p
        library.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_long,
        ]
        library.mincore.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ]
        library.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        length = start + size
        vector = (ctypes.c_ubyte * math.ceil(length / page_bytes))()
        with open(path, "rb") as handle:
            # PROT_NONE, MAP_SHARED: inspect residency without touching pages.
            address = library.mmap(None, length, 0, 1, handle.fileno(), 0)
            if address == ctypes.c_void_p(-1).value:
                return order
            try:
                if library.mincore(address, length, vector):
                    return order
                cached = np.ctypeslib.as_array(vector)
                fractions = []
                for number in order:
                    first = (start + number * BLOCK_BYTES) // page_bytes
                    stop = math.ceil(
                        (start + min(size, (number + 1) * BLOCK_BYTES)) / page_bytes
                    )
                    fractions.append(
                        np.count_nonzero(cached[first:stop] & 1) / (stop - first)
                    )
                return sorted(order, key=lambda number: (-fractions[number], number))
            finally:
                library.munmap(address, length)
    except (AttributeError, OSError, OverflowError):
        # Restricted or unsupported residency inspection must not block IO.
        return order


def _load_counts_cuda(path, header, start, device, verbose, started):
    """Read and authenticate bounded blocks in parallel before uploading them."""
    import cupy as cp

    shape = tuple(header["shape"])
    selected = cuda_device_index(device)
    valid = decode_valid(header["valid"], shape[2:])
    with cp.cuda.Device(selected):
        source = StreamedCounts(shape, np.dtype(header["dtype"]), valid)
        stream = cp.cuda.Stream(non_blocking=True)
        # The arena may be pool blocks the caller has just freed with reads of them still queued.
        start_after_queued_work(stream)
        arena = cp.empty(header["bytes"], cp.uint8)
        capacity = min(BLOCK_BYTES, header["bytes"])
        workers = min(16, len(header["sha256"]))
        owners = [cp.cuda.alloc_pinned_memory(capacity) for _ in range(workers)]
        buffers = [np.frombuffer(owner, np.uint8, count=capacity) for owner in owners]
        order = _cached_block_order(path, start, header["bytes"])
        try:
            with ExitStack() as stack:
                handles = [
                    stack.enter_context(open(path, "rb", buffering=0))
                    for _ in range(workers)
                ]
                reader = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
                def read(number, slot, uploaded=None):
                    # A slot cannot be overwritten until its previous upload
                    # finishes. Waiting in the worker leaves other reads active.
                    if uploaded is not None:
                        with cp.cuda.Device(selected):
                            uploaded.synchronize()
                    count = min(BLOCK_BYTES, header["bytes"] - number * BLOCK_BYTES)
                    view = memoryview(buffers[slot][:count])
                    # Independent file positions let reads and SHA-256 checks
                    # overlap without sharing a seek cursor between workers.
                    handle = handles[slot]
                    handle.seek(start + number * BLOCK_BYTES)
                    cursor = 0
                    while cursor < count:
                        length = handle.readinto(view[cursor:])
                        if not length:
                            raise ValueError("ANS snapshot ended during loading; recopy it.")
                        cursor += length
                    if hashlib.sha256(view).hexdigest() != header["sha256"][number]:
                        raise ValueError(f"ANS checksum mismatch in block {number}; recopy the file.")
                    return count

                block_iterator = iter(order)
                pending = {}
                for slot in range(workers):
                    number = next(block_iterator)
                    pending[reader.submit(read, number, slot)] = (number, slot)
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        number, slot = pending.pop(future)
                        count = future.result()
                        with stream:
                            first = number * BLOCK_BYTES
                            arena[first:first + count].set(
                                buffers[slot][:count], stream=stream
                            )
                            uploaded = cp.cuda.Event()
                            uploaded.record(stream)
                        following = next(block_iterator, None)
                        if following is not None:
                            pending[reader.submit(read, following, slot, uploaded)] = (
                                following, slot
                            )
                stream.synchronize()
            for chunk in header["chunks"]:
                arrays = []
                for spec, dtype in zip(chunk["arrays"], ARRAY_DTYPES):
                    end = spec["offset"] + spec["count"] * np.dtype(dtype).itemsize
                    arrays.append(arena[spec["offset"]:end].view(dtype))
                source.chunks.append(Chunk(chunk["first"], chunk["scans"], tuple(arrays)))
            source.ready_scans = math.prod(shape[:2])
            metadata = _counts_metadata(path, header, source, selected, started, dict(
                encode_seconds=0.0, index_seconds=0.0, pinned_staging_bytes=workers * capacity,
                verified_read_workers=workers, verified_encoded_bytes=header["bytes"]))
            if verbose:
                print(f"Restored exact CUDA ANS {shape} in {time.perf_counter() - started:.3f} s.")
            return Dataset4dstemGPU(source, metadata)
        except BaseException:
            stream.synchronize()
            source.release()
            raise


def _counts_metadata(path, header, source, selected, started, timings) -> dict:
    """The metadata of a CUDA counts resident restored from ``header``."""
    shape = tuple(header["shape"])
    metadata = qem_metadata.effective_metadata(dict(header["metadata"]), header["scientific_metadata"])
    metadata["scientific_metadata"] = header["scientific_metadata"]
    metadata["container"] = header["container"]
    metadata["container_version"] = header["container_version"]
    metadata.setdefault("original_source_path", metadata.get("source_path"))
    # source_dtype, file_counts_exact and working_counts_exact stay as the header states them: they are the
    # evidence of how the stored counts relate to the original acquisition (format 0.0.4, retained block).
    # Replaced here by what this load knows (the stored dtype, exact), a copy of a file rewrote a uint32
    # source as uint16 and counts changed by a pixel correction as exact.
    metadata.update(
        resident_metadata(shape, source.dtype, source.nbytes, backend="cuda"),
        source_kind="resident", source_path=str(Path(path).resolve()),
        source_logical_tensor_bytes=math.prod(shape) * source.dtype.itemsize,
        device=f"cuda:{selected}", resident_profile=INTEGER_CODEC,
        index_bytes=source.index_nbytes,
        lossless_exact=True,
        load_timings=dict(resident_ready_seconds=time.perf_counter() - started, **timings),
    )
    return metadata


def read_metadata(path) -> tuple[dict, dict]:
    """The header and the effective metadata of a ``.qem`` file, without reading its body."""
    header, _ = read_header_with_digest(path)[:2]
    return header, qem_metadata.effective_metadata(dict(header["metadata"]), header["scientific_metadata"])


def _blocks_in_order(path, header, start, workers: int = 8):
    """Yield ``(offset, pinned bytes)`` of every verified 64 MiB body block in file order.

    Workers read and hash the next blocks while the caller uploads one; a block's buffer is reused only after the
    caller asks for the following block, so the caller must finish with it (a synchronous upload) first.
    """
    import cupy as cp
    from collections import deque

    total, blocks = header["bytes"], len(header["sha256"])
    workers = max(1, min(workers, blocks))
    owners = [cp.cuda.alloc_pinned_memory(min(BLOCK_BYTES, total)) for _ in range(workers)]
    buffers = [np.frombuffer(owner, np.uint8, count=min(BLOCK_BYTES, total)) for owner in owners]
    with ExitStack() as stack:
        handles = [stack.enter_context(open(path, "rb", buffering=0)) for _ in range(workers)]
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))

        def read(number, slot):
            count = min(BLOCK_BYTES, total - number * BLOCK_BYTES)
            view = memoryview(buffers[slot][:count])
            handles[slot].seek(start + number * BLOCK_BYTES)
            cursor = 0
            while cursor < count:
                length = handles[slot].readinto(view[cursor:])
                if not length:
                    raise ValueError("ANS snapshot ended during loading; recopy it.")
                cursor += length
            if hashlib.sha256(view).hexdigest() != header["sha256"][number]:
                raise ValueError(f"ANS checksum mismatch in block {number}; recopy the file.")
            return count

        pending = deque((number, pool.submit(read, number, number)) for number in range(workers))
        following = workers
        while pending:
            number, future = pending.popleft()
            count = future.result()
            yield number * BLOCK_BYTES, buffers[number % workers][:count]
            if following < blocks:
                pending.append((following, pool.submit(read, following, following % workers)))
                following += 1


def _chunks_in_order(path, header, start):
    """Yield ``(chunk entry, device arrays)`` of every chunk, staged through a bounded device window.

    The window holds one chunk and the blocks around it (about 64 MiB more than the largest chunk, twice), so a
    file larger than the GPU can be read chunk by chunk; the arrays are views into it, valid until the next chunk.
    """
    import cupy as cp

    spans = []
    for chunk in header["chunks"]:
        first = chunk["arrays"][0]["offset"]
        last = chunk["arrays"][-1]
        dtype = ARRAY_DTYPES[len(chunk["arrays"]) - 1]
        spans.append((first, last["offset"] + last["count"] * np.dtype(dtype).itemsize))
    capacity = max(stop - first for first, stop in spans) + 2 * BLOCK_BYTES
    windows = [cp.empty(capacity, cp.uint8), cp.empty(capacity, cp.uint8)]
    current, origin, filled = 0, 0, 0
    blocks = _blocks_in_order(path, header, start)
    for chunk, (first, stop) in zip(header["chunks"], spans):
        aligned = (first // BLOCK_BYTES) * BLOCK_BYTES
        if aligned > origin:
            kept = origin + filled - aligned
            other = 1 - current
            if kept > 0:
                windows[other][:kept] = windows[current][aligned - origin:aligned - origin + kept]
            current, origin, filled = other, aligned, max(kept, 0)
        while origin + filled < stop:
            offset, block = next(blocks)
            if offset != origin + filled:
                raise ValueError("QEM body blocks arrived out of order; reopen the file.")
            windows[current][filled:filled + block.size].set(block)
            cp.cuda.get_current_stream().synchronize()
            filled += block.size
        arrays = []
        for spec, dtype in zip(chunk["arrays"], ARRAY_DTYPES):
            begin = spec["offset"] - origin
            arrays.append(windows[current][begin:begin + spec["count"] * np.dtype(dtype).itemsize].view(dtype))
        yield chunk, arrays
    for _ in blocks:  # every block is checked, the padding after the last chunk included
        pass


def load_counts_pixels(path, select, *, device=None, verbose: bool = False) -> Dataset4dstemGPU:
    """Restore a CUDA counts ``.qem`` keeping only the detector pixels ``select`` names, chunk by chunk.

    A first pass sums every pixel over the scan (exact uint64); ``select(total, valid)`` returns the boolean
    detector mask to keep. A second pass copies the kept pixels' count streams into compact arrays; every other
    pixel becomes an empty constant-zero stream, so it reads 0, and the spatial index is not loaded. The exact
    full-detector total stays with the resident (``detector.mean`` is unchanged). Memory holds the kept streams
    and one staged chunk, never the whole file, so an acquisition larger than the GPU loads when its kept pixels
    fit. The resident records ``kept_pixels``; a reader that needs another pixel must refuse. Callers announce the
    selection; this is for an algorithm that owns the resident and reads only those pixels.
    """
    import cupy as cp

    started = time.perf_counter()
    header, start, header_digest = read_header_with_digest(path)
    if header["codec"] != INTEGER_CODEC:
        raise NotImplementedError("Selected-pixel loading needs an integer counts .qem.")
    shape = tuple(header["shape"])
    selected = cuda_device_index(device)
    valid = decode_valid(header["valid"], shape[2:])
    pixels = math.prod(shape[2:])
    with cp.cuda.Device(selected):
        source = StreamedCounts(shape, np.dtype(header["dtype"]), valid)
        try:
            errors = cp.zeros(1, cp.uint32)
            with cp.cuda.using_allocator():  # retained, so outside the memory pool (see counts.retain)
                total = cp.zeros(shape[2:], cp.uint64)
            for chunk, arrays in _chunks_in_order(path, header, start):
                streams = math.ceil(chunk["scans"] / source.interval) * pixels
                source.kernels["total"](
                    ((streams + 127) // 128,), (128,),
                    (*arrays[:3], source.decoding, total, errors, np.uint32(chunk["scans"]), np.uint32(pixels),
                     np.uint32(source.interval)),
                )
            if int(errors.get()[0]):
                raise ValueError("An encoded count stream failed reconstruction.")
            summed = time.perf_counter() - started
            keep = np.asarray(select(total, valid), bool)
            if keep.shape != shape[2:]:
                raise ValueError("select must return a detector-shaped mask.")
            kept = cp.asarray(keep.ravel())
            for chunk, arrays in _chunks_in_order(path, header, start):
                payload, offsets, models = arrays[:3]
                mask = kept[cp.arange(models.size, dtype=cp.int64) % pixels]
                sizes = cp.diff(offsets).astype(cp.uint32) * mask
                streams = cp.flatnonzero(mask).astype(cp.uint32)
                with cp.cuda.using_allocator():
                    compact_offsets = cp.empty(models.size + 1, cp.uint32)
                    compact_offsets[0] = 0
                    cp.cumsum(sizes, dtype=cp.uint32, out=compact_offsets[1:])
                    compact = cp.empty(max(1, int(compact_offsets[-1].get())), cp.uint8)
                    # model 253: an empty stream of the constant 0
                    compact_models = cp.where(mask, models, np.uint8(253)).astype(cp.uint8)
                _copy_streams((max(1, math.ceil(streams.size / 128)),), (128,),
                              (payload, offsets, compact_offsets, streams, compact, np.uint32(streams.size)))
                cp.cuda.get_current_stream().synchronize()
                source.chunks.append(Chunk(chunk["first"], chunk["scans"], (compact, compact_offsets, compact_models)))
            source.ready_scans = math.prod(shape[:2])
            source.spatial_index = False
            source._detector_total = total
            source.kept_pixels = keep
            cp.get_default_memory_pool().free_all_blocks()
            metadata = _counts_metadata(path, header, source, selected, started, dict(
                detector_sum_seconds=summed, verified_encoded_bytes=header["bytes"]))
            metadata["kept_detector_pixels"] = int(keep.sum())
            with open(path, "rb") as handle:
                if handle.read(56)[24:].hex() != header_digest:
                    raise ValueError("QEM changed while loading; reopen the completed file.")
            metadata["qem_header_sha256"] = header_digest
            if verbose:
                print(f"Restored {int(keep.sum())} of {pixels} detector pixels of CUDA ANS {shape} "
                      f"in {time.perf_counter() - started:.3f} s.")
            return Dataset4dstemGPU(source, metadata)
        except BaseException:
            source.release()
            raise


def _load_counts_mps(path, header, start, verbose):
    """Restore on Metal the same exact integer-codec streams written on CUDA, without re-encoding."""
    started = time.perf_counter()
    shape = tuple(header["shape"])
    valid = decode_valid(header["valid"], shape[2:])
    resident = MPSStreamedCounts(shape, np.dtype(header["dtype"]), valid)
    try:
        spans = [
            (spec["offset"], spec["count"] * np.dtype(dtype).itemsize)
            for chunk in header["chunks"]
            for spec, dtype in zip(chunk["arrays"], ARRAY_DTYPES)
        ]
        buffers = _read_verified(path, header, start, spans)
        # Keep the exact spatial index for indexed detector kernels.
        for number, chunk in enumerate(header["chunks"]):
            arrays = buffers[6 * number : 6 * number + 6]
            resident.chunks.append(_Chunk(chunk["first"], chunk["scans"], tuple(arrays[:3])))
            resident.spatial_chunks.append(tuple(arrays[3:]))
        resident.ready_scans = math.prod(shape[:2])
        metadata = dict(
            header["metadata"],
            source_kind="resident",
            original_source_path=header["metadata"].get("source_path"),
            source_path=str(path),
        )
        metadata = qem_metadata.effective_metadata(metadata, header["scientific_metadata"])
        metadata["scientific_metadata"] = header["scientific_metadata"]
        metadata["container"] = header["container"]
        metadata["container_version"] = header["container_version"]
        return mps_counts_dataset(resident, metadata, started, verbose)
    except BaseException:
        resident.release()
        raise


def _load_scaled_mps(path, header, start, verbose):
    """Reopen saved scaled uint16 codes and their regional calibration unchanged.

    Each chunk's streams stay ANS encoded in their own resident; values are
    restored as ``float32(code * scale + offset)`` only inside queries.
    """
    started = time.perf_counter()
    shape = tuple(header["shape"])
    pixels = math.prod(shape[2:])
    spans = [
        (spec["offset"], spec["count"] * itemsize)
        for chunk in header["chunks"]
        for spec, itemsize in zip(chunk["arrays"], (1, 4, 1))
    ]
    first = MPSStreamedCounts((1, header["chunks"][0]["scans"], *shape[2:]), np.uint16)
    parts = [_ANSPart(first, first.shape)]
    buffers = []
    try:
        buffers = _read_verified(path, header, start, spans)
        for number, chunk in enumerate(header["chunks"]):
            scans = chunk["scans"]
            streams = math.ceil(scans / 512) * pixels
            payload, offsets, models = buffers[3 * number : 3 * number + 3]
            stops = np.frombuffer(buffer_view(offsets, (streams + 1) * 4), np.uint32)
            kinds = np.frombuffer(buffer_view(models, streams), np.uint8)
            if (stops[0] != 0 or stops[-1] != chunk["arrays"][0]["count"]
                    or np.any(stops[1:] < stops[:-1]) or np.any((kinds > 63) & (kinds < 252))):
                raise ValueError(
                    f"Scaled .qem chunk {number} has invalid stream offsets or models; "
                    "save the result again."
                )
            owner = parts[0].owner if number == 0 else MPSStreamedCounts(
                (1, scans, *shape[2:]), np.uint16
            )
            if number:
                parts.append(_ANSPart(owner, owner.shape))
            owner.chunks.append(_Chunk(0, scans, (payload, offsets, models)))
            owner.ready_scans = scans
            buffers[3 * number : 3 * number + 3] = [None] * 3
        report = header["intensity_calibration"]
        source = PrecisionSource(parts, shape, report)
    except BaseException:
        for part in parts:
            part.release()
        for buffer in buffers:
            release_buffer(buffer)
        raise
    metadata = qem_metadata.effective_metadata({}, header["scientific_metadata"])
    metadata.update(
        resident_metadata(shape, np.float32, source.nbytes, backend="mps"),
        scientific_metadata=header["scientific_metadata"],
        container=header["container"],
        container_version=header["container_version"],
        # a merged result's records parsed, under the keys the CPU and HDF5 readers give them
        **qem_metadata.maped_records(header.get("attributes", {})),
        precision=report,
        conversion_report_origin="saved",
        source_kind="resident",
        source_path=str(Path(path).resolve()),
        device="mps",
        resident_codec="ans",
        resident_profile=SCALED_CODEC,
        source_dtype=report["source_dtype"],
        storage_dtype="uint16",
        lossless_exact=report["changed"] == 0,
        file_counts_exact=False,
        working_counts_exact=False,
        load_timings=dict(
            resident_ready_seconds=time.perf_counter() - started,
            verified_encoded_bytes=header["bytes"],
        ),
    )
    if verbose:
        print(
            f"Loaded scaled uint16 ANS on Metal in {time.perf_counter() - started:.3f} s "
            f"(RMSE {report['rmse']:.3g}, max error {report['max_abs_error']:.3g})."
        )
    return Dataset4dstemGPU(source, metadata)


def _read_verified(path, header, start, spans):
    """Read body arrays straight into shared Metal buffers, checking each 64 MiB block.

    ``spans`` lists ``(offset, bytes)`` in file order. Workers read and hash whole
    blocks in parallel, alignment padding included, so every byte is read once and
    nothing reaches a decoder unless all checksums match.
    """
    buffers = []
    try:
        for _, size in spans:
            buffers.append(allocate_shared(max(1, size), "saved ANS"))
        views = [buffer_view(buffer, size) for buffer, (_, size) in zip(buffers, spans)]
        starts = [offset for offset, _ in spans]
        total = header["bytes"]

        def read(number):
            lower, upper = number * BLOCK_BYTES, min(total, (number + 1) * BLOCK_BYTES)
            digest = hashlib.sha256()
            index, cursor = max(0, bisect.bisect_right(starts, lower) - 1), lower
            while cursor < upper:
                while index < len(spans) and spans[index][0] + spans[index][1] <= cursor:
                    index += 1
                if index < len(spans) and spans[index][0] <= cursor:
                    offset, size = spans[index]
                    stop = min(upper, offset + size)
                    target = views[index][cursor - offset : stop - offset]
                else:
                    stop = min(upper, spans[index][0] if index < len(spans) else upper)
                    target = memoryview(bytearray(stop - cursor))
                read_exact(descriptor, target, start + cursor, ".qem body")
                digest.update(target)
                cursor = stop
            if digest.hexdigest() != header["sha256"][number]:
                raise ValueError(f"ANS checksum mismatch in block {number}; recopy the file.")

        descriptor = os.open(path, os.O_RDONLY)
        try:
            blocks = len(header["sha256"])
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1, blocks)) as workers:
                list(workers.map(read, range(blocks)))
        finally:
            os.close(descriptor)
        return buffers
    except BaseException:
        for buffer in buffers:
            release_buffer(buffer)
        raise


def _load_bits(path, header, start, backend, device):
    """Authenticate and upload encoded measurement bit lanes; never expand a complete source."""
    started = time.perf_counter()
    source = (FloatANSResident if header["codec"] == FLOAT_CODEC else BitANSResident)(header, backend, device)
    digest, block_bytes, block_index = hashlib.sha256(), 0, 0
    arrays = []
    try:
        with open(path, "rb", buffering=0) as handle:
            handle.seek(start)
            for chunk in header["chunks"]:
                arrays = []
                for name, dtype in (
                    ("payload", "u1"),
                    ("offset", "<u4"),
                    ("model", "u1"),
                ):
                    raw = handle.read(chunk[name + "_bytes"])
                    if len(raw) != chunk[name + "_bytes"]:
                        raise ValueError("QEM ended during loading; recopy the file.")
                    view = memoryview(raw)
                    while view:
                        part = min(len(view), BLOCK_BYTES - block_bytes)
                        digest.update(view[:part])
                        block_bytes += part
                        view = view[part:]
                        if block_bytes == BLOCK_BYTES:
                            if digest.hexdigest() != header["sha256"][block_index]:
                                raise ValueError(
                                    "QEM body checksum mismatch; recopy the file."
                                )
                            digest, block_bytes, block_index = (
                                hashlib.sha256(),
                                0,
                                block_index + 1,
                            )
                    values = np.frombuffer(raw, dtype)
                    if backend == "cuda":
                        arrays.append(source._array(values))
                    else:
                        arrays.append(upload_shared(values, "Float ANS encoded bytes"))
                if backend == "cuda":
                    source._lanes.chunks.append(
                        Chunk(chunk["first"], chunk["scans"], tuple(arrays))
                    )
                else:
                    source._lanes.chunks.append(
                        _Chunk(chunk["first"], chunk["scans"], tuple(arrays))
                    )
                arrays = []  # The resident now owns these buffers.
            if block_bytes and digest.hexdigest() != header["sha256"][block_index]:
                raise ValueError("QEM body checksum mismatch; recopy the file.")
        source._lanes.ready_scans = math.prod(source.shape[:2])
        background = header.get("empad", {}).get("background")
        if background is not None:
            raw = base64.b64decode(background["values_float32_le"], validate=True)
            if len(raw) != source.frame_bytes:
                raise ValueError(
                    f"QEM mean-dark plane must contain {source.detector_shape} float32 values."
                )
            source._background = source._array(
                np.frombuffer(raw, "<f4").copy().reshape(source.detector_shape)
            )
        source.synchronize()
        metadata = qem_metadata.effective_metadata(
            header.get("metadata", {}), header["scientific_metadata"]
        )
        metadata.update(
            resident_metadata(source.shape, source.dtype, source.nbytes, backend=backend),
            scientific_metadata=header["scientific_metadata"],
            qem_empad=header.get("empad", {}),
            valid_pixels=source.valid_pixels,
            container=header["container"],
            container_version=header["container_version"],
            source_path=str(Path(path).resolve()),
            source_kind="resident",
            device=f"cuda:{source.device}" if backend == "cuda" else "mps",
            shape=source.shape,
            resident_profile=header["codec"],
            source_logical_tensor_bytes=source.logical_nbytes,
            lossless_exact=True,
            background_applied=False,
            background_applied_by_reader=background is not None,
            load_timings={
                "resident_ready_seconds": time.perf_counter() - started,
                "encode_seconds": 0.0,
                "verified_encoded_bytes": header["bytes"],
            },
        )
        return Dataset4dstemGPU(source, metadata)
    except BaseException:
        if backend == "mps":
            for buffer in arrays:
                release_buffer(buffer)
        source.release()
        raise
