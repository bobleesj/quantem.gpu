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
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_device_index
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
    BLOCK_BYTES,
    FLOAT_CODEC,
    INTEGER_CODEC,
    SCALED_CODEC,
    decode_valid,
    encode_valid,
    read_header,
    write_envelope,
)
from quantem.gpu.io.dataset import Dataset4dstemGPU, resident_metadata
from quantem.gpu.io.hdf5.cpu import record_dense_representation
from quantem.gpu.resident.cuda.counts import Chunk, StreamedCounts
from quantem.gpu.resident.float_ans import FloatANSResident
from quantem.gpu.resident.mps.counts import MPSStreamedCounts, _Chunk
from quantem.gpu.resident.mps.precision import PrecisionSource, _ANSPart


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
    table = []
    with tempfile.TemporaryFile(dir=path.parent) as body:
        for number, chunk in enumerate(source.chunks):
            if metal:
                arrays = [np.frombuffer(buffer_view(buffer), np.dtype(dtype))
                          for buffer, dtype in zip((*chunk.buffers, *source.spatial_chunks[number]), ARRAY_DTYPES)]
            else:
                arrays = [array.get() for array in chunk.arrays]
            entries = []
            for array in arrays:
                # Arrays start on 8-byte boundaries so readers can view them in place.
                body.write(b"\0" * ((-body.tell()) % 8))
                entries.append(dict(offset=body.tell(), count=int(array.size)))
                body.write(memoryview(array).cast("B"))
            table.append(dict(first=chunk.first, scans=chunk.scans, arrays=entries))
        header = dict(
            version=1, profile=INTEGER_CODEC, interval=512,
            shape=list(source.shape), dtype=source.dtype.name,
            valid=encode_valid(source.valid_pixels), chunks=table, metadata=metadata,
            container=qem_metadata.CONTAINER,
            container_version=qem_metadata.CONTAINER_VERSION,
            codec=INTEGER_CODEC,
            scientific_metadata=qem_metadata.acquisition_metadata(source.shape, metadata),
        )
        write_envelope(path, header, body)


def save_float(path: str | Path, source: FloatANSResident, metadata: dict | None = None) -> None:
    """Save encoded float32 bit lanes as ``.qem`` without a decoded cube or the original file."""
    source._check()
    path = Path(path)
    if path.suffix.lower() != ".qem" or path.exists():
        raise ValueError("Choose a non-existing .qem destination.")
    source.synchronize()
    header = copy.deepcopy(source.header)
    if metadata is not None:
        header["metadata"] = dict(metadata)
        header["scientific_metadata"] = qem_metadata.acquisition_metadata(
            source.shape, metadata
        )
    # Build the authenticated directory from resident bytes, including newly
    # ingested sources. Only encoded blocks visit host storage for file writing.
    with tempfile.TemporaryFile(dir=path.parent) as body:
        header["chunks"] = []
        for chunk in source._lanes.chunks:
            if source.backend == "cuda":
                arrays = (array.get() for array in chunk.arrays)
            else:
                arrays = (buffer_view(buffer) for buffer in chunk.buffers)
            entry = {"first": chunk.first, "scans": chunk.scans}
            for name, array in zip(("payload", "offset", "model"), arrays):
                view = memoryview(array).cast("B")
                entry[name + "_offset"] = body.tell()
                entry[name + "_bytes"] = len(view)
                body.write(view)
            header["chunks"].append(entry)
        write_envelope(path, header, body)


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
    header, start = read_header(path)
    shape = tuple(header["shape"])
    if scan_shape is not None and tuple(scan_shape) != shape[:2]:
        raise ValueError("scan_shape disagrees with the saved acquisition; omit scan_shape.")
    if header["codec"] == FLOAT_CODEC:
        return _load_float(path, header, start, selected_backend, device)
    if header["codec"] == SCALED_CODEC:
        if selected_backend != "mps":
            raise NotImplementedError(
                "Scaled uint16 .qem results reopen on Apple GPUs (backend='mps'), or "
                "as a dense CPU reference with backend='cpu'."
            )
        return _load_scaled_mps(path, header, start, verbose)
    if selected_backend == "mps":
        return _load_counts_mps(path, header, start, verbose)
    return _load_counts_cuda(path, header, start, device, verbose, started)


def mps_counts_dataset(resident: MPSStreamedCounts, metadata: dict, started: float, verbose: bool) -> Dataset4dstemGPU:
    """Wrap an exact native-count Metal resident with the common loaded-data metadata.

    Saved ``.qem`` copies and native DigitalMicrograph cameras both end here,
    so the two report identical fields.
    """
    shape = resident.shape
    record = dict(metadata)
    record.update(
        resident_metadata(shape, resident.dtype, resident.nbytes, backend="mps"),
        device="mps",
        source_dtype=resident.dtype.name,
        source_logical_tensor_bytes=math.prod(shape) * resident.dtype.itemsize,
        file_counts_exact=True,
        lossless_exact=True,
        working_counts_exact=True,
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


def _load_counts_cuda(path, header, start, device, verbose, started):
    """Upload checked integer-codec arrays to CUDA through two bounded pinned buffers."""
    import cupy as cp

    shape = tuple(header["shape"])
    selected = cuda_device_index(device)
    valid = decode_valid(header["valid"], shape[2:])
    with cp.cuda.Device(selected):
        source = StreamedCounts(shape, np.dtype(header["dtype"]), valid)
        stream = cp.cuda.Stream(non_blocking=True)
        arena = cp.empty(header["bytes"], cp.uint8)
        capacity = min(BLOCK_BYTES, header["bytes"])
        owners = [cp.cuda.alloc_pinned_memory(capacity) for _ in range(2)]
        buffers = [np.frombuffer(owner, np.uint8, count=capacity) for owner in owners]
        events = [None, None]
        try:
            with open(path, "rb", buffering=0) as handle, ThreadPoolExecutor(max_workers=1) as reader:
                handle.seek(start)

                def read(number):
                    slot = number % 2
                    count = min(BLOCK_BYTES, header["bytes"] - number * BLOCK_BYTES)
                    view = memoryview(buffers[slot][:count])
                    cursor = 0
                    while cursor < count:
                        length = handle.readinto(view[cursor:])
                        if not length:
                            raise ValueError("ANS snapshot ended during loading; recopy it.")
                        cursor += length
                    if hashlib.sha256(view).hexdigest() != header["sha256"][number]:
                        raise ValueError(f"ANS checksum mismatch in block {number}; recopy the file.")
                    return count

                pending = reader.submit(read, 0)
                for number in range(len(header["sha256"])):
                    count = pending.result()
                    slot = number % 2
                    if number + 1 < len(header["sha256"]):
                        if events[1 - slot] is not None:
                            events[1 - slot].synchronize()
                        pending = reader.submit(read, number + 1)
                    with stream:
                        arena[number * BLOCK_BYTES:number * BLOCK_BYTES + count].set(buffers[slot][:count], stream=stream)
                        events[slot] = cp.cuda.Event()
                        events[slot].record(stream)
                stream.synchronize()
            for chunk in header["chunks"]:
                arrays = []
                for spec, dtype in zip(chunk["arrays"], ARRAY_DTYPES):
                    end = spec["offset"] + spec["count"] * np.dtype(dtype).itemsize
                    arrays.append(arena[spec["offset"]:end].view(dtype))
                source.chunks.append(Chunk(chunk["first"], chunk["scans"], tuple(arrays)))
            source.ready_scans = math.prod(shape[:2])
            metadata = dict(header["metadata"])
            if "scientific_metadata" in header:
                metadata = qem_metadata.effective_metadata(metadata, header["scientific_metadata"])
                metadata["scientific_metadata"] = header["scientific_metadata"]
                metadata["container"] = header["container"]
                metadata["container_version"] = header["container_version"]
            metadata.setdefault("original_source_path", metadata.get("source_path"))
            metadata.update(
                resident_metadata(shape, source.dtype, source.nbytes, backend="cuda"),
                source_kind="resident", source_path=str(Path(path).resolve()),
                source_dtype=source.dtype.name,
                source_logical_tensor_bytes=math.prod(shape) * source.dtype.itemsize,
                device=f"cuda:{selected}", resident_profile=INTEGER_CODEC,
                index_bytes=source.index_nbytes,
                file_counts_exact=True, working_counts_exact=True, lossless_exact=True,
                load_timings=dict(resident_ready_seconds=time.perf_counter() - started,
                                  encode_seconds=0.0, index_seconds=0.0,
                                  pinned_staging_bytes=2 * capacity,
                                  verified_encoded_bytes=header["bytes"]),
            )
            if verbose:
                print(f"Restored exact CUDA ANS {shape} in {time.perf_counter() - started:.3f} s.")
            return Dataset4dstemGPU(source, metadata)
        except BaseException:
            stream.synchronize()
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
        if "scientific_metadata" in header:
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
        attributes=dict(header.get("attributes", {})),
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
            with ThreadPoolExecutor(max_workers=min(8, blocks)) as workers:
                list(workers.map(read, range(blocks)))
        finally:
            os.close(descriptor)
        return buffers
    except BaseException:
        for buffer in buffers:
            release_buffer(buffer)
        raise


def _load_float(path, header, start, backend, device):
    """Authenticate and upload encoded float32 bit lanes; never expand a complete source."""
    started = time.perf_counter()
    source = FloatANSResident(header, backend, device)
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
        background = header["empad"].get("background")
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
            resident_metadata(source.shape, np.float32, source.nbytes, backend=backend),
            scientific_metadata=header["scientific_metadata"],
            qem_empad=header["empad"],
            container=header["container"],
            container_version=header["container_version"],
            source_path=str(Path(path).resolve()),
            source_kind="resident",
            device=f"cuda:{source.device}" if backend == "cuda" else "mps",
            shape=source.shape,
            source_dtype="float32",
            resident_profile=FLOAT_CODEC,
            source_logical_tensor_bytes=source.logical_nbytes,
            file_counts_exact=True,
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
