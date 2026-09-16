"""Checksummed snapshots of native GPU ANS streams and their spatial indexes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import _qem_metadata

MAGIC = b"QGPUSTRM"
PROFILE = "runtime-column-rans-spatial-v2"
BLOCK = 64 << 20
DTYPES = ("uint8", "uint32", "uint8", "uint32", "uint64", "uint8")


def is_streamed_file(path) -> bool:
    """Identify a snapshot without reading detector payloads."""
    if not isinstance(path, (str, os.PathLike)):
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(8) in (MAGIC, _qem_metadata.MAGIC)
    except (OSError, TypeError):
        return False


def read_header(path) -> tuple[dict, int]:
    """Validate the version, geometry and every array span before allocation."""
    from quantem.gpu._compact.streamed import field_count

    with open(path, "rb") as handle:
        prefix = handle.read(24)
        if len(prefix) != 24 or prefix[:8] not in (MAGIC, _qem_metadata.MAGIC):
            raise ValueError("Not a CUDA ANS snapshot; choose a file saved by io.save.")
        length, start = struct.unpack("<QQ", prefix[8:])
        size = os.fstat(handle.fileno()).st_size
        if not 0 < length <= 16 << 20 or start != 56 + length or start > size:
            raise ValueError("Invalid ANS snapshot header length; recopy the file.")
        digest = handle.read(32)
        blob = handle.read(length)
        if hashlib.sha256(blob).digest() != digest:
            raise ValueError("ANS snapshot header checksum mismatch; recopy the file.")
        header = json.loads(blob)
        if prefix[:8] == _qem_metadata.MAGIC:
            _qem_metadata.validate_header(header)
            if header.get("codec") != PROFILE:
                raise NotImplementedError(
                    f"This Python reader does not support QEM codec {header.get('codec')!r}. "
                    "Open EMPAD float32 QEM files in Live4DSTEM, or use the original acquisition."
                )
    try:
        shape = header["shape"]
        if (header["profile"] != PROFILE or header["version"] != 1
                or header["interval"] != 512 or header["dtype"] not in ("uint8", "uint16")
                or len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError("Unsupported ANS snapshot geometry or codec version.")
        pixels, fields = math.prod(shape[2:]), field_count(shape[2:])
        if len(bytes.fromhex(header["valid"])) != (pixels + 7) // 8:
            raise ValueError("Invalid detector validity mask.")
        cursor = first = 0
        for chunk in header["chunks"]:
            scans = chunk["scans"]
            if chunk["first"] != first or type(scans) is not int or scans <= 0:
                raise ValueError("Invalid ANS chunk scan coverage.")
            blocks = math.ceil(scans / 512)
            if blocks * pixels * (2 * min(scans, 512) + 4) >= 2**32:
                raise ValueError("ANS chunk offsets exceed the codec's uint32 range.")
            first += scans
            if len(chunk["arrays"]) != 6:
                raise ValueError("Incomplete ANS chunk arrays.")
            for index, spec in enumerate(chunk["arrays"]):
                count = spec["count"]
                expected = {1: blocks * pixels + 1, 2: blocks * pixels,
                            4: blocks * fields + 1, 5: blocks * fields}
                cursor = (cursor + 7) & ~7
                if (type(count) is not int or count < 0 or spec["offset"] != cursor
                        or index in expected and count != expected[index]):
                    raise ValueError("Invalid ANS array span.")
                cursor += count * np.dtype(DTYPES[index]).itemsize
        if first != math.prod(shape[:2]) or header["bytes"] != cursor or start + cursor != size:
            raise ValueError("Incomplete ANS snapshot; recopy the complete file.")
        if len(header["sha256"]) != math.ceil(cursor / BLOCK):
            raise ValueError("Incomplete ANS snapshot checksum table.")
        if not isinstance(header["metadata"], dict):
            raise ValueError("Invalid acquisition metadata.")
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed ANS snapshot header; recopy the file.") from error
    return header, start


def save_streamed(path, source, metadata: dict | None = None) -> None:
    """Atomically save exact encoded arrays; never decode or overwrite a file."""
    from quantem.gpu._compact.streamed import StreamedCounts

    from .backends.mps._streamed import MPSStreamedCounts
    metal = isinstance(source, MPSStreamedCounts)
    if not metal and (type(source) is not StreamedCounts or source.native_source is not None):
        raise TypeError("Snapshot saving requires native CUDA or Metal streamed ANS counts.")
    if metal and len(getattr(source, "spatial_chunks", [])) != len(source.chunks):
        raise ValueError("This Metal resident has no spatial indexes; load native DM4 or a saved camera snapshot.")
    if source.is_released or source.ready_scans != math.prod(source.shape[:2]):
        raise ValueError("Save a complete, live CUDA source; reload the acquisition first.")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new destination.")
    table, cursor = [], 0
    with tempfile.TemporaryFile(dir=path.parent) as body:
        for number, chunk in enumerate(source.chunks):
            entries = []
            if metal:
                from .backends.mps.packed import _buffer_view
                arrays = [np.frombuffer(_buffer_view(buffer), np.dtype(dtype))
                          for buffer, dtype in zip((*chunk.buffers, *source.spatial_chunks[number]), DTYPES)]
            else:
                arrays = chunk.arrays
            for array in arrays:
                aligned = (cursor + 7) & ~7
                body.write(b"\0" * (aligned - cursor))
                entries.append(dict(offset=aligned, count=int(array.size)))
                body.write(memoryview(array if metal else array.get()).cast("B"))
                cursor = aligned + array.nbytes
            table.append(dict(first=chunk.first, scans=chunk.scans, arrays=entries))
        body.seek(0)
        checksums = []
        while block := body.read(BLOCK):
            checksums.append(hashlib.sha256(block).hexdigest())
        header = dict(version=1, profile=PROFILE, interval=512,
                      shape=list(source.shape), dtype=source.dtype.name,
                      valid=np.packbits(source.valid_pixels.ravel()).tobytes().hex(),
                      chunks=table, bytes=cursor, sha256=checksums,
                      metadata={} if metadata is None else metadata)
        from ._ans import _json_metadata

        qem = path.suffix.lower() == ".qem"
        if qem:
            header.update(container="quantem.qem", container_version=1, codec=PROFILE,
                          scientific_metadata=_qem_metadata.acquisition_metadata(source.shape, header["metadata"]))

        blob = json.dumps(header, default=_json_metadata, allow_nan=False).encode()
        if len(blob) > 16 << 20:
            raise ValueError("Acquisition metadata exceeds the 16 MiB snapshot header limit.")
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                signature = _qem_metadata.MAGIC if qem else MAGIC
                output.write(signature + struct.pack("<QQ", len(blob), 56 + len(blob)) + hashlib.sha256(blob).digest() + blob)
                body.seek(0)
                while block := body.read(BLOCK):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, path)  # publish only a complete file, without replacement
        finally:
            os.unlink(temporary)


def load_streamed(path, *, backend, representation, scan_shape, device, verbose):
    """Restore checked encoded bytes using two bounded pinned buffers, no encoding."""
    from .backends import resolve_backend
    from .models import FourDSTEMData
    from quantem.gpu._compact.streamed import Chunk, StreamedCounts

    selected_backend = resolve_backend(backend)
    if selected_backend not in ("cuda", "mps") or representation not in (None, "encoded"):
        raise NotImplementedError("ANS snapshots reopen as encoded counts; choose backend='cuda' or 'mps'.")
    started = time.perf_counter()
    header, start = read_header(path)
    shape = tuple(header["shape"])
    if scan_shape is not None and tuple(scan_shape) != shape[:2]:
        raise ValueError("scan_shape disagrees with the saved acquisition; omit scan_shape.")
    if selected_backend == "mps":
        from ._camera_mps import load_snapshot_mps

        return load_snapshot_mps(path, header, start, verbose=verbose)
    import cupy as cp

    selected = cp.cuda.Device().id if device is None else int(str(device).removeprefix("cuda:"))
    valid = np.unpackbits(np.frombuffer(bytes.fromhex(header["valid"]), np.uint8))
    valid = valid[:math.prod(shape[2:])].astype(bool).reshape(shape[2:])
    with cp.cuda.Device(selected):
        source = StreamedCounts(shape, np.dtype(header["dtype"]), valid)
        stream = cp.cuda.Stream(non_blocking=True)
        arena = cp.empty(header["bytes"], cp.uint8)
        capacity = min(BLOCK, header["bytes"])
        owners = [cp.cuda.alloc_pinned_memory(capacity) for _ in range(2)]
        buffers = [np.frombuffer(owner, np.uint8, count=capacity) for owner in owners]
        events = [None, None]
        try:
            with open(path, "rb", buffering=0) as handle, ThreadPoolExecutor(max_workers=1) as reader:
                handle.seek(start)

                def read(number):
                    slot = number % 2
                    count = min(BLOCK, header["bytes"] - number * BLOCK)
                    view = memoryview(buffers[slot][:count])
                    at = 0
                    while at < count:
                        got = handle.readinto(view[at:])
                        if not got:
                            raise ValueError("ANS snapshot ended during loading; recopy it.")
                        at += got
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
                        arena[number * BLOCK:number * BLOCK + count].set(buffers[slot][:count], stream=stream)
                        events[slot] = cp.cuda.Event()
                        events[slot].record(stream)
                stream.synchronize()
            for chunk in header["chunks"]:
                arrays = []
                for spec, dtype in zip(chunk["arrays"], DTYPES):
                    end = spec["offset"] + spec["count"] * np.dtype(dtype).itemsize
                    arrays.append(arena[spec["offset"]:end].view(dtype))
                source.chunks.append(Chunk(chunk["first"], chunk["scans"], tuple(arrays)))
            source.ready_scans = math.prod(shape[:2])
            metadata = dict(header["metadata"])
            if "scientific_metadata" in header:
                metadata["scientific_metadata"] = header["scientific_metadata"]
                metadata["container"] = header["container"]
                metadata["container_version"] = header["container_version"]
            metadata.setdefault("original_source_path", metadata.get("source_path"))
            metadata.update(source_kind="resident", source_path=str(Path(path).resolve()),
                            source_shape=shape, working_shape=shape, scan_shape=shape[:2],
                            detector_shape=shape[2:], dtype=source.dtype.name,
                            source_dtype=source.dtype.name, working_dtype=source.dtype.name,
                            n_frames=math.prod(shape[:2]), scan_bin=1, detector_bin=1, crop=None,
                            source_logical_tensor_bytes=math.prod(shape) * source.dtype.itemsize,
                            working_logical_tensor_bytes=math.prod(shape) * source.dtype.itemsize,
                            backend="cuda", device=f"cuda:{selected}", representation="encoded",
                            residency="device", resident_profile=PROFILE,
                            physical_resident_bytes=source.nbytes, index_bytes=source.index_nbytes,
                            file_counts_exact=True, working_counts_exact=True, lossless_exact=True,
                            load_timings=dict(resident_ready_seconds=time.perf_counter() - started,
                                              encode_seconds=0.0, index_seconds=0.0,
                                              pinned_staging_bytes=2 * capacity,
                                              verified_encoded_bytes=header["bytes"]))
            if verbose:
                print(f"Restored exact CUDA ANS {shape} in {time.perf_counter() - started:.3f} s.")
            return FourDSTEMData(source, metadata)
        except BaseException:
            stream.synchronize()
            source.release()
            raise
