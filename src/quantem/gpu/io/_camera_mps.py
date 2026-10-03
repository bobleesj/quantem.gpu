"""Native camera and saved ANS loading through bounded Metal allocations."""

from quantem.gpu.io.models import create_dataset

import bisect
import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .backends.mps._streamed import MPSStreamedCounts, _Chunk
from .backends.mps.packed import _allocate_shared, _buffer_view, _pread_exact, _release


def load_camera_mps(source, *, verbose=False):
    """Read unchanged DM counts into a shared staging buffer and encode on Metal."""
    started = time.perf_counter()
    resident = MPSStreamedCounts(source.shape, source.dtype)
    scans, pixels = math.prod(source.shape[:2]), math.prod(source.shape[2:])
    block = min(scans, 512)
    staging = None
    resident.spatial_chunks = []
    try:
        staging = _allocate_shared(
            resident._device,
            resident._metal,
            block * pixels * source.dtype.itemsize,
            "DM4 staging",
        )
        with source.path.open("rb", buffering=0) as handle:
            handle.seek(source.offset)
            for first in range(0, scans, block):
                count = min(block, scans - first)
                view = _buffer_view(staging, count * pixels * source.dtype.itemsize)
                at = 0
                while at < len(view):
                    got = handle.readinto(view[at:])
                    if not got:
                        raise ValueError("Incomplete DM4 counts; finish the download.")
                    at += got
                resident.append(
                    SimpleNamespace(
                        _mtl=staging,
                        ndim=3,
                        shape=(count, *source.shape[2:]),
                        dtype=source.dtype,
                    )
                )
                from .backends.mps._spatial import build_index

                resident.spatial_chunks.append(build_index(resident, staging, count))
        source.assert_unchanged()
        return _result(resident, source.metadata, started, verbose)
    except BaseException:
        resident.release()
        raise
    finally:
        _release(staging)


def _read_verified(path, header, start, device, metal, spans):
    """Read body arrays straight into shared Metal buffers, checking each 64 MiB block.

    ``spans`` lists ``(offset, bytes)`` in file order. Workers read and hash whole
    blocks in parallel, alignment padding included, so every byte is read once and
    nothing reaches a decoder unless all checksums match.
    """
    from ._streamed_file import BLOCK

    buffers = []
    try:
        for _, size in spans:
            buffers.append(_allocate_shared(device, metal, max(1, size), "saved ANS"))
        views = [_buffer_view(buffer, size) for buffer, (_, size) in zip(buffers, spans)]
        starts = [offset for offset, _ in spans]
        total = header["bytes"]

        def read(number):
            lower, upper = number * BLOCK, min(total, (number + 1) * BLOCK)
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
                _pread_exact(descriptor, target, start + cursor, ".qem body")
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
            _release(buffer)
        raise


def load_snapshot_mps(path, header, start, *, verbose=False):
    """Restore the same exact ANS streams written on CUDA without re-encoding."""
    from ._streamed_file import DTYPES

    started = time.perf_counter()
    shape = tuple(header["shape"])
    valid = np.unpackbits(np.frombuffer(bytes.fromhex(header["valid"]), np.uint8))
    valid = valid[: math.prod(shape[2:])].astype(bool).reshape(shape[2:])
    resident = MPSStreamedCounts(shape, np.dtype(header["dtype"]), valid)
    try:
        spans = [
            (spec["offset"], spec["count"] * np.dtype(dtype).itemsize)
            for chunk in header["chunks"]
            for spec, dtype in zip(chunk["arrays"], DTYPES)
        ]
        buffers = _read_verified(
            path, header, start, resident._device, resident._metal, spans
        )
        # Keep the exact spatial index for indexed detector kernels.
        resident.spatial_chunks = []
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
            from ._qem_metadata import effective_metadata

            metadata = effective_metadata(metadata, header["scientific_metadata"])
            metadata["scientific_metadata"] = header["scientific_metadata"]
            metadata["container"] = header["container"]
            metadata["container_version"] = header["container_version"]
        return _result(resident, metadata, started, verbose)
    except BaseException:
        resident.release()
        raise


def load_scaled_snapshot_mps(path, header, start, *, verbose=False):
    """Reopen saved scaled uint16 codes and their regional calibration unchanged.

    Each chunk's streams stay ANS encoded in their own resident; values are
    restored as ``float32(code * scale + offset)`` only inside queries.
    """
    from ._qem_metadata import effective_metadata
    from ._streamed_file import SCALED
    from .backends.mps.precision import PrecisionSource, _ANSPart

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
        buffers = _read_verified(path, header, start, first._device, first._metal, spans)
        for number, chunk in enumerate(header["chunks"]):
            scans = chunk["scans"]
            streams = math.ceil(scans / 512) * pixels
            payload, offsets, models = buffers[3 * number : 3 * number + 3]
            stops = np.frombuffer(_buffer_view(offsets, (streams + 1) * 4), np.uint32)
            kinds = np.frombuffer(_buffer_view(models, streams), np.uint8)
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
            _release(buffer)
        raise
    metadata = effective_metadata({}, header["scientific_metadata"])
    metadata.update(
        scientific_metadata=header["scientific_metadata"],
        container=header["container"],
        container_version=header["container_version"],
        attributes=dict(header.get("attributes", {})),
        precision=report,
        conversion_report_origin="saved",
        source_kind="resident",
        source_path=str(Path(path).resolve()),
        backend="mps",
        device="mps",
        representation="encoded",
        resident_codec="ans",
        resident_profile=SCALED,
        residency="device",
        source_shape=shape,
        working_shape=shape,
        scan_shape=shape[:2],
        detector_shape=shape[2:],
        n_frames=math.prod(shape[:2]),
        dtype="float32",
        source_dtype=report["source_dtype"],
        storage_dtype="uint16",
        working_dtype="float32",
        physical_resident_bytes=source.nbytes,
        working_logical_tensor_bytes=math.prod(shape) * 4,
        lossless_exact=report["changed"] == 0,
        file_counts_exact=False,
        working_counts_exact=False,
        scan_bin=1,
        detector_bin=1,
        crop=None,
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
    return create_dataset(source, metadata)


def _result(resident, metadata, started, verbose):
    shape = resident.shape
    record = dict(metadata)
    record.update(
        backend="mps",
        device="mps",
        representation="encoded",
        residency="device",
        source_shape=shape,
        working_shape=shape,
        scan_shape=shape[:2],
        detector_shape=shape[2:],
        dtype=resident.dtype.name,
        source_dtype=resident.dtype.name,
        working_dtype=resident.dtype.name,
        n_frames=math.prod(shape[:2]),
        physical_resident_bytes=resident.nbytes,
        source_logical_tensor_bytes=math.prod(shape) * resident.dtype.itemsize,
        working_logical_tensor_bytes=math.prod(shape) * resident.dtype.itemsize,
        file_counts_exact=True,
        lossless_exact=True,
        working_counts_exact=True,
        scan_bin=1,
        detector_bin=1,
        crop=None,
        resident_profile="runtime-column-rans-spatial-v2",
        load_timings=dict(
            resident.load_metrics, resident_ready_seconds=time.perf_counter() - started
        ),
    )
    if verbose:
        print(
            f"Loaded exact native camera ANS on Metal in {time.perf_counter() - started:.3f} s."
        )
    return create_dataset(resident, record)
