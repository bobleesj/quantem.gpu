"""Native camera and saved ANS loading through bounded Metal allocations."""

import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from .backends.mps._streamed import MPSStreamedCounts, _Chunk
from .backends.mps.packed import _allocate_shared, _buffer_view, _release
from .models import FourDSTEMData


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


def load_snapshot_mps(path, header, start, *, verbose=False):
    """Restore the same exact ANS streams written on CUDA without re-encoding."""
    from ._streamed_file import BLOCK, DTYPES

    started = time.perf_counter()
    shape = tuple(header["shape"])
    valid = np.unpackbits(np.frombuffer(bytes.fromhex(header["valid"]), np.uint8))
    valid = valid[: math.prod(shape[2:])].astype(bool).reshape(shape[2:])
    resident = MPSStreamedCounts(shape, np.dtype(header["dtype"]), valid)
    try:
        mapping = np.memmap(
            path, mode="r", dtype=np.uint8, offset=start, shape=(header["bytes"],)
        )

        def verify(number):
            offset = number * BLOCK
            digest = hashlib.sha256(
                memoryview(mapping[offset : offset + BLOCK])
            ).hexdigest()
            if digest != header["sha256"][number]:
                raise ValueError(
                    f"ANS checksum mismatch in block {number}; recopy the file."
                )

        with ThreadPoolExecutor(max_workers=min(8, len(header["sha256"]))) as workers:
            list(workers.map(verify, range(len(header["sha256"]))))
        for chunk in header["chunks"]:
            buffers = []
            try:
                for spec, dtype in zip(chunk["arrays"], DTYPES):
                    size = spec["count"] * np.dtype(dtype).itemsize
                    buffer = _allocate_shared(
                        resident._device, resident._metal, max(1, size), "saved ANS"
                    )
                    buffers.append(buffer)
                    if size:
                        _buffer_view(buffer, size)[:] = memoryview(
                            mapping[spec["offset"] : spec["offset"] + size]
                        )
                resident.chunks.append(
                    _Chunk(chunk["first"], chunk["scans"], tuple(buffers[:3]))
                )
                # Keep the exact spatial index for indexed detector kernels.
                if not hasattr(resident, "spatial_chunks"):
                    resident.spatial_chunks = []
                resident.spatial_chunks.append(tuple(buffers[3:]))
            except BaseException:
                for buffer in buffers:
                    _release(buffer)
                raise
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
    return FourDSTEMData(resident, record)
