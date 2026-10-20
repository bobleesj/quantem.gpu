"""Byte-bounded ingestion into the shared integer and float ANS residents.

Arrays already in memory, and HDF5 layouts outside the direct chunk decoder, both
reach the residents through ``load_array_resident``.
"""

import bisect
import hashlib
import math
import os
import time
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch

from quantem.gpu.device.cuda_runtime import cuda_device_index
from quantem.gpu.device.metal_runtime import shared_array
from quantem.gpu.formats.hdf5.frames import detector_sources
from quantem.gpu.formats.float8 import hdf5_dtype, is_float8
from quantem.gpu.formats.qem import metadata as qem_metadata
from quantem.gpu.formats.qem.snapshot import BIT_CODEC, BIT_DTYPES, FLOAT_CODEC, INTEGER_CODEC, encode_valid
from quantem.gpu.io.dataset import Dataset4dstemGPU, resident_metadata
from quantem.gpu.resident import float_ans
from quantem.gpu.resident.bit_ans import BitANSResident
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.cuda.hot_pixels import CUDAHotPixelCorrector
from quantem.gpu.resident.hot_pixels import hot_pixel_record
from quantem.gpu.resident.mps import precision as metal_precision
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.hot_pixels import MPSHotPixelCorrector
from quantem.gpu.resident.mps.spatial import build_index

MAX_INGEST_BYTES = 32 << 20


def read_frame_block(data, shape: tuple[int, ...], first: int, stop: int) -> np.ndarray | torch.Tensor:
    """Copy file-backed rows without reshaping a noncontiguous acquisition."""
    if torch.is_tensor(data):
        bits_t = data.detach().view(torch.uint8)
        block_t = torch.empty(
            (stop - first, *shape[2:]), dtype=torch.uint8, device=data.device
        )
        cursor = first
        while cursor < stop:
            row, column = divmod(cursor, shape[1])
            count = min(shape[1] - column, stop - cursor)
            block_t[cursor - first:cursor - first + count] = bits_t[row, column:column + count]
            cursor += count
        return block_t.view(data.dtype)
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
    dtype: np.dtype | str | torch.dtype,
    read_frames: Callable[[int, int], np.ndarray | torch.Tensor],
    metadata: dict,
    *,
    backend: str,
    device: int | str | None = None,
    pixel_mask: np.ndarray | None = None,
    hot_pixel_correction: str = "none",
    verbose: bool = False,
) -> Dataset4dstemGPU:
    """Encode bounded input windows; never allocate a dense acquisition."""
    started = time.perf_counter()
    shape = tuple(shape)
    source_dtype = np.dtype(str(dtype).removeprefix("torch.")) if isinstance(dtype, torch.dtype) else np.dtype(dtype)
    dtype = source_dtype.newbyteorder("=")
    peak_bytes = 0
    if dtype.name not in BIT_DTYPES:
        raise NotImplementedError(
            f"ANS supports real integer and floating dtypes through 64 bits; got {dtype}. "
            "Keep the original source; no cast or upload occurred."
        )
    correction = hot_pixel_record(pixel_mask, hot_pixel_correction, backend=backend)
    if correction["applied"] and dtype.name not in ("uint8", "uint16", "uint32"):
        raise ValueError(
            f"The source flags {correction['pixel_count']} detector pixels; hot_pixel_correction="
            f"{hot_pixel_correction!r} fills them in uint8/uint16/uint32 counts only, and this acquisition is "
            f"{dtype}. Load it with hot_pixel_correction='none' to keep the stored values; flagged pixels "
            "stay excluded from detector products."
        )
    floating = dtype == np.dtype("float32")
    bit_lanes = dtype not in (np.dtype("uint8"), np.dtype("uint16"))
    # The word codec uses two bytes even for signed 8-bit measurements.
    frame_bytes = math.prod(shape[2:]) * max(2, dtype.itemsize)
    if frame_bytes > MAX_INGEST_BYTES:
        raise MemoryError("One detector frame exceeds the 32 MiB ingestion limit.")
    block_scans = min(512, MAX_INGEST_BYTES // frame_bytes)
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
            if bit_lanes:
                source = bit_resident(shape, dtype, metadata, valid, backend, device)
                resident = source._lanes
            elif backend == "cuda":
                source = resident = StreamedCounts(shape, dtype, valid)
            else:
                source = resident = MPSStreamedCounts(shape, dtype, valid)
            corrector_type = CUDAHotPixelCorrector if backend == "cuda" else MPSHotPixelCorrector
            corrector = corrector_type(pixel_mask, hot_pixel_correction)

            for first in range(0, math.prod(shape[:2]), block_scans):
                stop = min(first + block_scans, math.prod(shape[:2]))
                block = read_frames(first, stop)
                tensor_input = torch.is_tensor(block)
                block_dtype = np.dtype(str(block.dtype).removeprefix("torch.")) if tensor_input else block.dtype
                if (
                    block.shape != (stop - first, *shape[2:])
                    or block_dtype != source_dtype
                ):
                    raise ValueError(
                        "The source changed shape or dtype while loading; reopen it."
                    )
                if not tensor_input:
                    block = np.ascontiguousarray(block)
                    if not block.dtype.isnative:
                        block = block.byteswap().view(dtype)
                peak_bytes = max(peak_bytes, block.numel() * block.element_size() if tensor_input else block.nbytes)
                if floating:
                    logical.update(memoryview(block).cast("B"))
                if backend == "cuda":
                    raw = (
                        cp.from_dlpack(block.view(torch.uint8).to(f"cuda:{selected}"))
                        if tensor_input else cp.asarray(block.view(np.uint8) if is_float8(dtype) else block)
                    )
                    if corrector is not None:
                        corrector.apply(raw)
                    (source.append_device if bit_lanes else resident.append)(raw)
                    del raw
                else:
                    if tensor_input:
                        from quantem.gpu.device.metal_runtime import tensor_buffer

                        block_t = block.view(torch.uint8).to("mps")
                        torch.mps.synchronize()
                        source.append_device(shared_array(tensor_buffer(block_t), dtype, block_t.shape))
                        del block_t, block
                        continue
                    raw = metal_precision.upload(block.view(np.uint8) if is_float8(dtype) else block)
                    try:
                        if corrector is not None:
                            corrector.apply(raw)
                        (source.append_device if bit_lanes else resident.append)(
                            shared_array(raw._mtl, raw.dtype, raw.shape)
                        )
                        if not bit_lanes:
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
                resident_profile=source.header["codec"] if bit_lanes else INTEGER_CODEC,
                **correction_metadata(correction),
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


def bit_resident(shape, dtype, metadata, valid, backend, device=None):
    """Construct an empty typed GPU word resident with acquisition metadata."""
    dtype = np.dtype(dtype)
    floating = dtype == np.dtype("float32") and bool(np.all(valid))
    codec = FLOAT_CODEC if floating else BIT_CODEC
    scientific = qem_metadata.acquisition_metadata(shape, metadata)
    header = dict(
        container="quantem.qem", container_version=1, version=1,
        profile=codec, codec=codec, shape=list(shape), dtype=dtype.name,
        metadata=qem_metadata.retained_metadata(metadata),
        scientific_metadata=scientific,
    )
    if floating:
        header["empad"] = dict(metadata.get("qem_empad") or {})
        header["empad"].setdefault("format_identifier", "float32")
        header["empad"].setdefault("format_name", "Float32")
        header["empad"].setdefault("microscope_metadata", {
            str(key): str(value) for key, value in scientific["source_metadata"].items()
        })
    else:
        header["valid"] = encode_valid(valid)
    resident_type = float_ans.FloatANSResident if floating else BitANSResident
    return resident_type(header, backend, device)


def _inflate_frames(data, result, first) -> None:
    """Fill ``result`` with consecutive one-frame gzip chunks of ``data``, inflated on several cores.

    HDF5 inflates one chunk at a time on one core: 3.9 s of the 5.8 s load of
    10000 patterns of 256 x 256. The stored chunk bytes are read as they are
    and zlib, which releases the interpreter lock, inflates them in parallel.
    A chunk whose filter the writer skipped is copied as stored.
    """

    def inflate(offset):
        filter_mask, stored = data.id.read_direct_chunk((first + offset, 0, 0))
        result[offset] = np.frombuffer(stored if filter_mask else zlib.decompress(stored), result.dtype).reshape(result.shape[1:])

    with ThreadPoolExecutor(min(8, os.cpu_count())) as pool:
        list(pool.map(inflate, range(len(result))))


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
        if (
            selected is None
            and direct
            and np.dtype(info.dtype)
            in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"))
        ):
            return None
        starts = [0]
        for data in datasets:
            if (
                data.ndim not in (3, 4)
                or data.shape[-2:] != shape[2:]
                or hdf5_dtype(data) != np.dtype(info.dtype)
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

        # One gzip chunk per frame in the host's byte order, every chunk written: inflated in parallel.
        frame_gzip = [
            data.chunks == (1, *shape[2:])
            and [data.id.get_create_plist().get_filter(index)[0] for index in range(data.id.get_create_plist().get_nfilters())]
            == [h5py.h5z.FILTER_DEFLATE]
            and data.dtype.isnative
            and data.id.get_num_chunks() == data.shape[0]
            for data in datasets
        ]

        def read_frames(first, stop):
            result = np.empty((stop - first, *shape[2:]), datasets[0].dtype)
            cursor = first
            while cursor < stop:
                index = bisect.bisect_right(starts, cursor) - 1
                end = min(stop, starts[index + 1])
                data = datasets[index]
                local = cursor - starts[index]
                if frame_gzip[index]:
                    _inflate_frames(data, result[cursor - first : end - first], local)
                elif data.ndim == 3:
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
            return result.view(np.dtype(info.dtype)) if is_float8(info.dtype) else result

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


def correction_metadata(record: dict) -> dict:
    """Record how stored detector-mask pixels were handled and what that means for exactness.

    A corrected pixel no longer holds the file's count, so the working data is
    exact (its own codes decode bit for bit) but no longer the file's counts.
    Uncorrected loading keeps every original bit, including detector markers.
    """
    applied = record["applied"]
    if applied:
        policy = f"gpu-{record['method']}-corrected"
    else:
        policy = "preserve-stored-counts"
    exact = not applied
    return dict(
        lossless_exact=exact,
        file_counts_exact=exact,
        detector_mask_policy=policy,
        hot_pixel_correction=record,
        working_counts_exact=True,
        flagged_markers_stored_as_zero=0,
    )
