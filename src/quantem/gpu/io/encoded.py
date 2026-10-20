"""Stream complete HDF5 acquisitions into native-count GPU residents.

A bitshuffle+LZ4 master is read in bounded blocks of scan positions: the
selected compressed chunks are read on the host, decoded on the GPU, stored
detector-mask pixels are corrected, and the counts are ANS encoded, so the
dense acquisition never exists. The host reads the next block while the GPU
decodes and encodes the current one. Other HDF5 layouts go through
``io.arrays.load_hdf5_array_resident``.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import h5py
import numpy as np

from quantem.gpu.device.cuda_runtime import _release_pinned, cp, cuda_device_index
from quantem.gpu.device.metal_runtime import release_buffer
from quantem.gpu.formats.hdf5.reads import FrameReader
from quantem.gpu.formats.qem.snapshot import INTEGER_CODEC
from quantem.gpu.io import arrays
from quantem.gpu.io.dataset import Dataset4dstemGPU, resident_metadata
from quantem.gpu.io.hdf5.cuda.decode import decompress_prepared
from quantem.gpu.io.inspect import inspect
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.cuda.hot_pixels import CUDAHotPixelCorrector
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.hot_pixels import MPSHotPixelCorrector
from quantem.gpu.resident.mps.spatial import build_index

_CUDA_MAX_STAGING_SCANS = 16384


def load_h5_ans(
    path,
    *,
    scan_shape,
    dataset_path,
    device,
    verbose,
    backend,
    hot_pixel_correction,
):
    """Load bounded native chunks into a backend-owned runtime encoded resident."""
    started = time.perf_counter()
    info = inspect(path, scan_shape=scan_shape, dataset_path=dataset_path)
    # workflows over several acquisitions (MAPED tilts, focal series) label each
    # one by the file it came from; every branch below copies info.metadata
    info.metadata["source_path"] = str(path)
    generic = arrays.load_hdf5_array_resident(
        path, dataset_path=dataset_path, backend=backend,
        device=device, verbose=verbose, hot_pixel_correction=hot_pixel_correction,
        info=info,
    )
    if generic is not None:
        return generic
    if backend == "mps":
        return _load_h5_ans_mps(path, info, hot_pixel_correction, verbose)
    if backend != "cuda":
        raise NotImplementedError("Runtime HDF5-to-encoded needs CUDA or MPS.")

    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    stored_dtype = np.dtype(info.dtype)
    dtype = stored_dtype
    bit_lanes = dtype == np.dtype("uint32")
    if dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32")):
        raise TypeError(
            "Direct HDF5 decoding supports uint8/uint16/uint32. Use the bounded array reader for other dtypes."
        )
    selected = cuda_device_index(device)
    with cp.cuda.Device(selected), ExitStack() as stack:
        # Runs last: hand this load's read, decode and encode scratch back once
        # the reader and corrector are closed. The encoded chunks are not pooled.
        stack.callback(cp.get_default_memory_pool().free_all_blocks)
        corrector = CUDAHotPixelCorrector(info.pixel_mask, hot_pixel_correction)
        valid = np.ones(info.detector_shape, bool)
        if info.pixel_mask is not None and not corrector.record["applied"]:
            valid &= np.asarray(info.pixel_mask) == 0
        cp.get_default_memory_pool().free_all_blocks()
        source = (arrays.bit_resident(shape, dtype, info.metadata, valid, "cuda", selected)
                  if bit_lanes else StreamedCounts(shape, dtype, valid))
        resident = source._lanes if bit_lanes else source
        # A failed load frees the chunks it encoded: they live outside the memory
        # pool, and a traceback keeps this frame (and the source) alive.
        stack.push(lambda error_type, *_: error_type is not None and source.release())
        free, _ = cp.cuda.runtime.memGetInfo()
        # Amortize HDF5/read/decode dispatch across larger batches, while
        # reserving at least three quarters of free memory for resident output
        # and other documents. The estimate includes decoded and codec scratch.
        per_scan = math.prod(info.detector_shape) * 8
        chunk_scans = min(
            _CUDA_MAX_STAGING_SCANS,
            max(1, int((free - 128 * 1024**2) // (per_scan * 8))),
        )
        if chunk_scans < min(512, math.prod(info.scan_shape)):
            raise MemoryError(
                "Insufficient CUDA staging space; close another resident document "
                "before loading."
            )
        chunk_scans = (min(512, arrays.MAX_INGEST_BYTES // (math.prod(shape[2:]) * dtype.itemsize))
                       if bit_lanes else max(512, chunk_scans // 512 * 512))
        dataset_path = dataset_path or info.metadata.get("dataset_path")
        dataset = None
        reader = None
        if dataset_path is not None:
            handle = stack.enter_context(h5py.File(path, "r"))
            dataset = handle[dataset_path]
            if tuple(dataset.shape) != shape or np.dtype(dataset.dtype) != stored_dtype:
                raise ValueError(
                    "The selected H5 dataset must match its complete inspected "
                    "native geometry and dtype."
                )
        else:
            reader = FrameReader(str(path))
            stack.callback(reader.close)
        scans = math.prod(info.scan_shape)
        blocks = [(first, min(first + chunk_scans, scans)) for first in range(0, scans, chunk_scans)]
        # One block of compressed chunks is read ahead on a worker while the GPU decodes,
        # corrects and encodes the current block: read after read, the GPU sat idle for
        # every read and the host for every encode.
        read_ahead = stack.enter_context(ThreadPoolExecutor(max_workers=1))

        def prepare(block):
            # The staging buffer is page-locked in this GPU's context, which a worker thread enters itself.
            with cp.cuda.Device(selected):
                return reader.prepare(np.arange(*block))

        def release_unread_block():
            # A block read ahead of a failed decode was never copied to the GPU.
            if pending is not None and pending.exception() is None:
                _release_pinned(pending.result()["read_buffer"])

        pending = None if reader is None else read_ahead.submit(prepare, blocks[0])
        stack.callback(release_unread_block)
        for number, (first, stop) in enumerate(blocks):
            if dataset is not None:
                # A chunk can cut a rectangular scan row. h5py reads and decodes
                # the stored chunks on the host for this generic layout.
                raw = cp.asarray(arrays.read_frame_block(dataset, shape, first, stop))
            else:
                prepared, pending = pending.result(), None
                if number + 1 < len(blocks):
                    pending = read_ahead.submit(prepare, blocks[number + 1])
                raw = decompress_prepared(prepared, batch_bytes_target=128 * 1024**2)
                raw = raw.reshape(stop - first, *info.detector_shape)
            corrector.apply(raw)
            (source.append_device if bit_lanes else source.append)(raw)
            del raw
        metadata = dict(info.metadata)
        metadata.update(
            resident_metadata(shape, dtype, source.nbytes, backend="cuda"),
            source_dtype=stored_dtype.name,
            source_read_passes=1,
            source_logical_tensor_bytes=math.prod(shape) * stored_dtype.itemsize,
            index_bytes=0 if bit_lanes else source.index_nbytes,
            pixel_mask=info.pixel_mask,
            resident_profile=source.header["codec"] if bit_lanes else INTEGER_CODEC,
            **arrays.correction_metadata(corrector.record),
            load_timings=dict(
                resident.load_metrics,
                resident_ready_seconds=time.perf_counter() - started,
            ),
        )
        if verbose:
            print(_loaded_report(source.shape, "NVIDIA GPU", metadata))
        return Dataset4dstemGPU(source, metadata)


class _ExceedsUint16(Exception):
    """A decoded uint32 value does not fit 16 bits, so the scan keeps uint32 storage."""


def _load_h5_ans_mps(path, info, hot_pixel_correction, verbose):
    """Decode bounded HDF5 blocks and encode native-count ANS with Metal.

    uint32 counts that all fit 16 bits (Arina masters, once any requested
    bad-pixel correction has replaced the flagged pixels) are stored as uint16
    count ANS, whose detector sums use the spatial index; every value reads
    back unchanged. A single value above 65535, such as a kept 0xffffffff
    bad-pixel marker, keeps the whole scan in uint32 bit lanes.
    """
    if np.dtype(info.dtype) == np.dtype("uint32"):
        try:
            return _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, np.dtype("uint16"))
        except _ExceedsUint16:
            pass
    return _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, np.dtype(info.dtype))


def _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, dtype):
    """Load the master into a resident of ``dtype``: the stored dtype, or uint16 for uint32 counts."""
    # Metal decoding is imported only on the MPS path, like every platform decoder.
    from quantem.gpu.io.hdf5.mps.decode import MPSDecompressor, exact_uint16

    started = time.perf_counter()
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    stored_dtype = np.dtype(info.dtype)
    bit_lanes = dtype == np.dtype("uint32")
    narrowed = dtype != stored_dtype
    if stored_dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32")):
        raise TypeError(
            "Direct HDF5 decoding supports uint8/uint16/uint32."
        )
    corrector = MPSHotPixelCorrector(info.pixel_mask, hot_pixel_correction)
    valid = np.ones(info.detector_shape, bool)
    if info.pixel_mask is not None and not corrector.record["applied"]:
        valid &= np.asarray(info.pixel_mask) == 0
    source = (arrays.bit_resident(shape, dtype, info.metadata, valid, "mps")
              if bit_lanes else MPSStreamedCounts(shape, dtype, valid))
    resident = source._lanes if bit_lanes else source
    reader = FrameReader(str(path))
    decoder = None
    scans = math.prod(info.scan_shape)
    # Bound simultaneous decoded counts and ANS encoding scratch on smaller Macs;
    # uint32 batches hold half the scans, so their decoded bytes stay the same.
    # Keep batches aligned to the codec interval so exact stored counts are unchanged.
    chunk_scans = min(8192 if stored_dtype.itemsize <= 2 else 4096, scans)
    if bit_lanes:
        chunk_scans = min(512, arrays.MAX_INGEST_BYTES // (math.prod(shape[2:]) * dtype.itemsize))
    elif chunk_scans >= 512:
        chunk_scans = chunk_scans // 512 * 512
    try:
        ranges = [
            (first, min(first + chunk_scans, scans))
            for first in range(0, scans, chunk_scans)
        ]
        # Keep one host-side HDF5 read queued while Metal decodes, corrects,
        # and ANS-encodes the preceding batch. Scientific array work remains
        # on Metal; the worker only prepares compressed file bytes.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(reader.prepare, np.arange(*ranges[0]))
            for index in range(len(ranges)):
                prepared = pending.result()
                if index + 1 < len(ranges):
                    pending = pool.submit(reader.prepare, np.arange(*ranges[index + 1]))
                compressed_bytes = max(1, len(prepared["read_buffer"]))
                if decoder is None or compressed_bytes > decoder.max_compressed_bytes:
                    if decoder is not None:
                        decoder.free()
                    # Reuse bounded decode scratch across batches. Reserve room
                    # for compressed-size variation without rebuilding every
                    # block; decoded output still has independent ownership.
                    frame_bytes = int(prepared["frame_bytes"])
                    decoder = MPSDecompressor(
                        max_compressed_bytes=2 * compressed_bytes,
                        max_frames=chunk_scans,
                        frame_bytes=frame_bytes,
                        n_blocks_per_frame=(frame_bytes + 8191) // 8192,
                    )
                raw = decoder.load_prepared_frames(prepared)
                try:
                    corrector.apply(raw)
                    if narrowed:
                        counts = exact_uint16(raw)
                        if counts is None:
                            raise _ExceedsUint16
                        buffer, raw._mtl = raw._mtl, None
                        release_buffer(buffer)
                        raw = counts
                    (source.append_device if bit_lanes else source.append)(raw)
                    if not bit_lanes:
                        index_started = time.perf_counter()
                        source.spatial_chunks.append(
                            build_index(source, raw._mtl, int(raw.shape[0]))
                        )
                        source.load_metrics["index_seconds"] = (
                            source.load_metrics.get("index_seconds", 0.0)
                            + time.perf_counter() - index_started
                        )
                finally:
                    buffer, raw._mtl = raw._mtl, None
                    release_buffer(buffer)
                    del raw
    except BaseException:
        source.release()
        raise
    finally:
        reader.close()
        if decoder is not None:
            decoder.free()
        corrector.close()
    metadata = dict(info.metadata)
    metadata.update(
        resident_metadata(shape, dtype, source.nbytes, backend="mps"),
        source_dtype=stored_dtype.name,
        source_read_passes=1,
        source_logical_tensor_bytes=math.prod(shape) * stored_dtype.itemsize,
        index_bytes=0 if bit_lanes else sum(
            int(buffer.length())
            for chunk in source.spatial_chunks
            for buffer in chunk
        ),
        pixel_mask=info.pixel_mask,
        resident_profile=source.header["codec"] if bit_lanes else INTEGER_CODEC,
        **arrays.correction_metadata(corrector.record),
        load_timings=dict(
            resident.load_metrics,
            resident_ready_seconds=time.perf_counter() - started,
        ),
    )
    if verbose:
        print(_loaded_report(source.shape, "Apple GPU", metadata))
        if narrowed:
            print(f"Stored as {dtype.name}: every {stored_dtype.name} value fits in 16 bits, so every count reads back unchanged.")
    return Dataset4dstemGPU(source, metadata)


def _loaded_report(shape, device_name: str, metadata: dict) -> str:
    """One terse line per loaded scan: size, device, time, bad-pixel handling.

    Report whether flagged values stayed original or were explicitly replaced, and the setting that does the other.
    """
    correction = metadata["hot_pixel_correction"]
    count = int(correction["pixel_count"])
    flagged = f"{count} bad pixel{'s' if count != 1 else ''}"
    if count == 0:
        outcome = "no bad pixels flagged"
    elif correction["method"] == "median":
        outcome = f'{flagged} replaced by neighbor median (hot_pixel_correction="none" keeps the recorded values)'
    elif correction["method"] == "zero":
        outcome = f'{flagged} set to zero (hot_pixel_correction="none" keeps the recorded values)'
    else:
        outcome = f'{flagged} left as recorded (hot_pixel_correction="median" replaces them with the neighbor median)'
    seconds = metadata["load_timings"]["resident_ready_seconds"]
    return (
        f"Loaded {shape[0]} x {shape[1]} scan, {shape[2]} x {shape[3]} detector, "
        f"{device_name}, {seconds:.1f} s. {outcome}."
    )
