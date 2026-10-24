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
from queue import Empty, SimpleQueue

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
from quantem.gpu.resident.hot_pixels import hot_pixel_outcome
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.hot_pixels import MPSHotPixelCorrector
from quantem.gpu.resident.mps.spatial import build_index

_CUDA_MAX_STAGING_SCANS = 16384


def _prepare_cuda_loader() -> None:
    """Initialize reusable loader code before an acquisition arrives.

    No source is opened and no detector counts are allocated. Empty header
    arrays select the same compiled parser as bounded reads; the existing
    codec modules and small shared coding tables retain their normal caches.
    The caller must select its device before preparing the loader.
    """
    from quantem.gpu.formats.hdf5.frames import parse_headers
    from quantem.gpu.io.hdf5.cuda.decode import kernel
    from quantem.gpu.resident.cuda.counts import kernels, tables

    empty_bytes = np.empty(0, dtype=np.uint8)
    empty_words = np.empty(0, dtype=np.uint32)
    empty_offsets = np.empty(0, dtype=np.uint64)
    parse_headers(
        empty_bytes, empty_words, empty_offsets, empty_words, empty_words, 0, 1,
    )
    kernel("h5lz4dc_batched")
    selected = cp.cuda.Device().id
    kernels(selected)
    tables(selected)
    cp.cuda.get_current_stream().synchronize()


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
        # Two bounded compressed blocks read ahead while the GPU consumes
        # the current block, so reading can also overlap header preparation.
        # Only the current block is decoded on the GPU.
        read_ahead = stack.enter_context(ThreadPoolExecutor(max_workers=2))

        def prepare(block):
            # The staging buffer is page-locked in this GPU's context, which a worker thread enters itself.
            with cp.cuda.Device(selected):
                return reader.prepare(np.arange(*block))

        def release_unread_block():
            # Unconsumed prepared blocks have never been copied to the GPU.
            for future in pending.values():
                try:
                    prepared = future.result()
                except BaseException:  # noqa: BLE001, S110
                    pass  # Preserve the foreground failure.
                else:
                    _release_pinned(prepared["read_buffer"])

        pending = {} if reader is None else {
            number: read_ahead.submit(prepare, blocks[number])
            for number in range(min(2, len(blocks)))
        }
        stack.callback(release_unread_block)
        for number, (first, stop) in enumerate(blocks):
            if dataset is not None:
                # A chunk can cut a rectangular scan row. h5py reads and decodes
                # the stored chunks on the host for this generic layout.
                raw = cp.asarray(arrays.read_frame_block(dataset, shape, first, stop))
            else:
                prepared = pending.pop(number).result()
                following = number + 2
                if following < len(blocks):
                    pending[following] = read_ahead.submit(prepare, blocks[following])
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

    Counts are stored in the narrowest unsigned type that holds every value
    after any requested bad-pixel correction: uint8 when every value is at
    most 255, else uint16, else uint32 bit lanes (a single value above 65535,
    such as a kept 0xffffffff bad-pixel marker, keeps the whole scan uint32).
    Every value reads back unchanged; the uint8 and uint16 count residents use
    the spatial index for detector sums.
    """
    if np.dtype(info.dtype) == np.dtype("uint32"):
        try:
            return _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, np.dtype("uint16"))
        except _ExceedsUint16:
            pass
    return _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, np.dtype(info.dtype))


def _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, dtype):
    """Load the master into a resident of ``dtype`` (the stored dtype, or uint16 for uint32 counts).

    uint16 counts whose every value is at most 255 end as a uint8 resident.
    """
    # Metal decoding is imported only on the MPS path, like every platform decoder.
    from quantem.gpu.io.hdf5.mps.decode import (
        MPSDecompressor,
        exact_uint16,
    )

    started = time.perf_counter()
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    stored_dtype = np.dtype(info.dtype)
    bit_lanes = dtype == np.dtype("uint32")
    narrowed = dtype != stored_dtype
    # uint16 counts whose every value fits a byte are stored as uint8 once every batch is checked.
    byte_check, above_byte = dtype == np.dtype("uint16"), False
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
    decoded_buffers = SimpleQueue()
    scans = math.prod(info.scan_shape)
    # Bound simultaneous decoded counts and ANS encoding scratch on smaller Macs;
    # uint32 batches hold half the scans, so their decoded bytes stay the same.
    # Keep batches aligned to the codec interval so exact stored counts are unchanged.
    chunk_scans = min(8192 if stored_dtype.itemsize <= 2 else 4096, scans)
    if bit_lanes:
        chunk_scans = min(512, arrays.MAX_INGEST_BYTES // (math.prod(shape[2:]) * dtype.itemsize))
    elif chunk_scans >= 512:
        chunk_scans = chunk_scans // 512 * 512
    chunk_scans = resident._encoding_scan_limit(chunk_scans)
    staging_bytes = chunk_scans * math.prod(shape[2:]) * stored_dtype.itemsize

    def recycle_frames(raw):
        """Return an idle full staging buffer; release narrowed or short outputs."""
        buffer, raw._mtl = raw._mtl, None
        if int(buffer.length()) >= staging_bytes:
            decoded_buffers.put(buffer)
        else:
            release_buffer(buffer)

    try:
        ranges = [
            (first, min(first + chunk_scans, scans))
            for first in range(0, scans, chunk_scans)
        ]
        # One decoded batch is prepared ahead while the caller encodes and
        # indexes the preceding batch. Only this worker owns decoder scratch.
        def decode_next(bounds):
            """Prepare one batch on the host and decode into idle Metal staging."""
            nonlocal decoder
            prepared = reader.prepare(np.arange(*bounds))
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
            try:
                output = decoded_buffers.get_nowait()
            except Empty:
                output = None
            try:
                return decoder.load_prepared_frames(prepared, out_buffer=output)
            except BaseException:
                release_buffer(output)
                raise

        pending = None
        with ExitStack() as workspace, ThreadPoolExecutor(max_workers=1) as pool:
            if not bit_lanes:
                workspace.enter_context(source.encoding_workspace(chunk_scans))
            try:
                pending = pool.submit(decode_next, ranges[0])
                for index in range(len(ranges)):
                    raw = pending.result()
                    pending = None
                    try:
                        if index + 1 < len(ranges):
                            pending = pool.submit(decode_next, ranges[index + 1])
                        corrector.apply(raw)
                        if narrowed:
                            counts, byte = exact_uint16(raw)
                            if counts is None:
                                raise _ExceedsUint16
                            recycle_frames(raw)
                            raw = counts
                            above_byte |= byte
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
                        recycle_frames(raw)
                        del raw
            finally:
                # A failed encode/correction must also return an unread prefetched
                # output before decoder scratch and the reader are torn down.
                if pending is not None:
                    try:
                        unused = pending.result()
                    except BaseException:  # noqa: BLE001, S110 - keep the active ingest failure.
                        pass
                    else:
                        buffer, unused._mtl = unused._mtl, None
                        release_buffer(buffer)
    except BaseException:
        source.release()
        raise
    finally:
        reader.close()
        while not decoded_buffers.empty():
            release_buffer(decoded_buffers.get_nowait())
        if decoder is not None:
            decoder.free()
        corrector.close()
    if byte_check and not (above_byte or source._encoding_exceeds_uint8):
        source.store_as_uint8()
        dtype = source.dtype
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
        if dtype != stored_dtype:
            print(f"Stored as {dtype.name}: every {stored_dtype.name} value fits in {8 * dtype.itemsize} bits, so every "
                  f'count reads back unchanged (dtype="native" reads {stored_dtype.name}).')
    return Dataset4dstemGPU(source, metadata)


def _loaded_report(shape, device_name: str, metadata: dict) -> str:
    """One terse line per loaded scan: size, device, time, bad-pixel handling.

    Report whether flagged values stayed original or were explicitly replaced, and the setting that does the other.
    """
    outcome = hot_pixel_outcome(metadata["hot_pixel_correction"])
    seconds = metadata["load_timings"]["resident_ready_seconds"]
    return (
        f"Loaded {shape[0]} x {shape[1]} scan, {shape[2]} x {shape[3]} detector, "
        f"{device_name}, {seconds:.1f} s. {outcome}."
    )
