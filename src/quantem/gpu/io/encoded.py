"""Stream complete HDF5 acquisitions into native-count GPU residents.

A bitshuffle+LZ4 master is read in bounded blocks of scan positions: the
selected compressed chunks are read on the host, decoded on the GPU, stored
detector-mask pixels are corrected, and the counts are ANS encoded, so the
dense acquisition never exists. Other HDF5 layouts go through
``io.arrays.load_hdf5_array_resident``.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import h5py
import numpy as np

from quantem.gpu.device.cuda_runtime import cp, cuda_device_index
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
    backend="cuda",
    hot_pixel_correction="median",
    auto_narrow=True,
):
    """Load bounded native chunks into a backend-owned runtime encoded resident."""
    started = time.perf_counter()
    info = inspect(path, scan_shape=scan_shape, dataset_path=dataset_path)
    # workflows over several acquisitions (MAPED tilts, focal series) label each
    # one by the file it came from; every branch below copies info.metadata
    info.metadata["source_path"] = str(path)
    inspected = time.perf_counter()
    generic = arrays.load_hdf5_array_resident(
        path, dataset_path=dataset_path, backend=backend,
        device=device, verbose=verbose, hot_pixel_correction=hot_pixel_correction,
        info=info, auto_narrow=auto_narrow,
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
    # Arina writes uint32 files whose counts still fit 16 bits. They are encoded
    # as uint16 only after every stored chunk proves that no valid count is changed.
    dtype = np.dtype("uint16") if stored_dtype == np.dtype("uint32") else stored_dtype
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(
            "Compact count loading preserves native uint8/uint16 counts, and "
            "uint32 counts that fit uint16. Use dense loading for other dtypes."
        )
    selected = cuda_device_index(device)
    with cp.cuda.Device(selected), ExitStack() as stack:
        # Runs last: hand this load's read, decode and encode scratch back once
        # the reader and corrector are closed. The encoded chunks are not pooled.
        stack.callback(cp.get_default_memory_pool().free_all_blocks)
        corrector = CUDAHotPixelCorrector(info.pixel_mask, hot_pixel_correction)
        stack.callback(corrector.close)
        valid = np.ones(info.detector_shape, bool)
        if info.pixel_mask is not None and not corrector.record["applied"]:
            valid &= np.asarray(info.pixel_mask) == 0
        flagged = cp.asarray(np.flatnonzero(~valid), dtype=cp.int32)
        zeroed_markers = 0
        cp.get_default_memory_pool().free_all_blocks()
        source = StreamedCounts(shape, dtype, valid)
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
        chunk_scans = max(512, chunk_scans // 512 * 512)
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
        setup_finished = time.perf_counter()
        read_seconds = 0.0
        transfer_decode_seconds = 0.0
        preparation_timings = {}
        for first in range(0, math.prod(info.scan_shape), chunk_scans):
            stop = min(first + chunk_scans, math.prod(info.scan_shape))
            before = time.perf_counter()
            if dataset is not None:
                # A chunk can cut a rectangular scan row. Only storage copying
                # runs on CPU; counts and all scientific reductions stay native.
                raw = cp.asarray(arrays.read_frame_block(dataset, shape, first, stop))
            else:
                prepared = reader.prepare(np.arange(first, stop))
                for key, value in prepared["prepare_timing_s"].items():
                    preparation_timings[key] = preparation_timings.get(key, 0.0) + value
                decode_started = time.perf_counter()
                raw = decompress_prepared(prepared, batch_bytes_target=128 * 1024**2)
                transfer_decode_seconds += time.perf_counter() - decode_started
                raw = raw.reshape(stop - first, *info.detector_shape)
            cp.cuda.get_current_stream().synchronize()
            read_seconds += time.perf_counter() - before
            if stored_dtype != dtype:
                if corrector.record["applied"]:
                    # Flagged uint32 pixels can contain 0xffffffff sentinels.
                    # Their requested correction replaces them anyway; clear
                    # only those pixels before validating every retained count.
                    raw.reshape(raw.shape[0], -1)[:, corrector.bad] = 0
                elif flagged.size:
                    # Kept flagged pixels hold the detector's 0xffffffff marker,
                    # which no uint16 can hold. pixel_mask keeps their positions
                    # and reads exclude them, so a flagged value above 65535 is
                    # stored as 0 and counted; every valid count must still fit.
                    pixels = raw.reshape(raw.shape[0], -1)
                    kept = pixels[:, flagged]
                    markers = kept > 0xFFFF
                    zeroed_markers += int(cp.count_nonzero(markers))
                    pixels[:, flagged] = cp.where(markers, 0, kept)
                if bool(cp.any(raw > 0xFFFF)):
                    raise ValueError(
                        f"Frames {first} to {stop - 1} hold counts above 65535 at unflagged pixels, "
                        "so the uint32 acquisition cannot be encoded as exact uint16 counts. Keep "
                        "the original file; this encoded writer does not support these uint32 values."
                    )
                raw = raw.astype(cp.uint16)
            corrector.apply(raw)
            source.append(raw)
            del raw
        metadata = dict(info.metadata)
        metadata.update(
            resident_metadata(shape, dtype, source.nbytes, backend="cuda"),
            source_dtype=stored_dtype.name,
            source_read_passes=1,
            source_logical_tensor_bytes=math.prod(shape) * stored_dtype.itemsize,
            index_bytes=source.index_nbytes,
            pixel_mask=info.pixel_mask,
            resident_profile=INTEGER_CODEC,
            **_correction_metadata(corrector.record, zeroed_markers),
            load_timings=dict(
                source.load_metrics,
                read_upload_seconds=read_seconds,
                resident_ready_seconds=time.perf_counter() - started,
                max_chunk_scans=chunk_scans,
                inspect_seconds=inspected - started,
                setup_seconds=setup_finished - inspected,
                preparation_seconds=preparation_timings,
                transfer_decode_seconds=transfer_decode_seconds,
            ),
        )
        if verbose:
            print(_loaded_report(source.shape, "NVIDIA GPU", metadata))
        return Dataset4dstemGPU(source, metadata)


def _load_h5_ans_mps(path, info, hot_pixel_correction, verbose):
    """Decode bounded HDF5 blocks and encode native-count ANS with Metal."""
    # Metal decoding is imported only on the MPS path, like every platform decoder.
    from quantem.gpu.io.hdf5.mps.decode import exact_uint16, load_prepared_frames

    started = time.perf_counter()
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    stored_dtype = np.dtype(info.dtype)
    # As on CUDA, Arina uint32 counts are stored as uint16 only after every
    # decoded batch proves that no valid count changes.
    dtype = np.dtype("uint16") if stored_dtype == np.dtype("uint32") else stored_dtype
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(
            "Encoded loading preserves native uint8/uint16 detector counts, and "
            "uint32 counts that fit uint16."
        )
    corrector = MPSHotPixelCorrector(info.pixel_mask, hot_pixel_correction)
    valid = np.ones(info.detector_shape, bool)
    if info.pixel_mask is not None and not corrector.record["applied"]:
        valid &= np.asarray(info.pixel_mask) == 0
    source = MPSStreamedCounts(shape, dtype, valid)
    zeroed_markers = 0
    reader = FrameReader(str(path))
    scans = math.prod(info.scan_shape)
    # Bound simultaneous decoded counts and ANS encoding scratch on smaller Macs;
    # uint32 batches hold half the scans, so their decoded bytes stay the same.
    # Keep batches aligned to the codec interval so exact stored counts are unchanged.
    chunk_scans = min(8192 if stored_dtype.itemsize <= 2 else 4096, scans)
    if chunk_scans >= 512:
        chunk_scans = chunk_scans // 512 * 512
    read_decode_seconds = 0.0
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
                before = time.perf_counter()
                prepared = pending.result()
                if index + 1 < len(ranges):
                    pending = pool.submit(reader.prepare, np.arange(*ranges[index + 1]))
                raw = load_prepared_frames(prepared)
                read_decode_seconds += time.perf_counter() - before
                try:
                    # The median reads only valid neighbors, so flagged uint32
                    # sentinels are replaced before the range check, as on CUDA.
                    corrector.apply(raw)
                    if stored_dtype != dtype:
                        narrowed, zeroed = exact_uint16(raw, ranges[index][0], valid)
                        zeroed_markers += zeroed
                        buffer, raw._mtl = raw._mtl, None
                        release_buffer(buffer)
                        raw = narrowed
                    source.append(raw)
                    # Pack the exact spatial index from the same staging window so
                    # detector products and .qem saving stay available after the
                    # counts are encoded; the buffer handle is consumed in-place.
                    index_started = time.perf_counter()
                    source.spatial_chunks.append(
                        build_index(source, raw._mtl, int(raw.shape[0]))
                    )
                    source.load_metrics["index_seconds"] = (
                        source.load_metrics.get("index_seconds", 0.0)
                        + time.perf_counter()
                        - index_started
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
        corrector.close()
    metadata = dict(info.metadata)
    metadata.update(
        resident_metadata(shape, dtype, source.nbytes, backend="mps"),
        source_dtype=stored_dtype.name,
        source_read_passes=1,
        source_logical_tensor_bytes=math.prod(shape) * stored_dtype.itemsize,
        index_bytes=sum(
            int(buffer.length())
            for chunk in source.spatial_chunks
            for buffer in chunk
        ),
        pixel_mask=info.pixel_mask,
        resident_profile=INTEGER_CODEC,
        **_correction_metadata(corrector.record, zeroed_markers),
        load_timings=dict(
            source.load_metrics,
            read_upload_seconds=read_decode_seconds,
            resident_ready_seconds=time.perf_counter() - started,
            max_chunk_scans=chunk_scans,
        ),
    )
    if verbose:
        print(_loaded_report(source.shape, "Apple GPU", metadata))
    return Dataset4dstemGPU(source, metadata)


def _correction_metadata(record: dict, zeroed_markers: int = 0) -> dict:
    """Record how stored detector-mask pixels were handled and what that means for exactness.

    A corrected pixel no longer holds the file's count, so the working data is
    exact (its own codes decode bit for bit) but no longer the file's counts.
    ``zeroed_markers`` counts flagged-pixel values above 65535 (the uint32
    detector marker) stored as 0 when uncorrected uint32 counts become uint16:
    every valid count and every other flagged value is still the file's, and
    ``pixel_mask`` keeps the flagged positions.
    """
    applied = record["applied"]
    if applied:
        policy = f"gpu-{record['method']}-corrected"
    elif zeroed_markers:
        policy = "flagged-markers-stored-as-zero"
    else:
        policy = "preserve-stored-counts"
    exact = not applied and not zeroed_markers
    return dict(
        lossless_exact=exact,
        file_counts_exact=exact,
        detector_mask_policy=policy,
        hot_pixel_correction=record,
        working_counts_exact=True,
        flagged_markers_stored_as_zero=int(zeroed_markers),
    )


def _loaded_report(shape, device_name: str, metadata: dict) -> str:
    """One terse line per loaded scan: size, device, time, bad-pixel handling.

    The loader replaces the detector's flagged bad pixels before compressing the
    counts, which changes the working data; the reader must know how many and by
    which rule, without codec or mask vocabulary.
    """
    correction = metadata["hot_pixel_correction"]
    count = int(correction["pixel_count"])
    flagged = f"{count} bad pixel{'s' if count != 1 else ''}"
    if count == 0:
        outcome = "no bad pixels flagged"
    elif correction["method"] == "median":
        outcome = f"{flagged} replaced by neighbor median"
    elif correction["method"] == "zero":
        outcome = f"{flagged} set to zero"
    elif metadata.get("flagged_markers_stored_as_zero"):
        outcome = f"{flagged} kept in the mask, their uint32 marker stored as 0"
    else:
        outcome = f"{flagged} left as recorded"
    seconds = metadata["load_timings"]["resident_ready_seconds"]
    return (
        f"Loaded {shape[0]} x {shape[1]} scan, {shape[2]} x {shape[3]} detector, "
        f"{device_name}, {seconds:.1f} s. {outcome}."
    )
