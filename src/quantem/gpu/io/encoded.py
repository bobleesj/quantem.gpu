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
from quantem.gpu.device.metal_runtime import release_buffer, shared_array
from quantem.gpu.formats.hdf5.reads import FrameReader
from quantem.gpu.formats.hdf5.master import HOT_PIXEL_CORRECTION_ATTRIBUTE
from quantem.gpu.formats.qem.snapshot import INTEGER_CODEC
from quantem.gpu.io import arrays
from quantem.gpu.io.dataset import Dataset4dstem, resident_metadata
from quantem.gpu.io.hdf5.cuda.decode import decompress_prepared
from quantem.gpu.io.inspect import inspect
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.cuda.hot_pixels import CUDAHotPixelCorrector
from quantem.gpu.resident.hot_pixels import hot_pixel_outcome
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.hot_pixels import MPSHotPixelCorrector
from quantem.gpu.resident.mps.spatial import build_index

_CUDA_MAX_STAGING_SCANS = 16384
_CUDA_MIN_STAGING_SCANS = 64          # the smallest decode block (a small GPU), positions


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
    detector_bin: int = 1,
    detector_transpose: bool = False,
    progress: bool = False,
):
    """Load bounded native chunks into a backend-owned runtime encoded resident.

    ``detector_transpose`` (CUDA): each decoded block, corrected and binned, has its frames transposed on the GPU just
    before it is encoded (``arrays.transposed_frames``, one tiled pass in device memory); the record is the stored one
    with every detector axis swapped.

    ``detector_bin`` (2, 4 or 8, CUDA, 8- and 16-bit ARINA masters): each decoded block has its flagged pixels
    corrected, is summed ``detector_bin`` x ``detector_bin`` on the GPU (``bin``: every electron kept) and only then
    encoded, so the resident is about ``detector_bin**2`` times smaller (a GPU too small for the full scan, 11-17).
    A binned sum that does not fit the stored dtype raises; nothing is clipped."""
    started = time.perf_counter()
    info = inspect(path, scan_shape=scan_shape, dataset_path=dataset_path)
    saved_correction = info.metadata.get(HOT_PIXEL_CORRECTION_ATTRIBUTE)
    if saved_correction is not None:
        # Saved measurements already have their recorded correction policy.
        hot_pixel_correction = "none"
    # workflows over several acquisitions (MAPED tilts, focal series) label each
    # one by the file it came from; every branch below copies info.metadata
    info.metadata["source_path"] = str(path)
    if detector_transpose:
        arrays.refuse_second_transpose(info.metadata)
    generic = arrays.load_hdf5_array_resident(
        path, dataset_path=dataset_path, backend=backend,
        device=device, verbose=verbose, hot_pixel_correction=hot_pixel_correction,
        info=info, detector_transpose=detector_transpose,
    )
    if generic is not None and detector_bin != 1:
        generic.close()
        raise NotImplementedError("detector_bin applies to ARINA masters (direct chunk decoding); this layout "
                                  "loads at its native detector size.")
    if generic is not None:
        if saved_correction is not None:
            generic.metadata.update(arrays.correction_metadata(
                arrays.transposed_record({"hot_pixel_correction": saved_correction})["hot_pixel_correction"]
                if detector_transpose else saved_correction))
        return generic
    if backend == "mps":
        if detector_bin != 1:
            raise NotImplementedError("detector_bin needs CUDA.")
        return _load_h5_ans_mps(path, info, hot_pixel_correction, verbose, detector_transpose)
    if backend != "cuda":
        raise NotImplementedError("Runtime HDF5-to-encoded needs CUDA or MPS.")

    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    stored_dtype = np.dtype(info.dtype)
    dtype = stored_dtype
    bit_lanes = dtype == np.dtype("uint32")
    factor = int(detector_bin)
    if factor not in (1, 2, 4, 8):
        raise ValueError(f"detector_bin is 1, 2, 4 or 8; got {detector_bin!r}.")
    if factor != 1 and bit_lanes:
        raise NotImplementedError("detector_bin applies to 8- and 16-bit counts; this master stores 32-bit counts.")
    binned = (*info.scan_shape, info.detector_shape[0] // factor, info.detector_shape[1] // factor)
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
        unfilled = None
        if factor != 1:
            # a flagged pixel not filled by the correction is zeroed before the sum and its binned pixel is invalid:
            # its marker (65535) never reaches a neighbour's count (11-17)
            if not corrector.record["applied"] and not valid.all():
                unfilled = cp.asarray(~valid)
            rows, cols = binned[2] * factor, binned[3] * factor
            valid = valid[:rows, :cols].reshape(binned[2], factor, binned[3], factor).all(axis=(1, 3))
        # the resident's geometry and validity mask, transposed after the correction and the bin
        stored = (*binned[:2], binned[3], binned[2]) if detector_transpose else binned
        if detector_transpose:
            valid = np.ascontiguousarray(valid.T)
        source = (arrays.bit_resident(stored, dtype,
                                      arrays.transposed_record(info.metadata) if detector_transpose else info.metadata,
                                      valid, "cuda", selected)
                  if bit_lanes else StreamedCounts(stored, dtype, valid))
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
        # A small GPU (the ARINA PC's 4 GB A400) decodes in blocks of 64 positions: 0.15 GB of staging instead of
        # 1.2 GB, the same counts (11-17); larger GPUs keep blocks of 512 or more.
        if chunk_scans < min(_CUDA_MIN_STAGING_SCANS, math.prod(info.scan_shape)):
            raise MemoryError(
                f"Not enough free GPU memory to decode this scan ({free / 1e9:.2f} GB free): close or release the "
                "scans loaded before, then load it again."
            )
        chunk_scans = (min(512, arrays.MAX_INGEST_BYTES // (math.prod(shape[2:]) * dtype.itemsize))
                       if bit_lanes else (chunk_scans // 512 * 512 if chunk_scans >= 512
                                          else chunk_scans // _CUDA_MIN_STAGING_SCANS * _CUDA_MIN_STAGING_SCANS))
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
        bar = None
        if progress:                                # a bar over the scan positions decoded and encoded (11-17)
            from tqdm.auto import tqdm

            bar = stack.enter_context(tqdm(total=scans, desc=f"Loading {path.name}", unit=" positions",
                                           unit_scale=True, leave=False, mininterval=0.3))
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
            if factor != 1:
                from quantem.gpu.io.load import bin as bin_blocks

                if unfilled is not None:
                    raw[:, unfilled] = 0
                summed = bin_blocks(raw, factor=factor, axes="detector")          # exact, in 64-bit integers
                largest = int(summed.max()) if summed.size else 0
                if largest > np.iinfo(dtype).max:
                    raise ValueError(f"A {factor} x {factor} detector sum reaches {largest}, beyond {dtype.name}; "
                                     "load with a smaller detector_bin.")
                del raw
                raw = summed.astype(dtype)
                del summed
            if detector_transpose:
                raw = arrays.transposed_frames(raw)
            (source.append_device if bit_lanes else source.append)(raw)
            del raw
            if bar is not None:
                bar.update(stop - first)
        if not bit_lanes and source.host_nbytes:
            print(f"quantem.gpu: the device had no room for {source.host_nbytes / 1e9:.2f} GB of this scan's encoded "
                  "counts; they stay in page-locked host memory and the GPU reads them there (a whole-scan "
                  "reconstruction about 5x slower, a new virtual image about 50x).")
        metadata = dict(info.metadata)
        metadata.update(
            resident_metadata(binned if not bit_lanes else shape, dtype, source.nbytes, backend="cuda"),
            source_dtype=stored_dtype.name,
            source_read_passes=1,
            source_logical_tensor_bytes=math.prod(shape) * stored_dtype.itemsize,
            index_bytes=0 if bit_lanes else source.index_nbytes,
            pixel_mask=info.pixel_mask,
            resident_profile=source.header["codec"] if bit_lanes else INTEGER_CODEC,
            **arrays.correction_metadata(saved_correction or corrector.record),
            load_timings=dict(
                resident.load_metrics,
                resident_ready_seconds=time.perf_counter() - started,
            ),
        )
        if factor != 1:                          # after the resident's own fields, which say detector_bin=1
            metadata.update(_binned_record(info, factor, binned))
        if detector_transpose:                   # last: the stored record, every detector axis swapped
            metadata = arrays.transposed_record(metadata)
        if verbose:
            print(_loaded_report(source.shape, "NVIDIA GPU", metadata))
        return Dataset4dstem(source, metadata)


class _ExceedsUint16(Exception):
    """A decoded uint32 value does not fit 16 bits, so the scan keeps uint32 storage."""


def _binned_record(info, factor: int, binned) -> dict:
    """What a binned load changes in the record: the detector's shape and sampling (``factor`` times coarser), the
    binned mask (a binned pixel is flagged when any of its pixels is), and the bin itself."""
    record = dict(detector_bin=factor, detector_shape=list(binned[2:]), binned_from_detector_shape=list(info.detector_shape),
                  image_size_x=binned[3], image_size_y=binned[2])
    for key in ("detector_sampling_mrad", "detector_sampling_inv_A"):
        value = info.metadata.get(key)
        if value is not None:
            try:
                record[key] = [float(item) * factor for item in np.atleast_1d(np.asarray(value, dtype=float))]
            except (TypeError, ValueError):
                pass
    if info.pixel_mask is not None:
        mask = np.asarray(info.pixel_mask)[:binned[2] * factor, :binned[3] * factor]
        record["pixel_mask"] = mask.reshape(binned[2], factor, binned[3], factor).max(axis=(1, 3))
    disk = info.metadata.get("bright_field_disk")
    if isinstance(disk, str):
        import json

        try:
            disk = json.loads(disk)
        except json.JSONDecodeError:
            disk = None
    if isinstance(disk, dict):
        # the fitted disk in binned pixels: the sampling is alpha over its radius, so a radius left in unbinned pixels
        # gave the unbinned 0.577 mrad again (11-17); a pixel centre c becomes (c + 0.5) / factor - 0.5
        disk = dict(disk)
        if isinstance(disk.get("bright_field_edge_radius_px"), (int, float)):
            disk["bright_field_edge_radius_px"] = float(disk["bright_field_edge_radius_px"]) / factor
        if isinstance(disk.get("bright_field_center_px"), (list, tuple)):
            disk["bright_field_center_px"] = [(float(value) + 0.5) / factor - 0.5 for value in disk["bright_field_center_px"]]
        record["bright_field_disk"] = disk
    scientific = info.metadata.get("scientific_metadata")
    if isinstance(scientific, str):
        import json

        try:
            scientific = json.loads(scientific)
        except json.JSONDecodeError:
            scientific = None
    if isinstance(scientific, dict) and isinstance(scientific.get("axes"), list):
        # the detector axes as binned: half the size, twice the step (the record must match the scan, 11-17)
        import copy

        scientific = copy.deepcopy(scientific)
        for axis in scientific["axes"]:
            if isinstance(axis, dict) and str(axis.get("name", "")).startswith("det"):
                axis["size"] = int(axis.get("size", 0)) // factor
                sampling = axis.get("sampling")
                if isinstance(sampling, dict) and isinstance(sampling.get("value"), (int, float)):
                    sampling["value"] = float(sampling["value"]) * factor
        # and the detector step recorded as a calibration (the disk fit at the receiver): it wins over the axes
        from quantem.gpu.formats.qem.metadata import DETECTOR_PIXEL_SIZE

        for path, quantity in (scientific.get("calibration_overrides") or {}).items():
            if path in DETECTOR_PIXEL_SIZE and isinstance(quantity, dict) and isinstance(quantity.get("value"), (int, float)):
                quantity["value"] = float(quantity["value"]) * factor
        record["scientific_metadata"] = scientific
    return record


def _load_h5_ans_mps(path, info, hot_pixel_correction, verbose, detector_transpose=False):
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
            return _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, np.dtype("uint16"),
                                       detector_transpose)
        except _ExceedsUint16:
            pass
    return _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, np.dtype(info.dtype), detector_transpose)


def _load_h5_counts_mps(path, info, hot_pixel_correction, verbose, dtype, detector_transpose=False):
    """Load the master into a resident of ``dtype`` (the stored dtype, or uint16 for uint32 counts).

    uint16 counts whose every value is at most 255 end as a uint8 resident. ``detector_transpose``: each decoded batch,
    corrected and narrowed, is transposed on the GPU (``arrays.transposed_frames``) before it is encoded and indexed.
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
    stored = (*shape[:2], shape[3], shape[2]) if detector_transpose else shape
    if detector_transpose:
        valid = np.ascontiguousarray(valid.T)
    source = (arrays.bit_resident(stored, dtype,
                                  arrays.transposed_record(info.metadata) if detector_transpose else info.metadata,
                                  valid, "mps")
              if bit_lanes else MPSStreamedCounts(stored, dtype, valid))
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
                swapped = None
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
                        frames = raw
                        if detector_transpose:
                            swapped = arrays.transposed_frames(raw)
                            frames = shared_array(swapped._mtl, swapped.dtype, swapped.shape)
                        (source.append_device if bit_lanes else source.append)(frames)
                        if not bit_lanes:
                            index_started = time.perf_counter()
                            source.spatial_chunks.append(
                                build_index(source, frames._mtl, int(frames.shape[0]))
                            )
                            source.load_metrics["index_seconds"] = (
                                source.load_metrics.get("index_seconds", 0.0)
                                + time.perf_counter() - index_started
                            )
                    finally:
                        if swapped is not None:
                            swapped.release()
                            swapped = None
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
        **arrays.correction_metadata(info.metadata.get(HOT_PIXEL_CORRECTION_ATTRIBUTE) or corrector.record),
        load_timings=dict(
            resident.load_metrics,
            resident_ready_seconds=time.perf_counter() - started,
        ),
    )
    if detector_transpose:                       # last: the stored record, every detector axis swapped
        metadata = arrays.transposed_record(metadata)
    if verbose:
        print(_loaded_report(source.shape, "Apple GPU", metadata))
        if dtype != stored_dtype:
            print(f"Stored as {dtype.name}: every {stored_dtype.name} value fits in {8 * dtype.itemsize} bits, so every "
                  f'count reads back unchanged (dtype="native" reads {stored_dtype.name}).')
    return Dataset4dstem(source, metadata)


def _loaded_report(shape, device_name: str, metadata: dict) -> str:
    """Loaded scan geometry, device, time, ANS storage and bad-pixel handling.

    Report whether flagged values stayed original or were explicitly replaced, and the setting that does the other.
    """
    outcome = hot_pixel_outcome(metadata["hot_pixel_correction"])
    seconds = metadata["load_timings"]["resident_ready_seconds"]
    return (
        f"Loaded {shape[0]} x {shape[1]} scan, {shape[2]} x {shape[3]} detector, "
        f"{device_name}, {seconds:.1f} s. "
        f"ANS: {metadata['working_logical_tensor_bytes'] / 1e9:.2f} GB uncompressed "
        f"to {metadata['physical_resident_bytes'] / 1e9:.2f} GB resident on GPU. {outcome}."
        + (" Detector rows and columns swapped (detector_transpose)." if metadata.get("detector_transpose") else "")
    )
