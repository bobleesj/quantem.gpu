"""Stream complete H5 acquisitions into native-count GPU residents."""

import math
import time
from contextlib import ExitStack

import h5py
import numpy as np

from .inspect import inspect
from .models import FourDSTEMData

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
    from ._hdf5_array_resident import load_hdf5_array_resident

    started = time.perf_counter()
    info = inspect(path, scan_shape=scan_shape, dataset_path=dataset_path)
    inspected = time.perf_counter()
    generic = load_hdf5_array_resident(
        path, scan_shape=scan_shape, dataset_path=dataset_path, backend=backend,
        device=device, verbose=verbose, hot_pixel_correction=hot_pixel_correction,
        info=info, auto_narrow=auto_narrow,
    )
    if generic is not None:
        return generic
    if backend == "mps":
        return _load_h5_ans_mps(
            path,
            scan_shape=scan_shape,
            dataset_path=dataset_path,
            verbose=verbose,
            hot_pixel_correction=hot_pixel_correction,
            info=info,
        )
    if backend != "cuda":
        raise NotImplementedError("Runtime HDF5-to-encoded needs CUDA or MPS.")
    import cupy as cp

    from quantem.gpu._compact.streamed import StreamedCounts

    from .load import (
        _decompress_prepared,
        _discover_chunk_names,
        _prepare_master_frames,
        _SparseFrameReadSession,
    )

    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    stored_dtype = np.dtype(info.dtype)
    # Arina writes uint32 files whose counts still fit 16 bits. They are encoded
    # as uint16 only after every stored chunk proves that no count is changed.
    dtype = np.dtype("uint16") if stored_dtype == np.dtype("uint32") else stored_dtype
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(
            "Compact count loading preserves native uint8/uint16 counts, and "
            "uint32 counts that fit uint16. Use dense loading for other dtypes."
        )
    selected = (
        cp.cuda.Device().id
        if device is None
        else int(str(device).removeprefix("cuda:"))
    )
    with cp.cuda.Device(selected), ExitStack() as stack:
        from .backends.cuda.hot_pixels import CUDAHotPixelCorrector

        corrector = CUDAHotPixelCorrector(info.pixel_mask, hot_pixel_correction)
        stack.callback(corrector.close)
        valid = np.ones(info.detector_shape, bool)
        if info.pixel_mask is not None and not corrector.record["applied"]:
            valid &= np.asarray(info.pixel_mask) == 0
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
        session = None
        if dataset_path is not None:
            handle = stack.enter_context(h5py.File(path, "r"))
            dataset = handle[dataset_path]
            if tuple(dataset.shape) != shape or np.dtype(dataset.dtype) != stored_dtype:
                raise ValueError(
                    "The selected H5 dataset must match its complete inspected "
                    "native geometry and dtype."
                )
        else:
            names = _discover_chunk_names(str(path))
            if not names:
                names = ["data"]
            session = _SparseFrameReadSession(str(path), names, apply_mask=False)
            stack.callback(session.close)
        setup_finished = time.perf_counter()
        read_seconds = 0.0
        transfer_decode_seconds = 0.0
        preparation_timings = {}
        for first in range(0, math.prod(info.scan_shape), chunk_scans):
            stop = min(first + chunk_scans, math.prod(info.scan_shape))
            before = time.perf_counter()
            if dataset is not None:
                host = np.empty((stop - first, *info.detector_shape), stored_dtype)
                # A chunk can cut a rectangular scan row. Only storage copying
                # runs on CPU; counts and all scientific reductions stay native.
                cursor = first
                while cursor < stop:
                    row, col = divmod(cursor, shape[1])
                    count = min(shape[1] - col, stop - cursor)
                    host[cursor - first : cursor - first + count] = dataset[
                        row, col : col + count
                    ]
                    cursor += count
                raw = cp.asarray(host)
            else:
                prepared = _prepare_master_frames(
                    str(path),
                    names,
                    np.arange(first, stop),
                    apply_mask=False,
                    read_session=session,
                )
                for key, value in prepared["prepare_timing_s"].items():
                    preparation_timings[key] = preparation_timings.get(key, 0.0) + value
                decode_started = time.perf_counter()
                raw = _decompress_prepared(
                    prepared,
                    auto_narrow=False,
                    output_dtype=stored_dtype,
                    batch_bytes_target=128 * 1024**2,
                    prune_device_pool=False,
                )
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
                raw = _exact_uint16_counts(raw, first, stop)
            corrector.apply(raw)
            source.append(raw)
            del raw
        metadata = dict(info.metadata)
        metadata.update(
            backend="cuda",
            representation="encoded",
            residency="device",
            source_shape=shape,
            working_shape=shape,
            scan_shape=shape[:2],
            detector_shape=shape[2:],
            source_dtype=stored_dtype.name,
            working_dtype=dtype.name,
            dtype=dtype.name,
            n_frames=math.prod(shape[:2]),
            source_read_passes=1,
            source_logical_tensor_bytes=math.prod(shape) * stored_dtype.itemsize,
            working_logical_tensor_bytes=math.prod(shape) * dtype.itemsize,
            physical_resident_bytes=source.nbytes,
            index_bytes=source.index_nbytes,
            pixel_mask=info.pixel_mask,
            lossless_exact=not corrector.record["applied"],
            file_counts_exact=not corrector.record["applied"],
            resident_profile="runtime-column-rans-spatial-v2",
            detector_mask_policy=(
                "gpu-median-corrected"
                if corrector.record["applied"]
                and corrector.record["method"] == "median"
                else "gpu-zero-corrected"
                if corrector.record["applied"]
                else "preserve-stored-counts"
            ),
            hot_pixel_correction=corrector.record,
            working_counts_exact=True,
            scan_bin=1,
            detector_bin=1,
            crop=None,
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
            correction = metadata["hot_pixel_correction"]
            print(
                f"Loaded resident encoded data on CUDA in "
                f"{metadata['load_timings']['resident_ready_seconds']:.2f} s: "
                f"{correction['method']} correction, "
                f"{correction['pixel_count']} stored detector-mask pixels."
            )
        return FourDSTEMData(source, metadata)


def _exact_uint16_counts(raw, first: int, stop: int):
    """Narrow only when every stored value, including flagged pixels, fits."""
    import cupy as cp

    if bool(cp.any(raw > 0xFFFF)):
        raise ValueError(
            f"Frames {first} to {stop - 1} hold counts above 65535, so the uint32 "
            "acquisition cannot be encoded as exact uint16 counts. Keep the original "
            "file; this encoded writer does not support these uint32 values."
        )
    return raw.astype(cp.uint16)


def _load_h5_ans_mps(
    path, *, scan_shape, dataset_path, verbose, hot_pixel_correction="median", info=None
):
    """Decode bounded HDF5 blocks and encode native-count ANS with Metal."""
    from .backends.mps._spatial import build_index
    from .backends.mps._streamed import MPSStreamedCounts
    from .backends.mps.dense import load_prepared_frames, _release_metal_buffer
    from .load import (
        _discover_chunk_names,
        _prepare_master_frames,
        _SparseFrameReadSession,
    )

    started = time.perf_counter()
    if info is None:
        info = inspect(path, scan_shape=scan_shape)
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    dtype = np.dtype(info.dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(
            "Encoded loading preserves native uint8/uint16 detector counts."
        )
    from .backends.mps.hot_pixels import MPSHotPixelCorrector

    corrector = MPSHotPixelCorrector(info.pixel_mask, hot_pixel_correction)
    valid = np.ones(info.detector_shape, bool)
    if info.pixel_mask is not None and not corrector.record["applied"]:
        valid &= np.asarray(info.pixel_mask) == 0
    source = MPSStreamedCounts(shape, dtype, valid)
    names = _discover_chunk_names(str(path)) or ["data"]
    session = _SparseFrameReadSession(str(path), names, apply_mask=False)
    scans = math.prod(info.scan_shape)
    # Bound simultaneous decoded counts and ANS encoding scratch on smaller Macs.
    # Keep batches aligned to the codec interval so exact stored counts are unchanged.
    chunk_scans = min(8192, scans)
    if chunk_scans >= 512:
        chunk_scans = chunk_scans // 512 * 512
    read_decode_seconds = 0.0
    try:
        from concurrent.futures import ThreadPoolExecutor

        ranges = [
            (first, min(first + chunk_scans, scans))
            for first in range(0, scans, chunk_scans)
        ]

        def prepare_frames(bounds):
            first, stop = bounds
            return _prepare_master_frames(
                str(path),
                names,
                np.arange(first, stop),
                apply_mask=False,
                read_session=session,
            )

        # Keep one host-side HDF5 read queued while Metal decodes, corrects,
        # and ANS-encodes the preceding batch. Scientific array work remains
        # on Metal; the worker only prepares compressed file bytes.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(prepare_frames, ranges[0])
            for index, _ in enumerate(ranges):
                before = time.perf_counter()
                prepared = pending.result()
                if index + 1 < len(ranges):
                    pending = pool.submit(prepare_frames, ranges[index + 1])
                raw = load_prepared_frames(
                    prepared,
                    det_bin=1,
                    pixel_mask=None,
                    verbose=False,
                    output_dtype=dtype,
                )
                read_decode_seconds += time.perf_counter() - before
                try:
                    corrector.apply(raw)
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
                    _release_metal_buffer(buffer)
                    del raw
    except BaseException:
        source.release()
        raise
    finally:
        session.close()
        corrector.close()
    metadata = dict(info.metadata)
    metadata.update(
        backend="mps",
        representation="encoded",
        residency="device",
        source_shape=shape,
        working_shape=shape,
        scan_shape=shape[:2],
        detector_shape=shape[2:],
        source_dtype=dtype.name,
        working_dtype=dtype.name,
        dtype=dtype.name,
        n_frames=scans,
        source_read_passes=1,
        source_logical_tensor_bytes=math.prod(shape) * dtype.itemsize,
        working_logical_tensor_bytes=math.prod(shape) * dtype.itemsize,
        physical_resident_bytes=source.nbytes,
        index_bytes=sum(
            int(buffer.length())
            for chunk in source.spatial_chunks
            for buffer in chunk
        ),
        pixel_mask=info.pixel_mask,
        lossless_exact=not corrector.record["applied"],
        file_counts_exact=not corrector.record["applied"],
        resident_profile="runtime-column-rans-spatial-v2",
        detector_mask_policy=(
            "gpu-median-corrected"
            if corrector.record["applied"]
            and corrector.record["method"] == "median"
            else "gpu-zero-corrected"
            if corrector.record["applied"]
            else "preserve-stored-counts"
        ),
        hot_pixel_correction=corrector.record,
        working_counts_exact=True,
        scan_bin=1,
        detector_bin=1,
        crop=None,
        load_timings=dict(
            source.load_metrics,
            read_upload_seconds=read_decode_seconds,
            resident_ready_seconds=time.perf_counter() - started,
            max_chunk_scans=chunk_scans,
        ),
    )
    if verbose:
        correction = metadata["hot_pixel_correction"]
        print(
            f"Loaded resident encoded data on MPS in "
            f"{metadata['load_timings']['resident_ready_seconds']:.2f} s: "
            f"{correction['method']} correction, "
            f"{correction['pixel_count']} stored detector-mask pixels."
        )
    return FourDSTEMData(source, metadata)
