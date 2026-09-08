"""Stream complete H5 acquisitions into lossless CUDA count residents."""

import math
import time
from contextlib import ExitStack

import h5py
import numpy as np

from .inspect import inspect
from .models import FourDSTEMData


def load_h5_ans(path, *, scan_shape, dataset_path, device, verbose):
    """Load bounded native chunks; build data-derived indexes without disk caches."""
    import cupy as cp

    from quantem.gpu._compact.streamed import StreamedCounts

    from .load import (
        _decompress_prepared,
        _discover_chunk_names,
        _prepare_master_frames,
        _SparseFrameReadSession,
    )

    started = time.perf_counter()
    info = inspect(path, scan_shape=scan_shape)
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    dtype = np.dtype(info.dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(
            "Compact count loading preserves native uint8/uint16. Use dense loading for other dtypes."
        )
    selected = (
        cp.cuda.Device().id
        if device is None
        else int(str(device).removeprefix("cuda:"))
    )
    valid = np.ones(info.detector_shape, bool)
    if info.pixel_mask is not None:
        valid &= np.asarray(info.pixel_mask) == 0
    with cp.cuda.Device(selected), ExitStack() as stack:
        cp.get_default_memory_pool().free_all_blocks()
        source = StreamedCounts(shape, dtype, valid)
        free, _ = cp.cuda.runtime.memGetInfo()
        # Bound raw + encoding scratch + H5 decompression + output allocation.
        per_scan = math.prod(info.detector_shape) * 8
        chunk_scans = min(2048, max(1, int((free - 128 * 1024**2) // (per_scan * 2))))
        if chunk_scans < min(512, math.prod(info.scan_shape)):
            raise MemoryError(
                "Insufficient CUDA staging space; close another resident document before loading."
            )
        chunk_scans = max(512, chunk_scans // 512 * 512)
        dataset_path = dataset_path or info.metadata.get("dataset_path")
        dataset = None
        session = None
        if dataset_path is not None:
            handle = stack.enter_context(h5py.File(path, "r"))
            dataset = handle[dataset_path]
            if tuple(dataset.shape) != shape or np.dtype(dataset.dtype) != dtype:
                raise ValueError(
                    "The selected H5 dataset must match its complete inspected native geometry and dtype."
                )
        else:
            names = _discover_chunk_names(str(path))
            if not names:
                names = ["data"]
            session = _SparseFrameReadSession(str(path), names, apply_mask=False)
            stack.callback(session.close)
        read_seconds = 0.0
        for first in range(0, math.prod(info.scan_shape), chunk_scans):
            stop = min(first + chunk_scans, math.prod(info.scan_shape))
            before = time.perf_counter()
            if dataset is not None:
                host = np.empty((stop - first, *info.detector_shape), dtype)
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
                raw = _decompress_prepared(
                    prepared,
                    auto_narrow=False,
                    output_dtype=dtype,
                    batch_bytes_target=32 * 1024**2,
                    prune_device_pool=False,
                )
                raw = raw.reshape(stop - first, *info.detector_shape)
            cp.cuda.get_current_stream().synchronize()
            read_seconds += time.perf_counter() - before
            source.append(raw)
            del raw
            if verbose:
                print(f"Encoded {stop}/{math.prod(info.scan_shape)} native scans.")
        metadata = dict(info.metadata)
        metadata.update(
            backend="cuda",
            representation="ans",
            residency="device",
            source_shape=shape,
            working_shape=shape,
            scan_shape=shape[:2],
            detector_shape=shape[2:],
            source_dtype=dtype.name,
            working_dtype=dtype.name,
            dtype=dtype.name,
            n_frames=math.prod(shape[:2]),
            source_logical_tensor_bytes=math.prod(shape) * dtype.itemsize,
            working_logical_tensor_bytes=math.prod(shape) * dtype.itemsize,
            physical_resident_bytes=source.nbytes,
            index_bytes=source.index_nbytes,
            pixel_mask=info.pixel_mask,
            lossless_exact=True,
            file_counts_exact=True,
            resident_profile="runtime-column-rans-spatial-v2",
            detector_mask_policy="preserve-stored-counts",
            scan_bin=1,
            detector_bin=1,
            crop=None,
            load_timings=dict(
                source.load_metrics,
                read_upload_seconds=read_seconds,
                resident_ready_seconds=time.perf_counter() - started,
                max_chunk_scans=chunk_scans,
            ),
        )
        return FourDSTEMData(source, metadata)
