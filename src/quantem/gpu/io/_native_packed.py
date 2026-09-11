"""Two-pass bounded native packing through existing GPU kernels."""

import math
from contextlib import ExitStack
from pathlib import Path

import cupy as cp
import h5py
import numpy as np

from .inspect import inspect
from .models import FourDSTEMData
from quantem.gpu.io.backends.cuda._ans import CudaPackedResidentCounts, _kernels
from quantem.gpu.io.load import (
    _decompress_prepared,
    _discover_chunk_names,
    _prepare_master_frames,
    _SparseFrameReadSession,
    _file_source_signature,
)


def load_h5_packed(
    path: str | Path,
    *,
    scan_shape: tuple[int, int] | None,
    dataset_path: str | None,
    device: int | str | None,
    verbose: bool,
) -> FourDSTEMData:
    """Measure widths, then write exact packed words using bounded input buffers."""
    path = Path(path)
    # Amortize HDF5 preparation and CUDA decompressor launches while keeping
    # the raw staging buffer bounded for the 24 GiB laptop workflow.
    chunk_scans = 8192
    batch_bytes_target = 1024 * 1024**2
    info = inspect(path, scan_shape=scan_shape)
    if not info.ready:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    dtype = np.dtype(info.dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(
            f"Packed HDF5 loading requires native uint8/uint16; got {dtype}. "
            "Request representation='dense' explicitly for this dtype."
        )
    selected = (
        cp.cuda.Device().id
        if device is None
        else int(str(device).removeprefix("cuda:"))
    )
    with cp.cuda.Device(selected):
        scans, pixels = math.prod(shape[:2]), math.prod(shape[2:])
        block = 128
        streams = math.ceil(scans / block) * pixels
        widths = cp.empty(streams, cp.uint8)
        lengths = cp.empty(streams, cp.uint64)
        kernels = _kernels(cp.cuda.Device().id)
        signatures = {str(path): _file_source_signature(path)}
        with ExitStack() as stack:
            dataset_path = dataset_path or info.metadata.get("dataset_path")
            dataset = None
            session = None
            if dataset_path:
                dataset = stack.enter_context(h5py.File(path, "r"))[dataset_path]
                signatures[dataset.file.filename] = _file_source_signature(
                    dataset.file.filename
                )
                if dataset.id.get_create_plist().get_nfilters():
                    raise NotImplementedError(
                        "Filtered generic HDF5 requires a supported GPU decoder; "
                        "use an acquisition with the supported native master layout."
                    )
                if tuple(dataset.shape) != shape or np.dtype(dataset.dtype) != dtype:
                    raise ValueError(
                        "The dataset must preserve its full inspected geometry."
                    )
            else:
                names = _discover_chunk_names(str(path)) or ["data"]
                session = _SparseFrameReadSession(str(path), names, apply_mask=False)
                stack.callback(session.close)
                signatures.update(
                    {
                        entry["path"]: _file_source_signature(entry["path"])
                        for entry in session.source_infos
                    }
                )
            for phase in range(2):
                if any(
                    _file_source_signature(name) != initial
                    for name, initial in signatures.items()
                ):
                    raise RuntimeError(
                        "Source changed during packed loading; retry with immutable inputs."
                    )
                for first in range(0, scans, chunk_scans):
                    stop = min(first + chunk_scans, scans)
                    if dataset is not None:
                        host = np.empty((stop - first, *shape[2:]), dtype)
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
                            batch_bytes_target=batch_bytes_target,
                            prune_device_pool=False,
                        )
                    start_stream = first // block * pixels
                    count_streams = math.ceil((stop - first) / block) * pixels
                    end_stream = start_stream + count_streams
                    dimensions = (
                        np.uint64(stop - first),
                        np.uint32(pixels),
                        np.uint32(block),
                        np.uint64(count_streams),
                    )
                    launch = (((count_streams + 127) // 128,), (128,))
                    if phase == 0:
                        kernels["dense_measure_packed"](
                            *launch,
                            (
                                raw,
                                np.uint32(dtype.itemsize),
                                widths[start_stream:end_stream],
                                lengths[start_stream:end_stream],
                                *dimensions,
                            ),
                        )
                    else:
                        kernels["dense_write_packed"](
                            *launch,
                            (
                                raw,
                                np.uint32(dtype.itemsize),
                                widths[start_stream:end_stream],
                                offsets[start_stream : end_stream + 1],
                                words,
                                *dimensions,
                            ),
                        )
                    cp.cuda.get_current_stream().synchronize()
                    del raw
                if phase == 0:
                    offsets = cp.zeros(streams + 1, cp.uint64)
                    cp.cumsum(lengths, out=offsets[1:])
                    word_count = int(offsets[-1].get())
                    del lengths
                    cp.get_default_memory_pool().free_all_blocks()
                    words = cp.empty(word_count, cp.uint32)
        if any(
            _file_source_signature(name) != initial
            for name, initial in signatures.items()
        ):
            raise RuntimeError(
                "Source changed during packed loading; retry with immutable inputs."
            )
        owner = CudaPackedResidentCounts(
            shape, block, dtype, words, offsets, widths, cp.cuda.Device().id, kernels
        )
        metadata = dict(info.metadata)
        metadata.update(
            backend="cuda",
            representation="packed",
            residency="device",
            source_shape=shape,
            working_shape=shape,
            scan_shape=shape[:2],
            detector_shape=shape[2:],
            source_dtype=dtype.name,
            working_dtype=dtype.name,
            dtype=dtype.name,
            pixel_mask=info.pixel_mask,
            physical_resident_bytes=owner.resident_bytes,
            lossless_exact=True,
            detector_mask_policy="preserve-stored-counts",
        )
        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f"Packed {scans} native scans using bounded input buffers.")
        return FourDSTEMData(owner, metadata)
