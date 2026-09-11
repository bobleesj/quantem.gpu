"""Bounded native packing through existing GPU kernels."""

import hashlib
import json
import math
import os
import struct
import tempfile
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

_PACKING_PLAN_CACHE_ENV = "QUANTEM_GPU_PACKING_PLAN_CACHE_DIR"
_PACKING_PLAN_CACHE_VERSION = 1
_PACKING_PLAN_MAGIC = b"QGPUPLAN"


def _packing_plan_cache_path(path: Path, dataset_path: str | None) -> Path | None:
    """Return the private cache path for one source layout."""
    configured = os.environ.get(_PACKING_PLAN_CACHE_ENV)
    if configured is None:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        root = root / "quantem-gpu" / "packing-plans"
    elif not configured.strip():
        return None
    else:
        root = Path(configured).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    source = f"{path.absolute()}\0{dataset_path or ''}".encode()
    return root / f"{hashlib.sha256(source).hexdigest()}.bin"


def _packing_plan_identity(
    signatures: dict[str, dict],
    shape: tuple[int, ...],
    dtype: np.dtype,
    block_frames: int,
) -> dict:
    """Build the exact source identity stored with cached packing widths."""
    return {
        "version": _PACKING_PLAN_CACHE_VERSION,
        "sources": signatures,
        "shape": [int(value) for value in shape],
        "dtype": dtype.str,
        "block_frames": int(block_frames),
    }


def _load_packing_widths(
    cache_path: Path | None,
    identity: dict,
    stream_count: int,
    maximum_width: int,
) -> np.ndarray | None:
    """Load validated bit widths without trusting cache contents."""
    if cache_path is None:
        return None
    try:
        with cache_path.open("rb") as handle:
            if handle.read(len(_PACKING_PLAN_MAGIC)) != _PACKING_PLAN_MAGIC:
                return None
            encoded_size = handle.read(8)
            if len(encoded_size) != 8:
                return None
            header_size = struct.unpack("<Q", encoded_size)[0]
            if header_size > 1024 * 1024:
                return None
            header = handle.read(header_size)
            if len(header) != header_size or json.loads(header) != identity:
                return None
            widths = np.fromfile(handle, dtype=np.uint8, count=stream_count)
            if widths.size != stream_count or handle.read(1):
                return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if int(widths.max(initial=0)) > maximum_width:
        return None
    return widths


def _packing_plan_ready(
    path: str | Path,
    dataset_path: str | None,
    scan_shape: tuple[int, int] | None,
) -> bool:
    """Return whether a source-validated plan can support one-pass loading."""
    cache_path = _packing_plan_cache_path(Path(path), dataset_path)
    if cache_path is None:
        return False
    try:
        with cache_path.open("rb") as handle:
            if handle.read(len(_PACKING_PLAN_MAGIC)) != _PACKING_PLAN_MAGIC:
                return False
            encoded_size = handle.read(8)
            if len(encoded_size) != 8:
                return False
            header_size = struct.unpack("<Q", encoded_size)[0]
            if header_size > 1024 * 1024:
                return False
            header = json.loads(handle.read(header_size))
        shape = tuple(int(value) for value in header["shape"])
        block_frames = int(header["block_frames"])
        if (
            header.get("version") != _PACKING_PLAN_CACHE_VERSION
            or len(shape) != 4
            or block_frames < 1
            or (scan_shape is not None and tuple(scan_shape) != shape[:2])
        ):
            return False
        sources = header["sources"]
        if not isinstance(sources, dict) or any(
            _file_source_signature(name) != signature
            for name, signature in sources.items()
        ):
            return False
        stream_count = math.ceil(math.prod(shape[:2]) / block_frames) * math.prod(
            shape[2:]
        )
        expected_size = len(_PACKING_PLAN_MAGIC) + 8 + header_size + stream_count
        return cache_path.stat().st_size == expected_size
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_packing_widths(
    cache_path: Path | None,
    identity: dict,
    widths: np.ndarray,
) -> None:
    """Atomically retain measured widths for later single-pass loads."""
    if cache_path is None:
        return
    header = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    try:
        file_descriptor, temporary = tempfile.mkstemp(
            prefix=f".{cache_path.name}.", dir=cache_path.parent
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(_PACKING_PLAN_MAGIC)
                handle.write(struct.pack("<Q", len(header)))
                handle.write(header)
                np.asarray(widths, dtype=np.uint8).tofile(handle)
            os.replace(temporary, cache_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _offsets_from_widths(widths, scans: int, pixels: int, block_frames: int):
    """Construct exact packed word offsets from already measured widths."""
    lengths = widths.astype(cp.uint64)
    complete_blocks, remainder = divmod(scans, block_frames)
    complete_streams = complete_blocks * pixels
    if complete_streams:
        lengths[:complete_streams] *= block_frames // 32
    if remainder:
        tail = lengths[complete_streams:]
        tail *= remainder
        tail += 31
        tail //= 32
    offsets = cp.zeros(widths.size + 1, cp.uint64)
    cp.cumsum(lengths, dtype=cp.uint64, out=offsets[1:])
    del lengths
    return offsets


def load_h5_packed(
    path: str | Path,
    *,
    scan_shape: tuple[int, int] | None,
    dataset_path: str | None,
    device: int | str | None,
    verbose: bool,
) -> FourDSTEMData:
    """Pack native counts with bounded buffers and an automatic width plan."""
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
            cache_path = _packing_plan_cache_path(path, dataset_path)
            identity = _packing_plan_identity(signatures, shape, dtype, block)
            cached_widths = _load_packing_widths(
                cache_path, identity, streams, dtype.itemsize * 8
            )
            if cached_widths is None:
                source_passes = 2
                widths = cp.empty(streams, cp.uint8)
                lengths = cp.empty(streams, cp.uint64)
                phases = (0, 1)
            else:
                source_passes = 1
                widths = cp.asarray(cached_widths)
                offsets = _offsets_from_widths(widths, scans, pixels, block)
                words = cp.empty(int(offsets[-1].get()), cp.uint32)
                phases = (1,)
            for phase in phases:
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
                    measured_widths = widths.get()
                    _write_packing_widths(cache_path, identity, measured_widths)
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
            source_read_passes=source_passes,
        )
        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f"Packed {scans} native scans in {source_passes} source pass(es).")
        return FourDSTEMData(owner, metadata)
