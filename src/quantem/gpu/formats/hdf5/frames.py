"""Locate the detector frames of an HDF5 master without reading them.

A bitshuffle+LZ4 master stores one detector frame per HDF5 chunk, spread over
external data files. Bounded GPU loads need the byte offset and size of every
frame chunk and the start of every LZ4 block inside it, so they can read only
the selected frames and decode their blocks in parallel. Everything here reads
file headers and compressed bytes on the host; nothing touches a GPU.
"""

import ctypes
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache

import h5py
import numpy as np
from h5py._objects import phil
from numba import carray, cfunc, njit, prange, types

# Bitshuffle block size; must match the hdf5plugin format so other readers decode the files.
BLOCK_SIZE: int = 8192

# Parsed chunk indexes per (master, chunk names), with the file signatures
# that prove they still describe the files on disk.
_MASTER_FRAME_SOURCE_CACHE: dict[
    tuple[str, tuple[str, ...]], tuple[list[dict], list[dict]]
] = {}
_FRAME_SOURCE_CACHE_ENV = "QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR"
_TRUST_FRAME_SOURCE_CACHE_ENV = "QUANTEM_GPU_HDF5_TRUST_FRAME_SOURCE_CACHE"


@dataclass(frozen=True)
class DetectorSource:
    """One detector dataset of a master: its member name, file, dataset path and link kind."""

    name: str
    path: str
    dataset_path: str
    external: bool


def detector_sources(master: h5py.File) -> list[DetectorSource]:
    """List the detector datasets of an open master: ``data_NNNNNN`` members, or one ``data``.

    Arina writes each block of frames to its own file and links it from the
    master as ``entry/data/data_000001``, ``data_000002``, ...; a
    self-contained master stores ``entry/data/data``. Link targets resolve
    against the master's folder, never a guessed file name, so a renamed master
    still reads the files it links. Returns an empty list without ``entry/data``.
    """
    data_group = master.get("entry/data")
    if data_group is None:
        return []
    names = sorted(name for name in data_group if re.fullmatch(r"data_\d{6}", name))
    if not names and "data" in data_group:
        names = ["data"]
    master_path = os.path.abspath(master.filename)
    sources = []
    for name in names:
        link = data_group.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            filename = os.fsdecode(os.fspath(link.filename))
            if not os.path.isabs(filename):
                filename = os.path.join(os.path.dirname(master_path), filename)
            path = os.path.abspath(os.path.expanduser(filename))
            sources.append(DetectorSource(name, path, str(link.path), True))
        else:
            sources.append(DetectorSource(name, master_path, f"{data_group.name}/{name}", False))
    return sources


def master_frame_sources(filepath: str) -> list[dict]:
    """Return the detector-frame chunk index for one master.

    Sparse reads need the byte offset of every frame chunk. Indexing is
    reused from process memory, then from the disk cache, and rebuilt only
    when a file signature changed.
    """
    with h5py.File(filepath, "r") as master:
        sources = detector_sources(master)
    chunk_names = [source.name for source in sources]
    abs_path = os.path.abspath(filepath)
    key = (abs_path, tuple(chunk_names))
    cached = _MASTER_FRAME_SOURCE_CACHE.get(key)
    if cached is not None:
        signatures, infos = cached
        if all(_file_stat_signature(item["path"]) == item for item in signatures):
            return infos
        del _MASTER_FRAME_SOURCE_CACHE[key]

    disk_cache_path = _frame_source_cache_path(filepath, chunk_names)
    if (
        disk_cache_path is not None
        and os.environ.get(_TRUST_FRAME_SOURCE_CACHE_ENV, "").lower()
        in {"1", "true", "yes", "on"}
    ):
        cached = _load_frame_source_disk_cache(
            disk_cache_path,
            signature=None,
        )
        if cached is not None:
            _MASTER_FRAME_SOURCE_CACHE[key] = (
                [_file_stat_signature(path) for path in
                 [filepath, *(info["path"] for info in cached)]], cached
            )
            return cached

    # The file signature that keys the chunk-index caches.
    signature = {
        "master": _file_stat_signature(filepath),
        "sources": [
            {
                **_file_stat_signature(source.path),
                "dataset_path": source.dataset_path,
            }
            for source in sources
        ],
        "chunk_names": chunk_names,
    }
    file_signatures = [signature["master"], *[
        {name: value for name, value in item.items() if name != "dataset_path"}
        for item in signature["sources"]
    ]]
    if disk_cache_path is not None:
        cached = _load_frame_source_disk_cache(
            disk_cache_path,
            signature=signature,
        )
        if cached is not None:
            _MASTER_FRAME_SOURCE_CACHE[key] = (file_signatures, cached)
            return cached

    source_infos: list[dict] = []
    for source in sources:
        with h5py.File(source.path, "r") as df:
            ds = df[source.dataset_path]
            if ds.ndim != 3:
                raise ValueError(
                    "load(..., scan_region=...) currently supports flattened "
                    f"3D detector chunks; got {ds.shape} in {source.path}"
                )
            if ds.chunks is None or int(ds.chunks[0]) != 1:
                raise ValueError(
                    "load(..., scan_region=...) requires one detector frame per "
                    f"HDF5 chunk; got chunks={ds.chunks} in {source.path}"
                )
            chunk_infos = chunk_locations(ds)
            source_infos.append(
                {
                    "path": source.path,
                    "dataset_path": source.dataset_path,
                    "n_frames": int(ds.shape[0]),
                    "frame_shape": tuple(int(v) for v in ds.shape[1:]),
                    "dtype": ds.dtype,
                    "chunk_infos": np.asarray(chunk_infos, dtype=np.uint64),
                }
            )
    _MASTER_FRAME_SOURCE_CACHE[key] = (file_signatures, source_infos)
    if disk_cache_path is not None:
        _write_frame_source_disk_cache(
            disk_cache_path,
            signature=signature,
            source_infos=source_infos,
        )
    return source_infos


def _file_stat_signature(path: str) -> dict[str, str | int]:
    """Return the small file signature needed to validate an IO metadata cache."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _frame_source_cache_path(filepath: str, chunk_names: list[str]) -> str | None:
    """Return the private disk-cache path for source chunk metadata."""
    configured = os.environ.get(_FRAME_SOURCE_CACHE_ENV)
    if configured is None:
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "quantem-gpu",
            "frame-sources",
        )
    elif not configured.strip():
        return None
    else:
        cache_dir = os.path.expanduser(configured)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None
    payload = repr(
        (os.path.abspath(filepath), tuple(str(name) for name in chunk_names))
    ).encode("utf-8")
    key = hashlib.sha256(payload).hexdigest()
    return os.path.join(cache_dir, f"{key}.npz")


def _load_frame_source_disk_cache(
    cache_path: str,
    *,
    signature: dict | None,
) -> list[dict] | None:
    """Load cached frame-source metadata when every file signature still matches.

    A missing, stale, or unreadable cache returns None so the caller rebuilds
    the chunk index from the HDF5 files.
    """
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(cached["metadata"].tobytes())
            if signature is not None and metadata.get("signature") != signature:
                return None
            source_infos = []
            for index, info in enumerate(metadata.get("source_infos", [])):
                item = dict(info)
                item["dtype"] = np.dtype(item["dtype"])
                item["frame_shape"] = tuple(int(v) for v in item["frame_shape"])
                chunk_infos = np.asarray(cached[f"chunks_{index}"], dtype=np.uint64)
                if chunk_infos.ndim != 2 or chunk_infos.shape[1:] != (2,):
                    return None
                item["chunk_infos"] = chunk_infos
                source_infos.append(item)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return source_infos or None


def _write_frame_source_disk_cache(
    cache_path: str,
    *,
    signature: dict,
    source_infos: list[dict],
) -> None:
    """Write cached frame-source metadata atomically for future worker processes.

    Indexing every HDF5 chunk of a large master costs seconds; the cache lets
    fresh processes skip it. A failed write leaves no partial file behind.
    """
    serial_infos: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for index, info in enumerate(source_infos):
        item = dict(info)
        item["dtype"] = np.dtype(item["dtype"]).str
        item["frame_shape"] = [int(v) for v in item["frame_shape"]]
        arrays[f"chunks_{index}"] = np.asarray(
            item.pop("chunk_infos"), dtype=np.uint64
        )
        serial_infos.append(item)
    metadata = {"signature": signature, "source_infos": serial_infos}
    arrays["metadata"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
        dtype=np.uint8,
    )
    cache_dir = os.path.dirname(cache_path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".frame_source_", suffix=".tmp", dir=cache_dir
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(tmp_path, cache_path)
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def chunk_locations(dataset: h5py.Dataset) -> np.ndarray:
    """Return allocated chunk offsets and sizes in HDF5 iteration order."""
    # Use h5py's lock because its linked HDF5 library need not be thread-safe.
    # It also serializes lazy callback compilation across simultaneous loads.
    with phil:
        native = _native_iterator()
        if native is None:
            return _chunk_locations_python(dataset)
        # Size for acquired chunks, not the potentially huge declared extent
        # of an unfinished scan. Enumeration performs no evidence reads.
        capacity = dataset.id.get_num_chunks()
        output = np.empty(2 + 2 * capacity, dtype=np.uint64)
        output[:2] = capacity, 0
        _, iterate, callback = native
        status = iterate(dataset.id.id, 0, callback.address, output.ctypes.data)
        if status != 0:
            raise OSError(
                f"Cannot enumerate HDF5 chunks in {dataset.name!r}; "
                "check that the source is complete and readable."
            )
        count = int(output[1])
        return output[2:2 + 2 * count].reshape(-1, 2)


@lru_cache(maxsize=1)
def _native_iterator():
    """Resolve symbols from h5py's own library, never a second HDF5 install."""
    try:
        library = ctypes.CDLL(h5py.h5d.__file__)
        iterate = library.H5Dchunk_iter
    except (OSError, AttributeError):
        return None
    iterate.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
    ]
    iterate.restype = ctypes.c_int
    signature = types.intc(
        types.CPointer(types.uint64), types.uintc,
        types.uint64, types.uint64, types.voidptr,
    )
    return library, iterate, cfunc(signature, cache=True)(_collect_chunk)


def _collect_chunk(offset, filter_mask, address, size, context):
    """Append to a capacity-checked array owned by the synchronous caller."""
    header = carray(context, (2,), dtype=np.uint64)
    capacity, count = header[0], header[1]
    if count >= capacity:
        return -1
    output = carray(context, (2 + 2 * capacity,), dtype=np.uint64)
    output[2 + 2 * count] = address
    output[3 + 2 * count] = size
    header[1] = count + 1
    return 0


def _chunk_locations_python(dataset: h5py.Dataset) -> np.ndarray:
    """Retain the h5py reference when direct library symbols are unavailable."""
    locations = []
    dataset.id.chunk_iter(
        lambda info: locations.append((info.byte_offset, info.size))
    )
    return np.asarray(locations, dtype=np.uint64).reshape(-1, 2)


def parse_headers(
    pinned_buffer, chunk_sizes, chunk_offsets, block_starts_out,
    block_counts_out, n_frames, n_blocks_per_frame, *, thread_pool=None,
):
    """Parse chunk headers with the cheapest scheduler for the batch size.

    Launching Numba's parallel worker team costs more than it saves on
    bounded streaming batches, so only very large batches use it.
    """
    frame_count = int(n_frames)
    if thread_pool is not None and 8192 <= frame_count <= 32768:
        blocks_per_frame = int(n_blocks_per_frame)
        frames_per_worker = (frame_count + 3) // 4

        def parse_slice(first):
            stop = min(first + frames_per_worker, frame_count)
            _parse_headers_serial(
                pinned_buffer, chunk_sizes[first:stop], chunk_offsets[first:stop],
                block_starts_out[first * blocks_per_frame:stop * blocks_per_frame],
                block_counts_out[first:stop], stop - first, blocks_per_frame,
            )

        # Reads have completed. Reuse their workers and disjoint output slices
        # instead of starting Numba's much larger parallel worker team.
        list(thread_pool.map(parse_slice, range(0, frame_count, frames_per_worker)))
        return
    parser = _parse_headers_serial if frame_count <= 32768 else _parse_headers
    parser(
        pinned_buffer, chunk_sizes, chunk_offsets, block_starts_out,
        block_counts_out, n_frames, n_blocks_per_frame,
    )


@njit(cache=True, inline="always")
def _parse_frame_header(
    pinned_buffer,
    chunk_sizes,
    chunk_offsets,
    block_starts_out,
    block_counts_out,
    i,
    n_blocks_per_frame,
):
    """Locate the compressed LZ4 blocks of one bitshuffle+LZ4 HDF5 chunk.

    The chunk starts with a 12-byte big-endian header (8-byte uncompressed
    size, 4-byte block size); each block carries a 4-byte big-endian length.
    The GPU decoder needs every block's start offset to decode in parallel.
    """
    offset = chunk_offsets[i]
    chunk = pinned_buffer[offset : offset + chunk_sizes[i]]
    uncomp_size = (
        int(chunk[0]) << 56
        | int(chunk[1]) << 48
        | int(chunk[2]) << 40
        | int(chunk[3]) << 32
        | int(chunk[4]) << 24
        | int(chunk[5]) << 16
        | int(chunk[6]) << 8
        | int(chunk[7])
    )
    block_size = (
        int(chunk[8]) << 24
        | int(chunk[9]) << 16
        | int(chunk[10]) << 8
        | int(chunk[11])
    )
    n_blocks = (uncomp_size + block_size - 1) // block_size
    block_counts_out[i] = n_blocks
    pos = 12
    base_idx = i * n_blocks_per_frame
    for b in range(n_blocks):
        block_starts_out[base_idx + b] = pos
        comp_size = (
            int(chunk[pos]) << 24
            | int(chunk[pos + 1]) << 16
            | int(chunk[pos + 2]) << 8
            | int(chunk[pos + 3])
        )
        pos += 4 + comp_size


@njit(cache=True, parallel=True)
def _parse_headers(
    pinned_buffer, chunk_sizes, chunk_offsets, block_starts_out,
    block_counts_out, n_frames, n_blocks_per_frame,
):
    """Parse bitshuffle+LZ4 chunk headers in parallel."""
    for i in prange(n_frames):
        _parse_frame_header(
            pinned_buffer, chunk_sizes, chunk_offsets, block_starts_out,
            block_counts_out, i, n_blocks_per_frame,
        )


# Bounded streaming batches do too little header work to amortize a parallel
# team launch. Compile the identical parser without parallel scheduling there.
@njit(cache=True, nogil=True)
def _parse_headers_serial(
    pinned_buffer, chunk_sizes, chunk_offsets, block_starts_out,
    block_counts_out, n_frames, n_blocks_per_frame,
):
    """Parse bounded batches without parallel team-launch overhead."""
    for i in range(n_frames):
        _parse_frame_header(
            pinned_buffer, chunk_sizes, chunk_offsets, block_starts_out,
            block_counts_out, i, n_blocks_per_frame,
        )
