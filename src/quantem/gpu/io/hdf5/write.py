"""Write 4D-STEM frames as Arina-style master/data HDF5 files.

The layout is the one detector software and hdf5plugin readers expect: a
``*_master.h5`` that links ``*_data_000001.h5``, ... files, each holding
``/entry/data/data`` with one detector frame per chunk. GPU compressors hand
finished bitshuffle+LZ4 chunks to a background writer thread, so compression
of the next batch overlaps the file writes of this one.
"""

import queue
import threading
from collections.abc import Mapping
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.formats.hdf5.frames import BLOCK_SIZE
from quantem.gpu.io.hdf5.cuda.encode import compress_batch
from quantem.gpu.io.hdf5.mps.encode import NativeU16Compressor, compress_tensor_batch
from quantem.gpu.resident.mps.arrays import MetalArray

try:
    import torch
except ImportError:  # pragma: no cover - only for minimal IO-only installs
    torch = None

# Compression codecs available to save() / H5Writer. Default 'lz4' uses the
# GPU bitshuffle+LZ4 fast path that streams compressed bytes via
# write_direct_chunk. The other codecs let HDF5's CPU plugin pipeline run
# (bigger ratio, slower): pick them for archival writes where size matters
# more than write throughput.
_COMPRESSION_CODECS = ("lz4", "zstd", "blosc2_zstd")
_DEFAULT_CLEVELS = {"zstd": 3, "blosc2_zstd": 5}
_DEFAULT_BATCH_SIZE = 512
_MPS_DEFAULT_BATCH_SIZE = 2048
_SAVE_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.float16),
    np.dtype(np.float32),
)
_DTYPE_ALIASES = {
    "u8": np.uint8,
    "uint8": np.uint8,
    "u16": np.uint16,
    "uint16": np.uint16,
    "u32": np.uint32,
    "uint32": np.uint32,
    "f32": np.float32,
    "float32": np.float32,
    "f16": np.float16,
    "float16": np.float16,
}
# Backpressure bounds completed byte batches without stopping the GPU whenever
# an arbitrary batch count is reached. The writer drains while compute proceeds.
_write_queue = queue.Queue(maxsize=2)
_write_thread = None
_write_thread_lock = threading.Lock()
_write_error = None


class H5Writer:
    """Streaming Arina-style GPU writer for bitshuffle+LZ4 4D-STEM files.

    The output is a master HDF5 file with external data files:
    ``name_master.h5`` plus ``name_data_000001.h5``, etc. Each external
    data file contains ``entry/data/data`` with shape
    ``(n_frames_in_file, detector_row, detector_col)`` and HDF5 chunks
    ``(1, detector_row, detector_col)``. This matches the row/column
    Arina/Dectris layout closely enough for existing chunked readers.

    ``float32`` is the intended dtype for canonical drift-corrected 4D-STEM
    archives because bilinear correction creates fractional detector values.
    Pass a different dtype only when intentionally making a non-default export.
    """

    def __init__(self, filepath, n_frames, det_shape, scan_shape=None,
                 metadata=None, dtype=np.float32, source_master=None,
                 frames_per_file=32768, compression="lz4",
                 compression_level=0):
        # Until construction finishes there is nothing for __del__ to close.
        self._closed = True
        self._filepath = Path(filepath)
        self._n_frames = int(n_frames)
        self._det_row, self._det_col = (int(det_shape[0]), int(det_shape[1]))
        self._dtype = _normalize_save_dtype(dtype)
        self._compression, self._compression_level = _normalize_compression(
            compression, compression_level
        )
        self._frame_bytes = self._det_row * self._det_col * self._dtype.itemsize
        self._n_8kb = (self._frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE
        self._scan_shape = None if scan_shape is None else tuple(int(x) for x in scan_shape)
        self._metadata = metadata
        self._source_master = source_master
        self._frames_per_file = int(frames_per_file)
        if self._frames_per_file <= 0:
            raise ValueError("frames_per_file must be positive")
        self._frame_offset = 0
        self._file_index = 0
        self._current_file = None
        self._current_ds = None
        self._current_file_n = 0
        self._current_file_start = 0
        self._current_file_offset = 0
        self._data_files = []
        self._frame_ranges = []
        self._closed = False
        self._metal_compressor = None
        self._prefix = _master_prefix(self._filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

        wait_for_saves()
        _ensure_writer_thread()

    def _open_data_file(self):
        self._file_index += 1
        self._current_file_start = self._frame_offset
        remaining = self._n_frames - self._frame_offset
        self._current_file_n = min(self._frames_per_file, remaining)
        self._current_file_offset = 0
        data_path = self._filepath.with_name(f"{self._prefix}_data_{self._file_index:06d}.h5")
        self._data_files.append(data_path)
        lo = self._current_file_start + 1
        hi = self._current_file_start + self._current_file_n
        self._frame_ranges.append((lo, hi))

        self._current_file = h5py.File(data_path, "w")
        self._current_ds = self._current_file.create_dataset(
            "entry/data/data",
            shape=(self._current_file_n, self._det_row, self._det_col),
            dtype=self._dtype,
            chunks=(1, self._det_row, self._det_col),
            **_hdf5_filter(self._compression, self._compression_level),
        )
        self._current_ds.attrs["image_nr_low"] = np.uint64(lo)
        self._current_ds.attrs["image_nr_high"] = np.uint64(hi)

    def _close_data_file(self):
        if self._current_file is None:
            return
        wait_for_saves()
        self._current_file.close()
        self._current_file = None
        self._current_ds = None
        self._current_file_n = 0
        self._current_file_offset = 0

    def write(self, data_gpu):
        """Compress and queue a frame batch.

        ``data_gpu`` may be a CuPy or NumPy array with shape
        ``(n_batch, detector_row, detector_col)``. The dtype is cast to the
        writer dtype before compression.
        """
        if self._closed:
            raise RuntimeError("H5Writer is closed")
        _raise_write_error()
        # Precision exports hand over 16-bit codes already in a shared Metal buffer.
        native_metal = isinstance(data_gpu, MetalArray) and data_gpu.dtype == self._dtype
        native_mps = (
            torch is not None and torch.is_tensor(data_gpu)
            and data_gpu.device.type == "mps" and np.dtype(str(data_gpu.dtype).removeprefix("torch.")) == self._dtype
        )
        if (native_metal or native_mps) and (self._dtype.itemsize != 2 or self._compression != "lz4"):
            raise ValueError("Native Metal precision writing requires 16-bit bitshuffle/LZ4 storage.")
        if not native_metal and not native_mps and not isinstance(data_gpu, cp.ndarray):
            data_gpu = cp.asarray(np.asarray(data_gpu))
        input_dtype = (
            np.dtype(str(data_gpu.dtype).removeprefix("torch."))
            if native_mps else data_gpu.dtype
        )
        if input_dtype != self._dtype:
            # Float→integer cast: round to nearest BEFORE casting so bilinear-merged
            # 4D-STEM keeps max-error 0.5 counts (sub-noise-floor) instead of the 1.0
            # max-error you get from truncation. Numpy/CuPy default float->uint cast
            # truncates fractional parts.
            if (np.issubdtype(input_dtype, np.floating)
                    and np.issubdtype(self._dtype, np.integer)):
                lo, hi = (int(np.iinfo(self._dtype).min),
                          int(np.iinfo(self._dtype).max))
                data_gpu = cp.clip(cp.rint(data_gpu), lo, hi).astype(self._dtype)
            else:
                data_gpu = data_gpu.astype(self._dtype)
        if data_gpu.ndim != 3 or data_gpu.shape[1:] != (self._det_row, self._det_col):
            raise ValueError(
                f"Expected batch shape (n, {self._det_row}, {self._det_col}), "
                f"got {tuple(data_gpu.shape)}"
            )
        if self._frame_offset + int(data_gpu.shape[0]) > self._n_frames:
            raise ValueError("Batch would exceed declared n_frames")

        if not native_metal and not native_mps:
            data_gpu = cp.ascontiguousarray(data_gpu)
        batch_start = 0
        batch_n = int(data_gpu.shape[0])
        while batch_start < batch_n:
            if self._current_file is None:
                self._open_data_file()
            room = self._current_file_n - self._current_file_offset
            n_part = min(room, batch_n - batch_start)
            if native_metal:
                if self._metal_compressor is None:
                    self._metal_compressor = NativeU16Compressor(
                        batch_n, self._frame_bytes, self._n_8kb
                    )
                packed, starts, sizes = self._metal_compressor.compress(
                    data_gpu, batch_start, n_part
                )
                _write_queue.put((
                    _write_batch_to_h5,
                    (self._current_ds, packed, starts, sizes,
                     self._current_file_offset, n_part),
                ))
            elif native_mps:
                part = data_gpu[batch_start:batch_start + n_part]
                packed, starts, sizes = compress_tensor_batch(
                    part, self._n_8kb, self._frame_bytes, self._dtype
                )
                _write_queue.put((
                    _write_batch_to_h5,
                    (self._current_ds, packed, starts, sizes,
                     self._current_file_offset, n_part),
                ))
            elif self._compression == "lz4":
                part = data_gpu[batch_start:batch_start + n_part]
                packed, starts, sizes = compress_batch(
                    part, self._n_8kb, self._frame_bytes
                )
                _write_queue.put((
                    _write_batch_to_h5,
                    (self._current_ds, packed, starts, sizes,
                     self._current_file_offset, n_part),
                ))
            else:
                part = data_gpu[batch_start:batch_start + n_part]
                # Non-LZ4 codecs run inside HDF5's filter pipeline on CPU.
                # Pull the batch to host once, queue the filtered write so
                # GPU work continues while compression happens on a worker.
                host_frames = cp.asnumpy(part)
                _write_queue.put((
                    _write_batch_via_filter,
                    (self._current_ds, host_frames,
                     self._current_file_offset, n_part),
                ))
            self._current_file_offset += n_part
            self._frame_offset += n_part
            batch_start += n_part
            if self._current_file_offset == self._current_file_n:
                self._close_data_file()

    def close(self, wait: bool = False):
        """Finalize external data files and write the master file."""
        if self._closed:
            if wait:
                wait_for_saves()
            return
        self._closed = True
        self._close_data_file()
        if self._metal_compressor is not None:
            self._metal_compressor.close()
            self._metal_compressor = None
        if self._frame_offset != self._n_frames:
            raise RuntimeError(f"H5Writer wrote {self._frame_offset} of {self._n_frames} frames")
        _write_master_file(
            self._filepath,
            self._data_files,
            self._frame_ranges,
            self._scan_shape,
            (self._det_row, self._det_col),
            self._dtype,
            self._metadata,
            self._source_master,
        )
        if wait:
            wait_for_saves()

    def abort(self) -> None:
        """Close and remove an incomplete streamed output after a failed producer."""
        self._closed = True
        # Cleanup after a failed producer must not replace the producer's
        # error, so write and close failures here are ignored.
        try:
            self._close_data_file()
        except (RuntimeError, OSError):
            # wait_for_saves has already drained the queue before surfacing a
            # writer error, so closing the handle here cannot race queued work.
            try:
                if self._current_file is not None:
                    self._current_file.close()
            except (RuntimeError, OSError):
                pass
            self._current_file = None
            self._current_ds = None
        if self._metal_compressor is not None:
            try:
                self._metal_compressor.close()
            except (RuntimeError, OSError):
                pass
            self._metal_compressor = None
        for path in self._data_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            self._filepath.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(wait=True)
        return False

    def __del__(self):
        if not self._closed:
            self.close(wait=False)


def save_compressed_arina_h5(
    filepath: str | Path,
    data,
    *,
    scan_shape: tuple[int, int] | None = None,
    metadata: Mapping | None = None,
    dtype="u16",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    frames_per_file: int = 32768,
    compression: str | None = "lz4",
    compression_level: int | None = 4,
    shuffle: bool = True,
    source_master: str | None = None,
    compression_backend: str = "auto",
) -> None:
    """Save 4D-STEM as Arina-style master/data HDF5 using portable filters.

    This uses the same master-file/external-data layout as :class:`H5Writer`:
    ``*_master.h5`` links to ``*_data_000001.h5`` files containing
    ``/entry/data/data`` with shape ``(n_frames, det_row, det_col)`` and HDF5
    chunks ``(1, det_row, det_col)``. Unlike :func:`save`, it does not require
    CUDA/CuPy and can stream MPS ``torch.Tensor`` data to host in batches.
    With ``compression_backend="mps"`` or ``"auto"`` on an MPS tensor, the
    bitshuffle+LZ4 chunk bytes are produced by Metal kernels and injected with
    ``write_direct_chunk``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dtype = _normalize_save_dtype(dtype)
    shape = tuple(int(x) for x in data.shape)
    if len(shape) == 4:
        inferred_scan = shape[:2]
        if scan_shape is not None and tuple(scan_shape) != inferred_scan:
            raise ValueError(f"scan_shape={scan_shape} does not match data shape {inferred_scan}")
        scan_shape = inferred_scan
        n_frames = shape[0] * shape[1]
        det_shape = shape[2:]
        flat = data.reshape(n_frames, *det_shape)
    elif len(shape) == 3:
        if scan_shape is None:
            raise ValueError("scan_shape is required for 3D frame stacks")
        n_frames = shape[0]
        det_shape = shape[1:]
        flat = data
    else:
        raise ValueError("save_compressed_arina_h5 expects 3D frames or 4D-STEM data")

    det_row, det_col = (int(det_shape[0]), int(det_shape[1]))
    frames_per_file = int(frames_per_file)
    batch_size = int(batch_size)
    if frames_per_file <= 0 or batch_size <= 0:
        raise ValueError("frames_per_file and batch_size must be positive")
    compression_backend = str(compression_backend).lower()
    if compression_backend not in {"auto", "hdf5", "mps"}:
        raise ValueError("compression_backend must be 'auto', 'hdf5', or 'mps'")
    use_mps_backend = (
        compression_backend in {"auto", "mps"}
        and compression in {"lz4", "bslz4", "bitshuffle_lz4"}
        and dtype in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))
        and _is_mps_array(flat)
    )
    if compression_backend == "mps" and not use_mps_backend:
        raise ValueError(
            "compression_backend='mps' requires an MPS torch.Tensor and "
            "compression='lz4'."
        )
    if use_mps_backend and batch_size == _DEFAULT_BATCH_SIZE:
        batch_size = _MPS_DEFAULT_BATCH_SIZE

    prefix = _master_prefix(filepath)
    data_files = []
    frame_ranges = []
    for file_index, start in enumerate(range(0, n_frames, frames_per_file), start=1):
        end = min(start + frames_per_file, n_frames)
        data_path = filepath.with_name(f"{prefix}_data_{file_index:06d}.h5")
        data_files.append(data_path)
        frame_ranges.append((start + 1, end))
        tmp_data = data_path.with_suffix(data_path.suffix + ".tmp")
        with h5py.File(tmp_data, "w") as handle:
            dset = handle.create_dataset(
                "entry/data/data",
                shape=(end - start, det_row, det_col),
                dtype=dtype,
                chunks=(1, det_row, det_col),
                **_portable_h5_filters(compression, compression_level, shuffle),
            )
            dset.attrs["image_nr_low"] = np.uint64(start + 1)
            dset.attrs["image_nr_high"] = np.uint64(end)
            if use_mps_backend:
                _raise_write_error()
                _ensure_writer_thread()
                frame_bytes = det_row * det_col * dtype.itemsize
                n_8kb = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE
                for batch_start in range(start, end, batch_size):
                    batch_end = min(batch_start + batch_size, end)
                    packed, chunk_starts, chunk_sizes = compress_tensor_batch(
                        flat[batch_start:batch_end], n_8kb, frame_bytes, dtype,
                    )
                    _write_queue.put((
                        _write_batch_to_h5,
                        (
                            dset,
                            packed,
                            chunk_starts,
                            chunk_sizes,
                            batch_start - start,
                            batch_end - batch_start,
                        ),
                    ))
                wait_for_saves()
            else:
                out_offset = 0
                for batch_start in range(start, end, batch_size):
                    batch_end = min(batch_start + batch_size, end)
                    dset[out_offset:out_offset + (batch_end - batch_start)] = (
                        _cast_host_for_save(flat[batch_start:batch_end], dtype)
                    )
                    out_offset += batch_end - batch_start
        tmp_data.replace(data_path)

    tmp_master = filepath.with_suffix(filepath.suffix + ".tmp")
    _write_master_file(
        tmp_master,
        data_files,
        frame_ranges,
        scan_shape,
        det_shape,
        dtype,
        metadata,
        source_master,
    )
    tmp_master.replace(filepath)


def wait_for_saves() -> None:
    """Block until queued HDF5 write jobs finish, then raise write errors."""
    _write_queue.join()
    _raise_write_error()


def _raise_write_error():
    """Re-raise, once, an error the background HDF5 writer recorded."""
    global _write_error
    if _write_error is None:
        return
    err = _write_error
    _write_error = None
    raise RuntimeError("Background HDF5 write failed") from err


def _writer_loop():
    """Run queued HDF5 writes on the background thread.

    Errors cannot rise from this thread, so the first one is recorded for
    :func:`wait_for_saves` to raise in the caller.
    """
    global _write_error
    while True:
        job = _write_queue.get()
        try:
            if job is None:
                return
            func, args = job
            func(*args)
        except BaseException as exc:  # propagate on wait_for_saves()
            _write_error = exc
        finally:
            _write_queue.task_done()


def _ensure_writer_thread():
    """Start the background HDF5 writer if it is not already running."""
    global _write_thread
    with _write_thread_lock:
        if _write_thread is None or not _write_thread.is_alive():
            _write_thread = threading.Thread(target=_writer_loop, daemon=True)
            _write_thread.start()


def _write_batch_to_h5(ds, packed, chunk_starts, chunk_sizes, frame_offset,
                       n_frames):
    """Write packed compressed chunks to an open flattened HDF5 dataset."""
    packed_mv = memoryview(packed)
    for i in range(n_frames):
        start = int(chunk_starts[i])
        size = int(chunk_sizes[i])
        ds.id.write_direct_chunk(
            (frame_offset + i, 0, 0), packed_mv[start:start + size]
        )


def _write_batch_via_filter(ds, host_frames, frame_offset, n_frames):
    """Write a CPU host-side frame batch through the dataset's filter pipeline.

    Used for non-LZ4 codecs (zstd, blosc2_zstd) where compression runs on CPU
    inside HDF5 instead of in our GPU pipeline. ``host_frames`` is a contiguous
    numpy array of shape (n_frames, det_row, det_col).
    """
    ds[frame_offset:frame_offset + n_frames] = host_frames


def _write_master_file(
    master_path,
    data_files,
    frame_ranges,
    scan_shape,
    det_shape,
    dtype,
    metadata,
    source_master,
):
    """Write the master file that links every external data file.

    Readers find the frames, scan and detector shape, and frame ranges here;
    ``source_master`` carries over the original acquisition metadata.
    """
    master_path = Path(master_path)
    if source_master is not None:
        _copy_master_shell(source_master, master_path)
        mode = "r+"
    else:
        mode = "w"

    with h5py.File(master_path, mode) as f:
        entry = f.require_group("entry")
        if "data" in entry:
            del entry["data"]
        data_group = entry.create_group("data")
        if source_master is not None:
            with h5py.File(source_master, "r") as src:
                src_data = src.get("entry/data")
                if src_data is not None:
                    for key, val in src_data.attrs.items():
                        data_group.attrs[key] = val
        else:
            data_group.attrs["NX_class"] = np.bytes_(b"NXdata")
        data_group.attrs["signal"] = np.bytes_(b"data_000001")
        for index, data_file in enumerate(data_files, start=1):
            data_group[f"data_{index:06d}"] = h5py.ExternalLink(
                Path(data_file).name, "/entry/data/data"
            )
        if scan_shape is not None:
            data_group.attrs["scan_shape"] = tuple(int(x) for x in scan_shape)
            f.attrs["scan_shape"] = tuple(int(x) for x in scan_shape)
        data_group.attrs["det_shape"] = tuple(int(x) for x in det_shape)
        data_group.attrs["dtype"] = str(np.dtype(dtype))
        data_group.attrs["n_frames"] = int(sum(hi - lo + 1 for lo, hi in frame_ranges))
        data_group.attrs["data_file_frame_ranges"] = np.asarray(frame_ranges, dtype=np.uint64)
        f.attrs["detector_shape"] = tuple(int(x) for x in det_shape)
        f.attrs["dtype"] = str(np.dtype(dtype))
        f.attrs["n_frames"] = int(sum(hi - lo + 1 for lo, hi in frame_ranges))
        _metadata_attrs(f, metadata)


def _copy_master_shell(source_master, output_master):
    """Copy source Arina master metadata, excluding entry/data links."""
    with h5py.File(source_master, "r") as src, h5py.File(output_master, "w") as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        for name in src:
            if name != "entry":
                src.copy(name, dst, name=name, expand_external=False)
        if "entry" in src:
            src_entry = src["entry"]
            dst_entry = dst.require_group("entry")
            for key, val in src_entry.attrs.items():
                dst_entry.attrs[key] = val
            for name in src_entry:
                if name == "data":
                    continue
                src.copy(src_entry[name], dst_entry, name=name, expand_external=False)
        else:
            dst.require_group("entry")


def _metadata_attrs(f, metadata):
    """Write metadata as HDF5 attributes, storing unsupported values as text."""
    if metadata is None:
        return
    for key, val in metadata.items():
        if val is None:
            continue
        try:
            f.attrs[key] = val
        except (TypeError, ValueError):
            f.attrs[key] = str(val)


def _master_prefix(master_path):
    """Return the shared file prefix, so ``x_master.h5`` names ``x_data_000001.h5``."""
    return master_path.stem.removesuffix("_master")


def _output_file_size(master_path):
    """Return the bytes on disk of a master file plus its external data files."""
    master_path = Path(master_path)
    prefix = _master_prefix(master_path)
    total = master_path.stat().st_size if master_path.exists() else 0
    for data_path in master_path.parent.glob(f"{prefix}_data_*.h5"):
        total += data_path.stat().st_size
    return total


def _normalize_compression(compression, compression_level):
    """Return ``(codec, level)`` for H5Writer, rejecting levels LZ4 cannot use.

    The GPU LZ4 path has a fixed level; accepting a level there would let a
    caller believe a setting took effect when it did not.
    """
    if compression is None:
        compression = "lz4"
    compression = str(compression).lower()
    if compression not in _COMPRESSION_CODECS:
        raise ValueError(
            f"Unsupported compression {compression!r}; choose from "
            f"{_COMPRESSION_CODECS}."
        )
    if compression == "lz4":
        # LZ4 path is GPU bitshuffle+LZ4; level is fixed in the kernel.
        if compression_level not in (None, 0):
            raise ValueError(
                "compression_level is only meaningful for zstd / blosc2_zstd; "
                "leave it 0 for the default LZ4 path."
            )
        return compression, 0
    level = compression_level if compression_level else _DEFAULT_CLEVELS[compression]
    return compression, int(level)


def _hdf5_filter(compression, compression_level):
    """Return the hdf5plugin filter mapping for a given codec+level."""
    if compression == "lz4":
        return hdf5plugin.Bitshuffle(cname="lz4")
    if compression == "zstd":
        return hdf5plugin.Bitshuffle(cname="zstd", clevel=compression_level)
    if compression == "blosc2_zstd":
        return hdf5plugin.Blosc2(
            cname="zstd",
            clevel=compression_level,
            filters=hdf5plugin.Blosc2.BITSHUFFLE,
        )
    raise ValueError(f"Unsupported compression {compression!r}")


def _portable_h5_filters(compression, compression_level, shuffle):
    """Return ``create_dataset`` filter keywords readable by standard HDF5 tools."""
    if compression is None:
        return {}
    compression = str(compression).lower()
    if compression in {"lz4", "bslz4", "bitshuffle_lz4"}:
        return hdf5plugin.Bitshuffle(cname="lz4")
    if compression == "zstd":
        level = 3 if compression_level is None else int(compression_level)
        return hdf5plugin.Bitshuffle(cname="zstd", clevel=level)
    if compression == "gzip":
        if compression_level is None:
            compression_level = 4
        return {
            "compression": "gzip",
            "compression_opts": int(compression_level),
            "shuffle": bool(shuffle),
        }
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": bool(shuffle)}
    raise ValueError(
        "portable compressed HDF5 supports compression='lz4', 'zstd', "
        "'gzip', 'lzf', or None"
    )


def _normalize_save_dtype(dtype):
    """Resolve a dtype or short alias such as ``"u16"`` to a supported save dtype."""
    if isinstance(dtype, str):
        dtype = _DTYPE_ALIASES.get(dtype.lower(), dtype)
    dtype = np.dtype(dtype)
    if dtype not in _SAVE_DTYPES:
        raise ValueError(
            "save() supports float32 for canonical drift-corrected 4D-STEM "
            "archives, with integer dtypes reserved for explicit raw-data, "
            "compatibility, or display exports."
        )
    return dtype


def _default_save_dtype(data_dtype):
    """Return the storage dtype that preserves the input values exactly.

    Floating inputs store as float32; supported unsigned counts keep their
    width, so an omitted ``dtype`` never changes measurements.
    """
    data_dtype = np.dtype(data_dtype)
    if np.issubdtype(data_dtype, np.floating):
        return np.dtype(np.float32)
    if data_dtype == np.dtype(np.uint8):
        return np.dtype(np.uint8)
    if data_dtype == np.dtype(np.uint16):
        return np.dtype(np.uint16)
    if data_dtype == np.dtype(np.uint32):
        return np.dtype(np.uint32)
    raise ValueError(
        f"Unsupported input dtype {data_dtype}; pass dtype=np.float32, "
        "np.uint8, np.uint16, or np.uint32 explicitly."
    )


def _infer_save_dtype(data):
    """Return the default output dtype for a public save call."""
    if torch is not None and torch.is_tensor(data):
        torch_to_numpy = {
            torch.uint8: np.uint8,
            torch.uint16: np.uint16,
            torch.uint32: np.uint32,
            torch.float32: np.float32,
        }
        try:
            return _default_save_dtype(torch_to_numpy[data.dtype])
        except KeyError as exc:
            raise ValueError(
                f"Unsupported input dtype {data.dtype}; pass dtype='u16', "
                "'u8', 'u32', or 'f32' explicitly."
            ) from exc
    return _default_save_dtype(np.asarray(data).dtype)


def _cast_host_for_save(block, dtype):
    """Return a contiguous host batch in the save dtype.

    Floats saved as integers are rounded to nearest and clipped first, because
    a plain cast truncates and wraps out-of-range values.
    """
    if cp is not None and isinstance(block, cp.ndarray):
        arr = cp.asnumpy(block)
    elif torch is not None and torch.is_tensor(block):
        arr = block.detach().cpu().numpy()
    else:
        arr = np.asarray(block)
    if arr.dtype == dtype:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        lo, hi = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
        arr = np.clip(np.rint(arr), lo, hi).astype(dtype)
    else:
        arr = arr.astype(dtype)
    return np.ascontiguousarray(arr)


def _is_mps_array(data) -> bool:
    """Return True for a Torch MPS tensor without importing torch on CUDA hosts."""
    return torch is not None and torch.is_tensor(data) and data.device.type == "mps"
