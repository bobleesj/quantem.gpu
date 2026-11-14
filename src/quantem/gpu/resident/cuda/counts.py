"""Bounded, lossless CUDA encoding with runtime detector geometry.

This private resident profile uses byte rANS, sparse events, constants and uint16 literal
streams. Spatial sums are derived from original counts while each input chunk is
available. Neither coding probabilities nor indexes change scientific values.
"""

import gc
import math
from dataclasses import dataclass
from functools import cache
from itertools import accumulate
from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_module
from quantem.gpu.formats.qem.reference import count_tables
from quantem.gpu.formats.qem.snapshot import field_count

_SOURCE = (Path(__file__).with_name("kernels") / "streamed.cu").read_text()
_NAMES = (
    "encode",
    "compact",
    "decode_range_u8",
    "decode_range_u16",
    "columns_u16",
    "columns_fft",
    "total",
    "fields",
    "field_roots",
    "field_sizes",
    "pack_fields",
    *(f"{op}_u{bits}" for op in ("index", "residual") for bits in (32, 64)),
    "moment_pair",
    "selected",
    "frame_u8",
    "frame_u16",
)


def kernels(device: int) -> dict:
    """Return the codec and spatial-index kernels compiled for ``device``, by short name."""
    import cupy as cp

    with cp.cuda.Device(device):
        functions = cuda_module(_SOURCE, tuple(f"sc_{name}" for name in _NAMES), ("--std=c++17",))
    return {name: functions[f"sc_{name}"] for name in _NAMES}


@cache
def tables(device: int):
    """Upload the shared rANS encoding and decoding tables (``count_tables``) to ``device``."""
    import cupy as cp

    encoding, decoding = count_tables()
    # kept for the life of the process, so outside the memory pool (see retain)
    with cp.cuda.Device(device), cp.cuda.using_allocator():
        return cp.asarray(encoding), cp.asarray(decoding)


def retain(arrays, blocks: "Blocks | None" = None, expected: int = 0) -> tuple:
    """Copy the arrays a resident keeps into memory outside the memory pool, back to back at 256-byte offsets.

    The arrays are returned as views. A kept array carved from a pooled block
    that also served decode or encode scratch holds the whole block until the
    resident is released, so neither that scratch nor, after release, the
    resident returns to the GPU. Without ``blocks`` the arrays get one exact
    allocation of their own; with ``blocks`` they are placed in its current
    block, or in a new one sized to the ``expected`` bytes still to come.
    """
    import cupy as cp

    sizes = [math.ceil(array.nbytes / 256) * 256 for array in arrays]
    storage = _allocate(sum(sizes)) if blocks is None else blocks.take(sum(sizes), expected)
    if storage is None:
        raise MemoryError(f"The device has no room for {sum(sizes)} bytes of encoded counts.")
    views = []
    for array, start in zip(arrays, [0, *accumulate(sizes)]):
        view = storage[start : start + array.nbytes].view(array.dtype).reshape(array.shape)
        if isinstance(array, np.ndarray):
            view.set(array)  # a parked chunk returning from host memory (see StreamedCounts.restore)
        else:
            view[...] = array
        views.append(view)
    # The sources go back to the memory pool when the caller drops them, and the
    # loader's next decode can reuse that memory before these copies have run;
    # without this wait, loads under GPU contention kept another chunk's counts.
    cp.cuda.get_current_stream().synchronize()
    return tuple(views)


class Blocks:
    """Allocations outside the memory pool that a resident's chunks are packed into, back to back.

    The device rounds every allocation up to whole 2 MiB pages: 100 allocations of 8.45 MiB took 1000 MiB, so one
    allocation per chunk lost about 1 MiB a chunk, 260 to 400 MiB over the 256 chunks of a 2.2 GB encoded scan, and on
    a 4 GB card that scan's load ran out on its last chunk (2026-11-13). A block holds up to ``limit`` bytes, sized to
    what is still expected, so only the tail a chunk did not fit in is lost. A block returns to the device when its
    last chunk is freed, so a source read once in scan order shrinks a block at a time. With ``device`` set, a block
    the device has no room for, and every block after it, is page-locked host memory the device reads in place
    (``host_mapped``); without it, ``take`` returns None there.
    """

    def __init__(self, limit: int | None = 256 << 20, device: int | None = None):
        self.limit = limit
        self.device = device
        self.block = None
        self.used = 0
        self.on_host = False

    def take(self, nbytes: int, expected: int):
        """``nbytes`` of memory from the current block, or from a new one of the ``expected`` bytes (at least
        ``nbytes``, at most ``limit``) when the current block has no room."""
        if self.block is None or self.used + nbytes > self.block.nbytes:
            size = max(nbytes, expected if self.limit is None else min(self.limit, expected))
            size = math.ceil(size / (2 << 20)) * (2 << 20)  # the pages the device allocates anyway
            self.block = None if self.on_host else _allocate(size)
            if self.block is None:
                if self.device is None:
                    return None
                self.on_host = True
                self.block = host_mapped(pinned(size), self.device)
            self.used = 0
        storage = self.block[self.used : self.used + nbytes]
        self.used += nbytes
        return storage


def _allocate(nbytes: int):
    """One exact allocation of ``nbytes`` outside the memory pool (see retain), or None when the device has no room."""
    import cupy as cp

    for attempt in range(2):
        try:
            with cp.cuda.using_allocator():
                return cp.empty(nbytes, cp.uint8)
        except cp.cuda.runtime.CUDARuntimeError as error:
            if "memory" not in str(error).lower():
                raise
        # the pool's idle blocks (a loader's scratch from earlier batches) hold the room; on a 4 GB card a 2.2 GB
        # scan loading from HDF5 ran out here at 3700 MiB free while the pool kept hundreds of MiB idle
        cp.get_default_memory_pool().free_all_blocks()
    return None


def pinned(nbytes: int) -> np.ndarray:
    """``nbytes`` of page-locked host memory as a uint8 array."""
    import cupy as cp

    return np.frombuffer(cp.cuda.alloc_pinned_memory(max(nbytes, 1)), np.uint8, nbytes)


def host_mapped(host: np.ndarray, device: int):
    """A page-locked host array as a device array that kernels read and write in place, over the bus.

    Under unified addressing a kernel dereferences a page-locked host pointer directly, so an encoded scan the device
    has no room for still decodes, slower: on a 512 x 512 x 192 x 192 scan (RTX PRO 6000, PCIe 5.0 x8, 2026-11-13) a
    whole-scan phase took 0.96 s against 0.20 s from device memory, a new virtual image 237 ms against 5 ms, a 1-pixel
    detector drag 53 ms against 2.3 ms, one pattern 4 ms against 2.4 ms, with the same bits.
    """
    import cupy as cp

    if not _unified_addressing(device):
        raise MemoryError("The device has no room for the encoded counts, and it cannot read them from host memory "
                          "(no unified addressing).")
    memory = cp.cuda.UnownedMemory(host.ctypes.data, max(host.nbytes, 1), host, device)
    return cp.ndarray(host.shape, host.dtype, cp.cuda.MemoryPointer(memory, 0))


@cache
def _unified_addressing(device: int) -> bool:
    """Whether kernels on ``device`` read page-locked host memory by its own address (asked once: a restore that maps
    256 chunks asked 1536 times and took 3 s)."""
    import cupy as cp

    return bool(cp.cuda.runtime.getDeviceProperties(device)["unifiedAddressing"])


def on_host(array) -> bool:
    """Whether a device array lives in page-locked host memory (``host_mapped``)."""
    import cupy as cp

    host = cp.cuda.runtime.memoryTypeHost
    return bool(array.size) and cp.cuda.runtime.pointerGetAttributes(array.data.ptr).type == host


class Staging:
    """One device buffer that chunks held in host memory (``host_mapped``) are copied into, a chunk at a time.

    For a pass that decodes every stream of every chunk: read in place over the bus, the thousands of short streams
    are latency bound (a detector total of a 2.5 GB scan took 8.7 s, its centre-of-mass moments 8.6 s); one bulk copy
    of each chunk first runs at the bus's rate. One buffer the size of the largest chunk serves every chunk: copies
    and kernels queue on one stream, so a chunk's copy waits for the kernels that read the one before (2026-11-13).
    """

    def __init__(self, chunks):
        import cupy as cp

        sizes = [sum(math.ceil(array.nbytes / 256) * 256 for array in arrays)
                 for arrays in chunks if any(on_host(array) for array in arrays)]
        self.buffer = cp.empty(max(sizes), cp.uint8) if sizes else None

    def __call__(self, arrays) -> tuple:
        """``arrays`` on the device: those in host memory copied into the buffer, valid until the next call."""
        import cupy as cp

        if self.buffer is None:
            return tuple(arrays)
        staged, offset = [], 0
        for array in arrays:
            if on_host(array):
                view = self.buffer[offset : offset + array.nbytes].view(array.dtype).reshape(array.shape)
                cp.cuda.runtime.memcpyAsync(view.data.ptr, array.data.ptr, array.nbytes, cp.cuda.runtime.memcpyDefault,
                                            cp.cuda.get_current_stream().ptr)
                offset += math.ceil(array.nbytes / 256) * 256
                array = view
            staged.append(array)
        return tuple(staged)


def _to_host(array) -> np.ndarray:
    """Copy one device array into page-locked host memory.

    A parked acquisition is gigabytes: into ordinary host memory the copy of 2.4 GB took 4 to 38 s across runs,
    into page-locked memory 0.11 s after 1 to 2 s locking the pages (2026-11-12).
    """
    import cupy as cp

    if not array.size:
        return np.empty(array.shape, array.dtype)
    memory = cp.cuda.alloc_pinned_memory(array.nbytes)
    host = np.frombuffer(memory, array.dtype, array.size).reshape(array.shape)
    array.get(out=host)
    return host


def error_flags(errors, device: int):
    """The caller's decode-failure flags, or new zeroed ones when the caller passes none.

    Decode kernels OR their stream failures into one uint32 on the source
    device. A caller that passes its own flags decodes several ranges and checks
    them once, so no decode synchronizes; a decode given none checks its own.
    """
    import cupy as cp

    if errors is None:
        return cp.zeros(1, cp.uint32)
    if (
        not isinstance(errors, cp.ndarray)
        or errors.shape != (1,)
        or errors.dtype != cp.uint32
        or errors.device.id != device
    ):
        raise ValueError("errors must be one uint32 value on the source CUDA device.")
    return errors


@dataclass
class Chunk:
    """Own one bounded count payload and its exact bitpacked spatial sums."""

    first: int
    scans: int
    arrays: tuple

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.arrays)


class StreamedCounts:
    """Keep complete native counts in independent, bounded encoded chunks."""

    interval = 512
    # Exact uint64 sum over the scans appended so far, per detector pixel (before the valid-pixel
    # mask), or None until it is known: ``append`` adds each chunk's counts while it encodes them,
    # and a source whose chunks came from a file adds its streams at the first request.
    _detector_total = None
    # Exact spatial sums and the detector total, for detector queries; float bit lanes have no use for them.
    spatial_index = True
    # The encoded chunks as host arrays while they are parked off the device (see park), else None.
    _parked = None
    # The blocks ``append`` packs chunks into until the last scan arrives, and the bytes it has encoded so far.
    _blocks = None
    _appended_bytes = 0

    def __init__(self, shape: tuple[int, int, int, int], dtype, valid=None):
        import cupy as cp

        self.shape = tuple(map(int, shape))
        self.dtype = np.dtype(dtype)
        if (
            len(self.shape) != 4
            or min(self.shape) < 1
            or self.dtype not in (np.dtype("uint8"), np.dtype("uint16"))
        ):
            raise ValueError(
                "Streamed counts require a positive 4D uint8/uint16 acquisition."
            )
        self.device = cp.cuda.Device().id
        self.valid_pixels = (
            np.ones(self.shape[2:], bool)
            if valid is None
            else np.asarray(valid, bool).copy()
        )
        if self.valid_pixels.shape != self.shape[2:]:
            raise ValueError("Detector validity must match the native detector shape.")
        with cp.cuda.using_allocator():  # retained, so outside the memory pool (see retain)
            self.valid = cp.asarray(self.valid_pixels, dtype=cp.uint8)
        self.encoding, self.decoding = tables(self.device)
        self.kernels = kernels(self.device)
        self.chunks: list[Chunk] = []
        self.is_released = False
        self.ready_scans = 0
        self.released_scans = 0
        self.load_metrics = {"encode_seconds": 0.0, "index_seconds": 0.0}

    @property
    def nbytes(self) -> int:
        total = 0 if self._detector_total is None else self._detector_total.nbytes
        return sum(chunk.nbytes for chunk in self.chunks) + self.valid.nbytes + total

    @property
    def host_nbytes(self) -> int:
        """Bytes of the encoded chunks the device reads from page-locked host memory (``host_mapped``): those a load
        or a ``restore`` found no device room for."""
        return sum(array.nbytes for chunk in self.chunks for array in chunk.arrays if on_host(array))

    @property
    def index_nbytes(self) -> int:
        return sum(sum(array.nbytes for array in chunk.arrays[3:]) for chunk in self.chunks)

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "Encoded counts stay on CUDA; request a detector product or an explicit bounded decode."
        )

    def append(self, raw) -> None:
        """Encode a consecutive bounded scan chunk and derive its spatial sums."""
        import cupy as cp

        if self.is_released:
            raise ValueError("The resident source has been released.")
        if (
            not isinstance(raw, cp.ndarray)
            or raw.device.id != self.device
            or raw.dtype != self.dtype
            or not raw.flags.c_contiguous
        ):
            raise ValueError(
                "Provide contiguous native counts on the source CUDA device."
            )
        if raw.ndim != 3 or raw.shape[1:] != self.shape[2:]:
            raise ValueError(
                "Chunk detector geometry must match the complete acquisition."
            )
        scans, pixels = raw.shape[0], math.prod(self.shape[2:])
        if scans < 1 or self.ready_scans + scans > math.prod(self.shape[:2]):
            raise ValueError("Chunk scan coverage exceeds the declared acquisition.")
        streams = math.ceil(scans / self.interval) * pixels
        # Offset arithmetic is local to each chunk and must fit uint32.
        if streams * (2 * min(scans, self.interval) + 4) >= 2**32:
            raise ValueError("Use smaller chunks so encoded offsets fit uint32.")
        u32 = np.uint32
        with cp.cuda.Device(self.device):
            # Events time each phase on the stream; host waits would only stall it.
            begin, encoded, indexed = cp.cuda.Event(), cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            # This chunk's counts join the kept total only once the chunk is held, so an append that
            # fails part way leaves the total as it was. A source that holds scans without a total
            # (chunks read from a file, or scans freed since) has none to add to.
            keeps_total = self.spatial_index and (self._detector_total is not None or not self.ready_scans)
            counted = cp.zeros(self.shape[2:], cp.uint64) if keeps_total else np.intp(0)
            scratch = cp.empty((2 * min(scans, self.interval) + 4, streams), cp.uint8)
            sizes, states = cp.empty(streams, cp.uint32), cp.empty(streams, cp.uint32)
            models = cp.empty(streams, cp.uint8)
            grid = ((streams + 127) // 128,)
            args = (
                raw,
                np.int32(raw.dtype.itemsize),
                u32(scans),
                u32(pixels),
                u32(self.interval),
            )
            self.kernels["encode"](
                grid,
                (128,),
                (*args, self.encoding, scratch, sizes, states, models, u32(streams), counted),
            )
            offsets = cp.empty(streams + 1, cp.uint32)
            offsets[0] = 0
            cp.cumsum(sizes, dtype=cp.uint32, out=offsets[1:])
            payload = cp.empty(int(offsets[-1].get()), cp.uint8)
            self.kernels["compact"](
                grid,
                (128,),
                (*args, scratch, offsets, states, models, payload, u32(streams)),
            )
            encoded.record()
            kept = (payload, offsets, models)
            if self.spatial_index:
                # Return the codec scratch to the pool before the index allocates its own.
                del scratch, sizes, states
                fields = field_count(self.shape[2:])
                leaves = math.ceil(self.shape[2] / 8) * math.ceil(self.shape[3] / 8)
                values = cp.empty((scans, fields), cp.uint32)
                geometry = (u32(scans), u32(self.shape[2]), u32(self.shape[3]), u32(fields))
                self.kernels["fields"](
                    ((scans * leaves + 127) // 128,),
                    (128,),
                    (raw, np.int32(raw.dtype.itemsize), self.valid, values, *geometry),
                )
                self.kernels["field_roots"](
                    ((scans * (fields - leaves) + 127) // 128,), (128,), (values, *geometry)
                )
                index_streams = math.ceil(scans / self.interval) * fields
                widths, lengths = (
                    cp.empty(index_streams, cp.uint8),
                    cp.empty(index_streams, cp.uint64),
                )
                index_args = (u32(scans), u32(fields), u32(self.interval))
                index_grid = ((index_streams + 127) // 128,)
                self.kernels["field_sizes"](index_grid, (128,), (values, widths, lengths, *index_args))
                starts = cp.empty(index_streams + 1, cp.uint64)
                starts[0] = 0
                cp.cumsum(lengths, dtype=cp.uint64, out=starts[1:])
                words = cp.empty(int(starts[-1].get()), cp.uint32)
                self.kernels["pack_fields"](
                    index_grid, (128,), (values, widths, starts, words, *index_args)
                )
                kept += (words, starts, widths)
            indexed.record()
            # A new block holds the scans still to come at the bytes per scan so far: a short guess costs one more
            # block, a long one leaves memory unused at the end.
            self._appended_bytes += sum(math.ceil(array.nbytes / 256) * 256 for array in kept)
            remaining = math.prod(self.shape[:2]) - self.ready_scans
            expected = self._appended_bytes * remaining // (self.ready_scans + scans)
            self._blocks = self._blocks or Blocks(device=self.device)
            # retain() waits for the stream, so the events are complete when read.
            self.chunks.append(Chunk(self.ready_scans, scans, retain(kept, self._blocks, expected)))
            self.ready_scans += scans
            if self.ready_scans == math.prod(self.shape[:2]):
                self._blocks = None  # the chunks alone hold their blocks, so freeing them frees the memory
            if keeps_total:
                if self._detector_total is None:
                    with cp.cuda.using_allocator():  # retained, so outside the memory pool (see retain)
                        self._detector_total = cp.zeros(self.shape[2:], cp.uint64)
                self._detector_total += counted
            self.load_metrics["encode_seconds"] += cp.cuda.get_elapsed_time(begin, encoded) / 1e3
            self.load_metrics["index_seconds"] += cp.cuda.get_elapsed_time(encoded, indexed) / 1e3

    def decode_scan_range_device(
        self, first: int, stop: int, *, errors=None, detector_region=None
    ):
        """Decode a contiguous scan range without expanding the acquisition."""
        import cupy as cp

        first, stop = int(first), int(stop)
        self.check_readable(first, stop)
        owns_errors = errors is None
        with cp.cuda.Device(self.device):
            errors = error_flags(errors, self.device)
            pixels = math.prod(self.shape[2:])
            if detector_region is None:
                detector_region = (0, self.shape[2], 0, self.shape[3])
            row0, row1, col0, col1 = map(int, detector_region)
            if not (0 <= row0 < row1 <= self.shape[2]
                    and 0 <= col0 < col1 <= self.shape[3]):
                raise ValueError("Detector region must lie within the detector shape.")
            crop_rows, crop_cols = row1 - row0, col1 - col0
            selected_pixels = crop_rows * crop_cols
            output = cp.empty((stop - first, crop_rows, crop_cols), self.dtype)
            for chunk in self.chunks:
                overlap_first = max(first, chunk.first)
                overlap_stop = min(stop, chunk.first + chunk.scans)
                if overlap_first >= overlap_stop:
                    continue
                local_first = overlap_first - chunk.first
                local_stop = overlap_stop - chunk.first
                first_stream = (local_first // self.interval) * selected_pixels
                stop_stream = math.ceil(local_stop / self.interval) * selected_pixels
                result = output[overlap_first - first : overlap_stop - first]
                self.kernels[f"decode_range_u{self.dtype.itemsize * 8}"](
                    ((stop_stream - first_stream + 127) // 128,),
                    (128,),
                    (
                        *chunk.arrays[:3],
                        self.decoding,
                        result,
                        errors,
                        np.uint32(chunk.scans),
                        np.uint32(pixels),
                        np.uint32(self.interval),
                        np.uint32(local_first),
                        np.uint32(local_stop - local_first),
                        np.uint32(first_stream),
                        np.uint32(stop_stream),
                        np.uint32(self.shape[3]),
                        np.uint32(row0),
                        np.uint32(col0),
                        np.uint32(crop_rows),
                        np.uint32(crop_cols),
                    ),
                )
            if owns_errors and int(errors.get()[0]):
                raise ValueError("An encoded count stream failed reconstruction.")
            return output

    def check_readable(self, first: int, stop: int) -> None:
        """Refuse a scan range this source does not hold.

        A decode writes only the scans its chunks hold, so a scan not yet
        appended (an acquisition still arriving), freed by
        ``release_scans_before`` or dropped by ``release`` would come back as
        whatever the output memory held before.
        """
        if self.is_released:
            raise ValueError("The resident source has been released.")
        if not 0 <= first < stop <= self.ready_scans:
            raise ValueError(
                f"Scan range ({first}, {stop}) is not resident; scans 0 to "
                f"{self.ready_scans} of {math.prod(self.shape[:2])} are loaded."
            )
        if self._parked is not None:
            if not self.chunks or first < self.chunks[0].first or stop > self.chunks[-1].first + self.chunks[-1].scans:
                raise ValueError(
                    f"Scan range ({first}, {stop}) is parked in host memory; restore() it to the device first."
                )
            return
        if first < self.released_scans:
            raise ValueError(
                f"Scans before {self.released_scans} were released; load the source again to read them."
            )

    def detector_total_device(self):
        """Return the exact valid-pixel detector sum of the whole scan without dense expansion.

        ``append`` already added the counts it encoded. A source whose chunks were read from a file
        adds its streams here once (each thread sums one stream, nothing is written out) and keeps
        the total: the chunks never change. The sum, and the mean pattern divided from it, cover
        every scan position, so a source with scans still arriving or freed raises, as a read does.
        """
        import cupy as cp

        self.check_readable(0, math.prod(self.shape[:2]))
        with cp.cuda.Device(self.device):
            if self._detector_total is None:
                with cp.cuda.using_allocator():  # retained, so outside the memory pool (see retain)
                    total = cp.zeros(self.shape[2:], cp.uint64)
                errors = cp.zeros(1, cp.uint32)
                pixels = math.prod(self.shape[2:])
                staging = Staging([chunk.arrays[:3] for chunk in self.chunks])
                for chunk in self.chunks:
                    streams = math.ceil(chunk.scans / self.interval) * pixels
                    self.kernels["total"](
                        ((streams + 127) // 128,),
                        (128,),
                        (
                            *staging(chunk.arrays[:3]),
                            self.decoding,
                            total,
                            errors,
                            np.uint32(chunk.scans),
                            np.uint32(pixels),
                            np.uint32(self.interval),
                        ),
                    )
                if int(errors.get()[0]):
                    raise ValueError("An encoded count stream failed reconstruction.")
                self._detector_total = total
            return self._detector_total * self.valid

    def release_scans_before(self, stop: int) -> None:
        """Free the chunks that end by scan ``stop``, so a source read once in scan order shrinks as it is read.

        Only whole chunks are freed, and their memory returns to the GPU once
        every chunk sharing its block is freed (see Blocks). Decoding a freed
        scan raises.
        """
        kept = [chunk for chunk in self.chunks if chunk.first + chunk.scans > stop]
        if len(kept) != len(self.chunks):
            self._detector_total = None   # no request can be answered once scans are freed
        self.chunks = kept
        self.released_scans = self.chunks[0].first if self.chunks else self.ready_scans

    def park(self) -> None:
        """Move every encoded chunk to host memory and free it on the device, until ``restore``.

        The bytes stay encoded and are not read on the host: they wait there while another computation needs the
        device (a small card cannot hold a whole encoded scan and a fit's working arrays together). A second call
        frees the chunks a partial ``restore`` returned. The device memory returns only when no other consumer
        still holds the chunks, so park a source its session alone owns. Decoding a parked scan raises.
        """
        import cupy as cp

        if self._parked is None:
            with cp.cuda.Device(self.device):
                # chunks already in host memory (``host_mapped``) stay where they are
                self._parked = [
                    Chunk(chunk.first, chunk.scans,
                          tuple(array if on_host(array) else _to_host(array) for array in chunk.arrays))
                    for chunk in self.chunks
                ]
        self.chunks = []
        self._blocks = None
        # Chunks a finished decode or detector pass still refers to through a reference cycle keep their device
        # memory until the collector next runs: on a 3400 MiB card the fit that parked the scan for room then ran
        # out in about every second run, the whole scan (2.5 GB) still there. Collected now, the room is there.
        gc.collect()

    def restore(self, first: int = 0, stop: int | None = None, *, keep_host: bool = False) -> None:
        """Return parked chunks to the device: those holding scans ``first`` to ``stop``, or all of them.

        A partial restore keeps the host copies, so another range or the whole scan can follow; restoring the
        whole scan drops them unless ``keep_host``, with which the next ``park`` only frees the device. The chunks
        return into one exact allocation (see Blocks); when the device has no room for them they stay in host memory
        and the device reads them there (``host_mapped``, slower; ``host_nbytes`` says how much).
        """
        import cupy as cp

        if self._parked is None:
            return
        complete = first <= 0 and (stop is None or stop >= self.ready_scans)
        stop = self.ready_scans if stop is None else stop
        self.chunks = []
        wanted = [chunk for chunk in self._parked if chunk.first < stop and chunk.first + chunk.scans > first]
        total = sum(math.ceil(array.nbytes / 256) * 256 for chunk in wanted for array in chunk.arrays)
        with cp.cuda.Device(self.device):
            try:
                blocks = Blocks(limit=None)
                self.chunks = [Chunk(chunk.first, chunk.scans, retain(chunk.arrays, blocks, total)) for chunk in wanted]
            except MemoryError:
                try:
                    self.chunks = [Chunk(chunk.first, chunk.scans, tuple(
                        array if isinstance(array, cp.ndarray) else host_mapped(array, self.device)
                        for array in chunk.arrays)) for chunk in wanted]
                except MemoryError as error:
                    self.chunks = []
                    raise MemoryError(
                        f"The device has no room to take back {sum(chunk.nbytes for chunk in wanted) / 1e9:.2f} GB of "
                        "the encoded acquisition. It is intact in host memory: free device memory (close other "
                        "notebooks' kernels, or restart this one), then call restore() again."
                    ) from error
        if complete and not keep_host:
            self._parked = None
            # the page-locked host copies would otherwise wait in the pinned pool, as host memory nothing else can use
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def release(self) -> None:
        """Drop this owner's references without invalidating active borrowed sessions."""
        self._parked = None
        self._blocks = None
        self.chunks.clear()
        self._detector_total = None
        self.is_released = True
