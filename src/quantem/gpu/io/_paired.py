"""Original bitshuffle+LZ4 HDF5 to the paired resident layout at the drive's rate.

Reader threads read whole shards with direct I/O into page-locked staging and
parse the LZ4 block headers there. One producer thread only enqueues device work:
host-to-device copy of the compressed bytes, LZ4 decode, bitshuffle straight into
a ring of rolling frame buffers. A filled buffer is handed to the consumer with an
event; the consumer encodes and indexes it on a second stream and returns the
buffer with an event the producer waits for on the device. Reads run ahead across
acquisitions, so a series streams at the drive's rate rather than one file at a
time.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np

from quantem.gpu._compact.paired import QUERY_ABI, PairedCounts

from .constants import BLOCK_SIZE
from .inspect import inspect
from .models import FourDSTEMData

_ALIGN = 4096
_ROLLING_SCANS = 8192


class PairedLoader:
    """Stream complete native acquisitions into :class:`PairedCounts` sources.

    One loader owns pinned staging, reader threads and two CUDA streams; reuse it
    across a series so shard reads run ahead over file boundaries.

    Parameters
    ----------
    readers
        Direct-read threads; each holds one staging slot while its shard is read
        and parsed.
    slots
        Pinned staging slots; at least ``readers`` plus two copies in flight.
    capacity
        Bytes per staging slot; the largest shard must fit.
    rolling_scans
        Scans per rolling frame buffer, rounded down to whole 512-scan blocks;
        each buffer holds ``rolling_scans`` native frames on the device.
    rings
        Rolling frame buffers in flight between the decode and encode streams.
        Two small buffers (``rolling_scans=512, rings=2``) admit the last
        acquisitions of a series when little device memory remains.

    Examples
    --------
    >>> with PairedLoader() as loader:
    ...     for path, source, timings in loader.load_many(paths, scan_shape=(512, 512)):
    ...         resident.append(source)
    """

    def __init__(self, *, readers: int = 6, slots: int = 10, capacity: int = 128 * 1024**2, rolling_scans: int = _ROLLING_SCANS, rings: int = 3):
        import cupy as cp

        from ._memory import _alloc_pinned_fast, _release_pinned

        if slots < readers + 2:
            raise ValueError(f"slots ({slots}) must be at least readers ({readers}) plus two copies in flight.")
        if rolling_scans < PairedCounts.interval or rings < 2:
            raise ValueError(f"rolling_scans must be at least {PairedCounts.interval} and rings at least 2; got {rolling_scans} and {rings}.")
        self.rolling_scans, self.rings = int(rolling_scans), int(rings)
        self._release = _release_pinned
        self.capacity = (int(capacity) + _ALIGN - 1) & ~(_ALIGN - 1)
        self.staging = [_alloc_pinned_fast(self.capacity) for _ in range(slots)]
        if any(int(view.ctypes.data) % _ALIGN for view in self.staging):
            raise ValueError("Direct I/O needs page-aligned staging; the pinned allocator returned an unaligned buffer.")
        self.free_slots: Queue = Queue()
        for slot in range(slots):
            self.free_slots.put(slot)
        self.lookahead = readers + 2
        self.executor = ThreadPoolExecutor(max_workers=readers)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.producer_stream = cp.cuda.Stream(non_blocking=True)
        self.consumer_stream = cp.cuda.Stream(non_blocking=True)
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.worker.shutdown(wait=True)
        self.executor.shutdown(wait=True)
        self.producer_stream.synchronize()
        self.consumer_stream.synchronize()
        for view in self.staging:
            self._release(view, prune=False)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def load(self, path, *, scan_shape=None, device=None):
        """Load one acquisition; returns ``(source, timings)``. See :meth:`load_many`."""
        for _, source, timings in self.load_many([path], scan_shape=scan_shape, device=device):
            return source, timings
        raise ValueError(f"{path} was not admitted.")

    def load_many(self, paths, *, scan_shape=None, device=None, admit=None):
        """Yield ``(path, source, timings)`` for each acquisition, in order.

        ``admit(path)`` is asked once per acquisition when it enters the read-ahead
        window (up to two acquisitions before its first shard read), while earlier
        ones are still streaming; returning ``False`` ends the series after the
        acquisitions already admitted. Closing the generator early stops the
        stream and releases its staging.
        """
        import cupy as cp

        if self.closed:
            raise RuntimeError("The paired loader is closed.")
        paths = [str(path) for path in paths]
        selected = cp.cuda.Device().id if device is None else int(str(device).removeprefix("cuda:"))
        if not paths or (admit is not None and not admit(paths[0])):
            return
        handoff: Queue = Queue()
        free: Queue = Queue()
        failure, stop, timings = [], [], {}
        with cp.cuda.Device(selected):
            first = self._describe(paths[0], scan_shape)
            first["source"] = PairedCounts(first["shape"], first["dtype"], first["valid"])  # builds the coding tables once
            buffers = _DeviceBuffers(first, self.capacity, self.rolling_scans, self.rings)
            cp.cuda.Stream.null.synchronize()  # tables and buffers were prepared on the null stream; the loader streams never wait for it
            for ring in range(len(buffers.rolling)):
                free.put((ring, None))
            timings[paths[0]] = _fresh_timings(first)
            future = self.worker.submit(self._produce, paths, scan_shape, selected, admit, [first], buffers, handoff, free, failure, stop, timings)
            finished = False
            try:
                with self.consumer_stream:
                    while True:
                        item = handoff.get()
                        if item is None:
                            finished = True
                            break
                        ring, count, event, description, last = item
                        stats = timings[description["path"]]
                        self.consumer_stream.wait_event(event)
                        started = time.perf_counter()
                        source = description["source"]
                        source.append(buffers.rolling[ring][:count])
                        stats["consumer_seconds"] += time.perf_counter() - started
                        stats["chunks"] += 1
                        freed = cp.cuda.Event()
                        freed.record(self.consumer_stream)
                        free.put((ring, freed))
                        if last:
                            self.consumer_stream.synchronize()
                            source = description.pop("source")
                            if source.ready_scans != math.prod(source.shape[:2]):
                                raise RuntimeError("Producer and consumer disagree on the resident scan count.")
                            stats.update(
                                resident_ready_seconds=time.perf_counter() - stats.pop("started"),
                                encode_seconds=source.load_metrics["encode_seconds"], index_seconds=source.load_metrics["index_seconds"],
                                read_seconds=float(np.sum(stats["read_seconds"])), header_seconds=float(np.sum(stats["header_seconds"])),
                                shards=len(description["shards"]), resident_bytes=source.nbytes,
                            )
                            yield description["path"], source, dict(stats, metadata=description["metadata"], pixel_mask=description["pixel_mask"], shape=description["shape"], dtype=description["dtype"])
                            source = None
                future.result()
                if failure:
                    raise failure[0]
            finally:
                stop.append(True)
                while not finished:  # unblock a producer waiting for a ring after an early close
                    item = handoff.get()
                    finished = item is None
                    if not finished:
                        free.put((item[0], None))
                future.result()
                self.producer_stream.synchronize()
                self.consumer_stream.synchronize()
                buffers.release()

    def _describe(self, path, scan_shape):
        """Inspect one master and list its shards with their chunk tables."""
        from .load import _discover_chunk_names, _get_master_frame_sources

        started = time.perf_counter()
        info = inspect(path, scan_shape=scan_shape)
        if not info.ready or info.scan_shape is None or info.detector_shape is None:
            raise ValueError(f"{info.reason}: {info.action}")
        dtype = np.dtype(info.dtype)
        if dtype != np.dtype("uint16"):
            raise NotImplementedError("The paired loader streams native uint16 detectors; use representation='ans' for uint8.")
        shape = (*info.scan_shape, *info.detector_shape)
        if math.prod(info.scan_shape) % PairedCounts.interval:
            raise ValueError(f"The paired layout needs a scan with a multiple of {PairedCounts.interval} positions; got {info.scan_shape}.")
        frame_bytes = math.prod(info.detector_shape) * dtype.itemsize
        if (frame_bytes % BLOCK_SIZE) % 16:
            raise ValueError(f"A partial final LZ4 block must hold a multiple of 8 detector values; got detector shape {info.detector_shape}.")
        valid = np.ones(info.detector_shape, bool)
        if info.pixel_mask is not None:
            valid &= np.asarray(info.pixel_mask) == 0
        names = _discover_chunk_names(str(path)) or ["data"]
        infos, _ = _get_master_frame_sources(str(path), names, apply_mask=False)
        blocks = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE
        shards = []
        for source in infos:
            if tuple(source["frame_shape"]) != tuple(info.detector_shape) or np.dtype(source["dtype"]) != dtype:
                raise ValueError(f"{source['path']} does not match the inspected detector geometry.")
            shards.append(dict(path=source["path"], n_frames=int(source["n_frames"]), chunk_infos=np.asarray(source["chunk_infos"], np.uint64), blocks=blocks))
        if sum(shard["n_frames"] for shard in shards) != math.prod(info.scan_shape):
            raise ValueError(f"{path}: shards hold {sum(s['n_frames'] for s in shards)} frames for {math.prod(info.scan_shape)} scan positions.")
        return dict(path=str(path), shape=shape, dtype=dtype, valid=valid, shards=shards, frame_bytes=frame_bytes, blocks=blocks,
                    metadata=dict(info.metadata), pixel_mask=info.pixel_mask, describe_seconds=time.perf_counter() - started)

    def _read_and_parse(self, shard: dict) -> dict:
        """Reader thread: direct read of one shard plus its LZ4 block header parse."""
        from .load import _parse_headers

        slot = self.free_slots.get()
        try:
            staging = self.staging[slot]
            size, read_seconds = _read_direct(shard["path"], staging)
            offsets = np.ascontiguousarray(shard["chunk_infos"][:, 0], np.uint64)
            sizes = np.ascontiguousarray(shard["chunk_infos"][:, 1], np.uint32)
            n = shard["n_frames"]
            if int(offsets[-1] + sizes[-1]) > size:
                raise ValueError(f"{shard['path']}: chunk table extends past the file end.")
            starts = np.zeros(n * shard["blocks"], np.uint32)
            counts = np.zeros(n, np.uint32)
            started = time.perf_counter()
            _parse_headers(staging, sizes, offsets, starts, counts, n, shard["blocks"])
            if not np.all(counts == shard["blocks"]):
                raise ValueError(f"{shard['path']}: every frame chunk must hold {shard['blocks']} LZ4 blocks.")
            return dict(slot=slot, offsets=offsets, starts=starts, frames=n, size=size, read_seconds=read_seconds, header_seconds=time.perf_counter() - started)
        except BaseException:
            self.free_slots.put(slot)
            raise

    def _produce(self, paths, scan_shape, selected, admit, descriptions, buffers, handoff, free, failure, stop, timings):
        """Producer thread: keep reads ahead and enqueue decode work in shard order."""
        import cupy as cp

        from .load import _bitshuffle_kernel_u16, _bitshuffle_tail_kernel_u16, _h5lz4dc_kernel

        pending, order, copy_events = {}, [], [None, None]
        submitted = consumed = 0
        next_source = 1
        current = None
        describing = []  # (path, future) for admitted acquisitions whose metadata is being read
        stream = self.producer_stream
        try:
            with cp.cuda.Device(selected):
                def extend():
                    nonlocal next_source
                    # Metadata reads (HDF5 chunk tables) run on the reader pool so the producer
                    # keeps enqueueing decode work; admission is asked once per acquisition.
                    while next_source < len(paths) and len(describing) < 2 and len(order) - submitted < self.lookahead + 64:
                        path = paths[next_source]
                        if admit is not None and not admit(path):
                            del paths[next_source:]
                            break
                        describing.append((path, self.executor.submit(self._describe, path, scan_shape)))
                        next_source += 1
                    while describing and (submitted >= len(order) or describing[0][1].done()):
                        path, future = describing.pop(0)
                        description = future.result()
                        if description["shape"][2:] != descriptions[0]["shape"][2:]:
                            raise ValueError(f"{path}: every acquisition in one series must share the detector geometry {descriptions[0]['shape'][2:]}.")
                        description["source"] = PairedCounts(description["shape"], description["dtype"], description["valid"])
                        descriptions.append(description)
                        timings[path] = _fresh_timings(description)
                        for shard in description["shards"]:
                            order.append((len(descriptions) - 1, shard))

                def submit_ahead():
                    nonlocal submitted
                    while submitted < len(order) and submitted - consumed < self.lookahead:
                        pending[submitted] = self.executor.submit(self._read_and_parse, order[submitted][1])
                        submitted += 1

                for shard in descriptions[0]["shards"]:
                    order.append((0, shard))
                extend()
                submit_ahead()
                frame_bytes, blocks = descriptions[0]["frame_bytes"], descriptions[0]["blocks"]
                full_blocks, tail_bytes = divmod(frame_bytes, BLOCK_SIZE)
                rolling_scans = buffers.rolling[0].shape[0]
                filled = ready = 0
                source_index = -1
                pair = 0
                ring, ring_event = free.get()
                while consumed < len(order) and not stop:
                    index, shard = order[consumed]
                    if index != source_index:
                        if filled:
                            raise ValueError("Shards do not cover the declared scan exactly.")
                        source_index, ready = index, 0
                        description = descriptions[index]
                        total = math.prod(description["shape"][:2])
                    waited = time.perf_counter()
                    current = pending.pop(consumed).result()
                    consumed += 1
                    stats = timings[description["path"]]
                    stats["producer_wait_seconds"] += time.perf_counter() - waited
                    stats["read_seconds"].append(current["read_seconds"])
                    stats["header_seconds"].append(current["header_seconds"])
                    extend()
                    submit_ahead()
                    n, size = current["frames"], current["size"]
                    buffers.reserve(n, stream)
                    if copy_events[pair] is not None:
                        copy_events[pair].synchronize()
                        self.free_slots.put(copy_events[pair].slot)
                        copy_events[pair] = None
                    host_offsets = np.frombuffer(buffers.host_offsets[pair], np.uint64, n)
                    host_starts = np.frombuffer(buffers.host_starts[pair], np.uint32, blocks * n)
                    host_offsets[:] = current["offsets"]
                    host_starts[:] = current["starts"]
                    with stream:
                        buffers.compressed[pair][:size].set(self.staging[current["slot"]][:size], stream=stream)
                        buffers.offsets[pair][:n].set(host_offsets, stream=stream)
                        buffers.starts[pair][: blocks * n].set(host_starts, stream=stream)
                        done = cp.cuda.Event()
                        done.record(stream)
                        done.slot = current["slot"]
                        copy_events[pair] = done
                        current = None
                        _h5lz4dc_kernel(((blocks + 1) // 2, 1, n), (32, 2, 1), (buffers.compressed[pair], buffers.offsets[pair], buffers.starts[pair], buffers.counts, buffers.block_offsets, np.uint32(BLOCK_SIZE), np.uint32(frame_bytes), buffers.lz4[pair]), stream=stream)
                        at = 0
                        while at < n:
                            take = min(rolling_scans - filled, n - at)
                            if filled == 0 and ring_event is not None:
                                stream.wait_event(ring_event)
                            decoded = buffers.lz4[pair][at * frame_bytes :]
                            target = buffers.rolling[ring][filled:]
                            if full_blocks:
                                _bitshuffle_kernel_u16((full_blocks, 1, take), (256, 1, 1), (decoded, target, np.uint32(frame_bytes)), stream=stream)
                            if tail_bytes:
                                _bitshuffle_tail_kernel_u16(((tail_bytes // 2 + 255) // 256, 1, take), (256, 1, 1), (decoded, target, np.uint32(frame_bytes)), stream=stream)
                            at += take
                            filled += take
                            ready += take
                            last = ready == total
                            if filled == rolling_scans or last:
                                if filled % PairedCounts.interval:
                                    raise ValueError(f"A shard ends inside a {PairedCounts.interval}-scan block; the paired layout codes complete blocks.")
                                event = cp.cuda.Event()
                                event.record(stream)
                                handoff.put((ring, filled, event, description, last))
                                filled = 0
                                ring, ring_event = free.get()
                    pair ^= 1
        except BaseException as error:
            failure.append(error)
        finally:
            for _, future in describing:
                future.cancel()
            if current is not None:
                self.free_slots.put(current["slot"])
            for future in pending.values():
                try:
                    self.free_slots.put(future.result()["slot"])
                except BaseException:
                    continue
            for event in copy_events:
                if event is not None:
                    event.synchronize()
                    self.free_slots.put(event.slot)
            stream.synchronize()
            handoff.put(None)


class _DeviceBuffers:
    """Device staging for one series: compressed shards, LZ4 output, rolling frames."""

    def __init__(self, description: dict, capacity: int, rolling_scans: int, rings: int):
        import cupy as cp

        self.frame_bytes, self.blocks = description["frame_bytes"], description["blocks"]
        det_shape = description["shape"][2:]
        scans = min(rolling_scans, math.prod(description["shape"][:2])) // PairedCounts.interval * PairedCounts.interval
        self.rolling = [cp.empty((scans, *det_shape), cp.uint16) for _ in range(rings)]
        self.compressed = [cp.empty(capacity, cp.uint8) for _ in range(2)]
        self.frames = 0
        self.reserve(max(shard["n_frames"] for shard in description["shards"]), None)

    def reserve(self, frames: int, stream) -> None:
        """Size the per-shard tables for ``frames``; grows only after draining the stream."""
        import cupy as cp

        if frames <= self.frames:
            return
        if stream is not None:
            stream.synchronize()
        self.frames = frames
        self.lz4 = [cp.empty(frames * self.frame_bytes, cp.uint8) for _ in range(2)]
        self.offsets = [cp.empty(frames, cp.uint64) for _ in range(2)]
        self.starts = [cp.empty(frames * self.blocks, cp.uint32) for _ in range(2)]
        self.counts = cp.full(frames, self.blocks, cp.uint32)
        self.block_offsets = cp.arange(frames + 1, dtype=cp.uint32) * self.blocks
        self.host_offsets = [cp.cuda.alloc_pinned_memory(frames * 8) for _ in range(2)]
        self.host_starts = [cp.cuda.alloc_pinned_memory(frames * self.blocks * 4) for _ in range(2)]
        cp.cuda.Stream.null.synchronize()  # the tables are filled on the null stream; the producer stream does not wait for it

    def release(self) -> None:
        self.rolling = self.compressed = self.lz4 = self.offsets = self.starts = self.counts = self.block_offsets = None
        self.host_offsets = self.host_starts = None


def _fresh_timings(description: dict) -> dict:
    return dict(describe_seconds=description["describe_seconds"], read_seconds=[], header_seconds=[], producer_wait_seconds=0.0, consumer_seconds=0.0, chunks=0, started=time.perf_counter())


def _read_direct(path: str, staging: np.ndarray) -> tuple[int, float]:
    """Read one whole shard with direct I/O into page-aligned pinned staging."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    started = time.perf_counter()
    try:
        before = os.fstat(fd)
        size = before.st_size
        padded = (size + _ALIGN - 1) & ~(_ALIGN - 1)
        if size <= 0 or padded > staging.size:
            raise ValueError(f"{path} ({size} bytes) exceeds the staging capacity of {staging.size} bytes; raise PairedLoader(capacity=).")
        view = memoryview(staging)
        offset = 0
        while offset < size:
            got = os.preadv(fd, [view[offset:padded]], offset)
            if got <= 0 or (offset + got < size and got % _ALIGN):
                raise OSError(f"Short direct read of {path} at byte {offset}.")
            offset += got
        after = os.fstat(fd)
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"{path} changed while it was being read.")
        return size, time.perf_counter() - started
    finally:
        os.close(fd)


# --- io.load entry points -------------------------------------------------------


def load_h5_paired(paths, *, scan_shape, device, verbose):
    """Load original H5 acquisitions into the paired resident layout, one source each."""
    results = []
    with PairedLoader() as loader:
        for _, source, timings in loader.load_many(paths, scan_shape=scan_shape, device=device):
            results.append(_result(source, timings, verbose))
    return results


def load_paired_file(path, *, device, verbose):
    """Reopen a saved paired resident form as an exact source without decoding."""
    started = time.perf_counter()
    source = PairedCounts.load(path, device=device)
    timings = dict(metadata={}, pixel_mask=None, shape=source.shape, dtype=source.dtype, resident_ready_seconds=time.perf_counter() - started, resident_bytes=source.nbytes)
    return _result(source, timings, verbose)


def _result(source, timings, verbose):
    shape, dtype = tuple(source.shape), np.dtype(source.dtype)
    metadata = dict(timings["metadata"])
    metadata.update(
        backend="cuda", representation="paired", residency="device", source_shape=shape, working_shape=shape,
        scan_shape=shape[:2], detector_shape=shape[2:], source_dtype=dtype.name, working_dtype=dtype.name, dtype=dtype.name,
        n_frames=math.prod(shape[:2]), source_logical_tensor_bytes=math.prod(shape) * dtype.itemsize,
        working_logical_tensor_bytes=math.prod(shape) * dtype.itemsize, physical_resident_bytes=source.nbytes,
        index_bytes=source.index_nbytes, pixel_mask=timings["pixel_mask"], lossless_exact=True, file_counts_exact=True,
        resident_profile=QUERY_ABI, resident_codec="paired-tans", detector_mask_policy="preserve-stored-counts",
        scan_bin=1, detector_bin=1, crop=None,
        load_timings={key: value for key, value in timings.items() if key not in ("metadata", "pixel_mask", "shape", "dtype")},
    )
    if verbose:
        print(f"Paired resident source {shape} ready in {metadata['load_timings']['resident_ready_seconds']:.2f} s.")
    return FourDSTEMData(source, metadata)
