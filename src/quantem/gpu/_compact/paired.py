"""Paired-count tANS resident layout with an adaptive polar interaction index.

This is an opt-in second resident layout next to the byte-rANS ``StreamedCounts``
default. Every original count is retained exactly; only the bytes and the
query kernels differ. Compared with the default layout it decodes the residual
pixels of a detector query about twice as fast on the same counts, stores the
spatial index as radial-angular pixel groups so translated circular detectors
need fewer residual pixels, and can be written once to disk as its exact
resident arrays and reopened without decoding.

Constraints that the default layout does not have: chunks must be appended as
complete multiples of the 512-scan coding interval, and the detector must be
at most 65,535 pixels per stream group of 32 (grouped 16-bit offsets).
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from .interaction import StreamedSeriesCompute
from .streamed import Chunk, StreamedCounts, field_count

QUERY_ABI = "paired-polar-counts-v1"
RESIDENT_MAGIC = "quantem-paired-resident-v1"
FILE_MAGIC = b"QGPUPAIR"  # first eight bytes of a saved form; io.load detects it
MODELS = 32
STATES = 1024
GUARD_BYTES = 8
ALIGN = 4096
_ARRAY_NAMES = ("payload", "offsets", "models", "index_words", "index_starts", "index_widths")


@cache
def kernels(device: int) -> dict:
    """Compile the paired layout kernels once per device."""
    import cupy as cp

    with cp.cuda.Device(device):
        module = cp.RawModule(
            code=Path(__file__).with_name("kernels").joinpath("paired.cu").read_text(),
            options=("--std=c++17",),
        )
        names = [
            "pack_offsets", "unpack_offsets", "tables", "encode", "compact", "decode", "decode_range",
            "frame_u8", "frame_u16", "plan_u32", "plan_u64", "residual_u32", "residual_u64",
            "fields", "field_sizes", "pack_fields", "unpack_fields", "index_u32", "index_u64", "weights",
        ]
        return {name: module.get_function(f"pm_{name}") for name in names}


@cache
def frequencies(device: int):
    """Pair-symbol frequencies for 32 Poisson means, each with the best of 12 alphabet supports.

    Symbol ``a * 33 + b`` codes the count pair ``(a, b)`` with both below 32; symbol
    1088 escapes to a 13-bit literal word for pairs below 64, or to two 16-bit
    literals. Frequencies sum to 1024 per model. Model choice only changes bytes.
    """
    import cupy as cp

    with cp.cuda.Device(device):
        means = cp.exp(cp.linspace(cp.log(cp.float64(0.002)), cp.log(cp.float64(32)), MODELS))
        choices = cp.asarray([4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768], cp.int32)
        k = cp.arange(33, dtype=cp.float64)
        factorial = cp.concatenate((cp.zeros(1), cp.cumsum(cp.log(cp.arange(1, 33, dtype=cp.float64)))))
        p = cp.exp(k[None, :] * cp.log(means[:, None]) - factorial[None, :] - means[:, None])
        a = cp.arange(1089) // 33
        b = cp.arange(1089) % 33
        joint = cp.where((a < 32) & (b < 32), p[:, a] * p[:, b], 0)
        ranking = cp.argsort(cp.argsort(-joint, axis=1), axis=1)
        supported = ranking[None, :, :] < choices[:, None, None]
        supported[:, :, 1088] = True
        probability = cp.where(supported, joint[None, :, :], 0)
        probability[:, :, 1088] = 0
        probability[:, :, 1088] = cp.maximum(0, 1 - probability.sum(axis=2))
        count = supported.sum(axis=2)
        allocation = probability * (STATES - count)[:, :, None]
        frequency = cp.where(supported, cp.floor(allocation).astype(cp.uint32) + 1, 0)
        fraction = cp.where(supported, allocation - cp.floor(allocation), -cp.inf)
        ranks = cp.argsort(cp.argsort(-fraction, axis=2), axis=2)
        frequency += (ranks < (STATES - frequency.sum(axis=2))[:, :, None]).astype(cp.uint32)
        cost = (probability * cp.log2(STATES / cp.maximum(frequency, 1))).sum(axis=2) + 13 * probability[:, :, 1088]
        chosen = cp.argmin(cost, axis=0)
        frequency = frequency[chosen, cp.arange(MODELS)]
        starts = cp.cumsum(frequency, axis=1, dtype=cp.uint32) - frequency
        return (frequency << 16) | starts


@cache
def tables(device: int):
    """Return the encoding (state per symbol rank) and decoding (pair, bits, base) tables."""
    import cupy as cp

    with cp.cuda.Device(device):
        encoding = cp.empty((MODELS, STATES), cp.uint16)
        decoding = cp.empty((MODELS, STATES), cp.uint32)
        kernels(device)["tables"](((MODELS * 1089 + 127) // 128,), (128,), (frequencies(device), encoding, decoding))
        return encoding, decoding


@cache
def polar_layout(shape: tuple[int, int]):
    """Order detector pixels by radial band then angle into 64-pixel leaves and 16-leaf roots.

    Radial bands follow ``floor(radius ** 1.5 / 45)`` so bands narrow with radius and a
    translated annulus edge crosses fewer leaves. The layout only decides which pixels
    share an exact index sum; sums are integers over original counts.
    """
    rows, cols = shape
    row, col = np.indices(shape)
    row = row - (rows - 1) / 2
    col = col - (cols - 1) / 2
    radius = np.hypot(row, col)
    angle = np.arctan2(row, col)
    radial = np.floor(radius**1.5 / 45)
    order = np.lexsort((radius.ravel(), angle.ravel(), radial.ravel())).astype(np.int32)
    leaves = math.ceil(rows / 8) * math.ceil(cols / 8)
    roots = field_count(shape) - leaves
    permutation = np.full(leaves * 64, -1, np.int32)
    permutation[: order.size] = order
    return permutation, leaves, roots


@cache
def _device_permutation(device: int, shape: tuple[int, int]):
    import cupy as cp

    with cp.cuda.Device(device):
        return cp.asarray(polar_layout(shape)[0])


def polar_planner(shape: tuple[int, int], weights: np.ndarray | None = None):
    """Return ``(plan, cost)`` decomposing a signed mask into index fields plus residual pixels.

    Each leaf takes the value (0, 1 or -1) that most of its pixels hold, weighted by
    the expected decode cost of each pixel when weights are given; roots do the same
    over their leaves. Pixels that disagree with their leaf become signed residual
    corrections. ``cost`` estimates decode work for the planner's choice between a
    full mask and a difference against the previous mask.
    """
    permutation, leaves, roots = polar_layout(shape)
    real = permutation >= 0
    pixels = permutation[real]
    tile_weight = np.ones(permutation.size, np.float64)
    if weights is not None:
        tile_weight = np.zeros(permutation.size, np.float64)
        tile_weight[real] = np.asarray(weights, np.float64).ravel()[pixels]
    tile_weight = tile_weight.reshape(leaves, 64)
    options = np.array([0, 1, -1], np.int32)

    def plan(mask):
        values = np.zeros(permutation.size, np.int32)
        values[real] = mask.ravel()[pixels]
        tiles = values.reshape(leaves, 64)
        counts = np.stack([((tiles == value) * tile_weight).sum(axis=1) for value in options])
        leaf = options[counts.argmax(axis=0)]
        residual = np.zeros(mask.size, np.int32)
        residual[pixels] = (values - np.repeat(leaf, 64))[real]
        padded = np.zeros(roots * 16, np.int32)
        padded[:leaves] = leaf
        counts = np.stack([(padded.reshape(roots, 16) == value).sum(axis=1) for value in options])
        root = options[counts.argmax(axis=0)]
        leaf = leaf - np.repeat(root, 16)[:leaves]
        fields = np.concatenate((leaf, root))
        selected_fields = np.flatnonzero(fields).astype(np.uint32)
        selected_pixels = np.flatnonzero(residual).astype(np.uint32)
        return selected_fields, fields[selected_fields], selected_pixels, residual[selected_pixels]

    if weights is None:
        return plan, lambda p: len(p[0]) + len(p[2]) * 4
    flat = np.asarray(weights, np.float64).ravel()
    return plan, lambda p: len(p[0]) + float(flat[p[2]].sum())


class PairedCounts(StreamedCounts):
    """Exact paired-count tANS resident source with an adaptive polar index.

    Append complete 512-scan blocks of native counts; every count is retained. The
    source joins ``detector.prepare`` like any streamed source and selects the
    paired query kernels automatically.

    Examples
    --------
    >>> import cupy as cp
    >>> source = PairedCounts((16, 32, 192, 192), "uint16")
    >>> source.append(cp.zeros((512, 192, 192), cp.uint16))
    >>> source.ready_scans
    512
    """

    query_abi = QUERY_ABI

    def __init__(self, shape: tuple[int, int, int, int], dtype, valid=None):
        super().__init__(shape, dtype, valid)
        if math.prod(self.shape[:2]) % self.interval:
            raise ValueError(
                f"The paired layout codes complete {self.interval}-scan blocks; a "
                f"{self.shape[0]}x{self.shape[1]} scan has {math.prod(self.shape[:2])} positions. "
                "Use StreamedCounts for other scan sizes."
            )
        self.encoding, self.decoding = tables(self.device)
        self.kernels = kernels(self.device)
        self.polar_permutation = _device_permutation(self.device, tuple(self.shape[2:]))

    def append(self, raw) -> None:
        """Encode one complete block of consecutive native scans and index it."""
        import cupy as cp

        if self.is_released:
            raise ValueError("The resident source has been released.")
        if not isinstance(raw, cp.ndarray) or raw.device.id != self.device or raw.dtype != self.dtype or not raw.flags.c_contiguous:
            raise ValueError("Provide contiguous native counts on the source CUDA device.")
        if raw.ndim != 3 or raw.shape[1:] != self.shape[2:]:
            raise ValueError("Chunk detector geometry must match the complete acquisition.")
        scans, pixels = raw.shape[0], math.prod(self.shape[2:])
        if scans < 1 or scans % self.interval or self.ready_scans + scans > math.prod(self.shape[:2]):
            raise ValueError(
                f"Append a positive multiple of {self.interval} scans within the acquisition; got {scans} "
                f"after {self.ready_scans} of {math.prod(self.shape[:2])}."
            )
        streams = scans // self.interval * pixels
        u32 = np.uint32
        ks = self.kernels
        with cp.cuda.Device(self.device):
            started = time.perf_counter()
            # One coded pair can emit seven bytes past the 2*interval guard before the
            # encoder gives up on a stream, plus one flush byte; keep those in bounds.
            scratch = cp.empty((2 * self.interval + 16, streams), cp.uint8)
            sizes, states = cp.empty(streams, cp.uint32), cp.empty(streams, cp.uint32)
            models = cp.empty(streams, cp.uint8)
            grid = ((streams + 127) // 128,)
            geometry = (raw, np.int32(raw.dtype.itemsize), u32(scans), u32(pixels), u32(self.interval))
            ks["encode"](grid, (128,), (*geometry, frequencies(self.device), self.encoding, scratch, sizes, states, models, u32(streams)))
            offsets = cp.empty(streams + 1, cp.uint32)
            offsets[0] = 0
            cp.cumsum(sizes, dtype=cp.uint32, out=offsets[1:])
            payload = cp.zeros(int(offsets[-1].get()) + GUARD_BYTES, cp.uint8)
            ks["compact"](grid, (128,), (*geometry, scratch, offsets, states, models, payload, u32(streams)))
            groups = (streams + 31) // 32
            records = cp.zeros((groups + 1) * 17, cp.uint32)
            errors = cp.zeros(1, cp.uint32)
            ks["pack_offsets"](((groups * 32 + 127) // 128,), (128,), (offsets, records, errors, u32(streams)))
            if int(errors.get()[0]):
                raise ValueError("A stream group exceeds 65,535 bytes; the paired layout needs at most 65,535 detector pixels per 32 streams.")
            del scratch, sizes, states, offsets
            cp.cuda.get_current_stream().synchronize()
            self.load_metrics["encode_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            fields = field_count(self.shape[2:])
            values = cp.empty((scans, fields), cp.uint32)
            ks["fields"](((scans * fields + 3) // 4,), (128,), (raw, np.int32(raw.dtype.itemsize), self.valid, values, u32(scans), u32(self.shape[2]), u32(self.shape[3]), u32(fields), self.polar_permutation))
            nstreams = scans // self.interval * fields
            widths, lengths = cp.empty(nstreams, cp.uint8), cp.empty(nstreams, cp.uint64)
            index_args = (u32(scans), u32(fields), u32(self.interval))
            index_grid = ((nstreams + 127) // 128,)
            ks["field_sizes"](index_grid, (128,), (values, widths, lengths, *index_args))
            starts = cp.empty(nstreams + 1, cp.uint64)
            starts[0] = 0
            cp.cumsum(lengths, dtype=cp.uint64, out=starts[1:])
            words = cp.empty(int(starts[-1].get()), cp.uint32)
            ks["pack_fields"](index_grid, (128,), (values, widths, starts, words, *index_args))
            cp.cuda.get_current_stream().synchronize()
            self.load_metrics["index_seconds"] += time.perf_counter() - started
            self.chunks.append(Chunk(self.ready_scans, scans, (payload, records, models, words, starts, widths)))
            self.ready_scans += scans

    def decode_blocks(self, first: int, scans: int, *, out=None):
        """Decode ``scans`` consecutive scans from ``first`` into native counts.

        The range must start on a coding block (multiple of 512 scans) and lie inside
        one resident chunk; ``out`` is an optional ``(scans, rows, cols)`` uint16 device
        array. Kernels run on the current stream, so a caller may prefetch blocks on a
        side stream; see :class:`PairedFeed` for the double-buffered iterator.

        Examples
        --------
        >>> block = source.decode_blocks(4096, 512)
        >>> block.shape
        (512, 192, 192)
        """
        import cupy as cp

        if self.is_released:
            raise ValueError("The resident source has been released.")
        if first % self.interval or scans < 1:
            raise ValueError(f"Decode ranges start on a {self.interval}-scan block; got first={first}, scans={scans}.")
        chunk = next((c for c in self.chunks if c.first <= first < c.first + c.scans), None)
        if chunk is None or first + scans > chunk.first + chunk.scans:
            raise ValueError(f"Scans {first}..{first + scans} are not inside one resident chunk.")
        pixels = math.prod(self.shape[2:])
        blocks = math.ceil(scans / self.interval)
        if scans % self.interval and first + scans != chunk.first + chunk.scans:
            raise ValueError(f"Decode whole {self.interval}-scan blocks unless the range ends the chunk.")
        with cp.cuda.Device(self.device):
            if out is None:
                out = cp.empty((scans, *self.shape[2:]), cp.uint16)
            elif out.shape != (scans, *self.shape[2:]) or out.dtype != cp.uint16 or not out.flags.c_contiguous:
                raise ValueError(f"out must be a contiguous uint16 array of shape {(scans, *self.shape[2:])}.")
            errors = cp.zeros(1, cp.uint32)
            count = blocks * pixels
            self.kernels["decode_range"](((count + 127) // 128,), (128,), (
                *chunk.arrays[:3], self.decoding, out, errors, np.uint32(chunk.scans), np.uint32(pixels),
                np.uint32(self.interval), np.uint32((first - chunk.first) // self.interval * pixels), np.uint32(count)))
            if int(errors.get()[0]):
                raise ValueError("An encoded count stream failed exact decoding.")
            return out

    def save(self, path) -> dict:
        """Write the exact resident arrays once so the source reopens without decoding.

        The file starts with the fixed ``QGPUPAIR`` magic, a small JSON header and
        every chunk array at a 4096-byte aligned offset, in chunk order. Reopening reads it with direct I/O straight into
        device memory (``PairedCounts.load``); bytes are identical to this source.

        Examples
        --------
        >>> written = source.save("acquisition.paired")
        >>> reopened = PairedCounts.load("acquisition.paired")
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Saved resident form already exists: {path}. Choose a new destination.")
        if self.ready_scans != math.prod(self.shape[:2]):
            raise ValueError(f"Save a complete source; {self.ready_scans} of {math.prod(self.shape[:2])} scans are resident.")
        table, at, arrays = [], 0, []
        for chunk in self.chunks:
            entries = []
            for name, array in zip(_ARRAY_NAMES, chunk.arrays):
                at = (at + ALIGN - 1) & ~(ALIGN - 1)
                entries.append(dict(name=name, dtype=str(array.dtype), shape=list(array.shape), offset=at, nbytes=int(array.nbytes)))
                arrays.append((at, array))
                at += int(array.nbytes)
            table.append(dict(first=int(chunk.first), scans=int(chunk.scans), arrays=entries))
        header = dict(
            magic=RESIDENT_MAGIC, query_abi=QUERY_ABI, shape=list(self.shape), dtype=str(self.dtype),
            interval=int(self.interval), models=MODELS, states=STATES,
            valid=np.packbits(self.valid_pixels.ravel()).tobytes().hex(), chunks=table,
        )
        blob = json.dumps(header).encode()
        data_start = (len(blob) + 24 + ALIGN - 1) & ~(ALIGN - 1)
        started = time.perf_counter()
        with open(path, "wb") as handle:
            handle.write(FILE_MAGIC + len(blob).to_bytes(8, "little") + data_start.to_bytes(8, "little") + blob)
            handle.truncate(data_start)
            for offset, array in arrays:
                handle.seek(data_start + offset)
                handle.write(array.get().tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        return dict(path=str(path), bytes=path.stat().st_size, write_seconds=time.perf_counter() - started)

    @classmethod
    def load(cls, path, *, device: int | None = None, reader: ResidentFileReader | None = None, allocate=None):
        """Reopen a saved resident form as an exact source without decoding.

        Parameters
        ----------
        path
            File written by :meth:`save`.
        device
            CUDA device index; defaults to the current device.
        reader
            Shared :class:`ResidentFileReader` (pinned staging and threads) when
            opening many files; a private one is created otherwise.
        allocate
            ``allocate(shape, dtype)`` returning the destination device array, for
            callers that manage residency themselves; defaults to ``cupy.empty``.
        """
        import cupy as cp

        header, data_start = _read_header(path)
        if header.get("magic") != RESIDENT_MAGIC or header.get("query_abi") != QUERY_ABI or header.get("models") != MODELS or header.get("states") != STATES:
            raise ValueError(f"{path} is not a {RESIDENT_MAGIC} file for query ABI {QUERY_ABI}.")
        shape = tuple(header["shape"])
        valid = np.unpackbits(np.frombuffer(bytes.fromhex(header["valid"]), np.uint8))[: shape[2] * shape[3]].astype(bool).reshape(shape[2:])
        with cp.cuda.Device(cp.cuda.Device().id if device is None else device):
            source = cls(shape, np.dtype(header["dtype"]), valid)
            owned = reader or ResidentFileReader()
            try:
                source.chunks = owned.read(path, header, data_start, allocate or (lambda shape, dtype: cp.empty(shape, dtype)))
            finally:
                if reader is None:
                    owned.close()
            source.ready_scans = int(shape[0] * shape[1])
            return source


def _read_header(path):
    with open(path, "rb") as handle:
        if handle.read(8) != FILE_MAGIC:
            raise ValueError(f"{path} is not a saved paired resident form (missing {FILE_MAGIC!r} magic).")
        length = int.from_bytes(handle.read(8), "little")
        data_start = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(length)), data_start


class PairedFeed:
    """Iterate decoded scan blocks of one or more paired sources with device prefetch.

    Blocks are decoded on a private stream ``depth`` buffers ahead of the consumer;
    each yielded block is already ordered before the caller's current stream, and
    the buffer is reused only after the caller's stream has passed the next
    iteration. ``order="scan"`` yields block 0 of every source, then block 1, so a
    joint time-series reconstruction sees all acquisitions of one scan range
    together; ``order="source"`` walks each acquisition to its end first.

    Parameters
    ----------
    sources
        Complete :class:`PairedCounts` sources with one detector geometry.
    block_scans
        Scans per block; a multiple of 512.
    depth
        Blocks decoded ahead of the consumer (buffers in flight).
    amplitude
        Also provide ``block.amplitude``, ``sqrt`` of the counts as float32,
        computed on the prefetch stream.

    Examples
    --------
    >>> for block in PairedFeed(sources, amplitude=True):
    ...     update(block.source, block.first, block.amplitude)
    """

    class Block:
        __slots__ = ("source", "first", "scans", "raw", "amplitude")

        def __init__(self, source, first, scans, raw, amplitude):
            self.source, self.first, self.scans, self.raw, self.amplitude = source, first, scans, raw, amplitude

    def __init__(self, sources, *, block_scans: int = 512, depth: int = 2, amplitude: bool = False, order: str = "scan"):
        import cupy as cp

        self.sources = list(sources)
        if not self.sources or any(tuple(s.shape[2:]) != tuple(self.sources[0].shape[2:]) for s in self.sources):
            raise ValueError("Feed sources must share one detector geometry.")
        interval = self.sources[0].interval
        if block_scans < interval or block_scans % interval or depth < 1:
            raise ValueError(f"block_scans must be a positive multiple of {interval} and depth at least 1; got {block_scans}, {depth}.")
        if order not in ("scan", "source"):
            raise ValueError(f"order must be 'scan' or 'source'; got {order!r}.")
        self.block_scans, self.depth, self.amplitude, self.order = int(block_scans), int(depth), bool(amplitude), order
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.plan = self._plan()

    def _plan(self):
        """(source index, first scan, scans) per block, never crossing a resident chunk."""
        per_source = []
        for index, source in enumerate(self.sources):
            items = []
            for chunk in source.chunks:
                at = chunk.first
                while at < chunk.first + chunk.scans:
                    scans = min(self.block_scans, chunk.first + chunk.scans - at)
                    items.append((index, at, scans))
                    at += scans
            per_source.append(items)
        if self.order == "source":
            return [item for items in per_source for item in items]
        return [item for group in zip(*per_source) for item in group] if len({len(i) for i in per_source}) == 1 else [item for items in per_source for item in items]

    def __len__(self):
        return len(self.plan)

    def __iter__(self):
        import cupy as cp

        shape = (self.block_scans, *self.sources[0].shape[2:])
        with cp.cuda.Device(self.sources[0].device):
            raws = [cp.empty(shape, cp.uint16) for _ in range(self.depth + 1)]
            amps = [cp.empty(shape, cp.float32) for _ in range(self.depth + 1)] if self.amplitude else [None] * (self.depth + 1)
            released = [None] * (self.depth + 1)  # event on the consumer stream after it finished with that buffer
            ready = {}
            consumer = cp.cuda.get_current_stream()

            def launch(k):
                slot = k % (self.depth + 1)
                index, first, scans = self.plan[k]
                if released[slot] is not None:
                    self.stream.wait_event(released[slot])
                with self.stream:
                    self.sources[index].decode_blocks(first, scans, out=raws[slot][:scans])
                    if self.amplitude:
                        cp.sqrt(raws[slot][:scans], out=amps[slot][:scans])
                    event = cp.cuda.Event()
                    event.record(self.stream)
                ready[k] = event

            for k in range(min(self.depth, len(self.plan))):
                launch(k)
            for k in range(len(self.plan)):
                slot = k % (self.depth + 1)
                index, first, scans = self.plan[k]
                consumer.wait_event(ready.pop(k))
                yield PairedFeed.Block(index, first, scans, raws[slot][:scans], amps[slot][:scans] if self.amplitude else None)
                done = cp.cuda.Event()
                done.record(consumer)
                released[slot] = done
                if k + self.depth < len(self.plan):
                    launch(k + self.depth)
            self.stream.synchronize()


class ResidentFileReader:
    """Direct I/O reader for saved resident forms: aligned spans through pinned staging.

    Reader threads fill pinned slots with ``O_DIRECT`` reads while the main thread
    enqueues asynchronous copies into the destination device arrays on one stream.
    Nothing is decoded; the host synchronises once per file.
    """

    def __init__(self, readers: int = 6, slots: int = 12, span: int = 64 << 20):
        import cupy as cp

        self.span = span
        self.executor = ThreadPoolExecutor(max_workers=readers)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.slots = [cp.cuda.alloc_pinned_memory(span + ALIGN) for _ in range(slots)]
        self.free = list(range(slots))
        self.closed = False

    def _host_view(self, slot: int, length: int):
        buffer = self.slots[slot]
        pad = (-buffer.ptr) % ALIGN
        return np.frombuffer(buffer, np.uint8, pad + length)[pad:]

    def _read(self, fd: int, offset: int, length: int, slot: int):
        aligned = offset & ~(ALIGN - 1)
        lead = offset - aligned
        want = (lead + length + ALIGN - 1) & ~(ALIGN - 1)
        target = self._host_view(slot, want)
        got = 0
        while got < want:
            n = os.preadv(fd, [memoryview(target)[got:]], aligned + got)
            if n <= 0:
                break
            got += n
        return slot, lead, got

    def read(self, path, header: dict, data_start: int, allocate) -> list[Chunk]:
        import cupy as cp

        chunks, pieces = [], []
        for entry in header["chunks"]:
            arrays = []
            for spec in entry["arrays"]:
                array = allocate(tuple(spec["shape"]), np.dtype(spec["dtype"]))
                arrays.append(array)
                flat = array.view(cp.uint8).ravel() if array.nbytes else None
                at = 0
                while at < spec["nbytes"]:
                    take = min(self.span, spec["nbytes"] - at)
                    pieces.append((data_start + spec["offset"] + at, take, flat, at))
                    at += take
            chunks.append(Chunk(entry["first"], entry["scans"], tuple(arrays)))
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        pending, inflight = {}, []
        try:
            def submit(index):
                while not self.free:
                    event, slot = inflight.pop(0)
                    event.synchronize()
                    self.free.append(slot)
                slot = self.free.pop()
                offset, length, _, _ = pieces[index]
                pending[index] = self.executor.submit(self._read, fd, offset, length, slot)

            ahead = max(1, len(self.slots) - 2)
            for index in range(min(ahead, len(pieces))):
                submit(index)
            with self.stream:
                for index in range(len(pieces)):
                    slot, lead, got = pending.pop(index).result()
                    offset, length, flat, at = pieces[index]
                    if got < lead + length:
                        raise OSError(f"Short read of {path} at byte {offset}.")
                    flat[at : at + length].set(self._host_view(slot, lead + length)[lead:], stream=self.stream)
                    event = cp.cuda.Event()
                    event.record(self.stream)
                    inflight.append((event, slot))
                    if index + ahead < len(pieces):
                        submit(index + ahead)
            for event, slot in inflight:
                event.synchronize()
                self.free.append(slot)
            inflight.clear()
            self.stream.synchronize()
        finally:
            os.close(fd)
            for future in pending.values():
                try:
                    slot, _, _ = future.result()
                    self.free.append(slot)
                except OSError:
                    pass
        return chunks

    def close(self) -> None:
        if not self.closed:
            self.executor.shutdown(wait=True)
            self.stream.synchronize()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PairedSeriesCompute(StreamedSeriesCompute):
    """Joint detector queries over paired-layout sources with the polar planner.

    Selected by ``detector.prepare`` when every acquisition is a
    :class:`PairedCounts`. The descriptor rows, output ownership and the private
    baseline for incremental masks are inherited; the plan, the index summation and
    the residual decoder are the paired kernels.
    """

    def __init__(self, acquisitions):
        import cupy as cp

        sources = [item.data if hasattr(item, "_fields") and "data" in item._fields else item for item in acquisitions]
        if not sources or not all(isinstance(source, PairedCounts) for source in sources):
            raise TypeError("Paired queries need paired-layout sources only; mix nothing else into the series.")
        super().__init__(acquisitions)
        if any(chunk.scans % source.interval for source in sources for chunk in source.chunks):
            raise ValueError("Every paired chunk must hold complete 512-scan blocks.")
        self.backend_metadata["query_abi"] = QUERY_ABI
        ks = kernels(self.device)
        blocks = math.ceil(self.max_scans / self.interval)
        total = self.chunk_count * blocks
        with cp.cuda.Device(self.device):
            self.work_counts = cp.empty(total, cp.uint32)
            self.work = {"capacity": 0, "array": None}
            weights = cp.zeros(self.pixels, cp.float64)
            ks["weights"](((self.pixels + 255) // 256, self.chunk_count), (256,), (self.descriptors, weights, np.uint32(self.pixels), np.uint32(self.chunk_count)))
            blocks_total = sum(chunk.scans / self.interval for source in self.index_owners for chunk in source.chunks)
            self.pixel_weights = weights.get() / blocks_total
        self._polar_plan, self._polar_cost = polar_planner(self.det_shape, self.pixel_weights)
        shared = 16 * self.fields
        self.kernels = dict(self.kernels)
        for bits in (32, 64):
            def index(grid, block, args, bits=bits):
                ks[f"index_u{bits}"](grid, block, (*args[:8], args[9]), shared_mem=shared)

            def residual(grid, block, args, bits=bits):
                count = int(args[3])
                if count > 65535:
                    raise ValueError("At most 65,535 residual pixels per query.")
                if count > self.work["capacity"]:
                    self.work["array"] = cp.empty((total, count), cp.uint16)
                    self.work["capacity"] = count
                self.work_counts.fill(0)
                extra = (self.work["array"], self.work_counts, np.uint32(blocks))
                ks[f"plan_u{bits}"]((total,), (256,), (*args[:8], *extra))
                ks[f"residual_u{bits}"](((count + 255) // 256, total), (256,), (*args[:8], *extra))

            self.kernels[f"index_u{bits}"] = index
            self.kernels[f"residual_u{bits}"] = residual
        for bits in (8, 16):
            def frame(grid, block, args, bits=bits):
                ks[f"frame_u{bits}"](grid, block, args[:5])

            self.kernels[f"frame_u{bits}"] = frame

    def _plan(self, values):
        return self._polar_plan(values)

    def _cost(self, selection):
        return self._polar_cost(selection)
