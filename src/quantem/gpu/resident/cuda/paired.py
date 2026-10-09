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

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_module
from quantem.gpu.resident.cuda.counts import Chunk, StreamedCounts, field_count

QUERY_ABI = "paired-polar-counts-v1"
RESIDENT_MAGIC = "quantem-paired-resident-v1"
FILE_MAGIC = b"QGPUPAIR"  # first eight bytes of a saved form; io.load detects it
MODELS = 32
STATES = 1024
GUARD_BYTES = 8
ALIGN = 4096
_ARRAY_NAMES = ("payload", "offsets", "models", "index_words", "index_starts", "index_widths")


def _round_up(value: int, alignment: int) -> int:
    """Next multiple of ``alignment``: saved arrays start where direct I/O can read them."""
    return (value + alignment - 1) // alignment * alignment


_SOURCE = (Path(__file__).with_name("kernels") / "paired.cu").read_text()
_NAMES = (
    "pack_offsets", "unpack_offsets", "tables", "encode", "compact", "decode_range",
    "frame_u8", "frame_u16", "plan_u32", "plan_u64", "residual_u32", "residual_u64",
    "fields", "field_sizes", "pack_fields", "unpack_fields", "index_u32", "index_u64", "weights",
)


def kernels(device: int) -> dict:
    """Return the paired layout kernels compiled for ``device``, by short name."""
    import cupy as cp

    with cp.cuda.Device(device):
        functions = cuda_module(_SOURCE, tuple(f"pm_{name}" for name in _NAMES), ("--std=c++17",))
    return {name: functions[f"pm_{name}"] for name in _NAMES}


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
    """Upload the polar pixel order once per device and detector shape; every append reads it."""
    import cupy as cp

    with cp.cuda.Device(device):
        return cp.asarray(polar_layout(shape)[0])


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
        self._detector_total = None
        self.hot_pixel_correction = None

    @property
    def nbytes(self) -> int:
        return super().nbytes + (0 if self._detector_total is None else self._detector_total.nbytes)

    def detector_total_device(self):
        """Reuse exact scan totals, or reconstruct them once after saved-form load."""
        import cupy as cp

        if self.is_released:
            raise ValueError("The resident source has been released.")
        with cp.cuda.Device(self.device):
            if self._detector_total is None:
                self._detector_total = cp.zeros(self.shape[2:], cp.uint64)
                for chunk in self.chunks:
                    raw = self.decode_blocks(chunk.first, chunk.scans)
                    self._detector_total += cp.sum(raw, axis=0, dtype=cp.uint64)
            return self._detector_total * self.valid

    def release(self) -> None:
        super().release()
        self._detector_total = None

    def decode_scan_range_device(self, first: int, stop: int, *, errors=None):
        """Decode an arbitrary scan interval with the paired stream decoder."""
        import cupy as cp

        if not 0 <= first < stop <= self.ready_scans:
            raise ValueError(f"Scan interval {(first, stop)} is outside the resident source.")
        with cp.cuda.Device(self.device):
            output = cp.empty((stop - first, *self.shape[2:]), self.dtype)
            for chunk in self.chunks:
                begin, end = max(first, chunk.first), min(stop, chunk.first + chunk.scans)
                if begin >= end:
                    continue
                aligned = begin // self.interval * self.interval
                rounded_end = min(math.ceil(end / self.interval) * self.interval,
                                  chunk.first + chunk.scans)
                decoded = self.decode_blocks(aligned, rounded_end - aligned)
                output[begin - first:end - first] = decoded[begin - aligned:end - aligned]
            return output

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
        with cp.cuda.Device(self.device):
            started = time.perf_counter()
            # One coded pair can emit seven bytes past the 2*interval guard before the
            # encoder gives up on a stream, plus one flush byte; keep those in bounds.
            scratch = cp.empty((2 * self.interval + 16, streams), cp.uint8)
            sizes, states = cp.empty(streams, cp.uint32), cp.empty(streams, cp.uint32)
            models = cp.empty(streams, cp.uint8)
            grid = ((streams + 127) // 128,)
            geometry = (raw, np.int32(raw.dtype.itemsize), u32(scans), u32(pixels), u32(self.interval))
            self.kernels["encode"](grid, (128,), (*geometry, frequencies(self.device), self.encoding, scratch, sizes, states, models, u32(streams)))
            offsets = cp.empty(streams + 1, cp.uint32)
            offsets[0] = 0
            cp.cumsum(sizes, dtype=cp.uint32, out=offsets[1:])
            payload = cp.zeros(int(offsets[-1].get()) + GUARD_BYTES, cp.uint8)
            self.kernels["compact"](grid, (128,), (*geometry, scratch, offsets, states, models, payload, u32(streams)))
            groups = (streams + 31) // 32
            records = cp.zeros((groups + 1) * 17, cp.uint32)
            errors = cp.zeros(1, cp.uint32)
            self.kernels["pack_offsets"](((groups * 32 + 127) // 128,), (128,), (offsets, records, errors, u32(streams)))
            if int(errors.get()[0]):
                raise ValueError("A stream group exceeds 65,535 bytes; the paired layout needs at most 65,535 detector pixels per 32 streams.")
            del scratch, sizes, states, offsets
            cp.cuda.get_current_stream().synchronize()
            self.load_metrics["encode_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            fields = field_count(self.shape[2:])
            values = cp.empty((scans, fields), cp.uint32)
            self.kernels["fields"](((scans * fields + 3) // 4,), (128,), (raw, np.int32(raw.dtype.itemsize), self.valid, values, u32(scans), u32(self.shape[2]), u32(self.shape[3]), u32(fields), self.polar_permutation))
            index_streams = scans // self.interval * fields
            widths, lengths = cp.empty(index_streams, cp.uint8), cp.empty(index_streams, cp.uint64)
            index_args = (u32(scans), u32(fields), u32(self.interval))
            index_grid = ((index_streams + 127) // 128,)
            self.kernels["field_sizes"](index_grid, (128,), (values, widths, lengths, *index_args))
            starts = cp.empty(index_streams + 1, cp.uint64)
            starts[0] = 0
            cp.cumsum(lengths, dtype=cp.uint64, out=starts[1:])
            words = cp.empty(int(starts[-1].get()), cp.uint32)
            self.kernels["pack_fields"](index_grid, (128,), (values, widths, starts, words, *index_args))
            cp.cuda.get_current_stream().synchronize()
            self.load_metrics["index_seconds"] += time.perf_counter() - started
            self.chunks.append(Chunk(self.ready_scans, scans, (payload, records, models, words, starts, widths)))
            self.ready_scans += scans
            # Keep only one detector-sized sum while the decoded batch is already here.
            started = time.perf_counter()
            if self._detector_total is None:
                self._detector_total = cp.zeros(self.shape[2:], cp.uint64)
            self._detector_total += cp.sum(raw, axis=0, dtype=cp.uint64)
            cp.cuda.get_current_stream().synchronize()
            self.load_metrics["detector_total_seconds"] = (
                self.load_metrics.get("detector_total_seconds", 0.0)
                + time.perf_counter() - started
            )

    def decode_blocks(self, first: int, scans: int, *, out=None):
        """Decode ``scans`` consecutive scans from ``first`` into native counts.

        The range must start on a coding block (multiple of 512 scans) and lie inside
        one resident chunk; ``out`` is an optional ``(scans, rows, cols)`` uint16 device
        array. Kernels run on the current stream, so a caller may prefetch blocks on a
        side stream.

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
        chunk = next((candidate for candidate in self.chunks if candidate.first <= first < candidate.first + candidate.scans), None)
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
                at = _round_up(at, ALIGN)
                entries.append({
                    "name": name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "offset": at,
                    "nbytes": int(array.nbytes),
                })
                arrays.append((at, array))
                at += int(array.nbytes)
            table.append({"first": int(chunk.first), "scans": int(chunk.scans), "arrays": entries})
        header = {
            "magic": RESIDENT_MAGIC,
            "query_abi": QUERY_ABI,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "interval": int(self.interval),
            "models": MODELS,
            "states": STATES,
            "valid": np.packbits(self.valid_pixels.ravel()).tobytes().hex(),
            "chunks": table,
            "hot_pixel_correction": self.hot_pixel_correction,
        }
        blob = json.dumps(header).encode()
        data_start = _round_up(len(blob) + 24, ALIGN)
        started = time.perf_counter()
        with open(path, "wb") as handle:
            handle.write(FILE_MAGIC + len(blob).to_bytes(8, "little") + data_start.to_bytes(8, "little") + blob)
            handle.truncate(data_start)
            for offset, array in arrays:
                handle.seek(data_start + offset)
                handle.write(array.get().tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "write_seconds": time.perf_counter() - started,
        }

    @classmethod
    def load(cls, path, *, device: int | None = None, reader=None, allocate=None):
        """Reopen a saved resident form as an exact source without decoding.

        Parameters
        ----------
        path
            File written by :meth:`save`.
        device
            CUDA device index; defaults to the current device.
        reader
            A :class:`ResidentFileReader` to share across a series of files. Its
            pinned staging is a large page-locked allocation with reader threads,
            so an application opening tens of files makes it once and closes it
            itself. Without one, a private reader is made and closed per file.
        allocate
            ``allocate(shape, dtype)`` returning the device array that each saved
            array is read into. An application that reserves one device block for
            a whole series places every file inside it (Live4DSTEM keeps about 70
            acquisitions on one GPU this way), so a load beside a running viewer
            makes no driver allocation per file. Without it, every array of the
            file is a view into one device allocation, so a series of many files
            does not lose the driver's per-allocation granularity to hundreds of
            small chunk arrays.
        """
        import cupy as cp

        header, data_start = _read_header(path)
        if header.get("magic") != RESIDENT_MAGIC or header.get("query_abi") != QUERY_ABI or header.get("models") != MODELS or header.get("states") != STATES:
            raise ValueError(f"{path} is not a {RESIDENT_MAGIC} file for query ABI {QUERY_ABI}.")
        shape = tuple(header["shape"])
        valid = np.unpackbits(np.frombuffer(bytes.fromhex(header["valid"]), np.uint8))[: shape[2] * shape[3]].astype(bool).reshape(shape[2:])
        with cp.cuda.Device(cp.cuda.Device().id if device is None else device):
            source = cls(shape, np.dtype(header["dtype"]), valid)
            source.hot_pixel_correction = header.get("hot_pixel_correction")
            if allocate is None:
                # Carve the file's arrays, in chunk order, from one 512-byte aligned device
                # buffer: separate allocations would each round up to the driver granularity.
                total = sum(
                    _round_up(spec["nbytes"], 512)
                    for chunk in header["chunks"]
                    for spec in chunk["arrays"]
                )
                arena = cp.empty(max(total, 1), cp.uint8)
                cursor = 0

                def allocate(shape, dtype):
                    nonlocal cursor
                    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    if nbytes == 0:
                        return cp.empty(shape, dtype)
                    view = arena[cursor : cursor + nbytes].view(dtype).reshape(shape)
                    cursor += _round_up(nbytes, 512)
                    return view

            file_reader = ResidentFileReader() if reader is None else reader
            try:
                source.chunks = file_reader.read(path, header, data_start, allocate)
            finally:
                # A private reader's pinned staging and threads must not outlive this
                # file, even after a failed read; a shared reader stays with its owner.
                if reader is None:
                    file_reader.close()
            source.ready_scans = int(shape[0] * shape[1])
            return source


def _read_header(path):
    """Read the JSON header and data offset of a saved form without touching its arrays."""
    with open(path, "rb") as handle:
        if handle.read(8) != FILE_MAGIC:
            raise ValueError(f"{path} is not a saved paired resident form (missing {FILE_MAGIC!r} magic).")
        length = int.from_bytes(handle.read(8), "little")
        data_start = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(length)), data_start


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
        """View a pinned slot from its first page boundary; direct I/O rejects unaligned buffers."""
        buffer = self.slots[slot]
        pad = (-buffer.ptr) % ALIGN
        return np.frombuffer(buffer, np.uint8, pad + length)[pad:]

    def _read(self, fd: int, offset: int, length: int, slot: int):
        """Read one span on a worker thread; direct I/O needs page-aligned offsets and lengths."""
        aligned = offset // ALIGN * ALIGN
        lead = offset - aligned
        want = _round_up(lead + length, ALIGN)
        target = self._host_view(slot, want)
        got = 0
        while got < want:
            count = os.preadv(fd, [memoryview(target)[got:]], aligned + got)
            if count <= 0:
                break
            got += count
        return slot, lead, got

    def read(self, path, header: dict, data_start: int, allocate) -> list[Chunk]:
        """Copy every saved array into ``allocate``d device storage; nothing is decoded."""
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
        # Pieces that follow each other in the file are read together, up to ``span``
        # bytes per read: a saved form holds thousands of small pieces, and one read
        # plus one future per piece kept the interpreter busy enough to starve a GUI
        # thread in the same process (2026-09-10). The device copies stay per piece.
        groups = []   # (file offset, length, [(piece index, offset inside the group)])
        for index, (offset, length, _, _) in enumerate(pieces):
            if groups and groups[-1][0] + groups[-1][1] == offset and groups[-1][1] + length <= self.span:
                groups[-1][2].append((index, groups[-1][1]))
                groups[-1][1] += length
            else:
                groups.append([offset, length, [(index, 0)]])
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        pending, inflight = {}, []
        try:
            def submit(group):
                while not self.free:
                    event, slot = inflight.pop(0)
                    event.synchronize()
                    self.free.append(slot)
                slot = self.free.pop()
                offset, length, _ = groups[group]
                pending[group] = self.executor.submit(self._read, fd, offset, length, slot)

            ahead = max(1, len(self.slots) - 2)
            for group in range(min(ahead, len(groups))):
                submit(group)
            with self.stream:
                for group in range(len(groups)):
                    slot, lead, got = pending.pop(group).result()
                    offset, length, members = groups[group]
                    if got < lead + length:
                        raise OSError(f"Short read of {path} at byte {offset}.")
                    view = self._host_view(slot, lead + length)
                    for index, inside in members:
                        _, piece_length, flat, at = pieces[index]
                        flat[at : at + piece_length].set(view[lead + inside : lead + inside + piece_length], stream=self.stream)
                    event = cp.cuda.Event()
                    event.record(self.stream)
                    inflight.append((event, slot))
                    if group + ahead < len(groups):
                        submit(group + ahead)
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
