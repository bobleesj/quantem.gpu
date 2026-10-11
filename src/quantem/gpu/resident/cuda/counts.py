"""Bounded, lossless CUDA encoding with runtime detector geometry.

This private resident profile uses byte rANS, sparse events, constants and uint16 literal
streams. Spatial sums are derived from original counts while each input chunk is
available. Neither coding probabilities nor indexes change scientific values.
"""

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


def retain(arrays) -> tuple:
    """Copy the arrays a resident keeps into one exact allocation outside the memory pool.

    The arrays are packed back to back at 256-byte offsets and returned as views.
    A kept array carved from a pooled block that also served decode or encode
    scratch holds the whole block until the resident is released, so neither
    that scratch nor, after release, the resident returns to the GPU. One
    allocation per chunk, freed with its last view, because separate
    allocations are each rounded up to whole 2 MiB pages.
    """
    import cupy as cp

    sizes = [math.ceil(array.nbytes / 256) * 256 for array in arrays]
    with cp.cuda.using_allocator():
        storage = cp.empty(sum(sizes), cp.uint8)
    views = []
    for array, start in zip(arrays, [0, *accumulate(sizes)]):
        view = storage[start : start + array.nbytes].view(array.dtype).reshape(array.shape)
        view[...] = array
        views.append(view)
    # The sources go back to the memory pool when the caller drops them, and the
    # loader's next decode can reuse that memory before these copies have run;
    # without this wait, loads under GPU contention kept another chunk's counts.
    cp.cuda.get_current_stream().synchronize()
    return tuple(views)


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
                    ((scans * leaves + 3) // 4,),
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
            # retain() waits for the stream, so the events are complete when read.
            self.chunks.append(Chunk(self.ready_scans, scans, retain(kept)))
            self.ready_scans += scans
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
                for chunk in self.chunks:
                    streams = math.ceil(chunk.scans / self.interval) * pixels
                    self.kernels["total"](
                        ((streams + 127) // 128,),
                        (128,),
                        (
                            *chunk.arrays[:3],
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

        Only whole chunks are freed; each owns its own allocation (see retain),
        so its memory returns to the GPU at once. Decoding a freed scan raises.
        """
        kept = [chunk for chunk in self.chunks if chunk.first + chunk.scans > stop]
        if len(kept) != len(self.chunks):
            self._detector_total = None   # no request can be answered once scans are freed
        self.chunks = kept
        self.released_scans = self.chunks[0].first if self.chunks else self.ready_scans

    def release(self) -> None:
        """Drop this owner's references without invalidating active borrowed sessions."""
        self.chunks.clear()
        self._detector_total = None
        self.is_released = True
