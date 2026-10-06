"""Bounded, lossless CUDA encoding with runtime detector geometry.

This private resident profile uses byte rANS, sparse events, constants and uint16 literal
streams. Spatial sums are derived from original counts while each input chunk is
available. Neither coding probabilities nor indexes change scientific values.
"""

import math
import time
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
    "decode",
    "decode_range_u8",
    "decode_range_u16",
    "fields",
    "field_sizes",
    "pack_fields",
    *(f"{op}_u{bits}" for op in ("index", "residual") for bits in (32, 64)),
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
    # Exact detector totals kept while encoding; only the paired layout keeps them.
    _detector_total = None

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
        return sum(chunk.nbytes for chunk in self.chunks) + self.valid.nbytes

    @property
    def index_nbytes(self) -> int:
        return sum(sum(a.nbytes for a in chunk.arrays[3:]) for chunk in self.chunks)

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
            started = time.perf_counter()
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
                (*args, self.encoding, scratch, sizes, states, models, u32(streams)),
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
            cp.cuda.get_current_stream().synchronize()
            self.load_metrics["encode_seconds"] += time.perf_counter() - started
            del scratch, sizes, states
            words, starts, widths = self._index(raw)
            self.chunks.append(
                Chunk(
                    self.ready_scans,
                    scans,
                    retain((payload, offsets, models, words, starts, widths)),
                )
            )
            self.ready_scans += scans

    def _index(self, raw):
        """Build bounded exact spatial fields from a native CUDA chunk."""
        import cupy as cp

        scans = raw.shape[0]
        u32 = np.uint32
        with cp.cuda.Device(self.device):
            started = time.perf_counter()
            fields = field_count(self.shape[2:])
            values = cp.empty((scans, fields), cp.uint32)
            self.kernels["fields"](
                ((scans * fields + 3) // 4,),
                (128,),
                (
                    raw,
                    np.int32(raw.dtype.itemsize),
                    self.valid,
                    values,
                    u32(scans),
                    u32(self.shape[2]),
                    u32(self.shape[3]),
                    u32(fields),
                ),
            )
            nstreams = math.ceil(scans / self.interval) * fields
            widths, lengths = (
                cp.empty(nstreams, cp.uint8),
                cp.empty(nstreams, cp.uint64),
            )
            args = (u32(scans), u32(fields), u32(self.interval))
            grid = ((nstreams + 127) // 128,)
            self.kernels["field_sizes"](grid, (128,), (values, widths, lengths, *args))
            starts = cp.empty(nstreams + 1, cp.uint64)
            starts[0] = 0
            cp.cumsum(lengths, dtype=cp.uint64, out=starts[1:])
            words = cp.empty(int(starts[-1].get()), cp.uint32)
            self.kernels["pack_fields"](
                grid, (128,), (values, widths, starts, words, *args)
            )
            cp.cuda.get_current_stream().synchronize()
            self.load_metrics["index_seconds"] += time.perf_counter() - started
            return words, starts, widths

    def decode_chunk(self, index: int):
        """Return one complete chunk on CUDA for explicit reconstruction checks."""
        import cupy as cp

        if self.is_released:
            raise ValueError("The resident source has been released.")
        chunk = self.chunks[index]
        pixels = math.prod(self.shape[2:])
        streams = math.ceil(chunk.scans / self.interval) * pixels
        with cp.cuda.Device(self.device):
            raw = cp.empty((chunk.scans, *self.shape[2:]), cp.uint16)
            errors = cp.zeros(1, cp.uint32)
            self.kernels["decode"](
                ((streams + 127) // 128,),
                (128,),
                (
                    *chunk.arrays[:3],
                    self.decoding,
                    raw,
                    errors,
                    np.uint32(chunk.scans),
                    np.uint32(pixels),
                    np.uint32(self.interval),
                    np.uint32(streams),
                ),
            )
            if int(errors.get()[0]):
                raise ValueError("An encoded count stream failed reconstruction.")
            return raw.astype(self.dtype, copy=False)

    def decode_scan_range_device(
        self, first: int, stop: int, *, errors=None, detector_region=None
    ):
        """Decode a contiguous scan range without expanding the acquisition."""
        import cupy as cp

        if self.is_released:
            raise ValueError("The resident source has been released.")
        scan_count = math.prod(self.shape[:2])
        first, stop = int(first), int(stop)
        if not 0 <= first < stop <= scan_count:
            raise ValueError(
                f"Scan range must be nonempty and inside [0, {scan_count}); "
                f"got ({first}, {stop})."
            )
        if first < self.released_scans:
            raise ValueError(
                f"Scans before {self.released_scans} were released; load the source again to read them."
            )
        owns_errors = errors is None
        with cp.cuda.Device(self.device):
            if errors is None:
                errors = cp.zeros(1, cp.uint32)
            elif (
                not isinstance(errors, cp.ndarray)
                or errors.shape != (1,)
                or errors.dtype != cp.uint32
                or errors.device.id != self.device
            ):
                raise ValueError(
                    "errors must be one uint32 value on the source CUDA device."
                )
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

    def detector_total_device(self):
        """Return the exact valid-pixel detector sum without dense expansion."""
        import cupy as cp

        if self.is_released:
            raise ValueError("The resident source has been released.")
        with cp.cuda.Device(self.device):
            total = cp.zeros(self.shape[2:], cp.uint64)
            errors = cp.zeros(1, cp.uint32)
            valid = self.valid.astype(self.dtype, copy=False)
            blocks = (
                self.decode_scan_range_device(
                    chunk.first,
                    chunk.first + chunk.scans,
                    errors=errors,
                )
                for chunk in self.chunks
            )
            for decoded in blocks:
                decoded *= valid[None]
                total += cp.sum(decoded, axis=0, dtype=cp.uint64)
                del decoded
            cp.cuda.get_current_stream().synchronize()
            if int(errors.get()[0]):
                raise ValueError("An encoded count stream failed reconstruction.")
            return total

    def release_scans_before(self, stop: int) -> None:
        """Free the chunks that end by scan ``stop``, so a source read once in scan order shrinks as it is read.

        Only whole chunks are freed; each owns its own allocation (see retain),
        so its memory returns to the GPU at once. Decoding a freed scan raises.
        """
        self.chunks = [chunk for chunk in self.chunks if chunk.first + chunk.scans > stop]
        self.released_scans = self.chunks[0].first if self.chunks else self.ready_scans

    def release(self) -> None:
        """Drop this owner's references without invalidating active borrowed sessions."""
        self.chunks.clear()
        self.is_released = True
