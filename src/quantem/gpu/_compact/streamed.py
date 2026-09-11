"""Bounded, lossless CUDA encoding with runtime detector geometry.

This private resident profile uses byte rANS, sparse events, constants and uint16 literal
streams. Spatial sums are derived from original counts while each input chunk is
available. Neither coding probabilities nor indexes change scientific values.
"""

import math
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np


@cache
def kernels(device: int):
    import cupy as cp

    with cp.cuda.Device(device):
        from quantem.gpu.io.backends.cuda import _ans

        module = cp.RawModule(
            code=Path(_ans.__file__).with_suffix(".cu").read_text()
            + Path(__file__).with_name("kernels").joinpath("streamed.cu").read_text(),
            options=("--std=c++17",),
        )
        names = [
            "encode",
            "compact",
            "decode",
            "decode_range_u8",
            "decode_range_u16",
            "fields",
            "field_sizes",
            "pack_fields",
        ]
        names += [f"{op}_u{bits}" for op in ("index", "residual") for bits in (32, 64)]
        names += ["frame_u8", "frame_u16"]
        return {name: module.get_function(f"sc_{name}") for name in names}


@cache
def tables(device: int):
    """Construct generic entropy tables; all supported symbols have nonzero mass."""
    import cupy as cp

    encoding = np.empty((64, 33), np.uint32)
    decoding = np.empty((64, 1024), np.uint32)
    for model, mean in enumerate(np.geomspace(0.002, 32, 64)):
        logp = np.array([k * math.log(mean) - math.lgamma(k + 1) for k in range(33)])
        p = np.exp(logp - mean)
        p[32] = max(0.0, 1.0 - p[:32].sum())
        allocation = p / p.sum() * (1024 - 33)
        frequencies = np.floor(allocation).astype(np.uint32) + 1
        order = np.argsort(-(allocation - np.floor(allocation)), kind="stable")
        frequencies[order[: 1024 - int(frequencies.sum())]] += 1
        first = 0
        for symbol, frequency in enumerate(frequencies):
            frequency = int(frequency)
            encoding[model, symbol] = (frequency << 16) | first
            decoding[model, first : first + frequency] = (
                (frequency << 16) | (first << 6) | symbol
            )
            first += frequency
    with cp.cuda.Device(device):
        return cp.asarray(encoding), cp.asarray(decoding)


def field_count(shape: tuple[int, int]) -> int:
    return sum(
        math.ceil(shape[0] / side) * math.ceil(shape[1] / side) for side in (8, 32)
    )


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
        self.valid = cp.asarray(self.valid_pixels, dtype=cp.uint8)
        self.encoding, self.decoding = tables(self.device)
        self.kernels = kernels(self.device)
        self.native_source = None
        self.chunks: list[Chunk] = []
        self.is_released = False
        self.ready_scans = 0
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
                    (payload, offsets, models, words, starts, widths),
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

    @classmethod
    def index_encoded(cls, source, valid):
        """Borrow portable ANS/packed storage and derive only its spatial index."""
        import cupy as cp

        if source.is_released:
            raise ValueError("The encoded input has been released; reload it first.")
        with cp.cuda.Device(source._device_id):
            result = cls(source.shape, source.dtype, valid)
            result.native_source = source
            result.interval = source.block_frames
            scans = math.prod(source.shape[:2])
            # Cap decoded staging in bytes, including the concatenation copy.
            cp.get_default_memory_pool().free_all_blocks()
            free, _ = cp.cuda.runtime.memGetInfo()
            budget = min(256 * 1024**2, max(0, free - 128 * 1024**2))
            block_scans = min(scans, source.block_frames)
            block_bytes = block_scans * (
                math.prod(source.shape[2:]) * 2 + field_count(source.shape[2:]) * 4
            )
            group_blocks = min(4, budget // max(1, 2 * block_bytes))
            if group_blocks < 1:
                raise MemoryError(
                    "One encoded block exceeds available indexing staging; free CUDA memory or encode with a smaller block_frames value."
                )
            for first in range(0, scans, source.block_frames * group_blocks):
                parts = [
                    source.decode_block_device(i)
                    for i in range(
                        first // source.block_frames,
                        min(
                            math.ceil(scans / source.block_frames),
                            first // source.block_frames + group_blocks,
                        ),
                    )
                ]
                raw = parts[0] if len(parts) == 1 else cp.concatenate(parts)
                arrays = (
                    cp.empty(0, cp.uint8),
                    cp.empty(0, cp.uint32),
                    cp.empty(0, cp.uint8),
                    *result._index(raw),
                )
                result.chunks.append(Chunk(first, raw.shape[0], arrays))
                result.ready_scans += raw.shape[0]
                del raw, parts
            return result

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

    def decode_scan_range_device(self, first: int, stop: int, *, errors=None):
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
            output = cp.empty((stop - first, *self.shape[2:]), self.dtype)
            for chunk in self.chunks:
                overlap_first = max(first, chunk.first)
                overlap_stop = min(stop, chunk.first + chunk.scans)
                if overlap_first >= overlap_stop:
                    continue
                local_first = overlap_first - chunk.first
                local_stop = overlap_stop - chunk.first
                first_stream = (local_first // self.interval) * pixels
                stop_stream = math.ceil(local_stop / self.interval) * pixels
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
                    ),
                )
            if owns_errors and int(errors.get()[0]):
                raise ValueError("An encoded count stream failed reconstruction.")
            return output

    def release(self) -> None:
        """Drop this owner's references without invalidating active borrowed sessions."""
        self.chunks.clear()
        self.is_released = True
