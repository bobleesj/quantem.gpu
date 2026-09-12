"""Bounded Metal encoding for exact runtime count-ANS residents."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

from ._ans import MPSANSArray
from .packed import _allocate_shared, _buffer_view, _complete, _metal_module, _release


@lru_cache(maxsize=1)
def _tables_numpy() -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed probability tables shared with the CUDA resident codec."""
    encoding = np.empty((64, 33), np.uint32)
    decoding = np.empty((64, 1024), np.uint32)
    for model, mean in enumerate(np.geomspace(0.002, 32, 64)):
        logp = np.array(
            [index * math.log(mean) - math.lgamma(index + 1) for index in range(33)]
        )
        probability = np.exp(logp - mean)
        probability[32] = max(0.0, 1.0 - probability[:32].sum())
        allocation = probability / probability.sum() * (1024 - 33)
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
    return encoding, decoding


@lru_cache(maxsize=1)
def _runtime():
    metal = _metal_module()
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("Runtime count-ANS needs an available Metal GPU.")
    options = metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(False)
    source = (Path(__file__).parent / "kernels" / "streamed_counts.msl").read_text()
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        raise RuntimeError(f"Runtime count-ANS Metal compilation failed: {error}")
    pipelines = {}
    for name in (
        "encode",
        "compact",
        "decode_range",
        "detector_total",
        "normalize",
        "reduce",
    ):
        function = library.newFunctionWithName_(f"streamed_counts_{name}")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None:
            raise RuntimeError(f"Runtime count-ANS {name} pipeline failed: {error}")
        pipelines[name] = pipeline
    return device, metal, device.newCommandQueue(), pipelines


def _upload(device, metal, values: np.ndarray, label: str):
    values = np.ascontiguousarray(values)
    buffer = _allocate_shared(device, metal, max(1, values.nbytes), label)
    if values.nbytes:
        _buffer_view(buffer, values.nbytes)[:] = memoryview(values).cast("B")
    return buffer


@dataclass
class _Chunk:
    first: int
    scans: int
    buffers: tuple

    @property
    def nbytes(self) -> int:
        return sum(int(buffer.length()) for buffer in self.buffers)


class MPSStreamedCounts:
    """Keep complete HDF5 counts in the CUDA-equivalent runtime ANS layout."""

    interval = 512
    summary_batch_mode = "individual"

    def __init__(self, shape, dtype, valid=None):
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
        self.valid_pixels = (
            np.ones(self.shape[2:], bool)
            if valid is None
            else np.asarray(valid, bool).copy()
        )
        if self.valid_pixels.shape != self.shape[2:]:
            raise ValueError("Detector validity must match the native detector shape.")
        self._device, self._metal, self._queue, self._pipelines = _runtime()
        encoding, decoding = _tables_numpy()
        self._encoding = _upload(self._device, self._metal, encoding, "ANS encoding")
        self._decoding = _upload(self._device, self._metal, decoding, "ANS decoding")
        self._valid = _upload(
            self._device,
            self._metal,
            self.valid_pixels.astype(np.uint8),
            "ANS valid pixels",
        )
        self._errors = _allocate_shared(self._device, self._metal, 4, "ANS errors")
        self.chunks: list[_Chunk] = []
        self.ready_scans = 0
        self.is_released = False
        self.load_metrics = {"encode_seconds": 0.0}

    @property
    def device(self):
        import torch

        return torch.device("mps")

    @property
    def resident_bytes(self) -> int:
        shared = sum(
            int(buffer.length())
            for buffer in (self._encoding, self._decoding, self._valid, self._errors)
            if buffer is not None
        )
        return shared + sum(chunk.nbytes for chunk in self.chunks)

    @property
    def nbytes(self) -> int:
        return self.resident_bytes

    @property
    def logical_nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def _check_resident(self):
        if self.is_released:
            raise RuntimeError("The ANS source was released; load it again.")

    def _clear_errors(self):
        _buffer_view(self._errors)[:] = b"\0" * 4

    def _check_errors(self):
        if int(np.frombuffer(_buffer_view(self._errors), np.uint32)[0]):
            raise ValueError("An encoded count stream failed exact reconstruction.")

    def _dispatch_threads(self, encoder, count: int):
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._metal.MTLSizeMake(int(count), 1, 1),
            self._metal.MTLSizeMake(128, 1, 1),
        )

    def append(self, raw) -> None:
        """Encode one consecutive Metal-backed native-count block."""
        import time

        self._check_resident()
        if (
            not hasattr(raw, "_mtl")
            or raw._mtl is None
            or np.dtype(raw.dtype) != self.dtype
            or raw.ndim != 3
            or tuple(raw.shape[1:]) != self.shape[2:]
        ):
            raise ValueError(
                "Provide a contiguous Metal-backed native-count scan block."
            )
        scans = int(raw.shape[0])
        pixels = math.prod(self.shape[2:])
        if scans < 1 or self.ready_scans + scans > math.prod(self.shape[:2]):
            raise ValueError("ANS input block exceeds the declared scan geometry.")
        streams = math.ceil(scans / self.interval) * pixels
        scratch_bytes = (2 * min(scans, self.interval) + 4) * streams
        if scratch_bytes >= int(self._device.maxBufferLength()):
            raise MemoryError("ANS encoding scratch exceeds Metal's buffer limit.")
        scratch = sizes = states = models = offsets = payload = None
        started = time.perf_counter()
        try:
            scratch = _allocate_shared(
                self._device, self._metal, scratch_bytes, "ANS encoding scratch"
            )
            sizes = _allocate_shared(self._device, self._metal, streams * 4, "ANS sizes")
            states = _allocate_shared(self._device, self._metal, streams * 4, "ANS states")
            models = _allocate_shared(self._device, self._metal, streams, "ANS models")
            parameters = np.asarray(
                [scans, pixels, self.interval, streams, self.dtype.itemsize],
                dtype=np.uint64,
            ).tobytes()
            command = self._queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipelines["encode"])
            for index, buffer in enumerate(
                (raw._mtl, self._encoding, scratch, sizes, states, models)
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
            self._dispatch_threads(encoder, streams)
            encoder.endEncoding()
            _complete(command, "ANS encode")

            offsets = _allocate_shared(
                self._device, self._metal, (streams + 1) * 4, "ANS offsets"
            )
            offsets_view = np.frombuffer(_buffer_view(offsets), np.uint32)
            offsets_view[0] = 0
            np.cumsum(
                np.frombuffer(_buffer_view(sizes), np.uint32),
                dtype=np.uint32,
                out=offsets_view[1:],
            )
            payload_bytes = int(offsets_view[-1])
            payload = _allocate_shared(
                self._device, self._metal, max(1, payload_bytes), "ANS payload"
            )
            command = self._queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipelines["compact"])
            for index, buffer in enumerate(
                (raw._mtl, scratch, offsets, states, models, payload)
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
            self._dispatch_threads(encoder, streams)
            encoder.endEncoding()
            _complete(command, "ANS compact")
            self.chunks.append(
                _Chunk(self.ready_scans, scans, (payload, offsets, models))
            )
            self.ready_scans += scans
            payload = offsets = models = None
            self.load_metrics["encode_seconds"] += time.perf_counter() - started
        finally:
            for buffer in (scratch, sizes, states, models, offsets, payload):
                _release(buffer)

    def _encode_decode(
        self,
        command,
        chunk: _Chunk,
        local_first: int,
        count: int,
        output,
        output_offset_bytes: int = 0,
    ):
        pixels = math.prod(self.shape[2:])
        first_stream = (local_first // self.interval) * pixels
        stop_stream = math.ceil((local_first + count) / self.interval) * pixels
        parameters = np.asarray(
            [
                chunk.scans,
                pixels,
                self.interval,
                local_first,
                count,
                first_stream,
                stop_stream,
                self.dtype.itemsize,
            ],
            dtype=np.uint64,
        ).tobytes()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipelines["decode_range"])
        for index, buffer in enumerate(
            (*chunk.buffers, self._decoding, self._errors)
        ):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBuffer_offset_atIndex_(output, int(output_offset_bytes), 5)
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
        self._dispatch_threads(encoder, stop_stream - first_stream)
        encoder.endEncoding()

    def decode_scan_range_device(self, first: int, stop: int):
        """Decode a contiguous scan range into one caller-owned Metal buffer."""
        self._check_resident()
        first, stop = int(first), int(stop)
        scan_count = math.prod(self.shape[:2])
        if not 0 <= first < stop <= scan_count:
            raise ValueError(
                f"Scan range must be nonempty and inside [0, {scan_count})."
            )
        output = MPSANSArray(
            self._device,
            self._metal,
            (stop - first, *self.shape[2:]),
            self.dtype,
        )
        try:
            command = self._queue.commandBuffer()
            self._encode_scan_range_into(command, first, stop, output.buffer)
            _complete(command, "ANS range decode")
            self._check_errors()
            return output
        except BaseException:
            output.release()
            raise

    def _encode_scan_range_into(self, command, first: int, stop: int, output):
        """Append an exact range decode to an existing Metal command buffer."""
        self._check_resident()
        first, stop = int(first), int(stop)
        scan_count = math.prod(self.shape[:2])
        if not 0 <= first < stop <= scan_count:
            raise ValueError(
                f"Scan range must be nonempty and inside [0, {scan_count})."
            )
        self._clear_errors()
        for chunk in self.chunks:
            overlap_first = max(first, chunk.first)
            overlap_stop = min(stop, chunk.first + chunk.scans)
            if overlap_first >= overlap_stop:
                continue
            self._encode_decode(
                command,
                chunk,
                overlap_first - chunk.first,
                overlap_stop - overlap_first,
                output,
                (overlap_first - first)
                * math.prod(self.shape[2:])
                * self.dtype.itemsize,
            )

    def decode_block_device(self, block_index: int):
        """Decode one retained input chunk for exact parity testing."""
        if not 0 <= int(block_index) < len(self.chunks):
            raise IndexError("block_index is outside the retained ANS chunks.")
        chunk = self.chunks[int(block_index)]
        return self.decode_scan_range_device(chunk.first, chunk.first + chunk.scans)

    def mean_dp_device(self):
        """Compute the float32 mean diffraction pattern on Metal."""
        self._check_resident()
        pixels = math.prod(self.shape[2:])
        sums = MPSANSArray(self._device, self._metal, (pixels,), np.uint64)
        result = MPSANSArray(self._device, self._metal, self.shape[2:], np.float32)
        _buffer_view(sums.buffer)[:] = b"\0" * sums.nbytes
        self._clear_errors()
        try:
            command = self._queue.commandBuffer()
            for chunk in self.chunks:
                parameters = np.asarray(
                    [chunk.scans, pixels, self.interval], dtype=np.uint64
                ).tobytes()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(self._pipelines["detector_total"])
                for index, buffer in enumerate(
                    (*chunk.buffers, self._decoding, self._errors, sums.buffer)
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
                encoder.setBuffer_offset_atIndex_(self._valid, 0, 7)
                self._dispatch_threads(encoder, pixels)
                encoder.endEncoding()
            _complete(command, "ANS detector total")
            self._check_errors()
            command = self._queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipelines["normalize"])
            encoder.setBuffer_offset_atIndex_(sums.buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(result.buffer, 0, 1)
            values = np.asarray([pixels, math.prod(self.shape[:2])], np.uint64)
            encoder.setBytes_length_atIndex_(values.tobytes(), values.nbytes, 2)
            self._dispatch_threads(encoder, pixels)
            encoder.endEncoding()
            _complete(command, "ANS mean diffraction")
            return result
        except BaseException:
            result.release()
            raise
        finally:
            sums.release()

    def detector_sum_device(self, mask):
        """Compute exact masked detector sums with bounded decode staging."""
        self._check_resident()
        values = np.asarray(mask)
        if values.shape != self.shape[2:] or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"mask must have detector shape {self.shape[2:]} and be binary."
            )
        pixels = math.prod(self.shape[2:])
        mask_buffer = _upload(
            self._device,
            self._metal,
            (values.astype(bool) & self.valid_pixels).astype(np.uint8).reshape(-1),
            "ANS detector mask",
        )
        output = MPSANSArray(
            self._device, self._metal, self.shape[:2], np.uint64
        )
        decoded = MPSANSArray(
            self._device,
            self._metal,
            (max((chunk.scans for chunk in self.chunks), default=1), *self.shape[2:]),
            self.dtype,
        )
        self._clear_errors()
        try:
            command = self._queue.commandBuffer()
            for chunk in self.chunks:
                self._encode_decode(
                    command,
                    chunk,
                    0,
                    chunk.scans,
                    decoded.buffer,
                )
                parameters = np.asarray(
                    [pixels, self.dtype.itemsize, chunk.first, 0], np.uint64
                ).tobytes()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(self._pipelines["reduce"])
                for index, buffer in enumerate(
                    (decoded.buffer, mask_buffer, output.buffer)
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 3)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    self._metal.MTLSizeMake(chunk.scans, 1, 1),
                    self._metal.MTLSizeMake(128, 1, 1),
                )
                encoder.endEncoding()
            _complete(command, "ANS detector reduction")
            self._check_errors()
            return output
        except BaseException:
            output.release()
            raise
        finally:
            decoded.release()
            _release(mask_buffer)

    def detector_mean_device(self):
        """Return the per-scan mean over valid detector pixels as float32."""
        sums = self.detector_sum_device(np.ones(self.shape[2:], dtype=bool))
        result = MPSANSArray(self._device, self._metal, self.shape[:2], np.float32)
        try:
            command = self._queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipelines["normalize"])
            encoder.setBuffer_offset_atIndex_(sums.buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(result.buffer, 0, 1)
            values = np.asarray(
                [math.prod(self.shape[:2]), math.prod(self.shape[2:])],
                np.uint64,
            )
            encoder.setBytes_length_atIndex_(values.tobytes(), values.nbytes, 2)
            self._dispatch_threads(encoder, math.prod(self.shape[:2]))
            encoder.endEncoding()
            _complete(command, "ANS detector mean")
            return result
        except BaseException:
            result.release()
            raise
        finally:
            sums.release()

    def release(self) -> None:
        """Release every retained ANS buffer immediately."""
        chunks, self.chunks = self.chunks, []
        for chunk in chunks:
            for buffer in chunk.buffers:
                _release(buffer)
        for name in ("_encoding", "_decoding", "_valid", "_errors"):
            buffer = getattr(self, name, None)
            setattr(self, name, None)
            _release(buffer)
        self.is_released = True
