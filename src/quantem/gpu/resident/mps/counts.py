"""Bounded Metal encoding for exact runtime count-ANS residents."""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from quantem.gpu.device.metal_runtime import (
    SharedArray,
    allocate_shared,
    buffer_view,
    complete_command,
    metal_device,
    metal_module,
    metal_pipelines,
    metal_queue,
    release_buffer,
    tensor_buffer,
    upload_shared,
)
from quantem.gpu.formats.qem.reference import count_tables
from quantem.gpu.resident.mps import spatial
from quantem.gpu.resident.mps.arrays import MetalArray

_KERNELS = Path(__file__).parent / "kernels"
# runtime_spatial.msl is shared with native Swift; the codec kernels are Python's.
_SOURCE = (
    (_KERNELS / "streamed_counts.msl").read_text()
    + "\n"
    + (_KERNELS / "runtime_spatial.msl").read_text()
)
_CODEC = ("encode", "compact", "decode_range", "detector_total", "masked_sums")
_SPATIAL = (
    "camera_mask_leaves",
    "camera_mask_roots",
    "camera_index_sum_u64_simd",
    "camera_delta_u64",
    "camera_frame_native",
    "camera_fields",
    "camera_field_widths",
    "camera_pack_fields",
)


def _pipelines() -> dict:
    """Return the count-ANS codec and camera spatial-index pipelines, keyed by short name.

    Every resident shares one compiled set; compiling per source would add
    seconds to each load.
    """
    names = tuple(f"streamed_counts_{name}" for name in _CODEC) + _SPATIAL
    pipelines = metal_pipelines(_SOURCE, names, fast_math=False)
    return {name.removeprefix("streamed_counts_"): pipeline for name, pipeline in pipelines.items()}


@dataclass
class _Chunk:
    """Encoded streams for the consecutive scans ``first`` to ``first + scans``."""

    first: int
    scans: int
    buffers: tuple

    @property
    def nbytes(self) -> int:
        return sum(int(buffer.length()) for buffer in self.buffers)


class MPSStreamedCounts:
    """Keep complete HDF5 counts in the CUDA-equivalent runtime ANS layout.

    Viewers use it like a 4D array (``shape``, ``device``, ``source[row, col]``);
    every pattern and reduction is decoded from the resident streams.
    """

    interval = 512
    # quantem.widget's Show4DSTEM reads this flag to keep the source on its GPU path, without a NumPy copy.
    _is_gpu_frames = True
    ndim = 4
    det_bin = 1

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
        self._metal = metal_module()
        self._pipelines = _pipelines()
        encoding, decoding = count_tables()
        self._encoding = upload_shared(encoding, "ANS encoding")
        self._decoding = upload_shared(decoding, "ANS decoding")
        self._valid = upload_shared(self.valid_pixels.astype(np.uint8), "ANS valid pixels")
        self._errors = allocate_shared(4, "ANS errors")
        self.chunks: list[_Chunk] = []
        self.spatial_chunks: list[tuple] = []
        self.ready_scans = 0
        self.released_scans = 0
        self.is_released = False
        self.load_metrics = {"encode_seconds": 0.0}

    @property
    def device(self):
        return torch.device("mps")

    @property
    def scan_shape(self) -> tuple:
        return self.shape[:2]

    @property
    def det_shape(self) -> tuple:
        return self.shape[2:]

    @property
    def n_frames(self) -> int:
        return math.prod(self.shape[:2])

    def numel(self) -> int:
        """Logical count of native values; nothing is expanded."""
        return math.prod(self.shape)

    def __getitem__(self, position):
        """Decode one pattern, ``source[index]`` or ``source[row, col]``, on MPS."""
        if isinstance(position, (int, np.integer)):
            index = int(position)
        elif isinstance(position, tuple) and len(position) == 2:
            row, column = map(int, position)
            if not (0 <= row < self.shape[0] and 0 <= column < self.shape[1]):
                raise IndexError("Scan position lies outside this acquisition.")
            index = row * self.shape[1] + column
        else:
            raise TypeError(
                "Select one diffraction pattern with source[index] or source[row, col]."
            )
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Scan index must be in [0, {self.n_frames}); got {index}.")
        return self._decode_scan_range_torch(index, index + 1)[0]

    @property
    def resident_bytes(self) -> int:
        shared = sum(
            int(buffer.length())
            for buffer in (self._encoding, self._decoding, self._valid, self._errors)
            if buffer is not None
        )
        index_bytes = sum(
            int(buffer.length()) for chunk in self.spatial_chunks for buffer in chunk
        )
        return shared + sum(chunk.nbytes for chunk in self.chunks) + index_bytes

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
        buffer_view(self._errors)[:] = b"\0" * 4

    def _check_errors(self):
        if int(np.frombuffer(buffer_view(self._errors), np.uint32)[0]):
            raise ValueError("An encoded count stream failed exact reconstruction.")

    def _dispatch_threads(self, encoder, count: int):
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._metal.MTLSizeMake(int(count), 1, 1),
            self._metal.MTLSizeMake(128, 1, 1),
        )

    def append(self, raw) -> None:
        """Encode one consecutive ``(scan, row, col)`` native-count block held in a ``SharedArray``."""
        self._check_resident()
        if (
            not isinstance(raw, SharedArray)
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
        if scratch_bytes >= int(metal_device().maxBufferLength()):
            raise MemoryError("ANS encoding scratch exceeds Metal's buffer limit.")
        scratch = sizes = states = models = offsets = payload = None
        started = time.perf_counter()
        try:
            scratch = allocate_shared(scratch_bytes, "ANS encoding scratch")
            sizes = allocate_shared(streams * 4, "ANS sizes")
            states = allocate_shared(streams * 4, "ANS states")
            models = allocate_shared(streams, "ANS models")
            parameters = np.asarray(
                [scans, pixels, self.interval, streams, self.dtype.itemsize],
                dtype=np.uint64,
            ).tobytes()
            command = metal_queue().commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipelines["encode"])
            for index, buffer in enumerate(
                (raw._mtl, self._encoding, scratch, sizes, states, models)
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
            self._dispatch_threads(encoder, streams)
            encoder.endEncoding()
            complete_command(command, "ANS encode")

            offsets = allocate_shared((streams + 1) * 4, "ANS offsets")
            offsets_view = np.frombuffer(buffer_view(offsets), np.uint32)
            offsets_view[0] = 0
            np.cumsum(
                np.frombuffer(buffer_view(sizes), np.uint32),
                dtype=np.uint32,
                out=offsets_view[1:],
            )
            payload_bytes = int(offsets_view[-1])
            payload = allocate_shared(max(1, payload_bytes), "ANS payload")
            command = metal_queue().commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipelines["compact"])
            for index, buffer in enumerate(
                (raw._mtl, scratch, offsets, states, models, payload)
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
            self._dispatch_threads(encoder, streams)
            encoder.endEncoding()
            complete_command(command, "ANS compact")
            self.chunks.append(
                _Chunk(self.ready_scans, scans, (payload, offsets, models))
            )
            self.ready_scans += scans
            payload = offsets = models = None
            self.load_metrics["encode_seconds"] += time.perf_counter() - started
        finally:
            for buffer in (scratch, sizes, states, models, offsets, payload):
                release_buffer(buffer)

    def _encode_decode(
        self,
        command,
        chunk: _Chunk,
        local_first: int,
        count: int,
        output,
        output_offset_bytes: int = 0,
    ):
        """Append one chunk's decode of ``count`` scans from ``local_first`` to ``command``.

        Only the interval streams that overlap those scans are dispatched, so a
        range read never decodes the rest of the chunk. Values land
        ``output_offset_bytes`` into ``output``.
        """
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
        encoder.setComputePipelineState_(
            self._pipelines["camera_frame_native" if count == 1 else "decode_range"]
        )
        for index, buffer in enumerate((*chunk.buffers, self._decoding, self._errors)):
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
        output = MetalArray((stop - first, *self.shape[2:]), self.dtype)
        try:
            command = metal_queue().commandBuffer()
            self._encode_scan_range_into(command, first, stop, output._mtl)
            complete_command(command, "ANS range decode")
            self._check_errors()
            return output
        except BaseException:
            output.release()
            raise

    def extract_diffraction_device(self, scan_row: int, scan_column: int):
        """Return one exact native-count pattern in caller-owned Metal storage."""
        self._check_resident()
        if not 0 <= scan_row < self.shape[0] or not 0 <= scan_column < self.shape[1]:
            raise IndexError(
                "Choose a scan row and column inside the loaded acquisition."
            )
        first = scan_row * self.shape[1] + scan_column
        output = self.decode_scan_range_device(first, first + 1)
        output.shape = self.shape[2:]
        return output

    def _decode_scan_range_torch(self, first: int, stop: int):
        """Decode directly into independently owned Torch accelerator storage."""
        self._check_resident()
        first, stop = int(first), int(stop)
        scan_count = math.prod(self.shape[:2])
        if not 0 <= first < stop <= scan_count:
            raise ValueError(
                f"Scan range must be nonempty and inside [0, {scan_count})."
            )
        output = torch.empty(
            (stop - first, *self.shape[2:]),
            dtype=getattr(torch, self.dtype.name),
            device="mps",
        )
        # Finish prior Torch work before the native queue writes recycled
        # allocator storage. The native command finishes before returning.
        torch.mps.synchronize()
        buffer = tensor_buffer(output)
        if int(buffer.length()) < output.numel() * output.element_size() or int(
            buffer.device().registryID()
        ) != int(metal_device().registryID()):
            raise RuntimeError(
                "Torch storage must belong to the same Metal device as the resident."
            )
        command = metal_queue().commandBuffer()
        self._encode_scan_range_into(command, first, stop, buffer)
        complete_command(command, "ANS Torch range read")
        self._check_errors()
        return output

    def _encode_scan_range_into(self, command, first: int, stop: int, output):
        """Append an exact range decode to an existing Metal command buffer."""
        self._check_resident()
        first, stop = int(first), int(stop)
        scan_count = math.prod(self.shape[:2])
        if not 0 <= first < stop <= scan_count:
            raise ValueError(
                f"Scan range must be nonempty and inside [0, {scan_count})."
            )
        if first < self.released_scans:
            raise ValueError(
                f"Scans before {self.released_scans} were released; load the source again to read them."
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

    def detector_total_device(self):
        """Exact uint64 sum of every scan's counts per detector pixel, on Metal.

        Each pixel's streams are decoded once, without a frame buffer. Flagged
        pixels are 0, as in CUDA's ``StreamedCounts.detector_total_device``, so
        the mean pattern counts the same pixels as every exact detector sum.
        """
        self._check_resident()
        pixels = math.prod(self.shape[2:])
        total = MetalArray(self.shape[2:], np.uint64)
        buffer_view(total._mtl)[:] = b"\0" * total.nbytes
        self._clear_errors()
        try:
            command = metal_queue().commandBuffer()
            for chunk in self.chunks:
                parameters = np.asarray(
                    [chunk.scans, pixels, self.interval], dtype=np.uint64
                ).tobytes()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(self._pipelines["detector_total"])
                for index, buffer in enumerate(
                    (*chunk.buffers, self._decoding, self._errors, total._mtl)
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
                encoder.setBuffer_offset_atIndex_(self._valid, 0, 7)
                self._dispatch_threads(encoder, pixels)
                encoder.endEncoding()
            complete_command(command, "ANS detector total")
            self._check_errors()
            return total
        except BaseException:
            total.release()
            raise

    def mean_dp_device(self):
        """Return the float32 mean diffraction pattern in Metal storage.

        The exact uint64 detector total is summed on Metal; Metal has no
        float64, so the small pattern is divided by the scan count in float64
        on the host and rounded once to float32, the rule of every mean
        pattern and the CUDA result. A float32 division on Metal rounded the
        total first and lost counts above 2^24.
        """
        sums = self.detector_total_device()
        try:
            mean = (sums.get() / math.prod(self.shape[:2])).astype(np.float32)
        finally:
            sums.release()
        result = MetalArray(self.shape[2:], np.float32)
        buffer_view(result._mtl)[: mean.nbytes] = memoryview(mean).cast("B")
        return result

    def detector_sum_device(self, mask):
        """Compute exact masked detector sums from the resident streams."""
        self._check_resident()
        values = np.asarray(mask)
        if values.shape != self.shape[2:] or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"mask must have detector shape {self.shape[2:]} and be binary."
            )
        if len(self.spatial_chunks) == len(self.chunks) and self.chunks:
            return spatial.detector_sum(self, values)
        output = MetalArray(self.shape[:2], np.uint64)
        try:
            np.frombuffer(buffer_view(output._mtl), np.uint64)[:] = (
                self.masked_code_sums(values)
            )
            return output
        except BaseException:
            output.release()
            raise

    def weighted_code_sums(self, weights) -> np.ndarray:
        """Exact uint64 per-scan sums of nonnegative integer pixel weights times counts.

        Centre-of-mass moments weight each pixel by its row or column, which a
        binary mask cannot express. Flagged pixels weigh nothing.
        """
        self._check_resident()
        return spatial.weighted_sum(self, weights)

    def masked_code_sums(self, mask) -> np.ndarray:
        """Exact uint64 per-scan sums over the mask's valid pixels.

        Only the listed pixels' streams are decoded, never the whole detector.
        Each pass lists at most 65,536 pixels, so every device total stays
        below 2^32; passes are added exactly on the host.
        """
        self._check_resident()
        values = np.asarray(mask)
        if values.shape != self.shape[2:] or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"mask must have detector shape {self.shape[2:]} and be binary."
            )
        listed = np.flatnonzero(values.astype(bool) & self.valid_pixels)
        scans, pixels = math.prod(self.shape[:2]), math.prod(self.shape[2:])
        totals = np.zeros(scans, np.uint64)
        if not listed.size:
            return totals
        sums = allocate_shared(scans * 4, "ANS masked sums")
        try:
            view = np.frombuffer(buffer_view(sums, scans * 4), np.uint32)
            for first in range(0, listed.size, 1 << 16):
                batch = listed[first : first + (1 << 16)].astype(np.uint32)
                pixel_buffer = upload_shared(batch, "ANS pixels")
                try:
                    view[:] = 0
                    self._clear_errors()
                    command = metal_queue().commandBuffer()
                    for chunk in self.chunks:
                        parameters = np.asarray(
                            [chunk.scans, pixels, self.interval, batch.size], np.uint64
                        ).tobytes()
                        encoder = command.computeCommandEncoder()
                        encoder.setComputePipelineState_(self._pipelines["masked_sums"])
                        for index, buffer in enumerate(
                            (*chunk.buffers, self._decoding, self._errors)
                        ):
                            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                        encoder.setBuffer_offset_atIndex_(sums, chunk.first * 4, 5)
                        encoder.setBuffer_offset_atIndex_(pixel_buffer, 0, 6)
                        encoder.setBytes_length_atIndex_(parameters, len(parameters), 7)
                        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                            self._metal.MTLSizeMake(
                                math.ceil(batch.size / 256),
                                math.ceil(chunk.scans / self.interval),
                                1,
                            ),
                            self._metal.MTLSizeMake(256, 1, 1),
                        )
                        encoder.endEncoding()
                    complete_command(command, "ANS masked sums")
                    self._check_errors()
                    totals += view
                finally:
                    release_buffer(pixel_buffer)
        finally:
            release_buffer(sums)
        return totals

    def detector_delta_device(self, mask, previous=None, output=None):
        """Update an exact uint64 detector image from changed mask membership."""
        self._check_resident()
        if previous is None or output is None:
            return self.detector_sum_device(mask)
        return spatial.detector_delta(self, mask, previous, output)

    def release_scans_before(self, stop: int) -> None:
        """Free the chunks that end by scan ``stop``, so a source read once in scan order shrinks as it is read.

        Only whole chunks are freed, with their spatial sums; decoding a freed
        scan raises.
        """
        count = sum(chunk.first + chunk.scans <= stop for chunk in self.chunks)
        for chunk in self.chunks[:count]:
            for buffer in chunk.buffers:
                release_buffer(buffer)
        for buffers in self.spatial_chunks[:count]:
            for buffer in buffers:
                release_buffer(buffer)
        self.chunks, self.spatial_chunks = self.chunks[count:], self.spatial_chunks[count:]
        self.released_scans = self.chunks[0].first if self.chunks else self.ready_scans

    def release(self) -> None:
        """Release every retained ANS buffer immediately."""
        for chunk in self.spatial_chunks:
            for buffer in chunk:
                release_buffer(buffer)
        self.spatial_chunks = []
        chunks, self.chunks = self.chunks, []
        for chunk in chunks:
            for buffer in chunk.buffers:
                release_buffer(buffer)
        for buffer in (self._encoding, self._decoding, self._valid, self._errors):
            release_buffer(buffer)
        self._encoding = self._decoding = self._valid = self._errors = None
        self.is_released = True
