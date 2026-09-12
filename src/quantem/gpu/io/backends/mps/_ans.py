"""Private exact count-rANS operations over authenticated normalized arrays.

Disk ownership stays with io.load. This adapter does not recognize research
folders or choose a scientific mask, dtype conversion, binning, or crop.
"""

from pathlib import Path

import numpy as np

from ..._ans_contract import _validate_arrays
from .packed import _allocate_shared, _buffer_view, _complete, _metal_module, _release

_PIPELINES = {}
_REDUCTION_STAGING_BYTES = 32 * 1024 * 1024


def _pipelines(device, metal):
    """Compile the exact stream recurrence once per physical device."""
    key = int(device.registryID())
    if key not in _PIPELINES:
        source = (Path(__file__).parent / "kernels" / "ans_counts.msl").read_text()
        library, error = device.newLibraryWithSource_options_error_(source, None, None)
        if library is None or error is not None:
            raise RuntimeError(f"Count-rANS Metal compilation failed: {error}")
        pipelines = {}
        for name in (
            "validate",
            "decode",
            "gather",
            "reduce",
            "measure_packed",
            "write_packed",
            "packed_decode",
            "packed_gather",
        ):
            function = library.newFunctionWithName_(f"ans_counts_{name}")
            pipeline, error = device.newComputePipelineStateWithFunction_error_(
                function, None
            )
            if pipeline is None or error is not None:
                raise RuntimeError(f"Count-rANS {name} pipeline failed: {error}")
            pipelines[name] = pipeline
        _PIPELINES[key] = pipelines
    return _PIPELINES[key]


class MPSANSArray:
    """Own one exact native-count or uint64 product buffer until release.

    ``buffer`` may be consumed directly by a same-device Metal operation.
    ``to_numpy`` copies the requested result explicitly; it does not borrow a
    mutable view that outlives the buffer. Call ``release`` after the consumer
    finishes. These private ownership objects are not a second public API.
    """

    def __init__(self, device, metal, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
        self.buffer = (
            _allocate_shared(device, metal, self.nbytes, "ANS output")
            if self.nbytes
            else None
        )
        self.is_released = False

    def to_numpy(self) -> np.ndarray:
        """Copy only this requested result, preserving shape and native dtype."""
        if self.is_released:
            raise RuntimeError("The ANS output was released; request a new result.")
        if self.buffer is None:
            return np.empty(self.shape, self.dtype)
        return (
            np.frombuffer(_buffer_view(self.buffer), self.dtype)
            .reshape(self.shape)
            .copy()
        )

    def to_torch(self):
        """Copy this bounded Metal result into a Torch MPS tensor."""
        if self.is_released:
            raise RuntimeError("The ANS output was released; request a new result.")
        import torch

        view = np.frombuffer(_buffer_view(self.buffer), self.dtype).reshape(self.shape)
        return torch.from_numpy(view).to("mps")

    def release(self):
        """Release exactly this buffer, leaving its source and other results intact."""
        buffer, self.buffer = self.buffer, None
        _release(buffer)
        self.is_released = True


class _MPSResidentCounts:
    """Keep exact count streams resident with bounded decode and product scratch.

    The constructor validates complete streams on the physical accelerator
    without materializing a dense volume. Selected patterns decode only their
    entropy-stream prefixes; binary detector products use bounded native-count
    staging and exact uint64 reduction. Returned outputs are caller-owned.
    This is a correctness primitive, not a real-time performance claim.
    """

    def __init__(
        self,
        *,
        shape,
        block_frames,
        scale,
        payload,
        offsets,
        model_ids,
        context_offsets,
        symbols,
        cumulative,
        frequencies,
        literal,
        dtype="uint16",
    ):
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError("Native count dtype must be uint8 or uint16.")
        self.shape, arrays = _validate_arrays(
            shape=shape,
            block_frames=block_frames,
            scale=scale,
            payload=payload,
            offsets=offsets,
            model_ids=model_ids,
            context_offsets=context_offsets,
            symbols=symbols,
            cumulative=cumulative,
            frequencies=frequencies,
            literal=literal,
        )
        self.block_frames = int(block_frames)
        self.scale = int(scale)
        self._scan_count = self.shape[0] * self.shape[1]
        self._detector_count = self.shape[2] * self.shape[3]
        self._block_count = (
            self._scan_count + self.block_frames - 1
        ) // self.block_frames
        self._stream_count = self._block_count * self._detector_count
        if self._stream_count >= 2**32:
            raise ValueError(
                "This Metal stream index requires fewer than 2**32 streams."
            )
        self._metal = _metal_module()
        self._device = self._metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Count-rANS requires a physical Metal device.")
        self._pipelines = _pipelines(self._device, self._metal)
        self._queue = self._device.newCommandQueue()
        self._buffers = []
        self._errors = None
        self.is_released = False
        try:
            for array in arrays:
                buffer = _allocate_shared(
                    self._device, self._metal, max(1, array.nbytes), "ANS source"
                )
                self._buffers.append(buffer)
                if array.nbytes:
                    _buffer_view(buffer, array.nbytes)[:] = memoryview(array).cast("B")
            self._errors = _allocate_shared(self._device, self._metal, 4, "ANS errors")
            self._dispatch("validate", self._parameters(), self._stream_count)
        except BaseException:
            self.release()
            raise

    @property
    def resident_bytes(self) -> int:
        """Actual owned payload, tables, index, and error-buffer byte lengths."""
        return sum(int(buffer.length()) for buffer in self._buffers) + (
            4 if self._errors is not None else 0
        )

    @property
    def nbytes(self) -> int:
        return self.resident_bytes

    @property
    def logical_nbytes(self) -> int:
        return self._scan_count * self._detector_count * self.dtype.itemsize

    def _require_resident(self):
        if self.is_released:
            raise RuntimeError(
                "The ANS source was released; load a new source before reading it."
            )

    def _parameters(self, *, block=0, first=0, count=0):
        return np.asarray(
            [
                self._scan_count,
                self._detector_count,
                self.block_frames,
                self.scale,
                np.iinfo(self.dtype).max,
                block,
                first,
                count,
                self.dtype.itemsize,
            ],
            dtype=np.uint64,
        ).tobytes()

    def _check(self):
        status = int(np.frombuffer(_buffer_view(self._errors), dtype=np.uint32)[0])
        if status & 2:
            raise ValueError(
                "Decoded counts exceed the declared native dtype; preserve uint16."
            )
        if status:
            raise ValueError(
                "Malformed rANS stream: normalization or exact terminal state failed."
            )

    def _encode(
        self,
        command,
        name,
        parameters,
        width,
        height=1,
        output=None,
        requested=None,
        word_offsets=None,
    ):
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipelines[name])
        for index, buffer in enumerate(self._buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBuffer_offset_atIndex_(self._errors, 0, 8)
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 9)
        if output is not None:
            encoder.setBuffer_offset_atIndex_(output, 0, 10)
        if requested is not None:
            encoder.setBuffer_offset_atIndex_(requested, 0, 11)
        if word_offsets is not None:
            encoder.setBuffer_offset_atIndex_(word_offsets, 0, 12)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._metal.MTLSizeMake(width, height, 1),
            self._metal.MTLSizeMake(128, 1, 1),
        )
        encoder.endEncoding()

    def _dispatch(self, name, parameters, width, height=1, output=None, requested=None):
        _buffer_view(self._errors)[:] = b"\0" * 4
        command = self._queue.commandBuffer()
        self._encode(command, name, parameters, width, height, output, requested)
        _complete(command, f"ANS {name}")
        self._check()

    def decode_block_device(self, block_index: int):
        """Decode one encoded block to a caller-owned native-count Metal buffer."""
        self._require_resident()
        if (
            not isinstance(block_index, (int, np.integer))
            or not 0 <= block_index < self._block_count
        ):
            raise IndexError(
                f"block_index must be in [0, {self._block_count}); got {block_index}."
            )
        count = min(
            self.block_frames, self._scan_count - block_index * self.block_frames
        )
        output = MPSANSArray(
            self._device, self._metal, (count, *self.shape[2:]), self.dtype
        )
        try:
            self._dispatch(
                "decode",
                self._parameters(block=block_index, count=count),
                self._detector_count,
                output=output.buffer,
            )
            return output
        except BaseException:
            output.release()
            raise

    def gather_diffraction_device(self, scan_positions):
        """Return only requested DPs, retaining order, duplicates, and native dtype."""
        self._require_resident()
        positions = np.asarray(scan_positions)
        if (
            positions.ndim != 2
            or positions.shape[1] != 2
            or positions.dtype.kind not in "iu"
        ):
            raise ValueError(
                "scan_positions must be an integer (N, 2) array in row, column order."
            )
        if np.any(positions < 0) or np.any(positions >= self.shape[:2]):
            raise IndexError(f"Scan positions must be inside {self.shape[:2]}.")
        output = MPSANSArray(
            self._device, self._metal, (len(positions), *self.shape[2:]), self.dtype
        )
        requested = None
        try:
            if len(positions):
                coordinates = positions.astype(np.uint64, copy=False)
                flat = coordinates[:, 0] * self.shape[1] + coordinates[:, 1]
                requested = _allocate_shared(
                    self._device, self._metal, flat.nbytes, "ANS selected positions"
                )
                _buffer_view(requested)[:] = memoryview(
                    np.ascontiguousarray(flat)
                ).cast("B")
                self._dispatch(
                    "gather",
                    self._parameters(first=len(positions)),
                    self._detector_count,
                    len(positions),
                    output.buffer,
                    requested,
                )
            return output
        except BaseException:
            output.release()
            raise
        finally:
            _release(requested)

    def extract_diffraction_device(self, scan_row: int, scan_column: int):
        output = self.gather_diffraction_device(np.asarray([[scan_row, scan_column]]))
        output.shape = self.shape[2:]
        return output

    def detector_sum_device(self, mask):
        """Sum a binary mask exactly with bounded native-count decode staging."""
        self._require_resident()
        values = np.asarray(mask)
        if values.shape != self.shape[2:] or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"mask must have detector shape {self.shape[2:]} and contain only zero or one."
            )
        mask_values = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
        output = MPSANSArray(self._device, self._metal, self.shape[:2], np.uint64)
        staging = mask_buffer = None
        capacity = min(
            self.block_frames,
            max(
                1,
                _REDUCTION_STAGING_BYTES
                // (self._detector_count * self.dtype.itemsize),
            ),
        )
        try:
            staging = _allocate_shared(
                self._device,
                self._metal,
                capacity * self._detector_count * self.dtype.itemsize,
                "ANS reduction staging",
            )
            mask_buffer = _allocate_shared(
                self._device, self._metal, mask_values.nbytes, "ANS detector mask"
            )
            _buffer_view(mask_buffer)[:] = memoryview(mask_values).cast("B")
            for block in range(self._block_count):
                block_count = min(
                    self.block_frames, self._scan_count - block * self.block_frames
                )
                for first in range(0, block_count, capacity):
                    count = min(capacity, block_count - first)
                    _buffer_view(self._errors)[:] = b"\0" * 4
                    command = self._queue.commandBuffer()
                    self._encode(
                        command,
                        "decode",
                        self._parameters(block=block, first=first, count=count),
                        self._detector_count,
                        output=staging,
                    )
                    encoder = command.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._pipelines["reduce"])
                    for index, buffer in enumerate(
                        (staging, mask_buffer, output.buffer)
                    ):
                        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                    parameters = np.asarray(
                        [
                            self._detector_count,
                            self.dtype.itemsize,
                            block * self.block_frames + first,
                            0,
                        ],
                        dtype=np.uint64,
                    ).tobytes()
                    encoder.setBytes_length_atIndex_(parameters, len(parameters), 3)
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        self._metal.MTLSizeMake(count, 1, 1),
                        self._metal.MTLSizeMake(128, 1, 1),
                    )
                    encoder.endEncoding()
                    _complete(command, "ANS detector sum")
                    self._check()
            return output
        except BaseException:
            output.release()
            raise
        finally:
            _release(mask_buffer)
            _release(staging)

    def release(self):
        """Release only this source; already-returned outputs remain valid."""
        buffers, self._buffers = self._buffers, []
        errors, self._errors = self._errors, None
        for buffer in buffers:
            _release(buffer)
        _release(errors)
        self.is_released = True


class MPSANSResidentCounts(_MPSResidentCounts):
    """Own validated native-count ANS streams, independently of disk framing."""

    def to_packed(self):
        """Transcode to exact direct bit streams without a dense count tensor.

        Stream widths are computed on the accelerator. The host scans only
        the small uint64 word-length index to allocate the final word buffer;
        no decoded count array crosses the CPU boundary. The second GPU pass
        decodes each stream directly into its final bit-packed storage.
        The original source remains valid until its caller releases it.
        """
        self._require_resident()
        widths = lengths = offsets = words = errors = None
        try:
            widths = _allocate_shared(
                self._device, self._metal, self._stream_count, "packed stream widths"
            )
            lengths = _allocate_shared(
                self._device,
                self._metal,
                self._stream_count * 8,
                "packed stream word lengths",
            )
            self._dispatch(
                "measure_packed",
                self._parameters(),
                self._stream_count,
                output=widths,
                requested=lengths,
            )
            offsets = _allocate_shared(
                self._device,
                self._metal,
                (self._stream_count + 1) * 8,
                "packed stream offsets",
            )
            word_offsets = np.frombuffer(_buffer_view(offsets), dtype=np.uint64)
            word_offsets[0] = 0
            np.cumsum(
                np.frombuffer(_buffer_view(lengths), dtype=np.uint64),
                dtype=np.uint64,
                out=word_offsets[1:],
            )
            word_count = int(word_offsets[-1])
            if word_count * 4 > int(self._device.maxBufferLength()):
                raise MemoryError(
                    "The exact packed word buffer exceeds this device's maxBufferLength."
                )
            words = _allocate_shared(
                self._device, self._metal, max(4, word_count * 4), "packed stream words"
            )
            _buffer_view(self._errors)[:] = b"\0" * 4
            command = self._queue.commandBuffer()
            self._encode(
                command,
                "write_packed",
                self._parameters(),
                self._stream_count,
                output=words,
                requested=widths,
                word_offsets=offsets,
            )
            _complete(command, "ANS to packed")
            self._check()
            errors = _allocate_shared(self._device, self._metal, 4, "packed errors")
            result = MPSPackedResidentCounts(self, [words, offsets, widths], errors)
            result.conversion_owned_buffer_peak_bytes = (
                self.resident_bytes + result.resident_bytes + int(lengths.length())
            )
            result.conversion_host_index_bytes = (
                self._stream_count * 8 + (self._stream_count + 1) * 8
            )
            words = offsets = widths = errors = None
            return result
        finally:
            for buffer in (widths, lengths, offsets, words, errors):
                _release(buffer)


class MPSPackedResidentCounts(_MPSResidentCounts):
    """Own independently retained word-aligned streams produced by to_packed.

    This private device layout is shared with CUDA: words uint32, word offsets
    uint64, one width byte per stream. Widths range from zero through sixteen.
    No source lifetime or ANS table is retained after successful conversion.
    """

    def __init__(self, source, buffers, errors):
        for name in (
            "shape",
            "dtype",
            "block_frames",
            "scale",
            "_scan_count",
            "_detector_count",
            "_block_count",
            "_stream_count",
            "_metal",
            "_device",
            "_pipelines",
            "_queue",
        ):
            setattr(self, name, getattr(source, name))
        self._buffers = buffers
        self._errors = errors
        self.is_released = False
        self.conversion_owned_buffer_peak_bytes = None
        self.conversion_host_index_bytes = None

    def _encode(
        self, command, name, parameters, width, height=1, output=None, requested=None
    ):
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipelines[f"packed_{name}"])
        for index, buffer in enumerate(self._buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBuffer_offset_atIndex_(self._errors, 0, 8)
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 9)
        encoder.setBuffer_offset_atIndex_(output, 0, 10)
        if requested is not None:
            encoder.setBuffer_offset_atIndex_(requested, 0, 11)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._metal.MTLSizeMake(width, height, 1),
            self._metal.MTLSizeMake(128, 1, 1),
        )
        encoder.endEncoding()
