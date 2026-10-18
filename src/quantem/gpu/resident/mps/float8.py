"""Float32 calculations from preserved FP8 bytes or ANS words on Metal."""

from pathlib import Path

import numpy as np
import torch

from quantem.gpu.device.metal_runtime import (
    buffer_view,
    complete_command,
    metal_module,
    metal_pipelines,
    metal_queue,
    release_buffer,
    tensor_buffer,
    upload_shared,
)
from quantem.gpu.formats.float8 import FORMAT_CODES
from quantem.gpu.resident.mps.arrays import MetalArray

SOURCE = (Path(__file__).with_name("kernels") / "float8.msl").read_text()


def float32_view(values, dtype):
    """Convert on GPU without asking Torch to calculate with FP8 tensors."""
    result = torch.empty(values.shape, dtype=torch.float32, device="mps")
    width = values.element_size() if torch.is_tensor(values) else values.dtype.itemsize
    parameters = np.asarray(
        [result.numel(), width, FORMAT_CODES[str(dtype).removeprefix("torch.")]],
        np.uint32,
    )
    invalid = MetalArray((1,), np.uint32)
    try:
        np.frombuffer(buffer_view(invalid._mtl), np.uint32, count=1)[0] = 0
        torch.mps.synchronize()
        pipeline = metal_pipelines(SOURCE, ("float8_decode",), fast_math=False)[
            "float8_decode"
        ]
        command = metal_queue().commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        if torch.is_tensor(values):
            offset = values.storage_offset() * values.element_size()
            encoder.setBuffer_offset_atIndex_(tensor_buffer(values), offset, 0)
        else:
            encoder.setBuffer_offset_atIndex_(values._mtl, 0, 0)
        encoder.setBuffer_offset_atIndex_(tensor_buffer(result), 0, 1)
        encoder.setBuffer_offset_atIndex_(invalid._mtl, 0, 2)
        encoder.setBytes_length_atIndex_(parameters.tobytes(), parameters.nbytes, 3)
        metal = metal_module()
        encoder.dispatchThreads_threadsPerThreadgroup_(
            metal.MTLSizeMake(result.numel(), 1, 1), metal.MTLSizeMake(256, 1, 1)
        )
        encoder.endEncoding()
        complete_command(command, "FP8 calculation conversion")
        if width == 2 and invalid.get()[0]:
            raise ValueError("Decoded FP8 word exceeds one byte; recopy the QEM file.")
    finally:
        invalid.release()
    return result


def decode_bytes(lanes, first: int, stop: int):
    """Decode and validate byte lanes in one Metal command buffer."""
    lanes.check_readable(first, stop)
    shape = (stop - first, *lanes.shape[2:])
    words = MetalArray(shape, np.uint16)
    result = None
    try:
        result = MetalArray(shape, np.uint8)
        command = metal_queue().commandBuffer()
        lanes._encode_scan_range_into(command, first, stop, words._mtl)
        pipeline = metal_pipelines(SOURCE, ("float8_bytes",), fast_math=False)[
            "float8_bytes"
        ]
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate((words._mtl, result._mtl, lanes._errors)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        parameters = np.uint32(result.nbytes).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 3)
        lanes._dispatch_threads(encoder, result.nbytes)
        encoder.endEncoding()
        complete_command(command, "FP8 byte decode")
        lanes._check_errors()
        return result
    except BaseException:
        if result is not None:
            result.release()
        raise
    finally:
        words.release()


def reduce_patterns(lanes, indices, dtype, valid, *, mean: bool):
    """Reduce FP8 scan selections in one command, retaining only one word window."""
    pixels = int(np.prod(lanes.shape[2:]))
    flags = np.zeros(int(np.prod(lanes.shape[:2])), np.uint8)
    flags[indices] = 1
    total = torch.zeros((pixels, 4), dtype=torch.float32, device="mps")
    result = torch.empty(lanes.shape[2:], dtype=torch.float32, device="mps")
    window = MetalArray((max(chunk.scans for chunk in lanes.chunks), pixels), np.uint16)
    selected = validity = None
    try:
        selected = upload_shared(flags, "FP8 selected scans")
        validity = upload_shared(valid.astype(np.uint8), "FP8 valid pixels")
        pipelines = metal_pipelines(
            SOURCE, ("float8_pattern_total", "float8_pattern_finish"), fast_math=False
        )
        torch.mps.synchronize()
        command = metal_queue().commandBuffer()
        for chunk in lanes.chunks:
            local = np.flatnonzero(flags[chunk.first : chunk.first + chunk.scans])
            if not local.size:
                continue
            first, stop = chunk.first + int(local[0]), chunk.first + int(local[-1]) + 1
            lanes._encode_scan_range_into(command, first, stop, window._mtl)
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(pipelines["float8_pattern_total"])
            for index, buffer in enumerate(
                (window._mtl, tensor_buffer(total), selected, validity, lanes._errors)
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)
            parameters = np.asarray(
                [stop - first, pixels, first, FORMAT_CODES[str(dtype)]], np.uint32
            ).tobytes()
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 5)
            lanes._dispatch_threads(encoder, pixels)
            encoder.endEncoding()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipelines["float8_pattern_finish"])
        encoder.setBuffer_offset_atIndex_(tensor_buffer(total), 0, 0)
        encoder.setBuffer_offset_atIndex_(tensor_buffer(result), 0, 1)
        parameters = np.asarray(
            [pixels, len(indices) if mean else 1, FORMAT_CODES[str(dtype)]], np.uint32
        ).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 2)
        lanes._dispatch_threads(encoder, pixels)
        encoder.endEncoding()
        complete_command(command, "FP8 pattern reduction")
        lanes._check_errors()
        return result
    finally:
        window.release()
        release_buffer(selected)
        release_buffer(validity)
