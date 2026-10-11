"""Metal-backed PyTorch queries with parallel literal bit-lane access."""

from pathlib import Path

import numpy as np
import torch

from quantem.gpu.device.metal_runtime import (
    allocate_shared,
    complete_command,
    metal_module,
    metal_pipelines,
    metal_queue,
    release_buffer,
    tensor_buffer,
    upload_shared,
)
from quantem.gpu.resident.mps.counts import MPSStreamedCounts

_KERNELS = Path(__file__).parent / "kernels"
# The count codec's stream reader, without its encoder and count-product kernels.
_READER_SOURCE = (_KERNELS / "streamed_counts.msl").read_text().split("kernel void streamed_counts_encode")[0]
_FLOAT_SOURCE = (_KERNELS / "float_ans.msl").read_text()
_TOTAL_SOURCE = (_KERNELS / "pattern_total.msl").read_text()
_NAMES = ("float_ans_direct", "float_ans_entropy", "float_ans_selected_entropy", "float_ans_detector")


def pattern_total(windows, pixels: int) -> np.ndarray:
    """Sum float32 Torch MPS windows of ``(patterns, row, col)`` over their patterns: float64, one value per pixel.

    Metal has no float64, and a float32 running total is several units in the
    last place off after 16384 patterns. The kernel keeps each detector
    pixel's float32 sum with two orders of rounding residual
    (Kahan-Babuska-Klein); added in float64 on the host they give the float64
    sum of the measurements.
    """
    pipeline = metal_pipelines(_TOTAL_SOURCE, ("pattern_total",), fast_math=False)["pattern_total"]
    total = torch.zeros((pixels, 3), dtype=torch.float32, device="mps")
    for values in windows:
        values = values.contiguous()
        # Pending Torch work (a dark subtraction, a gather) must finish before the native queue reads.
        torch.mps.synchronize()
        command = metal_queue().commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(tensor_buffer(values), values.storage_offset() * 4, 0)
        encoder.setBuffer_offset_atIndex_(tensor_buffer(total), 0, 1)
        parameters = np.asarray([len(values), pixels], np.uint32).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            metal_module().MTLSizeMake(pixels, 1, 1), metal_module().MTLSizeMake(128, 1, 1)
        )
        encoder.endEncoding()
        complete_command(command, "Float pattern total")
        del values
    return total.cpu().numpy().astype(np.float64).sum(axis=1)


def _kernels(lanes):
    """Return the float bit-lane pipelines; the lane count is a compile-time constant of the source."""
    source = _READER_SOURCE + f"\n#define FLOAT_LANES {lanes}u\n" + _FLOAT_SOURCE
    pipelines = metal_pipelines(source, _NAMES, fast_math=False)
    return [pipelines[name] for name in _NAMES]


class MPSFloatLanes(MPSStreamedCounts):
    """Write the original bit lanes directly into caller-owned Metal storage."""

    def _encode_decode(
        self, command, chunk, local_first, count, output, output_offset_bytes=0
    ):
        lanes = int(np.prod(self.shape[2:]))
        direct, entropy = _kernels(lanes)[:2]
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(direct)
        for index, buffer in enumerate(chunk.buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBuffer_offset_atIndex_(output, output_offset_bytes, 3)
        parameters = np.asarray([local_first, count], np.uint32).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 4)
        self._dispatch_threads(encoder, count * lanes)
        encoder.endEncoding()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(entropy)
        for index, buffer in enumerate((*chunk.buffers, self._decoding)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBuffer_offset_atIndex_(output, output_offset_bytes, 4)
        encoder.setBuffer_offset_atIndex_(self._errors, 0, 5)
        parameters = np.asarray([chunk.scans, local_first, count], np.uint32).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
        self._dispatch_threads(encoder, lanes)
        encoder.endEncoding()

    def float_detector(self, mask, background=None):
        """Submit ordered selected decode/reduction pairs with one completion wait."""
        lanes = int(np.prod(self.shape[2:]))
        entropy, reduce = _kernels(lanes)[2:]
        output = torch.empty(self.shape[:2], dtype=torch.float32, device="mps")
        torch.mps.synchronize()
        target = tensor_buffer(output)
        dark = target if background is None else tensor_buffer(background)
        selected = scratch = None
        try:
            selected = upload_shared(mask.astype(np.uint8), "Float detector mask")
            scratch = allocate_shared(max(chunk.scans for chunk in self.chunks) * lanes * 2, "Bounded float entropy scratch")
            self._clear_errors()
            command = metal_queue().commandBuffer()
            for chunk in self.chunks:
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(entropy)
                for index, buffer in enumerate(
                    (*chunk.buffers, self._decoding, scratch, self._errors, selected)
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                parameters = np.uint32(chunk.scans).tobytes()
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 7)
                self._dispatch_threads(encoder, lanes)
                encoder.endEncoding()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(reduce)
                for index, buffer in enumerate(
                    (*chunk.buffers, scratch, selected, dark, target)
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                parameters = np.asarray(
                    [chunk.first, background is not None], np.uint32
                ).tobytes()
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 7)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    self._metal.MTLSizeMake(chunk.scans, 1, 1),
                    self._metal.MTLSizeMake(128, 1, 1),
                )
                encoder.endEncoding()
            complete_command(command, "Float ANS detector")
            self._check_errors()
            return output
        finally:
            release_buffer(scratch)
            release_buffer(selected)
