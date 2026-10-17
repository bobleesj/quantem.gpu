"""Float32 display products from preserved IEEE float64 measurement bits."""

from pathlib import Path

import numpy as np
import torch

from quantem.gpu.device.metal_runtime import (
    complete_command,
    metal_module,
    metal_pipelines,
    metal_queue,
    tensor_buffer,
)

_SOURCE = (Path(__file__).with_name("kernels") / "float64_display.msl").read_text()


def float32_view(values):
    """Round a bounded float64 Metal array to float32 for display on the GPU.

    The source buffer remains unchanged. This is explicitly product precision,
    never a storage conversion or a claim of float64 Metal arithmetic.
    """
    result = torch.empty(values.shape, dtype=torch.float32, device="mps")
    torch.mps.synchronize()
    pipeline = metal_pipelines(_SOURCE, ("float64_display",), fast_math=False)[
        "float64_display"
    ]
    command = metal_queue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(values._mtl, 0, 0)
    encoder.setBuffer_offset_atIndex_(tensor_buffer(result), 0, 1)
    count = np.uint32(result.numel()).tobytes()
    encoder.setBytes_length_atIndex_(count, len(count), 2)
    metal = metal_module()
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(result.numel(), 1, 1), metal.MTLSizeMake(128, 1, 1)
    )
    encoder.endEncoding()
    complete_command(command, "Float64 display conversion")
    return result
