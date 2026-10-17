"""Correctly rounded display means from exact 64-bit Metal totals."""

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

_SOURCE = (Path(__file__).with_name("kernels") / "integer_mean.msl").read_text()


def integer_mean(total, count):
    """Divide signed/unsigned totals by a scan count on GPU and round once."""
    if not 0 < count < 1 << 31 or total.dtype not in (torch.int64, torch.uint64):
        raise ValueError("Provide 64-bit totals and a scan count below 2^31.")
    result = torch.empty(total.shape, dtype=torch.float32, device="mps")
    torch.mps.synchronize()
    pipeline = metal_pipelines(_SOURCE, ("integer_mean",), fast_math=False)[
        "integer_mean"
    ]
    command = metal_queue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(tensor_buffer(total), 0, 0)
    encoder.setBuffer_offset_atIndex_(tensor_buffer(result), 0, 1)
    parameters = np.asarray(
        [total.numel(), count, total.dtype == torch.int64], np.uint64
    ).tobytes()
    encoder.setBytes_length_atIndex_(parameters, len(parameters), 2)
    metal = metal_module()
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(total.numel(), 1, 1), metal.MTLSizeMake(128, 1, 1)
    )
    encoder.endEncoding()
    complete_command(command, "Integer mean")
    return result
