"""Transpose bounded blocks of diffraction frames on the Apple GPU (``io.load(..., detector_transpose=True)``)."""

import math

import numpy as np

from quantem.gpu.device.metal_runtime import complete_command, metal_module, metal_pipelines, metal_queue
from quantem.gpu.resident.mps.arrays import MetalArray

# One 32 x 32 tile of a frame per threadgroup, staged through threadgroup memory so that both the read of the frame
# and the write of its transpose are rows of consecutive elements. Elements move as unsigned words of their size,
# so every value keeps its bits.
_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline void transpose_tile(device const T* raw, device T* out, uint rows, uint cols, threadgroup T* tile,
                           uint3 group, uint3 local) {
    const ulong frame = ulong(group.z) * rows * cols;
    uint col = group.x * 32 + local.x, row = group.y * 32 + local.y;
    for (uint step = 0; step < 32; step += 8)
        if (row + step < rows && col < cols) tile[(local.y + step) * 33 + local.x] = raw[frame + ulong(row + step) * cols + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    col = group.y * 32 + local.x; row = group.x * 32 + local.y;
    for (uint step = 0; step < 32; step += 8)
        if (row + step < cols && col < rows) out[frame + ulong(row + step) * rows + col] = tile[local.x * 33 + local.y + step];
}

#define TRANSPOSE(NAME, T) \
kernel void NAME(device const T* raw [[buffer(0)]], device T* out [[buffer(1)]], \
                 constant uint2& shape [[buffer(2)]], uint3 group [[threadgroup_position_in_grid]], \
                 uint3 local [[thread_position_in_threadgroup]]) { \
    threadgroup T tile[32 * 33]; \
    transpose_tile(raw, out, shape.x, shape.y, tile, group, local); \
}
TRANSPOSE(transpose_1, uchar)
TRANSPOSE(transpose_2, ushort)
TRANSPOSE(transpose_4, uint)
TRANSPOSE(transpose_8, ulong)
"""
_NAMES = ("transpose_1", "transpose_2", "transpose_4", "transpose_8")


def transposed_frames(raw) -> MetalArray:
    """A new Metal array of ``raw``'s frames with each frame's rows and columns swapped: ``out[f, i, j] = raw[f, j, i]``.

    ``raw`` is a Metal array or a ``SharedArray`` of shape (frames, detector rows, detector cols); the caller releases
    both. The swap runs on the GPU between the flagged-pixel correction and the encoder, as on CUDA
    (``io.arrays.transposed_frames``): no host copy.
    """
    frames, rows, cols = (int(side) for side in raw.shape)
    dtype = np.dtype(raw.dtype)
    out = MetalArray((frames, cols, rows), dtype)
    if not frames * rows * cols:
        return out
    metal = metal_module()
    command = metal_queue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(metal_pipelines(_SOURCE, _NAMES)[f"transpose_{dtype.itemsize}"])
    encoder.setBuffer_offset_atIndex_(raw._mtl, 0, 0)
    encoder.setBuffer_offset_atIndex_(out._mtl, 0, 1)
    shape = np.asarray([rows, cols], dtype=np.uint32).tobytes()
    encoder.setBytes_length_atIndex_(shape, len(shape), 2)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        metal.MTLSizeMake(math.ceil(cols / 32), math.ceil(rows / 32), frames), metal.MTLSizeMake(32, 8, 1)
    )
    encoder.endEncoding()
    complete_command(command, "Detector transpose")
    return out
