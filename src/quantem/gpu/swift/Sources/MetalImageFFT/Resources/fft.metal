#include <metal_stdlib>
using namespace metal;

kernel void image_fft_pack_u8(
    device const uchar *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    destination[index] = float(source[index]);
}

kernel void image_fft_pack_u16(
    device const ushort *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    destination[index] = float(source[index]);
}

kernel void image_fft_pack_u32(
    device const uint *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    destination[index] = float(source[index]);
}

kernel void image_fft_pack_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    destination[index] = isfinite(source[index]) ? source[index] : 0.0f;
}
