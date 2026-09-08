#include <metal_stdlib>
using namespace metal;

// A block is one 128-pixel detector row. Common leading/trailing XOR bits
// disappear from storage; no floating-point arithmetic occurs during packing.
kernel void empad_describe(device const uint* input [[buffer(0)]],
                           device uint4* descriptors [[buffer(1)]],
                           uint block [[thread_position_in_grid]]) {
    uint base = input[block * 128];
    uint changed = 0;
    for (uint i = 0; i < 128; ++i) changed |= input[block * 128 + i] ^ base;
    uint shift = changed ? ctz(changed) : 0;
    uint width = changed ? 32 - clz(changed) - shift : 0;
    descriptors[block] = uint4(base, width, shift, 0);
}

kernel void empad_pack(device const uint* input [[buffer(0)]],
                      device const uint4* descriptors [[buffer(1)]],
                      device uint* packed [[buffer(2)]],
                      uint block [[thread_position_in_grid]]) {
    uint4 d = descriptors[block];
    if (!d.y) return;
    for (uint word = 0; word < d.y * 4; ++word) packed[d.w + word] = 0;
    for (uint i = 0; i < 128; ++i) {
        uint delta = (input[block * 128 + i] ^ d.x) >> d.z;
        uint bit = i * d.y, word = d.w + bit / 32, shift = bit % 32;
        packed[word] |= delta << shift;
        if (shift + d.y > 32) packed[word + 1] |= delta >> (32 - shift);
    }
}

inline uint empad_word(device const uint* packed, device const uint4* descriptors,
                       uint index) {
    uint4 d = descriptors[index / 128];
    if (!d.y) return d.x;
    uint bit = (index % 128) * d.y, shift = bit % 32, word = d.w + bit / 32;
    uint value = packed[word] >> shift;
    if (shift + d.y > 32) value |= packed[word + 1] << (32 - shift);
    if (d.y < 32) value &= (1u << d.y) - 1;
    return d.x ^ (value << d.z);
}

kernel void empad_diffraction(device const uint* packed [[buffer(0)]],
                             device const uint4* descriptors [[buffer(1)]],
                             device uint* output [[buffer(2)]],
                             constant uint& frame [[buffer(3)]],
                             uint pixel [[thread_position_in_grid]]) {
    output[pixel] = empad_word(packed, descriptors, frame * 16384 + pixel);
}

kernel void empad_virtual_image(device const uint* packed [[buffer(0)]],
                               device const uint4* descriptors [[buffer(1)]],
                               device const uchar* mask [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& offset [[buffer(4)]],
                               uint frame [[thread_position_in_grid]]) {
    float sum = 0, correction = 0;
    for (uint pixel = 0; pixel < 16384; ++pixel) {
        // Unselected NaNs do not contaminate the selected detector aperture.
        if (mask[pixel]) {
            float value = as_type<float>(empad_word(packed, descriptors, frame * 16384 + pixel));
            if (isfinite(value) && isfinite(sum)) {
                float adjusted = value - correction;
                float next = sum + adjusted;
                correction = (next - sum) - adjusted;
                sum = next;
            } else {
                // Keep IEEE non-finite semantics; compensation is not defined
                // for infinity minus infinity.
                sum += value;
                correction = 0;
            }
        }
    }
    output[offset + frame] = sum;
}
