#include <metal_stdlib>
using namespace metal;

kernel void empad_dark_mean(device const float* input [[buffer(0)]],
                            device float2* accumulator [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            constant uint3& dimensions [[buffer(3)]],
                            uint pixel [[thread_position_in_grid]]) {
    float2 prior = dimensions.x == 0 ? float2(0) : accumulator[pixel];
    float sum = prior.x, correction = prior.y;
    for (uint frame = 0; frame < dimensions.y; ++frame) {
        float value = input[frame * 16384 + pixel] / float(dimensions.z);
        float adjusted = value - correction;
        float next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    accumulator[pixel] = float2(sum, correction);
    output[pixel] = sum;
}

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

// One SIMD group owns a complete row. Contiguous lanes read contiguous words,
// and each output word has exactly one writer (no atomics or dense expansion).
kernel void empad_describe_simd(device const uint* input [[buffer(0)]],
                               device uint4* descriptors [[buffer(1)]],
                               uint block [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    uint base = input[block * 128], changed = 0;
    for (uint i = lane; i < 128; i += 32) changed |= input[block * 128 + i] ^ base;
    changed = simd_or(changed);
    if (lane == 0) {
        uint shift = changed ? ctz(changed) : 0;
        descriptors[block] = uint4(base, changed ? 32 - clz(changed) - shift : 0, shift, 0);
    }
}

kernel void empad_pack_simd(device const uint* input [[buffer(0)]],
                          device const uint4* descriptors [[buffer(1)]],
                          device uint* packed [[buffer(2)]],
                          uint block [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    uint4 d = descriptors[block];
    if (!d.y) return;
    for (uint word = lane; word < d.y * 4; word += 32) {
        uint bit = word * 32, pixel = bit / d.y, skip = bit % d.y;
        uint value = ((input[block * 128 + pixel] ^ d.x) >> d.z) >> skip;
        uint filled = d.y - skip;
        while (filled < 32 && ++pixel < 128) {
            value |= ((input[block * 128 + pixel] ^ d.x) >> d.z) << filled;
            filled += d.y;
        }
        packed[d.w + word] = value;
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

inline float empad_value(device const uint* packed, device const uint4* descriptors,
                         uint index, device const float* background, uint corrected) {
    float value = as_type<float>(empad_word(packed, descriptors, index));
    return corrected ? value - background[index % 16384] : value;
}

kernel void empad_diffraction(device const uint* packed [[buffer(0)]],
                             device const uint4* descriptors [[buffer(1)]],
                             device uint* output [[buffer(2)]],
                             constant uint& frame [[buffer(3)]],
                             device const float* background [[buffer(8)]],
                             constant uint& corrected [[buffer(9)]],
                             uint pixel [[thread_position_in_grid]]) {
    uint word = empad_word(packed, descriptors, frame * 16384 + pixel);
    output[pixel] = corrected ? as_type<uint>(as_type<float>(word) - background[pixel]) : word;
}

kernel void empad_virtual_image_serial(device const uint* packed [[buffer(0)]],
                               device const uint4* descriptors [[buffer(1)]],
    device const float* background [[buffer(8)]],
    constant uint& corrected [[buffer(9)]],
                               device const uchar* mask [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& offset [[buffer(4)]],
                               uint frame [[thread_position_in_grid]]) {
    float sum = 0, correction = 0;
    for (uint pixel = 0; pixel < 16384; ++pixel) {
        // Unselected NaNs do not contaminate the selected detector aperture.
        if (mask[pixel]) {
            float value = empad_value(packed, descriptors, frame * 16384 + pixel, background, corrected);
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

inline void empad_add(float value, thread float& sum, thread float& correction) {
    if (isfinite(value) && isfinite(sum)) {
        float adjusted = value - correction;
        float next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    } else { sum += value; correction = 0; }
}

// Preserve the low-order part through the reduction tree. EMPAD measurements
// can be signed: dropping a lane's compensation before combining lanes loses
// absolute accuracy when a nearly empty aperture cancels to a small value.
inline void empad_accumulate(float value, thread float& sum, thread float& residual) {
    float next = sum + value;
    if (isfinite(value) && isfinite(sum) && isfinite(next)) {
        residual += abs(sum) >= abs(value) ? (sum - next) + value : (value - next) + sum;
    } else { residual = 0; }
    sum = next;
}

// Four SIMD groups cooperate on a scan position. Neighboring lanes read
// neighboring detector pixels, rather than each lane walking a separate DP.
// Compensation is retained within lanes and at both reduction levels.
// A one-group launch remains the controlled benchmark topology. Both use only
// 32 bytes of transient threadgroup scratch and the same packed resident.
kernel void empad_virtual_image(device const uint* packed [[buffer(0)]],
                               device const uint4* descriptors [[buffer(1)]],
    device const float* background [[buffer(8)]],
    constant uint& corrected [[buffer(9)]],
                               device const uchar* mask [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& offset [[buffer(4)]],
                               uint frame [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort group [[simdgroup_index_in_threadgroup]],
                               uint localIndex [[thread_index_in_threadgroup]],
                               uint width [[threads_per_threadgroup]]) {
    threadgroup float2 partials[4];
    float sum = 0, correction = 0;
    for (uint pixel = localIndex; pixel < 16384; pixel += width) {
        if (mask[pixel]) {
            empad_accumulate(empad_value(packed, descriptors, frame * 16384 + pixel, background, corrected), sum, correction);
        }
    }
    float total = 0, compensation = 0;
    for (ushort i = 0; i < 32; ++i) {
        float partial = simd_broadcast(sum, i);
        float residual = simd_broadcast(correction, i);
        if (lane == 0) {
            empad_accumulate(partial, total, compensation);
            empad_accumulate(residual, total, compensation);
        }
    }
    if (lane == 0) partials[group] = float2(total, compensation);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localIndex == 0) {
        total = 0; compensation = 0;
        for (uint i = 0; i < width / 32; ++i) {
            empad_accumulate(partials[i].x, total, compensation);
            empad_accumulate(partials[i].y, total, compensation);
        }
        output[offset + frame] = total + compensation;
    }
}

// Incremental integration retains a high/low sum per scan position. It is an
// image-sized cache, not a second 4D representation. Nonfinite previous sums
// are recomputed from the complete current mask so removing a NaN recovers.
kernel void empad_virtual_image_changes(device const uint* packed [[buffer(0)]],
                                       device const uint4* descriptors [[buffer(1)]],
    device const float* background [[buffer(8)]],
    constant uint& corrected [[buffer(9)]],
                                       device const int2* entries [[buffer(2)]],
                                       device float* output [[buffer(3)]],
                                       constant uint& offset [[buffer(4)]],
                                       device float2* accumulated [[buffer(5)]],
                                       constant uint2& parameters [[buffer(6)]],
                                       device const uchar* mask [[buffer(7)]],
                                       uint frame [[threadgroup_position_in_grid]],
                                       ushort lane [[thread_index_in_simdgroup]],
                                       ushort group [[simdgroup_index_in_threadgroup]],
                                       uint local [[thread_index_in_threadgroup]],
                                       uint width [[threads_per_threadgroup]]) {
    threadgroup float2 partials[4];
    bool reset = parameters.y != 0;
    float2 previous = reset ? float2(0) : accumulated[offset + frame];
    if (!reset && parameters.x == 0) {
        if (local == 0) output[offset + frame] = previous.x + previous.y;
        return;
    }
    bool recover = !reset && (!isfinite(previous.x) || !isfinite(previous.y));
    float sum = 0, residual = 0;
    uint count = recover ? 16384 : parameters.x;
    for (uint entry = local; entry < count; entry += width) {
        int2 item = recover ? int2(entry, mask[entry] != 0) : entries[entry];
        if (item.y) {
            float value = empad_value(packed, descriptors, frame * 16384 + uint(item.x), background, corrected);
            empad_accumulate(value * float(item.y), sum, residual);
        }
    }
    float high = 0, low = 0;
    for (ushort i = 0; i < 32; ++i) {
        float a = simd_broadcast(sum, i), b = simd_broadcast(residual, i);
        if (lane == 0) { empad_accumulate(a, high, low); empad_accumulate(b, high, low); }
    }
    if (lane == 0) partials[group] = float2(high, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local == 0) {
        high = 0; low = 0;
        if (!reset && !recover) {
            empad_accumulate(previous.x, high, low);
            empad_accumulate(previous.y, high, low);
        }
        for (uint i = 0; i < width / 32; ++i) {
            empad_accumulate(partials[i].x, high, low);
            empad_accumulate(partials[i].y, high, low);
        }
        accumulated[offset + frame] = float2(high, low);
        output[offset + frame] = high + low;
    }
}

// Full-detector intensity-weighted coordinates. Normalize the weights first
// to avoid overflowing a row moment for large finite float measurements.
kernel void empad_center_of_mass(device const uint* packed [[buffer(0)]],
                                device const uint4* descriptors [[buffer(1)]],
    device const float* background [[buffer(8)]],
    constant uint& corrected [[buffer(9)]],
                                device float* rows [[buffer(2)]],
                                device float* columns [[buffer(3)]],
                                constant uint& offset [[buffer(4)]],
                                uint frame [[thread_position_in_grid]]) {
    float magnitude = 0;
    bool valid = true;
    for (uint pixel = 0; pixel < 16384; ++pixel) {
        float value = empad_value(packed, descriptors, frame * 16384 + pixel, background, corrected);
        valid = valid && isfinite(value);
        magnitude = max(magnitude, abs(value));
    }
    float total = 0, row = 0, column = 0, ct = 0, cr = 0, cc = 0;
    if (valid && magnitude > 0) {
        for (uint pixel = 0; pixel < 16384; ++pixel) {
            float value = empad_value(packed, descriptors, frame * 16384 + pixel, background, corrected) / magnitude;
            empad_add(value, total, ct);
            empad_add(value * float(pixel / 128), row, cr);
            empad_add(value * float(pixel % 128), column, cc);
        }
    }
    rows[offset + frame] = valid && total != 0 ? row / total : NAN;
    columns[offset + frame] = valid && total != 0 ? column / total : NAN;
}

kernel void empad_center_of_mass_simd(device const uint* packed [[buffer(0)]],
                                     device const uint4* descriptors [[buffer(1)]],
    device const float* background [[buffer(8)]],
    constant uint& corrected [[buffer(9)]],
                                     device float* rows [[buffer(2)]],
                                     device float* columns [[buffer(3)]],
                                     constant uint& offset [[buffer(4)]],
                                     uint frame [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort group [[simdgroup_index_in_threadgroup]],
                                     uint local [[thread_index_in_threadgroup]]) {
    threadgroup float maxima[4];
    threadgroup uint validity[4];
    threadgroup float2 partials[12];
    float magnitude = 0;
    bool valid = true;
    for (uint pixel = local; pixel < 16384; pixel += 128) {
        float value = empad_value(packed, descriptors, frame * 16384 + pixel, background, corrected);
        valid = valid && isfinite(value);
        magnitude = max(magnitude, abs(value));
    }
    magnitude = simd_max(magnitude);
    valid = simd_all(valid);
    if (lane == 0) { maxima[group] = magnitude; validity[group] = valid; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    magnitude = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
    valid = validity[0] && validity[1] && validity[2] && validity[3];
    float sum[3] = {0, 0, 0}, residual[3] = {0, 0, 0};
    if (valid && magnitude > 0) {
        for (uint pixel = local; pixel < 16384; pixel += 128) {
            float value = empad_value(packed, descriptors, frame * 16384 + pixel, background, corrected) / magnitude;
            empad_accumulate(value, sum[0], residual[0]);
            empad_accumulate(value * float(pixel / 128), sum[1], residual[1]);
            empad_accumulate(value * float(pixel % 128), sum[2], residual[2]);
        }
    }
    for (uint component = 0; component < 3; ++component) {
        float total = 0, correction = 0;
        for (ushort i = 0; i < 32; ++i) {
            float high = simd_broadcast(sum[component], i);
            float low = simd_broadcast(residual[component], i);
            if (lane == 0) {
                empad_accumulate(high, total, correction);
                empad_accumulate(low, total, correction);
            }
        }
        if (lane == 0) partials[component * 4 + group] = float2(total, correction);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local == 0) {
        float3 result;
        for (uint component = 0; component < 3; ++component) {
            float total = 0, correction = 0;
            for (uint i = 0; i < 4; ++i) {
                empad_accumulate(partials[component * 4 + i].x, total, correction);
                empad_accumulate(partials[component * 4 + i].y, total, correction);
            }
            result[component] = total + correction;
        }
        rows[offset + frame] = valid && result.x != 0 ? result.y / result.x : NAN;
        columns[offset + frame] = valid && result.x != 0 ? result.z / result.x : NAN;
    }
}

kernel void empad_mean_diffraction(device const uint* packed [[buffer(0)]],
                                   device const uint4* descriptors [[buffer(1)]],
    device const float* background [[buffer(8)]],
    constant uint& corrected [[buffer(9)]],
                                   device float2* accumulator [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant uint3& dimensions [[buffer(4)]],
                                   uint pixel [[thread_position_in_grid]]) {
    float2 previous = dimensions.x == 0 ? float2(0) : accumulator[pixel];
    float sum = previous.x, correction = previous.y;
    for (uint frame = 0; frame < dimensions.y; ++frame) {
        float value = empad_value(packed, descriptors, frame * 16384 + pixel, background, corrected);
        empad_add(value / float(dimensions.z), sum, correction);
    }
    accumulator[pixel] = float2(sum, correction);
    output[pixel] = sum;
}
