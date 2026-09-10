#include <metal_stdlib>
using namespace metal;

struct OriginalPackingShape { uint scans, pixels, columns, sourceBytes; };
inline uint original_count(const device uchar *source, uint scan, uint pixel,
                           constant OriginalPackingShape &s) {
    ulong index = ulong(scan) * s.pixels + pixel;
    return s.sourceBytes == 1 ? uint(source[index])
        : (s.sourceBytes == 4 ? ((const device uint *)source)[index]
                             : uint(((const device ushort *)source)[index]));
}

// Independent checkpoint-sized workers preserve the same exact packed bytes.
// Requires the same validated nonoverlapping header partition as the baseline.
kernel void original_packing_values_verified_checkpoints(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    for (uint tile = checkpoint * 32; tile < min(tiles, (checkpoint + 1) * 32); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        uint originals[32];
        uint packed = 0, occupied = 0, outputWord = 0;
        for (uint sample = 0; sample < 32; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            originals[sample] = value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32;
                    packed = occupied ? value >> (bits - occupied) : 0;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0; sample < 32; ++sample) {
            uint bit = sample * bits, shift = bit % 32;
            ulong value = bits ? ulong(payload[offset + bit / 32]) >> shift : 0;
            if (shift + bits > 32) value |= ulong(payload[offset + bit / 32 + 1]) << (32 - shift);
            if ((value & mask) != originals[sample])
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        }
        offset += bits;
    }
}

inline uint originalEvenBits(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1u)) & 0x33333333u;
    value = (value | (value >> 2u)) & 0x0f0f0f0fu;
    value = (value | (value >> 4u)) & 0x00ff00ffu;
    return (value | (value >> 8u)) & 0xffffu;
}

inline uint originalThirdBits(uint value) {
    uint high = (value >> 20u) & 0x400u;
    value &= 0x09249249u;
    value = (value ^ (value >> 2u)) & 0x030c30c3u;
    value = (value ^ (value >> 4u)) & 0x0300f00fu;
    value = (value ^ (value >> 8u)) & 0x030000ffu;
    value = (value ^ (value >> 16u)) & 0x000003ffu;
    return value | high;
}

inline uint originalFourthBits(uint value) {
    value &= 0x11111111u;
    value = (value | (value >> 3u)) & 0x03030303u;
    value = (value | (value >> 6u)) & 0x000f000fu;
    return (value | (value >> 12u)) & 0xffu;
}

// First-load path: verify the bounded decoded source against interleaved
// words, then transpose the verified cell in registers and verify its stores.
// Widths zero and one need no conversion; common two- to four-bit cases
// use exact register-level transposition.
inline uint originalPackCountsAsPlanes(
    const device uchar *source, const device uint *headers,
    volatile device uint *payload, device atomic_uint *errors,
    constant OriginalPackingShape &s, uint index) {
    if (!s.pixels || !s.scans || s.scans % 32u || (s.sourceBytes != 1u && s.sourceBytes != 2u && s.sourceBytes != 4u)) {
        atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return 0u;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return 0u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    uint sum = 0u;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_count(source, tile * 32u + sample, pixel, s);
            originals[sample] = value;
            sum += value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint bit = sample * bits, shift = bit % 32u;
            ulong value = bits ? ulong(payload[offset + bit / 32u]) >> shift : 0u;
            if (shift + bits > 32u) value |= ulong(payload[offset + bit / 32u + 1u]) << (32u - shift);
            if ((value & mask) != originals[sample])
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        }
        if (bits > 1u) {
            uint4 words = uint4(0u);
            if (bits <= 4u) {
                for (uint word = 0u; word < bits; ++word) words[word] = payload[offset + word];
            }
            for (uint plane = 0u; plane < bits; ++plane) {
                uint value = 0u;
                if (bits == 2u) {
                    value = originalEvenBits(words[0] >> plane)
                        | (originalEvenBits(words[1] >> plane) << 16u);
                } else if (bits == 3u) {
                    uint secondShift = (plane + 1u) % 3u, thirdShift = (plane + 2u) % 3u;
                    uint firstCount = (34u - plane) / 3u, secondCount = (34u - secondShift) / 3u;
                    value = originalThirdBits(words[0] >> plane)
                        | (originalThirdBits(words[1] >> secondShift) << firstCount)
                        | (originalThirdBits(words[2] >> thirdShift) << (firstCount + secondCount));
                } else if (bits == 4u) {
                    for (uint word = 0u; word < 4u; ++word)
                        value |= originalFourthBits(words[word] >> plane) << (8u * word);
                } else {
                    for (uint sample = 0u; sample < 32u; ++sample)
                        value |= ((originals[sample] >> plane) & 1u) << sample;
                }
                payload[offset + plane] = value;
                if (payload[offset + plane] != value)
                    atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
            }
        }
        offset += bits;
    }
    return sum;
}

kernel void original_packing_values_planes_checkpoints(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    originalPackCountsAsPlanes(source, headers, payload, errors, s, index);
}

kernel void original_packing_values_planes_checkpoints_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (!s.pixels || index >= s.pixels * ((s.scans / 32u + 31u) / 32u)) return;
    partialSums[index] = originalPackCountsAsPlanes(source, headers, payload, errors, s, index);
}

// Input is complete checked LZ4-expanded uint16
// bitshuffle blocks, not dense counts. All frames and all 16 planes remain.
// Count verification below compares actual packed payload with reconstructed
// counts, including values in zero-width tiles.
inline uint original_bitshuffle_count(
    const device uchar *scratch, uint scan, uint pixel,
    constant OriginalPackingShape &s
) {
    const device uint *planes = (const device uint *)(scratch
        + ulong(scan) * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
    uint group = (pixel % 4096u) / 32u, lane = pixel % 32u;
    uint value = 0u;
    #pragma unroll
    for (uint bit = 0u; bit < 16u; ++bit) {
        value |= ((planes[bit * 128u + group] >> lane) & 1u) << bit;
    }
    return value;
}

// Cooperative gather for one aligned group of 32 detector pixels. Every lane
// participates, including lanes whose eventual packed tile has zero width.
inline uint original_bitshuffle_transpose_count(
    const device uchar *scratch, uint scan, uint pixel,
    constant OriginalPackingShape &s, uint lane
) {
    const device uint *planes = (const device uint *)(scratch
        + ulong(scan) * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
    uint group = (pixel % 4096u) / 32u;
    uint value = lane < 16u ? planes[lane * 128u + group] : 0u;
    #pragma unroll
    for (uint shift = 1u; shift <= 16u; shift *= 2u) {
        uint mask = 0xffffffffu / ((1u << shift) + 1u);
        uint other = simd_shuffle_xor(value, shift);
        value = (lane & shift)
            ? (value & ~mask) | ((other & ~mask) >> shift)
            : (value & mask) | ((other & mask) << shift);
    }
    return value;
}

// Separately selected candidate: the independent scalar-gather entry point
// below remains unchanged. Requires 32-lane SIMD groups and aligned pixels.
kernel void original_packing_bitshuffle_transpose_verified_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane
        || groupThreads % 32u != 0u;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints) return;
    // A different writer can fail as this group starts. Vote before returning
    // so that no lane leaves peers executing a transpose with missing inputs.
    if (simd_any(atomic_load_explicit(errors, memory_order_relaxed) != 0u)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    uint sum = 0u, maximum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_bitshuffle_transpose_count(source, tile * 32u + sample, pixel, s, lane);
            originals[sample] = value;
            sum += value;
            maximum = max(maximum, value);
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint bit = sample * bits, shift = bit % 32u;
            ulong value = bits ? ulong(payload[offset + bit / 32u]) >> shift : 0u;
            if (shift + bits > 32u) value |= ulong(payload[offset + bit / 32u + 1u]) << (32u - shift);
            if ((value & mask) != originals[sample]) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
        }
        // Never leave this loop per lane: even a failed lane participates in
        // every later transpose. No summaries are consumed if any count fails.
        offset += bits;
    }
    if (valid) {
        partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
        partialMaximums[ulong(checkpoint) * s.pixels + pixel] = maximum;
    }
}

// Transpose source bit planes across 32 scan positions directly into the
// resident cell. Every source bit is checked, including absent high planes;
// this proves all counts without constructing 32 scalar counts per lane.
inline void originalPackingBitshufflePlanes(
    const device uchar *source, const device uint *headers,
    volatile device uint *payload, device atomic_uint *errors,
    constant OriginalPackingShape &s, device uint *partialSums,
    device uint *partialMaximums, uint index, uint lane, uint simdWidth,
    uint groupThreads) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane
        || groupThreads % 32u != 0u;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints) return;
    if (simd_any(atomic_load_explicit(errors, memory_order_relaxed) != 0u)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    uint sum = 0u, maximum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint candidates = 0xffffffffu, tileMaximum = 0u;
        for (int plane = 15; plane >= 0; --plane) {
            ulong scan = ulong(tile) * 32ul + lane;
            const device uint *input = (const device uint *)(source
                + scan * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
            uint value = input[uint(plane) * 128u + (pixel % 4096u) / 32u];
            if (simd_any(value != 0u)) {
                #pragma unroll
                for (uint shift = 1u; shift <= 16u; shift *= 2u) {
                    uint mask = 0xffffffffu / ((1u << shift) + 1u);
                    uint other = simd_shuffle_xor(value, shift);
                    value = (lane & shift)
                        ? (value & ~mask) | ((other & ~mask) >> shift)
                        : (value & mask) | ((other & mask) << shift);
                }
            }
            uint restored = 0u;
            if (uint(plane) < bits) {
                payload[offset + uint(plane)] = value;
                restored = payload[offset + uint(plane)];
            }
            if (restored != value) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
            sum += popcount(value) << uint(plane);
            uint nextCandidates = candidates & value;
            if (nextCandidates != 0u) {
                tileMaximum |= 1u << uint(plane);
                candidates = nextCandidates;
            }
        }
        maximum = max(maximum, tileMaximum);
        offset += bits;
    }
    if (valid) {
        partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
        partialMaximums[ulong(checkpoint) * s.pixels + pixel] = maximum;
    }
}

kernel void original_packing_bitshuffle_planes_verified_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlanes(source, headers, payload, errors, s,
        partialSums, partialMaximums, index, lane, simdWidth, groupThreads);
}


// Four neighboring source words per lane improve transaction utilization.
// Each component remains an independent exact detector column. No temporary
// count volume or auxiliary resident representation is constructed.
template<typename Vector, uint columns, bool hasZeroTail = false>
inline void originalPackingBitshufflePlaneVectors(
    const device uchar *source, const device uint *headers,
    volatile device uint *payload, device atomic_uint *errors,
    constant OriginalPackingShape &s, device uint *partialSums,
    device uint *partialMaximums, uint index,
    uint lane, uint simdWidth,
    uint groupThreads, const device uint *zeroTails = nullptr) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane
        || groupThreads % 32u != 0u;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint checkpoint = index / (s.pixels / columns);
    if (checkpoint >= checkpoints) return;
    if (simd_any(atomic_load_explicit(errors, memory_order_relaxed) != 0u)) return;
    uint pixelGroup = ((index % (s.pixels / columns)) / 32u) * (32u * columns);
    Vector pixels;
    #pragma unroll
    for (uint part = 0u; part < columns; ++part) pixels[part] = pixelGroup + lane + part * 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    Vector offset, sum = Vector(0u), maximum = Vector(0u);
    #pragma unroll
    for (uint part = 0u; part < columns; ++part) {
        offset[part] = headers[pixels[part] * stride];
        if (checkpoint) offset[part] += headers[pixels[part] * stride + checkpoint];
    }
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        Vector bits;
        #pragma unroll
        for (uint part = 0u; part < columns; ++part) {
            uint width = (headers[pixels[part] * stride + checkpoints + tile / 8u]
                >> ((tile % 8u) * 4u)) & 15u;
            bits[part] = width == 15u ? 16u : width;
        }
        Vector candidates = Vector(0xffffffffu), tileMaximum = Vector(0u);
        uint zeroTail = 8192u;
        if (hasZeroTail) zeroTail = zeroTails[(tile * 32u + lane) * (s.pixels / 4096u) + pixelGroup / 4096u];
        for (int plane = 15; plane >= 0; --plane) {
            ulong scan = ulong(tile) * 32ul + lane;
            const device Vector *input = (const device Vector *)(source
                + scan * s.pixels * 2ul + ulong(pixelGroup / 4096u) * 8192ul);
            uint sourceByte = uint(plane) * 512u + (pixelGroup % 4096u) / 8u;
            Vector value = sourceByte < zeroTail
                ? input[sourceByte / (4u * columns)] : Vector(0u);
            if (simd_any(any(value != Vector(0u)))) {
                #pragma unroll
                for (uint shift = 1u; shift <= 16u; shift *= 2u) {
                    uint mask = 0xffffffffu / ((1u << shift) + 1u);
                    Vector other = simd_shuffle_xor(value, shift);
                    value = (lane & shift)
                        ? (value & ~mask) | ((other & ~mask) >> shift)
                        : (value & mask) | ((other & mask) << shift);
                }
            }
            #pragma unroll
            for (uint part = 0u; part < columns; ++part) {
                if (uint(plane) < bits[part]) payload[offset[part] + uint(plane)] = value[part];
            }
            #pragma unroll
            for (uint part = 0u; part < columns; ++part) {
                uint restored = uint(plane) < bits[part]
                    ? payload[offset[part] + uint(plane)] : 0u;
                if (restored != value[part]) {
                    atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                    valid = false;
                }
            }
            sum += popcount(value) << uint(plane);
            Vector nextCandidates = candidates & value;
            auto hasCandidate = nextCandidates != Vector(0u);
            tileMaximum |= select(Vector(0u), Vector(1u << uint(plane)), hasCandidate);
            candidates = select(candidates, nextCandidates, hasCandidate);
        }
        maximum = max(maximum, tileMaximum);
        offset += bits;
    }
    if (valid) {
        #pragma unroll
        for (uint part = 0u; part < columns; ++part) {
            partialSums[ulong(checkpoint) * s.pixels + pixels[part]] = sum[part];
            partialMaximums[ulong(checkpoint) * s.pixels + pixels[part]] = maximum[part];
        }
    }
}


kernel void original_packing_bitshuffle_planes_vector4_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u>(source, headers, payload, errors, s,
        partialSums, partialMaximums, index, lane, simdWidth, groupThreads);
}

kernel void original_packing_bitshuffle_planes_zero_tail_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], const device uint *zeroTails [[buffer(7)]],
    uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]], uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, true>(source, headers, payload, errors, s,
        partialSums, partialMaximums, index, lane, simdWidth, groupThreads, zeroTails);
}

kernel void original_packing_bitshuffle_verified_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (!s.pixels || !s.scans || s.scans % 32u || (s.sourceBytes != 2u || s.pixels % 4096u)) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    // At most 1024 full uint16 counts, bounded by 67,107,840.
    uint sum = 0u, maximum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_bitshuffle_count(source, tile * 32u + sample, pixel, s);
            originals[sample] = value;
            sum += value;
            maximum = max(maximum, value);
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint bit = sample * bits, shift = bit % 32u;
            ulong value = bits ? ulong(payload[offset + bit / 32u]) >> shift : 0u;
            if (shift + bits > 32u) value |= ulong(payload[offset + bit / 32u + 1u]) << (32u - shift);
            if ((value & mask) != originals[sample]) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
        }
        offset += bits;
    }
    // A different worker may fail concurrently. The following reduction must
    // run in a later encoder and observe the completed global error counter.
    if (valid) {
        partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
        partialMaximums[ulong(checkpoint) * s.pixels + pixel] = maximum;
    }
}

kernel void original_packing_reduce_bitshuffle_summary(
    const device uint *partialSums [[buffer(0)]], const device uint *headers [[buffer(1)]],
    device ulong *sums [[buffer(2)]], device uint *maximumWidths [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]], constant OriginalPackingShape &s [[buffer(5)]],
    const device uint *partialMaximums [[buffer(6)]], device uint *maximumCounts [[buffer(7)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels || atomic_load_explicit(errors, memory_order_relaxed)) return;
    if (!s.scans || s.scans % 32u || s.sourceBytes != 2u || s.pixels % 4096u) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    ulong sum = 0ul;
    // These cumulative buffers start at zero for each new acquisition and
    // contain only previously validated current-load windows, never cached
    // maxima inferred from wider-than-minimal stored encoding headers.
    uint maximumCount = maximumCounts[pixel];
    for (uint checkpoint = 0u; checkpoint < checkpoints; ++checkpoint) {
        ulong index = ulong(checkpoint) * s.pixels + pixel;
        sum += ulong(partialSums[index]);
        maximumCount = max(maximumCount, partialMaximums[index]);
    }
    uint maximumWidth = maximumWidths[pixel];
    for (uint tile = 0u; tile < tiles; ++tile) {
        uint nibble = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        maximumWidth = max(maximumWidth, nibble == 15u ? 16u : nibble);
    }
    sums[pixel] = sum;
    maximumWidths[pixel] = maximumWidth;
    maximumCounts[pixel] = maximumCount;
}



// Private cached-plan experiment. Range validation must precede this writer
// in a separate encoder with immutable headers and the same error buffer.
kernel void original_packing_values_verified_checkpoints_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (!s.pixels || !s.scans || s.scans % 32u || (s.sourceBytes != 1u && s.sourceBytes != 2u && s.sourceBytes != 4u)) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    // At most 1024 full uint16 counts, bounded by 67,107,840.
    uint sum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_count(source, tile * 32u + sample, pixel, s);
            originals[sample] = value;
            sum += value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint bit = sample * bits, shift = bit % 32u;
            ulong value = bits ? ulong(payload[offset + bit / 32u]) >> shift : 0u;
            if (shift + bits > 32u) value |= ulong(payload[offset + bit / 32u + 1u]) << (32u - shift);
            if ((value & mask) != originals[sample]) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
        }
        offset += bits;
    }
    // A different worker may fail concurrently. The following reduction must
    // run in a later encoder and observe the completed global error counter.
    if (valid) partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
}

kernel void original_packing_reduce_verified_summary(
    const device uint *partialSums [[buffer(0)]], const device uint *headers [[buffer(1)]],
    device ulong *sums [[buffer(2)]], device uint *maximumWidths [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]], constant OriginalPackingShape &s [[buffer(5)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels || atomic_load_explicit(errors, memory_order_relaxed)) return;
    if (!s.scans || s.scans % 32u || (s.sourceBytes != 1u && s.sourceBytes != 2u && s.sourceBytes != 4u)) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    ulong sum = 0ul;
    for (uint checkpoint = 0u; checkpoint < checkpoints; ++checkpoint) {
        sum += ulong(partialSums[ulong(checkpoint) * s.pixels + pixel]);
    }
    // Infer the encoding width from the validated payload plan, never a
    // cached calibration/product summary. This also respects the all-zero
    // stream's one-bit sentinel tile and the nibble 15 -> width 16 convention.
    uint maximumWidth = maximumWidths[pixel];
    for (uint tile = 0u; tile < tiles; ++tile) {
        uint nibble = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        maximumWidth = max(maximumWidth, nibble == 15u ? 16u : nibble);
    }
    sums[pixel] = sum;
    maximumWidths[pixel] = maximumWidth;
}

// Each lane owns a detector pixel. Adjacent lanes read adjacent source counts.
kernel void original_packing_headers(
    const device uchar *source [[buffer(0)]], device uint *headers [[buffer(1)]],
    device uint *sizes [[buffer(2)]], device ulong *sums [[buffer(3)]],
    device uint *maximumWidths [[buffer(4)]],
    constant OriginalPackingShape &s [[buffer(5)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint words = 0;
    ulong sum = 0;
    uint maximumWidth = maximumWidths[pixel];
    for (uint tile = 0; tile < tiles; ++tile) {
        if (tile % 32 == 0) headers[pixel * stride + tile / 32] = words;
        uint maximum = 0;
        for (uint sample = 0; sample < 32; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            maximum = max(maximum, value); sum += value;
        }
        uint bits = maximum == 0 ? 0 : 32 - clz(maximum);
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        maximumWidth = max(maximumWidth, bits);
        uint word = pixel * stride + checkpoints + tile / 8;
        if (tile % 8 == 0) headers[word] = 0;
        headers[word] |= min(bits, 15u) << ((tile % 8) * 4);
        words += bits;
    }
    sizes[pixel] = words;
    sums[pixel] = sum;
    maximumWidths[pixel] = maximumWidth;
}

kernel void original_packing_values(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    device uint *payload [[buffer(2)]], constant OriginalPackingShape &s [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    for (uint tile = 0; tile < tiles; ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        for (uint word = 0; word < bits; ++word) payload[offset + word] = 0;
        for (uint sample = 0; sample < 32 && bits; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            uint bit = sample * bits, word = bit / 32, shift = bit % 32;
            payload[offset + word] |= value << shift;
            if (shift + bits > 32) payload[offset + word + 1] |= value >> (32 - shift);
        }
        offset += bits;
    }
}

// Verify every packed count against decoded source before publishing a shard.
kernel void original_packing_verify(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    const device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    for (uint tile = 0; tile < tiles; ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0; sample < 32; ++sample) {
            uint bit = sample * bits, shift = bit % 32;
            ulong value = bits ? ulong(payload[offset + bit / 32]) >> shift : 0;
            if (shift + bits > 32) value |= ulong(payload[offset + bit / 32 + 1]) << (32 - shift);
            if ((value & mask) != original_count(source, tile * 32 + sample, pixel, s))
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        }
        offset += bits;
    }
}

// Exact intensity and row/column weighted sums directly from source bit planes.
// A 32-pixel word stays within one detector row (columns must be divisible by 32).
// Five bit-position masks recover the column weight without dense unshuffle.
kernel void original_packing_bitshuffle_dpc(
    const device uchar *source [[buffer(0)]], device ulong4 *dpc [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
    if (s.sourceBytes != 2u || s.pixels % 4096u || !s.columns || s.columns % 32u
        || simdWidth != 32u) {
        if (index == 0u) atomic_fetch_or_explicit(errors, 4u, memory_order_relaxed);
        return;
    }
    uint scan = index / 32u;
    if (scan >= s.scans) return;
    ulong total = 0ul, row = 0ul, column = 0ul;
    for (uint wordIndex = lane; wordIndex < s.pixels / 32u; wordIndex += 32u) {
        uint pixel = wordIndex * 32u;
        const device uint *block = (const device uint *)(source
            + ulong(scan) * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
        uint rowIndex = pixel / s.columns, columnIndex = pixel % s.columns;
        for (uint plane = 0u; plane < 16u; ++plane) {
            uint word = block[plane * 128u + (pixel % 4096u) / 32u];
            if (!word) continue;
            ulong count = ulong(popcount(word));
            uint localColumn = popcount(word & 0xaaaaaaaau)
                + 2u * popcount(word & 0xccccccccu)
                + 4u * popcount(word & 0xf0f0f0f0u)
                + 8u * popcount(word & 0xff00ff00u)
                + 16u * popcount(word & 0xffff0000u);
            total += count << plane;
            row += (count * rowIndex) << plane;
            column += (count * columnIndex + ulong(localColumn)) << plane;
        }
    }
    for (uint delta = 16u; delta; delta >>= 1u) {
        total += ulong(simd_shuffle_down(uint(total), delta))
            | (ulong(simd_shuffle_down(uint(total >> 32u), delta)) << 32u);
        row += ulong(simd_shuffle_down(uint(row), delta))
            | (ulong(simd_shuffle_down(uint(row >> 32u), delta)) << 32u);
        column += ulong(simd_shuffle_down(uint(column), delta))
            | (ulong(simd_shuffle_down(uint(column >> 32u), delta)) << 32u);
    }
    if (lane == 0u) dpc[scan] = ulong4(total, row, column, 0ul);
}

kernel void original_packing_moments(
    const device uchar *source [[buffer(0)]], device ulong4 *moments [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]]) {
    uint scan = index / simdWidth;
    if (scan >= s.scans) return;
    ulong total = 0, row = 0, column = 0;
    for (uint pixel = lane; pixel < s.pixels; pixel += simdWidth) {
        ulong value = original_count(source, scan, pixel, s);
        total += value; row += value * (pixel / s.columns); column += value * (pixel % s.columns);
    }
    // Read adjacent counts together. Shuffle the two halves separately, then
    // add in 64 bits so carries preserve high-count DPC products exactly.
    for (uint delta = simdWidth / 2; delta; delta >>= 1) {
        total += ulong(simd_shuffle_down(uint(total), delta))
               | (ulong(simd_shuffle_down(uint(total >> 32), delta)) << 32);
        row += ulong(simd_shuffle_down(uint(row), delta))
             | (ulong(simd_shuffle_down(uint(row >> 32), delta)) << 32);
        column += ulong(simd_shuffle_down(uint(column), delta))
                | (ulong(simd_shuffle_down(uint(column >> 32), delta)) << 32);
    }
    if (lane == 0) moments[scan] = ulong4(total, row, column, 0);
}

kernel void original_packing_u8(
    const device uchar *source [[buffer(0)]], device uchar *output [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < s.scans * s.pixels)
        output[index] = uchar(original_count(source, index / s.pixels, index % s.pixels, s));
}

// Prove the declared payload ranges form a complete, disjoint partition before
// permitting any writer to use them. Checkpoints must match the tile widths.
kernel void original_packing_validate_ranges(
    const device uint *headers [[buffer(0)]], device atomic_uint *errors [[buffer(1)]],
    const device uint *payloadWords [[buffer(2)]],
    constant OriginalPackingShape &s [[buffer(3)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint base = headers[pixel * stride];
    uint end = pixel + 1 < s.pixels ? headers[(pixel + 1) * stride] : payloadWords[0];
    bool invalid = base > end || end > payloadWords[0] || (pixel == 0 && base != 0);
    ulong words = 0;
    for (uint tile = 0; tile < tiles; ++tile) {
        if (tile && tile % 32 == 0 && ulong(headers[pixel * stride + tile / 32]) != words)
            invalid = true;
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        words += bits == 15 ? (s.sourceBytes == 4 ? 32 : 16) : bits;
    }
    if (ulong(base) + words != ulong(end)) invalid = true;
    if (invalid) atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
}

// The caller must encode the range validator first, with immutable headers and
// the same error counter. Each tile then checks every actual stored count via
// volatile readback against its original input, including zero-width tiles.
kernel void original_packing_values_verified(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    if (atomic_load_explicit(errors, memory_order_relaxed) != 0) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    for (uint tile = 0; tile < tiles; ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        uint originals[32];
        uint packed = 0, occupied = 0, outputWord = 0;
        for (uint sample = 0; sample < 32; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            originals[sample] = value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32;
                    packed = occupied ? value >> (bits - occupied) : 0;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0; sample < 32; ++sample) {
            uint bit = sample * bits, shift = bit % 32;
            ulong value = bits ? ulong(payload[offset + bit / 32]) >> shift : 0;
            if (shift + bits > 32) value |= ulong(payload[offset + bit / 32 + 1]) << (32 - shift);
            if ((value & mask) != originals[sample])
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        }
        offset += bits;
    }
}
