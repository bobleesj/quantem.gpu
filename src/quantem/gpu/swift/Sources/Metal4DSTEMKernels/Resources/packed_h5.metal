#include <metal_stdlib>
using namespace metal;

// Sum the exact, masked per-scan totals already prepared for DPC. This reads
// only the small summary buffer, never the packed 4D payload. The host checks
// a conservative UInt64 sum bound before dispatching one 256-thread group.
kernel void compact_h5_total_counts(
    device const uint *moments [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup ulong partial[256];
    ulong sum = 0;
    for (ulong scan = tid; scan < count; scan += 256) {
        ulong base = scan * 8;
        sum += ulong(moments[base]) | (ulong(moments[base + 1]) << 32);
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[0] = partial[0];
}

struct CompactLZ4Chunk {
    uint inputOffset;
    uint inputBytes;
    uint outputWord;
    uint outputBytes;
};

struct CompactLZ4Parameters {
    uint chunkCount;
    uint compressedBytes;
};

struct CompactScanParameters {
    uint count;
    // 0: bit widths to words; 1: encoded chunk lengths to bytes; 2: uint sums.
    uint kind;
};

struct CompactTableParameters {
    uint descriptorCount;
    uint chunkCount;
    uint decodedWords;
    uint compressedBytes;
    uint chunkBytes;
};

// Hierarchical exclusive scan. Every level stays on the GPU. Overflow is
// reported, never silently used as a wrapped pointer into scientific evidence.
kernel void compact_h5_scan_offsets(
    device const uint *input [[buffer(0)]],
    device uint *offsets [[buffer(1)]],
    device uint *blockSums [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant CompactScanParameters &p [[buffer(4)]],
    uint index [[thread_position_in_grid]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint sums[8];
    threadgroup uint carries[8];
    uint value = 0u;
    if (index < p.count) {
        value = p.kind == 2u ? input[index]
            : uint(reinterpret_cast<device const uchar *>(input)[index]);
        if (p.kind == 0u) value *= 4u;
        if (p.kind == 1u) value += 1u;
    }
    uint prefix = simd_prefix_exclusive_sum(value);
    if (prefix + value < prefix) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    uint sum = simd_sum(value);
    if (lane == 0u) sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        uint groupValue = lane < 8u ? sums[lane] : 0u;
        uint groupPrefix = simd_prefix_exclusive_sum(groupValue);
        if (groupPrefix + groupValue < groupPrefix)
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        if (lane < 8u) carries[lane] = groupPrefix;
        if (lane == 7u) blockSums[block] = groupPrefix + groupValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = prefix + carries[simd];
    if (offset < prefix) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    if (index < p.count) offsets[index] = offset;
}

kernel void compact_h5_add_offset_carries(
    device uint *offsets [[buffer(0)]],
    device const uint *blockOffsets [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    uint value = offsets[index];
    uint sum = value + blockOffsets[index / 256u];
    if (sum < value) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    offsets[index] = sum;
}

kernel void compact_h5_prepare_descriptors(
    device const uchar *widths [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device uint *descriptors [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant CompactTableParameters &p [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= p.descriptorCount) return;
    uint width = widths[index];
    uint offset = offsets[index];
    if (width > 16u) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    if (offset >= (1u << 27u)
        || (index + 1u == p.descriptorCount && ulong(offset) + ulong(width) * 4ul != p.decodedWords))
        atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
    descriptors[index] = (offset << 5u) | width;
}

kernel void compact_h5_prepare_chunks(
    device const uchar *lengths [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device CompactLZ4Chunk *chunks [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant CompactTableParameters &p [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= p.chunkCount) return;
    uint inputBytes = uint(lengths[index]) + 1u;
    uint outputWord = index * (p.chunkBytes / 4u);
    if (outputWord >= p.decodedWords
        || (index + 1u == p.chunkCount && ulong(offsets[index]) + inputBytes != p.compressedBytes))
        atomic_fetch_or_explicit(status, 8u, memory_order_relaxed);
    chunks[index] = CompactLZ4Chunk {
        offsets[index], inputBytes, outputWord,
        min(p.chunkBytes / 4u, p.decodedWords - min(outputWord, p.decodedWords)) * 4u
    };
}

kernel void compact_h5_reduce_decode_status(
    device const uint *chunkStatus [[buffer(0)]],
    device atomic_uint *status [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint error = simd_or(index < count ? chunkStatus[index] : 0u);
    if (lane == 0u && error != 0u)
        atomic_fetch_or_explicit(status, 16u, memory_order_relaxed);
}

struct CompactDescriptorParameters {
    uint descriptorCount;
    uint payloadWords;
    uint tileCount;
    uint headerWordsPerPixel;
    uint scanTile;
    uint headerEncoding;
};

struct CompactSelectedParameters {
    uint scan;
    uint tileCount;
    uint pixelCount;
    uint scanTile;
    uint headerWordsPerPixel;
    uint headerEncoding;
    uint payloadLayout;
};

struct CompactDetectorEntry {
    uint pixel;
    int coefficient;
};

struct CompactDetectorParameters {
    uint scanCount;
    uint tileCount;
    uint entryCount;
    uint outputOffset;
    uint mode;
    uint scanTile;
    uint headerWordsPerPixel;
    uint headerEncoding;
    uint payloadLayout;
};

struct CompactFullDecodeParameters {
    uint scanCount;
    uint pixelCount;
    uint tileCount;
    uint scanTile;
    uint headerWordsPerPixel;
    uint headerEncoding;
    uint outputWordCount;
    uint payloadLayout;
};

struct CompactDetectorSumParameters {
    uint scanCount;
    uint tileCount;
    uint pixelCount;
    uint scanTile;
    uint headerWordsPerPixel;
    uint headerEncoding;
    uint payloadLayout;
};

inline uint compactReadByte(
    device const uint *source,
    uint address
) {
    return (source[address >> 2u] >> ((address & 3u) * 8u)) & 0xffu;
}

inline uint compactReadU16LE(
    device const uint *source,
    uint address
) {
    uint shift = (address & 3u) * 8u;
    uint value = source[address >> 2u] >> shift;
    if (shift == 24u) value |= source[(address >> 2u) + 1u] << 8u;
    return value & 0xffffu;
}

inline uint compactReadDecodedByte(
    device const uint *decoded,
    uint outputBaseWord,
    uint address
) {
    return (
        decoded[outputBaseWord + (address >> 2u)]
        >> ((address & 3u) * 8u)
    ) & 0xffu;
}

inline bool compactExtendLength(
    device const uint *source,
    thread uint &position,
    uint end,
    thread uint &length
) {
    if (length != 15u) return true;
    uint next = 255u;
    while (next == 255u) {
        if (position >= end) return false;
        next = compactReadByte(source, position++);
        if (length > 0xffffffffu - next) return false;
        length += next;
    }
    return true;
}

inline uint compactFirstOwnedWord(uint firstWord, uint lane) {
    constexpr uint threadCount = 64u;
    return firstWord + ((lane + threadCount - (firstWord & 63u)) & 63u);
}

inline void compactCopyLiterals(
    device const uint *source,
    device uint *decoded,
    uint outputBaseWord,
    uint destination,
    uint input,
    uint count,
    uint lane
) {
    if (count == 0u) return;
    uint firstWord = destination >> 2u;
    uint lastWord = (destination + count - 1u) >> 2u;
    for (
        uint wordIndex = compactFirstOwnedWord(firstWord, lane);
        wordIndex <= lastWord;
        wordIndex += 64u
    ) {
        uint value = decoded[outputBaseWord + wordIndex];
        uint wordByte = wordIndex * 4u;
        for (uint byteIndex = 0u; byteIndex < 4u; ++byteIndex) {
            uint address = wordByte + byteIndex;
            if (address >= destination && address - destination < count) {
                uint shift = byteIndex * 8u;
                uint byteValue = compactReadByte(
                    source, input + address - destination
                );
                value = (value & ~(0xffu << shift)) | (byteValue << shift);
            }
        }
        decoded[outputBaseWord + wordIndex] = value;
    }
}

inline void compactCopyMatch(
    device uint *decoded,
    uint outputBaseWord,
    uint destination,
    uint source,
    uint period,
    uint count,
    uint lane
) {
    if (count == 0u) return;
    uint firstWord = destination >> 2u;
    uint lastWord = (destination + count - 1u) >> 2u;
    uint periodMask = period - 1u;
    bool powerOfTwo = period != 0u && (period & periodMask) == 0u;
    for (
        uint wordIndex = compactFirstOwnedWord(firstWord, lane);
        wordIndex <= lastWord;
        wordIndex += 64u
    ) {
        uint value = decoded[outputBaseWord + wordIndex];
        uint wordByte = wordIndex * 4u;
        for (uint byteIndex = 0u; byteIndex < 4u; ++byteIndex) {
            uint address = wordByte + byteIndex;
            if (address >= destination && address - destination < count) {
                uint relative = address - destination;
                uint repeated = powerOfTwo
                    ? (relative & periodMask)
                    : (relative % period);
                uint shift = byteIndex * 8u;
                uint byteValue = compactReadDecodedByte(
                    decoded, outputBaseWord, source + repeated
                );
                value = (value & ~(0xffu << shift)) | (byteValue << shift);
            }
        }
        decoded[outputBaseWord + wordIndex] = value;
    }
}

kernel void compact_h5_lz4_decode(
    device const uint *compressed [[buffer(0)]],
    device const CompactLZ4Chunk *chunks [[buffer(1)]],
    device uint *decoded [[buffer(2)]],
    device uint *status [[buffer(3)]],
    constant CompactLZ4Parameters &parameters [[buffer(4)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (chunk >= parameters.chunkCount) return;
    CompactLZ4Chunk record = chunks[chunk];
    uint inputPosition = record.inputOffset;
    uint inputEnd = record.inputOffset + record.inputBytes;
    uint outputPosition = 0u;
    uint tokenCount = 0u;
    uint error = 0u;
    if (
        record.inputBytes == 0u
        || record.inputOffset > parameters.compressedBytes
        || record.inputBytes > parameters.compressedBytes - record.inputOffset
        || record.outputBytes == 0u
        || (record.outputBytes & 3u) != 0u
    ) {
        error = 2u;
    }
    // Partial-word literal/match writes preserve neighboring bytes. Initialize
    // each disjoint chunk here instead of clearing the full volume on the CPU.
    if (error == 0u) {
        for (uint word = lane; word < record.outputBytes / 4u; word += 64u) {
            decoded[record.outputWord + word] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    while (
        error == 0u
        && inputPosition < inputEnd
        && outputPosition < record.outputBytes
    ) {
        if (++tokenCount > record.outputBytes) {
            error = 3u;
            break;
        }
        uint token = compactReadByte(compressed, inputPosition++);
        uint literalCount = token >> 4u;
        if (
            !compactExtendLength(
                compressed, inputPosition, inputEnd, literalCount
            )
            || literalCount > inputEnd - inputPosition
            || literalCount > record.outputBytes - outputPosition
        ) {
            error = 4u;
            break;
        }
        compactCopyLiterals(
            compressed,
            decoded,
            record.outputWord,
            outputPosition,
            inputPosition,
            literalCount,
            lane
        );
        inputPosition += literalCount;
        outputPosition += literalCount;
        if (outputPosition == record.outputBytes || inputPosition == inputEnd) {
            break;
        }
        if (inputEnd - inputPosition < 2u) {
            error = 5u;
            break;
        }
        uint matchOffset = compactReadU16LE(compressed, inputPosition);
        inputPosition += 2u;
        if (matchOffset == 0u || matchOffset > outputPosition) {
            error = 6u;
            break;
        }
        uint matchCount = token & 15u;
        if (
            !compactExtendLength(
                compressed, inputPosition, inputEnd, matchCount
            )
            || matchCount > 0xfffffffbu
        ) {
            error = 7u;
            break;
        }
        matchCount += 4u;
        if (matchCount > record.outputBytes - outputPosition) {
            error = 8u;
            break;
        }
        threadgroup_barrier(mem_flags::mem_device);
        compactCopyMatch(
            decoded,
            record.outputWord,
            outputPosition,
            outputPosition - matchOffset,
            matchOffset,
            matchCount,
            lane
        );
        outputPosition += matchCount;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0u) {
        status[chunk] = error != 0u
            ? error
            : (
                outputPosition == record.outputBytes
                && inputPosition == inputEnd
                    ? 0u
                    : 9u
            );
    }
}

// One SIMD group owns a complete 128-byte block. Each lane retains one output
// word; match references use lane shuffles instead of repeatedly reading and
// barriering device memory. Eight independent blocks share one threadgroup.
kernel void compact_h5_lz4_decode_simd32(
    device const uint *compressed [[buffer(0)]],
    device const CompactLZ4Chunk *chunks [[buffer(1)]],
    device uint *decoded [[buffer(2)]],
    device uint *status [[buffer(3)]],
    constant CompactLZ4Parameters &parameters [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint chunk = group * 8u + simd;
    if (chunk >= parameters.chunkCount) return;
    CompactLZ4Chunk record = chunks[chunk];
    uint input = record.inputOffset;
    uint end = input + record.inputBytes;
    uint output = 0u;
    uint word = 0u;
    uint tokens = 0u;
    uint error = 0u;
    if (record.inputBytes == 0u || input > parameters.compressedBytes
        || record.inputBytes > parameters.compressedBytes - input
        || record.outputBytes == 0u || record.outputBytes > 128u
        || (record.outputBytes & 3u) != 0u) error = 2u;
    while (error == 0u && input < end && output < record.outputBytes) {
        if (++tokens > record.outputBytes) { error = 3u; break; }
        uint token = compactReadByte(compressed, input++);
        uint literals = token >> 4u;
        if (!compactExtendLength(compressed, input, end, literals)
            || literals > end - input || literals > record.outputBytes - output) {
            error = 4u; break;
        }
        for (uint byte = 0u; byte < 4u; ++byte) {
            uint address = lane * 4u + byte;
            if (address >= output && address - output < literals) {
                uint value = compactReadByte(compressed, input + address - output);
                word = (word & ~(0xffu << (byte * 8u))) | (value << (byte * 8u));
            }
        }
        input += literals;
        output += literals;
        if (output == record.outputBytes || input == end) break;
        if (end - input < 2u) { error = 5u; break; }
        uint period = compactReadU16LE(compressed, input);
        input += 2u;
        if (period == 0u || period > output) { error = 6u; break; }
        uint count = token & 15u;
        if (!compactExtendLength(compressed, input, end, count) || count > 0xfffffffbu) {
            error = 7u; break;
        }
        count += 4u;
        if (count > record.outputBytes - output) { error = 8u; break; }
        // All lanes execute every shuffle, including lanes not writing a byte.
        // Its source always belongs to the already-decoded prefix. Keep that
        // prefix immutable while constructing this overlapping LZ4 match.
        uint prefixWord = word;
        bool powerOfTwo = (period & (period - 1u)) == 0u;
        for (uint byte = 0u; byte < 4u; ++byte) {
            uint address = lane * 4u + byte;
            bool active = address >= output && address - output < count;
            uint relative = active ? address - output : 0u;
            uint reference = output - period
                + (powerOfTwo ? relative & (period - 1u) : relative % period);
            uint source = simd_shuffle(prefixWord, reference >> 2u);
            if (active) {
                uint value = (source >> ((reference & 3u) * 8u)) & 0xffu;
                word = (word & ~(0xffu << (byte * 8u))) | (value << (byte * 8u));
            }
        }
        output += count;
    }
    if (error == 0u && (output != record.outputBytes || input != end)) error = 9u;
    if (error == 0u && lane < record.outputBytes / 4u) decoded[record.outputWord + lane] = word;
    if (lane == 0u) status[chunk] = error;
}

kernel void compact_h5_validate_descriptors(
    device const uint *descriptors [[buffer(0)]],
    device atomic_uint *status [[buffer(1)]],
    constant CompactDescriptorParameters &parameters [[buffer(2)]],
    device atomic_uint *maximumWidths [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (parameters.headerEncoding != 0u) {
        if (index >= parameters.descriptorCount) return;
        uint checkpointWords = (parameters.tileCount + 31u) / 32u;
        uint widthWords = (parameters.tileCount + 7u) / 8u;
        uint errors = 0u;
        if (
            parameters.scanTile != 32u
            || parameters.headerWordsPerPixel != checkpointWords + widthWords
        ) {
            errors |= 16u;
        } else {
            uint headerBase = index * parameters.headerWordsPerPixel;
            uint payloadBase = descriptors[headerBase];
            uint cursor = payloadBase;
            uint maximumWidth = 0u;
            if (payloadBase >= (1u << 27u)) errors |= 2u;
            for (uint tile = 0u; tile < parameters.tileCount; ++tile) {
                if (
                    tile != 0u
                    && (tile & 31u) == 0u
                    && descriptors[headerBase + tile / 32u] != cursor - payloadBase
                ) {
                    errors |= 32u;
                }
                uint packed = descriptors[
                    headerBase + checkpointWords + tile / 8u
                ];
                uint width = (packed >> ((tile & 7u) * 4u)) & 15u;
                if (parameters.headerEncoding == 2u && width == 15u) width = 16u;
                maximumWidth = max(maximumWidth, width);
                if (width > 16u || cursor > 0xffffffffu - width) {
                    errors |= 1u;
                    break;
                }
                cursor += width;
            }
            uint expectedNext = index + 1u < parameters.descriptorCount
                ? descriptors[headerBase + parameters.headerWordsPerPixel]
                : parameters.payloadWords;
            if (cursor != expectedNext) errors |= 4u;
            if (cursor > parameters.payloadWords) errors |= 8u;
            atomic_fetch_max_explicit(
                &maximumWidths[index], maximumWidth, memory_order_relaxed
            );
        }
        if (errors != 0u) {
            atomic_fetch_or_explicit(status, errors, memory_order_relaxed);
        }
        return;
    }
    if (index >= parameters.descriptorCount) return;
    uint descriptor = descriptors[index];
    uint width = descriptor & 31u;
    uint offset = descriptor >> 5u;
    uint errors = 0u;
    if (width > 16u) errors |= 1u;
    if (offset >= (1u << 27u)) errors |= 2u;
    if (index == 0u && offset != 0u) errors |= 4u;
    uint expectedNext = offset + width * 4u;
    atomic_fetch_max_explicit(
        &maximumWidths[index / parameters.tileCount],
        width,
        memory_order_relaxed
    );
    if (index + 1u < parameters.descriptorCount) {
        if ((descriptors[index + 1u] >> 5u) != expectedNext) errors |= 4u;
    } else if (expectedNext != parameters.payloadWords) {
        errors |= 8u;
    }
    if (errors != 0u) {
        atomic_fetch_or_explicit(status, errors, memory_order_relaxed);
    }
}

inline uint compactSumWidthNibbles(uint packed, uint count, uint headerEncoding) {
    uint mask = count >= 8u
        ? 0xffffffffu
        : (count == 0u ? 0u : (1u << (count * 4u)) - 1u);
    packed &= mask;
    uint bytes = (packed & 0x0f0f0f0fu) + ((packed >> 4u) & 0x0f0f0f0fu);
    uint total = (bytes & 0xffu) + ((bytes >> 8u) & 0xffu)
        + ((bytes >> 16u) & 0xffu) + ((bytes >> 24u) & 0xffu);
    if (headerEncoding == 2u || headerEncoding == 3u) {
        // One additional word for each nibble 15, which represents width 16.
        uint full = packed & (packed >> 1u) & (packed >> 2u) & (packed >> 3u);
        total += popcount(full & 0x11111111u) * (headerEncoding == 3u ? 17u : 1u);
    }
    return total;
}

inline uint compactDescriptorFor(
    device const uint *descriptors,
    uint tileCount,
    uint headerWordsPerPixel,
    uint headerEncoding,
    uint pixel,
    uint tile
) {
    if (headerEncoding == 0u) {
        return descriptors[pixel * tileCount + tile];
    }
    uint checkpointWords = (tileCount + 31u) / 32u;
    uint headerBase = pixel * headerWordsPerPixel;
    uint checkpoint = tile / 32u;
    uint offset = descriptors[headerBase];
    if (checkpoint != 0u) offset += descriptors[headerBase + checkpoint];
    uint firstWidthWord = checkpoint * 4u;
    uint tileWidthWord = tile / 8u;
    for (uint word = firstWidthWord; word < tileWidthWord; ++word) {
        offset += compactSumWidthNibbles(
            descriptors[headerBase + checkpointWords + word], 8u, headerEncoding
        );
    }
    uint packed = descriptors[headerBase + checkpointWords + tileWidthWord];
    offset += compactSumWidthNibbles(packed, tile & 7u, headerEncoding);
    uint width = (packed >> ((tile & 7u) * 4u)) & 15u;
    if (headerEncoding == 2u && width == 15u) width = 16u;
    // Descriptor width 31 is reserved for a full 32-bit cell. Its five-bit
    // width field and existing word offset are otherwise unchanged.
    if (headerEncoding == 3u && width == 15u) width = 31u;
    return (offset << 5u) | width;
}


inline uint compactCellValue(
    device const uint *payload, uint descriptor, uint scanInTile,
    uint payloadLayout
) {
    uint width = descriptor & 31u;
    if (width == 31u) width = 32u;
    if (width == 0u) return 0u;
    uint offset = descriptor >> 5u;
    if (payloadLayout == 1u) {
        uint value = 0u;
        for (uint plane = 0u; plane < width; ++plane) {
            value |= ((payload[offset + plane] >> scanInTile) & 1u) << plane;
        }
        return value;
    }
    uint bit = scanInTile * width;
    uint shift = bit % 32u;
    uint index = offset + bit / 32u;
    uint value = payload[index] >> shift;
    if (shift + width > 32u) value |= payload[index + 1u] << (32u - shift);
    return value & uint((1ul << width) - 1ul);
}

inline uint compactSampleValue(
    device const uint *payload,
    device const uint *descriptors,
    uint tileCount,
    uint scanTile,
    uint headerWordsPerPixel,
    uint headerEncoding,
    uint pixel,
    uint scan,
    uint payloadLayout = 0u
) {
    uint descriptor = compactDescriptorFor(
        descriptors,
        tileCount,
        headerWordsPerPixel,
        headerEncoding,
        pixel,
        scan / scanTile
    );
    return compactCellValue(payload, descriptor, scan % scanTile, payloadLayout);
}

// Wide source counts retain exact UInt64 detector sums. Floating-point values
// are produced only for the independent display snapshot, never the resident.
kernel void compact_h5_detector_update_u64(
    const device uint *payload [[buffer(0)]], const device uint *descriptors [[buffer(1)]],
    const device CompactDetectorEntry *entries [[buffer(2)]],
    const device ulong *previous [[buffer(3)]], device ulong *output [[buffer(4)]],
    constant CompactDetectorParameters &p [[buffer(5)]],
    const device ulong4 *moments [[buffer(6)]],
    uint scan [[thread_position_in_grid]]) {
    if (scan >= p.scanCount) return;
    long delta = 0l;
    for (uint i = 0u; i < p.entryCount; ++i) {
        uint value = compactSampleValue(payload, descriptors, p.tileCount, p.scanTile,
            p.headerWordsPerPixel, p.headerEncoding, entries[i].pixel, scan, p.payloadLayout);
        delta += long(value) * entries[i].coefficient;
    }
    uint index = p.outputOffset + scan;
    ulong base = p.mode == 2u ? moments[index].x : (p.mode == 1u ? 0ul : previous[index]);
    output[index] = delta < 0l ? base - ulong(-delta) : base + ulong(delta);
}

kernel void compact_h5_u64_display(
    const device ulong *values [[buffer(0)]], device float *display [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < count) display[index] = float(values[index]);
}

// Experimental auxiliary exact block sums. The original packed evidence remains
// resident; these sums replace only complete equal-coefficient mask interiors.

kernel void compact_h5_selected_diffraction(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const uint *excluded [[buffer(2)]],
    device uint *diffraction [[buffer(3)]],
    constant CompactSelectedParameters &parameters [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= parameters.pixelCount) return;
    diffraction[pixel] = excluded[pixel] != 0u
        ? 0u
        : compactSampleValue(
            payload,
            descriptors,
            parameters.tileCount,
            parameters.scanTile,
            parameters.headerWordsPerPixel,
            parameters.headerEncoding,
            pixel,
            parameters.scan,
            parameters.payloadLayout
        );
}

kernel void compact_h5_full_decode_u8(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const uint *excluded [[buffer(2)]],
    device uint *output [[buffer(3)]],
    constant CompactFullDecodeParameters &parameters [[buffer(4)]],
    uint outputWord [[thread_position_in_grid]]
) {
    if (outputWord >= parameters.outputWordCount) return;
    uint packed = 0u;
    for (uint byteIndex = 0u; byteIndex < 4u; ++byteIndex) {
        uint linear = outputWord * 4u + byteIndex;
        uint scan = linear / parameters.pixelCount;
        uint pixel = linear - scan * parameters.pixelCount;
        uint value = excluded[pixel] != 0u
            ? 0u
            : compactSampleValue(
                payload,
                descriptors,
                parameters.tileCount,
                parameters.scanTile,
                parameters.headerWordsPerPixel,
                parameters.headerEncoding,
                pixel,
                scan,
                parameters.payloadLayout
            );
        packed |= value << (byteIndex * 8u);
    }
    output[outputWord] = packed;
}

kernel void compact_h5_detector_update(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const CompactDetectorEntry *entries [[buffer(2)]],
    device const uint *previous [[buffer(3)]],
    device uint *next [[buffer(4)]],
    constant CompactDetectorParameters &parameters [[buffer(5)]],
    device const uint *preparedMoments [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    constexpr uint lanesPerScan = 4u;
    constexpr uint scansPerGroup = 32u;
    uint scanLane = lane % scansPerGroup;
    uint entryLane = lane / scansPerGroup;
    uint scan = group * scansPerGroup + scanLane;
    uint partial = 0u;
    if (scan < parameters.scanCount) {
        for (
            uint index = entryLane;
            index < parameters.entryCount;
            index += lanesPerScan
        ) {
            CompactDetectorEntry entry = entries[index];
            // All 32 SIMD lanes inspect one packed scan tile. Resolve its
            // checkpoint once, then share the descriptor across the lanes.
            uint descriptor = 0u;
            if (scanLane == 0u) {
                descriptor = compactDescriptorFor(
                    descriptors, parameters.tileCount,
                    parameters.headerWordsPerPixel, parameters.headerEncoding,
                    entry.pixel, scan / parameters.scanTile
                );
            }
            descriptor = simd_broadcast_first(descriptor);
            uint value = compactCellValue(
                payload, descriptor, scan % parameters.scanTile, parameters.payloadLayout
            );
            if (entry.coefficient == 1) partial += value;
            else partial -= value;
        }
    }
    threadgroup uint partials[128];
    partials[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (entryLane == 0u && scan < parameters.scanCount) {
        uint output = parameters.mode == 1u
            ? 0u
            : (parameters.mode == 2u
                ? preparedMoments[(parameters.outputOffset + scan) * 8u]
                : previous[parameters.outputOffset + scan]);
        // Modulo-u32 total-minus-complement is exact because the host bounds
        // the final selected sum to u32, even if the full total exceeds it.
        for (uint part = 0u; part < lanesPerScan; ++part) {
            output += partials[scanLane + part * scansPerGroup];
        }
        next[parameters.outputOffset + scan] = output;
    }
}

inline uint compactTransposeStep(uint value, uint lane, uint distance, uint mask) {
    uint partner = simd_shuffle_xor(value, distance);
    return (lane & distance) != 0u
        ? ((partner >> distance) & mask) | (value & ~mask)
        : (value & mask) | ((partner & mask) << distance);
}

// Input lane is detector entry and input bit is scan. Output lane is scan
// and output bit is detector entry, without reversing either coordinate.
inline uint compactTransposeBits(uint value, uint lane) {
    value = compactTransposeStep(value, lane, 16u, 0x0000ffffu);
    value = compactTransposeStep(value, lane, 8u, 0x00ff00ffu);
    value = compactTransposeStep(value, lane, 4u, 0x0f0f0f0fu);
    value = compactTransposeStep(value, lane, 2u, 0x33333333u);
    return compactTransposeStep(value, lane, 1u, 0x55555555u);
}

// Pack the existing exact one-shard auxiliary sums directly into bitplanes.

inline uint compactEvenBits(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1u)) & 0x33333333u;
    value = (value | (value >> 2u)) & 0x0f0f0f0fu;
    value = (value | (value >> 4u)) & 0x00ff00ffu;
    return (value | (value >> 8u)) & 0x0000ffffu;
}

inline uint compactLowBitScanSums(
    device const uint *payload,
    uint descriptor,
    int coefficient,
    constant CompactDetectorParameters &parameters,
    uint scanBase,
    uint scanLane,
    uint maximumWidth
) {
    if (maximumWidth == 0u) return 0u;
    uint width = descriptor & 31u;
    uint low = 0u;
    uint high = 0u;
    if (width != 0u) {
        // Every group begins at a multiple of32 scans, so these one-/two-bit
        // cells start on a word boundary for both32- and128-scan packing.
        uint word = (descriptor >> 5u) + (scanBase % parameters.scanTile) * width / 32u;
        uint first = payload[word];
        if (width == 1u) {
            low = first;
        } else {
            uint second = payload[word + 1u];
            low = compactEvenBits(first) | (compactEvenBits(second) << 16u);
            high = compactEvenBits(first >> 1u) | (compactEvenBits(second >> 1u) << 16u);
        }
    }
    // Distinct lane bits add without carry; this is the coefficient bitmask.
    uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
    uint negative = ~positive;
    low = compactTransposeBits(low, scanLane);
    uint sum = popcount(low & positive) - popcount(low & negative);
    if (maximumWidth == 2u) {
        high = compactTransposeBits(high, scanLane);
        sum += 2u * (popcount(high & positive) - popcount(high & negative));
    }
    return sum;
}

inline uint compactThirdBits(uint value) {
    uint high = (value >> 20u) & 0x400u;
    value &= 0x09249249u;
    value = (value ^ (value >> 2u)) & 0x030c30c3u;
    value = (value ^ (value >> 4u)) & 0x0300f00fu;
    value = (value ^ (value >> 8u)) & 0x030000ffu;
    value = (value ^ (value >> 16u)) & 0x000003ffu;
    return value | high;
}

inline uint compactFourthBits(uint value) {
    value &= 0x11111111u;
    value = (value | (value >> 3u)) & 0x03030303u;
    value = (value | (value >> 6u)) & 0x000f000fu;
    return (value | (value >> 12u)) & 0xffu;
}

// Separate from the measured <=2-bit path: three/four-bit groups transpose
// up to four exact bit planes instead of repeatedly reducing packed fields.
inline uint compactMediumBitScanSums(
    device const uint *payload, uint descriptor, int coefficient,
    constant CompactDetectorParameters &parameters,
    uint scanBase, uint scanLane, uint maximumWidth
) {
    uint width = descriptor & 31u;
    uint4 planes = uint4(0u);
    if (width != 0u && scanBase < parameters.scanCount) {
        uint offset = (descriptor >> 5u) + ((scanBase % parameters.scanTile) * width) / 32u;
        uint4 words = uint4(0u);
        for (uint word = 0u; word < width; ++word) words[word] = payload[offset + word];
        if (width == 1u) {
            planes[0] = words[0];
        } else if (width == 2u) {
            planes[0] = compactEvenBits(words[0]) | (compactEvenBits(words[1]) << 16u);
            planes[1] = compactEvenBits(words[0] >> 1u) | (compactEvenBits(words[1] >> 1u) << 16u);
        } else if (width == 3u) {
            for (uint plane = 0u; plane < 3u; ++plane) {
                uint secondShift = (plane + 1u) % 3u;
                uint thirdShift = (plane + 2u) % 3u;
                uint firstCount = (34u - plane) / 3u;
                uint secondCount = (34u - secondShift) / 3u;
                planes[plane] = compactThirdBits(words[0] >> plane)
                    | (compactThirdBits(words[1] >> secondShift) << firstCount)
                    | (compactThirdBits(words[2] >> thirdShift) << (firstCount + secondCount));
            }
        } else {
            for (uint plane = 0u; plane < 4u; ++plane) {
                for (uint word = 0u; word < 4u; ++word) {
                    planes[plane] |= compactFourthBits(words[word] >> plane) << (8u * word);
                }
            }
        }
    }
    uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
    uint sum = 0u;
    for (uint plane = 0u; plane < maximumWidth; ++plane) {
        uint bits = compactTransposeBits(planes[plane], scanLane);
        sum += (1u << plane) * (popcount(bits & positive) - popcount(bits & ~positive));
    }
    return sum;
}

// Sum independent scan fields without carry between fields. The caller must
// bound 32 detector entries to each field: width <= 3 for eight-bit fields,
// or width <= 11 for sixteen-bit fields. Coefficients are reduced separately.
template<uint samplesPerPack, uint fieldBits>
inline uint compactPackedScanSums(
    device const uint *payload,
    uint descriptor,
    int coefficient,
    constant CompactDetectorParameters &parameters,
    uint scanBase,
    uint scanLane
) {
    uint width = descriptor & 31u;
    uint partial = 0u;
    for (uint block = 0u; block < 32u / samplesPerPack; ++block) {
        uint packedSamples = 0u;
        uint scan = scanBase + block * samplesPerPack;
        if (width != 0u && scan < parameters.scanCount) {
            uint bit = (scan % parameters.scanTile) * width;
            uint word = (descriptor >> 5u) + bit / 32u;
            uint shift = bit % 32u;
            uint bits = payload[word] >> shift;
            if (shift + width * samplesPerPack > 32u) {
                bits |= payload[word + 1u] << (32u - shift);
            }
            uint mask = (1u << width) - 1u;
            for (uint sample = 0u; sample < samplesPerPack; ++sample) {
                if (scan + sample < parameters.scanCount) {
                    packedSamples |= ((bits >> (sample * width)) & mask)
                        << (sample * fieldBits);
                }
            }
        }
        uint positive = simd_sum(coefficient == 1 ? packedSamples : 0u);
        uint negative = simd_sum(coefficient == 1 ? 0u : packedSamples);
        if (scanLane / samplesPerPack == block) {
            uint shift = (scanLane % samplesPerPack) * fieldBits;
            uint mask = (1u << fieldBits) - 1u;
            partial += ((positive >> shift) & mask) - ((negative >> shift) & mask);
        }
    }
    return partial;
}

// Latency-oriented topology for nontrivial detector deltas. Each SIMD lane
// owns a detector entry, not a scan sample; small deltas use the scan-lane
// kernel above to avoid the fixed cost of the transposed reduction.
kernel void compact_h5_detector_update_pixel_lanes(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const CompactDetectorEntry *entries [[buffer(2)]],
    device const uint *previous [[buffer(3)]],
    device uint *next [[buffer(4)]],
    constant CompactDetectorParameters &parameters [[buffer(5)]],
    device const uint *preparedMoments [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    constexpr uint scansPerGroup = 32u;
    constexpr uint entryGroups = 4u;
    uint scanLane = lane % scansPerGroup;
    uint entryGroup = lane / scansPerGroup;
    uint scanBase = group * scansPerGroup;
    uint partial = 0u;
    for (uint first = 0u; first < parameters.entryCount; first += 128u) {
        uint entryIndex = first + lane;
        uint descriptor = 0u;
        int coefficient = 1;
        if (entryIndex < parameters.entryCount) {
            CompactDetectorEntry entry = entries[entryIndex];
            coefficient = entry.coefficient;
            descriptor = compactDescriptorFor(
                descriptors, parameters.tileCount,
                parameters.headerWordsPerPixel, parameters.headerEncoding,
                entry.pixel, scanBase / parameters.scanTile
            );
        }
        uint width = descriptor & 31u;
        uint maximumWidth = simd_max(width);
        if (parameters.payloadLayout == 1u) {
            uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
            for (uint plane = 0u; plane < maximumWidth; ++plane) {
                uint bits = width > plane && scanBase < parameters.scanCount
                    ? payload[(descriptor >> 5u) + plane] : 0u;
                bits = compactTransposeBits(bits, scanLane);
                partial += (1u << plane)
                    * (popcount(bits & positive) - popcount(bits & ~positive));
            }
            continue;
        }
        if (maximumWidth <= 2u) {
            partial += compactLowBitScanSums(
                payload, descriptor, coefficient, parameters,
                scanBase, scanLane, maximumWidth
            );
            continue;
        }
        if (maximumWidth <= 4u) {
            partial += compactMediumBitScanSums(
                payload, descriptor, coefficient, parameters,
                scanBase, scanLane, maximumWidth
            );
            continue;
        }
        if (maximumWidth <= 11u) {
            // 32 * 2047 = 65504 fits each independent sixteen-bit sum.
            partial += compactPackedScanSums<2u, 16u>(
                payload, descriptor, coefficient, parameters, scanBase, scanLane
            );
            continue;
        }
        // All lanes participate in every reduction, including padded entries
        // and the final partial scan group. No divergent SIMD reduction.
        for (uint sample = 0u; sample < scansPerGroup; ++sample) {
            uint value = 0u;
            uint scan = scanBase + sample;
            if (width != 0u && scan < parameters.scanCount) {
                uint bit = (scan % parameters.scanTile) * width;
                uint word = (descriptor >> 5u) + bit / 32u;
                uint shift = bit % 32u;
                value = payload[word] >> shift;
                if (shift + width > 32u) {
                    value |= payload[word + 1u] << (32u - shift);
                }
                value &= (1u << width) - 1u;
            }
            uint signedValue = coefficient == 1 ? value : 0u - value;
            uint sum = simd_sum(signedValue);
            if (scanLane == sample) partial += sum;
        }
    }
    threadgroup uint partials[128];
    partials[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint scan = scanBase + scanLane;
    if (entryGroup == 0u && scan < parameters.scanCount) {
        uint output = parameters.mode == 1u
            ? 0u
            : (parameters.mode == 2u
                ? preparedMoments[(parameters.outputOffset + scan) * 8u]
                : previous[parameters.outputOffset + scan]);
        for (uint part = 0u; part < entryGroups; ++part) {
            output += partials[scanLane + part * scansPerGroup];
        }
        next[parameters.outputOffset + scan] = output;
    }
}

// Experimental raw-planar ILP: expose two independent transpose chains while
// preserving the single-plane tail and exact signed modulo-u32 accumulation.
kernel void compact_h5_detector_update_planar_ilp(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const CompactDetectorEntry *entries [[buffer(2)]],
    device const uint *previous [[buffer(3)]],
    device uint *next [[buffer(4)]],
    constant CompactDetectorParameters &p [[buffer(5)]],
    device const uint *preparedMoments [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint scanLane = lane % 32u;
    uint entryGroup = lane / 32u;
    uint scanBase = group * 32u;
    uint partial = 0u;
    for (uint first = 0u; first < p.entryCount; first += 128u) {
        uint descriptor = 0u;
        int coefficient = 1;
        if (first + lane < p.entryCount) {
            CompactDetectorEntry entry = entries[first + lane];
            coefficient = entry.coefficient;
            descriptor = compactDescriptorFor(descriptors, p.tileCount,
                p.headerWordsPerPixel, p.headerEncoding, entry.pixel, scanBase / p.scanTile);
        }
        uint width = descriptor & 31u;
        uint maximumWidth = simd_max(width);
        uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
        uint plane = 0u;
        for (; plane + 1u < maximumWidth; plane += 2u) {
            uint a = width > plane && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane] : 0u;
            uint b = width > plane + 1u && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane + 1u] : 0u;
            a = compactTransposeStep(a, scanLane, 16u, 0x0000ffffu);
            b = compactTransposeStep(b, scanLane, 16u, 0x0000ffffu);
            a = compactTransposeStep(a, scanLane, 8u, 0x00ff00ffu);
            b = compactTransposeStep(b, scanLane, 8u, 0x00ff00ffu);
            a = compactTransposeStep(a, scanLane, 4u, 0x0f0f0f0fu);
            b = compactTransposeStep(b, scanLane, 4u, 0x0f0f0f0fu);
            a = compactTransposeStep(a, scanLane, 2u, 0x33333333u);
            b = compactTransposeStep(b, scanLane, 2u, 0x33333333u);
            a = compactTransposeStep(a, scanLane, 1u, 0x55555555u);
            b = compactTransposeStep(b, scanLane, 1u, 0x55555555u);
            uint low = popcount(a & positive) - popcount(a & ~positive);
            uint high = popcount(b & positive) - popcount(b & ~positive);
            partial += (low + (high << 1u)) << plane;
        }
        if (plane < maximumWidth) {
            uint bits = width > plane && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane] : 0u;
            bits = compactTransposeBits(bits, scanLane);
            partial += (popcount(bits & positive) - popcount(bits & ~positive)) << plane;
        }
    }
    threadgroup uint partials[128];
    partials[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint scan = scanBase + scanLane;
    if (entryGroup == 0u && scan < p.scanCount) {
        uint output = p.mode == 1u ? 0u : (p.mode == 2u
            ? preparedMoments[(p.outputOffset + scan) * 8u] : previous[p.outputOffset + scan]);
        for (uint part = 0u; part < 4u; ++part) output += partials[scanLane + part * 32u];
        next[p.outputOffset + scan] = output;
    }
}

// Experimental adjacent-tile ownership: each SIMD produces its own 32 scans.
// Neighboring SIMDs read adjacent cells of identical detector-entry batches;
// no cross-SIMD barrier or intermediate reduction is needed.
kernel void compact_h5_detector_update_planar_scan_cooperative(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const CompactDetectorEntry *entries [[buffer(2)]],
    device const uint *previous [[buffer(3)]],
    device uint *next [[buffer(4)]],
    constant CompactDetectorParameters &p [[buffer(5)]],
    device const uint *preparedMoments [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint scanLane = lane % 32u;
    uint tileLane = lane / 32u;
    uint scanBase = (group * 4u + tileLane) * 32u;
    if (scanBase >= p.scanCount) return; // Whole SIMD groups only.
    uint partial = 0u;
    for (uint first = 0u; first < p.entryCount; first += 32u) {
        uint descriptor = 0u;
        int coefficient = 1;
        if (first + scanLane < p.entryCount) {
            CompactDetectorEntry entry = entries[first + scanLane];
            coefficient = entry.coefficient;
            descriptor = compactDescriptorFor(descriptors, p.tileCount,
                p.headerWordsPerPixel, p.headerEncoding, entry.pixel, scanBase / p.scanTile);
        }
        uint width = descriptor & 31u;
        uint maximumWidth = simd_max(width);
        uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
        uint plane = 0u;
        for (; plane + 1u < maximumWidth; plane += 2u) {
            uint a = width > plane && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane] : 0u;
            uint b = width > plane + 1u && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane + 1u] : 0u;
            a = compactTransposeStep(a, scanLane, 16u, 0x0000ffffu);
            b = compactTransposeStep(b, scanLane, 16u, 0x0000ffffu);
            a = compactTransposeStep(a, scanLane, 8u, 0x00ff00ffu);
            b = compactTransposeStep(b, scanLane, 8u, 0x00ff00ffu);
            a = compactTransposeStep(a, scanLane, 4u, 0x0f0f0f0fu);
            b = compactTransposeStep(b, scanLane, 4u, 0x0f0f0f0fu);
            a = compactTransposeStep(a, scanLane, 2u, 0x33333333u);
            b = compactTransposeStep(b, scanLane, 2u, 0x33333333u);
            a = compactTransposeStep(a, scanLane, 1u, 0x55555555u);
            b = compactTransposeStep(b, scanLane, 1u, 0x55555555u);
            uint low = popcount(a & positive) - popcount(a & ~positive);
            uint high = popcount(b & positive) - popcount(b & ~positive);
            partial += (low + (high << 1u)) << plane;
        }
        if (plane < maximumWidth) {
            uint bits = width > plane && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane] : 0u;
            bits = compactTransposeBits(bits, scanLane);
            partial += (popcount(bits & positive) - popcount(bits & ~positive)) << plane;
        }
    }
    uint scan = scanBase + scanLane;
    if (scan < p.scanCount) {
        uint output = p.mode == 1u ? 0u : (p.mode == 2u
            ? preparedMoments[(p.outputOffset + scan) * 8u] : previous[p.outputOffset + scan]);
        next[p.outputOffset + scan] = output + partial;
    }
}



// Carry four independent scan-tile bit matrices through the shuffle network
// together, exposing their instruction independence without another buffer.
template <bool constantWide = false>
inline void compactPlanarQuadVector(
    device const uint *payload, device const uint *descriptors,
    device const CompactDetectorEntry *entries,
    device const uint *previous, device uint *next,
    constant CompactDetectorParameters &p, device const uint *preparedMoments,
    uint group, uint lane, uint groupSize, threadgroup uint *partials) {
    uint scanLane = lane % 32u, entryGroup = lane / 32u;
    uint scanBase = group * 128u;
    uint4 partial(0u);
    for (uint first = 0u; first < p.entryCount; first += groupSize) {
        uint4 widths(0u), offsets(0u);
        int coefficient = 1;
        if (first + lane < p.entryCount) {
            CompactDetectorEntry entry = entries[first + lane];
            coefficient = entry.coefficient;
            uint tile = scanBase / p.scanTile;
            uint descriptor = compactDescriptorFor(descriptors, p.tileCount,
                p.headerWordsPerPixel, p.headerEncoding, entry.pixel, tile);
            uint widthWord = 0u, offset = descriptor >> 5u;
            if (p.headerEncoding != 0u) {
                uint checkpoints = (p.tileCount + 31u) / 32u;
                widthWord = descriptors[entry.pixel * p.headerWordsPerPixel + checkpoints + tile / 8u]
                    >> ((tile % 8u) * 4u);
            }
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                if (scanBase + part * 32u < p.scanCount) {
                    if (p.headerEncoding == 0u) {
                        uint cell = descriptors[entry.pixel * p.tileCount + tile + part];
                        offsets[part] = cell >> 5u;
                        widths[part] = cell & 31u;
                    } else {
                        uint width = (widthWord >> (part * 4u)) & 15u;
                        if (p.headerEncoding == 2u && width == 15u) width = 16u;
                        offsets[part] = offset;
                        widths[part] = width;
                        offset += width;
                    }
                }
            }
        }
        if (constantWide && simd_any(any(widths > uint4(8u)))) {
            // A uniform bit plane is either all zero or all one. Authenticate
            // every plane before replacing a wide tile with its exact constant.
            // Nonuniform tiles retain the ordinary transposition below.
            uint4 constants(0u);
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                if (widths[part] > 8u) {
                    uint value = 0u;
                    bool uniform = true;
                    for (uint plane = 0u; plane < widths[part]; ++plane) {
                        uint bits = payload[offsets[part] + plane];
                        if (bits != 0u && bits != 0xffffffffu) {
                            uniform = false;
                            break;
                        }
                        value |= (bits & 1u) << plane;
                    }
                    if (uniform) {
                        constants[part] = coefficient == 1 ? value : 0u - value;
                        widths[part] = 0u;
                    }
                }
            }
            partial += simd_sum(constants);
        }
        uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
        uint maximumWidth = simd_max(max(max(widths.x, widths.y), max(widths.z, widths.w)));
        for (uint plane = 0u; plane < maximumWidth; ++plane) {
            uint4 value(0u);
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                if (widths[part] > plane) value[part] = payload[offsets[part] + plane];
            }
            #pragma unroll
            for (uint distance = 16u; distance; distance >>= 1u) {
                uint mask = 0xffffffffu / ((1u << distance) + 1u);
                uint4 partner = simd_shuffle_xor(value, distance);
                value = (scanLane & distance)
                    ? ((partner >> distance) & mask) | (value & ~mask)
                    : (value & mask) | ((partner & mask) << distance);
            }
            partial += (popcount(value & positive) - popcount(value & ~positive)) << plane;
        }
    }
    for (uint tile = 0u; tile < 4u; ++tile) partials[tile * groupSize + lane] = partial[tile];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (entryGroup == 0u) {
        for (uint tile = 0u; tile < 4u; ++tile) {
            uint scan = scanBase + tile * 32u + scanLane;
            if (scan < p.scanCount) {
                uint output = p.mode == 1u ? 0u : (p.mode == 2u
                    ? preparedMoments[(p.outputOffset + scan) * 8u] : previous[p.outputOffset + scan]);
                for (uint part = 0u; part < groupSize / 32u; ++part) {
                    output += partials[tile * groupSize + scanLane + part * 32u];
                }
                next[p.outputOffset + scan] = output;
            }
        }
    }
}

kernel void compact_h5_detector_update_planar_quad_vector(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const CompactDetectorEntry *entries [[buffer(2)]],
    device const uint *previous [[buffer(3)]], device uint *next [[buffer(4)]],
    constant CompactDetectorParameters &p [[buffer(5)]],
    device const uint *preparedMoments [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]) {
    threadgroup uint partials[512];
    compactPlanarQuadVector(payload, descriptors, entries, previous, next,
        p, preparedMoments, group, lane, groupSize, partials);
}

kernel void compact_h5_detector_update_planar_quad_constant(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const CompactDetectorEntry *entries [[buffer(2)]],
    device const uint *previous [[buffer(3)]], device uint *next [[buffer(4)]],
    constant CompactDetectorParameters &p [[buffer(5)]],
    device const uint *preparedMoments [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]) {
    threadgroup uint partials[512];
    compactPlanarQuadVector<true>(payload, descriptors, entries, previous, next,
        p, preparedMoments, group, lane, groupSize, partials);
}


// Experimental fused planar reduction: both exact inputs accumulate before
// a single output store. The existing two-phase path remains the fallback.
kernel void compact_h5_detector_update_planar_fused(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const CompactDetectorEntry *entries [[buffer(2)]],
    device const uint *previous [[buffer(3)]],
    device uint *next [[buffer(4)]],
    constant CompactDetectorParameters &p [[buffer(5)]],
    device const uint *preparedMoments [[buffer(6)]],
    device const uint *auxiliaryPayload [[buffer(7)]],
    device const uint *auxiliaryDescriptors [[buffer(8)]],
    device const CompactDetectorEntry *auxiliaryEntries [[buffer(9)]],
    constant uint &auxiliaryEntryCount [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint scanLane = lane % 32u;
    uint tileLane = lane / 32u;
    uint scanBase = (group * 4u + tileLane) * 32u;
    if (scanBase >= p.scanCount) return; // Whole SIMD groups only.
    uint partial = 0u;
    for (uint first = 0u; first < p.entryCount; first += 32u) {
        uint descriptor = 0u;
        int coefficient = 1;
        if (first + scanLane < p.entryCount) {
            CompactDetectorEntry entry = entries[first + scanLane];
            coefficient = entry.coefficient;
            descriptor = compactDescriptorFor(descriptors, p.tileCount,
                p.headerWordsPerPixel, p.headerEncoding, entry.pixel, scanBase / p.scanTile);
        }
        uint width = descriptor & 31u;
        uint maximumWidth = simd_max(width);
        uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
        uint plane = 0u;
        for (; plane + 1u < maximumWidth; plane += 2u) {
            uint a = width > plane && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane] : 0u;
            uint b = width > plane + 1u && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane + 1u] : 0u;
            a = compactTransposeStep(a, scanLane, 16u, 0x0000ffffu);
            b = compactTransposeStep(b, scanLane, 16u, 0x0000ffffu);
            a = compactTransposeStep(a, scanLane, 8u, 0x00ff00ffu);
            b = compactTransposeStep(b, scanLane, 8u, 0x00ff00ffu);
            a = compactTransposeStep(a, scanLane, 4u, 0x0f0f0f0fu);
            b = compactTransposeStep(b, scanLane, 4u, 0x0f0f0f0fu);
            a = compactTransposeStep(a, scanLane, 2u, 0x33333333u);
            b = compactTransposeStep(b, scanLane, 2u, 0x33333333u);
            a = compactTransposeStep(a, scanLane, 1u, 0x55555555u);
            b = compactTransposeStep(b, scanLane, 1u, 0x55555555u);
            uint low = popcount(a & positive) - popcount(a & ~positive);
            uint high = popcount(b & positive) - popcount(b & ~positive);
            partial += (low + (high << 1u)) << plane;
        }
        if (plane < maximumWidth) {
            uint bits = width > plane && scanBase < p.scanCount ? payload[(descriptor >> 5u) + plane] : 0u;
            bits = compactTransposeBits(bits, scanLane);
            partial += (popcount(bits & positive) - popcount(bits & ~positive)) << plane;
        }
    }
    // Auxiliary sums are UInt32 with up to 22 planes, never narrowed to UInt16.
    for (uint first = 0u; first < auxiliaryEntryCount; first += 32u) {
        uint descriptor = 0u;
        int coefficient = 1;
        if (first + scanLane < auxiliaryEntryCount) {
            CompactDetectorEntry entry = auxiliaryEntries[first + scanLane];
            coefficient = entry.coefficient;
            descriptor = auxiliaryDescriptors[entry.pixel * p.tileCount + scanBase / 32u];
        }
        uint width = descriptor & 31u;
        uint maximumWidth = simd_max(width);
        uint positive = simd_sum(coefficient == 1 ? (1u << scanLane) : 0u);
        uint plane = 0u;
        for (; plane + 1u < maximumWidth; plane += 2u) {
            uint a = width > plane && scanBase < p.scanCount ? auxiliaryPayload[(descriptor >> 5u) + plane] : 0u;
            uint b = width > plane + 1u && scanBase < p.scanCount ? auxiliaryPayload[(descriptor >> 5u) + plane + 1u] : 0u;
            a = compactTransposeStep(a, scanLane, 16u, 0x0000ffffu);
            b = compactTransposeStep(b, scanLane, 16u, 0x0000ffffu);
            a = compactTransposeStep(a, scanLane, 8u, 0x00ff00ffu);
            b = compactTransposeStep(b, scanLane, 8u, 0x00ff00ffu);
            a = compactTransposeStep(a, scanLane, 4u, 0x0f0f0f0fu);
            b = compactTransposeStep(b, scanLane, 4u, 0x0f0f0f0fu);
            a = compactTransposeStep(a, scanLane, 2u, 0x33333333u);
            b = compactTransposeStep(b, scanLane, 2u, 0x33333333u);
            a = compactTransposeStep(a, scanLane, 1u, 0x55555555u);
            b = compactTransposeStep(b, scanLane, 1u, 0x55555555u);
            uint low = popcount(a & positive) - popcount(a & ~positive);
            uint high = popcount(b & positive) - popcount(b & ~positive);
            partial += (low + (high << 1u)) << plane;
        }
        if (plane < maximumWidth) {
            uint bits = width > plane && scanBase < p.scanCount ? auxiliaryPayload[(descriptor >> 5u) + plane] : 0u;
            bits = compactTransposeBits(bits, scanLane);
            partial += (popcount(bits & positive) - popcount(bits & ~positive)) << plane;
        }
    }
    uint scan = scanBase + scanLane;
    if (scan < p.scanCount) {
        uint output = p.mode == 1u ? 0u : (p.mode == 2u
            ? preparedMoments[(p.outputOffset + scan) * 8u] : previous[p.outputOffset + scan]);
        next[p.outputOffset + scan] = output + partial;
    }
}



kernel void compact_h5_detector_sum_u64(
    device const uint *payload [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const uint *excluded [[buffer(2)]],
    device ulong *detectorSum [[buffer(3)]],
    constant CompactDetectorSumParameters &parameters [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= parameters.pixelCount || excluded[pixel] != 0u) return;
    ulong total = 0ul;
    if (parameters.payloadLayout == 1u) {
        for (uint tile = 0u; tile < parameters.tileCount; ++tile) {
            uint descriptor = compactDescriptorFor(
                descriptors, parameters.tileCount, parameters.headerWordsPerPixel,
                parameters.headerEncoding, pixel, tile
            );
            uint width = descriptor & 31u;
            for (uint plane = 0u; plane < width; ++plane) {
                total += ulong(popcount(payload[(descriptor >> 5u) + plane])) << plane;
            }
        }
        detectorSum[pixel] += total;
        return;
    }
    for (uint scan = 0u; scan < parameters.scanCount; ++scan) {
        total += ulong(compactSampleValue(
            payload,
            descriptors,
            parameters.tileCount,
            parameters.scanTile,
            parameters.headerWordsPerPixel,
            parameters.headerEncoding,
            pixel,
            scan,
            parameters.payloadLayout
        ));
    }
    detectorSum[pixel] += total;
}
