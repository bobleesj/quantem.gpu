#include <metal_stdlib>
using namespace metal;

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
};

struct CompactFullDecodeParameters {
    uint scanCount;
    uint pixelCount;
    uint tileCount;
    uint scanTile;
    uint headerWordsPerPixel;
    uint headerEncoding;
    uint outputWordCount;
};

struct CompactDetectorSumParameters {
    uint scanCount;
    uint tileCount;
    uint pixelCount;
    uint scanTile;
    uint headerWordsPerPixel;
    uint headerEncoding;
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

inline uint compactSumWidthNibbles(uint packed, uint count) {
    uint mask = count >= 8u
        ? 0xffffffffu
        : (count == 0u ? 0u : (1u << (count * 4u)) - 1u);
    packed &= mask;
    uint bytes = (packed & 0x0f0f0f0fu) + ((packed >> 4u) & 0x0f0f0f0fu);
    return (bytes & 0xffu) + ((bytes >> 8u) & 0xffu)
        + ((bytes >> 16u) & 0xffu) + ((bytes >> 24u) & 0xffu);
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
            descriptors[headerBase + checkpointWords + word], 8u
        );
    }
    uint packed = descriptors[headerBase + checkpointWords + tileWidthWord];
    offset += compactSumWidthNibbles(packed, tile & 7u);
    uint width = (packed >> ((tile & 7u) * 4u)) & 15u;
    return (offset << 5u) | width;
}

inline uint compactSampleValue(
    device const uint *payload,
    device const uint *descriptors,
    uint tileCount,
    uint scanTile,
    uint headerWordsPerPixel,
    uint headerEncoding,
    uint pixel,
    uint scan
) {
    uint descriptor = compactDescriptorFor(
        descriptors,
        tileCount,
        headerWordsPerPixel,
        headerEncoding,
        pixel,
        scan / scanTile
    );
    uint width = descriptor & 31u;
    if (width == 0u) return 0u;
    uint bit = (scan % scanTile) * width;
    uint index = (descriptor >> 5u) + bit / 32u;
    uint shift = bit % 32u;
    uint value = payload[index] >> shift;
    if (shift + width > 32u) {
        value |= payload[index + 1u] << (32u - shift);
    }
    return value & ((1u << width) - 1u);
}

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
            parameters.scan
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
                scan
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
            uint value = compactSampleValue(
                payload,
                descriptors,
                parameters.tileCount,
                parameters.scanTile,
                parameters.headerWordsPerPixel,
                parameters.headerEncoding,
                entry.pixel,
                scan
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
            : previous[parameters.outputOffset + scan];
        for (uint part = 0u; part < lanesPerScan; ++part) {
            output += partials[scanLane + part * scansPerGroup];
        }
        next[parameters.outputOffset + scan] = output;
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
    for (uint scan = 0u; scan < parameters.scanCount; ++scan) {
        total += ulong(compactSampleValue(
            payload,
            descriptors,
            parameters.tileCount,
            parameters.scanTile,
            parameters.headerWordsPerPixel,
            parameters.headerEncoding,
            pixel,
            scan
        ));
    }
    detectorSum[pixel] += total;
}
