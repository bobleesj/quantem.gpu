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
