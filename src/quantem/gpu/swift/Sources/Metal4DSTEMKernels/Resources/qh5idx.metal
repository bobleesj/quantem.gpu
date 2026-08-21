#include <metal_stdlib>
using namespace metal;

// Ported from quantem.gpu.io.backends.mps.kernels/bslz4.msl. The native entry
// point consumes QH5IDX01 block metadata, so compressed bytes remain in the
// original HDF5 file instead of being repackaged into an extracted raw file.

constant uint kLZ4Threads = 32;
constant uint kInputBufferBytes = 256;
constant uint kInputPrefetchDistance = 128;
constant uint kBslz4BlockBytes = 8192;

inline void bslz4Barrier() {
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

inline ushort bslz4ReadWordDevice(const device uchar *address) {
    return ushort(address[0]) | (ushort(address[1]) << 8);
}

inline ushort bslz4ReadWordThreadgroup(threadgroup const uchar *address) {
    return ushort(address[0]) | (ushort(address[1]) << 8);
}

struct BSLZ4Token {
    uint literals;
    uint matches;
};

inline BSLZ4Token bslz4DecodeToken(uchar value) {
    return BSLZ4Token{uint((value & 0xf0) >> 4), uint(value & 0x0f)};
}

inline void bslz4LoadInput(
    threadgroup uchar *buffer,
    const device uchar *compressed,
    uint compressedLength,
    thread long &bufferOffset,
    uint threadIndex
) {
    ulong address = ulong(compressed) + ulong(bufferOffset);
    ulong aligned = (address / 8) * 8;
    bufferOffset = long(aligned - ulong(compressed));
    if (uint(bufferOffset) + kInputBufferBytes <= compressedLength) {
        const device ulong *source = (const device ulong *)(compressed + bufferOffset);
        threadgroup ulong *destination = (threadgroup ulong *)buffer;
        destination[threadIndex] = source[threadIndex];
    } else {
        for (uint index = threadIndex; index < kInputBufferBytes; index += kLZ4Threads) {
            if (uint(bufferOffset) + index < compressedLength) {
                buffer[index] = compressed[bufferOffset + index];
            }
        }
    }
    bslz4Barrier();
}

inline uint bslz4ReadLength(
    threadgroup const uchar *buffer,
    const device uchar *compressed,
    long bufferOffset,
    uint bufferEnd,
    thread uint &index
) {
    uint length = 0;
    uchar next = 0xff;
    while (next == 0xff && index < bufferEnd) {
        next = buffer[index - uint(bufferOffset)];
        index++;
        length += next;
    }
    while (next == 0xff) {
        next = compressed[index];
        index++;
        length += next;
    }
    return length;
}

// The native detector is uint16 bitshuffle/LZ4 on disk, but its audited
// scientific range fits exactly in uint8 after masking the detector sentinels.
// These copies keep one decompressed 8192-byte bitshuffle block in threadgroup
// memory so the fused kernel below can unshuffle it without a device-memory
// round trip through a full uint16 frame.
inline void bslz4CopyDeviceToThreadgroup(
    threadgroup uchar *destination,
    const device uchar *source,
    uint length,
    uint threadIndex
) {
    for (uint index = threadIndex; index < length; index += kLZ4Threads) {
        destination[index] = source[index];
    }
}

inline void bslz4CopyRepeatDeviceToThreadgroup(
    threadgroup uchar *destination,
    const device uchar *source,
    uint distance,
    uint length,
    uint threadIndex
) {
    uint repeatMask = distance - 1u;
    if (distance == 2u) {
        // A 32-lane stride preserves lane parity, so every write by this lane
        // uses the same one of the two stable history bytes. Load it once.
        uchar repeated = source[threadIndex & 1u];
        for (uint index = threadIndex; index < length; index += kLZ4Threads) {
            destination[index] = repeated;
        }
    } else if (distance != 0u && (distance & repeatMask) == 0u) {
        for (uint index = threadIndex; index < length; index += kLZ4Threads) {
            destination[index] = source[index & repeatMask];
        }
    } else {
        for (uint index = threadIndex; index < length; index += kLZ4Threads) {
            destination[index] = source[index % distance];
        }
    }
}

inline void bslz4CopyThreadgroupToThreadgroup(
    threadgroup uchar *destination,
    threadgroup const uchar *source,
    uint length,
    uint threadIndex
) {
    for (uint index = threadIndex; index < length; index += kLZ4Threads) {
        destination[index] = source[index];
    }
}

inline void bslz4CopyRepeatThreadgroupToThreadgroup(
    threadgroup uchar *destination,
    threadgroup const uchar *source,
    uint distance,
    uint length,
    uint threadIndex
) {
    // Every lane reads only the already-produced history window. Keep this
    // byte-addressed: packed threadgroup stores were not reliably aligned on
    // all Apple GPU generations and corrupted rare legal LZ4 repeats.
    uint repeatMask = distance - 1u;
    if (distance == 2u) {
        // A 32-lane stride preserves lane parity, so every write by this lane
        // uses the same one of the two stable history bytes. Load it once.
        uchar repeated = source[threadIndex & 1u];
        for (uint index = threadIndex; index < length; index += kLZ4Threads) {
            destination[index] = repeated;
        }
    } else if (distance != 0u && (distance & repeatMask) == 0u) {
        for (uint index = threadIndex; index < length; index += kLZ4Threads) {
            destination[index] = source[index & repeatMask];
        }
    } else {
        for (uint index = threadIndex; index < length; index += kLZ4Threads) {
            destination[index] = source[index % distance];
        }
    }
}

inline void bslz4CopyOverlapThreadgroupToThreadgroup(
    threadgroup uchar *destination,
    threadgroup const uchar *source,
    uint distance,
    uint length,
    uint threadIndex
) {
    if (distance < length) {
        bslz4CopyRepeatThreadgroupToThreadgroup(
            destination, source, distance, length, threadIndex
        );
    } else {
        bslz4CopyThreadgroupToThreadgroup(
            destination, source, length, threadIndex
        );
    }
}

inline void bslz4DecompressStreamToThreadgroup(
    threadgroup uchar *inputCache,
    threadgroup uchar *decompressed,
    const device uchar *compressed,
    uint compressedLength,
    uint threadIndex
) {
    long bufferOffset = 0;
    bslz4LoadInput(
        inputCache, compressed, compressedLength, bufferOffset, threadIndex
    );
    uint outputIndex = 0;
    uint inputIndex = 0;
    while (inputIndex < compressedLength) {
        uint bufferEnd = uint(bufferOffset) + kInputBufferBytes;
        if (inputIndex + kInputPrefetchDistance > bufferEnd) {
            bufferOffset = long(inputIndex);
            bslz4LoadInput(
                inputCache, compressed, compressedLength, bufferOffset, threadIndex
            );
        }
        bufferEnd = uint(bufferOffset) + kInputBufferBytes;
        BSLZ4Token token = bslz4DecodeToken(
            inputCache[inputIndex - uint(bufferOffset)]
        );
        inputIndex++;
        uint literalCount = token.literals;
        if (token.literals == 15) {
            literalCount += bslz4ReadLength(
                inputCache, compressed, bufferOffset, bufferEnd, inputIndex
            );
        }
        uint literalStart = inputIndex;
        if (literalCount + inputIndex > bufferEnd) {
            bslz4CopyDeviceToThreadgroup(
                decompressed + outputIndex,
                compressed + inputIndex,
                literalCount,
                threadIndex
            );
        } else {
            bslz4CopyThreadgroupToThreadgroup(
                decompressed + outputIndex,
                inputCache + (inputIndex - uint(bufferOffset)),
                literalCount,
                threadIndex
            );
        }
        inputIndex += literalCount;
        outputIndex += literalCount;
        if (inputIndex < compressedLength) {
            ushort matchOffset;
            if (inputIndex + 2 > bufferEnd) {
                matchOffset = bslz4ReadWordDevice(compressed + inputIndex);
            } else {
                matchOffset = bslz4ReadWordThreadgroup(
                    inputCache + (inputIndex - uint(bufferOffset))
                );
            }
            inputIndex += 2;
            uint matchLength = 4 + token.matches;
            if (token.matches == 15) {
                matchLength += bslz4ReadLength(
                    inputCache, compressed, bufferOffset, bufferEnd, inputIndex
                );
            }
            if (
                matchOffset <= literalCount
                && long(literalStart) >= bufferOffset
                && literalStart + literalCount <= bufferEnd
            ) {
                bslz4CopyOverlapThreadgroupToThreadgroup(
                    decompressed + outputIndex,
                    inputCache + literalStart + literalCount - matchOffset
                        - uint(bufferOffset),
                    matchOffset,
                    matchLength,
                    threadIndex
                );
                bslz4Barrier();
            } else {
                bslz4Barrier();
                bslz4CopyOverlapThreadgroupToThreadgroup(
                    decompressed + outputIndex,
                    decompressed + outputIndex - matchOffset,
                    matchOffset,
                    matchLength,
                    threadIndex
                );
            }
            outputIndex += matchLength;
        }
    }
    bslz4Barrier();
}

// Full-precision direct-input variant. It preserves every byte of the complete
// LZ4 stream while removing the 256-byte compressed-input cache and its refill
// barriers. Matches that point into the immediately preceding literal run read
// the same compressed bytes directly; all other matches retain the exact
// dependency-safe threadgroup history copy.
inline void bslz4DecompressStreamDirectToThreadgroup(
    threadgroup uchar *decompressed,
    const device uchar *compressed,
    uint compressedLength,
    uint threadIndex
) {
    uint outputIndex = 0;
    uint inputIndex = 0;
    while (inputIndex < compressedLength) {
        BSLZ4Token token = bslz4DecodeToken(compressed[inputIndex++]);
        uint literalCount = token.literals;
        if (token.literals == 15) {
            uchar next = 0xff;
            while (next == 0xff) {
                next = compressed[inputIndex++];
                literalCount += next;
            }
        }
        uint literalStart = inputIndex;
        bslz4CopyDeviceToThreadgroup(
            decompressed + outputIndex,
            compressed + inputIndex,
            literalCount,
            threadIndex
        );
        inputIndex += literalCount;
        outputIndex += literalCount;
        if (inputIndex < compressedLength) {
            ushort matchOffset = bslz4ReadWordDevice(compressed + inputIndex);
            inputIndex += 2;
            uint matchLength = 4 + token.matches;
            if (token.matches == 15) {
                uchar next = 0xff;
                while (next == 0xff) {
                    next = compressed[inputIndex++];
                    matchLength += next;
                }
            }
            if (matchOffset <= literalCount) {
                bslz4CopyRepeatDeviceToThreadgroup(
                    decompressed + outputIndex,
                    compressed + literalStart + literalCount - matchOffset,
                    matchOffset,
                    matchLength,
                    threadIndex
                );
                bslz4Barrier();
            } else {
                bslz4Barrier();
                bslz4CopyOverlapThreadgroupToThreadgroup(
                    decompressed + outputIndex,
                    decompressed + outputIndex - matchOffset,
                    matchOffset,
                    matchLength,
                    threadIndex
                );
            }
            outputIndex += matchLength;
        }
    }
    bslz4Barrier();
}

// Decode only a proven prefix of the LZ4 output. This is used after an exact
// source-identity-bound audit has established that omitted uint16 high planes
// contain no valid detector values. Boundary-crossing tokens are clipped
// without changing any byte before the prefix limit.
inline void bslz4DecompressPrefixToThreadgroup(
    threadgroup uchar *inputCache,
    threadgroup uchar *decompressed,
    const device uchar *compressed,
    uint compressedLength,
    uint outputLimit,
    uint threadIndex
) {
    long bufferOffset = 0;
    bslz4LoadInput(
        inputCache, compressed, compressedLength, bufferOffset, threadIndex
    );
    uint outputIndex = 0;
    uint inputIndex = 0;
    while (inputIndex < compressedLength && outputIndex < outputLimit) {
        uint bufferEnd = uint(bufferOffset) + kInputBufferBytes;
        if (inputIndex + kInputPrefetchDistance > bufferEnd) {
            bufferOffset = long(inputIndex);
            bslz4LoadInput(
                inputCache, compressed, compressedLength, bufferOffset, threadIndex
            );
        }
        bufferEnd = uint(bufferOffset) + kInputBufferBytes;
        BSLZ4Token token = bslz4DecodeToken(
            inputCache[inputIndex - uint(bufferOffset)]
        );
        inputIndex++;
        uint literalCount = token.literals;
        if (token.literals == 15) {
            literalCount += bslz4ReadLength(
                inputCache, compressed, bufferOffset, bufferEnd, inputIndex
            );
        }
        uint literalStart = inputIndex;
        uint literalOutput = min(literalCount, outputLimit - outputIndex);
        if (literalOutput + inputIndex > bufferEnd) {
            bslz4CopyDeviceToThreadgroup(
                decompressed + outputIndex,
                compressed + inputIndex,
                literalOutput,
                threadIndex
            );
        } else {
            bslz4CopyThreadgroupToThreadgroup(
                decompressed + outputIndex,
                inputCache + (inputIndex - uint(bufferOffset)),
                literalOutput,
                threadIndex
            );
        }
        inputIndex += literalCount;
        outputIndex += literalCount;
        if (inputIndex >= compressedLength || outputIndex >= outputLimit) break;

        ushort matchOffset;
        if (inputIndex + 2 > bufferEnd) {
            matchOffset = bslz4ReadWordDevice(compressed + inputIndex);
        } else {
            matchOffset = bslz4ReadWordThreadgroup(
                inputCache + (inputIndex - uint(bufferOffset))
            );
        }
        inputIndex += 2;
        uint matchLength = 4 + token.matches;
        if (token.matches == 15) {
            matchLength += bslz4ReadLength(
                inputCache, compressed, bufferOffset, bufferEnd, inputIndex
            );
        }
        uint matchOutput = min(matchLength, outputLimit - outputIndex);
        if (
            matchOffset <= literalCount
            && long(literalStart) >= bufferOffset
            && literalStart + literalCount <= bufferEnd
        ) {
            bslz4CopyOverlapThreadgroupToThreadgroup(
                decompressed + outputIndex,
                inputCache + literalStart + literalCount - matchOffset
                    - uint(bufferOffset),
                matchOffset,
                matchOutput,
                threadIndex
            );
            bslz4Barrier();
        } else {
            bslz4Barrier();
            bslz4CopyOverlapThreadgroupToThreadgroup(
                decompressed + outputIndex,
                decompressed + outputIndex - matchOffset,
                matchOffset,
                matchOutput,
                threadIndex
            );
        }
        outputIndex += matchLength;
    }
    bslz4Barrier();
}

// Direct-input variant for the audited low-plane path. It removes the input
// cache and its refill barriers, leaving exactly one 4 KiB decoded tile in
// threadgroup memory so constrained Apple GPUs can maximize resident blocks.
inline void bslz4DecompressPrefixDirectToThreadgroup(
    threadgroup uchar *decompressed,
    const device uchar *compressed,
    uint compressedLength,
    uint outputLimit,
    uint threadIndex
) {
    uint outputIndex = 0;
    uint inputIndex = 0;
    while (inputIndex < compressedLength && outputIndex < outputLimit) {
        BSLZ4Token token = bslz4DecodeToken(compressed[inputIndex++]);
        uint literalCount = token.literals;
        if (token.literals == 15) {
            uchar next = 0xff;
            while (next == 0xff) {
                next = compressed[inputIndex++];
                literalCount += next;
            }
        }
        uint literalOutput = min(literalCount, outputLimit - outputIndex);
        bslz4CopyDeviceToThreadgroup(
            decompressed + outputIndex,
            compressed + inputIndex,
            literalOutput,
            threadIndex
        );
        inputIndex += literalCount;
        outputIndex += literalCount;
        if (inputIndex >= compressedLength || outputIndex >= outputLimit) break;

        ushort matchOffset = bslz4ReadWordDevice(compressed + inputIndex);
        inputIndex += 2;
        uint matchLength = 4 + token.matches;
        if (token.matches == 15) {
            uchar next = 0xff;
            while (next == 0xff) {
                next = compressed[inputIndex++];
                matchLength += next;
            }
        }
        uint matchOutput = min(matchLength, outputLimit - outputIndex);
        bslz4Barrier();
        bslz4CopyOverlapThreadgroupToThreadgroup(
            decompressed + outputIndex,
            decompressed + outputIndex - matchOffset,
            matchOffset,
            matchOutput,
            threadIndex
        );
        outputIndex += matchLength;
    }
    bslz4Barrier();
}

kernel void h5lz4dc_unshuffle_source_u8_qh5idx(
    const device uchar *h5File [[buffer(0)]],
    const device uint2 *blockMetadata [[buffer(1)]],
    constant ulong &rangeStart [[buffer(2)]],
    constant uint &blocksPerFrame [[buffer(3)]],
    constant uint &frameElements [[buffer(4)]],
    device uchar *output [[buffer(5)]],
    constant uint &metadataFrameOffset [[buffer(6)]],
    const device uchar *badPixelMask [[buffer(7)]],
    device atomic_uint *countAudit [[buffer(8)]],
    uint3 threadgroupPosition [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]
) {
    uint frame = threadgroupPosition.x;
    uint block = threadgroupPosition.z;
    threadgroup uchar inputCache[kInputBufferBytes];
    threadgroup uchar shuffledBlock[kBslz4BlockBytes];
    if (simdgroup == 0 && block < blocksPerFrame) {
        uint2 metadata = blockMetadata[
            ulong(metadataFrameOffset + frame) * blocksPerFrame + block
        ];
        bslz4DecompressStreamToThreadgroup(
            inputCache,
            shuffledBlock,
            h5File + rangeStart + ulong(metadata.x),
            metadata.y,
            lane
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (block < blocksPerFrame) {
        const threadgroup uint *planes = (const threadgroup uint *)shuffledBlock;
        uint detectorStart = block * 8192;
        // Preserve the compile-time 256-word stride for every full block. Only
        // the final partial uint8 block uses its shorter bit-plane stride.
        if (detectorStart + 8192 <= frameElements) {
            for (uint group = simdgroup; group < 256; group += 8) {
                uchar value = 0;
                for (uint bit = 0; bit < 8; ++bit) {
                    if (planes[bit * 256 + group] & (1u << lane)) {
                        value |= uchar(1u << bit);
                    }
                }
                uint detectorIndex = detectorStart + group * 32 + lane;
                bool valid = badPixelMask[detectorIndex] == 0;
                if (valid) {
                    atomic_fetch_max_explicit(
                        &countAudit[0], uint(value), memory_order_relaxed
                    );
                }
                output[ulong(frame) * frameElements + detectorIndex] =
                    valid ? value : uchar(0);
            }
        } else {
            uint groups = (frameElements - detectorStart) / 32;
            for (uint group = simdgroup; group < groups; group += 8) {
                uchar value = 0;
                for (uint bit = 0; bit < 8; ++bit) {
                    if (planes[bit * groups + group] & (1u << lane)) {
                        value |= uchar(1u << bit);
                    }
                }
                uint detectorIndex = detectorStart + group * 32 + lane;
                bool valid = badPixelMask[detectorIndex] == 0;
                if (valid) {
                    atomic_fetch_max_explicit(
                        &countAudit[0], uint(value), memory_order_relaxed
                    );
                }
                output[ulong(frame) * frameElements + detectorIndex] =
                    valid ? value : uchar(0);
            }
        }
    }
}

// Native-precision path with a fused exact count-range audit. One global
// atomic pair is emitted per frame/block pair rather than per detector value.
kernel void h5lz4dc_unshuffle_u16_qh5idx(
    const device uchar *h5File [[buffer(0)]],
    const device uint2 *blockMetadata [[buffer(1)]],
    constant ulong &rangeStart [[buffer(2)]],
    constant uint &blocksPerFrame [[buffer(3)]],
    constant uint &frameElements [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    constant uint &metadataFrameOffset [[buffer(6)]],
    const device uchar *badPixelMask [[buffer(7)]],
    device atomic_uint *countAudit [[buffer(8)]],
    constant uint &globalFrameOffset [[buffer(9)]],
    uint3 threadgroupPosition [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]
) {
    uint frame = threadgroupPosition.x;
    uint blockSlot = simdgroup / 4;
    uint block = threadgroupPosition.z * 2 + blockSlot;
    threadgroup uchar inputCache[kInputBufferBytes * 2];
    threadgroup uchar shuffledBlocks[kBslz4BlockBytes * 2];
    threadgroup atomic_uint batchMax;
    threadgroup atomic_uint batchAbove255;
    if (simdgroup == 0 && lane == 0) {
        atomic_store_explicit(&batchMax, 0u, memory_order_relaxed);
        atomic_store_explicit(&batchAbove255, 0u, memory_order_relaxed);
    }

    if ((simdgroup % 4) == 0 && block < blocksPerFrame) {
        uint2 metadata = blockMetadata[
            ulong(metadataFrameOffset + frame) * blocksPerFrame + block
        ];
        bslz4DecompressStreamToThreadgroup(
            inputCache + blockSlot * kInputBufferBytes,
            shuffledBlocks + blockSlot * kBslz4BlockBytes,
            h5File + rangeStart + ulong(metadata.x),
            metadata.y,
            lane
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint localMax = 0;
    uint localAbove255 = 0;

    if (block < blocksPerFrame) {
        const threadgroup uint *planes = (const threadgroup uint *)(
            shuffledBlocks + blockSlot * kBslz4BlockBytes
        );
        for (uint group = simdgroup % 4; group < 128; group += 4) {
            ushort value = 0;
            for (uint bit = 0; bit < 16; bit++) {
                if (planes[bit * 128 + group] & (1u << lane)) {
                    value |= ushort(1u << bit);
                }
            }
            uint detectorIndex = block * 4096 + group * 32 + lane;
            if (detectorIndex < frameElements) {
                ushort stored = badPixelMask[detectorIndex] ? ushort(0) : value;
                output[ulong(frame) * frameElements + detectorIndex] = stored;
                localMax = max(localMax, uint(stored));
                localAbove255 += stored > ushort(255) ? 1u : 0u;
            }
        }
        atomic_fetch_max_explicit(&batchMax, localMax, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &batchAbove255, localAbove255, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0 && lane == 0) {
        uint frameAudit = 2u * (globalFrameOffset + frame);
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&batchMax, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &countAudit[frameAudit + 1u],
            atomic_load_explicit(&batchAbove255, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

// One compressed bitshuffle block per 128-thread threadgroup. The two-block
// kernel above uses fewer threadgroups, but its 16.5 KiB threadgroup-memory
// footprint limits concurrent groups on smaller Apple GPUs. This topology uses
// 8 KiB and therefore exposes more independent LZ4 streams while
// preserving the identical uint16 output and exact count audit.
kernel void h5lz4dc_unshuffle_u16_single_block_qh5idx(
    const device uchar *h5File [[buffer(0)]],
    const device uint2 *blockMetadata [[buffer(1)]],
    constant ulong &rangeStart [[buffer(2)]],
    constant uint &blocksPerFrame [[buffer(3)]],
    constant uint &frameElements [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    constant uint &metadataFrameOffset [[buffer(6)]],
    const device uchar *badPixelMask [[buffer(7)]],
    device atomic_uint *countAudit [[buffer(8)]],
    constant uint &globalFrameOffset [[buffer(9)]],
    uint3 threadgroupPosition [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]
) {
    uint frame = threadgroupPosition.x;
    uint block = threadgroupPosition.z;
    threadgroup uchar shuffledBlock[kBslz4BlockBytes];
    threadgroup atomic_uint blockMax;
    threadgroup atomic_uint blockAbove255;
    if (simdgroup == 0 && lane == 0) {
        atomic_store_explicit(&blockMax, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockAbove255, 0u, memory_order_relaxed);
    }
    if (simdgroup == 0 && block < blocksPerFrame) {
        uint2 metadata = blockMetadata[
            ulong(metadataFrameOffset + frame) * blocksPerFrame + block
        ];
        bslz4DecompressStreamDirectToThreadgroup(
            shuffledBlock,
            h5File + rangeStart + ulong(metadata.x),
            metadata.y,
            lane
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localMax = 0;
    uint localAbove255 = 0;
    if (block < blocksPerFrame) {
        const threadgroup uint *planes =
            (const threadgroup uint *)shuffledBlock;
        for (uint group = simdgroup; group < 128; group += 4) {
            ushort value = 0;
            for (uint bit = 0; bit < 16; bit++) {
                if (planes[bit * 128 + group] & (1u << lane)) {
                    value |= ushort(1u << bit);
                }
            }
            uint detectorIndex = block * 4096 + group * 32 + lane;
            if (detectorIndex < frameElements) {
                ushort stored = badPixelMask[detectorIndex] ? ushort(0) : value;
                output[ulong(frame) * frameElements + detectorIndex] = stored;
                localMax = max(localMax, uint(stored));
                localAbove255 += stored > ushort(255) ? 1u : 0u;
            }
        }
        atomic_fetch_max_explicit(&blockMax, localMax, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &blockAbove255, localAbove255, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0 && lane == 0) {
        uint frameAudit = 2u * (globalFrameOffset + frame);
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&blockMax, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &countAudit[frameAudit + 1u],
            atomic_load_explicit(&blockAbove255, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

// Decode a uint16 bitshuffle/LZ4 source into a compact uint8 staging buffer
// while auditing every omitted high bit. The caller may use this output only
// when the exact `above255` audit is zero; otherwise it must rerun the native
// uint16 path. This is an internal staging optimization, never a dtype claim.
kernel void h5lz4dc_unshuffle_u16_lossless_u8_qh5idx(
    const device uchar *h5File [[buffer(0)]],
    const device uint2 *blockMetadata [[buffer(1)]],
    constant ulong &rangeStart [[buffer(2)]],
    constant uint &blocksPerFrame [[buffer(3)]],
    constant uint &frameElements [[buffer(4)]],
    device uchar *output [[buffer(5)]],
    constant uint &metadataFrameOffset [[buffer(6)]],
    const device uchar *badPixelMask [[buffer(7)]],
    device atomic_uint *countAudit [[buffer(8)]],
    constant uint &globalFrameOffset [[buffer(9)]],
    uint3 threadgroupPosition [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]
) {
    uint frame = threadgroupPosition.x;
    uint block = threadgroupPosition.z;
    threadgroup uchar inputCache[kInputBufferBytes];
    threadgroup uchar shuffledBlock[kBslz4BlockBytes];
    threadgroup atomic_uint blockMax;
    threadgroup atomic_uint blockAbove255;
    if (simdgroup == 0 && lane == 0) {
        atomic_store_explicit(&blockMax, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockAbove255, 0u, memory_order_relaxed);
    }
    if (simdgroup == 0 && block < blocksPerFrame) {
        uint2 metadata = blockMetadata[
            ulong(metadataFrameOffset + frame) * blocksPerFrame + block
        ];
        bslz4DecompressStreamToThreadgroup(
            inputCache,
            shuffledBlock,
            h5File + rangeStart + ulong(metadata.x),
            metadata.y,
            lane
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localMax = 0;
    uint localAbove255 = 0;
    if (block < blocksPerFrame) {
        const threadgroup uint *planes =
            (const threadgroup uint *)shuffledBlock;
        for (uint group = simdgroup; group < 128; group += 4) {
            uchar value = 0;
            for (uint bit = 0; bit < 8; bit++) {
                if (planes[bit * 128 + group] & (1u << lane)) {
                    value |= uchar(1u << bit);
                }
            }
            uint highWord = 0;
            for (uint bit = 8; bit < 16; bit++) {
                highWord |= planes[bit * 128 + group];
            }
            uint detectorIndex = block * 4096 + group * 32 + lane;
            if (detectorIndex < frameElements) {
                bool valid = badPixelMask[detectorIndex] == 0;
                uchar stored = valid ? value : uchar(0);
                output[ulong(frame) * frameElements + detectorIndex] = stored;
                localMax = max(localMax, uint(stored));
                localAbove255 +=
                    valid && (highWord & (1u << lane)) != 0u ? 1u : 0u;
            }
        }
        atomic_fetch_max_explicit(&blockMax, localMax, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &blockAbove255, localAbove255, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0 && lane == 0) {
        uint frameAudit = 2u * (globalFrameOffset + frame);
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&blockMax, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &countAudit[frameAudit + 1u],
            atomic_load_explicit(&blockAbove255, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

// Compact path for a source whose valid high planes were already audited
// against the same source identity and bad-pixel mask. Decode the complete LZ4
// block before selecting its low bitshuffle planes: stopping at the plane
// boundary is not byte-exact for every legal overlapping LZ4 match sequence.
// This entry point requires the durable audit contract and skips only the
// high-plane unshuffle/count work.
kernel void h5lz4dc_unshuffle_u16_audited_low8_qh5idx(
    const device uchar *h5File [[buffer(0)]],
    const device uint2 *blockMetadata [[buffer(1)]],
    constant ulong &rangeStart [[buffer(2)]],
    constant uint &blocksPerFrame [[buffer(3)]],
    constant uint &frameElements [[buffer(4)]],
    device uchar *output [[buffer(5)]],
    constant uint &metadataFrameOffset [[buffer(6)]],
    const device uchar *badPixelMask [[buffer(7)]],
    device atomic_uint *countAudit [[buffer(8)]],
    constant uint &globalFrameOffset [[buffer(9)]],
    uint3 threadgroupPosition [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]
) {
    uint frame = threadgroupPosition.x;
    uint block = threadgroupPosition.z;
    threadgroup uchar inputCache[kInputBufferBytes];
    threadgroup uchar shuffledBlock[kBslz4BlockBytes];
    threadgroup atomic_uint blockMax;
    if (simdgroup == 0 && lane == 0) {
        atomic_store_explicit(&blockMax, 0u, memory_order_relaxed);
    }
    if (simdgroup == 0 && block < blocksPerFrame) {
        uint2 metadata = blockMetadata[
            ulong(metadataFrameOffset + frame) * blocksPerFrame + block
        ];
        bslz4DecompressStreamToThreadgroup(
            inputCache,
            shuffledBlock,
            h5File + rangeStart + ulong(metadata.x),
            metadata.y,
            lane
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localMax = 0;
    if (block < blocksPerFrame) {
        const threadgroup uint *planes =
            (const threadgroup uint *)shuffledBlock;
        for (uint group = simdgroup; group < 128; group += 4) {
            uchar value = 0;
            for (uint bit = 0; bit < 8; bit++) {
                if (planes[bit * 128 + group] & (1u << lane)) {
                    value |= uchar(1u << bit);
                }
            }
            uint detectorIndex = block * 4096 + group * 32 + lane;
            if (detectorIndex < frameElements) {
                uchar stored = badPixelMask[detectorIndex] ? uchar(0) : value;
                output[ulong(frame) * frameElements + detectorIndex] = stored;
                localMax = max(localMax, uint(stored));
            }
        }
        atomic_fetch_max_explicit(&blockMax, localMax, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0 && lane == 0) {
        uint frameAudit = 2u * (globalFrameOffset + frame);
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&blockMax, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

struct QH5DirectDetectorBinParams {
    uint sourceDetectorRows;
    uint sourceDetectorColumns;
    uint outputDetectorColumns;
    uint outputScanCount;
};

struct QH5WordMajorClearParams {
    uint outputScanCount;
    uint destinationScanOffset;
    uint batchScanCount;
    uint detectorWords;
};

// Experimental synchronization-free block decoder. Each Metal thread owns one
// independent 4 KiB low-plane prefix and therefore needs no threadgroup fence
// between LZ4 tokens. The bounded batch scratch is consumed immediately by the
// GPU bin kernel below; it is never represented as a native-resolution result.
inline void bslz4DecompressPrefixSerialToDevice(
    device uchar *destination,
    const device uchar *compressed,
    uint compressedLength,
    uint outputLimit
) {
    uint outputIndex = 0u;
    uint inputIndex = 0u;
    while (inputIndex < compressedLength && outputIndex < outputLimit) {
        uchar token = compressed[inputIndex++];
        uint literalCount = uint(token >> 4u);
        if (literalCount == 15u) {
            uchar next = 0xff;
            while (next == 0xff && inputIndex < compressedLength) {
                next = compressed[inputIndex++];
                literalCount += uint(next);
            }
        }
        uint literalOutput = min(literalCount, outputLimit - outputIndex);
        uint literalWords = literalOutput / 4u;
        device packed_uchar4 *literalDestination =
            (device packed_uchar4 *)(destination + outputIndex);
        const device packed_uchar4 *literalSource =
            (const device packed_uchar4 *)(compressed + inputIndex);
        for (uint index = 0u; index < literalWords; ++index) {
            literalDestination[index] = literalSource[index];
        }
        for (uint index = literalWords * 4u; index < literalOutput; ++index) {
            destination[outputIndex + index] = compressed[inputIndex + index];
        }
        inputIndex += literalCount;
        outputIndex += literalCount;
        if (inputIndex >= compressedLength || outputIndex >= outputLimit) break;

        uint matchOffset = uint(bslz4ReadWordDevice(compressed + inputIndex));
        inputIndex += 2u;
        uint matchLength = 4u + uint(token & 15u);
        if ((token & 15u) == 15u) {
            uchar next = 0xff;
            while (next == 0xff && inputIndex < compressedLength) {
                next = compressed[inputIndex++];
                matchLength += uint(next);
            }
        }
        uint matchOutput = min(matchLength, outputLimit - outputIndex);
        if (matchOffset == 1u) {
            uchar value = destination[outputIndex - 1u];
            packed_uchar4 pattern(value, value, value, value);
            uint words = matchOutput / 4u;
            device packed_uchar4 *wordDestination =
                (device packed_uchar4 *)(destination + outputIndex);
            uint index = 0u;
            for (; index + 3u < words; index += 4u) {
                wordDestination[index] = pattern;
                wordDestination[index + 1u] = pattern;
                wordDestination[index + 2u] = pattern;
                wordDestination[index + 3u] = pattern;
            }
            for (; index < words; ++index) {
                wordDestination[index] = pattern;
            }
            for (uint index = words * 4u; index < matchOutput; ++index) {
                destination[outputIndex + index] = value;
            }
        } else if (matchOffset == 2u) {
            uchar first = destination[outputIndex - 2u];
            uchar second = destination[outputIndex - 1u];
            packed_uchar4 pattern(first, second, first, second);
            uint words = matchOutput / 4u;
            device packed_uchar4 *wordDestination =
                (device packed_uchar4 *)(destination + outputIndex);
            uint index = 0u;
            for (; index + 3u < words; index += 4u) {
                wordDestination[index] = pattern;
                wordDestination[index + 1u] = pattern;
                wordDestination[index + 2u] = pattern;
                wordDestination[index + 3u] = pattern;
            }
            for (; index < words; ++index) {
                wordDestination[index] = pattern;
            }
            for (uint index = words * 4u; index < matchOutput; ++index) {
                destination[outputIndex + index] =
                    (index & 1u) == 0u ? first : second;
            }
        } else if (matchOffset >= 16u) {
            uint words = matchOutput / 4u;
            device packed_uchar4 *wordDestination =
                (device packed_uchar4 *)(destination + outputIndex);
            const device packed_uchar4 *wordSource =
                (const device packed_uchar4 *)(
                    destination + outputIndex - matchOffset
                );
            uint index = 0u;
            for (; index + 3u < words; index += 4u) {
                wordDestination[index] = wordSource[index];
                wordDestination[index + 1u] = wordSource[index + 1u];
                wordDestination[index + 2u] = wordSource[index + 2u];
                wordDestination[index + 3u] = wordSource[index + 3u];
            }
            for (; index < words; ++index) {
                wordDestination[index] = wordSource[index];
            }
            for (uint index = words * 4u; index < matchOutput; ++index) {
                destination[outputIndex + index] =
                    destination[outputIndex + index - matchOffset];
            }
        } else if (matchOffset >= 4u) {
            uint words = matchOutput / 4u;
            device packed_uchar4 *wordDestination =
                (device packed_uchar4 *)(destination + outputIndex);
            const device packed_uchar4 *wordSource =
                (const device packed_uchar4 *)(
                    destination + outputIndex - matchOffset
                );
            for (uint index = 0u; index < words; ++index) {
                wordDestination[index] = wordSource[index];
            }
            for (uint index = words * 4u; index < matchOutput; ++index) {
                destination[outputIndex + index] =
                    destination[outputIndex + index - matchOffset];
            }
        } else {
            for (uint index = 0u; index < matchOutput; ++index) {
                destination[outputIndex + index] =
                    destination[outputIndex + index - matchOffset];
            }
        }
        outputIndex += matchLength;
    }
}

kernel void h5lz4dc_u16_audited_low8_scalar_qh5idx(
    const device uchar *h5File [[buffer(0)]],
    const device uint2 *blockMetadata [[buffer(1)]],
    constant ulong &rangeStart [[buffer(2)]],
    constant uint &blocksPerFrame [[buffer(3)]],
    constant uint &frameElements [[buffer(4)]],
    device uchar *lowPlaneScratch [[buffer(5)]],
    constant uint &metadataFrameOffset [[buffer(6)]],
    uint linearBlock [[thread_position_in_grid]]
) {
    uint frame = linearBlock / blocksPerFrame;
    uint block = linearBlock - frame * blocksPerFrame;
    uint2 metadata = blockMetadata[
        ulong(metadataFrameOffset + frame) * blocksPerFrame + block
    ];
    uint blockStart = block * 4096u;
    uint outputLimit = min(4096u, frameElements - blockStart);
    bslz4DecompressPrefixSerialToDevice(
        lowPlaneScratch + ulong(frame) * frameElements + blockStart,
        h5File + rangeStart + ulong(metadata.x),
        metadata.y,
        outputLimit
    );
}

kernel void clear_u16_word_major_range_qh5idx(
    device atomic_uint *output [[buffer(0)]],
    constant QH5WordMajorClearParams &params [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    if (position.x >= params.detectorWords
        || position.y >= params.batchScanCount) return;
    ulong index = ulong(position.x) * params.outputScanCount
        + params.destinationScanOffset + position.y;
    atomic_store_explicit(&output[index], 0u, memory_order_relaxed);
}

// Fused path for an identity-audited uint16 source loaded as exact detector
// 4x4 sums. Each compressed bitshuffle block is decoded once and contributes
// directly to packed uint16 detector-word-major storage. The destination must
// be zero-filled before dispatch. Atomic additions are exact because the
// caller separately proves that every completed 4x4 sum fits uint16.
kernel void h5lz4dc_unshuffle_u16_audited_low8_bin4_u16_word_major_qh5idx(
    const device uchar *h5File [[buffer(0)]],
    const device uint2 *blockMetadata [[buffer(1)]],
    constant ulong &rangeStart [[buffer(2)]],
    constant uint &blocksPerFrame [[buffer(3)]],
    constant uint &frameElements [[buffer(4)]],
    device atomic_uint *output [[buffer(5)]],
    constant uint &metadataFrameOffset [[buffer(6)]],
    const device uchar *badPixelMask [[buffer(7)]],
    device atomic_uint *countAudit [[buffer(8)]],
    constant uint &globalFrameOffset [[buffer(9)]],
    constant QH5DirectDetectorBinParams &params [[buffer(10)]],
    uint3 threadgroupPosition [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]
) {
    uint linearBlock = threadgroupPosition.x;
    uint frame = linearBlock / blocksPerFrame;
    uint block = linearBlock - frame * blocksPerFrame;
    threadgroup uchar lowPlanes[kBslz4BlockBytes / 2];
    threadgroup atomic_uint blockMax;
    if (threadIndex == 0) {
        atomic_store_explicit(&blockMax, 0u, memory_order_relaxed);
    }
    if (simdgroup == 0 && block < blocksPerFrame) {
        uint2 metadata = blockMetadata[
            ulong(metadataFrameOffset + frame) * blocksPerFrame + block
        ];
        bslz4DecompressPrefixDirectToThreadgroup(
            lowPlanes,
            h5File + rangeStart + ulong(metadata.x),
            metadata.y,
            kBslz4BlockBytes / 2,
            lane
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint blockStart = block * 4096u;
    uint blockStop = min(frameElements, blockStart + 4096u);
    uint firstDetectorRow = blockStart / params.sourceDetectorColumns;
    uint lastDetectorRow = (max(blockStart + 1u, blockStop) - 1u)
        / params.sourceDetectorColumns;
    uint firstOutputRow = firstDetectorRow / 4u;
    uint lastOutputRow = lastDetectorRow / 4u;
    uint candidateCount =
        (lastOutputRow - firstOutputRow + 1u) * params.outputDetectorColumns;
    uint localMax = 0u;
    for (uint candidate = threadIndex; candidate < candidateCount; candidate += 128u) {
        uint outputRow = firstOutputRow + candidate / params.outputDetectorColumns;
        uint outputColumn = candidate % params.outputDetectorColumns;
        uint sum = 0u;
        for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
            uint sourceRow = outputRow * 4u + rowOffset;
            if (sourceRow >= params.sourceDetectorRows) continue;
            for (uint columnOffset = 0u; columnOffset < 4u; ++columnOffset) {
                uint sourceColumn = outputColumn * 4u + columnOffset;
                if (sourceColumn >= params.sourceDetectorColumns) continue;
                uint sourcePixel =
                    sourceRow * params.sourceDetectorColumns + sourceColumn;
                if (sourcePixel < blockStart || sourcePixel >= blockStop
                    || badPixelMask[sourcePixel]) continue;
                uint blockPixel = sourcePixel - blockStart;
                uint group = blockPixel >> 5u;
                uint bitMask = 1u << (blockPixel & 31u);
                const threadgroup uint *planes =
                    (const threadgroup uint *)lowPlanes;
                uint value = 0u;
                for (uint bit = 0u; bit < 8u; ++bit) {
                    if (planes[bit * 128u + group] & bitMask) {
                        value |= 1u << bit;
                    }
                }
                sum += value;
                localMax = max(localMax, value);
            }
        }
        if (sum != 0u) {
            uint outputPixel = outputRow * params.outputDetectorColumns + outputColumn;
            ulong wordIndex =
                ulong(outputPixel >> 1u) * params.outputScanCount
                + globalFrameOffset + frame;
            uint packedContribution =
                (outputPixel & 1u) == 0u ? sum : (sum << 16u);
            atomic_fetch_add_explicit(
                &output[wordIndex], packedContribution, memory_order_relaxed
            );
        }
    }
    localMax = simd_max(localMax);
    if (lane == 0) {
        atomic_fetch_max_explicit(&blockMax, localMax, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIndex == 0) {
        uint frameAudit = 2u * (globalFrameOffset + frame);
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&blockMax, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

// Companion to the scalar decoder. It consumes only the transient shuffled
// low-plane batch and writes the same explicit detector-bin-4 uint16 result as
// the accepted fused kernel.
kernel void h5lz4dc_bin_u16_audited_low8_scalar_u16_word_major_qh5idx(
    const device uchar *lowPlaneScratch [[buffer(0)]],
    device atomic_uint *output [[buffer(1)]],
    const device uchar *badPixelMask [[buffer(2)]],
    device atomic_uint *countAudit [[buffer(3)]],
    constant uint &globalFrameOffset [[buffer(4)]],
    constant uint &blocksPerFrame [[buffer(5)]],
    constant uint &frameElements [[buffer(6)]],
    constant QH5DirectDetectorBinParams &params [[buffer(7)]],
    const device uchar *detectorBands [[buffer(8)]],
    device atomic_uint *bfMap [[buffer(9)]],
    device atomic_uint *abfMap [[buffer(10)]],
    device atomic_uint *dfMap [[buffer(11)]],
    device atomic_uint *comTotal [[buffer(12)]],
    device atomic_uint *comRowMoment [[buffer(13)]],
    device atomic_uint *comColumnMoment [[buffer(14)]],
    uint linearBlock [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint frame = linearBlock / blocksPerFrame;
    uint block = linearBlock - frame * blocksPerFrame;
    const device uint *planes = (const device uint *)(
        lowPlaneScratch + ulong(frame) * frameElements + block * 4096u
    );
    threadgroup atomic_uint blockMax;
    threadgroup atomic_uint blockBF;
    threadgroup atomic_uint blockABF;
    threadgroup atomic_uint blockDF;
    threadgroup atomic_uint blockTotal;
    threadgroup atomic_uint blockRowMoment;
    threadgroup atomic_uint blockColumnMoment;
    if (threadIndex == 0u) {
        atomic_store_explicit(&blockMax, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockBF, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockABF, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockDF, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockTotal, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockRowMoment, 0u, memory_order_relaxed);
        atomic_store_explicit(&blockColumnMoment, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint blockStart = block * 4096u;
    uint blockStop = min(frameElements, blockStart + 4096u);
    uint firstDetectorRow = blockStart / params.sourceDetectorColumns;
    uint lastDetectorRow = (max(blockStart + 1u, blockStop) - 1u)
        / params.sourceDetectorColumns;
    uint firstOutputRow = firstDetectorRow / 4u;
    uint lastOutputRow = lastDetectorRow / 4u;
    uint candidateCount =
        (lastOutputRow - firstOutputRow + 1u) * params.outputDetectorColumns;
    uint localMax = 0u;
    uint localBF = 0u;
    uint localABF = 0u;
    uint localDF = 0u;
    uint localTotal = 0u;
    uint localRowMoment = 0u;
    uint localColumnMoment = 0u;
    for (uint candidate = threadIndex; candidate < candidateCount; candidate += 128u) {
        uint outputRow = firstOutputRow + candidate / params.outputDetectorColumns;
        uint outputColumn = candidate % params.outputDetectorColumns;
        uint sum = 0u;
        for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
            uint sourceRow = outputRow * 4u + rowOffset;
            if (sourceRow >= params.sourceDetectorRows) continue;
            uint rowPixel = sourceRow * params.sourceDetectorColumns
                + outputColumn * 4u;
            uint sourceRowStop = (sourceRow + 1u) * params.sourceDetectorColumns;
            uint rowStop = min(rowPixel + 4u, min(blockStop, sourceRowStop));
            uint rowStart = max(rowPixel, blockStart);
            if (rowStart >= rowStop) continue;
            uint validMask = 0u;
            for (uint sourcePixel = rowStart; sourcePixel < rowStop; ++sourcePixel) {
                if (!badPixelMask[sourcePixel]) {
                    validMask |= 1u << (sourcePixel - rowStart);
                }
            }
            uint blockPixel = rowStart - blockStart;
            uint group = blockPixel >> 5u;
            uint shift = blockPixel & 31u;
            if (shift + (rowStop - rowStart) <= 32u) {
                uint packedValues = 0u;
                for (uint bit = 0u; bit < 8u; ++bit) {
                    uint nibble =
                        (planes[bit * 128u + group] >> shift) & validMask;
                    packedValues |= (nibble & 1u) << bit;
                    packedValues |= (nibble & 2u) << (bit + 7u);
                    packedValues |= (nibble & 4u) << (bit + 14u);
                    packedValues |= (nibble & 8u) << (bit + 21u);
                }
                uint value0 = packedValues & 0xffu;
                uint value1 = (packedValues >> 8u) & 0xffu;
                uint value2 = (packedValues >> 16u) & 0xffu;
                uint value3 = packedValues >> 24u;
                sum += value0 + value1 + value2 + value3;
                localMax = max(
                    localMax, max(max(value0, value1), max(value2, value3))
                );
            } else {
                for (uint sourcePixel = rowStart; sourcePixel < rowStop; ++sourcePixel) {
                    if (badPixelMask[sourcePixel]) continue;
                    uint localPixel = sourcePixel - blockStart;
                    uint localGroup = localPixel >> 5u;
                    uint bitMask = 1u << (localPixel & 31u);
                    uint value = 0u;
                    for (uint bit = 0u; bit < 8u; ++bit) {
                        if (planes[bit * 128u + localGroup] & bitMask) {
                            value |= 1u << bit;
                        }
                    }
                    sum += value;
                    localMax = max(localMax, value);
                }
            }
        }
        if (sum != 0u) {
            uint outputPixel = outputRow * params.outputDetectorColumns + outputColumn;
            ulong wordIndex =
                ulong(outputPixel >> 1u) * params.outputScanCount
                + globalFrameOffset + frame;
            uint packedContribution =
                (outputPixel & 1u) == 0u ? sum : (sum << 16u);
            atomic_fetch_add_explicit(
                &output[wordIndex], packedContribution, memory_order_relaxed
            );
            uchar bands = detectorBands[outputPixel];
            localBF += (bands & 1u) == 0u ? 0u : sum;
            localABF += (bands & 2u) == 0u ? 0u : sum;
            localDF += (bands & 4u) == 0u ? 0u : sum;
            localTotal += sum;
            localRowMoment += sum * outputRow;
            localColumnMoment += sum * outputColumn;
        }
    }
    localMax = simd_max(localMax);
    localBF = simd_sum(localBF);
    localABF = simd_sum(localABF);
    localDF = simd_sum(localDF);
    localTotal = simd_sum(localTotal);
    localRowMoment = simd_sum(localRowMoment);
    localColumnMoment = simd_sum(localColumnMoment);
    if (lane == 0u) {
        atomic_fetch_max_explicit(&blockMax, localMax, memory_order_relaxed);
        atomic_fetch_add_explicit(&blockBF, localBF, memory_order_relaxed);
        atomic_fetch_add_explicit(&blockABF, localABF, memory_order_relaxed);
        atomic_fetch_add_explicit(&blockDF, localDF, memory_order_relaxed);
        atomic_fetch_add_explicit(&blockTotal, localTotal, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &blockRowMoment, localRowMoment, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &blockColumnMoment, localColumnMoment, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIndex == 0u) {
        uint frameAudit = 2u * (globalFrameOffset + frame);
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&blockMax, memory_order_relaxed),
            memory_order_relaxed
        );
        uint outputFrame = globalFrameOffset + frame;
        atomic_fetch_add_explicit(
            &bfMap[outputFrame],
            atomic_load_explicit(&blockBF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &abfMap[outputFrame],
            atomic_load_explicit(&blockABF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &dfMap[outputFrame],
            atomic_load_explicit(&blockDF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comTotal[outputFrame],
            atomic_load_explicit(&blockTotal, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comRowMoment[outputFrame],
            atomic_load_explicit(&blockRowMoment, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comColumnMoment[outputFrame],
            atomic_load_explicit(&blockColumnMoment, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

inline uint bslz4Low8AlignedRow4(
    const device uchar *lowPlaneScratch,
    const device uchar *badPixelMask,
    uint frame,
    uint frameElements,
    uint sourcePixel,
    thread uint &localMax
) {
    uint block = sourcePixel >> 12u;
    uint localPixel = sourcePixel & 4095u;
    uint group = localPixel >> 5u;
    uint shift = localPixel & 31u;
    const device uint *planes = (const device uint *)(
        lowPlaneScratch + ulong(frame) * frameElements + block * 4096u
    );
    uint validMask = 0u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        if (!badPixelMask[sourcePixel + offset]) {
            validMask |= 1u << offset;
        }
    }
    uint packedValues = 0u;
    for (uint bit = 0u; bit < 8u; ++bit) {
        uint nibble = (planes[bit * 128u + group] >> shift) & validMask;
        packedValues |= (nibble & 1u) << bit;
        packedValues |= (nibble & 2u) << (bit + 7u);
        packedValues |= (nibble & 4u) << (bit + 14u);
        packedValues |= (nibble & 8u) << (bit + 21u);
    }
    uint value0 = packedValues & 0xffu;
    uint value1 = (packedValues >> 8u) & 0xffu;
    uint value2 = (packedValues >> 16u) & 0xffu;
    uint value3 = packedValues >> 24u;
    localMax = max(localMax, max(max(value0, value1), max(value2, value3)));
    return value0 + value1 + value2 + value3;
}

// Frame-owned detector-bin-4 path. Each thread owns one packed pair of output
// pixels and writes it exactly once, so the resident volume needs neither a
// full clear nor atomic additions. Product and audit reductions remain exact.
kernel void h5lz4dc_bin_u16_audited_low8_scalar_u16_word_major_frame_owned_qh5idx(
    const device uchar *lowPlaneScratch [[buffer(0)]],
    device uint *output [[buffer(1)]],
    const device uchar *badPixelMask [[buffer(2)]],
    device atomic_uint *countAudit [[buffer(3)]],
    constant uint &globalFrameOffset [[buffer(4)]],
    constant uint &blocksPerFrame [[buffer(5)]],
    constant uint &frameElements [[buffer(6)]],
    constant QH5DirectDetectorBinParams &params [[buffer(7)]],
    const device uchar *detectorBands [[buffer(8)]],
    device atomic_uint *bfMap [[buffer(9)]],
    device atomic_uint *abfMap [[buffer(10)]],
    device atomic_uint *dfMap [[buffer(11)]],
    device atomic_uint *comTotal [[buffer(12)]],
    device atomic_uint *comRowMoment [[buffer(13)]],
    device atomic_uint *comColumnMoment [[buffer(14)]],
    uint linearGroup [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)blocksPerFrame;
    uint outputRows = (params.sourceDetectorRows + 3u) / 4u;
    uint outputPixels = outputRows * params.outputDetectorColumns;
    uint outputWords = (outputPixels + 1u) / 2u;
    uint groupsPerFrame = (outputWords + 127u) / 128u;
    uint frame = linearGroup / groupsPerFrame;
    uint groupInFrame = linearGroup - frame * groupsPerFrame;
    uint outputWord = groupInFrame * 128u + threadIndex;

    threadgroup atomic_uint groupMax;
    threadgroup atomic_uint groupBF;
    threadgroup atomic_uint groupABF;
    threadgroup atomic_uint groupDF;
    threadgroup atomic_uint groupTotal;
    threadgroup atomic_uint groupRowMoment;
    threadgroup atomic_uint groupColumnMoment;
    if (threadIndex == 0u) {
        atomic_store_explicit(&groupMax, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupBF, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupABF, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupDF, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupTotal, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupRowMoment, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupColumnMoment, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localMax = 0u;
    uint localBF = 0u;
    uint localABF = 0u;
    uint localDF = 0u;
    uint localTotal = 0u;
    uint localRowMoment = 0u;
    uint localColumnMoment = 0u;
    uint packedOutput = 0u;
    if (outputWord < outputWords) {
        for (uint pair = 0u; pair < 2u; ++pair) {
            uint outputPixel = outputWord * 2u + pair;
            if (outputPixel >= outputPixels) continue;
            uint outputRow = outputPixel / params.outputDetectorColumns;
            uint outputColumn = outputPixel - outputRow * params.outputDetectorColumns;
            uint sum = 0u;
            for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
                uint sourceRow = outputRow * 4u + rowOffset;
                if (sourceRow >= params.sourceDetectorRows) continue;
                uint sourcePixel = sourceRow * params.sourceDetectorColumns
                    + outputColumn * 4u;
                sum += bslz4Low8AlignedRow4(
                    lowPlaneScratch,
                    badPixelMask,
                    frame,
                    frameElements,
                    sourcePixel,
                    localMax
                );
            }
            packedOutput |= pair == 0u ? sum : (sum << 16u);
            uchar bands = detectorBands[outputPixel];
            localBF += (bands & 1u) == 0u ? 0u : sum;
            localABF += (bands & 2u) == 0u ? 0u : sum;
            localDF += (bands & 4u) == 0u ? 0u : sum;
            localTotal += sum;
            localRowMoment += sum * outputRow;
            localColumnMoment += sum * outputColumn;
        }
        ulong destination = ulong(outputWord) * params.outputScanCount
            + globalFrameOffset + frame;
        output[destination] = packedOutput;
    }

    localMax = simd_max(localMax);
    localBF = simd_sum(localBF);
    localABF = simd_sum(localABF);
    localDF = simd_sum(localDF);
    localTotal = simd_sum(localTotal);
    localRowMoment = simd_sum(localRowMoment);
    localColumnMoment = simd_sum(localColumnMoment);
    if (lane == 0u) {
        atomic_fetch_max_explicit(&groupMax, localMax, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupBF, localBF, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupABF, localABF, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupDF, localDF, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupTotal, localTotal, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &groupRowMoment, localRowMoment, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &groupColumnMoment, localColumnMoment, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIndex == 0u) {
        uint outputFrame = globalFrameOffset + frame;
        uint frameAudit = 2u * outputFrame;
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&groupMax, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &bfMap[outputFrame],
            atomic_load_explicit(&groupBF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &abfMap[outputFrame],
            atomic_load_explicit(&groupABF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &dfMap[outputFrame],
            atomic_load_explicit(&groupDF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comTotal[outputFrame],
            atomic_load_explicit(&groupTotal, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comRowMoment[outputFrame],
            atomic_load_explicit(&groupRowMoment, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comColumnMoment[outputFrame],
            atomic_load_explicit(&groupColumnMoment, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

// Frame-major companion for Python MPS consumers. Each thread owns one packed
// pair of detector-bin-4 pixels and writes it exactly once. The source audit
// proves that the uint16 source has no populated high byte after masking and
// that each exact 4x4 sum fits in uint16. This keeps the public Python load
// layout (frame, detector row, detector column) without a 1.2 GB transpose.
kernel void h5lz4dc_bin_u16_audited_low8_scalar_u16_frame_major_qh5idx(
    const device uchar *lowPlaneScratch [[buffer(0)]],
    device uint *output [[buffer(1)]],
    const device uchar *badPixelMask [[buffer(2)]],
    device atomic_uint *countAudit [[buffer(3)]],
    constant uint &globalFrameOffset [[buffer(4)]],
    constant uint &frameElements [[buffer(5)]],
    constant QH5DirectDetectorBinParams &params [[buffer(6)]],
    uint linearGroup [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint outputRows = params.sourceDetectorRows / 4u;
    uint outputPixels = outputRows * params.outputDetectorColumns;
    uint outputWords = outputPixels / 2u;
    uint groupsPerFrame = (outputWords + 127u) / 128u;
    uint frame = linearGroup / groupsPerFrame;
    uint groupInFrame = linearGroup - frame * groupsPerFrame;
    uint outputWord = groupInFrame * 128u + threadIndex;

    threadgroup atomic_uint groupMax;
    if (threadIndex == 0u) {
        atomic_store_explicit(&groupMax, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localMax = 0u;
    if (outputWord < outputWords) {
        uint packedOutput = 0u;
        for (uint pair = 0u; pair < 2u; ++pair) {
            uint outputPixel = outputWord * 2u + pair;
            uint outputRow = outputPixel / params.outputDetectorColumns;
            uint outputColumn = outputPixel - outputRow * params.outputDetectorColumns;
            uint sum = 0u;
            for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
                uint sourceRow = outputRow * 4u + rowOffset;
                uint sourcePixel = sourceRow * params.sourceDetectorColumns
                    + outputColumn * 4u;
                sum += bslz4Low8AlignedRow4(
                    lowPlaneScratch,
                    badPixelMask,
                    frame,
                    frameElements,
                    sourcePixel,
                    localMax
                );
            }
            packedOutput |= pair == 0u ? sum : (sum << 16u);
        }
        ulong outputFrame = ulong(globalFrameOffset + frame);
        output[outputFrame * outputWords + outputWord] = packedOutput;
    }

    localMax = simd_max(localMax);
    if (lane == 0u) {
        atomic_fetch_max_explicit(&groupMax, localMax, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIndex == 0u) {
        atomic_fetch_max_explicit(
            &countAudit[globalFrameOffset + frame],
            atomic_load_explicit(&groupMax, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

inline uint2 bslz4Low8AlignedRow8(
    const device uchar *lowPlaneScratch,
    const device uchar *badPixelMask,
    uint frame,
    uint frameElements,
    uint sourcePixel,
    thread uint &localMax
) {
    uint block = sourcePixel >> 12u;
    uint localPixel = sourcePixel & 4095u;
    uint group = localPixel >> 5u;
    uint shift = localPixel & 31u;
    const device uint *planes = (const device uint *)(
        lowPlaneScratch + ulong(frame) * frameElements + block * 4096u
    );
    uint validMask = 0u;
    for (uint offset = 0u; offset < 8u; ++offset) {
        if (!badPixelMask[sourcePixel + offset]) {
            validMask |= 1u << offset;
        }
    }
    ulong bitMatrix = 0ul;
    for (uint bit = 0u; bit < 8u; ++bit) {
        uint byteValues = (planes[bit * 128u + group] >> shift) & validMask;
        bitMatrix |= ulong(byteValues) << (bit * 8u);
    }
    ulong swap = (bitMatrix ^ (bitMatrix >> 7u)) & 0x00AA00AA00AA00AAul;
    bitMatrix ^= swap ^ (swap << 7u);
    swap = (bitMatrix ^ (bitMatrix >> 14u)) & 0x0000CCCC0000CCCCul;
    bitMatrix ^= swap ^ (swap << 14u);
    swap = (bitMatrix ^ (bitMatrix >> 28u)) & 0x00000000F0F0F0F0ul;
    bitMatrix ^= swap ^ (swap << 28u);
    uint packedLow = uint(bitMatrix);
    uint packedHigh = uint(bitMatrix >> 32u);
    uint value0 = packedLow & 0xffu;
    uint value1 = (packedLow >> 8u) & 0xffu;
    uint value2 = (packedLow >> 16u) & 0xffu;
    uint value3 = packedLow >> 24u;
    uint value4 = packedHigh & 0xffu;
    uint value5 = (packedHigh >> 8u) & 0xffu;
    uint value6 = (packedHigh >> 16u) & 0xffu;
    uint value7 = packedHigh >> 24u;
    localMax = max(
        localMax,
        max(
            max(max(value0, value1), max(value2, value3)),
            max(max(value4, value5), max(value6, value7))
        )
    );
    return uint2(
        value0 + value1 + value2 + value3,
        value4 + value5 + value6 + value7
    );
}

// Two adjacent detector-bin-4 pixels share each source-row bit-plane load.
// The caller selects this kernel only when detector rows begin on 32-pixel
// bit-plane groups and the output detector width is even, so every row-8 read
// stays within one block word. Masking and runtime count auditing remain exact.
kernel void h5lz4dc_bin_u16_audited_low8_scalar_u16_frame_major_row8_qh5idx(
    const device uchar *lowPlaneScratch [[buffer(0)]],
    device uint *output [[buffer(1)]],
    const device uchar *badPixelMask [[buffer(2)]],
    device atomic_uint *countAudit [[buffer(3)]],
    constant uint &globalFrameOffset [[buffer(4)]],
    constant uint &frameElements [[buffer(5)]],
    constant QH5DirectDetectorBinParams &params [[buffer(6)]],
    uint linearGroup [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint outputRows = params.sourceDetectorRows / 4u;
    uint outputPixels = outputRows * params.outputDetectorColumns;
    uint outputWords = outputPixels / 2u;
    uint groupsPerFrame = (outputWords + 127u) / 128u;
    uint frame = linearGroup / groupsPerFrame;
    uint groupInFrame = linearGroup - frame * groupsPerFrame;
    uint outputWord = groupInFrame * 128u + threadIndex;

    threadgroup atomic_uint groupMax;
    if (threadIndex == 0u) {
        atomic_store_explicit(&groupMax, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localMax = 0u;
    if (outputWord < outputWords) {
        uint firstOutputPixel = outputWord * 2u;
        uint outputRow = firstOutputPixel / params.outputDetectorColumns;
        uint outputColumn = firstOutputPixel
            - outputRow * params.outputDetectorColumns;
        uint2 sums = uint2(0u);
        for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
            uint sourceRow = outputRow * 4u + rowOffset;
            uint sourcePixel = sourceRow * params.sourceDetectorColumns
                + outputColumn * 4u;
            sums += bslz4Low8AlignedRow8(
                lowPlaneScratch,
                badPixelMask,
                frame,
                frameElements,
                sourcePixel,
                localMax
            );
        }
        ulong outputFrame = ulong(globalFrameOffset + frame);
        output[outputFrame * outputWords + outputWord] = sums.x | (sums.y << 16u);
    }

    localMax = simd_max(localMax);
    if (lane == 0u) {
        atomic_fetch_max_explicit(&groupMax, localMax, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIndex == 0u) {
        atomic_fetch_max_explicit(
            &countAudit[globalFrameOffset + frame],
            atomic_load_explicit(&groupMax, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}

// Word-major companion for native consumers. One thread owns the two adjacent
// detector-bin-4 pixels packed into each destination word. When both pixels
// occupy one detector row, their eight source columns share each bit-plane
// load. The guarded scalar fallback preserves exact behavior for incomplete or
// unaligned detector geometry.
kernel void h5lz4dc_bin_u16_audited_low8_scalar_u16_word_major_frame_owned_row8_qh5idx(
    const device uchar *lowPlaneScratch [[buffer(0)]],
    device uint *output [[buffer(1)]],
    const device uchar *badPixelMask [[buffer(2)]],
    device atomic_uint *countAudit [[buffer(3)]],
    constant uint &globalFrameOffset [[buffer(4)]],
    constant uint &blocksPerFrame [[buffer(5)]],
    constant uint &frameElements [[buffer(6)]],
    constant QH5DirectDetectorBinParams &params [[buffer(7)]],
    const device uchar *detectorBands [[buffer(8)]],
    device atomic_uint *bfMap [[buffer(9)]],
    device atomic_uint *abfMap [[buffer(10)]],
    device atomic_uint *dfMap [[buffer(11)]],
    device atomic_uint *comTotal [[buffer(12)]],
    device atomic_uint *comRowMoment [[buffer(13)]],
    device atomic_uint *comColumnMoment [[buffer(14)]],
    uint linearGroup [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    (void)blocksPerFrame;
    uint outputRows = (params.sourceDetectorRows + 3u) / 4u;
    uint outputPixels = outputRows * params.outputDetectorColumns;
    uint outputWords = (outputPixels + 1u) / 2u;
    uint groupsPerFrame = (outputWords + 127u) / 128u;
    uint frame = linearGroup / groupsPerFrame;
    uint groupInFrame = linearGroup - frame * groupsPerFrame;
    uint outputWord = groupInFrame * 128u + threadIndex;

    threadgroup atomic_uint groupMax;
    threadgroup atomic_uint groupBF;
    threadgroup atomic_uint groupABF;
    threadgroup atomic_uint groupDF;
    threadgroup atomic_uint groupTotal;
    threadgroup atomic_uint groupRowMoment;
    threadgroup atomic_uint groupColumnMoment;
    if (threadIndex == 0u) {
        atomic_store_explicit(&groupMax, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupBF, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupABF, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupDF, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupTotal, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupRowMoment, 0u, memory_order_relaxed);
        atomic_store_explicit(&groupColumnMoment, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint localMax = 0u;
    uint localBF = 0u;
    uint localABF = 0u;
    uint localDF = 0u;
    uint localTotal = 0u;
    uint localRowMoment = 0u;
    uint localColumnMoment = 0u;
    if (outputWord < outputWords) {
        uint firstOutputPixel = outputWord * 2u;
        uint firstOutputRow = firstOutputPixel / params.outputDetectorColumns;
        uint firstOutputColumn = firstOutputPixel
            - firstOutputRow * params.outputDetectorColumns;
        uint2 sums = uint2(0u);
        bool completePair = firstOutputPixel + 1u < outputPixels
            && firstOutputColumn + 1u < params.outputDetectorColumns
            && firstOutputRow * 4u + 3u < params.sourceDetectorRows
            && firstOutputColumn * 4u + 7u < params.sourceDetectorColumns;
        if (completePair) {
            bool aligned = true;
            for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
                uint sourceRow = firstOutputRow * 4u + rowOffset;
                uint sourcePixel = sourceRow * params.sourceDetectorColumns
                    + firstOutputColumn * 4u;
                uint localPixel = sourcePixel & 4095u;
                aligned = aligned
                    && (sourcePixel & 31u) <= 24u
                    && localPixel <= 4088u;
            }
            if (aligned) {
                for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
                    uint sourceRow = firstOutputRow * 4u + rowOffset;
                    uint sourcePixel = sourceRow * params.sourceDetectorColumns
                        + firstOutputColumn * 4u;
                    sums += bslz4Low8AlignedRow8(
                        lowPlaneScratch,
                        badPixelMask,
                        frame,
                        frameElements,
                        sourcePixel,
                        localMax
                    );
                }
            } else {
                completePair = false;
            }
        }
        if (!completePair) {
            sums = uint2(0u);
            for (uint pair = 0u; pair < 2u; ++pair) {
                uint outputPixel = firstOutputPixel + pair;
                if (outputPixel >= outputPixels) continue;
                uint outputRow = outputPixel / params.outputDetectorColumns;
                uint outputColumn = outputPixel
                    - outputRow * params.outputDetectorColumns;
                uint sum = 0u;
                for (uint rowOffset = 0u; rowOffset < 4u; ++rowOffset) {
                    uint sourceRow = outputRow * 4u + rowOffset;
                    if (sourceRow >= params.sourceDetectorRows) continue;
                    uint sourcePixel = sourceRow * params.sourceDetectorColumns
                        + outputColumn * 4u;
                    sum += bslz4Low8AlignedRow4(
                        lowPlaneScratch,
                        badPixelMask,
                        frame,
                        frameElements,
                        sourcePixel,
                        localMax
                    );
                }
                sums[pair] = sum;
            }
        }
        output[ulong(outputWord) * params.outputScanCount
            + globalFrameOffset + frame] = sums.x | (sums.y << 16u);

        for (uint pair = 0u; pair < 2u; ++pair) {
            uint outputPixel = firstOutputPixel + pair;
            if (outputPixel >= outputPixels) continue;
            uint outputRow = outputPixel / params.outputDetectorColumns;
            uint outputColumn = outputPixel - outputRow * params.outputDetectorColumns;
            uint sum = sums[pair];
            uchar bands = detectorBands[outputPixel];
            localBF += (bands & 1u) == 0u ? 0u : sum;
            localABF += (bands & 2u) == 0u ? 0u : sum;
            localDF += (bands & 4u) == 0u ? 0u : sum;
            localTotal += sum;
            localRowMoment += sum * outputRow;
            localColumnMoment += sum * outputColumn;
        }
    }

    localMax = simd_max(localMax);
    localBF = simd_sum(localBF);
    localABF = simd_sum(localABF);
    localDF = simd_sum(localDF);
    localTotal = simd_sum(localTotal);
    localRowMoment = simd_sum(localRowMoment);
    localColumnMoment = simd_sum(localColumnMoment);
    if (lane == 0u) {
        atomic_fetch_max_explicit(&groupMax, localMax, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupBF, localBF, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupABF, localABF, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupDF, localDF, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupTotal, localTotal, memory_order_relaxed);
        atomic_fetch_add_explicit(&groupRowMoment, localRowMoment, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &groupColumnMoment, localColumnMoment, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIndex == 0u) {
        uint outputFrame = globalFrameOffset + frame;
        uint frameAudit = 2u * outputFrame;
        atomic_fetch_max_explicit(
            &countAudit[frameAudit],
            atomic_load_explicit(&groupMax, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &bfMap[outputFrame],
            atomic_load_explicit(&groupBF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &abfMap[outputFrame],
            atomic_load_explicit(&groupABF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &dfMap[outputFrame],
            atomic_load_explicit(&groupDF, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comTotal[outputFrame],
            atomic_load_explicit(&groupTotal, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comRowMoment[outputFrame],
            atomic_load_explicit(&groupRowMoment, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &comColumnMoment[outputFrame],
            atomic_load_explicit(&groupColumnMoment, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}
