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
    for (uint index = threadIndex; index < length; index += kLZ4Threads) {
        destination[index] = source[index % distance];
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
        atomic_fetch_max_explicit(
            &countAudit[0],
            atomic_load_explicit(&batchMax, memory_order_relaxed),
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &countAudit[1 + globalFrameOffset + frame],
            atomic_load_explicit(&batchAbove255, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}
