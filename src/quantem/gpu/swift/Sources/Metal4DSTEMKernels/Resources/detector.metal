#include <metal_stdlib>
using namespace metal;

struct DetectorParams {
    uint frameCount;
    uint detectorPixels;
    uint globalFrameOffset;
    uint padding;
};

template <typename Sample>
inline void detectorProducts(
    device const Sample *data,
    device uint *bfMap,
    device uint *abfMap,
    device uint *dfMap,
    constant DetectorParams &params,
    device const uchar *bands,
    threadgroup atomic_uint *sums,
    uint frame,
    uint threadIndex,
    uint lane,
    uint threads
) {
    if (frame >= params.frameCount) return;
    if (threadIndex < 3) {
        atomic_store_explicit(&sums[threadIndex], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const Sample *source =
        data + ulong(frame) * ulong(params.detectorPixels);
    uint bf = 0;
    uint abf = 0;
    uint df = 0;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value = uint(source[pixel]);
        uchar band = bands[pixel];
        if (band & 1u) bf += value;
        if (band & 2u) abf += value;
        if (band & 4u) df += value;
    }
    bf = simd_sum(bf);
    abf = simd_sum(abf);
    df = simd_sum(df);
    if (lane == 0) {
        atomic_fetch_add_explicit(&sums[0], bf, memory_order_relaxed);
        atomic_fetch_add_explicit(&sums[1], abf, memory_order_relaxed);
        atomic_fetch_add_explicit(&sums[2], df, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadIndex != 0) return;
    uint destination = params.globalFrameOffset + frame;
    bfMap[destination] = atomic_load_explicit(&sums[0], memory_order_relaxed);
    abfMap[destination] = atomic_load_explicit(&sums[1], memory_order_relaxed);
    dfMap[destination] = atomic_load_explicit(&sums[2], memory_order_relaxed);
}

// Produce BF/ABF/DF while the decoded batch is hot. One threadgroup owns each
// frame, reads contiguous detector values, and reduces eight SIMD-group sums.
kernel void detector_products_u16(
    device const ushort *data [[buffer(0)]],
    device uint *bfMap [[buffer(1)]],
    device uint *abfMap [[buffer(2)]],
    device uint *dfMap [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint sums[3];
    detectorProducts(
        data, bfMap, abfMap, dfMap, params, bands, sums,
        frame, threadIndex, lane, threads
    );
}

kernel void detector_products_u8(
    device const uchar *data [[buffer(0)]],
    device uint *bfMap [[buffer(1)]],
    device uint *abfMap [[buffer(2)]],
    device uint *dfMap [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint sums[3];
    detectorProducts(
        data, bfMap, abfMap, dfMap, params, bands, sums,
        frame, threadIndex, lane, threads
    );
}

template <typename Sample>
inline void detectorSum(
    device const Sample *data,
    device atomic_uint *output,
    uint detectorPixels,
    uint frameCount,
    uint pixel
) {
    if (pixel >= detectorPixels) return;
    uint sum = 0;
    for (uint frame = 0; frame < frameCount; ++frame) {
        sum += uint(data[ulong(frame) * detectorPixels + pixel]);
    }
    atomic_fetch_add_explicit(&output[pixel], sum, memory_order_relaxed);
}

// The maximum observed source count is below 2000, so summing 262,144 frames
// into uint32 is exact for this dataset. The app independently verifies that
// contract before treating this product as parity evidence.
kernel void detector_sum_u16(
    device const ushort *data [[buffer(0)]],
    device atomic_uint *output [[buffer(1)]],
    constant uint &detectorPixels [[buffer(2)]],
    constant uint &frameCount [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]
) {
    detectorSum(data, output, detectorPixels, frameCount, pixel);
}

kernel void detector_sum_u8(
    device const uchar *data [[buffer(0)]],
    device atomic_uint *output [[buffer(1)]],
    constant uint &detectorPixels [[buffer(2)]],
    constant uint &frameCount [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]
) {
    detectorSum(data, output, detectorPixels, frameCount, pixel);
}

// Convert scan-major packed words to detector-word-major storage once after
// load. Each 16x16 threadgroup moves a 32x32 tile, cutting dispatch count by
// four while keeping source reads and destination writes coalesced.
kernel void transpose_scan_words(
    device const uint *scanMajor [[buffer(0)]],
    device uint *wordMajor [[buffer(1)]],
    constant uint &sourceScanCount [[buffer(2)]],
    constant uint &detectorWords [[buffer(3)]],
    constant uint &destinationScanCount [[buffer(4)]],
    constant uint &destinationScanOffset [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    threadgroup uint tile[32][33];
    for (uint rowOffset = 0; rowOffset < 32; rowOffset += 16) {
        for (uint colOffset = 0; colOffset < 32; colOffset += 16) {
            uint tileRow = local.y + rowOffset;
            uint tileCol = local.x + colOffset;
            uint sourceScan = group.y * 32u + tileRow;
            uint sourceWord = group.x * 32u + tileCol;
            if (sourceWord < detectorWords && sourceScan < sourceScanCount) {
                tile[tileRow][tileCol] =
                    scanMajor[ulong(sourceScan) * detectorWords + sourceWord];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint rowOffset = 0; rowOffset < 32; rowOffset += 16) {
        for (uint colOffset = 0; colOffset < 32; colOffset += 16) {
            uint tileRow = local.x + colOffset;
            uint tileCol = local.y + rowOffset;
            uint destinationScan = group.y * 32u + tileRow;
            uint destinationWord = group.x * 32u + tileCol;
            if (destinationWord < detectorWords && destinationScan < sourceScanCount) {
                wordMajor[
                    ulong(destinationWord) * destinationScanCount
                    + destinationScanOffset + destinationScan
                ] =
                    tile[tileRow][tileCol];
            }
        }
    }
}

// Each entry is (detector uint32 word, 2-bit coefficients for its four uint8
// lanes). Coeff 1 includes a lane; coeff 2 subtracts it. Consecutive SIMD lanes
// process consecutive scan positions while reading one detector-major word,
// which reproduces the proven Show4DSTEM WebGPU drag topology on native Metal.
inline int signed_u8_word(uint word, uint coefficients) {
    int value = 0;
    for (uint lane = 0; lane < 4; ++lane) {
        uint coefficient = (coefficients >> (lane * 2u)) & 3u;
        int sample = int((word >> (lane * 8u)) & 0xffu);
        if (coefficient == 1u) value += sample;
        else if (coefficient == 2u) value -= sample;
    }
    return value;
}

kernel void full_sum_u8_word_major(
    device const uint *data [[buffer(0)]],
    device const uint2 *entries [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &entryCount [[buffer(4)]],
    uint scan [[thread_position_in_grid]]
) {
    if (scan >= scanCount) return;
    uint sum = 0;
    for (uint entry = 0; entry < entryCount; ++entry) {
        uint2 spec = entries[entry];
        uint word = data[ulong(spec.x) * scanCount + scan];
        sum += uint(signed_u8_word(word, spec.y));
    }
    output[scan] = sum;
}

kernel void signed_delta_u8_word_major(
    device const uint *data [[buffer(0)]],
    device const uint2 *entries [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &entryCount [[buffer(4)]],
    uint scan [[thread_position_in_grid]]
) {
    if (scan >= scanCount) return;
    int delta = 0;
    for (uint entry = 0; entry < entryCount; ++entry) {
        uint2 spec = entries[entry];
        uint word = data[ulong(spec.x) * scanCount + scan];
        delta += signed_u8_word(word, spec.y);
    }
    output[scan] = uint(int(output[scan]) + delta);
}

kernel void extract_u8_word_major_frame(
    device const uint *data [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &scanIndex [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= pixelCount) return;
    uint word = data[ulong(pixel >> 2u) * scanCount + scanIndex];
    output[pixel] = uchar((word >> ((pixel & 3u) * 8u)) & 0xffu);
}

inline int signed_u16_word(uint word, uint coefficients) {
    int value = 0;
    for (uint lane = 0; lane < 2; ++lane) {
        uint coefficient = (coefficients >> (lane * 2u)) & 3u;
        int sample = int((word >> (lane * 16u)) & 0xffffu);
        if (coefficient == 1u) value += sample;
        else if (coefficient == 2u) value -= sample;
    }
    return value;
}

kernel void full_sum_u16_word_major(
    device const uint *data [[buffer(0)]],
    device const uint2 *entries [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &entryCount [[buffer(4)]],
    uint scan [[thread_position_in_grid]]
) {
    if (scan >= scanCount) return;
    uint sum = 0;
    for (uint entry = 0; entry < entryCount; ++entry) {
        uint2 spec = entries[entry];
        uint word = data[ulong(spec.x) * scanCount + scan];
        sum += uint(signed_u16_word(word, spec.y));
    }
    output[scan] = sum;
}

kernel void signed_delta_u16_word_major(
    device const uint *data [[buffer(0)]],
    device const uint2 *entries [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &entryCount [[buffer(4)]],
    uint scan [[thread_position_in_grid]]
) {
    if (scan >= scanCount) return;
    int delta = 0;
    for (uint entry = 0; entry < entryCount; ++entry) {
        uint2 spec = entries[entry];
        uint word = data[ulong(spec.x) * scanCount + scan];
        delta += signed_u16_word(word, spec.y);
    }
    output[scan] = uint(int(output[scan]) + delta);
}

kernel void extract_u16_word_major_frame(
    device const uint *data [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant uint &scanIndex [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= pixelCount) return;
    uint word = data[ulong(pixel >> 1u) * scanCount + scanIndex];
    output[pixel] = ushort((word >> ((pixel & 1u) * 16u)) & 0xffffu);
}
