#include <metal_stdlib>
using namespace metal;

// Establish every possible Apple GPU virtual-memory page before a large exact
// resident load. The loader overwrites every byte afterward. The host supplies
// its VM page size so this avoids a redundant full-volume clear while
// preserving deterministic synchronization.
kernel void prepare_private_resident_pages(
    device uint *destination [[buffer(0)]],
    constant ulong &byteCount [[buffer(1)]],
    constant ulong &byteStride [[buffer(2)]],
    uint pageIndex [[thread_position_in_grid]]
) {
    ulong byteOffset = ulong(pageIndex) * byteStride;
    if (byteOffset >= byteCount) return;
    destination[byteOffset / sizeof(uint)] = 0u;
}

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

// Produce exact virtual-detector sums and center-of-mass integer moments in
// the same scan-major pass. The 64-bit moments are safe for the complete
// uint16 range and avoid rereading the much larger word-major resident volume.
kernel void detector_products_u16_with_u64_moments(
    device const ushort *data [[buffer(0)]],
    device uint *bfMap [[buffer(1)]],
    device uint *abfMap [[buffer(2)]],
    device uint *dfMap [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    device ulong *totalMap [[buffer(6)]],
    device ulong *rowMomentMap [[buffer(7)]],
    device ulong *columnMomentMap [[buffer(8)]],
    constant uint &detectorColumns [[buffer(9)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (frame >= params.frameCount) return;
    threadgroup atomic_uint bandSums[3];
    threadgroup ulong totalScratch[256];
    threadgroup ulong rowScratch[256];
    threadgroup ulong columnScratch[256];
    if (threadIndex < 3u) {
        atomic_store_explicit(&bandSums[threadIndex], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const ushort *source =
        data + ulong(frame) * ulong(params.detectorPixels);
    uint bf = 0u;
    uint abf = 0u;
    uint df = 0u;
    ulong total = 0ul;
    ulong rowMoment = 0ul;
    ulong columnMoment = 0ul;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value = uint(source[pixel]);
        uchar band = bands[pixel];
        if (band & 1u) bf += value;
        if (band & 2u) abf += value;
        if (band & 4u) df += value;
        uint row = pixel / detectorColumns;
        uint column = pixel - row * detectorColumns;
        ulong wide = ulong(value);
        total += wide;
        rowMoment += wide * ulong(row);
        columnMoment += wide * ulong(column);
    }
    bf = simd_sum(bf);
    abf = simd_sum(abf);
    df = simd_sum(df);
    if (lane == 0u) {
        atomic_fetch_add_explicit(&bandSums[0], bf, memory_order_relaxed);
        atomic_fetch_add_explicit(&bandSums[1], abf, memory_order_relaxed);
        atomic_fetch_add_explicit(&bandSums[2], df, memory_order_relaxed);
    }
    totalScratch[threadIndex] = total;
    rowScratch[threadIndex] = rowMoment;
    columnScratch[threadIndex] = columnMoment;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threads >> 1u; offset > 0u; offset >>= 1u) {
        if (threadIndex < offset) {
            totalScratch[threadIndex] += totalScratch[threadIndex + offset];
            rowScratch[threadIndex] += rowScratch[threadIndex + offset];
            columnScratch[threadIndex] += columnScratch[threadIndex + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (threadIndex == 0u) {
        uint destination = params.globalFrameOffset + frame;
        bfMap[destination] = atomic_load_explicit(&bandSums[0], memory_order_relaxed);
        abfMap[destination] = atomic_load_explicit(&bandSums[1], memory_order_relaxed);
        dfMap[destination] = atomic_load_explicit(&bandSums[2], memory_order_relaxed);
        totalMap[destination] = totalScratch[0];
        rowMomentMap[destination] = rowScratch[0];
        columnMomentMap[destination] = columnScratch[0];
    }
}

// Produce exact, overflow-safe screening sufficient statistics from one
// frame-major uint16 decode window. One threadgroup owns each frame. Every
// partial sum uses ulong and the fixed tree reduction is deterministic; no
// dataset-specific count ceiling or unordered atomic sum is assumed.
kernel void detector_products_u16_exact_u64(
    device const ushort *data [[buffer(0)]],
    device ulong *band1Map [[buffer(1)]],
    device ulong *band2Map [[buffer(2)]],
    device ulong *band4Map [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    device ulong *totalMap [[buffer(6)]],
    device ulong *rowMomentMap [[buffer(7)]],
    device ulong *columnMomentMap [[buffer(8)]],
    constant uint &detectorColumns [[buffer(9)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (frame >= params.frameCount) return;
    threadgroup ulong band1Scratch[256];
    threadgroup ulong band2Scratch[256];
    threadgroup ulong band4Scratch[256];
    threadgroup ulong totalScratch[256];
    threadgroup ulong rowScratch[256];
    threadgroup ulong columnScratch[256];

    device const ushort *source =
        data + ulong(frame) * ulong(params.detectorPixels);
    ulong band1 = 0ul;
    ulong band2 = 0ul;
    ulong band4 = 0ul;
    ulong total = 0ul;
    ulong rowMoment = 0ul;
    ulong columnMoment = 0ul;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        ulong value = ulong(source[pixel]);
        uchar membership = bands[pixel];
        if (membership & 1u) band1 += value;
        if (membership & 2u) band2 += value;
        if (membership & 4u) band4 += value;
        uint row = pixel / detectorColumns;
        uint column = pixel - row * detectorColumns;
        total += value;
        rowMoment += value * ulong(row);
        columnMoment += value * ulong(column);
    }
    band1Scratch[threadIndex] = band1;
    band2Scratch[threadIndex] = band2;
    band4Scratch[threadIndex] = band4;
    totalScratch[threadIndex] = total;
    rowScratch[threadIndex] = rowMoment;
    columnScratch[threadIndex] = columnMoment;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threads >> 1u; offset > 0u; offset >>= 1u) {
        if (threadIndex < offset) {
            band1Scratch[threadIndex] += band1Scratch[threadIndex + offset];
            band2Scratch[threadIndex] += band2Scratch[threadIndex + offset];
            band4Scratch[threadIndex] += band4Scratch[threadIndex + offset];
            totalScratch[threadIndex] += totalScratch[threadIndex + offset];
            rowScratch[threadIndex] += rowScratch[threadIndex + offset];
            columnScratch[threadIndex] += columnScratch[threadIndex + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (threadIndex == 0u) {
        uint destination = params.globalFrameOffset + frame;
        band1Map[destination] = band1Scratch[0];
        band2Map[destination] = band2Scratch[0];
        band4Map[destination] = band4Scratch[0];
        totalMap[destination] = totalScratch[0];
        rowMomentMap[destination] = rowScratch[0];
        columnMomentMap[destination] = columnScratch[0];
    }
}

// Exact counterpart for an identity-audited lossless uint8 staging window.
// Every accumulator and public product remains uint64, matching the native
// uint16 path byte-for-byte when the durable audit proves all high bytes zero.
kernel void detector_products_u8_exact_u64(
    device const uchar *data [[buffer(0)]],
    device ulong *band1Map [[buffer(1)]],
    device ulong *band2Map [[buffer(2)]],
    device ulong *band4Map [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    device ulong *totalMap [[buffer(6)]],
    device ulong *rowMomentMap [[buffer(7)]],
    device ulong *columnMomentMap [[buffer(8)]],
    constant uint &detectorColumns [[buffer(9)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (frame >= params.frameCount) return;
    threadgroup ulong band1Scratch[256];
    threadgroup ulong band2Scratch[256];
    threadgroup ulong band4Scratch[256];
    threadgroup ulong totalScratch[256];
    threadgroup ulong rowScratch[256];
    threadgroup ulong columnScratch[256];

    device const uchar *source =
        data + ulong(frame) * ulong(params.detectorPixels);
    ulong band1 = 0ul;
    ulong band2 = 0ul;
    ulong band4 = 0ul;
    ulong total = 0ul;
    ulong rowMoment = 0ul;
    ulong columnMoment = 0ul;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        ulong value = ulong(source[pixel]);
        uchar membership = bands[pixel];
        if (membership & 1u) band1 += value;
        if (membership & 2u) band2 += value;
        if (membership & 4u) band4 += value;
        uint row = pixel / detectorColumns;
        uint column = pixel - row * detectorColumns;
        total += value;
        rowMoment += value * ulong(row);
        columnMoment += value * ulong(column);
    }
    band1Scratch[threadIndex] = band1;
    band2Scratch[threadIndex] = band2;
    band4Scratch[threadIndex] = band4;
    totalScratch[threadIndex] = total;
    rowScratch[threadIndex] = rowMoment;
    columnScratch[threadIndex] = columnMoment;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threads >> 1u; offset > 0u; offset >>= 1u) {
        if (threadIndex < offset) {
            band1Scratch[threadIndex] += band1Scratch[threadIndex + offset];
            band2Scratch[threadIndex] += band2Scratch[threadIndex + offset];
            band4Scratch[threadIndex] += band4Scratch[threadIndex + offset];
            totalScratch[threadIndex] += totalScratch[threadIndex + offset];
            rowScratch[threadIndex] += rowScratch[threadIndex + offset];
            columnScratch[threadIndex] += columnScratch[threadIndex + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (threadIndex == 0u) {
        uint destination = params.globalFrameOffset + frame;
        band1Map[destination] = band1Scratch[0];
        band2Map[destination] = band2Scratch[0];
        band4Map[destination] = band4Scratch[0];
        totalMap[destination] = totalScratch[0];
        rowMomentMap[destination] = rowScratch[0];
        columnMomentMap[destination] = columnScratch[0];
    }
}

// Fast exact counterpart for audited uint8 staging when the host has proved
// that every per-frame band sum and row/column moment fits uint32. Integer
// SIMD reductions replace six 256-element ulong trees; results are widened to
// the unchanged public uint64 product buffers only after the exact reduction.
kernel void detector_products_u8_exact_u32_simd_to_u64(
    device const uchar *data [[buffer(0)]],
    device ulong *band1Map [[buffer(1)]],
    device ulong *band2Map [[buffer(2)]],
    device ulong *band4Map [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    device ulong *totalMap [[buffer(6)]],
    device ulong *rowMomentMap [[buffer(7)]],
    device ulong *columnMomentMap [[buffer(8)]],
    constant uint &detectorColumns [[buffer(9)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (frame >= params.frameCount) return;
    threadgroup uint partials[6 * 8];

    device const uchar *source =
        data + ulong(frame) * ulong(params.detectorPixels);
    uint band1 = 0u;
    uint band2 = 0u;
    uint band4 = 0u;
    uint total = 0u;
    uint rowMoment = 0u;
    uint columnMoment = 0u;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value = uint(source[pixel]);
        uchar membership = bands[pixel];
        if (membership & 1u) band1 += value;
        if (membership & 2u) band2 += value;
        if (membership & 4u) band4 += value;
        uint row = pixel / detectorColumns;
        uint column = pixel - row * detectorColumns;
        total += value;
        rowMoment += value * row;
        columnMoment += value * column;
    }
    band1 = simd_sum(band1);
    band2 = simd_sum(band2);
    band4 = simd_sum(band4);
    total = simd_sum(total);
    rowMoment = simd_sum(rowMoment);
    columnMoment = simd_sum(columnMoment);
    if (lane == 0u) {
        partials[simdgroup] = band1;
        partials[8u + simdgroup] = band2;
        partials[16u + simdgroup] = band4;
        partials[24u + simdgroup] = total;
        partials[32u + simdgroup] = rowMoment;
        partials[40u + simdgroup] = columnMoment;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        uint simdgroups = (threads + 31u) / 32u;
        uint finalBand1 = lane < simdgroups ? partials[lane] : 0u;
        uint finalBand2 = lane < simdgroups ? partials[8u + lane] : 0u;
        uint finalBand4 = lane < simdgroups ? partials[16u + lane] : 0u;
        uint finalTotal = lane < simdgroups ? partials[24u + lane] : 0u;
        uint finalRowMoment = lane < simdgroups ? partials[32u + lane] : 0u;
        uint finalColumnMoment = lane < simdgroups ? partials[40u + lane] : 0u;
        finalBand1 = simd_sum(finalBand1);
        finalBand2 = simd_sum(finalBand2);
        finalBand4 = simd_sum(finalBand4);
        finalTotal = simd_sum(finalTotal);
        finalRowMoment = simd_sum(finalRowMoment);
        finalColumnMoment = simd_sum(finalColumnMoment);
        if (lane == 0u) {
            uint destination = params.globalFrameOffset + frame;
            band1Map[destination] = ulong(finalBand1);
            band2Map[destination] = ulong(finalBand2);
            band4Map[destination] = ulong(finalBand4);
            totalMap[destination] = ulong(finalTotal);
            rowMomentMap[destination] = ulong(finalRowMoment);
            columnMomentMap[destination] = ulong(finalColumnMoment);
        }
    }
}

// Exact 192x192 specialization. One thread owns one detector column and walks
// rows directly, eliminating per-pixel division/modulo from the generic path.
// The host retains the same proven uint32 accumulator bound and widens every
// published value to the unchanged uint64 product representation.
kernel void detector_products_u8_detector192_exact_u32_simd_to_u64(
    device const uchar *data [[buffer(0)]],
    device ulong *band1Map [[buffer(1)]],
    device ulong *band2Map [[buffer(2)]],
    device ulong *band4Map [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    device ulong *totalMap [[buffer(6)]],
    device ulong *rowMomentMap [[buffer(7)]],
    device ulong *columnMomentMap [[buffer(8)]],
    constant uint &detectorColumns [[buffer(9)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (frame >= params.frameCount) return;
    threadgroup uint partials[6 * 8];

    device const uchar *source =
        data + ulong(frame) * ulong(params.detectorPixels);
    uint band1 = 0u;
    uint band2 = 0u;
    uint band4 = 0u;
    uint total = 0u;
    uint rowMoment = 0u;
    uint columnMoment = 0u;
    uint column = threadIndex;
    for (uint row = 0u; row < 192u; ++row) {
        uint pixel = row * 192u + column;
        uint value = uint(source[pixel]);
        uchar membership = bands[pixel];
        if (membership & 1u) band1 += value;
        if (membership & 2u) band2 += value;
        if (membership & 4u) band4 += value;
        total += value;
        rowMoment += value * row;
        columnMoment += value * column;
    }
    band1 = simd_sum(band1);
    band2 = simd_sum(band2);
    band4 = simd_sum(band4);
    total = simd_sum(total);
    rowMoment = simd_sum(rowMoment);
    columnMoment = simd_sum(columnMoment);
    if (lane == 0u) {
        partials[simdgroup] = band1;
        partials[8u + simdgroup] = band2;
        partials[16u + simdgroup] = band4;
        partials[24u + simdgroup] = total;
        partials[32u + simdgroup] = rowMoment;
        partials[40u + simdgroup] = columnMoment;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        uint finalBand1 = lane < 6u ? partials[lane] : 0u;
        uint finalBand2 = lane < 6u ? partials[8u + lane] : 0u;
        uint finalBand4 = lane < 6u ? partials[16u + lane] : 0u;
        uint finalTotal = lane < 6u ? partials[24u + lane] : 0u;
        uint finalRowMoment = lane < 6u ? partials[32u + lane] : 0u;
        uint finalColumnMoment = lane < 6u ? partials[40u + lane] : 0u;
        finalBand1 = simd_sum(finalBand1);
        finalBand2 = simd_sum(finalBand2);
        finalBand4 = simd_sum(finalBand4);
        finalTotal = simd_sum(finalTotal);
        finalRowMoment = simd_sum(finalRowMoment);
        finalColumnMoment = simd_sum(finalColumnMoment);
        if (lane == 0u) {
            uint destination = params.globalFrameOffset + frame;
            band1Map[destination] = ulong(finalBand1);
            band2Map[destination] = ulong(finalBand2);
            band4Map[destination] = ulong(finalBand4);
            totalMap[destination] = ulong(finalTotal);
            rowMomentMap[destination] = ulong(finalRowMoment);
            columnMomentMap[destination] = ulong(finalColumnMoment);
        }
    }
}

kernel void detector_products_u8_with_u64_moments(
    device const uchar *data [[buffer(0)]],
    device uint *bfMap [[buffer(1)]],
    device uint *abfMap [[buffer(2)]],
    device uint *dfMap [[buffer(3)]],
    constant DetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    device ulong *totalMap [[buffer(6)]],
    device ulong *rowMomentMap [[buffer(7)]],
    device ulong *columnMomentMap [[buffer(8)]],
    constant uint &detectorColumns [[buffer(9)]],
    uint frame [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (frame >= params.frameCount) return;
    threadgroup atomic_uint bandSums[3];
    threadgroup ulong totalScratch[256];
    threadgroup ulong rowScratch[256];
    threadgroup ulong columnScratch[256];
    if (threadIndex < 3u) {
        atomic_store_explicit(&bandSums[threadIndex], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar *source =
        data + ulong(frame) * ulong(params.detectorPixels);
    uint bf = 0u;
    uint abf = 0u;
    uint df = 0u;
    ulong total = 0ul;
    ulong rowMoment = 0ul;
    ulong columnMoment = 0ul;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value = uint(source[pixel]);
        uchar band = bands[pixel];
        if (band & 1u) bf += value;
        if (band & 2u) abf += value;
        if (band & 4u) df += value;
        uint row = pixel / detectorColumns;
        uint column = pixel - row * detectorColumns;
        ulong wide = ulong(value);
        total += wide;
        rowMoment += wide * ulong(row);
        columnMoment += wide * ulong(column);
    }
    bf = simd_sum(bf);
    abf = simd_sum(abf);
    df = simd_sum(df);
    if (lane == 0u) {
        atomic_fetch_add_explicit(&bandSums[0], bf, memory_order_relaxed);
        atomic_fetch_add_explicit(&bandSums[1], abf, memory_order_relaxed);
        atomic_fetch_add_explicit(&bandSums[2], df, memory_order_relaxed);
    }
    totalScratch[threadIndex] = total;
    rowScratch[threadIndex] = rowMoment;
    columnScratch[threadIndex] = columnMoment;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threads >> 1u; offset > 0u; offset >>= 1u) {
        if (threadIndex < offset) {
            totalScratch[threadIndex] += totalScratch[threadIndex + offset];
            rowScratch[threadIndex] += rowScratch[threadIndex + offset];
            columnScratch[threadIndex] += columnScratch[threadIndex + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (threadIndex == 0u) {
        uint destination = params.globalFrameOffset + frame;
        bfMap[destination] = atomic_load_explicit(&bandSums[0], memory_order_relaxed);
        abfMap[destination] = atomic_load_explicit(&bandSums[1], memory_order_relaxed);
        dfMap[destination] = atomic_load_explicit(&bandSums[2], memory_order_relaxed);
        totalMap[destination] = totalScratch[0];
        rowMomentMap[destination] = rowScratch[0];
        columnMomentMap[destination] = columnScratch[0];
    }
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

// Accumulate one decoded uint16 window into an exact detector-space sum. Each
// detector pixel has one unique writer, so command-queue order makes repeated
// windows deterministic without atomics.
kernel void detector_accumulate_u16_u64(
    device const ushort *data [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant uint &detectorPixels [[buffer(2)]],
    constant uint &frameCount [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= detectorPixels) return;
    ulong sum = 0ul;
    for (uint frame = 0u; frame < frameCount; ++frame) {
        sum += ulong(data[ulong(frame) * ulong(detectorPixels) + pixel]);
    }
    output[pixel] += sum;
}

// Exact detector-space accumulator for frame-major uint8 staging. Eight SIMD
// groups read eight source frames concurrently, so every 32-byte transaction
// remains contiguous in detector space while latency is hidden across frames.
// One thread then combines the eight exact uint64 partials for each pixel.
kernel void detector_accumulate_u8_u64_frame_tiled(
    device const uchar *data [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant uint &detectorPixels [[buffer(2)]],
    constant uint &frameCount [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    uint pixel = group.x * 32u + local.x;
    threadgroup ulong partials[8][33];
    ulong sum = 0ul;
    if (pixel < detectorPixels) {
        for (uint frame = local.y; frame < frameCount; frame += 8u) {
            sum += ulong(data[ulong(frame) * detectorPixels + pixel]);
        }
    }
    partials[local.y][local.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local.y == 0u && pixel < detectorPixels) {
        ulong total = 0ul;
        for (uint frameLane = 0u; frameLane < 8u; ++frameLane) {
            total += partials[frameLane][local.x];
        }
        output[pixel] += total;
    }
}

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

// Canonical 32x32 tiled transpose. A 32x8 threadgroup issues full-width
// coalesced transactions and moves four rows per thread with one barrier.
kernel void transpose_scan_words_32x8(
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
    for (uint rowOffset = 0u; rowOffset < 32u; rowOffset += 8u) {
        uint tileRow = local.y + rowOffset;
        uint sourceScan = group.y * 32u + tileRow;
        uint sourceWord = group.x * 32u + local.x;
        if (sourceWord < detectorWords && sourceScan < sourceScanCount) {
            tile[tileRow][local.x] =
                scanMajor[ulong(sourceScan) * detectorWords + sourceWord];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint wordOffset = 0u; wordOffset < 32u; wordOffset += 8u) {
        uint tileColumn = local.y + wordOffset;
        uint destinationScan = group.y * 32u + local.x;
        uint destinationWord = group.x * 32u + tileColumn;
        if (destinationWord < detectorWords && destinationScan < sourceScanCount) {
            wordMajor[
                ulong(destinationWord) * destinationScanCount
                + destinationScanOffset + destinationScan
            ] = tile[local.x][tileColumn];
        }
    }
}

struct ScanBinParams {
    uint sourceRows;
    uint sourceCols;
    uint detectorPixels;
    uint scanBin;
    uint outputScanCount;
    uint outputCols;
    uint destinationRowOffset;
    uint padding;
};

template <typename Sample>
inline void scanBinToU32WordMajor(
    device const Sample *source,
    device uint *destination,
    constant ScanBinParams &params,
    uint2 position
) {
    uint detectorPixel = position.x;
    uint localOutputScan = position.y;
    uint localOutputRows = (params.sourceRows + params.scanBin - 1u) / params.scanBin;
    uint localOutputCount = localOutputRows * params.outputCols;
    if (detectorPixel >= params.detectorPixels || localOutputScan >= localOutputCount) return;

    uint outputRow = localOutputScan / params.outputCols;
    uint outputCol = localOutputScan - outputRow * params.outputCols;
    uint sourceRowStart = outputRow * params.scanBin;
    uint sourceColStart = outputCol * params.scanBin;
    uint sourceRowStop = min(params.sourceRows, sourceRowStart + params.scanBin);
    uint sourceColStop = min(params.sourceCols, sourceColStart + params.scanBin);
    uint sum = 0;
    for (uint row = sourceRowStart; row < sourceRowStop; ++row) {
        for (uint col = sourceColStart; col < sourceColStop; ++col) {
            ulong sourceScan = ulong(row) * params.sourceCols + col;
            sum += uint(source[sourceScan * params.detectorPixels + detectorPixel]);
        }
    }
    uint destinationScan =
        (params.destinationRowOffset + outputRow) * params.outputCols + outputCol;
    destination[ulong(detectorPixel) * params.outputScanCount + destinationScan] = sum;
}

// Scan binning preserves exact integer sums and writes one uint32 value per
// detector pixel. Edge bins include every acquired scan position rather than
// cropping incomplete bins.
kernel void scan_bin_u8_to_u32_word_major(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    scanBinToU32WordMajor(source, destination, params, position);
}

kernel void scan_bin_u16_to_u32_word_major(
    device const ushort *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    scanBinToU32WordMajor(source, destination, params, position);
}

struct ScanDetectorBinParams {
    uint sourceRows;
    uint sourceCols;
    uint sourceDetectorRows;
    uint sourceDetectorCols;
    uint scanBin;
    uint detectorBin;
    uint outputScanCount;
    uint outputScanCols;
    uint outputDetectorRows;
    uint outputDetectorCols;
    uint destinationScanRowOffset;
    uint padding;
};

template <typename Sample>
inline uint scanDetectorBinValue(
    device const Sample *source,
    constant ScanDetectorBinParams &params,
    uint outputDetectorPixel,
    uint localOutputScan
) {
    uint outputScanRow = localOutputScan / params.outputScanCols;
    uint outputScanCol = localOutputScan - outputScanRow * params.outputScanCols;
    uint outputDetectorRow = outputDetectorPixel / params.outputDetectorCols;
    uint outputDetectorCol =
        outputDetectorPixel - outputDetectorRow * params.outputDetectorCols;
    uint sourceScanRowStart = outputScanRow * params.scanBin;
    uint sourceScanColStart = outputScanCol * params.scanBin;
    uint sourceScanRowStop = min(
        params.sourceRows, sourceScanRowStart + params.scanBin
    );
    uint sourceScanColStop = min(
        params.sourceCols, sourceScanColStart + params.scanBin
    );
    uint sourceDetectorRowStart = outputDetectorRow * params.detectorBin;
    uint sourceDetectorColStart = outputDetectorCol * params.detectorBin;
    uint sourceDetectorRowStop = min(
        params.sourceDetectorRows, sourceDetectorRowStart + params.detectorBin
    );
    uint sourceDetectorColStop = min(
        params.sourceDetectorCols, sourceDetectorColStart + params.detectorBin
    );
    uint sourceDetectorPixels = params.sourceDetectorRows * params.sourceDetectorCols;
    uint sum = 0u;
    for (uint scanRow = sourceScanRowStart; scanRow < sourceScanRowStop; ++scanRow) {
        for (uint scanCol = sourceScanColStart; scanCol < sourceScanColStop; ++scanCol) {
            uint sourceScan = scanRow * params.sourceCols + scanCol;
            for (uint detectorRow = sourceDetectorRowStart;
                 detectorRow < sourceDetectorRowStop; ++detectorRow) {
                for (uint detectorCol = sourceDetectorColStart;
                     detectorCol < sourceDetectorColStop; ++detectorCol) {
                    uint sourceDetectorPixel =
                        detectorRow * params.sourceDetectorCols + detectorCol;
                    sum += uint(
                        source[ulong(sourceScan) * sourceDetectorPixels + sourceDetectorPixel]
                    );
                }
            }
        }
    }
    return sum;
}

template <typename Sample>
inline void scanDetectorBinToU32WordMajor(
    device const Sample *source,
    device uint *destination,
    constant ScanDetectorBinParams &params,
    uint2 position
) {
    uint outputDetectorPixel = position.x;
    uint localOutputScan = position.y;
    uint localOutputScanRows =
        (params.sourceRows + params.scanBin - 1u) / params.scanBin;
    uint localOutputScanCount = localOutputScanRows * params.outputScanCols;
    uint outputDetectorPixels = params.outputDetectorRows * params.outputDetectorCols;
    if (outputDetectorPixel >= outputDetectorPixels ||
        localOutputScan >= localOutputScanCount) return;

    uint outputScanRow = localOutputScan / params.outputScanCols;
    uint outputScanCol = localOutputScan - outputScanRow * params.outputScanCols;
    uint destinationScan =
        (params.destinationScanRowOffset + outputScanRow) * params.outputScanCols
        + outputScanCol;
    destination[ulong(outputDetectorPixel) * params.outputScanCount + destinationScan] =
        scanDetectorBinValue(source, params, outputDetectorPixel, localOutputScan);
}

kernel void scan_detector_bin_u8_to_u32_word_major(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    scanDetectorBinToU32WordMajor(source, destination, params, position);
}

// Exact compact resident path. Callers must prove that the scan-and-detector
// contribution bound fits uint16 and expose source, staging, and output dtypes
// in provenance; otherwise they use the uint32 path.
template <typename Sample>
inline void scanDetectorBinToU16WordMajor(
    device const Sample *source,
    device uint *destination,
    constant ScanDetectorBinParams &params,
    uint2 position
) {
    uint outputDetectorWord = position.x;
    uint localOutputScan = position.y;
    uint localOutputScanRows =
        (params.sourceRows + params.scanBin - 1u) / params.scanBin;
    uint localOutputScanCount = localOutputScanRows * params.outputScanCols;
    uint outputDetectorPixels = params.outputDetectorRows * params.outputDetectorCols;
    uint firstPixel = outputDetectorWord * 2u;
    if (firstPixel >= outputDetectorPixels ||
        localOutputScan >= localOutputScanCount) return;

    uint low = scanDetectorBinValue(source, params, firstPixel, localOutputScan);
    uint high = firstPixel + 1u < outputDetectorPixels
        ? scanDetectorBinValue(source, params, firstPixel + 1u, localOutputScan)
        : 0u;
    uint outputScanRow = localOutputScan / params.outputScanCols;
    uint outputScanCol = localOutputScan - outputScanRow * params.outputScanCols;
    uint destinationScan =
        (params.destinationScanRowOffset + outputScanRow) * params.outputScanCols
        + outputScanCol;
    destination[ulong(outputDetectorWord) * params.outputScanCount + destinationScan] =
        low | (high << 16u);
}

kernel void scan_detector_bin_u8_to_u16_word_major(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    scanDetectorBinToU16WordMajor(source, destination, params, position);
}

kernel void scan_detector_bin_u16_to_u16_word_major(
    device const ushort *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    scanDetectorBinToU16WordMajor(source, destination, params, position);
}

kernel void scan_detector_bin_u16_to_u32_word_major(
    device const ushort *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    scanDetectorBinToU32WordMajor(source, destination, params, position);
}

// A prepared QH5 slice is a contiguous frame interval, but chunk/shard
// boundaries need not align to complete scan rows. This exact path therefore
// maps one decoded frame directly to one working scan position while retaining
// detector-word-major packed uint16 storage. Scan binning and crop are excluded
// by the typed Swift contract.
struct ContiguousDetectorBinParams {
    uint frameCount;
    uint sourceDetectorRows;
    uint sourceDetectorCols;
    uint detectorBin;
    uint destinationScanCount;
    uint destinationScanOffset;
    uint outputDetectorRows;
    uint outputDetectorCols;
};

inline uint contiguousDetectorBinU16Value(
    device const ushort *source,
    constant ContiguousDetectorBinParams &params,
    uint outputDetectorPixel,
    uint localFrame
) {
    uint outputDetectorRow = outputDetectorPixel / params.outputDetectorCols;
    uint outputDetectorCol =
        outputDetectorPixel - outputDetectorRow * params.outputDetectorCols;
    uint sourceDetectorRowStart = outputDetectorRow * params.detectorBin;
    uint sourceDetectorColStart = outputDetectorCol * params.detectorBin;
    uint sourceDetectorRowStop = min(
        params.sourceDetectorRows,
        sourceDetectorRowStart + params.detectorBin
    );
    uint sourceDetectorColStop = min(
        params.sourceDetectorCols,
        sourceDetectorColStart + params.detectorBin
    );
    uint sourceDetectorPixels = params.sourceDetectorRows * params.sourceDetectorCols;
    ulong frameBase = ulong(localFrame) * sourceDetectorPixels;
    uint sum = 0u;
    for (uint row = sourceDetectorRowStart; row < sourceDetectorRowStop; ++row) {
        ulong rowBase = frameBase + ulong(row) * params.sourceDetectorCols;
        for (uint col = sourceDetectorColStart; col < sourceDetectorColStop; ++col) {
            sum += source[rowBase + col];
        }
    }
    return sum;
}

kernel void contiguous_detector_bin_u16_to_u16_word_major(
    device const ushort *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ContiguousDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint outputDetectorWord = position.x;
    uint localFrame = position.y;
    uint outputDetectorPixels = params.outputDetectorRows * params.outputDetectorCols;
    uint firstPixel = outputDetectorWord * 2u;
    if (firstPixel >= outputDetectorPixels || localFrame >= params.frameCount) return;

    uint low = contiguousDetectorBinU16Value(
        source, params, firstPixel, localFrame
    );
    uint high = firstPixel + 1u < outputDetectorPixels
        ? contiguousDetectorBinU16Value(
            source, params, firstPixel + 1u, localFrame
        )
        : 0u;
    uint destinationScan = params.destinationScanOffset + localFrame;
    destination[ulong(outputDetectorWord) * params.destinationScanCount + destinationScan] =
        low | (high << 16u);
}

inline uint contiguousDetectorBinU8Value(
    device const uchar *source,
    constant ContiguousDetectorBinParams &params,
    uint outputDetectorPixel,
    uint localFrame
) {
    uint outputDetectorRow = outputDetectorPixel / params.outputDetectorCols;
    uint outputDetectorCol =
        outputDetectorPixel - outputDetectorRow * params.outputDetectorCols;
    uint sourceDetectorRowStart = outputDetectorRow * params.detectorBin;
    uint sourceDetectorColStart = outputDetectorCol * params.detectorBin;
    uint sourceDetectorRowStop = min(
        params.sourceDetectorRows,
        sourceDetectorRowStart + params.detectorBin
    );
    uint sourceDetectorColStop = min(
        params.sourceDetectorCols,
        sourceDetectorColStart + params.detectorBin
    );
    uint sourceDetectorPixels = params.sourceDetectorRows * params.sourceDetectorCols;
    ulong frameBase = ulong(localFrame) * sourceDetectorPixels;
    uint sum = 0u;
    for (uint row = sourceDetectorRowStart; row < sourceDetectorRowStop; ++row) {
        ulong rowBase = frameBase + ulong(row) * params.sourceDetectorCols;
        for (uint col = sourceDetectorColStart; col < sourceDetectorColStop; ++col) {
            sum += source[rowBase + col];
        }
    }
    return sum;
}

kernel void contiguous_detector_bin_u8_to_u16_word_major(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ContiguousDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint outputDetectorWord = position.x;
    uint localFrame = position.y;
    uint outputDetectorPixels = params.outputDetectorRows * params.outputDetectorCols;
    uint firstPixel = outputDetectorWord * 2u;
    if (firstPixel >= outputDetectorPixels || localFrame >= params.frameCount) return;

    uint low = contiguousDetectorBinU8Value(
        source, params, firstPixel, localFrame
    );
    uint high = firstPixel + 1u < outputDetectorPixels
        ? contiguousDetectorBinU8Value(
            source, params, firstPixel + 1u, localFrame
        )
        : 0u;
    uint destinationScan = params.destinationScanOffset + localFrame;
    destination[ulong(outputDetectorWord) * params.destinationScanCount + destinationScan] =
        low | (high << 16u);
}

// Exact 192x192 detector-bin1 specialization. A 16x16 threadgroup widens a
// 32x32 tile of adjacent uint8 samples to packed uint16 and transposes it so
// both frame-major source reads and detector-word-major writes are coalesced.
kernel void contiguous_detector_bin1_u8_to_u16_word_major_tiled32(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ContiguousDetectorBinParams &params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    threadgroup uint tile[32][33];
    uint wordBase = group.x * 32u;
    uint frameBase = group.y * 32u;
    uint outputDetectorPixels = params.outputDetectorRows * params.outputDetectorCols;
    uint outputDetectorWords = (outputDetectorPixels + 1u) / 2u;
    uint sourceDetectorPixels = params.sourceDetectorRows * params.sourceDetectorCols;

    for (uint frameDelta = 0u; frameDelta < 32u; frameDelta += 16u) {
        uint localFrame = frameBase + local.y + frameDelta;
        for (uint wordDelta = 0u; wordDelta < 32u; wordDelta += 16u) {
            uint outputWord = wordBase + local.x + wordDelta;
            uint packed = 0u;
            if (localFrame < params.frameCount && outputWord < outputDetectorWords) {
                ulong sourceOffset =
                    ulong(localFrame) * ulong(sourceDetectorPixels)
                    + ulong(outputWord * 2u);
                uint adjacent = uint(*((device const ushort *)(source + sourceOffset)));
                packed = (adjacent & 0xffu) | ((adjacent & 0xff00u) << 8u);
            }
            tile[local.y + frameDelta][local.x + wordDelta] = packed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint wordDelta = 0u; wordDelta < 32u; wordDelta += 16u) {
        uint outputWord = wordBase + local.y + wordDelta;
        for (uint frameDelta = 0u; frameDelta < 32u; frameDelta += 16u) {
            uint localFrame = frameBase + local.x + frameDelta;
            if (outputWord < outputDetectorWords && localFrame < params.frameCount) {
                uint destinationScan = params.destinationScanOffset + localFrame;
                destination[ulong(outputWord) * params.destinationScanCount + destinationScan] =
                    tile[local.x + frameDelta][local.y + wordDelta];
            }
        }
    }
}

// Exact 192x192 -> 96x96 detector-bin2 specialization. A 16x16 threadgroup
// fills and transposes a 32x32 packed-word tile so source reads and
// detector-word-major destination writes remain coalesced. No atomics or
// changed arithmetic order are used.
kernel void contiguous_detector_bin2_u8_to_u16_word_major_tiled32(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ContiguousDetectorBinParams &params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    threadgroup uint tile[32][33];
    uint wordBase = group.x * 32u;
    uint frameBase = group.y * 32u;
    uint outputDetectorPixels = params.outputDetectorRows * params.outputDetectorCols;
    uint outputDetectorWords = (outputDetectorPixels + 1u) / 2u;
    uint sourceDetectorPixels = params.sourceDetectorRows * params.sourceDetectorCols;

    for (uint frameDelta = 0u; frameDelta < 32u; frameDelta += 16u) {
        uint localFrame = frameBase + local.y + frameDelta;
        for (uint wordDelta = 0u; wordDelta < 32u; wordDelta += 16u) {
            uint outputWord = wordBase + local.x + wordDelta;
            uint packed = 0u;
            if (localFrame < params.frameCount && outputWord < outputDetectorWords) {
                uint firstOutputPixel = outputWord * 2u;
                uint outputRow = firstOutputPixel / 96u;
                uint outputColumn = firstOutputPixel - outputRow * 96u;
                ulong sourceOffset =
                    ulong(localFrame) * ulong(sourceDetectorPixels)
                    + ulong(outputRow * 2u * 192u + outputColumn * 2u);
                uint firstRow = *((device const uint *)(source + sourceOffset));
                uint secondRow = *((device const uint *)(source + sourceOffset + 192u));
                uint low =
                    (firstRow & 0xffu) + ((firstRow >> 8u) & 0xffu)
                    + (secondRow & 0xffu) + ((secondRow >> 8u) & 0xffu);
                uint high =
                    ((firstRow >> 16u) & 0xffu) + (firstRow >> 24u)
                    + ((secondRow >> 16u) & 0xffu) + (secondRow >> 24u);
                packed = low | (high << 16u);
            }
            tile[local.y + frameDelta][local.x + wordDelta] = packed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint wordDelta = 0u; wordDelta < 32u; wordDelta += 16u) {
        uint outputWord = wordBase + local.y + wordDelta;
        for (uint frameDelta = 0u; frameDelta < 32u; frameDelta += 16u) {
            uint localFrame = frameBase + local.x + frameDelta;
            if (outputWord < outputDetectorWords && localFrame < params.frameCount) {
                uint destinationScan = params.destinationScanOffset + localFrame;
                destination[ulong(outputWord) * params.destinationScanCount + destinationScan] =
                    tile[local.x + frameDelta][local.y + wordDelta];
            }
        }
    }
}

struct ContiguousBin2ProductsParams {
    uint frameCount;
    uint sourceDetectorRows;
    uint sourceDetectorCols;
    uint destinationScanCount;
    uint destinationScanOffset;
    uint globalFrameOffset;
    uint outputDetectorRows;
    uint outputDetectorCols;
};

struct ExactProductU32 {
    uint band1;
    uint band2;
    uint band4;
    uint total;
    uint rowMoment;
    uint columnMoment;
};

// Exact detector-bin1 counterpart of the fused bin2 path. Each source byte is
// read once to produce the packed uint16 resident volume, exact virtual-image
// and CoM numerators, and bounded 32-frame detector-sum partials.
kernel void contiguous_detector_bin1_u8_products_detector_partials_tiled32x8(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ContiguousBin2ProductsParams &params [[buffer(2)]],
    device const uchar *bands [[buffer(3)]],
    device uint *band1Map [[buffer(4)]],
    device uint *band2Map [[buffer(5)]],
    device uint *band4Map [[buffer(6)]],
    device uint *totalMap [[buffer(7)]],
    device uint *rowMomentMap [[buffer(8)]],
    device uint *columnMomentMap [[buffer(9)]],
    device ushort *detectorPartials [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint tile[32][65];
    threadgroup uint detectorTile[4][8][33];
    uint frameBase = group.x * 32u;
    uint sourceDetectorPixels =
        params.sourceDetectorRows * params.sourceDetectorCols;
    uint outputDetectorPixels =
        params.outputDetectorRows * params.outputDetectorCols;
    uint outputDetectorWords = (outputDetectorPixels + 1u) / 2u;
    ExactProductU32 products[4];
    for (uint slot = 0u; slot < 4u; ++slot) {
        products[slot] = ExactProductU32{0u, 0u, 0u, 0u, 0u, 0u};
    }

    for (uint wordBase = 0u; wordBase < outputDetectorWords; wordBase += 64u) {
        uint detectorPartial[4] = {0u, 0u, 0u, 0u};
        for (uint frameSlot = 0u; frameSlot < 4u; ++frameSlot) {
            uint localFrame = frameBase + local.y + frameSlot * 8u;
            for (uint wordLane = 0u; wordLane < 2u; ++wordLane) {
                uint tileWord = local.x + wordLane * 32u;
                uint outputWord = wordBase + tileWord;
                uint packed = 0u;
                if (localFrame < params.frameCount && outputWord < outputDetectorWords) {
                    uint firstPixel = outputWord * 2u;
                    ulong sourceOffset =
                        ulong(localFrame) * ulong(sourceDetectorPixels)
                        + ulong(firstPixel);
                    uint adjacent =
                        uint(*((device const ushort *)(source + sourceOffset)));
                    uint low = adjacent & 0xffu;
                    uint high = (adjacent >> 8u) & 0xffu;
                    packed = low | (high << 16u);
                    detectorPartial[wordLane * 2u] += low;
                    detectorPartial[wordLane * 2u + 1u] += high;

                    uint row = firstPixel / params.sourceDetectorCols;
                    uint column = firstPixel - row * params.sourceDetectorCols;
                    uchar lowMembership = bands[firstPixel];
                    uchar highMembership = bands[firstPixel + 1u];
                    if (lowMembership & 1u) products[frameSlot].band1 += low;
                    if (lowMembership & 2u) products[frameSlot].band2 += low;
                    if (lowMembership & 4u) products[frameSlot].band4 += low;
                    if (highMembership & 1u) products[frameSlot].band1 += high;
                    if (highMembership & 2u) products[frameSlot].band2 += high;
                    if (highMembership & 4u) products[frameSlot].band4 += high;
                    products[frameSlot].total += low + high;
                    products[frameSlot].rowMoment += (low + high) * row;
                    products[frameSlot].columnMoment +=
                        low * column + high * (column + 1u);
                }
                tile[local.y + frameSlot * 8u][tileWord] = packed;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint localFrame = frameBase + local.x;
        for (uint wordSlot = 0u; wordSlot < 8u; ++wordSlot) {
            uint outputWord = wordBase + local.y + wordSlot * 8u;
            if (localFrame < params.frameCount && outputWord < outputDetectorWords) {
                uint destinationScan = params.destinationScanOffset + localFrame;
                destination[
                    ulong(outputWord) * params.destinationScanCount + destinationScan
                ] = tile[local.x][local.y + wordSlot * 8u];
            }
        }
        for (uint valueIndex = 0u; valueIndex < 4u; ++valueIndex) {
            detectorTile[valueIndex][local.y][local.x] =
                detectorPartial[valueIndex];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local.y == 0u) {
            for (uint wordLane = 0u; wordLane < 2u; ++wordLane) {
                uint outputWord = wordBase + local.x + wordLane * 32u;
                if (outputWord < outputDetectorWords) {
                    uint low = 0u;
                    uint high = 0u;
                    for (uint frameLane = 0u; frameLane < 8u; ++frameLane) {
                        low += detectorTile[wordLane * 2u][frameLane][local.x];
                        high += detectorTile[
                            wordLane * 2u + 1u
                        ][frameLane][local.x];
                    }
                    uint firstPixel = outputWord * 2u;
                    detectorPartials[
                        ulong(group.x) * sourceDetectorPixels + firstPixel
                    ] = ushort(low);
                    detectorPartials[
                        ulong(group.x) * sourceDetectorPixels + firstPixel + 1u
                    ] = ushort(high);
                }
            }
        }
    }

    for (uint frameSlot = 0u; frameSlot < 4u; ++frameSlot) {
        uint localFrame = frameBase + local.y + frameSlot * 8u;
        uint band1 = simd_sum(products[frameSlot].band1);
        uint band2 = simd_sum(products[frameSlot].band2);
        uint band4 = simd_sum(products[frameSlot].band4);
        uint total = simd_sum(products[frameSlot].total);
        uint rowMoment = simd_sum(products[frameSlot].rowMoment);
        uint columnMoment = simd_sum(products[frameSlot].columnMoment);
        if (lane == 0u && localFrame < params.frameCount) {
            uint outputFrame = params.globalFrameOffset + localFrame;
            band1Map[outputFrame] = band1;
            band2Map[outputFrame] = band2;
            band4Map[outputFrame] = band4;
            totalMap[outputFrame] = total;
            rowMomentMap[outputFrame] = rowMoment;
            columnMomentMap[outputFrame] = columnMoment;
        }
    }
}

// The same exact tile additionally writes one uint16 detector sum per 32-frame
// group. A following coalesced reduction accumulates those bounded partials to
// the public uint64 mean-DP numerator, avoiding a second full staging read.
kernel void contiguous_detector_bin2_u8_products_detector_partials_tiled32x8(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ContiguousBin2ProductsParams &params [[buffer(2)]],
    device const uchar *bands [[buffer(3)]],
    device uint *band1Map [[buffer(4)]],
    device uint *band2Map [[buffer(5)]],
    device uint *band4Map [[buffer(6)]],
    device uint *totalMap [[buffer(7)]],
    device uint *rowMomentMap [[buffer(8)]],
    device uint *columnMomentMap [[buffer(9)]],
    device ushort *detectorPartials [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint tile[32][33];
    threadgroup uint detectorTile[8][8][33];
    uint frameBase = group.x * 32u;
    uint sourceDetectorPixels =
        params.sourceDetectorRows * params.sourceDetectorCols;
    uint outputDetectorPixels =
        params.outputDetectorRows * params.outputDetectorCols;
    uint outputDetectorWords = (outputDetectorPixels + 1u) / 2u;
    ExactProductU32 products[4];
    for (uint slot = 0u; slot < 4u; ++slot) {
        products[slot] = ExactProductU32{0u, 0u, 0u, 0u, 0u, 0u};
    }

    for (uint wordBase = 0u; wordBase < outputDetectorWords; wordBase += 32u) {
        uint detectorPartial[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        for (uint frameSlot = 0u; frameSlot < 4u; ++frameSlot) {
            uint localFrame = frameBase + local.y + frameSlot * 8u;
            uint outputWord = wordBase + local.x;
            uint packed = 0u;
            if (localFrame < params.frameCount && outputWord < outputDetectorWords) {
                uint firstOutputPixel = outputWord * 2u;
                uint outputRow = firstOutputPixel / 96u;
                uint outputColumn = firstOutputPixel - outputRow * 96u;
                uint sourceRow = outputRow * 2u;
                uint sourceColumn = outputColumn * 2u;
                uint firstPixel = sourceRow * 192u + sourceColumn;
                ulong sourceOffset =
                    ulong(localFrame) * ulong(sourceDetectorPixels)
                    + ulong(firstPixel);
                uint firstRow = *((device const uint *)(source + sourceOffset));
                uint secondRow = *((device const uint *)(source + sourceOffset + 192u));
                uint values[8] = {
                    firstRow & 0xffu,
                    (firstRow >> 8u) & 0xffu,
                    (firstRow >> 16u) & 0xffu,
                    firstRow >> 24u,
                    secondRow & 0xffu,
                    (secondRow >> 8u) & 0xffu,
                    (secondRow >> 16u) & 0xffu,
                    secondRow >> 24u,
                };
                uint low = values[0] + values[1] + values[4] + values[5];
                uint high = values[2] + values[3] + values[6] + values[7];
                packed = low | (high << 16u);

                for (uint valueIndex = 0u; valueIndex < 8u; ++valueIndex) {
                    uint row = sourceRow + valueIndex / 4u;
                    uint column = sourceColumn + valueIndex % 4u;
                    uint pixel = row * 192u + column;
                    uint value = values[valueIndex];
                    detectorPartial[valueIndex] += value;
                    uchar membership = bands[pixel];
                    if (membership & 1u) products[frameSlot].band1 += value;
                    if (membership & 2u) products[frameSlot].band2 += value;
                    if (membership & 4u) products[frameSlot].band4 += value;
                    products[frameSlot].total += value;
                    products[frameSlot].rowMoment += value * row;
                    products[frameSlot].columnMoment += value * column;
                }
            }
            tile[local.y + frameSlot * 8u][local.x] = packed;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint localFrame = frameBase + local.x;
        for (uint wordSlot = 0u; wordSlot < 4u; ++wordSlot) {
            uint outputWord = wordBase + local.y + wordSlot * 8u;
            if (localFrame < params.frameCount && outputWord < outputDetectorWords) {
                uint destinationScan = params.destinationScanOffset + localFrame;
                destination[
                    ulong(outputWord) * params.destinationScanCount + destinationScan
                ] = tile[local.x][local.y + wordSlot * 8u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint valueIndex = 0u; valueIndex < 8u; ++valueIndex) {
            detectorTile[valueIndex][local.y][local.x] =
                detectorPartial[valueIndex];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local.y == 0u) {
            uint outputWord = wordBase + local.x;
            if (outputWord < outputDetectorWords) {
                uint firstOutputPixel = outputWord * 2u;
                uint outputRow = firstOutputPixel / 96u;
                uint outputColumn = firstOutputPixel - outputRow * 96u;
                uint sourceRow = outputRow * 2u;
                uint sourceColumn = outputColumn * 2u;
                for (uint valueIndex = 0u; valueIndex < 8u; ++valueIndex) {
                    uint partial = 0u;
                    for (uint frameLane = 0u; frameLane < 8u; ++frameLane) {
                        partial += detectorTile[valueIndex][frameLane][local.x];
                    }
                    uint row = sourceRow + valueIndex / 4u;
                    uint column = sourceColumn + valueIndex % 4u;
                    uint pixel = row * 192u + column;
                    detectorPartials[ulong(group.x) * sourceDetectorPixels + pixel] =
                        ushort(partial);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint frameSlot = 0u; frameSlot < 4u; ++frameSlot) {
        uint localFrame = frameBase + local.y + frameSlot * 8u;
        uint band1 = simd_sum(products[frameSlot].band1);
        uint band2 = simd_sum(products[frameSlot].band2);
        uint band4 = simd_sum(products[frameSlot].band4);
        uint total = simd_sum(products[frameSlot].total);
        uint rowMoment = simd_sum(products[frameSlot].rowMoment);
        uint columnMoment = simd_sum(products[frameSlot].columnMoment);
        if (lane == 0u && localFrame < params.frameCount) {
            uint outputFrame = params.globalFrameOffset + localFrame;
            band1Map[outputFrame] = band1;
            band2Map[outputFrame] = band2;
            band4Map[outputFrame] = band4;
            totalMap[outputFrame] = total;
            rowMomentMap[outputFrame] = rowMoment;
            columnMomentMap[outputFrame] = columnMoment;
        }
    }
}

// Exact detector-bin4 specialization for native detector-192 data. Four
// source rows are folded into each output row while the same source bytes also
// produce exact products and bounded 32-frame detector-sum partials. The
// detector tile is reused one source row at a time to stay within Apple GPU
// threadgroup-memory limits.
kernel void contiguous_detector_bin4_u8_products_detector_partials_tiled32x8(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ContiguousBin2ProductsParams &params [[buffer(2)]],
    device const uchar *bands [[buffer(3)]],
    device uint *band1Map [[buffer(4)]],
    device uint *band2Map [[buffer(5)]],
    device uint *band4Map [[buffer(6)]],
    device uint *totalMap [[buffer(7)]],
    device uint *rowMomentMap [[buffer(8)]],
    device uint *columnMomentMap [[buffer(9)]],
    device ushort *detectorPartials [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint tile[32][33];
    threadgroup uint detectorTile[8][8][33];
    uint frameBase = group.x * 32u;
    uint sourceDetectorPixels =
        params.sourceDetectorRows * params.sourceDetectorCols;
    uint outputDetectorPixels =
        params.outputDetectorRows * params.outputDetectorCols;
    uint outputDetectorWords = (outputDetectorPixels + 1u) / 2u;
    ExactProductU32 products[4];
    for (uint slot = 0u; slot < 4u; ++slot) {
        products[slot] = ExactProductU32{0u, 0u, 0u, 0u, 0u, 0u};
    }

    for (uint wordBase = 0u; wordBase < outputDetectorWords; wordBase += 32u) {
        uint lowSums[4] = {0u, 0u, 0u, 0u};
        uint highSums[4] = {0u, 0u, 0u, 0u};
        uint outputWord = wordBase + local.x;
        uint firstOutputPixel = outputWord * 2u;
        uint outputRow = firstOutputPixel / params.outputDetectorCols;
        uint outputColumn =
            firstOutputPixel - outputRow * params.outputDetectorCols;
        uint sourceRow = outputRow * 4u;
        uint sourceColumn = outputColumn * 4u;

        for (uint sourceRowDelta = 0u; sourceRowDelta < 4u; ++sourceRowDelta) {
            uint detectorPartial[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
            for (uint frameSlot = 0u; frameSlot < 4u; ++frameSlot) {
                uint localFrame = frameBase + local.y + frameSlot * 8u;
                if (localFrame < params.frameCount && outputWord < outputDetectorWords) {
                    uint row = sourceRow + sourceRowDelta;
                    ulong sourceOffset =
                        ulong(localFrame) * ulong(sourceDetectorPixels)
                        + ulong(row * params.sourceDetectorCols + sourceColumn);
                    ulong packedRow =
                        *((device const ulong *)(source + sourceOffset));
                    for (uint valueIndex = 0u; valueIndex < 8u; ++valueIndex) {
                        uint value = uint(
                            (packedRow >> (valueIndex * 8u)) & ulong(0xffu)
                        );
                        uint column = sourceColumn + valueIndex;
                        uint pixel = row * params.sourceDetectorCols + column;
                        detectorPartial[valueIndex] += value;
                        if (valueIndex < 4u) {
                            lowSums[frameSlot] += value;
                        } else {
                            highSums[frameSlot] += value;
                        }
                        uchar membership = bands[pixel];
                        if (membership & 1u) products[frameSlot].band1 += value;
                        if (membership & 2u) products[frameSlot].band2 += value;
                        if (membership & 4u) products[frameSlot].band4 += value;
                        products[frameSlot].total += value;
                        products[frameSlot].rowMoment += value * row;
                        products[frameSlot].columnMoment += value * column;
                    }
                }
            }

            for (uint valueIndex = 0u; valueIndex < 8u; ++valueIndex) {
                detectorTile[valueIndex][local.y][local.x] =
                    detectorPartial[valueIndex];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (local.y == 0u && outputWord < outputDetectorWords) {
                uint row = sourceRow + sourceRowDelta;
                for (uint valueIndex = 0u; valueIndex < 8u; ++valueIndex) {
                    uint partial = 0u;
                    for (uint frameLane = 0u; frameLane < 8u; ++frameLane) {
                        partial += detectorTile[valueIndex][frameLane][local.x];
                    }
                    uint pixel =
                        row * params.sourceDetectorCols + sourceColumn + valueIndex;
                    detectorPartials[
                        ulong(group.x) * sourceDetectorPixels + pixel
                    ] = ushort(partial);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint frameSlot = 0u; frameSlot < 4u; ++frameSlot) {
            tile[local.y + frameSlot * 8u][local.x] =
                lowSums[frameSlot] | (highSums[frameSlot] << 16u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint localFrame = frameBase + local.x;
        for (uint wordSlot = 0u; wordSlot < 4u; ++wordSlot) {
            uint transposedWord = wordBase + local.y + wordSlot * 8u;
            if (localFrame < params.frameCount
                && transposedWord < outputDetectorWords) {
                uint destinationScan = params.destinationScanOffset + localFrame;
                destination[
                    ulong(transposedWord) * params.destinationScanCount
                    + destinationScan
                ] = tile[local.x][local.y + wordSlot * 8u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint frameSlot = 0u; frameSlot < 4u; ++frameSlot) {
        uint localFrame = frameBase + local.y + frameSlot * 8u;
        uint band1 = simd_sum(products[frameSlot].band1);
        uint band2 = simd_sum(products[frameSlot].band2);
        uint band4 = simd_sum(products[frameSlot].band4);
        uint total = simd_sum(products[frameSlot].total);
        uint rowMoment = simd_sum(products[frameSlot].rowMoment);
        uint columnMoment = simd_sum(products[frameSlot].columnMoment);
        if (lane == 0u && localFrame < params.frameCount) {
            uint outputFrame = params.globalFrameOffset + localFrame;
            band1Map[outputFrame] = band1;
            band2Map[outputFrame] = band2;
            band4Map[outputFrame] = band4;
            totalMap[outputFrame] = total;
            rowMomentMap[outputFrame] = rowMoment;
            columnMomentMap[outputFrame] = columnMoment;
        }
    }
}

kernel void detector_accumulate_u16_partials_u64(
    device const ushort *partials [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant uint &detectorPixels [[buffer(2)]],
    constant uint &partialCount [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= detectorPixels) return;
    ulong sum = 0ul;
    for (uint partial = 0u; partial < partialCount; ++partial) {
        sum += ulong(partials[ulong(partial) * detectorPixels + pixel]);
    }
    output[pixel] += sum;
}

struct WordMajorDetectorParams {
    uint scanCount;
    uint detectorPixels;
};

inline uint word_major_u8_sample(
    device const uint *data, uint pixel, uint scan, uint scanCount
) {
    uint word = data[ulong(pixel >> 2u) * scanCount + scan];
    return (word >> ((pixel & 3u) * 8u)) & 0xffu;
}

inline uint word_major_u16_sample(
    device const uint *data, uint pixel, uint scan, uint scanCount
) {
    uint word = data[ulong(pixel >> 1u) * scanCount + scan];
    return (word >> ((pixel & 1u) * 16u)) & 0xffffu;
}

inline uint word_major_u32_sample(
    device const uint *data, uint pixel, uint scan, uint scanCount
) {
    return data[ulong(pixel) * scanCount + scan];
}

struct ResidentRebinParams {
    uint sourceRows;
    uint sourceCols;
    uint sourceScanCount;
    uint sourceDetectorRows;
    uint sourceDetectorCols;
    uint sourceRowOffset;
    uint sourceColOffset;
    uint selectedRows;
    uint selectedCols;
    uint scanBin;
    uint detectorBin;
    uint outputRows;
    uint outputCols;
    uint outputScanCount;
    uint outputDetectorRows;
    uint outputDetectorCols;
};

template <uint StorageBits>
inline void residentRebinToU32WordMajor(
    device const uint *source,
    device uint *destination,
    constant ResidentRebinParams &params,
    uint2 position
) {
    uint outputDetectorPixel = position.x;
    uint outputScan = position.y;
    uint outputDetectorPixels = params.outputDetectorRows * params.outputDetectorCols;
    if (outputDetectorPixel >= outputDetectorPixels ||
        outputScan >= params.outputScanCount) return;

    uint outputRow = outputScan / params.outputCols;
    uint outputCol = outputScan - outputRow * params.outputCols;
    uint selectedRowStop = params.sourceRowOffset + params.selectedRows;
    uint selectedColStop = params.sourceColOffset + params.selectedCols;
    uint sourceRowStart = params.sourceRowOffset + outputRow * params.scanBin;
    uint sourceColStart = params.sourceColOffset + outputCol * params.scanBin;
    uint sourceRowStop = min(selectedRowStop, sourceRowStart + params.scanBin);
    uint sourceColStop = min(selectedColStop, sourceColStart + params.scanBin);
    uint outputDetectorRow = outputDetectorPixel / params.outputDetectorCols;
    uint outputDetectorCol =
        outputDetectorPixel - outputDetectorRow * params.outputDetectorCols;
    uint sourceDetectorRowStart = outputDetectorRow * params.detectorBin;
    uint sourceDetectorColStart = outputDetectorCol * params.detectorBin;
    uint sourceDetectorRowStop = min(
        params.sourceDetectorRows, sourceDetectorRowStart + params.detectorBin
    );
    uint sourceDetectorColStop = min(
        params.sourceDetectorCols, sourceDetectorColStart + params.detectorBin
    );
    uint sum = 0u;
    for (uint row = sourceRowStart; row < sourceRowStop; ++row) {
        for (uint col = sourceColStart; col < sourceColStop; ++col) {
            uint scan = row * params.sourceCols + col;
            for (uint detectorRow = sourceDetectorRowStart;
                 detectorRow < sourceDetectorRowStop; ++detectorRow) {
                for (uint detectorCol = sourceDetectorColStart;
                     detectorCol < sourceDetectorColStop; ++detectorCol) {
                    uint sourceDetectorPixel =
                        detectorRow * params.sourceDetectorCols + detectorCol;
                    if constexpr (StorageBits == 8u) {
                        sum += word_major_u8_sample(
                            source, sourceDetectorPixel, scan, params.sourceScanCount
                        );
                    } else if constexpr (StorageBits == 16u) {
                        sum += word_major_u16_sample(
                            source, sourceDetectorPixel, scan, params.sourceScanCount
                        );
                    } else {
                        sum += word_major_u32_sample(
                            source, sourceDetectorPixel, scan, params.sourceScanCount
                        );
                    }
                }
            }
        }
    }
    destination[ulong(outputDetectorPixel) * params.outputScanCount + outputScan] = sum;
}

#define DEFINE_RESIDENT_REBIN_KERNEL(NAME, BITS)                                    \
kernel void NAME(                                                                    \
    device const uint *source [[buffer(0)]],                                         \
    device uint *destination [[buffer(1)]],                                          \
    constant ResidentRebinParams &params [[buffer(2)]],                              \
    uint2 position [[thread_position_in_grid]]                                       \
) {                                                                                  \
    residentRebinToU32WordMajor<BITS>(source, destination, params, position);         \
}

// Rebin an already resident detector-word-major volume without a source-file
// reread. The destination remains an exact uint32 sum, including edge bins.
DEFINE_RESIDENT_REBIN_KERNEL(resident_rebin_u8_word_major_to_u32_word_major, 8u)
DEFINE_RESIDENT_REBIN_KERNEL(resident_rebin_u16_word_major_to_u32_word_major, 16u)
DEFINE_RESIDENT_REBIN_KERNEL(resident_rebin_u32_word_major_to_u32_word_major, 32u)

template <uint StorageBits>
inline void detectorCenterOfMass(
    device const uint *data,
    device float *rowOutput,
    device float *columnOutput,
    constant WordMajorDetectorParams &params,
    uint detectorColumns,
    threadgroup ulong *totalScratch,
    threadgroup ulong *rowScratch,
    threadgroup ulong *columnScratch,
    uint scan,
    uint threadIndex,
    uint threads
) {
    if (scan >= params.scanCount) return;
    ulong total = 0ul;
    ulong rowMoment = 0ul;
    ulong columnMoment = 0ul;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value;
        if constexpr (StorageBits == 8u) {
            value = word_major_u8_sample(data, pixel, scan, params.scanCount);
        } else if constexpr (StorageBits == 16u) {
            value = word_major_u16_sample(data, pixel, scan, params.scanCount);
        } else {
            value = word_major_u32_sample(data, pixel, scan, params.scanCount);
        }
        ulong wide = ulong(value);
        uint row = pixel / detectorColumns;
        uint column = pixel - row * detectorColumns;
        total += wide;
        rowMoment += wide * ulong(row);
        columnMoment += wide * ulong(column);
    }
    totalScratch[threadIndex] = total;
    rowScratch[threadIndex] = rowMoment;
    columnScratch[threadIndex] = columnMoment;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threads >> 1u; offset > 0u; offset >>= 1u) {
        if (threadIndex < offset) {
            totalScratch[threadIndex] += totalScratch[threadIndex + offset];
            rowScratch[threadIndex] += rowScratch[threadIndex + offset];
            columnScratch[threadIndex] += columnScratch[threadIndex + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (threadIndex == 0u) {
        float inverse = totalScratch[0] > 0ul ? 1.0f / float(totalScratch[0]) : 0.0f;
        rowOutput[scan] = float(rowScratch[0]) * inverse;
        columnOutput[scan] = float(columnScratch[0]) * inverse;
    }
}

#define DEFINE_WORD_MAJOR_COM_KERNEL(NAME, BITS)                                      \
kernel void NAME(                                                                      \
    device const uint *data [[buffer(0)]],                                             \
    device float *rowOutput [[buffer(1)]],                                             \
    device float *columnOutput [[buffer(2)]],                                          \
    constant WordMajorDetectorParams &params [[buffer(3)]],                            \
    constant uint &detectorColumns [[buffer(4)]],                                      \
    uint scan [[threadgroup_position_in_grid]],                                        \
    uint threadIndex [[thread_index_in_threadgroup]],                                  \
    uint threads [[threads_per_threadgroup]]                                           \
) {                                                                                    \
    threadgroup ulong totalScratch[256];                                               \
    threadgroup ulong rowScratch[256];                                                 \
    threadgroup ulong columnScratch[256];                                              \
    detectorCenterOfMass<BITS>(                                                        \
        data, rowOutput, columnOutput, params, detectorColumns,                        \
        totalScratch, rowScratch, columnScratch, scan, threadIndex, threads            \
    );                                                                                 \
}

DEFINE_WORD_MAJOR_COM_KERNEL(center_of_mass_u8_word_major, 8u)
DEFINE_WORD_MAJOR_COM_KERNEL(center_of_mass_u16_word_major, 16u)
DEFINE_WORD_MAJOR_COM_KERNEL(center_of_mass_u32_word_major, 32u)

// Convert exact integer moments accumulated while decoding into the same
// float center-of-mass representation as the standalone word-major kernels.
kernel void center_of_mass_u32_moments(
    device const uint *total [[buffer(0)]],
    device const uint *rowMoment [[buffer(1)]],
    device const uint *columnMoment [[buffer(2)]],
    device float *rowOutput [[buffer(3)]],
    device float *columnOutput [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    float inverse = total[index] > 0u ? 1.0f / float(total[index]) : 0.0f;
    rowOutput[index] = float(rowMoment[index]) * inverse;
    columnOutput[index] = float(columnMoment[index]) * inverse;
}

kernel void center_of_mass_u64_moments(
    device const ulong *total [[buffer(0)]],
    device const ulong *rowMoment [[buffer(1)]],
    device const ulong *columnMoment [[buffer(2)]],
    device float *rowOutput [[buffer(3)]],
    device float *columnOutput [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    float inverse = total[index] > 0ul ? 1.0f / float(total[index]) : 0.0f;
    rowOutput[index] = float(rowMoment[index]) * inverse;
    columnOutput[index] = float(columnMoment[index]) * inverse;
}

// Preserve exact fused-load accumulators in the uint64 summary schema without
// traversing the resident 4D volume a second time. The caller must first prove
// that the fused uint32 accumulation is safe for the selected source and load
// geometry.
kernel void widen_u32_accumulator_triplet_to_u64(
    device const uint *firstInput [[buffer(0)]],
    device const uint *secondInput [[buffer(1)]],
    device const uint *thirdInput [[buffer(2)]],
    device ulong *firstOutput [[buffer(3)]],
    device ulong *secondOutput [[buffer(4)]],
    device ulong *thirdOutput [[buffer(5)]],
    constant uint &count [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    firstOutput[index] = ulong(firstInput[index]);
    secondOutput[index] = ulong(secondInput[index]);
    thirdOutput[index] = ulong(thirdInput[index]);
}


// Produce BF/ABF/ADF from an exact uint32 scan-binned cache. One threadgroup
// owns each output scan position, matching the native detector reduction.
kernel void detector_products_u32_word_major(
    device const uint *data [[buffer(0)]],
    device uint *bfMap [[buffer(1)]],
    device uint *abfMap [[buffer(2)]],
    device uint *dfMap [[buffer(3)]],
    constant WordMajorDetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    uint scan [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (scan >= params.scanCount) return;
    threadgroup atomic_uint sums[3];
    if (threadIndex < 3) {
        atomic_store_explicit(&sums[threadIndex], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bf = 0;
    uint abf = 0;
    uint df = 0;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value = data[ulong(pixel) * params.scanCount + scan];
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
    bfMap[scan] = atomic_load_explicit(&sums[0], memory_order_relaxed);
    abfMap[scan] = atomic_load_explicit(&sums[1], memory_order_relaxed);
    dfMap[scan] = atomic_load_explicit(&sums[2], memory_order_relaxed);
}

// Produce BF/ABF/ADF from two packed uint16 detector pixels per word. The
// layout is detector-word-major: each word row spans all scan positions.
kernel void detector_products_u16_word_major(
    device const uint *data [[buffer(0)]],
    device uint *bfMap [[buffer(1)]],
    device uint *abfMap [[buffer(2)]],
    device uint *dfMap [[buffer(3)]],
    constant WordMajorDetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    uint scan [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (scan >= params.scanCount) return;
    threadgroup atomic_uint sums[3];
    if (threadIndex < 3) {
        atomic_store_explicit(&sums[threadIndex], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bf = 0;
    uint abf = 0;
    uint df = 0;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value = word_major_u16_sample(data, pixel, scan, params.scanCount);
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
    bfMap[scan] = atomic_load_explicit(&sums[0], memory_order_relaxed);
    abfMap[scan] = atomic_load_explicit(&sums[1], memory_order_relaxed);
    dfMap[scan] = atomic_load_explicit(&sums[2], memory_order_relaxed);
}

// Produce exact virtual-detector sums and 64-bit center-of-mass moments while
// traversing a packed uint16 resident cache once. This is the reusable summary
// path for fast, provenance-bound cache reopen.
kernel void detector_products_u16_word_major_with_u64_moments(
    device const uint *data [[buffer(0)]],
    device uint *bfMap [[buffer(1)]],
    device uint *abfMap [[buffer(2)]],
    device uint *dfMap [[buffer(3)]],
    constant WordMajorDetectorParams &params [[buffer(4)]],
    device const uchar *bands [[buffer(5)]],
    device ulong *totalMap [[buffer(6)]],
    device ulong *rowMomentMap [[buffer(7)]],
    device ulong *columnMomentMap [[buffer(8)]],
    constant uint &detectorColumns [[buffer(9)]],
    uint scan [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (scan >= params.scanCount) return;
    threadgroup atomic_uint bandSums[3];
    threadgroup ulong totalScratch[256];
    threadgroup ulong rowScratch[256];
    threadgroup ulong columnScratch[256];
    if (threadIndex < 3u) {
        atomic_store_explicit(&bandSums[threadIndex], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint bf = 0u;
    uint abf = 0u;
    uint df = 0u;
    ulong total = 0ul;
    ulong rowMoment = 0ul;
    ulong columnMoment = 0ul;
    for (uint pixel = threadIndex; pixel < params.detectorPixels; pixel += threads) {
        uint value = word_major_u16_sample(data, pixel, scan, params.scanCount);
        uchar band = bands[pixel];
        if (band & 1u) bf += value;
        if (band & 2u) abf += value;
        if (band & 4u) df += value;
        uint row = pixel / detectorColumns;
        uint column = pixel - row * detectorColumns;
        ulong wide = ulong(value);
        total += wide;
        rowMoment += wide * ulong(row);
        columnMoment += wide * ulong(column);
    }
    bf = simd_sum(bf);
    abf = simd_sum(abf);
    df = simd_sum(df);
    if (lane == 0u) {
        atomic_fetch_add_explicit(&bandSums[0], bf, memory_order_relaxed);
        atomic_fetch_add_explicit(&bandSums[1], abf, memory_order_relaxed);
        atomic_fetch_add_explicit(&bandSums[2], df, memory_order_relaxed);
    }
    totalScratch[threadIndex] = total;
    rowScratch[threadIndex] = rowMoment;
    columnScratch[threadIndex] = columnMoment;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threads >> 1u; offset > 0u; offset >>= 1u) {
        if (threadIndex < offset) {
            totalScratch[threadIndex] += totalScratch[threadIndex + offset];
            rowScratch[threadIndex] += rowScratch[threadIndex + offset];
            columnScratch[threadIndex] += columnScratch[threadIndex + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (threadIndex == 0u) {
        bfMap[scan] = atomic_load_explicit(&bandSums[0], memory_order_relaxed);
        abfMap[scan] = atomic_load_explicit(&bandSums[1], memory_order_relaxed);
        dfMap[scan] = atomic_load_explicit(&bandSums[2], memory_order_relaxed);
        totalMap[scan] = totalScratch[0];
        rowMomentMap[scan] = rowScratch[0];
        columnMomentMap[scan] = columnScratch[0];
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

// Display-facing extraction always expands exact integer counts to uint32.
// This keeps one stable Metal surface across source uint8/uint16 and exact
// uint32 scan-sum data, avoiding an NSImage allocation on every scan drag.
kernel void extract_u8_word_major_frame_to_u32(
    device const uint *data [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &scanIndex [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= pixelCount) return;
    uint word = data[ulong(pixel >> 2u) * scanCount + scanIndex];
    output[pixel] = (word >> ((pixel & 3u) * 8u)) & 0xffu;
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

// Convert a bounded batch from the private detector-word-major resident layout
// into the backend-neutral logical pixel order used by parity evidence. The
// output is frame-major, with detector column varying fastest. Odd detector
// counts intentionally omit the unused high lane of the final packed word.
kernel void extract_u16_word_major_frames(
    device const uint *data [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant uint &scanStart [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint pixel = position.x;
    uint localScan = position.y;
    if (pixel >= pixelCount || scanStart + localScan >= scanCount) return;
    uint word = data[ulong(pixel >> 1u) * scanCount + scanStart + localScan];
    output[ulong(localScan) * pixelCount + pixel] =
        ushort((word >> ((pixel & 1u) * 16u)) & 0xffffu);
}

kernel void extract_u16_word_major_frame_to_u32(
    device const uint *data [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &scanIndex [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= pixelCount) return;
    uint word = data[ulong(pixel >> 1u) * scanCount + scanIndex];
    output[pixel] = (word >> ((pixel & 1u) * 16u)) & 0xffffu;
}

inline long signed_u32_word(uint word, uint coefficients) {
    uint coefficient = coefficients & 3u;
    if (coefficient == 1u) return long(word);
    if (coefficient == 2u) return -long(word);
    return 0;
}

kernel void full_sum_u32_word_major(
    device const uint *data [[buffer(0)]],
    device const uint2 *entries [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &entryCount [[buffer(4)]],
    uint scan [[thread_position_in_grid]]
) {
    if (scan >= scanCount) return;
    long sum = 0;
    for (uint entry = 0; entry < entryCount; ++entry) {
        uint2 spec = entries[entry];
        uint word = data[ulong(spec.x) * scanCount + scan];
        sum += signed_u32_word(word, spec.y);
    }
    output[scan] = uint(sum);
}

kernel void signed_delta_u32_word_major(
    device const uint *data [[buffer(0)]],
    device const uint2 *entries [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &entryCount [[buffer(4)]],
    uint scan [[thread_position_in_grid]]
) {
    if (scan >= scanCount) return;
    long delta = 0;
    for (uint entry = 0; entry < entryCount; ++entry) {
        uint2 spec = entries[entry];
        uint word = data[ulong(spec.x) * scanCount + scan];
        delta += signed_u32_word(word, spec.y);
    }
    output[scan] = uint(long(output[scan]) + delta);
}

kernel void extract_u32_word_major_frame(
    device const uint *data [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &scanIndex [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= pixelCount) return;
    output[pixel] = data[ulong(pixel) * scanCount + scanIndex];
}

kernel void extract_u32_word_major_frames(
    device const uint *data [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &scanStart [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint pixel = position.x;
    uint localScan = position.y;
    if (pixel >= pixelCount || scanStart + localScan >= scanCount) return;
    output[ulong(localScan) * pixelCount + pixel] =
        data[ulong(pixel) * scanCount + scanStart + localScan];
}

kernel void extract_u32_word_major_frame_to_u32(
    device const uint *data [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &scanIndex [[buffer(2)]],
    constant uint &scanCount [[buffer(3)]],
    constant uint &pixelCount [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]
) {
    if (pixel >= pixelCount) return;
    output[pixel] = data[ulong(pixel) * scanCount + scanIndex];
}

struct ScanRegionSumParams {
    uint scanRows;
    uint scanColumns;
    uint scanCount;
    uint detectorPixels;
    float centerRow;
    float centerColumn;
    float radius;
    uint shape;  // 0 = circle, 1 = square
    uint reduction;  // 0 = sum/mean numerator, 1 = maximum
};

template <uint StorageBits>
inline void scanRegionSumToU32(
    device const uint *data,
    device uint *output,
    constant ScanRegionSumParams &params,
    uint pixel
) {
    if (pixel >= params.detectorPixels) return;
    int rowStart = max(0, int(ceil(params.centerRow - params.radius)));
    int rowStop = min(int(params.scanRows) - 1, int(floor(params.centerRow + params.radius)));
    int columnStart = max(0, int(ceil(params.centerColumn - params.radius)));
    int columnStop = min(
        int(params.scanColumns) - 1,
        int(floor(params.centerColumn + params.radius))
    );
    ulong sum = 0ul;
    float radiusSquared = params.radius * params.radius;
    for (int row = rowStart; row <= rowStop; ++row) {
        float rowOffset = float(row) - params.centerRow;
        for (int column = columnStart; column <= columnStop; ++column) {
            float columnOffset = float(column) - params.centerColumn;
            if (params.shape == 0u &&
                rowOffset * rowOffset + columnOffset * columnOffset > radiusSquared) {
                continue;
            }
            uint scan = uint(row) * params.scanColumns + uint(column);
            ulong value;
            if constexpr (StorageBits == 8u) {
                value = ulong(word_major_u8_sample(data, pixel, scan, params.scanCount));
            } else if constexpr (StorageBits == 16u) {
                value = ulong(word_major_u16_sample(data, pixel, scan, params.scanCount));
            } else {
                value = ulong(word_major_u32_sample(data, pixel, scan, params.scanCount));
            }
            sum = params.reduction == 1u ? max(sum, value) : sum + value;
        }
    }
    output[pixel] = uint(sum);
}

#define DEFINE_SCAN_REGION_SUM_KERNEL(NAME, BITS)                                   \
kernel void NAME(                                                                    \
    device const uint *data [[buffer(0)]],                                           \
    device uint *output [[buffer(1)]],                                               \
    constant ScanRegionSumParams &params [[buffer(2)]],                              \
    uint pixel [[thread_position_in_grid]]                                           \
) {                                                                                  \
    scanRegionSumToU32<BITS>(data, output, params, pixel);                           \
}

// Sum/mean or maximize one circle/square directly from detector-major
// residency. Sum/mean accumulation remains uint64; max stays integer-exact.
DEFINE_SCAN_REGION_SUM_KERNEL(scan_region_sum_u8_word_major_to_u32, 8u)
DEFINE_SCAN_REGION_SUM_KERNEL(scan_region_sum_u16_word_major_to_u32, 16u)
DEFINE_SCAN_REGION_SUM_KERNEL(scan_region_sum_u32_word_major_to_u32, 32u)
