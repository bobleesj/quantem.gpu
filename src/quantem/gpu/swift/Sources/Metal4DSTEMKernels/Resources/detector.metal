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

// Exact compact resident path for an audited uint8 staging range.  Callers
// must prove that the detector-bin contribution bound fits uint16 and expose
// the actual resident dtype in provenance; otherwise they use the uint32 path.
kernel void scan_detector_bin_u8_to_u16_word_major(
    device const uchar *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
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

kernel void scan_detector_bin_u16_to_u32_word_major(
    device const ushort *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant ScanDetectorBinParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    scanDetectorBinToU32WordMajor(source, destination, params, position);
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
