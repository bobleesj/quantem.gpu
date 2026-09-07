// Exact sums for complete detector regions. Original evidence stays resident;
// detector boundaries always use the original counts.
struct CompactDetectorRegionsParameters {
    uint scans;
    uint tiles;
    uint columns;
    uint blocks;
    uint headerWords;
    uint headerEncoding;
    uint payloadWords;
    uint blockSide;
    uint maximumWidth;
};

kernel void compact_detector_regions_sum(
    device const uint *payload [[buffer(0)]],
    device const uint *headers [[buffer(1)]],
    device uint *sums [[buffer(2)]],
    constant CompactDetectorRegionsParameters &p [[buffer(3)]],
    uint linear [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint block = linear / p.scans;
    uint scan = linear % p.scans;
    if (block >= p.blocks) return;
    uint row = (block / (p.columns / p.blockSide)) * p.blockSide;
    uint col = (block % (p.columns / p.blockSide)) * p.blockSide;
    uint sum = 0u;
    for (uint r = 0u; r < p.blockSide; ++r) {
        for (uint c = 0u; c < p.blockSide; ++c) {
            uint descriptor = 0u;
            if (lane == 0u) {
                descriptor = compactDescriptorFor(headers, p.tiles, p.headerWords,
                    p.headerEncoding, (row + r) * p.columns + col + c, scan / 32u);
            }
            descriptor = simd_broadcast_first(descriptor);
            sum += compactCellValue(payload, descriptor, scan % 32u, 0u);
        }
    }
    // Eight-by-eight regions sum to at most 64 * 65535, exactly 22 bits.
    sums[linear] = sum;
}

kernel void compact_detector_regions_widths(
    device const uint *sums [[buffer(0)]],
    device uint *descriptors [[buffer(1)]],
    constant CompactDetectorRegionsParameters &p [[buffer(2)]],
    uint cell [[thread_position_in_grid]]
) {
    if (cell >= p.blocks * p.tiles) return;
    uint maximum = 0u;
    for (uint scan = 0u; scan < 32u; ++scan) maximum = max(maximum, sums[cell * 32u + scan]);
    descriptors[cell] = 32u - clz(maximum);
}

kernel void compact_detector_regions_prefix(
    device uint *descriptors [[buffer(0)]],
    device uint *totalWords [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant CompactDetectorRegionsParameters &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0u) return;
    uint offset = 0u;
    for (uint cell = 0u; cell < p.blocks * p.tiles; ++cell) {
        uint width = descriptors[cell];
        if (width > p.maximumWidth || offset >= (1u << 27u) || width >= (1u << 27u) - offset) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            return;
        }
        descriptors[cell] = (offset << 5u) | width;
        offset += width;
    }
    totalWords[0] = offset;
}

kernel void compact_detector_regions_pack(
    device const uint *sums [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device uint *payload [[buffer(2)]],
    constant CompactDetectorRegionsParameters &p [[buffer(3)]],
    uint cell [[thread_position_in_grid]]
) {
    if (cell >= p.blocks * p.tiles) return;
    uint descriptor = descriptors[cell];
    uint width = descriptor & 31u;
    uint offset = descriptor >> 5u;
    // The completed prefix command and host checked every width and total.
    for (uint word = 0u; word < width; ++word) {
        uint result = 0u;
        uint firstBit = word * 32u;
        uint firstScan = firstBit / width;
        uint lastScan = min(31u, (firstBit + 31u) / width);
        for (uint scan = firstScan; scan <= lastScan; ++scan) {
            uint bit = scan * width;
            uint value = sums[cell * 32u + scan];
            result |= bit >= firstBit ? value << (bit - firstBit) : value >> (firstBit - bit);
        }
        payload[offset + word] = result;
    }
}

kernel void compact_detector_regions_verify(
    device const uint *sums [[buffer(0)]],
    device const uint *descriptors [[buffer(1)]],
    device const uint *payload [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant CompactDetectorRegionsParameters &p [[buffer(4)]],
    uint linear [[thread_position_in_grid]]
) {
    if (linear >= p.blocks * p.scans) return;
    uint cell = linear / 32u;
    uint descriptor = descriptors[cell];
    uint width = descriptor & 31u;
    uint offset = descriptor >> 5u;
    uint expectedNext = cell + 1u == p.blocks * p.tiles
        ? p.payloadWords : descriptors[cell + 1u] >> 5u;
    if (width > p.maximumWidth || offset > p.payloadWords || width > p.payloadWords - offset
        || expectedNext != offset + width || (cell == 0u && offset != 0u)) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        return;
    }
    uint actual = compactCellValue(payload, descriptor, linear % 32u, 0u);
    if (actual != sums[linear]) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
}
