// Exact 8/32-pixel spatial indexes and SIMD mask decomposition for camera counts.
kernel void camera_mask_leaves(
    device const uchar* mask [[buffer(0)]], device const uchar* valid [[buffer(1)]],
    device int* leaves [[buffer(2)]], device uint* pixels [[buffer(3)]],
    device int* coefficients [[buffer(4)]], device atomic_uint* counts [[buffer(5)]],
    constant uint2& shape [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint tileCols = (shape.y + 7) / 8;
    uint row = tile / tileCols * 8 + lane / 8, col = tile % tileCols * 8 + lane % 8;
    bool ok0 = row < shape.x && col < shape.y, ok1 = row + 4 < shape.x && col < shape.y;
    int a = ok0 && valid[row * shape.y + col] && mask[row * shape.y + col];
    int b = ok1 && valid[(row + 4) * shape.y + col] && mask[(row + 4) * shape.y + col];
    int base = simd_sum(a + b) > 32 ? 1 : 0;
    if (lane == 0) leaves[tile] = base;
    uint n0 = ok0 && valid[row * shape.y + col] && a != base;
    uint n1 = ok1 && valid[(row + 4) * shape.y + col] && b != base;
    uint prefix = simd_prefix_exclusive_sum(n0 + n1), total = simd_sum(n0 + n1), start = 0;
    if (lane == 0 && total) start = atomic_fetch_add_explicit(counts + 1, total, memory_order_relaxed);
    start = simd_broadcast_first(start) + prefix;
    if (n0) { pixels[start] = row * shape.y + col; coefficients[start++] = a - base; }
    if (n1) { pixels[start] = (row + 4) * shape.y + col; coefficients[start] = b - base; }
}

kernel void camera_mask_roots(
    device const int* leaves [[buffer(0)]], device uint* fields [[buffer(1)]],
    device int* coefficients [[buffer(2)]], device atomic_uint* counts [[buffer(3)]],
    constant uint2& shape [[buffer(4)]], uint root [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint nr = (shape.x + 7) / 8, nc = (shape.y + 7) / 8, rootCols = (nc + 3) / 4;
    uint row = root / rootCols * 4 + lane / 4, col = root % rootCols * 4 + lane % 4;
    bool valid = lane < 16 && row < nr && col < nc;
    int value = valid ? leaves[row * nc + col] : 0;
    int base = simd_sum(value) > 8 ? 1 : 0;
    if (lane == 0 && base) {
        uint at = atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
        fields[at] = nr * nc + root; coefficients[at] = base;
    }
    if (valid && value != base) {
        uint at = atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
        fields[at] = row * nc + col; coefficients[at] = value - base;
    }
}

inline uint camera_field(device const uint* words, device const ulong* starts,
                         device const uchar* widths, uint scan, uint field, uint fields) {
    uint stream = scan / 512 * fields + field, width = widths[stream];
    if (!width) return 0;
    ulong bit = ulong(scan % 512) * width, at = starts[stream] + bit / 32;
    ulong value = words[at];
    if (bit % 32 + width > 32) value |= ulong(words[at + 1]) << 32;
    return uint(value >> (bit % 32)) & (width == 32 ? 0xffffffffu : (1u << width) - 1);
}

kernel void camera_index_sum(
    device const uint* words [[buffer(0)]], device const ulong* starts [[buffer(1)]],
    device const uchar* widths [[buffer(2)]], device const uint* selected [[buffer(3)]],
    device const int* coefficients [[buffer(4)]], device uint* output [[buffer(5)]],
    constant uint4& p [[buffer(6)]], uint scan [[thread_position_in_grid]]) {
    if (scan >= p.x) return;
    uint sum = 0;
    for (uint index = 0; index < p.z; ++index)
        sum += uint(coefficients[index]) * camera_field(words, starts, widths, scan, selected[index], p.y);
    output[p.w + scan] = sum;
}

kernel void camera_fields(
    device const uchar* raw [[buffer(0)]], device const uchar* valid [[buffer(1)]],
    device uint* values [[buffer(2)]], constant uint4& p [[buffer(3)]],
    uint item [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint rows = p.y, cols = p.z, leaves = ((rows + 7) / 8) * ((cols + 7) / 8);
    uint fields = leaves + ((rows + 31) / 32) * ((cols + 31) / 32);
    uint scan = item / fields, field = item % fields;
    if (scan >= p.x) return;
    uint side = field < leaves ? 8 : 32;
    if (field >= leaves) field -= leaves;
    uint fieldCols = (cols + side - 1) / side, r0 = field / fieldCols * side, c0 = field % fieldCols * side;
    uint sum = 0;
    for (uint i = lane; i < side * side; i += 32) {
        uint row = r0 + i / side, col = c0 + i % side;
        if (row < rows && col < cols && valid[row * cols + col])
            sum += sc_raw(raw, (ulong(scan) * rows + row) * cols + col, p.w);
    }
    sum = simd_sum(sum);
    if (lane == 0) values[item] = sum;
}

kernel void camera_field_widths(
    device const uint* values [[buffer(0)]], device uchar* widths [[buffer(1)]],
    device uint* lengths [[buffer(2)]], constant uint2& p [[buffer(3)]],
    uint stream [[thread_position_in_grid]]) {
    if (stream >= ((p.x + 511) / 512) * p.y) return;
    uint first = stream / p.y * 512, field = stream % p.y, bits = 0, count = min(512u, p.x - first);
    for (uint i = 0; i < count; ++i) bits |= values[ulong(first + i) * p.y + field];
    uint width = bits ? 32 - clz(bits) : 0;
    widths[stream] = width; lengths[stream] = (count * width + 31) / 32;
}

kernel void camera_pack_fields(
    device const uint* values [[buffer(0)]], device const uchar* widths [[buffer(1)]],
    device const ulong* starts [[buffer(2)]], device uint* words [[buffer(3)]],
    constant uint2& p [[buffer(4)]], uint stream [[thread_position_in_grid]]) {
    if (stream >= ((p.x + 511) / 512) * p.y) return;
    uint width = widths[stream]; if (!width) return;
    uint first = stream / p.y * 512, field = stream % p.y, available = 0;
    ulong reservoir = 0, at = starts[stream];
    for (uint i = 0; i < min(512u, p.x - first); ++i) {
        reservoir |= ulong(values[ulong(first + i) * p.y + field]) << available;
        available += width;
        if (available >= 32) { words[at++] = uint(reservoir); reservoir >>= 32; available -= 32; }
    }
    if (available) words[at] = uint(reservoir);
}

inline uint camera_entropy(thread StreamReader& reader) {
    uint slot = reader.state & 1023u, code = reader.table[slot];
    reader.state = (code >> 16) * (reader.state >> 10) + slot - ((code >> 6) & 1023u);
    while (reader.state < SC_LOWER) {
        if (reader.cursor >= reader.end) { reader.valid = false; return 0; }
        reader.state = (reader.state << 8) | uint(reader.payload[reader.cursor++]);
    }
    uint symbol = code & 63u;
    if (symbol == 32) {
        if (reader.end - reader.cursor < 2) { reader.valid = false; return 0; }
        symbol = uint(reader.payload[reader.cursor]) | (uint(reader.payload[reader.cursor + 1]) << 8);
        reader.cursor += 2;
    }
    return symbol;
}

inline uint camera_frame_value(thread StreamReader& reader, uint position) {
    if (!reader.valid) return 0;
    if (reader.model == 253 || reader.model == 255) return reader.constant_value;
    if (reader.model == 254) {
        uint at = reader.cursor + position * 2;
        if (at + 2 > reader.end) { reader.valid = false; return 0; }
        return uint(reader.payload[at]) | (uint(reader.payload[at + 1]) << 8);
    }
    if (reader.model == 252) {
        while (reader.cursor + 2 <= reader.end) {
            uint event = uint(reader.payload[reader.cursor]) | (uint(reader.payload[reader.cursor + 1]) << 8);
            reader.cursor += 2;
            if ((event >> 7) == position) return (event & 127) + 1;
            if ((event >> 7) > position) break;
        }
        return 0;
    }
    uint value = 0;
    for (uint i = 0; i <= position; ++i) value = camera_entropy(reader);
    return value;
}

kernel void camera_frame(
    device const uchar* payload [[buffer(0)]], device const uint* offsets [[buffer(1)]],
    device const uchar* models [[buffer(2)]], device const uint* decoding [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]], device uint* output [[buffer(5)]],
    constant ulong* p [[buffer(6)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= p[1]) return;
    StreamReader reader(payload, offsets, models, decoding, uint(p[3] / 512 * p[1]) + pixel);
    output[pixel] = camera_frame_value(reader, uint(p[3] % 512));
    if (!reader.valid) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

kernel void camera_frame_native(
    device const uchar* payload [[buffer(0)]], device const uint* offsets [[buffer(1)]],
    device const uchar* models [[buffer(2)]], device const uint* decoding [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]], device uchar* output [[buffer(5)]],
    constant ulong* p [[buffer(6)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= p[1]) return;
    StreamReader reader(payload, offsets, models, decoding, uint(p[3] / 512 * p[1]) + pixel);
    uint value = camera_frame_value(reader, uint(p[3] % 512));
    if (p[7] == 1) output[pixel] = uchar(value);
    else reinterpret_cast<device ushort*>(output)[pixel] = ushort(value);
    if (!reader.valid) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

// Split-word atomic addition preserves exact uint64 sums on Apple GPUs.
// Readers consume the two words only after command completion.
inline void camera_atomic_signed(device atomic_uint* output, uint scan, int value) {
    uint old = atomic_fetch_add_explicit(output + 2 * scan, uint(value), memory_order_relaxed);
    if (value >= 0 && old > 0xffffffffu - uint(value))
        atomic_fetch_add_explicit(output + 2 * scan + 1, 1u, memory_order_relaxed);
    else if (value < 0 && old < uint(-value))
        atomic_fetch_sub_explicit(output + 2 * scan + 1, 1u, memory_order_relaxed);
}

kernel void camera_delta_u64(
    device const uchar* payload [[buffer(0)]], device const uint* offsets [[buffer(1)]],
    device const uchar* models [[buffer(2)]], device const uint* table [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]], device const uint* selected [[buffer(5)]],
    device const int* signs [[buffer(6)]], device atomic_uint* output [[buffer(7)]],
    constant uint4& p [[buffer(8)]], uint job [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint groups = (p.z + 31) / 32, block = job / groups, ordinal = job % groups * 32 + lane;
    uint first = block * 512, count = min(512u, p.x - first);
    bool active = ordinal < p.z;
    uint pixel = active ? selected[ordinal] : 0;
    int sign = active ? signs[ordinal] : 0;
    StreamReader reader(payload, offsets, models, table, block * p.y + pixel);
    bool sparse = active && reader.model == 252;
    bool dense = active && reader.model != 252 && reader.model != 253;
    if (simd_any(dense)) {
        for (uint scan = 0; scan < count; ++scan) {
            uint value = dense ? (reader.model < SC_MODELS ? camera_entropy(reader) : reader.next()) : 0;
            int subtotal = simd_sum(int(value) * sign);
            if (lane == 0 && subtotal) camera_atomic_signed(output, p.w + first + scan, subtotal);
        }
    }
    if (sparse) {
        uint previous = 0; bool firstEvent = true;
        while (reader.cursor + 2 <= reader.end) {
            uint event = uint(payload[reader.cursor]) | (uint(payload[reader.cursor + 1]) << 8);
            reader.cursor += 2;
            uint position = event >> 7;
            if (position >= count || (!firstEvent && position <= previous)) { reader.valid = false; break; }
            camera_atomic_signed(output, p.w + first + position, int((event & 127) + 1) * sign);
            previous = position; firstEvent = false;
        }
    }
    if (active && !reader.finished()) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

kernel void camera_index_sum_u64_simd(
    device const uint* words [[buffer(0)]], device const ulong* starts [[buffer(1)]],
    device const uchar* widths [[buffer(2)]], device const uint* selected [[buffer(3)]],
    device const int* coefficients [[buffer(4)]], device ulong* output [[buffer(5)]],
    constant uint4& p [[buffer(6)]], uint scan [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (scan >= p.x) return;
    ulong sum = 0;
    for (uint index = lane; index < p.z; index += 32)
        sum += ulong(long(coefficients[index])) * camera_field(words, starts, widths, scan, selected[index], p.y);
    for (uint offset = 16; offset; offset /= 2) {
        uint low = simd_shuffle_down(uint(sum), offset);
        uint high = simd_shuffle_down(uint(sum >> 32), offset);
        sum += ulong(low) | (ulong(high) << 32);
    }
    if (lane == 0) output[p.w + scan] = sum;
}

kernel void camera_index_sum_simd(
    device const uint* words [[buffer(0)]], device const ulong* starts [[buffer(1)]],
    device const uchar* widths [[buffer(2)]], device const uint* selected [[buffer(3)]],
    device const int* coefficients [[buffer(4)]], device uint* output [[buffer(5)]],
    constant uint4& p [[buffer(6)]], uint scan [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (scan >= p.x) return;
    uint sum = 0;
    for (uint index = lane; index < p.z; index += 32)
        sum += uint(coefficients[index]) * camera_field(words, starts, widths, scan, selected[index], p.y);
    sum = simd_sum(sum);
    if (lane == 0) output[p.w + scan] = sum;
}
