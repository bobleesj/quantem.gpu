#include <metal_stdlib>
using namespace metal;

// Same byte-normalized recurrence as the independently retained count streams.
struct ANSReader {
    device const uchar *payload;
    device const ushort *symbols;
    device const ushort *cumulative;
    device const ushort *frequencies;
    ulong cursor, end;
    uint state, first, stop, scale;
    bool raw, valid;

    ANSReader(device const uchar *bytes, device const ulong *offsets,
        device const uint *models, device const uint *contexts,
        device const ushort *values, device const ushort *starts,
        device const ushort *weights, device const uchar *literals,
        ulong stream, uint bits)
        : payload(bytes), symbols(values), cumulative(starts), frequencies(weights),
          cursor(offsets[stream]), end(offsets[stream + 1]), state(1u << 23),
          scale(bits), valid(true) {
        uint model = models[stream];
        first = contexts[model];
        stop = contexts[model + 1];
        raw = literals[model] != 0;
        if (!raw) {
            if (end - cursor < 4) { valid = false; return; }
            state = 0;
            for (uint byte = 0; byte < 4; ++byte)
                state |= uint(payload[cursor++]) << (8 * byte);
            valid = state >= (1u << 23) && state < (1u << 31);
        }
    }

    ushort next() {
        if (!valid) return 0;
        if (raw) {
            if (end - cursor < 2) { valid = false; return 0; }
            ushort value = ushort(payload[cursor]) | (ushort(payload[cursor + 1]) << 8);
            cursor += 2;
            return value;
        }
        uint slot = state & ((1u << scale) - 1);
        uint left = first, right = stop;
        while (left + 1 < right) {
            uint middle = left + (right - left) / 2;
            if (cumulative[middle] <= slot) left = middle;
            else right = middle;
        }
        ushort value = symbols[left];
        state = frequencies[left] * (state >> scale) + slot - cumulative[left];
        while (state < (1u << 23)) {
            if (cursor >= end) { valid = false; return 0; }
            state = (state << 8) | payload[cursor++];
        }
        return value;
    }

    bool finished() const {
        return valid && cursor == end && (raw || state == (1u << 23));
    }
};

#define ANS_INPUTS \
    device const uchar *payload [[buffer(0)]], \
    device const ulong *offsets [[buffer(1)]], \
    device const uint *models [[buffer(2)]], \
    device const uint *contexts [[buffer(3)]], \
    device const ushort *symbols [[buffer(4)]], \
    device const ushort *cumulative [[buffer(5)]], \
    device const ushort *frequencies [[buffer(6)]], \
    device const uchar *literal [[buffer(7)]], \
    device atomic_uint *errors [[buffer(8)]], \
    constant ulong *parameters [[buffer(9)]]

#define ANS_READER(stream) ANSReader reader(payload, offsets, models, contexts, \
    symbols, cumulative, frequencies, literal, stream, uint(parameters[3]))

inline void store_count(device uchar *output, ulong index, ushort value, ulong bytes) {
    if (bytes == 1) output[index] = uchar(value);
    else reinterpret_cast<device ushort *>(output)[index] = value;
}

kernel void ans_counts_validate(ANS_INPUTS, uint position [[thread_position_in_grid]]) {
    ulong stream = position;
    ulong blockCount = (parameters[0] + parameters[2] - 1) / parameters[2];
    if (stream >= blockCount * parameters[1]) return;
    ulong begin = (stream / parameters[1]) * parameters[2];
    uint count = uint(min(parameters[2], parameters[0] - begin));
    ANS_READER(stream);
    for (uint scan = 0; scan < count && reader.valid; ++scan)
        if (reader.next() > parameters[4]) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
    if (!reader.finished()) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

kernel void ans_counts_decode(ANS_INPUTS, device uchar *output [[buffer(10)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= parameters[1]) return;
    ANS_READER(parameters[5] * parameters[1] + pixel);
    uint end = uint(parameters[6] + parameters[7]);
    if (reader.raw) reader.cursor += parameters[6] * 2;
    for (uint scan = reader.raw ? uint(parameters[6]) : 0u; scan < end && reader.valid; ++scan) {
        ushort value = reader.next();
        if (scan >= parameters[6])
            store_count(output, (scan - parameters[6]) * parameters[1] + pixel,
                        value, parameters[8]);
    }
    if (!reader.valid) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

kernel void ans_counts_gather(ANS_INPUTS, device uchar *output [[buffer(10)]],
    device const ulong *requested [[buffer(11)]], uint2 position [[thread_position_in_grid]]) {
    uint pixel = position.x;
    uint request = position.y;
    if (pixel >= parameters[1] || request >= parameters[6]) return;
    ulong scan = requested[request];
    if (scan >= parameters[0]) {
        atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    ANS_READER((scan / parameters[2]) * parameters[1] + pixel);
    uint within = uint(scan % parameters[2]);
    ushort value = 0;
    if (reader.raw) {
        reader.cursor += ulong(within) * 2;
        value = reader.next();
    } else {
        for (uint step = 0; step <= within && reader.valid; ++step) value = reader.next();
    }
    if (!reader.valid) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    else store_count(output, ulong(request) * parameters[1] + pixel, value, parameters[8]);
}

kernel void ans_counts_reduce(device const uchar *counts [[buffer(0)]],
    device const uchar *mask [[buffer(1)]], device ulong *output [[buffer(2)]],
    constant ulong4 &parameters [[buffer(3)]], uint scan [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    ulong subtotal = 0;
    for (ulong pixel = lane; pixel < parameters.x; pixel += 128) {
        ulong index = ulong(scan) * parameters.x + pixel;
        uint value = parameters.y == 1 ? counts[index]
            : reinterpret_cast<device const ushort *>(counts)[index];
        if (mask[pixel]) subtotal += value;
    }
    threadgroup ulong partials[128];
    partials[lane] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride /= 2) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[parameters.z + scan] = partials[0];
}
