#include <metal_stdlib>
using namespace metal;

constant uint SC_LOWER = 1u << 23;
constant uint SC_SCALE = 10;
constant uint SC_MODELS = 64;

inline uint sc_raw(device const uchar *data, ulong at, uint item_bytes) {
    return item_bytes == 1 ? uint(data[at])
        : uint(reinterpret_cast<device const ushort *>(data)[at]);
}

struct StreamReader {
    device const uchar *payload;
    device const uint *table;
    uint cursor, end, state, model, constant_value;
    bool valid;

    StreamReader(device const uchar *bytes, device const uint *offsets,
                 device const uchar *models, device const uint *decoding,
                 uint stream)
        : payload(bytes), cursor(offsets[stream]), end(offsets[stream + 1]),
          state(SC_LOWER), model(models[stream]), constant_value(0), valid(true) {
        table = decoding + (model < SC_MODELS ? model * 1024 : 0);
        if (model == 252) {
            state = 0;
            valid = ((end - cursor) & 1u) == 0;
        } else if (model == 253) {
            return;
        } else if (model == 255) {
            valid = end - cursor == 2;
            if (valid) {
                constant_value = uint(payload[cursor])
                    | (uint(payload[cursor + 1]) << 8);
                cursor += 2;
            }
        } else if (model != 254) {
            valid = model < SC_MODELS && end - cursor >= 4;
            if (valid) {
                state = 0;
                for (uint byte = 0; byte < 4; ++byte)
                    state |= uint(payload[cursor++]) << (8 * byte);
                valid = state >= SC_LOWER && state < (1u << 31);
            }
        }
    }

    uint next() {
        if (!valid) return 0;
        if (model == 252) {
            uint position = state++;
            if (cursor == end) return 0;
            uint event = uint(payload[cursor])
                | (uint(payload[cursor + 1]) << 8);
            if ((event >> 7) < position) {
                valid = false;
                return 0;
            }
            if ((event >> 7) != position) return 0;
            cursor += 2;
            return (event & 127u) + 1;
        }
        if (model == 253 || model == 255) return constant_value;
        if (model == 254) {
            if (end - cursor < 2) {
                valid = false;
                return 0;
            }
            uint value = uint(payload[cursor])
                | (uint(payload[cursor + 1]) << 8);
            cursor += 2;
            return value;
        }
        uint slot = state & 1023u;
        uint code = table[slot];
        state = (code >> 16) * (state >> SC_SCALE)
            + slot - ((code >> 6) & 1023u);
        while (state < SC_LOWER) {
            if (cursor >= end) {
                valid = false;
                return 0;
            }
            state = (state << 8) | uint(payload[cursor++]);
        }
        uint symbol = code & 63u;
        if (symbol == 32) {
            if (end - cursor < 2) {
                valid = false;
                return 0;
            }
            symbol = uint(payload[cursor])
                | (uint(payload[cursor + 1]) << 8);
            cursor += 2;
        }
        return symbol;
    }

    uint next_entropy() {
        uint slot = state & 1023u;
        uint code = table[slot];
        state = (code >> 16) * (state >> SC_SCALE)
            + slot - ((code >> 6) & 1023u);
        while (state < SC_LOWER) {
            if (cursor >= end) {
                valid = false;
                return 0;
            }
            state = (state << 8) | uint(payload[cursor++]);
        }
        uint symbol = code & 63u;
        if (symbol == 32) {
            if (end - cursor < 2) {
                valid = false;
                return 0;
            }
            symbol = uint(payload[cursor])
                | (uint(payload[cursor + 1]) << 8);
            cursor += 2;
        }
        return symbol;
    }

    bool finished() const {
        return valid && cursor == end && (model >= 252 || state == SC_LOWER);
    }
};

kernel void streamed_counts_encode(
    device const uchar *raw [[buffer(0)]],
    device const uint *encoding [[buffer(1)]],
    device uchar *scratch [[buffer(2)]],
    device uint *sizes [[buffer(3)]],
    device uint *states [[buffer(4)]],
    device uchar *models [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    uint stream [[thread_position_in_grid]]) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    uint streams = uint(p[3]), item_bytes = uint(p[4]);
    if (stream >= streams) return;
    uint pixel = stream % pixels;
    uint first = (stream / pixels) * interval;
    uint count = min(interval, scans - first);
    uint maximum = 0, minimum = 65535, sum = 0, nonzero = 0;
    for (uint i = 0; i < count; ++i) {
        uint value = sc_raw(raw, ulong(first + i) * pixels + pixel, item_bytes);
        maximum = max(maximum, value);
        minimum = min(minimum, value);
        sum += min(value, 32u);
        nonzero += value != 0;
    }
    if (minimum == maximum) {
        models[stream] = maximum ? 255 : 253;
        sizes[stream] = maximum ? 2 : 0;
        states[stream] = maximum;
        return;
    }
    if (maximum <= 128 && nonzero <= 2) {
        models[stream] = 252;
        sizes[stream] = 2 * nonzero;
        return;
    }
    float mean = float(sum) / float(count);
    float scaled = (log(max(mean, 0.002f)) - log(0.002f))
        * (63.0f / log(16000.0f));
    int model = clamp(int(rint(scaled)), 0, 63);
    uint state = SC_LOWER, emitted = 0;
    bool literal = false;
    for (uint i = count; i > 0; --i) {
        if (emitted + 4 >= 2 * count) {
            literal = true;
            break;
        }
        uint value = sc_raw(raw, ulong(first + i - 1) * pixels + pixel, item_bytes);
        if (value >= 32) {
            scratch[ulong(emitted++) * streams + stream] = uchar(value >> 8);
            scratch[ulong(emitted++) * streams + stream] = uchar(value);
        }
        uint code = encoding[model * 33 + min(value, 32u)];
        uint frequency = code >> 16, start = code & 65535u;
        uint limit = ((SC_LOWER >> SC_SCALE) << 8) * frequency;
        while (state >= limit) {
            scratch[ulong(emitted++) * streams + stream] = uchar(state);
            state >>= 8;
        }
        uint quotient = state / frequency;
        state = (quotient << SC_SCALE) + (state - quotient * frequency) + start;
    }
    uint bytes = literal ? 2 * count : min(emitted + 4, 2 * count);
    models[stream] = bytes < 2 * count ? uchar(model) : uchar(254);
    if (maximum <= 128 && 2 * nonzero < bytes) {
        models[stream] = 252;
        bytes = 2 * nonzero;
    }
    sizes[stream] = bytes;
    states[stream] = state;
}

kernel void streamed_counts_compact(
    device const uchar *raw [[buffer(0)]],
    device const uchar *scratch [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device const uint *states [[buffer(3)]],
    device const uchar *models [[buffer(4)]],
    device uchar *payload [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    uint stream [[thread_position_in_grid]]) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    uint streams = uint(p[3]), item_bytes = uint(p[4]);
    if (stream >= streams) return;
    uint begin = offsets[stream], size = offsets[stream + 1] - begin;
    uint model = models[stream];
    if (model == 253) return;
    uint first = (stream / pixels) * interval, pixel = stream % pixels;
    if (model == 252) {
        uint at = begin;
        for (uint i = 0; i < min(interval, scans - first); ++i) {
            uint value = sc_raw(raw, ulong(first + i) * pixels + pixel, item_bytes);
            if (value) {
                uint event = (i << 7) | (value - 1);
                payload[at++] = uchar(event);
                payload[at++] = uchar(event >> 8);
            }
        }
        return;
    }
    if (model == 255) {
        payload[begin] = uchar(states[stream]);
        payload[begin + 1] = uchar(states[stream] >> 8);
        return;
    }
    if (model == 254) {
        for (uint i = 0; i < size / 2; ++i) {
            uint value = sc_raw(raw, ulong(first + i) * pixels + pixel, item_bytes);
            payload[begin + 2 * i] = uchar(value);
            payload[begin + 2 * i + 1] = uchar(value >> 8);
        }
        return;
    }
    for (uint i = 0; i < 4; ++i)
        payload[begin + i] = uchar(states[stream] >> (8 * i));
    for (uint i = 4; i < size; ++i)
        payload[begin + i] = scratch[ulong(size - 1 - i) * streams + stream];
}

kernel void streamed_counts_decode_range(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device uchar *output [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    uint local_stream [[thread_position_in_grid]]) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    uint first = uint(p[3]), count = uint(p[4]), first_stream = uint(p[5]);
    uint stop_stream = uint(p[6]);
    uint stream = first_stream + local_stream;
    if (stream >= stop_stream) return;
    uint block_first = (stream / pixels) * interval, pixel = stream % pixels;
    uint block_count = min(interval, scans - block_first), stop = first + count;
    StreamReader reader(payload, offsets, models, decoding, stream);
    if (reader.valid && reader.model < SC_MODELS) {
        for (uint i = 0; i < block_count; ++i) {
            uint value = reader.next_entropy(), scan = block_first + i;
            if (scan >= first && scan < stop) {
                ulong at = ulong(scan - first) * pixels + pixel;
            reinterpret_cast<device uint *>(output)[at] = value;
            }
        }
    } else {
        for (uint i = 0; i < block_count; ++i) {
            uint value = reader.next(), scan = block_first + i;
            if (scan >= first && scan < stop) {
                ulong at = ulong(scan - first) * pixels + pixel;
                reinterpret_cast<device uint *>(output)[at] = value;
            }
        }
    }
    if (!reader.finished())
        atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

// Decode selected detector columns directly to plane-major scan storage. Padding
// is initialized separately; every original scan sample retains its position.
kernel void streamed_counts_detector_columns(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device uint *output [[buffer(5)]],
    constant uint *p [[buffer(6)]],
    device const uint *selected [[buffer(7)]],
    device const uchar *valid [[buffer(8)]],
    uint job [[thread_position_in_grid]]) {
    uint scans = p[0], pixels = p[1], interval = p[2], columns = p[3];
    uint blocks = (scans + interval - 1) / interval;
    if (job >= blocks * columns) return;
    uint ordinal = job % columns, block = job / columns;
    uint pixel = selected[ordinal], first = block * interval;
    if (!valid[pixel]) return;
    StreamReader reader(payload, offsets, models, decoding, block * pixels + pixel);
    for (uint i = 0; i < min(interval, scans - first); ++i) {
        uint scan = p[4] + first + i;
        uint value = reader.next();
        output[ulong(ordinal) * p[6] * p[7] + (scan / p[5]) * p[7] + scan % p[5]] = value;
    }
    if (!reader.finished()) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

kernel void streamed_counts_detector_total(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device ulong *output [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    device const uchar *valid [[buffer(7)]],
    uint pixel [[thread_position_in_grid]]) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    if (pixel >= pixels) return;
    if (!valid[pixel]) {
        output[pixel] = 0;
        return;
    }
    ulong total = output[pixel];
    uint streams_per_pixel = (scans + interval - 1) / interval;
    for (uint block = 0; block < streams_per_pixel; ++block) {
        uint first = block * interval;
        StreamReader reader(payload, offsets, models, decoding, block * pixels + pixel);
        for (uint i = 0; i < min(interval, scans - first); ++i)
            total += ulong(reader.next());
        if (!reader.finished())
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    }
    output[pixel] = total;
}

// Exact raw-count mean-DP numerator. One lane owns one detector pixel; chunks
// are ordered with a buffer barrier, so UInt64 sums need no atomic operations.
kernel void streamed_counts_region_total(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device ulong *output [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    device const uchar *membership [[buffer(7)]],
    uint pixel [[thread_position_in_grid]]) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    if (pixel >= pixels) return;
    ulong total = output[pixel];
    ulong region_first = p[5] * p[4] + p[7];
    ulong region_stop = (p[6] - 1) * p[4] + p[8];
    for (uint block = 0; block < (scans + interval - 1) / interval; ++block) {
        uint first = block * interval, count = min(interval, scans - first);
        if (p[3] + first >= region_stop || p[3] + first + count <= region_first) continue;
        StreamReader reader(payload, offsets, models, decoding, block * pixels + pixel);
        if (reader.model == 253) {
            // A validated zero stream contributes nothing to any scan region.
        } else if (reader.model == 252) {
            uint previous = 0;
            bool first_event = true;
            while (reader.valid && reader.cursor < reader.end) {
                uint event = uint(payload[reader.cursor]) | (uint(payload[reader.cursor + 1]) << 8);
                reader.cursor += 2;
                uint position = event >> 7;
                if (position >= count || (!first_event && position <= previous)) {
                    reader.valid = false;
                    break;
                }
                first_event = false;
                previous = position;
                if (membership[first + position]) total += ulong((event & 127u) + 1u);
            }
        } else if (reader.model < SC_MODELS) {
            uint subtotal = 0;
            for (uint i = 0; i < count && reader.valid; ++i) {
                uint value = reader.next_entropy();
                if (membership[first + i]) subtotal += value;
            }
            // One block has at most 512 uint16 values, safely within UInt32.
            total += ulong(subtotal);
        } else {
            uint subtotal = 0;
            for (uint i = 0; i < count; ++i) {
                uint value = reader.next();
                if (membership[first + i]) subtotal += value;
            }
            total += ulong(subtotal);
        }
        if (!reader.finished()) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    }
    output[pixel] = total;
}

// Apply only detector pixels whose binary membership changed. One lane owns one
// entropy stream; SIMD reduction turns 32 competing per-scan atomics into one.
// UInt32 is exact for one uint16 192x192 detector sum (maximum 2,415,882,240).
kernel void streamed_counts_detector_delta(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device const uint *selected [[buffer(5)]],
    device const int *coefficients [[buffer(6)]],
    device atomic_uint *output [[buffer(7)]],
    constant ulong *p [[buffer(8)]],
    uint job [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]]) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    uint changed = uint(p[3]), output_first = uint(p[4]), width = uint(p[5]);
    uint blocks = (scans + interval - 1) / interval;
    uint groups = (changed + width - 1u) / width;
    if (job >= blocks * groups) return;
    uint block = job / groups, ordinal = (job % groups) * width + lane;
    bool active = ordinal < changed;
    uint pixel = active ? selected[ordinal] : 0u;
    if (active && pixel >= pixels) {
        atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
        active = false;
    }
    int coefficient = active ? coefficients[ordinal] : 0;
    uint first = block * interval;
    StreamReader reader(payload, offsets, models, decoding, block * pixels + pixel);
    uint count = min(interval, scans - first);
    bool sparse = active && reader.model == 252;
    bool dense = active && reader.model != 252 && reader.model != 253;
    bool anyDense = simd_any(dense);
    if (anyDense) {
        bool entropy = dense && reader.model < SC_MODELS;
        for (uint i = 0; i < count; ++i) {
            uint value = 0;
            if (dense) value = entropy ? reader.next_entropy() : reader.next();
            int contribution = int(value) * coefficient;
            // All SIMD lanes reconverge before the collective reduction.
            int subtotal = simd_sum(contribution);
            if (simdLane == 0)
                atomic_fetch_add_explicit(
                    output + output_first + first + i, uint(subtotal), memory_order_relaxed);
        }
    }
    if (sparse) {
        bool firstEvent = true;
        uint previousPosition = 0;
        while (reader.cursor < reader.end) {
            if (reader.end - reader.cursor < 2) {
                reader.valid = false;
                break;
            }
            uint event = uint(reader.payload[reader.cursor])
                | (uint(reader.payload[reader.cursor + 1]) << 8);
            reader.cursor += 2;
            uint position = event >> 7;
            if (position >= count || (!firstEvent && position <= previousPosition)) {
                reader.valid = false;
                break;
            }
            previousPosition = position;
            firstEvent = false;
            int contribution = int((event & 127u) + 1u) * coefficient;
            atomic_fetch_add_explicit(
                output + output_first + first + position,
                uint(contribution), memory_order_relaxed);
        }
    }
    if (active && !reader.finished())
        atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

// One SIMD group owns one 512-scan packet. Dense columns reduce into local
// memory, zero columns disappear, and sparse columns scatter only their stored
// events. Every output scan is published once after all selected columns.
kernel void streamed_counts_detector_packet(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device const uint *selected [[buffer(5)]],
    device const int *coefficients [[buffer(6)]],
    device uint *output [[buffer(7)]],
    constant ulong *p [[buffer(8)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    uint changed = uint(p[3]), output_first = uint(p[4]);
    uint blocks = (scans + interval - 1u) / interval;
    if (block >= blocks || lane >= 32u) return;
    uint first = block * interval;
    uint count = min(interval, scans - first);
    threadgroup atomic_int partials[512];
    for (uint scan = lane; scan < count; scan += 32u)
        atomic_store_explicit(partials + scan, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < changed; base += 32u) {
        uint ordinal = base + lane;
        bool active = ordinal < changed;
        uint pixel = active ? selected[ordinal] : 0u;
        if (active && pixel >= pixels) {
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
            active = false;
        }
        int coefficient = active ? coefficients[ordinal] : 0;
        StreamReader reader(payload, offsets, models, decoding, block * pixels + pixel);
        bool sparse = active && reader.model == 252;
        bool dense = active && reader.model != 252 && reader.model != 253;
        bool anyDense = simd_any(dense);
        if (anyDense) {
            bool entropy = dense && reader.model < SC_MODELS;
            for (uint scan = 0; scan < count; ++scan) {
                uint value = 0;
                if (dense) value = entropy ? reader.next_entropy() : reader.next();
                int subtotal = simd_sum(int(value) * coefficient);
                if (lane == 0)
                    atomic_fetch_add_explicit(
                        partials + scan, subtotal, memory_order_relaxed);
            }
        }
        if (sparse) {
            bool firstEvent = true;
            uint previousPosition = 0;
            while (reader.cursor < reader.end) {
                if (reader.end - reader.cursor < 2) {
                    reader.valid = false;
                    break;
                }
                uint event = uint(reader.payload[reader.cursor])
                    | (uint(reader.payload[reader.cursor + 1]) << 8);
                reader.cursor += 2;
                uint position = event >> 7;
                if (position >= count || (!firstEvent && position <= previousPosition)) {
                    reader.valid = false;
                    break;
                }
                previousPosition = position;
                firstEvent = false;
                int contribution = int((event & 127u) + 1u) * coefficient;
                atomic_fetch_add_explicit(
                    partials + position, contribution, memory_order_relaxed);
            }
        }
        if (active && !reader.finished())
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint scan = lane; scan < count; scan += 32u)
        output[output_first + first + scan] += uint(
            atomic_load_explicit(partials + scan, memory_order_relaxed));
}

// Exact scheduling experiment: four independent packet-owner SIMD groups share
// one 128-thread threadgroup. Each SIMD owns a disjoint 512-scan packet, a
// disjoint 512-word scratch slice, and a disjoint output slice. The scalar rANS
// streams and their per-lane decode order are unchanged.
kernel void streamed_counts_detector_packet4(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device const uint *selected [[buffer(5)]],
    device const int *coefficients [[buffer(6)]],
    device uint *output [[buffer(7)]],
    constant ulong *p [[buffer(8)]],
    uint block_group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    uint simd = thread_id >> 5;
    uint lane = thread_id & 31u;
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    uint changed = uint(p[3]), output_first = uint(p[4]);
    uint blocks = (scans + interval - 1u) / interval;
    uint block = block_group * 4u + simd;
    if (block >= blocks) return;
    uint first = block * interval;
    uint count = min(interval, scans - first);
    threadgroup atomic_int partial_storage[4 * 512];
    threadgroup atomic_int *partials = partial_storage + simd * 512u;
    for (uint scan = lane; scan < count; scan += 32u)
        atomic_store_explicit(partials + scan, 0, memory_order_relaxed);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < changed; base += 32u) {
        uint ordinal = base + lane;
        bool active = ordinal < changed;
        uint pixel = active ? selected[ordinal] : 0u;
        if (active && pixel >= pixels) {
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
            active = false;
        }
        int coefficient = active ? coefficients[ordinal] : 0;
        StreamReader reader(payload, offsets, models, decoding, block * pixels + pixel);
        bool sparse = active && reader.model == 252;
        bool dense = active && reader.model != 252 && reader.model != 253;
        bool anyDense = simd_any(dense);
        if (anyDense) {
            bool entropy = dense && reader.model < SC_MODELS;
            for (uint scan = 0; scan < count; ++scan) {
                uint value = 0;
                if (dense) value = entropy ? reader.next_entropy() : reader.next();
                int subtotal = simd_sum(int(value) * coefficient);
                if (lane == 0)
                    atomic_fetch_add_explicit(
                        partials + scan, subtotal, memory_order_relaxed);
            }
        }
        if (sparse) {
            bool firstEvent = true;
            uint previousPosition = 0;
            while (reader.cursor < reader.end) {
                if (reader.end - reader.cursor < 2) {
                    reader.valid = false;
                    break;
                }
                uint event = uint(reader.payload[reader.cursor])
                    | (uint(reader.payload[reader.cursor + 1]) << 8);
                reader.cursor += 2;
                uint position = event >> 7;
                if (position >= count || (!firstEvent && position <= previousPosition)) {
                    reader.valid = false;
                    break;
                }
                previousPosition = position;
                firstEvent = false;
                int contribution = int((event & 127u) + 1u) * coefficient;
                atomic_fetch_add_explicit(
                    partials + position, contribution, memory_order_relaxed);
            }
        }
        if (active && !reader.finished())
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint scan = lane; scan < count; scan += 32u)
        output[output_first + first + scan] += uint(
            atomic_load_explicit(partials + scan, memory_order_relaxed));
}

kernel void streamed_counts_normalize(
    device const ulong *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ulong2 &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < p.x) output[index] = float(input[index]) / float(p.y);
}

kernel void streamed_counts_reduce(
    device const uchar *counts [[buffer(0)]],
    device const uchar *mask [[buffer(1)]],
    device ulong *output [[buffer(2)]],
    constant ulong4 &p [[buffer(3)]],
    uint scan [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    ulong subtotal = 0;
    for (uint pixel = lane; pixel < p.x; pixel += 128) {
        ulong at = ulong(scan) * p.x + pixel;
        uint value = p.y == 1 ? uint(counts[at])
            : uint(reinterpret_cast<device const ushort *>(counts)[at]);
        if (mask[pixel]) subtotal += value;
    }
    threadgroup ulong partials[128];
    partials[lane] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[p.z + scan] = partials[0];
}
