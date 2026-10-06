#include <metal_stdlib>
using namespace metal;
constant uint float_ans_pixels [[function_constant(20)]];

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

    StreamReader()
        : payload(nullptr), table(nullptr), cursor(0), end(0), state(0), model(0),
          constant_value(0), valid(false) {}

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

// Begin both independent rANS table lookups before consuming either result.
// The two streams and their byte cursors remain separate; output word order is
// unchanged. Invalid streams still set the existing error flag after decode.
inline uint2 float_ans_next_entropy_pair(
    thread StreamReader& low, thread StreamReader& high) {
    uint lowSlot = low.state & 1023u;
    uint highSlot = high.state & 1023u;
    uint lowCode = low.table[lowSlot];
    uint highCode = high.table[highSlot];
    low.state = (lowCode >> 16) * (low.state >> SC_SCALE)
        + lowSlot - ((lowCode >> 6) & 1023u);
    high.state = (highCode >> 16) * (high.state >> SC_SCALE)
        + highSlot - ((highCode >> 6) & 1023u);
    bool lowFailed = false, highFailed = false;
    while (low.state < SC_LOWER) {
        if (low.cursor >= low.end) {
            low.valid = false;
            lowFailed = true;
            break;
        }
        low.state = (low.state << 8) | uint(low.payload[low.cursor++]);
    }
    while (high.state < SC_LOWER) {
        if (high.cursor >= high.end) {
            high.valid = false;
            highFailed = true;
            break;
        }
        high.state = (high.state << 8) | uint(high.payload[high.cursor++]);
    }
    uint lowSymbol = lowFailed ? 0 : lowCode & 63u;
    uint highSymbol = highFailed ? 0 : highCode & 63u;
    if (lowSymbol == 32) {
        if (low.end - low.cursor < 2) {
            low.valid = false;
            lowSymbol = 0;
        } else {
            lowSymbol = uint(low.payload[low.cursor])
                | (uint(low.payload[low.cursor + 1]) << 8);
            low.cursor += 2;
        }
    }
    if (highSymbol == 32) {
        if (high.end - high.cursor < 2) {
            high.valid = false;
            highSymbol = 0;
        } else {
            highSymbol = uint(high.payload[high.cursor])
                | (uint(high.payload[high.cursor + 1]) << 8);
            high.cursor += 2;
        }
    }
    return uint2(lowSymbol, highSymbol);
}

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

// Restore IEEE-754 words without converting their numerical value. Row
// descriptors expose only this bounded scratch window to existing consumers.
kernel void float_ans_recovery_needed(
    device const uint2 *accumulated [[buffer(0)]],
    device atomic_uint *recover [[buffer(1)]],
    uint frame [[thread_position_in_grid]]) {
    uint2 previous = accumulated[frame];
    if ((previous.x & 0x7f800000u) == 0x7f800000u
        || (previous.y & 0x7f800000u) == 0x7f800000u)
        atomic_fetch_or_explicit(recover, 1u, memory_order_relaxed);
}

// Nonfinite prior sums require current-mask entropy columns too, so removal of
// a NaN can recover. Independent codes are consumed directly by the reducer.
kernel void float_ans_decode_selected(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device uint *words [[buffer(5)]],
    device uint4 *descriptors [[buffer(6)]],
    device const uchar *changed [[buffer(7)]],
    device const uchar *mask [[buffer(8)]],
    device const uint *recover [[buffer(9)]],
    constant uint &scans [[buffer(10)]],
    uint pixel [[thread_position_in_grid]]) {
    // Changed columns are handled by the parallel literal/constant decoder.
    // This dispatch supplies only the exceptional nonfinite recovery columns.
    if (!recover[0] || !mask[pixel] || changed[pixel]) return;
    if (models[pixel * 2] >= 253 && models[pixel * 2 + 1] >= 253) return;
    StreamReader low(payload, offsets, models, decoding, pixel * 2);
    StreamReader high(payload, offsets, models, decoding, pixel * 2 + 1);
    for (uint frame = 0; frame < scans; ++frame)
        words[frame * float_ans_pixels + pixel] = low.next() | (high.next() << 16);
    if (!low.finished() || !high.finished())
        atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

// Decode only changed entropy columns into bounded scratch. Literal/constant
// streams have no state dependency and are read directly by the reducer.
kernel void float_ans_decode_changes(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device uint *words [[buffer(5)]],
    device const int2 *entries [[buffer(6)]],
    constant uint &scans [[buffer(7)]],
    uint entry [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint pixel = uint(entries[entry].x);
    uint stream = pixel * 2;
    uint low_model = models[stream], high_model = models[stream + 1];
    if (low_model >= 253 && high_model >= 253) {
        // The detector consumer can read these independent codes in place.
        return;
    } else if (lane == 0) {
        StreamReader low(payload, offsets, models, decoding, stream);
        StreamReader high(payload, offsets, models, decoding, stream + 1);
        for (uint frame = 0; frame < scans; ++frame)
            words[frame * float_ans_pixels + pixel] = low.next() | (high.next() << 16);
        if (!low.finished() || !high.finished())
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    }
}

// One independent ANS stream pair per SIMD lane. Each lane preserves its
// original frame order; only the schedule of independent detector columns
// changes. The reducer still consumes exactly the same decoded IEEE words.
kernel void float_ans_decode_changes_parallel(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device uint *words [[buffer(5)]],
    device const int2 *entries [[buffer(6)]],
    constant uint &scans [[buffer(7)]],
    constant uint &entryCount [[buffer(13)]],
    uint entry [[thread_position_in_grid]]) {
    if (entry >= entryCount) return;
    uint pixel = uint(entries[entry].x);
    uint stream = pixel * 2;
    if (models[stream] >= 253 && models[stream + 1] >= 253) return;
    StreamReader low(payload, offsets, models, decoding, stream);
    StreamReader high(payload, offsets, models, decoding, stream + 1);
    for (uint frame = 0; frame < scans; ++frame)
        words[frame * float_ans_pixels + pixel] = low.next() | (high.next() << 16);
    if (!low.finished() || !high.finished())
        atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

// Specialize the common paired-entropy case once per changed column. Both
// independent table lookups can proceed before either rANS state is updated.
kernel void float_ans_decode_changes_parallel_entropy(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device uint *words [[buffer(5)]],
    device const int2 *entries [[buffer(6)]],
    constant uint &scans [[buffer(7)]],
    constant uint &entryCount [[buffer(13)]],
    uint entry [[thread_position_in_grid]]) {
    if (entry >= entryCount) return;
    uint pixel = uint(entries[entry].x);
    uint stream = pixel * 2;
    uint lowModel = models[stream], highModel = models[stream + 1];
    if (lowModel >= 253 && highModel >= 253) return;
    StreamReader low(payload, offsets, models, decoding, stream);
    StreamReader high(payload, offsets, models, decoding, stream + 1);
    if (lowModel < 64 && highModel < 64 && low.valid && high.valid) {
        for (uint frame = 0; frame < scans; ++frame) {
            uint2 pair = float_ans_next_entropy_pair(low, high);
            words[frame * float_ans_pixels + pixel] = pair.x | (pair.y << 16);
        }
    } else {
        for (uint frame = 0; frame < scans; ++frame)
            words[frame * float_ans_pixels + pixel] = low.next() | (high.next() << 16);
    }
    if (!low.finished() || !high.finished())
        atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
}

// The same compensated, scan-ordered mean as empad_mean_diffraction. Each
// detector pixel owns its accumulator across ordered chunk dispatches. Unlike
// a point query, a region must not decode unrelated chunks on every movement.
inline void float_ans_mean_add(float value, thread float& sum, thread float& correction) {
    if (isfinite(value) && isfinite(sum)) {
        float adjusted = value - correction;
        float next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    } else { sum += value; correction = 0; }
}

inline uint float_ans_independent(device const uchar* payload, uint start,
                                  uint model, uint frame) {
    if (model == 253) return 0;
    uint at = start + (model == 254 ? frame * 2 : 0);
    return uint(payload[at]) | (uint(payload[at + 1]) << 8);
}

kernel void float_ans_region_mean(
    device const uchar* payload [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const uchar* models [[buffer(2)]],
    device const uint* table [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]],
    device float2* accumulator [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* background [[buffer(7)]],
    constant uint& corrected [[buffer(8)]],
    constant uint4& parameters [[buffer(9)]],
    constant uint* frames [[buffer(10)]],
    uint pixel [[thread_position_in_grid]]) {
    uint stream = pixel * 2;
    float2 previous = parameters.w ? float2(0) : accumulator[pixel];
    float sum = previous.x, correction = previous.y;
    uint lowModel = models[stream], highModel = models[stream + 1];
    if (lowModel >= 253 && highModel >= 253) {
        uint lowStart = offsets[stream], highStart = offsets[stream + 1];
        uint lowBytes = lowModel == 253 ? 0 : lowModel == 255 ? 2 : parameters.x * 2;
        uint highBytes = highModel == 253 ? 0 : highModel == 255 ? 2 : parameters.x * 2;
        if (highStart - lowStart != lowBytes || offsets[stream + 2] - highStart != highBytes) {
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
            return;
        }
        for (uint index = 0; index < parameters.y; ++index) {
            uint frame = frames[index];
            uint word = float_ans_independent(payload, lowStart, lowModel, frame)
                | (float_ans_independent(payload, highStart, highModel, frame) << 16);
            float value = as_type<float>(word);
            if (corrected) value -= background[pixel];
            float_ans_mean_add(value / float(parameters.z), sum, correction);
        }
    } else {
        StreamReader low(payload, offsets, models, table, stream);
        StreamReader high(payload, offsets, models, table, stream + 1);
        uint index = 0;
        for (uint frame = 0; frame < parameters.x; ++frame) {
            uint word = low.next() | (high.next() << 16);
            if (index < parameters.y && frame == frames[index]) {
                float value = as_type<float>(word);
                if (corrected) value -= background[pixel];
                float_ans_mean_add(value / float(parameters.z), sum, correction);
                ++index;
            }
        }
        if (!low.finished() || !high.finished())
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    }
    accumulator[pixel] = float2(sum, correction);
    output[pixel] = sum;
}

kernel void float_ans_join_words(
    device const uint *lanes [[buffer(0)]],
    device uint *words [[buffer(1)]],
    device uint4 *descriptors [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    words[index] = lanes[index * 2] | (lanes[index * 2 + 1] << 16);
    if (index % 128 == 0) descriptors[index / 128] = uint4(0, 32, 0, index);
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


// Exact per-scan DPC moments (total, row moment, column moment) over every
// stored original count, for the runtime ANS representation.
//
// One SIMD group owns one stored packet and walks the detector in stripes:
// lane l decodes the streams of pixels {stripe * 32 * S + lane + 32 * i} for
// i < S, so consecutive lanes touch consecutive streams. Every reader consumes
// its stream exactly once, which keeps the terminal-state check in
// StreamReader::finished meaningful for a whole pass. The S streams of a lane
// are unrolled into scalars rather than arrays: this loop is the whole cost of
// the pass, and the indexed form was measurably slower than the plain decode.
//
// Per-scan moments are reduced across the lanes and published straight to the
// output with split-word atomic adds, which keep UInt64 sums exact on Apple
// GPUs that have no 64-bit atomics. No threadgroup memory and no barriers are
// used, so the serial rANS decode is the only critical path. Values come from
// the stored payload through the same StreamReader the decode path uses, so the
// basis is the exact integer quantity of the retained counts. Nothing is
// written into the `.qem` payload and no dense copy of the acquisition is made.
//
// errors[0] records failure, errors[1..3] the first failing stream index, its
// model, and a reason code, all through atomic min so a diagnosis survives.
// Compile-time constants: `constant uint` is a runtime value, which leaves the
// reader loop below with a non-constant trip count and forces its state into
// thread-local memory. These must be macros to stay in registers.
#define SC_MOMENT_WORDS 6
#define SC_MOMENT_STREAMS 4
#define SC_MOMENT_REASON_INVALID 1
#define SC_MOMENT_REASON_UNCONSUMED 2
#define SC_MOMENT_REASON_STATE 3

inline void sc_atomic_add_u64(device atomic_uint *target, ulong value) {
    uint add = uint(value), carry = uint(value >> 32);
    uint previous = atomic_fetch_add_explicit(target, add, memory_order_relaxed);
    if (previous > 0xffffffffu - add) carry += 1u;
    if (carry) atomic_fetch_add_explicit(target + 1, carry, memory_order_relaxed);
}

// Shuffle the two halves of every 64-bit partial separately and add in 64 bits
// so carries never squeeze a moment through 32 bits.
inline ulong sc_reduce_u64(ulong value, uint width) {
    for (uint delta = width / 2; delta; delta >>= 1) {
        uint low = simd_shuffle_down(uint(value), delta);
        uint high = simd_shuffle_down(uint(value >> 32), delta);
        value += ulong(low) | (ulong(high) << 32);
    }
    return value;
}

// The narrow path keeps every lane partial inside 32 bits, which the host
// guarantees by only selecting it when `streams * maximumValue * maximumWeight`
// fits UInt32. Splitting into 16-bit halves keeps each `simd_sum` inside 21 bits
// and the recombination happens in 64 bits, so the cross-lane sum is exact.
inline ulong sc_reduce_u32_split(uint value) {
    return (ulong(simd_sum(value >> 16)) << 16) + ulong(simd_sum(value & 0xffffu));
}

inline void sc_moment_failure(
    device atomic_uint *errors, uint stream, uint model, uint reason) {
    atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    atomic_fetch_min_explicit(errors + 1, stream, memory_order_relaxed);
    atomic_fetch_min_explicit(errors + 2, model, memory_order_relaxed);
    atomic_fetch_min_explicit(errors + 3, reason, memory_order_relaxed);
}

// One stored stream of one detector pixel. `active` is false for a validated
// zero stream, which contributes nothing to any scan position and needs no
// decoding; `entropy` hoists the model test out of the scan loop.
struct SCMomentStream {
    StreamReader reader;
    bool active;
    bool entropy;
    bool report;
    uint rowWeight;
    uint columnWeight;
};

inline SCMomentStream sc_moment_stream(
    device const uchar *payload, device const uint *offsets, device const uchar *models,
    device const uint *decoding, uint stream, uint pixel, uint pixels, uint columns) {
    SCMomentStream state;
    state.reader = StreamReader(payload, offsets, models, decoding, stream);
    state.report = pixel < pixels;
    state.active = state.report && state.reader.model != 253;
    state.entropy = state.reader.model < SC_MODELS;
    state.rowWeight = pixel / columns;
    state.columnWeight = pixel % columns;
    return state;
}

// The accumulator is a scalar triple; the wide and narrow variants differ only
// in whether a lane's partial fits UInt32.
struct SCMomentLane {
    uint total;
    uint smallRow;
    uint smallColumn;
    ulong row;
    ulong column;
};

template <bool Narrow>
inline void sc_moment_take(
    thread SCMomentLane *lane, thread const SCMomentStream *stream, uint value) {
    lane->total += value;
    if (Narrow) {
        lane->smallRow += value * stream->rowWeight;
        lane->smallColumn += value * stream->columnWeight;
    } else {
        lane->row += ulong(value) * ulong(stream->rowWeight);
        lane->column += ulong(value) * ulong(stream->columnWeight);
    }
}

inline uint sc_moment_next(thread SCMomentStream *stream) {
    return stream->entropy ? stream->reader.next_entropy() : stream->reader.next();
}

// Every reader of a validated stream must have consumed its payload exactly and
// finished in the canonical state; anything else is reported through `errors`.
inline void sc_moment_check(
    device atomic_uint *errors, thread const SCMomentStream *stream, uint streamIndex) {
    if (!stream->report) return;
    StreamReader reader = stream->reader;
    if (!reader.valid) {
        sc_moment_failure(errors, streamIndex, uint(reader.model), SC_MOMENT_REASON_INVALID);
    } else if (reader.cursor != reader.end) {
        sc_moment_failure(errors, streamIndex, uint(reader.model), SC_MOMENT_REASON_UNCONSUMED);
    } else if (reader.model < SC_MODELS && reader.state != SC_LOWER) {
        sc_moment_failure(errors, streamIndex, uint(reader.model), SC_MOMENT_REASON_STATE);
    }
}

// `EmitTotals` additionally sums the same decoded counts over every scan
// position of every detector pixel, which is exactly the quantity the detector
// column pass derives on its own. Folding it in here means one decode of the
// stored payload produces both products, so an acquisition that wants both
// never pays for the counts twice. The flag is a template constant, so the
// moment-only instantiation keeps the code it had.
template <bool Narrow, bool EmitTotals>
inline void sc_exact_moments_body(
    device const uchar *payload, device const uint *offsets, device const uchar *models,
    device const uint *decoding, device atomic_uint *errors, device atomic_uint *output,
    constant ulong *p, uint job, uint localIndex,
    device atomic_uint *totals, device const uchar *valid, device uint *partials) {
    uint scans = uint(p[0]), pixels = uint(p[1]), interval = uint(p[2]);
    uint firstScan = uint(p[3]), columns = uint(p[4]), outputFirst = uint(p[5]);
    uint stripeGroups = max(1u, uint(p[6]));
    // A threadgroup may carry several SIMD groups; each owns its own stripe run.
    uint unitsPerGroup = max(1u, uint(p[7]));
    uint lane = localIndex % 32u;
    uint unit = job * unitsPerGroup + localIndex / 32u;
    uint blocks = (scans + interval - 1) / interval;
    if (unit >= blocks * stripeGroups) return;
    uint block = unit / stripeGroups, stripeGroup = unit % stripeGroups;
    uint localFirst = block * interval;
    uint count = min(interval, scans - localFirst);
    // A stripe's partial is folded into this unit's own slice of `partials`
    // rather than added to the shared per-scan slot. That slot is the same
    // address for every stripe of a packet, so an atomic add there serializes a
    // whole detector on one word and costs more than the decode it publishes.
    // The slice is private to this unit, so the fold is a plain read, add and
    // write: no atomic, no contention, and the combine pass resolves the slices
    // afterwards. The host zeroes the slice before every use.
    device uint *slice = partials + ulong(unit) * ulong(interval) * SC_MOMENT_WORDS;
    uint width = 32u * SC_MOMENT_STREAMS;
    uint stripes = (pixels + width - 1) / width;
    for (uint stripe = stripeGroup; stripe < stripes; stripe += stripeGroups) {
        uint base = stripe * width + lane;
        // The four stored streams of this lane are named rather than held in an
        // array: an indexed reader set lands in thread-local memory and costs
        // roughly 3x the plain decode loop. This is the hot path of the pass.
        uint origin = block * pixels + base;
        SCMomentStream s0 = sc_moment_stream(
            payload, offsets, models, decoding, origin, base, pixels, columns);
        SCMomentStream s1 = sc_moment_stream(
            payload, offsets, models, decoding, origin + 32u, base + 32u, pixels, columns);
        SCMomentStream s2 = sc_moment_stream(
            payload, offsets, models, decoding, origin + 64u, base + 64u, pixels, columns);
        SCMomentStream s3 = sc_moment_stream(
            payload, offsets, models, decoding, origin + 96u, base + 96u, pixels, columns);
        uint pixelTotal0 = 0u, pixelTotal1 = 0u, pixelTotal2 = 0u, pixelTotal3 = 0u;
        for (uint scan = 0; scan < count; ++scan) {
            SCMomentLane lane_totals = SCMomentLane{0u, 0u, 0u, 0ul, 0ul};
            if (s0.active) {
                uint value = sc_moment_next(&s0);
                sc_moment_take<Narrow>(&lane_totals, &s0, value);
                if (EmitTotals) pixelTotal0 += value;
            }
            if (s1.active) {
                uint value = sc_moment_next(&s1);
                sc_moment_take<Narrow>(&lane_totals, &s1, value);
                if (EmitTotals) pixelTotal1 += value;
            }
            if (s2.active) {
                uint value = sc_moment_next(&s2);
                sc_moment_take<Narrow>(&lane_totals, &s2, value);
                if (EmitTotals) pixelTotal2 += value;
            }
            if (s3.active) {
                uint value = sc_moment_next(&s3);
                sc_moment_take<Narrow>(&lane_totals, &s3, value);
                if (EmitTotals) pixelTotal3 += value;
            }
            // Every lane reaches the reductions: a stripe with no pixel for a
            // lane is skipped by marking the stream inactive, never by leaving
            // the loop.
            uint summedTotal = simd_sum(lane_totals.total);
            ulong summedRow = 0ul, summedColumn = 0ul;
            if (Narrow) {
                summedRow = sc_reduce_u32_split(lane_totals.smallRow);
                summedColumn = sc_reduce_u32_split(lane_totals.smallColumn);
            } else {
                summedRow = sc_reduce_u64(lane_totals.row, 32u);
                summedColumn = sc_reduce_u64(lane_totals.column, 32u);
            }
            if (lane == 0) {
                device uint *slot = slice + ulong(scan) * SC_MOMENT_WORDS;
                // Split-word accumulation: each stripe adds its partial to the
                // words and carries the 32-bit overflow into the next word. A
                // stripe total always fits in 32 bits, so the carry is one bit.
                uint totalSum = slot[0] + summedTotal;
                slot[1] += uint(ulong(summedTotal) >> 32) + (totalSum < slot[0] ? 1u : 0u);
                slot[0] = totalSum;
                uint rowLow = slot[2] + uint(summedRow);
                slot[3] += uint(summedRow >> 32) + (rowLow < slot[2] ? 1u : 0u);
                slot[2] = rowLow;
                uint columnLow = slot[4] + uint(summedColumn);
                slot[5] += uint(summedColumn >> 32) + (columnLow < slot[4] ? 1u : 0u);
                slot[4] = columnLow;
            }
        }
        if (EmitTotals) {
            // One lane owns these four pixels for this stripe and no other lane
            // touches them in this dispatch, so only the cross-chunk sum needs
            // to be atomic. A bad pixel keeps the zero the caller pre-filled.
            if (base < pixels && valid[base])
                sc_atomic_add_u64(totals + ulong(base) * 2, ulong(pixelTotal0));
            if (base + 32u < pixels && valid[base + 32u])
                sc_atomic_add_u64(totals + ulong(base + 32u) * 2, ulong(pixelTotal1));
            if (base + 64u < pixels && valid[base + 64u])
                sc_atomic_add_u64(totals + ulong(base + 64u) * 2, ulong(pixelTotal2));
            if (base + 96u < pixels && valid[base + 96u])
                sc_atomic_add_u64(totals + ulong(base + 96u) * 2, ulong(pixelTotal3));
        }
        sc_moment_check(errors, &s0, origin);
        sc_moment_check(errors, &s1, origin + 32u);
        sc_moment_check(errors, &s2, origin + 64u);
        sc_moment_check(errors, &s3, origin + 96u);
    }
}

kernel void streamed_counts_exact_moments(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device atomic_uint *output [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    uint job [[threadgroup_position_in_grid]],
    device uint *partials [[buffer(9)]],
    uint localIndex [[thread_index_in_threadgroup]]) {
    sc_exact_moments_body<false, false>(
        payload, offsets, models, decoding, errors, output, p, job, localIndex, nullptr, nullptr, partials);
}

// Same basis for count ranges whose lanes stay inside 32 bits: the host selects
// this entry point only when streams * maximumValue * maximumWeight fits
// UInt32, so every product and every lane partial is exact by construction.
kernel void streamed_counts_exact_moments_narrow(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device atomic_uint *output [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    uint job [[threadgroup_position_in_grid]],
    device uint *partials [[buffer(9)]],
    uint localIndex [[thread_index_in_threadgroup]]) {
    sc_exact_moments_body<true, false>(
        payload, offsets, models, decoding, errors, output, p, job, localIndex, nullptr, nullptr, partials);
}

// Folds one packet's per-unit stripe partials into the per-scan basis. Each scan
// position is written by exactly one thread, so this pass needs no atomics, and
// it runs after the decode has finished rather than inside it.
kernel void streamed_counts_exact_moments_combine(
    device const uint *partials [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    uint scans = uint(p[0]), interval = uint(p[1]), groups = uint(p[2]);
    uint firstScan = uint(p[3]), outputFirst = uint(p[4]);
    if (i >= scans) return;
    uint block = i / interval, within = i % interval;
    if (within >= min(interval, scans - block * interval)) return;
    ulong total = 0ul, row = 0ul, column = 0ul;
    device const uint *slice =
        partials + (ulong(block) * ulong(groups) * ulong(interval) + ulong(within)) * SC_MOMENT_WORDS;
    for (uint group = 0; group < groups; ++group) {
        total += ulong(slice[0]) | (ulong(slice[1]) << 32);
        row += ulong(slice[2]) | (ulong(slice[3]) << 32);
        column += ulong(slice[4]) | (ulong(slice[5]) << 32);
        slice += ulong(interval) * SC_MOMENT_WORDS;
    }
    device uint *slot = output + ulong(outputFirst + firstScan + i) * SC_MOMENT_WORDS;
    slot[0] = uint(total);
    slot[1] = uint(total >> 32);
    slot[2] = uint(row);
    slot[3] = uint(row >> 32);
    slot[4] = uint(column);
    slot[5] = uint(column >> 32);
}

// Same basis, and in the same pass the exact sum of stored counts over every
// scan position of every detector pixel. `totals` is held as split-word
// atomic pairs so the UInt64 per-pixel sum stays exact on Apple GPUs.
kernel void streamed_counts_exact_moments_totals(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device atomic_uint *output [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    device atomic_uint *totals [[buffer(7)]],
    device const uchar *valid [[buffer(8)]],
    uint job [[threadgroup_position_in_grid]],
    device uint *partials [[buffer(9)]],
    uint localIndex [[thread_index_in_threadgroup]]) {
    sc_exact_moments_body<false, true>(
        payload, offsets, models, decoding, errors, output, p, job, localIndex, totals, valid, partials);
}

kernel void streamed_counts_exact_moments_totals_narrow(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *models [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]],
    device atomic_uint *output [[buffer(5)]],
    constant ulong *p [[buffer(6)]],
    device atomic_uint *totals [[buffer(7)]],
    device const uchar *valid [[buffer(8)]],
    uint job [[threadgroup_position_in_grid]],
    device uint *partials [[buffer(9)]],
    uint localIndex [[thread_index_in_threadgroup]]) {
    sc_exact_moments_body<true, true>(
        payload, offsets, models, decoding, errors, output, p, job, localIndex, totals, valid, partials);
}
