#include <metal_stdlib>
using namespace metal;

constant uint PRT_STATES = 1024u;
constant uint PRT_SYMBOLS = 1089u;
constant uint PRT_ESCAPE = 1088u;
constant uint prt_sparse_slack_requested [[function_constant(4)]];
constant uint prt_sparse_slack = is_function_constant_defined(
    prt_sparse_slack_requested) ? prt_sparse_slack_requested : 0u;
// Speed-first event coding: a stream whose values are all <= 128 and whose
// nonzero count is at most this limit is stored as exact (position, value)
// events even when entropy coding would be smaller. 0 keeps byte-optimal modes.
constant uint prt_sparse_max_nonzero_requested [[function_constant(27)]];
constant uint prt_sparse_max_nonzero = is_function_constant_defined(
    prt_sparse_max_nonzero_requested) ? prt_sparse_max_nonzero_requested : 0u;
// FC42: stream j of every 512-scan packet holds detector pixel stream_pixels[j]
// (radial1 order), so the streams of nearby detector pixels are adjacent in the
// payload, offsets and modes. Every consumer maps pixel ids to stream ranks.
constant bool prt_stream_permutation_requested [[function_constant(42)]];
constant bool prt_stream_permutation = is_function_constant_defined(
    prt_stream_permutation_requested) ? prt_stream_permutation_requested : false;
constant bool prt_scratchless_encode_requested [[function_constant(7)]];
constant bool prt_scratchless_encode = is_function_constant_defined(
    prt_scratchless_encode_requested) ? prt_scratchless_encode_requested : false;
constant bool prt_compact_offsets_requested [[function_constant(20)]];
constant bool prt_compact_offsets = is_function_constant_defined(
    prt_compact_offsets_requested) ? prt_compact_offsets_requested : false;
// Synthetic-codec experiment only. Resident detector paths keep
// prt_entropy_mode's original 64..95 range and reject this opt-in format.
constant bool prt_interleaved_states_requested [[function_constant(24)]];
constant bool prt_interleaved_states = is_function_constant_defined(
    prt_interleaved_states_requested) ? prt_interleaved_states_requested : false;
constant bool prt_vector_pair_reduction_requested [[function_constant(25)]];
constant bool prt_vector_pair_reduction = is_function_constant_defined(
    prt_vector_pair_reduction_requested) ? prt_vector_pair_reduction_requested : false;
constant uint PRT_INTERVAL = 512u;
// FC45, speed-first exact compact events (mode 251): a stream with 3..T nonzero
// counts, all at most 263, is stored as one byte per event, (gap << 3) | value,
// where position = previous + 1 + gap (first event: gap). gap field 31 adds the
// next byte (and one more byte when that byte is 255); value field 0 means the
// value is 8 plus the next byte. 0 keeps the default modes.
// FC46: mode-251 residual decode takes two plain event bytes per trip when neither
// needs an extension, and falls back to the single-byte state machine otherwise.
constant bool prt_compact_pairs_requested [[function_constant(46)]];
constant bool prt_compact_pairs = is_function_constant_defined(
    prt_compact_pairs_requested) ? prt_compact_pairs_requested : false;
// FC47: with FC46, take four plain event bytes per trip when all four are plain.
constant bool prt_compact_quads_requested [[function_constant(47)]];
constant bool prt_compact_quads = is_function_constant_defined(
    prt_compact_quads_requested) ? prt_compact_quads_requested : false;
constant uint prt_compact_event_max_nonzero_requested [[function_constant(45)]];
constant uint prt_compact_event_max_nonzero = is_function_constant_defined(
    prt_compact_event_max_nonzero_requested) ? prt_compact_event_max_nonzero_requested : 0u;

struct PRTCompactEvents {
    uint cursor;
    uint end;
    uint next_position;
    bool valid;
};

// Decode the next mode-251 event; false at the end of the stream or on malformed data
// (then valid is false unless the stream ended cleanly).
inline bool prt_compact_next(
    device const uchar *payload, thread PRTCompactEvents &events,
    thread uint &position, thread uint &value) {
    if (!events.valid || events.cursor >= events.end) return false;
    uint byte = uint(payload[events.cursor++]);
    uint gap = byte >> 3u;
    uint count = byte & 7u;
    if (gap == 31u) {
        if (events.cursor >= events.end) { events.valid = false; return false; }
        uint extension = uint(payload[events.cursor++]);
        gap += extension;
        if (extension == 255u) {
            if (events.cursor >= events.end) { events.valid = false; return false; }
            gap += uint(payload[events.cursor++]);
        }
    }
    if (count == 0u) {
        if (events.cursor >= events.end) { events.valid = false; return false; }
        count = 8u + uint(payload[events.cursor++]);
    }
    position = events.next_position + gap;
    if (position >= PRT_INTERVAL) { events.valid = false; return false; }
    events.next_position = position + 1u;
    value = count;
    return true;
}

inline uint prt_source_offset(
    device const uint *offsets, uint stream, uint stream_count) {
    if (!prt_compact_offsets) return offsets[stream];
    uint base_count = (stream_count >> 5u) + 1u;
    device const uchar *bytes = reinterpret_cast<device const uchar *>(offsets);
    device const ushort *starts = reinterpret_cast<device const ushort *>(
        bytes + ulong(base_count) * sizeof(uint));
    return offsets[stream >> 5u] + uint(starts[stream]);
}

inline uint prt_raw(
    device const uchar *raw, ulong index, uint item_bytes) {
    return item_bytes == 1u
        ? uint(raw[index])
        : uint(reinterpret_cast<device const ushort *>(raw)[index]);
}

inline void prt_scratch_byte(
    device uchar *scratch, device const uint *direct_offsets,
    uint streams, uint stream, uint byte, uchar value, bool direct_write) {
    if (prt_scratchless_encode) {
        if (direct_write) {
            // FC42: (first, end) pairs per pixel-order stream point at rank positions.
            uint first = direct_offsets[prt_stream_permutation ? 2u * stream : stream];
            uint end = direct_offsets[prt_stream_permutation ? 2u * stream + 1u : stream + 1u];
            if (end >= first && byte < end - first)
                scratch[first + byte] = value;
        }
    } else {
        scratch[ulong(byte) * streams + stream] = value;
    }
}

inline void prt_append_bits(
    device uchar *scratch, uint streams, uint stream,
    thread ulong &buffer, thread uint &available, thread uint &emitted,
    thread bool &overflow, uint scratch_stride, uint header_bytes,
    device const uint *direct_offsets, bool direct_write,
    uint value, uint count) {
    if (count == 0u || overflow) return;
    buffer |= ulong(value) << available;
    available += count;
    while (available >= 8u) {
        if (header_bytes + emitted >= scratch_stride) {
            overflow = true;
            return;
        }
        prt_scratch_byte(
            scratch, direct_offsets, streams, stream,
            header_bytes + emitted, uchar(buffer), direct_write);
        ++emitted;
        buffer >>= 8u;
        available -= 8u;
    }
}

inline uint prt_model(uint clipped_sum, uint count) {
    float mean = float(clipped_sum) / float(count);
    float scaled = (log(max(mean, 0.002f)) - log(0.002f))
        * float(31.0 / log(16000.0));
    return uint(clamp(int(rint(scaled)), 0, 31));
}

inline bool prt_entropy_mode(uint mode) {
    return mode >= 64u && mode < 96u;
}

inline uint prt_entropy_model(uint mode) {
    return mode - 64u;
}

kernel void paired_runtime_tans_encode(
    device const uchar *raw [[buffer(0)]],
    device const uint *frequency_starts [[buffer(1)]],
    device const ushort *encoding [[buffer(2)]],
    device uchar *scratch [[buffer(3)]],
    device uint *sizes [[buffer(4)]],
    device uchar *modes [[buffer(5)]],
    device atomic_uint *failure [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    device const uint *direct_offsets [[buffer(8)]],
    device const uint *stream_pixels [[buffer(9)]],
    uint stream [[thread_position_in_grid]]) {
    uint scans = p[0], pixels = p[1], streams = p[2], item_bytes = p[3];
    uint scratch_stride = p[4];
    bool direct_write = prt_scratchless_encode && p[5] != 0u;
    uint header_bytes = prt_interleaved_states ? 3u : 2u;
    if (stream >= streams) return;
    // Streams are always encoded in pixel order; FC42 only changes where the
    // scratchless pass writes them (see prt_scratch_byte) and compaction.
    uint pixel = stream % pixels;
    (void)stream_pixels;
    uint first = (stream / pixels) * PRT_INTERVAL;
    uint count = min(PRT_INTERVAL, scans - first);
    if (count == 0u || scratch_stride < 2u * count || (item_bytes != 1u && item_bytes != 2u)) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
        return;
    }

    uint minimum = 65535u, maximum = 0u, nonzero = 0u, clipped_sum = 0u;
    uint first_event = 0u, second_event = 0u;
    for (uint scan = 0u; scan < count; ++scan) {
        uint value = prt_raw(raw, ulong(first + scan) * pixels + pixel, item_bytes);
        minimum = min(minimum, value);
        maximum = max(maximum, value);
        if (value != 0u && nonzero < 2u) {
            uint event = (scan << 7u) | (value - 1u);
            if (nonzero == 0u) first_event = event;
            else second_event = event;
        }
        nonzero += value != 0u;
        clipped_sum += min(value, 32u);
    }
    if (minimum == maximum) {
        if (maximum == 0u) {
            modes[stream] = 253u;
            sizes[stream] = 0u;
        } else {
            modes[stream] = 255u;
            sizes[stream] = 2u;
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 0u, uchar(maximum), direct_write);
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 1u, uchar(maximum >> 8u), direct_write);
        }
        return;
    }
    if (maximum <= 128u && nonzero <= 2u) {
        if (nonzero > 0u) {
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 0u, uchar(first_event), direct_write);
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 1u, uchar(first_event >> 8u), direct_write);
        }
        if (nonzero > 1u) {
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 2u, uchar(second_event), direct_write);
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 3u, uchar(second_event >> 8u), direct_write);
        }
        modes[stream] = 252u;
        sizes[stream] = 2u * nonzero;
        return;
    }
    if (prt_compact_event_max_nonzero != 0u && nonzero >= 3u
        && nonzero <= prt_compact_event_max_nonzero && maximum <= 263u) {
        uint compact_size = 0u;
        uint previous_end = 0u;
        for (uint scan = 0u; scan < count; ++scan) {
            uint value = prt_raw(raw, ulong(first + scan) * pixels + pixel, item_bytes);
            if (value == 0u) continue;
            uint gap = scan - previous_end;
            compact_size += 1u + (gap >= 31u ? (gap - 31u >= 255u ? 2u : 1u) : 0u) + (value > 7u ? 1u : 0u);
            previous_end = scan + 1u;
        }
        if (compact_size <= scratch_stride) {
            uint emitted_compact = 0u;
            previous_end = 0u;
            for (uint scan = 0u; scan < count; ++scan) {
                uint value = prt_raw(raw, ulong(first + scan) * pixels + pixel, item_bytes);
                if (value == 0u) continue;
                uint gap = scan - previous_end;
                uint gap_field = min(gap, 31u);
                uint value_field = value <= 7u ? value : 0u;
                prt_scratch_byte(scratch, direct_offsets, streams, stream, emitted_compact++,
                    uchar((gap_field << 3u) | value_field), direct_write);
                if (gap >= 31u) {
                    uint extension = gap - 31u;
                    if (extension < 255u) {
                        prt_scratch_byte(scratch, direct_offsets, streams, stream, emitted_compact++,
                            uchar(extension), direct_write);
                    } else {
                        prt_scratch_byte(scratch, direct_offsets, streams, stream, emitted_compact++,
                            uchar(255u), direct_write);
                        prt_scratch_byte(scratch, direct_offsets, streams, stream, emitted_compact++,
                            uchar(extension - 255u), direct_write);
                    }
                }
                if (value > 7u)
                    prt_scratch_byte(scratch, direct_offsets, streams, stream, emitted_compact++,
                        uchar(value - 8u), direct_write);
                previous_end = scan + 1u;
            }
            modes[stream] = 251u;
            sizes[stream] = emitted_compact;
            return;
        }
    }

    uint model = prt_model(clipped_sum, count);
    // Declared here because the header written after the walk reads both states.
    uint states[2];
    states[0] = 0u;
    states[1] = 0u;
    uint total_bits = 0u, emitted = 0u, available = 0u;
    bool entropy_overflow = false;
    ulong bit_buffer = 0u;
    uint pairs = (count + 1u) / 2u;
    if (item_bytes == 2u && !prt_interleaved_states) {
        // Software-pipelined reverse walk: the next pair's two loads are issued
        // before the current pair's coder runs, so each warp keeps more lines in
        // flight than the plain backward loop. Coding order, emitted bytes, and
        // state updates are unchanged.
        uint state = 0u;
        device const ushort *rows = reinterpret_cast<device const ushort *>(raw)
            + ulong(first + 2u * (pairs - 1u)) * pixels + pixel;
        uint a = uint(rows[0]);
        uint b = 2u * (pairs - 1u) + 1u < count ? uint(rows[pixels]) : 0u;
        for (uint reverse_pair = pairs; reverse_pair > 0u; --reverse_pair) {
            uint next_a = 0u, next_b = 0u;
            if (reverse_pair > 1u) {
                device const ushort *next_rows = rows - 2u * pixels;
                uint next_index = 2u * reverse_pair - 4u;
                next_a = uint(next_rows[0]);
                next_b = next_index + 1u < count ? uint(next_rows[pixels]) : 0u;
                rows = next_rows;
            }
            uint symbol = a < 32u && b < 32u ? a * 33u + b : PRT_ESCAPE;
            uint frequency_start = frequency_starts[model * PRT_SYMBOLS + symbol];
            uint frequency = frequency_start >> 16u;
            if (symbol == PRT_ESCAPE || frequency == 0u) {
                symbol = PRT_ESCAPE;
                frequency_start = frequency_starts[model * PRT_SYMBOLS + symbol];
                frequency = frequency_start >> 16u;
                if (a < 64u && b < 64u) {
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write,
                        a | (b << 6u), 13u);
                    total_bits += 13u;
                } else {
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write, b, 16u);
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write, a, 16u);
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write, 4096u, 13u);
                    total_bits += 45u;
                }
            }
            uint start = frequency_start & 65535u;
            uint y = PRT_STATES + state;
            uint bits = 10u - (31u - clz(frequency));
            if (y < (frequency << bits)) --bits;
            uint rank = (y >> bits) - frequency;
            uint low = bits == 0u ? 0u : y & ((1u << bits) - 1u);
            prt_append_bits(
                scratch, streams, stream, bit_buffer, available, emitted,
                entropy_overflow, scratch_stride, header_bytes,
                direct_offsets, direct_write, low, bits);
            total_bits += bits;
            state = uint(encoding[model * PRT_STATES + start + rank]);
            if (entropy_overflow) break;
            a = next_a;
            b = next_b;
        }
        // Publish the single-stream chain state for the non-interleaved header.
        states[0] = state;
    } else {
        for (uint reverse_pair = pairs; reverse_pair > 0u; --reverse_pair) {
            uint pair_index = reverse_pair - 1u;
            uint state_index = prt_interleaved_states ? (pair_index & 1u) : 0u;
            uint state = states[state_index];
            uint index = pair_index * 2u;
            uint a = prt_raw(raw, ulong(first + index) * pixels + pixel, item_bytes);
            uint b = index + 1u < count
                ? prt_raw(raw, ulong(first + index + 1u) * pixels + pixel, item_bytes)
                : 0u;
            uint symbol = a < 32u && b < 32u ? a * 33u + b : PRT_ESCAPE;
            uint frequency_start = frequency_starts[model * PRT_SYMBOLS + symbol];
            uint frequency = frequency_start >> 16u;
            if (symbol == PRT_ESCAPE || frequency == 0u) {
                symbol = PRT_ESCAPE;
                frequency_start = frequency_starts[model * PRT_SYMBOLS + symbol];
                frequency = frequency_start >> 16u;
                if (a < 64u && b < 64u) {
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write,
                        a | (b << 6u), 13u);
                    total_bits += 13u;
                } else {
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write, b, 16u);
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write, a, 16u);
                    prt_append_bits(
                        scratch, streams, stream, bit_buffer, available, emitted,
                        entropy_overflow, scratch_stride, header_bytes,
                        direct_offsets, direct_write, 4096u, 13u);
                    total_bits += 45u;
                }
            }
            uint start = frequency_start & 65535u;
            uint y = PRT_STATES + state;
            uint bits = 10u - (31u - clz(frequency));
            if (y < (frequency << bits)) --bits;
            uint rank = (y >> bits) - frequency;
            uint low = bits == 0u ? 0u : y & ((1u << bits) - 1u);
            prt_append_bits(
                scratch, streams, stream, bit_buffer, available, emitted,
                entropy_overflow, scratch_stride, header_bytes,
                direct_offsets, direct_write, low, bits);
            total_bits += bits;
            states[state_index] = uint(encoding[model * PRT_STATES + start + rank]);
            if (entropy_overflow) break;
        }
    }
    if (!entropy_overflow && available != 0u) {
        if (header_bytes + emitted >= scratch_stride) {
            entropy_overflow = true;
        } else {
            prt_scratch_byte(
                scratch, direct_offsets, streams, stream,
                header_bytes + emitted, uchar(bit_buffer), direct_write);
            ++emitted;
        }
    }
    bool use_entropy = !entropy_overflow
        && emitted + header_bytes < 2u * count && total_bits < 16384u;
    uint size = use_entropy ? emitted + header_bytes : 2u * count;
    bool byte_optimal_sparse = 2u * nonzero <= size + 1u;
    bool speed_sparse = prt_sparse_slack != 0u && nonzero <= 64u
        && 2u * nonzero <= size + 1u + prt_sparse_slack;
    bool event_first_sparse = prt_sparse_max_nonzero != 0u
        && nonzero <= prt_sparse_max_nonzero;
    if (maximum <= 128u && (byte_optimal_sparse || speed_sparse || event_first_sparse)) {
        emitted = 0u;
        for (uint scan = 0u; scan < count; ++scan) {
            uint value = prt_raw(raw, ulong(first + scan) * pixels + pixel, item_bytes);
            if (value == 0u) continue;
            uint event = (scan << 7u) | (value - 1u);
            prt_scratch_byte(scratch, direct_offsets, streams, stream, emitted++, uchar(event), direct_write);
            prt_scratch_byte(scratch, direct_offsets, streams, stream, emitted++, uchar(event >> 8u), direct_write);
        }
        modes[stream] = 252u;
        sizes[stream] = emitted;
        return;
    }
    if (!use_entropy) {
        for (uint scan = 0u; scan < count; ++scan) {
            uint value = prt_raw(raw, ulong(first + scan) * pixels + pixel, item_bytes);
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 2u * scan, uchar(value), direct_write);
            prt_scratch_byte(scratch, direct_offsets, streams, stream, 2u * scan + 1u, uchar(value >> 8u), direct_write);
        }
        modes[stream] = 254u;
        sizes[stream] = 2u * count;
        return;
    }
    if (prt_interleaved_states) {
        // Reverse emission makes the reader encounter pair 0, then pair 1.
        // Their even/odd state chains therefore share one ordered bitstream.
        uint header = states[0] | (states[1] << 10u) | ((total_bits & 7u) << 20u);
        prt_scratch_byte(scratch, direct_offsets, streams, stream, 0u, uchar(header), direct_write);
        prt_scratch_byte(
            scratch, direct_offsets, streams, stream, 1u, uchar(header >> 8u), direct_write);
        prt_scratch_byte(
            scratch, direct_offsets, streams, stream, 2u, uchar(header >> 16u), direct_write);
        modes[stream] = uchar(96u + model);
    } else {
        uint header = (states[0] << 6u) | (total_bits & 7u);
        prt_scratch_byte(scratch, direct_offsets, streams, stream, 0u, uchar(header), direct_write);
        prt_scratch_byte(
            scratch, direct_offsets, streams, stream, 1u, uchar(header >> 8u), direct_write);
        modes[stream] = uchar(64u + model);
    }
    sizes[stream] = size;
}

// Reorder per-stream modes from pixel order to radial1 rank order.
kernel void paired_runtime_tans_rank_modes(
    device const uchar *encoded_modes [[buffer(0)]],
    device uchar *ranked_modes [[buffer(1)]],
    device const uint *stream_pixels [[buffer(2)]],
    device atomic_uint *failure [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint stream [[thread_position_in_grid]]) {
    uint streams = p[0], pixels = p[1];
    if (stream >= streams) return;
    uint pixel = stream_pixels[stream % pixels];
    if (pixels == 0u || pixel >= pixels) {
        atomic_store_explicit(failure, 4u, memory_order_relaxed);
        return;
    }
    ranked_modes[stream] = encoded_modes[(stream / pixels) * pixels + pixel];
}

kernel void paired_runtime_tans_compact(
    device const uchar *scratch [[buffer(0)]],
    device const uint *sizes [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device uchar *payload [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    constant uint *p [[buffer(5)]],
    device const uint *stream_pixels [[buffer(6)]],
    device const uchar *encoded_modes [[buffer(7)]],
    device uchar *ranked_modes [[buffer(8)]],
    uint stream [[thread_position_in_grid]]) {
    uint streams = p[0], scratch_stride = p[1], payload_bytes = p[2];
    if (stream >= streams) return;
    // FC42: output stream rank `stream` holds the encoded pixel-order stream
    // (packet, stream_pixels[rank]); offsets are prefix sums in rank order.
    uint source = stream;
    if (prt_stream_permutation) {
        uint pixels = p[3];
        uint pixel = stream_pixels[stream % pixels];
        if (pixels == 0u || pixel >= pixels) {
            atomic_store_explicit(failure, 3u, memory_order_relaxed);
            return;
        }
        source = (stream / pixels) * pixels + pixel;
        ranked_modes[stream] = encoded_modes[source];
    }
    uint first = offsets[stream], end = offsets[stream + 1u];
    if (end < first || end - first != sizes[source] || end > payload_bytes
        || sizes[source] > scratch_stride) {
        atomic_store_explicit(failure, 2u, memory_order_relaxed);
        return;
    }
    for (uint byte = 0u; byte < sizes[source]; ++byte)
        payload[first + byte] = scratch[ulong(byte) * streams + source];
}

struct PRTBitReader {
    device const uchar *payload;
    uint first;
    uint remaining;
    bool valid;

    PRTBitReader(device const uchar *bytes, uint body_first, uint meaningful)
        : payload(bytes), first(body_first), remaining(meaningful), valid(true) {}

    uint pop(uint count) {
        if (count > remaining) {
            valid = false;
            return 0u;
        }
        remaining -= count;
        uint value = 0u;
        for (uint bit = 0u; bit < count; ++bit) {
            uint position = remaining + bit;
            uint next = (uint(payload[first + position / 8u]) >> (position & 7u)) & 1u;
            value |= next << bit;
        }
        return value;
    }
};

kernel void paired_runtime_tans_decode(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device ushort *output [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    constant uint *p [[buffer(6)]],
    uint stream [[thread_position_in_grid]]) {
    uint scans = p[0], pixels = p[1], streams = p[2], item_bytes = p[3];
    uint payload_bytes = p[4];
    if (stream >= streams) return;
    uint pixel = stream % pixels;
    uint first_scan = (stream / pixels) * PRT_INTERVAL;
    uint count = min(PRT_INTERVAL, scans - first_scan);
    uint first = prt_source_offset(offsets, stream, streams);
    uint end = prt_source_offset(offsets, stream + 1u, streams);
    if (count == 0u || end < first || end > payload_bytes
        || (item_bytes != 1u && item_bytes != 2u)) {
        atomic_store_explicit(failure, 3u, memory_order_relaxed);
        return;
    }
    uint mode = uint(modes[stream]);
    if (mode == 253u) {
        if (first != end) atomic_store_explicit(failure, 4u, memory_order_relaxed);
        for (uint scan = 0u; scan < count; ++scan)
            output[ulong(first_scan + scan) * pixels + pixel] = 0u;
        return;
    }
    if (mode == 255u) {
        if (end - first != 2u) {
            atomic_store_explicit(failure, 5u, memory_order_relaxed);
            return;
        }
        uint value = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
        if (item_bytes == 1u && value > 255u) {
            atomic_store_explicit(failure, 6u, memory_order_relaxed);
            return;
        }
        for (uint scan = 0u; scan < count; ++scan)
            output[ulong(first_scan + scan) * pixels + pixel] = ushort(value);
        return;
    }
    if (mode == 254u) {
        if (end - first != 2u * count) {
            atomic_store_explicit(failure, 7u, memory_order_relaxed);
            return;
        }
        for (uint scan = 0u; scan < count; ++scan) {
            uint value = uint(payload[first + 2u * scan])
                | (uint(payload[first + 2u * scan + 1u]) << 8u);
            if (item_bytes == 1u && value > 255u) {
                atomic_store_explicit(failure, 8u, memory_order_relaxed);
                return;
            }
            output[ulong(first_scan + scan) * pixels + pixel] = ushort(value);
        }
        return;
    }
    if (mode == 252u) {
        if (((end - first) & 1u) != 0u) {
            atomic_store_explicit(failure, 9u, memory_order_relaxed);
            return;
        }
        uint cursor = first, previous = 0u;
        bool has_previous = false;
        for (uint scan = 0u; scan < count; ++scan)
            output[ulong(first_scan + scan) * pixels + pixel] = 0u;
        while (cursor < end) {
            uint event = uint(payload[cursor]) | (uint(payload[cursor + 1u]) << 8u);
            cursor += 2u;
            uint scan = event >> 7u, value = (event & 127u) + 1u;
            if (scan >= count || (has_previous && scan <= previous)) {
                atomic_store_explicit(failure, 10u, memory_order_relaxed);
                return;
            }
            output[ulong(first_scan + scan) * pixels + pixel] = ushort(value);
            previous = scan;
            has_previous = true;
        }
        return;
    }
    bool interleaved = mode >= 96u && mode < 128u;
    if ((!prt_entropy_mode(mode) && !interleaved)
        || end - first < (interleaved ? 3u : 2u)) {
        atomic_store_explicit(failure, 11u, memory_order_relaxed);
        return;
    }
    uint tail, body_first;
    uint states[2];
    if (interleaved) {
        uint header = uint(payload[first])
            | (uint(payload[first + 1u]) << 8u)
            | (uint(payload[first + 2u]) << 16u);
        states[0] = header & 1023u;
        states[1] = (header >> 10u) & 1023u;
        tail = (header >> 20u) & 7u;
        body_first = first + 3u;
        if ((header >> 23u) != 0u) {
            atomic_store_explicit(failure, 12u, memory_order_relaxed);
            return;
        }
    } else {
        uint header = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
        states[0] = header >> 6u;
        states[1] = 0u;
        tail = header & 7u;
        body_first = first + 2u;
        if (((header >> 3u) & 7u) != 0u || states[0] >= PRT_STATES) {
            atomic_store_explicit(failure, 12u, memory_order_relaxed);
            return;
        }
    }
    uint body_bytes = end - body_first;
    if (tail != 0u && body_bytes == 0u) {
        atomic_store_explicit(failure, 12u, memory_order_relaxed);
        return;
    }
    uint padding = tail == 0u ? 0u : 8u - tail;
    uint meaningful = body_bytes * 8u - padding;
    if (tail != 0u && (uint(payload[end - 1u]) >> tail) != 0u) {
        atomic_store_explicit(failure, 13u, memory_order_relaxed);
        return;
    }
    PRTBitReader reader(payload, body_first, meaningful);
    uint model = interleaved ? mode - 96u : prt_entropy_model(mode);
    for (uint index = 0u; index < count; index += 2u) {
        uint state_index = interleaved ? ((index >> 1u) & 1u) : 0u;
        uint state = states[state_index];
        uint code = decoding[model * PRT_STATES + state];
        uint pair = code & 4095u, bits = (code >> 12u) & 15u;
        state = (code >> 16u) + reader.pop(bits);
        states[state_index] = state;
        uint a, b;
        if (pair == 4095u) {
            uint word = reader.pop(13u);
            if (word < 4096u) {
                a = word & 63u;
                b = word >> 6u;
            } else if (word == 4096u) {
                a = reader.pop(16u);
                b = reader.pop(16u);
            } else {
                reader.valid = false;
                a = b = 0u;
            }
        } else {
            a = pair & 63u;
            b = pair >> 6u;
        }
        if (!reader.valid || state >= PRT_STATES || (item_bytes == 1u && (a > 255u || b > 255u))) {
            atomic_store_explicit(failure, 14u, memory_order_relaxed);
            return;
        }
        output[ulong(first_scan + index) * pixels + pixel] = ushort(a);
        if (index + 1u < count)
            output[ulong(first_scan + index + 1u) * pixels + pixel] = ushort(b);
        else if (b != 0u) {
            atomic_store_explicit(failure, 15u, memory_order_relaxed);
            return;
        }
    }
    if (reader.remaining != 0u || states[0] != 0u || states[1] != 0u)
        atomic_store_explicit(failure, 16u, memory_order_relaxed);
}

// Reverse stack reader used by the interactive kernels. The encoder appends
// fields least-significant bit first while walking pairs backwards; decoding
// therefore consumes fields from the high end of the stored bit sequence.
// At most 23 bits are resident because every field is at most 16 bits and a
// refill happens only when fewer than that remain.
struct PRTReverseReader {
    device const uchar *payload;
    uint body_first, cursor, last, reservoir, available, remaining, last_bits;
    bool valid;
};

inline PRTReverseReader prt_reverse_reader(
    device const uchar *payload, uint body_first, uint end, uint meaningful,
    uint last_bits) {
    PRTReverseReader reader;
    reader.payload = payload;
    reader.body_first = body_first;
    reader.cursor = end;
    reader.last = end - 1u;
    reader.reservoir = 0u;
    reader.available = 0u;
    reader.remaining = meaningful;
    reader.last_bits = last_bits;
    reader.valid = true;
    return reader;
}

inline uint prt_reverse_pop(thread PRTReverseReader &reader, uint count) {
    if (!reader.valid || count > reader.remaining) {
        reader.valid = false;
        return 0u;
    }
    while (reader.available < count) {
        if (reader.cursor <= reader.body_first) {
            reader.valid = false;
            return 0u;
        }
        --reader.cursor;
        uint bits = reader.cursor == reader.last ? reader.last_bits : 8u;
        uint value = uint(reader.payload[reader.cursor]) & ((1u << bits) - 1u);
        reader.reservoir = value | (reader.reservoir << bits);
        reader.available += bits;
    }
    reader.remaining -= count;
    reader.available -= count;
    uint value = count == 0u
        ? 0u : (reader.reservoir >> reader.available) & ((1u << count) - 1u);
    reader.reservoir &= reader.available == 0u
        ? 0u : (1u << reader.available) - 1u;
    return value;
}

inline bool prt_entropy_begin(
    device const uchar *payload, uint first, uint end, uint payload_bytes,
    thread uint &state, thread PRTReverseReader &reader) {
    if (end < first || end > payload_bytes || end - first < 2u) return false;
    uint header = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
    uint tail = header & 7u;
    state = header >> 6u;
    uint body_first = first + 2u;
    uint body_bytes = end - body_first;
    if (((header >> 3u) & 7u) != 0u || state >= PRT_STATES
        || (tail != 0u && body_bytes == 0u)) return false;
    uint last_bits = tail == 0u ? 8u : tail;
    if (body_bytes == 0u || (tail != 0u && (uint(payload[end - 1u]) >> tail) != 0u))
        return false;
    uint meaningful = (body_bytes - 1u) * 8u + last_bits;
    reader = prt_reverse_reader(payload, body_first, end, meaningful, last_bits);
    return true;
}

inline bool prt_entropy_pair(
    device const uint *table, thread uint &state,
    thread PRTReverseReader &reader, thread uint &a, thread uint &b) {
    uint code = table[state];
    uint pair = code & 4095u;
    uint bits = (code >> 12u) & 15u;
    state = (code >> 16u) + prt_reverse_pop(reader, bits);
    if (pair == 4095u) {
        uint word = prt_reverse_pop(reader, 13u);
        if (word < 4096u) {
            a = word & 63u;
            b = word >> 6u;
        } else if (word == 4096u) {
            a = prt_reverse_pop(reader, 16u);
            b = prt_reverse_pop(reader, 16u);
        } else {
            reader.valid = false;
            a = b = 0u;
        }
    } else {
        a = pair & 63u;
        b = pair >> 6u;
    }
    return reader.valid && state < PRT_STATES;
}

constant bool prt_macro_requested [[function_constant(2)]];
constant bool prt_macro_enabled = is_function_constant_defined(
    prt_macro_requested) ? prt_macro_requested : false;
constant uint prt_macro_lookahead_bits_requested [[function_constant(22)]];
constant uint prt_macro_lookahead_bits = is_function_constant_defined(
    prt_macro_lookahead_bits_requested) ? prt_macro_lookahead_bits_requested : 4u;
// Experimental scan512 topology. Striped independent accumulators preserve
// exact modulo-UInt32 arithmetic while shortening each dependency chain.
constant uint prt_polar_scan512_stripes_requested [[function_constant(23)]];
constant uint prt_polar_scan512_stripes = is_function_constant_defined(
    prt_polar_scan512_stripes_requested) ? prt_polar_scan512_stripes_requested : 1u;
constant bool prt_polar_scan512_contiguous_quad_requested [[function_constant(26)]];
constant bool prt_polar_scan512_contiguous_quad = is_function_constant_defined(
    prt_polar_scan512_contiguous_quad_requested)
    ? prt_polar_scan512_contiguous_quad_requested : false;
constant bool prt_reader32_requested [[function_constant(3)]];
constant bool prt_reader32_enabled = is_function_constant_defined(
    prt_reader32_requested) ? prt_reader32_requested : false;
constant uint prt_partial_leaf_width_requested [[function_constant(5)]];
constant uint prt_partial_leaf_width = is_function_constant_defined(
    prt_partial_leaf_width_requested) ? prt_partial_leaf_width_requested : 64u;
constant uint prt_partial_output_fields_requested [[function_constant(6)]];
constant uint prt_partial_output_fields = is_function_constant_defined(
    prt_partial_output_fields_requested) ? prt_partial_output_fields_requested : 0u;
// Direct partial stores are safe because each partials dispatch group owns a
// disjoint (group, packet) field.  The specialization still initializes every
// field and fences before sparse atomic additions, so mixed dense/sparse masks
// retain the exact modulo-UInt32 reduction contract.
constant bool prt_partial_stores_requested [[function_constant(8)]];
constant bool prt_partial_stores = is_function_constant_defined(
    prt_partial_stores_requested) ? prt_partial_stores_requested : false;
// Optional packet splitting reuses the existing four-SIMD-group owner.  Each
// split gets a disjoint selected-stream subset and the same 2 KiB packet-local
// partial; only the final publication changes to a global atomic merge.
constant uint prt_packet_split_count_requested [[function_constant(9)]];
constant uint prt_packet_split_count = is_function_constant_defined(
    prt_packet_split_count_requested) ? prt_packet_split_count_requested : 1u;
// Each nonterminal reverse-reader refill advances by four bytes.  The next
// eight-byte prime therefore overlaps the previous prime by one word; the
// specialization reuses that word and loads only the newly exposed word.
// Keep the full prime for the final 0...7-byte boundary, where clamping can
// make the old and new aligned windows coincide.
constant bool prt_single_word_refill_requested [[function_constant(10)]];
constant bool prt_single_word_refill = is_function_constant_defined(
    prt_single_word_refill_requested) ? prt_single_word_refill_requested : false;
// Dense packet contributions can be accumulated by the unique lane that owns
// each output scan. Sparse events still use the packet-local atomic partials.
constant bool prt_register_reduction_requested [[function_constant(11)]];
constant bool prt_register_reduction = is_function_constant_defined(
    prt_register_reduction_requested) ? prt_register_reduction_requested : false;
// Plain dense updates are safe in packet_owner2 when the SIMD group fences
// before and after its sparse atomic phase. This keeps the existing 8 KiB
// packet partial and avoids the register footprint of lane-owned sums.
constant bool prt_plain_packet_owner2_requested [[function_constant(12)]];
constant bool prt_plain_packet_owner2 = is_function_constant_defined(
    prt_plain_packet_owner2_requested) ? prt_plain_packet_owner2_requested : false;
// The deterministic internal table factory proves base + (1 << bits) <=
// PRT_STATES for every ordinary and macro transition. This opt-in removes
// only the per-pair state-range compare; payload validity remains checked.
constant bool prt_trusted_decode_table_requested [[function_constant(13)]];
constant bool prt_trusted_decode_table = is_function_constant_defined(
    prt_trusted_decode_table_requested) ? prt_trusted_decode_table_requested : false;
// Diagnostic-only decode coverage mode. It preserves every decoded scalar in
// a lane-local checksum and publishes one reduced checksum per packet instead
// of producing an image or performing per-pair output atomics.
constant bool prt_decode_checksum_requested [[function_constant(14)]];
constant bool prt_decode_checksum = is_function_constant_defined(
    prt_decode_checksum_requested) ? prt_decode_checksum_requested : false;
// FC15 defers reverse-reader refills until the exact state-table bit count is
// known. The eager refill remains the default for all existing pipelines.
constant bool prt_lazy_refill_requested [[function_constant(15)]];
constant bool prt_lazy_refill = is_function_constant_defined(
    prt_lazy_refill_requested) ? prt_lazy_refill_requested : false;
constant bool prt_branchless_pop_requested [[function_constant(16)]];
constant bool prt_branchless_pop = is_function_constant_defined(
    prt_branchless_pop_requested) ? prt_branchless_pop_requested : false;
constant uint prt_refill_threshold_requested [[function_constant(17)]];
constant uint prt_refill_threshold = is_function_constant_defined(
    prt_refill_threshold_requested) ? prt_refill_threshold_requested : 32u;
// Load independent ordinary transition entries before advancing any reader.
// Macro readers retain their queued-pair/lookahead path without preloading.
constant bool prt_phased_table_loads_requested [[function_constant(18)]];
constant bool prt_phased_table_loads = is_function_constant_defined(
    prt_phased_table_loads_requested) ? prt_phased_table_loads_requested : false;
// Supported specializations are 1, 2, 4 and 8, all divisors of 256 pairs.
// Each unrolled iteration retains the ordinary per-pair reduction and stores.
constant uint prt_pair_unroll_requested [[function_constant(19)]];
constant uint prt_pair_unroll = is_function_constant_defined(
    prt_pair_unroll_requested) ? prt_pair_unroll_requested : 1u;
// Experimental only: specialize a 32-lane stream group when every lane has a
// successfully initialized entropy reader. The default packet-owner2 pipeline
// does not define this constant and keeps the general per-stream decoder.
constant bool prt_simd_entropy_fast_path_requested [[function_constant(21)]];
constant bool prt_simd_entropy_fast_path = is_function_constant_defined(
    prt_simd_entropy_fast_path_requested) ? prt_simd_entropy_fast_path_requested : false;

// Four-byte reverse reader. It may prefetch a few bytes below one stream's
// beginning, but never consumes them; the terminal check proves that only the
// declared stream bits affected decoded counts.
struct PRTFastReader {
    device const uchar *payload;
    device const uint *table;
    uint begin, cursor, state, available;
    uint pending_low, pending_high, pending_shift;
    ulong reservoir;
    uint reservoir_low, reservoir_high;
    ulong queued_pairs;
    uint queued_count, remaining_pairs;
    bool valid;
};

inline void prt_fast_prime(thread PRTFastReader &reader) {
    uint low = reader.cursor >= 4u ? reader.cursor - 4u : 0u;
    uint aligned = low & ~3u;
    device const uint *words = reinterpret_cast<device const uint *>(
        reader.payload + aligned);
    reader.pending_low = words[0];
    reader.pending_high = words[1];
    reader.pending_shift = (low & 3u) * 8u;
}

inline void prt_fast_refill(thread PRTFastReader &reader) {
    uint word;
    if (prt_reader32_enabled) {
        uint shift = reader.pending_shift;
        word = shift == 0u
            ? reader.pending_low
            : (reader.pending_low >> shift) | (reader.pending_high << (32u - shift));
    } else {
        ulong joined = ulong(reader.pending_low) | (ulong(reader.pending_high) << 32u);
        word = uint(joined >> reader.pending_shift);
    }
    if (reader.cursor < 4u) {
        uint count = reader.cursor;
        uint mask = count == 0u ? 0u : (1u << (count * 8u)) - 1u;
        word &= mask;
        uint pushed = count * 8u;
        if (prt_reader32_enabled) {
            if (pushed != 0u) {
                reader.reservoir_high = (reader.reservoir_high << pushed)
                    | (reader.reservoir_low >> (32u - pushed));
                reader.reservoir_low = (reader.reservoir_low << pushed) | word;
            }
        } else {
            reader.reservoir = (reader.reservoir << pushed) | ulong(word);
        }
        reader.available += count * 8u;
        reader.cursor = 0u;
    } else {
        if (prt_reader32_enabled) {
            reader.reservoir_high = reader.reservoir_low;
            reader.reservoir_low = word;
        } else {
            reader.reservoir = (reader.reservoir << 32u) | ulong(word);
        }
        reader.available += 32u;
        reader.cursor -= 4u;
    }
    if (prt_single_word_refill && reader.cursor >= 4u) {
        uint low = reader.cursor - 4u;
        uint aligned = low & ~3u;
        device const uint *words = reinterpret_cast<device const uint *>(
            reader.payload + aligned);
        reader.pending_high = reader.pending_low;
        reader.pending_low = words[0];
        reader.pending_shift = (low & 3u) * 8u;
    } else {
        prt_fast_prime(reader);
    }
}

inline uint prt_fast_pop(thread PRTFastReader &reader, uint count) {
    if (!reader.valid || count > reader.available) {
        reader.valid = false;
        return 0u;
    }
    reader.available -= count;
    // The 64-bit reservoir stays below 64 available bits with eager refill.
    // A zero-bit mask already produces zero without a dependent branch.
    if ((!prt_branchless_pop || prt_reader32_enabled) && count == 0u) return 0u;
    uint mask = (1u << count) - 1u;
    if (prt_reader32_enabled) {
        uint shift = reader.available;
        uint value;
        if (shift == 0u) value = reader.reservoir_low;
        else if (shift < 32u) {
            value = (reader.reservoir_low >> shift)
                | (reader.reservoir_high << (32u - shift));
        } else value = reader.reservoir_high >> (shift - 32u);
        return value & mask;
    }
    return uint(reader.reservoir >> reader.available) & mask;
}

inline bool prt_fast_begin(
    device const uchar *payload, uint first, uint end, uint payload_bytes,
    device const uint *table, thread PRTFastReader &reader) {
    reader.payload = payload;
    reader.table = table;
    reader.begin = first;
    reader.cursor = end;
    reader.state = 0u;
    reader.available = 0u;
    reader.pending_low = reader.pending_high = reader.pending_shift = 0u;
    reader.reservoir = 0u;
    if (prt_reader32_enabled) {
        reader.reservoir_low = 0u;
        reader.reservoir_high = 0u;
    }
    if (prt_macro_enabled) {
        reader.queued_pairs = 0u;
        reader.queued_count = 0u;
        reader.remaining_pairs = PRT_INTERVAL / 2u;
    }
    reader.valid = end >= first && end <= payload_bytes
        && end - first >= 2u;
    if (!reader.valid) return false;
    uint header0 = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
    uint tail = header0 & 7u;
    reader.state = header0 >> 6u;
    reader.begin = first + 2u;
    reader.valid = ((header0 >> 3u) & 7u) == 0u
        && reader.state < PRT_STATES
        && (tail == 0u || end > reader.begin);
    if (!reader.valid) return false;
    uint bits = (end - reader.begin) * 8u - ((8u - tail) & 7u);
    if ((bits & 7u) != 0u) {
        reader.available = bits & 7u;
        reader.reservoir = uint(payload[--reader.cursor]);
        if (prt_reader32_enabled) reader.reservoir_low = uint(reader.reservoir);
        reader.valid = uint(reader.reservoir) < (1u << reader.available);
    }
    prt_fast_prime(reader);
    return reader.valid;
}

// Experimental exact midpoint checkpoint for the 512-value detector record.
// The tANS state needs 10 bits. The encoder admits entropy records only when
// their meaningful bit count is below 2^14, so the unread reverse-bit position
// fits 14 more bits. The original payload is immutable: reconstructing the
// reservoir from that bit position avoids storing a 64-bit copy per stream.
constant uint PRT_MIDPOINT_PAIRS = PRT_INTERVAL / 4u;
constant uint PRT_MIDPOINT_CHECKPOINT_BYTES = 3u;

inline int prt_midpoint_unread_bits(thread PRTFastReader &reader) {
    return int(reader.available)
        + 8 * (int(reader.cursor) - int(reader.begin));
}

inline bool prt_midpoint_checkpoint_store(
    thread PRTFastReader &reader, device uchar *checkpoints, uint record) {
    int unread = prt_midpoint_unread_bits(reader);
    if (!reader.valid || reader.state >= PRT_STATES
        || unread < 0 || unread >= 16384) return false;
    uint packed = reader.state | (uint(unread) << 10u);
    device uchar *destination = checkpoints + record * PRT_MIDPOINT_CHECKPOINT_BYTES;
    destination[0] = uchar(packed);
    destination[1] = uchar(packed >> 8u);
    destination[2] = uchar(packed >> 16u);
    return true;
}

inline bool prt_midpoint_checkpoint_matches(
    thread PRTFastReader &reader, device const uchar *checkpoints, uint record) {
    int unread = prt_midpoint_unread_bits(reader);
    if (!reader.valid || reader.state >= PRT_STATES
        || unread < 0 || unread >= 16384) return false;
    uint expected = record * PRT_MIDPOINT_CHECKPOINT_BYTES;
    uint packed = reader.state | (uint(unread) << 10u);
    return checkpoints[expected] == uchar(packed)
        && checkpoints[expected + 1u] == uchar(packed >> 8u)
        && checkpoints[expected + 2u] == uchar(packed >> 16u);
}

inline bool prt_fast_restore_midpoint(
    device const uchar *payload, uint first, uint end, uint payload_bytes,
    device const uint *table, device const uchar *checkpoints, uint record,
    thread PRTFastReader &reader) {
    if (end < first || end > payload_bytes || end - first < 3u) return false;
    uint header = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
    uint tail = header & 7u;
    if (((header >> 3u) & 7u) != 0u || (tail != 0u && end <= first + 2u))
        return false;
    uint body_first = first + 2u;
    uint padding = (8u - tail) & 7u;
    uint total_bits = (end - body_first) * 8u - padding;
    uint packed_record = record * PRT_MIDPOINT_CHECKPOINT_BYTES;
    uint packed = uint(checkpoints[packed_record])
        | (uint(checkpoints[packed_record + 1u]) << 8u)
        | (uint(checkpoints[packed_record + 2u]) << 16u);
    uint state = packed & 1023u;
    uint unread = packed >> 10u;
    if (state >= PRT_STATES || unread > total_bits) return false;

    reader.payload = payload;
    reader.table = table;
    reader.begin = body_first;
    reader.state = state;
    reader.available = unread & 7u;
    reader.cursor = body_first + (unread >> 3u);
    reader.pending_low = reader.pending_high = reader.pending_shift = 0u;
    reader.reservoir = 0ul;
    if (prt_reader32_enabled) {
        reader.reservoir_low = 0u;
        reader.reservoir_high = 0u;
    }
    if (reader.available != 0u) {
        if (reader.cursor >= end) return false;
        uint partial = uint(payload[reader.cursor])
            & ((1u << reader.available) - 1u);
        reader.reservoir = ulong(partial);
        if (prt_reader32_enabled) reader.reservoir_low = partial;
    }
    reader.queued_pairs = 0ul;
    reader.queued_count = 0u;
    reader.remaining_pairs = PRT_MIDPOINT_PAIRS;
    reader.valid = true;
    prt_fast_prime(reader);
    return true;
}

inline bool prt_fast_pair_code(
    thread PRTFastReader &reader, uint code, thread uint &a, thread uint &b) {
    uint bits = (code >> 12u) & 15u;
    if (prt_lazy_refill) {
        if (reader.available < bits) prt_fast_refill(reader);
    } else if (reader.available < prt_refill_threshold) {
        prt_fast_refill(reader);
    }
    reader.state = (code >> 16u) + prt_fast_pop(reader, bits);
    uint pair = code & 4095u;
    if (pair == 4095u) {
        if (reader.available < 13u) prt_fast_refill(reader);
        uint word = prt_fast_pop(reader, 13u);
        if (word < 4096u) {
            a = word & 63u;
            b = word >> 6u;
        } else if (word == 4096u) {
            if (reader.available < 16u) prt_fast_refill(reader);
            a = prt_fast_pop(reader, 16u);
            if (reader.available < 16u) prt_fast_refill(reader);
            b = prt_fast_pop(reader, 16u);
        } else {
            reader.valid = false;
            a = b = 0u;
        }
    } else {
        a = pair & 63u;
        b = pair >> 6u;
    }
    if (prt_macro_enabled) reader.remaining_pairs -= 1u;
    return reader.valid && (prt_trusted_decode_table || reader.state < PRT_STATES);
}

inline bool prt_fast_pair(
    thread PRTFastReader &reader, thread uint &a, thread uint &b) {
    if (prt_macro_enabled && reader.queued_count != 0u) {
        uint pair = uint(reader.queued_pairs) & 4095u;
        reader.queued_pairs >>= 12u;
        reader.queued_count -= 1u;
        reader.remaining_pairs -= 1u;
        a = pair & 63u;
        b = pair >> 6u;
        return reader.valid && (prt_trusted_decode_table || reader.state < PRT_STATES);
    }
    if (prt_macro_enabled && reader.remaining_pairs >= 3u) {
        int actual_remaining = int(reader.available)
            + 8 * (int(reader.cursor) - int(reader.begin));
        uint lookahead_bits = prt_macro_lookahead_bits;
        uint lookahead_count = 1u << lookahead_bits;
        if ((lookahead_bits == 2u || lookahead_bits == 4u)
            && actual_remaining >= int(lookahead_bits)) {
            if (reader.available < lookahead_bits) prt_fast_refill(reader);
            uint lookahead = uint(reader.reservoir >> (reader.available - lookahead_bits))
                & (lookahead_count - 1u);
            uint macro_index = PRT_STATES
                + 2u * (reader.state * lookahead_count + lookahead);
            ulong entry = ulong(reader.table[macro_index])
                | (ulong(reader.table[macro_index + 1u]) << 32u);
            uint consumed = uint(entry >> 46u) & 7u;
            uint count = uint(entry >> 49u) & 3u;
            if (count != 0u && consumed <= uint(actual_remaining)) {
                (void)prt_fast_pop(reader, consumed);
                reader.state = uint(entry >> 36u) & 1023u;
                reader.queued_pairs = entry & ((1ul << 36u) - 1ul);
                reader.queued_count = count;
                uint pair = uint(reader.queued_pairs) & 4095u;
                reader.queued_pairs >>= 12u;
                reader.queued_count -= 1u;
                reader.remaining_pairs -= 1u;
                a = pair & 63u;
                b = pair >> 6u;
                return reader.valid && (prt_trusted_decode_table || reader.state < PRT_STATES);
            }
        }
    }
    return prt_fast_pair_code(reader, reader.table[reader.state], a, b);
}

inline bool prt_fast_finished(thread PRTFastReader &reader) {
    return reader.valid && reader.cursor <= reader.begin
        && reader.available == 8u * (reader.begin - reader.cursor)
        && reader.state == 0u
        && (!prt_macro_enabled
            || (reader.remaining_pairs == 0u && reader.queued_count == 0u));
}

// Concatenate bounded record offsets without exposing private metadata to the
// host. Adjacent record boundaries intentionally write the same exact value.
kernel void paired_runtime_tans_rebase_offsets(
    device const uint *source [[buffer(0)]],
    device uint *destination [[buffer(1)]],
    constant uint *p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    uint source_streams = p[0], destination_first = p[1], payload_first = p[2];
    if (index <= source_streams)
        destination[destination_first + index] = source[index] + payload_first;
}

// Rebase one record directly into the packed acquisition-wide block-32
// directory. Every stream start is a uint16 delta from its group's uint32
// payload base; a raw fallback is at most 1024 bytes, so a group spans at most
// 32768 bytes. The final record sentinel is checked against the producer's
// host-visible receipt before it is published.
kernel void paired_runtime_tans_rebase_block32_offsets(
    device const uint *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    device atomic_uint *failure [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint source_streams = p[0], destination_first = p[1];
    uint payload_first = p[2], destination_streams = p[3];
    uint expected_record_payload_bytes = p[4];
    if (index > source_streams) return;
    if (source_streams == 0u || (source_streams & 31u) != 0u
        || (destination_first & 31u) != 0u
        || (destination_streams & 31u) != 0u
        || destination_first > destination_streams
        || source_streams > destination_streams - destination_first) {
        atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        return;
    }
    if (source[0] != 0u || source[source_streams] != expected_record_payload_bytes) {
        atomic_fetch_or_explicit(failure, 2u, memory_order_relaxed);
        return;
    }
    uint group_first = index & ~31u;
    uint value = source[index], base = source[group_first];
    bool invalid_span = value < base || value - base > 32768u;
    if (index > 0u) {
        uint previous = source[index - 1u];
        invalid_span = invalid_span || value < previous || value - previous > 1024u;
    }
    if (index > 0u && (index & 31u) == 0u) {
        uint previous_group_base = source[index - 32u];
        invalid_span = invalid_span || value < previous_group_base
            || value - previous_group_base > 32768u;
    }
    if (invalid_span || value > 0xffffffffu - payload_first) {
        atomic_fetch_or_explicit(failure, 4u, memory_order_relaxed);
        return;
    }
    uint destination_stream = destination_first + index;
    uint base_count = (destination_streams >> 5u) + 1u;
    device uint *bases = reinterpret_cast<device uint *>(destination);
    device ushort *starts = reinterpret_cast<device ushort *>(
        destination + ulong(base_count) * sizeof(uint));
    if ((index & 31u) == 0u)
        bases[destination_stream >> 5u] = payload_first + value;
    starts[destination_stream] = ushort(value - base);
}

// Exact diffraction pattern at one scan of one flat 16K record. The kernel
// decodes only the dependency prefix ending at the selected scan's pair and
// widens the native count to uint32. It has no archive or HDF5 dependency.
kernel void paired_runtime_tans_selected_dp(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device uint *output [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    constant uint *p [[buffer(6)]],
    uint pixel [[thread_position_in_grid]]) {
    uint pixels = p[0], packets = p[1], scan = p[2], payload_bytes = p[3];
    if (pixel >= pixels) return;
    if (scan >= packets * PRT_INTERVAL) {
        atomic_store_explicit(failure, 20u, memory_order_relaxed);
        return;
    }
    uint packet = scan / PRT_INTERVAL;
    uint local = scan % PRT_INTERVAL;
    uint stream = packet * pixels + pixel;
    uint total_streams = pixels * packets;
    uint first = prt_source_offset(offsets, stream, total_streams);
    uint end = prt_source_offset(offsets, stream + 1u, total_streams);
    if (end < first || end > payload_bytes) {
        atomic_store_explicit(failure, 21u, memory_order_relaxed);
        return;
    }
    uint mode = uint(modes[stream]);
    if (mode == 253u) {
        if (first != end) atomic_store_explicit(failure, 22u, memory_order_relaxed);
        output[pixel] = 0u;
        return;
    }
    if (mode == 255u) {
        if (end - first != 2u) {
            atomic_store_explicit(failure, 23u, memory_order_relaxed);
            return;
        }
        output[pixel] = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
        return;
    }
    if (mode == 254u) {
        if (end - first != 2u * PRT_INTERVAL) {
            atomic_store_explicit(failure, 24u, memory_order_relaxed);
            return;
        }
        uint at = first + 2u * local;
        output[pixel] = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
        return;
    }
    if (mode == 252u) {
        if (((end - first) & 1u) != 0u) {
            atomic_store_explicit(failure, 25u, memory_order_relaxed);
            return;
        }
        uint value = 0u, previous = 0u;
        bool has_previous = false;
        for (uint cursor = first; cursor < end; cursor += 2u) {
            uint event = uint(payload[cursor]) | (uint(payload[cursor + 1u]) << 8u);
            uint position = event >> 7u;
            if (position >= PRT_INTERVAL || (has_previous && position <= previous)) {
                atomic_store_explicit(failure, 26u, memory_order_relaxed);
                return;
            }
            if (position == local) value = (event & 127u) + 1u;
            previous = position;
            has_previous = true;
        }
        output[pixel] = value;
        return;
    }
    if (mode == 251u) {
        PRTCompactEvents events = {first, end, 0u, true};
        uint position = 0u, value = 0u, result = 0u;
        while (prt_compact_next(payload, events, position, value)) {
            if (position == local) result = value;
            if (position >= local) break;
        }
        if (!events.valid) {
            atomic_store_explicit(failure, 30u, memory_order_relaxed);
            return;
        }
        output[pixel] = result;
        return;
    }
    if (!prt_entropy_mode(mode)) {
        atomic_store_explicit(failure, 27u, memory_order_relaxed);
        return;
    }
    uint state = 0u;
    PRTFastReader reader;
    device const uint *table = decoding + prt_entropy_model(mode) * PRT_STATES;
    if (!prt_fast_begin(payload, first, end, payload_bytes, table, reader)) {
        atomic_store_explicit(failure, 28u, memory_order_relaxed);
        return;
    }
    uint target_pair = local >> 1u;
    uint a = 0u, b = 0u;
    for (uint pair = 0u; pair <= target_pair; ++pair) {
        if (!prt_fast_pair(reader, a, b)) {
            atomic_store_explicit(failure, 29u, memory_order_relaxed);
            return;
        }
    }
    output[pixel] = (local & 1u) == 0u ? a : b;
}

// Bounded direct transcoder. Counts stay in registers; only compact headers
// and per-detector summaries are visible to the allocation planner.
struct PRTPackedReader {
    uint mode, first, end, scan, cursor, next_position, next_value, pair_b;
    bool valid, has_event;
    PRTCompactEvents events;
    PRTFastReader entropy;
};

inline bool prt_packed_begin(device const uchar *payload, device const uint *table,
    uint first, uint end, uint bytes, uint mode, thread PRTPackedReader &r) {
    r.mode = mode; r.first = first; r.end = end; r.scan = 0u;
    r.cursor = first; r.valid = end >= first && end <= bytes; r.has_event = false;
    if (!r.valid) return false;
    if (mode == 253u) return r.valid = first == end;
    if (mode == 255u) return r.valid = end - first == 2u;
    if (mode == 254u) return r.valid = end - first == 2u * PRT_INTERVAL;
    if (mode == 252u || mode == 251u) {
        r.events = {first, end, 0u, true};
        r.next_position = 0u; r.next_value = 0u;
        if (mode == 252u) return r.valid = ((end - first) & 1u) == 0u;
        return true;
    }
    if (!prt_entropy_mode(mode)) return r.valid = false;
    return r.valid = prt_fast_begin(payload, first, end, bytes,
        table + prt_entropy_model(mode) * PRT_STATES, r.entropy);
}

inline uint prt_packed_next(device const uchar *payload, thread PRTPackedReader &r) {
    uint scan = r.scan++;
    if (r.mode == 253u) return 0u;
    if (r.mode == 255u) return uint(payload[r.first]) | (uint(payload[r.first + 1u]) << 8u);
    if (r.mode == 254u) {
        uint at = r.first + 2u * scan;
        return uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
    }
    if (r.mode == 252u || r.mode == 251u) {
        if (!r.has_event) {
            if (r.mode == 252u && r.cursor < r.end) {
                uint code = uint(payload[r.cursor]) | (uint(payload[r.cursor + 1u]) << 8u);
                r.cursor += 2u; r.next_position = code >> 7u;
                r.next_value = (code & 127u) + 1u; r.has_event = true;
            } else if (r.mode == 251u) {
                r.has_event = prt_compact_next(payload, r.events, r.next_position, r.next_value);
                r.valid = r.valid && r.events.valid;
            }
        }
        if (r.has_event && (r.next_position < scan || r.next_position >= PRT_INTERVAL)) r.valid = false;
        if (r.has_event && r.next_position == scan) {
            r.has_event = false; return r.next_value;
        }
        return 0u;
    }
    if ((scan & 1u) != 0u) return r.pair_b;
    uint a = 0u;
    r.valid = prt_fast_pair(r.entropy, a, r.pair_b) && r.valid;
    return a;
}

inline bool prt_packed_finished(thread PRTPackedReader &r) {
    if (!r.valid || r.scan != PRT_INTERVAL) return false;
    if (r.mode == 252u) return !r.has_event && r.cursor == r.end;
    if (r.mode == 251u) return !r.has_event && r.events.cursor == r.end && r.events.valid;
    return !prt_entropy_mode(r.mode) || prt_fast_finished(r.entropy);
}

// One exact partial per detector stream and selected scan packet. Counts never
// leave registers; a packet sum is at most 512 * 65535 and fits in uint32.
kernel void paired_runtime_tans_region_partials(
    device const uchar *payload [[buffer(0)]], device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]], device const uint *table [[buffer(3)]],
    device uint *partials [[buffer(4)]], device atomic_uint *failure [[buffer(5)]],
    constant uint *p [[buffer(6)]], device const uchar *membership [[buffer(7)]],
    device const uint *selected_counts [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    uint pixels = p[0], packets = p[1], packet_count = p[4];
    if (index >= pixels * packet_count) return;
    uint packet = p[3] + index / pixels;
    uint stream = packet * pixels + index % pixels;
    uint first = prt_source_offset(offsets, stream, pixels * packets);
    uint end = prt_source_offset(offsets, stream + 1u, pixels * packets);
    PRTPackedReader reader;
    if (!prt_packed_begin(payload, table, first, end, p[2], uint(modes[stream]), reader)) {
        atomic_store_explicit(failure, 110u, memory_order_relaxed); return;
    }
    if (reader.mode == 253u || selected_counts[packet] == 0u) {
        partials[index] = 0u; return;
    }
    if (reader.mode == 255u) {
        partials[index] = (uint(payload[first]) | (uint(payload[first + 1u]) << 8u))
            * selected_counts[packet]; return;
    }
    if (reader.mode == 252u || reader.mode == 251u) {
        uint sum = 0u, previous = 0u;
        bool has_previous = false;
        PRTCompactEvents events = {first, end, 0u, true};
        while (events.cursor < end) {
            uint position = 0u, value = 0u;
            if (reader.mode == 252u) {
                uint code = uint(payload[events.cursor]) | (uint(payload[events.cursor + 1u]) << 8u);
                events.cursor += 2u; position = code >> 7u; value = (code & 127u) + 1u;
            } else if (!prt_compact_next(payload, events, position, value)) {
                atomic_store_explicit(failure, 111u, memory_order_relaxed); return;
            }
            if (position >= PRT_INTERVAL || (has_previous && position <= previous)) {
                atomic_store_explicit(failure, 111u, memory_order_relaxed); return;
            }
            if (membership[packet * PRT_INTERVAL + position] != 0u) sum += value;
            previous = position; has_previous = true;
        }
        partials[index] = sum; return;
    }
    uint sum = 0u;
    for (uint scan = 0u; scan < PRT_INTERVAL; ++scan) {
        uint value = prt_packed_next(payload, reader);
        if (membership[packet * PRT_INTERVAL + scan] != 0u) sum += value;
    }
    partials[index] = sum;
    if (!prt_packed_finished(reader))
        atomic_store_explicit(failure, 111u, memory_order_relaxed);
}

// Ordered encoders combine bounded batches into an exact 64-bit detector sum.
kernel void paired_runtime_tans_region_combine(
    device const uint *partials [[buffer(0)]], device ulong *sums [[buffer(1)]],
    constant uint *p [[buffer(2)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= p[0]) return;
    ulong sum = sums[pixel];
    for (uint packet = 0u; packet < p[1]; ++packet)
        sum += ulong(partials[packet * p[0] + pixel]);
    sums[pixel] = sum;
}

constant bool prt_packed_staging_requested [[function_constant(80)]];
constant bool prt_packed_staging = is_function_constant_defined(prt_packed_staging_requested)
    ? prt_packed_staging_requested : false;

kernel void paired_runtime_packed_measure(
    device const uchar *payload [[buffer(0)]], device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]], device const uint *table [[buffer(3)]],
    device uint *headers [[buffer(4)]], device atomic_uint *sums [[buffer(5)]],
    device const uint *pixel_of_rank [[buffer(6)]], device atomic_uint *failure [[buffer(7)]],
    constant uint *p [[buffer(8)]], device ushort *staging [[buffer(9)]],
    uint local_stream [[thread_position_in_grid]]) {
    uint pixels=p[0], packets=p[1], packet_count=p[3], stride=p[4], first_packet=p[5];
    if (local_stream >= pixels * packet_count) return;
    uint packet=local_stream/pixels, pixel=pixel_of_rank[local_stream%pixels];
    uint stream=(first_packet+packet)*pixels+local_stream%pixels;
    uint first=prt_source_offset(offsets,stream,pixels*packets);
    uint end=prt_source_offset(offsets,stream+1u,pixels*packets);
    PRTPackedReader r;
    if (!prt_packed_begin(payload,table,first,end,p[2],uint(modes[stream]),r)) {
        atomic_store_explicit(failure,101u,memory_order_relaxed); return;
    }
    uint checkpoints=(packet_count*16u+31u)/32u, word=0u, sum=0u;
    for (uint tile=0u; tile<16u; ++tile) {
        uint combined=0u;
        for (uint scan=0u; scan<32u; ++scan) {
            uint value=prt_packed_next(payload,r); combined|=value; sum+=value;
            if (prt_packed_staging) staging[(packet*512u+tile*32u+scan)*pixels+local_stream%pixels]=ushort(value);
        }
        uint width=combined==0u ? 0u : 32u-clz(combined);
        // The existing uint16 header reserves nibble 15 for width 16.
        if (width==15u) width=16u;
        word |= min(width,15u) << ((tile&7u)*4u);
        if ((tile&7u)==7u) {
            headers[pixel*stride+checkpoints+packet*2u+tile/8u]=word; word=0u;
        }
    }
    atomic_fetch_add_explicit(sums+pixel,sum,memory_order_relaxed);
    if (!prt_packed_finished(r)) atomic_store_explicit(failure,102u,memory_order_relaxed);
}

kernel void paired_runtime_packed_headers(
    device uint *headers [[buffer(0)]], device uint *lengths [[buffer(1)]],
    device uint *maximum_widths [[buffer(2)]], constant uint *p [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel>=p[0]) return;
    uint tiles=p[3]*16u, checkpoints=(tiles+31u)/32u, stride=p[4], sum=0u, maximum=0u;
    for (uint tile=0u; tile<tiles; ++tile) {
        if ((tile&31u)==0u) headers[pixel*stride+tile/32u]=sum;
        uint width=(headers[pixel*stride+checkpoints+tile/8u]>>((tile&7u)*4u))&15u;
        if (width==15u) width=16u;
        sum+=width; maximum=max(maximum,width);
    }
    lengths[pixel]=sum; maximum_widths[pixel]=maximum;
}

kernel void paired_runtime_packed_write(
    device const uchar *payload [[buffer(0)]], device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]], device const uint *table [[buffer(3)]],
    device const uint *headers [[buffer(4)]], device uint *output [[buffer(5)]],
    device const uint *pixel_of_rank [[buffer(6)]], device atomic_uint *failure [[buffer(7)]],
    constant uint *p [[buffer(8)]], device const ushort *staging [[buffer(9)]],
    uint local_stream [[thread_position_in_grid]]) {
    uint pixels=p[0], packets=p[1], packet_count=p[3], stride=p[4], first_packet=p[5];
    if (local_stream>=pixels*packet_count) return;
    uint packet=local_stream/pixels, pixel=pixel_of_rank[local_stream%pixels];
    uint stream=(first_packet+packet)*pixels+local_stream%pixels;
    uint first=prt_source_offset(offsets,stream,pixels*packets);
    uint end=prt_source_offset(offsets,stream+1u,pixels*packets);
    PRTPackedReader r;
    if (!prt_packed_staging && !prt_packed_begin(payload,table,first,end,p[2],uint(modes[stream]),r)) {
        atomic_store_explicit(failure,103u,memory_order_relaxed); return;
    }
    uint checkpoints=(packet_count*16u+31u)/32u;
    uint offset=headers[pixel*stride];
    if (packet/2u!=0u) offset+=headers[pixel*stride+packet/2u];
    if ((packet&1u)!=0u) {
        for (uint word=0u; word<2u; ++word) {
            uint bits=headers[pixel*stride+checkpoints+(packet-1u)*2u+word];
            for (uint nibble=0u; nibble<8u; ++nibble) {
                uint width=(bits>>(nibble*4u))&15u; offset+=width==15u?16u:width;
            }
        }
    }
    for (uint tile=0u; tile<16u; ++tile) {
        uint width=(headers[pixel*stride+checkpoints+packet*2u+tile/8u]>>((tile&7u)*4u))&15u;
        if (width==15u) width=16u;
        // Low-count tiles dominate diffraction data. Fixed registers avoid
        // dynamically indexing a 16-word private array for their bit planes.
        if (width<=2u) {
            uint low=0u, high=0u;
            for (uint scan=0u; scan<32u; ++scan) {
                uint value=prt_packed_staging ? uint(staging[(packet*512u+tile*32u+scan)*pixels+local_stream%pixels]) : prt_packed_next(payload,r);
                low|=(value&1u)<<scan;
                high|=((value>>1u)&1u)<<scan;
            }
            if (width>0u) output[offset]=low;
            if (width>1u) output[offset+1u]=high;
        } else {
            uint4 a=0u, b=0u, c=0u, d=0u;
            for (uint scan=0u; scan<32u; ++scan) {
                uint value=prt_packed_staging ? uint(staging[(packet*512u+tile*32u+scan)*pixels+local_stream%pixels]) : prt_packed_next(payload,r);
                a|=((uint4(value)>>uint4(0u,1u,2u,3u))&1u)<<scan;
                if (width>4u) b|=((uint4(value)>>uint4(4u,5u,6u,7u))&1u)<<scan;
                if (width>8u) c|=((uint4(value)>>uint4(8u,9u,10u,11u))&1u)<<scan;
                if (width>12u) d|=((uint4(value)>>uint4(12u,13u,14u,15u))&1u)<<scan;
            }
            for (uint plane=0u; plane<min(width,4u); ++plane) output[offset+plane]=a[plane];
            for (uint plane=4u; plane<min(width,8u); ++plane) output[offset+plane]=b[plane-4u];
            for (uint plane=8u; plane<min(width,12u); ++plane) output[offset+plane]=c[plane-8u];
            for (uint plane=12u; plane<width; ++plane) output[offset+plane]=d[plane-12u];
        }
        offset+=width;
    }
    if (!prt_packed_staging && !prt_packed_finished(r)) atomic_store_explicit(failure,104u,memory_order_relaxed);
}

// Exact detector-mask delta for one flat 16K record. Four SIMD groups share a
// threadgroup; each owns one 512-scan packet and a disjoint 2 KiB scratch
// slice. Dense streams reduce within that SIMD, sparse streams scatter their
// at-most-two events atomically, and every output scan is written once.
kernel void paired_runtime_tans_detector_packet_owner(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device const uint *selected [[buffer(4)]],
    device const int *coefficients [[buffer(5)]],
    device uint *output [[buffer(6)]],
    device atomic_uint *failure [[buffer(7)]],
    constant uint *p [[buffer(8)]],
    uint packet_group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    uint simd = thread_id >> 5u;
    uint lane = thread_id & 31u;
    uint pixels = p[0], packets = p[1], changed = p[2];
    uint output_first = p[3], payload_bytes = p[4];
    bool include_sparse = p[5] != 0u;
    uint packet = packet_group * 4u + simd;
    if (packet >= packets) return;
    threadgroup atomic_int partial_storage[4u * PRT_INTERVAL];
    threadgroup atomic_int *partials = partial_storage + simd * PRT_INTERVAL;
    for (uint scan = lane; scan < PRT_INTERVAL; scan += 32u)
        atomic_store_explicit(partials + scan, 0, memory_order_relaxed);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0u; base < changed; base += 32u) {
        uint ordinal = base + lane;
        bool active = ordinal < changed;
        uint pixel = active ? selected[ordinal] : 0u;
        if (active && pixel >= pixels) {
            atomic_store_explicit(failure, 30u, memory_order_relaxed);
            active = false;
        }
        int coefficient = active ? coefficients[ordinal] : 0;
        uint stream = packet * pixels + pixel;
        uint total_streams = pixels * packets;
        uint first = active ? prt_source_offset(offsets, stream, total_streams) : 0u;
        uint end = active ? prt_source_offset(offsets, stream + 1u, total_streams) : 0u;
        uint mode = active ? uint(modes[stream]) : 253u;
        if (active && (end < first || end > payload_bytes)) {
            atomic_store_explicit(failure, 31u, memory_order_relaxed);
            active = false;
            mode = 253u;
        }
        bool sparse = include_sparse && active && mode == 252u;
        bool dense = active && mode != 252u && mode != 253u;
        uint constant_value = 0u, state = 0u;
        PRTFastReader reader;
        bool entropy = dense && prt_entropy_mode(mode);
        if (dense && mode == 255u) {
            if (end - first == 2u)
                constant_value = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
            else active = dense = false;
        } else if (dense && mode == 254u) {
            if (end - first != 2u * PRT_INTERVAL) active = dense = false;
        } else if (entropy) {
            device const uint *table = decoding + prt_entropy_model(mode) * PRT_STATES;
            if (!prt_fast_begin(payload, first, end, payload_bytes, table, reader))
                active = dense = entropy = false;
        } else if (dense) {
            active = dense = false;
        }
        if (!active && ordinal < changed)
            atomic_store_explicit(failure, 32u, memory_order_relaxed);

        if (simd_any(dense)) {
            for (uint pair = 0u; pair < PRT_INTERVAL / 2u; ++pair) {
                uint a = 0u, b = 0u;
                if (dense) {
                    if (entropy) {
                        if (!prt_fast_pair(reader, a, b)) {
                            atomic_store_explicit(failure, 33u, memory_order_relaxed);
                            dense = false;
                        }
                    } else if (mode == 254u) {
                        uint at = first + 4u * pair;
                        a = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                        b = uint(payload[at + 2u]) | (uint(payload[at + 3u]) << 8u);
                    } else {
                        a = b = constant_value;
                    }
                }
                int sum_a = simd_sum(int(a) * coefficient);
                int sum_b = simd_sum(int(b) * coefficient);
                if (lane == 0u) {
                    atomic_fetch_add_explicit(
                        partials + 2u * pair, sum_a, memory_order_relaxed);
                    atomic_fetch_add_explicit(
                        partials + 2u * pair + 1u, sum_b, memory_order_relaxed);
                }
            }
            if (entropy && dense && !prt_fast_finished(reader))
                atomic_store_explicit(failure, 34u, memory_order_relaxed);
        }
        if (sparse) {
            if (((end - first) & 1u) != 0u) {
                atomic_store_explicit(failure, 35u, memory_order_relaxed);
            } else {
                uint previous = 0u;
                bool has_previous = false;
                for (uint cursor = first; cursor < end; cursor += 2u) {
                    uint event = uint(payload[cursor])
                        | (uint(payload[cursor + 1u]) << 8u);
                    uint position = event >> 7u;
                    if (position >= PRT_INTERVAL || (has_previous && position <= previous)) {
                        atomic_store_explicit(failure, 36u, memory_order_relaxed);
                        break;
                    }
                    int contribution = int((event & 127u) + 1u) * coefficient;
                    atomic_fetch_add_explicit(
                        partials + position, contribution, memory_order_relaxed);
                    previous = position;
                    has_previous = true;
                }
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint packet_first = output_first + packet * PRT_INTERVAL;
    for (uint scan = lane; scan < PRT_INTERVAL; scan += 32u)
        output[packet_first + scan] += uint(
            atomic_load_explicit(partials + scan, memory_order_relaxed));
}

// Sparse streams have at most two events in the runtime ABI. Give every
// changed-pixel/packet stream its own thread so sparse ADF boundaries do not
// serialize behind dense packet-owner batches.
kernel void paired_runtime_tans_detector_sparse_scatter(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device const int *coefficients [[buffer(4)]],
    device atomic_uint *output [[buffer(5)]],
    device atomic_uint *failure [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    uint job [[thread_position_in_grid]]) {
    uint pixels = p[0], packets = p[1], changed = p[2], payload_bytes = p[3];
    if (job >= packets * changed) return;
    uint packet = job / changed;
    uint ordinal = job - packet * changed;
    uint pixel = selected[ordinal];
    if (pixel >= pixels) {
        atomic_store_explicit(failure, 50u, memory_order_relaxed);
        return;
    }
    uint stream = packet * pixels + pixel;
    if (modes[stream] != 252u) return;
    uint total_streams = pixels * packets;
    uint first = prt_source_offset(offsets, stream, total_streams);
    uint end = prt_source_offset(offsets, stream + 1u, total_streams);
    if (end < first || end > payload_bytes || ((end - first) & 1u) != 0u) {
        atomic_store_explicit(failure, 51u, memory_order_relaxed);
        return;
    }
    uint previous = 0u;
    bool has_previous = false;
    uint sign = uint(coefficients[ordinal]);
    for (uint cursor = first; cursor < end; cursor += 2u) {
        uint event = uint(payload[cursor]) | (uint(payload[cursor + 1u]) << 8u);
        uint position = event >> 7u;
        if (position >= PRT_INTERVAL || (has_previous && position <= previous)) {
            atomic_store_explicit(failure, 52u, memory_order_relaxed);
            return;
        }
        uint contribution = ((event & 127u) + 1u) * sign;
        atomic_fetch_add_explicit(
            output + packet * PRT_INTERVAL + position, contribution,
            memory_order_relaxed);
        previous = position;
        has_previous = true;
    }
}

// Benchmark-only selected-stream mode histogram. The 256 counters are the only
// CPU-readable result; the acquisition-wide private mode array stays resident.
kernel void paired_runtime_tans_detector_mode_histogram(
    device const uchar *modes [[buffer(0)]],
    device const uint *selected [[buffer(1)]],
    device atomic_uint *histogram [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint job [[thread_position_in_grid]]) {
    uint pixels = p[0], packets = p[1], changed = p[2];
    if (job >= packets * changed) return;
    uint packet = job / changed;
    uint ordinal = job - packet * changed;
    uint pixel = selected[ordinal];
    if (pixel >= pixels) return;
    uint mode = uint(modes[packet * pixels + pixel]);
    atomic_fetch_add_explicit(histogram + mode, 1u, memory_order_relaxed);
}

// Benchmark-only entropy-mode census. Reduce 128 stream flags in threadgroup
// memory so a full detector mask does not serialize millions of global atomics.
kernel void paired_runtime_tans_detector_entropy_mode_count(
    device const uchar *modes [[buffer(0)]],
    device const uint *selected [[buffer(1)]],
    device atomic_uint *entropy_count [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint job [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint partial[128];
    uint pixels = p[0], packets = p[1], changed = p[2];
    uint jobs = packets * changed;
    uint entropy = 0u;
    if (job < jobs) {
        uint packet = job / changed;
        uint ordinal = job - packet * changed;
        uint pixel = selected[ordinal];
        if (pixel < pixels) {
            uint mode = uint(modes[packet * pixels + pixel]);
            entropy = mode >= 64u && mode <= 95u ? 1u : 0u;
        }
    }
    partial[lane] = entropy;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        atomic_fetch_add_explicit(entropy_count, partial[0], memory_order_relaxed);
    }
}

// Benchmark-only census of the decoder's 64-stream outer chunks and 32-lane
// SIMD halves. Each job checks one packet/64-stream chunk and increments four
// outer-chunk counters, four SIMD-half counters, and one invalid-index counter.
// The resident mode array remains private and no detector-sized output is
// allocated or copied back.
kernel void paired_runtime_tans_entropy_chunk_census(
    device const uchar *modes [[buffer(0)]],
    device const uint *selected [[buffer(1)]],
    device atomic_uint *counts [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint job [[thread_position_in_grid]]) {
    uint pixels = p[0], packets = p[1], selected_count = p[2];
    uint chunk_width = p[3], full_chunks = p[4], tail_streams = p[5];
    uint chunks_per_packet = full_chunks + (tail_streams == 0u ? 0u : 1u);
    if (job >= packets * chunks_per_packet) return;

    uint packet = job / chunks_per_packet;
    uint chunk = job - packet * chunks_per_packet;
    bool is_tail = chunk == full_chunks;
    uint begin = chunk * chunk_width;
    uint count = is_tail ? tail_streams : chunk_width;
    uint second_count = count > 32u ? count - 32u : 0u;
    bool first_all_entropy = true;
    bool second_all_entropy = true;
    for (uint offset = 0u; offset < count; ++offset) {
        uint ordinal = begin + offset;
        if (ordinal >= selected_count) {
            atomic_fetch_add_explicit(counts + 8u, 1u, memory_order_relaxed);
            return;
        }
        uint pixel = selected[ordinal];
        if (pixel >= pixels) {
            atomic_fetch_add_explicit(counts + 8u, 1u, memory_order_relaxed);
            return;
        }
        uint mode = uint(modes[packet * pixels + pixel]);
        bool entropy = mode >= 64u && mode <= 95u;
        if (offset < 32u) first_all_entropy = first_all_entropy && entropy;
        else second_all_entropy = second_all_entropy && entropy;
    }
    bool all_entropy = first_all_entropy && second_all_entropy;
    uint full64_counter = is_tail ? (all_entropy ? 2u : 3u) : (all_entropy ? 0u : 1u);
    atomic_fetch_add_explicit(counts + full64_counter, 1u, memory_order_relaxed);

    // Every outer group starts with a complete 32-stream SIMD half. The
    // second half is complete for ordinary chunks and may be partial at tail.
    uint first_counter = first_all_entropy ? 4u : 5u;
    atomic_fetch_add_explicit(counts + first_counter, 1u, memory_order_relaxed);
    if (second_count == 32u) {
        uint second_counter = second_all_entropy ? 4u : 5u;
        atomic_fetch_add_explicit(counts + second_counter, 1u, memory_order_relaxed);
    } else if (second_count != 0u) {
        uint second_counter = second_all_entropy ? 6u : 7u;
        atomic_fetch_add_explicit(counts + second_counter, 1u, memory_order_relaxed);
    }
}

constant uint prt_streams_per_lane_requested [[function_constant(0)]];
constant uint prt_streams_per_lane = is_function_constant_defined(
    prt_streams_per_lane_requested) ? prt_streams_per_lane_requested : 2u;
constant bool prt_plain_scratch_requested [[function_constant(1)]];
constant bool prt_plain_scratch = is_function_constant_defined(
    prt_plain_scratch_requested) ? prt_plain_scratch_requested : false;

// FC28 (opt-in, trusted table only): cadence window reverse reader.
// The eager reader refills whenever one lane's reservoir drops below 32 bits.
// In a lockstep 32-lane SIMD group some lane needs that refill in most pair
// iterations (CPU census of seven-tilt ADF edge streams: 58-88% of group
// iterations, 4-10% of lanes), so every lane pays the refill instructions.
// This reader keeps only the unread bit count and one 64-bit payload window.
// Every prt_window_cadence pairs all lanes reload at the same iteration. A
// 4-byte-aligned window whose top lies within three bytes above the unread
// bits exposes at least 33 unread bits, and the validated table proves at most
// 10 state bits per ordinary pair, so cadence 1, 2 or 3 never reads outside
// the window. Escapes reload before their 13-bit word and between and after
// 16-bit fields. Loads stay within the eager reader's [end - 8, end + 3]
// bound. Payload failures are sticky: a malformed stream keeps decoding
// bounded values and fails the same terminal check (all meaningful bits
// consumed, final state zero); any failure code discards the update.
constant bool prt_window_reader_requested [[function_constant(28)]];
constant bool prt_window_reader = is_function_constant_defined(
    prt_window_reader_requested) ? prt_window_reader_requested : false;
// FC30: exact lane-private event accumulation. Each lane scatters its event
// streams with plain adds into its own 128 x uint4 row (2,048 B of thread stack),
// both stream slots share one flat loop to the SIMD-group maximum, and the rows
// are reduced once per packet with 128 uint4 simd_sum calls. Lane L publishes
// scans 16L...16L+15, so every scan is still published exactly once.
// FC44: FC30's flat event loop, but events are added atomically into the shared
// packet partials (no lane-private rows, clears or extra reductions).
constant bool prt_flat_events_requested [[function_constant(44)]];
constant bool prt_flat_events = is_function_constant_defined(
    prt_flat_events_requested) ? prt_flat_events_requested : false;
constant bool prt_event_rows_requested [[function_constant(30)]];
constant bool prt_event_rows = is_function_constant_defined(
    prt_event_rows_requested) ? prt_event_rows_requested : false;
// FC43: lane L decodes residual ordinals base + 2L and base + 2L + 1, so with
// rank-ordered residuals a lane's two streams are adjacent in the payload.
constant bool prt_adjacent_lane_streams_requested [[function_constant(43)]];
constant bool prt_adjacent_lane_streams = is_function_constant_defined(
    prt_adjacent_lane_streams_requested) ? prt_adjacent_lane_streams_requested : false;
constant uint prt_window_cadence_requested [[function_constant(29)]];
constant uint prt_window_cadence = is_function_constant_defined(
    prt_window_cadence_requested) ? prt_window_cadence_requested : 3u;
// Entropy records are shorter than 2 * PRT_INTERVAL bytes (fewer than 8192
// meaningful bits). An unread count at or above this bound can only be a
// wrapped over-consumption and is rejected before it forms a payload address.
constant uint PRT_WINDOW_WRAP_BITS = 16384u;
// FC40: timing-only ablations of the FC28 loop. Any nonzero level produces wrong maps.
// 1 skip the pair loop (stream setup only); 2 loop without pair decode; 3 decode without
// reductions or partial adds; 4 decode and SIMD sums without partial adds; 6 decode without
// window reloads (no payload reads after the header).
constant uint prt_window_diag_requested [[function_constant(40)]];
constant uint prt_window_diag = is_function_constant_defined(
    prt_window_diag_requested) ? prt_window_diag_requested : 0u;

struct PRTWindowReader {
    device const uint *table;
    ulong window;
    uint begin, unread, window_bit, state;
    bool valid;
};

inline bool prt_window_begin(
    device const uchar *payload, uint first, uint end, uint payload_bytes,
    device const uint *table, thread PRTWindowReader &reader) {
    reader.table = table;
    reader.window = 0ul;
    reader.begin = first + 2u;
    reader.unread = 0u;
    reader.window_bit = 0u;
    reader.state = 0u;
    reader.valid = end >= first && end <= payload_bytes && end - first >= 2u
        && end - first < 2u * PRT_INTERVAL;
    if (!reader.valid) return false;
    uint header = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
    uint tail = header & 7u;
    reader.state = header >> 6u;
    reader.valid = ((header >> 3u) & 7u) == 0u && reader.state < PRT_STATES
        && (tail == 0u || end > reader.begin);
    if (!reader.valid) return false;
    reader.unread = (end - reader.begin) * 8u - ((8u - tail) & 7u);
    if (tail != 0u)
        reader.valid = (uint(payload[end - 1u]) >> tail) == 0u;
    return reader.valid;
}

// FC41: trusted stream setup for FC28. Every stream of the resident was decoded with
// full validation when the exact regional-sum index was built at load, and resident
// buffers are immutable afterwards, so per-update bounds and header checks are
// redundant. Reload wrap guards stay in place.
constant bool prt_trusted_setup_requested [[function_constant(41)]];
constant bool prt_trusted_setup = is_function_constant_defined(
    prt_trusted_setup_requested) ? prt_trusted_setup_requested : false;

inline void prt_window_begin_trusted(
    device const uchar *payload, uint first, uint end,
    device const uint *table, thread PRTWindowReader &reader) {
    reader.table = table;
    reader.window = 0ul;
    reader.begin = first + 2u;
    reader.window_bit = 0u;
    reader.valid = true;
    // FC40 level 7 (timing only): take the header from a hash instead of payload memory
    // to measure the stream-header cache miss. Maps are wrong at this level.
    uint header = prt_window_diag == 7u
        ? ((first * 2654435761u) >> 16u)
        : (uint(payload[first]) | (uint(payload[first + 1u]) << 8u));
    reader.state = (header >> 6u) & 1023u;
    reader.unread = (end - reader.begin) * 8u - ((8u - (header & 7u)) & 7u);
}

inline void prt_window_reload(
    device const uchar *payload, thread PRTWindowReader &reader) {
    bool wrapped = reader.unread >= PRT_WINDOW_WRAP_BITS;
    reader.valid = reader.valid && !wrapped;
    reader.unread = wrapped ? 0u : reader.unread;
    uint top = reader.begin + ((reader.unread + 7u) >> 3u);
    uint aligned = (top >= 5u ? top - 5u : 0u) & ~3u;
    device const uint *words = reinterpret_cast<device const uint *>(
        payload + aligned);
    reader.window = ulong(words[0]) | (ulong(words[1]) << 32u);
    // Window start relative to the record body, modulo 2^32. It wraps when
    // the window starts below the body; (unread - window_bit) stays exact.
    reader.window_bit = (aligned - reader.begin) * 8u;
}

inline uint prt_window_take(thread PRTWindowReader &reader, uint count) {
    reader.unread -= count;
    uint shift = (reader.unread - reader.window_bit) & 63u;
    return uint(reader.window >> shift) & ((1u << count) - 1u);
}

inline void prt_window_pair(
    device const uchar *payload, thread PRTWindowReader &reader,
    thread uint &a, thread uint &b) {
    uint code = reader.table[reader.state];
    reader.state = (code >> 16u) + prt_window_take(reader, (code >> 12u) & 15u);
    uint pair = code & 4095u;
    if (pair != 4095u) {
        a = pair & 63u;
        b = pair >> 6u;
        return;
    }
    prt_window_reload(payload, reader);
    uint word = prt_window_take(reader, 13u);
    if (word < 4096u) {
        a = word & 63u;
        b = word >> 6u;
    } else if (word == 4096u) {
        a = prt_window_take(reader, 16u);
        prt_window_reload(payload, reader);
        b = prt_window_take(reader, 16u);
        prt_window_reload(payload, reader);
    } else {
        reader.valid = false;
        a = b = 0u;
    }
}

inline bool prt_window_finished(thread PRTWindowReader &reader) {
    return reader.valid && reader.unread == 0u && reader.state == 0u;
}

// Independent detector streams per lane expose tANS instruction-level
// parallelism and share SIMD reductions. Exact sparse alternatives remain
// event-driven and share the same packet-local accumulator. The production
// specializations use two or four streams per lane.
kernel void paired_runtime_tans_detector_packet_owner2(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device const uint *selected [[buffer(4)]],
    device const int *coefficients [[buffer(5)]],
    device uint *output [[buffer(6)]],
    device atomic_uint *failure [[buffer(7)]],
    constant uint *p [[buffer(8)]],
    uint packet_group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    uint simd = thread_id >> 5u;
    uint lane = thread_id & 31u;
    uint pixels = p[0], packets = p[1], changed = p[2];
    uint output_first = p[3], payload_bytes = p[4];
    bool include_sparse = p[5] != 0u;
    if (prt_decode_checksum && prt_packet_split_count != 1u) {
        if (thread_id == 0u) atomic_store_explicit(failure, 100u, memory_order_relaxed);
        return;
    }
    if (prt_packet_split_count != 1u && prt_packet_split_count != 2u
        && prt_packet_split_count != 4u && prt_packet_split_count != 8u) {
        if (thread_id == 0u) atomic_store_explicit(failure, 98u, memory_order_relaxed);
        return;
    }
    uint split_index = packet_group % prt_packet_split_count;
    uint packet_batch = packet_group / prt_packet_split_count;
    uint packet = packet_batch * 4u + simd;
    // Block stride: logical packet L maps to physical 512-scan block phase + L * stride.
    // Callers that do not use it pass stride 1 and phase 0.
    uint packet_stride = p[6] == 0u ? 1u : p[6];
    uint packet_phase = p[7];
    if (packet_phase >= packet_stride) {
        if (thread_id == 0u) atomic_store_explicit(failure, 96u, memory_order_relaxed);
        return;
    }
    uint logical_packets = packets > packet_phase
        ? (packets - packet_phase + packet_stride - 1u) / packet_stride : 0u;
    if (packet >= logical_packets) return;
    packet = packet_phase + packet * packet_stride;
    if (packet >= packets) return;
    if (prt_register_reduction && prt_plain_packet_owner2) {
        if (thread_id == 0u) atomic_store_explicit(failure, 99u, memory_order_relaxed);
        return;
    }
    if (prt_window_reader && (!prt_trusted_decode_table || prt_macro_enabled
        || prt_reader32_enabled || prt_decode_checksum || prt_register_reduction
        || prt_vector_pair_reduction
        || prt_simd_entropy_fast_path || prt_phased_table_loads
        || prt_pair_unroll != 1u || prt_window_cadence == 0u
        || prt_window_cadence > 3u)) {
        if (thread_id == 0u) atomic_store_explicit(failure, 106u, memory_order_relaxed);
        return;
    }
    threadgroup uint partial_storage[4u * PRT_INTERVAL];
    threadgroup uint *partials = partial_storage + simd * PRT_INTERVAL;
    threadgroup atomic_uint *atomic_partials =
        reinterpret_cast<threadgroup atomic_uint *>(partials);
    uint decode_checksum = 0u;
    uint diag_sink = 0u;
    if (prt_event_rows && (prt_streams_per_lane != 2u || prt_register_reduction
        || prt_decode_checksum || prt_plain_packet_owner2)) {
        if (thread_id == 0u) atomic_store_explicit(failure, 67u, memory_order_relaxed);
        return;
    }
    uint4 event_rows[PRT_INTERVAL / 4u];
    uint4 event_owned[4];
    bool event_rows_ready = false;
    uint event_failure = 0u;
    if (prt_event_rows) {
        for (uint block = 0u; block < 4u; ++block)
            event_owned[block] = uint4(0u);
    }
    uint dense_accum[16];
    if (prt_register_reduction) {
        #pragma unroll
        for (uint slot = 0u; slot < 16u; ++slot)
            dense_accum[slot] = 0u;
    }
    if (!prt_decode_checksum) {
        for (uint scan = lane; scan < PRT_INTERVAL; scan += 32u) {
            if (prt_plain_packet_owner2)
                partials[scan] = 0u;
            else
                atomic_store_explicit(atomic_partials + scan, 0u, memory_order_relaxed);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint base = split_index * 32u * prt_streams_per_lane;
    uint base_stride = prt_packet_split_count * 32u * prt_streams_per_lane;
    for (; base < changed; base += base_stride) {
        bool active[4], sparse[4], dense[4], entropy[4], simd_all_entropy[4], compact[4];
        uint first[4], end[4], mode[4], constant_value[4];
        int coefficient[4];
        PRTFastReader reader[4];
        PRTWindowReader window_reader[4];
        #pragma unroll
        for (uint j = 0u; j < prt_streams_per_lane; ++j) {
            uint ordinal = prt_adjacent_lane_streams && prt_streams_per_lane == 2u
                ? base + 2u * lane + j : base + j * 32u + lane;
            if (prt_trusted_setup && prt_window_reader) {
                active[j] = ordinal < changed;
                uint trusted_pixel = active[j] ? selected[ordinal] : 0u;
                coefficient[j] = active[j] ? coefficients[ordinal] : 0;
                uint trusted_stream = packet * pixels + trusted_pixel;
                uint trusted_total = pixels * packets;
                first[j] = active[j]
                    ? prt_source_offset(offsets, trusted_stream, trusted_total) : 0u;
                end[j] = active[j]
                    ? prt_source_offset(offsets, trusted_stream + 1u, trusted_total) : 0u;
                mode[j] = active[j] ? uint(modes[trusted_stream]) : 253u;
                sparse[j] = include_sparse && active[j] && mode[j] == 252u;
                compact[j] = active[j] && mode[j] == 251u;
                dense[j] = active[j] && mode[j] != 252u && mode[j] != 253u && mode[j] != 251u;
                entropy[j] = dense[j] && prt_entropy_mode(mode[j]);
                constant_value[j] = dense[j] && mode[j] == 255u
                    ? (uint(payload[first[j]]) | (uint(payload[first[j] + 1u]) << 8u)) : 0u;
                if (entropy[j]) {
                    prt_window_begin_trusted(
                        payload, first[j], end[j],
                        decoding + prt_entropy_model(mode[j]) * PRT_STATES, window_reader[j]);
                }
                simd_all_entropy[j] = false;
                continue;
            }
            active[j] = ordinal < changed;
            uint pixel = active[j] ? selected[ordinal] : 0u;
            if (active[j] && pixel >= pixels) {
                atomic_store_explicit(failure, 60u, memory_order_relaxed);
                active[j] = false;
            }
            coefficient[j] = active[j] ? coefficients[ordinal] : 0;
            uint stream = packet * pixels + pixel;
            uint total_streams = pixels * packets;
            first[j] = active[j] ? prt_source_offset(offsets, stream, total_streams) : 0u;
            end[j] = active[j]
                ? prt_source_offset(offsets, stream + 1u, total_streams) : 0u;
            mode[j] = active[j] ? uint(modes[stream]) : 253u;
            if (active[j] && (end[j] < first[j] || end[j] > payload_bytes)) {
                atomic_store_explicit(failure, 61u, memory_order_relaxed);
                active[j] = false;
                mode[j] = 253u;
            }
            sparse[j] = include_sparse && active[j] && mode[j] == 252u;
            compact[j] = active[j] && mode[j] == 251u;
            dense[j] = active[j] && mode[j] != 252u && mode[j] != 253u && mode[j] != 251u;
            entropy[j] = dense[j] && prt_entropy_mode(mode[j]);
            constant_value[j] = 0u;
            if (dense[j] && mode[j] == 255u) {
                if (end[j] - first[j] == 2u)
                    constant_value[j] = uint(payload[first[j]])
                        | (uint(payload[first[j] + 1u]) << 8u);
                else active[j] = dense[j] = false;
            } else if (dense[j] && mode[j] == 254u) {
                if (end[j] - first[j] != 2u * PRT_INTERVAL)
                    active[j] = dense[j] = false;
            } else if (entropy[j]) {
                uint table_stride = PRT_STATES + (prt_macro_enabled
                    ? PRT_STATES * (1u << prt_macro_lookahead_bits) * 2u : 0u);
                device const uint *table = decoding
                    + prt_entropy_model(mode[j]) * table_stride;
                bool reader_ready = prt_window_reader
                    ? prt_window_begin(
                        payload, first[j], end[j], payload_bytes, table, window_reader[j])
                    : prt_fast_begin(
                        payload, first[j], end[j], payload_bytes, table, reader[j]);
                if (!reader_ready)
                    active[j] = dense[j] = entropy[j] = false;
            } else if (dense[j]) {
                active[j] = dense[j] = false;
            }
            if (!active[j] && ordinal < changed)
                atomic_store_explicit(failure, 62u, memory_order_relaxed);
            // The reduction is deliberately per unrolled stream j (one
            // 32-stream census unit), after offset, mode, and reader-header
            // validation. Inactive tail lanes and malformed records force the
            // ordinary decoder below; we never trust the file's mode alone.
            simd_all_entropy[j] = prt_simd_entropy_fast_path
                && simd_all(dense[j] && entropy[j]);
        }
        bool any_dense = false;
        #pragma unroll
        for (uint j = 0u; j < prt_streams_per_lane; ++j)
            any_dense = any_dense || dense[j];
        bool run_dense = simd_any(any_dense);
        if (prt_window_reader && prt_window_diag == 1u) run_dense = false;
        if (run_dense && prt_window_reader) {
            // FC28: every lane reloads its windows at the same pair index, so
            // no lane waits on another lane's data-dependent refill. Payload
            // failures are sticky and reported once per stream after the loop.
            for (uint block_first = 0u; block_first < PRT_INTERVAL / 2u;
                 block_first += prt_window_cadence) {
                #pragma unroll
                for (uint j = 0u; j < prt_streams_per_lane; ++j) {
                    if (entropy[j] && prt_window_diag != 6u)
                        prt_window_reload(payload, window_reader[j]);
                }
                uint block_end = min(block_first + prt_window_cadence, PRT_INTERVAL / 2u);
                for (uint pair = block_first; pair < block_end; ++pair) {
                    uint sum_a = 0u, sum_b = 0u;
                    #pragma unroll
                    for (uint j = 0u; j < prt_streams_per_lane; ++j) {
                        uint a = 0u, b = 0u;
                        if (entropy[j]) {
                            if (prt_window_diag == 2u) { a = 0u; b = 0u; }
                            else prt_window_pair(payload, window_reader[j], a, b);
                        } else if (dense[j] && mode[j] == 254u) {
                            uint at = first[j] + 4u * pair;
                            a = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                            b = uint(payload[at + 2u]) | (uint(payload[at + 3u]) << 8u);
                        } else if (dense[j]) {
                            a = b = constant_value[j];
                        }
                        sum_a += uint(int(a) * coefficient[j]);
                        sum_b += uint(int(b) * coefficient[j]);
                    }
                    if (prt_window_diag != 3u) {
                    sum_a = simd_sum(sum_a);
                    sum_b = simd_sum(sum_b);
                    }
                    if (prt_window_diag == 4u) diag_sink += sum_a ^ sum_b;
                    if (lane == 0u && prt_window_diag != 3u && prt_window_diag != 4u) {
                        // FC12 with FC28: only lane 0 writes this packet's partials
                        // inside the dense loop, so plain adds are race-free here.
                        if (prt_plain_packet_owner2) {
                            partials[2u * pair] += sum_a;
                            partials[2u * pair + 1u] += sum_b;
                        } else {
                            atomic_fetch_add_explicit(
                                atomic_partials + 2u * pair, sum_a, memory_order_relaxed);
                            atomic_fetch_add_explicit(
                                atomic_partials + 2u * pair + 1u, sum_b, memory_order_relaxed);
                        }
                    }
                }
            }
            #pragma unroll
            for (uint j = 0u; j < prt_streams_per_lane; ++j) {
                if (!entropy[j] || prt_trusted_setup || prt_window_diag == 1u
                    || prt_window_diag == 2u || prt_window_diag == 6u) continue;
                if (!window_reader[j].valid)
                    atomic_store_explicit(failure, 63u, memory_order_relaxed);
                else if (!prt_window_finished(window_reader[j]))
                    atomic_store_explicit(failure, 64u, memory_order_relaxed);
            }
        } else if (run_dense) {
            for (uint pair_base = 0u; pair_base < PRT_INTERVAL / 2u;
                 pair_base += prt_pair_unroll) {
                #pragma unroll
                for (uint pair_offset = 0u; pair_offset < prt_pair_unroll; ++pair_offset) {
                    uint pair = pair_base + pair_offset;
                    uint sum_a = 0u, sum_b = 0u;
                    uint preloaded_code[4];
                    if (prt_phased_table_loads && !prt_macro_enabled) {
                        #pragma unroll
                        for (uint j = 0u; j < prt_streams_per_lane; ++j) {
                            // The same guards protect the original pair call.
                            // Failed readers have dense cleared before the next pair.
                            if (dense[j] && entropy[j])
                                preloaded_code[j] = reader[j].table[reader[j].state];
                        }
                    }
                    #pragma unroll
                    for (uint j = 0u; j < prt_streams_per_lane; ++j) {
                        uint a = 0u, b = 0u;
                        if (dense[j]) {
                            if (simd_all_entropy[j]) {
                                if (!prt_fast_pair(reader[j], a, b)) {
                                    atomic_store_explicit(
                                        failure, 63u, memory_order_relaxed);
                                    dense[j] = false;
                                }
                            } else if (entropy[j]) {
                                bool decoded = prt_phased_table_loads && !prt_macro_enabled
                                    ? prt_fast_pair_code(reader[j], preloaded_code[j], a, b)
                                    : prt_fast_pair(reader[j], a, b);
                                if (!decoded) {
                                    atomic_store_explicit(failure, 63u, memory_order_relaxed);
                                    dense[j] = false;
                                }
                            } else if (mode[j] == 254u) {
                                uint at = first[j] + 4u * pair;
                                a = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                                b = uint(payload[at + 2u]) | (uint(payload[at + 3u]) << 8u);
                            } else {
                                a = b = constant_value[j];
                            }
                        }
                        sum_a += uint(int(a) * coefficient[j]);
                        sum_b += uint(int(b) * coefficient[j]);
                    }
                    if (prt_decode_checksum) {
                        decode_checksum += sum_a + sum_b;
                    } else if (prt_vector_pair_reduction) {
                        // Reduce both exact UInt32 scan components in one vector-valued
                        // SIMD-group operation. Component-wise modulo-UInt32 addition
                        // matches the two scalar reductions and preserves negative
                        // residual totals as their exact two's-complement bit patterns.
                        uint2 pair_sum = simd_sum(uint2(sum_a, sum_b));
                        sum_a = pair_sum.x;
                        sum_b = pair_sum.y;
                    } else {
                        sum_a = simd_sum(sum_a);
                        sum_b = simd_sum(sum_b);
                    }
                    if (!prt_decode_checksum && lane == 0u) {
                        if (prt_plain_packet_owner2) {
                            partials[2u * pair] += sum_a;
                            partials[2u * pair + 1u] += sum_b;
                        } else if (!prt_register_reduction) {
                            atomic_fetch_add_explicit(
                                atomic_partials + 2u * pair, sum_a, memory_order_relaxed);
                            atomic_fetch_add_explicit(
                                atomic_partials + 2u * pair + 1u, sum_b, memory_order_relaxed);
                        }
                    }
                    if (prt_register_reduction && (lane >> 1u) == (pair & 15u)) {
                        dense_accum[pair >> 4u] += (lane & 1u) == 0u ? sum_a : sum_b;
                    }
                }
            }
            #pragma unroll
            for (uint j = 0u; j < prt_streams_per_lane; ++j)
                if (entropy[j] && dense[j] && !prt_fast_finished(reader[j]))
                    atomic_store_explicit(failure, 64u, memory_order_relaxed);
        }
        if (prt_flat_events && !prt_event_rows && !prt_decode_checksum) {
            uint length0 = sparse[0] ? end[0] - first[0] : 0u;
            uint length1 = sparse[1] ? end[1] - first[1] : 0u;
            if (((length0 | length1) & 1u) != 0u)
                atomic_store_explicit(failure, 65u, memory_order_relaxed);
            uint count0 = length0 >> 1u;
            uint count01 = count0 + (length1 >> 1u);
            uint trips = simd_max(count01);
            uint cursor = first[0];
            uint weight = uint(coefficient[0]);
            uint second_weight = uint(coefficient[1]);
            uint next_position = 0u;
            uint disorder = 0u;
            for (uint trip = 0u; trip < trips; ++trip) {
                bool second = trip == count0;
                cursor = second ? first[1] : cursor;
                weight = second ? second_weight : weight;
                next_position = second ? 0u : next_position;
                bool live = trip < count01;
                uint at = live ? cursor : 0u;
                uint event = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                uint position = event >> 7u;
                disorder |= uint(live && position < next_position);
                next_position = position + 1u;
                uint contribution = live ? ((event & 127u) + 1u) * weight : 0u;
                atomic_fetch_add_explicit(
                    atomic_partials + position, contribution, memory_order_relaxed);
                cursor += 2u;
            }
            if (disorder != 0u)
                atomic_store_explicit(failure, 66u, memory_order_relaxed);
            sparse[0] = false;
            sparse[1] = false;
        }
        if (prt_event_rows) {
            uint length0 = sparse[0] ? end[0] - first[0] : 0u;
            uint length1 = sparse[1] ? end[1] - first[1] : 0u;
            if (((length0 | length1) & 1u) != 0u)
                event_failure = 65u;
            uint count0 = length0 >> 1u;
            uint count01 = count0 + (length1 >> 1u);
            uint trips = simd_max(count01);
            if (trips != 0u) {
                if (!event_rows_ready) {
                    for (uint block = 0u; block < PRT_INTERVAL / 4u; ++block)
                        event_rows[block] = uint4(0u);
                    event_rows_ready = true;
                }
                uint cursor = first[0];
                uint weight = uint(coefficient[0]);
                uint second_weight = uint(coefficient[1]);
                uint next_position = 0u;
                uint disorder = 0u;
                for (uint trip = 0u; trip < trips; ++trip) {
                    bool second = trip == count0;
                    cursor = second ? first[1] : cursor;
                    weight = second ? second_weight : weight;
                    next_position = second ? 0u : next_position;
                    bool live = trip < count01;
                    // Some lane has a live event, so payload bytes 0 and 1 exist;
                    // dead trips read them and add zero.
                    uint at = live ? cursor : 0u;
                    uint event = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                    uint position = event >> 7u;
                    disorder |= uint(live && position < next_position);
                    next_position = position + 1u;
                    uint contribution = live ? ((event & 127u) + 1u) * weight : 0u;
                    event_rows[position >> 2u][position & 3u] += contribution;
                    cursor += 2u;
                }
                if (disorder != 0u)
                    event_failure = 66u;
            }
            sparse[0] = false;
            sparse[1] = false;
        }
        bool any_sparse = false;
        #pragma unroll
        for (uint j = 0u; j < prt_streams_per_lane; ++j)
            any_sparse = any_sparse || sparse[j];
        if (prt_plain_packet_owner2 && simd_any(any_sparse))
            simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint j = 0u; j < prt_streams_per_lane; ++j) {
            if (!sparse[j]) continue;
            if (((end[j] - first[j]) & 1u) != 0u) {
                atomic_store_explicit(failure, 65u, memory_order_relaxed);
                continue;
            }
            uint previous = 0u;
            bool has_previous = false;
            for (uint cursor = first[j]; cursor < end[j]; cursor += 2u) {
                uint event = uint(payload[cursor])
                    | (uint(payload[cursor + 1u]) << 8u);
                uint position = event >> 7u;
                if (position >= PRT_INTERVAL || (has_previous && position <= previous)) {
                    atomic_store_explicit(failure, 66u, memory_order_relaxed);
                    break;
                }
                int contribution = int((event & 127u) + 1u) * coefficient[j];
                if (prt_decode_checksum)
                    decode_checksum += uint(contribution);
                else
                    atomic_fetch_add_explicit(
                        atomic_partials + position, uint(contribution), memory_order_relaxed);
                previous = position;
                has_previous = true;
            }
        }
        if (prt_plain_packet_owner2 && simd_any(any_sparse))
            simdgroup_barrier(mem_flags::mem_threadgroup);
        // Mode 251 compact events: one flat byte loop over both stream slots, to the
        // SIMD-group maximum byte count. Dead trips read byte 0 and write nothing.
        if (prt_streams_per_lane == 4u && (compact[2] || compact[3]))
            atomic_store_explicit(failure, 88u, memory_order_relaxed);
        {
            uint clen0 = compact[0] ? end[0] - first[0] : 0u;
            uint clen1 = compact[1] ? end[1] - first[1] : 0u;
            uint ctotal = clen0 + clen1;
            uint ctrips = simd_max(ctotal);
            if (ctrips != 0u && !prt_decode_checksum && prt_compact_pairs) {
                uint slot = clen0 != 0u ? 0u : 1u;
                uint cursor = slot == 0u ? first[0] : first[1];
                uint slot_end = slot == 0u ? first[0] + clen0 : first[1] + clen1;
                uint weight = uint(coefficient[slot]);
                uint next_position = 0u;
                uint stage = 0u, pending_gap = 0u, pending_value = 0u;
                uint bad = 0u;
                for (uint trip = 0u; trip < ctrips; ++trip) {
                    if (slot == 0u && cursor >= slot_end) {
                        bad |= uint(stage != 0u);
                        slot = 1u;
                        cursor = first[1];
                        slot_end = first[1] + clen1;
                        weight = uint(coefficient[1]);
                        next_position = 0u;
                        stage = 0u;
                    }
                    bool live = cursor < slot_end;
                    if (!simd_any(live))
                        break;
                    bool two_live = live && cursor + 1u < slot_end;
                    uint byte = uint(payload[live ? cursor : 0u]);
                    uint byte1 = uint(payload[two_live ? cursor + 1u : 0u]);
                    bool fast = two_live && stage == 0u
                        && (byte >> 3u) != 31u && (byte & 7u) != 0u
                        && (byte1 >> 3u) != 31u && (byte1 & 7u) != 0u;
                    bool four_live = prt_compact_quads && fast && cursor + 3u < slot_end;
                    uint byte2 = uint(payload[four_live ? cursor + 2u : 0u]);
                    uint byte3 = uint(payload[four_live ? cursor + 3u : 0u]);
                    bool fast4 = four_live
                        && (byte2 >> 3u) != 31u && (byte2 & 7u) != 0u
                        && (byte3 >> 3u) != 31u && (byte3 & 7u) != 0u;
                    if (fast4) {
                        uint position0 = next_position + (byte >> 3u);
                        uint position1 = position0 + 1u + (byte1 >> 3u);
                        uint position2 = position1 + 1u + (byte2 >> 3u);
                        uint position3 = position2 + 1u + (byte3 >> 3u);
                        bad |= uint(position3 >= PRT_INTERVAL);
                        atomic_fetch_add_explicit(
                            atomic_partials + min(position0, PRT_INTERVAL - 1u),
                            (byte & 7u) * weight, memory_order_relaxed);
                        atomic_fetch_add_explicit(
                            atomic_partials + min(position1, PRT_INTERVAL - 1u),
                            (byte1 & 7u) * weight, memory_order_relaxed);
                        atomic_fetch_add_explicit(
                            atomic_partials + min(position2, PRT_INTERVAL - 1u),
                            (byte2 & 7u) * weight, memory_order_relaxed);
                        atomic_fetch_add_explicit(
                            atomic_partials + min(position3, PRT_INTERVAL - 1u),
                            (byte3 & 7u) * weight, memory_order_relaxed);
                        next_position = position3 + 1u;
                        cursor += 4u;
                    } else if (fast) {
                        uint position0 = next_position + (byte >> 3u);
                        uint position1 = position0 + 1u + (byte1 >> 3u);
                        bad |= uint(position1 >= PRT_INTERVAL);
                        atomic_fetch_add_explicit(
                            atomic_partials + min(position0, PRT_INTERVAL - 1u),
                            (byte & 7u) * weight, memory_order_relaxed);
                        atomic_fetch_add_explicit(
                            atomic_partials + min(position1, PRT_INTERVAL - 1u),
                            (byte1 & 7u) * weight, memory_order_relaxed);
                        next_position = position1 + 1u;
                        cursor += 2u;
                    } else if (live) {
                        bool emit = false;
                        uint value = 0u;
                        uint next_stage = 0u;
                        if (stage == 0u) {
                            pending_gap = byte >> 3u;
                            pending_value = byte & 7u;
                            if (pending_gap == 31u) next_stage = 1u;
                            else if (pending_value == 0u) next_stage = 3u;
                            else { emit = true; value = pending_value; }
                        } else if (stage == 1u) {
                            pending_gap += byte;
                            if (byte == 255u) next_stage = 2u;
                            else if (pending_value == 0u) next_stage = 3u;
                            else { emit = true; value = pending_value; }
                        } else if (stage == 2u) {
                            pending_gap += byte;
                            if (pending_value == 0u) next_stage = 3u;
                            else { emit = true; value = pending_value; }
                        } else {
                            emit = true;
                            value = 8u + byte;
                        }
                        if (emit) {
                            uint position = next_position + pending_gap;
                            bad |= uint(position >= PRT_INTERVAL);
                            atomic_fetch_add_explicit(
                                atomic_partials + min(position, PRT_INTERVAL - 1u),
                                value * weight, memory_order_relaxed);
                            next_position = position + 1u;
                        }
                        stage = next_stage;
                        cursor += 1u;
                    }
                }
                bad |= uint(stage != 0u);
                if (bad != 0u)
                    atomic_store_explicit(failure, 68u, memory_order_relaxed);
            } else if (ctrips != 0u && !prt_decode_checksum) {
                uint cursor = first[0];
                uint weight = uint(coefficient[0]);
                uint next_position = 0u;
                uint stage = 0u, pending_gap = 0u, pending_value = 0u;
                uint bad = 0u;
                for (uint trip = 0u; trip < ctrips; ++trip) {
                    if (trip == clen0) {
                        bad |= uint(clen0 != 0u && stage != 0u);
                        cursor = first[1];
                        weight = uint(coefficient[1]);
                        next_position = 0u;
                        stage = 0u;
                    }
                    bool live = trip < ctotal;
                    uint byte = uint(payload[live ? cursor : 0u]);
                    bool emit = false;
                    uint value = 0u;
                    uint next_stage = 0u;
                    if (stage == 0u) {
                        pending_gap = byte >> 3u;
                        pending_value = byte & 7u;
                        if (pending_gap == 31u) next_stage = 1u;
                        else if (pending_value == 0u) next_stage = 3u;
                        else { emit = true; value = pending_value; }
                    } else if (stage == 1u) {
                        pending_gap += byte;
                        if (byte == 255u) next_stage = 2u;
                        else if (pending_value == 0u) next_stage = 3u;
                        else { emit = true; value = pending_value; }
                    } else if (stage == 2u) {
                        pending_gap += byte;
                        if (pending_value == 0u) next_stage = 3u;
                        else { emit = true; value = pending_value; }
                    } else {
                        emit = true;
                        value = 8u + byte;
                    }
                    if (live) {
                        if (emit) {
                            uint position = next_position + pending_gap;
                            bad |= uint(position >= PRT_INTERVAL);
                            position = min(position, PRT_INTERVAL - 1u);
                            atomic_fetch_add_explicit(
                                atomic_partials + position, value * weight, memory_order_relaxed);
                            next_position = position + 1u;
                        }
                        stage = next_stage;
                        cursor += 1u;
                    }
                }
                bad |= uint(stage != 0u);
                if (bad != 0u)
                    atomic_store_explicit(failure, 68u, memory_order_relaxed);
            }
        }
    }
    if (prt_event_rows && event_rows_ready) {
        // Every lane joins all 128 reductions; lane L keeps blocks 4L...4L+3.
        for (uint owner = 0u; owner < 32u; ++owner) {
            uint block = owner << 2u;
            uint4 total0 = simd_sum(event_rows[block]);
            uint4 total1 = simd_sum(event_rows[block + 1u]);
            uint4 total2 = simd_sum(event_rows[block + 2u]);
            uint4 total3 = simd_sum(event_rows[block + 3u]);
            if (owner == lane) {
                event_owned[0] = total0;
                event_owned[1] = total1;
                event_owned[2] = total2;
                event_owned[3] = total3;
            }
        }
    }
    if (prt_event_rows && event_failure != 0u)
        atomic_store_explicit(failure, event_failure, memory_order_relaxed);
    if (prt_window_diag == 4u && lane == 0u && diag_sink == 0x9e3779b9u)
        atomic_store_explicit(failure, 0u, memory_order_relaxed);
    if (prt_decode_checksum) {
        uint packet_checksum = simd_sum(decode_checksum);
        if (lane == 0u) output[packet] = packet_checksum;
        return;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint packet_first = output_first + packet * PRT_INTERVAL;
    if (prt_packet_split_count == 1u) {
        for (uint slot = 0u; slot < 16u; ++slot) {
            uint scan = prt_event_rows ? (lane << 4u) + slot : lane + 32u * slot;
            uint contribution = prt_plain_packet_owner2
                ? partials[scan]
                : uint(atomic_load_explicit(atomic_partials + scan, memory_order_relaxed));
            if (prt_register_reduction)
                contribution += dense_accum[slot];
            if (prt_event_rows && event_rows_ready)
                contribution += event_owned[slot >> 2u][slot & 3u];
            output[packet_first + scan] += contribution;
        }
    } else {
        device atomic_uint *atomic_output = reinterpret_cast<device atomic_uint *>(output);
        for (uint slot = 0u; slot < 16u; ++slot) {
            uint scan = prt_event_rows ? (lane << 4u) + slot : lane + 32u * slot;
            uint contribution = prt_plain_packet_owner2
                ? partials[scan]
                : uint(atomic_load_explicit(atomic_partials + scan, memory_order_relaxed));
            if (prt_register_reduction)
                contribution += dense_accum[slot];
            if (prt_event_rows && event_rows_ready)
                contribution += event_owned[slot >> 2u][slot & 3u];
            atomic_fetch_add_explicit(
                atomic_output + packet_first + scan,
                contribution,
                memory_order_relaxed);
        }
    }
}

// Dense-only packet owner used with the independent sparse scatter. Each lane
// advances through its strided selected ordinals until it owns a dense stream,
// so sparse and zero gaps cannot force an otherwise idle lane through 256 pair
// iterations. One stream per lane keeps the experimental register footprint
// bounded while preserving the packet-local exact integer reduction.
kernel void paired_runtime_tans_detector_dense_compaction(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device const uint *selected [[buffer(4)]],
    device const int *coefficients [[buffer(5)]],
    device uint *output [[buffer(6)]],
    device atomic_uint *failure [[buffer(7)]],
    constant uint *p [[buffer(8)]],
    uint packet_group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    uint simd = thread_id >> 5u;
    uint lane = thread_id & 31u;
    uint pixels = p[0], packets = p[1], changed = p[2];
    uint output_first = p[3], payload_bytes = p[4];
    uint packet = packet_group * 4u + simd;
    if (packet >= packets) return;
    threadgroup uint partial_storage[4u * PRT_INTERVAL];
    threadgroup uint *partials = partial_storage + simd * PRT_INTERVAL;
    threadgroup atomic_uint *atomic_partials =
        reinterpret_cast<threadgroup atomic_uint *>(partials);
    for (uint scan = lane; scan < PRT_INTERVAL; scan += 32u) {
        if (prt_plain_scratch) partials[scan] = 0u;
        else atomic_store_explicit(atomic_partials + scan, 0u, memory_order_relaxed);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint ordinal = lane;
    while (true) {
        bool dense = false, entropy = false;
        uint first = 0u, end = 0u, mode = 253u, constant_value = 0u;
        int coefficient = 0;
        PRTFastReader reader;
        while (ordinal < changed && !dense) {
            uint candidate = ordinal;
            ordinal += 32u;
            uint pixel = selected[candidate];
            if (pixel >= pixels) {
                atomic_store_explicit(failure, 70u, memory_order_relaxed);
                continue;
            }
            uint stream = packet * pixels + pixel;
            uint total_streams = pixels * packets;
            first = prt_source_offset(offsets, stream, total_streams);
            end = prt_source_offset(offsets, stream + 1u, total_streams);
            mode = uint(modes[stream]);
            if (end < first || end > payload_bytes) {
                atomic_store_explicit(failure, 71u, memory_order_relaxed);
                continue;
            }
            if (mode == 252u || mode == 253u) continue;
            coefficient = coefficients[candidate];
            entropy = prt_entropy_mode(mode);
            if (mode == 255u) {
                if (end - first != 2u) {
                    atomic_store_explicit(failure, 72u, memory_order_relaxed);
                    continue;
                }
                constant_value = uint(payload[first])
                    | (uint(payload[first + 1u]) << 8u);
            } else if (mode == 254u) {
                if (end - first != 2u * PRT_INTERVAL) {
                    atomic_store_explicit(failure, 72u, memory_order_relaxed);
                    continue;
                }
            } else if (entropy) {
                device const uint *table = decoding + prt_entropy_model(mode) * PRT_STATES;
                if (!prt_fast_begin(payload, first, end, payload_bytes, table, reader)) {
                    atomic_store_explicit(failure, 72u, memory_order_relaxed);
                    continue;
                }
            } else {
                atomic_store_explicit(failure, 72u, memory_order_relaxed);
                continue;
            }
            dense = true;
        }
        if (!simd_any(dense)) break;
        for (uint pair = 0u; pair < PRT_INTERVAL / 2u; ++pair) {
            uint a = 0u, b = 0u;
            if (dense) {
                if (entropy) {
                    if (!prt_fast_pair(reader, a, b)) {
                        atomic_store_explicit(failure, 73u, memory_order_relaxed);
                        dense = false;
                    }
                } else if (mode == 254u) {
                    uint at = first + 4u * pair;
                    a = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                    b = uint(payload[at + 2u]) | (uint(payload[at + 3u]) << 8u);
                } else {
                    a = b = constant_value;
                }
            }
            int sum_a = simd_sum(int(a) * coefficient);
            int sum_b = simd_sum(int(b) * coefficient);
            if (lane == 0u) {
                if (prt_plain_scratch) {
                    partials[2u * pair] += uint(sum_a);
                    partials[2u * pair + 1u] += uint(sum_b);
                } else {
                    atomic_fetch_add_explicit(
                        atomic_partials + 2u * pair, uint(sum_a), memory_order_relaxed);
                    atomic_fetch_add_explicit(
                        atomic_partials + 2u * pair + 1u, uint(sum_b),
                        memory_order_relaxed);
                }
            }
        }
        if (entropy && dense && !prt_fast_finished(reader))
            atomic_store_explicit(failure, 74u, memory_order_relaxed);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint packet_first = output_first + packet * PRT_INTERVAL;
    for (uint scan = lane; scan < PRT_INTERVAL; scan += 32u) {
        uint value = prt_plain_scratch
            ? partials[scan]
            : atomic_load_explicit(atomic_partials + scan, memory_order_relaxed);
        output[packet_first + scan] += value;
    }
}

// Four SIMD groups cooperate on one packet while retaining the same 8 KiB
// threadgroup footprint as the four-packet owner. Each SIMD owns disjoint
// detector ordinals and a disjoint 512-scan partial; the whole threadgroup
// publishes each exact output once after reducing those four partials.
kernel void paired_runtime_tans_detector_cooperative(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device const uint *selected [[buffer(4)]],
    device const int *coefficients [[buffer(5)]],
    device uint *output [[buffer(6)]],
    device atomic_uint *failure [[buffer(7)]],
    constant uint *p [[buffer(8)]],
    uint packet [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    uint simd = thread_id >> 5u;
    uint lane = thread_id & 31u;
    uint pixels = p[0], packets = p[1], changed = p[2];
    uint output_first = p[3], payload_bytes = p[4];
    bool include_sparse = p[5] != 0u;
    if (packet >= packets) return;
    threadgroup uint partial_storage[4u * PRT_INTERVAL];
    threadgroup uint *partials = partial_storage + simd * PRT_INTERVAL;
    for (uint scan = lane; scan < PRT_INTERVAL; scan += 32u)
        partials[scan] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = simd * 32u; base < changed; base += 128u) {
        uint ordinal = base + lane;
        bool active = ordinal < changed;
        uint pixel = active ? selected[ordinal] : 0u;
        if (active && pixel >= pixels) {
            atomic_store_explicit(failure, 80u, memory_order_relaxed);
            active = false;
        }
        int coefficient = active ? coefficients[ordinal] : 0;
        uint stream = packet * pixels + pixel;
        uint total_streams = pixels * packets;
        uint first = active ? prt_source_offset(offsets, stream, total_streams) : 0u;
        uint end = active ? prt_source_offset(offsets, stream + 1u, total_streams) : 0u;
        uint mode = active ? uint(modes[stream]) : 253u;
        if (active && (end < first || end > payload_bytes)) {
            atomic_store_explicit(failure, 81u, memory_order_relaxed);
            active = false;
            mode = 253u;
        }
        bool sparse = include_sparse && active && mode == 252u;
        bool dense = active && mode != 252u && mode != 253u;
        uint constant_value = 0u;
        PRTFastReader reader;
        bool entropy = dense && prt_entropy_mode(mode);
        if (dense && mode == 255u) {
            if (end - first == 2u)
                constant_value = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
            else active = dense = false;
        } else if (dense && mode == 254u) {
            if (end - first != 2u * PRT_INTERVAL) active = dense = false;
        } else if (entropy) {
            device const uint *table = decoding + prt_entropy_model(mode) * PRT_STATES;
            if (!prt_fast_begin(payload, first, end, payload_bytes, table, reader))
                active = dense = entropy = false;
        } else if (dense) {
            active = dense = false;
        }
        if (!active && ordinal < changed)
            atomic_store_explicit(failure, 82u, memory_order_relaxed);

        if (simd_any(dense)) {
            for (uint pair = 0u; pair < PRT_INTERVAL / 2u; ++pair) {
                uint a = 0u, b = 0u;
                if (dense) {
                    if (entropy) {
                        if (!prt_fast_pair(reader, a, b)) {
                            atomic_store_explicit(failure, 83u, memory_order_relaxed);
                            dense = false;
                        }
                    } else if (mode == 254u) {
                        uint at = first + 4u * pair;
                        a = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                        b = uint(payload[at + 2u]) | (uint(payload[at + 3u]) << 8u);
                    } else {
                        a = b = constant_value;
                    }
                }
                uint sum_a = simd_sum(uint(int(a) * coefficient));
                uint sum_b = simd_sum(uint(int(b) * coefficient));
                if (lane == 0u) {
                    partials[2u * pair] += sum_a;
                    partials[2u * pair + 1u] += sum_b;
                }
            }
            if (entropy && dense && !prt_fast_finished(reader))
                atomic_store_explicit(failure, 84u, memory_order_relaxed);
        }
        if (sparse) {
            if (((end - first) & 1u) != 0u) {
                atomic_store_explicit(failure, 85u, memory_order_relaxed);
            } else {
                uint previous = 0u;
                bool has_previous = false;
                for (uint cursor = first; cursor < end; cursor += 2u) {
                    uint event = uint(payload[cursor])
                        | (uint(payload[cursor + 1u]) << 8u);
                    uint position = event >> 7u;
                    if (position >= PRT_INTERVAL
                        || (has_previous && position <= previous)) {
                        atomic_store_explicit(failure, 86u, memory_order_relaxed);
                        break;
                    }
                    uint contribution = uint(
                        int((event & 127u) + 1u) * coefficient);
                    atomic_fetch_add_explicit(
                        reinterpret_cast<threadgroup atomic_uint *>(partials) + position,
                        contribution, memory_order_relaxed);
                    previous = position;
                    has_previous = true;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint packet_first = output_first + packet * PRT_INTERVAL;
    for (uint scan = thread_id; scan < PRT_INTERVAL; scan += 128u) {
        uint value = partial_storage[scan]
            + partial_storage[PRT_INTERVAL + scan]
            + partial_storage[2u * PRT_INTERVAL + scan]
            + partial_storage[3u * PRT_INTERVAL + scan];
        output[packet_first + scan] += value;
    }
}

// High-parallelism detector delta for medium masks. One SIMD owns 64 changed
// detector pixels in one packet, decodes two independent streams per lane, and
// publishes an independent 512-scan partial.
// FC51: the index build fills leaf partials for packets p[5] ..< p[5] + p[6] into a
// batch-local field buffer; p[1] stays the source's packet count for stream addresses.
constant bool prt_partial_packet_batch_requested [[function_constant(51)]];
constant bool prt_partial_packet_batch = is_function_constant_defined(
    prt_partial_packet_batch_requested) ? prt_partial_packet_batch_requested : false;

kernel void paired_runtime_tans_detector_partials(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device const uint *selected [[buffer(4)]],
    device const int *coefficients [[buffer(5)]],
    device atomic_uint *partials [[buffer(6)]],
    device atomic_uint *failure [[buffer(7)]],
    constant uint *p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint pixels = p[0], packets = p[1], changed = p[2], groups = p[3];
    uint payload_bytes = p[4];
    uint first_packet = prt_partial_packet_batch ? p[5] : 0u;
    uint batch_packets = prt_partial_packet_batch ? p[6] : packets;
    if (first_packet + batch_packets > packets) {
        if (lane == 0u) atomic_store_explicit(failure, 87u, memory_order_relaxed);
        return;
    }
    if (group.x >= groups || group.y >= batch_packets) return;
    if (prt_partial_leaf_width != 16u && prt_partial_leaf_width != 32u
        && prt_partial_leaf_width != 64u) {
        if (lane == 0u) atomic_store_explicit(failure, 47u, memory_order_relaxed);
        return;
    }
    bool active[2], sparse[2], dense[2], entropy[2], compact[2];
    uint first[2], end[2], mode[2], constant_value[2];
    int coefficient[2];
    PRTFastReader reader[2];
    #pragma unroll
    for (uint j = 0u; j < 2u; ++j) {
        uint local = j * 32u + lane;
        uint ordinal = group.x * prt_partial_leaf_width + local;
        active[j] = local < prt_partial_leaf_width && ordinal < changed;
        uint pixel = active[j] ? selected[ordinal] : 0u;
        if (active[j] && pixel >= pixels) {
            atomic_store_explicit(failure, 40u, memory_order_relaxed);
            active[j] = false;
        }
        coefficient[j] = active[j] ? coefficients[ordinal] : 0;
        uint stream = (first_packet + group.y) * pixels + pixel;
        uint total_streams = pixels * packets;
        first[j] = active[j] ? prt_source_offset(offsets, stream, total_streams) : 0u;
        end[j] = active[j]
            ? prt_source_offset(offsets, stream + 1u, total_streams) : 0u;
        mode[j] = active[j] ? uint(modes[stream]) : 253u;
        if (active[j] && (end[j] < first[j] || end[j] > payload_bytes)) {
            atomic_store_explicit(failure, 41u, memory_order_relaxed);
            active[j] = false;
            mode[j] = 253u;
        }
        sparse[j] = active[j] && mode[j] == 252u;
        compact[j] = active[j] && mode[j] == 251u;
        dense[j] = active[j] && mode[j] != 252u && mode[j] != 253u && mode[j] != 251u;
        entropy[j] = dense[j] && prt_entropy_mode(mode[j]);
        constant_value[j] = 0u;
        if (dense[j] && mode[j] == 255u) {
            if (end[j] - first[j] == 2u)
                constant_value[j] = uint(payload[first[j]])
                    | (uint(payload[first[j] + 1u]) << 8u);
            else active[j] = dense[j] = false;
        } else if (dense[j] && mode[j] == 254u) {
            if (end[j] - first[j] != 2u * PRT_INTERVAL)
                active[j] = dense[j] = false;
        } else if (entropy[j]) {
            device const uint *table = decoding
                + prt_entropy_model(mode[j]) * PRT_STATES;
            if (!prt_fast_begin(
                payload, first[j], end[j], payload_bytes, table, reader[j]))
                active[j] = dense[j] = entropy[j] = false;
        } else if (dense[j]) {
            active[j] = dense[j] = false;
        }
        if (!active[j] && local < prt_partial_leaf_width && ordinal < changed)
            atomic_store_explicit(failure, 42u, memory_order_relaxed);
    }

    uint output_fields = prt_partial_output_fields == 0u
        ? groups : prt_partial_output_fields;
    if (output_fields < groups) {
        if (lane == 0u) atomic_store_explicit(failure, 48u, memory_order_relaxed);
        return;
    }
    uint partial_first = (group.y * output_fields + group.x) * PRT_INTERVAL;
    device uint *plain_partials = reinterpret_cast<device uint *>(partials);
    if (prt_partial_stores) {
        // The host normally clears this lazy, reusable buffer.  The direct
        // specialization clears only its owned field on-GPU, permitting the
        // host-side memset to be skipped without leaving stale scans behind.
        for (uint scan = lane; scan < PRT_INTERVAL; scan += 32u)
            plain_partials[partial_first + scan] = 0u;
        threadgroup_barrier(mem_flags::mem_device);
    }
    bool any_sparse = simd_any(sparse[0] || sparse[1]);
    if (simd_any(dense[0] || dense[1])) {
        for (uint pair = 0u; pair < PRT_INTERVAL / 2u; ++pair) {
            int sum_a = 0, sum_b = 0;
            #pragma unroll
            for (uint j = 0u; j < 2u; ++j) {
                uint a = 0u, b = 0u;
                if (dense[j]) {
                    if (entropy[j]) {
                        if (!prt_fast_pair(reader[j], a, b)) {
                            atomic_store_explicit(failure, 43u, memory_order_relaxed);
                            dense[j] = false;
                        }
                    } else if (mode[j] == 254u) {
                        uint at = first[j] + 4u * pair;
                        a = uint(payload[at]) | (uint(payload[at + 1u]) << 8u);
                        b = uint(payload[at + 2u]) | (uint(payload[at + 3u]) << 8u);
                    } else {
                        a = b = constant_value[j];
                    }
                }
                sum_a += int(a) * coefficient[j];
                sum_b += int(b) * coefficient[j];
            }
            sum_a = simd_sum(sum_a);
            sum_b = simd_sum(sum_b);
            if (lane == 0u) {
                if (prt_partial_stores) {
                    plain_partials[partial_first + 2u * pair] = uint(sum_a);
                    plain_partials[partial_first + 2u * pair + 1u] = uint(sum_b);
                } else {
                    atomic_fetch_add_explicit(
                        partials + partial_first + 2u * pair, uint(sum_a), memory_order_relaxed);
                    atomic_fetch_add_explicit(
                        partials + partial_first + 2u * pair + 1u, uint(sum_b),
                        memory_order_relaxed);
                }
            }
        }
        #pragma unroll
        for (uint j = 0u; j < 2u; ++j)
            if (entropy[j] && dense[j] && !prt_fast_finished(reader[j]))
                atomic_store_explicit(failure, 44u, memory_order_relaxed);
    }
    if (prt_partial_stores && any_sparse)
        threadgroup_barrier(mem_flags::mem_device);
    #pragma unroll
    for (uint j = 0u; j < 2u; ++j) {
        if (sparse[j]) {
            if (((end[j] - first[j]) & 1u) != 0u) {
                atomic_store_explicit(failure, 45u, memory_order_relaxed);
            } else {
                uint previous = 0u;
                bool has_previous = false;
                for (uint cursor = first[j]; cursor < end[j]; cursor += 2u) {
                    uint event = uint(payload[cursor])
                        | (uint(payload[cursor + 1u]) << 8u);
                    uint position = event >> 7u;
                    if (position >= PRT_INTERVAL || (has_previous && position <= previous)) {
                        atomic_store_explicit(failure, 46u, memory_order_relaxed);
                        break;
                    }
                    uint contribution = uint(int((event & 127u) + 1u) * coefficient[j]);
                    atomic_fetch_add_explicit(
                        partials + partial_first + position, contribution,
                        memory_order_relaxed);
                    previous = position;
                    has_previous = true;
                }
            }
        }
        if (compact[j]) {
            PRTCompactEvents events = {first[j], end[j], 0u, true};
            uint position = 0u, value = 0u;
            while (prt_compact_next(payload, events, position, value)) {
                atomic_fetch_add_explicit(
                    partials + partial_first + position, value * uint(coefficient[j]),
                    memory_order_relaxed);
            }
            if (!events.valid)
                atomic_store_explicit(failure, 49u, memory_order_relaxed);
        }
    }
}

kernel void paired_runtime_tans_detector_finish(
    device const uint *partials [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint *p [[buffer(2)]],
    uint scan [[thread_position_in_grid]]) {
    uint packets = p[0], groups = p[1];
    if (scan >= packets * PRT_INTERVAL) return;
    uint packet = scan / PRT_INTERVAL;
    uint local = scan % PRT_INTERVAL;
    uint value = output[scan];
    for (uint group = 0u; group < groups; ++group)
        value += partials[(packet * groups + group) * PRT_INTERVAL + local];
    output[scan] = value;
}

// Exact bounded polar-field index. Existing leaves use packet-major layout
// ((packet * 576 + leaf) * 512 + scan). The output uses the corresponding
// 612-field stride and contains copied leaves followed by 36 exact roots.
kernel void paired_runtime_tans_polar_roots(
    device const uint *leaves [[buffer(0)]],
    device uint *fields [[buffer(1)]],
    device atomic_uint *failure [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint job [[thread_position_in_grid]]) {
    uint packets = p[0];
    if (job >= packets * 612u * PRT_INTERVAL) return;
    uint scan = job % PRT_INTERVAL;
    uint field = (job / PRT_INTERVAL) % 612u;
    uint packet = job / (612u * PRT_INTERVAL);
    if (packet >= packets) {
        atomic_store_explicit(failure, 90u, memory_order_relaxed);
        return;
    }
    uint output_at = (packet * 612u + field) * PRT_INTERVAL + scan;
    if (field < 576u) {
        fields[output_at] = leaves[(packet * 576u + field) * PRT_INTERVAL + scan];
        return;
    }
    uint root = field - 576u;
    uint value = 0u;
    for (uint leaf = 0u; leaf < 16u; ++leaf) {
        uint source_field = root * 16u + leaf;
        value += leaves[(packet * 576u + source_field) * PRT_INTERVAL + scan];
    }
    fields[output_at] = value;
}

// Append roots to a packet-major field buffer whose leaf region was written
// directly by the leaf-width-specialized detector-partials kernel.
kernel void paired_runtime_tans_polar_roots_in_place(
    device uint *fields [[buffer(0)]],
    device atomic_uint *failure [[buffer(1)]],
    constant uint *p [[buffer(2)]],
    uint job [[thread_position_in_grid]]) {
    uint packets = p[0], leaves = p[1], field_count = p[2];
    if (leaves == 0u || leaves > 4096u || (leaves & 15u) != 0u
        || field_count != leaves + leaves / 16u || field_count > 4352u) {
        if (job == 0u) atomic_store_explicit(failure, 103u, memory_order_relaxed);
        return;
    }
    uint roots = leaves / 16u;
    if (job >= packets * roots * PRT_INTERVAL) return;
    uint scan = job % PRT_INTERVAL;
    uint root = (job / PRT_INTERVAL) % roots;
    uint packet = job / (roots * PRT_INTERVAL);
    uint value = 0u;
    for (uint offset = 0u; offset < 16u; ++offset) {
        uint leaf = root * 16u + offset;
        value += fields[(packet * field_count + leaf) * PRT_INTERVAL + scan];
    }
    uint output_field = leaves + root;
    fields[(packet * field_count + output_field) * PRT_INTERVAL + scan] = value;
}

// One job per (field, packet). A tag's low six bits are width 0...32 and bit7
// means payload begins with one UInt32 minimum followed by packed deltas.
// sizes are measured in UInt32 words and may be CPU-prefixed into offsets.
kernel void paired_runtime_tans_polar_sizes(
    device const uint *fields [[buffer(0)]],
    device uchar *tags [[buffer(1)]],
    device uint *sizes [[buffer(2)]],
    device atomic_uint *failure [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint stream [[thread_position_in_grid]]) {
    uint packets = p[0], field_count = p[1];
    if (stream >= packets * field_count) return;
    if (field_count > 4352u) {
        atomic_store_explicit(failure, 91u, memory_order_relaxed);
        return;
    }
    uint packet = stream / field_count, field = stream - packet * field_count;
    uint first = (packet * field_count + field) * PRT_INTERVAL;
    uint lo = 0xffffffffu, hi = 0u;
    for (uint scan = 0u; scan < PRT_INTERVAL; ++scan) {
        uint value = fields[first + scan];
        lo = min(lo, value);
        hi = max(hi, value);
    }
    uint range = hi - lo;
    uint raw_width = hi == 0u ? 0u : 32u - clz(hi);
    uint delta_width = range == 0u ? 0u : 32u - clz(range);
    uint raw_words = (PRT_INTERVAL * raw_width + 31u) / 32u;
    uint delta_words = 1u + (PRT_INTERVAL * delta_width + 31u) / 32u;
    bool use_delta = delta_words < raw_words;
    tags[stream] = uchar((use_delta ? 128u : 0u)
        | (use_delta ? delta_width : raw_width));
    sizes[stream] = use_delta ? delta_words : raw_words;
}

kernel void paired_runtime_tans_polar_pack(
    device const uint *fields [[buffer(0)]],
    device const uchar *tags [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device uint *payload [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    constant uint *p [[buffer(5)]],
    uint stream [[thread_position_in_grid]]) {
    uint packets = p[0], field_count = p[1], payload_words = p[2];
    uint streams = packets * field_count;
    if (stream >= streams) return;
    uint begin = offsets[stream], end = offsets[stream + 1u];
    uint tag = uint(tags[stream]), width = tag & 63u;
    if (field_count > 4352u || (tag & 64u) != 0u || width > 32u
        || end < begin || end > payload_words) {
        atomic_store_explicit(failure, 92u, memory_order_relaxed);
        return;
    }
    uint expected = (tag & 128u ? 1u : 0u)
        + (PRT_INTERVAL * width + 31u) / 32u;
    if (end - begin != expected) {
        atomic_store_explicit(failure, 93u, memory_order_relaxed);
        return;
    }
    uint packet = stream / field_count, field = stream - packet * field_count;
    uint first = (packet * field_count + field) * PRT_INTERVAL;
    uint at = begin, base = 0u;
    if ((tag & 128u) != 0u) {
        base = 0xffffffffu;
        for (uint scan = 0u; scan < PRT_INTERVAL; ++scan)
            base = min(base, fields[first + scan]);
        payload[at++] = base;
    }
    if (width == 0u) return;
    ulong reservoir = 0u;
    uint available = 0u;
    for (uint scan = 0u; scan < PRT_INTERVAL; ++scan) {
        reservoir |= ulong(fields[first + scan] - base) << available;
        available += width;
        if (available >= 32u) {
            if (at >= end) {
                atomic_store_explicit(failure, 94u, memory_order_relaxed);
                return;
            }
            payload[at++] = uint(reservoir);
            reservoir >>= 32u;
            available -= 32u;
        }
    }
    if (available != 0u) {
        if (at >= end) {
            atomic_store_explicit(failure, 94u, memory_order_relaxed);
            return;
        }
        payload[at++] = uint(reservoir);
    }
    if (at != end) atomic_store_explicit(failure, 95u, memory_order_relaxed);
}

// One 128-thread group owns 128 consecutive scans within a 512-scan packet.
// The caller supplies dynamic threadgroup memory of selected_count*16 bytes.
kernel void paired_runtime_tans_polar_query(
    device const uint *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *tags [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device const int *coefficients [[buffer(4)]],
    device uint *output [[buffer(5)]],
    device atomic_uint *failure [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint packets = p[0], field_count = p[1], selected_count = p[2];
    uint payload_words = p[3], output_first = p[4];
    if (selected_count > field_count || field_count > 4352u) {
        if (lane == 0u) atomic_store_explicit(failure, 96u, memory_order_relaxed);
        return;
    }
    uint packet = group / 4u, quarter = group & 3u;
    if (packet >= packets) return;
    threadgroup uint *word_first = shared;
    threadgroup uint *bases = shared + selected_count;
    threadgroup uint *widths = shared + 2u * selected_count;
    threadgroup int *coefs = reinterpret_cast<threadgroup int *>(
        shared + 3u * selected_count);
    for (uint ordinal = lane; ordinal < selected_count; ordinal += 128u) {
        uint field = selected[ordinal];
        if (field >= field_count) {
            atomic_store_explicit(failure, 97u, memory_order_relaxed);
            field = 0u;
        }
        uint stream = packet * field_count + field;
        uint begin = offsets[stream], end = offsets[stream + 1u];
        uint tag = uint(tags[stream]), width = tag & 63u;
        uint base = 0u;
        if ((tag & 64u) != 0u || width > 32u || end < begin || end > payload_words) {
            atomic_store_explicit(failure, 98u, memory_order_relaxed);
            begin = end = 0u;
            width = 0u;
        } else if ((tag & 128u) != 0u) {
            if (begin >= end) {
                atomic_store_explicit(failure, 99u, memory_order_relaxed);
                begin = end = 0u;
                width = 0u;
            } else base = payload[begin++];
        }
        uint expected_words = (PRT_INTERVAL * width + 31u) / 32u;
        if (end - begin != expected_words) {
            atomic_store_explicit(failure, 100u, memory_order_relaxed);
            begin = end = 0u;
            width = 0u;
            base = 0u;
        }
        word_first[ordinal] = begin;
        bases[ordinal] = base;
        widths[ordinal] = width;
        coefs[ordinal] = coefficients[ordinal];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint local = quarter * 128u + lane;
    uint value = output[output_first + packet * PRT_INTERVAL + local];
    for (uint ordinal = 0u; ordinal < selected_count; ++ordinal) {
        uint width = widths[ordinal], field_value = bases[ordinal];
        if (width != 0u) {
            uint bit = local * width;
            uint at = word_first[ordinal] + (bit >> 5u);
            uint shift = bit & 31u;
            if (at >= payload_words) {
                atomic_store_explicit(failure, 101u, memory_order_relaxed);
                continue;
            }
            ulong words = ulong(payload[at]);
            if (shift + width > 32u) {
                if (at + 1u >= payload_words) {
                    atomic_store_explicit(failure, 102u, memory_order_relaxed);
                    continue;
                }
                words |= ulong(payload[at + 1u]) << 32u;
            }
            uint extracted = uint(words >> shift);
            field_value += width == 32u
                ? extracted : extracted & ((1u << width) - 1u);
        }
        value += uint(coefs[ordinal]) * field_value;
    }
    output[output_first + packet * PRT_INTERVAL + local] = value;
}

inline uint prt_polar_window_value(
    thread const uint *window, uint first_shift, uint width, uint sample) {
    uint bit = first_shift + sample * width;
    uint word = bit >> 5u, shift = bit & 31u;
    ulong words = ulong(window[word]);
    if (shift + width > 32u) words |= ulong(window[word + 1u]) << 32u;
    uint extracted = uint(words >> shift);
    return width == 32u ? extracted : extracted & ((1u << width) - 1u);
}

// Experimental scan-axis variant. One group stages field metadata once, then
// each lane applies the selected fields to four scan outputs. The contiguous
// specialization reuses the packed word window for four adjacent scans.
// Striped variants reassociate only modulo-UInt32 additions; no count is
// dropped, clipped, or converted, and the stripe-1 path retains the baseline
// accumulation order.
kernel void paired_runtime_tans_polar_query_scan512(
    device const uint *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *tags [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device const int *coefficients [[buffer(4)]],
    device uint *output [[buffer(5)]],
    device atomic_uint *failure [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint logical_packet [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint packets = p[0], field_count = p[1], selected_count = p[2];
    uint payload_words = p[3], output_first = p[4];
    if (selected_count > field_count || field_count > 4352u) {
        if (lane == 0u) atomic_store_explicit(failure, 96u, memory_order_relaxed);
        return;
    }
    if (prt_polar_scan512_stripes != 1u && prt_polar_scan512_stripes != 2u
        && prt_polar_scan512_stripes != 4u && prt_polar_scan512_stripes != 8u) {
        if (lane == 0u) atomic_store_explicit(failure, 104u, memory_order_relaxed);
        return;
    }
    uint packet_stride = p[5] == 0u ? 1u : p[5];
    uint packet_phase = p[6];
    if (packet_phase >= packet_stride) {
        if (lane == 0u) atomic_store_explicit(failure, 105u, memory_order_relaxed);
        return;
    }
    uint packet = packet_phase + logical_packet * packet_stride;
    if (packet >= packets) return;
    threadgroup uint *word_first = shared;
    threadgroup uint *bases = shared + selected_count;
    threadgroup uint *widths = shared + 2u * selected_count;
    threadgroup int *coefs = reinterpret_cast<threadgroup int *>(
        shared + 3u * selected_count);
    for (uint ordinal = lane; ordinal < selected_count; ordinal += 128u) {
        uint field = selected[ordinal];
        if (field >= field_count) {
            atomic_store_explicit(failure, 97u, memory_order_relaxed);
            field = 0u;
        }
        uint stream = packet * field_count + field;
        uint begin = offsets[stream], end = offsets[stream + 1u];
        uint tag = uint(tags[stream]), width = tag & 63u;
        uint base = 0u;
        if ((tag & 64u) != 0u || width > 32u || end < begin || end > payload_words) {
            atomic_store_explicit(failure, 98u, memory_order_relaxed);
            begin = end = 0u;
            width = 0u;
        } else if ((tag & 128u) != 0u) {
            if (begin >= end) {
                atomic_store_explicit(failure, 99u, memory_order_relaxed);
                begin = end = 0u;
                width = 0u;
            } else base = payload[begin++];
        }
        uint expected_words = (PRT_INTERVAL * width + 31u) / 32u;
        if (end - begin != expected_words) {
            atomic_store_explicit(failure, 100u, memory_order_relaxed);
            begin = end = 0u;
            width = 0u;
            base = 0u;
        }
        word_first[ordinal] = begin;
        bases[ordinal] = base;
        widths[ordinal] = width;
        coefs[ordinal] = coefficients[ordinal];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local0 = prt_polar_scan512_contiguous_quad ? lane * 4u : lane;
    uint local1 = local0 + (prt_polar_scan512_contiguous_quad ? 1u : 128u);
    uint local2 = local0 + (prt_polar_scan512_contiguous_quad ? 2u : 256u);
    uint local3 = local0 + (prt_polar_scan512_contiguous_quad ? 3u : 384u);
    uint outputBase = output_first + packet * PRT_INTERVAL;
    uint4 values = uint4(
        output[outputBase + local0], output[outputBase + local1],
        output[outputBase + local2], output[outputBase + local3]);
    uint4 accumulators[8];
    for (uint stripe = 0u; stripe < 8u; ++stripe)
        accumulators[stripe] = uint4(0u);
    for (uint first_ordinal = 0u; first_ordinal < selected_count;
        first_ordinal += prt_polar_scan512_stripes) {
        #pragma unroll
        for (uint stripe = 0u; stripe < prt_polar_scan512_stripes; ++stripe) {
        uint ordinal = first_ordinal + stripe;
        if (ordinal >= selected_count) continue;
        uint width = widths[ordinal];
        uint base = bases[ordinal];
        int coefficient = coefs[ordinal];
        uint field0 = base, field1 = base, field2 = base, field3 = base;
        bool valid0 = true, valid1 = true, valid2 = true, valid3 = true;
        if (prt_polar_scan512_contiguous_quad) {
            uint first_bit = local0 * width;
            uint first_word = first_bit >> 5u;
            uint first_shift = first_bit & 31u;
            uint window_words = (first_shift + 4u * width + 31u) >> 5u;
            uint window[5] = {0u, 0u, 0u, 0u, 0u};
            bool window_valid = true;
            for (uint word = 0u; word < window_words; ++word) {
                uint at = word_first[ordinal] + first_word + word;
                if (at >= payload_words) {
                    atomic_store_explicit(failure, 101u, memory_order_relaxed);
                    window_valid = false;
                } else {
                    window[word] = payload[at];
                }
            }
            if (width != 0u && window_valid) {
                field0 += prt_polar_window_value(window, first_shift, width, 0u);
                field1 += prt_polar_window_value(window, first_shift, width, 1u);
                field2 += prt_polar_window_value(window, first_shift, width, 2u);
                field3 += prt_polar_window_value(window, first_shift, width, 3u);
            }
            valid0 = window_valid;
            valid1 = window_valid;
            valid2 = window_valid;
            valid3 = window_valid;
        } else if (width != 0u) {
            uint bit0 = local0 * width, bit1 = local1 * width;
            uint bit2 = local2 * width, bit3 = local3 * width;
            uint at0 = word_first[ordinal] + (bit0 >> 5u);
            uint at1 = word_first[ordinal] + (bit1 >> 5u);
            uint at2 = word_first[ordinal] + (bit2 >> 5u);
            uint at3 = word_first[ordinal] + (bit3 >> 5u);
            uint shift0 = bit0 & 31u, shift1 = bit1 & 31u;
            uint shift2 = bit2 & 31u, shift3 = bit3 & 31u;
            if (at0 >= payload_words) {
                atomic_store_explicit(failure, 101u, memory_order_relaxed);
                valid0 = false;
            }
            if (at1 >= payload_words) {
                atomic_store_explicit(failure, 101u, memory_order_relaxed);
                valid1 = false;
            }
            if (at2 >= payload_words) {
                atomic_store_explicit(failure, 101u, memory_order_relaxed);
                valid2 = false;
            }
            if (at3 >= payload_words) {
                atomic_store_explicit(failure, 101u, memory_order_relaxed);
                valid3 = false;
            }
            ulong words0 = valid0 ? ulong(payload[at0]) : 0u;
            ulong words1 = valid1 ? ulong(payload[at1]) : 0u;
            ulong words2 = valid2 ? ulong(payload[at2]) : 0u;
            ulong words3 = valid3 ? ulong(payload[at3]) : 0u;
            if (valid0 && shift0 + width > 32u) {
                if (at0 + 1u >= payload_words) {
                    atomic_store_explicit(failure, 102u, memory_order_relaxed);
                    valid0 = false;
                } else words0 |= ulong(payload[at0 + 1u]) << 32u;
            }
            if (valid1 && shift1 + width > 32u) {
                if (at1 + 1u >= payload_words) {
                    atomic_store_explicit(failure, 102u, memory_order_relaxed);
                    valid1 = false;
                } else words1 |= ulong(payload[at1 + 1u]) << 32u;
            }
            if (valid2 && shift2 + width > 32u) {
                if (at2 + 1u >= payload_words) {
                    atomic_store_explicit(failure, 102u, memory_order_relaxed);
                    valid2 = false;
                } else words2 |= ulong(payload[at2 + 1u]) << 32u;
            }
            if (valid3 && shift3 + width > 32u) {
                if (at3 + 1u >= payload_words) {
                    atomic_store_explicit(failure, 102u, memory_order_relaxed);
                    valid3 = false;
                } else words3 |= ulong(payload[at3 + 1u]) << 32u;
            }
            if (valid0) {
                uint extracted = uint(words0 >> shift0);
                field0 += width == 32u ? extracted : extracted & ((1u << width) - 1u);
            }
            if (valid1) {
                uint extracted = uint(words1 >> shift1);
                field1 += width == 32u ? extracted : extracted & ((1u << width) - 1u);
            }
            if (valid2) {
                uint extracted = uint(words2 >> shift2);
                field2 += width == 32u ? extracted : extracted & ((1u << width) - 1u);
            }
            if (valid3) {
                uint extracted = uint(words3 >> shift3);
                field3 += width == 32u ? extracted : extracted & ((1u << width) - 1u);
            }
        }
        uint4 field_values = uint4(field0, field1, field2, field3);
        if (!valid0) field_values.x = 0u;
        if (!valid1) field_values.y = 0u;
        if (!valid2) field_values.z = 0u;
        if (!valid3) field_values.w = 0u;
        accumulators[stripe] += uint(coefficient) * field_values;
        }
    }
    #pragma unroll
    for (uint stripe = 0u; stripe < prt_polar_scan512_stripes; ++stripe)
        values += accumulators[stripe];
    output[outputBase + local0] = values.x;
    output[outputBase + local1] = values.y;
    output[outputBase + local2] = values.z;
    output[outputBase + local3] = values.w;
}

// Experimental field-axis query. Four SIMD groups reduce disjoint subsets of
// selected fields for each 32-scan tile, shortening the per-lane field loop
// while keeping every scan's signed UInt32 sum exact.
// Packet-major exact polar query. One thread owns one 512-scan packet and walks
// every selected field sequentially: each field's fixed-width packed values are
// unpacked in scan order with a 64-bit bit buffer and accumulated into a
// thread-local row, which is added to the output once. This replaces the
// scan-major access pattern (every scan position touching every field) without
// changing field values, coefficients, or wraparound semantics.
kernel void paired_runtime_tans_polar_query_packet_major(
    device const uint *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *tags [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device const int *coefficients [[buffer(4)]],
    device uint *output [[buffer(5)]],
    device atomic_uint *failure [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    uint packet [[thread_position_in_grid]]) {
    uint packets = p[0], field_count = p[1], selected_count = p[2];
    uint payload_words = p[3], output_first = p[4];
    if (selected_count > field_count || field_count > 4352u) {
        atomic_store_explicit(failure, 96u, memory_order_relaxed);
        return;
    }
    if (packet >= packets) return;
    uint row[512];
    for (uint scan = 0u; scan < PRT_INTERVAL; ++scan)
        row[scan] = 0u;
    for (uint ordinal = 0u; ordinal < selected_count; ++ordinal) {
        uint field = selected[ordinal];
        if (field >= field_count) {
            atomic_store_explicit(failure, 97u, memory_order_relaxed);
            continue;
        }
        uint stream = packet * field_count + field;
        uint begin = offsets[stream], end = offsets[stream + 1u];
        uint tag = uint(tags[stream]), width = tag & 63u;
        if ((tag & 64u) != 0u || width > 32u || end < begin || end > payload_words) {
            atomic_store_explicit(failure, 98u, memory_order_relaxed);
            continue;
        }
        uint base = 0u;
        if ((tag & 128u) != 0u) {
            if (begin >= end) {
                atomic_store_explicit(failure, 99u, memory_order_relaxed);
                continue;
            }
            base = payload[begin++];
        }
        if (end - begin != (PRT_INTERVAL * width + 31u) / 32u) {
            atomic_store_explicit(failure, 100u, memory_order_relaxed);
            continue;
        }
        uint coefficient = uint(coefficients[ordinal]);
        uint scaled_base = coefficient * base;
        if (width == 0u) {
            for (uint scan = 0u; scan < PRT_INTERVAL; ++scan)
                row[scan] += scaled_base;
            continue;
        }
        uint mask = width == 32u ? 0xffffffffu : ((1u << width) - 1u);
        ulong buffer = 0ul;
        uint available = 0u;
        uint at = begin;
        for (uint scan = 0u; scan < PRT_INTERVAL; ++scan) {
            if (available < width) {
                buffer |= ulong(payload[at++]) << available;
                available += 32u;
            }
            uint value = uint(buffer) & mask;
            buffer >>= width;
            available -= width;
            row[scan] += scaled_base + coefficient * value;
        }
    }
    uint first = output_first + packet * PRT_INTERVAL;
    for (uint scan = 0u; scan < PRT_INTERVAL; ++scan)
        output[first + scan] += row[scan];
}

kernel void paired_runtime_tans_polar_query_field4(
    device const uint *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *tags [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device const int *coefficients [[buffer(4)]],
    device uint *output [[buffer(5)]],
    device atomic_uint *failure [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    threadgroup uint *shared [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    uint simd = thread_id >> 5u;
    uint lane = thread_id & 31u;
    uint packets = p[0], field_count = p[1], selected_count = p[2];
    uint payload_words = p[3], output_first = p[4];
    if (selected_count > field_count || field_count > 4352u) {
        if (thread_id == 0u) atomic_store_explicit(failure, 96u, memory_order_relaxed);
        return;
    }
    uint packet = group / 16u, tile = group & 15u;
    if (packet >= packets) return;

    threadgroup uint *word_first = shared;
    threadgroup uint *bases = shared + selected_count;
    threadgroup uint *widths = shared + 2u * selected_count;
    threadgroup int *coefs = reinterpret_cast<threadgroup int *>(
        shared + 3u * selected_count);
    threadgroup uint *partial_sums = shared + 4u * selected_count;
    for (uint ordinal = thread_id; ordinal < selected_count; ordinal += 128u) {
        uint field = selected[ordinal];
        if (field >= field_count) {
            atomic_store_explicit(failure, 97u, memory_order_relaxed);
            field = 0u;
        }
        uint stream = packet * field_count + field;
        uint begin = offsets[stream], end = offsets[stream + 1u];
        uint tag = uint(tags[stream]), width = tag & 63u;
        uint base = 0u;
        if ((tag & 64u) != 0u || width > 32u || end < begin || end > payload_words) {
            atomic_store_explicit(failure, 98u, memory_order_relaxed);
            begin = end = 0u;
            width = 0u;
        } else if ((tag & 128u) != 0u) {
            if (begin >= end) {
                atomic_store_explicit(failure, 99u, memory_order_relaxed);
                begin = end = 0u;
                width = 0u;
            } else base = payload[begin++];
        }
        uint expected_words = (PRT_INTERVAL * width + 31u) / 32u;
        if (end - begin != expected_words) {
            atomic_store_explicit(failure, 100u, memory_order_relaxed);
            begin = end = 0u;
            width = 0u;
            base = 0u;
        }
        word_first[ordinal] = begin;
        bases[ordinal] = base;
        widths[ordinal] = width;
        coefs[ordinal] = coefficients[ordinal];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local = tile * 32u + lane;
    uint partial = 0u;
    for (uint ordinal = simd; ordinal < selected_count; ordinal += 4u) {
        uint width = widths[ordinal];
        uint field_value = bases[ordinal];
        if (width != 0u) {
            uint bit = local * width;
            uint at = word_first[ordinal] + (bit >> 5u);
            uint shift = bit & 31u;
            if (at >= payload_words) {
                atomic_store_explicit(failure, 101u, memory_order_relaxed);
                continue;
            }
            ulong words = ulong(payload[at]);
            if (shift + width > 32u) {
                if (at + 1u >= payload_words) {
                    atomic_store_explicit(failure, 102u, memory_order_relaxed);
                    continue;
                }
                words |= ulong(payload[at + 1u]) << 32u;
            }
            uint extracted = uint(words >> shift);
            field_value += width == 32u
                ? extracted : extracted & ((1u << width) - 1u);
        }
        partial += uint(coefs[ordinal]) * field_value;
    }
    partial_sums[simd * 32u + lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        uint total = partial_sums[lane]
            + partial_sums[32u + lane]
            + partial_sums[64u + lane]
            + partial_sums[96u + lane];
        uint at = output_first + packet * PRT_INTERVAL + local;
        output[at] += total;
    }
}
