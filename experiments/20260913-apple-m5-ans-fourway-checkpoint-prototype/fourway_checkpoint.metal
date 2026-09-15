#include <metal_stdlib>
using namespace metal;

constant uint FW_STATE_COUNT = 1024u;
constant uint FW_INTERVAL = 512u;
constant uint FW_SEGMENT_PAIRS = 64u;
constant uint FW_SEGMENT_COUNT = 4u;
constant uint FW_CHECKPOINT_COUNT = FW_SEGMENT_COUNT - 1u;
constant uint FW_CHECKPOINT_BYTES = 3u;
constant uint FW_CHECKPOINT_RECORD_BYTES = FW_CHECKPOINT_COUNT * FW_CHECKPOINT_BYTES;

struct FWReverseReader {
    device const uchar *payload;
    uint body_first;
    uint cursor;
    uint last;
    uint reservoir;
    uint available;
    uint remaining;
    uint last_bits;
    bool valid;
};

inline FWReverseReader fw_reverse_reader(
    device const uchar *payload,
    uint body_first,
    uint end,
    uint meaningful_bits,
    uint last_bits) {
    FWReverseReader reader;
    reader.payload = payload;
    reader.body_first = body_first;
    reader.cursor = end;
    reader.last = end - 1u;
    reader.reservoir = 0u;
    reader.available = 0u;
    reader.remaining = meaningful_bits;
    reader.last_bits = last_bits;
    reader.valid = true;
    return reader;
}

inline uint fw_reverse_pop(thread FWReverseReader &reader, uint count) {
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
        uint bit_count = reader.cursor == reader.last ? reader.last_bits : 8u;
        uint value = uint(reader.payload[reader.cursor]) & ((1u << bit_count) - 1u);
        reader.reservoir = value | (reader.reservoir << bit_count);
        reader.available += bit_count;
    }
    reader.remaining -= count;
    reader.available -= count;
    uint value = count == 0u
        ? 0u : (reader.reservoir >> reader.available) & ((1u << count) - 1u);
    reader.reservoir &= reader.available == 0u
        ? 0u : (1u << reader.available) - 1u;
    return value;
}

inline bool fw_entropy_begin(
    device const uchar *payload,
    uint first,
    uint end,
    uint payload_bytes,
    thread uint &state,
    thread uint &body_first,
    thread uint &meaningful_bits,
    thread uint &last_bits,
    thread FWReverseReader &reader) {
    if (end < first || end > payload_bytes || end - first < 3u) return false;
    uint header = uint(payload[first]) | (uint(payload[first + 1u]) << 8u);
    uint tail = header & 7u;
    state = header >> 6u;
    body_first = first + 2u;
    uint body_bytes = end - body_first;
    if (((header >> 3u) & 7u) != 0u || state >= FW_STATE_COUNT
        || (tail != 0u && body_bytes == 0u)) return false;
    last_bits = tail == 0u ? 8u : tail;
    if (tail != 0u && (uint(payload[end - 1u]) >> tail) != 0u) return false;
    meaningful_bits = body_bytes * 8u - ((8u - tail) & 7u);
    reader = fw_reverse_reader(
        payload, body_first, end, meaningful_bits, last_bits);
    return true;
}

inline bool fw_reverse_reader_at_bit_position(
    device const uchar *payload,
    uint body_first,
    uint end,
    uint unread_bits,
    thread FWReverseReader &reader) {
    if (end < body_first) return false;
    uint total_bits = (end - body_first) * 8u;
    if (unread_bits > total_bits) return false;
    uint cursor = body_first + ((unread_bits + 7u) >> 3u);
    if (cursor > end) return false;
    reader.payload = payload;
    reader.body_first = body_first;
    reader.cursor = cursor;
    reader.last = cursor == body_first ? body_first - 1u : cursor - 1u;
    reader.reservoir = 0u;
    reader.available = 0u;
    reader.remaining = unread_bits;
    reader.last_bits = (unread_bits & 7u) == 0u ? 8u : (unread_bits & 7u);
    reader.valid = true;
    return true;
}

inline bool fw_decode_pair(
    device const uint *table,
    thread uint &state,
    thread FWReverseReader &reader,
    thread uint &first,
    thread uint &second) {
    uint code = table[state];
    uint pair = code & 4095u;
    uint bits = (code >> 12u) & 15u;
    state = (code >> 16u) + fw_reverse_pop(reader, bits);
    if (pair == 4095u) {
        uint word = fw_reverse_pop(reader, 13u);
        if (word < 4096u) {
            first = word & 63u;
            second = word >> 6u;
        } else if (word == 4096u) {
            first = fw_reverse_pop(reader, 16u);
            second = fw_reverse_pop(reader, 16u);
        } else {
            reader.valid = false;
            first = second = 0u;
        }
    } else {
        first = pair & 63u;
        second = pair >> 6u;
    }
    return reader.valid && state < FW_STATE_COUNT;
}

inline uint fw_checkpoint_word(
    device const uchar *checkpoints,
    uint selected_ordinal,
    uint checkpoint_index) {
    uint offset = selected_ordinal * FW_CHECKPOINT_RECORD_BYTES
        + checkpoint_index * FW_CHECKPOINT_BYTES;
    return uint(checkpoints[offset])
        | (uint(checkpoints[offset + 1u]) << 8u)
        | (uint(checkpoints[offset + 2u]) << 16u);
}

inline uint fw_source_offset(
    device const uint *offsets,
    uint stream,
    uint stream_count,
    bool compact_offsets) {
    if (!compact_offsets) return offsets[stream];
    uint base_count = (stream_count >> 5u) + 1u;
    device const uchar *bytes = reinterpret_cast<device const uchar *>(offsets);
    device const ushort *starts = reinterpret_cast<device const ushort *>(
        bytes + ulong(base_count) * sizeof(uint));
    return offsets[stream >> 5u] + uint(starts[stream]);
}

inline void fw_checkpoint_store(
    device uchar *checkpoints,
    uint selected_ordinal,
    uint checkpoint_index,
    uint state,
    uint unread_bits) {
    uint packed = state | (unread_bits << 10u);
    uint offset = selected_ordinal * FW_CHECKPOINT_RECORD_BYTES
        + checkpoint_index * FW_CHECKPOINT_BYTES;
    checkpoints[offset] = uchar(packed);
    checkpoints[offset + 1u] = uchar(packed >> 8u);
    checkpoints[offset + 2u] = uchar(packed >> 16u);
}

// status: 1 exact entropy checkpoints, 2 invalid selection, 3 unsupported mode,
// 4 malformed record, 5 decode/terminal failure, 6 fallback mode.
kernel void fw_inspect_selected_modes(
    device const uchar *modes [[buffer(0)]],
    device const uint *selected_streams [[buffer(1)]],
    device uchar *selected_modes [[buffer(2)]],
    constant uint *parameters [[buffer(3)]],
    uint selected_ordinal [[thread_position_in_grid]]) {
    if (selected_ordinal < parameters[0])
        selected_modes[selected_ordinal] = modes[selected_streams[selected_ordinal]];
}

kernel void fw_capture_selected_checkpoints(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device const uint *selected_streams [[buffer(4)]],
    device uchar *checkpoints [[buffer(5)]],
    device uint *record_status [[buffer(6)]],
    constant uint *parameters [[buffer(7)]],
    uint selected_ordinal [[thread_position_in_grid]]) {
    uint stream_count = parameters[0];
    uint selected_count = parameters[1];
    uint payload_bytes = parameters[2];
    bool compact_offsets = parameters[3] != 0u;
    uint count = parameters[4];
    if (selected_ordinal >= selected_count) return;
    for (uint byte = 0u; byte < FW_CHECKPOINT_RECORD_BYTES; ++byte)
        checkpoints[selected_ordinal * FW_CHECKPOINT_RECORD_BYTES + byte] = 0u;
    record_status[selected_ordinal] = 4u;

    uint stream = selected_streams[selected_ordinal];
    if (stream >= stream_count) {
        record_status[selected_ordinal] = 2u;
        return;
    }
    uint mode = uint(modes[stream]);
    if (mode == 252u || mode == 253u || mode == 254u || mode == 255u) {
        record_status[selected_ordinal] = 6u;
        return;
    }
    if (mode < 64u || mode >= 96u) {
        record_status[selected_ordinal] = 3u;
        return;
    }

    uint first = fw_source_offset(offsets, stream, stream_count, compact_offsets);
    uint end = fw_source_offset(offsets, stream + 1u, stream_count, compact_offsets);
    if (count == 0u || count > FW_INTERVAL || end < first
        || end > payload_bytes) {
        record_status[selected_ordinal] = 4u;
        return;
    }

    uint state, body_first, meaningful_bits, last_bits;
    FWReverseReader reader;
    if (!fw_entropy_begin(
            payload, first, end, payload_bytes, state, body_first,
            meaningful_bits, last_bits, reader)) {
        record_status[selected_ordinal] = 4u;
        return;
    }
    device const uint *table = decoding + (mode - 64u) * FW_STATE_COUNT;
    uint pair_count = (count + 1u) >> 1u;
    uint checkpoint_index = 0u;
    for (uint pair_index = 0u; pair_index < pair_count; ++pair_index) {
        uint first_value, second_value;
        if (!fw_decode_pair(table, state, reader, first_value, second_value)) {
            record_status[selected_ordinal] = 5u;
            return;
        }
        if ((pair_index + 1u) == (checkpoint_index + 1u) * FW_SEGMENT_PAIRS
            && checkpoint_index < FW_CHECKPOINT_COUNT) {
            if (reader.remaining >= 16384u) {
                record_status[selected_ordinal] = 5u;
                return;
            }
            fw_checkpoint_store(
                checkpoints, selected_ordinal, checkpoint_index,
                state, reader.remaining);
            ++checkpoint_index;
        }
        if ((count & 1u) != 0u && pair_index + 1u == pair_count
            && second_value != 0u) {
            record_status[selected_ordinal] = 5u;
            return;
        }
    }
    if (checkpoint_index != FW_CHECKPOINT_COUNT || state != 0u
        || reader.remaining != 0u || !reader.valid) {
        record_status[selected_ordinal] = 5u;
        return;
    }
    record_status[selected_ordinal] = 1u;
}

// Each thread writes one exact 64-pair value range. status: 1 exact segment,
// 2 missing/invalid checkpoint, 3 boundary mismatch, 4 malformed record,
// 5 pair decode failure.
kernel void fw_decode_fourway_segment_values(
    device const uchar *payload [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *modes [[buffer(2)]],
    device const uint *decoding [[buffer(3)]],
    device const uint *selected_streams [[buffer(4)]],
    device const uchar *checkpoints [[buffer(5)]],
    device const uint *record_status [[buffer(6)]],
    device ushort *decoded_values [[buffer(7)]],
    device uint *terminal_state [[buffer(8)]],
    device uint *terminal_unread_bits [[buffer(9)]],
    device uint *segment_status [[buffer(10)]],
    constant uint *parameters [[buffer(11)]],
    uint lane [[thread_position_in_grid]]) {
    uint stream_count = parameters[0];
    uint selected_count = parameters[1];
    uint payload_bytes = parameters[2];
    bool compact_offsets = parameters[3] != 0u;
    uint count = parameters[4];
    uint segment_item_count = selected_count * FW_SEGMENT_COUNT;
    if (lane >= segment_item_count) return;
    uint selected_ordinal = lane / FW_SEGMENT_COUNT;
    uint segment_index = lane % FW_SEGMENT_COUNT;
    terminal_state[lane] = 0u;
    terminal_unread_bits[lane] = 0u;
    segment_status[lane] = 0u;
    if (record_status[selected_ordinal] != 1u) {
        segment_status[lane] = 2u;
        return;
    }

    uint stream = selected_streams[selected_ordinal];
    if (stream >= stream_count) {
        segment_status[lane] = 4u;
        return;
    }
    uint mode = uint(modes[stream]);
    uint first = fw_source_offset(offsets, stream, stream_count, compact_offsets);
    uint end = fw_source_offset(offsets, stream + 1u, stream_count, compact_offsets);
    if (mode < 64u || mode >= 96u || count == 0u || count > FW_INTERVAL
        || end < first || end > payload_bytes) {
        segment_status[lane] = 4u;
        return;
    }

    uint initial_state, body_first, meaningful_bits, last_bits;
    FWReverseReader initial_reader;
    if (!fw_entropy_begin(
            payload, first, end, payload_bytes, initial_state, body_first,
            meaningful_bits, last_bits, initial_reader)) {
        segment_status[lane] = 4u;
        return;
    }
    uint state = initial_state;
    uint unread_bits = meaningful_bits;
    if (segment_index != 0u) {
        uint packed = fw_checkpoint_word(
            checkpoints, selected_ordinal, segment_index - 1u);
        state = packed & 1023u;
        unread_bits = packed >> 10u;
    }
    if (state >= FW_STATE_COUNT || unread_bits > meaningful_bits) {
        segment_status[lane] = 2u;
        return;
    }
    FWReverseReader reader;
    if (!fw_reverse_reader_at_bit_position(
            payload, body_first, end, unread_bits, reader)) {
        segment_status[lane] = 4u;
        return;
    }

    device const uint *table = decoding + (mode - 64u) * FW_STATE_COUNT;
    uint first_pair = segment_index * FW_SEGMENT_PAIRS;
    uint pair_count = (count + 1u) >> 1u;
    uint last_pair = min(first_pair + FW_SEGMENT_PAIRS, pair_count);
    for (uint pair_index = first_pair; pair_index < last_pair; ++pair_index) {
        uint first_value, second_value;
        if (!fw_decode_pair(table, state, reader, first_value, second_value)) {
            segment_status[lane] = 5u;
            return;
        }
        uint first_scan = pair_index * 2u;
        uint output_first = selected_ordinal * count + first_scan;
        if (first_scan < count) decoded_values[output_first] = ushort(first_value);
        if (first_scan + 1u < count)
            decoded_values[output_first + 1u] = ushort(second_value);
    }

    if (segment_index + 1u < FW_SEGMENT_COUNT) {
        uint expected = fw_checkpoint_word(
            checkpoints, selected_ordinal, segment_index);
        uint expected_state = expected & 1023u;
        uint expected_unread = expected >> 10u;
        if (state != expected_state || reader.remaining != expected_unread) {
            segment_status[lane] = 3u;
            return;
        }
    } else if (last_pair != pair_count || state != 0u || reader.remaining != 0u) {
        segment_status[lane] = 3u;
        return;
    }
    terminal_state[lane] = state;
    terminal_unread_bits[lane] = reader.remaining;
    segment_status[lane] = 1u;
}
