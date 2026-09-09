// Exact, bounded source112 decoding. No interaction-index data is an input.
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
constexpr int Q253 = 36864, D253 = 17466, C253 = 19398;

__device__ __forceinline__ bool bits253(const u32* payload, u32& cursor,
    u32 end, u64& reservoir, u32& available, u32& remaining, u32 count, u32& value) {
    if (count > remaining) return false;
    if (available < count) {
        if (cursor >= end) return false;
        reservoir |= u64(payload[cursor++]) << available;
        available += 32;
    }
    value = u32(reservoir) & ((1u << count) - 1);
    reservoir >>= count;
    available -= count;
    remaining -= count;
    return true;
}

extern "C" __global__ void decode_dense253(
    const u32* payload, u32 payload_words, const u32* offsets,
    const int* columns, const u8* ids, const u32* decoding, u32 models,
    int first_packet, int packets, u16* raw, u32* errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= packets * D253) return;
    const int rank = i % D253, packet = first_packet + i / D253;
    const int q = columns[rank], stream = packet * D253 + rank;
    const u8* lengths = reinterpret_cast<const u8*>(offsets + D253);
    u64 begin = offsets[stream >> 5];
    for (int j = stream & ~31; j < stream; ++j) begin += u32(lengths[j]) + 1;
    const u64 end64 = begin + u32(lengths[stream]) + 1;
    if (q < 0 || q >= Q253 || begin >= end64 || end64 > payload_words) {
        atomicOr(errors, 1u); return;
    }
    const int scan = (i / D253) * 512;
    const u32 model = ids[q];
    if (model == 255) {
        if (end64 - begin != 256) { atomicOr(errors, 1u); return; }
        for (int pair = 0; pair < 256; ++pair) {
            const u32 word = payload[begin + pair];
            raw[u64(scan + pair * 2) * Q253 + q] = word & 65535u;
            raw[u64(scan + pair * 2 + 1) * Q253 + q] = word >> 16;
        }
        return;
    }
    if (model >= models || model == 254) { atomicOr(errors, 2u); return; }
    u32 cursor = u32(begin), end = u32(end64), header = payload[cursor++];
    u32 state = header & 1023u, available = 22, remaining = 22 + 32 * (end - cursor);
    u64 reservoir = header >> 10;
    for (int pair_index = 0; pair_index < 256; ++pair_index) {
        if (state >= 1024) { atomicOr(errors, 2u); return; }
        const u32 code = decoding[model * 1024 + state], count = (code >> 12) & 15u;
        u32 low;
        if (!bits253(payload, cursor, end, reservoir, available, remaining, count, low)) {
            atomicOr(errors, 4u); return;
        }
        state = (code >> 16) + low;
        u32 pair = code & 4095u;
        if (pair == 4095u && !bits253(payload, cursor, end, reservoir, available, remaining, 12, pair)) {
            atomicOr(errors, 4u); return;
        }
        raw[u64(scan + pair_index * 2) * Q253 + q] = pair & 63u;
        raw[u64(scan + pair_index * 2 + 1) * Q253 + q] = pair >> 6;
    }
    if (state >= 1024) atomicOr(errors, 2u);
}

extern "C" __global__ void decode_sparse253(
    const u32* packed, u32 words, const u32* offsets, const int* columns,
    int first_packet, int packets, u16* raw, u32* errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= packets * C253) return;
    if (words < 4) { atomicOr(errors, 8u); return; }
    const u32 n = packed[0], pwords = packed[1], fwords = packed[2], rwords = packed[3];
    const u64 value_word = 4ull + pwords + fwords + rwords;
    if (pwords != (u64(n) * 9 + 31) / 32 || fwords != (u64(n) + 31) / 32
        || rwords != (u64(n) + 255) / 256 || value_word > words) {
        atomicOr(errors, 8u); return;
    }
    const int rank = i % C253, packet = first_packet + i / C253;
    const int stream = packet * C253 + rank, q = columns[rank];
    const u8* lengths = reinterpret_cast<const u8*>(offsets + C253 + 1);
    u64 begin = offsets[stream >> 5];
    for (int j = stream & ~31; j < stream; ++j) begin += lengths[j];
    const u64 end = begin + lengths[stream];
    if (q < 0 || q >= Q253 || begin > end || end > n || offsets[C253] != n) {
        atomicOr(errors, 16u); return;
    }
    const u32 *positions = packed + 4, *flags = positions + pwords, *ranks = flags + fwords;
    const u8* values = reinterpret_cast<const u8*>(packed + value_word);
    int previous = -1;
    for (u32 at = u32(begin); at < end; ++at) {
        const u64 bit = u64(at) * 9;
        const u32 word = u32(bit >> 5), shift = bit & 31u;
        u64 pair = positions[word];
        if (word + 1 < pwords) pair |= u64(positions[word + 1]) << 32;
        const u32 position = u32(pair >> shift) & 511u, flag = flags[at >> 5];
        u32 value = 1;
        if ((flag >> (at & 31)) & 1u) {
            u64 value_index = ranks[at >> 8];
            for (u32 j = (at >> 8) * 8; j < (at >> 5); ++j) value_index += __popc(flags[j]);
            value_index += __popc(flag & ((1u << (at & 31)) - 1));
            if (value_index >= (u64(words) - value_word) * 4) { atomicOr(errors, 32u); return; }
            value = values[value_index];
        }
        if (int(position) <= previous || value == 0 || value > 127) { atomicOr(errors, 32u); return; }
        previous = position;
        raw[u64((i / C253) * 512 + position) * Q253 + q] = value;
    }
}
