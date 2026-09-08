// Byte-normalized rANS recurrence retained from the detector-rANS experiments.
// Every stream is one detector column within a bounded block of scan positions.
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
static const u32 LOWER = 1u << 23;

struct Reader {
    __device__ Reader() = default;
    const u8* payload;
    const u16* symbols;
    const u16* cumulative;
    const u16* frequencies;
    u64 cursor;
    u64 end;
    u32 state;
    u32 first;
    u32 stop;
    u32 scale;
    bool raw;
    bool valid;

    __device__ Reader(
        const u8* bytes, const u64* offsets, const u32* model_ids,
        const u32* contexts, const u16* values, const u16* starts,
        const u16* weights, const u8* literals, u64 stream, u32 bits
    ) : payload(bytes), symbols(values), cumulative(starts),
        frequencies(weights), cursor(offsets[stream]), end(offsets[stream + 1]),
        state(LOWER), scale(bits), valid(true) {
        u32 model = model_ids[stream];
        first = contexts[model];
        stop = contexts[model + 1];
        raw = literals[model] != 0;
        if (!raw) {
            if (end - cursor < 4) { valid = false; return; }
            state = 0;
            for (u32 byte = 0; byte < 4; ++byte)
                state |= ((u32)payload[cursor++]) << (8 * byte);
            valid = state >= LOWER && state < (1u << 31);
        }
    }

    __device__ u16 next() {
        if (!valid) return 0;
        if (raw) {
            if (end - cursor < 2) { valid = false; return 0; }
            u16 value = payload[cursor] | ((u16)payload[cursor + 1] << 8);
            cursor += 2;
            return value;
        }
        u32 slot = state & ((1u << scale) - 1u);
        u32 left = first, right = stop;
        while (left + 1 < right) {
            u32 middle = left + (right - left) / 2;
            if (cumulative[middle] <= slot) left = middle;
            else right = middle;
        }
        // Host validation proves the context partitions every possible slot.
        u16 value = symbols[left];
        state = frequencies[left] * (state >> scale) + slot - cumulative[left];
        while (state < LOWER) {
            if (cursor >= end) { valid = false; return 0; }
            state = (state << 8) | payload[cursor++];
        }
        return value;
    }

    __device__ bool finished() const {
        return valid && cursor == end && (raw || state == LOWER);
    }
};

#define INPUTS const u8* payload, const u64* offsets, const u32* model_ids, \
    const u32* contexts, const u16* symbols, const u16* cumulative, \
    const u16* frequencies, const u8* literal
#define READER(stream) Reader reader(payload, offsets, model_ids, contexts, \
    symbols, cumulative, frequencies, literal, stream, scale)

extern "C" __global__ void ans_validate(
    INPUTS, u32* errors, u64 scan_count, u32 detector_count,
    u32 block_frames, u32 scale, u64 stream_count, u32 maximum_count
) {
    u64 stream = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= stream_count) return;
    u64 start = (stream / detector_count) * block_frames;
    u32 count = (u32)min((u64)block_frames, scan_count - start);
    READER(stream);
    for (u32 scan = 0; scan < count && reader.valid; ++scan)
        if (reader.next() > maximum_count) atomicOr(errors, 2u);
    if (!reader.finished()) atomicOr(errors, 1u);
}

extern "C" __global__ void ans_decode_block(
    INPUTS, u16* output, u32* errors, u64 block,
    u32 detector_count, u32 count, u32 scale
) {
    u32 pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= detector_count) return;
    READER(block * detector_count + pixel);
    for (u32 scan = 0; scan < count && reader.valid; ++scan)
        output[(u64)scan * detector_count + pixel] = reader.next();
    if (!reader.finished()) atomicOr(errors, 1u);
}

extern "C" __global__ void ans_diffraction(
    INPUTS, const u64* requested, u16* output, u32* errors,
    u64 request_count, u32 detector_count, u32 block_frames, u32 scale
) {
    u64 item = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 total = request_count * detector_count;
    if (item >= total) return;
    u64 request = item / detector_count;
    u32 pixel = item % detector_count;
    u64 scan = requested[request];
    u64 stream = (scan / block_frames) * detector_count + pixel;
    READER(stream);
    u16 value = 0;
    u32 within = scan % block_frames;
    if (reader.raw) {
        reader.cursor += (u64)within * 2;
        value = reader.next();
    } else {
        for (u32 step = 0; step <= within && reader.valid; ++step)
            value = reader.next();
    }
    if (!reader.valid) atomicOr(errors, 1u);
    else output[item] = value;
}

extern "C" __global__ void ans_detector_sum(
    INPUTS, const u8* mask, u64* output, u32* errors,
    u64 scan_count, u32 detector_count, u32 block_frames,
    u32 scale, u64 stream_count
) {
    u64 stream = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= stream_count) return;
    u32 pixel = stream % detector_count;
    if (!mask[pixel]) return;
    u64 start = (stream / detector_count) * block_frames;
    u32 count = (u32)min((u64)block_frames, scan_count - start);
    READER(stream);
    for (u32 scan = 0; scan < count && reader.valid; ++scan) {
        u16 value = reader.next();
        atomicAdd(output + start + scan, (u64)value);
    }
    if (!reader.finished()) atomicOr(errors, 1u);
}

extern "C" __global__ void ans_measure_packed(
    INPUTS, u8* widths, u64* lengths, u32* errors,
    u64 scan_count, u32 detector_count, u32 block_frames,
    u32 scale, u64 stream_count
) {
    u64 stream = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= stream_count) return;
    u64 start = (stream / detector_count) * block_frames;
    u32 count = (u32)min((u64)block_frames, scan_count - start);
    READER(stream);
    u32 combined = 0;
    for (u32 scan = 0; scan < count && reader.valid; ++scan) combined |= reader.next();
    u32 width = combined == 0 ? 0 : 32 - __clz(combined);
    widths[stream] = width;
    lengths[stream] = ((u64)count * width + 31) / 32;
    if (!reader.finished()) atomicOr(errors, 1u);
}

extern "C" __global__ void ans_write_packed(
    INPUTS, const u8* widths, const u64* word_offsets, u32* words,
    u32* errors, u64 scan_count, u32 detector_count,
    u32 block_frames, u32 scale, u64 stream_count
) {
    u64 stream = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= stream_count) return;
    u64 start = (stream / detector_count) * block_frames;
    u32 count = (u32)min((u64)block_frames, scan_count - start);
    u32 width = widths[stream];
    u64 cursor = word_offsets[stream], accumulator = 0;
    u32 used = 0;
    READER(stream);
    for (u32 scan = 0; scan < count && reader.valid; ++scan) {
        accumulator |= ((u64)reader.next()) << used;
        used += width;
        if (used >= 32) {
            words[cursor++] = (u32)accumulator;
            accumulator >>= 32;
            used -= 32;
        }
    }
    if (used) words[cursor++] = (u32)accumulator;
    if (!reader.finished() || cursor != word_offsets[stream + 1]) atomicOr(errors, 1u);
}

__device__ u16 packed_count(
    const u32* words, const u64* word_offsets, const u8* widths,
    u64 stream, u32 scan
) {
    u32 width = widths[stream];
    if (!width) return 0;
    u64 bit = (u64)scan * width;
    u64 word = word_offsets[stream] + bit / 32;
    u32 shift = bit % 32;
    u64 value = words[word];
    if (shift + width > 32) value |= (u64)words[word + 1] << 32;
    return (value >> shift) & ((1u << width) - 1u);
}

#define PACKED_INPUTS const u32* words, const u64* word_offsets, const u8* widths

extern "C" __global__ void packed_diffraction(
    PACKED_INPUTS, const u64* requested, u16* output,
    u64 request_count, u32 detector_count, u32 block_frames
) {
    u64 item = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= request_count * detector_count) return;
    u64 scan = requested[item / detector_count];
    u64 stream = (scan / block_frames) * detector_count + item % detector_count;
    output[item] = packed_count(words, word_offsets, widths, stream, scan % block_frames);
}

extern "C" __global__ void packed_decode_block(
    PACKED_INPUTS, u16* output, u64 block, u32 count, u32 detector_count
) {
    u64 item = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= (u64)count * detector_count) return;
    u64 stream = block * detector_count + item % detector_count;
    output[item] = packed_count(words, word_offsets, widths, stream, item / detector_count);
}

extern "C" __global__ void packed_detector_sum(
    PACKED_INPUTS, const u8* mask, u64* output, u64 scan_count,
    u32 detector_count, u32 block_frames, u64 stream_count
) {
    u64 stream = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= stream_count || !mask[stream % detector_count]) return;
    u64 start = (stream / detector_count) * block_frames;
    u32 count = (u32)min((u64)block_frames, scan_count - start);
    for (u32 scan = 0; scan < count; ++scan)
        atomicAdd(output + start + scan,
                  (u64)packed_count(words, word_offsets, widths, stream, scan));
}
