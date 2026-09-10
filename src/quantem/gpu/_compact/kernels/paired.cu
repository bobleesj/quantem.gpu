// Paired-count tANS resident layout with an adaptive polar interaction index.
//
// Every 512-scan stream of one detector pixel is coded as consecutive count pairs
// with a 1024-state tANS table chosen from 32 Poisson pair models. Streams that are
// empty, constant, sparse or incompressible keep exact alternative forms. The
// spatial index stores exact per-scan sums over radial-angular pixel groups so a
// detector mask decomposes into whole groups plus a few residual pixels. No kernel
// here changes a count; probabilities only change bytes.
//
// Stream modes (one byte per stream):
//   64 + m   paired tANS with model m (0..31); two-byte header: state<<6 | tail bits
//   252      sparse events: (scan<<7 | count-1) little-endian 16-bit, counts <= 128
//   253      all zero, no bytes
//   254      literal uint16 counts, two bytes per scan
//   255      constant nonzero count, one 16-bit value
//
// Grouped offsets: 32 streams share one 32-bit base followed by 32 relative 16-bit
// offsets (17 words per group); a final group holds the payload end.
//
// Descriptor rows (23 x u64 per chunk): 0 payload, 1 offset records, 2 models,
// 3 pair decoding table, 4 index words, 5 index starts, 6 index widths, 7 first scan,
// 8 scans, 9 acquisition, 10 validity mask, 12 interval.

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

static const u32 PM_MODELS = 32;
static const u32 PM_STATES = 1024;
static const u32 PM_SYMBOLS = 1089;      // 33 x 33 small pairs plus one escape
static const u32 PM_ESCAPE = 1088;
static const u32 PM_DESCRIPTOR = 23;

__device__ __forceinline__ u32 pm_raw(const void* data, u64 at, int itemsize) {
    return itemsize == 1 ? ((const u8*)data)[at] : ((const u16*)data)[at];
}

// ---------------------------------------------------------------------------
// Grouped stream offsets

__device__ __forceinline__ u32 pm_begin(const u32* records, u32 stream) {
    u32 group = stream >> 5, lane = stream & 31;
    return records[group * 17] + ((const u16*)(records + group * 17 + 1))[lane];
}
__device__ __forceinline__ u32 pm_end(const u32* records, u32 stream) {
    return (stream & 31) == 31 ? records[((stream >> 5) + 1) * 17] : pm_begin(records, stream + 1);
}
extern "C" __global__ void pm_pack_offsets(const u32* offsets, u32* records, u32* errors, u32 streams) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x, groups = (streams + 31) / 32;
    if (s < groups * 32) {
        u32 group = s >> 5, lane = s & 31, first = group * 32, at = min(s, streams), base = offsets[first];
        if (!lane) records[group * 17] = base;
        u32 delta = offsets[at] - base;
        if (delta > 65535u || offsets[at] < base) atomicOr(errors, 1u);
        ((u16*)(records + group * 17 + 1))[lane] = u16(delta);
    }
    if (!s) records[groups * 17] = offsets[streams];
}
extern "C" __global__ void pm_unpack_offsets(const u32* records, u32* offsets, u32 streams) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < streams) offsets[s] = pm_begin(records, s);
    else if (s == streams) offsets[s] = streams ? pm_end(records, streams - 1) : 0;
}

// ---------------------------------------------------------------------------
// Tables: frequencies (32 x 1089, (f << 16) | start) to encoding (state per rank)
// and decoding (pair | bits << 12 | base << 16) with the classical spread 643.

extern "C" __global__ void pm_tables(const u32* frequencies, u16* encoding, u32* decoding) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= PM_MODELS * PM_SYMBOLS) return;
    u32 model = id / PM_SYMBOLS, symbol = id % PM_SYMBOLS, code = frequencies[id];
    u32 f = code >> 16, start = code & 65535u, rank = 0;
    if (!f) return;
    for (u32 state = 0; state < PM_STATES; ++state) {
        u32 sequence = (state * 43u) & (PM_STATES - 1);  // inverse of spread step 643
        if (sequence < start || sequence >= start + f) continue;
        u32 n = f + rank, bits = 10 - (31 - __clz(n)), base = (n << bits) - PM_STATES;
        u32 pair = symbol == PM_ESCAPE ? 4095u : (symbol / 33) | ((symbol % 33) << 6);
        decoding[model * PM_STATES + state] = pair | (bits << 12) | (base << 16);
        encoding[model * PM_STATES + start + rank++] = u16(state);
    }
}

// ---------------------------------------------------------------------------
// Encoder: one thread per stream, reverse order so the decoder reads forward.

__device__ __forceinline__ void pm_push(u32 value, u32 n, u64& buffer, u32& available, u32& emitted,
                                        u8* scratch, u32 streams, u32 s) {
    buffer |= u64(value) << available;
    available += n;
    while (available >= 8) {
        scratch[u64(emitted++) * streams + s] = u8(buffer);
        buffer >>= 8;
        available -= 8;
    }
}

extern "C" __global__ void pm_encode(const void* raw, int itemsize, u32 scans, u32 pixels, u32 interval,
                                     const u32* frequencies, const u16* encoding, u8* scratch,
                                     u32* sizes, u32* states, u8* models, u32 streams) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= streams) return;
    u32 first = (s / pixels) * interval, pixel = s % pixels, length = min(interval, scans - first);
    u32 lo = 65535, hi = 0, sum = 0, nonzero = 0;
    for (u32 i = 0; i < length; ++i) {
        u32 v = pm_raw(raw, u64(first + i) * pixels + pixel, itemsize);
        lo = min(lo, v); hi = max(hi, v); sum += min(v, 32u); nonzero += v != 0;
    }
    if (lo == hi) { models[s] = hi ? 255 : 253; sizes[s] = hi ? 2 : 0; states[s] = hi; return; }
    if (hi <= 128 && nonzero <= 2) { models[s] = 252; sizes[s] = 2 * nonzero; states[s] = 0; return; }
    float mean = float(sum) / length;
    u32 m = max(0, min(int(PM_MODELS) - 1, __float2int_rn((logf(fmaxf(mean, .002f)) - logf(.002f)) * (float(PM_MODELS - 1) / logf(16000.0f)))));
    u32 state = 0, available = 0, emitted = 0, total = 0;
    u64 buffer = 0;
    bool valid = true;
    for (u32 j = (length + 1) / 2; j > 0; --j) {
        if (emitted >= 2 * length) { valid = false; break; }
        u32 i = (j - 1) * 2;
        u32 a = pm_raw(raw, u64(first + i) * pixels + pixel, itemsize);
        u32 b = i + 1 < length ? pm_raw(raw, u64(first + i + 1) * pixels + pixel, itemsize) : 0;
        u32 symbol = a < 32 && b < 32 ? a * 33 + b : PM_ESCAPE, code = frequencies[m * PM_SYMBOLS + symbol];
        if (symbol == PM_ESCAPE || !(code >> 16)) {
            code = frequencies[m * PM_SYMBOLS + PM_ESCAPE];
            if (a < 64 && b < 64) { pm_push(a | (b << 6), 13, buffer, available, emitted, scratch, streams, s); total += 13; }
            else {
                pm_push(b, 16, buffer, available, emitted, scratch, streams, s);
                pm_push(a, 16, buffer, available, emitted, scratch, streams, s);
                pm_push(4096, 13, buffer, available, emitted, scratch, streams, s);
                total += 45;
            }
        }
        u32 f = code >> 16, start = code & 65535u, y = PM_STATES + state, bits = 10 - (31 - __clz(f));
        if (y < (f << bits)) --bits;
        u32 rank = (y >> bits) - f, low = y & ((1u << bits) - 1);
        pm_push(low, bits, buffer, available, emitted, scratch, streams, s);
        total += bits;
        state = encoding[m * PM_STATES + start + rank];
    }
    if (available) scratch[u64(emitted++) * streams + s] = u8(buffer);
    bool use = valid && emitted + 2 < 2 * length && total < 16384;
    u32 bytes = use ? emitted + 2 : 2 * length;
    models[s] = use ? 64 + m : 254;
    states[s] = (state << 6) | (total & 7u);
    // Sparse events decode with a few atomics; prefer them unless the paired stream saves at least two bytes.
    if (hi <= 128 && 2 * nonzero <= bytes + 1u) { models[s] = 252; bytes = 2 * nonzero; }
    sizes[s] = bytes;
}

extern "C" __global__ void pm_compact(const void* raw, int itemsize, u32 scans, u32 pixels, u32 interval,
                                      const u8* scratch, const u32* offsets, const u32* states,
                                      const u8* models, u8* payload, u32 streams) {
    u32 stream = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= streams) return;
    u32 begin = offsets[stream], size = offsets[stream + 1] - begin, model = models[stream];
    if (model == 253) return;
    if (model == 252) {
        u32 first = (stream / pixels) * interval, pixel = stream % pixels, at = begin;
        for (u32 i = 0; i < min(interval, scans - first); ++i) {
            u32 value = pm_raw(raw, u64(first + i) * pixels + pixel, itemsize);
            if (value) { u32 event = (i << 7) | (value - 1); payload[at++] = u8(event); payload[at++] = u8(event >> 8); }
        }
        return;
    }
    if (model == 255) { payload[begin] = u8(states[stream]); payload[begin + 1] = u8(states[stream] >> 8); return; }
    if (model == 254) {
        u32 first = (stream / pixels) * interval, pixel = stream % pixels;
        for (u32 i = 0; i < size / 2; ++i) {
            u32 v = pm_raw(raw, u64(first + i) * pixels + pixel, itemsize);
            payload[begin + 2 * i] = u8(v); payload[begin + 2 * i + 1] = u8(v >> 8);
        }
        return;
    }
    for (int i = 0; i < 2; ++i) payload[begin + i] = u8(states[stream] >> (8 * i));
    for (u32 i = 2; i < size; ++i) payload[begin + i] = scratch[u64(i - 2) * streams + stream];
}

// ---------------------------------------------------------------------------
// Readers

// Exact alternative forms: sparse events, zero, literal and constant streams.
struct SparseReader {
    const u8* payload;
    u32 cursor, end, position, model, constant;
    bool valid;
    __device__ SparseReader() = default;
    __device__ SparseReader(const u8* bytes, const u32* records, const u8* models, u32 stream)
        : payload(bytes), cursor(pm_begin(records, stream)), end(pm_end(records, stream)), position(0),
          model(models[stream]), constant(0), valid(true) {
        if (model == 252) { valid = (end - cursor) % 2 == 0; return; }
        if (model == 253) return;
        if (model == 255) {
            valid = end - cursor == 2;
            if (valid) { constant = payload[cursor] | (u32(payload[cursor + 1]) << 8); cursor += 2; }
            return;
        }
        valid = model == 254;
    }
    __device__ u32 next() {
        if (!valid) return 0;
        if (model == 252) {
            u32 scan = position++;
            if (cursor == end) return 0;
            u32 event = payload[cursor] | (u32(payload[cursor + 1]) << 8);
            if ((event >> 7) < scan) { valid = false; return 0; }
            if ((event >> 7) != scan) return 0;
            cursor += 2;
            return (event & 127u) + 1;
        }
        if (model == 253 || model == 255) return constant;
        if (end - cursor < 2) { valid = false; return 0; }
        u32 value = payload[cursor] | (u32(payload[cursor + 1]) << 8);
        cursor += 2;
        return value;
    }
    __device__ bool finished() const { return valid && cursor == end; }
};

// Paired tANS stream read from its end: 64-bit reservoir refilled four bytes at a time
// from a register-prefetched window, so the load latency overlaps the symbol decodes.
// A refill may run a few bytes below the stream start (into the previous stream, never
// below the payload); those bits are never consumed by a well-formed stream and
// finished() requires exactly them to remain. Callers guarantee bits before decode():
// ensure() gives at least 32 bits, enough for three coded pairs, and the escape path
// tops up for itself. A short or malformed stream is reported by finished().
struct PairReader {
    const u8* bytes;
    const u32* table;
    u32 begin, cursor, state, available, pending_low, pending_high, pending_shift;
    u64 buffer;
    bool valid;
    __device__ __forceinline__ void prime() {
        u32 low = cursor >= 4 ? cursor - 4 : 0;
        const u32* p = (const u32*)(bytes + (low & ~3u));
        pending_low = p[0]; pending_high = p[1]; pending_shift = (low & 3u) * 8;
    }
    __device__ PairReader(const u8* p, const u32* records, const u8* models, const u32* decoding, u32 stream)
        : bytes(p), begin(pm_begin(records, stream)), cursor(pm_end(records, stream)), state(0), available(0),
          pending_low(0), pending_high(0), pending_shift(0), buffer(0), valid(true) {
        u32 m = models[stream];
        valid = m >= 64 && m < 64 + PM_MODELS && cursor >= begin && cursor - begin >= 2;
        table = decoding + ((m >= 64 && m < 64 + PM_MODELS) ? (m - 64) * PM_STATES : 0);
        if (!valid) { cursor = begin; return; }
        u32 header = bytes[begin] | (u32(bytes[begin + 1]) << 8), tail = header & 7u;
        valid = ((header >> 3) & 7u) == 0 && (tail == 0 || cursor - begin > 2);
        state = header >> 6;
        begin += 2;
        if (!valid) { cursor = begin; return; }
        u32 bits = (cursor - begin) * 8 - ((8 - tail) & 7u);
        if (bits % 8) { available = bits % 8; buffer = bytes[--cursor]; valid = buffer < (1u << available); }
        prime();
    }
    // Requires available < 32. Takes the prefetched window below the cursor, then prefetches again.
    __device__ __forceinline__ void refill() {
        u32 word = __funnelshift_r(pending_low, pending_high, pending_shift);
        if (cursor < 4) {
            // Payload start: only `cursor` bytes remain below; the window began at byte 0.
            u32 count = cursor;
            word &= (1u << (count * 8)) - 1u;
            buffer = (buffer << (count * 8)) | word;
            available += count * 8;
            cursor = 0;
        } else {
            buffer = (buffer << 32) | word;
            available += 32;
            cursor -= 4;
        }
        prime();
    }
    __device__ __forceinline__ void ensure() { if (available < 32) refill(); }
    __device__ __forceinline__ u32 pop(u32 n) {
        available -= n;
        return u32(buffer >> available) & ((1u << n) - 1u);
    }
    // One coded pair; the caller has ensured its bits (ten for the state step).
    __device__ __forceinline__ void decode(u32& a, u32& b, bool& wide) {
        u32 code = table[state], pair = code & 4095u;
        state = (code >> 16) + pop((code >> 12) & 15u);
        if (pair != 4095u) { a = pair & 63u; b = pair >> 6; return; }
        if (available < 13) refill();
        u32 word = pop(13);
        if (word < 4096u) { a = word & 63u; b = word >> 6; }
        else if (word == 4096u) {
            if (available < 16) refill();
            a = pop(16);
            if (available < 16) refill();
            b = pop(16);
            wide = true;
        }
        else { valid = false; a = 0; b = 0; }
        if (available < 20) refill();   // the rest of the caller's three-pair group
    }
    __device__ __forceinline__ void next(u32& a, u32& b, bool& wide) { ensure(); decode(a, b, wide); }
    __device__ bool finished() const {
        return valid && cursor <= begin && available == 8 * (begin - cursor) && state == 0;
    }
};

// ---------------------------------------------------------------------------
// Whole-chunk decode and single-frame extraction

extern "C" __global__ void pm_decode(const u8* payload, const u32* records, const u8* models, const u32* decoding,
                                     u16* raw, u32* errors, u32 scans, u32 pixels, u32 interval, u32 streams) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= streams) return;
    u32 first = (s / pixels) * interval, pixel = s % pixels, length = min(interval, scans - first), m = models[s];
    if (m >= 64 && m < 64 + PM_MODELS) {
        PairReader r(payload, records, models, decoding, s);
        bool wide = false;
        for (u32 i = 0; i < length; i += 2) {
            u32 a, b;
            r.next(a, b, wide);
            raw[u64(first + i) * pixels + pixel] = u16(a);
            if (i + 1 < length) raw[u64(first + i + 1) * pixels + pixel] = u16(b);
            else if (b) r.valid = false;
        }
        if (!r.finished()) atomicOr(errors, 1u);
        return;
    }
    SparseReader r(payload, records, models, s);
    for (u32 i = 0; i < length; ++i) raw[u64(first + i) * pixels + pixel] = u16(r.next());
    if (!r.finished()) atomicOr(errors, 1u);
}

// Same decode for a run of whole blocks: streams first_stream .. first_stream+count, written from scan origin.
extern "C" __global__ void pm_decode_range(const u8* payload, const u32* records, const u8* models, const u32* decoding,
                                           u16* raw, u32* errors, u32 scans, u32 pixels, u32 interval,
                                           u32 first_stream, u32 count) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u32 s = first_stream + i, first = (s / pixels) * interval, pixel = s % pixels;
    u32 length = min(interval, scans - first), origin = (first_stream / pixels) * interval, m = models[s];
    u64 base = u64(first - origin) * pixels + pixel;
    if (m >= 64 && m < 64 + PM_MODELS) {
        PairReader r(payload, records, models, decoding, s);
        bool wide = false;
        for (u32 j = 0; j < length; j += 2) {
            u32 a, b;
            r.next(a, b, wide);
            raw[base + u64(j) * pixels] = u16(a);
            if (j + 1 < length) raw[base + u64(j + 1) * pixels] = u16(b);
            else if (b) r.valid = false;
        }
        if (!r.finished()) atomicOr(errors, 1u);
        return;
    }
    SparseReader r(payload, records, models, s);
    for (u32 j = 0; j < length; ++j) raw[base + u64(j) * pixels] = u16(r.next());
    if (!r.finished()) atomicOr(errors, 1u);
}

template<typename Output>
__device__ void pm_frame(const u64* descriptors, Output* output, u32* errors, u32 index, u32 pixels) {
    const u64* d = descriptors + u64(blockIdx.y) * PM_DESCRIPTOR;
    u32 pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels || index < d[7] || index >= d[7] + d[8]) return;
    u32 interval = d[12], local = index - d[7], offset = local % interval, s = (local / interval) * pixels + pixel;
    const u8* payload = (const u8*)d[0];
    const u32* records = (const u32*)d[1];
    const u8* models = (const u8*)d[2];
    u32 m = models[s], value = 0;
    if (m >= 64 && m < 64 + PM_MODELS) {
        PairReader r(payload, records, models, (const u32*)d[3], s);
        bool wide = false;
        u32 a = 0, b = 0;
        for (u32 i = 0; i <= offset; i += 2) r.next(a, b, wide);
        if (!r.valid) atomicOr(errors, 1u);
        value = offset & 1 ? b : a;
    } else {
        SparseReader r(payload, records, models, s);
        for (u32 i = 0; i <= offset; ++i) value = r.next();
        if (!r.valid) atomicOr(errors, 1u);
    }
    output[d[9] * pixels + pixel] = Output(value);
}
extern "C" __global__ void pm_frame_u8(const u64* d, u8* o, u32* e, u32 index, u32 pixels) { pm_frame(d, o, e, index, pixels); }
extern "C" __global__ void pm_frame_u16(const u64* d, u16* o, u32* e, u32 index, u32 pixels) { pm_frame(d, o, e, index, pixels); }

// ---------------------------------------------------------------------------
// Residual pixel decoding for a detector query: a work list of paired streams per
// (chunk, 512-scan block), cheap modes applied directly, then one thread per paired
// stream with a warp-wide packed reduction per 32 scans.

template<typename Output>
__device__ void pm_plan(const u64* descriptors, const u32* selected, const int* coefficients, u32 count,
                        Output* output, u32* errors, u64 scans, u32 pixels, u16* work, u32* counts, u32 max_blocks, u32 stride) {
    // `max_blocks` work items per chunk cover blocks 0, stride, 2*stride, ...: a viewer may
    // sum every k-th 512-scan block of a moving mask and leave the other rows untouched.
    u32 sb = blockIdx.x, chunk = sb / max_blocks, block = (sb % max_blocks) * stride, lane = threadIdx.x & 31;
    const u64* d = descriptors + u64(chunk) * PM_DESCRIPTOR;
    u32 interval = d[12], first = block * interval;
    if (first >= d[8]) return;
    u32 length = min(interval, u32(d[8]) - first);
    int constant = 0;
    const u8* payload = (const u8*)d[0];
    const u32* records = (const u32*)d[1];
    const u8* models = (const u8*)d[2];
    for (u32 base = 0; base < count; base += blockDim.x) {
        u32 at = base + threadIdx.x, pixel = at < count ? selected[at] : 0;
        int coefficient = at < count && ((const u8*)d[10])[pixel] ? coefficients[at] : 0;
        bool paired = false;
        if (coefficient) {
            u32 s = block * pixels + pixel, m = models[s];
            if (m >= 64 && m < 64 + PM_MODELS) paired = true;
            else if (m == 252) {
                u32 begin = pm_begin(records, s), end = pm_end(records, s), previous = 0;
                bool seen = false;
                if (end < begin || (end - begin) % 2) atomicOr(errors, 1u);
                else for (u32 pos = begin; pos < end; pos += 2) {
                    u32 event = payload[pos] | (u32(payload[pos + 1]) << 8), i = event >> 7;
                    if (i >= length || (seen && i <= previous)) { atomicOr(errors, 1u); break; }
                    atomicAdd(output + d[9] * scans + d[7] + first + i, Output((long long)coefficient * ((event & 127u) + 1)));
                    previous = i; seen = true;
                }
            } else if (m == 254) {
                u32 begin = pm_begin(records, s);
                if (pm_end(records, s) - begin != 2 * length) atomicOr(errors, 1u);
                else for (u32 i = 0; i < length; ++i) {
                    u32 v = payload[begin + 2 * i] | (u32(payload[begin + 2 * i + 1]) << 8);
                    if (v) atomicAdd(output + d[9] * scans + d[7] + first + i, Output((long long)coefficient * v));
                }
            } else if (m == 255) {
                u32 begin = pm_begin(records, s);
                if (pm_end(records, s) - begin != 2) atomicOr(errors, 1u);
                else constant += coefficient * int(payload[begin] | (u32(payload[begin + 1]) << 8));
            } else if (m == 253) {
                if (pm_begin(records, s) != pm_end(records, s)) atomicOr(errors, 1u);
            } else atomicOr(errors, 1u);
        }
        u32 mask = __ballot_sync(0xffffffffu, paired), start = 0;
        if (!lane && mask) start = atomicAdd(counts + sb, __popc(mask));
        start = __shfl_sync(0xffffffffu, start, 0);
        if (paired) work[u64(sb) * count + start + __popc(mask & ((1u << lane) - 1))] = u16(at);
    }
    constant = __reduce_add_sync(0xffffffffu, constant);
    if (constant) for (u32 i = lane; i < length; i += 32) atomicAdd(output + d[9] * scans + d[7] + first + i, Output((long long)constant));
}

template<typename Output>
__device__ void pm_residual(const u64* descriptors, const u32* selected, const int* coefficients, u32 count,
                            Output* output, u32* errors, u64 scans, u32 pixels, const u16* work, const u32* counts,
                            u32 max_blocks, u32 stride) {
    // Work items (chunk, block) go in grid x: a series of many small chunks exceeds the 65,535 limit of y.
    u32 sb = blockIdx.x, chunk = sb / max_blocks, block = (sb % max_blocks) * stride, lane = threadIdx.x & 31;
    u32 at = blockIdx.y * blockDim.x + threadIdx.x, active = counts[sb];
    if ((at / 32) * 32 >= active) return;
    const u64* d = descriptors + u64(chunk) * PM_DESCRIPTOR;
    u32 interval = d[12], first = block * interval, length = min(interval, u32(d[8]) - first);
    if (length != interval) { atomicOr(errors, 2u); return; }
    // Inactive lanes decode a duplicate of their warp's first stream with coefficient zero.
    u32 selected_at = work[u64(sb) * count + min(at, active - 1)], pixel = selected[selected_at];
    int coefficient = at < active ? coefficients[selected_at] : 0;
    PairReader reader((const u8*)d[0], (const u32*)d[1], (const u8*)d[2], (const u32*)d[3], block * pixels + pixel);
    // Two counts below 64 with |coefficient| <= 2 sum to at most 4032 over 32 lanes, so a
    // 16-bit biased pack cannot overflow; the escape literal path and larger
    // coefficients take the full-width reduction.
    bool lane_wide = coefficient < -2 || coefficient > 2;
    // Per-lane constants of the packed reduction: which of the four reduced words and
    // which half of it hold this lane's scan, and the group that produces it.
    u32 which = (lane >> 1) & 3u, half_shift = (lane & 1u) * 16, group = lane & ~7u;
    for (u32 batch = 0; batch < length; batch += 32) {
        int result = 0;
        #pragma unroll
        for (u32 g = 0; g < 32; g += 8) {
            // Biased products: a count times a coefficient of magnitude at most two lies
            // in [-126, 126], so 1024 keeps every 16-bit field positive across 32 lanes.
            u32 p[8];
            bool wide = lane_wide;
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                u32 a, b;
                if (((g + j) / 2) % 3 == 0) reader.ensure();   // 32 bits cover three coded pairs
                reader.decode(a, b, wide);
                p[j] = u32(int(a) * coefficient + 1024);
                p[j + 1] = u32(int(b) * coefficient + 1024);
            }
            if (__any_sync(0xffffffffu, wide)) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) { int sum = __reduce_add_sync(0xffffffffu, int(p[j]) - 1024); if (lane == g + j) result = sum; }
            } else {
                u32 s0 = __reduce_add_sync(0xffffffffu, p[0] | (p[1] << 16));
                u32 s1 = __reduce_add_sync(0xffffffffu, p[2] | (p[3] << 16));
                u32 s2 = __reduce_add_sync(0xffffffffu, p[4] | (p[5] << 16));
                u32 s3 = __reduce_add_sync(0xffffffffu, p[6] | (p[7] << 16));
                u32 word = (which & 2u) ? ((which & 1u) ? s3 : s2) : ((which & 1u) ? s1 : s0);
                if (group == g) result = int((word >> half_shift) & 65535u) - 32768;
            }
        }
        if (result) atomicAdd(output + d[9] * scans + d[7] + first + batch + lane, Output((long long)result));
    }
    if (coefficient && !reader.finished()) atomicOr(errors, 1u);
}

#define PM_RESIDUAL(BITS, TYPE) \
extern "C" __global__ void pm_plan_u##BITS(const u64* d, const u32* s, const int* c, u32 n, TYPE* o, u32* e, u64 scans, u32 pixels, u16* work, u32* counts, u32 blocks, u32 stride) { pm_plan(d, s, c, n, o, e, scans, pixels, work, counts, blocks, stride); } \
extern "C" __global__ void pm_residual_u##BITS(const u64* d, const u32* s, const int* c, u32 n, TYPE* o, u32* e, u64 scans, u32 pixels, const u16* work, const u32* counts, u32 blocks, u32 stride) { pm_residual(d, s, c, n, o, e, scans, pixels, work, counts, blocks, stride); }
PM_RESIDUAL(32, u32)
PM_RESIDUAL(64, u64)

// ---------------------------------------------------------------------------
// Spatial index: exact sums over 64-pixel polar leaves and 16-leaf roots, then
// frame-of-reference bit packing per (512-scan interval, field).

extern "C" __global__ void pm_fields(const void* raw, int itemsize, const u8* valid, u32* fields, u32 scans,
                                     u32 rows, u32 cols, u32 field_count, const int* permutation) {
    u64 item = (u64(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    u32 lane = threadIdx.x & 31;
    if (item >= u64(scans) * field_count) return;
    u32 scan = item / field_count, field = item % field_count, leaves = ((rows + 7) / 8) * ((cols + 7) / 8);
    u32 first = field < leaves ? field * 64 : (field - leaves) * 1024, size = field < leaves ? 64 : 1024, sum = 0;
    for (u32 i = lane; i < size; i += 32) {
        u32 at = first + i;
        int pixel = at < leaves * 64 ? permutation[at] : -1;
        if (pixel >= 0 && valid[pixel]) sum += pm_raw(raw, u64(scan) * rows * cols + pixel, itemsize);
    }
    sum = __reduce_add_sync(0xffffffffu, sum);
    if (!lane) fields[item] = sum;
}

extern "C" __global__ void pm_field_sizes(const u32* values, u8* widths, u64* sizes, u32 scans, u32 fields, u32 interval) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= ((scans + interval - 1) / interval) * fields) return;
    u32 first = (s / fields) * interval, f = s % fields, n = min(interval, scans - first), lo = 0xffffffffu, hi = 0;
    for (u32 i = 0; i < n; ++i) { u32 v = values[u64(first + i) * fields + f]; lo = min(lo, v); hi = max(hi, v); }
    u32 range = hi - lo, w = hi ? 32 - __clz(hi) : 0, r = range ? 32 - __clz(range) : 0;
    u64 old = (u64(n) * w + 31) / 32, proposed = 1 + (u64(n) * r + 31) / 32;
    bool use = proposed < old;
    widths[s] = use ? r | 128u : w;
    sizes[s] = use ? proposed : old;
}

extern "C" __global__ void pm_pack_fields(const u32* values, const u8* widths, const u64* offsets, u32* payload,
                                          u32 scans, u32 fields, u32 interval) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= ((scans + interval - 1) / interval) * fields) return;
    u32 tag = widths[s], w = tag & 63u, first = (s / fields) * interval, f = s % fields, n = min(interval, scans - first), base = 0;
    u64 at = offsets[s];
    if (tag & 128u) { base = 0xffffffffu; for (u32 i = 0; i < n; ++i) base = min(base, values[u64(first + i) * fields + f]); payload[at++] = base; }
    if (!w) return;
    u32 available = 0;
    u64 reservoir = 0;
    for (u32 i = 0; i < n; ++i) {
        reservoir |= u64(values[u64(first + i) * fields + f] - base) << available;
        available += w;
        if (available >= 32) { payload[at++] = u32(reservoir); reservoir >>= 32; available -= 32; }
    }
    if (available) payload[at] = u32(reservoir);
}

__device__ u32 pm_field(const u32* payload, const u64* offsets, const u8* widths, u32 scan, u32 field, u32 fields, u32 interval) {
    u32 s = (scan / interval) * fields + field, tag = widths[s], w = tag & 63u, base = 0;
    u64 at = offsets[s];
    if (tag & 128u) base = payload[at++];
    if (!w) return base;
    u64 bit = u64(scan % interval) * w;
    at += bit / 32;
    u64 value = payload[at];
    if ((bit % 32) + w > 32) value |= u64(payload[at + 1]) << 32;
    return base + (u32(value >> (bit % 32)) & (w == 32 ? 0xffffffffu : (1u << w) - 1));
}
extern "C" __global__ void pm_unpack_fields(const u32* p, const u64* o, const u8* w, u32* out, u32 scans, u32 fields, u32 interval) {
    u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < u64(scans) * fields) out[i] = pm_field(p, o, w, i / fields, i % fields, fields, interval);
}

// Index sums: one block of 128 consecutive scans lies inside one interval, so the
// per-field tag, word offset and minimum are loaded once per block.
template<typename Output>
__device__ void pm_index_sum(const u64* descriptors, const u32* selected, const int* coefficients, u32 count,
                             const Output* previous, Output* output, u64 scans, u32 fields, int delta, u32 stride,
                             u32* s_word, u32* s_base, u32* s_width, int* s_coef) {
    const u64* d = descriptors + u64(blockIdx.y) * PM_DESCRIPTOR;
    u32 interval = d[12], per_block = interval / blockDim.x;
    u32 scan0 = (blockIdx.x / per_block) * stride * interval + (blockIdx.x % per_block) * blockDim.x;
    if (scan0 >= d[8]) return;
    const u32* payload = (const u32*)d[4];
    const u64* offsets = (const u64*)d[5];
    const u8* widths = (const u8*)d[6];
    u32 block_interval = scan0 / interval;
    for (u32 i = threadIdx.x; i < count; i += blockDim.x) {
        u32 s = block_interval * fields + selected[i], tag = widths[s];
        u64 at = offsets[s];
        u32 base = 0;
        if (tag & 128u) base = payload[at++];
        s_word[i] = u32(at); s_base[i] = base; s_width[i] = tag & 63u; s_coef[i] = coefficients[i];
    }
    __syncthreads();
    u32 scan = scan0 + threadIdx.x;
    if (scan >= d[8]) return;
    u32 local = scan % interval;
    u64 at = d[9] * scans + d[7] + scan;
    Output value = delta ? previous[at] : 0;
    #pragma unroll 4
    for (u32 i = 0; i < count; ++i) {
        u32 w = s_width[i], v = s_base[i];
        if (w) {
            u32 bit = local * w, index = s_word[i] + (bit >> 5), shift = bit & 31u;
            u32 extracted = __funnelshift_r(payload[index], payload[index + 1], shift);
            v += w == 32 ? extracted : extracted & ((1u << w) - 1);
        }
        value += Output((long long)s_coef[i] * v);
    }
    output[at] = value;
}
#define PM_INDEX(BITS, TYPE) \
extern "C" __global__ void pm_index_u##BITS(const u64* d, const u32* selected, const int* coefficients, u32 count, const TYPE* previous, TYPE* output, u64 scans, u32 fields, int delta, u32 stride) { \
    extern __shared__ u32 shared[]; \
    pm_index_sum<TYPE>(d, selected, coefficients, count, previous, output, scans, fields, delta, stride, shared, shared + fields, shared + 2 * fields, (int*)(shared + 3 * fields)); }
PM_INDEX(32, u32)
PM_INDEX(64, u64)

// Planner weights: expected decode cost per detector pixel from the resident
// stream modes and byte lengths, averaged over 512-scan blocks and acquisitions.
extern "C" __global__ void pm_weights(const u64* descriptors, double* out, u32 pixels, u32 chunks) {
    u32 p = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (p >= pixels || c >= chunks) return;
    const u64* d = descriptors + u64(c) * PM_DESCRIPTOR;
    if (!((const u8*)d[10])[p]) return;
    const u32* records = (const u32*)d[1];
    const u8* models = (const u8*)d[2];
    u32 interval = d[12];
    double cost = 0;
    for (u32 first = 0; first < d[8]; first += interval) {
        u32 s = (first / interval) * pixels + p, m = models[s], begin = pm_begin(records, s), end = pm_end(records, s);
        double n = double(min(interval, u32(d[8]) - first)) / double(interval);
        if (m >= 64 && m < 64 + PM_MODELS) cost += 4 * n + (end - begin) / 128.0;
        else if (m == 252) cost += .01 * n + (end - begin) / 128.0;
        else if (m == 254) cost += 1.5 * n;
        else if (m == 255) cost += .05 * n;
        else cost += .005 * n;
    }
    atomicAdd(out + p, cost);
}
