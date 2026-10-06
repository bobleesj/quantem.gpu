#include <metal_stdlib>
using namespace metal;

struct OriginalPackingShape { uint scans, pixels, columns, sourceBytes; };
template<uint fixedSourceBytes = 0u>
inline uint original_count(const device uchar *source, uint scan, uint pixel,
                           constant OriginalPackingShape &s) {
    ulong index = ulong(scan) * s.pixels + pixel;
    return fixedSourceBytes == 1u
        ? uint(source[index])
        : (fixedSourceBytes == 4u
            ? ((const device uint *)source)[index]
            : (fixedSourceBytes == 2u
                ? uint(((const device ushort *)source)[index])
                : (s.sourceBytes == 1u
                    ? uint(source[index])
                    : (s.sourceBytes == 4u
                        ? ((const device uint *)source)[index]
                        : uint(((const device ushort *)source)[index])))));
}

// Independent checkpoint-sized workers preserve the same exact packed bytes.
// Requires the same validated nonoverlapping header partition as the baseline.
kernel void original_packing_values_verified_checkpoints(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    for (uint tile = checkpoint * 32; tile < min(tiles, (checkpoint + 1) * 32); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        uint originals[32];
        uint packed = 0, occupied = 0, outputWord = 0;
        for (uint sample = 0; sample < 32; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            originals[sample] = value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32;
                    packed = occupied ? value >> (bits - occupied) : 0;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0; sample < 32; ++sample) {
            uint bit = sample * bits, shift = bit % 32;
            ulong value = bits ? ulong(payload[offset + bit / 32]) >> shift : 0;
            if (shift + bits > 32) value |= ulong(payload[offset + bit / 32 + 1]) << (32 - shift);
            if ((value & mask) != originals[sample])
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        }
        offset += bits;
    }
}

inline uint originalEvenBits(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1u)) & 0x33333333u;
    value = (value | (value >> 2u)) & 0x0f0f0f0fu;
    value = (value | (value >> 4u)) & 0x00ff00ffu;
    return (value | (value >> 8u)) & 0xffffu;
}

inline uint originalThirdBits(uint value) {
    uint high = (value >> 20u) & 0x400u;
    value &= 0x09249249u;
    value = (value ^ (value >> 2u)) & 0x030c30c3u;
    value = (value ^ (value >> 4u)) & 0x0300f00fu;
    value = (value ^ (value >> 8u)) & 0x030000ffu;
    value = (value ^ (value >> 16u)) & 0x000003ffu;
    return value | high;
}

inline uint originalFourthBits(uint value) {
    value &= 0x11111111u;
    value = (value | (value >> 3u)) & 0x03030303u;
    value = (value | (value >> 6u)) & 0x000f000fu;
    return (value | (value >> 12u)) & 0xffu;
}

// Diagnostic-only pair transposes for the uncommon five- and six-bit tiles.
// Each table entry encodes two samples: plane p occupies bits 2*p and
// 2*p+1, with the low bit belonging to the first sample. Splitting the
// 10/12-bit pair code into an 8-bit low part and a small high part keeps the
// constants compact while retaining the same exact LSB-first layout.
constant ushort originalFifthPairLowLUT[256] = {
    0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015, 0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055,
    0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115, 0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155,
    0x0002, 0x0003, 0x0006, 0x0007, 0x0012, 0x0013, 0x0016, 0x0017, 0x0042, 0x0043, 0x0046, 0x0047, 0x0052, 0x0053, 0x0056, 0x0057,
    0x0102, 0x0103, 0x0106, 0x0107, 0x0112, 0x0113, 0x0116, 0x0117, 0x0142, 0x0143, 0x0146, 0x0147, 0x0152, 0x0153, 0x0156, 0x0157,
    0x0008, 0x0009, 0x000c, 0x000d, 0x0018, 0x0019, 0x001c, 0x001d, 0x0048, 0x0049, 0x004c, 0x004d, 0x0058, 0x0059, 0x005c, 0x005d,
    0x0108, 0x0109, 0x010c, 0x010d, 0x0118, 0x0119, 0x011c, 0x011d, 0x0148, 0x0149, 0x014c, 0x014d, 0x0158, 0x0159, 0x015c, 0x015d,
    0x000a, 0x000b, 0x000e, 0x000f, 0x001a, 0x001b, 0x001e, 0x001f, 0x004a, 0x004b, 0x004e, 0x004f, 0x005a, 0x005b, 0x005e, 0x005f,
    0x010a, 0x010b, 0x010e, 0x010f, 0x011a, 0x011b, 0x011e, 0x011f, 0x014a, 0x014b, 0x014e, 0x014f, 0x015a, 0x015b, 0x015e, 0x015f,
    0x0020, 0x0021, 0x0024, 0x0025, 0x0030, 0x0031, 0x0034, 0x0035, 0x0060, 0x0061, 0x0064, 0x0065, 0x0070, 0x0071, 0x0074, 0x0075,
    0x0120, 0x0121, 0x0124, 0x0125, 0x0130, 0x0131, 0x0134, 0x0135, 0x0160, 0x0161, 0x0164, 0x0165, 0x0170, 0x0171, 0x0174, 0x0175,
    0x0022, 0x0023, 0x0026, 0x0027, 0x0032, 0x0033, 0x0036, 0x0037, 0x0062, 0x0063, 0x0066, 0x0067, 0x0072, 0x0073, 0x0076, 0x0077,
    0x0122, 0x0123, 0x0126, 0x0127, 0x0132, 0x0133, 0x0136, 0x0137, 0x0162, 0x0163, 0x0166, 0x0167, 0x0172, 0x0173, 0x0176, 0x0177,
    0x0028, 0x0029, 0x002c, 0x002d, 0x0038, 0x0039, 0x003c, 0x003d, 0x0068, 0x0069, 0x006c, 0x006d, 0x0078, 0x0079, 0x007c, 0x007d,
    0x0128, 0x0129, 0x012c, 0x012d, 0x0138, 0x0139, 0x013c, 0x013d, 0x0168, 0x0169, 0x016c, 0x016d, 0x0178, 0x0179, 0x017c, 0x017d,
    0x002a, 0x002b, 0x002e, 0x002f, 0x003a, 0x003b, 0x003e, 0x003f, 0x006a, 0x006b, 0x006e, 0x006f, 0x007a, 0x007b, 0x007e, 0x007f,
    0x012a, 0x012b, 0x012e, 0x012f, 0x013a, 0x013b, 0x013e, 0x013f, 0x016a, 0x016b, 0x016e, 0x016f, 0x017a, 0x017b, 0x017e, 0x017f,
};

constant ushort originalFifthPairHighLUT[4] = {
    0x0000, 0x0080, 0x0200, 0x0280,
};

constant ushort originalSixthPairLowLUT[256] = {
    0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015, 0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055,
    0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115, 0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155,
    0x0400, 0x0401, 0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415, 0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455,
    0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515, 0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555,
    0x0002, 0x0003, 0x0006, 0x0007, 0x0012, 0x0013, 0x0016, 0x0017, 0x0042, 0x0043, 0x0046, 0x0047, 0x0052, 0x0053, 0x0056, 0x0057,
    0x0102, 0x0103, 0x0106, 0x0107, 0x0112, 0x0113, 0x0116, 0x0117, 0x0142, 0x0143, 0x0146, 0x0147, 0x0152, 0x0153, 0x0156, 0x0157,
    0x0402, 0x0403, 0x0406, 0x0407, 0x0412, 0x0413, 0x0416, 0x0417, 0x0442, 0x0443, 0x0446, 0x0447, 0x0452, 0x0453, 0x0456, 0x0457,
    0x0502, 0x0503, 0x0506, 0x0507, 0x0512, 0x0513, 0x0516, 0x0517, 0x0542, 0x0543, 0x0546, 0x0547, 0x0552, 0x0553, 0x0556, 0x0557,
    0x0008, 0x0009, 0x000c, 0x000d, 0x0018, 0x0019, 0x001c, 0x001d, 0x0048, 0x0049, 0x004c, 0x004d, 0x0058, 0x0059, 0x005c, 0x005d,
    0x0108, 0x0109, 0x010c, 0x010d, 0x0118, 0x0119, 0x011c, 0x011d, 0x0148, 0x0149, 0x014c, 0x014d, 0x0158, 0x0159, 0x015c, 0x015d,
    0x0408, 0x0409, 0x040c, 0x040d, 0x0418, 0x0419, 0x041c, 0x041d, 0x0448, 0x0449, 0x044c, 0x044d, 0x0458, 0x0459, 0x045c, 0x045d,
    0x0508, 0x0509, 0x050c, 0x050d, 0x0518, 0x0519, 0x051c, 0x051d, 0x0548, 0x0549, 0x054c, 0x054d, 0x0558, 0x0559, 0x055c, 0x055d,
    0x000a, 0x000b, 0x000e, 0x000f, 0x001a, 0x001b, 0x001e, 0x001f, 0x004a, 0x004b, 0x004e, 0x004f, 0x005a, 0x005b, 0x005e, 0x005f,
    0x010a, 0x010b, 0x010e, 0x010f, 0x011a, 0x011b, 0x011e, 0x011f, 0x014a, 0x014b, 0x014e, 0x014f, 0x015a, 0x015b, 0x015e, 0x015f,
    0x040a, 0x040b, 0x040e, 0x040f, 0x041a, 0x041b, 0x041e, 0x041f, 0x044a, 0x044b, 0x044e, 0x044f, 0x045a, 0x045b, 0x045e, 0x045f,
    0x050a, 0x050b, 0x050e, 0x050f, 0x051a, 0x051b, 0x051e, 0x051f, 0x054a, 0x054b, 0x054e, 0x054f, 0x055a, 0x055b, 0x055e, 0x055f,
};

constant ushort originalSixthPairHighLUT[16] = {
    0x0000, 0x0020, 0x0080, 0x00a0, 0x0200, 0x0220, 0x0280, 0x02a0, 0x0800, 0x0820, 0x0880, 0x08a0, 0x0a00, 0x0a20, 0x0a80, 0x0aa0,
};

inline uint originalWidth56PairTranspose(uint pair, uint bits) {
    if (bits == 5u) {
        return uint(originalFifthPairLowLUT[pair & 255u])
            | uint(originalFifthPairHighLUT[pair >> 8u]);
    }
    return uint(originalSixthPairLowLUT[pair & 255u])
        | uint(originalSixthPairHighLUT[pair >> 8u]);
}

// First-load path: verify the bounded decoded source against interleaved
// words, then transpose the verified cell in registers and verify its stores.
// Widths zero and one need no conversion; common two- to four-bit cases
// use exact register-level transposition.
template<bool skipStoreVerification = false, uint fixedSourceBytes = 0u,
         bool useWidth56PairLUT = false>
inline uint originalPackCountsAsPlanes(
    const device uchar *source, const device uint *headers,
    volatile device uint *payload, device atomic_uint *errors,
    constant OriginalPackingShape &s, uint index) {
    const bool validSourceBytes = fixedSourceBytes == 0u
        ? (s.sourceBytes == 1u || s.sourceBytes == 2u || s.sourceBytes == 4u)
        : s.sourceBytes == fixedSourceBytes;
    if (!s.pixels || !s.scans || s.scans % 32u || !validSourceBytes) {
        atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return 0u;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return 0u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    uint sum = 0u;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_count<fixedSourceBytes>(
                source, tile * 32u + sample, pixel, s);
            originals[sample] = value;
            sum += value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    if (payload[offset + outputWord - 1u] != packed)
                        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        // Header widths come from the same exact source tile, so the packed
        // fields cannot overflow their word partition. The write itself is
        // checked once per word above; the prior per-sample reread duplicated
        // those checks 32 times without adding output protection.
        if (bits > 1u) {
            if (useWidth56PairLUT && (bits == 5u || bits == 6u)) {
                uint packedWords[6] = {0u, 0u, 0u, 0u, 0u, 0u};
                uint wordCount = bits == 5u ? 5u : 6u;
                for (uint word = 0u; word < wordCount; ++word)
                    packedWords[word] = payload[offset + word];

                uint transposed[6] = {0u, 0u, 0u, 0u, 0u, 0u};
                uint pairMask = (1u << (2u * bits)) - 1u;
                for (uint pairIndex = 0u; pairIndex < 16u; ++pairIndex) {
                    uint bitOffset = pairIndex * (2u * bits);
                    uint word = bitOffset / 32u, shift = bitOffset % 32u;
                    ulong pairWords = ulong(packedWords[word]);
                    if (word + 1u < wordCount)
                        pairWords |= ulong(packedWords[word + 1u]) << 32u;
                    uint pair = uint((pairWords >> shift) & ulong(pairMask));
                    uint pairPlanes = originalWidth56PairTranspose(pair, bits);
                    for (uint plane = 0u; plane < bits; ++plane)
                        transposed[plane] |= ((pairPlanes >> (2u * plane)) & 3u)
                            << (2u * pairIndex);
                }
                for (uint plane = 0u; plane < bits; ++plane) {
                    uint value = transposed[plane];
                    payload[offset + plane] = value;
                    if (!skipStoreVerification && payload[offset + plane] != value)
                        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                }
            } else {
                uint4 words = uint4(0u);
                if (bits <= 4u) {
                    for (uint word = 0u; word < bits; ++word) words[word] = payload[offset + word];
                }
                for (uint plane = 0u; plane < bits; ++plane) {
                    uint value = 0u;
                    if (bits == 2u) {
                        value = originalEvenBits(words[0] >> plane)
                            | (originalEvenBits(words[1] >> plane) << 16u);
                    } else if (bits == 3u) {
                        uint secondShift = (plane + 1u) % 3u, thirdShift = (plane + 2u) % 3u;
                        uint firstCount = (34u - plane) / 3u, secondCount = (34u - secondShift) / 3u;
                        value = originalThirdBits(words[0] >> plane)
                            | (originalThirdBits(words[1] >> secondShift) << firstCount)
                            | (originalThirdBits(words[2] >> thirdShift) << (firstCount + secondCount));
                    } else if (bits == 4u) {
                        for (uint word = 0u; word < 4u; ++word)
                            value |= originalFourthBits(words[word] >> plane) << (8u * word);
                    } else {
                        for (uint sample = 0u; sample < 32u; ++sample)
                            value |= ((originals[sample] >> plane) & 1u) << sample;
                    }
                    payload[offset + plane] = value;
                    if (!skipStoreVerification && payload[offset + plane] != value)
                        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                }
            }
        }
        offset += bits;
    }
    return sum;
}

kernel void original_packing_values_planes_checkpoints(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    originalPackCountsAsPlanes(source, headers, payload, errors, s, index);
}

// The original-file first-load contract is uint16 for the BTO sources. This
// entry point keeps the same checked packing algorithm and layout, but makes
// the source width a compile-time constant so the hot count loop does not
// carry a per-sample source-width branch.
kernel void original_packing_values_planes_checkpoints_u16(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    originalPackCountsAsPlanes<false, 2u>(source, headers, payload, errors, s, index);
}

// Diagnostic-only width-5/6 specialization. Swift orchestration deliberately
// does not select this entry point yet; the existing checked kernels retain
// their current paths until focused parity and timing evidence is collected.
kernel void original_packing_values_planes_checkpoints_u16_width56_diagnostic(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    originalPackCountsAsPlanes<false, 2u, true>(source, headers, payload, errors, s, index);
}

// Diagnostic ceiling only; the normal first-load kernel verifies every
// packed word against its dense source before publication.
kernel void original_packing_values_planes_checkpoints_unchecked(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    originalPackCountsAsPlanes<true>(source, headers, payload, errors, s, index);
}

kernel void original_packing_values_planes_checkpoints_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (!s.pixels || index >= s.pixels * ((s.scans / 32u + 31u) / 32u)) return;
    partialSums[index] = originalPackCountsAsPlanes(source, headers, payload, errors, s, index);
}

// Input is complete checked LZ4-expanded uint16
// bitshuffle blocks, not dense counts. All frames and all 16 planes remain.
// Count verification below compares actual packed payload with reconstructed
// counts, including values in zero-width tiles.
inline uint original_bitshuffle_count(
    const device uchar *scratch, uint scan, uint pixel,
    constant OriginalPackingShape &s
) {
    const device uint *planes = (const device uint *)(scratch
        + ulong(scan) * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
    uint group = (pixel % 4096u) / 32u, lane = pixel % 32u;
    uint value = 0u;
    #pragma unroll
    for (uint bit = 0u; bit < 16u; ++bit) {
        value |= ((planes[bit * 128u + group] >> lane) & 1u) << bit;
    }
    return value;
}

// Cooperative gather for one aligned group of 32 detector pixels. Every lane
// participates, including lanes whose eventual packed tile has zero width.
inline uint original_bitshuffle_transpose_count(
    const device uchar *scratch, uint scan, uint pixel,
    constant OriginalPackingShape &s, uint lane
) {
    const device uint *planes = (const device uint *)(scratch
        + ulong(scan) * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
    uint group = (pixel % 4096u) / 32u;
    uint value = lane < 16u ? planes[lane * 128u + group] : 0u;
    #pragma unroll
    for (uint shift = 1u; shift <= 16u; shift *= 2u) {
        uint mask = 0xffffffffu / ((1u << shift) + 1u);
        uint other = simd_shuffle_xor(value, shift);
        value = (lane & shift)
            ? (value & ~mask) | ((other & ~mask) >> shift)
            : (value & mask) | ((other & mask) << shift);
    }
    return value;
}

// Separately selected candidate: the independent scalar-gather entry point
// below remains unchanged. Requires 32-lane SIMD groups and aligned pixels.
kernel void original_packing_bitshuffle_transpose_verified_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane
        || groupThreads % 32u != 0u;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints) return;
    // A different writer can fail as this group starts. Vote before returning
    // so that no lane leaves peers executing a transpose with missing inputs.
    if (simd_any(atomic_load_explicit(errors, memory_order_relaxed) != 0u)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    uint sum = 0u, maximum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_bitshuffle_transpose_count(source, tile * 32u + sample, pixel, s, lane);
            originals[sample] = value;
            sum += value;
            maximum = max(maximum, value);
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint bit = sample * bits, shift = bit % 32u;
            ulong value = bits ? ulong(payload[offset + bit / 32u]) >> shift : 0u;
            if (shift + bits > 32u) value |= ulong(payload[offset + bit / 32u + 1u]) << (32u - shift);
            if ((value & mask) != originals[sample]) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
        }
        // Never leave this loop per lane: even a failed lane participates in
        // every later transpose. No summaries are consumed if any count fails.
        offset += bits;
    }
    if (valid) {
        partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
        partialMaximums[ulong(checkpoint) * s.pixels + pixel] = maximum;
    }
}

// Transpose source bit planes across 32 scan positions directly into the
// resident cell. Every source bit is checked, including absent high planes;
// this proves all counts without constructing 32 scalar counts per lane.
inline void originalPackingBitshufflePlanes(
    const device uchar *source, const device uint *headers,
    volatile device uint *payload, device atomic_uint *errors,
    constant OriginalPackingShape &s, device uint *partialSums,
    device uint *partialMaximums, uint index, uint lane, uint simdWidth,
    uint groupThreads) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane
        || groupThreads % 32u != 0u;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints) return;
    if (simd_any(atomic_load_explicit(errors, memory_order_relaxed) != 0u)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    uint sum = 0u, maximum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint candidates = 0xffffffffu, tileMaximum = 0u;
        for (int plane = 15; plane >= 0; --plane) {
            ulong scan = ulong(tile) * 32ul + lane;
            const device uint *input = (const device uint *)(source
                + scan * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
            uint value = input[uint(plane) * 128u + (pixel % 4096u) / 32u];
            if (simd_any(value != 0u)) {
                #pragma unroll
                for (uint shift = 1u; shift <= 16u; shift *= 2u) {
                    uint mask = 0xffffffffu / ((1u << shift) + 1u);
                    uint other = simd_shuffle_xor(value, shift);
                    value = (lane & shift)
                        ? (value & ~mask) | ((other & ~mask) >> shift)
                        : (value & mask) | ((other & mask) << shift);
                }
            }
            uint restored = 0u;
            if (uint(plane) < bits) {
                payload[offset + uint(plane)] = value;
                restored = payload[offset + uint(plane)];
            }
            if (restored != value) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
            sum += popcount(value) << uint(plane);
            uint nextCandidates = candidates & value;
            if (nextCandidates != 0u) {
                tileMaximum |= 1u << uint(plane);
                candidates = nextCandidates;
            }
        }
        maximum = max(maximum, tileMaximum);
        offset += bits;
    }
    if (valid) {
        partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
        partialMaximums[ulong(checkpoint) * s.pixels + pixel] = maximum;
    }
}

kernel void original_packing_bitshuffle_planes_verified_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlanes(source, headers, payload, errors, s,
        partialSums, partialMaximums, index, lane, simdWidth, groupThreads);
}


// Four neighboring source words per lane improve transaction utilization.
// Each component remains an independent exact detector column. No temporary
// count volume or auxiliary resident representation is constructed.
template<typename Vector, uint columns, bool hasZeroTail = false,
         bool skipStoreVerification = false, bool widthAwareHighPlanes = false,
         bool low8Only = false, bool emitDPC = false,
         bool boundToHeaderWidth = false>
inline void originalPackingBitshufflePlaneVectors(
    const device uchar *source, const device uint *headers,
    volatile device uint *payload, device atomic_uint *errors,
    constant OriginalPackingShape &s, device uint *partialSums,
    device uint *partialMaximums, uint index,
    uint lane, uint simdWidth,
    uint groupThreads, const device uint *zeroTails = nullptr,
    device ulong4 *dpcPartials = nullptr) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane
        || groupThreads % 32u != 0u;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint checkpoint = index / (s.pixels / columns);
    if (checkpoint >= checkpoints) return;
    if (simd_any(atomic_load_explicit(errors, memory_order_relaxed) != 0u)) return;
    uint pixelGroup = ((index % (s.pixels / columns)) / 32u) * (32u * columns);
    uint dpcGroups = s.pixels / (32u * columns);
    uint dpcGroup = (index % (s.pixels / columns)) / 32u;
    Vector pixels;
    #pragma unroll
    for (uint part = 0u; part < columns; ++part) pixels[part] = pixelGroup + lane + part * 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    Vector offset, sum = Vector(0u), maximum = Vector(0u);
    #pragma unroll
    for (uint part = 0u; part < columns; ++part) {
        offset[part] = headers[pixels[part] * stride];
        if (checkpoint) offset[part] += headers[pixels[part] * stride + checkpoint];
    }
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        ulong scanTotal = 0ul, scanRow = 0ul, scanColumn = 0ul;
        Vector bits;
        #pragma unroll
        for (uint part = 0u; part < columns; ++part) {
            uint width = (headers[pixels[part] * stride + checkpoints + tile / 8u]
                >> ((tile % 8u) * 4u)) & 15u;
            bits[part] = width == 15u ? 16u : width;
        }
        if (low8Only) {
            bool unsupported = false;
            #pragma unroll
            for (uint part = 0u; part < columns; ++part)
                unsupported = unsupported || bits[part] > 8u;
            if (simd_any(unsupported)) {
                if (lane == 0u) atomic_fetch_or_explicit(errors, 8u, memory_order_relaxed);
                return;
            }
            // Some acquisition profiles publish uint8 working values, but the
            // source is still uint16. Validate the discarded eight planes directly in
            // the bitshuffle representation, then transpose only planes 0-7.
            // This preserves exactness while removing eight butterfly passes.
            bool highPlaneNonzero = false;
            for (int plane = 15; plane >= 8; --plane) {
                ulong scan = ulong(tile) * 32ul + lane;
                const device Vector *input = (const device Vector *)(source
                    + scan * s.pixels * 2ul + ulong(pixelGroup / 4096u) * 8192ul);
                uint sourceByte = uint(plane) * 512u + (pixelGroup % 4096u) / 8u;
                Vector value = input[sourceByte / (4u * columns)];
                highPlaneNonzero = highPlaneNonzero || any(value != Vector(0u));
            }
            if (simd_any(highPlaneNonzero)) {
                if (lane == 0u) atomic_fetch_or_explicit(errors, 8u, memory_order_relaxed);
                return;
            }
        }
        Vector candidates = Vector(0xffffffffu), tileMaximum = Vector(0u);
        uint maximumBits = 16u;
        if (boundToHeaderWidth) {
            maximumBits = bits[0];
            #pragma unroll
            for (uint part = 1u; part < columns; ++part)
                maximumBits = max(maximumBits, bits[part]);
            // Every lane in this SIMD group must execute the same number of
            // butterfly passes. Reduce the local header maximum across lanes;
            // pixels whose own width is smaller are masked below.
            for (uint shift = 16u; shift > 0u; shift >>= 1u)
                maximumBits = max(maximumBits, simd_shuffle_xor(maximumBits, shift));
            maximumBits = simd_broadcast_first(maximumBits);
        }
        uint zeroTail = 8192u;
        if (hasZeroTail) zeroTail = zeroTails[(tile * 32u + lane) * (s.pixels / 4096u) + pixelGroup / 4096u];
        for (int plane = int(low8Only ? min(8u, maximumBits) : maximumBits) - 1;
             plane >= 0; --plane) {
            ulong scan = ulong(tile) * 32ul + lane;
            const device Vector *input = (const device Vector *)(source
                + scan * s.pixels * 2ul + ulong(pixelGroup / 4096u) * 8192ul);
            uint sourceByte = uint(plane) * 512u + (pixelGroup % 4096u) / 8u;
            Vector value(0u);
            if (!boundToHeaderWidth || simd_any(any(plane < bits))) {
                value = sourceByte < zeroTail
                    ? input[sourceByte / (4u * columns)] : Vector(0u);
            }
            if (emitDPC) {
                #pragma unroll
                for (uint part = 0u; part < columns; ++part) {
                    uint word = value[part];
                    if (!word) continue;
                    ulong count = ulong(popcount(word));
                    uint pixel = pixelGroup + part * 32u;
                    uint rowIndex = pixel / s.columns;
                    uint columnIndex = pixel % s.columns;
                    uint localColumn = popcount(word & 0xaaaaaaaau)
                        + 2u * popcount(word & 0xccccccccu)
                        + 4u * popcount(word & 0xf0f0f0f0u)
                        + 8u * popcount(word & 0xff00ff00u)
                        + 16u * popcount(word & 0xffff0000u);
                    scanTotal += count << uint(plane);
                    scanRow += (count * ulong(rowIndex)) << uint(plane);
                    scanColumn += (count * ulong(columnIndex) + ulong(localColumn))
                        << uint(plane);
                }
            }
            Vector activeValue = value;
            if (widthAwareHighPlanes) {
                // `bits[part]` varies by SIMD lane, while `value[part]` is a
                // source word containing all 32 detector lanes. We may skip
                // the transpose only when this whole SIMD group is already
                // above the plane; otherwise transpose first and test the
                // per-pixel result below.
                #pragma unroll
                for (uint part = 0u; part < columns; ++part) {
                    bool allHigh = simd_all(uint(plane) >= bits[part]);
                    if (allHigh) {
                        if (simd_any(value[part] != 0u) && lane == 0u)
                            atomic_fetch_or_explicit(errors, 8u, memory_order_relaxed);
                        activeValue[part] = 0u;
                    }
                }
            }
            if (simd_any(any(activeValue != Vector(0u)))) {
                #pragma unroll
                for (uint shift = 1u; shift <= 16u; shift *= 2u) {
                    uint mask = 0xffffffffu / ((1u << shift) + 1u);
                    Vector other = simd_shuffle_xor(activeValue, shift);
                    activeValue = (lane & shift)
                        ? (activeValue & ~mask) | ((other & ~mask) >> shift)
                        : (activeValue & mask) | ((other & mask) << shift);
                }
            }
            if (boundToHeaderWidth) {
                // `value[part]` still contains the 32 detector pixels in this
                // source word. Apply the per-pixel width after the butterfly,
                // when each SIMD lane owns one detector pixel; masking before
                // the transpose would erase unrelated detector columns.
                #pragma unroll
                for (uint part = 0u; part < columns; ++part)
                    if (uint(plane) >= bits[part]) activeValue[part] = 0u;
            }
            if (widthAwareHighPlanes) {
                bool invalidHigh = false;
                #pragma unroll
                for (uint part = 0u; part < columns; ++part) {
                    if (uint(plane) >= bits[part]) {
                        invalidHigh = invalidHigh || activeValue[part] != 0u;
                        activeValue[part] = 0u;
                    }
                }
                if (invalidHigh)
                    atomic_fetch_or_explicit(errors, 8u, memory_order_relaxed);
            }
            #pragma unroll
            for (uint part = 0u; part < columns; ++part) {
                if (uint(plane) < bits[part]) payload[offset[part] + uint(plane)] = activeValue[part];
            }
            if (!skipStoreVerification) {
                #pragma unroll
                for (uint part = 0u; part < columns; ++part) {
                    uint restored = uint(plane) < bits[part]
                        ? payload[offset[part] + uint(plane)] : 0u;
                    if (restored != activeValue[part]) {
                        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                        valid = false;
                    }
                }
            }
            sum += popcount(activeValue) << uint(plane);
            Vector nextCandidates = candidates & activeValue;
            auto hasCandidate = nextCandidates != Vector(0u);
            tileMaximum |= select(Vector(0u), Vector(1u << uint(plane)), hasCandidate);
            candidates = select(candidates, nextCandidates, hasCandidate);
        }
        // The header pass established the exact per-pixel widths from the
        // same unchanged source. For this SIMD group, `maximumBits` proves
        // that every source word above it is zero, so those DPC planes are
        // omitted without changing the exact moments.
        if (emitDPC) {
            dpcPartials[ulong(tile * 32u + lane) * dpcGroups + dpcGroup] =
                ulong4(scanTotal, scanRow, scanColumn, 0ul);
        }
        maximum = max(maximum, tileMaximum);
        offset += bits;
    }
    if (valid) {
        #pragma unroll
        for (uint part = 0u; part < columns; ++part) {
            partialSums[ulong(checkpoint) * s.pixels + pixels[part]] = sum[part];
            partialMaximums[ulong(checkpoint) * s.pixels + pixels[part]] = maximum[part];
        }
    }
}


kernel void original_packing_bitshuffle_planes_vector4_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u>(source, headers, payload, errors, s,
        partialSums, partialMaximums, index, lane, simdWidth, groupThreads);
}

// Diagnostic ceiling only. Source planes and all summaries remain checked;
// this omits the per-word payload readback so the cost of store verification
// can be measured before any trusted-plan contract is considered.
kernel void original_packing_bitshuffle_planes_vector4_unchecked_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, false, true>(source, headers, payload, errors, s,
        partialSums, partialMaximums, index, lane, simdWidth, groupThreads);
}

// Diagnostic width-aware variant. Persisted layout headers already state the
// exact width of each detector tile. This kernel still rejects any nonzero bit
// above that width, but masks those planes before the butterfly so narrow BTO
// tiles do not pay for 16 full transposes.
kernel void original_packing_bitshuffle_planes_vector4_widthaware_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, false, false, true>(source, headers, payload,
        errors, s, partialSums, partialMaximums, index, lane, simdWidth, groupThreads);
}

// Exact diagnostic specialization for uint8 BTO working values. The source
// remains uint16 and all high planes are checked before the low-plane
// transpose; no source information is silently discarded.
kernel void original_packing_bitshuffle_planes_vector4_low8_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, false, false, false, true>(source, headers,
        payload, errors, s, partialSums, partialMaximums, index, lane, simdWidth, groupThreads);
}

kernel void original_packing_bitshuffle_planes_zero_tail_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], const device uint *zeroTails [[buffer(7)]],
    uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]], uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, true>(source, headers, payload, errors, s,
        partialSums, partialMaximums, index, lane, simdWidth, groupThreads, zeroTails);
}

// Diagnostic direct-scratch variant. The exact payload and per-pixel summary
// work is unchanged, but each SIMD group also emits one DPC partial for every
// scan it reconstructs. A later small reduction replaces the separate DPC
// reread of the complete 3 GiB bitshuffle scratch.
kernel void original_packing_bitshuffle_planes_vector4_dpc_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], device ulong4 *dpcPartials [[buffer(7)]],
    uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]], uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, false, false, false, false, true>(source,
        headers, payload, errors, s, partialSums, partialMaximums, index, lane, simdWidth,
        groupThreads, nullptr, dpcPartials);
}

// Exact width-bounded companion. Header generation has already established
// the maximum bit width for every detector pixel in every scan tile. This
// variant uses the SIMD-group maximum as a uniform loop bound and masks each
// narrower component after the butterfly, so no unproven source plane is
// discarded and the packed output remains byte-for-byte identical.
kernel void original_packing_bitshuffle_planes_vector4_widthbounded_dpc_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], device ulong4 *dpcPartials [[buffer(7)]],
    uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]], uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, false, false, false, false, true, true>(
        source, headers, payload, errors, s, partialSums, partialMaximums, index, lane,
        simdWidth, groupThreads, nullptr, dpcPartials);
}

// Width-bounded payload-only diagnostic. This isolates the variable-width
// payload optimization from fused DPC accounting so parity can identify which
// contract is responsible for any mismatch.
kernel void original_packing_bitshuffle_planes_vector4_widthbounded_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, false, false, false, false, false, true>(
        source, headers, payload, errors, s, partialSums, partialMaximums, index, lane,
        simdWidth, groupThreads, nullptr, nullptr);
}

// Zero-tail companion for the direct first-load path. The decoder records the
// first byte after a validated terminal zero match; reading only the live
// prefix avoids touching stale bytes in a reused private scratch buffer while
// preserving the same packed payload, summaries, and exact DPC moments.
kernel void original_packing_bitshuffle_planes_vector4_zero_tail_dpc_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]], const device uint *zeroTails [[buffer(7)]],
    device ulong4 *dpcPartials [[buffer(8)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]],
    uint groupThreads [[threads_per_threadgroup]]) {
    originalPackingBitshufflePlaneVectors<uint4, 4u, true, false, false, false, true>(source,
        headers, payload, errors, s, partialSums, partialMaximums, index, lane, simdWidth,
        groupThreads, zeroTails, dpcPartials);
}

kernel void original_packing_reduce_bitshuffle_dpc_fused(
    const device ulong4 *partialDPC [[buffer(0)]], device ulong4 *moments [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    uint scan [[thread_position_in_grid]]) {
    uint groups = s.pixels / 128u;
    if (!groups || s.pixels % 128u || scan >= s.scans) {
        if (scan == 0u) atomic_fetch_or_explicit(errors, 4u, memory_order_relaxed);
        return;
    }
    ulong4 result = ulong4(0ul);
    for (uint group = 0u; group < groups; ++group)
        result += partialDPC[ulong(scan) * groups + group];
    moments[scan] = result;
}

kernel void original_packing_bitshuffle_verified_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    device uint *partialMaximums [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (!s.pixels || !s.scans || s.scans % 32u || (s.sourceBytes != 2u || s.pixels % 4096u)) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    // At most 1024 full uint16 counts, bounded by 67,107,840.
    uint sum = 0u, maximum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_bitshuffle_count(source, tile * 32u + sample, pixel, s);
            originals[sample] = value;
            sum += value;
            maximum = max(maximum, value);
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint bit = sample * bits, shift = bit % 32u;
            ulong value = bits ? ulong(payload[offset + bit / 32u]) >> shift : 0u;
            if (shift + bits > 32u) value |= ulong(payload[offset + bit / 32u + 1u]) << (32u - shift);
            if ((value & mask) != originals[sample]) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
        }
        offset += bits;
    }
    // A different worker may fail concurrently. The following reduction must
    // run in a later encoder and observe the completed global error counter.
    if (valid) {
        partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
        partialMaximums[ulong(checkpoint) * s.pixels + pixel] = maximum;
    }
}

kernel void original_packing_reduce_bitshuffle_summary(
    const device uint *partialSums [[buffer(0)]], const device uint *headers [[buffer(1)]],
    device ulong *sums [[buffer(2)]], device uint *maximumWidths [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]], constant OriginalPackingShape &s [[buffer(5)]],
    const device uint *partialMaximums [[buffer(6)]], device uint *maximumCounts [[buffer(7)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels || atomic_load_explicit(errors, memory_order_relaxed)) return;
    if (!s.scans || s.scans % 32u || s.sourceBytes != 2u || s.pixels % 4096u) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    ulong sum = 0ul;
    // These cumulative buffers start at zero for each new acquisition and
    // contain only previously validated current-load windows, never cached
    // maxima inferred from wider-than-minimal stored encoding headers.
    uint maximumCount = maximumCounts[pixel];
    for (uint checkpoint = 0u; checkpoint < checkpoints; ++checkpoint) {
        ulong index = ulong(checkpoint) * s.pixels + pixel;
        sum += ulong(partialSums[index]);
        maximumCount = max(maximumCount, partialMaximums[index]);
    }
    uint maximumWidth = maximumWidths[pixel];
    for (uint tile = 0u; tile < tiles; ++tile) {
        uint nibble = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        maximumWidth = max(maximumWidth, nibble == 15u ? 16u : nibble);
    }
    sums[pixel] = sum;
    maximumWidths[pixel] = maximumWidth;
    maximumCounts[pixel] = maximumCount;
}



// Private cached-plan experiment. Range validation must precede this writer
// in a separate encoder with immutable headers and the same error buffer.
kernel void original_packing_values_verified_checkpoints_summary(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], device uint *partialSums [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (!s.pixels || !s.scans || s.scans % 32u || (s.sourceBytes != 1u && s.sourceBytes != 2u && s.sourceBytes != 4u)) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint pixel = index % s.pixels, checkpoint = index / s.pixels;
    if (checkpoint >= checkpoints || atomic_load_explicit(errors, memory_order_relaxed)) return;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    if (checkpoint) offset += headers[pixel * stride + checkpoint];
    // At most 1024 full uint16 counts, bounded by 67,107,840.
    uint sum = 0u;
    bool valid = true;
    for (uint tile = checkpoint * 32u; tile < min(tiles, (checkpoint + 1u) * 32u); ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        if (bits >= 15u) bits = s.sourceBytes == 4u ? 32u : 16u;
        uint originals[32];
        uint packed = 0u, occupied = 0u, outputWord = 0u;
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint value = original_count(source, tile * 32u + sample, pixel, s);
            originals[sample] = value;
            sum += value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32u) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32u;
                    packed = occupied ? value >> (bits - occupied) : 0u;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0u; sample < 32u; ++sample) {
            uint bit = sample * bits, shift = bit % 32u;
            ulong value = bits ? ulong(payload[offset + bit / 32u]) >> shift : 0u;
            if (shift + bits > 32u) value |= ulong(payload[offset + bit / 32u + 1u]) << (32u - shift);
            if ((value & mask) != originals[sample]) {
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
                valid = false;
            }
        }
        offset += bits;
    }
    // A different worker may fail concurrently. The following reduction must
    // run in a later encoder and observe the completed global error counter.
    if (valid) partialSums[ulong(checkpoint) * s.pixels + pixel] = sum;
}

kernel void original_packing_reduce_verified_summary(
    const device uint *partialSums [[buffer(0)]], const device uint *headers [[buffer(1)]],
    device ulong *sums [[buffer(2)]], device uint *maximumWidths [[buffer(3)]],
    device atomic_uint *errors [[buffer(4)]], constant OriginalPackingShape &s [[buffer(5)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels || atomic_load_explicit(errors, memory_order_relaxed)) return;
    if (!s.scans || s.scans % 32u || (s.sourceBytes != 1u && s.sourceBytes != 2u && s.sourceBytes != 4u)) {
        atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    ulong sum = 0ul;
    for (uint checkpoint = 0u; checkpoint < checkpoints; ++checkpoint) {
        sum += ulong(partialSums[ulong(checkpoint) * s.pixels + pixel]);
    }
    // Infer the encoding width from the validated payload plan, never a
    // cached calibration/product summary. This also respects the all-zero
    // stream's one-bit sentinel tile and the nibble 15 -> width 16 convention.
    uint maximumWidth = maximumWidths[pixel];
    for (uint tile = 0u; tile < tiles; ++tile) {
        uint nibble = (headers[pixel * stride + checkpoints + tile / 8u] >> ((tile % 8u) * 4u)) & 15u;
        maximumWidth = max(maximumWidth, nibble == 15u ? 16u : nibble);
    }
    sums[pixel] = sum;
    maximumWidths[pixel] = maximumWidth;
}

// Each lane owns a detector pixel. Adjacent lanes read adjacent source counts.
kernel void original_packing_headers(
    const device uchar *source [[buffer(0)]], device uint *headers [[buffer(1)]],
    device uint *sizes [[buffer(2)]], device ulong *sums [[buffer(3)]],
    device uint *maximumWidths [[buffer(4)]],
    constant OriginalPackingShape &s [[buffer(5)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint words = 0;
    ulong sum = 0;
    uint maximumWidth = maximumWidths[pixel];
    for (uint tile = 0; tile < tiles; ++tile) {
        if (tile % 32 == 0) headers[pixel * stride + tile / 32] = words;
        uint maximum = 0;
        for (uint sample = 0; sample < 32; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            maximum = max(maximum, value); sum += value;
        }
        uint bits = maximum == 0 ? 0 : 32 - clz(maximum);
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        maximumWidth = max(maximumWidth, bits);
        uint word = pixel * stride + checkpoints + tile / 8;
        if (tile % 8 == 0) headers[word] = 0;
        headers[word] |= min(bits, 15u) << ((tile % 8) * 4);
        words += bits;
    }
    sizes[pixel] = words;
    sums[pixel] = sum;
    maximumWidths[pixel] = maximumWidth;
}

// First-load direct-scratch header pass. The scalar checked BSLZ4 decoder
// already emits the exact uint16 bitshuffle blocks used by the resident plane
// packer. Reconstruct the same per-tile widths and sums from those planes so
// first load can skip the dense uint16 unshuffle volume. One SIMD group owns a
// contiguous group of detector pixels; its lanes are the 32 scan samples in a
// bitshuffle tile and the butterfly below restores each pixel's 32-bit plane.
// This kernel is diagnostic-only until its full-count and packed-layout gates
// have passed on every supported original source.
kernel void original_packing_bitshuffle_headers(
    const device uchar *source [[buffer(0)]], device uint *headers [[buffer(1)]],
    device uint *sizes [[buffer(2)]], device ulong *sums [[buffer(3)]],
    device uint *maximumWidths [[buffer(4)]], device atomic_uint *errors [[buffer(5)]],
    device atomic_uint *highPlaneWords [[buffer(6)]], constant OriginalPackingShape &s [[buffer(7)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]]) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint pixel = index;
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32u;
    uint checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint words = 0u;
    ulong sum = 0ul;
    uint maximumWidth = maximumWidths[pixel];
    for (uint tile = 0u; tile < tiles; ++tile) {
        if (tile % 32u == 0u) headers[pixel * stride + tile / 32u] = words;
        uint tileMaximum = 0u;
        for (int plane = 15; plane >= 0; --plane) {
            uint scan = tile * 32u + lane;
            uint detectorWord = (pixel % 4096u) / 32u;
            const device uint *input = (const device uint *)(source
                + ulong(scan) * s.pixels * 2ul
                + ulong(pixel / 4096u) * 8192ul);
            uint value = input[uint(plane) * 128u + detectorWord];
            // A zero bit plane remains zero after transpose. Vote across the
            // complete 32-scan tile before skipping the five shuffle stages;
            // every lane must take the same branch to preserve SIMD lockstep.
            if (simd_any(value != 0u)) {
                #pragma unroll
                for (uint shift = 1u; shift <= 16u; shift *= 2u) {
                    uint mask = 0xffffffffu / ((1u << shift) + 1u);
                    uint other = simd_shuffle_xor(value, shift);
                    value = (lane & shift)
                        ? (value & ~mask) | ((other & ~mask) >> shift)
                        : (value & mask) | ((other & mask) << shift);
                }
            }
            sum += ulong(popcount(value)) << uint(plane);
            if (value != 0u) tileMaximum |= 1u << uint(plane);
        }
        uint bits = tileMaximum == 0u ? 0u : 32u - clz(tileMaximum);
        if (bits >= 15u) bits = 16u;
        maximumWidth = max(maximumWidth, bits);
        uint headerWord = pixel * stride + checkpoints + tile / 8u;
        if (tile % 8u == 0u) headers[headerWord] = 0u;
        headers[headerWord] |= min(bits, 15u) << ((tile % 8u) * 4u);
        words += bits;
    }
    sizes[pixel] = words;
    sums[pixel] = sum;
    maximumWidths[pixel] = maximumWidth;
    if (maximumWidth > 8u)
        atomic_fetch_or_explicit(&highPlaneWords[pixel / 32u], 1u, memory_order_relaxed);
}

// Diagnostic direct-scratch fusion. One SIMD group owns 128 detector pixels
// for the complete packing window, transposes each source plane once, and
// writes a fixed 16-word slot per tile. The host later compacts only the
// declared width for each tile after prefixing the exact variable payload.
// This removes the separate header transpose followed by a second transpose
// in the variable-width payload writer. DPC partials are emitted from the
// pre-transpose source words, where one lane still denotes one scan.
kernel void original_packing_bitshuffle_direct_combined(
    const device uchar *source [[buffer(0)]], device uint *headers [[buffer(1)]],
    device uint *sizes [[buffer(2)]], device ulong *sums [[buffer(3)]],
    device uint *maximumWidths [[buffer(4)]], device atomic_uint *errors [[buffer(5)]],
    device uint *fixedPayload [[buffer(6)]], device ulong4 *dpcPartials [[buffer(7)]],
    constant OriginalPackingShape &s [[buffer(8)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]]) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint pixelGroup = ((index % (s.pixels / 4u)) / 32u) * 128u;
    uint dpcGroups = s.pixels / 128u;
    uint dpcGroup = (index % (s.pixels / 4u)) / 32u;
    uint4 words(0u), total(0u), maximumWidth(0u);
    // The complete width plan is written by one lane per pixel group. No
    // checkpoint split is needed because all tile widths are produced here.
    for (uint tile = 0u; tile < tiles; ++tile) {
        if (tile % 32u == 0u) {
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part)
                headers[(pixelGroup + lane + part * 32u) * stride + tile / 32u] = words[part];
        }
        uint tileMaximum[4] = {0u, 0u, 0u, 0u};
        ulong scanTotal = 0ul, scanRow = 0ul, scanColumn = 0ul;
        for (int plane = 15; plane >= 0; --plane) {
            ulong scan = ulong(tile) * 32ul + lane;
            const device uint4 *input = (const device uint4 *)(source
                + scan * s.pixels * 2ul + ulong(pixelGroup / 4096u) * 8192ul);
            uint sourceByte = uint(plane) * 512u + (pixelGroup % 4096u) / 8u;
            uint4 value = input[sourceByte / 16u];
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                uint word = value[part];
                if (word) {
                    ulong count = ulong(popcount(word));
                    uint pixel = pixelGroup + part * 32u;
                    uint rowIndex = pixel / s.columns;
                    uint columnIndex = pixel % s.columns;
                    uint localColumn = popcount(word & 0xaaaaaaaau)
                        + 2u * popcount(word & 0xccccccccu)
                        + 4u * popcount(word & 0xf0f0f0f0u)
                        + 8u * popcount(word & 0xff00ff00u)
                        + 16u * popcount(word & 0xffff0000u);
                    scanTotal += count << uint(plane);
                    scanRow += (count * ulong(rowIndex)) << uint(plane);
                    scanColumn += (count * ulong(columnIndex) + ulong(localColumn))
                        << uint(plane);
                }
            }
            if (simd_any(any(value != uint4(0u)))) {
                #pragma unroll
                for (uint shift = 1u; shift <= 16u; shift *= 2u) {
                    uint mask = 0xffffffffu / ((1u << shift) + 1u);
                    uint4 other = simd_shuffle_xor(value, shift);
                    value = (lane & shift)
                        ? (value & ~mask) | ((other & ~mask) >> shift)
                        : (value & mask) | ((other & mask) << shift);
                }
            }
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                uint pixel = pixelGroup + lane + part * 32u;
                uint fixedOffset = (pixel * tiles + tile) * 16u + uint(plane);
                fixedPayload[fixedOffset] = value[part];
                total[part] += popcount(value[part]) << uint(plane);
                tileMaximum[part] |= value[part] ? (1u << uint(plane)) : 0u;
            }
        }
        dpcPartials[ulong(tile * 32u + lane) * dpcGroups + dpcGroup] =
            ulong4(scanTotal, scanRow, scanColumn, 0ul);
        #pragma unroll
        for (uint part = 0u; part < 4u; ++part) {
            uint pixel = pixelGroup + lane + part * 32u;
            uint bits = tileMaximum[part] ? 32u - clz(tileMaximum[part]) : 0u;
            if (bits >= 15u) bits = 16u;
            maximumWidth[part] = max(maximumWidth[part], bits);
            uint headerWord = pixel * stride + checkpoints + tile / 8u;
            if (tile % 8u == 0u) headers[headerWord] = 0u;
            headers[headerWord] |= min(bits, 15u) << ((tile % 8u) * 4u);
            words[part] += bits;
        }
    }
    #pragma unroll
    for (uint part = 0u; part < 4u; ++part) {
        uint pixel = pixelGroup + lane + part * 32u;
        sizes[pixel] = words[part];
        sums[pixel] = ulong(total[part]);
        maximumWidths[pixel] = maximumWidth[part];
    }
}

// Compact the exact fixed-plane intermediate into the variable-width
// resident payload after the host has prefixed the validated headers.
kernel void original_packing_compact_fixed_planes(
    const device uint *fixedPayload [[buffer(0)]], const device uint *headers [[buffer(1)]],
    device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels || atomic_load_explicit(errors, memory_order_relaxed)) return;
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint offset = headers[pixel * stride];
    for (uint tile = 0u; tile < tiles; ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8u]
            >> ((tile % 8u) * 4u)) & 15u;
        if (bits == 15u) bits = 16u;
        uint fixedBase = (pixel * tiles + tile) * 16u;
        for (uint plane = 0u; plane < bits; ++plane)
            payload[offset + plane] = fixedPayload[fixedBase + plane];
        offset += bits;
    }
}

// Diagnostic header-only vector4 variant. It keeps the established variable
// payload writer and exact header contract, but processes four adjacent
// detector words per SIMD lane so header generation has the same source-load
// topology as the retained vectorized payload pass.
kernel void original_packing_bitshuffle_headers_vector4(
    const device uchar *source [[buffer(0)]], device uint *headers [[buffer(1)]],
    device uint *sizes [[buffer(2)]], device ulong *sums [[buffer(3)]],
    device uint *maximumWidths [[buffer(4)]], device atomic_uint *errors [[buffer(5)]],
    device atomic_uint *highPlaneWords [[buffer(6)]], constant OriginalPackingShape &s [[buffer(7)]],
    const device uint *zeroTails [[buffer(8)]], constant uint &zeroTailEnabled [[buffer(9)]],
    uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
    bool invalid = !s.pixels || !s.scans || s.scans % 32u || s.sourceBytes != 2u
        || s.pixels % 4096u || simdWidth != 32u || index % 32u != lane;
    if (simd_any(invalid)) {
        if (lane == 0u) atomic_fetch_or_explicit(errors, 2u, memory_order_relaxed);
        return;
    }
    uint tiles = s.scans / 32u, checkpoints = (tiles + 31u) / 32u;
    uint stride = checkpoints + (tiles + 7u) / 8u;
    uint pixelGroup = ((index % (s.pixels / 4u)) / 32u) * 128u;
    uint4 words(0u), total(0u), maximumWidth(0u);
    for (uint tile = 0u; tile < tiles; ++tile) {
        if (tile % 32u == 0u) {
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part)
                headers[(pixelGroup + lane + part * 32u) * stride + tile / 32u] = words[part];
        }
        uint4 tileMaximum(0u);
        for (int plane = 15; plane >= 0; --plane) {
            ulong scan = ulong(tile) * 32ul + lane;
            const device uint4 *input = (const device uint4 *)(source
                + scan * s.pixels * 2ul + ulong(pixelGroup / 4096u) * 8192ul);
            uint sourceByte = uint(plane) * 512u + (pixelGroup % 4096u) / 8u;
            uint zeroTail = zeroTailEnabled
                ? zeroTails[scan * (s.pixels / 4096u) + pixelGroup / 4096u] : 8192u;
            uint4 value = sourceByte < zeroTail ? input[sourceByte / 16u] : uint4(0u);
            if (simd_any(any(value != uint4(0u)))) {
                #pragma unroll
                for (uint shift = 1u; shift <= 16u; shift *= 2u) {
                    uint mask = 0xffffffffu / ((1u << shift) + 1u);
                    uint4 other = simd_shuffle_xor(value, shift);
                    value = (lane & shift)
                        ? (value & ~mask) | ((other & ~mask) >> shift)
                        : (value & mask) | ((other & mask) << shift);
                }
            }
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                total[part] += popcount(value[part]) << uint(plane);
                if (value[part]) tileMaximum[part] |= 1u << uint(plane);
            }
        }
        #pragma unroll
        for (uint part = 0u; part < 4u; ++part) {
            uint pixel = pixelGroup + lane + part * 32u;
            uint bits = tileMaximum[part] ? 32u - clz(tileMaximum[part]) : 0u;
            if (bits >= 15u) bits = 16u;
            maximumWidth[part] = max(maximumWidth[part], bits);
            uint headerWord = pixel * stride + checkpoints + tile / 8u;
            if (tile % 8u == 0u) headers[headerWord] = 0u;
            headers[headerWord] |= min(bits, 15u) << ((tile % 8u) * 4u);
            words[part] += bits;
        }
    }
    #pragma unroll
    for (uint part = 0u; part < 4u; ++part) {
        uint pixel = pixelGroup + lane + part * 32u;
        sizes[pixel] = words[part];
        sums[pixel] = ulong(total[part]);
        maximumWidths[pixel] = maximumWidth[part];
        if (maximumWidth[part] > 8u)
            atomic_fetch_or_explicit(&highPlaneWords[pixel / 32u], 1u, memory_order_relaxed);
    }
}

kernel void original_packing_values(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    device uint *payload [[buffer(2)]], constant OriginalPackingShape &s [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    for (uint tile = 0; tile < tiles; ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        for (uint word = 0; word < bits; ++word) payload[offset + word] = 0;
        for (uint sample = 0; sample < 32 && bits; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            uint bit = sample * bits, word = bit / 32, shift = bit % 32;
            payload[offset + word] |= value << shift;
            if (shift + bits > 32) payload[offset + word + 1] |= value >> (32 - shift);
        }
        offset += bits;
    }
}

// Verify every packed count against decoded source before publishing a shard.
kernel void original_packing_verify(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    const device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    for (uint tile = 0; tile < tiles; ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0; sample < 32; ++sample) {
            uint bit = sample * bits, shift = bit % 32;
            ulong value = bits ? ulong(payload[offset + bit / 32]) >> shift : 0;
            if (shift + bits > 32) value |= ulong(payload[offset + bit / 32 + 1]) << (32 - shift);
            if ((value & mask) != original_count(source, tile * 32 + sample, pixel, s))
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        }
        offset += bits;
    }
}

// Exact intensity and row/column weighted sums directly from source bit planes.
// A 32-pixel word stays within one detector row (columns must be divisible by 32).
// Five bit-position masks recover the column weight without dense unshuffle.
inline ulong originalPackingDPCSumUInt64(ulong value) {
    for (uint delta = 16u; delta; delta >>= 1u) {
        value += ulong(simd_shuffle_down(uint(value), delta))
            | (ulong(simd_shuffle_down(uint(value >> 32u), delta)) << 32u);
    }
    return value;
}

kernel void original_packing_bitshuffle_dpc(
    const device uchar *source [[buffer(0)]], device ulong4 *dpc [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
    if (s.sourceBytes != 2u || s.pixels % 4096u || !s.columns || s.columns % 32u
        || simdWidth != 32u) {
        if (index == 0u) atomic_fetch_or_explicit(errors, 4u, memory_order_relaxed);
        return;
    }
    uint scan = index / 32u;
    if (scan >= s.scans) return;
    ulong total = 0ul, row = 0ul, column = 0ul;
    for (uint wordIndex = lane; wordIndex < s.pixels / 32u; wordIndex += 32u) {
        uint pixel = wordIndex * 32u;
        const device uint *block = (const device uint *)(source
            + ulong(scan) * s.pixels * 2ul + ulong(pixel / 4096u) * 8192ul);
        uint rowIndex = pixel / s.columns, columnIndex = pixel % s.columns;
        for (uint plane = 0u; plane < 16u; ++plane) {
            uint word = block[plane * 128u + (pixel % 4096u) / 32u];
            if (!word) continue;
            ulong count = ulong(popcount(word));
            uint localColumn = popcount(word & 0xaaaaaaaau)
                + 2u * popcount(word & 0xccccccccu)
                + 4u * popcount(word & 0xf0f0f0f0u)
                + 8u * popcount(word & 0xff00ff00u)
                + 16u * popcount(word & 0xffff0000u);
            total += count << plane;
            row += (count * rowIndex) << plane;
            column += (count * columnIndex + ulong(localColumn)) << plane;
        }
    }
    for (uint delta = 16u; delta; delta >>= 1u) {
        total += ulong(simd_shuffle_down(uint(total), delta))
            | (ulong(simd_shuffle_down(uint(total >> 32u), delta)) << 32u);
        row += ulong(simd_shuffle_down(uint(row), delta))
            | (ulong(simd_shuffle_down(uint(row >> 32u), delta)) << 32u);
        column += ulong(simd_shuffle_down(uint(column), delta))
            | (ulong(simd_shuffle_down(uint(column >> 32u), delta)) << 32u);
    }
    if (lane == 0u) dpc[scan] = ulong4(total, row, column, 0ul);
}

// Exact diagnostic DPC variant. The header pass has already inspected every
// source bit and records the rare 32-pixel words containing a value wider than
// eight bits. All other words are proven zero in planes 8-15, so the DPC
// reduction can avoid rereading those eight planes without changing results.
kernel void original_packing_bitshuffle_dpc_pruned_high_planes(
    const device uchar *source [[buffer(0)]], device ulong4 *dpc [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    const device uint *highPlaneWords [[buffer(4)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]]) {
    if (s.sourceBytes != 2u || s.pixels % 4096u || !s.columns || s.columns % 32u
        || simdWidth != 32u) {
        if (index == 0u) atomic_fetch_or_explicit(errors, 4u, memory_order_relaxed);
        return;
    }
    uint scan = index / 32u;
    if (scan >= s.scans) return;
    ulong total = 0ul, row = 0ul, column = 0ul;
    for (uint wordIndex = lane; wordIndex < s.pixels / 32u; wordIndex += 32u) {
        uint planeLimit = highPlaneWords[wordIndex] ? 16u : 8u;
        const device uint *block = (const device uint *)(source
            + ulong(scan) * s.pixels * 2ul + ulong(wordIndex / 128u) * 8192ul);
        uint detectorWord = wordIndex % 128u;
        uint rowIndex = (wordIndex * 32u) / s.columns;
        uint columnIndex = (wordIndex * 32u) % s.columns;
        for (uint plane = 0u; plane < planeLimit; ++plane) {
            uint word = block[plane * 128u + detectorWord];
            if (!word) continue;
            ulong count = ulong(popcount(word));
            uint localColumn = popcount(word & 0xaaaaaaaau)
                + 2u * popcount(word & 0xccccccccu)
                + 4u * popcount(word & 0xf0f0f0f0u)
                + 8u * popcount(word & 0xff00ff00u)
                + 16u * popcount(word & 0xffff0000u);
            total += count << plane;
            row += (count * rowIndex) << plane;
            column += (count * columnIndex + ulong(localColumn)) << plane;
        }
    }
    for (uint delta = 16u; delta; delta >>= 1u) {
        total += ulong(simd_shuffle_down(uint(total), delta))
            | (ulong(simd_shuffle_down(uint(total >> 32u), delta)) << 32u);
        row += ulong(simd_shuffle_down(uint(row), delta))
            | (ulong(simd_shuffle_down(uint(row >> 32u), delta)) << 32u);
        column += ulong(simd_shuffle_down(uint(column), delta))
            | (ulong(simd_shuffle_down(uint(column >> 32u), delta)) << 32u);
    }
    if (lane == 0u) dpc[scan] = ulong4(total, row, column, 0ul);
}

// Exact diagnostic DPC variant with four independent SIMD groups per scan.
// The original kernel gives one SIMD group 36 detector words per lane. This
// layout gives each group a disjoint quarter of the detector words, reducing
// that serial loop to nine words per lane while preserving the same 64-bit
// subgroup and cross-group reductions.
kernel void original_packing_bitshuffle_dpc_wide_groups(
    const device uchar *source [[buffer(0)]], device ulong4 *dpc [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    uint scan [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]], uint threads [[threads_per_threadgroup]]) {
    if (s.sourceBytes != 2u || s.pixels % 4096u || !s.columns || s.columns % 32u
        || simdWidth != 32u || threads != 128u || simdgroup >= 4u) {
        if (simdgroup == 0u && lane == 0u)
            atomic_fetch_or_explicit(errors, 4u, memory_order_relaxed);
        return;
    }
    if (scan >= s.scans) return;
    ulong total = 0ul, row = 0ul, column = 0ul;
    uint wordCount = s.pixels / 32u;
    for (uint wordIndex = simdgroup * 32u + lane;
         wordIndex < wordCount; wordIndex += 128u) {
        const device uint *block = (const device uint *)(source
            + ulong(scan) * s.pixels * 2ul + ulong(wordIndex / 128u) * 8192ul);
        uint rowIndex = (wordIndex * 32u) / s.columns;
        uint columnIndex = (wordIndex * 32u) % s.columns;
        for (uint plane = 0u; plane < 16u; ++plane) {
            uint word = block[plane * 128u + (wordIndex % 128u)];
            if (!word) continue;
            ulong count = ulong(popcount(word));
            uint localColumn = popcount(word & 0xaaaaaaaau)
                + 2u * popcount(word & 0xccccccccu)
                + 4u * popcount(word & 0xf0f0f0f0u)
                + 8u * popcount(word & 0xff00ff00u)
                + 16u * popcount(word & 0xffff0000u);
            total += count << plane;
            row += (count * rowIndex) << plane;
            column += (count * columnIndex + ulong(localColumn)) << plane;
        }
    }
    total = originalPackingDPCSumUInt64(total);
    row = originalPackingDPCSumUInt64(row);
    column = originalPackingDPCSumUInt64(column);
    threadgroup ulong4 partial[4];
    if (lane == 0u) partial[simdgroup] = ulong4(total, row, column, 0ul);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u && lane == 0u)
        dpc[scan] = partial[0] + partial[1] + partial[2] + partial[3];
}

kernel void original_packing_moments(
    const device uchar *source [[buffer(0)]], device ulong4 *moments [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdWidth [[threads_per_simdgroup]]) {
    uint scan = index / simdWidth;
    if (scan >= s.scans) return;
    ulong total = 0, row = 0, column = 0;
    for (uint pixel = lane; pixel < s.pixels; pixel += simdWidth) {
        ulong value = original_count(source, scan, pixel, s);
        total += value; row += value * (pixel / s.columns); column += value * (pixel % s.columns);
    }
    // Read adjacent counts together. Shuffle the two halves separately, then
    // add in 64 bits so carries preserve high-count DPC products exactly.
    for (uint delta = simdWidth / 2; delta; delta >>= 1) {
        total += ulong(simd_shuffle_down(uint(total), delta))
               | (ulong(simd_shuffle_down(uint(total >> 32), delta)) << 32);
        row += ulong(simd_shuffle_down(uint(row), delta))
             | (ulong(simd_shuffle_down(uint(row >> 32), delta)) << 32);
        column += ulong(simd_shuffle_down(uint(column), delta))
                | (ulong(simd_shuffle_down(uint(column >> 32), delta)) << 32);
    }
    if (lane == 0) moments[scan] = ulong4(total, row, column, 0);
}

kernel void original_packing_u8(
    const device uchar *source [[buffer(0)]], device uchar *output [[buffer(1)]],
    constant OriginalPackingShape &s [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < s.scans * s.pixels)
        output[index] = uchar(original_count(source, index / s.pixels, index % s.pixels, s));
}

// Prove the declared payload ranges form a complete, disjoint partition before
// permitting any writer to use them. Checkpoints must match the tile widths.
kernel void original_packing_validate_ranges(
    const device uint *headers [[buffer(0)]], device atomic_uint *errors [[buffer(1)]],
    const device uint *payloadWords [[buffer(2)]],
    constant OriginalPackingShape &s [[buffer(3)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint base = headers[pixel * stride];
    uint end = pixel + 1 < s.pixels ? headers[(pixel + 1) * stride] : payloadWords[0];
    bool invalid = base > end || end > payloadWords[0] || (pixel == 0 && base != 0);
    ulong words = 0;
    for (uint tile = 0; tile < tiles; ++tile) {
        if (tile && tile % 32 == 0 && ulong(headers[pixel * stride + tile / 32]) != words)
            invalid = true;
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        words += bits == 15 ? (s.sourceBytes == 4 ? 32 : 16) : bits;
    }
    if (ulong(base) + words != ulong(end)) invalid = true;
    if (invalid) atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
}

// The caller must encode the range validator first, with immutable headers and
// the same error counter. Each tile then checks every actual stored count via
// volatile readback against its original input, including zero-width tiles.
kernel void original_packing_values_verified(
    const device uchar *source [[buffer(0)]], const device uint *headers [[buffer(1)]],
    volatile device uint *payload [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
    constant OriginalPackingShape &s [[buffer(4)]], uint pixel [[thread_position_in_grid]]) {
    if (pixel >= s.pixels) return;
    if (atomic_load_explicit(errors, memory_order_relaxed) != 0) return;
    uint tiles = s.scans / 32, checkpoints = (tiles + 31) / 32;
    uint stride = checkpoints + (tiles + 7) / 8;
    uint offset = headers[pixel * stride];
    for (uint tile = 0; tile < tiles; ++tile) {
        uint bits = (headers[pixel * stride + checkpoints + tile / 8] >> ((tile % 8) * 4)) & 15;
        if (bits >= 15) bits = s.sourceBytes == 4 ? 32 : 16;
        uint originals[32];
        uint packed = 0, occupied = 0, outputWord = 0;
        for (uint sample = 0; sample < 32; ++sample) {
            uint value = original_count(source, tile * 32 + sample, pixel, s);
            originals[sample] = value;
            if (bits) {
                packed |= value << occupied;
                occupied += bits;
                if (occupied >= 32) {
                    payload[offset + outputWord++] = packed;
                    occupied -= 32;
                    packed = occupied ? value >> (bits - occupied) : 0;
                }
            }
        }
        uint mask = uint((1ul << bits) - 1ul);
        for (uint sample = 0; sample < 32; ++sample) {
            uint bit = sample * bits, shift = bit % 32;
            ulong value = bits ? ulong(payload[offset + bit / 32]) >> shift : 0;
            if (shift + bits > 32) value |= ulong(payload[offset + bit / 32 + 1]) << (32 - shift);
            if ((value & mask) != originals[sample])
                atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);
        }
        offset += bits;
    }
}
