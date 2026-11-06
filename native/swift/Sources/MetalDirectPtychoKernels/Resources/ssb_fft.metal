#include <metal_stdlib>
using namespace metal;

inline float2 complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline uint reverse_base8_digits(uint value, uint digits) {
    uint reversed = 0u;
    for (uint digit = 0u; digit < digits; ++digit) {
        reversed = (reversed << 3u) | (value & 7u);
        value >>= 3u;
    }
    return reversed;
}

// 512 = 8^3. Reverse the three octal digits so each thread can load the
// first radix-8 inputs directly into registers. This removes the initial
// threadgroup-memory scatter/read round trip from the native 512 hot path.
inline uint octal_reverse_512(uint value) {
    const uint d0 = value & 7u;
    const uint d1 = (value >> 3u) & 7u;
    const uint d2 = value >> 6u;
    return (d0 << 6u) | (d1 << 3u) | d2;
}

inline void fft8_butterfly(
    float2 x0, float2 x1, float2 x2, float2 x3,
    float2 x4, float2 x5, float2 x6, float2 x7,
    bool inverse,
    thread float2 &y0, thread float2 &y1, thread float2 &y2, thread float2 &y3,
    thread float2 &y4, thread float2 &y5, thread float2 &y6, thread float2 &y7) {
    const float2 ea0 = x0 + x4;
    const float2 ea1 = x0 - x4;
    const float2 ea2 = x2 + x6;
    const float2 ea3 = x2 - x6;
    const float2 er = inverse ? float2(-ea3.y, ea3.x) : float2(ea3.y, -ea3.x);
    const float2 e0 = ea0 + ea2;
    const float2 e1 = ea1 + er;
    const float2 e2 = ea0 - ea2;
    const float2 e3 = ea1 - er;

    const float2 oa0 = x1 + x5;
    const float2 oa1 = x1 - x5;
    const float2 oa2 = x3 + x7;
    const float2 oa3 = x3 - x7;
    const float2 ora = inverse ? float2(-oa3.y, oa3.x) : float2(oa3.y, -oa3.x);
    const float2 o0 = oa0 + oa2;
    const float2 o1 = oa1 + ora;
    const float2 o2 = oa0 - oa2;
    const float2 o3 = oa1 - ora;

    constexpr float s = 0.7071067811865475244f;
    const float2 w1 = inverse
        ? float2(s * (o1.x - o1.y), s * (o1.x + o1.y))
        : float2(s * (o1.x + o1.y), s * (o1.y - o1.x));
    const float2 w2 = inverse ? float2(-o2.y, o2.x) : float2(o2.y, -o2.x);
    const float2 w3 = inverse
        ? float2(-s * (o3.x + o3.y), s * (o3.x - o3.y))
        : float2(s * (o3.y - o3.x), -s * (o3.x + o3.y));
    y0 = e0 + o0; y4 = e0 - o0;
    y1 = e1 + w1; y5 = e1 - w1;
    y2 = e2 + w2; y6 = e2 - w2;
    y3 = e3 + w3; y7 = e3 - w3;
}

inline float2 shuffle_xor_float2(float2 value, ushort mask) {
    return float2(
        simd_shuffle_xor(value.x, mask),
        simd_shuffle_xor(value.y, mask)
    );
}

// Exact 512-point inverse FFT used by the CUDA t64 path: stage one remains
// entirely in registers, the middle exchange uses SIMD-lane shuffles, and
// only the final cross-SIMD exchange uses threadgroup memory. Twiddles in the
// shared table use the forward sign, so their imaginary component is flipped
// here for the inverse transform.
inline void ifft512_radix8_registers(
    thread float2 &r0, thread float2 &r1,
    thread float2 &r2, thread float2 &r3,
    thread float2 &r4, thread float2 &r5,
    thread float2 &r6, thread float2 &r7,
    uint tid,
    device const float2 *twiddle,
    threadgroup float2 *scratch) {
    float2 y0, y1, y2, y3, y4, y5, y6, y7;
    fft8_butterfly(
        r0, r1, r2, r3, r4, r5, r6, r7, true,
        y0, y1, y2, y3, y4, y5, y6, y7
    );
    r0 = y0; r1 = y1; r2 = y2; r3 = y3;
    r4 = y4; r5 = y5; r6 = y6; r7 = y7;

    float2 sent;
    float2 received;
    sent = (tid & 1u) != 0u ? r0 : r1;
    received = shuffle_xor_float2(sent, 1u);
    if ((tid & 1u) != 0u) r0 = received; else r1 = received;
    sent = (tid & 1u) != 0u ? r2 : r3;
    received = shuffle_xor_float2(sent, 1u);
    if ((tid & 1u) != 0u) r2 = received; else r3 = received;
    sent = (tid & 1u) != 0u ? r4 : r5;
    received = shuffle_xor_float2(sent, 1u);
    if ((tid & 1u) != 0u) r4 = received; else r5 = received;
    sent = (tid & 1u) != 0u ? r6 : r7;
    received = shuffle_xor_float2(sent, 1u);
    if ((tid & 1u) != 0u) r6 = received; else r7 = received;

    sent = (tid & 2u) != 0u ? r0 : r2;
    received = shuffle_xor_float2(sent, 2u);
    if ((tid & 2u) != 0u) r0 = received; else r2 = received;
    sent = (tid & 2u) != 0u ? r1 : r3;
    received = shuffle_xor_float2(sent, 2u);
    if ((tid & 2u) != 0u) r1 = received; else r3 = received;
    sent = (tid & 2u) != 0u ? r4 : r6;
    received = shuffle_xor_float2(sent, 2u);
    if ((tid & 2u) != 0u) r4 = received; else r6 = received;
    sent = (tid & 2u) != 0u ? r5 : r7;
    received = shuffle_xor_float2(sent, 2u);
    if ((tid & 2u) != 0u) r5 = received; else r7 = received;

    sent = (tid & 4u) != 0u ? r0 : r4;
    received = shuffle_xor_float2(sent, 4u);
    if ((tid & 4u) != 0u) r0 = received; else r4 = received;
    sent = (tid & 4u) != 0u ? r1 : r5;
    received = shuffle_xor_float2(sent, 4u);
    if ((tid & 4u) != 0u) r1 = received; else r5 = received;
    sent = (tid & 4u) != 0u ? r2 : r6;
    received = shuffle_xor_float2(sent, 4u);
    if ((tid & 4u) != 0u) r2 = received; else r6 = received;
    sent = (tid & 4u) != 0u ? r3 : r7;
    received = shuffle_xor_float2(sent, 4u);
    if ((tid & 4u) != 0u) r3 = received; else r7 = received;

    const uint stage2 = tid & 7u;
    float2 w1 = twiddle[(stage2 * 1u * 8u) & 511u]; w1.y = -w1.y;
    float2 w2 = twiddle[(stage2 * 2u * 8u) & 511u]; w2.y = -w2.y;
    float2 w3 = twiddle[(stage2 * 3u * 8u) & 511u]; w3.y = -w3.y;
    float2 w4 = twiddle[(stage2 * 4u * 8u) & 511u]; w4.y = -w4.y;
    float2 w5 = twiddle[(stage2 * 5u * 8u) & 511u]; w5.y = -w5.y;
    float2 w6 = twiddle[(stage2 * 6u * 8u) & 511u]; w6.y = -w6.y;
    float2 w7 = twiddle[(stage2 * 7u * 8u) & 511u]; w7.y = -w7.y;
    r1 = complex_mul(w1, r1); r2 = complex_mul(w2, r2);
    r3 = complex_mul(w3, r3); r4 = complex_mul(w4, r4);
    r5 = complex_mul(w5, r5); r6 = complex_mul(w6, r6);
    r7 = complex_mul(w7, r7);
    fft8_butterfly(
        r0, r1, r2, r3, r4, r5, r6, r7, true,
        y0, y1, y2, y3, y4, y5, y6, y7
    );
    r0 = y0; r1 = y1; r2 = y2; r3 = y3;
    r4 = y4; r5 = y5; r6 = y6; r7 = y7;

    const uint outer = tid >> 3u;
    const uint base = outer * 64u + stage2;
    scratch[base + 0u] = r0; scratch[base + 8u] = r1;
    scratch[base + 16u] = r2; scratch[base + 24u] = r3;
    scratch[base + 32u] = r4; scratch[base + 40u] = r5;
    scratch[base + 48u] = r6; scratch[base + 56u] = r7;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    r0 = scratch[tid + 0u];
    w1 = twiddle[(tid * 1u) & 511u]; w1.y = -w1.y;
    w2 = twiddle[(tid * 2u) & 511u]; w2.y = -w2.y;
    w3 = twiddle[(tid * 3u) & 511u]; w3.y = -w3.y;
    w4 = twiddle[(tid * 4u) & 511u]; w4.y = -w4.y;
    w5 = twiddle[(tid * 5u) & 511u]; w5.y = -w5.y;
    w6 = twiddle[(tid * 6u) & 511u]; w6.y = -w6.y;
    w7 = twiddle[(tid * 7u) & 511u]; w7.y = -w7.y;
    r1 = complex_mul(w1, scratch[tid + 64u]);
    r2 = complex_mul(w2, scratch[tid + 128u]);
    r3 = complex_mul(w3, scratch[tid + 192u]);
    r4 = complex_mul(w4, scratch[tid + 256u]);
    r5 = complex_mul(w5, scratch[tid + 320u]);
    r6 = complex_mul(w6, scratch[tid + 384u]);
    r7 = complex_mul(w7, scratch[tid + 448u]);
    fft8_butterfly(
        r0, r1, r2, r3, r4, r5, r6, r7, true,
        y0, y1, y2, y3, y4, y5, y6, y7
    );
    r0 = y0; r1 = y1; r2 = y2; r3 = y3;
    r4 = y4; r5 = y5; r6 = y6; r7 = y7;
}
