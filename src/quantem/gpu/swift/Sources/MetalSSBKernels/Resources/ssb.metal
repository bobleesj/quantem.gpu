#include <metal_stdlib>
using namespace metal;

struct FFTParams {
    uint n;
    uint log2n;
    uint batch;
    uint inverse;
};

struct SSBParams {
    uint n;
    uint batch;
    uint logical_bf;
    uint bf_offset;
    float wavelength;
    float semiangle;
    float ang_y;
    float ang_x;
    float c10;
    float c12;
    float cos2phi12;
    float sin2phi12;
    float factor;
    float dc_r;
    float dc_i;
    float aperture_inner_k2;
    float aperture_outer_k2;
};

struct HalfExtractParams {
    uint source_n;
    uint batch;
};

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

kernel void uint8_to_complex(
    device const uchar *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = float2(float(input[index]), 0.0f);
    }
}

// 512 = 8^3. This is the parity-checked radix-8 kernel retained from the
// native-device FFT implementation. One threadgroup transforms one row.
kernel void fft_rows_radix8(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    device const float2 *twiddle [[buffer(2)]],
    constant FFTParams &params [[buffer(3)]],
    threadgroup float2 *scratch [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {

    const uint n = params.n;
    const uint eighth_n = n >> 3u;
    const uint radix8_digits = params.log2n / 3u;
    const uint row = group % n;
    const uint batch_index = group / n;
    if (batch_index >= params.batch || params.log2n % 3u != 0u) return;
    const size_t base = ((size_t)batch_index * n + row) * n;
    const bool inverse = params.inverse != 0u;

    for (uint butterfly = tid; butterfly < eighth_n; butterfly += threads) {
        uint pos[8];
        float2 x[8];
        for (uint lane = 0u; lane < 8u; ++lane) {
            pos[lane] = butterfly + lane * eighth_n;
            x[lane] = input[base + pos[lane]];
        }
        float2 y0, y1, y2, y3, y4, y5, y6, y7;
        fft8_butterfly(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], inverse,
            y0, y1, y2, y3, y4, y5, y6, y7);
        scratch[reverse_base8_digits(pos[0], radix8_digits)] = y0;
        scratch[reverse_base8_digits(pos[1], radix8_digits)] = y1;
        scratch[reverse_base8_digits(pos[2], radix8_digits)] = y2;
        scratch[reverse_base8_digits(pos[3], radix8_digits)] = y3;
        scratch[reverse_base8_digits(pos[4], radix8_digits)] = y4;
        scratch[reverse_base8_digits(pos[5], radix8_digits)] = y5;
        scratch[reverse_base8_digits(pos[6], radix8_digits)] = y6;
        scratch[reverse_base8_digits(pos[7], radix8_digits)] = y7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 64u; width <= n; width <<= 3u) {
        const uint eighth = width >> 3u;
        for (uint butterfly = tid; butterfly < eighth_n; butterfly += threads) {
            const uint block = butterfly / eighth;
            const uint j = butterfly - block * eighth;
            const uint first = block * width + j;
            const uint twiddle_index = j * (n / width);
            float2 x[8];
            x[0] = scratch[first];
            for (uint lane = 1u; lane < 8u; ++lane) {
                float2 w = twiddle[twiddle_index * lane * (512u / n)];
                if (inverse) w.y = -w.y;
                x[lane] = complex_mul(w, scratch[first + lane * eighth]);
            }
            float2 y0, y1, y2, y3, y4, y5, y6, y7;
            fft8_butterfly(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], inverse,
                y0, y1, y2, y3, y4, y5, y6, y7);
            scratch[first] = y0;
            scratch[first + eighth] = y1;
            scratch[first + 2u * eighth] = y2;
            scratch[first + 3u * eighth] = y3;
            scratch[first + 4u * eighth] = y4;
            scratch[first + 5u * eighth] = y5;
            scratch[first + 6u * eighth] = y6;
            scratch[first + 7u * eighth] = y7;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = inverse ? 1.0f / float(n) : 1.0f;
    for (uint index = tid; index < n; index += threads) {
        output[base + index] = scratch[index] * scale;
    }
}

kernel void extract_hermitian_half(
    device const float2 *full_g [[buffer(0)]],
    device float2 *half_g [[buffer(1)]],
    constant HalfExtractParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint half_cols = params.source_n / 2u + 1u;
    const uint half_plane = params.source_n * half_cols;
    const uint total = params.batch * half_plane;
    if (index >= total) return;
    const uint local_bf = index / half_plane;
    const uint pixel = index - local_bf * half_plane;
    const uint row = pixel / half_cols;
    const uint col = pixel - row * half_cols;
    const size_t full_plane = (size_t)params.source_n * params.source_n;
    half_g[index] = full_g[(size_t)local_bf * full_plane +
        (size_t)row * params.source_n + col];
}

kernel void transpose_complex32(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant FFTParams &params [[buffer(2)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float2 tile[32][33];
    const uint n = params.n;
    const uint batch_index = group.z;
    if (batch_index >= params.batch) return;
    const size_t plane = (size_t)batch_index * n * n;
    const uint input_x = group.x * 32u + tid.x;
    const uint input_y = group.y * 32u + tid.y;
    for (uint offset = 0; offset < 32u; offset += 8u) {
        if (input_x < n && input_y + offset < n) {
            tile[tid.y + offset][tid.x] =
                input[plane + (size_t)(input_y + offset) * n + input_x];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint output_x = group.y * 32u + tid.x;
    const uint output_y = group.x * 32u + tid.y;
    for (uint offset = 0; offset < 32u; offset += 8u) {
        if (output_x < n && output_y + offset < n) {
            output[plane + (size_t)(output_y + offset) * n + output_x] =
                tile[tid.x][tid.y + offset];
        }
    }
}

inline float4 compute_geometry(float dx, float dy, constant SSBParams &params) {
    const float dx2 = dx * dx;
    const float dy2 = dy * dy;
    const float r2 = dx2 + dy2;
    const float r = sqrt(r2);
    const float alpha = r * params.wavelength;
    const float alpha2 = alpha * alpha;
    const float inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
    const float cos2 = (dx2 - dy2) * inv_r2;
    const float sin2 = 2.0f * dx * dy * inv_r2;
    const float denom_num2 =
        (dx * params.ang_y) * (dx * params.ang_y) +
        (dy * params.ang_x) * (dy * params.ang_x);
    const float inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
    const float denom = sqrt(denom_num2) * inv_r;
    const float edge = denom > 1.0e-15f
        ? (params.semiangle - alpha) / denom + 0.5f
        : 1.0f;
    return float4(alpha2, cos2, sin2, clamp(edge, 0.0f, 1.0f));
}

inline float2 exp_neg_i(float phase, float aperture) {
    float cosine;
    const float sine = metal::sincos(phase, cosine);
    return float2(aperture * cosine, -aperture * sine);
}

inline float aperture_at(float x, float y, constant SSBParams &params) {
    const float r2 = x * x + y * y;
    if (params.ang_x == params.ang_y) {
        if (r2 <= params.aperture_inner_k2) return 1.0f;
        if (r2 >= params.aperture_outer_k2) return 0.0f;
        const float alpha = sqrt(r2) * params.wavelength;
        const float edge = params.ang_x > 1.0e-15f
            ? (params.semiangle - alpha) / params.ang_x + 0.5f
            : 1.0f;
        return clamp(edge, 0.0f, 1.0f);
    }
    const float r = sqrt(r2);
    const float alpha = r * params.wavelength;
    const float denom_num2 =
        (x * params.ang_y) * (x * params.ang_y) +
        (y * params.ang_x) * (y * params.ang_x);
    const float denom = r > 1.0e-15f ? sqrt(denom_num2) / r : 0.0f;
    const float edge = denom > 1.0e-15f
        ? (params.semiangle - alpha) / denom + 0.5f
        : 1.0f;
    return clamp(edge, 0.0f, 1.0f);
}

kernel void ssb_gamma_accumulate(
    device const float2 *g [[buffer(0)]],
    device const float4 *bf_geometry [[buffer(1)]],
    device const float2 *bf_trig [[buffer(2)]],
    device const float *q_row [[buffer(3)]],
    device const float *q_col [[buffer(4)]],
    device float2 *accumulator [[buffer(5)]],
    constant SSBParams &params [[buffer(6)]],
    uint pixel [[thread_position_in_grid]]) {
    const uint plane = params.n * params.n;
    if (pixel >= plane || pixel == 0u) return;
    const uint row = pixel / params.n;
    const uint col = pixel - row * params.n;
    const float qx = q_row[row];
    const float qy = q_col[col];
    float2 sum = accumulator[pixel];

    for (uint local = 0u; local < params.batch; ++local) {
        const uint bf = params.bf_offset + local;
        const float4 bg = bf_geometry[bf];
        const float2 bt = bf_trig[bf];
        const float chi_k = params.factor * bg.z *
            (params.c12 * (bt.x * params.cos2phi12 + bt.y * params.sin2phi12)
             + params.c10);
        const float2 pk = exp_neg_i(chi_k, bg.w);

        const float4 gm = compute_geometry(qx - bg.x, qy - bg.y, params);
        const float4 gp = compute_geometry(qx + bg.x, qy + bg.y, params);
        const float chi_m = params.factor * gm.x *
            (params.c12 * (gm.y * params.cos2phi12 + gm.z * params.sin2phi12)
             + params.c10);
        const float chi_p = params.factor * gp.x *
            (params.c12 * (gp.y * params.cos2phi12 + gp.z * params.sin2phi12)
             + params.c10);
        const float2 pm = exp_neg_i(chi_m, gm.w);
        const float2 pp = exp_neg_i(chi_p, gp.w);

        // gamma = p(q-k) conj(p(k)) - conj(p(q+k)) p(k)
        const float gamma_r =
            (pm.x * pk.x + pm.y * pk.y) -
            (pp.x * pk.x + pp.y * pk.y);
        const float gamma_i =
            (pm.y * pk.x - pm.x * pk.y) -
            (pp.x * pk.y - pp.y * pk.x);
        const float magnitude = sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
        const float inverse_magnitude = 1.0f / max(magnitude, 1.0e-8f);
        const float conjugate_r = gamma_r * inverse_magnitude;
        const float conjugate_i = -gamma_i * inverse_magnitude;
        const float2 value = g[(size_t)local * plane + pixel];
        sum += float2(
            value.x * conjugate_r - value.y * conjugate_i,
            value.x * conjugate_i + value.y * conjugate_r);
    }
    accumulator[pixel] = sum;
}

inline float2 load_g_cache_chunk(
    uint chunk,
    size_t index,
    device const float2 *g0,
    device const float2 *g1,
    device const float2 *g2,
    device const float2 *g3,
    device const float2 *g4,
    device const float2 *g5,
    device const float2 *g6,
    device const float2 *g7,
    device const float2 *g8,
    device const float2 *g9,
    device const float2 *g10,
    device const float2 *g11) {
    switch (chunk) {
        case 0u: return g0[index];
        case 1u: return g1[index];
        case 2u: return g2[index];
        case 3u: return g3[index];
        case 4u: return g4[index];
        case 5u: return g5[index];
        case 6u: return g6[index];
        case 7u: return g7[index];
        case 8u: return g8[index];
        case 9u: return g9[index];
        case 10u: return g10[index];
        default: return g11[index];
    }
}

kernel void ssb_gamma_accumulate_half(
    device const float2 *g0 [[buffer(0)]],
    device const float2 *g1 [[buffer(1)]],
    device const float2 *g2 [[buffer(2)]],
    device const float2 *g3 [[buffer(3)]],
    device const float2 *g4 [[buffer(4)]],
    device const float2 *g5 [[buffer(5)]],
    device const float2 *g6 [[buffer(6)]],
    device const float2 *g7 [[buffer(7)]],
    device const float2 *g8 [[buffer(8)]],
    device const float2 *g9 [[buffer(9)]],
    device const float2 *g10 [[buffer(10)]],
    device const float2 *g11 [[buffer(11)]],
    device const float4 *bf_geometry [[buffer(12)]],
    device const float *q_row [[buffer(13)]],
    device const float *q_col [[buffer(14)]],
    device float2 *accumulator [[buffer(15)]],
    constant SSBParams &params [[buffer(16)]],
    uint pair_index [[thread_position_in_grid]]) {
    const uint interior_cols = params.n / 2u - 1u;
    const uint interior_pair_count = (params.n - 1u) * interior_cols;
    const uint boundary_rows = params.n / 2u + 1u;
    const uint half_cols = params.n / 2u + 1u;
    const uint half_plane = params.n * half_cols;
    if (pair_index >= half_plane) return;

    uint row;
    uint col;
    bool should_pair;
    if (pair_index < interior_pair_count) {
        const uint compact_row = pair_index / interior_cols;
        row = compact_row < params.n / 2u ? compact_row : compact_row + 1u;
        col = pair_index - compact_row * interior_cols + 1u;
        should_pair = true;
    } else if (pair_index < interior_pair_count + boundary_rows) {
        row = pair_index - interior_pair_count;
        col = 0u;
        should_pair = true;
    } else if (pair_index < interior_pair_count + boundary_rows + params.n) {
        row = pair_index - interior_pair_count - boundary_rows;
        col = params.n / 2u;
        should_pair = false;
    } else {
        const uint nyquist_row_offset =
            pair_index - interior_pair_count - boundary_rows - params.n;
        row = params.n / 2u;
        col = nyquist_row_offset < interior_cols
            ? nyquist_row_offset + 1u
            : params.n / 2u + 1u + nyquist_row_offset - interior_cols;
        should_pair = false;
    }
    const uint pixel = row * params.n + col;
    if (pixel == 0u) return;
    const uint mirror_row = row == 0u ? 0u : params.n - row;
    const uint mirror_col = col == 0u ? 0u : params.n - col;
    const uint mirror_pixel = mirror_row * params.n + mirror_col;
    const bool is_self_pair = mirror_pixel == pixel;
    const bool write_mirror = should_pair && !is_self_pair;
    const float qx = q_row[row];
    const float qy = q_col[col];
    const float lambda2 = params.wavelength * params.wavelength;
    const float quadratic_factor = params.factor * lambda2;
    const float xx = params.c10 + params.c12 * params.cos2phi12;
    const float xy = params.c12 * params.sin2phi12;
    const float yy = params.c10 - params.c12 * params.cos2phi12;
    const float chi_q = quadratic_factor *
        (xx * qx * qx + 2.0f * xy * qx * qy + yy * qy * qy);
    float cosine_q;
    const float sine_q = metal::sincos(chi_q, cosine_q);
    float2 sum = accumulator[pixel];
    float2 mirror_sum = write_mirror ? accumulator[mirror_pixel] : float2(0.0f);

    for (uint local = 0u; local < params.batch; ++local) {
        const uint bf = params.bf_offset + local;
        const float4 bg = bf_geometry[bf];
        if (bg.w <= 0.0f) continue;
        const float aperture_m = bg.w * aperture_at(qx - bg.x, qy - bg.y, params);
        const float aperture_p = bg.w * aperture_at(qx + bg.x, qy + bg.y, params);
        if (aperture_m <= 0.0f && aperture_p <= 0.0f) continue;
        const float cross_phase = 2.0f * quadratic_factor *
            (xx * qx * bg.x + xy * (qx * bg.y + qy * bg.x) +
             yy * qy * bg.y);
        float cosine_cross;
        const float sine_cross = metal::sincos(cross_phase, cosine_cross);
        const float bracket_r = (aperture_m - aperture_p) * cosine_q;
        const float bracket_i = -(aperture_m + aperture_p) * sine_q;
        const float gamma_r =
            cosine_cross * bracket_r - sine_cross * bracket_i;
        const float gamma_i =
            sine_cross * bracket_r + cosine_cross * bracket_i;
        const float magnitude = sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
        const float inverse_magnitude = 1.0f / max(magnitude, 1.0e-8f);
        const float conjugate_r = gamma_r * inverse_magnitude;
        const float conjugate_i = -gamma_i * inverse_magnitude;

        const uint global_bf = params.bf_offset + local;
        const uint cache_chunk = global_bf / 512u;
        const uint cache_local = global_bf - cache_chunk * 512u;
        const size_t source_offset = (size_t)cache_local * half_plane;
        float2 value;
        if (col <= params.n / 2u) {
            value = load_g_cache_chunk(
                cache_chunk, source_offset + (size_t)row * half_cols + col,
                g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11);
        } else {
            value = load_g_cache_chunk(
                cache_chunk,
                source_offset + (size_t)mirror_row * half_cols + mirror_col,
                g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11);
            value.y = -value.y;
        }
        const float2 contribution = float2(
            value.x * conjugate_r - value.y * conjugate_i,
            value.x * conjugate_i + value.y * conjugate_r);
        sum += contribution;
        if (write_mirror) {
            float2 mirror_value;
            if (col == 0u || col == params.n / 2u) {
                mirror_value = load_g_cache_chunk(
                    cache_chunk,
                    source_offset + (size_t)mirror_row * half_cols + mirror_col,
                    g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11);
            } else {
                mirror_value = float2(value.x, -value.y);
            }
            const float mirror_conjugate_r = -conjugate_r;
            const float mirror_conjugate_i = conjugate_i;
            mirror_sum += float2(
                mirror_value.x * mirror_conjugate_r -
                    mirror_value.y * mirror_conjugate_i,
                mirror_value.x * mirror_conjugate_i +
                    mirror_value.y * mirror_conjugate_r);
        }
    }
    accumulator[pixel] = sum;
    if (write_mirror) accumulator[mirror_pixel] = mirror_sum;
}

kernel void ssb_finalize_fourier_sum(
    device float2 *accumulator [[buffer(0)]],
    constant SSBParams &params [[buffer(1)]],
    uint pixel [[thread_position_in_grid]]) {
    const uint plane = params.n * params.n;
    if (pixel >= plane) return;
    if (pixel == 0u) {
        accumulator[pixel] = float2(params.dc_r, params.dc_i);
    } else {
        accumulator[pixel] /= float(params.logical_bf);
    }
}

inline float2 ssb_corrected_value(
    float2 value,
    uint row,
    uint col,
    uint bf,
    device const float4 *bf_geometry,
    device const float *q_row,
    device const float *q_col,
    constant SSBParams &params) {
    if (row == 0u && col == 0u) {
        return float2(params.dc_r, params.dc_i);
    }
    const float4 bg = bf_geometry[bf];
    if (bg.w <= 0.0f) return float2(0.0f);
    const float qx = q_row[row];
    const float qy = q_col[col];
    const float aperture_m = bg.w * aperture_at(qx - bg.x, qy - bg.y, params);
    const float aperture_p = bg.w * aperture_at(qx + bg.x, qy + bg.y, params);
    if (aperture_m <= 0.0f && aperture_p <= 0.0f) return float2(0.0f);

    const float lambda2 = params.wavelength * params.wavelength;
    const float quadratic_factor = params.factor * lambda2;
    const float xx = params.c10 + params.c12 * params.cos2phi12;
    const float xy = params.c12 * params.sin2phi12;
    const float yy = params.c10 - params.c12 * params.cos2phi12;
    const float chi_q = quadratic_factor *
        (xx * qx * qx + 2.0f * xy * qx * qy + yy * qy * qy);
    float cosine_q;
    const float sine_q = metal::sincos(chi_q, cosine_q);
    const float cross_phase = 2.0f * quadratic_factor *
        (xx * qx * bg.x + xy * (qx * bg.y + qy * bg.x) +
         yy * qy * bg.y);
    float cosine_cross;
    const float sine_cross = metal::sincos(cross_phase, cosine_cross);
    const float bracket_r = (aperture_m - aperture_p) * cosine_q;
    const float bracket_i = -(aperture_m + aperture_p) * sine_q;
    const float gamma_r = cosine_cross * bracket_r - sine_cross * bracket_i;
    const float gamma_i = sine_cross * bracket_r + cosine_cross * bracket_i;
    const float inverse_magnitude = rsqrt(max(
        gamma_r * gamma_r + gamma_i * gamma_i,
        1.0e-16f
    ));
    const float conjugate_r = gamma_r * inverse_magnitude;
    const float conjugate_i = -gamma_i * inverse_magnitude;
    return float2(
        value.x * conjugate_r - value.y * conjugate_i,
        value.x * conjugate_i + value.y * conjugate_r
    );
}

// Apply the identical gamma correction with chi(q) supplied by a per-candidate
// 512x512 cache. chi(q) is independent of BF, so evaluating its float32 sincos
// once and reusing the result removes exact duplicate work across all BF planes.
inline float2 ssb_corrected_value_precomputed_chi(
    float2 value,
    uint row,
    uint col,
    uint bf,
    device const float4 *bf_geometry,
    device const float *q_row,
    device const float *q_col,
    device const float2 *chi_trig,
    device const float2 *cross_trig,
    constant SSBParams &params) {
    if (row == 0u && col == 0u) {
        return float2(params.dc_r, params.dc_i);
    }
    const float4 bg = bf_geometry[bf];
    if (bg.w <= 0.0f) return float2(0.0f);
    const float qx = q_row[row];
    const float qy = q_col[col];
    const float aperture_m = bg.w * aperture_at(qx - bg.x, qy - bg.y, params);
    const float aperture_p = bg.w * aperture_at(qx + bg.x, qy + bg.y, params);
    if (aperture_m <= 0.0f && aperture_p <= 0.0f) return float2(0.0f);

    const float2 cached_chi = chi_trig[row * 512u + col];
    const float cosine_q = cached_chi.x;
    const float sine_q = cached_chi.y;
    const size_t cross_base = (size_t)bf * 1024u;
    const float2 row_cross = cross_trig[cross_base + row];
    const float2 col_cross = cross_trig[cross_base + 512u + col];
    const float cosine_cross =
        row_cross.x * col_cross.x - row_cross.y * col_cross.y;
    const float sine_cross =
        row_cross.y * col_cross.x + row_cross.x * col_cross.y;
    const float bracket_r = (aperture_m - aperture_p) * cosine_q;
    const float bracket_i = -(aperture_m + aperture_p) * sine_q;
    const float gamma_r = cosine_cross * bracket_r - sine_cross * bracket_i;
    const float gamma_i = sine_cross * bracket_r + cosine_cross * bracket_i;
    const float inverse_magnitude = rsqrt(max(
        gamma_r * gamma_r + gamma_i * gamma_i,
        1.0e-16f
    ));
    const float conjugate_r = gamma_r * inverse_magnitude;
    const float conjugate_i = -gamma_i * inverse_magnitude;
    return float2(
        value.x * conjugate_r - value.y * conjugate_i,
        value.x * conjugate_i + value.y * conjugate_r
    );
}

kernel void ssb_precompute_chi_trig512(
    device const float *q_row [[buffer(0)]],
    device const float *q_col [[buffer(1)]],
    device float2 *chi_trig [[buffer(2)]],
    constant SSBParams &params [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]) {
    constexpr uint n = 512u;
    if (pixel >= n * n) return;
    const uint row = pixel / n;
    const uint col = pixel - row * n;
    const float qx = q_row[row];
    const float qy = q_col[col];
    const float lambda2 = params.wavelength * params.wavelength;
    const float quadratic_factor = params.factor * lambda2;
    const float xx = params.c10 + params.c12 * params.cos2phi12;
    const float xy = params.c12 * params.sin2phi12;
    const float yy = params.c10 - params.c12 * params.cos2phi12;
    const float chi_q = quadratic_factor *
        (xx * qx * qx + 2.0f * xy * qx * qy + yy * qy * qy);
    float cosine_q;
    const float sine_q = metal::sincos(chi_q, cosine_q);
    chi_trig[pixel] = float2(cosine_q, sine_q);
}

kernel void ssb_precompute_cross_trig512(
    device const float4 *bf_geometry [[buffer(0)]],
    device const float *q_row [[buffer(1)]],
    device const float *q_col [[buffer(2)]],
    device float2 *cross_trig [[buffer(3)]],
    constant SSBParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint n = 512u;
    const uint active_bf = params.batch;
    const uint values_per_bf = 2u * n;
    if (index >= active_bf * values_per_bf) return;
    const uint bf = index / values_per_bf;
    const uint coordinate = index - bf * values_per_bf;
    const float4 bg = bf_geometry[bf];
    const float lambda2 = params.wavelength * params.wavelength;
    const float twice_quadratic = 2.0f * params.factor * lambda2;
    const float xx = params.c10 + params.c12 * params.cos2phi12;
    const float xy = params.c12 * params.sin2phi12;
    const float yy = params.c10 - params.c12 * params.cos2phi12;
    float phase;
    if (coordinate < n) {
        phase = twice_quadratic * q_row[coordinate] *
            (xx * bg.x + xy * bg.y);
    } else {
        const uint col = coordinate - n;
        phase = twice_quadratic * q_col[col] *
            (xy * bg.x + yy * bg.y);
    }
    float cosine;
    const float sine = metal::sincos(phase, cosine);
    cross_trig[index] = float2(cosine, sine);
}

kernel void ssb_transpose_half_to_column_major(
    device const float2 *source [[buffer(0)]],
    device float2 *destination [[buffer(1)]],
    constant uint &batch [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint rows = 512u;
    constexpr uint cols = 257u;
    constexpr uint plane = rows * cols;
    if (index >= batch * plane) return;
    const uint bf = index / plane;
    const uint within = index - bf * plane;
    const uint row = within / cols;
    const uint col = within - row * cols;
    destination[(size_t)bf * plane + (size_t)col * rows + row] = source[index];
}

kernel void ssb_transpose_half_to_row_major(
    device const float2 *source [[buffer(0)]],
    device float2 *destination [[buffer(1)]],
    constant uint &batch [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint rows = 512u;
    constexpr uint cols = 257u;
    constexpr uint plane = rows * cols;
    if (index >= batch * plane) return;
    const uint bf = index / plane;
    const uint within = index - bf * plane;
    const uint row = within / cols;
    const uint col = within - row * cols;
    destination[index] = source[
        (size_t)bf * plane + (size_t)col * rows + row
    ];
}

// Produce one complete corrected complex64 Fourier plane per BF term. The
// source contains a contiguous local batch from the resident Hermitian cache;
// bf_offset identifies the matching geometry in the full CUDA-selected BF set.
kernel void ssb_correct_half_for_phase_loss(
    device const float2 *half_g [[buffer(0)]],
    device const float4 *bf_geometry [[buffer(1)]],
    device const float *q_row [[buffer(2)]],
    device const float *q_col [[buffer(3)]],
    device float2 *corrected [[buffer(4)]],
    constant SSBParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint plane = params.n * params.n;
    const uint total = params.batch * plane;
    if (index >= total) return;
    const uint local_bf = index / plane;
    const uint pixel = index - local_bf * plane;
    const uint row = pixel / params.n;
    const uint col = pixel - row * params.n;
    const uint half_cols = params.n / 2u + 1u;
    const uint half_plane = params.n * half_cols;
    float2 value;
    if (col <= params.n / 2u) {
        value = half_g[(size_t)local_bf * half_plane +
            (size_t)row * half_cols + col];
    } else {
        const uint mirror_row = row == 0u ? 0u : params.n - row;
        const uint mirror_col = params.n - col;
        value = half_g[(size_t)local_bf * half_plane +
            (size_t)mirror_row * half_cols + mirror_col];
        value.y = -value.y;
    }
    corrected[index] = ssb_corrected_value(
        value, row, col, params.bf_offset + local_bf,
        bf_geometry, q_row, q_col, params
    );
}

// Equivalent exact path for a freshly transformed resident uint8 BF batch.
// This keeps hybrid-cache datasets correct without persisting another G cache.
kernel void ssb_correct_full_for_phase_loss(
    device const float2 *full_g [[buffer(0)]],
    device const float4 *bf_geometry [[buffer(1)]],
    device const float *q_row [[buffer(2)]],
    device const float *q_col [[buffer(3)]],
    device float2 *corrected [[buffer(4)]],
    constant SSBParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint plane = params.n * params.n;
    const uint total = params.batch * plane;
    if (index >= total) return;
    const uint local_bf = index / plane;
    const uint pixel = index - local_bf * plane;
    const uint row = pixel / params.n;
    const uint col = pixel - row * params.n;
    corrected[index] = ssb_corrected_value(
        full_g[index], row, col, params.bf_offset + local_bf,
        bf_geometry, q_row, q_col, params
    );
}

// One thread owns one native scan pixel and accumulates a local BF batch. This
// avoids atomic float additions while preserving the established BF order.
kernel void ssb_accumulate_phase_moments(
    device const float2 *objects [[buffer(0)]],
    device float *phase_sum [[buffer(1)]],
    device float *phase_sumsq [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    uint pixel [[thread_position_in_grid]]) {
    constexpr uint plane = 512u * 512u;
    if (pixel >= plane) return;
    float sum = phase_sum[pixel];
    float sumsq = phase_sumsq[pixel];
    for (uint local = 0u; local < batch; ++local) {
        const float2 value = objects[(size_t)local * plane + pixel];
        const float phase = metal::atan2(value.y, value.x);
        sum += phase;
        sumsq += phase * phase;
    }
    phase_sum[pixel] = sum;
    phase_sumsq[pixel] = sumsq;
}


// The corrected non-DC spectrum obeys F(-q) = -conj(F(q)). Multiplying by
// -i makes it Hermitian, so only the native 512x257 half-plane is transformed
// along the first dimension. The real/complex DC value is restored after the
// second transform. This is an exact symmetry reduction, not BF subsampling.
kernel void ssb_correct_half_column_ifft512_hermitian(
    device const float2 *half_g [[buffer(0)]],
    device const float4 *bf_geometry [[buffer(1)]],
    device const float *q_row [[buffer(2)]],
    device const float *q_col [[buffer(3)]],
    device const float2 *twiddle [[buffer(4)]],
    device float2 *column_ifft_half [[buffer(5)]],
    constant SSBParams &params [[buffer(6)]],
    device const float2 *chi_trig [[buffer(7)]],
    device const float2 *cross_trig [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr uint n = 512u;
    constexpr uint half_cols = 257u;
    const uint col = group.x;
    const uint local_bf = group.y;
    if (tid >= 64u || col >= half_cols || local_bf >= params.batch) return;

    threadgroup float2 scratch[512];
    const uint src0 = octal_reverse_512(tid * 8u + 0u);
    const uint src1 = octal_reverse_512(tid * 8u + 1u);
    const uint src2 = octal_reverse_512(tid * 8u + 2u);
    const uint src3 = octal_reverse_512(tid * 8u + 3u);
    const uint src4 = octal_reverse_512(tid * 8u + 4u);
    const uint src5 = octal_reverse_512(tid * 8u + 5u);
    const uint src6 = octal_reverse_512(tid * 8u + 6u);
    const uint src7 = octal_reverse_512(tid * 8u + 7u);
    const size_t half_base = (size_t)local_bf * n * half_cols;
    const uint bf = params.bf_offset + local_bf;
#define LOAD_HERMITIAN(slot) \
    float2 r##slot; \
    if (src##slot == 0u && col == 0u) { \
        r##slot = float2(0.0f); \
    } else { \
        const float2 corrected = ssb_corrected_value_precomputed_chi( \
            half_g[half_base + (size_t)col * n + src##slot], \
            src##slot, col, bf, bf_geometry, q_row, q_col, \
            chi_trig, cross_trig, params); \
        r##slot = float2(corrected.y, -corrected.x); \
    }
    LOAD_HERMITIAN(0); LOAD_HERMITIAN(1);
    LOAD_HERMITIAN(2); LOAD_HERMITIAN(3);
    LOAD_HERMITIAN(4); LOAD_HERMITIAN(5);
    LOAD_HERMITIAN(6); LOAD_HERMITIAN(7);
#undef LOAD_HERMITIAN
    ifft512_radix8_registers(
        r0, r1, r2, r3, r4, r5, r6, r7, tid, twiddle, scratch
    );

    constexpr float scale = 1.0f / 512.0f;
#define STORE_HALF(slot, row) \
    column_ifft_half[half_base + (size_t)(row) * half_cols + col] = \
        r##slot * scale
    STORE_HALF(0, tid + 0u); STORE_HALF(1, tid + 64u);
    STORE_HALF(2, tid + 128u); STORE_HALF(3, tid + 192u);
    STORE_HALF(4, tid + 256u); STORE_HALF(5, tid + 320u);
    STORE_HALF(6, tid + 384u); STORE_HALF(7, tid + 448u);
#undef STORE_HALF
}


// Complete the Hermitian inverse transform along x and accumulate the phase
// of DC/(512^2) + i*h. h is real by construction; its tiny numerical imaginary
// residue is intentionally ignored because the exact Hermitian transform is
// real-valued.
kernel void ssb_ifft512_rows_hermitian_phase_moments(
    device const float2 *column_ifft_half [[buffer(0)]],
    device const float2 *twiddle [[buffer(1)]],
    device float *phase_sum [[buffer(2)]],
    device float *phase_sumsq [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant float2 &dc [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    constexpr uint n = 512u;
    constexpr uint half_cols = 257u;
    if (tid >= 64u || row >= n) return;

    threadgroup float2 shared_rows[2][512];
    const uint src[8] = {
        octal_reverse_512(tid * 8u + 0u), octal_reverse_512(tid * 8u + 1u),
        octal_reverse_512(tid * 8u + 2u), octal_reverse_512(tid * 8u + 3u),
        octal_reverse_512(tid * 8u + 4u), octal_reverse_512(tid * 8u + 5u),
        octal_reverse_512(tid * 8u + 6u), octal_reverse_512(tid * 8u + 7u)
    };
    float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};
    float sumsq[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f};
    constexpr float scale = 1.0f / 512.0f;
    const float2 dc_spatial = dc * (scale * scale);
    const bool positive_dc = dc_spatial.x > 0.0f;
    const float inverse_dc_real = positive_dc
        ? 1.0f / dc_spatial.x : 0.0f;
    for (uint local_bf = 0u; local_bf < batch; ++local_bf) {
        const size_t half_base = (size_t)local_bf * n * half_cols +
            (size_t)row * half_cols;
        float2 r[8];
        for (uint lane = 0u; lane < 8u; ++lane) {
            const uint col = src[lane];
            if (col < half_cols) {
                r[lane] = column_ifft_half[half_base + col];
            } else {
                const float2 mirrored = column_ifft_half[half_base + (n - col)];
                r[lane] = float2(mirrored.x, -mirrored.y);
            }
        }
        threadgroup float2 *scratch = &shared_rows[local_bf & 1u][0];
        ifft512_radix8_registers(
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
            tid, twiddle, scratch
        );
        for (uint lane = 0u; lane < 8u; ++lane) {
            const float h = r[lane].x * scale;
            const float imaginary = dc_spatial.y + h;
            const float phase = positive_dc
                ? metal::atan(imaginary * inverse_dc_real)
                : metal::atan2(imaginary, dc_spatial.x);
            sum[lane] += phase;
            sumsq[lane] += phase * phase;
        }
    }
    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint col = tid + lane * 64u;
        const uint pixel = row * n + col;
        phase_sum[pixel] += sum[lane];
        phase_sumsq[pixel] += sumsq[lane];
    }
}
