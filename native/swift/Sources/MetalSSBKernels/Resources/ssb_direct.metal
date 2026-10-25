#include <metal_stdlib>
using namespace metal;
// Regularized direct reconstruction (methods "phase" and "complex"): the normal equations of every scan frequency
// with all its supported output aliases, as Python's CUDA `ssb.cuda.direct` (`direct_normal_equations`).
// Compiled after ssb_joint.metal (joint_geometry_xy, joint_sinc) and ssb_fifth_order.metal (ssb_chi_full).
//
// For T = exp(eta + i phi) ~ 1 + eta + i phi, the measured spectrum of BF pixel k at scan frequency Q is modelled as
// sum over the supported aliases q of Q of (t1 + t2) eta_q + i (t1 - t2) phi_q with t1 = A(k-q) A(k) w1
// exp(-i (chi(k-q) - chi(k))) and t2 = A(k+q) A(k) w2 exp(i (chi(k+q) - chi(k))) (w1, w2 the depth sincs with the
// tilt shift). Per frequency the augmented vector V = (T_1 .. T_D, G) at each BF pixel gives the weighted Gram
// V^H W V: the normal matrix, the right side and the signal power.
//
// One thread owns one frequency and walks a batch of BF pixels (the engine's cached spectra, so neighbouring
// threads read neighbouring columns of one spectrum). It evaluates the transfer and adds the entries
// [DIRECT_FIRST, DIRECT_LAST) of the augmented upper triangle (row-major, a <= b) to float32 running sums, as the
// CUDA lanes do; the batch sums then meet in a float-float accumulator, so the sum over all BF pixels keeps about
// float64 precision across batches. Larger systems split their entries over several dispatches (parts), each
// evaluating the transfer again, rather than spilling their sums out of registers.

constant uint DIRECT_ALIASES [[function_constant(60)]];  // supported aliases per frequency (1 ... 16)
constant bool DIRECT_COMPLEX [[function_constant(61)]];  // log-amplitude and phase (true) or phase only
constant uint DIRECT_FIRST [[function_constant(62)]];    // first augmented entry of this part
constant uint DIRECT_LAST [[function_constant(63)]];     // one past the last entry of this part
constexpr constant uint DIRECT_MAX_ALIASES = 16u;
constexpr constant uint DIRECT_MAX_SHARE = 96u;
constant uint DIRECT_DIM = DIRECT_COMPLEX ? 2u * DIRECT_ALIASES : DIRECT_ALIASES;
constant uint DIRECT_WIDTH = DIRECT_DIM + 1u;
constant uint DIRECT_OUTPUTS = DIRECT_WIDTH * (DIRECT_WIDTH + 1u) / 2u;

struct DirectGrid {
    uint frequencies;  // frequencies of this alias group
    uint count;        // BF pixels of this batch
    uint bfOffset;     // active index of the batch's first BF pixel
    uint planeStride;  // complex values per BF spectrum of the source (half or full plane)
    uint gCols;        // stored columns per spectrum row
    float wl, semiangle, angularRow, angularColumn;
    float factor;      // pi / wavelength, as the SSB transfer
    float depth;       // effective depth spread, A
    float tiltRow, tiltColumn;  // rad
    float outer2;      // r^2 beyond which the soft aperture is exactly zero, with a margin
};

// The k-only terms of every active BF pixel, two float4 each: (kr, kc, chi(k), detector weight) and
// (alpha^2(k), aperture(k), 0, 0).
kernel void direct_pixel_terms(device const float2* k [[buffer(0)]], device const float* weights [[buffer(1)]],
    device const float* coefficients [[buffer(2)]], device float4* pixel [[buffer(3)]],
    constant DirectGrid& p [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    if (j >= p.count) return;
    const float2 kj = k[j];
    const float4 g = joint_geometry_xy(kj.x, kj.y, p.wl, p.semiangle, p.angularRow, p.angularColumn);
    pixel[2 * j] = float4(kj.x, kj.y, ssb_chi_full(kj.x, kj.y, p.wl, coefficients), weights[j]);
    pixel[2 * j + 1] = float4(g.x, g.w, 0.0f, 0.0f);
}

// sin(x) / x (1 at |x| < 1e-6, as joint_sinc), the sine of x reduced to [-pi, pi] by fast::sin.
inline float direct_sinc(float x) {
    if (metal::abs(x) < 1e-6f) return 1.0f;
    const float turns = metal::rint(x * 0.159154943f);
    const float reduced = metal::fma(-turns, -1.74845553e-7f, metal::fma(-turns, 6.28318548f, x));
    return metal::fast::sin(reduced) / x;
}

// One side of the transfer: amplitude * exp(sign * i (chi(k +- q) - chi(k))), zero where the aperture is closed.
// The soft aperture is CUDA's thick_probe_geometry (square detector pixels: the edge width is the constant sampling).
inline float2 direct_side(float dr, float dc, float sign, float shift, float4 gk, float ck, constant DirectGrid& p,
                          device const float* coefficients) {
    const float r2 = dr * dr + dc * dc;
    if (r2 >= p.outer2) return float2(0.0f);
    float4 g;
    if (p.angularRow == p.angularColumn) {
        const float edge = r2 > 1e-30f ? metal::fma(p.semiangle - metal::sqrt(r2) * p.wl, 1.0f / p.angularColumn, 0.5f)
                                       : 1.0f;
        g = float4(r2 * p.wl * p.wl, 0.0f, 0.0f, metal::clamp(edge, 0.0f, 1.0f));
    } else {
        g = joint_geometry_xy(dr, dc, p.wl, p.semiangle, p.angularRow, p.angularColumn);
    }
    if (g.w * gk.y == 0.0f) return float2(0.0f);
    const float chi = ssb_chi_full(dr, dc, p.wl, coefficients);
    const float w = p.depth != 0.0f ? direct_sinc(0.5f * p.depth * (sign * p.factor * (g.x - gk.x) - shift)) : 1.0f;
    const float amplitude = g.w * gk.y * w;
    // chi - ck reduced to [-pi, pi] (2 pi split in two floats, fused), then the fast sine and cosine.
    const float x = chi - ck;
    const float turns = metal::rint(x * 0.159154943f);
    const float reduced = metal::fma(-turns, -1.74845553e-7f, metal::fma(-turns, 6.28318548f, x));
    float c;
    const float s = metal::fast::sincos(reduced, c);
    return float2(amplitude * c, sign * amplitude * s);
}

kernel void direct_normal_equations(device const float2* g [[buffer(0)]],
    device const uint2* frequency [[buffer(1)]], device const float2* alias [[buffer(2)]],
    device const float4* pixel [[buffer(3)]], device const float* coefficients [[buffer(4)]],
    device float4* sums [[buffer(5)]], constant DirectGrid& p [[buffer(6)]], uint f [[thread_position_in_grid]]) {
    if (f >= p.frequencies) return;
    const uint2 rc = frequency[f];
    const uint column = rc.x * p.gCols + rc.y;
    float qr[DIRECT_MAX_ALIASES], qc[DIRECT_MAX_ALIASES], shift[DIRECT_MAX_ALIASES];
#pragma unroll
    for (uint a = 0; a < DIRECT_ALIASES; ++a) {
        const float2 q = alias[f * DIRECT_ALIASES + a];
        qr[a] = q.x;
        qc[a] = q.y;
        shift[a] = 6.283185307f * (q.x * p.tiltRow + q.y * p.tiltColumn);
    }
    float2 sum[DIRECT_MAX_SHARE];
#pragma unroll
    for (uint i = 0; i < DIRECT_LAST - DIRECT_FIRST; ++i) sum[i] = float2(0.0f);
    for (uint j = 0; j < p.count; ++j) {
        const float4 pk = pixel[2 * (p.bfOffset + j)], gk = pixel[2 * (p.bfOffset + j) + 1];
        const float kr = pk.x, kc = pk.y, ck = pk.z, w = pk.w;
        float2 v[2u * DIRECT_MAX_ALIASES + 1u];
        v[DIRECT_DIM] = g[(size_t)j * p.planeStride + column];
#pragma unroll
        for (uint a = 0; a < DIRECT_ALIASES; ++a) {
            const float2 t1 = direct_side(kr - qr[a], kc - qc[a], -1.0f, shift[a], gk, ck, p, coefficients);
            const float2 t2 = direct_side(kr + qr[a], kc + qc[a], 1.0f, shift[a], gk, ck, p, coefficients);
            // i (t1 - t2): the phase column; t1 + t2: the log-amplitude column
            v[DIRECT_DIM - DIRECT_ALIASES + a] = float2(-(t1.y - t2.y), t1.x - t2.x);
            if (DIRECT_COMPLEX) v[a] = t1 + t2;
        }
        // Columns of an alias with both sidebands closed are zero: skip their products (the SIMD group's
        // neighbouring frequencies mostly agree, so the branches stay coherent).
#pragma unroll
        for (uint a = 0; a < DIRECT_WIDTH; ++a) {
            if (a < DIRECT_DIM && v[a].x == 0.0f && v[a].y == 0.0f) continue;
            // w conj(v_a) v_b
            const float ur = w * v[a].x, ui = -w * v[a].y;
#pragma unroll
            for (uint b = a; b < DIRECT_WIDTH; ++b) {
                const uint i = a * DIRECT_WIDTH - a * (a - 1u) / 2u + (b - a);
                if (i >= DIRECT_FIRST && i < DIRECT_LAST) {
                    if (b < DIRECT_DIM && v[b].x == 0.0f && v[b].y == 0.0f) continue;
                    float2 s = sum[i - DIRECT_FIRST];
                    s.x = metal::fma(ur, v[b].x, metal::fma(-ui, v[b].y, s.x));
                    s.y = metal::fma(ur, v[b].y, metal::fma(ui, v[b].x, s.y));
                    sum[i - DIRECT_FIRST] = s;
                }
            }
        }
    }
    // Float-float accumulation of this batch's sums (two-sum on each component).
    device float4* out = sums + (size_t)f * DIRECT_OUTPUTS + DIRECT_FIRST;
#pragma unroll
    for (uint i = 0; i < DIRECT_LAST - DIRECT_FIRST; ++i) {
        float4 a = out[i];
        const float re = a.x + sum[i].x, rv = re - a.x;
        a.z += (a.x - (re - rv)) + (sum[i].x - rv);
        const float im = a.y + sum[i].y, iv = im - a.y;
        a.w += (a.y - (im - iv)) + (sum[i].y - iv);
        a.x = re;
        a.y = im;
        out[i] = a;
    }
}

// The DC of each spectrum of a batch (the scan sum of each BF pixel), for the incident dose and detector weights.
kernel void direct_dc(device const float2* g [[buffer(0)]], device float* dc [[buffer(1)]],
    constant uint3& p [[buffer(2)]], uint j [[thread_position_in_grid]]) {
    // p = (count, planeStride, first active index)
    if (j < p.x) dc[p.z + j] = g[(size_t)j * p.y].x;
}

// The inverse FFT of eta + i phi (both spectra Hermitian, packed as one) to the wave exp(eta + i phi): the phase
// angle(wave) and, for the complex method, the relative amplitude |wave|. Flags a non-finite result.
kernel void direct_wave(device const float2* field [[buffer(0)]], device float* phase [[buffer(1)]],
    device float* amplitude [[buffer(2)]], device atomic_uint* invalid [[buffer(3)]],
    constant uint2& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    // p = (count, write amplitude)
    if (i >= p.x) return;
    const float eta = field[i].x, phi = field[i].y;
    const float scale = metal::precise::exp(eta);
    float c;
    const float s = metal::precise::sincos(phi, c);
    const float re = scale * c, im = scale * s;
    const float angle = metal::precise::atan2(im, re);
    const float magnitude = metal::precise::sqrt(re * re + im * im);
    phase[i] = angle;
    if (p.y != 0u) amplitude[i] = magnitude;
    if (!metal::isfinite(re) || !metal::isfinite(im)) atomic_store_explicit(invalid, 1u, memory_order_relaxed);
}
