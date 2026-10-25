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
constexpr constant uint DIRECT_MAX_SHARE = 192u;
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

// ssb_chi_full with the coefficients in the constant address space (uniform across the dispatch).
inline float direct_chi(float row, float col, float wl, constant float* coefficients) {
    float x = row * wl, y = col * wl;
    float r2 = x * x + y * y;
    float radial[4] = {1.0f, r2, r2 * r2, r2 * r2 * r2};
    float re[7], im[7];
    re[0] = 1.0f; im[0] = 0.0f;
    for (int m = 1; m <= 6; ++m) {
        re[m] = re[m-1] * x - im[m-1] * y;
        im[m] = im[m-1] * x + re[m-1] * y;
    }
    const int azimuth[14] = {0,2,1,3,0,2,4,1,3,5,0,2,4,6};
    const int power[14] = {1,0,1,0,2,1,0,2,1,0,3,2,1,0};
    float chi = 0.0f;
    for (int i = 0; i < 14; ++i) {
        int m = azimuth[i];
        chi += radial[power[i]] * (re[m] * coefficients[2*i] + im[m] * coefficients[2*i+1]);
    }
    return (6.283185307179586f / wl) * chi;
}

// One side of the transfer: amplitude * exp(sign * i (chi(k +- q) - chi(k))), zero where the aperture is closed.
// The soft aperture is CUDA's thick_probe_geometry (square detector pixels: the edge width is the constant sampling).
inline float2 direct_side(float dr, float dc, float sign, float shift, float4 gk, float ck, constant DirectGrid& p,
                          constant float* coefficients) {
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
    const float chi = direct_chi(dr, dc, p.wl, coefficients);
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

// w conj(v_a) v_b onto entry i (a <= b) of this part's running sums, in CUDA's order of operations.
inline void direct_add(thread float2* sum, uint i, float2 va, float2 vb, float w) {
    if (i < DIRECT_FIRST || i >= DIRECT_LAST) return;
    const float ur = w * va.x, ui = -w * va.y;
    float2 s = sum[i - DIRECT_FIRST];
    s.x = metal::fma(ur, vb.x, metal::fma(-ui, vb.y, s.x));
    s.y = metal::fma(ur, vb.y, metal::fma(ui, vb.x, s.y));
    sum[i - DIRECT_FIRST] = s;
}

// Entry (a, b), a <= b, of the row-major augmented upper triangle.
inline uint direct_entry(uint a, uint b) { return a * DIRECT_WIDTH - a * (a - 1u) / 2u + (b - a); }

// One BF pixel's products w conj(v_a) v_b. An alias whose sidebands are both closed has zero columns, so its products
// are exact zeros and skipped; the open aliases are visited in pairs (neighbouring frequencies of a SIMD group mostly
// agree, so these branches stay coherent). Every entry still receives its products in BF pixel order.
inline void direct_accumulate(thread float2* sum, thread const float2* v, float w) {
    const uint C = DIRECT_ALIASES, P = DIRECT_DIM - DIRECT_ALIASES;  // P: first phase column
    bool open[DIRECT_MAX_ALIASES];
#pragma unroll
    for (uint a = 0; a < C; ++a) {
        open[a] = v[P + a].x != 0.0f || v[P + a].y != 0.0f;
        if (DIRECT_COMPLEX) open[a] = open[a] || v[a].x != 0.0f || v[a].y != 0.0f;
    }
#pragma unroll
    for (uint a = 0; a < C; ++a) {
        if (!open[a]) continue;
#pragma unroll
        for (uint b = a; b < C; ++b) {
            if (!open[b]) continue;
            direct_add(sum, direct_entry(P + a, P + b), v[P + a], v[P + b], w);
            if (DIRECT_COMPLEX) {
                direct_add(sum, direct_entry(a, b), v[a], v[b], w);
                direct_add(sum, direct_entry(a, P + b), v[a], v[P + b], w);
                if (b != a) direct_add(sum, direct_entry(b, P + a), v[b], v[P + a], w);
            }
        }
        direct_add(sum, direct_entry(P + a, DIRECT_DIM), v[P + a], v[DIRECT_DIM], w);
        if (DIRECT_COMPLEX) direct_add(sum, direct_entry(a, DIRECT_DIM), v[a], v[DIRECT_DIM], w);
    }
    direct_add(sum, direct_entry(DIRECT_DIM, DIRECT_DIM), v[DIRECT_DIM], v[DIRECT_DIM], w);
}

kernel void direct_normal_equations(device const float2* g [[buffer(0)]],
    device const uint2* frequency [[buffer(1)]], device const float2* alias [[buffer(2)]],
    device const float4* pixel [[buffer(3)]], constant float* coefficients [[buffer(4)]],
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
        direct_accumulate(sum, v, w);
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

// The same normal equations with one thread per (frequency, alias): a threadgroup holds 32 frequencies of one alias
// group, SIMD group a evaluating alias a of each. For a tile of BF pixels every thread writes its alias's columns
// (log-amplitude and phase, or phase) to threadgroup memory; after a barrier it adds the products it owns: its
// alias against itself and against the aliases a + 1 ... a + C/2 (cyclically, so each pair is owned once), its
// right sides, and on alias 0 the signal power. Each product is formed in the upper triangle's orientation with
// the arithmetic of direct_add, so the per-pixel values and the order of the sums over BF pixels are those of
// direct_normal_equations; a thread holds about C/2 * 4 sums instead of the whole triangle.
constexpr constant uint DIRECT_LANES = 32u;
constant uint DIRECT_TILE [[function_constant(64)]];  // BF pixels per threadgroup tile
constant uint DIRECT_BLOCKS [[function_constant(65)]];  // blocks of 32 frequencies per threadgroup
constant uint DIRECT_COLUMNS = DIRECT_COMPLEX ? 2u : 1u;  // columns per alias (eta, phi) or (phi)
constant uint DIRECT_PARTNERS = DIRECT_ALIASES / 2u;  // pairs (a, a + d), d = 1 ... C/2
constant uint DIRECT_OWNED = DIRECT_COMPLEX ? 3u + 4u * DIRECT_PARTNERS + 2u : 1u + DIRECT_PARTNERS + 1u;
constexpr constant uint DIRECT_MAX_OWNED = 3u + 4u * (DIRECT_MAX_ALIASES / 2u) + 2u;

// Entry (row, column), row <= column, of the row-major augmented upper triangle (as direct_entry).
inline uint direct_index(uint row, uint column) { return row * DIRECT_WIDTH - row * (row - 1u) / 2u + (column - row); }

// w conj(va) vb onto one running sum (direct_add's arithmetic).
inline float2 direct_product(float2 sum, float2 va, float2 vb, float w) {
    const float ur = w * va.x, ui = -w * va.y;
    return float2(metal::fma(ur, vb.x, metal::fma(-ui, vb.y, sum.x)), metal::fma(ur, vb.y, metal::fma(ui, vb.x, sum.y)));
}

kernel void direct_normal_tiled(device const float2* g [[buffer(0)]],
    device const uint2* frequency [[buffer(1)]], device const float2* alias [[buffer(2)]],
    device const float4* pixel [[buffer(3)]], constant float* coefficients [[buffer(4)]],
    device float4* sums [[buffer(5)]], constant DirectGrid& p [[buffer(6)]],
    threadgroup float2* columns [[threadgroup(0)]], threadgroup float2* spectra [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    const uint C = DIRECT_ALIASES, S = DIRECT_COLUMNS;
    // R blocks of 32 frequencies per threadgroup when the aliases alone would leave it small
    const uint R = DIRECT_BLOCKS, simd = tid / DIRECT_LANES, lane = tid % DIRECT_LANES;
    const uint a = simd % C, block = simd / C;
    const uint f = (group * R + block) * DIRECT_LANES + lane;
    columns += block * DIRECT_TILE * C * S * DIRECT_LANES;
    spectra += block * DIRECT_TILE * DIRECT_LANES;
    const bool valid = f < p.frequencies;
    const uint fc = valid ? f : p.frequencies - 1u;
    const uint2 rc = frequency[fc];
    const uint column = rc.x * p.gCols + rc.y;
    const float2 q = alias[fc * C + a];
    const float shift = 6.283185307f * (q.x * p.tiltRow + q.y * p.tiltColumn);
    // Column indices of this alias: log-amplitude a, phase P + a; the right side is column C * S.
    const uint P = DIRECT_COMPLEX ? C : 0u, D = C * S;
    float2 sum[DIRECT_MAX_OWNED];
#pragma unroll
    for (uint i = 0; i < DIRECT_OWNED; ++i) sum[i] = float2(0.0f);
    float2 power = float2(0.0f);  // the signal power, on alias 0
    for (uint base = 0; base < p.count; base += DIRECT_TILE) {
        const uint tile = metal::min(DIRECT_TILE, p.count - base);
        for (uint t = 0; t < tile; ++t) {
            const uint j = base + t;
            const float4 pk = pixel[2 * (p.bfOffset + j)], gk = pixel[2 * (p.bfOffset + j) + 1];
            const float2 t1 = direct_side(pk.x - q.x, pk.y - q.y, -1.0f, shift, gk, pk.z, p, coefficients);
            const float2 t2 = direct_side(pk.x + q.x, pk.y + q.y, 1.0f, shift, gk, pk.z, p, coefficients);
            const uint at = (t * C + a) * S * DIRECT_LANES + lane;
            columns[at + (S - 1u) * DIRECT_LANES] = float2(-(t1.y - t2.y), t1.x - t2.x);
            if (DIRECT_COMPLEX) columns[at] = t1 + t2;
        }
        // the tile's spectra values, loaded once per threadgroup (SIMD group a takes pixels a, a + C, ...)
        for (uint t = a; t < tile; t += C) {
            spectra[t * DIRECT_LANES + lane] = g[(size_t)(base + t) * p.planeStride + column];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < tile; ++t) {
            const uint j = base + t;
            const float w = pixel[2 * (p.bfOffset + j)].w;
            const uint own = (t * C + a) * S * DIRECT_LANES + lane;
            const float2 phi = columns[own + (S - 1u) * DIRECT_LANES];
            const float2 eta = DIRECT_COMPLEX ? columns[own] : float2(0.0f);
            const bool closed = phi.x == 0.0f && phi.y == 0.0f && eta.x == 0.0f && eta.y == 0.0f;
            if (closed && a != 0u) continue;
            const float2 gv = spectra[t * DIRECT_LANES + lane];
            if (a == 0u) power = direct_product(power, gv, gv, w);
            if (closed) continue;
            uint i = 0;
            // the alias against itself
            if (DIRECT_COMPLEX) {
                sum[i] = direct_product(sum[i], eta, eta, w); ++i;
                sum[i] = direct_product(sum[i], eta, phi, w); ++i;
            }
            sum[i] = direct_product(sum[i], phi, phi, w); ++i;
#pragma unroll
            for (uint d = 1; d <= DIRECT_PARTNERS; ++d) {
                const uint b = (a + d) % C;
                const bool owned = 2u * d < C || a < C / 2u;
                const uint other = (t * C + b) * S * DIRECT_LANES + lane;
                const float2 phib = columns[other + (S - 1u) * DIRECT_LANES];
                const float2 etab = DIRECT_COMPLEX ? columns[other] : float2(0.0f);
                const float use = owned ? w : 0.0f;
                // products in the upper triangle's orientation: the lower column index is conjugated
                if (DIRECT_COMPLEX) {
                    sum[i] = a < b ? direct_product(sum[i], eta, etab, use) : direct_product(sum[i], etab, eta, use); ++i;
                    sum[i] = direct_product(sum[i], eta, phib, use); ++i;  // (a, P + b)
                    sum[i] = direct_product(sum[i], etab, phi, use); ++i;  // (b, P + a)
                }
                sum[i] = a < b ? direct_product(sum[i], phi, phib, use) : direct_product(sum[i], phib, phi, use); ++i;
            }
            if (DIRECT_COMPLEX) { sum[i] = direct_product(sum[i], eta, gv, w); ++i; }
            sum[i] = direct_product(sum[i], phi, gv, w); ++i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!valid) return;
    device float4* out = sums + (size_t)f * DIRECT_OUTPUTS;
    uint at[DIRECT_MAX_OWNED];
    uint i = 0;
    if (DIRECT_COMPLEX) { at[i++] = direct_index(a, a); at[i++] = direct_index(a, P + a); }
    at[i++] = direct_index(P + a, P + a);
#pragma unroll
    for (uint d = 1; d <= DIRECT_PARTNERS; ++d) {
        const uint b = (a + d) % C;
        if (DIRECT_COMPLEX) {
            at[i++] = direct_index(metal::min(a, b), metal::max(a, b));
            at[i++] = direct_index(a, P + b);
            at[i++] = direct_index(b, P + a);
        }
        at[i++] = direct_index(P + metal::min(a, b), P + metal::max(a, b));
    }
    if (DIRECT_COMPLEX) at[i++] = direct_index(a, D);
    at[i++] = direct_index(P + a, D);
    const uint count = i;
#pragma unroll
    for (uint e = 0; e < DIRECT_OWNED; ++e) {
        if (e >= count) break;
        // pairs at d = C/2 of the upper half are the other half's (their sums stayed zero)
        float4 acc = out[at[e]];
        const float2 value = sum[e];
        if (value.x == 0.0f && value.y == 0.0f) continue;
        const float re = acc.x + value.x, rv = re - acc.x;
        acc.z += (acc.x - (re - rv)) + (value.x - rv);
        const float im = acc.y + value.y, iv = im - acc.y;
        acc.w += (acc.y - (im - iv)) + (value.y - iv);
        acc.x = re;
        acc.y = im;
        out[at[e]] = acc;
    }
    if (a == 0u) {
        float4 acc = out[direct_index(D, D)];
        const float re = acc.x + power.x, rv = re - acc.x;
        acc.z += (acc.x - (re - rv)) + (power.x - rv);
        const float im = acc.y + power.y, iv = im - acc.y;
        acc.w += (acc.y - (im - iv)) + (power.y - iv);
        acc.x = re;
        acc.y = im;
        out[direct_index(D, D)] = acc;
    }
}

// Drag windows: a Chebyshev series in one control of every prepared entry (Python's CUDA `DragWindow`). Each node's
// exact normal equations (float-float, as direct_normal_* leave them) add weight[n] times their value to coefficient n
// of every entry, in float-float; a complete series is rounded to float32 per coefficient, and a preview evaluates
// sum_n c_n T_n(x) in float-float for the regularized solve.

// a + b of two float-float values (hi, lo), each component of a complex entry.
inline float4 direct_two_sum(float4 a, float2 bh, float2 bl) {
    const float re = a.x + bh.x, rv = re - a.x;
    const float im = a.y + bh.y, iv = im - a.y;
    const float lre = a.z + bl.x + (a.x - (re - rv)) + (bh.x - rv);
    const float lim = a.w + bl.y + (a.y - (im - iv)) + (bh.y - iv);
    const float hre = re + lre, him = im + lim;
    return float4(hre, him, lre - (hre - re), lim - (him - im));
}

// coefficient[n][i] += weight[n] * node[i] for n = 0 ... count - 1 (p = (entries, count)).
kernel void direct_window_accumulate(device const float4* node [[buffer(0)]], device float4* coefficient [[buffer(1)]],
    constant float* weight [[buffer(2)]], constant uint2& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    const float4 v = node[i];
    for (uint n = 0; n < p.y; ++n) {
        const float w = weight[n];
        // w * (hi + lo): the rounded product and its error, plus w * lo
        const float2 hi = float2(w * v.x, w * v.y);
        const float2 lo = float2(metal::fma(w, v.x, -hi.x) + w * v.z, metal::fma(w, v.y, -hi.y) + w * v.w);
        device float4* c = coefficient + (size_t)n * p.x + i;
        *c = direct_two_sum(*c, hi, lo);
    }
}

// The float32 coefficients of a complete series: hi + lo rounded once (p = entries * count).
kernel void direct_window_finalize(device const float4* coefficient [[buffer(0)]], device float2* rounded [[buffer(1)]],
    constant uint& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p) return;
    const float4 c = coefficient[i];
    rounded[i] = float2(c.x + c.z, c.y + c.w);
}

// entry[i] = sum_n c_n[i] T_n(x) in float-float, from T_n(x) as (hi, lo) (p = (entries, count)); the result in the
// prepared sums' layout (hi re, hi im, lo re, lo im).
kernel void direct_window_evaluate(device const float2* coefficient [[buffer(0)]], device float4* entry [[buffer(1)]],
    constant float2* basis [[buffer(2)]], constant uint2& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    float4 sum = float4(0.0f);
    for (uint n = 0; n < p.y; ++n) {
        const float2 c = coefficient[(size_t)n * p.x + i], t = basis[n];
        const float2 hi = c * t.x;
        const float2 lo = float2(metal::fma(c.x, t.x, -hi.x) + c.x * t.y, metal::fma(c.y, t.x, -hi.y) + c.y * t.y);
        sum = direct_two_sum(sum, hi, lo);
    }
    entry[i] = sum;
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
