#include <metal_stdlib>
using namespace metal;
// Interactive coherent (phase_of_mean) redraw at the native or a finer output grid, every BF pixel and every
// aberration term (C10 through C56), sample tilt and depth spread. The pair arithmetic is that of
// fifth_model_alias_corrected in ssb_fifth_order.metal, reorganized:
//
// 1. Only the source half (rows 0 ... N/2) is evaluated. Gamma(-q, k) = -conj(Gamma(q, k)) holds for odd orders
//    too (the minus sideband uses chi(k - q)), and G(-q) = conj(G(q)), so redraw_mirror fills the other half.
//    The output Nyquist column of the other half has no partner on the grid and is evaluated directly.
// 2. One thread owns one source frequency q0 and visits its output aliases q0 + m * (scan frequency period)
//    within each batch of 32 BF pixels, so the aliases reread that batch's G(q0, k) from cache.
// 3. The phase differences chi(k + q) - chi(k) and chi(k - q) - chi(k) are evaluated as polynomials in q expanded
//    about each BF pixel k (redraw_taylor, once per redraw): for one BF pixel and one output row they are
//    degree <= 6 polynomials in the output column frequency whose coefficients the threadgroup computes once and
//    shares, so each pair costs one Horner evaluation. Expanding about k keeps the rounding proportional to the
//    difference, not to chi itself: at the lowest frequencies the two sidebands nearly cancel and the sign of
//    their second difference chi(k + q) + chi(k - q) - 2 chi(k) decides the correction, which differences of
//    separately rounded chi values of tens of radians cannot resolve. The absolute bivariate coefficients are
//    prepared on the host in double precision from the same packed float32 terms (`SSBRedrawPolynomial`).
// 4. Each SIMD group skips BF pixels whose sidebands are closed for all its outputs (a conservative distance
//    test), and pixels with one possible sideband use conj(gamma) / |gamma| = +-exp(i delta chi). Pixels with both
//    sidebands possible form gamma from the half sum and half difference of the two phase differences, so the
//    rounding of sin and cos is not amplified where the sidebands nearly cancel.
//
// Rounding differs from the per-term kernel (chi differences expanded about k, mirrored half, fast trigonometry);
// the redraw parity tests state the tolerance.

constant uint REDRAW_F [[function_constant(0)]];       // output factor: aliases per axis (1 ... 4)
constant uint REDRAW_DEGREE [[function_constant(1)]];  // chi polynomial degree: 2 (C10/C12) or 6 (fifth order)
constant bool REDRAW_DEPTH [[function_constant(2)]];   // nonzero depth spread: sinc depth averaging
constexpr constant uint REDRAW_BATCH = 32u;            // BF pixels per shared batch (one ballot word)
constexpr constant uint REDRAW_WIDTH = 128u;           // threads per threadgroup, along the source column

struct RedrawGrid {
    uint sourceN, outputN, gCols, count, bfOffset, sourceHalf;
    uint rowLimit, colLimit;  // largest |output frequency index| with support, per axis
    float support2;           // squared outward-rounded support radius (max |k| + aperture)
    uint pixels;              // all BF pixels: the stride of one Taylor coefficient in `taylor`
};

// One BF pixel at one alias row, REDRAW_ROW floats: the plus (q + k) and minus (k - q) row coordinates and their
// squares, then chi(k +- q) - chi(k) = sum_j c[j] (+-q_col)^j for the plus and the minus sideband, each padded to
// whole float4s.
constant uint REDRAW_SET = REDRAW_DEGREE > 3u ? 8u : 4u;
constant uint REDRAW_ROW = 4u + 2u * REDRAW_SET;

inline float redraw_horner(threadgroup const float* c, float y) {
    float4 a = *(threadgroup const float4*)c;
    if (REDRAW_DEGREE <= 3u) return metal::fma(metal::fma(metal::fma(a.w, y, a.z), y, a.y), y, a.x);
    float4 b = *(threadgroup const float4*)(c + 4);
    float value = metal::fma(metal::fma(b.z, y, b.y), y, b.x);
    return metal::fma(metal::fma(metal::fma(metal::fma(value, y, a.w), y, a.z), y, a.y), y, a.x);
}

// sum_i d[i][j] (+-x)^i for each column power j: the row-collapsed coefficients of one pixel's expansion for the
// plus (+x) and the minus (-x) sideband, zero-padded. d[i][j] is at d[(i * 7 + j) * stride].
inline void redraw_collapse(device const float* d, uint stride, float x, threadgroup float* plus,
    threadgroup float* minus) {
    for (uint j = 0; j < REDRAW_SET; ++j) {
        float p = 0.0f, m = 0.0f;
        if (j <= REDRAW_DEGREE) {
            for (int i = int(REDRAW_DEGREE - j); i >= 0; --i) {
                float v = d[(uint(i) * 7u + j) * stride];
                p = metal::fma(p, x, v);
                m = metal::fma(m, -x, v);
            }
        }
        plus[j] = p;
        minus[j] = m;
    }
}

// chi(k + s) - chi(k) = sum_{i + j >= 1} d[i][j] s_row^i s_col^j for each BF pixel k: the Taylor coefficients
// d[i][j] = sum_{a >= i, b >= j} C(a, i) C(b, j) poly[a][b] k_row^(a - i) k_col^(b - j) of the absolute chi
// polynomial, coefficient-major (d[i][j] of pixel k at (i * 7 + j) * count + k) with d[0][0] = 0. One thread per
// BF pixel, before every redraw; it also writes the pixel's probe geometry (model_geometry: alpha^2, cos 2 phi,
// sin 2 phi, aperture) to probe[k * 5 ... k * 5 + 3]. The redraw does not read chi(k) (probe[k * 5 + 4]).
kernel void redraw_taylor(device const float2* kk [[buffer(0)]], constant float* params [[buffer(1)]],
    constant float* poly [[buffer(2)]], device float* probe [[buffer(3)]], device float* taylor [[buffer(4)]],
    constant uint& count [[buffer(5)]], uint k [[thread_position_in_grid]]) {
    if (k >= count) return;
    float kr = kk[k].x, kc = kk[k].y;
    float4 gk = model_geometry(kr, kc, params[0], params[1], params[2], params[3]);
    for (uint j = 0; j < 4; ++j) probe[k * 5 + j] = gk[j];
    float d[7][7];
    // Shift the row variable by kr, then the column variable by kc (repeated synthetic division, Taylor shift).
    for (uint b = 0; b < 7; ++b) {
        float c[7];
        for (uint a = 0; a < 7; ++a) c[a] = poly[a * 7 + b];
        for (uint i = 0; i < 7; ++i)
            for (int a = 5; a >= int(i); --a) c[a] = metal::fma(c[a + 1], kr, c[a]);
        for (uint a = 0; a < 7; ++a) d[a][b] = c[a];
    }
    for (uint a = 0; a < 7; ++a) {
        float c[7];
        for (uint b = 0; b < 7; ++b) c[b] = d[a][b];
        for (uint j = 0; j < 7; ++j)
            for (int b = 5; b >= int(j); --b) c[b] = metal::fma(c[b + 1], kc, c[b]);
        for (uint b = 0; a + b < 7; ++b) taylor[(a * 7 + b) * count + k] = (a + b == 0u) ? 0.0f : c[b];
    }
}

// The soft aperture weight of a shifted probe inside the outer radius (model_alias_geometry), without branches:
// (semi - alpha) / denom + 1/2 with denom = |(dx ay, dy ax)| / r, clamped to [0, 1]. Inside the inner radius the
// clamp gives exactly 1, as the per-term kernel's shortcut does.
inline float redraw_edge(float dx, float dy, float r2, float wl, float semi, float ay, float ax) {
    float r = metal::fast::sqrt(r2);
    float d2 = (dx * ay) * (dx * ay) + (dy * ax) * (dy * ax);
    float edge = metal::fma((semi - r * wl) * r, metal::fast::rsqrt(d2), 0.5f);
    return d2 > 1e-30f ? metal::clamp(edge, 0.0f, 1.0f) : 1.0f;
}

// One sideband's aperture weight: 1 or 0 unless some lane of the SIMD group lies in the soft edge ring.
inline float redraw_weight(float dx, float dy, float r2, float outer2, float inner2, float wl, float semi,
    float ay, float ax) {
    float w = r2 <= outer2 ? 1.0f : 0.0f;
    if (metal::simd_any(r2 >= inner2 && r2 <= outer2) && r2 <= outer2)
        w = redraw_edge(dx, dy, r2, wl, semi, ay, ax);
    return w;
}

inline float redraw_sinc(float x) {
    return (metal::abs(x) < 1e-6f) ? 1.0f : metal::fast::sin(x) * metal::fast::divide(1.0f, x);
}

// a / max(|a|, 1e-8) for the real sideband weight a = w * sinc(x), w >= 0: the normalized correction of a pair
// with one open sideband is this scale times exp(i delta chi). Away from the floor it is the sign of a.
inline float redraw_scale(float x, float w) {
    bool flat = !REDRAW_DEPTH || metal::abs(x) < 1e-6f;
    float num = flat ? w : w * metal::fast::sin(x), den = flat ? 1.0f : x;  // a = num / den
    if (metal::abs(num) > 2e-8f * metal::abs(den)) return num * den > 0.0f ? 1.0f : -1.0f;
    float a = num * metal::fast::divide(1.0f, den);
    return a * metal::fast::divide(1.0f, metal::max(metal::abs(a), 1e-8f));
}

// The single-sideband pairs of one alias and batch: MINUS evaluates chi(k - q), else chi(q + k).
template <bool MINUS>
inline float2 redraw_side(uint mask, threadgroup const float4* pixels, threadgroup const float* rows,
    device const float2* source, uint plane, bool mirror, float qc, float depth_q, float depth_k, float outer2,
    float inner2, float wl, float semi, float ay, float ax) {
    float h = MINUS ? -qc : qc;
    float2 acc = float2(0.0f);
    while (mask != 0u) {
        uint local = metal::ctz(mask);
        mask &= mask - 1u;
        float4 pixel = pixels[local];
        threadgroup const float* row = rows + local * REDRAW_F * REDRAW_ROW;
        float y = MINUS ? pixel.x - qc : qc + pixel.x;
        float r2 = row[MINUS ? 3 : 1] + y * y;
        float w = redraw_weight(row[MINUS ? 2 : 0], y, r2, outer2, inner2, wl, semi, ay, ax) * pixel.z;
        float x = MINUS ? metal::fma(-depth_k, r2, pixel.w + depth_q) : metal::fma(depth_k, r2, depth_q - pixel.w);
        float scale = redraw_scale(x, w);
        float2 value = source[local * plane];
        float vr = value.x, vi = mirror ? -value.y : value.y;
        float c, s = metal::fast::sincos(redraw_horner(row + (MINUS ? 4u + REDRAW_SET : 4u), h), c);
        acc += MINUS ? scale * float2(vr * c - vi * s, vi * c + vr * s)
                     : -scale * float2(vr * c + vi * s, vi * c - vr * s);
    }
    return acc;
}

// params: the 12 floats of ssb_model.metal (wavelength, semiangle, angular row, angular column, C10, C12,
// cos 2 phi12, sin 2 phi12, pi / wavelength, depth spread, tilt row, tilt column). taylor: redraw_taylor's
// expansion of chi about every BF pixel. A threadgroup owns one source row and REDRAW_WIDTH source columns; rows
// above N/2 evaluate only the output Nyquist column. One dispatch per stored batch of BF spectra adds onto `total`.
kernel void redraw_coherent(device const float2* g [[buffer(0)]], device const float2* kk [[buffer(1)]],
    constant float* params [[buffer(2)]], device const float* probe [[buffer(3)]],
    device const float* taylor [[buffer(4)]], device float2* total [[buffer(5)]],
    constant RedrawGrid& p [[buffer(6)]], device const float* q_row [[buffer(7)]],
    device const float* q_col [[buffer(8)]],
    threadgroup float4* pixels [[threadgroup(0)]], threadgroup float* rows [[threadgroup(1)]],
    threadgroup float4* intervals [[threadgroup(2)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_index [[simdgroup_index_in_threadgroup]]) {
    const uint F = REDRAW_F, A = REDRAW_F * REDRAW_F;
    uint sc = group.x * REDRAW_WIDTH + lane, sr = group.y;
    uint n = p.outputN;
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float factor = params[8], half_t = 0.5f * params[9], wl2 = wl * wl;
    float outer = (semi + 0.5f * metal::max(ay, ax)) / wl;
    outer *= 1.0f + 8.0f * 1.1920928955078125e-7f;
    float outer2 = outer * outer;
    // Inside the inner radius the soft aperture is exactly 1 (model_alias_geometry); rounded inwards.
    float inner = (semi - 0.5f * metal::max(ay, ax)) * (1.0f - 8.0f * 1.1920928955078125e-7f) / wl;
    float inner2 = inner > 0.0f ? inner * inner * (1.0f - 1e-6f) : -1.0f;
    // Depth sinc arguments: x = -depth_k |k - q|^2 + pixel term + depth_q (minus), depth_k |q + k|^2 - pixel
    // term + depth_q (plus).
    float depth_k = half_t * factor * wl2;

    // The aliases of this source frequency with support.
    uint active = 0u;
    bool upper = sr > p.sourceN / 2u;
    for (uint a = 0; a < A; ++a) {
        uint orow = sr + (a / F) * p.sourceN, ocol = sc + (a % F) * p.sourceN;
        int ir = orow < (n - 1u) / 2u + 1u ? int(orow) : int(orow) - int(n);
        int ic = ocol < (n - 1u) / 2u + 1u ? int(ocol) : int(ocol) - int(n);
        float qr = q_row[orow], qc = q_col[metal::min(ocol, n - 1u)];
        bool support = sc < p.sourceN && !(sr == 0u && sc == 0u)  // scan-mean replicas stay zero
            && (!upper || ocol == n / 2u)
            && uint(metal::abs(ir)) <= p.rowLimit && uint(metal::abs(ic)) <= p.colLimit
            && qr * qr + qc * qc <= p.support2;
        if (support) active |= 1u << a;
    }
    // A threadgroup without support leaves together, before any batch barrier.
    threadgroup uint any;
    if (lane == 0u) any = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active != 0u) any = 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (any == 0u) return;
    uint simd_active = metal::simd_or(active);

    uint source_row = sr, source_col = metal::min(sc, p.sourceN - 1u);
    bool mirror = p.sourceHalf != 0u && source_col > p.sourceN / 2u;
    if (mirror) {
        source_row = source_row == 0u ? 0u : p.sourceN - source_row;
        source_col = p.sourceN - source_col;
    }
    uint plane = p.sourceN * p.gCols;
    uint offset = source_row * p.gCols + source_col;
    // Per alias, fixed over the batches: this SIMD group's interval of active output columns (outward by an
    // ulp-scale margin) and the alias row frequency.
    threadgroup float4* own_intervals = intervals + simd_index * A;
    for (uint a = 0; a < A; ++a) {
        bool alias_active = ((active >> a) & 1u) != 0u;
        float qr = q_row[sr + (a / F) * p.sourceN], qc = q_col[metal::min(sc + (a % F) * p.sourceN, n - 1u)];
        float qmin = metal::simd_min(alias_active ? qc : INFINITY) - 1e-5f * outer;
        float qmax = metal::simd_max(alias_active ? qc : -INFINITY) + 1e-5f * outer;
        if (simd_lane == 0u) own_intervals[a] = float4(qmin, qmax, qr, 0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < p.count; base += REDRAW_BATCH) {
        uint batch = metal::min(REDRAW_BATCH, p.count - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = lane; index < batch * F; index += REDRAW_WIDTH) {
            uint local = index / F, row_alias = index % F;
            uint k = p.bfOffset + base + local;
            float2 kv = kk[k];
            threadgroup float* row = rows + index * REDRAW_ROW;
            float q = q_row[sr + row_alias * p.sourceN];
            float pr = q + kv.x, mr = kv.x - q;
            row[0] = pr; row[1] = pr * pr; row[2] = mr; row[3] = mr * mr;
            redraw_collapse(taylor + k, p.pixels, q, row + 4, row + 4 + REDRAW_SET);
            // column coordinate, (unused), probe aperture, the pixel's depth-sinc term
            if (row_alias == 0u) pixels[local] = float4(kv.y, 0.0f, probe[k * 5 + 3], half_t * factor * probe[k * 5]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float2* source = g + (size_t)base * plane + offset;
        // Lane = BF pixel of the batch for the relevance ballots.
        uint kl = metal::min(simd_lane, batch - 1u);
        float4 own = pixels[kl];
        bool lit = simd_lane < batch && own.z != 0.0f;
        for (uint a = 0; a < A; ++a) {
            if (((simd_active >> a) & 1u) == 0u) continue;
            uint ar = a / F, ocol = sc + (a % F) * p.sourceN;
            float4 interval = own_intervals[a];
            float qmin = interval.x, qmax = interval.y, qr = interval.z;
            float qc = q_col[metal::min(ocol, n - 1u)];
            threadgroup const float* rows_a = rows + ar * REDRAW_ROW;  // pixel `local` at + local * F * REDRAW_ROW
            float dp = metal::max(0.0f, metal::max(qmin + own.x, -(qmax + own.x)));
            float dm = metal::max(0.0f, metal::max(own.x - qmax, qmin - own.x));
            threadgroup const float* own_row = rows_a + kl * F * REDRAW_ROW;
            uint minus = uint(uint64_t(metal::simd_ballot(lit && own_row[3] + dm * dm <= outer2)));
            uint plus = uint(uint64_t(metal::simd_ballot(lit && own_row[1] + dp * dp <= outer2)));
            if ((minus | plus) == 0u || ((active >> a) & 1u) == 0u) continue;
            float depth_q = -half_t * 6.283185307179586f * (qr * params[10] + qc * params[11]);
            float2 acc = float2(0.0f);
            // Both sidebands possible for this SIMD group: the general formula. With delta_m = chi(k - q) - chi(k),
            // delta_p = chi(k + q) - chi(k), mu = (delta_p - delta_m) / 2 and nu = -(delta_m + delta_p) / 2,
            // gamma = a exp(-i delta_m) - b exp(+i delta_p) = exp(i mu) ((a - b) cos nu + i (a + b) sin nu): no
            // difference of two nearly equal unit vectors where the sidebands nearly cancel (|nu| << 1), so the
            // rounding of sin and cos is not amplified by 1 / |gamma|.
            uint mask = minus & plus;
            while (mask != 0u) {
                uint local = metal::ctz(mask);
                mask &= mask - 1u;
                float4 pixel = pixels[local];
                threadgroup const float* row = rows_a + local * F * REDRAW_ROW;
                float pc = qc + pixel.x, mc = pixel.x - qc;
                float r2p = row[1] + pc * pc, r2m = row[3] + mc * mc;
                float2 value = source[local * plane];
                float vr = value.x, vi = mirror ? -value.y : value.y;
                float wm = 0.0f, wp = 0.0f, dm = 0.0f, dp = 0.0f;
                #pragma clang loop unroll(disable)
                for (uint side = 0; side < 2u; ++side) {
                    bool use_m = side == 0u;
                    float y = use_m ? mc : pc, r2 = use_m ? r2m : r2p;
                    float w = redraw_weight(use_m ? row[2] : row[0], y, r2, outer2, inner2, wl, semi, ay, ax)
                        * pixel.z;
                    float x = use_m ? metal::fma(-depth_k, r2, pixel.w + depth_q)
                                    : metal::fma(depth_k, r2, depth_q - pixel.w);
                    float weight = REDRAW_DEPTH ? redraw_sinc(x) * w : w;
                    float delta = redraw_horner(row + (use_m ? 4u + REDRAW_SET : 4u), use_m ? -qc : qc);
                    wm = use_m ? weight : wm;
                    dm = use_m ? delta : dm;
                    wp = use_m ? wp : weight;
                    dp = use_m ? dp : delta;
                }
                float cm, sm = metal::fast::sincos(0.5f * (dp - dm), cm);
                float cn, sn = metal::fast::sincos(-0.5f * (dm + dp), cn);
                float zr = (wm - wp) * cn, zi = (wm + wp) * sn;
                float re = cm * zr - sm * zi, im = sm * zr + cm * zi;
                float inv = metal::fast::divide(1.0f, metal::max(metal::fast::sqrt(zr * zr + zi * zi), 1e-8f));
                acc += float2((vr * re + vi * im) * inv, (vi * re - vr * im) * inv);
            }
            // One possible sideband for the whole SIMD group: minus (chi(k - q)), then plus (chi(q + k)).
            acc += redraw_side<true>(minus & ~plus, pixels, rows_a, source, plane, mirror, qc, depth_q, depth_k,
                                     outer2, inner2, wl, semi, ay, ax);
            acc += redraw_side<false>(plus & ~minus, pixels, rows_a, source, plane, mirror, qc, depth_q, depth_k,
                                      outer2, inner2, wl, semi, ay, ax);
            total[(size_t)(sr + ar * p.sourceN) * n + ocol] += acc;
        }
    }
}

// The unevaluated half: C(q) = -conj(C(-q)) for every output whose source row lies above N/2, except the
// output Nyquist column (its partner is not on the grid), which redraw_coherent evaluates directly.
kernel void redraw_mirror(device float2* total [[buffer(0)]], constant uint2& size [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint n = size.x, source = size.y;
    uint col = tid.x, row = tid.y;
    if (col >= n || row >= n || row % source <= source / 2u || col == n / 2u) return;
    float2 v = total[(size_t)((n - row) % n) * n + (n - col) % n];
    total[(size_t)row * n + col] = float2(-v.x, v.y);
}
