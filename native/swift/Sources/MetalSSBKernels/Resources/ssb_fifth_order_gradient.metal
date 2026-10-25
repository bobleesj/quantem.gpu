#include <metal_stdlib>
using namespace metal;
// Exact gradient of the fifth-order agreement S = sum_q w_q |N_q|^2 / D_q (ssb-fifth-gradient v0.2), on all BF
// pixels or every kStride-th, after the value pass `fifth_agreement_rows` (ssb_fifth_agreement.metal) of the same
// candidates: N_q = sum_k G conj(gamma), D_q = sum_k |gamma|^2, gamma = b1 e^{-i phi1} - b2 e^{i phi2},
// phi1 = chi(k-q) - chi(k), phi2 = chi(k+q) - chi(k). With a_q = N_q / D_q and R = conj(a_q) (G - a_q gamma),
// dS/dtheta = sum_{q,k} 2 w_q Re(dgamma/dtheta conj(R)).
// chi is linear in the 28 packed coefficients: dgamma/dc_j = -i (dphi1_j u1 + dphi2_j u2), u1 = b1 e^{-i phi1},
// u2 = b2 e^{i phi2}, so dS/dc_j = sum 2w [alpha1 dphi1_j + alpha2 dphi2_j], alpha1 = Im(u1 conj R),
// alpha2 = Im(u2 conj R). The depth t and tilt enter through b1 = a1 sinc(x1), b2 = a2 sinc(x2):
// dgamma/db1 = e^{-i phi1}, dgamma/db2 = -e^{i phi2}.
//
// The per-pair terms are those of fifth_agreement_rows: a threadgroup owns up to 128 band frequencies of at most two
// band rows and shares the row-collapsed chi polynomial of each BF pixel (one degree-6 Horner evaluation per
// sideband); the soft aperture is exact outside the one-pixel edge ring; fast::sincos and Markstein division.
// The coefficient gradient needs no per-pair basis: chi is a polynomial of degree <= 6 in k, so dS/dc is linear in
// the monomial sums T_a = sum 2w [alpha1 ((k-q)^a - k^a) + alpha2 ((k+q)^a - k^a)], 2 <= |a| <= 6, and by the
// binomial expansion T_a = sum_{a' < a} C(a, a') q^(a - a') [alpha2 + (-1)^|a - a'| alpha1] k^a'. A thread (one
// band frequency) accumulates the moments E_a' = sum_k (alpha1 + alpha2) k^a' (|a'| <= 4) and
// O_a' = sum_k (alpha2 - alpha1) k^a' (|a'| <= 5) with the monomials k^a' shared, 36 FMA per pair, shifts them by
// its q once, and the host maps the 25 monomial sums to the 28 packed coefficients in double. Per-term arithmetic is
// float32 like the objective; the threadgroup sums are reduced on the host in double.

// a_q = N_q / D_q of the value pass (in the chunk order of `fifth_power`), |a_q|^2 and 2 w_q per band frequency and
// candidate. A frequency without model support (D_q = 0) carries no gradient.
kernel void fifth_gradient_scale(device const float4* partial [[buffer(0)]], device const float* weight [[buffer(1)]],
    device float4* scale [[buffer(2)]], constant FifthGrid& p [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.bandCount || tid.y >= p.batch) return;
    float3 sum = float3(0.0f);
    for (uint chunk = 0; chunk < p.chunks; ++chunk)
        sum += partial[((size_t)chunk * p.batch + tid.y) * p.bandCount + tid.x].xyz;
    float4 s = float4(0.0f);
    if (sum.z > 0.0f) {
        float ar = sum.x / sum.z, ai = sum.y / sum.z;
        s = float4(ar, ai, ar * ar + ai * ai, 2.0f * weight[tid.x]);
    }
    scale[(size_t)tid.y * p.bandCount + tid.x] = s;
}

struct FifthGradientRowsGrid {
    uint count, bandCount, batch, kStride;  // count: all BF pixels; used pixel j is k[j * kStride]
    uint span, spans, tiles;                // used pixels per threadgroup, spans of them, band tiles
    float wl, semiangle, angularRow, angularColumn, factor;
    float inner2, outer2;                   // squared radii where the soft aperture is exactly 1 / exactly 0
};

// sinc(x) as `fifth_sinc` and its slope (cos x - sinc x) / x, by its series where the difference cancels.
inline float fifth_sinc_slope(float x, thread float& sinc) {
    float c;
    float s = metal::fast::sincos(x, c);
    sinc = (metal::abs(x) < 1e-6f) ? 1.0f : fifth_quotient(s, x);
    float x2 = x * x;
    if (metal::abs(x) < 0.25f) return x * (-1.0f / 3.0f + x2 * (1.0f / 30.0f - x2 * (1.0f / 840.0f)));
    return fifth_quotient(c - sinc, x);
}

// Monomials x^a y^b of a BF pixel (row, column) in degree order, index (a + b)(a + b + 1) / 2 + b: the even moments
// use degree <= 4 (15), the odd moments degree <= 5 (21).
constant float FIFTH_BINOMIAL[7][7] = {
    {1, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 2, 1, 0, 0, 0, 0}, {1, 3, 3, 1, 0, 0, 0},
    {1, 4, 6, 4, 1, 0, 0}, {1, 5, 10, 10, 5, 1, 0}, {1, 6, 15, 20, 15, 6, 1}};

// tiles: as fifth_agreement_rows. poly: 49 floats per candidate (SSBRedrawPolynomial). chi: chi(k) per (pixel,
// candidate) of the value pass. scale: fifth_gradient_scale. Grid: (tile, span, candidate), one candidate per
// thread. Output per threadgroup: 28 floats [d/d depth, d/d tilt row, d/d tilt column, then T_a for 2 <= |a| <= 6
// in degree order] at ((candidate * spans + span) * tiles + tile) * 28.
kernel void fifth_gradient_rows(device const float2* g [[buffer(0)]], device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]], device float* partial [[buffer(3)]],
    constant FifthGradientRowsGrid& p [[buffer(4)]], device const float* trial [[buffer(5)]],
    device const float4* geometry [[buffer(6)]], device const float* poly [[buffer(7)]],
    device const uint4* tiles [[buffer(8)]], device const float* chi [[buffer(9)]],
    device const float4* scale [[buffer(10)]],
    threadgroup float4* coeff [[threadgroup(0)]], threadgroup float4* mono [[threadgroup(1)]],
    threadgroup float* shared [[threadgroup(2)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 size [[threads_per_threadgroup]]) {
    const uint S = FIFTH_S, W = size.x;
    uint4 tile = tiles[group.x];
    uint span = group.y, b = group.z;
    bool valid = lane < tile.y;
    uint qi = tile.x + metal::min(lane, tile.y - 1u);
    float qx = q[qi].x, qy = q[qi].y;
    uint slot = lane >= tile.z ? 1u : 0u, rows = tile.w;
    float qrow0 = q[tile.x].x, qrow1 = q[tile.x + metal::min(tile.z, tile.y - 1u)].x;
    threadgroup float* polys = shared;      // 49 floats
    threadgroup float* sums = shared + 52;  // 28 floats per simdgroup
    for (uint i = lane; i < 49u; i += W) polys[i] = poly[b * 49u + i];
    float4 s = scale[(size_t)b * p.bandCount + qi];
    bool active = valid && s.w != 0.0f;
    device const float* row = trial + b * 31u;
    float depth = row[0];
    float shift = 6.283185307179586f * (qx * row[1] + qy * row[2]);
    float ar = s.x, ai = s.y, aa = s.z;
    // Moments of the monomials 4j ... 4j + 3 (vector FMAs; even uses 15 of 16, odd 21 of 24).
    float4 even[4], odd[6];
    for (int j = 0; j < 4; ++j) even[j] = float4(0.0f);
    for (int j = 0; j < 6; ++j) odd[j] = float4(0.0f);
    float depthSum = 0.0f, tiltSum = 0.0f;
    uint used = (p.count + p.kStride - 1u) / p.kStride;
    uint k0 = span * p.span, k1 = metal::min(used, k0 + p.span);
    for (uint start = k0; start < k1; start += S) {
        uint n = metal::min(S, k1 - start);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // chi(x, y) = sum_ij c_ij x^i y^j collapsed at x = k_row -/+ q_row for every pixel and band row.
        for (uint index = lane; index < rows * n * 2u; index += W) {
            uint row_slot = index / (n * 2u), rest = index % (n * 2u);
            uint pix = rest >> 1, side = rest & 1u;
            float qrow = row_slot == 0u ? qrow0 : qrow1;
            float x = k[(start + pix) * p.kStride].x + (side == 0u ? -qrow : qrow);
            float c[7];
            for (uint j = 0; j < 7; ++j) {
                float value = 0.0f;
                for (int i = int(6 - j); i >= 0; --i) value = metal::fma(value, x, polys[i * 7 + j]);
                c[j] = value;
            }
            threadgroup float4* out = coeff + ((row_slot * S + pix) * 2u + side) * 2u;
            out[0] = float4(c[0], c[1], c[2], c[3]);
            out[1] = float4(c[4], c[5], c[6], 0.0f);
        }
        // The monomials k_row^a k_col^b, a + b <= 5, of every pixel.
        for (uint pix = lane; pix < n; pix += W) {
            float2 kv = k[(start + pix) * p.kStride];
            float xp[6], yp[6];
            xp[0] = 1.0f;
            yp[0] = 1.0f;
            for (int i = 1; i < 6; ++i) {
                xp[i] = xp[i - 1] * kv.x;
                yp[i] = yp[i - 1] * kv.y;
            }
            float m[24];
            for (int d = 0; d < 6; ++d)
                for (int bb = 0; bb <= d; ++bb) m[d * (d + 1) / 2 + bb] = xp[d - bb] * yp[bb];
            m[21] = 0.0f;
            m[22] = 0.0f;
            m[23] = 0.0f;
            threadgroup float4* out = mono + pix * 6u;
            for (int j = 0; j < 6; ++j) out[j] = float4(m[4 * j], m[4 * j + 1], m[4 * j + 2], m[4 * j + 3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) continue;
        for (uint pix = 0; pix < n; ++pix) {
            uint kk = (start + pix) * p.kStride;
            float kxv = k[kk].x, kyv = k[kk].y;
            float4 gk = geometry[kk];
            float mx = qx - kxv, my = qy - kyv, px = qx + kxv, py = qy + kyv;
            float rm = mx * mx + my * my, rp = px * px + py * py;
            float sm = metal::sqrt(rm), sq = metal::sqrt(rp);
            float am = sm * p.wl, ap = sq * p.wl;
            bool ring_m = rm > p.inner2 && rm < p.outer2, ring_p = rp > p.inner2 && rp < p.outer2;
            float em = rm <= p.inner2 ? 1.0f : 0.0f, eq = rp <= p.inner2 ? 1.0f : 0.0f;
            FIFTH_UNIFORM_IF(ring_m) em = fifth_edge(mx, my, sm, am, p.semiangle, p.angularRow, p.angularColumn);
            FIFTH_UNIFORM_IF(ring_p) eq = fifth_edge(px, py, sq, ap, p.semiangle, p.angularRow, p.angularColumn);
            float a1 = em * gk.w, a2 = eq * gk.w;
            bool open_m = a1 != 0.0f, open_p = a2 != 0.0f;
            if (!open_m && !open_p) continue;
            float alpha2m = am * am;
            float alpha2p = ap * ap;
            float2 value = g[(size_t)kk * p.bandCount + qi];
            float gre = value.x, gim = value.y;
            float chi_k = chi[(size_t)kk * p.batch + b];
            threadgroup const float4* own = coeff + (slot * S + pix) * 4u;
            float b1 = 0.0f, c1 = 0.0f, s1 = 0.0f, d1 = 0.0f, h1 = 0.0f;
            float b2 = 0.0f, cp = 0.0f, sp = 0.0f, d2 = 0.0f, h2 = 0.0f;
            FIFTH_UNIFORM_IF(open_m) {
                s1 = metal::fast::sincos(fifth_horner(own, kyv - qy) - chi_k, c1);
                b1 = a1;
                FIFTH_UNIFORM_IF(depth != 0.0f) {
                    h1 = -p.factor * (alpha2m - gk.x) - shift;
                    float sinc;
                    d1 = fifth_sinc_slope(0.5f * depth * h1, sinc) * a1;
                    b1 = sinc * a1;
                }
            }
            FIFTH_UNIFORM_IF(open_p) {
                sp = metal::fast::sincos(fifth_horner(own + 2u, qy + kyv) - chi_k, cp);
                b2 = a2;
                FIFTH_UNIFORM_IF(depth != 0.0f) {
                    h2 = p.factor * (alpha2p - gk.x) - shift;
                    float sinc;
                    d2 = fifth_sinc_slope(0.5f * depth * h2, sinc) * a2;
                    b2 = sinc * a2;
                }
            }
            float gr = b1 * c1 - b2 * cp;
            float gi = -b1 * s1 - b2 * sp;
            float rr = ar * gre + ai * gim - aa * gr;
            float ri = ar * gim - ai * gre - aa * gi;
            float alpha1 = -b1 * (c1 * ri + s1 * rr);
            float alpha2 = b2 * (sp * rr - cp * ri);
            float beta1 = c1 * rr - s1 * ri;
            float beta2 = -(cp * rr + sp * ri);
            tiltSum += beta1 * d1 + beta2 * d2;
            depthSum += beta1 * d1 * h1 + beta2 * d2 * h2;
            float e = alpha1 + alpha2, o = alpha2 - alpha1;
            threadgroup const float4* mp = mono + pix * 6u;
            for (int j = 0; j < 4; ++j) even[j] = metal::fma(float4(e), mp[j], even[j]);
            for (int j = 0; j < 6; ++j) odd[j] = metal::fma(float4(o), mp[j], odd[j]);
        }
    }
    // Shift the moments by this thread's q: T_a = sum_{a' < a} C(a, a') q^(a - a') (odd or even moment of a').
    float qxp[7], qyp[7];
    qxp[0] = 1.0f;
    qyp[0] = 1.0f;
    for (int i = 1; i < 7; ++i) {
        qxp[i] = qxp[i - 1] * qx;
        qyp[i] = qyp[i - 1] * qy;
    }
    float w2 = active ? s.w : 0.0f;
    float out[28];
    out[0] = w2 * 0.5f * depthSum;
    out[1] = w2 * (-3.141592653589793f) * depth * qx * tiltSum;
    out[2] = w2 * (-3.141592653589793f) * depth * qy * tiltSum;
    for (int d = 2; d <= 6; ++d) {
        for (int bb = 0; bb <= d; ++bb) {
            int a = d - bb;
            float t = 0.0f;
            for (int a2 = 0; a2 <= a; ++a2) {
                for (int b2 = 0; b2 <= bb; ++b2) {
                    int order = (a - a2) + (bb - b2);
                    if (order == 0) continue;
                    int index = (a2 + b2) * (a2 + b2 + 1) / 2 + b2;
                    float moment = (order & 1) ? odd[index / 4][index % 4] : even[index / 4][index % 4];
                    float c = FIFTH_BINOMIAL[a][a2] * FIFTH_BINOMIAL[bb][b2];
                    t = metal::fma(c * qxp[a - a2] * qyp[bb - b2], moment, t);
                }
            }
            out[d * (d + 1) / 2 + bb] = w2 * t;
        }
    }
    for (int j = 0; j < 28; ++j) {
        float total = simd_sum(out[j]);
        if (simd_lane == 0u) sums[simd_group * 28u + j] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 28u) {
        float total = 0.0f;
        for (uint i = 0; i < (W + 31u) / 32u; ++i) total += sums[i * 28u + lane];
        partial[(((size_t)b * p.spans + span) * p.tiles + group.x) * 28u + lane] = total;
    }
}
