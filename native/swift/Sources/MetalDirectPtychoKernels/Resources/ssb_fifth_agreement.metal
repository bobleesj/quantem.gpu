#include <metal_stdlib>
using namespace metal;
// The fifth-order agreement objective (ssb-fifth-agreement since v0.3): per band frequency q and candidate,
// N = sum_k G conj(gamma), D = sum_k |gamma|^2 over the BF pixels k in chunks of 256, written as the chunk partials
// that fifth_power and fifth_score (ssb_fifth_order.metal) reduce. kStride > 1 restricts the sums to every
// kStride-th BF pixel (the coarse steering objective of the exact-gradient search); chunks then count used pixels.
// Each per-pair term is the one of v0.1's per-term kernel fifth_agreement (deleted 2026-10-25), reorganized:
//
// 1. A threadgroup owns up to 128 consecutive band frequencies spanning at most two band rows. For one BF pixel and
//    one band row, chi(k -/+ q) is a degree <= 6 polynomial in the column of q whose coefficients the threadgroup
//    collapses once from the bivariate chi polynomial (SSBRedrawPolynomial, double precision on the host from the
//    same packed float32 terms) and shares, so each sideband costs one Horner evaluation instead of the 14-term
//    harmonic sum. chi(k) is the fifth_probe value. This reassociation is the only change of rounding.
// 2. The soft aperture of joint_geometry_xy is 1 inside and 0 outside conservatively rounded radii exactly, so the
//    full expression runs only for SIMD groups with a lane in the one-pixel edge ring. 1/r and the edge quotient
//    use the correctly rounded reciprocal with a Markstein correction, which reproduces IEEE division.
// 3. Candidates sorted by (depth, tilt) on the host share the depth-sinc weights of their predecessor.
// 4. fast:: sin and sincos, which equal precise:: here (fast math off) apart from 1-ulp differences in about one
//    argument in ten million, and the sinc quotient through the Markstein-corrected reciprocal.
// Uniform branches are written as one-trip ballot loops so the compiler cannot if-convert them.

constant uint FIFTH_B [[function_constant(40)]];  // candidates per thread (1 ... 4)
constexpr constant uint FIFTH_S = 16u;            // BF pixels per shared batch
#define FIFTH_UNIFORM_IF(cond) \
    for (uint64_t fifth_once = uint64_t(metal::simd_ballot(cond)); fifth_once != 0ull; fifth_once = 0ull) \
        if (cond)

struct FifthRowsGrid {
    uint count, bandCount, chunks, batch, first;  // first: candidate offset of this dispatch within the batch
    uint kStride;                                 // BF pixel k_j = k[j * kStride]; count is the full BF count
    float wl, semiangle, angularRow, angularColumn, factor;
    float inner2, outer2;  // squared radii where the soft aperture is exactly 1 / exactly 0
};

inline float fifth_quotient(float a, float b) {
    float y = metal::fast::divide(1.0f, b);
    float q0 = a * y;
    return metal::fma(metal::fma(-b, q0, a), y, q0);
}
inline float fifth_sinc(float x) {
    return (metal::abs(x) < 1e-6f) ? 1.0f : fifth_quotient(metal::fast::sin(x), x);
}
// The clamped edge of joint_geometry_xy given r = sqrt(r2) and alpha = r * wl.
inline float fifth_edge(float dx, float dy, float r, float alpha, float semiangle, float ang_y, float ang_x) {
    float inv_r = (r > 1e-15f) ? metal::fast::divide(1.0f, r) : 0.0f;
    float denom = metal::sqrt(dx * ang_y * dx * ang_y + dy * ang_x * dy * ang_x) * inv_r;
    float edge = (denom > 1e-15f) ? (fifth_quotient(semiangle - alpha, denom) + 0.5f) : 1.0f;
    return metal::clamp(edge, 0.0f, 1.0f);
}
inline float fifth_horner(threadgroup const float4* c, float y) {
    float4 a = c[0], b = c[1];
    float value = metal::fma(metal::fma(b.z, y, b.y), y, b.x);
    return metal::fma(metal::fma(metal::fma(metal::fma(value, y, a.w), y, a.z), y, a.y), y, a.x);
}

// tiles: (first band index, count, lanes in the first row, rows). trial: 31 floats per candidate (depth, tilt row,
// tilt column, 28 packed). poly: 49 floats per candidate. chi: chi(k) per (pixel, candidate) from fifth_probe.
kernel void fifth_agreement_rows(device const float2* g [[buffer(0)]], device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]], device float4* partial [[buffer(3)]],
    constant FifthRowsGrid& p [[buffer(4)]], device const float* trial [[buffer(5)]],
    device const float4* geometry [[buffer(6)]], device const float* poly [[buffer(7)]],
    device const uint4* tiles [[buffer(8)]], device const float* chi [[buffer(9)]],
    threadgroup float4* coeff [[threadgroup(0)]], threadgroup float* polys [[threadgroup(1)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint3 size [[threads_per_threadgroup]]) {
    const uint B = FIFTH_B, S = FIFTH_S, W = size.x;
    uint4 tile = tiles[group.x];
    uint chunk = group.y, base = p.first + group.z * B;
    bool valid = lane < tile.y;
    uint qi = tile.x + metal::min(lane, tile.y - 1u);
    float qx = q[qi].x, qy = q[qi].y;
    uint slot = lane >= tile.z ? 1u : 0u, rows = tile.w;
    float qrow0 = q[tile.x].x, qrow1 = q[tile.x + metal::min(tile.z, tile.y - 1u)].x;
    for (uint i = lane; i < B * 49u; i += W) polys[i] = poly[base * 49u + i];
    float nr[4], ni[4], dd[4], shift[4], depth[4];
    bool same[4];
    for (uint b = 0; b < B; ++b) {
        nr[b] = 0.0f; ni[b] = 0.0f; dd[b] = 0.0f;
        device const float* row = trial + (base + b) * 31u;
        shift[b] = 6.283185307179586f * (qx * row[1] + qy * row[2]);
        depth[b] = row[0];
        same[b] = false;
        if (b > 0) {
            device const float* previous = row - 31;
            same[b] = row[0] == previous[0] && row[1] == previous[1] && row[2] == previous[2];
        }
    }
    uint used = (p.count + p.kStride - 1u) / p.kStride;
    uint k0 = chunk * 256u, k1 = metal::min(used, k0 + 256u);
    for (uint start = k0; start < k1; start += S) {
        uint n = metal::min(S, k1 - start);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Collapse chi(x, y) = sum_ij c_ij x^i y^j at x = k_row -/+ q_row for every pixel, candidate and row.
        for (uint index = lane; index < rows * n * B * 2u; index += W) {
            uint row_slot = index / (n * B * 2u), rest = index % (n * B * 2u);
            uint pix = rest / (2u * B), r = rest % (2u * B), b = r >> 1, side = r & 1u;
            float qrow = row_slot == 0u ? qrow0 : qrow1;
            float x = k[(start + pix) * p.kStride].x + (side == 0u ? -qrow : qrow);
            threadgroup const float* c7 = polys + b * 49u;
            float c[7];
            for (uint j = 0; j < 7; ++j) {
                float value = 0.0f;
                for (int i = int(6 - j); i >= 0; --i) value = metal::fma(value, x, c7[i * 7 + j]);
                c[j] = value;
            }
            threadgroup float4* out = coeff + (((row_slot * S + pix) * B + b) * 2u + side) * 2u;
            out[0] = float4(c[0], c[1], c[2], c[3]);
            out[1] = float4(c[4], c[5], c[6], 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!valid) continue;
        for (uint pix = 0; pix < n; ++pix) {
            uint kk = (start + pix) * p.kStride;
            float kxv = k[kk].x, kyv = k[kk].y;
            float4 gk = geometry[kk];
            float mx = qx - kxv, my = qy - kyv, px = qx + kxv, py = qy + kyv;
            float rm = mx * mx + my * my, rp = px * px + py * py;
            float sm = metal::sqrt(rm), sq = metal::sqrt(rp);
            float am = sm * p.wl, ap = sq * p.wl;
            bool ring_m = rm > p.inner2 && rm < p.outer2, ring_p = rp > p.inner2 && rp < p.outer2;
            float em = rm <= p.inner2 ? 1.0f : 0.0f, ep = rp <= p.inner2 ? 1.0f : 0.0f;
            FIFTH_UNIFORM_IF(ring_m) em = fifth_edge(mx, my, sm, am, p.semiangle, p.angularRow, p.angularColumn);
            FIFTH_UNIFORM_IF(ring_p) ep = fifth_edge(px, py, sq, ap, p.semiangle, p.angularRow, p.angularColumn);
            float a1 = em * gk.w, a2 = ep * gk.w;
            bool open_m = a1 != 0.0f, open_p = a2 != 0.0f;
            if (!open_m && !open_p) continue;
            // alpha^2 rounded on its own, and each depth-sinc argument as one expression, as in joint_geometry_xy and
            // v0.1's fifth_agreement (fp contraction stays within an expression).
            float alpha2m = am * am;
            float alpha2p = ap * ap;
            float2 value = g[(size_t)kk * p.bandCount + qi];
            float gre = value.x, gim = value.y;
            threadgroup const float4* own = coeff + (slot * S + pix) * B * 4u;
            float chi_k[4], b1[4], c1[4], s1[4], b2[4], cp[4], sp[4];
            for (uint b = 0; b < B; ++b) {
                chi_k[b] = chi[(size_t)kk * p.batch + base + b];
                b1[b] = 0.0f; c1[b] = 0.0f; s1[b] = 0.0f; b2[b] = 0.0f; cp[b] = 0.0f; sp[b] = 0.0f;
            }
            FIFTH_UNIFORM_IF(open_m) {
                for (uint b = 0; b < B; ++b) {
                    s1[b] = metal::fast::sincos(fifth_horner(own + b * 4u, kyv - qy) - chi_k[b], c1[b]);
                    if (same[b]) b1[b] = b1[b - 1];
                    else {
                        b1[b] = a1;
                        FIFTH_UNIFORM_IF(depth[b] != 0.0f) b1[b] = fifth_sinc(0.5f * depth[b] * (-p.factor * (alpha2m - gk.x) - shift[b])) * a1;
                    }
                }
            }
            FIFTH_UNIFORM_IF(open_p) {
                for (uint b = 0; b < B; ++b) {
                    sp[b] = metal::fast::sincos(fifth_horner(own + b * 4u + 2u, qy + kyv) - chi_k[b], cp[b]);
                    if (same[b]) b2[b] = b2[b - 1];
                    else {
                        b2[b] = a2;
                        FIFTH_UNIFORM_IF(depth[b] != 0.0f) b2[b] = fifth_sinc(0.5f * depth[b] * (p.factor * (alpha2p - gk.x) - shift[b])) * a2;
                    }
                }
            }
            for (uint b = 0; b < B; ++b) {
                float gr = b1[b] * c1[b] - b2[b] * cp[b];
                float gi = -b1[b] * s1[b] - b2[b] * sp[b];
                nr[b] += gre * gr + gim * gi;
                ni[b] += gim * gr - gre * gi;
                dd[b] += gr * gr + gi * gi;
            }
        }
    }
    if (!valid) return;
    for (uint b = 0; b < B; ++b)
        partial[((size_t)chunk * p.batch + base + b) * p.bandCount + qi] = float4(nr[b], ni[b], dd[b], 0.0f);
}
