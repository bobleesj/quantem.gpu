#include <metal_stdlib>
using namespace metal;
// The low-order agreement objective with angle-sum phases (ssb-joint-agreement since v0.2): the arithmetic of
// v0.1's joint_objective_angle_sums (deleted 2026-10-25) per band frequency, chunk of 256 BF pixels and candidate,
// with
// 1. the soft aperture of joint_geometry_xy exactly 1 inside and 0 outside conservatively rounded radii, its full
//    expression only for SIMD groups with a lane in the one-pixel edge ring (the angle sums need no alpha^2, so no
//    square root elsewhere); 1/r and quotients through the correctly rounded reciprocal with a Markstein
//    correction, which reproduces IEEE division, also in the depth sinc s / x;
// 2. fast:: sincos, equal to precise:: here (fast math off) apart from 1-ulp differences in about one argument in
//    ten million;
// 3. phasor anchors as a real branch (a one-trip ballot loop, so the compiler cannot if-convert the sincos into
//    every pixel), and JOINT_B candidates per thread (1, 2 or 4) so a short batch computes no padding candidates.

constant uint JOINT_B [[function_constant(50)]];
#define JOINT_UNIFORM_IF(cond) \
    for (uint64_t joint_once = uint64_t(metal::simd_ballot(cond)); joint_once != 0ull; joint_once = 0ull) \
        if (cond)

struct JointAgreementExtra {
    float inner2, outer2;  // squared radii where the soft aperture is exactly 1 / exactly 0
    uint first;            // first candidate of this dispatch within the group
};

inline float joint_quotient(float a, float b) {
    float y = metal::fast::divide(1.0f, b);
    float q0 = a * y;
    return metal::fma(metal::fma(-b, q0, a), y, q0);
}
inline float joint_agreement_sinc(float x, float s) {
    float x2 = x * x;
    float series = 1.0f - x2 * (1.0f / 6.0f) * (1.0f - x2 * (1.0f / 20.0f) * (1.0f - x2 * (1.0f / 42.0f)
                 * (1.0f - x2 * (1.0f / 72.0f) * (1.0f - x2 * (1.0f / 110.0f)))));
    return metal::abs(x) < 1.0f ? series : joint_quotient(s, x);
}
// The clamped edge of joint_geometry_xy.
inline float joint_agreement_edge(float dx, float dy, float r2, float wl, float semiangle, float ang_y,
                                  float ang_x) {
    float r = metal::sqrt(r2);
    float alpha = r * wl;
    float inv_r = (r > 1e-15f) ? metal::fast::divide(1.0f, r) : 0.0f;
    float denom = metal::sqrt(dx * ang_y * dx * ang_y + dy * ang_x * dy * ang_x) * inv_r;
    float edge = (denom > 1e-15f) ? (joint_quotient(semiangle - alpha, denom) + 0.5f) : 1.0f;
    return metal::clamp(edge, 0.0f, 1.0f);
}

kernel void joint_agreement_angle_sums(device const float2* g [[buffer(0)]], device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]], device float4* partial [[buffer(3)]],
    constant JointGrid& p [[buffer(4)]], device const float* trial [[buffer(5)]],
    constant uint& TOTAL [[buffer(6)]], device const float4* probe_geometry [[buffer(7)]],
    device const float* neighbour [[buffer(8)]], constant float2& step_k [[buffer(9)]],
    constant JointAgreementExtra& extra [[buffer(10)]], uint3 tid [[thread_position_in_grid]]) {
    const uint B = JOINT_B;
    constexpr uint ANCHOR = 8;
    uint qi = tid.x, chunk = tid.y, base = extra.first + tid.z * B;
    if (qi >= p.bandCount || chunk >= p.chunks || base >= TOTAL) return;
    float wl = p.wl, semi = p.semiangle, ang_y = p.angularRow, ang_x = p.angularColumn, factor = p.factor;
    float f = factor * wl * wl;
    float qx = q[qi].x, qy = q[qi].y;
    float ux[4], uy[4], cos_psi[4], sin_psi[4], c_q[4], cos_p[4], sin_p[4], p_q[4], bx[4], by[4];
    float2 n_minus[4], n_plus[4];
    float s_minus[4], s_plus[4];
    bool thick[4];
    for (uint b = 0; b < B; ++b) {
        uint i = base + b;
        float c10 = trial[i * 7], c12 = trial[i * 7 + 1], c2 = trial[i * 7 + 2], s2 = trial[i * 7 + 3];
        float t = trial[i * 7 + 4], thr = trial[i * 7 + 5], thc = trial[i * 7 + 6];
        float m00 = f * (c10 + c12 * c2), m01 = f * (c12 * s2), m11 = f * (c10 - c12 * c2);
        ux[b] = 2.0f * (m00 * qx + m01 * qy);
        uy[b] = 2.0f * (m01 * qx + m11 * qy);
        float psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
        sin_psi[b] = metal::fast::sincos(psi, cos_psi[b]);
        c_q[b] = -3.141592653589793f * t * (qx * thr + qy * thc);
        p_q[b] = 0.5f * t * f * (qx * qx + qy * qy);
        sin_p[b] = metal::fast::sincos(p_q[b], cos_p[b]);
        bx[b] = t * f * qx;
        by[b] = t * f * qy;
        thick[b] = p_q[b] != 0.0f || c_q[b] != 0.0f || bx[b] != 0.0f || by[b] != 0.0f;
        n_minus[b] = n_plus[b] = float2(0.0f);
        s_minus[b] = s_plus[b] = 0.0f;
    }
    uint k0 = chunk * 256;
    uint k1 = metal::min(p.totalBF, k0 + 256u);
    float2 eu[4], ey[4], su_step[4], sy_step[4];
    for (uint b = 0; b < B; ++b) {
        float c, sn = metal::fast::sincos(ux[b] * step_k[0] + uy[b] * step_k[1], c);
        su_step[b] = float2(c, sn);
        sn = metal::fast::sincos(bx[b] * step_k[0] + by[b] * step_k[1], c);
        sy_step[b] = float2(c, sn);
        eu[b] = ey[b] = float2(1.0f, 0.0f);
    }
    for (uint kk = k0; kk < k1; ++kk) {
        float kxv = k[kk].x, kyv = k[kk].y;
        bool anchor = kk == k0 || neighbour[kk] == 0.0f || (kk - k0) % ANCHOR == 0;
        JOINT_UNIFORM_IF(anchor) {
            for (uint b = 0; b < B; ++b) {
                float c, sn = metal::fast::sincos(ux[b] * kxv + uy[b] * kyv, c);
                eu[b] = float2(c, sn);
                sn = metal::fast::sincos(c_q[b] + (bx[b] * kxv + by[b] * kyv), c);
                ey[b] = float2(c, sn);
            }
        }
        if (!anchor) {
            for (uint b = 0; b < B; ++b) {
                eu[b] = float2(eu[b].x * su_step[b].x - eu[b].y * su_step[b].y,
                               eu[b].x * su_step[b].y + eu[b].y * su_step[b].x);
                ey[b] = float2(ey[b].x * sy_step[b].x - ey[b].y * sy_step[b].y,
                               ey[b].x * sy_step[b].y + ey[b].y * sy_step[b].x);
            }
        }
        float4 gk = probe_geometry[kk];
        float mx = qx - kxv, my = qy - kyv, px = qx + kxv, py = qy + kyv;
        float rm = mx * mx + my * my, rp = px * px + py * py;
        bool ring_m = rm > extra.inner2 && rm < extra.outer2, ring_p = rp > extra.inner2 && rp < extra.outer2;
        float em = rm <= extra.inner2 ? 1.0f : 0.0f, ep = rp <= extra.inner2 ? 1.0f : 0.0f;
        JOINT_UNIFORM_IF(ring_m) em = joint_agreement_edge(mx, my, rm, wl, semi, ang_y, ang_x);
        JOINT_UNIFORM_IF(ring_p) ep = joint_agreement_edge(px, py, rp, wl, semi, ang_y, ang_x);
        float a1 = em * gk.w, a2 = ep * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;
        float2 value = g[(size_t)kk * p.bandCount + qi];
        float gre = value.x, gim = value.y;
        for (uint b = 0; b < B; ++b) {
            float cu = eu[b].x, su = eu[b].y;
            float hr = gre * cu + gim * su, hi = gim * cu - gre * su;
            float wa = a1, wb = a2;
            if (thick[b]) {
                float y = c_q[b] + (bx[b] * kxv + by[b] * kyv);
                float cy = ey[b].x, sy = ey[b].y;
                if (a1 != 0.0f) wa = joint_agreement_sinc(y - p_q[b], sy * cos_p[b] - cy * sin_p[b]) * a1;
                if (a2 != 0.0f) wb = joint_agreement_sinc(y + p_q[b], sy * cos_p[b] + cy * sin_p[b]) * a2;
            }
            float d = wa - wb, sm = wa + wb;
            n_minus[b] += d * float2(hr, hi);
            n_plus[b] += sm * float2(hr, hi);
            s_minus[b] += d * d;
            s_plus[b] += sm * sm;
        }
    }
    for (uint b = 0; b < B; ++b) {
        partial[((size_t)chunk * TOTAL + base + b) * p.bandCount + qi] = float4(
            cos_psi[b] * n_minus[b].x - sin_psi[b] * n_plus[b].y,
            cos_psi[b] * n_minus[b].y + sin_psi[b] * n_plus[b].x,
            cos_psi[b] * cos_psi[b] * s_minus[b] + sin_psi[b] * sin_psi[b] * s_plus[b], 0);
    }
}
