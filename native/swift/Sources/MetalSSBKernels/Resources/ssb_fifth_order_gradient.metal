#include <metal_stdlib>
using namespace metal;
// Exact gradient of the fifth-order agreement S = sum_q w_q |N_q|^2 / D_q (`fifth_agreement`), with
// N_q = sum_k G conj(gamma), D_q = sum_k |gamma|^2, gamma = b1 e^{-i phi1} - b2 e^{i phi2},
// phi1 = chi(k-q) - chi(k), phi2 = chi(k+q) - chi(k). With a_q = N_q / D_q and
// R = conj(a_q) (G - a_q gamma), dS/dtheta = sum_{q,k} 2 w_q Re(dgamma/dtheta conj(R)).
// chi is linear in the 28 packed coefficients: dgamma/dc_j = -i (dphi1_j u1 + dphi2_j u2), u1 = b1 e^{-i phi1},
// u2 = b2 e^{i phi2}, so dS/dc_j = sum 2w [Im(u1 conj R) dphi1_j + Im(u2 conj R) dphi2_j]. The depth t and tilt
// enter through b1 = a1 sinc(x1), b2 = a2 sinc(x2): dgamma/db1 = e^{-i phi1}, dgamma/db2 = -e^{i phi2}.
// Per-term arithmetic is float32 like the objective; one simdgroup of band frequencies reduces its sums and the
// host reduces the simdgroup partials in double.

// bandCount counts the used band frequencies q_i = band[i * qStride]; qStride 1 is the full objective.
// chunks counts the 256-pixel objective partials, spans the `span`-pixel gradient partials.
struct FifthGradientGrid {
    uint count, bandCount, chunks, batch;
    float wl, semiangle, angularRow, angularColumn, factor;
    uint span, spans, qGroups, qStride, fullBandCount;
};

// `fifth_agreement` on every qStride-th band frequency (the Hessian of a band subsample steers the search; the
// full objective and gradient define the optimum). Partials are indexed by the subsample.
kernel void fifth_strided_agreement(device const float2* g [[buffer(0)]], device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]], device float4* partial [[buffer(3)]],
    constant FifthGradientGrid& p [[buffer(4)]], device const float* trial [[buffer(5)]],
    device const float4* geometry [[buffer(6)]], device const float* chi [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]) {
    constexpr int B = 4;
    uint qi = tid.x, chunk = tid.y, base = tid.z * B;
    if (qi >= p.bandCount || chunk >= p.chunks || base >= p.batch) return;
    uint qf = qi * p.qStride;
    float qx = q[qf].x, qy = q[qf].y;
    float nr[B], ni[B], dd[B];
    for (int b = 0; b < B; ++b) { nr[b] = 0.0f; ni[b] = 0.0f; dd[b] = 0.0f; }
    uint k1 = metal::min(p.count, chunk * 256u + 256u);
    for (uint kk = chunk * 256u; kk < k1; ++kk) {
        float kxv = k[kk].x, kyv = k[kk].y;
        float4 gk = geometry[kk];
        float4 gm = joint_geometry_xy(qx-kxv, qy-kyv, p.wl, p.semiangle, p.angularRow, p.angularColumn);
        float4 gp = joint_geometry_xy(qx+kxv, qy+kyv, p.wl, p.semiangle, p.angularRow, p.angularColumn);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;
        float2 value = g[(size_t)kk * p.fullBandCount + qf];
        float gre = value.x, gim = value.y;
        for (int b = 0; b < B; ++b) {
            if (base + b >= p.batch) break;
            device const float* row = trial + (base + b) * 31;
            float t = row[0], thr = row[1], thc = row[2];
            float chi_k = chi[(size_t)kk * p.batch + base + b];
            float b1 = 0.0f, b2 = 0.0f, c1 = 0.0f, cp = 0.0f, s1 = 0.0f, sp = 0.0f;
            bool thick = t != 0.0f;
            float shift = 6.283185307179586f * (qx * thr + qy * thc);
            if (a1 != 0.0f) {
                float chi_m = ssb_chi_full(kxv-qx, kyv-qy, p.wl, row + 3);
                s1 = metal::precise::sincos(chi_m-chi_k, c1);
                b1 = thick ? joint_sinc(0.5f*t*(-p.factor*(gm.x-gk.x)-shift))*a1 : a1;
            }
            if (a2 != 0.0f) {
                float chi_p = ssb_chi_full(qx+kxv, qy+kyv, p.wl, row + 3);
                sp = metal::precise::sincos(chi_p-chi_k, cp);
                b2 = thick ? joint_sinc(0.5f*t*(p.factor*(gp.x-gk.x)-shift))*a2 : a2;
            }
            float gr = b1*c1-b2*cp;
            float gi = -b1*s1-b2*sp;
            nr[b] += gre*gr+gim*gi;
            ni[b] += gim*gr-gre*gi;
            dd[b] += gr*gr+gi*gi;
        }
    }
    for (int b = 0; b < B; ++b) {
        if (base + b >= p.batch) break;
        partial[((size_t)chunk*p.batch+base+b)*p.bandCount+qi] = float4(nr[b], ni[b], dd[b], 0.0f);
    }
}

// Per used band frequency and candidate: a_q = N_q / D_q from the objective partials (in the chunk order of
// `fifth_power`), |a_q|^2 and 2 w_q. A frequency without model support (D_q = 0) carries no gradient.
kernel void fifth_gradient_scale(device const float4* partial [[buffer(0)]], device const float* weight [[buffer(1)]],
    device float4* scale [[buffer(2)]], constant FifthGradientGrid& p [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.bandCount || tid.y >= p.batch) return;
    float3 sum = float3(0.0f);
    for (uint chunk = 0; chunk < p.chunks; ++chunk)
        sum += partial[((size_t)chunk*p.batch+tid.y)*p.bandCount+tid.x].xyz;
    float4 s = float4(0.0f);
    if (sum.z > 0.0f) {
        float ar = sum.x / sum.z, ai = sum.y / sum.z;
        s = float4(ar, ai, ar * ar + ai * ai, 2.0f * weight[tid.x * p.qStride]);
    }
    scale[(size_t)tid.y*p.bandCount+tid.x] = s;
}

// sinc(x) as `joint_sinc` and its slope (cos x - sinc x) / x, by its series where the difference cancels.
inline float fifth_sinc_slope(float x, thread float& sinc) {
    float x2 = x * x;
    float c;
    float s = metal::precise::sincos(x, c);
    sinc = metal::abs(x) < 1e-6f ? 1.0f : s / x;
    if (metal::abs(x) < 0.25f) return x * (-1.0f / 3.0f + x2 * (1.0f / 30.0f - x2 * (1.0f / 840.0f)));
    return (c - sinc) / x;
}

// acc[2i], acc[2i+1] += weight * r2^power_i * (Re, Im)(z^m_i), the `ssb_chi_full` basis without 2 pi / wl.
inline void fifth_basis_add(thread float* acc, float row, float col, float wl, float weight) {
    float x = row * wl, y = col * wl;
    float r2 = x * x + y * y;
    float w1 = weight * r2, w2 = w1 * r2, w3 = w2 * r2;
    acc[0] += w1;
    acc[8] += w2;
    acc[20] += w3;
    float re = x, im = y;
    acc[4] += w1 * re; acc[5] += w1 * im;
    acc[14] += w2 * re; acc[15] += w2 * im;
    float t = re * x - im * y; im = im * x + re * y; re = t;
    acc[2] += weight * re; acc[3] += weight * im;
    acc[10] += w1 * re; acc[11] += w1 * im;
    acc[22] += w2 * re; acc[23] += w2 * im;
    t = re * x - im * y; im = im * x + re * y; re = t;
    acc[6] += weight * re; acc[7] += weight * im;
    acc[16] += w1 * re; acc[17] += w1 * im;
    t = re * x - im * y; im = im * x + re * y; re = t;
    acc[12] += weight * re; acc[13] += weight * im;
    acc[24] += w1 * re; acc[25] += w1 * im;
    t = re * x - im * y; im = im * x + re * y; re = t;
    acc[18] += weight * re; acc[19] += weight * im;
    t = re * x - im * y; im = im * x + re * y; re = t;
    acc[26] += weight * re; acc[27] += weight * im;
}

// One thread per (band frequency, span of BF pixels, candidate) accumulates the 28 coefficient sums, the depth
// and the tilt sums; each simdgroup of 32 frequencies writes 32 floats: [depth, tilt row, tilt column,
// 28 coefficients (the ssb_chi_full basis without 2 pi / wl), 0] at
// ((candidate * chunks + span) * qGroups + q / 32) * 32.
kernel void fifth_gradient(device const float2* g [[buffer(0)]], device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]], device float* partial [[buffer(3)]],
    constant FifthGradientGrid& p [[buffer(4)]], device const float* trial [[buffer(5)]],
    device const float4* geometry [[buffer(6)]], device const float* chi [[buffer(7)]],
    device const float4* scale [[buffer(8)]], uint3 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint qi = tid.x, chunk = tid.y, b = tid.z;
    if (chunk >= p.spans || b >= p.batch) return;
    float acc[28];
    for (int j = 0; j < 28; ++j) acc[j] = 0.0f;
    float depthSum = 0.0f, tiltSum = 0.0f;
    float4 s = qi < p.bandCount ? scale[(size_t)b * p.bandCount + qi] : float4(0.0f);
    float qx = 0.0f, qy = 0.0f;
    uint qf = qi * p.qStride;
    device const float* row = trial + b * 31;
    float t = row[0], thr = row[1], thc = row[2];
    if (s.w != 0.0f) {
        qx = q[qf].x; qy = q[qf].y;
        float ar = s.x, ai = s.y, aa = s.z;
        bool thick = t != 0.0f;
        float shift = 6.283185307179586f * (qx * thr + qy * thc);
        uint k0 = chunk * p.span, k1 = metal::min(p.count, k0 + p.span);
        for (uint kk = k0; kk < k1; ++kk) {
            float kxv = k[kk].x, kyv = k[kk].y;
            float4 gk = geometry[kk];
            float4 gm = joint_geometry_xy(qx-kxv, qy-kyv, p.wl, p.semiangle, p.angularRow, p.angularColumn);
            float4 gp = joint_geometry_xy(qx+kxv, qy+kyv, p.wl, p.semiangle, p.angularRow, p.angularColumn);
            float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
            if (a1 == 0.0f && a2 == 0.0f) continue;
            float2 value = g[(size_t)kk * p.fullBandCount + qf];
            float gre = value.x, gim = value.y;
            float chi_k = chi[(size_t)kk * p.batch + b];
            float b1 = 0.0f, b2 = 0.0f, c1 = 0.0f, cp = 0.0f, s1 = 0.0f, sp = 0.0f;
            float d1 = 0.0f, d2 = 0.0f, h1 = 0.0f, h2 = 0.0f;
            if (a1 != 0.0f) {
                float chi_m = ssb_chi_full(kxv-qx, kyv-qy, p.wl, row + 3);
                s1 = metal::precise::sincos(chi_m-chi_k, c1);
                if (thick) {
                    h1 = -p.factor*(gm.x-gk.x)-shift;
                    float sinc;
                    d1 = fifth_sinc_slope(0.5f*t*h1, sinc)*a1;
                    b1 = sinc*a1;
                } else {
                    b1 = a1;
                }
            }
            if (a2 != 0.0f) {
                float chi_p = ssb_chi_full(qx+kxv, qy+kyv, p.wl, row + 3);
                sp = metal::precise::sincos(chi_p-chi_k, cp);
                if (thick) {
                    h2 = p.factor*(gp.x-gk.x)-shift;
                    float sinc;
                    d2 = fifth_sinc_slope(0.5f*t*h2, sinc)*a2;
                    b2 = sinc*a2;
                } else {
                    b2 = a2;
                }
            }
            float gr = b1*c1-b2*cp;
            float gi = -b1*s1-b2*sp;
            float rr = ar*gre + ai*gim - aa*gr;
            float ri = ar*gim - ai*gre - aa*gi;
            float alpha1 = -b1*(c1*ri + s1*rr);
            float alpha2 = b2*(sp*rr - cp*ri);
            float beta1 = c1*rr - s1*ri;
            float beta2 = -(cp*rr + sp*ri);
            tiltSum += beta1*d1 + beta2*d2;
            depthSum += beta1*d1*h1 + beta2*d2*h2;
            fifth_basis_add(acc, kxv-qx, kyv-qy, p.wl, alpha1);
            fifth_basis_add(acc, kxv+qx, kyv+qy, p.wl, alpha2);
            fifth_basis_add(acc, kxv, kyv, p.wl, -(alpha1+alpha2));
        }
    }
    // d b / d t = 0.5 h d sinc, d b / d theta_row = -pi t q_row d sinc.
    float w2 = s.w;
    float out0 = simd_sum(w2 * 0.5f * depthSum);
    float out1 = simd_sum(w2 * (-3.141592653589793f) * t * qx * tiltSum);
    float out2 = simd_sum(w2 * (-3.141592653589793f) * t * qy * tiltSum);
    float sums[28];
    for (int j = 0; j < 28; ++j) sums[j] = simd_sum(w2 * acc[j]);
    if (lane == 0u) {
        device float* out = partial + (((size_t)b * p.spans + chunk) * p.qGroups + qi / 32u) * 32u;
        out[0] = out0; out[1] = out1; out[2] = out2;
        for (int j = 0; j < 28; ++j) out[3 + j] = sums[j];
        out[31] = 0.0f;
    }
}
