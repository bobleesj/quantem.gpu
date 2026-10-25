// The low-order agreement with its exact gradient and Hessian (ssb-low-order-derivatives v0.2), in the six fit
// coordinates z = (C10, a = C12 cos 2 phi12, b = C12 sin 2 phi12 [A], tilt row, tilt column [mrad], depth spread
// [A]), the arithmetic of `ssb_low_order.metal` (v0.1) with
// 1. the soft aperture exactly 1 inside and 0 outside conservatively rounded radii, its full expression only for
//    SIMD groups with a lane in the one-pixel edge ring (`joint_agreement_edge`, bitwise the v0.1 aperture);
//    fast:: sincos (equal to precise:: with fast math off apart from rare 1-ulp differences near +-pi), the
//    correctly rounded reciprocal (bitwise IEEE 1/x) and Markstein quotients; phasor anchors as a real branch;
// 2. the value in its own stage: `low_order_value_sums` and `low_order_value_reduce` form A, B, Sd, Ss and
//    w |N|^2 / D per band frequency, so the Newton asks for derivatives only at the centers whose step it uses;
// 3. 29 k-sums in place of 51, from two exact reductions of the derivative structure:
//    - the weights depend on the tilt only through tau = q.theta, so d/d(tilt row) = qx d/dtau and
//      d/d(tilt column) = qy d/dtau (x = t (-pi 1e-3 tau + f q.k -+ P), tau in mrad / A);
//    - the phase slopes v_j(k) of (C10, a, b) are linear in k: v_j = alpha_j . k with alpha_0 = 2f (qx, qy),
//      alpha_1 = 2f (qx, -qy), alpha_2 = 2f (qy, qx), so every j-sum is two k-moments and every (j, l)-sum three.
//    `low_order_gradient_combine` expands the 29 sums to v0.1's 51 per band frequency and forms its outputs.
//
// Pass 2 fields (n_k = h (c d + i s sm), m_k = h (-s d + i c sm), rho1 + i rho2 = conj(N) h; m over (tau, t) with
// d_tau = sum over the sidebands of +-amplitude sinc'(x), the factor -pi 1e-3 t applied in the combine):
//   0-3 sum kx n_k, sum ky n_k (complex); 4-7 Q_tau, Q_t = sum h (c d_m + i s s_m) (complex);
//   8-9 R_m = sum c^2 d d_m + s^2 sm s_m; 10-11 T_m = sum sm s_m - d d_m;
//   12-14 U_mn = sum c^2 (d_m d_n + d d_mn) + s^2 (s_m s_n + sm s_mn), (tau tau) (tau t) (t t), where d_tau tau and
//         d_tau t omit the factors of the combine (d_tau t's -pi 1e-3 d_tau term is added there);
//   15-16 sum k Im(conj N m_k); 17-19 sum kx kx, kx ky, ky ky Re(conj N n_k);
//   20-21 sum Re(conj N m_k,m); 22-25 sum kx, ky Im(conj N n_k,m) for m = tau then t; 26-28 sum Re(conj N n_k,mn).

constant uint LOW_V2_FIELDS = 29;
constant uint LOW_V2_OUTPUTS = 28;
constant uint LOW_V2_ANCHOR = 8;

struct LowOrderV2Setup {
    float qx, qy, ux, uy, psi, c_q, p_q, bx, by, cos_p, sin_p;
    float2 su_step, sy_step;
};

// Trial rows: C10, a, b, depth spread [A], tilt row, column [rad] (as v0.1).
inline LowOrderV2Setup low_order_v2_setup(float qx, float qy, device const float* t, constant JointGrid& p,
                                          float2 step_k) {
    LowOrderV2Setup s;
    float f = p.factor * p.wl * p.wl;
    float c10 = t[0], a = t[1], b = t[2], depth = t[3], thr = t[4], thc = t[5];
    float m00 = f * (c10 + a), m01 = f * b, m11 = f * (c10 - a);
    s.qx = qx; s.qy = qy;
    s.ux = 2.0f * (m00 * qx + m01 * qy);
    s.uy = 2.0f * (m01 * qx + m11 * qy);
    s.psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
    s.c_q = -3.141592653589793f * depth * (qx * thr + qy * thc);
    s.p_q = 0.5f * depth * f * (qx * qx + qy * qy);
    s.sin_p = metal::fast::sincos(s.p_q, s.cos_p);
    s.bx = depth * f * qx;
    s.by = depth * f * qy;
    float c, sn = metal::fast::sincos(s.ux * step_k[0] + s.uy * step_k[1], c);
    s.su_step = float2(c, sn);
    sn = metal::fast::sincos(s.bx * step_k[0] + s.by * step_k[1], c);
    s.sy_step = float2(c, sn);
    return s;
}

// sinc(x) = sin x / x and its first two derivatives, by their series below |x| = 1 (v0.1's expressions; the
// reciprocal is the correctly rounded one).
inline float3 low_order_v2_sinc(float x, float sx, float cx) {
    float x2 = x * x;
    if (metal::abs(x) < 1.0f) {
        float s0 = 1.0f - x2 * (1.0f / 6.0f) * (1.0f - x2 * (1.0f / 20.0f) * (1.0f - x2 * (1.0f / 42.0f)
                 * (1.0f - x2 * (1.0f / 72.0f) * (1.0f - x2 * (1.0f / 110.0f)))));
        float s1 = x * (-1.0f / 3.0f + x2 * (1.0f / 30.0f + x2 * (-1.0f / 840.0f + x2 * (1.0f / 45360.0f
                 + x2 * (-1.0f / 3991680.0f)))));
        float s2 = -1.0f / 3.0f + x2 * (1.0f / 10.0f + x2 * (-1.0f / 168.0f + x2 * (1.0f / 6480.0f
                 + x2 * (-1.0f / 443520.0f))));
        return float3(s0, s1, s2);
    }
    float inv = metal::fast::divide(1.0f, x);
    float s0 = sx * inv;
    float s1 = (cx - s0) * inv;
    float s2 = -(s0 + 2.0f * s1 * inv);
    return float3(s0, s1, s2);
}

// The soft aperture of both sidebands of pixel k at q (bitwise joint_geometry_xy(.).w).
inline float2 low_order_v2_aperture(float qx, float qy, float kxv, float kyv, constant JointGrid& p,
                                    constant JointAgreementExtra& extra) {
    float mx = qx - kxv, my = qy - kyv, px = qx + kxv, py = qy + kyv;
    float rm = mx * mx + my * my, rp = px * px + py * py;
    bool ring_m = rm > extra.inner2 && rm < extra.outer2, ring_p = rp > extra.inner2 && rp < extra.outer2;
    float em = rm <= extra.inner2 ? 1.0f : 0.0f, ep = rp <= extra.inner2 ? 1.0f : 0.0f;
    JOINT_UNIFORM_IF(ring_m) em = joint_agreement_edge(mx, my, rm, p.wl, p.semiangle, p.angularRow, p.angularColumn);
    JOINT_UNIFORM_IF(ring_p) ep = joint_agreement_edge(px, py, rp, p.wl, p.semiangle, p.angularRow, p.angularColumn);
    return float2(em, ep);
}

// Pass 1: per (band frequency, chunk of 256 BF pixels, candidate) the partial A, B (float4) and Sd, Ss.
kernel void low_order_value_sums(device const float2* g [[buffer(0)]], device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]], device float4* partial_ab [[buffer(3)]],
    device float2* partial_ss [[buffer(4)]], constant JointGrid& p [[buffer(5)]],
    device const float* trial [[buffer(6)]], constant uint& TOTAL [[buffer(7)]],
    device const float4* probe_geometry [[buffer(8)]], device const float* neighbour [[buffer(9)]],
    constant float2& step_k [[buffer(10)]], constant JointAgreementExtra& extra [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, chunk = tid.y, candidate = tid.z;
    if (qi >= p.bandCount || chunk >= p.chunks || candidate >= TOTAL) return;
    LowOrderV2Setup s = low_order_v2_setup(q[qi].x, q[qi].y, trial + candidate * 6, p, step_k);
    bool thick = s.p_q != 0.0f || s.c_q != 0.0f || s.bx != 0.0f || s.by != 0.0f;
    float2 a_sum = 0.0f, b_sum = 0.0f;
    float sd = 0.0f, ss = 0.0f;
    float2 eu = float2(1.0f, 0.0f), ey = float2(1.0f, 0.0f);
    uint k0 = chunk * 256;
    uint k1 = metal::min(p.totalBF, k0 + 256u);
    for (uint kk = k0; kk < k1; ++kk) {
        float kxv = k[kk].x, kyv = k[kk].y;
        bool anchor = kk == k0 || neighbour[kk] == 0.0f || (kk - k0) % LOW_V2_ANCHOR == 0;
        JOINT_UNIFORM_IF(anchor) {
            float c, sn = metal::fast::sincos(s.ux * kxv + s.uy * kyv, c);
            eu = float2(c, sn);
            sn = metal::fast::sincos(s.c_q + (s.bx * kxv + s.by * kyv), c);
            ey = float2(c, sn);
        }
        if (!anchor) {
            eu = float2(eu.x * s.su_step.x - eu.y * s.su_step.y, eu.x * s.su_step.y + eu.y * s.su_step.x);
            ey = float2(ey.x * s.sy_step.x - ey.y * s.sy_step.y, ey.x * s.sy_step.y + ey.y * s.sy_step.x);
        }
        float4 gk = probe_geometry[kk];
        float2 e = low_order_v2_aperture(s.qx, s.qy, kxv, kyv, p, extra);
        float a1 = e.x * gk.w, a2 = e.y * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;
        float2 value = g[(size_t)kk * p.bandCount + qi];
        float2 h = float2(value.x * eu.x + value.y * eu.y, value.y * eu.x - value.x * eu.y);
        float wa = a1, wb = a2;
        if (thick) {
            float y = s.c_q + (s.bx * kxv + s.by * kyv);
            if (a1 != 0.0f) wa = joint_agreement_sinc(y - s.p_q, ey.y * s.cos_p - ey.x * s.sin_p) * a1;
            if (a2 != 0.0f) wb = joint_agreement_sinc(y + s.p_q, ey.y * s.cos_p + ey.x * s.sin_p) * a2;
        }
        float d = wa - wb, sm = wa + wb;
        a_sum += d * h;
        b_sum += sm * h;
        sd += d * d;
        ss += sm * sm;
    }
    size_t index = ((size_t)candidate * p.chunks + chunk) * p.bandCount + qi;
    partial_ab[index] = float4(a_sum, b_sum);
    partial_ss[index] = float2(sd, ss);
}

// Pass 1 reduction in chunk order, A, B, Sd, Ss per (candidate, band frequency), and the band frequency's term of
// the value, w |N|^2 / D (v0.1's combine expression).
kernel void low_order_value_reduce(device const float4* partial_ab [[buffer(0)]],
    device const float2* partial_ss [[buffer(1)]], device float4* ab [[buffer(2)]],
    device float2* ss [[buffer(3)]], device float* value [[buffer(4)]], device const float2* q [[buffer(5)]],
    device const float* weights [[buffer(6)]], constant JointGrid& p [[buffer(7)]],
    device const float* trial [[buffer(8)]], constant uint& TOTAL [[buffer(9)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, candidate = tid.y;
    if (qi >= p.bandCount || candidate >= TOTAL) return;
    float4 x = 0.0f;
    float2 y = 0.0f;
    for (uint c = 0; c < p.chunks; ++c) {
        size_t index = ((size_t)candidate * p.chunks + c) * p.bandCount + qi;
        x += partial_ab[index];
        y += partial_ss[index];
    }
    size_t out = (size_t)candidate * p.bandCount + qi;
    ab[out] = x;
    ss[out] = y;
    device const float* t = trial + candidate * 6;
    float f = p.factor * p.wl * p.wl;
    float qx = q[qi].x, qy = q[qi].y;
    float m00 = f * (t[0] + t[1]), m01 = f * t[2], m11 = f * (t[0] - t[1]);
    float psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
    float cpsi, spsi = metal::fast::sincos(psi, cpsi);
    float D = cpsi * cpsi * y.x + spsi * spsi * y.y;
    if (!(D > 0.0f)) {
        value[out] = 0.0f;
        return;
    }
    float nx = cpsi * x.x - spsi * x.w, ny = cpsi * x.y + spsi * x.z;
    float E = nx * nx + ny * ny;
    float inv = 1.0f / D;
    value[out] = weights[qi] * E * inv;
}

// Pass 2: per (band frequency, chunk of `chunk_size` BF pixels, selected candidate) the 29 k-sums of the
// derivatives; `selected` maps the dispatch's slots to candidates of the value pass.
kernel void low_order_gradient_sums(device const float2* g [[buffer(0)]],
    device const float2* k [[buffer(1)]], device const float2* q [[buffer(2)]],
    device float* partial [[buffer(3)]], constant JointGrid& p [[buffer(4)]],
    device const float* trial [[buffer(5)]], constant uint& TOTAL [[buffer(6)]],
    device const float4* probe_geometry [[buffer(7)]], device const float* neighbour [[buffer(8)]],
    constant float2& step_k [[buffer(9)]], device const float4* ab [[buffer(10)]],
    constant uint& chunk_size [[buffer(11)]], constant uint& chunks [[buffer(12)]],
    constant JointAgreementExtra& extra [[buffer(13)]], device const uint* selected [[buffer(14)]],
    uint3 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, chunk = tid.y, slot = tid.z;
    if (qi >= p.bandCount || chunk >= chunks || slot >= TOTAL) return;
    uint candidate = selected[slot];
    float f = p.factor * p.wl * p.wl;
    device const float* t = trial + candidate * 6;
    LowOrderV2Setup s = low_order_v2_setup(q[qi].x, q[qi].y, t, p, step_k);
    float cpsi, spsi = metal::fast::sincos(s.psi, cpsi);
    float c2 = cpsi * cpsi, s2 = spsi * spsi;
    float4 sums = ab[(size_t)candidate * p.bandCount + qi];
    // N = c A + i s B.
    float nx = cpsi * sums.x - spsi * sums.w, ny = cpsi * sums.y + spsi * sums.z;
    float qx = s.qx, qy = s.qy;
    float p_rate = 0.5f * f * (qx * qx + qy * qy);
    float tilt_y = -3.141592653589793f * (qx * t[4] + qy * t[5]);
    float acc[29];
    for (uint i = 0; i < LOW_V2_FIELDS; ++i) acc[i] = 0.0f;
    float2 eu = float2(1.0f, 0.0f), ey = float2(1.0f, 0.0f);
    uint k0 = chunk * chunk_size;
    uint k1 = metal::min(p.totalBF, k0 + chunk_size);
    for (uint kk = k0; kk < k1; ++kk) {
        float kxv = k[kk].x, kyv = k[kk].y;
        bool anchor = kk == k0 || neighbour[kk] == 0.0f || (kk - k0) % LOW_V2_ANCHOR == 0;
        JOINT_UNIFORM_IF(anchor) {
            float c, sn = metal::fast::sincos(s.ux * kxv + s.uy * kyv, c);
            eu = float2(c, sn);
            sn = metal::fast::sincos(s.c_q + (s.bx * kxv + s.by * kyv), c);
            ey = float2(c, sn);
        }
        if (!anchor) {
            eu = float2(eu.x * s.su_step.x - eu.y * s.su_step.y, eu.x * s.su_step.y + eu.y * s.su_step.x);
            ey = float2(ey.x * s.sy_step.x - ey.y * s.sy_step.y, ey.x * s.sy_step.y + ey.y * s.sy_step.x);
        }
        float4 gk = probe_geometry[kk];
        float2 e = low_order_v2_aperture(qx, qy, kxv, kyv, p, extra);
        float a1 = e.x * gk.w, a2 = e.y * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;
        float2 value = g[(size_t)kk * p.bandCount + qi];
        float hx = value.x * eu.x + value.y * eu.y, hy = value.y * eu.x - value.x * eu.y;
        // Sideband weights and their x-derivatives; x = y -+ P.
        float y = s.c_q + (s.bx * kxv + s.by * kyv);
        float sy = ey.y, cy = ey.x;
        float3 za = low_order_v2_sinc(y - s.p_q, sy * s.cos_p - cy * s.sin_p, cy * s.cos_p + sy * s.sin_p);
        float3 zb = low_order_v2_sinc(y + s.p_q, sy * s.cos_p + cy * s.sin_p, cy * s.cos_p - sy * s.sin_p);
        float qk = qx * kxv + qy * kyv;
        float y_rate = tilt_y + f * qk;
        float ra = y_rate - p_rate, rb = y_rate + p_rate;
        float wa = a1 * za.x, wb = a2 * zb.x;
        float w1a = a1 * za.y, w1b = a2 * zb.y;
        float w2a = a1 * za.z, w2b = a2 * zb.z;
        float d = wa - wb, sm = wa + wb;
        // tau (factor applied in the combine) and depth.
        float d_u = w1a - w1b, s_u = w1a + w1b;
        float w1ar = w1a * ra, w1br = w1b * rb;
        float d_t = w1ar - w1br, s_t = w1ar + w1br;
        float d_uu = w2a - w2b, s_uu = w2a + w2b;
        float w2ar = w2a * ra, w2br = w2b * rb;
        float d_ut = w2ar - w2br, s_ut = w2ar + w2br;
        float w2arr = w2ar * ra, w2brr = w2br * rb;
        float d_tt = w2arr - w2brr, s_tt = w2arr + w2brr;
        // Projections on conj(N): rho1 = Re(conj N h), rho2 = Im(conj N h).
        float rho1 = nx * hx + ny * hy, rho2 = nx * hy - ny * hx;
        float cd = cpsi * d, ssm = spsi * sm;
        float n_re = cd * hx - ssm * hy, n_im = cd * hy + ssm * hx;
        acc[0] += kxv * n_re;
        acc[1] += kxv * n_im;
        acc[2] += kyv * n_re;
        acc[3] += kyv * n_im;
        float re_n = cd * rho1 - ssm * rho2;
        float im_m = cpsi * sm * rho1 - spsi * d * rho2;
        acc[15] += kxv * im_m;
        acc[16] += kyv * im_m;
        float kre = kxv * re_n;
        acc[17] += kxv * kre;
        acc[18] += kyv * kre;
        acc[19] += kyv * (kyv * re_n);
        float cdu = cpsi * d_u, ssu = spsi * s_u;
        acc[4] += cdu * hx - ssu * hy;
        acc[5] += cdu * hy + ssu * hx;
        acc[8] += c2 * d * d_u + s2 * sm * s_u;
        acc[10] += sm * s_u - d * d_u;
        acc[20] += -spsi * d_u * rho1 - cpsi * s_u * rho2;
        float im_u = cdu * rho2 + ssu * rho1;
        acc[22] += kxv * im_u;
        acc[23] += kyv * im_u;
        float cdt = cpsi * d_t, sst = spsi * s_t;
        acc[6] += cdt * hx - sst * hy;
        acc[7] += cdt * hy + sst * hx;
        acc[9] += c2 * d * d_t + s2 * sm * s_t;
        acc[11] += sm * s_t - d * d_t;
        acc[21] += -spsi * d_t * rho1 - cpsi * s_t * rho2;
        float im_t = cdt * rho2 + sst * rho1;
        acc[24] += kxv * im_t;
        acc[25] += kyv * im_t;
        acc[12] += c2 * (d_u * d_u + d * d_uu) + s2 * (s_u * s_u + sm * s_uu);
        acc[13] += c2 * (d_u * d_t + d * d_ut) + s2 * (s_u * s_t + sm * s_ut);
        acc[14] += c2 * (d_t * d_t + d * d_tt) + s2 * (s_t * s_t + sm * s_tt);
        acc[26] += cpsi * d_uu * rho1 - spsi * s_uu * rho2;
        acc[27] += cpsi * d_ut * rho1 - spsi * s_ut * rho2;
        acc[28] += cpsi * d_tt * rho1 - spsi * s_tt * rho2;
    }
    size_t base = ((size_t)slot * chunks + chunk) * LOW_V2_FIELDS;
    for (uint i = 0; i < LOW_V2_FIELDS; ++i) partial[(base + i) * p.bandCount + qi] = acc[i];
}

// Reduce pass 2 in chunk order, expand the 29 sums to v0.1's 51 and form w (dS_q/dz, d2S_q/dz2) for each band
// frequency: 27 floats, the six first derivatives and the upper triangle of the Hessian row by row (the value is
// `low_order_value_reduce`'s).
kernel void low_order_gradient_combine(device const float* partial [[buffer(0)]],
    device const float4* ab [[buffer(1)]], device const float2* ss [[buffer(2)]],
    device const float2* q [[buffer(3)]], device const float* weights [[buffer(4)]],
    device float* output [[buffer(5)]], constant JointGrid& p [[buffer(6)]],
    device const float* trial [[buffer(7)]], constant uint& TOTAL [[buffer(8)]],
    constant uint& chunks [[buffer(9)]], device const uint* selected [[buffer(10)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, slot = tid.y;
    if (qi >= p.bandCount || slot >= TOTAL) return;
    uint candidate = selected[slot];
    size_t out = (size_t)slot * (LOW_V2_OUTPUTS - 1);
    float b[29];
    for (uint i = 0; i < LOW_V2_FIELDS; ++i) b[i] = 0.0f;
    for (uint c = 0; c < chunks; ++c) {
        size_t base = ((size_t)slot * chunks + c) * LOW_V2_FIELDS;
        for (uint i = 0; i < LOW_V2_FIELDS; ++i) b[i] += partial[(base + i) * p.bandCount + qi];
    }
    device const float* t = trial + candidate * 6;
    float f = p.factor * p.wl * p.wl;
    float qx = q[qi].x, qy = q[qi].y;
    float m00 = f * (t[0] + t[1]), m01 = f * t[2], m11 = f * (t[0] - t[1]);
    float psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
    float cpsi, spsi = metal::fast::sincos(psi, cpsi);
    float4 sums = ab[(size_t)candidate * p.bandCount + qi];
    float2 sq = ss[(size_t)candidate * p.bandCount + qi];
    float D = cpsi * cpsi * sq.x + spsi * spsi * sq.y;
    if (!(D > 0.0f)) {
        for (uint i = 0; i < LOW_V2_OUTPUTS - 1; ++i) output[(out + i) * p.bandCount + qi] = 0.0f;
        return;
    }
    float nx = cpsi * sums.x - spsi * sums.w, ny = cpsi * sums.y + spsi * sums.z;
    float mx = -spsi * sums.x - cpsi * sums.w, my = -spsi * sums.y + cpsi * sums.z;
    // v0.1's 51 fields from the 29: tilt row / column = (qx, qy) kappa x the tau sums, kappa = -pi 1e-3 t;
    // j-sums alpha_j . (kx, ky) moments.
    float lambda = -3.141592653589793f * 1e-3f;
    float kappa = lambda * t[3];
    float ar = qx * kappa, ac = qy * kappa;
    float2 alpha[3] = {2.0f * f * float2(qx, qy), 2.0f * f * float2(qx, -qy), 2.0f * f * float2(qy, qx)};
    float a[51];
    for (int j = 0; j < 3; ++j) {
        a[2 * j] = alpha[j].x * b[0] + alpha[j].y * b[2];
        a[2 * j + 1] = alpha[j].x * b[1] + alpha[j].y * b[3];
        a[24 + j] = alpha[j].x * b[15] + alpha[j].y * b[16];
        float xu = alpha[j].x * b[22] + alpha[j].y * b[23];
        a[36 + 3 * j] = ar * xu;
        a[37 + 3 * j] = ac * xu;
        a[38 + 3 * j] = alpha[j].x * b[24] + alpha[j].y * b[25];
    }
    const int jl[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    for (int r = 0; r < 6; ++r) {
        float2 u = alpha[jl[r][0]], v = alpha[jl[r][1]];
        a[27 + r] = u.x * v.x * b[17] + (u.x * v.y + u.y * v.x) * b[18] + u.y * v.y * b[19];
    }
    a[6] = ar * b[4]; a[7] = ar * b[5]; a[8] = ac * b[4]; a[9] = ac * b[5]; a[10] = b[6]; a[11] = b[7];
    a[12] = ar * b[8]; a[13] = ac * b[8]; a[14] = b[9];
    a[15] = ar * b[10]; a[16] = ac * b[10]; a[17] = b[11];
    float u_ut = kappa * b[13] + lambda * b[8];
    a[18] = ar * ar * b[12]; a[19] = ar * ac * b[12]; a[20] = qx * u_ut;
    a[21] = ac * ac * b[12]; a[22] = qy * u_ut; a[23] = b[14];
    a[33] = ar * b[20]; a[34] = ac * b[20]; a[35] = b[21];
    float f_ut = kappa * b[27] + lambda * (nx * b[4] + ny * b[5]);
    a[45] = ar * ar * b[26]; a[46] = ar * ac * b[26]; a[47] = qx * f_ut;
    a[48] = ac * ac * b[26]; a[49] = qy * f_ut; a[50] = b[28];
    // v0.1's combine.
    float psi_j[3] = {f * (qx * qx + qy * qy), f * (qx * qx - qy * qy), 2.0f * f * qx * qy};
    float E = nx * nx + ny * ny;
    float split = sq.y - sq.x;
    float d_psi = 2.0f * spsi * cpsi * split;
    float d_psipsi = 2.0f * (cpsi * cpsi - spsi * spsi) * split;
    float Nre[6], Nim[6], Dd[6];
    for (int j = 0; j < 3; ++j) {
        // N_j = psi_j M - i P_j.
        Nre[j] = psi_j[j] * mx + a[2 * j + 1];
        Nim[j] = psi_j[j] * my - a[2 * j];
        Dd[j] = psi_j[j] * d_psi;
    }
    for (int m = 0; m < 3; ++m) {
        Nre[3 + m] = a[6 + 2 * m];
        Nim[3 + m] = a[7 + 2 * m];
        Dd[3 + m] = 2.0f * a[12 + m];
    }
    float Ed[6];
    for (int i = 0; i < 6; ++i) Ed[i] = 2.0f * (nx * Nre[i] + ny * Nim[i]);
    float w = weights[qi];
    float inv = 1.0f / D;
    for (int i = 0; i < 6; ++i)
        output[(out + i) * p.bandCount + qi] = w * (Ed[i] - E * Dd[i] * inv) * inv;
    const int pairs[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    int field = 6;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            float projected, Dij;
            if (j < 3) {
                int pair = 0;
                for (int r = 0; r < 6; ++r) if (pairs[r][0] == i && pairs[r][1] == j) pair = r;
                projected = -psi_j[i] * psi_j[j] * E + psi_j[i] * a[24 + j] + psi_j[j] * a[24 + i] - a[27 + pair];
                Dij = psi_j[i] * psi_j[j] * d_psipsi;
            } else if (i < 3) {
                int m = j - 3;
                projected = psi_j[i] * a[33 + m] + a[36 + 3 * i + m];
                Dij = 4.0f * psi_j[i] * spsi * cpsi * a[15 + m];
            } else {
                int m = i - 3, n = j - 3, pair = 0;
                for (int r = 0; r < 6; ++r) if (pairs[r][0] == m && pairs[r][1] == n) pair = r;
                projected = a[45 + pair];
                Dij = 2.0f * a[18 + pair];
            }
            float Eij = 2.0f * (Nre[i] * Nre[j] + Nim[i] * Nim[j]) + 2.0f * projected;
            float value = Eij * inv - (Ed[i] * Dd[j] + Ed[j] * Dd[i]) * inv * inv
                        - E * Dij * inv * inv + 2.0f * E * Dd[i] * Dd[j] * inv * inv * inv;
            output[(out + field) * p.bandCount + qi] = w * value;
            ++field;
        }
    }
}
