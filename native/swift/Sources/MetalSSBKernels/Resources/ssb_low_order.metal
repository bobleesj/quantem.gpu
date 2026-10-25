// The low-order agreement with its exact gradient and Hessian in the six fit coordinates
// z = (C10, C12 cos 2 phi12, C12 sin 2 phi12 [A], tilt row, tilt column [mrad], depth spread [A]).
//
// For one band frequency q the two-term chi is a quadratic form, so with h_k = G_kq e^{-i u.k},
// d_k = wa - wb, s_k = wa + wb (the depth-averaged sideband weights) the angle-sum objective is
//   N = cos(psi) A + i sin(psi) B,  A = sum_k d h,  B = sum_k s h,
//   D = cos^2(psi) Sd + sin^2(psi) Ss,  Sd = sum_k d^2,  Ss = sum_k s^2,   S_q = w |N|^2 / D.
// psi = f q.M.q and u.k = 2 f q.M.k are linear in (C10, a, b) with slopes psi_j (per q) and v_j(k); the
// weights depend on the depth spread t and the tilt only, through the sinc arguments
//   xa = t (Y - P), xb = t (Y + P),  Y = -pi q.theta + f q.k,  P = f |q|^2 / 2.
// Pass 1 (`low_order_sums`) forms A, B, Sd, Ss. Pass 2 (`low_order_derivative_sums`) knows N per q and
// accumulates every k-sum the first and second derivatives of |N|^2 and D need, the second derivatives of N
// already projected on conj(N) (51 floats). `low_order_derivative_combine` reduces them in chunk order and
// writes w (S, dS/dz, d2S/dz2) per q; the host sums the band in double precision.
// The phases use the angle sums of `joint_objective_angle_sums` (exact at a chunk start, a row break and every
// eighth pixel, advanced by one row step in between).

constant uint LOW_ORDER_FIELDS = 51;
constant uint LOW_ORDER_OUTPUTS = 28;
constant uint LOW_ORDER_ANCHOR = 8;

struct LowOrderSetup {
    float qx, qy, ux, uy, psi, c_q, p_q, bx, by, cos_p, sin_p;
    float2 su_step, sy_step;
};

// Trial rows: C10, a = C12 cos 2 phi12, b = C12 sin 2 phi12 [A], depth spread [A], tilt row, column [rad].
inline LowOrderSetup low_order_setup(float qx, float qy, device const float* t, constant JointGrid& p,
                                     float2 step_k) {
    LowOrderSetup s;
    float f = p.factor * p.wl * p.wl;
    float c10 = t[0], a = t[1], b = t[2], depth = t[3], thr = t[4], thc = t[5];
    float m00 = f * (c10 + a), m01 = f * b, m11 = f * (c10 - a);
    s.qx = qx; s.qy = qy;
    s.ux = 2.0f * (m00 * qx + m01 * qy);
    s.uy = 2.0f * (m01 * qx + m11 * qy);
    s.psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
    s.c_q = -3.141592653589793f * depth * (qx * thr + qy * thc);
    s.p_q = 0.5f * depth * f * (qx * qx + qy * qy);
    s.sin_p = metal::precise::sincos(s.p_q, s.cos_p);
    s.bx = depth * f * qx;
    s.by = depth * f * qy;
    float c, sn = metal::precise::sincos(s.ux * step_k[0] + s.uy * step_k[1], c);
    s.su_step = float2(c, sn);
    sn = metal::precise::sincos(s.bx * step_k[0] + s.by * step_k[1], c);
    s.sy_step = float2(c, sn);
    return s;
}

inline float2 low_order_rotate(float2 e, float2 step) {
    return float2(e.x * step.x - e.y * step.y, e.x * step.y + e.y * step.x);
}

// sinc(x) = sin x / x and its first two derivatives, by their series below |x| = 1.
inline float3 low_order_sinc(float x, float sx, float cx) {
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
    float inv = 1.0f / x;
    float s0 = sx * inv;
    float s1 = (cx - s0) * inv;
    float s2 = -(s0 + 2.0f * s1 * inv);
    return float3(s0, s1, s2);
}

// Pass 1: per (band frequency, chunk of 256 BF pixels, candidate) the partial A, B (float4) and Sd, Ss.
kernel void low_order_sums(device const float2* g [[buffer(0)]], device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]], device float4* partial_ab [[buffer(3)]],
    device float2* partial_ss [[buffer(4)]], constant JointGrid& p [[buffer(5)]],
    device const float* trial [[buffer(6)]], constant uint& TOTAL [[buffer(7)]],
    device const float4* probe_geometry [[buffer(8)]], device const float* neighbour [[buffer(9)]],
    constant float2& step_k [[buffer(10)]], uint3 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, chunk = tid.y, candidate = tid.z;
    if (qi >= p.bandCount || chunk >= p.chunks || candidate >= TOTAL) return;
    float wl = p.wl, semi = p.semiangle, ang_y = p.angularRow, ang_x = p.angularColumn;
    LowOrderSetup s = low_order_setup(q[qi].x, q[qi].y, trial + candidate * 6, p, step_k);
    bool thick = s.p_q != 0.0f || s.c_q != 0.0f || s.bx != 0.0f || s.by != 0.0f;
    float2 a_sum = 0.0f, b_sum = 0.0f;
    float sd = 0.0f, ss = 0.0f;
    float2 eu = float2(1.0f, 0.0f), ey = float2(1.0f, 0.0f);
    uint k0 = chunk * 256;
    uint k1 = metal::min(p.totalBF, k0 + 256u);
    for (uint kk = k0; kk < k1; ++kk) {
        float kxv = k[kk].x, kyv = k[kk].y;
        if (kk == k0 || neighbour[kk] == 0.0f || (kk - k0) % LOW_ORDER_ANCHOR == 0) {
            float c, sn = metal::precise::sincos(s.ux * kxv + s.uy * kyv, c);
            eu = float2(c, sn);
            sn = metal::precise::sincos(s.c_q + (s.bx * kxv + s.by * kyv), c);
            ey = float2(c, sn);
        } else {
            eu = low_order_rotate(eu, s.su_step);
            ey = low_order_rotate(ey, s.sy_step);
        }
        float4 gk = probe_geometry[kk];
        float4 gm = joint_geometry_xy(s.qx - kxv, s.qy - kyv, wl, semi, ang_y, ang_x);
        float4 gp = joint_geometry_xy(s.qx + kxv, s.qy + kyv, wl, semi, ang_y, ang_x);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;
        float2 value = g[(size_t)kk * p.bandCount + qi];
        float2 h = float2(value.x * eu.x + value.y * eu.y, value.y * eu.x - value.x * eu.y);
        float wa = a1, wb = a2;
        if (thick) {
            float y = s.c_q + (s.bx * kxv + s.by * kyv);
            if (a1 != 0.0f) wa = joint_depth_sinc(y - s.p_q, ey.y * s.cos_p - ey.x * s.sin_p) * a1;
            if (a2 != 0.0f) wb = joint_depth_sinc(y + s.p_q, ey.y * s.cos_p + ey.x * s.sin_p) * a2;
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

// Pass 1 reduction in chunk order: A, B, Sd, Ss per (candidate, band frequency).
kernel void low_order_reduce_sums(device const float4* partial_ab [[buffer(0)]],
    device const float2* partial_ss [[buffer(1)]], device float4* ab [[buffer(2)]],
    device float2* ss [[buffer(3)]], constant JointGrid& p [[buffer(4)]],
    constant uint& TOTAL [[buffer(5)]], uint2 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, candidate = tid.y;
    if (qi >= p.bandCount || candidate >= TOTAL) return;
    float4 x = 0.0f;
    float2 y = 0.0f;
    for (uint c = 0; c < p.chunks; ++c) {
        size_t index = ((size_t)candidate * p.chunks + c) * p.bandCount + qi;
        x += partial_ab[index];
        y += partial_ss[index];
    }
    ab[(size_t)candidate * p.bandCount + qi] = x;
    ss[(size_t)candidate * p.bandCount + qi] = y;
}

// Pass 2: per (band frequency, chunk of `chunk_size` BF pixels, candidate) the 51 k-sums of the derivatives.
// Fields, with m, n over (tilt row, tilt column, depth) and j, l over (C10, a, b):
//   0-5 P_j = sum v_j n_k (complex), 6-11 Q_m = sum n_k,m (complex), 12-14 R_m = sum c^2 d d_m + s^2 s s_m,
//   15-17 T_m = sum s s_m - d d_m, 18-23 U_mn = sum c^2 (d_m d_n + d d_mn) + s^2 (s_m s_n + s s_mn),
//   24-26 sum v_j Im(conj N m_k), 27-32 sum v_j v_l Re(conj N n_k), 33-35 sum Re(conj N m_k,m),
//   36-44 sum v_j Im(conj N n_k,m) (j major), 45-50 sum Re(conj N n_k,mn),
// where n_k = h (c d + i s s), m_k = h (-s d + i c s) with c, s = cos psi, sin psi, and n_k,m, n_k,mn use the
// weight derivatives. Pairs (m, n) and (j, l) run (0,0) (0,1) (0,2) (1,1) (1,2) (2,2).
kernel void low_order_derivative_sums(device const float2* g [[buffer(0)]],
    device const float2* k [[buffer(1)]], device const float2* q [[buffer(2)]],
    device float* partial [[buffer(3)]], constant JointGrid& p [[buffer(4)]],
    device const float* trial [[buffer(5)]], constant uint& TOTAL [[buffer(6)]],
    device const float4* probe_geometry [[buffer(7)]], device const float* neighbour [[buffer(8)]],
    constant float2& step_k [[buffer(9)]], device const float4* ab [[buffer(10)]],
    constant uint& chunk_size [[buffer(11)]], constant uint& chunks [[buffer(12)]],
    uint3 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, chunk = tid.y, candidate = tid.z;
    if (qi >= p.bandCount || chunk >= chunks || candidate >= TOTAL) return;
    float wl = p.wl, semi = p.semiangle, ang_y = p.angularRow, ang_x = p.angularColumn;
    float f = p.factor * wl * wl;
    device const float* t = trial + candidate * 6;
    LowOrderSetup s = low_order_setup(q[qi].x, q[qi].y, t, p, step_k);
    float depth = t[3];
    float cpsi, spsi = metal::precise::sincos(s.psi, cpsi);
    float c2 = cpsi * cpsi, s2 = spsi * spsi;
    float4 sums = ab[(size_t)candidate * p.bandCount + qi];
    // N = c A + i s B.
    float nx = cpsi * sums.x - spsi * sums.w, ny = cpsi * sums.y + spsi * sums.z;
    float qx = s.qx, qy = s.qy;
    float p_rate = 0.5f * f * (qx * qx + qy * qy);
    float tilt_y = -3.141592653589793f * (qx * t[4] + qy * t[5]);
    // d x / d tilt (mrad) and d2 x / d depth d tilt (mrad), the same for both sidebands.
    float xr = -3.141592653589793f * qx * 1e-3f, xc = -3.141592653589793f * qy * 1e-3f;
    float acc[51];
    for (uint i = 0; i < LOW_ORDER_FIELDS; ++i) acc[i] = 0.0f;
    float2 eu = float2(1.0f, 0.0f), ey = float2(1.0f, 0.0f);
    uint k0 = chunk * chunk_size;
    uint k1 = metal::min(p.totalBF, k0 + chunk_size);
    for (uint kk = k0; kk < k1; ++kk) {
        float kxv = k[kk].x, kyv = k[kk].y;
        if (kk == k0 || neighbour[kk] == 0.0f || (kk - k0) % LOW_ORDER_ANCHOR == 0) {
            float c, sn = metal::precise::sincos(s.ux * kxv + s.uy * kyv, c);
            eu = float2(c, sn);
            sn = metal::precise::sincos(s.c_q + (s.bx * kxv + s.by * kyv), c);
            ey = float2(c, sn);
        } else {
            eu = low_order_rotate(eu, s.su_step);
            ey = low_order_rotate(ey, s.sy_step);
        }
        float4 gk = probe_geometry[kk];
        float4 gm = joint_geometry_xy(qx - kxv, qy - kyv, wl, semi, ang_y, ang_x);
        float4 gp = joint_geometry_xy(qx + kxv, qy + kyv, wl, semi, ang_y, ang_x);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;
        float2 value = g[(size_t)kk * p.bandCount + qi];
        float hx = value.x * eu.x + value.y * eu.y, hy = value.y * eu.x - value.x * eu.y;
        // Sideband weights and their derivatives in (tilt row, tilt column, depth).
        float qk = qx * kxv + qy * kyv;
        float y_rate = tilt_y + f * qk;
        float wa = 0.0f, wb = 0.0f;
        float wa_m[3] = {0.0f, 0.0f, 0.0f}, wb_m[3] = {0.0f, 0.0f, 0.0f};
        float wa_mn[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float wb_mn[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float y = s.c_q + (s.bx * kxv + s.by * kyv);
        for (int side = 0; side < 2; ++side) {
            float amplitude = side == 0 ? a1 : a2;
            if (amplitude == 0.0f) continue;
            float x = side == 0 ? y - s.p_q : y + s.p_q;
            float sx = side == 0 ? ey.y * s.cos_p - ey.x * s.sin_p : ey.y * s.cos_p + ey.x * s.sin_p;
            float cx = side == 0 ? ey.x * s.cos_p + ey.y * s.sin_p : ey.x * s.cos_p - ey.y * s.sin_p;
            float3 sinc = low_order_sinc(x, sx, cx);
            float rate = side == 0 ? y_rate - p_rate : y_rate + p_rate;
            float dx[3] = {xr * depth, xc * depth, rate};
            float w = sinc.x * amplitude;
            float w_m[3], w_mn[6];
            for (int m = 0; m < 3; ++m) w_m[m] = amplitude * sinc.y * dx[m];
            w_mn[0] = amplitude * sinc.z * dx[0] * dx[0];
            w_mn[1] = amplitude * sinc.z * dx[0] * dx[1];
            w_mn[2] = amplitude * (sinc.z * dx[0] * dx[2] + sinc.y * xr);
            w_mn[3] = amplitude * sinc.z * dx[1] * dx[1];
            w_mn[4] = amplitude * (sinc.z * dx[1] * dx[2] + sinc.y * xc);
            w_mn[5] = amplitude * sinc.z * dx[2] * dx[2];
            if (side == 0) {
                wa = w;
                for (int m = 0; m < 3; ++m) wa_m[m] = w_m[m];
                for (int m = 0; m < 6; ++m) wa_mn[m] = w_mn[m];
            } else {
                wb = w;
                for (int m = 0; m < 3; ++m) wb_m[m] = w_m[m];
                for (int m = 0; m < 6; ++m) wb_mn[m] = w_mn[m];
            }
        }
        float d = wa - wb, sm = wa + wb;
        float d_m[3], s_m[3];
        for (int m = 0; m < 3; ++m) { d_m[m] = wa_m[m] - wb_m[m]; s_m[m] = wa_m[m] + wb_m[m]; }
        // Projections on conj(N): rho1 = Re(conj N h), rho2 = Im(conj N h).
        float rho1 = nx * hx + ny * hy, rho2 = nx * hy - ny * hx;
        float v[3] = {2.0f * f * qk, 2.0f * f * (qx * kxv - qy * kyv), 2.0f * f * (qx * kyv + qy * kxv)};
        // n_k = h (c d + i s sm).
        float n_re = cpsi * d * hx - spsi * sm * hy, n_im = cpsi * d * hy + spsi * sm * hx;
        for (int j = 0; j < 3; ++j) {
            acc[2 * j] += v[j] * n_re;
            acc[2 * j + 1] += v[j] * n_im;
        }
        float re_n = cpsi * d * rho1 - spsi * sm * rho2;
        float im_m = cpsi * sm * rho1 - spsi * d * rho2;
        for (int j = 0; j < 3; ++j) acc[24 + j] += v[j] * im_m;
        acc[27] += v[0] * v[0] * re_n;
        acc[28] += v[0] * v[1] * re_n;
        acc[29] += v[0] * v[2] * re_n;
        acc[30] += v[1] * v[1] * re_n;
        acc[31] += v[1] * v[2] * re_n;
        acc[32] += v[2] * v[2] * re_n;
        for (int m = 0; m < 3; ++m) {
            acc[6 + 2 * m] += cpsi * d_m[m] * hx - spsi * s_m[m] * hy;
            acc[7 + 2 * m] += cpsi * d_m[m] * hy + spsi * s_m[m] * hx;
            acc[12 + m] += c2 * d * d_m[m] + s2 * sm * s_m[m];
            acc[15 + m] += sm * s_m[m] - d * d_m[m];
            acc[33 + m] += -spsi * d_m[m] * rho1 - cpsi * s_m[m] * rho2;
            float im_n = cpsi * d_m[m] * rho2 + spsi * s_m[m] * rho1;
            for (int j = 0; j < 3; ++j) acc[36 + 3 * j + m] += v[j] * im_n;
        }
        int pair = 0;
        for (int m = 0; m < 3; ++m) {
            for (int n = m; n < 3; ++n) {
                float d_mn = wa_mn[pair] - wb_mn[pair], s_mn = wa_mn[pair] + wb_mn[pair];
                acc[18 + pair] += c2 * (d_m[m] * d_m[n] + d * d_mn) + s2 * (s_m[m] * s_m[n] + sm * s_mn);
                acc[45 + pair] += cpsi * d_mn * rho1 - spsi * s_mn * rho2;
                ++pair;
            }
        }
    }
    size_t base = ((size_t)candidate * chunks + chunk) * LOW_ORDER_FIELDS;
    for (uint i = 0; i < LOW_ORDER_FIELDS; ++i) partial[(base + i) * p.bandCount + qi] = acc[i];
}

// Reduce pass 2 in chunk order and form w (S_q, dS_q/dz, d2S_q/dz2) for each band frequency: 28 floats,
// value, the six first derivatives and the upper triangle of the Hessian row by row.
kernel void low_order_derivative_combine(device const float* partial [[buffer(0)]],
    device const float4* ab [[buffer(1)]], device const float2* ss [[buffer(2)]],
    device const float2* q [[buffer(3)]], device const float* weights [[buffer(4)]],
    device float* output [[buffer(5)]], constant JointGrid& p [[buffer(6)]],
    device const float* trial [[buffer(7)]], constant uint& TOTAL [[buffer(8)]],
    constant uint& chunks [[buffer(9)]], constant float2& step_k [[buffer(10)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint qi = tid.x, candidate = tid.y;
    if (qi >= p.bandCount || candidate >= TOTAL) return;
    size_t out = (size_t)candidate * LOW_ORDER_OUTPUTS;
    float a[51];
    for (uint i = 0; i < LOW_ORDER_FIELDS; ++i) a[i] = 0.0f;
    for (uint c = 0; c < chunks; ++c) {
        size_t base = ((size_t)candidate * chunks + c) * LOW_ORDER_FIELDS;
        for (uint i = 0; i < LOW_ORDER_FIELDS; ++i) a[i] += partial[(base + i) * p.bandCount + qi];
    }
    float f = p.factor * p.wl * p.wl;
    LowOrderSetup s = low_order_setup(q[qi].x, q[qi].y, trial + candidate * 6, p, step_k);
    float cpsi, spsi = metal::precise::sincos(s.psi, cpsi);
    float4 sums = ab[(size_t)candidate * p.bandCount + qi];
    float2 sq = ss[(size_t)candidate * p.bandCount + qi];
    float D = cpsi * cpsi * sq.x + spsi * spsi * sq.y;
    if (!(D > 0.0f)) {
        for (uint i = 0; i < LOW_ORDER_OUTPUTS; ++i) output[(out + i) * p.bandCount + qi] = 0.0f;
        return;
    }
    float nx = cpsi * sums.x - spsi * sums.w, ny = cpsi * sums.y + spsi * sums.z;
    float mx = -spsi * sums.x - cpsi * sums.w, my = -spsi * sums.y + cpsi * sums.z;
    float qx = s.qx, qy = s.qy;
    float psi_j[3] = {f * (qx * qx + qy * qy), f * (qx * qx - qy * qy), 2.0f * f * qx * qy};
    float E = nx * nx + ny * ny;
    float split = sq.y - sq.x;
    float d_psi = 2.0f * spsi * cpsi * split;
    float d_psipsi = 2.0f * (cpsi * cpsi - spsi * spsi) * split;
    // First derivatives of N (complex) and D over z = (C10, a, b, tilt row, tilt column, depth).
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
    output[(out + 0) * p.bandCount + qi] = w * E * inv;
    for (int i = 0; i < 6; ++i)
        output[(out + 1 + i) * p.bandCount + qi] = w * (Ed[i] - E * Dd[i] * inv) * inv;
    // Re(conj N N_ij) and D_ij.
    const int pairs[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    int slot = 7;
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
            output[(out + slot) * p.bandCount + qi] = w * value;
            ++slot;
        }
    }
}
