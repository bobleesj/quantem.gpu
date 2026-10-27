#include <metal_stdlib>
using namespace metal;
// Separable coherent C10/C12 SSB with the thick-sample depth weights: a port of Python MPS
// quantem.gpu.ssb.mps.upsampling (_SEPARABLE_HEADER, _separable_table_kernel, _SEPARABLE_SOURCE, _PAIRED_HEADER,
// _PAIRED_SOURCE, _DEPTH_HEADER, _SEPARABLE_DEPTH_SOURCE, _depth_table_kernel) with the same float32 expressions.
//
// chi(v) = v^T M v, so chi(q -/+ k) - chi(k) = Psi(q) -/+ u(q).k with u.k = rf P_row(k) + cf P_col(k): one row and
// one column phasor per BF pixel replace two soft-edge geometries, two chi values and two sincos per term. The
// depth rates are x1,2 = C -/+ P + B with B = t f q.k, again a row and a column phasor. The flat aperture classes
// skip the weights a margin proves to be 0, 1 or positive. The host builds the phasor parts reduced modulo 2 pi
// from double precision; the kernels accumulate onto a compact (support rows + Nyquist upper row) x columns
// spectrum, which sep_scatter mirrors into the full anti-Hermitian plane.
struct SepGrid {
    uint rows, cols, colsAll, extra, nyquistRI, sourceN, gCols, count;
    uint chunk, groups, groupOffset, tuples, bfOffset, n, pad0, pad1;
    float depthScale, pad2, pad3, pad4;
};

inline float sep_reduce_one(float x) {
    float n = metal::rint(x * 0x1.45f3060000000p-3f);
    float r = x - n * 0x1.9220000000000p+2f;
    r = r - n * -0x1.2b00000000000p-16f;
    return r - n * 0x1.10b4600000000p-28f;
}
inline float sep_reduce(float m, float p1, float p2, float p3) {
    return sep_reduce_one(m * p1) + sep_reduce_one(m * p2) + m * p3;
}
inline float2 sep_cmul(float2 x, float2 y) {
    return float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
inline float4 sep_alias_geometry(float dx, float dy, float wl, float semi, float ay, float ax) {
    float r2 = dx * dx + dy * dy;
    float outer = (semi + 0.5f * max(ay, ax)) / wl;
    outer *= 1.0f + 8.0f * 1.1920928955078125e-7f;
    if (r2 > outer * outer) return float4(0.0f);
    float r = metal::precise::sqrt(r2);
    float alpha = r * wl;
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2 = (dx * dx - dy * dy) * inv_r2;
    float sin2 = 2.0f * dx * dy * inv_r2;
    float inner = semi - 0.5f * max(ay, ax);
    inner *= 1.0f - 8.0f * 1.1920928955078125e-7f;
    if (inner > 0.0f && alpha < inner) return float4(alpha * alpha, cos2, sin2, 1.0f);
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = metal::precise::sqrt((dx * ay) * (dx * ay) + (dy * ax) * (dy * ax)) * inv_r;
    float edge = (denom > 1e-15f) ? ((semi - alpha) / denom + 0.5f) : 1.0f;
    return float4(alpha * alpha, cos2, sin2, metal::clamp(edge, 0.0f, 1.0f));
}

struct SepRadii { float r2_out, r2_zero, r2_one, r2_pos, r2_pos_g, r2_one_g, r2_zero_g; };
inline SepRadii sep_radii(constant float* params) {
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    SepRadii r;
    float radius = (semi + 0.5f * metal::max(ay, ax)) / wl;
    radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
    r.r2_out = radius * radius;
    r.r2_zero = r.r2_out * (1.0f + 32.0f * 1.1920928955078125e-7f);
    float inner = (semi - 0.5f * metal::max(ay, ax)) * (1.0f - 8.0f * 1.1920928955078125e-7f);
    float inner_q = inner / wl;
    r.r2_one = inner > 0.0f ? inner_q * inner_q * (1.0f - 64.0f * 1.1920928955078125e-7f) : -1.0f;
    float positive_q = (semi + 0.5f * metal::min(ay, ax)) / wl;
    r.r2_pos = positive_q * positive_q * (1.0f - 64.0f * 1.1920928955078125e-7f);
    r.r2_pos_g = r.r2_pos * (1.0f - 64.0f * 1.1920928955078125e-7f);
    r.r2_one_g = r.r2_one * (1.0f - 64.0f * 1.1920928955078125e-7f);
    r.r2_zero_g = r.r2_zero * (1.0f + 64.0f * 1.1920928955078125e-7f);
    return r;
}

// 0: no pixel of the group reaches q; 1: every pixel in one flat class (kind 0 a, 1 b, 2 both); 2: per pixel.
inline int sep_group_class(float qr, float qc, float4 box, float min_aperture, float r2_out,
                           float r2_pos_g, float r2_one_g, float r2_zero_g, bool both_ok, thread int &pure_kind) {
    float rm = max(max(box.x - qr, qr - box.y), 0.0f);
    float cm = max(max(box.z - qc, qc - box.w), 0.0f);
    float rp = max(max(box.x + qr, -qr - box.y), 0.0f);
    float cp = max(max(box.z + qc, -qc - box.w), 0.0f);
    float near_m = rm * rm + cm * cm, near_p = rp * rp + cp * cp;
    if (min(near_m, near_p) > r2_out) return 0;
    float rmf = max(metal::abs(qr - box.x), metal::abs(qr - box.y));
    float cmf = max(metal::abs(qc - box.z), metal::abs(qc - box.w));
    float rpf = max(metal::abs(qr + box.x), metal::abs(qr + box.y));
    float cpf = max(metal::abs(qc + box.z), metal::abs(qc + box.w));
    float far_m = rmf * rmf + cmf * cmf, far_p = rpf * rpf + cpf * cpf;
    bool pure_m = far_m < r2_pos_g && near_p > r2_zero_g;
    bool pure_p = far_p < r2_pos_g && near_m > r2_zero_g;
    bool pure_i = far_m < r2_one_g && far_p < r2_one_g && both_ok;
    if ((pure_m || pure_p || pure_i) && min_aperture >= 1e-3f) {
        pure_kind = pure_m ? 0 : (pure_p ? 1 : 2);
        return 1;
    }
    return 2;
}

inline void sep_mixed_term(float qr, float qc, float4 kd, float2 term, float2 epsi, bool both_ok,
                           SepRadii R, float wl, float semi, float ay, float ax,
                           thread float2 &ga, thread float2 &gb, thread float2 &gab, thread float2 &gi) {
    float mr = qr - kd.x, mc = qc - kd.y, pr = qr + kd.x, pc = qc + kd.y;
    float rm2 = mr * mr + mc * mc, rp2 = pr * pr + pc * pc;
    if (metal::min(rm2, rp2) > R.r2_out) return;
    bool closed_m = rm2 > R.r2_zero, closed_p = rp2 > R.r2_zero;
    bool single_m = rm2 < R.r2_pos && closed_p, single_p = rp2 < R.r2_pos && closed_m;
    bool both_one = rm2 < R.r2_one && rp2 < R.r2_one && both_ok;
    if ((single_m || single_p || both_one) && kd.z >= 1e-3f) {
        ga += single_m ? term : float2(0.0f);
        gb += single_p ? term : float2(0.0f);
        gi += both_one ? term : float2(0.0f);
        return;
    }
    float wa = closed_m ? 0.0f : (rm2 < R.r2_one ? 1.0f : sep_alias_geometry(mr, mc, wl, semi, ay, ax).w);
    float wb = closed_p ? 0.0f : (rp2 < R.r2_one ? 1.0f : sep_alias_geometry(pr, pc, wl, semi, ay, ax).w);
    float gkw = kd.z;
    if (gkw == 0.0f || (wa == 0.0f && wb == 0.0f)) return;
    float a = wa * gkw, b = wb * gkw;
    if (b == 0.0f) {
        ga += a >= 1e-8f ? term : term * (a / 1e-8f);
    } else if (a == 0.0f) {
        gb += b >= 1e-8f ? term : term * (b / 1e-8f);
    } else {
        float wr = (a - b) * epsi.x, wi = (a + b) * epsi.y;
        float inv = 1.0f / metal::max(metal::precise::sqrt(wr * wr + wi * wi), 1e-8f);
        gab += sep_cmul(term, float2(wr * inv, wi * inv));
    }
}

// Row and column phasors exp(-i rf P_row(k)) and exp(-i cf P_col(k)) of every stored BF pixel.
kernel void sep_tables(device const float* p_row [[buffer(0)]], device const float* p_col [[buffer(1)]],
    device const float* row_f [[buffer(2)]], device const float* col_f [[buffer(3)]],
    device float2* t_row [[buffer(4)]], device float2* t_col [[buffer(5)]], constant SepGrid& p [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint i = tid.x, k = tid.y;
    if (k >= p.count) return;
    if (i < p.rows) {
        float x = sep_reduce(row_f[i], p_row[k * 3], p_row[k * 3 + 1], p_row[k * 3 + 2]);
        float c;
        float s = metal::precise::sincos(x, c);
        t_row[(size_t)k * p.rows + i] = float2(c, -s);
    }
    if (i < p.colsAll) {
        float x = sep_reduce(col_f[i], p_col[k * 3], p_col[k * 3 + 1], p_col[k * 3 + 2]);
        float c;
        float s = metal::precise::sincos(x, c);
        t_col[(size_t)k * p.colsAll + i] = float2(c, -s);
    }
}

// Row and column float4 tables (transfer phasor, depth phasor) of every stored BF pixel.
kernel void sep_depth_tables(device const float* p_row [[buffer(0)]], device const float* p_col [[buffer(1)]],
    device const float* d_row [[buffer(2)]], device const float* d_col [[buffer(3)]],
    device const float* row_f [[buffer(4)]], device const float* col_f [[buffer(5)]],
    device float4* t_row [[buffer(6)]], device float4* t_col [[buffer(7)]], constant SepGrid& p [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint i = tid.x, k = tid.y;
    if (k >= p.count) return;
    if (i < p.rows) {
        float x = sep_reduce(row_f[i], p_row[k * 3], p_row[k * 3 + 1], p_row[k * 3 + 2]);
        float y = sep_reduce(row_f[i], d_row[k * 3], d_row[k * 3 + 1], d_row[k * 3 + 2]);
        float c, d;
        float s = metal::precise::sincos(x, c);
        float e = metal::precise::sincos(y, d);
        t_row[(size_t)k * p.rows + i] = float4(c, -s, d, -e);
    }
    if (i < p.colsAll) {
        float x = sep_reduce(col_f[i], p_col[k * 3], p_col[k * 3 + 1], p_col[k * 3 + 2]);
        float y = sep_reduce(col_f[i], d_col[k * 3], d_col[k * 3 + 1], d_col[k * 3 + 2]);
        float c, d;
        float s = metal::precise::sincos(x, c);
        float e = metal::precise::sincos(y, d);
        t_col[(size_t)k * p.colsAll + i] = float4(c, -s, d, -e);
    }
}

// The phasor epsi = exp(i Psi(q)) of one output frequency from its row, column and cross parts.
inline float2 sep_epsi(uint ri, uint ci, device const float2* psi_row, device const float2* psi_col,
                       device const float* psi_cross, device const float* col_f) {
    float xb = sep_reduce(col_f[ci], psi_cross[ri * 3], psi_cross[ri * 3 + 1], psi_cross[ri * 3 + 2]);
    float cb;
    float sb = metal::precise::sincos(xb, cb);
    return sep_cmul(sep_cmul(psi_row[ri], psi_col[ci]), float2(cb, sb));
}

// One output per thread (any factor without depth). One grid row past ROWS computes the upper half of the
// Nyquist row, whose anti-Hermitian partner is itself. `k` indexes the launch's BF pixels; tables, k_data and
// bounds are bound at the launch's first pixel and group.
kernel void sep_coherent(device const float2* g [[buffer(0)]], device const float* q_row [[buffer(1)]],
    device const float* q_col [[buffer(2)]], device const float4* k_data [[buffer(3)]],
    constant float* params [[buffer(4)]], device const float* bounds [[buffer(5)]],
    device const uint* row_indices [[buffer(6)]], device const uint* col_indices [[buffer(7)]],
    device const float2* t_row [[buffer(8)]], device const float2* t_col [[buffer(9)]],
    device const float2* psi_row [[buffer(10)]], device const float2* psi_col [[buffer(11)]],
    device const float* psi_cross [[buffer(12)]], device const float* col_f [[buffer(13)]],
    device float2* total [[buffer(14)]], constant SepGrid& p [[buffer(15)]],
    uint2 tid [[thread_position_in_grid]]) {
    constexpr uint PURE_UNROLL = 4;
    uint gx = tid.x, gy = tid.y;
    uint ri, ci;
    size_t out;
    if (gy < p.rows) {
        if (gx >= p.cols) return;
        ri = gy; ci = gx; out = (size_t)gy * p.cols + gx;
    } else {
        if (gy > p.rows || gx >= p.extra) return;
        ri = p.nyquistRI; ci = p.cols + gx; out = (size_t)p.rows * p.cols + gx;
    }
    uint row = row_indices[ri], col = col_indices[ci];
    if (row % p.sourceN == 0u && col % p.sourceN == 0u) return;
    float2 result = total[out];
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float qr = q_row[row], qc = q_col[col];
    SepRadii R = sep_radii(params);
    uint source_row = row % p.sourceN, source_col = col % p.sourceN;
    bool mirror = p.gCols != p.sourceN && source_col > p.sourceN / 2u;
    if (mirror) {
        source_row = source_row == 0u ? 0u : p.sourceN - source_row;
        source_col = p.sourceN - source_col;
    }
    size_t plane = (size_t)p.sourceN * p.gCols;
    size_t source_offset = size_t(source_row) * p.gCols + source_col;
    float2 epsi = sep_epsi(ri, ci, psi_row, psi_col, psi_cross, col_f);
    float2 sum_a = float2(0.0f), sum_b = float2(0.0f), sum_ab = float2(0.0f), sum_i = float2(0.0f);
    bool both_ok = metal::abs(epsi.y) >= 1e-4f;
    for (uint group = 0; group < p.groups; ++group) {
        uint start = group * p.chunk, stop = metal::min(start + p.chunk, p.count);
        float4 box = ((const device float4*)bounds)[2 * group];
        float min_aperture = bounds[8 * group + 4];
        int kind = 0;
        int cls = sep_group_class(qr, qc, box, min_aperture, R.r2_out, R.r2_pos_g, R.r2_one_g, R.r2_zero_g,
                                  both_ok, kind);
        if (cls == 0) continue;
        if (cls == 1) {
            float2 acc = float2(0.0f);
            uint k = start;
            for (; k + PURE_UNROLL <= stop; k += PURE_UNROLL) {
                float2 gv[PURE_UNROLL], tr[PURE_UNROLL], tc[PURE_UNROLL];
                for (uint j = 0; j < PURE_UNROLL; ++j) {
                    float2 value = g[(size_t)(k + j) * plane + source_offset];
                    gv[j] = float2(value.x, mirror ? -value.y : value.y);
                    tr[j] = t_row[(size_t)(k + j) * p.rows + ri];
                    tc[j] = t_col[(size_t)(k + j) * p.colsAll + ci];
                }
                for (uint j = 0; j < PURE_UNROLL; ++j) acc += sep_cmul(gv[j], sep_cmul(tr[j], tc[j]));
            }
            for (; k < stop; ++k) {
                float2 value = g[(size_t)k * plane + source_offset];
                float2 gv = float2(value.x, mirror ? -value.y : value.y);
                acc += sep_cmul(gv, sep_cmul(t_row[(size_t)k * p.rows + ri], t_col[(size_t)k * p.colsAll + ci]));
            }
            if (kind == 0) sum_a += acc; else if (kind == 1) sum_b += acc; else sum_i += acc;
            continue;
        }
        float2 ga = float2(0.0f), gb = float2(0.0f), gab = float2(0.0f), gi = float2(0.0f);
        for (uint k = start; k < stop; ++k) {
            float4 kd = k_data[k];
            float2 value = g[(size_t)k * plane + source_offset];
            float2 gv = float2(value.x, mirror ? -value.y : value.y);
            float2 term = sep_cmul(gv, sep_cmul(t_row[(size_t)k * p.rows + ri], t_col[(size_t)k * p.colsAll + ci]));
            sep_mixed_term(qr, qc, kd, term, epsi, both_ok, R, wl, semi, ay, ax, ga, gb, gab, gi);
        }
        sum_a += ga; sum_b += gb; sum_ab += gab; sum_i += gi;
    }
    float2 w_one = float2(0.0f, epsi.y > 0.0f ? 1.0f : -1.0f);
    result += sep_cmul(epsi, sum_a) - sep_cmul(float2(epsi.x, -epsi.y), sum_b) + sum_ab + sep_cmul(w_one, sum_i);
    total[out] = result;
}

// The 2x preview: the two fine rows of one source row (pairs[gy]) per thread, from one G read.
kernel void sep_coherent_paired(device const float2* g [[buffer(0)]], device const float* q_row [[buffer(1)]],
    device const float* q_col [[buffer(2)]], device const float4* k_data [[buffer(3)]],
    constant float* params [[buffer(4)]], device const float* bounds [[buffer(5)]],
    device const uint* row_indices [[buffer(6)]], device const uint* col_indices [[buffer(7)]],
    device const float2* t_row [[buffer(8)]], device const float2* t_col [[buffer(9)]],
    device const float2* psi_row [[buffer(10)]], device const float2* psi_col [[buffer(11)]],
    device const float* psi_cross [[buffer(12)]], device const float* col_f [[buffer(13)]],
    device float2* total [[buffer(14)]], constant SepGrid& p [[buffer(15)]],
    device const uint* pairs [[buffer(16)]], uint2 tid [[thread_position_in_grid]]) {
    uint gx = tid.x, gy = tid.y;
    uint ci, ris[2];
    size_t outs[2] = {0, 0};
    if (gy < p.tuples) {
        if (gx >= p.cols) return;
        ci = gx; ris[0] = pairs[2 * gy]; ris[1] = pairs[2 * gy + 1];
        outs[0] = (size_t)ris[0] * p.cols + gx;
        if (ris[1] != 0xFFFFFFFFu) outs[1] = (size_t)ris[1] * p.cols + gx;
    } else {
        if (gy > p.tuples || gx >= p.extra) return;
        ris[0] = p.nyquistRI; ris[1] = 0xFFFFFFFFu; ci = p.cols + gx; outs[0] = (size_t)p.rows * p.cols + gx;
    }
    bool has[2] = {true, ris[1] != 0xFFFFFFFFu};
    uint col = col_indices[ci];
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float qc = q_col[col];
    SepRadii R = sep_radii(params);
    uint source_row = row_indices[ris[0]] % p.sourceN, source_col = col % p.sourceN;
    bool mirror = p.gCols != p.sourceN && source_col > p.sourceN / 2u;
    if (mirror) {
        source_row = source_row == 0u ? 0u : p.sourceN - source_row;
        source_col = p.sourceN - source_col;
    }
    size_t plane = (size_t)p.sourceN * p.gCols;
    size_t source_offset = size_t(source_row) * p.gCols + source_col;
    float qrs[2];
    float2 epsi[2], sum_a[2], sum_b[2], sum_ab[2], sum_i[2];
    bool both_ok[2], active[2];
    for (uint j = 0; j < 2; ++j) {
        uint ri = has[j] ? ris[j] : ris[0];
        uint row = row_indices[ri];
        qrs[j] = q_row[row];
        active[j] = has[j] && (row % p.sourceN != 0u || col % p.sourceN != 0u);
        epsi[j] = sep_epsi(ri, ci, psi_row, psi_col, psi_cross, col_f);
        both_ok[j] = metal::abs(epsi[j].y) >= 1e-4f;
        sum_a[j] = sum_b[j] = sum_ab[j] = sum_i[j] = float2(0.0f);
    }
    if (active[0] || active[1]) {
        for (uint group = 0; group < p.groups; ++group) {
            uint start = group * p.chunk, stop = metal::min(start + p.chunk, p.count);
            float4 box = ((const device float4*)bounds)[2 * group];
            float min_aperture = bounds[8 * group + 4];
            int kinds[2] = {0, 0}, classes[2];
            for (uint j = 0; j < 2; ++j) {
                classes[j] = active[j] ? sep_group_class(qrs[j], qc, box, min_aperture, R.r2_out, R.r2_pos_g,
                                                         R.r2_one_g, R.r2_zero_g, both_ok[j], kinds[j]) : 0;
            }
            if (classes[0] == 0 && classes[1] == 0) continue;
            bool per_pixel = classes[0] == 2 || classes[1] == 2;
            float2 acc[2] = {float2(0.0f), float2(0.0f)}, ga[2] = {float2(0.0f), float2(0.0f)},
                   gb[2] = {float2(0.0f), float2(0.0f)}, gab[2] = {float2(0.0f), float2(0.0f)},
                   gi[2] = {float2(0.0f), float2(0.0f)};
            for (uint k = start; k < stop; ++k) {
                float2 value = g[(size_t)k * plane + source_offset];
                float2 gv = float2(value.x, mirror ? -value.y : value.y);
                float2 tc = t_col[(size_t)k * p.colsAll + ci];
                float4 kd = per_pixel ? k_data[k] : float4(0.0f);
                for (uint j = 0; j < 2; ++j) {
                    if (classes[j] == 0) continue;
                    float2 term = sep_cmul(gv, sep_cmul(t_row[(size_t)k * p.rows + ris[j]], tc));
                    if (classes[j] == 1) acc[j] += term;
                    else sep_mixed_term(qrs[j], qc, kd, term, epsi[j], both_ok[j], R, wl, semi, ay, ax,
                                        ga[j], gb[j], gab[j], gi[j]);
                }
            }
            for (uint j = 0; j < 2; ++j) {
                if (classes[j] == 1) {
                    if (kinds[j] == 0) sum_a[j] += acc[j]; else if (kinds[j] == 1) sum_b[j] += acc[j];
                    else sum_i[j] += acc[j];
                } else if (classes[j] == 2) {
                    sum_a[j] += ga[j]; sum_b[j] += gb[j]; sum_ab[j] += gab[j]; sum_i[j] += gi[j];
                }
            }
        }
    }
    for (uint j = 0; j < 2; ++j) {
        if (!active[j]) continue;
        float2 w_one = float2(0.0f, epsi[j].y > 0.0f ? 1.0f : -1.0f);
        total[outs[j]] += sep_cmul(epsi[j], sum_a[j]) - sep_cmul(float2(epsi[j].x, -epsi[j].y), sum_b[j])
                          + sum_ab[j] + sep_cmul(w_one, sum_i[j]);
    }
}

// sinc(x) as (numerator, denominator): its Taylor series to x^10 over 1 below |x| = 1, sin x over x above.
inline float2 sep_sinc_parts(float x, float s) {
    float x2 = x * x;
    float series = 1.0f - x2 * (1.0f / 6.0f) * (1.0f - x2 * (1.0f / 20.0f) * (1.0f - x2 * (1.0f / 42.0f)
                 * (1.0f - x2 * (1.0f / 72.0f) * (1.0f - x2 * (1.0f / 110.0f)))));
    return metal::abs(x) < 1.0f ? float2(series, 1.0f) : float2(s, x);
}
inline float sep_sinc(float x, float s) {
    float2 parts = sep_sinc_parts(x, s);
    return parts.x / parts.y;
}
inline float sep_side(float a) {
    return metal::abs(a) >= 1e-8f ? metal::sign(a) : a * 1e8f;
}
inline float2 sep_both(float2 p1, float2 p2, float2 epsi) {
    float a = p1.x * p2.y, b = p2.x * p1.y;
    float wr = (a - b) * epsi.x, wi = (a + b) * epsi.y;
    float m2 = wr * wr + wi * wi;
    float inv = m2 > 0.0f ? metal::sign(p1.y * p2.y) * metal::precise::rsqrt(m2) : 0.0f;
    return float2(wr * inv, wi * inv);
}
inline float2 sep_depth_pure(int kind, float2 gv, float4 tr, float4 tc, float2 kk, float bq_r, float bq_c,
                             float xm0, float xp0, float2 em, float2 ep, float2 epsi) {
    float2 term = sep_cmul(gv, sep_cmul(tr.xy, tc.xy));
    float2 eb = sep_cmul(tr.zw, tc.zw);
    float bk = metal::fma(bq_r, kk.x, bq_c * kk.y);
    if (kind == 0) {
        float2 p1 = sep_sinc_parts(xm0 + bk, em.x * eb.y + em.y * eb.x);
        return term * (metal::sign(p1.x) * metal::sign(p1.y));
    }
    if (kind == 1) {
        float2 p2 = sep_sinc_parts(xp0 + bk, ep.x * eb.y + ep.y * eb.x);
        return term * (metal::sign(p2.x) * metal::sign(p2.y));
    }
    float2 p1 = sep_sinc_parts(xm0 + bk, em.x * eb.y + em.y * eb.x);
    float2 p2 = sep_sinc_parts(xp0 + bk, ep.x * eb.y + ep.y * eb.x);
    return sep_cmul(term, sep_both(p1, p2, epsi));
}
inline void sep_depth_pixel(float qr, float qc, float4 kd, float2 gv, float4 tr, float4 tc, float bq_r, float bq_c,
                            float xm0, float xp0, float2 em, float2 ep, float2 epsi, SepRadii R,
                            float wl, float semi, float ay, float ax,
                            thread float2 &ga, thread float2 &gb, thread float2 &gab) {
    float mr = qr - kd.x, mc = qc - kd.y, pr = qr + kd.x, pc = qc + kd.y;
    float rm2 = mr * mr + mc * mc, rp2 = pr * pr + pc * pc;
    float gkw = kd.z;
    if (metal::min(rm2, rp2) > R.r2_out || gkw == 0.0f) return;
    bool closed_m = rm2 > R.r2_zero, closed_p = rp2 > R.r2_zero;
    float2 term = sep_cmul(gv, sep_cmul(tr.xy, tc.xy));
    float2 eb = sep_cmul(tr.zw, tc.zw);
    float bk = metal::fma(bq_r, kd.x, bq_c * kd.y);
    if (gkw >= 1e-3f) {
        if (rm2 < R.r2_pos && closed_p) {
            float2 p1 = sep_sinc_parts(xm0 + bk, em.x * eb.y + em.y * eb.x);
            ga += term * (metal::sign(p1.x) * metal::sign(p1.y));
            return;
        }
        if (rp2 < R.r2_pos && closed_m) {
            float2 p2 = sep_sinc_parts(xp0 + bk, ep.x * eb.y + ep.y * eb.x);
            gb += term * (metal::sign(p2.x) * metal::sign(p2.y));
            return;
        }
        if (rm2 < R.r2_one && rp2 < R.r2_one) {
            float2 p1 = sep_sinc_parts(xm0 + bk, em.x * eb.y + em.y * eb.x);
            float2 p2 = sep_sinc_parts(xp0 + bk, ep.x * eb.y + ep.y * eb.x);
            gab += sep_cmul(term, sep_both(p1, p2, epsi));
            return;
        }
    }
    float wa = closed_m ? 0.0f : (rm2 < R.r2_one ? 1.0f : sep_alias_geometry(mr, mc, wl, semi, ay, ax).w);
    float wb = closed_p ? 0.0f : (rp2 < R.r2_one ? 1.0f : sep_alias_geometry(pr, pc, wl, semi, ay, ax).w);
    if (wa == 0.0f && wb == 0.0f) return;
    float a = wa != 0.0f ? sep_sinc(xm0 + bk, em.x * eb.y + em.y * eb.x) * wa * gkw : 0.0f;
    float b = wb != 0.0f ? sep_sinc(xp0 + bk, ep.x * eb.y + ep.y * eb.x) * wb * gkw : 0.0f;
    if (b == 0.0f) {
        ga += term * sep_side(a);
    } else if (a == 0.0f) {
        gb += term * sep_side(b);
    } else {
        float wr = (a - b) * epsi.x, wi = (a + b) * epsi.y;
        float inv = 1.0f / metal::max(metal::precise::sqrt(wr * wr + wi * wi), 1e-8f);
        gab += sep_cmul(term, float2(wr * inv, wi * inv));
    }
}

// The depth-weighted kernel: the ALIASES fine rows of one column sharing a source row (tuples[gy]) per thread.
template <uint ALIASES>
kernel void sep_coherent_depth(device const float2* g [[buffer(0)]], device const float* q_row [[buffer(1)]],
    device const float* q_col [[buffer(2)]], device const float4* k_data [[buffer(3)]],
    constant float* params [[buffer(4)]], device const float* bounds [[buffer(5)]],
    device const uint* row_indices [[buffer(6)]], device const uint* col_indices [[buffer(7)]],
    device const float4* row_tables [[buffer(8)]], device const float4* col_tables [[buffer(9)]],
    device const float2* psi_row [[buffer(10)]], device const float2* psi_col [[buffer(11)]],
    device const float* psi_cross [[buffer(12)]], device const float* col_f [[buffer(13)]],
    device float2* total [[buffer(14)]], constant SepGrid& p [[buffer(15)]],
    device const uint* tuples [[buffer(16)]], device const float2* dm_row [[buffer(17)]],
    device const float2* dm_col [[buffer(18)]], device const float2* dp_row [[buffer(19)]],
    device const float2* dp_col [[buffer(20)]], device const float* x_row [[buffer(21)]],
    device const float* x_col [[buffer(22)]], uint2 tid [[thread_position_in_grid]]) {
    uint gx = tid.x, gy = tid.y;
    uint ci;
    uint ris[ALIASES];
    size_t outs[ALIASES];
    bool has[ALIASES];
    if (gy < p.tuples) {
        if (gx >= p.cols) return;
        ci = gx;
        for (uint j = 0; j < ALIASES; ++j) {
            ris[j] = tuples[gy * ALIASES + j];
            has[j] = ris[j] != 0xFFFFFFFFu;
            outs[j] = has[j] ? (size_t)ris[j] * p.cols + gx : 0;
        }
    } else {
        if (gy > p.tuples || gx >= p.extra) return;
        ci = p.cols + gx;
        for (uint j = 0; j < ALIASES; ++j) { ris[j] = p.nyquistRI; has[j] = j == 0; outs[j] = 0; }
        outs[0] = (size_t)p.rows * p.cols + gx;
    }
    uint col = col_indices[ci];
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float qc = q_col[col];
    SepRadii R = sep_radii(params);
    uint source_row = row_indices[ris[0]] % p.sourceN, source_col = col % p.sourceN;
    bool mirror = p.gCols != p.sourceN && source_col > p.sourceN / 2u;
    if (mirror) {
        source_row = source_row == 0u ? 0u : p.sourceN - source_row;
        source_col = p.sourceN - source_col;
    }
    size_t plane = (size_t)p.sourceN * p.gCols;
    size_t source_offset = size_t(source_row) * p.gCols + source_col;
    float2 dm_c = dm_col[ci], dp_c = dp_col[ci];
    float bq_c = p.depthScale * qc;
    float qrs[ALIASES], xm0[ALIASES], xp0[ALIASES], bq_r[ALIASES];
    float2 epsi[ALIASES], em[ALIASES], ep[ALIASES];
    float2 sum_a[ALIASES], sum_b[ALIASES], sum_ab[ALIASES];
    bool active[ALIASES];
    bool any_active = false;
    for (uint j = 0; j < ALIASES; ++j) {
        uint ri = has[j] ? ris[j] : ris[0];
        uint row = row_indices[ri];
        qrs[j] = q_row[row];
        sum_a[j] = sum_b[j] = sum_ab[j] = float2(0.0f);
        active[j] = has[j] && (row % p.sourceN != 0u || col % p.sourceN != 0u);
        any_active = any_active || active[j];
        epsi[j] = sep_epsi(ri, ci, psi_row, psi_col, psi_cross, col_f);
        em[j] = sep_cmul(dm_row[ri], dm_c);
        ep[j] = sep_cmul(dp_row[ri], dp_c);
        float c_q = x_row[2 * ri + 1] + x_col[2 * ci + 1], p_q = x_row[2 * ri] + x_col[2 * ci];
        xm0[j] = c_q - p_q;
        xp0[j] = c_q + p_q;
        bq_r[j] = p.depthScale * qrs[j];
    }
    if (any_active) {
        for (uint group = 0; group < p.groups; ++group) {
            uint start = group * p.chunk, stop = metal::min(start + p.chunk, p.count);
            float4 box = ((const device float4*)bounds)[2 * group];
            float min_aperture = bounds[8 * group + 4];
            int classes[ALIASES], kinds[ALIASES];
            bool any = false;
            for (uint j = 0; j < ALIASES; ++j) {
                kinds[j] = 0;
                classes[j] = active[j] ? sep_group_class(qrs[j], qc, box, min_aperture, R.r2_out, R.r2_pos_g,
                                                         R.r2_one_g, R.r2_zero_g, true, kinds[j]) : 0;
                any = any || classes[j] != 0;
            }
            if (!any) continue;
            float2 ga[ALIASES], gb[ALIASES], gab[ALIASES];
            for (uint j = 0; j < ALIASES; ++j) ga[j] = gb[j] = gab[j] = float2(0.0f);
            for (uint k = start; k < stop; ++k) {
                float2 value = g[(size_t)k * plane + source_offset];
                float2 gv = float2(value.x, mirror ? -value.y : value.y);
                float4 kd = k_data[k];
                float4 tc = col_tables[(size_t)k * p.colsAll + ci];
                for (uint j = 0; j < ALIASES; ++j) {
                    if (classes[j] == 0) continue;
                    float4 tr = row_tables[(size_t)k * p.rows + ris[j]];
                    if (classes[j] == 1) {
                        float2 v = sep_depth_pure(kinds[j], gv, tr, tc, kd.xy, bq_r[j], bq_c, xm0[j], xp0[j],
                                                  em[j], ep[j], epsi[j]);
                        if (kinds[j] == 0) ga[j] += v; else if (kinds[j] == 1) gb[j] += v; else gab[j] += v;
                        continue;
                    }
                    sep_depth_pixel(qrs[j], qc, kd, gv, tr, tc, bq_r[j], bq_c, xm0[j], xp0[j], em[j], ep[j],
                                    epsi[j], R, wl, semi, ay, ax, ga[j], gb[j], gab[j]);
                }
            }
            for (uint j = 0; j < ALIASES; ++j) { sum_a[j] += ga[j]; sum_b[j] += gb[j]; sum_ab[j] += gab[j]; }
        }
    }
    for (uint j = 0; j < ALIASES; ++j) {
        if (!has[j] || !active[j]) continue;
        total[outs[j]] += sep_cmul(epsi[j], sum_a[j]) - sep_cmul(float2(epsi[j].x, -epsi[j].y), sum_b[j])
                          + sum_ab[j];
    }
}
template [[host_name("sep_coherent_depth1")]] kernel void sep_coherent_depth<1>(
    device const float2*, device const float*, device const float*, device const float4*, constant float*,
    device const float*, device const uint*, device const uint*, device const float4*, device const float4*,
    device const float2*, device const float2*, device const float*, device const float*, device float2*,
    constant SepGrid&, device const uint*, device const float2*, device const float2*, device const float2*,
    device const float2*, device const float*, device const float*, uint2);
template [[host_name("sep_coherent_depth2")]] kernel void sep_coherent_depth<2>(
    device const float2*, device const float*, device const float*, device const float4*, constant float*,
    device const float*, device const uint*, device const uint*, device const float4*, device const float4*,
    device const float2*, device const float2*, device const float*, device const float*, device float2*,
    constant SepGrid&, device const uint*, device const float2*, device const float2*, device const float2*,
    device const float2*, device const float*, device const float*, uint2);
template [[host_name("sep_coherent_depth3")]] kernel void sep_coherent_depth<3>(
    device const float2*, device const float*, device const float*, device const float4*, constant float*,
    device const float*, device const uint*, device const uint*, device const float4*, device const float4*,
    device const float2*, device const float2*, device const float*, device const float*, device float2*,
    constant SepGrid&, device const uint*, device const float2*, device const float2*, device const float2*,
    device const float2*, device const float*, device const float*, uint2);

// The full n x n spectrum from the compact one: support rows and columns 0..n/2 directly, the Nyquist row's
// upper columns from the extra row, every other upper column from its anti-Hermitian partner -conj(O(-q)).
kernel void sep_scatter(device const float2* compact [[buffer(0)]], device float2* full [[buffer(1)]],
    device const int* row_position [[buffer(2)]], device const int* col_position [[buffer(3)]],
    constant SepGrid& p [[buffer(4)]], constant int& upper_first [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint c = tid.x, r = tid.y, n = p.n;
    if (c >= n || r >= n) return;
    uint halfN = n / 2;
    float2 value = float2(0.0f);
    if (c <= halfN) {
        int rp = row_position[r], cp = col_position[c];
        if (rp >= 0 && cp >= 0) value = compact[(size_t)rp * p.cols + cp];
    } else if (r == halfN && p.extra > 0) {
        if (upper_first >= 0 && int(c) >= upper_first) value = compact[(size_t)p.rows * p.cols + (c - upper_first)];
    } else {
        uint mr = (n - r) % n, mc = n - c;
        int rp = row_position[mr], cp = col_position[mc];
        if (rp >= 0 && cp >= 0) {
            float2 v = compact[(size_t)rp * p.cols + cp];
            value = float2(-v.x, v.y);
        }
    }
    full[(size_t)r * n + c] = value;
}
