#include <metal_stdlib>
using namespace metal;
// The 14-term Cartesian polynomial matches ssb.higher_order.CHI_SOURCE. Coefficients
// include 1/(n+1); phases use chi(k-q) for the negative sideband, including odd orders.

inline float ssb_chi_full(float row, float col, float wl, device const float* coefficients) {
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

struct FifthGrid {
    uint count, bandCount, chunks, batch;
    float wl, semiangle, angularRow, angularColumn, factor;
};
kernel void fifth_probe(device const float2* k [[buffer(0)]], device const float* trial [[buffer(1)]],
    device float* chi [[buffer(2)]], constant FifthGrid& p [[buffer(3)]], uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.count || tid.y >= p.batch) return;
    chi[(size_t)tid.x * p.batch + tid.y] = ssb_chi_full(k[tid.x].x, k[tid.x].y, p.wl,
                                                              trial + tid.y * 31 + 3);
}
// Float32 chunk and weighted-band reduction stays on the GPU. The host reads one score per candidate.
kernel void fifth_power(device const float4* partial [[buffer(0)]], device const float* weight [[buffer(1)]],
    device float* power [[buffer(2)]], constant FifthGrid& p [[buffer(3)]], uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.bandCount || tid.y >= p.batch) return;
    float3 sum = float3(0.0f);
    for (uint chunk = 0; chunk < p.chunks; ++chunk)
        sum += partial[((size_t)chunk*p.batch+tid.y)*p.bandCount+tid.x].xyz;
    power[(size_t)tid.y*p.bandCount+tid.x] = sum.z > 0.0f
        ? weight[tid.x] * ((sum.x*sum.x+sum.y*sum.y)/sum.z) : 0.0f;
}
kernel void fifth_score(device const float* power [[buffer(0)]], device float* scores [[buffer(1)]],
    constant FifthGrid& p [[buffer(2)]], uint candidate [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    float value = 0.0f;
    for (uint q = lane; q < p.bandCount; q += 256u) value += power[(size_t)candidate*p.bandCount+q];
    sums[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (lane < offset) sums[lane] += sums[lane+offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) scores[candidate] = sums[0];
}

kernel void fifth_model_probe(device const float2* k [[buffer(0)]], constant float* params [[buffer(1)]],
    device float* probe [[buffer(2)]], constant uint& count [[buffer(3)]],
    device const float* coefficients [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float4 gk = model_geometry(k[i].x, k[i].y, params[0], params[1], params[2], params[3]);
    for (uint j = 0; j < 4; ++j) probe[i*5+j] = gk[j];
    probe[i*5+4] = ssb_chi_full(k[i].x, k[i].y, params[0], coefficients);
}
inline float2 fifth_model_alias_corrected(uint k, float qr, float qc, float shift, float radius,
    device const float2* g, device const float2* kk, constant float* params, device const float* probe,
    size_t source, bool mirror, bool depth, device const float* coefficients) {
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float c10 = params[4], c12 = params[5], c2 = params[6], s2 = params[7];
    float factor = params[8], thickness = params[9];
    float kr = kk[k].x, kc = kk[k].y;
    float4 gk = float4(probe[k * 5], probe[k * 5 + 1], probe[k * 5 + 2], probe[k * 5 + 3]);
    float mr = qr - kr, mc = qc - kc, pr = qr + kr, pc = qc + kc;
    if (metal::min(mr * mr + mc * mc, pr * pr + pc * pc) > radius * radius) return float2(0.0f);
    float4 gm = model_alias_geometry(qr - kr, qc - kc, wl, semi, ay, ax);
    float4 gp = model_alias_geometry(qr + kr, qc + kc, wl, semi, ay, ax);
    if (gk.w == 0.0f || (gm.w == 0.0f && gp.w == 0.0f)) return float2(0.0f);
    float2 value = g[source];
    float vr = value.x, vi = mirror ? -value.y : value.y;
    float chi_k = probe[k * 5 + 4];
    float a = 0.0f, b = 0.0f, ca = 0.0f, cb = 0.0f, sa = 0.0f, sb = 0.0f;
    if (gm.w != 0.0f) {
        float chi_m = ssb_chi_full(kr - qr, kc - qc, wl, coefficients);
        a = gm.w * gk.w;
        if (depth) a = model_sinc(0.5f * thickness * (-factor * (gm.x - gk.x) - shift)) * gm.w * gk.w;
        sa = metal::precise::sincos(chi_m - chi_k, ca);
    }
    if (gp.w != 0.0f) {
        float chi_p = ssb_chi_full(qr + kr, qc + kc, wl, coefficients);
        b = gp.w * gk.w;
        if (depth) b = model_sinc(0.5f * thickness * (factor * (gp.x - gk.x) - shift)) * gp.w * gk.w;
        sb = metal::precise::sincos(chi_p - chi_k, cb);
    }
    float re = a * ca - b * cb, im = -a * sa - b * sb;
    float inv = 1.0f / metal::max(metal::precise::sqrt(re * re + im * im), 1e-8f);
    return float2((vr * re + vi * im) * inv, (vi * re - vr * im) * inv);
}

// One thread per retained output frequency accumulates the corrected waves of one batch of BF pixels, group
// by group, onto the running spectrum. The scan mean is background: its replicas at nonzero output
// frequencies stay zero, and the origin is set once from the common DC afterwards.
kernel void fifth_model_coherent(device const float2* g [[buffer(0)]], device const float* q_row [[buffer(1)]],
    device const float* q_col [[buffer(2)]], device const float2* kk [[buffer(3)]],
    constant float* params [[buffer(4)]], device const float* probe [[buffer(5)]],
    device const float* bounds [[buffer(6)]], device const uint* row_indices [[buffer(7)]],
    device const uint* col_indices [[buffer(8)]], device float2* total [[buffer(9)]],
    constant ModelGrid& p [[buffer(10)]], device const float* coefficients [[buffer(11)]], uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.cols || tid.y >= p.rows) return;
    uint row = row_indices[tid.y], col = col_indices[tid.x];
    if (row % p.sourceN == 0u && col % p.sourceN == 0u) return;
    size_t out = (size_t)row * p.outputN + col;
    float2 result = total[out];
    float qr = q_row[row], qc = q_col[col];
    float shift = 6.283185307179586f * (qr * params[10] + qc * params[11]);
    float radius = (params[1] + 0.5f * metal::max(params[2], params[3])) / params[0];
    radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
    uint source_row = row % p.sourceN, source_col = col % p.sourceN;
    bool mirror = p.sourceHalf != 0u && source_col > p.sourceN / 2u;
    if (mirror) {
        source_row = source_row == 0u ? 0u : p.sourceN - source_row;
        source_col = p.sourceN - source_col;
    }
    size_t plane = (size_t)p.sourceN * p.gCols;
    size_t offset = (size_t)source_row * p.gCols + source_col;
    for (uint group = 0; group < p.groups; ++group) {
        uint start = group * p.chunk, stop = metal::min(start + p.chunk, p.count);
        uint bound = (p.groupOffset + group) * 5;
        float row_min = bounds[bound], row_max = bounds[bound + 1];
        float col_min = bounds[bound + 2], col_max = bounds[bound + 3];
        float rm = max(max(row_min - qr, qr - row_max), 0.0f);
        float cm = max(max(col_min - qc, qc - col_max), 0.0f);
        float rp = max(max(row_min + qr, -qr - row_max), 0.0f);
        float cp = max(max(col_min + qc, -qc - col_max), 0.0f);
        if (min(rm * rm + cm * cm, rp * rp + cp * cp) > bounds[bound + 4]) continue;
        float2 sum = float2(0.0f);
        for (uint local = start; local < stop; ++local) {
            uint k = p.bfOffset + local;
            sum += fifth_model_alias_corrected(k, qr, qc, shift, radius, g + local * plane, kk, params, probe,
                                         offset, mirror, p.depth != 0u, coefficients);
        }
        result += sum;
    }
    total[out] = result;
}
