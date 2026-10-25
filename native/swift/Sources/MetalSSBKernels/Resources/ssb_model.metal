#include <metal_stdlib>
using namespace metal;
// Coherent (phase_of_mean) SSB reconstruction with the depth-averaged transfer model, at the native scan
// sampling or a finer output grid. Port of quantem.gpu.ssb.mps.upsampling (_alias_probe_kernel,
// _alias_group_bounds_kernel, _ALIAS_GEOMETRY, _ALIAS_TRANSFER, _coherent_alias_kernel): the same float32
// expressions in the same order, so the phase agrees with the Python MPS path to float32 rounding.
//
// params (12 floats, the MPS order): wavelength, semiangle, angular row, angular column, C10, C12,
// cos 2 phi12, sin 2 phi12, pi / wavelength, depth spread, tilt row (rad), tilt column (rad).
struct ModelGrid {
    uint sourceN, outputN, gCols, count, bfOffset, sourceHalf, rows, cols;
    uint chunk, groups, groupOffset, depth;
};

inline float4 model_geometry(float dx, float dy, float wl, float semiangle, float ang_y, float ang_x) {
    float r2 = dx * dx + dy * dy;
    float r = metal::precise::sqrt(r2);
    float alpha = r * wl;
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2 = (dx * dx - dy * dy) * inv_r2;
    float sin2 = 2.0f * dx * dy * inv_r2;
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = metal::precise::sqrt((dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x)) * inv_r;
    float edge = (denom > 1e-15f) ? ((semiangle - alpha) / denom + 0.5f) : 1.0f;
    return float4(alpha * alpha, cos2, sin2, metal::clamp(edge, 0.0f, 1.0f));
}
inline float model_sinc(float x) { return (metal::abs(x) < 1e-6f) ? 1.0f : metal::precise::sin(x) / x; }

// Fine-grid redraws evaluate a shifted probe well outside or inside the aperture; only conservative bounds
// skip the soft-edge arithmetic.
inline float4 model_alias_geometry(float dx, float dy, float wl, float semi, float ay, float ax) {
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
    if (inner > 0.0f && alpha < inner) {
        return float4(alpha * alpha, cos2, sin2, 1.0f);
    }
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = metal::precise::sqrt((dx * ay) * (dx * ay) + (dy * ax) * (dy * ax)) * inv_r;
    float edge = (denom > 1e-15f) ? ((semi - alpha) / denom + 0.5f) : 1.0f;
    return float4(alpha * alpha, cos2, sin2, metal::clamp(edge, 0.0f, 1.0f));
}

kernel void model_probe(device const float2* k [[buffer(0)]], constant float* params [[buffer(1)]],
    device float* probe [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float4 gk = model_geometry(k[i].x, k[i].y, params[0], params[1], params[2], params[3]);
    for (uint j = 0; j < 4; ++j) probe[i * 5 + j] = gk[j];
    probe[i * 5 + 4] = params[8] * gk.x * (params[5] * (gk.y * params[6] + gk.z * params[7]) + params[4]);
}

inline float2 model_alias_corrected(uint k, float qr, float qc, float shift, float radius,
    device const float2* g, device const float2* kk, constant float* params, device const float* probe,
    size_t source, bool mirror, bool depth) {
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
        float chi_m = factor * gm.x * (c12 * (gm.y * c2 + gm.z * s2) + c10);
        a = gm.w * gk.w;
        if (depth) a = model_sinc(0.5f * thickness * (-factor * (gm.x - gk.x) - shift)) * gm.w * gk.w;
        sa = metal::precise::sincos(chi_m - chi_k, ca);
    }
    if (gp.w != 0.0f) {
        float chi_p = factor * gp.x * (c12 * (gp.y * c2 + gp.z * s2) + c10);
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
kernel void model_coherent(device const float2* g [[buffer(0)]], device const float* q_row [[buffer(1)]],
    device const float* q_col [[buffer(2)]], device const float2* kk [[buffer(3)]],
    constant float* params [[buffer(4)]], device const float* probe [[buffer(5)]],
    device const float* bounds [[buffer(6)]], device const uint* row_indices [[buffer(7)]],
    device const uint* col_indices [[buffer(8)]], device float2* total [[buffer(9)]],
    constant ModelGrid& p [[buffer(10)]], uint2 tid [[thread_position_in_grid]]) {
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
            sum += model_alias_corrected(k, qr, qc, shift, radius, g + local * plane, kk, params, probe,
                                         offset, mirror, p.depth != 0u);
        }
        result += sum;
    }
    total[out] = result;
}

// For the even two-term probe the corrected spectrum is anti-Hermitian, C(-q) = -conj(C(q)); the host uses
// the half only when neither Nyquist edge has support.
kernel void model_mirror(device float2* total [[buffer(0)]], constant uint& n [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint col = tid.x + n / 2u + 1u, row = tid.y;
    if (col >= n || row >= n) return;
    float2 v = total[(size_t)((n - row) % n) * n + (n - col)];
    total[(size_t)row * n + col] = float2(-v.x, v.y);
}

kernel void model_set_origin(device float2* total [[buffer(0)]], constant float2& origin [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i == 0u) total[0] = origin;
}

kernel void model_wave_phase(device const float2* wave [[buffer(0)]], device float* phase [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) phase[i] = metal::precise::atan2(wave[i].y, wave[i].x);
}

// complex_wave: the object wave's amplitude, |inverse FFT of the summed spectrum| / scale (the BF count).
kernel void model_wave_amplitude(device const float2* wave [[buffer(0)]], device float* amplitude [[buffer(1)]],
    constant uint& count [[buffer(2)]], constant float& scale [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) amplitude[i] = metal::precise::sqrt(wave[i].x * wave[i].x + wave[i].y * wave[i].y) / scale;
}
