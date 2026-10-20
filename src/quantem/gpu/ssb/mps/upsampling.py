"""Bounded Metal reconstruction on finer scan grids from the retained BF FFT.

The measured Fourier aliases repeat at the native scan frequency. The probe
correction is evaluated at each expanded frequency; this is not interpolation
of a reconstructed phase image. The native aberration-search objective stays
unchanged.
"""

from functools import lru_cache

import numpy as np

from quantem.gpu.ssb._execution import checkpoint
from quantem.gpu.ssb.higher_order import CHI_SOURCE, pack_coefficients
from quantem.gpu.ssb.mps.hardware import require_mlx
from quantem.gpu.ssb.mps.kernels.phase_sums import phase_sum_from_complex
from quantem.gpu.ssb.mps.prepared import PreparedMpsSSB, ifft2_chunked
from quantem.gpu.ssb.mps.thick_sample import _THICK_FIT_BATCH_HEADER, _params

# Bound the correction and FFT working arrays independently of the BF count.
_WORK_BYTES = 512 << 20
_PENDING_CHUNKS = 4
_HEADER = ("#define SSB_PRECISE_GEOMETRY\n#define SSB_DEVICE inline\n"
           "#define SSB_COEFFICIENTS const device float*\n"
           + _THICK_FIT_BATCH_HEADER + CHI_SOURCE)


# Fine-grid redraws often evaluate a shifted probe well outside or inside the
# aperture. Keep the original geometry arithmetic at the soft edge, and avoid
# its extra square root/divisions only where conservative bounds prove 0 or 1.
_ALIAS_GEOMETRY = """
inline float4 alias_geometry(float dx, float dy, float wl, float semi,
                             float ay, float ax) {
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
    float denom = metal::precise::sqrt((dx * ay) * (dx * ay)
                                     + (dy * ax) * (dy * ax)) * inv_r;
    float edge = (denom > 1e-15f) ? ((semi - alpha) / denom + 0.5f) : 1.0f;
    return float4(alpha * alpha, cos2, sin2, metal::clamp(edge, 0.0f, 1.0f));
}
"""


# One transfer-function implementation for per-detector and coherent outputs.
# Keep precise geometry, trigonometry, and float32 evaluation on both paths.
# Pointer types retain MLX's constant/device address spaces for small BF groups.
_ALIAS_TRANSFER = """
template <uint SOURCE_N, uint G_COLS, bool HIGHER, bool INDEX32,
          typename G, typename KRow, typename KCol, typename Params,
          typename Probe, typename Coefficients>
inline float2 alias_corrected(
    uint k, float qr, float qc, float shift, float radius,
    G g, KRow k_row, KCol k_col, Params params, Probe probe,
    Coefficients coefficients, size_t source_offset, bool mirror
) {
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float c10 = params[4], c12 = params[5], c2 = params[6], s2 = params[7];
    float factor = params[8], depth = params[9];
    float kr = k_row[k], kc = k_col[k];
    float4 gk = float4(probe[k * 5], probe[k * 5 + 1], probe[k * 5 + 2], probe[k * 5 + 3]);
    // Bound the complete soft aperture before evaluating its geometry.
    float mr = qr - kr, mc = qc - kc, pr = qr + kr, pc = qc + kc;
    if (metal::min(mr * mr + mc * mc, pr * pr + pc * pc) > radius * radius) {
        return float2(0.0f);
    }
    float4 gm = alias_geometry(qr - kr, qc - kc, wl, semi, ay, ax);
    float4 gp = alias_geometry(qr + kr, qc + kc, wl, semi, ay, ax);
    if (gk.w == 0.0f || (gm.w == 0.0f && gp.w == 0.0f)) {
        return float2(0.0f);
    }
    // Start the source load before the optical arithmetic. Narrow addressing
    // is exact when the chunk fits; larger chunks retain size_t arithmetic.
    auto value = g[INDEX32
        ? size_t(k * (SOURCE_N * G_COLS) + uint(source_offset))
        : size_t(k) * SOURCE_N * G_COLS + source_offset];
    float vr = value.real, vi = mirror ? -value.imag : value.imag;
    float chi_k = probe[k * 5 + 4];
    // A closed aperture contributes exactly zero, independent of its phase.
    float a = 0.0f, b = 0.0f;
    float ca = 0.0f, cb = 0.0f, sa = 0.0f, sb = 0.0f;
    if (gm.w != 0.0f) {
        float chi_m = factor * gm.x * (c12 * (gm.y * c2 + gm.z * s2) + c10);
        if (HIGHER) chi_m = ssb_chi_full(qr - kr, qc - kc, wl, coefficients);
        a = gm.w * gk.w;
        if (SSB_ALIAS_DEPTH) {
            a = tf_sinc(0.5f * depth * (-factor * (gm.x - gk.x) - shift)) * gm.w * gk.w;
        }
        sa = metal::precise::sincos(chi_m - chi_k, ca);
    }
    if (gp.w != 0.0f) {
        float chi_p = factor * gp.x * (c12 * (gp.y * c2 + gp.z * s2) + c10);
        if (HIGHER) chi_p = ssb_chi_full(qr + kr, qc + kc, wl, coefficients);
        b = gp.w * gk.w;
        if (SSB_ALIAS_DEPTH) {
            b = tf_sinc(0.5f * depth * (factor * (gp.x - gk.x) - shift)) * gp.w * gk.w;
        }
        sb = metal::precise::sincos(chi_p - chi_k, cb);
    }
    float re = a * ca - b * cb, im = -a * sa - b * sb;
    float inv = 1.0f / metal::max(metal::precise::sqrt(re * re + im * im), 1e-8f);
    return float2((vr * re + vi * im) * inv, (vi * re - vr * im) * inv);
}
"""


@lru_cache(maxsize=1)
def _alias_probe_kernel():
    """Calculate the geometry and phase at k once for all fine-grid q."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_alias_probe",
        input_names=["k_row", "k_col", "params", "coefficients"],
        output_names=["probe"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source="""
        uint k = thread_position_in_grid.x;
        if (k >= K) return;
        float4 gk = tf_geometry(k_row[k], k_col[k], params[0], params[1], params[2], params[3]);
        for (uint i = 0; i < 4; ++i) probe[k * 5 + i] = gk[i];
        probe[k * 5 + 4] = params[8] * gk.x * (params[5] * (gk.y * params[6] + gk.z * params[7]) + params[4]);
        if (HIGHER) probe[k * 5 + 4] = ssb_chi_full(k_row[k], k_col[k], params[0], coefficients);
        """,
    )


@lru_cache(maxsize=1)
def _alias_group_bounds_kernel():
    """Bound groups of BF coordinates without changing their contributions."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_alias_group_bounds",
        input_names=["k_row", "k_col", "params"],
        output_names=["bounds"],
        compile_options={"math_mode": "safe"},
        source="""
        uint group = thread_position_in_grid.x;
        uint first = group * CHUNK;
        if (first >= K) return;
        float row_min = k_row[first], row_max = row_min;
        float col_min = k_col[first], col_max = col_min;
        for (uint k = first + 1u; k < min(first + CHUNK, uint(K)); ++k) {
            row_min = min(row_min, k_row[k]); row_max = max(row_max, k_row[k]);
            col_min = min(col_min, k_col[k]); col_max = max(col_max, k_col[k]);
        }
        float radius = (params[1] + 0.5f * max(params[2], params[3])) / params[0];
        radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
        bounds[group * 5] = row_min; bounds[group * 5 + 1] = row_max;
        bounds[group * 5 + 2] = col_min; bounds[group * 5 + 3] = col_max;
        bounds[group * 5 + 4] = radius * radius;
        """,
    )


@lru_cache(maxsize=2)
def _alias_correction_kernel(has_depth: bool = True):
    """Specialize the zero-depth model without changing optical arithmetic."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_alias_correction",
        input_names=["g", "q_row", "q_col", "k_row", "k_col", "params", "dc", "probe", "coefficients"],
        output_names=["corrected"],
        header=(f"#define SSB_ALIAS_DEPTH {int(has_depth)}\n"
                + _HEADER + _ALIAS_GEOMETRY + _ALIAS_TRANSFER),
        compile_options={"math_mode": "safe"},
        source="""
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint k = thread_position_in_grid.z;
        if (col >= N || row >= N || k >= K) return;
        size_t out = ((size_t)k * N + row) * N + col;
        if (col == 0u && row == 0u) {
            corrected[out] = dc[0];
            return;
        }
        // The scan mean is background, not measured nonzero-q structure.
        // Keep its normalization only at the true origin, never its aliases.
        if (row % SOURCE_N == 0u && col % SOURCE_N == 0u) {
            corrected[out].real = 0.0f;
            corrected[out].imag = 0.0f;
            return;
        }
        float qr = q_row[row], qc = q_col[col];
        float shift = 6.283185307179586f * (qr * params[10] + qc * params[11]);
        float radius = (params[1] + 0.5f * metal::max(params[2], params[3])) / params[0];
        radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
        // Only the source G has Hermitian symmetry. Corrected aliases do not.
        uint source_row = row % SOURCE_N, source_col = col % SOURCE_N;
        bool mirror = G_COLS != SOURCE_N && source_col > SOURCE_N / 2u;
        if (mirror) {
            source_row = source_row == 0u ? 0u : SOURCE_N - source_row;
            source_col = SOURCE_N - source_col;
        }
        size_t source_offset = size_t(source_row) * G_COLS + source_col;
        constexpr bool INDEX32 = size_t(K) * SOURCE_N * G_COLS <= (size_t(1) << 32);
        float2 value = alias_corrected<SOURCE_N, G_COLS, HIGHER, INDEX32>(
            k, qr, qc, shift, radius, g, k_row, k_col, params, probe,
            coefficients, source_offset, mirror);
        corrected[out].real = value.x;
        corrected[out].imag = value.y;
        """,
    )


@lru_cache(maxsize=2)
def _coherent_alias_kernel(has_depth: bool = True):
    """Correct detector groups and accumulate them without partial spectra.

    Zero depth omits only the identically-one thickness weights.
    Each thread retains the original BF group boundaries and addition order.
    Grid-dependent values are shared across groups, and only their accumulated
    spectrum is written to device memory.
    """
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_coherent_alias",
        input_names=["g", "q_row", "q_col", "k_row", "k_col", "params", "dc",
                     "probe", "coefficients", "bounds", "row_indices", "col_indices", "previous"],
        output_names=["total"],
        header=(f"#define SSB_ALIAS_DEPTH {int(has_depth)}\n"
                + _HEADER + _ALIAS_GEOMETRY + _ALIAS_TRANSFER),
        compile_options={"math_mode": "safe"},
        source="""
        uint col_index = thread_position_in_grid.x;
        uint row_index = thread_position_in_grid.y;
        if (col_index >= COLS || row_index >= ROWS) return;
        uint row = row_indices[row_index], col = col_indices[col_index];
        size_t out = (size_t)row_index * COLS + col_index;
        float2 result = float2(previous[out].real, previous[out].imag);
        float qr = q_row[row], qc = q_col[col];
        float shift = 6.283185307179586f * (qr * params[10] + qc * params[11]);
        float radius = (params[1] + 0.5f * metal::max(params[2], params[3])) / params[0];
        radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
        // Reuse the source address for every BF pixel at this output frequency.
        // Only the source G has Hermitian symmetry. Corrected aliases do not.
        uint source_row = row % SOURCE_N, source_col = col % SOURCE_N;
        bool mirror = G_COLS != SOURCE_N && source_col > SOURCE_N / 2u;
        if (mirror) {
            source_row = source_row == 0u ? 0u : SOURCE_N - source_row;
            source_col = SOURCE_N - source_col;
        }
        size_t source_offset = size_t(source_row) * G_COLS + source_col;
        constexpr bool INDEX32 = size_t(K) * SOURCE_N * G_COLS <= (size_t(1) << 32);
        for (uint group = 0; group < GROUPS; ++group) {
            uint start = group * CHUNK, stop = metal::min(start + CHUNK, uint(K));
            float2 sum = float2(0.0f);
            if (row == 0u && col == 0u) {
                for (uint k = start; k < stop; ++k) sum += float2(dc[0].real, dc[0].imag);
            } else if (row % SOURCE_N != 0u || col % SOURCE_N != 0u) {
                uint bound = group * 5;
                float row_min = bounds[bound], row_max = bounds[bound + 1];
                float col_min = bounds[bound + 2], col_max = bounds[bound + 3];
                float rm = max(max(row_min - qr, qr - row_max), 0.0f);
                float cm = max(max(col_min - qc, qc - col_max), 0.0f);
                float rp = max(max(row_min + qr, -qr - row_max), 0.0f);
                float cp = max(max(col_min + qc, -qc - col_max), 0.0f);
                if (min(rm * rm + cm * cm, rp * rp + cp * cp) <= bounds[bound + 4]) {
                    for (uint k = start; k < stop; ++k) {
                        sum += alias_corrected<SOURCE_N, G_COLS, HIGHER, INDEX32>(
                            k, qr, qc, shift, radius, g, k_row, k_col,
                            params, probe, coefficients, source_offset, mirror);
                    }
                }
            }
            result += sum;
        }
        total[out].real = result.x;
        total[out].imag = result.y;
        """,
    )


# --- Separable phasors for the two-term (C10, C12) coherent reconstruction ---------------------------
#
# chi(v) = v^T M v is quadratic, so the transfer phases split exactly:
#     chi(q - k) - chi(k) = Psi(q) - u(q).k,   chi(q + k) - chi(k) = Psi(q) + u(q).k,
# with Psi = q^T M q and u = 2 M q. With E = exp(-i u.k) the corrected term is
#     G conj(gamma) / |gamma| = G E w,   w = (a e^{i Psi} - b e^{-i Psi}) / |a e^{i Psi} - b e^{-i Psi}|,
# where a and b are the aperture products of the two sidebands. One open sideband reduces w to the pure
# phase e^{i Psi} or -e^{-i Psi}. On the q grid u.k = rf P_row(k) + cf P_col(k) for integer frequencies
# rf and cf, so E is the product of a row and a column phasor of each BF pixel. Those tables and Psi are
# reduced modulo 2 pi from exact float32 products of 12-bit splits of float64 coefficients, so every
# phasor carries about 1e-7 rad of error, below the float32 rounding of the per-term chi difference it
# replaces. Aperture weights and their soft edge, the BF selection and the normalisation are unchanged.

def _split_mod(values) -> np.ndarray:
    """Float64 values reduced modulo 2 pi as three float32 parts, the first two with 12 significant bits."""
    rest = np.remainder(np.asarray(values, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi
    parts = []
    for _ in range(2):
        scale = np.ldexp(1.0, 12 - np.frexp(rest)[1])
        part = np.round(rest * scale) / scale
        parts.append(part)
        rest = rest - part
    parts.append(rest)
    return np.stack(parts, axis=-1).astype(np.float32)


def _two_pi_parts() -> tuple[float, float, float]:
    """2 pi as two 14-bit parts and a float32 remainder: n times either part is exact for |n| < 512."""
    two_pi = 2.0 * np.pi
    c1 = np.round(two_pi * 2.0 ** 11) / 2.0 ** 11
    c2 = np.round((two_pi - c1) * 2.0 ** 25) / 2.0 ** 25
    return float(c1), float(c2), float(np.float32(two_pi - c1 - c2))


def _metal_float(value: float) -> str:
    return float(np.float32(value)).hex() + "f"


_TWO_PI_PARTS = _two_pi_parts()
_SEPARABLE_HEADER = f"""
// x is an exact float32 product; remove the nearest multiple of 2 pi (the first step is exact).
inline float sep_reduce_one(float x) {{
    float n = metal::rint(x * {_metal_float(1.0 / (2.0 * np.pi))});
    float r = x - n * {_metal_float(_TWO_PI_PARTS[0])};
    r = r - n * {_metal_float(_TWO_PI_PARTS[1])};
    return r - n * {_metal_float(_TWO_PI_PARTS[2])};
}}
// m (p1 + p2 + p3) modulo 2 pi for an integer |m| <= 1024 and 12-bit p1, p2, whose products are exact.
inline float sep_reduce(float m, float p1, float p2, float p3) {{
    return sep_reduce_one(m * p1) + sep_reduce_one(m * p2) + m * p3;
}}
inline float2 sep_cmul(float2 x, float2 y) {{
    return float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}}
"""


@lru_cache(maxsize=1)
def _separable_table_kernel():
    """Row and column phasors exp(-i rf P_row(k)) and exp(-i cf P_col(k)) of every stored BF pixel."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_separable_tables",
        input_names=["p_row", "p_col", "row_f", "col_f"],
        output_names=["t_row", "t_col"],
        header=_SEPARABLE_HEADER,
        compile_options={"math_mode": "safe"},
        source="""
        uint i = thread_position_in_grid.x;
        uint k = thread_position_in_grid.y;
        if (k >= K) return;
        if (i < ROWS) {
            float x = sep_reduce(row_f[i], p_row[k * 3], p_row[k * 3 + 1], p_row[k * 3 + 2]);
            float c;
            float s = metal::precise::sincos(x, c);
            t_row[k * ROWS + i].real = c;
            t_row[k * ROWS + i].imag = -s;
        }
        if (i < COLS) {
            float x = sep_reduce(col_f[i], p_col[k * 3], p_col[k * 3 + 1], p_col[k * 3 + 2]);
            float c;
            float s = metal::precise::sincos(x, c);
            t_col[k * COLS + i].real = c;
            t_col[k * COLS + i].imag = -s;
        }
        """,
    )


_SEPARABLE_SOURCE = """
        // ri and ci index the row and column tables. One grid row past ROWS computes the upper half of the
        // Nyquist row (EXTRA columns stored after the COLS main columns), whose partner is itself.
        uint gx = thread_position_in_grid.x, gy = thread_position_in_grid.y;
        uint ri, ci;
        size_t out;
        if (gy < ROWS) {
            if (gx >= COLS) return;
            ri = gy; ci = gx; out = (size_t)gy * COLS + gx;
        } else {
            if (gy > ROWS || gx >= EXTRA) return;
            ri = NYQUIST_RI; ci = COLS + gx; out = (size_t)ROWS * COLS + gx;
        }
        uint row = row_indices[ri], col = col_indices[ci];
        float2 result = float2(previous[out].real, previous[out].imag);
        float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
        float qr = q_row[row], qc = q_col[col];
        float radius = (semi + 0.5f * metal::max(ay, ax)) / wl;
        radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
        float r2_out = radius * radius;
        // Weights 0 and 1 where a margin proves the unchanged soft-edge geometry returns them; else evaluate it.
        float r2_zero = r2_out * (1.0f + 32.0f * 1.1920928955078125e-7f);
        float inner = (semi - 0.5f * metal::max(ay, ax)) * (1.0f - 8.0f * 1.1920928955078125e-7f);
        float inner_q = inner / wl;
        float r2_one = inner > 0.0f ? inner_q * inner_q * (1.0f - 64.0f * 1.1920928955078125e-7f) : -1.0f;
        // Inside this radius the soft edge (semi - alpha) / denom + 1/2 is positive for any edge direction,
        // with a margin far above its float32 rounding.
        float positive_q = (semi + 0.5f * metal::min(ay, ax)) / wl;
        float r2_pos = positive_q * positive_q * (1.0f - 64.0f * 1.1920928955078125e-7f);
        // Only the source G has Hermitian symmetry. Corrected aliases do not.
        uint source_row = row % SOURCE_N, source_col = col % SOURCE_N;
        bool mirror = G_COLS != SOURCE_N && source_col > SOURCE_N / 2u;
        if (mirror) {
            source_row = source_row == 0u ? 0u : SOURCE_N - source_row;
            source_col = SOURCE_N - source_col;
        }
        size_t source_offset = size_t(source_row) * G_COLS + source_col;
        constexpr bool INDEX32 = size_t(K) * SOURCE_N * G_COLS <= (size_t(1) << 32);
        if (row == 0u && col == 0u) {
            for (uint group = 0; group < GROUPS; ++group) {
                uint start = group * CHUNK, stop = metal::min(start + CHUNK, uint(K));
                float2 sum = float2(0.0f);
                for (uint k = start; k < stop; ++k) sum += float2(dc[0].real, dc[0].imag);
                result += sum;
            }
        } else if (row % SOURCE_N != 0u || col % SOURCE_N != 0u) {
            // The scan mean is background, not measured nonzero-q structure: its aliases stay zero.
            float2 psi_r = float2(psi_row[ri].real, psi_row[ri].imag);
            float2 psi_c = float2(psi_col[ci].real, psi_col[ci].imag);
            float xb = sep_reduce(col_f[ci], psi_cross[ri * 3], psi_cross[ri * 3 + 1], psi_cross[ri * 3 + 2]);
            float cb;
            float sb = metal::precise::sincos(xb, cb);
            float2 epsi = sep_cmul(sep_cmul(psi_r, psi_c), float2(cb, sb));
            float2 sum_a = float2(0.0f), sum_b = float2(0.0f), sum_ab = float2(0.0f), sum_i = float2(0.0f);
            // With a = b = A(k) >= 1e-3 and |sin Psi| >= 1e-4, |w| never reaches the 1e-8 floor.
            bool both_ok = metal::abs(epsi.y) >= 1e-4f;
            // Group-level margins cover the float32 rounding of box distances against per-pixel distances.
            float r2_pos_g = r2_pos * (1.0f - 64.0f * 1.1920928955078125e-7f);
            float r2_one_g = r2_one * (1.0f - 64.0f * 1.1920928955078125e-7f);
            float r2_zero_g = r2_zero * (1.0f + 64.0f * 1.1920928955078125e-7f);
            for (uint group = 0; group < GROUPS; ++group) {
                uint start = group * CHUNK, stop = metal::min(start + CHUNK, uint(K));
                float4 box = ((const device float4*)bounds)[2 * group];  // row min/max, col min/max
                float min_aperture = bounds[8 * group + 4];
                float rm = max(max(box.x - qr, qr - box.y), 0.0f);
                float cm = max(max(box.z - qc, qc - box.w), 0.0f);
                float rp = max(max(box.x + qr, -qr - box.y), 0.0f);
                float cp = max(max(box.z + qc, -qc - box.w), 0.0f);
                float near_m = rm * rm + cm * cm, near_p = rp * rp + cp * cp;
                if (min(near_m, near_p) > r2_out) continue;
                float rmf = max(metal::abs(qr - box.x), metal::abs(qr - box.y));
                float cmf = max(metal::abs(qc - box.z), metal::abs(qc - box.w));
                float rpf = max(metal::abs(qr + box.x), metal::abs(qr + box.y));
                float cpf = max(metal::abs(qc + box.z), metal::abs(qc + box.w));
                float far_m = rmf * rmf + cmf * cmf, far_p = rpf * rpf + cpf * cpf;
                // Every pixel of the group in the same single-sideband or flat double-sideband class:
                // the per-pixel classification below would give each the same accumulator and no weight.
                bool pure_m = far_m < r2_pos_g && near_p > r2_zero_g;
                bool pure_p = far_p < r2_pos_g && near_m > r2_zero_g;
                bool pure_i = far_m < r2_one_g && far_p < r2_one_g && both_ok;
                if ((pure_m || pure_p || pure_i) && min_aperture >= 1e-3f) {
                    // Independent loads of PURE_UNROLL pixels per step keep several requests in flight.
                    float2 acc = float2(0.0f);
                    uint k = start;
                    for (; k + PURE_UNROLL <= stop; k += PURE_UNROLL) {
                        float2 gv[PURE_UNROLL], tr[PURE_UNROLL], tc[PURE_UNROLL];
                        for (uint j = 0; j < PURE_UNROLL; ++j) {
                            auto value = g[INDEX32
                                ? size_t((k + j) * (SOURCE_N * G_COLS) + uint(source_offset))
                                : size_t(k + j) * SOURCE_N * G_COLS + source_offset];
                            gv[j] = float2(value.real, mirror ? -value.imag : value.imag);
                            tr[j] = float2(t_row[(k + j) * ROWS + ri].real, t_row[(k + j) * ROWS + ri].imag);
                            tc[j] = float2(t_col[(k + j) * COLS_ALL + ci].real, t_col[(k + j) * COLS_ALL + ci].imag);
                        }
                        for (uint j = 0; j < PURE_UNROLL; ++j) acc += sep_cmul(gv[j], sep_cmul(tr[j], tc[j]));
                    }
                    for (; k < stop; ++k) {
                        auto value = g[INDEX32
                            ? size_t(k * (SOURCE_N * G_COLS) + uint(source_offset))
                            : size_t(k) * SOURCE_N * G_COLS + source_offset];
                        float2 gv = float2(value.real, mirror ? -value.imag : value.imag);
                        float2 tr = float2(t_row[k * ROWS + ri].real, t_row[k * ROWS + ri].imag);
                        float2 tc = float2(t_col[k * COLS_ALL + ci].real, t_col[k * COLS_ALL + ci].imag);
                        acc += sep_cmul(gv, sep_cmul(tr, tc));
                    }
                    if (pure_m) sum_a += acc; else if (pure_p) sum_b += acc; else sum_i += acc;
                    continue;
                }
                float2 ga = float2(0.0f), gb = float2(0.0f), gab = float2(0.0f), gi = float2(0.0f);
                // Issue the loads of UNROLL pixels before using them (indices clamped inside the group).
                for (uint k0 = start; k0 < stop; k0 += UNROLL) {
                    float4 kd[UNROLL];
                    float2 gv[UNROLL], tr[UNROLL], tc[UNROLL];
                    for (uint j = 0; j < UNROLL; ++j) {
                        uint k = metal::min(k0 + j, stop - 1u);
                        kd[j] = ((const device float4*)k_data)[k];  // (k_row, k_col, aperture at k, 0)
                        auto value = g[INDEX32
                            ? size_t(k * (SOURCE_N * G_COLS) + uint(source_offset))
                            : size_t(k) * SOURCE_N * G_COLS + source_offset];
                        gv[j] = float2(value.real, mirror ? -value.imag : value.imag);
                        tr[j] = float2(t_row[k * ROWS + ri].real, t_row[k * ROWS + ri].imag);
                        tc[j] = float2(t_col[k * COLS_ALL + ci].real, t_col[k * COLS_ALL + ci].imag);
                    }
                    for (uint j = 0; j < UNROLL; ++j) {
                        float mr = qr - kd[j].x, mc = qc - kd[j].y, pr = qr + kd[j].x, pc = qc + kd[j].y;
                        float rm2 = mr * mr + mc * mc, rp2 = pr * pr + pc * pc;
                        if (k0 + j >= stop || metal::min(rm2, rp2) > r2_out) continue;
                        float2 term = sep_cmul(gv[j], sep_cmul(tr[j], tc[j]));
                        // One open sideband with a weight proven positive: its weight cancels in the normalisation.
                        // Both sidebands inside the flat aperture: a = b, so w is the per-q constant i sign(sin Psi).
                        bool closed_m = rm2 > r2_zero, closed_p = rp2 > r2_zero;
                        bool single_m = rm2 < r2_pos && closed_p, single_p = rp2 < r2_pos && closed_m;
                        bool both_one = rm2 < r2_one && rp2 < r2_one && both_ok;
                        if ((single_m || single_p || both_one) && kd[j].z >= 1e-3f) {
                            ga += single_m ? term : float2(0.0f);
                            gb += single_p ? term : float2(0.0f);
                            gi += both_one ? term : float2(0.0f);
                            continue;
                        }
                        float wa = closed_m ? 0.0f
                            : (rm2 < r2_one ? 1.0f : alias_geometry(mr, mc, wl, semi, ay, ax).w);
                        float wb = closed_p ? 0.0f
                            : (rp2 < r2_one ? 1.0f : alias_geometry(pr, pc, wl, semi, ay, ax).w);
                        float gkw = kd[j].z;
                        if (gkw == 0.0f || (wa == 0.0f && wb == 0.0f)) continue;
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
                }
                sum_a += ga; sum_b += gb; sum_ab += gab; sum_i += gi;
            }
            float2 w_one = float2(0.0f, epsi.y > 0.0f ? 1.0f : -1.0f);
            result += sep_cmul(epsi, sum_a) - sep_cmul(float2(epsi.x, -epsi.y), sum_b) + sum_ab
                      + sep_cmul(w_one, sum_i);
        }
        total[out].real = result.x;
        total[out].imag = result.y;
"""


_PAIRED_HEADER = """
// One output frequency of the separable sum: its row-table index, frequency, coherent factor and sums.
struct SepAlias {
    uint ri;
    float qr;
    float2 epsi;
    bool both_ok;
    float2 sum_a, sum_b, sum_ab, sum_i;
};

// 0: no pixel of the group reaches q; 1: every pixel in one flat class (pure_kind 0 a, 1 b, 2 both); 2: per pixel.
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

// The per-pixel classification of the separable kernel's mixed groups, for one pixel and one output.
inline void sep_mixed_term(float qr, float qc, float4 kd, float2 term, float2 epsi, bool both_ok,
                           float r2_out, float r2_zero, float r2_pos, float r2_one,
                           float wl, float semi, float ay, float ax,
                           thread float2 &ga, thread float2 &gb, thread float2 &gab, thread float2 &gi) {
    float mr = qr - kd.x, mc = qc - kd.y, pr = qr + kd.x, pc = qc + kd.y;
    float rm2 = mr * mr + mc * mc, rp2 = pr * pr + pc * pc;
    if (metal::min(rm2, rp2) > r2_out) return;
    bool closed_m = rm2 > r2_zero, closed_p = rp2 > r2_zero;
    bool single_m = rm2 < r2_pos && closed_p, single_p = rp2 < r2_pos && closed_m;
    bool both_one = rm2 < r2_one && rp2 < r2_one && both_ok;
    if ((single_m || single_p || both_one) && kd.z >= 1e-3f) {
        ga += single_m ? term : float2(0.0f);
        gb += single_p ? term : float2(0.0f);
        gi += both_one ? term : float2(0.0f);
        return;
    }
    float wa = closed_m ? 0.0f : (rm2 < r2_one ? 1.0f : alias_geometry(mr, mc, wl, semi, ay, ax).w);
    float wb = closed_p ? 0.0f : (rp2 < r2_one ? 1.0f : alias_geometry(pr, pc, wl, semi, ay, ax).w);
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
"""

_PAIRED_SOURCE = """
        // Fine-grid rows that share one source row read the same G values; each thread computes up to two of
        // them (pairs[gy]), every output with the separable kernel's own arithmetic and summation order.
        // One grid row past PAIRS computes the upper half of the Nyquist row, as in the single-output kernel.
        uint gx = thread_position_in_grid.x, gy = thread_position_in_grid.y;
        uint ci, ri_a, ri_b;
        size_t out_a, out_b = 0;
        if (gy < PAIRS) {
            if (gx >= COLS) return;
            ci = gx; ri_a = pairs[2 * gy]; ri_b = pairs[2 * gy + 1];
            out_a = (size_t)ri_a * COLS + gx;
            if (ri_b != 0xFFFFFFFFu) out_b = (size_t)ri_b * COLS + gx;
        } else {
            if (gy > PAIRS || gx >= EXTRA) return;
            ri_a = NYQUIST_RI; ri_b = 0xFFFFFFFFu; ci = COLS + gx; out_a = (size_t)ROWS * COLS + gx;
        }
        bool has_b = ri_b != 0xFFFFFFFFu;
        uint col = col_indices[ci];
        uint row_a = row_indices[ri_a], row_b = has_b ? row_indices[ri_b] : 0u;
        float2 result_a = float2(previous[out_a].real, previous[out_a].imag);
        float2 result_b = has_b ? float2(previous[out_b].real, previous[out_b].imag) : float2(0.0f);
        float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
        float qc = q_col[col];
        float radius = (semi + 0.5f * metal::max(ay, ax)) / wl;
        radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
        float r2_out = radius * radius;
        float r2_zero = r2_out * (1.0f + 32.0f * 1.1920928955078125e-7f);
        float inner = (semi - 0.5f * metal::max(ay, ax)) * (1.0f - 8.0f * 1.1920928955078125e-7f);
        float inner_q = inner / wl;
        float r2_one = inner > 0.0f ? inner_q * inner_q * (1.0f - 64.0f * 1.1920928955078125e-7f) : -1.0f;
        float positive_q = (semi + 0.5f * metal::min(ay, ax)) / wl;
        float r2_pos = positive_q * positive_q * (1.0f - 64.0f * 1.1920928955078125e-7f);
        // Both rows share the source row, so the G offset and its Hermitian mirror are common.
        uint source_row = row_a % SOURCE_N, source_col = col % SOURCE_N;
        bool mirror = G_COLS != SOURCE_N && source_col > SOURCE_N / 2u;
        if (mirror) {
            source_row = source_row == 0u ? 0u : SOURCE_N - source_row;
            source_col = SOURCE_N - source_col;
        }
        size_t source_offset = size_t(source_row) * G_COLS + source_col;
        constexpr bool INDEX32 = size_t(K) * SOURCE_N * G_COLS <= (size_t(1) << 32);
        // The DC output sums the common real DC term; the aliases of the scan mean stay zero.
        if (row_a == 0u && col == 0u) {
            for (uint group = 0; group < GROUPS; ++group) {
                uint start = group * CHUNK, stop = metal::min(start + CHUNK, uint(K));
                float2 sum = float2(0.0f);
                for (uint k = start; k < stop; ++k) sum += float2(dc[0].real, dc[0].imag);
                result_a += sum;
            }
        }
        bool active_a = row_a % SOURCE_N != 0u || col % SOURCE_N != 0u;
        bool active_b = has_b && (row_b % SOURCE_N != 0u || col % SOURCE_N != 0u);
        SepAlias A, B;
        float2 psi_c = float2(psi_col[ci].real, psi_col[ci].imag);
        for (uint which = 0; which < 2; ++which) {
            thread SepAlias &alias = which == 0 ? A : B;
            uint ri = which == 0 ? ri_a : (has_b ? ri_b : ri_a);
            alias.ri = ri;
            alias.qr = q_row[row_indices[ri]];
            float2 psi_r = float2(psi_row[ri].real, psi_row[ri].imag);
            float xb = sep_reduce(col_f[ci], psi_cross[ri * 3], psi_cross[ri * 3 + 1], psi_cross[ri * 3 + 2]);
            float cb;
            float sb = metal::precise::sincos(xb, cb);
            alias.epsi = sep_cmul(sep_cmul(psi_r, psi_c), float2(cb, sb));
            alias.both_ok = metal::abs(alias.epsi.y) >= 1e-4f;
            alias.sum_a = alias.sum_b = alias.sum_ab = alias.sum_i = float2(0.0f);
        }
        float r2_pos_g = r2_pos * (1.0f - 64.0f * 1.1920928955078125e-7f);
        float r2_one_g = r2_one * (1.0f - 64.0f * 1.1920928955078125e-7f);
        float r2_zero_g = r2_zero * (1.0f + 64.0f * 1.1920928955078125e-7f);
        if (active_a || active_b) {
            for (uint group = 0; group < GROUPS; ++group) {
                uint start = group * CHUNK, stop = metal::min(start + CHUNK, uint(K));
                float4 box = ((const device float4*)bounds)[2 * group];  // row min/max, col min/max
                float min_aperture = bounds[8 * group + 4];
                int kind_a = 0, kind_b = 0;
                int class_a = active_a ? sep_group_class(A.qr, qc, box, min_aperture, r2_out, r2_pos_g, r2_one_g,
                                                         r2_zero_g, A.both_ok, kind_a) : 0;
                int class_b = active_b ? sep_group_class(B.qr, qc, box, min_aperture, r2_out, r2_pos_g, r2_one_g,
                                                         r2_zero_g, B.both_ok, kind_b) : 0;
                if (class_a == 0 && class_b == 0) continue;
                bool per_pixel = class_a == 2 || class_b == 2;
                float2 acc_a = float2(0.0f), ga_a = float2(0.0f), gb_a = float2(0.0f), gab_a = float2(0.0f),
                       gi_a = float2(0.0f);
                float2 acc_b = float2(0.0f), ga_b = float2(0.0f), gb_b = float2(0.0f), gab_b = float2(0.0f),
                       gi_b = float2(0.0f);
                for (uint k = start; k < stop; ++k) {
                    auto value = g[INDEX32
                        ? size_t(k * (SOURCE_N * G_COLS) + uint(source_offset))
                        : size_t(k) * SOURCE_N * G_COLS + source_offset];
                    float2 gv = float2(value.real, mirror ? -value.imag : value.imag);
                    float2 tc = float2(t_col[k * COLS_ALL + ci].real, t_col[k * COLS_ALL + ci].imag);
                    float4 kd = per_pixel ? ((const device float4*)k_data)[k] : float4(0.0f);
                    if (class_a != 0) {
                        float2 tr = float2(t_row[k * ROWS + A.ri].real, t_row[k * ROWS + A.ri].imag);
                        float2 term = sep_cmul(gv, sep_cmul(tr, tc));
                        if (class_a == 1) acc_a += term;
                        else sep_mixed_term(A.qr, qc, kd, term, A.epsi, A.both_ok, r2_out, r2_zero, r2_pos, r2_one,
                                            wl, semi, ay, ax, ga_a, gb_a, gab_a, gi_a);
                    }
                    if (class_b != 0) {
                        float2 tr = float2(t_row[k * ROWS + B.ri].real, t_row[k * ROWS + B.ri].imag);
                        float2 term = sep_cmul(gv, sep_cmul(tr, tc));
                        if (class_b == 1) acc_b += term;
                        else sep_mixed_term(B.qr, qc, kd, term, B.epsi, B.both_ok, r2_out, r2_zero, r2_pos, r2_one,
                                            wl, semi, ay, ax, ga_b, gb_b, gab_b, gi_b);
                    }
                }
                if (class_a == 1) {
                    if (kind_a == 0) A.sum_a += acc_a; else if (kind_a == 1) A.sum_b += acc_a; else A.sum_i += acc_a;
                } else if (class_a == 2) {
                    A.sum_a += ga_a; A.sum_b += gb_a; A.sum_ab += gab_a; A.sum_i += gi_a;
                }
                if (class_b == 1) {
                    if (kind_b == 0) B.sum_a += acc_b; else if (kind_b == 1) B.sum_b += acc_b; else B.sum_i += acc_b;
                } else if (class_b == 2) {
                    B.sum_a += ga_b; B.sum_b += gb_b; B.sum_ab += gab_b; B.sum_i += gi_b;
                }
            }
        }
        if (active_a) {
            float2 w_one = float2(0.0f, A.epsi.y > 0.0f ? 1.0f : -1.0f);
            result_a += sep_cmul(A.epsi, A.sum_a) - sep_cmul(float2(A.epsi.x, -A.epsi.y), A.sum_b) + A.sum_ab
                        + sep_cmul(w_one, A.sum_i);
        }
        if (active_b) {
            float2 w_one = float2(0.0f, B.epsi.y > 0.0f ? 1.0f : -1.0f);
            result_b += sep_cmul(B.epsi, B.sum_a) - sep_cmul(float2(B.epsi.x, -B.epsi.y), B.sum_b) + B.sum_ab
                        + sep_cmul(w_one, B.sum_i);
        }
        total[out_a].real = result_a.x;
        total[out_a].imag = result_a.y;
        if (has_b) {
            total[out_b].real = result_b.x;
            total[out_b].imag = result_b.y;
        }
"""


@lru_cache(maxsize=1)
def _paired_coherent_kernel():
    """The separable kernel computing the two fine-grid rows of one source row per thread, from one G read."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_separable_coherent_paired",
        input_names=["g", "q_row", "q_col", "k_data", "params", "dc",
                     "bounds", "row_indices", "col_indices", "t_row", "t_col",
                     "psi_row", "psi_col", "psi_cross", "col_f", "pairs", "previous"],
        output_names=["total"],
        header=_HEADER + _ALIAS_GEOMETRY + _SEPARABLE_HEADER + _PAIRED_HEADER,
        compile_options={"math_mode": "safe"},
        source=_PAIRED_SOURCE,
    )


@lru_cache(maxsize=1)
def _separable_coherent_kernel():
    """Accumulate every BF contribution of the C10/C12 transfer in separable form."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_separable_coherent",
        input_names=["g", "q_row", "q_col", "k_data", "params", "dc",
                     "bounds", "row_indices", "col_indices", "t_row", "t_col",
                     "psi_row", "psi_col", "psi_cross", "col_f", "previous"],
        output_names=["total"],
        header=_HEADER + _ALIAS_GEOMETRY + _SEPARABLE_HEADER,
        compile_options={"math_mode": "safe"},
        source=_SEPARABLE_SOURCE,
    )


# The two-term coherent reconstruction uses the separable kernel; False selects the per-term reference.
_SEPARABLE = True
# A 2x preview computes the two fine rows of each source row in one thread; False gives one row per thread.
_PAIRED = True
# BF pixels per culling group and per launch, and the threadgroup of the separable kernel.
_SEPARABLE_CHUNK = 16
_SEPARABLE_LAUNCH_BF = 8192
_SEPARABLE_THREADGROUP = (32, 4, 1)
_SEPARABLE_UNROLL = 1
_SEPARABLE_PURE_UNROLL = 4


def _separable_geometry(prepared: PreparedMpsSSB, n: int, params) -> dict:
    """Aberration-independent state of the separable path for one output grid, cached on ``prepared``."""
    mx = prepared.mx
    cached = prepared.separable_geometry
    if cached is not None and cached[0] == n:
        return cached[1]
    stored = int(prepared.g_qk.shape[0])
    columns = n // 2 + 1
    row_indices, col_indices = _coherent_support(prepared, n, columns)
    frequencies = np.fft.fftfreq(n) * n
    # The Nyquist row's anti-Hermitian partner is itself, so its upper half is computed, not mirrored,
    # by one extra grid row of the same launch (a separate small launch would add a serial tail).
    nyquist = n // 2
    upper = np.zeros(0, dtype=np.uint32)
    if nyquist in set(row_indices.tolist()):
        upper = np.arange(columns, n, dtype=np.uint32)
        upper = upper[np.abs(frequencies[upper]) <= np.abs(frequencies[col_indices]).max()]
        if upper.size and upper[-1] != n - 1 or np.any(np.diff(upper) != 1):
            raise RuntimeError("The Nyquist row support is expected to be one run ending at the last column.")
        if upper.size > col_indices.size:
            raise RuntimeError("The Nyquist row's upper half is expected to fit one grid row.")
    all_cols = np.concatenate((col_indices, upper)).astype(np.uint32)
    # The probe aperture at k depends only on the geometry; its chi(k) cancels in the separable form.
    probe = _alias_probe_kernel()(
        inputs=[prepared.kx, prepared.ky, params, mx.array(np.zeros((14, 2), np.float32))],
        template=[("K", stored), ("HIGHER", False)],
        grid=(stored, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(stored, 5)], output_dtypes=[mx.float32],
    )[0]
    k_data = mx.contiguous(mx.stack((prepared.kx, prepared.ky, probe[:, 3], mx.zeros_like(prepared.kx)), axis=1))
    mx.eval(k_data)
    # Per group of BF pixels: row min/max, col min/max of k, and the smallest aperture at k.
    group_count = -(-stored // _SEPARABLE_CHUNK)
    padded = group_count * _SEPARABLE_CHUNK - stored
    values = np.asarray(k_data)
    edge = np.concatenate((values, np.repeat(values[-1:], padded, axis=0))).reshape(group_count, _SEPARABLE_CHUNK, 4)
    bounds_np = np.zeros((group_count, 8), np.float32)
    bounds_np[:, 0], bounds_np[:, 1] = edge[..., 0].min(axis=1), edge[..., 0].max(axis=1)
    bounds_np[:, 2], bounds_np[:, 3] = edge[..., 1].min(axis=1), edge[..., 1].max(axis=1)
    bounds_np[:, 4] = edge[..., 2].min(axis=1)
    bounds = mx.array(bounds_np)
    geometry = {
        "columns": columns, "bounds": bounds, "k_data": k_data,
        "kx": prepared.kx_np.astype(np.float64), "ky": prepared.ky_np.astype(np.float64),
        "steps": (float(prepared.q_row[1].item()), float(prepared.q_col[1].item())),
        "rows": row_indices, "cols": col_indices, "upper": upper,
        "nyquist_ri": int(np.flatnonzero(row_indices == nyquist)[0]) if upper.size else 0,
        "rows_m": mx.array(row_indices), "cols_m": mx.array(all_cols),
        "row_f": frequencies[row_indices], "col_f": frequencies[all_cols],
        "pairs_m": mx.array(_alias_row_pairs(row_indices, int(prepared.scan_shape[0]))),
        "row_f_m": mx.array(frequencies[row_indices].astype(np.float32)),
        "col_f_m": mx.array(frequencies[all_cols].astype(np.float32)),
    }
    prepared.separable_geometry = (n, geometry)
    return geometry


def _alias_row_pairs(row_indices: np.ndarray, source_n: int) -> np.ndarray:
    """Row-table indices in pairs that share a source row (index mod ``source_n``); 0xffffffff marks no partner."""
    by_source: dict[int, list[int]] = {}
    for index, row in enumerate(row_indices.tolist()):
        by_source.setdefault(row % source_n, []).append(index)
    pairs = []
    for members in by_source.values():
        for first in range(0, len(members), 2):
            pair = members[first:first + 2]
            pairs.append(pair + [0xFFFFFFFF] * (2 - len(pair)))
    return np.asarray(pairs, dtype=np.uint32).reshape(-1, 2)


def _separable_spectrum(prepared: PreparedMpsSSB, qr, qc, params, params_np: np.ndarray, dc, n: int):
    """The complete corrected C10/C12 spectrum (n x n), with every selected BF contribution."""
    mx = prepared.mx
    geometry = _separable_geometry(prepared, n, params)
    stored = int(prepared.g_qk.shape[0])
    source_n = int(prepared.scan_shape[0])
    # chi(v) = factor (|v| wl)^2 (c10 + c12 cos 2(theta - phi12)) = v^T M v, from the kernel's float32 inputs
    wl, c10, c12, c2, s2, factor = (float(params_np[i]) for i in (0, 4, 5, 6, 7, 8))
    scale = factor * wl * wl
    m00, m01, m11 = scale * (c10 + c12 * c2), scale * c12 * s2, scale * (c10 - c12 * c2)
    step_r, step_c = geometry["steps"]
    kx, ky = geometry["kx"], geometry["ky"]
    p_row = mx.array(_split_mod(2.0 * step_r * (m00 * kx + m01 * ky)))
    p_col = mx.array(_split_mod(2.0 * step_c * (m01 * kx + m11 * ky)))
    psi_a, psi_b, psi_c = m00 * step_r ** 2, 2.0 * m01 * step_r * step_c, m11 * step_c ** 2
    rows, cols, extra = len(geometry["rows"]), len(geometry["cols"]), len(geometry["upper"])
    row_f, col_f = geometry["row_f"], geometry["col_f"]
    psi_row = mx.array(np.exp(1j * np.remainder(psi_a * row_f ** 2, 2.0 * np.pi)).astype(np.complex64))
    psi_col = mx.array(np.exp(1j * np.remainder(psi_c * col_f ** 2, 2.0 * np.pi)).astype(np.complex64))
    psi_cross = mx.array(_split_mod(psi_b * row_f))
    t_row, t_col = _separable_table_kernel()(
        inputs=[p_row, p_col, geometry["row_f_m"], geometry["col_f_m"]],
        template=[("K", stored), ("ROWS", rows), ("COLS", cols + extra)],
        grid=(max(rows, cols + extra), stored, 1), threadgroup=(256, 1, 1),
        output_shapes=[(stored, rows), (stored, cols + extra)], output_dtypes=[mx.complex64, mx.complex64],
    )
    grid_rows = rows + (1 if extra else 0)
    total = mx.zeros((grid_rows, cols), dtype=mx.complex64)
    launches = -(-stored // _SEPARABLE_LAUNCH_BF)
    per_launch = -(-stored // launches)
    per_launch = -(-per_launch // _SEPARABLE_CHUNK) * _SEPARABLE_CHUNK
    # The two fine rows of one source row read the same G values; at 2x one thread computes both and reads
    # each G value once for the BF pixels that reach either row.
    paired = _PAIRED and n == 2 * source_n
    pairs = int(geometry["pairs_m"].shape[0])
    for start in range(0, stored, per_launch):
        checkpoint()
        stop = min(start + per_launch, stored)
        groups = -(-(stop - start) // _SEPARABLE_CHUNK)
        first = start // _SEPARABLE_CHUNK
        inputs = [prepared.g_qk[start:stop], qr, qc, geometry["k_data"][start:stop],
                  params, dc, geometry["bounds"][first:first + groups].reshape(-1),
                  geometry["rows_m"], geometry["cols_m"], t_row[start:stop], t_col[start:stop],
                  psi_row, psi_col, psi_cross, geometry["col_f_m"]]
        template = [("ROWS", rows), ("COLS", cols), ("COLS_ALL", cols + extra), ("EXTRA", extra),
                    ("NYQUIST_RI", geometry["nyquist_ri"]), ("SOURCE_N", source_n),
                    ("G_COLS", int(prepared.g_qk.shape[-1])),
                    ("K", stop - start), ("CHUNK", _SEPARABLE_CHUNK), ("GROUPS", groups)]
        if paired:
            total = _paired_coherent_kernel()(
                inputs=[*inputs, geometry["pairs_m"], total], template=[*template, ("PAIRS", pairs)],
                grid=(cols, pairs + (1 if extra else 0), 1), threadgroup=_SEPARABLE_THREADGROUP,
                output_shapes=[(grid_rows, cols)], output_dtypes=[mx.complex64],
            )[0]
            continue
        total = _separable_coherent_kernel()(
            inputs=[*inputs, total],
            template=[*template, ("UNROLL", _SEPARABLE_UNROLL), ("PURE_UNROLL", _SEPARABLE_PURE_UNROLL)],
            grid=(cols, grid_rows, 1), threadgroup=_SEPARABLE_THREADGROUP,
            output_shapes=[(grid_rows, cols)], output_dtypes=[mx.complex64],
        )[0]
    upper = total[rows:, :extra] if extra else None
    total = total[:rows] if extra else total
    # Restore the zero rows and columns outside the aperture support.
    columns = geometry["columns"]
    for axis, indices, count in ((0, geometry["rows"], n), (1, geometry["cols"], columns)):
        if len(indices) == count:
            continue
        gaps = np.flatnonzero(np.diff(indices) > 1)
        split = int(gaps[0] + 1) if len(gaps) else len(indices)
        shape = list(total.shape)
        shape[axis] = count - len(indices)
        before, after = mx.split(total, [split], axis=axis)
        total = mx.concatenate((before, mx.zeros(shape, dtype=mx.complex64), after), axis=axis)
    # Compacted inactive BF columns still carry the common real DC term.
    total[0, 0] = total[0, 0] + (prepared.num_bright_field_pixels - stored) * dc[0]
    # Paired corrected frequencies of the even C10/C12 probe are anti-Hermitian, O(-q) = -conj(O(q)),
    # except the Nyquist row, whose partner is itself and was computed above.
    mirror_rows = mx.array((-np.arange(n)) % n)
    negative = -mx.conj(total[mirror_rows, 1:n // 2][:, ::-1])
    if upper is not None:
        # The supported upper columns of the Nyquist row are one contiguous run ending at column n - 1.
        first = int(geometry["upper"][0])
        if first > columns:
            upper = mx.concatenate((mx.zeros((1, first - columns), dtype=mx.complex64), upper), axis=1)
        negative = mx.concatenate((negative[:n // 2], upper, negative[n // 2 + 1:]), axis=0)
    return mx.concatenate((total, negative), axis=1)


def _coherent_support(
    prepared: PreparedMpsSSB, n: int, columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain every nonzero aperture frequency, with an outward grid margin."""
    radius = (
        prepared.semiangle_rad
        + 0.5 * max(prepared.ang_y_rad, prepared.ang_x_rad)
    ) / prepared.wavelength
    radius *= 1.0 + 8.0 * np.finfo(np.float32).eps
    frequency = np.abs(np.fft.fftfreq(n) * n)
    indices = []
    for coordinates, step, count in (
        (prepared.kx_np, prepared.q_row[1], n),
        (prepared.ky_np, prepared.q_col[1], columns),
    ):
        support = float(np.max(np.abs(coordinates))) + radius
        # One extra grid point includes float32 frequency and geometry rounding.
        limit = np.ceil(support / abs(float(step))) + 1
        indices.append(np.flatnonzero(frequency[:count] <= limit).astype(np.uint32))
    return indices[0], indices[1]


def _coherent_spectrum(
    prepared: PreparedMpsSSB, qr, qc, params, dc, probe, coefficients,
    higher: bool, n: int, chunk: int, columns: int, has_depth: bool = True,
):
    """Accumulate every BF contribution in its original group and sum order."""
    mx = prepared.mx
    stored = int(prepared.g_qk.shape[0])
    source_n = int(prepared.scan_shape[0])
    # Bounds and support depend on the grid and rotated BF geometry, not on
    # aberrations, tilt or depth. Keep one latest set with the frequency grid.
    cached = prepared.upsampled_geometry
    geometry_key = (columns, chunk)
    if cached is not None and cached[0] == n and cached[3] == geometry_key:
        row_indices, col_indices, row_indices_m, col_indices_m, bounds = cached[4]
    else:
        row_indices, col_indices = _coherent_support(prepared, n, columns)
        row_indices_m = mx.array(row_indices)
        col_indices_m = mx.array(col_indices)
        group_count = (stored + chunk - 1) // chunk
        bounds = _alias_group_bounds_kernel()(
            inputs=[prepared.kx, prepared.ky, params],
            template=[("K", stored), ("CHUNK", chunk)],
            grid=(group_count, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(group_count, 5)], output_dtypes=[mx.float32],
        )[0]
        geometry = (row_indices, col_indices, row_indices_m, col_indices_m, bounds)
        prepared.upsampled_geometry = (n, qr, qc, geometry_key, geometry)
    rows, cols = len(row_indices), len(col_indices)
    total = mx.zeros((rows, cols), dtype=mx.complex64)
    # Preserve BF chunk boundaries and sequential float32 accumulation order.
    # Reuse each frequency and tilt term across groups without partial spectra.
    groups_per_batch = 64
    for start in range(0, stored, groups_per_batch * chunk):
        batch = start // (groups_per_batch * chunk)
        if batch % _PENDING_CHUNKS == 0:
            checkpoint()
        stop = min(start + groups_per_batch * chunk, stored)
        groups = (stop - start + chunk - 1) // chunk
        total = _coherent_alias_kernel(has_depth)(
            inputs=[prepared.g_qk[start:stop], qr, qc, prepared.kx[start:stop],
                    prepared.ky[start:stop], params, dc, probe[start:stop], coefficients,
                    bounds[start // chunk:start // chunk + groups],
                    row_indices_m, col_indices_m, total],
            template=[("N", n), ("ROWS", rows), ("COLS", cols), ("SOURCE_N", source_n),
                      ("G_COLS", int(prepared.g_qk.shape[-1])),
                      ("K", stop - start), ("HIGHER", higher), ("CHUNK", chunk), ("GROUPS", groups)],
            grid=(cols, rows, 1), threadgroup=(16, 8, 1),
            output_shapes=[(rows, cols)], output_dtypes=[mx.complex64],
        )[0]
        # Queue a bounded set of dependent batches without waiting on the CPU
        # after each dispatch. Synchronize before the next cancellation check.
        mx.async_eval(total)
        if (batch + 1) % _PENDING_CHUNKS == 0 or stop == stored:
            mx.eval(total)
    if rows == n and cols == columns:
        return total
    # Restore the full Fourier grid before the unchanged inverse FFT.
    for axis, indices, count in ((0, row_indices, n), (1, col_indices, columns)):
        if len(indices) == count:
            continue
        gaps = np.flatnonzero(np.diff(indices) > 1)
        split = int(gaps[0] + 1) if len(gaps) else len(indices)
        shape = list(total.shape)
        shape[axis] = count - len(indices)
        before, after = mx.split(total, [split], axis=axis)
        total = mx.concatenate((before, mx.zeros(shape, dtype=mx.complex64), after), axis=axis)
    return total


def _coherent_columns(prepared: PreparedMpsSSB, n: int) -> int:
    """Use the unique half only when both Nyquist edges have zero support."""
    radius = (
        prepared.semiangle_rad
        + 0.5 * max(prepared.ang_y_rad, prepared.ang_x_rad)
    ) / prepared.wavelength
    # Round the support bound outwards, including the soft aperture edge.
    radius *= 1.0 + 8.0 * np.finfo(np.float32).eps
    row_support = float(np.max(np.abs(prepared.kx_np))) + radius
    col_support = float(np.max(np.abs(prepared.ky_np))) + radius
    row_nyquist = n // 2 * abs(float(prepared.q_row[1]))
    col_nyquist = n // 2 * abs(float(prepared.q_col[1]))
    if row_nyquist > row_support and col_nyquist > col_support:
        return n // 2 + 1
    return n


def _reconstruct_upsampled(
    prepared: PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    sample_tilt_mrad: tuple[float, float],
    thickness: float,
    upsampling_factor: int,
    phase_estimator: str,
    higher_order_magnitudes: np.ndarray | None = None,
    higher_order_angles: np.ndarray | None = None,
    compute_loss: bool = False,
) -> tuple[np.ndarray, float | None]:
    """Return the full fine-grid phase with every selected BF contribution.

    ``mean_phase`` averages per-detector phases after their inverse FFTs.
    ``phase_of_mean`` accumulates corrected spectra before one inverse FFT.
    Both use complex64 arithmetic. Units match the prepared MPS engine:
    lengths in Angstrom, tilt in mrad, and ``phi12`` in radians.
    """
    if phase_estimator not in ("mean_phase", "phase_of_mean"):
        raise ValueError("Choose phase_estimator='mean_phase' or 'phase_of_mean'.")
    if compute_loss and (upsampling_factor != 1 or phase_estimator != "mean_phase"):
        raise ValueError("The phase-variance diagnostic requires native-grid mean_phase.")
    mx = prepared.mx
    source_n = int(prepared.scan_shape[0])
    n = source_n * upsampling_factor
    cached_geometry = prepared.upsampled_geometry
    if cached_geometry is not None and cached_geometry[0] == n:
        qr, qc = cached_geometry[1:3]
    else:
        frequencies = np.fft.fftfreq(n) * n
        qr = mx.array((frequencies * float(prepared.q_row[1])).astype(np.float32))
        qc = mx.array((frequencies * float(prepared.q_col[1])).astype(np.float32))
        prepared.upsampled_geometry = (n, qr, qc, None, None)
    params = _params(prepared, C10, C12, phi12, sample_tilt_mrad, thickness)
    higher = higher_order_magnitudes is not None
    packed = (pack_coefficients(higher_order_magnitudes, higher_order_angles)
              if higher else np.zeros((14, 2), np.float32))
    coefficients = mx.array(packed)
    dc = mx.array([complex(prepared.dc_value)], dtype=mx.complex64)
    wave_mean = phase_estimator == "phase_of_mean"
    if wave_mean and not higher and thickness == 0.0 and _SEPARABLE:
        total = _separable_spectrum(prepared, qr, qc, params, np.array(params), dc, n)
        wave = mx.fft.ifft2(total / prepared.num_bright_field_pixels)
        phase = mx.arctan2(mx.imag(wave), mx.real(wave))
        mx.eval(phase)
        checkpoint()
        return np.asarray(phase, dtype=np.float32), None
    chunk = max(1, _WORK_BYTES // (n * n * 8 * 3))
    stored = int(prepared.g_qk.shape[0])
    probe_key = (float(C10), float(C12), float(phi12), higher, packed.tobytes())
    cached_probe = prepared.upsampled_probe
    if cached_probe is not None and cached_probe[0] == probe_key:
        probe = cached_probe[1]
    else:
        probe = _alias_probe_kernel()(
            inputs=[prepared.kx, prepared.ky, params, coefficients],
            template=[("K", stored), ("HIGHER", higher)],
            grid=(stored, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(stored, 5)], output_dtypes=[mx.float32],
        )[0]
        prepared.upsampled_probe = (probe_key, probe)
    if wave_mean:
        columns = _coherent_columns(prepared, n) if not higher and upsampling_factor > 1 else n
        total = _coherent_spectrum(
            prepared, qr, qc, params, dc, probe, coefficients, higher, n,
            chunk, columns, has_depth=thickness != 0.0,
        )
        # Compacted inactive BF columns still carry the common real DC term.
        total[0, 0] = total[0, 0] + (prepared.num_bright_field_pixels - stored) * dc[0]
        if columns != n:
            # For the even C10/C12 probe, paired corrected frequencies are
            # anti-Hermitian. The guard excludes unpaired Nyquist support.
            mirror_rows = mx.array((-np.arange(n)) % n)
            negative = -mx.conj(total[mirror_rows, 1:n // 2][:, ::-1])
            total = mx.concatenate((total, negative), axis=1)
        wave = mx.fft.ifft2(total / prepared.num_bright_field_pixels)
        phase = mx.arctan2(mx.imag(wave), mx.real(wave))
        mx.eval(phase)
        checkpoint()
        return np.asarray(phase, dtype=np.float32), None

    total = mx.zeros((n, n), dtype=mx.float32)
    squares = mx.zeros((n, n), dtype=mx.float32) if compute_loss else None
    kernel = _alias_correction_kernel(thickness != 0.0)
    for start in range(0, stored, chunk):
        if (start // chunk) % _PENDING_CHUNKS == 0:
            checkpoint()
        stop = min(start + chunk, stored)
        corrected = kernel(
            inputs=[
                prepared.g_qk[start:stop],
                qr,
                qc,
                prepared.kx[start:stop],
                prepared.ky[start:stop],
                params,
                dc,
                probe[start:stop],
                coefficients,
            ],
            template=[
                ("N", n),
                ("SOURCE_N", source_n),
                ("G_COLS", int(prepared.g_qk.shape[-1])),
                ("K", stop - start),
                ("HIGHER", higher),
            ],
            grid=(n, n, stop - start),
            threadgroup=(32, 8, 1),
            output_shapes=[(stop - start, n, n)],
            output_dtypes=[mx.complex64],
        )[0]
        waves = ifft2_chunked(mx, corrected)
        total = total + phase_sum_from_complex(mx, waves)
        if compute_loss:
            angles = mx.arctan2(mx.imag(waves), mx.real(waves))
            squares = squares + mx.sum(angles * angles, axis=0)
            mx.eval(squares)
        # Keep the same accumulation order while overlapping chunk submission.
        # Periodic completion bounds the number of live temporary arrays.
        mx.async_eval(total)
        if (start // chunk + 1) % _PENDING_CHUNKS == 0:
            mx.eval(total)
    phase = total / prepared.num_bright_field_pixels
    mx.eval(phase)
    checkpoint()
    loss = float(mx.mean(squares / prepared.num_bright_field_pixels - phase * phase)) if compute_loss else None
    return np.asarray(phase, dtype=np.float32), loss
