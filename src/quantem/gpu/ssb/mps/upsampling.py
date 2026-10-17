"""Bounded Metal reconstruction on finer scan grids from the retained BF FFT.

The measured Fourier aliases repeat at the native scan frequency. The probe
correction is evaluated at each expanded frequency; this is not interpolation
of a reconstructed phase image. The native aberration-search objective stays
unchanged.
"""

from functools import lru_cache

import numpy as np

from quantem.gpu.ssb.higher_order import CHI_SOURCE, pack_coefficients
from quantem.gpu.ssb.mps.hardware import require_mlx
from quantem.gpu.ssb.mps.kernels.phase_sums import phase_sum_from_complex
from quantem.gpu.ssb.mps.prepared import PreparedMpsSSB, ifft2_chunked
from quantem.gpu.ssb.mps.thick_sample import _THICK_FIT_BATCH_HEADER, _params

# Bound the correction and FFT working arrays independently of the BF count.
_WORK_BYTES = 512 << 20
_PENDING_CHUNKS = 4
_COHERENT_GROUP = 32
_COHERENT_CHUNK = 256
_HEADER = ("#define SSB_PRECISE_GEOMETRY\n#define SSB_DEVICE inline\n"
           "#define SSB_COEFFICIENTS const device float*\n"
           + _THICK_FIT_BATCH_HEADER + CHI_SOURCE)


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


# Both estimators use the same per-BF correction. Coherent reconstruction
# reduces groups in registers before writing scratch spectra; mean_phase
# retains one complete spectrum per BF pixel for its individual inverse FFT.
_CORRECTION_HEADER = r"""
template <typename Complex>
inline float2 ssb_corrected_alias(
    uint row, uint col, uint k, uint source_n, uint g_cols, bool higher,
    const device Complex* g, const device float* q_row,
    const device float* q_col, float kr, float kc,
    const device float* params,
    Complex dc_value, const device float* probe,
    const device float* coefficients
) {
    if (col == 0u && row == 0u) {
        return float2(dc_value.real, dc_value.imag);
    }
    // The scan mean is background, not measured nonzero-q structure.
    // Keep its normalization only at the true origin, never its aliases.
    if (row % source_n == 0u && col % source_n == 0u) {
        return float2(0.0f);
    }
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float c10 = params[4], c12 = params[5], c2 = params[6], s2 = params[7];
    float factor = params[8], depth = params[9], tr = params[10], tc = params[11];
    float qr = q_row[row], qc = q_col[col];
    float4 gk = float4(probe[k * 5], probe[k * 5 + 1], probe[k * 5 + 2], probe[k * 5 + 3]);
    // Bound the complete soft aperture before evaluating its geometry.
    // The outward margin includes float32 roundoff and anisotropic pixels.
    float radius = (semi + 0.5f * metal::max(ay, ax)) / wl;
    radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
    float mr = qr - kr, mc = qc - kc, pr = qr + kr, pc = qc + kc;
    if (metal::min(mr * mr + mc * mc, pr * pr + pc * pc) > radius * radius) {
        return float2(0.0f);
    }
    float4 gm = tf_geometry(qr - kr, qc - kc, wl, semi, ay, ax);
    float4 gp = tf_geometry(qr + kr, qc + kc, wl, semi, ay, ax);
    if (gk.w == 0.0f || (gm.w == 0.0f && gp.w == 0.0f)) {
        return float2(0.0f);
    }
    float chi_k = probe[k * 5 + 4];
    float chi_m = factor * gm.x * (c12 * (gm.y * c2 + gm.z * s2) + c10);
    float chi_p = factor * gp.x * (c12 * (gp.y * c2 + gp.z * s2) + c10);
    if (higher) {
        chi_m = ssb_chi_full(qr - kr, qc - kc, wl, coefficients);
        chi_p = ssb_chi_full(qr + kr, qc + kc, wl, coefficients);
    }
    float shift = 6.283185307179586f * (qr * tr + qc * tc);
    float a = tf_sinc(0.5f * depth * (-factor * (gm.x - gk.x) - shift)) * gm.w * gk.w;
    float b = tf_sinc(0.5f * depth * (factor * (gp.x - gk.x) - shift)) * gp.w * gk.w;
    float ca, cb;
    float sa = metal::precise::sincos(chi_m - chi_k, ca);
    float sb = metal::precise::sincos(chi_p - chi_k, cb);
    float re = a * ca - b * cb, im = -a * sa - b * sb;
    float inv = 1.0f / metal::max(metal::precise::sqrt(re * re + im * im), 1e-8f);
    // Source G is Hermitian. The corrected non-DC spectrum is
    // anti-Hermitian only for an even probe and paired frequencies.
    uint source_row = row % source_n, source_col = col % source_n;
    bool mirror = g_cols != source_n && source_col > source_n / 2u;
    if (mirror) {
        source_row = source_row == 0u ? 0u : source_n - source_row;
        source_col = source_n - source_col;
    }
    auto value = g[((size_t)k * source_n + source_row) * g_cols + source_col];
    float vr = value.real, vi = mirror ? -value.imag : value.imag;
    return float2((vr * re + vi * im) * inv, (vi * re - vr * im) * inv);
}
"""


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
        uint first = group * BF_GROUP;
        if (first >= K) return;
        float row_min = k_row[first], row_max = row_min;
        float col_min = k_col[first], col_max = col_min;
        for (uint k = first + 1u; k < min(first + BF_GROUP, uint(K)); ++k) {
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


@lru_cache(maxsize=1)
def _alias_correction_kernel():
    """Correct each BF spectrum, optionally summing coherent groups on chip."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_alias_correction",
        input_names=["g", "q_row", "q_col", "k_row", "k_col", "params", "dc", "probe", "coefficients", "bounds"],
        output_names=["corrected"],
        header=_HEADER + _CORRECTION_HEADER,
        compile_options={"math_mode": "safe"},
        source="""
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        uint first = group * BF_GROUP;
        if (col >= COLS || row >= N || first >= K) return;
        // Outside the union of the two aperture bounds every term is zero.
        // Preserve true DC and its aliases through the common correction.
        if (BF_GROUP > 1 && !(row % SOURCE_N == 0u && col % SOURCE_N == 0u)) {
            float qr = q_row[row], qc = q_col[col];
            float row_min = bounds[group * 5], row_max = bounds[group * 5 + 1];
            float col_min = bounds[group * 5 + 2], col_max = bounds[group * 5 + 3];
            float rm = max(max(row_min - qr, qr - row_max), 0.0f);
            float cm = max(max(col_min - qc, qc - col_max), 0.0f);
            float rp = max(max(row_min + qr, -qr - row_max), 0.0f);
            float cp = max(max(col_min + qc, -qc - col_max), 0.0f);
            if (min(rm * rm + cm * cm, rp * rp + cp * cp) > bounds[group * 5 + 4]) {
                size_t out = ((size_t)group * N + row) * COLS + col;
                corrected[out].real = 0.0f;
                corrected[out].imag = 0.0f;
                return;
            }
        }
        float2 value = ssb_corrected_alias(
            row, col, first, SOURCE_N, G_COLS, HIGHER,
            g, q_row, q_col, k_row[first], k_col[first], params, dc[0], probe, coefficients);
        for (uint k = first + 1u; k < min(first + BF_GROUP, uint(K)); ++k) {
            value += ssb_corrected_alias(
                row, col, k, SOURCE_N, G_COLS, HIGHER,
                g, q_row, q_col, k_row[k], k_col[k], params, dc[0], probe, coefficients);
        }
        size_t out = ((size_t)group * N + row) * COLS + col;
        corrected[out].real = value.x;
        corrected[out].imag = value.y;
        """,
    )


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
    frequencies = np.fft.fftfreq(n) * n
    qr = mx.array((frequencies * float(prepared.q_row[1])).astype(np.float32))
    qc = mx.array((frequencies * float(prepared.q_col[1])).astype(np.float32))
    params = _params(prepared, C10, C12, phi12, sample_tilt_mrad, thickness)
    higher = higher_order_magnitudes is not None
    coefficients = mx.array(pack_coefficients(higher_order_magnitudes, higher_order_angles)
                            if higher else np.zeros((14, 2), np.float32))
    dc = mx.array([complex(prepared.dc_value)], dtype=mx.complex64)
    wave_mean = phase_estimator == "phase_of_mean"
    columns = n
    if wave_mean and not higher and upsampling_factor > 1:
        columns = _coherent_columns(prepared, n)
    total = mx.zeros((n, columns), dtype=mx.complex64 if wave_mean else mx.float32)
    squares = mx.zeros((n, n), dtype=mx.float32) if compute_loss else None
    chunk = max(1, _WORK_BYTES // (n * n * 8 * 3))
    bf_group = _COHERENT_GROUP if wave_mean else 1
    if wave_mean:
        # The working-memory bound applies to group sums, not all BF spectra.
        planes = max(1, _WORK_BYTES // (n * columns * 8 * 3))
        chunk = min(_COHERENT_CHUNK, planes * bf_group)
    stored = int(prepared.g_qk.shape[0])
    probe = _alias_probe_kernel()(
        inputs=[prepared.kx, prepared.ky, params, coefficients],
        template=[("K", stored), ("HIGHER", higher)], grid=(stored, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(stored, 5)], output_dtypes=[mx.float32],
    )[0]
    bounds = mx.zeros((1, 5), dtype=mx.float32)
    if wave_mean:
        groups = (stored + bf_group - 1) // bf_group
        bounds = _alias_group_bounds_kernel()(
            inputs=[prepared.kx, prepared.ky, params],
            template=[("K", stored), ("BF_GROUP", bf_group)],
            grid=(groups, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(groups, 5)], output_dtypes=[mx.float32],
        )[0]
    kernel = _alias_correction_kernel()
    for start in range(0, stored, chunk):
        stop = min(start + chunk, stored)
        groups = (stop - start + bf_group - 1) // bf_group
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
                bounds[start // bf_group:(stop + bf_group - 1) // bf_group]
                if wave_mean else bounds,
            ],
            template=[
                ("N", n),
                ("COLS", columns),
                ("SOURCE_N", source_n),
                ("G_COLS", int(prepared.g_qk.shape[-1])),
                ("K", stop - start),
                ("HIGHER", higher),
                ("BF_GROUP", bf_group),
            ],
            grid=(columns, n, groups),
            threadgroup=(32, 8, 1),
            output_shapes=[(groups, n, columns)],
            output_dtypes=[mx.complex64],
        )[0]
        if wave_mean:
            total = total + mx.sum(corrected, axis=0)
        else:
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
    if wave_mean:
        # Compacted inactive BF columns still carry the common real DC term.
        total[0, 0] = total[0, 0] + (prepared.num_bright_field_pixels - stored) * dc[0]
        if columns != n:
            # Gamma(-q,k) = -conj(Gamma(q,k)) for the C10/C12 probe.
            # Preserve the real DC separately and reflect only paired columns.
            mirror_rows = mx.array((-np.arange(n)) % n)
            negative = -mx.conj(total[mirror_rows, 1:n // 2][:, ::-1])
            total = mx.concatenate((total, negative), axis=1)
        wave = mx.fft.ifft2(total / prepared.num_bright_field_pixels)
        phase = mx.arctan2(mx.imag(wave), mx.real(wave))
    else:
        phase = total / prepared.num_bright_field_pixels
    mx.eval(phase)
    loss = float(mx.mean(squares / prepared.num_bright_field_pixels - phase * phase)) if compute_loss else None
    return np.asarray(phase, dtype=np.float32), loss
