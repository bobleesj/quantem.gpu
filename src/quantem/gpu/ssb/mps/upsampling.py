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
_HEADER = ("#define SSB_PRECISE_GEOMETRY\n#define SSB_DEVICE inline\n"
           "#define SSB_COEFFICIENTS const device float*\n"
           + _THICK_FIT_BATCH_HEADER + CHI_SOURCE)


# One transfer-function implementation for per-detector and coherent outputs.
# Keep precise geometry, trigonometry, and float32 evaluation on both paths.
# Pointer types retain MLX's constant/device address spaces for small BF groups.
_ALIAS_TRANSFER = """
template <uint SOURCE_N, uint G_COLS, bool HIGHER,
          typename G, typename KRow, typename KCol, typename Params,
          typename Probe, typename Coefficients>
inline float2 alias_corrected(
    uint row, uint col, uint k, float qr, float qc,
    G g, KRow k_row, KCol k_col, Params params, Probe probe, Coefficients coefficients
) {
    float wl = params[0], semi = params[1], ay = params[2], ax = params[3];
    float c10 = params[4], c12 = params[5], c2 = params[6], s2 = params[7];
    float factor = params[8], depth = params[9], tr = params[10], tc = params[11];
    float kr = k_row[k], kc = k_col[k];
    float4 gk = float4(probe[k * 5], probe[k * 5 + 1], probe[k * 5 + 2], probe[k * 5 + 3]);
    float4 gm = tf_geometry(qr - kr, qc - kc, wl, semi, ay, ax);
    float4 gp = tf_geometry(qr + kr, qc + kc, wl, semi, ay, ax);
    if (gk.w == 0.0f || (gm.w == 0.0f && gp.w == 0.0f)) {
        return float2(0.0f);
    }
    float chi_k = probe[k * 5 + 4];
    float chi_m = factor * gm.x * (c12 * (gm.y * c2 + gm.z * s2) + c10);
    float chi_p = factor * gp.x * (c12 * (gp.y * c2 + gp.z * s2) + c10);
    if (HIGHER) {
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
    // Only the source G has Hermitian symmetry. Corrected aliases do not.
    uint source_row = row % SOURCE_N, source_col = col % SOURCE_N;
    bool mirror = G_COLS != SOURCE_N && source_col > SOURCE_N / 2u;
    if (mirror) {
        source_row = source_row == 0u ? 0u : SOURCE_N - source_row;
        source_col = SOURCE_N - source_col;
    }
    auto value = g[((size_t)k * SOURCE_N + source_row) * G_COLS + source_col];
    float vr = value.real, vi = mirror ? -value.imag : value.imag;
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
def _alias_correction_kernel():
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_alias_correction",
        input_names=["g", "q_row", "q_col", "k_row", "k_col", "params", "dc", "probe", "coefficients"],
        output_names=["corrected"],
        header=_HEADER + _ALIAS_TRANSFER,
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
        float2 value = alias_corrected<SOURCE_N, G_COLS, HIGHER>(
            row, col, k, q_row[row], q_col[col],
            g, k_row, k_col, params, probe, coefficients);
        corrected[out].real = value.x;
        corrected[out].imag = value.y;
        """,
    )


@lru_cache(maxsize=1)
def _coherent_alias_kernel():
    """Correct and sum a detector group without materializing every spectrum."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_coherent_alias",
        input_names=["g", "q_row", "q_col", "k_row", "k_col", "params", "dc", "probe", "coefficients"],
        output_names=["partial"],
        header=_HEADER + _ALIAS_TRANSFER,
        compile_options={"math_mode": "safe"},
        source="""
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        uint start = group * CHUNK, stop = metal::min(start + CHUNK, uint(K));
        if (col >= COLS || row >= N || start >= K) return;
        size_t out = ((size_t)group * N + row) * COLS + col;
        float2 sum = float2(0.0f);
        if (row == 0u && col == 0u) {
            for (uint k = start; k < stop; ++k) sum += float2(dc[0].real, dc[0].imag);
        } else if (row % SOURCE_N != 0u || col % SOURCE_N != 0u) {
            float qr = q_row[row], qc = q_col[col];
            // Both apertures vanish beyond twice their maximum radius.
            // This bound includes the antialiased rim and a conservative
            // float32 margin. Frequencies inside it retain the full model.
            float radius = 2.0f * (params[1] + 0.5f * metal::max(params[2], params[3]))
                           / params[0] * 1.00001f + 1e-6f;
            if (qr * qr + qc * qc <= radius * radius) {
                for (uint k = start; k < stop; ++k) {
                    sum += alias_corrected<SOURCE_N, G_COLS, HIGHER>(
                        row, col, k, qr, qc, g, k_row, k_col, params, probe, coefficients);
                }
            }
        }
        partial[out].real = sum.x;
        partial[out].imag = sum.y;
        """,
    )


@lru_cache(maxsize=1)
def _ordered_alias_sum_kernel():
    """Retain detector-group accumulation order across bounded launches."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_ordered_alias_sum",
        input_names=["partial", "previous"],
        output_names=["total"],
        compile_options={"math_mode": "safe"},
        source="""
        uint i = thread_position_in_grid.x;
        if (i >= N * COLS) return;
        float re = previous[i].real, im = previous[i].imag;
        for (uint group = 0; group < GROUPS; ++group) {
            auto value = partial[(size_t)group * N * COLS + i];
            re += value.real;
            im += value.imag;
        }
        total[i].real = re;
        total[i].imag = im;
        """,
    )


def _coherent_spectrum(
    prepared: PreparedMpsSSB, qr, qc, params, dc, probe, coefficients,
    higher: bool, n: int, chunk: int, columns: int,
):
    """Accumulate every BF contribution with at most one bounded partial array."""
    mx = prepared.mx
    stored = int(prepared.g_qk.shape[0])
    source_n = int(prepared.scan_shape[0])
    total = mx.zeros((n, columns), dtype=mx.complex64)
    # Preserve the established BF chunk boundaries; only the inner sum changes
    # from MLX's reduction to sequential float32 addition in the fused kernel.
    groups_per_batch = max(1, _WORK_BYTES // (n * columns * 8))
    for start in range(0, stored, groups_per_batch * chunk):
        stop = min(start + groups_per_batch * chunk, stored)
        groups = (stop - start + chunk - 1) // chunk
        partial = _coherent_alias_kernel()(
            inputs=[prepared.g_qk[start:stop], qr, qc, prepared.kx[start:stop],
                    prepared.ky[start:stop], params, dc, probe[start:stop], coefficients],
            template=[("N", n), ("COLS", columns), ("SOURCE_N", source_n),
                      ("G_COLS", int(prepared.g_qk.shape[-1])),
                      ("K", stop - start), ("HIGHER", higher), ("CHUNK", chunk)],
            grid=(columns, n, groups), threadgroup=(32, 8, 1),
            output_shapes=[(groups, n, columns)], output_dtypes=[mx.complex64],
        )[0]
        total = _ordered_alias_sum_kernel()(
            inputs=[partial, total], template=[("N", n), ("COLS", columns), ("GROUPS", groups)],
            grid=(n * columns, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(n, columns)], output_dtypes=[mx.complex64],
        )[0]
        mx.eval(total)
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
    frequencies = np.fft.fftfreq(n) * n
    qr = mx.array((frequencies * float(prepared.q_row[1])).astype(np.float32))
    qc = mx.array((frequencies * float(prepared.q_col[1])).astype(np.float32))
    params = _params(prepared, C10, C12, phi12, sample_tilt_mrad, thickness)
    higher = higher_order_magnitudes is not None
    coefficients = mx.array(pack_coefficients(higher_order_magnitudes, higher_order_angles)
                            if higher else np.zeros((14, 2), np.float32))
    dc = mx.array([complex(prepared.dc_value)], dtype=mx.complex64)
    wave_mean = phase_estimator == "phase_of_mean"
    chunk = max(1, _WORK_BYTES // (n * n * 8 * 3))
    stored = int(prepared.g_qk.shape[0])
    probe = _alias_probe_kernel()(
        inputs=[prepared.kx, prepared.ky, params, coefficients],
        template=[("K", stored), ("HIGHER", higher)], grid=(stored, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(stored, 5)], output_dtypes=[mx.float32],
    )[0]
    if wave_mean:
        columns = _coherent_columns(prepared, n) if not higher and upsampling_factor > 1 else n
        total = _coherent_spectrum(
            prepared, qr, qc, params, dc, probe, coefficients, higher, n, chunk, columns)
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
        return np.asarray(phase, dtype=np.float32), None

    total = mx.zeros((n, n), dtype=mx.float32)
    squares = mx.zeros((n, n), dtype=mx.float32) if compute_loss else None
    kernel = _alias_correction_kernel()
    for start in range(0, stored, chunk):
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
    loss = float(mx.mean(squares / prepared.num_bright_field_pixels - phase * phase)) if compute_loss else None
    return np.asarray(phase, dtype=np.float32), loss
