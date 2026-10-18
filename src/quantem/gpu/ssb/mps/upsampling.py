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


# One transfer-function implementation for per-detector and coherent outputs.
# Keep precise geometry, trigonometry, and float32 evaluation on both paths.
# Pointer types retain MLX's constant/device address spaces for small BF groups.
_ALIAS_TRANSFER = """
template <uint SOURCE_N, uint G_COLS, bool HIGHER,
          typename G, typename KRow, typename KCol, typename Params,
          typename Probe, typename Coefficients>
inline float2 alias_corrected(
    uint row, uint col, uint k, float qr, float qc, float shift, float radius,
    G g, KRow k_row, KCol k_col, Params params, Probe probe, Coefficients coefficients
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
    // A closed aperture contributes exactly zero, independent of its phase.
    float a = 0.0f, b = 0.0f;
    float ca = 0.0f, cb = 0.0f, sa = 0.0f, sb = 0.0f;
    if (gm.w != 0.0f) {
        a = tf_sinc(0.5f * depth * (-factor * (gm.x - gk.x) - shift)) * gm.w * gk.w;
        sa = metal::precise::sincos(chi_m - chi_k, ca);
    }
    if (gp.w != 0.0f) {
        b = tf_sinc(0.5f * depth * (factor * (gp.x - gk.x) - shift)) * gp.w * gk.w;
        sb = metal::precise::sincos(chi_p - chi_k, cb);
    }
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
        float qr = q_row[row], qc = q_col[col];
        float shift = 6.283185307179586f * (qr * params[10] + qc * params[11]);
        // Preserve the outward roundoff margin and the original expressions.
        float radius = (params[1] + 0.5f * metal::max(params[2], params[3])) / params[0];
        radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
        float2 value = alias_corrected<SOURCE_N, G_COLS, HIGHER>(
            row, col, k, qr, qc, shift, radius,
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
        input_names=["g", "q_row", "q_col", "k_row", "k_col", "params", "dc",
                     "probe", "coefficients", "bounds", "row_indices", "col_indices"],
        output_names=["partial"],
        header=_HEADER + _ALIAS_TRANSFER,
        compile_options={"math_mode": "safe"},
        source="""
        uint col_index = thread_position_in_grid.x;
        uint row_index = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        uint start = group * CHUNK, stop = metal::min(start + CHUNK, uint(K));
        if (col_index >= COLS || row_index >= ROWS || start >= K) return;
        uint row = row_indices[row_index], col = col_indices[col_index];
        size_t out = ((size_t)group * ROWS + row_index) * COLS + col_index;
        float2 sum = float2(0.0f);
        if (row == 0u && col == 0u) {
            for (uint k = start; k < stop; ++k) sum += float2(dc[0].real, dc[0].imag);
        } else if (row % SOURCE_N != 0u || col % SOURCE_N != 0u) {
            float qr = q_row[row], qc = q_col[col];
            // Outside both expanded group bounds every contribution is zero.
            // True DC and its aliases retain the rules above.
            float row_min = bounds[group * 5], row_max = bounds[group * 5 + 1];
            float col_min = bounds[group * 5 + 2], col_max = bounds[group * 5 + 3];
            float rm = max(max(row_min - qr, qr - row_max), 0.0f);
            float cm = max(max(col_min - qc, qc - col_max), 0.0f);
            float rp = max(max(row_min + qr, -qr - row_max), 0.0f);
            float cp = max(max(col_min + qc, -qc - col_max), 0.0f);
            if (min(rm * rm + cm * cm, rp * rp + cp * cp) > bounds[group * 5 + 4]) {
                partial[out].real = 0.0f;
                partial[out].imag = 0.0f;
                return;
            }
            // These values depend on q, not the detector contribution k.
            float shift = 6.283185307179586f * (qr * params[10] + qc * params[11]);
            // Preserve the outward roundoff margin and the original expressions.
            float radius = (params[1] + 0.5f * metal::max(params[2], params[3])) / params[0];
            radius *= 1.0f + 8.0f * 1.1920928955078125e-7f;
            for (uint k = start; k < stop; ++k) {
                sum += alias_corrected<SOURCE_N, G_COLS, HIGHER>(
                    row, col, k, qr, qc, shift, radius, g, k_row, k_col, params, probe, coefficients);
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
    higher: bool, n: int, chunk: int, columns: int,
):
    """Accumulate every BF contribution with at most one bounded partial array."""
    mx = prepared.mx
    stored = int(prepared.g_qk.shape[0])
    source_n = int(prepared.scan_shape[0])
    row_indices, col_indices = _coherent_support(prepared, n, columns)
    rows, cols = len(row_indices), len(col_indices)
    row_indices_m = mx.array(row_indices)
    col_indices_m = mx.array(col_indices)
    total = mx.zeros((rows, cols), dtype=mx.complex64)
    group_count = (stored + chunk - 1) // chunk
    bounds = _alias_group_bounds_kernel()(
        inputs=[prepared.kx, prepared.ky, params],
        template=[("K", stored), ("CHUNK", chunk)],
        grid=(group_count, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(group_count, 5)], output_dtypes=[mx.float32],
    )[0]
    # Preserve BF chunk boundaries and sequential float32 accumulation order.
    # Omit only the identically zero frequencies from temporary storage.
    groups_per_batch = max(1, _WORK_BYTES // (rows * cols * 8))
    for start in range(0, stored, groups_per_batch * chunk):
        checkpoint()
        stop = min(start + groups_per_batch * chunk, stored)
        groups = (stop - start + chunk - 1) // chunk
        partial = _coherent_alias_kernel()(
            inputs=[prepared.g_qk[start:stop], qr, qc, prepared.kx[start:stop],
                    prepared.ky[start:stop], params, dc, probe[start:stop], coefficients,
                    bounds[start // chunk:start // chunk + groups],
                    row_indices_m, col_indices_m],
            template=[("N", n), ("ROWS", rows), ("COLS", cols), ("SOURCE_N", source_n),
                      ("G_COLS", int(prepared.g_qk.shape[-1])),
                      ("K", stop - start), ("HIGHER", higher), ("CHUNK", chunk)],
            grid=(cols, rows, groups), threadgroup=(32, 8, 1),
            output_shapes=[(groups, rows, cols)], output_dtypes=[mx.complex64],
        )[0]
        total = _ordered_alias_sum_kernel()(
            inputs=[partial, total], template=[("N", rows), ("COLS", cols), ("GROUPS", groups)],
            grid=(rows * cols, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(rows, cols)], output_dtypes=[mx.complex64],
        )[0]
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
        checkpoint()
        return np.asarray(phase, dtype=np.float32), None

    total = mx.zeros((n, n), dtype=mx.float32)
    squares = mx.zeros((n, n), dtype=mx.float32) if compute_loss else None
    kernel = _alias_correction_kernel()
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
