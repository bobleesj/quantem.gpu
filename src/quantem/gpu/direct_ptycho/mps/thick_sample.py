"""Thick-sample SSB (sample tilt + thickness) on MPS: MLX reference path.

Same model as the CUDA ``thick_correct_kernel`` / ``thick_fit_kernel`` in ``cuda/kernels/engine.py`` (read that comment for the physics). Standard SSB
treats the sample as one plane at the probe defocus. For a crystal of thickness t tilted by theta (straight columns), the
slice at depth z (from mid-depth) sees the defocus C10 + z and sits shifted by z theta. Averaging the two SSB terms over depth
gives each its own real weight (sinc of the depth-phase rate times t / 2):

    t1 = P(q - k) conj(P(k)),   rate1 = -factor (alpha_m^2 - alpha_k^2) - 2 pi q . theta
    t2 = conj(P(q + k)) P(k),   rate2 = +factor (alpha_p^2 - alpha_k^2) - 2 pi q . theta
    gamma = w1 t1 - w2 t2,      w = sinc(rate t / 2)

with P = aperture exp(-i chi) at the mid-depth aberrations, chi = factor alpha^2 (C10 + C12 cos 2(phi - phi12)),
factor = pi / lambda[A], and the soft aperture of ``prepared.compute_geometry``. Thickness 0 gives w = 1 and the standard
correction exactly. P is even in k for C10 and C12, so P(q - k) = P(k - q); the odd higher orders are not, and
``higher_order_agreement`` evaluates the first term at k - q. Units: q, k in 1/A; C10, C12, thickness in Angstrom (the engine unit); tilt in mrad, scan frame (row, col).

Two implementations of the thick preview. ``reconstruct_thick`` (the interactive path) runs the standard fused MPS
row kernel with the depth weights added per (k, q) (``kernels.row_ifft._row_ifft_small_dynamic_kernel(thick=True)``) and the same
column-IFFT phase kernel, so it costs about what the standard preview costs; 128/256/512/1024 scans. ``reconstruct_thick_reference``
is the element-wise model as one ``mx.compile`` graph with MLX inverse FFTs: any scan shape, the definition the fast path
is tested against, and the fallback for shapes the fused kernels do not cover.
"""

import math
from functools import lru_cache

import numpy as np

from quantem.gpu.direct_ptycho._execution import checkpoint
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx
from quantem.gpu.direct_ptycho.mps.kernels.phase_sums import phase_sums_from_complex
from quantem.gpu.direct_ptycho.mps.prepared import (
    PreparedMpsSSB,
    compute_geometry,
    expand_hermitian_mx,
    ifft2_chunked,
)
from quantem.gpu.direct_ptycho.mps.reconstruct import reconstruct_prepared

_FUSED_SHAPES = ((128, 128), (256, 256), (512, 512), (1024, 1024))
# A chunk of the element-wise model keeps _LIVE_PLANES complex64 plane-sized temporaries per BF pixel; _chunk_bf sizes
# chunks so they stay near _CHUNK_BYTES.
_CHUNK_BYTES = 256 << 20
_LIVE_PLANES = 6
THICK_FIT_MAX_BATCH = 8
# Keep search proposals in groups of eight. Smaller execution tiles reduce
# register pressure without changing proposal order or BF reduction order.
_FIT_DISPATCH_BATCH = 4
# Small scans benefit from scheduling Newton's independent candidate tiles
# together. Larger scans already supply enough work in each execution tile.
_FIT_PARALLEL_SHAPES = ((128, 128), (256, 256))
# Bound queued partial reductions while letting independent Metal tiles run
# without a CPU readback between every four candidates.
_FIT_PENDING_BYTES = 64 << 20
_K_CHUNK = 256      # bright-field pixels summed per thread; the partial sums over chunks are reduced afterwards


def reconstruct_thick(
    prepared: PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    sample_tilt_mrad: tuple[float, float],
    thickness: float,
    compute_loss: bool = True,
    chunk_bf: int = 4096,
) -> tuple[np.ndarray, float | None]:
    """Mean phase and phase-variance loss for a thick, tilted crystal (interactive path).

    Same outputs as ``reconstruct_thick_reference``. On 128/256/512/1024 scans the depth weights run inside the standard
    fused row kernel, ``chunk_bf`` BF pixels per dispatch (why: the per-call cost is then the standard preview's, not a
    separate element-wise graph + MLX FFT per chunk); other shapes use the reference graph.
    """
    if tuple(prepared.scan_shape) not in _FUSED_SHAPES:
        return reconstruct_thick_reference(prepared, C10=C10, C12=C12, phi12=phi12, sample_tilt_mrad=sample_tilt_mrad,
                                           thickness=thickness, compute_loss=compute_loss)
    thick = (float(thickness), float(sample_tilt_mrad[0]) * 1e-3, float(sample_tilt_mrad[1]) * 1e-3)
    _object, loss, phase = reconstruct_prepared(prepared, C10=float(C10), C12=float(C12), phi12=float(phi12),
                                                 chunk_bf=int(chunk_bf), compute_loss=compute_loss,
                                                 compute_object=False, thick=thick)
    return phase, None if loss is None else float(loss)


def reconstruct_thick_reference(
    prepared: PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    sample_tilt_mrad: tuple[float, float],
    thickness: float,
    compute_loss: bool = True,
) -> tuple[np.ndarray, float | None]:
    """Mean phase and phase-variance loss for a thick, tilted crystal: MLX element-wise reference, any scan shape.

    Same outputs and definitions as ``reconstruct.reconstruct_prepared`` (mean over bright-field pixels of the per-pixel phase
    of ifft2(G conj(gamma / |gamma|)) with the DC set to the stored DC value; loss = mean over the image of the per-pixel
    phase variance), with each pixel's correction averaged over the sample depth. Compacted inactive BF pixels (aperture 0)
    are not stored and contribute phase 0, exactly as in the standard path, so the sums divide by ``prepared.num_bright_field_pixels``.
    """
    mx = prepared.mx
    params = _params(prepared, C10, C12, phi12, sample_tilt_mrad, thickness)
    dc = mx.array(complex(np.complex64(prepared.dc_value)), dtype=mx.complex64)
    phase_sum = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    phase_sumsq = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    storage_bf = int(prepared.g_qk.shape[0])
    chunk = _chunk_bf(prepared)
    for start in range(0, storage_bf, chunk):
        checkpoint()
        stop = min(storage_bf, start + chunk)
        projected_real, projected_imag, weight2 = _chunk_terms(prepared, start, stop, params)
        # phase-only correction G conj(gamma) / |gamma|, same floor as the standard kernels
        inv_mag = 1.0 / mx.maximum(mx.sqrt(weight2), 1e-8)
        corrected = (projected_real * inv_mag) + 1j * (projected_imag * inv_mag)
        corrected = mx.where(prepared.dc_mask, dc, corrected)
        chunk_sum, chunk_sumsq = phase_sums_from_complex(mx, ifft2_chunked(mx, corrected))
        phase_sum = phase_sum + chunk_sum
        phase_sumsq = phase_sumsq + chunk_sumsq
        mx.eval(phase_sum, phase_sumsq)
    mean_phase = phase_sum / prepared.num_bright_field_pixels
    loss = None
    if compute_loss:
        loss = float(np.asarray(mx.mean(phase_sumsq / prepared.num_bright_field_pixels - mean_phase * mean_phase)))
    mx.eval(mean_phase)
    return np.asarray(mean_phase).astype(np.float32, copy=False), loss


def thick_fit(
    prepared: PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    sample_tilt_mrad: tuple[float, float],
    thickness: float,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
) -> float:
    """Least-squares fit of the thick-sample SSB model to G over a spatial-frequency band (larger = better).

    Per q the best object Psi(q) in G(q, k) = Psi(q) gamma(q, k) explains |sum_k G conj(gamma)|^2 / sum_k |gamma|^2 of the
    data (gamma unnormalised); this sums that over ``band_inv_A[0] < |q| < band_inv_A[1]``. It is the objective that
    recovers sample tilt (the phase-variance loss weights pixels equally and does not see it). Same definition as CUDA
    ``ThickSample.fit``.
    """
    mx = prepared.mx
    params = _params(prepared, C10, C12, phi12, sample_tilt_mrad, thickness)
    numerator_real = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    numerator_imag = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    denominator = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    storage_bf = int(prepared.g_qk.shape[0])
    chunk = _chunk_bf(prepared)
    for start in range(0, storage_bf, chunk):
        stop = min(storage_bf, start + chunk)
        projected_real, projected_imag, weight2 = _chunk_terms(prepared, start, stop, params)
        numerator_real = numerator_real + mx.sum(projected_real, axis=0)
        numerator_imag = numerator_imag + mx.sum(projected_imag, axis=0)
        denominator = denominator + mx.sum(weight2, axis=0)
        mx.eval(numerator_real, numerator_imag, denominator)
    band = _band_mask(prepared, band_inv_A)
    den = np.asarray(denominator)
    keep = band & (den > 0)
    power = np.asarray(numerator_real)[keep].astype(np.float64) ** 2 + np.asarray(numerator_imag)[keep].astype(np.float64) ** 2
    return float((power / den[keep]).sum())


def thick_fit_batch(prepared: PreparedMpsSSB, params, band_inv_A: tuple[float, float] = (0.2, 0.9)) -> np.ndarray:
    """``thick_fit`` for many parameter sets at once with the fused Metal kernel (fast path used by the fit search).

    ``params``: (B, 6) rows of (C10, C12, phi12, tilt_row_mrad, tilt_col_mrad, thickness), engine units. Same value as
    ``thick_fit`` row by row (rel ~1e-6); G is read once per execution tile, only at the band's q on
    the stored half-plane, and (k, q) pairs with no double overlap are skipped.

    Independent Newton tiles run together for small scans. The search proposal
    schedule, per-thread accumulation and final reductions remain unchanged.
    """
    mx = prepared.mx
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    flat, band_qx, band_qy, weight = _band(prepared, band_inv_A)
    n_band = int(flat.size)
    g = prepared.g_qk
    storage_bf = int(g.shape[0])
    plane = int(g.shape[1]) * int(g.shape[2])
    k_chunks = (storage_bf + _K_CHUNK - 1) // _K_CHUNK
    scalars = mx.array([prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad, prepared.ang_x_rad, prepared.factor],
                       dtype=mx.float32)
    kernel = _thick_fit_batch_kernel()
    execution_batch = _FIT_DISPATCH_BATCH
    if tuple(prepared.scan_shape) in _FIT_PARALLEL_SHAPES and len(params) > THICK_FIT_MAX_BATCH:
        bytes_per_candidate = (k_chunks + 1) * n_band * 3 * 4
        execution_batch = max(
            _FIT_DISPATCH_BATCH,
            (_FIT_PENDING_BYTES // bytes_per_candidate // _FIT_DISPATCH_BATCH) * _FIT_DISPATCH_BATCH,
        )
    out = np.empty(len(params))
    pending = []
    pending_bytes = 0
    for start in range(0, len(params), execution_batch):
        rows = params[start:start + execution_batch]
        batch_size = len(rows)
        tile_size = min(_FIT_DISPATCH_BATCH, batch_size)
        trial = np.stack([rows[:, 0], rows[:, 1], np.cos(2 * rows[:, 2]), np.sin(2 * rows[:, 2]), rows[:, 5],
                          rows[:, 3] * 1e-3, rows[:, 4] * 1e-3], axis=1).astype(np.float32)
        trial_mx = mx.array(trial.ravel())
        # These terms depend on k and the trial, but not on q. Calculate them
        # once per BF pixel instead of repeating them across the whole band.
        probe_geometry, probe_chi = _trial_probe_kernel()(
            inputs=[prepared.kx, prepared.ky, scalars, trial_mx],
            template=[("NUM_BF", storage_bf), ("B", batch_size)],
            grid=(storage_bf, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(storage_bf, 4), (storage_bf, batch_size)],
            output_dtypes=[mx.float32, mx.float32],
        )
        numer_real, numer_imag, denom = kernel(
            inputs=[g, flat, band_qx, band_qy, prepared.kx, prepared.ky,
                    trial_mx, scalars, probe_geometry, probe_chi],
            template=[
                ("B", tile_size), ("TOTAL", batch_size), ("N_BAND", n_band),
                ("NUM_BF", storage_bf), ("PLANE", plane), ("K_CHUNK", _K_CHUNK),
            ],
            grid=(n_band, k_chunks, (batch_size + tile_size - 1) // tile_size),
            threadgroup=(64, 1, 1),
            output_shapes=[(k_chunks, batch_size, n_band)] * 3,
            output_dtypes=[mx.float32] * 3,
        )
        sums = mx.stack([mx.sum(numer_real, axis=0), mx.sum(numer_imag, axis=0), mx.sum(denom, axis=0)])
        pending.append((start, batch_size, sums))
        pending_bytes += (k_chunks + 1) * batch_size * n_band * 3 * 4
        if (
            execution_batch > _FIT_DISPATCH_BATCH
            or pending_bytes >= _FIT_PENDING_BYTES
            or start + batch_size == len(params)
        ):
            mx.eval(*(item[2] for item in pending))
            for offset, count, partials in pending:
                # Keep the existing float32 BF reduction and float64 band sum
                # in exactly the same order; only dispatch scheduling changes.
                values = np.asarray(partials).astype(np.float64)
                den = values[2]
                ok = den > 0
                power = np.where(ok, (values[0] ** 2 + values[1] ** 2) / np.where(ok, den, 1.0), 0.0)
                out[offset:offset + count] = (weight[None] * power).sum(axis=1)
            pending.clear()
            pending_bytes = 0
    return out


def _chunk_bf(prepared: PreparedMpsSSB) -> int:
    """BF pixels per chunk so a chunk's complex temporaries stay near ``_CHUNK_BYTES``."""
    ny, nx = prepared.scan_shape
    return max(1, _CHUNK_BYTES // (int(ny) * int(nx) * 8 * _LIVE_PLANES))


@lru_cache(maxsize=1)
def _compiled_terms():
    """Compiled element-wise thick-sample model: returns (Re, Im) of G conj(gamma) and |gamma|^2 (gamma unnormalised).

    Why compiled: the model is ~60 element-wise ops per (k, q); uncompiled MLX would materialise each as a full
    (chunk, ny, nx) array. ``params`` holds every scalar as one array so the graph is traced once per chunk shape.
    """
    mx = require_mlx()

    def terms(g_real, g_imag, qx, qy, kx, ky, params):
        wavelength, semiangle, ang_y, ang_x = params[0], params[1], params[2], params[3]
        c10, c12, cos2phi12, sin2phi12 = params[4], params[5], params[6], params[7]
        factor, thickness, theta_row, theta_col = params[8], params[9], params[10], params[11]
        alpha_k2, cos2_k, sin2_k, ap_k = compute_geometry(mx, kx, ky, wavelength, semiangle, ang_y, ang_x)
        alpha_m2, cos2_m, sin2_m, ap_m = compute_geometry(mx, qx - kx, qy - ky, wavelength, semiangle, ang_y, ang_x)
        alpha_p2, cos2_p, sin2_p, ap_p = compute_geometry(mx, qx + kx, qy + ky, wavelength, semiangle, ang_y, ang_x)
        chi_k = factor * alpha_k2 * (c12 * (cos2_k * cos2phi12 + sin2_k * sin2phi12) + c10)
        chi_m = factor * alpha_m2 * (c12 * (cos2_m * cos2phi12 + sin2_m * sin2phi12) + c10)
        chi_p = factor * alpha_p2 * (c12 * (cos2_p * cos2phi12 + sin2_p * sin2phi12) + c10)
        # depth weights; thickness 0 gives x = 0 and w = 1 exactly (the standard SSB correction)
        shift = 2.0 * math.pi * (qx * theta_row + qy * theta_col)
        x1 = 0.5 * thickness * (-factor * (alpha_m2 - alpha_k2) - shift)
        x2 = 0.5 * thickness * (factor * (alpha_p2 - alpha_k2) - shift)
        small1 = mx.abs(x1) < 1e-6
        small2 = mx.abs(x2) < 1e-6
        safe1 = mx.where(small1, 1.0, x1)
        safe2 = mx.where(small2, 1.0, x2)
        w1 = mx.where(small1, 1.0, mx.sin(safe1) / safe1)
        w2 = mx.where(small2, 1.0, mx.sin(safe2) / safe2)
        # t1 = P(m) conj(P(k)) = a_m a_k exp(-i (chi_m - chi_k)); t2 = conj(P(p)) P(k) = a_p a_k exp(i (chi_p - chi_k))
        a1 = w1 * ap_m * ap_k
        a2 = w2 * ap_p * ap_k
        d1 = chi_m - chi_k
        d2 = chi_p - chi_k
        gamma_real = a1 * mx.cos(d1) - a2 * mx.cos(d2)
        gamma_imag = -a1 * mx.sin(d1) - a2 * mx.sin(d2)
        projected_real = g_real * gamma_real + g_imag * gamma_imag
        projected_imag = g_imag * gamma_real - g_real * gamma_imag
        return projected_real, projected_imag, gamma_real * gamma_real + gamma_imag * gamma_imag

    return mx.compile(terms)


def _params(prepared: PreparedMpsSSB, C10, C12, phi12, sample_tilt_mrad, thickness):
    """Pack every scalar of the thick-sample model into one float32 array, in the order ``_compiled_terms`` reads.

    One array argument keeps the compiled graph from being traced again for each new parameter value.
    """
    mx = prepared.mx
    return mx.array(
        [
            prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad, prepared.ang_x_rad,
            float(C10), float(C12), math.cos(2.0 * float(phi12)), math.sin(2.0 * float(phi12)),
            prepared.factor, float(thickness), float(sample_tilt_mrad[0]) * 1e-3, float(sample_tilt_mrad[1]) * 1e-3,
        ],
        dtype=mx.float32,
    )


def _chunk_terms(prepared: PreparedMpsSSB, start: int, stop: int, params):
    """Full-plane G for storage BF pixels [start, stop) and the model terms for them."""
    mx = prepared.mx
    _, nx = prepared.scan_shape
    # G is stored as the Hermitian half plane of a real BF image; the model needs every q
    g_full = expand_hermitian_mx(mx, prepared.g_qk[start:stop], int(nx))
    kx = prepared.kx[start:stop].reshape(-1, 1, 1)
    ky = prepared.ky[start:stop].reshape(-1, 1, 1)
    return _compiled_terms()(mx.real(g_full), mx.imag(g_full), prepared.qx, prepared.qy, kx, ky, params)


# Same model as ``_compiled_terms`` and CUDA ``thick_fit_batch_kernel``: gamma = w1 P(q-k) conj P(k) - w2 conj P(q+k) P(k).
_THICK_FIT_BATCH_HEADER = """
inline float4 tf_geometry(float dx, float dy, float wl, float semiangle, float ang_y, float ang_x) {
    float r2 = dx * dx + dy * dy;
#ifdef SSB_PRECISE_GEOMETRY
    float r = metal::precise::sqrt(r2);
#else
    float r = metal::sqrt(r2);
#endif
    float alpha = r * wl;
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2 = (dx * dx - dy * dy) * inv_r2;
    float sin2 = 2.0f * dx * dy * inv_r2;
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
#ifdef SSB_PRECISE_GEOMETRY
    // Match compute_geometry's squared angular components on the fine grid.
    float denom = metal::precise::sqrt((dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x)) * inv_r;
#else
    float denom = metal::sqrt(dx * ang_y * dx * ang_y + dy * ang_x * dy * ang_x) * inv_r;
#endif
    float edge = (denom > 1e-15f) ? ((semiangle - alpha) / denom + 0.5f) : 1.0f;
    return float4(alpha * alpha, cos2, sin2, metal::clamp(edge, 0.0f, 1.0f));
}
inline float tf_sinc(float x) { return (metal::abs(x) < 1e-6f) ? 1.0f : metal::precise::sin(x) / x; }
"""


@lru_cache(maxsize=1)
def _trial_probe_kernel():
    """Keep the original float32 geometry and chi(k) expression per trial."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_thick_trial_probe",
        input_names=["kx", "ky", "scalars", "trial"],
        output_names=["probe_geometry", "probe_chi"],
        header=_THICK_FIT_BATCH_HEADER,
        source="""
        uint k = thread_position_in_grid.x;
        if (k >= NUM_BF) return;
        float4 gk = tf_geometry(kx[k], ky[k], scalars[0], scalars[1], scalars[2], scalars[3]);
        for (uint i = 0; i < 4; ++i) probe_geometry[k * 4 + i] = gk[i];
        for (int b = 0; b < B; ++b) {
            float c10 = trial[b * 7], c12 = trial[b * 7 + 1];
            float c2 = trial[b * 7 + 2], s2 = trial[b * 7 + 3];
            probe_chi[k * B + b] = scalars[4] * gk.x * (c12 * (gk.y * c2 + gk.z * s2) + c10);
        }
        """,
    )


_THICK_FIT_BATCH_SOURCE = """
    uint q = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint base = thread_position_in_grid.z * B;
    if (q >= N_BAND) return;
    float wl = scalars[0], semiangle = scalars[1], ang_y = scalars[2], ang_x = scalars[3], factor = scalars[4];
    float qx = band_qx[q], qy = band_qy[q];
    uint off = band_flat[q];
    float nr[B], ni[B], dd[B];
    for (int b = 0; b < B; ++b) { nr[b] = 0.0f; ni[b] = 0.0f; dd[b] = 0.0f; }
    uint k0 = chunk * K_CHUNK;
    uint k1 = metal::min(uint(NUM_BF), k0 + uint(K_CHUNK));
    for (uint k = k0; k < k1; ++k) {
        float kxv = kx[k], kyv = ky[k];
        // geometry of k, q - k, q + k once per (k, q), shared by every parameter set
        float4 gk = float4(probe_geometry[k * 4], probe_geometry[k * 4 + 1],
                           probe_geometry[k * 4 + 2], probe_geometry[k * 4 + 3]);
        float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x);
        float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;      // no double overlap: gamma = 0
        auto g = G[(size_t)k * (size_t)PLANE + (size_t)off];
        float gre = g.real, gim = g.imag;
        for (int b = 0; b < B; ++b) {
            if (base + b >= TOTAL) break;
            float C10 = trial[(base + b) * 7 + 0], C12 = trial[(base + b) * 7 + 1];
            float c2 = trial[(base + b) * 7 + 2], s2 = trial[(base + b) * 7 + 3];
            float t = trial[(base + b) * 7 + 4], thr = trial[(base + b) * 7 + 5], thc = trial[(base + b) * 7 + 6];
            float chi_k = probe_chi[k * TOTAL + base + b];
            // A zero aperture contributes exactly zero. Retain precise range
            // reduction for each active side, including soft aperture edges.
            float b1 = 0.0f, b2 = 0.0f, c1 = 0.0f, cp = 0.0f, s1 = 0.0f, sp = 0.0f;
            // Depth spread 0 is standard SSB: both depth weights are exactly 1 (tf_sinc(0) = 1), so the sinc and
            // the tilt shift, which only enters the sinc, are skipped.
            bool thick = t != 0.0f;
            if (a1 != 0.0f) {
                float chi_m = factor * gm.x * (C12 * (gm.y * c2 + gm.z * s2) + C10);
                s1 = metal::precise::sincos(chi_m - chi_k, c1);
                b1 = a1;
                if (thick) {
                    float shift = 6.283185307179586f * (qx * thr + qy * thc);
                    b1 = tf_sinc(0.5f * t * (-factor * (gm.x - gk.x) - shift)) * a1;
                }
            }
            if (a2 != 0.0f) {
                float chi_p = factor * gp.x * (C12 * (gp.y * c2 + gp.z * s2) + C10);
                sp = metal::precise::sincos(chi_p - chi_k, cp);
                b2 = a2;
                if (thick) {
                    float shift = 6.283185307179586f * (qx * thr + qy * thc);
                    b2 = tf_sinc(0.5f * t * (factor * (gp.x - gk.x) - shift)) * a2;
                }
            }
            float gr = b1 * c1 - b2 * cp;
            float gi = -b1 * s1 - b2 * sp;
            nr[b] += gre * gr + gim * gi;        // G conj(gamma)
            ni[b] += gim * gr - gre * gi;
            dd[b] += gr * gr + gi * gi;
        }
    }
    for (int b = 0; b < B; ++b) {
        if (base + b >= TOTAL) break;
        size_t o = ((size_t)chunk * TOTAL + base + b) * N_BAND + q;
        numer_real[o] = nr[b];
        numer_imag[o] = ni[b];
        denom[o] = dd[b];
    }
"""


@lru_cache(maxsize=1)
def _thick_fit_batch_kernel():
    """Metal kernel: per band q (one thread) and chunk of ``_K_CHUNK`` BF pixels, the partial sums of G conj(gamma) and
    |gamma|^2 for up to ``THICK_FIT_MAX_BATCH`` parameter sets. Mirrors CUDA ``thick_fit_batch_kernel``; the chunk partials
    are written out and summed afterwards instead of atomically added, so the result is deterministic.
    Grid z schedules independent B-candidate tiles within the TOTAL candidates;
    the per-thread arrays retain their small execution-tile size.
    """
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="ssb_thick_fit_batch",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky",
                     "trial", "scalars", "probe_geometry", "probe_chi"],
        output_names=["numer_real", "numer_imag", "denom"],
        source=_THICK_FIT_BATCH_SOURCE,
        header=_THICK_FIT_BATCH_HEADER,
    )


def _band(prepared: PreparedMpsSSB, band_inv_A: tuple[float, float]):
    """Band q on the stored half-plane: flat index into a G plane, (qx, qy) and the mirror weight; cached on ``prepared``.

    The term at -q equals the term at q (gamma(k, -q) = -conj gamma(k, q) for any probe phase with the first term at
    k - q, higher orders included, the depth weights swap under q -> -q, and G(-q) = conj G(q)), so the half-plane sum
    with weight 2 equals the full-plane sum; columns 0 and nx/2 are their own mirror column and count once.
    """
    key = (float(band_inv_A[0]), float(band_inv_A[1]))
    if prepared.thick_band is not None and prepared.thick_band[0] == key:
        return prepared.thick_band[1]
    mx = prepared.mx
    _, nx = (int(size) for size in prepared.scan_shape)
    cols = int(prepared.g_qk.shape[-1])
    half = cols != nx
    q_row = np.asarray(prepared.q_row, dtype=np.float32).reshape(-1)
    q_col = np.asarray(prepared.q_col, dtype=np.float32).reshape(-1)[:cols]
    q = np.hypot(q_row[:, None], q_col[None, :])
    rows, columns = np.nonzero((q > key[0]) & (q < key[1]))
    if rows.size == 0:
        raise ValueError(
            f"The model-agreement fit needs spatial frequencies in {key[0]:g}–{key[1]:g} 1/A, "
            f"but this scan reaches only {float(q.max()):.3g} 1/A. "
            "Use the phase-variance search (tilt=False, objective='phase_variance'), or acquire a finer scan. "
            "Reconstruction upsampling does not add measured frequencies."
        )
    if half:
        self_mirror = (columns == 0) | ((nx % 2 == 0) & (columns == nx // 2))
        weight = np.where(self_mirror, 1.0, 2.0)
    else:
        weight = np.ones(rows.shape)
    band = (
        mx.array((rows * cols + columns).astype(np.uint32)),
        mx.array(q_row[rows]),
        mx.array(q_col[columns]),
        weight,
    )
    prepared.thick_band = (key, band)
    return band


def _band_mask(prepared: PreparedMpsSSB, band_inv_A: tuple[float, float]) -> np.ndarray:
    q_row = np.asarray(prepared.q_row, dtype=np.float32)[:, None]
    q_col = np.asarray(prepared.q_col, dtype=np.float32)[None, :]
    q = np.hypot(q_row, q_col)
    return (q > float(band_inv_A[0])) & (q < float(band_inv_A[1]))


__all__ = ["reconstruct_thick", "reconstruct_thick_reference", "thick_fit", "thick_fit_batch"]
