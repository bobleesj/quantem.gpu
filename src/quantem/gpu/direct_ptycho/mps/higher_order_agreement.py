"""Model agreement of the thick-sample SSB model with higher-order aberrations, on MPS.

Same objective as ``thick_sample.thick_fit_batch`` (the least-squares agreement of the SSB model with G(q, k) over a
band of q, larger is better), with the probe phase chi evaluated from the 14-term Krivanek polynomial of
``higher_order`` instead of C10 and C12 only. The fit's angle-sum kernel (``resident_fit``) depends on chi being a
quadratic form, which coma, trefoil and spherical aberration are not, so this is the per-term kernel: chi at k - q
and k + q per (k, q, trial), chi at k once per (k, trial).

Trials are rows of ``(thickness_A, tilt_row_mrad, tilt_col_mrad, z_1 ... z_{2 N})`` where the z are the first ``N``
Krivanek terms (``higher_order.ORDERS`` order) as Cartesian vectors, Angstrom: ``(C cos m phi, C sin m phi)`` for
each term, ``(C, 0)`` for a round term (C10, C30). The vector form is smooth through C = 0, where the azimuth is
undefined, so it is also the coordinate the Newton refinement steps in.
"""

from functools import lru_cache

import numpy as np

from quantem.gpu.direct_ptycho._execution import checkpoint
from quantem.gpu.direct_ptycho.higher_order_fit import num_terms, packed
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx
from quantem.gpu.direct_ptycho.mps.thick_sample import (
    _FIT_DISPATCH_BATCH,
    _K_CHUNK,
    _THICK_FIT_BATCH_HEADER,
    _band,
)

# Execution tile: trials per thread. Larger tiles reuse the (k, q) geometry more but raise register pressure.
_TILE = 4
# Queue of partial sums before a readback, as in ``thick_sample``.
_PENDING_BYTES = 64 << 20

_CHI_HEADER = """
// chi(row, col) / (2 pi / wl) of the first NT Krivanek terms; c holds 2 NT Cartesian coefficients already divided by
// (n + 1), as higher_order.pack_coefficients packs them.
template <int NT>
inline float ho_chi(float row, float col, float wl, const device float* c) {
    float x = row * wl, y = col * wl;
    float r2 = x * x + y * y;
    float radial[4] = {1.0f, r2, r2 * r2, r2 * r2 * r2};
    float re[7], im[7];
    re[0] = 1.0f; im[0] = 0.0f;
    for (int m = 1; m <= 6; ++m) {
        re[m] = re[m - 1] * x - im[m - 1] * y;
        im[m] = im[m - 1] * x + re[m - 1] * y;
    }
    const int azimuth[14] = {0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6};
    const int power[14] = {1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0};
    float chi = 0.0f;
    for (int i = 0; i < NT; ++i) {
        chi += radial[power[i]] * (re[azimuth[i]] * c[2 * i] + im[azimuth[i]] * c[2 * i + 1]);
    }
    return (6.283185307179586f / wl) * chi;
}
"""

_PROBE_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= NUM_BF) return;
    float4 gk = tf_geometry(kx[k], ky[k], scalars[0], scalars[1], scalars[2], scalars[3]);
    for (uint i = 0; i < 4; ++i) probe_geometry[k * 4 + i] = gk[i];
    for (int b = 0; b < TOTAL; ++b) {
        probe_chi[k * TOTAL + b] = ho_chi<NT>(kx[k], ky[k], scalars[0], trial + b * STRIDE + 3);
    }
"""

_AGREEMENT_SOURCE = """
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
            const device float* row = trial + (base + b) * STRIDE;
            float t = row[0], thr = row[1], thc = row[2];
            float chi_k = probe_chi[k * TOTAL + base + b];
            float b1 = 0.0f, b2 = 0.0f, c1 = 0.0f, cp = 0.0f, s1 = 0.0f, sp = 0.0f;
            bool thick = t != 0.0f;
            float shift = 6.283185307179586f * (qx * thr + qy * thc);
            if (a1 != 0.0f) {
                // chi(k - q), not chi(q - k): they differ for the odd terms (coma, trefoil). G(q, k) ~ P(k - q) conj P(k)
                // - P(k) conj P(k + q) for a thin object (Pennycook et al. 2015); chi(q - k) breaks G(-q) = conj G(q).
                float chi_m = ho_chi<NT>(kxv - qx, kyv - qy, wl, row + 3);
                s1 = metal::precise::sincos(chi_m - chi_k, c1);
                b1 = thick ? tf_sinc(0.5f * t * (-factor * (gm.x - gk.x) - shift)) * a1 : a1;
            }
            if (a2 != 0.0f) {
                float chi_p = ho_chi<NT>(qx + kxv, qy + kyv, wl, row + 3);
                sp = metal::precise::sincos(chi_p - chi_k, cp);
                b2 = thick ? tf_sinc(0.5f * t * (factor * (gp.x - gk.x) - shift)) * a2 : a2;
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
def _probe_kernel():
    return require_mlx().fast.metal_kernel(
        name="ssb_higher_order_probe",
        input_names=["kx", "ky", "scalars", "trial"],
        output_names=["probe_geometry", "probe_chi"],
        header=_THICK_FIT_BATCH_HEADER + _CHI_HEADER,
        source=_PROBE_SOURCE,
    )


@lru_cache(maxsize=1)
def _agreement_kernel():
    return require_mlx().fast.metal_kernel(
        name="ssb_higher_order_agreement",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky", "trial", "scalars", "probe_geometry",
                     "probe_chi"],
        output_names=["numer_real", "numer_imag", "denom"],
        header=_THICK_FIT_BATCH_HEADER + _CHI_HEADER,
        source=_AGREEMENT_SOURCE,
    )


def agreement_batch(prepared, rows, band_inv_A: tuple[float, float] = (0.2, 0.9)) -> np.ndarray:
    """The thick-sample model agreement for each trial row (see the module docstring); larger is better.

    Equals ``thick_sample.thick_fit_batch`` at the same C10, C12, phi12, tilt and depth spread when the higher-order
    coefficients are 0 (the same model, chi evaluated by the polynomial instead of alpha^2 (C10 + C12 cos 2 phi)).
    """
    mx = prepared.mx
    trial = packed(rows)
    terms = num_terms(trial)
    stride = trial.shape[1]
    flat, band_qx, band_qy, weight = _band(prepared, band_inv_A)
    n_band = int(flat.size)
    g = prepared.g_qk
    storage_bf = int(g.shape[0])
    plane = int(g.shape[1]) * int(g.shape[2])
    k_chunks = (storage_bf + _K_CHUNK - 1) // _K_CHUNK
    scalars = mx.array([prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad, prepared.ang_x_rad,
                        prepared.factor], dtype=mx.float32)
    per_candidate = (k_chunks + 1) * n_band * 3 * 4
    execution = max(_FIT_DISPATCH_BATCH, (_PENDING_BYTES // per_candidate // _TILE) * _TILE)
    out = np.empty(len(trial))
    for start in range(0, len(trial), execution):
        checkpoint()
        rows_np = trial[start:start + execution]
        batch = len(rows_np)
        tile = min(_TILE, batch)
        rows_mx = mx.array(rows_np.ravel())
        probe_geometry, probe_chi = _probe_kernel()(
            inputs=[prepared.kx, prepared.ky, scalars, rows_mx],
            template=[("NUM_BF", storage_bf), ("TOTAL", batch), ("STRIDE", stride), ("NT", terms)],
            grid=(storage_bf, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(storage_bf, 4), (storage_bf, batch)],
            output_dtypes=[mx.float32, mx.float32],
        )
        numer_real, numer_imag, denom = _agreement_kernel()(
            inputs=[g, flat, band_qx, band_qy, prepared.kx, prepared.ky, rows_mx, scalars, probe_geometry, probe_chi],
            template=[("B", tile), ("TOTAL", batch), ("N_BAND", n_band), ("NUM_BF", storage_bf), ("PLANE", plane),
                      ("K_CHUNK", _K_CHUNK), ("STRIDE", stride), ("NT", terms)],
            grid=(n_band, k_chunks, (batch + tile - 1) // tile),
            threadgroup=(64, 1, 1),
            output_shapes=[(k_chunks, batch, n_band)] * 3,
            output_dtypes=[mx.float32] * 3,
        )
        sums = mx.stack([mx.sum(numer_real, axis=0), mx.sum(numer_imag, axis=0), mx.sum(denom, axis=0)])
        values = np.asarray(sums).astype(np.float64)
        den = values[2]
        ok = den > 0
        power = np.where(ok, (values[0] ** 2 + values[1] ** 2) / np.where(ok, den, 1.0), 0.0)
        out[start:start + batch] = (weight[None] * power).sum(axis=1)
    return out


def band_power(prepared, band_inv_A: tuple[float, float] = (0.2, 0.9)) -> dict[str, float]:
    """G's power over the agreement band, sum_q w_q sum_k |G(q, k)|^2 (the agreement's upper bound), the band's weight
    sum_q w_q, and the noise variance of one G(q, k) for counted electrons: the mean over k of G(0, k), the counts at k
    summed over the scan (Poisson)."""
    mx = prepared.mx
    flat, _, _, weight = _band(prepared, band_inv_A)
    g = prepared.g_qk
    columns = mx.take(g.reshape(g.shape[0], -1), mx.array(np.asarray(flat)), axis=1)
    per_q = np.asarray(mx.sum(mx.real(columns) ** 2 + mx.imag(columns) ** 2, axis=0)).astype(np.float64)
    w = np.asarray(weight, dtype=np.float64)
    counts = np.asarray(mx.real(g[:, 0, 0])).astype(np.float64)
    return {"power": float((w * per_q).sum()), "band_weight": float(w.sum()), "noise_variance": float(counts.mean()),
            "num_bf": len(counts)}


__all__ = ["agreement_batch", "band_power"]
