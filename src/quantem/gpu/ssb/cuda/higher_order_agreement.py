"""Model agreement of the thick-sample SSB model with higher-order aberrations, on CUDA.

The CUDA twin of ``mps.higher_order_agreement``: the joint fit's least-squares agreement (``thick_fit_batch_kernel``,
larger is better) with the probe phase chi taken from the first ``N`` Krivanek terms of ``higher_order`` instead of
C10 and C12, so ``find_aberrations(higher_order=...)`` can fit coma, trefoil and spherical aberration. chi is evaluated
at k - q and k + q per (q, k, trial) and at k once per (k, trial) and block; the batch kernel's angle-sum form needs a
quadratic chi, which these terms are not.

Rows are ``(thickness_A, tilt_row_mrad, tilt_col_mrad, z_1 ... z_{2 N})`` as in the MPS kernel (``higher_order_fit``
packs them). The sum runs over the joint fit's band on the stored half plane with its mirror weights: with the first
term at k - q, gamma(k, -q) = -conj gamma(k, q) for any probe phase, so the term at -q equals the term at q here too.
Each block sums its slice of bright-field pixels into its own partial, and the slices are added in order afterwards,
so a value does not change from run to run.
"""

from functools import cache

import cupy as cp
import numpy as np

from quantem.gpu.ssb.higher_order_fit import num_terms, packed

# Trials per thread (registers) and bright-field pixels per block, as in the joint fit's batch kernel.
_TILE = 4
_K_CHUNK = 256
# Bytes of partial sums held at once; more trials run in further passes over G.
_PARTIAL_BYTES = 256 << 20

_SOURCE = r"""
#define NT __NT__
#define TILE __TILE__
#define K_CHUNK __K_CHUNK__
// The soft aperture and alpha^2 of a probe tilt (dx, dy), as thick_fit_batch's tf_geometry (.x = alpha^2, .w = aperture).
__device__ __forceinline__ float4 ho_geometry(float dx, float dy, float wl, float semiangle, float ang_y, float ang_x,
                                              float inv_ang) {
    float r2 = dx * dx + dy * dy;
    float wl2 = wl * wl;
    float edge = 1.0f;
    if (r2 > 1e-30f) {
        float r = sqrtf(r2);
        float excess = semiangle - r * wl;
        edge = inv_ang > 0.0f ? fmaf(excess, inv_ang, 0.5f)
                              : excess * r / sqrtf(dx * ang_y * dx * ang_y + dy * ang_x * dy * ang_x) + 0.5f;
    }
    return make_float4(r2 * wl2, (dx * dx - dy * dy) * wl2, 2.0f * dx * dy * wl2, fminf(fmaxf(edge, 0.0f), 1.0f));
}
__device__ __forceinline__ float ho_sinc(float x) { return (fabsf(x) < 1e-6f) ? 1.0f : sinf(x) / x; }
// chi(row, col) of the first NT Krivanek terms; c holds 2 NT Cartesian coefficients already divided by (n + 1).
__device__ __forceinline__ float ho_chi(float row, float col, float wl, const float* c) {
    const int azimuth[14] = {0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6};
    const int power[14] = {1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0};
    float x = row * wl, y = col * wl;
    float r2 = x * x + y * y;
    float radial[4] = {1.0f, r2, r2 * r2, r2 * r2 * r2};
    float re[7], im[7];
    re[0] = 1.0f; im[0] = 0.0f;
#pragma unroll
    for (int m = 1; m <= 6; ++m) {
        re[m] = re[m - 1] * x - im[m - 1] * y;
        im[m] = im[m - 1] * x + re[m - 1] * y;
    }
    float chi = 0.0f;
#pragma unroll
    for (int i = 0; i < NT; ++i) chi += radial[power[i]] * (re[azimuth[i]] * c[2 * i] + im[azimuth[i]] * c[2 * i + 1]);
    return (6.283185307179586f / wl) * chi;
}
extern "C" __global__ void __launch_bounds__(256) ho_agreement(
    const float2* __restrict__ G, const long long* __restrict__ flat, const float* __restrict__ qxb,
    const float* __restrict__ qyb, const float* __restrict__ kx, const float* __restrict__ ky,
    const float* __restrict__ trial, float* __restrict__ numer_re, float* __restrict__ numer_im,
    float* __restrict__ denom, int num_bf, long long plane, int n_band, int total, int stride,
    float wl, float semiangle, float ang_y, float ang_x, float factor)
{
    __shared__ float rows[TILE * (3 + 2 * NT)];
    __shared__ float4 geometry_k[K_CHUNK];
    __shared__ float chi_k[TILE * K_CHUNK];
    int base = blockIdx.z * TILE;
    for (int i = threadIdx.x; i < TILE * stride; i += blockDim.x) {
        int b = i / stride;
        rows[i] = base + b < total ? trial[(size_t)(base + b) * stride + i % stride] : 0.0f;
    }
    __syncthreads();
    float inv_ang = ang_y == ang_x ? 1.0f / ang_x : 0.0f;
    int k0 = blockIdx.y * K_CHUNK, k1 = min(num_bf, k0 + K_CHUNK);
    // the probe at k is the same for every q of the block: its geometry and chi(k) per trial, once
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
        geometry_k[k - k0] = ho_geometry(kx[k], ky[k], wl, semiangle, ang_y, ang_x, inv_ang);
#pragma unroll
        for (int b = 0; b < TILE; ++b) chi_k[b * K_CHUNK + k - k0] = ho_chi(kx[k], ky[k], wl, rows + b * stride + 3);
    }
    __syncthreads();
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_band) return;
    float qx = qxb[q], qy = qyb[q];
    long long off = flat[q];
    float nr[TILE], ni[TILE], dd[TILE], shift[TILE], half_t[TILE];
#pragma unroll
    for (int b = 0; b < TILE; ++b) {
        nr[b] = 0.0f; ni[b] = 0.0f; dd[b] = 0.0f;
        half_t[b] = 0.5f * rows[b * stride];
        shift[b] = 6.283185307179586f * (qx * rows[b * stride + 1] + qy * rows[b * stride + 2]);
    }
    for (int k = k0; k < k1; ++k) {
        float kxv = kx[k], kyv = ky[k];
        float4 gk = geometry_k[k - k0];
        float4 gm = ho_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x, inv_ang);
        float4 gp = ho_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x, inv_ang);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;          // no double overlap: gamma = 0
        float2 g = G[(long long)k * plane + off];
#pragma unroll
        for (int b = 0; b < TILE; ++b) {
            if (base + b >= total) break;
            const float* c = rows + b * stride + 3;
            float ck = chi_k[b * K_CHUNK + k - k0];
            float b1 = 0.0f, b2 = 0.0f, c1 = 0.0f, s1 = 0.0f, cp = 0.0f, sp = 0.0f;
            if (a1 != 0.0f) {
                // chi(k - q), not chi(q - k): G(q, k) ~ P(k - q) conj P(k) - P(k) conj P(k + q) for a thin object
                sincosf(ho_chi(kxv - qx, kyv - qy, wl, c) - ck, &s1, &c1);
                b1 = half_t[b] != 0.0f ? ho_sinc((-factor * (gm.x - gk.x) - shift[b]) * half_t[b]) * a1 : a1;
            }
            if (a2 != 0.0f) {
                sincosf(ho_chi(qx + kxv, qy + kyv, wl, c) - ck, &sp, &cp);
                b2 = half_t[b] != 0.0f ? ho_sinc((factor * (gp.x - gk.x) - shift[b]) * half_t[b]) * a2 : a2;
            }
            float gr = b1 * c1 - b2 * cp;          // gamma = b1 exp(-i(chi_m - chi_k)) - b2 exp(i(chi_p - chi_k))
            float gi = -b1 * s1 - b2 * sp;
            nr[b] += g.x * gr + g.y * gi;          // G conj(gamma)
            ni[b] += g.y * gr - g.x * gi;
            dd[b] += gr * gr + gi * gi;
        }
    }
#pragma unroll
    for (int b = 0; b < TILE; ++b) {
        if (base + b >= total) break;
        size_t slot = ((size_t)blockIdx.y * total + base + b) * n_band + q;
        numer_re[slot] = nr[b];
        numer_im[slot] = ni[b];
        denom[slot] = dd[b];
    }
}
"""


@cache
def _kernel(terms: int):
    source = (_SOURCE.replace("__NT__", str(terms)).replace("__TILE__", str(_TILE))
              .replace("__K_CHUNK__", str(_K_CHUNK)))
    return cp.RawKernel(source, "ho_agreement")


def agreement_batch(engine, rows, band_inv_A: tuple[float, float] = (0.2, 0.9)) -> np.ndarray:
    """The thick-sample model agreement for each row (see the module docstring); larger is better.

    ``engine`` is a CUDA ``SSBEngine`` at its current rotation. Equals ``ThickSample.fit_batch`` at the same C10, C12,
    phi12, tilt and depth spread when the higher-order coefficients are 0, and ``mps.higher_order_agreement`` on the
    same evidence.
    """
    trial = packed(rows)
    terms = num_terms(trial)
    stride = trial.shape[1]
    cache = engine._cache
    num_bf, _, _ = engine._bf_grid()
    _, _, flat, qx, qy, weight = engine.thick._prepare_fit_band(band_inv_A)
    n_band = int(flat.size)
    k_blocks = (num_bf + _K_CHUNK - 1) // _K_CHUNK
    plane = int(engine.G_qk.shape[1] * engine.G_qk.shape[2])
    per_trial = k_blocks * n_band * 3 * 4
    execution = max(_TILE, (_PARTIAL_BYTES // per_trial) // _TILE * _TILE)
    scalars = (np.float32(engine.wavelength), np.float32(cache["semiangle_rad"]), np.float32(cache["ang_y_rad"]),
               np.float32(cache["ang_x_rad"]), np.float32(engine._factor))
    qx, qy = qx.ravel(), qy.ravel()
    out = np.empty(len(trial))
    for start in range(0, len(trial), execution):
        chunk = cp.asarray(trial[start:start + execution])
        batch = len(chunk)
        numer_re = cp.empty((k_blocks, batch, n_band), dtype=cp.float32)
        numer_im = cp.empty_like(numer_re)
        denom = cp.empty_like(numer_re)
        _kernel(terms)(
            ((n_band + 255) // 256, k_blocks, (batch + _TILE - 1) // _TILE), (256,),
            (engine.G_qk, flat, qx, qy, cache["kx_bf"], cache["ky_bf"], chunk.ravel(), numer_re, numer_im, denom,
             np.int32(num_bf), np.int64(plane), np.int32(n_band), np.int32(batch), np.int32(stride), *scalars),
        )
        re, im, den = (cp.sum(part, axis=0).astype(cp.float64) for part in (numer_re, numer_im, denom))
        ok = den > 0
        power = cp.where(ok, (re * re + im * im) / cp.where(ok, den, 1.0), 0.0)
        out[start:start + batch] = cp.asnumpy((weight[None].astype(cp.float64) * power).sum(axis=1))
    return out


def band_power(engine, band_inv_A: tuple[float, float] = (0.2, 0.9)) -> dict[str, float]:
    """G's power over the agreement band, the band's weight and the Poisson noise variance (``mps.higher_order_agreement``)."""
    num_bf, _, _ = engine._bf_grid()
    _, _, flat, _, _, weight = engine.thick._prepare_fit_band(band_inv_A)
    g = engine.G_qk.reshape(num_bf, -1)
    per_q = cp.zeros(flat.size, dtype=cp.float64)
    for start in range(0, num_bf, 256):
        columns = g[start:start + 256][:, flat]
        per_q += (columns.real.astype(cp.float64) ** 2 + columns.imag.astype(cp.float64) ** 2).sum(axis=0)
    w = weight.astype(cp.float64)
    counts = engine.G_qk[:, 0, 0].real.astype(cp.float64)
    return {"power": float((w * per_q).sum()), "band_weight": float(w.sum()), "noise_variance": float(counts.mean()),
            "num_bf": int(num_bf)}


__all__ = ["agreement_batch", "band_power"]
