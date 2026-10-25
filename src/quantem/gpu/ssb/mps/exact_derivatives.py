"""Exact derivatives of the MPS SSB agreement for the trust-region fits (``cuda/exact_derivatives.py`` on Metal).

The agreement S = sum_q w_q |N_q|^2 / D_q and its derivatives follow the CUDA module's docstring: variable projection
with a_q = N_q / D_q and R = conj(a_q) (G - a_q gamma) gives dS/dtheta = sum 2 w Re(dgamma/dtheta conj R); the
low-order Hessian adds sum_q (2 w_q / D_q) Re(conj(V_j) V_i) to the per-pair terms. The value passes are the MPS
production kernels (``fifth_order.Objective``, ``resident_fit.band_sums``) with the band sum in float64 on the host;
the derivative passes repeat their per-pair arithmetic in float32 (precise sincos, as the objectives), reduce each SIMD
group on the GPU and sum the groups in float64 on the host.
"""

from functools import lru_cache

import numpy as np

from quantem.gpu.ssb._exact_newton import joint_trial
from quantem.gpu.ssb.higher_order import CHI_SOURCE
from quantem.gpu.ssb.mps import fifth_order
from quantem.gpu.ssb.mps.hardware import require_mlx
from quantem.gpu.ssb.mps.resident_fit import _probe_geometry, band_sums
from quantem.gpu.ssb.mps.thick_sample import _K_CHUNK, _THICK_FIT_BATCH_HEADER, _band

_HEADER = (
    "#define SSB_DEVICE inline\n#define SSB_COEFFICIENTS const device float*\n"
    + _THICK_FIT_BATCH_HEADER + CHI_SOURCE + """
inline float fg_sinc_slope(float x, thread float& sinc) {
    sinc = tf_sinc(x);
    float x2 = x * x;
    if (metal::abs(x) < 0.25f) return x * (-1.0f / 3.0f + x2 * (1.0f / 30.0f - x2 * (1.0f / 840.0f)));
    float c;
    float s = metal::precise::sincos(x, c);
    (void)s;
    return (c - sinc) / x;
}
constant float FG_BINOMIAL[7][7] = {
    {1, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 2, 1, 0, 0, 0, 0}, {1, 3, 3, 1, 0, 0, 0},
    {1, 4, 6, 4, 1, 0, 0}, {1, 5, 10, 10, 5, 1, 0}, {1, 6, 15, 20, 15, 6, 1}};
inline void ld_sinc(float x, thread float& f0, thread float& f1, thread float& f2) {
    f0 = tf_sinc(x);
    float x2 = x * x;
    if (metal::abs(x) < 1.0f) {
        float g = 1.0f / 3.0f + x2 * (-1.0f / 30.0f + x2 * (1.0f / 840.0f + x2 * (-1.0f / 45360.0f
                + x2 * (1.0f / 3991680.0f))));
        f1 = -x * g;
        f2 = 2.0f * g - f0;
        return;
    }
    float c;
    float s = metal::precise::sincos(x, c);
    (void)s;
    float inv = 1.0f / x;
    f1 = (c - f0) * inv;
    f2 = -(f0 + 2.0f * f1 * inv);
}
"""
)

# Grid (band frequency, BF chunk, candidate); one SIMD group of 32 band frequencies writes its 28 sums:
# [dS/d depth spread (A), dS/d tilt row, dS/d tilt column (rad), T_a for 2 <= |a| <= 6] (``Coordinates.gradient``).
_FIFTH_GRADIENT_SOURCE = """
    uint q = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    bool valid = q < N_BAND;
    uint qi = valid ? q : 0u;
    float wl = scalars[0], semiangle = scalars[1], ang_y = scalars[2], ang_x = scalars[3], factor = scalars[4];
    float qx = band_qx[qi], qy = band_qy[qi];
    uint off = band_flat[qi];
    size_t si = ((size_t)b * N_BAND + qi) * 4;
    float ar = scale[si], ai = scale[si + 1], aa = scale[si + 2], w2 = valid ? scale[si + 3] : 0.0f;
    bool active = valid && w2 != 0.0f;
    const device float* row = trial + b * 31;
    float t = row[0], thr = row[1], thc = row[2];
    bool thick = t != 0.0f;
    float shift = 6.283185307179586f * (qx * thr + qy * thc);
    float even[15], odd[21];
    for (int j = 0; j < 15; ++j) even[j] = 0.0f;
    for (int j = 0; j < 21; ++j) odd[j] = 0.0f;
    float depth_sum = 0.0f, tilt_sum = 0.0f;
    if (active) {
        uint k0 = chunk * K_CHUNK, k1 = metal::min(uint(NUM_BF), k0 + uint(K_CHUNK));
        for (uint k = k0; k < k1; ++k) {
            float kxv = kx[k], kyv = ky[k];
            float4 gk = float4(probe_geometry[k * 4], probe_geometry[k * 4 + 1],
                               probe_geometry[k * 4 + 2], probe_geometry[k * 4 + 3]);
            float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x);
            float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x);
            float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
            if (a1 == 0.0f && a2 == 0.0f) continue;
            auto g = G[(size_t)k * (size_t)PLANE + (size_t)off];
            float gre = g.real, gim = g.imag;
            float chi_k = probe_chi[k * TOTAL + b];
            float b1 = 0.0f, b2 = 0.0f, c1 = 0.0f, cp = 0.0f, s1 = 0.0f, sp = 0.0f;
            float d1 = 0.0f, d2 = 0.0f, h1 = 0.0f, h2 = 0.0f;
            if (a1 != 0.0f) {
                s1 = metal::precise::sincos(ssb_chi_full(kxv - qx, kyv - qy, wl, row + 3) - chi_k, c1);
                b1 = a1;
                if (thick) {
                    float sinc;
                    h1 = -factor * (gm.x - gk.x) - shift;
                    d1 = fg_sinc_slope(0.5f * t * h1, sinc) * a1;
                    b1 = sinc * a1;
                }
            }
            if (a2 != 0.0f) {
                sp = metal::precise::sincos(ssb_chi_full(qx + kxv, qy + kyv, wl, row + 3) - chi_k, cp);
                b2 = a2;
                if (thick) {
                    float sinc;
                    h2 = factor * (gp.x - gk.x) - shift;
                    d2 = fg_sinc_slope(0.5f * t * h2, sinc) * a2;
                    b2 = sinc * a2;
                }
            }
            float gr = b1 * c1 - b2 * cp;
            float gi = -b1 * s1 - b2 * sp;
            float rr = ar * gre + ai * gim - aa * gr;
            float ri = ar * gim - ai * gre - aa * gi;
            float alpha1 = -b1 * (c1 * ri + s1 * rr);
            float alpha2 = b2 * (sp * rr - cp * ri);
            float beta1 = c1 * rr - s1 * ri;
            float beta2 = -(cp * rr + sp * ri);
            tilt_sum += beta1 * d1 + beta2 * d2;
            depth_sum += beta1 * d1 * h1 + beta2 * d2 * h2;
            float e = alpha1 + alpha2, o = alpha2 - alpha1;
            const device float* mp = monomials + (size_t)k * 21;
            for (int j = 0; j < 15; ++j) even[j] = metal::fma(e, mp[j], even[j]);
            for (int j = 0; j < 21; ++j) odd[j] = metal::fma(o, mp[j], odd[j]);
        }
    }
    float qxp[7], qyp[7];
    qxp[0] = 1.0f; qyp[0] = 1.0f;
    for (int i = 1; i < 7; ++i) { qxp[i] = qxp[i - 1] * qx; qyp[i] = qyp[i - 1] * qy; }
    float w = active ? w2 : 0.0f;
    float out[28];
    out[0] = w * 0.5f * depth_sum;
    out[1] = w * (-3.141592653589793f) * t * qx * tilt_sum;
    out[2] = w * (-3.141592653589793f) * t * qy * tilt_sum;
    for (int d = 2; d <= 6; ++d) {
        for (int cc = 0; cc <= d; ++cc) {
            int a = d - cc;
            float total = 0.0f;
            for (int a2 = 0; a2 <= a; ++a2) {
                for (int c2 = 0; c2 <= cc; ++c2) {
                    int order = (a - a2) + (cc - c2);
                    if (order == 0) continue;
                    int index = (a2 + c2) * (a2 + c2 + 1) / 2 + c2;
                    float moment = (order & 1) ? odd[index] : even[metal::min(index, 14)];
                    total = metal::fma(FG_BINOMIAL[a][a2] * FG_BINOMIAL[cc][c2] * qxp[a - a2] * qyp[cc - c2],
                                       moment, total);
                }
            }
            out[d * (d + 1) / 2 + cc] = w * total;
        }
    }
    size_t o = (((size_t)b * CHUNKS + chunk) * GROUPS + q / 32) * 28;
    for (int j = 0; j < 28; ++j) {
        float v = simd_sum(out[j]);
        if (lane == 0) partial[o + j] = v;
    }
"""

# Low-order derivatives (``cuda.exact_derivatives._LOW_SOURCE``): per (band frequency, BF chunk, candidate)
# N_i, D_i (18 floats per frequency) and per SIMD group the 27 linear sums (gradient, Hessian upper triangle).
_LOW_SOURCE = """
    uint q = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    bool valid = q < N_BAND;
    uint qi = valid ? q : 0u;
    float wl = scalars[0], semiangle = scalars[1], ang_y = scalars[2], ang_x = scalars[3], factor = scalars[4];
    float qx = band_qx[qi], qy = band_qy[qi];
    uint off = band_flat[qi];
    size_t si = ((size_t)b * N_BAND + qi) * 4;
    float ar = scale[si], ai = scale[si + 1], aa = scale[si + 2], wq = valid ? scale[si + 3] : 0.0f;
    bool active = valid && wq != 0.0f;
    auto tr = trial + b * 7;   // MLX passes a small input (one 7-float trial) in the constant address space
    float c10 = tr[0], ca = tr[1] * tr[2], cb = tr[1] * tr[3], t = tr[4];
    float shift = 6.283185307179586f * (qx * tr[5] + qy * tr[6]);
    float xr = -3.141592653589793e-3f * qx * t, xc = -3.141592653589793e-3f * qy * t;
    float xrt = -3.141592653589793e-3f * qx, xct = -3.141592653589793e-3f * qy;
    float nr[6], ni[6], dn[6], lg[6], lh[21];
    for (int i = 0; i < 6; ++i) { nr[i] = 0.0f; ni[i] = 0.0f; dn[i] = 0.0f; lg[i] = 0.0f; }
    for (int i = 0; i < 21; ++i) lh[i] = 0.0f;
    if (active) {
        uint k0 = chunk * K_CHUNK, k1 = metal::min(uint(NUM_BF), k0 + uint(K_CHUNK));
        for (uint k = k0; k < k1; ++k) {
            float kxv = kx[k], kyv = ky[k];
            float4 gk = float4(probe_geometry[k * 4], probe_geometry[k * 4 + 1],
                               probe_geometry[k * 4 + 2], probe_geometry[k * 4 + 3]);
            float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x);
            float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x);
            float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
            if (a1 == 0.0f && a2 == 0.0f) continue;
            auto g = G[(size_t)k * (size_t)PLANE + (size_t)off];
            float gx = g.real, gy = g.imag;
            float m[3] = {factor * (gm.x - gk.x), factor * (gm.y * gm.x - gk.y * gk.x),
                          factor * (gm.z * gm.x - gk.z * gk.x)};
            float p[3] = {factor * (gp.x - gk.x), factor * (gp.y * gp.x - gk.y * gk.x),
                          factor * (gp.z * gp.x - gk.z * gk.x)};
            float c1, cp;
            float s1 = metal::precise::sincos(m[0] * c10 + m[1] * ca + m[2] * cb, c1);
            float sp = metal::precise::sincos(p[0] * c10 + p[1] * ca + p[2] * cb, cp);
            float h1 = -m[0] - shift, h2 = p[0] - shift;
            float f10, f11, f12, f20, f21, f22;
            ld_sinc(0.5f * t * h1, f10, f11, f12);
            ld_sinc(0.5f * t * h2, f20, f21, f22);
            float b1 = a1 * f10, b2 = a2 * f20;
            float x1[3] = {xr, xc, 0.5f * h1}, x2[3] = {xr, xc, 0.5f * h2};
            float db1[3], db2[3];
            for (int i = 0; i < 3; ++i) { db1[i] = a1 * f11 * x1[i]; db2[i] = a2 * f21 * x2[i]; }
            float u1r = b1 * c1, u1i = -b1 * s1, u2r = b2 * cp, u2i = b2 * sp;
            float gr = u1r - u2r, gi = u1i - u2i;
            float rr = ar * gx + ai * gy - aa * gr;
            float ri = ar * gy - ai * gx - aa * gi;
            float dr[6], di[6];
            for (int i = 0; i < 3; ++i) {
                float ur = u1r * m[i] + u2r * p[i], ui = u1i * m[i] + u2i * p[i];
                dr[i] = ui; di[i] = -ur;
                dr[3 + i] = db1[i] * c1 - db2[i] * cp;
                di[3 + i] = -db1[i] * s1 - db2[i] * sp;
            }
            for (int i = 0; i < 6; ++i) {
                nr[i] += gx * dr[i] + gy * di[i];
                ni[i] += gy * dr[i] - gx * di[i];
                dn[i] += 2.0f * (gr * dr[i] + gi * di[i]);
                lg[i] += 2.0f * (dr[i] * rr + di[i] * ri);
            }
            int index = 0;
            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    float hr, hi;
                    if (j < 3) {
                        float mm = m[i] * m[j], pp = p[i] * p[j];
                        hr = -u1r * mm + u2r * pp;
                        hi = -u1i * mm + u2i * pp;
                    } else if (i < 3) {
                        float vr = db1[j - 3] * c1 * m[i] + db2[j - 3] * cp * p[i];
                        float vi = -db1[j - 3] * s1 * m[i] + db2[j - 3] * sp * p[i];
                        hr = vi; hi = -vr;
                    } else {
                        int u = i - 3, v = j - 3;
                        float mixed = (u == 0 && v == 2) ? xrt : (u == 1 && v == 2) ? xct : 0.0f;
                        float e1 = a1 * (f12 * x1[u] * x1[v] + f11 * mixed);
                        float e2 = a2 * (f22 * x2[u] * x2[v] + f21 * mixed);
                        hr = e1 * c1 - e2 * cp;
                        hi = -e1 * s1 - e2 * sp;
                    }
                    lh[index] += 2.0f * (hr * rr + hi * ri) - 2.0f * aa * (dr[i] * dr[j] + di[i] * di[j]);
                    ++index;
                }
            }
        }
    }
    if (valid) {
        size_t base = ((size_t)b * CHUNKS + chunk) * 18;
        for (int i = 0; i < 6; ++i) {
            nd[(base + i) * N_BAND + q] = nr[i];
            nd[(base + 6 + i) * N_BAND + q] = ni[i];
            nd[(base + 12 + i) * N_BAND + q] = dn[i];
        }
    }
    float w = active ? wq : 0.0f;
    size_t o = (((size_t)b * CHUNKS + chunk) * GROUPS + q / 32) * 27;
    for (int j = 0; j < 27; ++j) {
        float v = simd_sum(w * (j < 6 ? lg[j] : lh[j - 6]));
        if (lane == 0) linear[o + j] = v;
    }
"""


@lru_cache(maxsize=1)
def _fifth_gradient_kernel():
    return require_mlx().fast.metal_kernel(
        name="ssb_exact_fifth_gradient",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky", "probe_geometry", "probe_chi",
                     "monomials", "trial", "scalars", "scale"],
        output_names=["partial"],
        header=_HEADER,
        source=_FIFTH_GRADIENT_SOURCE,
    )


@lru_cache(maxsize=1)
def _low_kernel():
    return require_mlx().fast.metal_kernel(
        name="ssb_exact_low_order_derivatives",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky", "probe_geometry", "trial", "scalars",
                     "scale"],
        output_names=["nd", "linear"],
        header=_HEADER,
        source=_LOW_SOURCE,
    )


def _scale(numerators, denominators, weight, gradient_weight):
    """(Re a_q, Im a_q, |a_q|^2, gradient_weight * w_q) per candidate and band frequency, zero where D_q = 0."""
    covered = denominators > 0
    safe = np.where(covered, denominators, 1)
    ar = np.where(covered, numerators.real / safe, 0).astype(np.float32)
    ai = np.where(covered, numerators.imag / safe, 0).astype(np.float32)
    w = np.where(covered, gradient_weight * weight[None], 0).astype(np.float32)
    return np.ascontiguousarray(np.stack([ar, ai, ar * ar + ai * ai, w], axis=-1), dtype=np.float32)


def _power_sum(numerators, denominators, weight):
    """w |N|^2 / D per band frequency in float64 (0 where D = 0), summed over the band."""
    real = numerators.real.astype(np.float64)
    imag = numerators.imag.astype(np.float64)
    covered = denominators > 0
    den = np.where(covered, denominators, 1).astype(np.float64)
    return np.sum(np.asarray(weight, np.float64)[None] * np.where(covered, (real * real + imag * imag) / den, 0.0),
                  axis=1)


class FifthOrderGradients(fifth_order.Objective):
    """The MPS fifth-order agreement with exact gradients in the 28 dimensionless coordinates z.

    The same interface as ``cuda.exact_derivatives.FifthOrderGradients``: ``values(z, stride)`` with the
    per-frequency powers summed in float64, ``gradients(z, stride)`` at the points of the latest ``values`` call.
    """

    def __init__(self, prepared, partial_bytes: int = 64 << 20) -> None:
        super().__init__(prepared, partial_bytes)
        self._pixels_by_stride = {}
        self._last = None
        self.value_passes = 0
        self.gradient_passes = 0

    @property
    def num_bright_field_pixels(self) -> int:
        return int(self.p.g_qk.shape[0])

    def _pixels(self, stride):
        if stride not in self._pixels_by_stride:
            p = self.p
            mx = p.mx
            kx = np.asarray(p.kx, dtype=np.float32)[::stride].copy()
            ky = np.asarray(p.ky, dtype=np.float32)[::stride].copy()
            geometry = np.asarray(_probe_geometry(p, self._scalars), dtype=np.float32)[::stride].copy()
            powers_x, powers_y = [np.ones_like(kx)], [np.ones_like(ky)]
            for _ in range(5):
                powers_x.append(powers_x[-1] * kx)
                powers_y.append(powers_y[-1] * ky)
            monomials = np.stack([powers_x[d - c] * powers_y[c] for d in range(6) for c in range(d + 1)],
                                 axis=1).astype(np.float32)
            self._pixels_by_stride[stride] = (mx.array(kx), mx.array(ky), mx.array(geometry), mx.array(monomials),
                                              int(kx.size))
        return self._pixels_by_stride[stride]

    def _band_arrays_for(self, band):
        """The band frequencies on the device and their float32 weights on the host."""
        flat, qx, qy, weight = _band(self.p, tuple(float(value) for value in band))
        return flat, qx, qy, np.asarray(weight, dtype=np.float32)

    def _sums(self, rows, packed, stride, band):
        """``score``'s kernels on every ``stride``-th BF pixel: per-frequency numerators, denominators, chi(k)."""
        p = self.p
        mx = p.mx
        rows = np.atleast_2d(rows)
        coefficients = np.asarray(packed, dtype=np.float32).reshape(len(rows), 28)
        trial = np.empty((len(rows), 31), dtype=np.float32)
        trial[:, 0] = rows[:, 5]
        trial[:, 1:3] = rows[:, 3:5] * 1e-3
        trial[:, 3:] = coefficients
        flat, qx, qy, weight = self._band_arrays_for(band)
        kx, ky, geometry, _, nk = self._pixels(stride)
        n_band = int(flat.size)
        plane = int(p.g_qk.shape[1]) * int(p.g_qk.shape[2]) * stride
        chunks = (nk + _K_CHUNK - 1) // _K_CHUNK
        trial_mx = mx.array(trial.ravel())
        (chi,) = fifth_order._probe_kernel()(
            inputs=[kx, ky, self._scalars, trial_mx],
            template=[("NUM_BF", nk), ("TOTAL", len(trial)), ("STRIDE", 31), ("NT", 14)],
            grid=(nk, len(trial), 1), threadgroup=(256, 1, 1),
            output_shapes=[(nk, len(trial))], output_dtypes=[mx.float32],
        )
        tile = min(4, len(trial))
        real, imag, denominator = fifth_order._agreement_kernel()(
            inputs=[p.g_qk, flat, qx, qy, kx, ky, trial_mx, self._scalars, geometry, chi],
            template=[("B", tile), ("TOTAL", len(trial)), ("N_BAND", n_band), ("NUM_BF", nk), ("PLANE", plane),
                      ("K_CHUNK", _K_CHUNK), ("STRIDE", 31), ("NT", 14)],
            grid=(n_band, chunks, (len(trial) + tile - 1) // tile), threadgroup=(64, 1, 1),
            output_shapes=[(chunks, len(trial), n_band)] * 3, output_dtypes=[mx.float32] * 3,
        )
        real, imag, denominator = (np.asarray(mx.sum(part, axis=0)) for part in (real, imag, denominator))
        return real + 1j * imag, denominator, weight, trial_mx, chi

    def values(self, z, stride=1, band=(0.2, 0.9)):
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        rows, packed = self.encode(z, 5)
        numerators, denominators, weight, trial, chi = self._sums(rows, packed, stride, band)
        self._last = (z.copy(), stride, tuple(band), numerators, denominators, weight, trial, chi)
        self.value_passes += len(z)
        self.calls += len(z)
        return _power_sum(numerators, denominators, weight)

    def gradients(self, z=None, stride=1, band=(0.2, 0.9)):
        if self._last is None:
            raise RuntimeError("Evaluate the values before their gradients.")
        last_z, last_stride, last_band, numerators, denominators, weight, trial, chi = self._last
        if (z is not None and not np.array_equal(np.atleast_2d(z), last_z)) or stride != last_stride \
                or tuple(band) != last_band:
            raise RuntimeError("The gradient pass needs the value pass of the same points, stride and band.")
        p = self.p
        mx = p.mx
        flat, qx, qy, _ = self._band_arrays_for(band)
        kx, ky, geometry, monomials, nk = self._pixels(stride)
        n_band = int(flat.size)
        count = len(last_z)
        chunks = (nk + _K_CHUNK - 1) // _K_CHUNK
        groups = (n_band + 255) // 256 * 8
        scale = mx.array(_scale(numerators, denominators, weight, 2.0).ravel())
        (partial,) = _fifth_gradient_kernel()(
            inputs=[p.g_qk, flat, qx, qy, kx, ky, geometry, chi, monomials, trial, self._scalars, scale],
            template=[("N_BAND", n_band), ("NUM_BF", nk), ("PLANE", int(p.g_qk.shape[1]) * int(p.g_qk.shape[2])
                                                                * stride),
                      ("K_CHUNK", _K_CHUNK), ("TOTAL", count), ("CHUNKS", chunks), ("GROUPS", groups)],
            grid=(groups * 32, chunks, count), threadgroup=(256, 1, 1),
            output_shapes=[(count, chunks, groups, 28)], output_dtypes=[mx.float32],
        )
        raw = np.asarray(partial).astype(np.float64).sum(axis=(1, 2))
        self.gradient_passes += count
        return self.gradient(raw)

    def values_and_gradients(self, z, stride=1, band=(0.2, 0.9)):
        values = self.values(z, stride, band)
        return values, self.gradients(z, stride, band)


class LowOrderDerivatives:
    """The MPS joint agreement (``resident_fit.evaluate``'s kernel) with exact gradient and Hessian in z.

    The interface of ``cuda.exact_derivatives.LowOrderDerivatives`` for ``_exact_newton``.
    """

    def __init__(self, prepared, band_inv_A=(0.2, 0.9)):
        self.p = prepared
        self.band = tuple(band_inv_A)
        mx = prepared.mx
        self._scalars = mx.array([prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad,
                                  prepared.ang_x_rad, prepared.factor], dtype=mx.float32)
        self._last = None
        self.value_passes = 0
        self.derivative_passes = 0

    def values(self, z, all_derivatives=False):
        p = self.p
        mx = p.mx
        trial = joint_trial(z)
        flat, qx, qy, weight = _band(p, self.band)
        sums = np.asarray(band_sums(p, mx.array(trial), flat, qx, qy, self._scalars))
        numerators, denominators = sums[0] + 1j * sums[1], sums[2]
        weight = np.asarray(weight, dtype=np.float32)
        values = _power_sum(numerators, denominators, weight)
        self._last = (trial, numerators, denominators, weight, values)
        self.value_passes += len(trial)
        return values

    def derivatives(self, indices):
        trial, numerators, denominators, weight, values = self._last
        indices = list(indices)
        if not indices:
            return []
        p = self.p
        mx = p.mx
        trial, numerators, denominators = trial[indices], numerators[indices], denominators[indices]
        flat, qx, qy, _ = _band(p, self.band)
        n_band = int(flat.size)
        nk = int(p.g_qk.shape[0])
        count = len(indices)
        chunks = (nk + _K_CHUNK - 1) // _K_CHUNK
        groups = (n_band + 255) // 256 * 8
        nd, linear = _low_kernel()(
            inputs=[p.g_qk, flat, qx, qy, p.kx, p.ky, _probe_geometry(p, self._scalars),
                    mx.array(np.ascontiguousarray(trial).ravel()), self._scalars,
                    mx.array(_scale(numerators, denominators, weight, 1.0).ravel())],
            template=[("N_BAND", n_band), ("NUM_BF", nk), ("PLANE", int(p.g_qk.shape[1]) * int(p.g_qk.shape[2])),
                      ("K_CHUNK", _K_CHUNK), ("CHUNKS", chunks), ("GROUPS", groups)],
            grid=(groups * 32, chunks, count), threadgroup=(256, 1, 1),
            output_shapes=[(count, chunks, 18, n_band), (count, chunks, groups, 27)],
            output_dtypes=[mx.float32, mx.float32],
        )
        lin = np.asarray(linear).astype(np.float64).sum(axis=(1, 2))
        sums = np.asarray(nd).astype(np.float64).sum(axis=1)
        covered = denominators > 0
        den = np.where(covered, denominators, 1).astype(np.float64)
        a = numerators.astype(np.complex128) / den
        vr = sums[:, 0:6] - a.real[:, None] * sums[:, 12:18]
        vi = sums[:, 6:12] - a.imag[:, None] * sums[:, 12:18]
        factor = np.where(covered, 2 * weight[None].astype(np.float64) / den, 0.0)[:, None, :]
        quadratic = np.matmul(vr * factor, vr.transpose(0, 2, 1)) + np.matmul(vi * factor, vi.transpose(0, 2, 1))
        self.derivative_passes += count
        out = []
        upper = np.triu_indices(6)
        for row in range(count):
            hessian = np.zeros((6, 6))
            hessian[upper] = lin[row, 6:]
            hessian = hessian + np.triu(hessian, 1).T + quadratic[row]
            out.append((float(values[indices[row]]), lin[row, :6].copy(), hessian))
        return out
