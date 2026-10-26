"""Exact derivatives of the MPS SSB agreement for the trust-region fits (``cuda/exact_derivatives.py`` on Metal).

The agreement S = sum_q w_q |N_q|^2 / D_q and its derivatives follow the CUDA module's docstring: variable projection
with a_q = N_q / D_q and R = conj(a_q) (G - a_q gamma) gives dS/dtheta = sum 2 w Re(dgamma/dtheta conj R). The
fifth-order value pass is the production kernel (``fifth_order.Objective``) with the band sum in float64 on the host;
its gradient pass repeats the per-pair arithmetic in float32 (precise sincos), reduces each SIMD group on the GPU and
sums the groups in float64 on the host. The low-order passes (``LowOrderDerivatives``) use the angle-sum arithmetic
of the joint objective (``resident_fit``) and form value, gradient and Hessian per band frequency on the GPU; the band
is summed in float64 on the host."""

import copy
from functools import cache, lru_cache

import numpy as np

from quantem.gpu.direct_ptycho._exact_newton import joint_trial
from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE
from quantem.gpu.direct_ptycho.mps import fifth_order
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx
from quantem.gpu.direct_ptycho.mps.resident_fit import ANCHOR, FIT_HEADER, _probe_geometry, _row_steps, aperture_radii
from quantem.gpu.direct_ptycho.mps.thick_sample import _K_CHUNK, _THICK_FIT_BATCH_HEADER, _band

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
        self._band_factor = None
        self.value_passes = 0
        self.gradient_passes = 0

    @property
    def num_bright_field_pixels(self) -> int:
        return int(self.p.g_qk.shape[0])

    def band_frequencies(self, band=(0.2, 0.9)):
        """(q row, q column) of the band frequencies on the stored half plane, 1/A, float32 on the host."""
        _, qx, qy, _ = _band(self.p, tuple(float(value) for value in band))
        return np.asarray(qx, dtype=np.float32).ravel(), np.asarray(qy, dtype=np.float32).ravel()

    def with_band_weights(self, factor):
        """This objective with each band frequency's weight multiplied by ``factor`` (one value per frequency of
        ``band_frequencies``; 0 leaves a frequency out of values and gradients)."""
        weighted = copy.copy(self)
        weighted._band_factor = np.asarray(factor, dtype=np.float32)
        weighted._last = None
        return weighted

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
        weight = np.asarray(weight, dtype=np.float32)
        return flat, qx, qy, weight if self._band_factor is None else weight * self._band_factor

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


# Low-order agreement with its exact gradient and Hessian in z = (C10, a = C12 cos 2 phi12, b = C12 sin 2 phi12 [A],
# tilt row, tilt column [mrad], depth spread [A]): the arithmetic of the native Metal fit
# (``ssb_low_order_derivatives.metal``) with the objective's helpers (``resident_fit.FIT_HEADER``):
# 1. the value in its own pass (``_LOW_VALUE_SUMS``, ``_LOW_VALUE_REDUCE``): per band frequency the angle-sum sums
#    A = sum (a - b) H, B = sum (a + b) H, Sd = sum (a - b)^2, Ss = sum (a + b)^2 over 256-pixel chunks, then
#    N = cos Psi A + i sin Psi B and D = cos^2 Psi Sd + sin^2 Psi Ss, so the Newton asks for derivatives only at the
#    centers whose step uses them; where it uses every center's, one fused pass forms both (``_LOW_FUSED_SUMS``,
#    the same value partials bitwise);
# 2. 29 k-sums in place of 51, from two exact reductions of the derivative structure:
#    - the weights depend on the tilt only through tau = q.theta, so d/d(tilt row) = qx d/dtau and
#      d/d(tilt column) = qy d/dtau (x = t (-pi 1e-3 tau + f q.k -+ P), tau in mrad / A);
#    - the phase slopes v_j(k) of (C10, a, b) are linear in k: v_j = alpha_j . k with alpha_0 = 2f (qx, qy),
#      alpha_1 = 2f (qx, -qy), alpha_2 = 2f (qy, qx), so every j-sum is two k-moments and every (j, l)-sum three;
#    ``low_outputs`` expands them to the 51 per band frequency and forms w (dS_q/dz, d2S_q/dz2), 27 floats; the band
#    is summed on the host in float64.
# Pass-2 fields (n_k = h (c d + i s sm), m_k = h (-s d + i c sm), rho1 + i rho2 = conj(N) h; m over (tau, t) with
# d_tau = sum over the sidebands of +-amplitude sinc'(x), the factor -pi 1e-3 t applied in the combine):
#   0-3 sum kx n_k, sum ky n_k (complex); 4-7 Q_tau, Q_t = sum h (c d_m + i s s_m) (complex);
#   8-9 R_m = sum c^2 d d_m + s^2 sm s_m; 10-11 T_m = sum sm s_m - d d_m;
#   12-14 U_mn = sum c^2 (d_m d_n + d d_mn) + s^2 (s_m s_n + sm s_mn), (tau tau) (tau t) (t t);
#   15-16 sum k Im(conj N m_k); 17-19 sum kx kx, kx ky, ky ky Re(conj N n_k);
#   20-21 sum Re(conj N m_k,m); 22-25 sum kx, ky Im(conj N n_k,m) for m = tau then t; 26-28 sum Re(conj N n_k,mn).
# The fused pass keeps the 43 complex sums that do not need N (15-18 sum kx m_k, sum ky m_k; 19-24 sum kx kx,
# kx ky, ky ky n_k; 25-28 sum m_k,tau, sum m_k,t; 29-36 sum kx n_k,tau, ky n_k,tau, kx n_k,t, ky n_k,t; 37-42
# sum n_k,tau tau, n_k,tau t, n_k,t t) and projects them on conj N in its combine.
_LOW_FIELDS = 29
_LOW_FUSED_FIELDS = 43
_LOW_OUTPUTS = 27
# BF pixels per derivative-pass chunk (a multiple of the value pass's 256-pixel chunks)
_LOW_CHUNK = 2048

_LOW_HEADER = FIT_HEADER + """
struct LowSetup {
    float qx, qy, ux, uy, psi, c_q, p_q, bx, by, cos_p, sin_p;
    float2 su_step, sy_step;
};
// Trial rows: C10, a, b, depth spread [A], tilt row, column [rad].
inline LowSetup low_setup(float qx, float qy, float c10, float a, float b, float depth, float thr, float thc, float f,
                          float step_x, float step_y) {
    LowSetup s;
    float m00 = f * (c10 + a), m01 = f * b, m11 = f * (c10 - a);
    s.qx = qx; s.qy = qy;
    s.ux = 2.0f * (m00 * qx + m01 * qy);
    s.uy = 2.0f * (m01 * qx + m11 * qy);
    s.psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
    s.c_q = -3.141592653589793f * depth * (qx * thr + qy * thc);
    s.p_q = 0.5f * depth * f * (qx * qx + qy * qy);
    s.sin_p = metal::precise::sincos(s.p_q, s.cos_p);
    s.bx = depth * f * qx;
    s.by = depth * f * qy;
    float c, sn = metal::precise::sincos(s.ux * step_x + s.uy * step_y, c);
    s.su_step = float2(c, sn);
    sn = metal::precise::sincos(s.bx * step_x + s.by * step_y, c);
    s.sy_step = float2(c, sn);
    return s;
}
inline float low_psi(float qx, float qy, float c10, float a, float b, float f) {
    float m00 = f * (c10 + a), m01 = f * b, m11 = f * (c10 - a);
    return qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
}
// sinc(x) and its first two derivatives from sx = sin x, cx = cos x. Below |x| = 1 by series (sinc and
// g = j1(x) / x, sinc' = -x g, sinc'' = 2 g - sinc); above it sinc is ``fit_sinc``'s quotient.
inline float3 low_sinc(float x, float sx, float cx) {
    float x2 = x * x;
    if (metal::abs(x) < 1.0f) {
        float s0 = 1.0f - x2 * (1.0f / 6.0f) * (1.0f - x2 * (1.0f / 20.0f) * (1.0f - x2 * (1.0f / 42.0f)
                 * (1.0f - x2 * (1.0f / 72.0f) * (1.0f - x2 * (1.0f / 110.0f)))));
        float g = 1.0f / 3.0f + x2 * (-1.0f / 30.0f + x2 * (1.0f / 840.0f + x2 * (-1.0f / 45360.0f
                + x2 * (1.0f / 3991680.0f))));
        return float3(s0, -x * g, 2.0f * g - s0);
    }
    float inv = metal::fast::divide(1.0f, x);
    float q0 = sx * inv;
    float s0 = metal::fma(metal::fma(-x, q0, sx), inv, q0);
    float s1 = (cx - s0) * inv;
    float s2 = -(s0 + 2.0f * s1 * inv);
    return float3(s0, s1, s2);
}
// w (dS_q/dz, d2S_q/dz2) of one band frequency from its 29 sums b (projected on conj N), A and B (sums) and Sd, Ss
// (sq): the 51 per-frequency fields, then their combine. 27 values: the six first derivatives and the upper
// triangle of the Hessian row by row.
inline void low_outputs(thread float* b, float qx, float qy, float c10, float ta, float tb, float depth, float4 sums,
                        float2 sq, float w, float f, thread float* res) {
    float psi = low_psi(qx, qy, c10, ta, tb, f);
    float cpsi, spsi = metal::precise::sincos(psi, cpsi);
    float D = cpsi * cpsi * sq.x + spsi * spsi * sq.y;
    if (!(D > 0.0f)) {
        for (int i = 0; i < 27; ++i) res[i] = 0.0f;
        return;
    }
    float nx = cpsi * sums.x - spsi * sums.w, ny = cpsi * sums.y + spsi * sums.z;
    float mx = -spsi * sums.x - cpsi * sums.w, my = -spsi * sums.y + cpsi * sums.z;
    float lambda = -3.141592653589793f * 1e-3f;
    float kappa = lambda * depth;
    float ar = qx * kappa, ac = qy * kappa;
    float2 alpha[3] = {2.0f * f * float2(qx, qy), 2.0f * f * float2(qx, -qy), 2.0f * f * float2(qy, qx)};
    float a[51];
    for (int j = 0; j < 3; ++j) {
        a[2 * j] = alpha[j].x * b[0] + alpha[j].y * b[2];
        a[2 * j + 1] = alpha[j].x * b[1] + alpha[j].y * b[3];
        a[24 + j] = alpha[j].x * b[15] + alpha[j].y * b[16];
        float xu = alpha[j].x * b[22] + alpha[j].y * b[23];
        a[36 + 3 * j] = ar * xu;
        a[37 + 3 * j] = ac * xu;
        a[38 + 3 * j] = alpha[j].x * b[24] + alpha[j].y * b[25];
    }
    const int jl[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    for (int r = 0; r < 6; ++r) {
        float2 u = alpha[jl[r][0]], v = alpha[jl[r][1]];
        a[27 + r] = u.x * v.x * b[17] + (u.x * v.y + u.y * v.x) * b[18] + u.y * v.y * b[19];
    }
    a[6] = ar * b[4]; a[7] = ar * b[5]; a[8] = ac * b[4]; a[9] = ac * b[5]; a[10] = b[6]; a[11] = b[7];
    a[12] = ar * b[8]; a[13] = ac * b[8]; a[14] = b[9];
    a[15] = ar * b[10]; a[16] = ac * b[10]; a[17] = b[11];
    float u_ut = kappa * b[13] + lambda * b[8];
    a[18] = ar * ar * b[12]; a[19] = ar * ac * b[12]; a[20] = qx * u_ut;
    a[21] = ac * ac * b[12]; a[22] = qy * u_ut; a[23] = b[14];
    a[33] = ar * b[20]; a[34] = ac * b[20]; a[35] = b[21];
    float f_ut = kappa * b[27] + lambda * (nx * b[4] + ny * b[5]);
    a[45] = ar * ar * b[26]; a[46] = ar * ac * b[26]; a[47] = qx * f_ut;
    a[48] = ac * ac * b[26]; a[49] = qy * f_ut; a[50] = b[28];
    float psi_j[3] = {f * (qx * qx + qy * qy), f * (qx * qx - qy * qy), 2.0f * f * qx * qy};
    float E = nx * nx + ny * ny;
    float split = sq.y - sq.x;
    float d_psi = 2.0f * spsi * cpsi * split;
    float d_psipsi = 2.0f * (cpsi * cpsi - spsi * spsi) * split;
    float Nre[6], Nim[6], Dd[6];
    for (int j = 0; j < 3; ++j) {
        // N_j = psi_j M - i P_j.
        Nre[j] = psi_j[j] * mx + a[2 * j + 1];
        Nim[j] = psi_j[j] * my - a[2 * j];
        Dd[j] = psi_j[j] * d_psi;
    }
    for (int m = 0; m < 3; ++m) {
        Nre[3 + m] = a[6 + 2 * m];
        Nim[3 + m] = a[7 + 2 * m];
        Dd[3 + m] = 2.0f * a[12 + m];
    }
    float Ed[6];
    for (int i = 0; i < 6; ++i) Ed[i] = 2.0f * (nx * Nre[i] + ny * Nim[i]);
    float inv = 1.0f / D;
    for (int i = 0; i < 6; ++i) res[i] = w * (Ed[i] - E * Dd[i] * inv) * inv;
    int field = 6;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            float projected, Dij;
            if (j < 3) {
                int pair = 0;
                for (int r = 0; r < 6; ++r) if (jl[r][0] == i && jl[r][1] == j) pair = r;
                projected = -psi_j[i] * psi_j[j] * E + psi_j[i] * a[24 + j] + psi_j[j] * a[24 + i] - a[27 + pair];
                Dij = psi_j[i] * psi_j[j] * d_psipsi;
            } else if (i < 3) {
                int m = j - 3;
                projected = psi_j[i] * a[33 + m] + a[36 + 3 * i + m];
                Dij = 4.0f * psi_j[i] * spsi * cpsi * a[15 + m];
            } else {
                int m = i - 3, n = j - 3, pair = 0;
                for (int r = 0; r < 6; ++r) if (jl[r][0] == m && jl[r][1] == n) pair = r;
                projected = a[45 + pair];
                Dij = 2.0f * a[18 + pair];
            }
            float Eij = 2.0f * (Nre[i] * Nre[j] + Nim[i] * Nim[j]) + 2.0f * projected;
            res[field++] = w * (Eij * inv - (Ed[i] * Dd[j] + Ed[j] * Dd[i]) * inv * inv
                                - E * Dij * inv * inv + 2.0f * E * Dd[i] * Dd[j] * inv * inv * inv);
        }
    }
}
"""

# Shared prologue: band frequency qi, the candidate's trial row (C10, a, b, depth, tilt row, tilt column) and setup.
_LOW_PROLOGUE = """
    float wl = scalars[0], semi = scalars[1], ang_y = scalars[2], ang_x = scalars[3], factor = scalars[4];
    float inner2 = radii[0], outer2 = radii[1];
    float f = factor * wl * wl;
    float qx = band_qx[qi], qy = band_qy[qi];
    uint off = band_flat[qi];
    float t0 = trial[candidate * 6], t1 = trial[candidate * 6 + 1], t2 = trial[candidate * 6 + 2];
    float t3 = trial[candidate * 6 + 3], t4 = trial[candidate * 6 + 4], t5 = trial[candidate * 6 + 5];
    LowSetup s = low_setup(qx, qy, t0, t1, t2, t3, t4, t5, f, step_k[0], step_k[1]);
"""

# Advance exp(i u.k) and exp(i y) to pixel kk (anchored at kk == k0, a row break and every ANCHOR pixels), the
# double-overlap aperture and H = G exp(-i u.k); `continue` where neither sideband overlaps.
_LOW_PIXEL = """
            float kxv = kx[kk], kyv = ky[kk];
            bool anchor = kk == k0 || neighbour[kk] == 0.0f || (kk - k0) % ANCHOR == 0;
            FIT_UNIFORM_IF(anchor) {
                float c, sn = metal::precise::sincos(s.ux * kxv + s.uy * kyv, c);
                eu = float2(c, sn);
                sn = metal::precise::sincos(s.c_q + (s.bx * kxv + s.by * kyv), c);
                ey = float2(c, sn);
            }
            if (!anchor) {
                eu = float2(eu.x * s.su_step.x - eu.y * s.su_step.y, eu.x * s.su_step.y + eu.y * s.su_step.x);
                ey = float2(ey.x * s.sy_step.x - ey.y * s.sy_step.y, ey.x * s.sy_step.y + ey.y * s.sy_step.x);
            }
            float gkw = probe_geometry[kk * 4 + 3];
            float2 e = fit_aperture(qx, qy, kxv, kyv, wl, semi, ang_y, ang_x, inner2, outer2);
            float a1 = e.x * gkw, a2 = e.y * gkw;
            if (a1 == 0.0f && a2 == 0.0f) continue;
            auto g = G[(size_t)kk * (size_t)PLANE + (size_t)off];
            float hx = g.real * eu.x + g.imag * eu.y, hy = g.imag * eu.x - g.real * eu.y;
"""

# Pass 1: per (band frequency, chunk of 256 BF pixels, candidate) the partial A, B (4 floats) and Sd, Ss (2 floats).
_LOW_VALUE_SUMS = """
    uint qi = thread_position_in_grid.x, chunk = thread_position_in_grid.y, candidate = thread_position_in_grid.z;
    if (qi >= N_BAND || candidate >= count[0]) return;
""" + _LOW_PROLOGUE + """
    bool thick = s.p_q != 0.0f || s.c_q != 0.0f || s.bx != 0.0f || s.by != 0.0f;
    float2 a_sum = 0.0f, b_sum = 0.0f;
    float sd = 0.0f, ss = 0.0f;
    float2 eu = float2(1.0f, 0.0f), ey = float2(1.0f, 0.0f);
    uint k0 = chunk * 256u;
    uint k1 = metal::min(uint(NUM_BF), k0 + 256u);
    for (uint kk = k0; kk < k1; ++kk) {
""" + _LOW_PIXEL + """
            float wa = a1, wb = a2;
            if (thick) {
                float y = s.c_q + (s.bx * kxv + s.by * kyv);
                if (a1 != 0.0f) wa = fit_sinc(y - s.p_q, ey.y * s.cos_p - ey.x * s.sin_p) * a1;
                if (a2 != 0.0f) wb = fit_sinc(y + s.p_q, ey.y * s.cos_p + ey.x * s.sin_p) * a2;
            }
            float d = wa - wb, sm = wa + wb;
            a_sum += d * float2(hx, hy);
            b_sum += sm * float2(hx, hy);
            sd += d * d;
            ss += sm * sm;
    }
    size_t index = ((size_t)candidate * CHUNKS + chunk) * N_BAND + qi;
    for (int i = 0; i < 2; ++i) { partial_ab[index * 4 + i] = a_sum[i]; partial_ab[index * 4 + 2 + i] = b_sum[i]; }
    partial_ss[index * 2] = sd;
    partial_ss[index * 2 + 1] = ss;
"""

# Pass 1 reduction in chunk order: A, B, Sd, Ss per (candidate, band frequency), and N (real, imaginary) and D.
_LOW_VALUE_REDUCE = """
    uint qi = thread_position_in_grid.x, candidate = thread_position_in_grid.y;
    if (qi >= N_BAND || candidate >= count[0]) return;
    float4 x = 0.0f;
    float2 y = 0.0f;
    for (uint c = 0; c < CHUNKS; ++c) {
        size_t index = ((size_t)candidate * CHUNKS + c) * N_BAND + qi;
        x += float4(partial_ab[index * 4], partial_ab[index * 4 + 1], partial_ab[index * 4 + 2],
                    partial_ab[index * 4 + 3]);
        y += float2(partial_ss[index * 2], partial_ss[index * 2 + 1]);
    }
    size_t out = (size_t)candidate * N_BAND + qi;
    for (int i = 0; i < 4; ++i) ab[out * 4 + i] = x[i];
    ss[out * 2] = y.x;
    ss[out * 2 + 1] = y.y;
    float f = scalars[4] * scalars[0] * scalars[0];
    float psi = low_psi(band_qx[qi], band_qy[qi], trial[candidate * 6], trial[candidate * 6 + 1],
                        trial[candidate * 6 + 2], f);
    float cpsi, spsi = metal::precise::sincos(psi, cpsi);
    // N = cos Psi A + i sin Psi B, D = cos^2 Psi Sd + sin^2 Psi Ss
    nd[((size_t)candidate * 3) * N_BAND + qi] = cpsi * x.x - spsi * x.w;
    nd[((size_t)candidate * 3 + 1) * N_BAND + qi] = cpsi * x.y + spsi * x.z;
    nd[((size_t)candidate * 3 + 2) * N_BAND + qi] = cpsi * cpsi * y.x + spsi * spsi * y.y;
"""

# Per-pixel sideband weights and their x-derivatives (x = y -+ P) of the derivative passes.
_LOW_WEIGHTS = """
            float y = s.c_q + (s.bx * kxv + s.by * kyv);
            float sy = ey.y, cy = ey.x;
            float3 za = low_sinc(y - s.p_q, sy * s.cos_p - cy * s.sin_p, cy * s.cos_p + sy * s.sin_p);
            float3 zb = low_sinc(y + s.p_q, sy * s.cos_p + cy * s.sin_p, cy * s.cos_p - sy * s.sin_p);
            float qk = qx * kxv + qy * kyv;
            float y_rate = tilt_y + f * qk;
            float ra = y_rate - p_rate, rb = y_rate + p_rate;
            float wa = a1 * za.x, wb = a2 * zb.x;
            float w1a = a1 * za.y, w1b = a2 * zb.y;
            float w2a = a1 * za.z, w2b = a2 * zb.z;
            float d = wa - wb, sm = wa + wb;
            float d_u = w1a - w1b, s_u = w1a + w1b;
            float w1ar = w1a * ra, w1br = w1b * rb;
            float d_t = w1ar - w1br, s_t = w1ar + w1br;
            float d_uu = w2a - w2b, s_uu = w2a + w2b;
            float w2ar = w2a * ra, w2br = w2b * rb;
            float d_ut = w2ar - w2br, s_ut = w2ar + w2br;
            float w2arr = w2ar * ra, w2brr = w2br * rb;
            float d_tt = w2arr - w2brr, s_tt = w2arr + w2brr;
"""

_LOW_DERIVATIVE_PROLOGUE = """
    float cpsi, spsi = metal::precise::sincos(s.psi, cpsi);
    float c2 = cpsi * cpsi, s2 = spsi * spsi;
    float p_rate = 0.5f * f * (qx * qx + qy * qy);
    float tilt_y = -3.141592653589793f * (qx * t4 + qy * t5);
    float2 eu = float2(1.0f, 0.0f), ey = float2(1.0f, 0.0f);
    uint k0 = chunk * LOW_CHUNK;
    uint k1 = metal::min(uint(NUM_BF), k0 + uint(LOW_CHUNK));
"""

# Pass 2: per (band frequency, chunk of LOW_CHUNK BF pixels, selected candidate) the 29 k-sums of the derivatives.
_LOW_GRADIENT_SUMS = """
    uint qi = thread_position_in_grid.x, chunk = thread_position_in_grid.y, slot = thread_position_in_grid.z;
    if (qi >= N_BAND || slot >= count[0]) return;
    uint candidate = selected[slot];
""" + _LOW_PROLOGUE + _LOW_DERIVATIVE_PROLOGUE + """
    size_t at = (size_t)candidate * N_BAND + qi;
    float4 sums = float4(ab[at * 4], ab[at * 4 + 1], ab[at * 4 + 2], ab[at * 4 + 3]);
    // N = c A + i s B.
    float nx = cpsi * sums.x - spsi * sums.w, ny = cpsi * sums.y + spsi * sums.z;
    float acc[29];
    for (int i = 0; i < 29; ++i) acc[i] = 0.0f;
    for (uint kk = k0; kk < k1; ++kk) {
""" + _LOW_PIXEL + _LOW_WEIGHTS + """
            // Projections on conj(N): rho1 = Re(conj N h), rho2 = Im(conj N h).
            float rho1 = nx * hx + ny * hy, rho2 = nx * hy - ny * hx;
            float cd = cpsi * d, ssm = spsi * sm;
            float n_re = cd * hx - ssm * hy, n_im = cd * hy + ssm * hx;
            acc[0] += kxv * n_re;
            acc[1] += kxv * n_im;
            acc[2] += kyv * n_re;
            acc[3] += kyv * n_im;
            float re_n = cd * rho1 - ssm * rho2;
            float im_m = cpsi * sm * rho1 - spsi * d * rho2;
            acc[15] += kxv * im_m;
            acc[16] += kyv * im_m;
            float kre = kxv * re_n;
            acc[17] += kxv * kre;
            acc[18] += kyv * kre;
            acc[19] += kyv * (kyv * re_n);
            float cdu = cpsi * d_u, ssu = spsi * s_u;
            acc[4] += cdu * hx - ssu * hy;
            acc[5] += cdu * hy + ssu * hx;
            acc[8] += c2 * d * d_u + s2 * sm * s_u;
            acc[10] += sm * s_u - d * d_u;
            acc[20] += -spsi * d_u * rho1 - cpsi * s_u * rho2;
            float im_u = cdu * rho2 + ssu * rho1;
            acc[22] += kxv * im_u;
            acc[23] += kyv * im_u;
            float cdt = cpsi * d_t, sst = spsi * s_t;
            acc[6] += cdt * hx - sst * hy;
            acc[7] += cdt * hy + sst * hx;
            acc[9] += c2 * d * d_t + s2 * sm * s_t;
            acc[11] += sm * s_t - d * d_t;
            acc[21] += -spsi * d_t * rho1 - cpsi * s_t * rho2;
            float im_t = cdt * rho2 + sst * rho1;
            acc[24] += kxv * im_t;
            acc[25] += kyv * im_t;
            acc[12] += c2 * (d_u * d_u + d * d_uu) + s2 * (s_u * s_u + sm * s_uu);
            acc[13] += c2 * (d_u * d_t + d * d_ut) + s2 * (s_u * s_t + sm * s_ut);
            acc[14] += c2 * (d_t * d_t + d * d_tt) + s2 * (s_t * s_t + sm * s_tt);
            acc[26] += cpsi * d_uu * rho1 - spsi * s_uu * rho2;
            acc[27] += cpsi * d_ut * rho1 - spsi * s_ut * rho2;
            acc[28] += cpsi * d_tt * rho1 - spsi * s_tt * rho2;
    }
    size_t base = ((size_t)slot * LOW_CHUNKS + chunk) * 29;
    for (int i = 0; i < 29; ++i) partial[(base + i) * N_BAND + qi] = acc[i];
"""

# Reduce pass 2 in chunk order and form the 27 outputs of each band frequency.
_LOW_GRADIENT_COMBINE = """
    uint qi = thread_position_in_grid.x, slot = thread_position_in_grid.y;
    if (qi >= N_BAND || slot >= count[0]) return;
    uint candidate = selected[slot];
    float b[29];
    for (int i = 0; i < 29; ++i) b[i] = 0.0f;
    for (uint c = 0; c < LOW_CHUNKS; ++c) {
        size_t base = ((size_t)slot * LOW_CHUNKS + c) * 29;
        for (int i = 0; i < 29; ++i) b[i] += partial[(base + i) * N_BAND + qi];
    }
""" + """
    size_t at = (size_t)candidate * N_BAND + qi;
    float4 sums = float4(ab[at * 4], ab[at * 4 + 1], ab[at * 4 + 2], ab[at * 4 + 3]);
    float2 sq = float2(ss[at * 2], ss[at * 2 + 1]);
    float f = scalars[4] * scalars[0] * scalars[0];
    float res[27];
    low_outputs(b, band_qx[qi], band_qy[qi], trial[candidate * 6], trial[candidate * 6 + 1], trial[candidate * 6 + 2],
                trial[candidate * 6 + 3], sums, sq, weight[qi], f, res);
    for (int i = 0; i < 27; ++i) output[((size_t)slot * 27 + i) * N_BAND + qi] = res[i];
"""

# The fused pass for centers whose derivatives the Newton will use: pass 1's 256-pixel partials (the same per-pixel
# terms and chunks, bitwise) and, per LOW_CHUNK pixels, the 43 k-sums that do not need N.
_LOW_FUSED_SUMS = """
    uint qi = thread_position_in_grid.x, chunk = thread_position_in_grid.y, candidate = thread_position_in_grid.z;
    if (qi >= N_BAND || candidate >= count[0]) return;
""" + _LOW_PROLOGUE + _LOW_DERIVATIVE_PROLOGUE + """
    float acc[43];
    for (int i = 0; i < 43; ++i) acc[i] = 0.0f;
    for (uint v0 = k0; v0 < k1; v0 += 256u) {
        uint v1 = metal::min(k1, v0 + 256u);
        float2 a_sum = 0.0f, b_sum = 0.0f;
        float sdv = 0.0f, ssv = 0.0f;
        for (uint kk = v0; kk < v1; ++kk) {
""" + _LOW_PIXEL + _LOW_WEIGHTS + """
            a_sum += d * float2(hx, hy);
            b_sum += sm * float2(hx, hy);
            sdv += d * d;
            ssv += sm * sm;
            float cd = cpsi * d, ssm = spsi * sm;
            float n_re = cd * hx - ssm * hy, n_im = cd * hy + ssm * hx;
            acc[0] += kxv * n_re;
            acc[1] += kxv * n_im;
            acc[2] += kyv * n_re;
            acc[3] += kyv * n_im;
            float sdd = spsi * d, csm = cpsi * sm;
            float m_re = -sdd * hx - csm * hy, m_im = csm * hx - sdd * hy;
            acc[15] += kxv * m_re;
            acc[16] += kxv * m_im;
            acc[17] += kyv * m_re;
            acc[18] += kyv * m_im;
            float kn_re = kxv * n_re, kn_im = kxv * n_im;
            acc[19] += kxv * kn_re;
            acc[20] += kxv * kn_im;
            acc[21] += kyv * kn_re;
            acc[22] += kyv * kn_im;
            acc[23] += kyv * (kyv * n_re);
            acc[24] += kyv * (kyv * n_im);
            float cdu = cpsi * d_u, ssu = spsi * s_u;
            float nu_re = cdu * hx - ssu * hy, nu_im = cdu * hy + ssu * hx;
            acc[4] += nu_re;
            acc[5] += nu_im;
            acc[8] += c2 * d * d_u + s2 * sm * s_u;
            acc[10] += sm * s_u - d * d_u;
            float sdu = spsi * d_u, csu = cpsi * s_u;
            acc[25] += -sdu * hx - csu * hy;
            acc[26] += csu * hx - sdu * hy;
            acc[29] += kxv * nu_re;
            acc[30] += kxv * nu_im;
            acc[31] += kyv * nu_re;
            acc[32] += kyv * nu_im;
            float cdt = cpsi * d_t, sst = spsi * s_t;
            float nt_re = cdt * hx - sst * hy, nt_im = cdt * hy + sst * hx;
            acc[6] += nt_re;
            acc[7] += nt_im;
            acc[9] += c2 * d * d_t + s2 * sm * s_t;
            acc[11] += sm * s_t - d * d_t;
            float sdt = spsi * d_t, cst = cpsi * s_t;
            acc[27] += -sdt * hx - cst * hy;
            acc[28] += cst * hx - sdt * hy;
            acc[33] += kxv * nt_re;
            acc[34] += kxv * nt_im;
            acc[35] += kyv * nt_re;
            acc[36] += kyv * nt_im;
            acc[12] += c2 * (d_u * d_u + d * d_uu) + s2 * (s_u * s_u + sm * s_uu);
            acc[13] += c2 * (d_u * d_t + d * d_ut) + s2 * (s_u * s_t + sm * s_ut);
            acc[14] += c2 * (d_t * d_t + d * d_tt) + s2 * (s_t * s_t + sm * s_tt);
            float cuu = cpsi * d_uu, suu = spsi * s_uu;
            acc[37] += cuu * hx - suu * hy;
            acc[38] += cuu * hy + suu * hx;
            float cut = cpsi * d_ut, sut = spsi * s_ut;
            acc[39] += cut * hx - sut * hy;
            acc[40] += cut * hy + sut * hx;
            float ctt = cpsi * d_tt, stt = spsi * s_tt;
            acc[41] += ctt * hx - stt * hy;
            acc[42] += ctt * hy + stt * hx;
        }
        size_t index = ((size_t)candidate * CHUNKS + v0 / 256u) * N_BAND + qi;
        for (int i = 0; i < 2; ++i) { partial_ab[index * 4 + i] = a_sum[i]; partial_ab[index * 4 + 2 + i] = b_sum[i]; }
        partial_ss[index * 2] = sdv;
        partial_ss[index * 2 + 1] = ssv;
    }
    size_t base = ((size_t)candidate * LOW_CHUNKS + chunk) * 43;
    for (int i = 0; i < 43; ++i) partial[(base + i) * N_BAND + qi] = acc[i];
"""

# Reduce the fused pass in chunk order, project its complex sums on conj N and form the 27 outputs.
_LOW_FUSED_COMBINE = """
    uint qi = thread_position_in_grid.x, candidate = thread_position_in_grid.y;
    if (qi >= N_BAND || candidate >= count[0]) return;
    float F[43];
    for (int i = 0; i < 43; ++i) F[i] = 0.0f;
    for (uint c = 0; c < LOW_CHUNKS; ++c) {
        size_t base = ((size_t)candidate * LOW_CHUNKS + c) * 43;
        for (int i = 0; i < 43; ++i) F[i] += partial[(base + i) * N_BAND + qi];
    }
    float f = scalars[4] * scalars[0] * scalars[0];
    float qx = band_qx[qi], qy = band_qy[qi];
    float c10 = trial[candidate * 6], ta = trial[candidate * 6 + 1], tb = trial[candidate * 6 + 2];
    float cpsi, spsi = metal::precise::sincos(low_psi(qx, qy, c10, ta, tb, f), cpsi);
    size_t at = (size_t)candidate * N_BAND + qi;
    float4 sums = float4(ab[at * 4], ab[at * 4 + 1], ab[at * 4 + 2], ab[at * 4 + 3]);
    float2 sq = float2(ss[at * 2], ss[at * 2 + 1]);
    float nx = cpsi * sums.x - spsi * sums.w, ny = cpsi * sums.y + spsi * sums.z;
    float b[29];
    for (int i = 0; i < 15; ++i) b[i] = F[i];
    // Re(conj N z) = nx z.re + ny z.im, Im(conj N z) = nx z.im - ny z.re.
    b[15] = nx * F[16] - ny * F[15];
    b[16] = nx * F[18] - ny * F[17];
    b[17] = nx * F[19] + ny * F[20];
    b[18] = nx * F[21] + ny * F[22];
    b[19] = nx * F[23] + ny * F[24];
    b[20] = nx * F[25] + ny * F[26];
    b[21] = nx * F[27] + ny * F[28];
    b[22] = nx * F[30] - ny * F[29];
    b[23] = nx * F[32] - ny * F[31];
    b[24] = nx * F[34] - ny * F[33];
    b[25] = nx * F[36] - ny * F[35];
    b[26] = nx * F[37] + ny * F[38];
    b[27] = nx * F[39] + ny * F[40];
    b[28] = nx * F[41] + ny * F[42];
    float res[27];
    low_outputs(b, qx, qy, c10, ta, tb, trial[candidate * 6 + 3], sums, sq, weight[qi], f, res);
    for (int i = 0; i < 27; ++i) output[((size_t)candidate * 27 + i) * N_BAND + qi] = res[i];
"""

_PIXEL_INPUTS = ["G", "band_flat", "band_qx", "band_qy", "kx", "ky", "trial", "scalars", "probe_geometry",
                 "neighbour", "step_k", "radii", "count"]


@cache
def _low_kernel(name):
    """The low-order passes: ``value_sums``, ``value_reduce``, ``gradient_sums``, ``gradient_combine``,
    ``fused_sums`` and ``fused_combine``."""
    inputs, outputs, source = {
        "value_sums": (_PIXEL_INPUTS, ["partial_ab", "partial_ss"], _LOW_VALUE_SUMS),
        "value_reduce": (["partial_ab", "partial_ss", "band_qx", "band_qy", "trial", "scalars", "count"],
                         ["ab", "ss", "nd"], _LOW_VALUE_REDUCE),
        "gradient_sums": (_PIXEL_INPUTS + ["ab", "selected"], ["partial"], _LOW_GRADIENT_SUMS),
        "gradient_combine": (["partial", "ab", "ss", "band_qx", "band_qy", "weight", "trial", "scalars", "selected",
                              "count"], ["output"], _LOW_GRADIENT_COMBINE),
        "fused_sums": (_PIXEL_INPUTS, ["partial", "partial_ab", "partial_ss"], _LOW_FUSED_SUMS),
        "fused_combine": (["partial", "ab", "ss", "band_qx", "band_qy", "weight", "trial", "scalars", "count"],
                          ["output"], _LOW_FUSED_COMBINE),
    }[name]
    return require_mlx().fast.metal_kernel(name=f"ssb_low_order_{name}", input_names=inputs, output_names=outputs,
                                           header=_LOW_HEADER, source=source)


class LowOrderDerivatives:
    """The MPS joint agreement with exact gradient and Hessian in z (the low-order passes above).

    The interface of ``cuda.exact_derivatives.LowOrderDerivatives`` for ``_exact_newton``: ``values(z,
    all_derivatives)`` evaluates rows of z and keeps their sums, forming every row's derivatives in the same pass when
    ``all_derivatives``; ``derivatives(indices)`` returns (value, gradient, Hessian) at rows of the latest ``values``
    call. Each band frequency's terms are formed in float32 on the GPU; the band is summed on the host in float64.
    """

    def __init__(self, prepared, band_inv_A=(0.2, 0.9)):
        self.p = prepared
        self.band = tuple(band_inv_A)
        mx = prepared.mx
        self._scalars = mx.array([prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad,
                                  prepared.ang_x_rad, prepared.factor], dtype=mx.float32)
        flat, _, _, weight = _band(prepared, self.band)
        self._weight = np.asarray(weight, dtype=np.float32)
        self._weight_mx = mx.array(self._weight)
        g = prepared.g_qk
        nk = int(g.shape[0])
        self._chunks = (nk + 255) // 256
        self._low_chunks = (nk + _LOW_CHUNK - 1) // _LOW_CHUNK
        self._n_band = int(flat.size)
        self._template = [("N_BAND", self._n_band), ("NUM_BF", nk), ("PLANE", int(g.shape[1]) * int(g.shape[2])),
                          ("CHUNKS", self._chunks), ("LOW_CHUNK", _LOW_CHUNK), ("LOW_CHUNKS", self._low_chunks),
                          ("ANCHOR", ANCHOR)]
        self._last = None
        self.value_passes = 0
        self.derivative_passes = 0

    def _pixel_inputs(self, trial, count):
        p = self.p
        flat, qx, qy, _ = _band(p, self.band)
        return [p.g_qk, flat, qx, qy, p.kx, p.ky, trial, self._scalars, _probe_geometry(p, self._scalars),
                *_row_steps(p), aperture_radii(p), count]

    def _run(self, name, inputs, grid, threadgroup, shapes):
        mx = self.p.mx
        return _low_kernel(name)(inputs=inputs, template=self._template, grid=grid, threadgroup=threadgroup,
                                 output_shapes=shapes, output_dtypes=[mx.float32] * len(shapes))

    def values(self, z, all_derivatives=False):
        p = self.p
        mx = p.mx
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        # C10, a, b, depth spread [A], tilt row, tilt column [rad]: the float32 values of ``joint_trial``
        rows = np.ascontiguousarray(joint_trial(z)[:, [0, 2, 3, 4, 5, 6]])
        count = len(rows)
        n = self._n_band
        trial = mx.array(rows.ravel())
        count_mx = mx.array([count], dtype=mx.uint32)
        _, qx, qy, _ = _band(p, self.band)
        pixel = self._pixel_inputs(trial, count_mx)
        output = None
        if all_derivatives:
            partial, partial_ab, partial_ss = self._run(
                "fused_sums", pixel, (n, self._low_chunks, count), (64, 1, 1),
                [(count * self._low_chunks * _LOW_FUSED_FIELDS * n,), (count * self._chunks * n * 4,),
                 (count * self._chunks * n * 2,)])
        else:
            partial_ab, partial_ss = self._run(
                "value_sums", pixel, (n, self._chunks, count), (64, 1, 1),
                [(count * self._chunks * n * 4,), (count * self._chunks * n * 2,)])
        ab, ss, nd = self._run("value_reduce", [partial_ab, partial_ss, qx, qy, trial, self._scalars, count_mx],
                               (n, count, 1), (128, 1, 1), [(count * n * 4,), (count * n * 2,), (count * 3 * n,)])
        if all_derivatives:
            (output,) = self._run("fused_combine", [partial, ab, ss, qx, qy, self._weight_mx, trial, self._scalars,
                                                    count_mx], (n, count, 1), (64, 1, 1),
                                  [(count * _LOW_OUTPUTS * n,)])
            mx.eval(nd, output, ab, ss)
            output = _band_totals(output, count, n)
        else:
            mx.eval(nd, ab, ss)
        nd = np.asarray(nd).reshape(count, 3, n)
        values = _power_sum(nd[:, 0] + 1j * nd[:, 1], nd[:, 2], self._weight)
        self._last = (z, trial, ab, ss, values, output)
        self.value_passes += count
        return values

    def derivatives(self, indices):
        z, trial, ab, ss, values, totals = self._last
        indices = list(indices)
        if not indices:
            return []
        if totals is None:
            p = self.p
            mx = p.mx
            n = self._n_band
            count = len(indices)
            count_mx = mx.array([count], dtype=mx.uint32)
            selected = mx.array(np.asarray(indices, dtype=np.uint32))
            _, qx, qy, _ = _band(p, self.band)
            (partial,) = self._run("gradient_sums", [*self._pixel_inputs(trial, count_mx), ab, selected],
                                   (n, self._low_chunks, count), (64, 1, 1),
                                   [(count * self._low_chunks * _LOW_FIELDS * n,)])
            (output,) = self._run("gradient_combine", [partial, ab, ss, qx, qy, self._weight_mx, trial,
                                                       self._scalars, selected, count_mx],
                                  (n, count, 1), (64, 1, 1), [(count * _LOW_OUTPUTS * n,)])
            mx.eval(output)
            rows = _band_totals(output, count, n)
        else:
            rows = totals[indices]
        self.derivative_passes += len(indices)
        out = []
        upper = np.triu_indices(6)
        for index, total in zip(indices, rows, strict=True):
            # the kernels see the depth spread |z5|: d|z5|/dz5 = sign(z5)
            sign = -1.0 if z[index, 5] < 0 else 1.0
            gradient = total[:6].copy()
            gradient[5] *= sign
            hessian = np.zeros((6, 6))
            hessian[upper] = total[6:]
            hessian[:5, 5] *= sign
            hessian = hessian + np.triu(hessian, 1).T
            out.append((float(values[index]), gradient, hessian))
        return out


def _band_totals(output, count, n_band):
    """The 27 per-frequency outputs of each candidate summed over the band in float64, shape (count, 27)."""
    return np.asarray(output).reshape(count, _LOW_OUTPUTS, n_band).astype(np.float64).sum(axis=2)

