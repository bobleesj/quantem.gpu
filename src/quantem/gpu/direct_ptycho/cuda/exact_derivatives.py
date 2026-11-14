"""Exact derivatives of the CUDA direct ptychography agreement for the trust-region fits.

Both fit objectives are the least-squares agreement S = sum_q w_q |N_q|^2 / D_q of the transfer model with the
measured double-overlap data, N_q = sum_k G(q, k) conj(gamma), D_q = sum_k |gamma|^2,
gamma = b1 exp(-i phi1) - b2 exp(i phi2), phi1 = chi(k - q) - chi(k), phi2 = chi(k + q) - chi(k), and b1, b2 the
soft apertures times the depth sinc of the sample tilt and depth spread. With a_q = N_q / D_q and
R = conj(a_q) (G - a_q gamma) (variable projection):

    dS/dtheta_i = sum_{q,k} 2 w_q Re(gamma_i conj R)
    d2S/dtheta_i dtheta_j = sum_{q,k} 2 w_q [Re(gamma_ij conj R) - |a_q|^2 Re(gamma_i conj gamma_j)]
                            + sum_q (2 w_q / D_q) Re(conj(V_j) V_i)

with gamma_i = dgamma/dtheta_i, V_i = N_i - a_q D_i, N_i = sum_k G conj(gamma_i) and D_i = sum_k 2 Re(conj(gamma)
gamma_i). The value passes are the production kernels themselves, so the values are the CUDA objectives; the
derivative passes repeat their per-pair arithmetic in float32 and sum the BF chunks and band frequencies in float64.

``FifthOrderGradients`` serves the fifth-order search (``broad_batch.Objective``): chi is linear in the 28 packed
Cartesian coefficients and a polynomial of degree <= 6 in k, so the coefficient gradient is linear in the monomial
sums T_a = sum 2w [alpha1 ((k-q)^a - k^a) + alpha2 ((k+q)^a - k^a)] (alpha1 = Im(u1 conj R), alpha2 = Im(u2 conj R),
u1 = b1 exp(-i phi1), u2 = b2 exp(i phi2)), and by the binomial expansion each thread accumulates the k-moments
E = sum (alpha1 + alpha2) k^a' (|a'| <= 4) and O = sum (alpha2 - alpha1) k^a' (|a'| <= 5) and shifts them by its q
once. The host maps the 25 monomial sums to the packed coefficients in float64. A BF subsample (every n-th pixel)
is a strided view of G and steers only.

``low_order_derivatives`` serves the low-order Newton (``ThickSample``'s ``thick_fit_batch``): value, gradient and
Hessian in z = (C10, C12 cos 2 phi12, C12 sin 2 phi12 [A], tilt row, tilt column [mrad], depth spread [A]).
"""

import copy
import itertools

import cupy as cp
import numpy as np

from quantem.gpu.direct_ptycho._exact_newton import joint_trial
from quantem.gpu.direct_ptycho.cuda.broad_batch import Objective as _BatchObjective
from quantem.gpu.direct_ptycho.cuda.kernels.engine import (
    THICK_FIT_MAX_BATCH,
    thick_fit_batch_kernel,
)
from quantem.gpu.direct_ptycho.cuda.thick import SlabSums
from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE

# tf_geometry and tf_sinc exactly as the production objective kernels define them.
_GEOMETRY_SOURCE = thick_fit_batch_kernel.code.split('extern "C"')[0]
_CHI_SOURCE = "#define DIRECT_DEVICE __device__ __forceinline__\n#define DIRECT_COEFFICIENTS const float*\n" + CHI_SOURCE

_K_CHUNK = 256

_FIFTH_SOURCE = _CHI_SOURCE + _GEOMETRY_SOURCE + r"""
__device__ __forceinline__ float fg_sinc_slope(float x, float &sinc) {
    // sinc as the objective (tf_sinc) and its slope (cos x - sinc x) / x, by its series where the difference cancels
    sinc = tf_sinc(x);
    float x2 = x * x;
    if (fabsf(x) < 0.25f) return x * (-1.0f / 3.0f + x2 * (1.0f / 30.0f - x2 * (1.0f / 840.0f)));
    float s, c;
    __sincosf(x, &s, &c);
    return __fdividef(c - sinc, x);
}

__device__ __constant__ float FG_BINOMIAL[7][7] = {
    {1, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 2, 1, 0, 0, 0, 0}, {1, 3, 3, 1, 0, 0, 0},
    {1, 4, 6, 4, 1, 0, 0}, {1, 5, 10, 10, 5, 1, 0}, {1, 6, 15, 20, 15, 6, 1}};

// chi(k) of every (strided) BF pixel and candidate: the term every pair of the candidate subtracts.
extern "C" __global__ void fifth_chi_k(const float* __restrict__ kx, const float* __restrict__ ky,
                                       const float* __restrict__ coefficients, float* __restrict__ chi,
                                       int num_bf, int B, float wl) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (k >= num_bf || b >= B) return;
    chi[(long long)b * num_bf + k] = direct_chi_full(kx[k], ky[k], wl, coefficients + b * 28);
}

// One thread per band frequency q, one block row per chunk of 256 BF pixels, one block plane per candidate.
// monomials: 21 floats per pixel, k_row^a k_col^c (a + c <= 5) at index (a + c)(a + c + 1) / 2 + c.
// scale: (Re a_q, Im a_q, |a_q|^2, 2 w_q) of the candidate's value pass, zero where D_q = 0.
// Output per block: 28 floats [dS/d thickness, dS/d tilt row, dS/d tilt column (rad), T_a for 2 <= |a| <= 6].
extern "C" __global__ void __launch_bounds__(256) fifth_gradient(
    const float2* __restrict__ G, const long long* __restrict__ flat, const float* __restrict__ qxb,
    const float* __restrict__ qyb, const float* __restrict__ kx, const float* __restrict__ ky,
    const float* __restrict__ monomials, const float* __restrict__ chik, const float* __restrict__ trial,
    const float* __restrict__ coefficients, const float4* __restrict__ scale, float* __restrict__ partial,
    int num_bf, long long plane, int n_band, float wl, float semiangle, float ang_y, float ang_x, float factor)
{
    const int b = blockIdx.z;
    __shared__ float coef[28];
    __shared__ float4 geometry_k[256];
    __shared__ float chis[256];
    __shared__ float mono[256 * 21];
    __shared__ float sums[8 * 28];
    const float* c = coefficients + b * 28;
    for (int i = threadIdx.x; i < 28; i += blockDim.x) coef[i] = c[i];
    float inv_ang = ang_y == ang_x ? 1.0f / ang_x : 0.0f;
    int k0 = blockIdx.y * 256, k1 = min(num_bf, k0 + 256);
    for (int offset = threadIdx.x; offset < k1 - k0; offset += blockDim.x) {
        int k = k0 + offset;
        geometry_k[offset] = tf_geometry(kx[k], ky[k], wl, semiangle, ang_y, ang_x, inv_ang);
        chis[offset] = chik[(long long)b * num_bf + k];
        for (int j = 0; j < 21; ++j) mono[offset * 21 + j] = monomials[(long long)k * 21 + j];
    }
    __syncthreads();
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = q < n_band;
    float4 s = valid ? scale[(long long)b * n_band + q] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool active = valid && s.w != 0.0f;
    float qx = valid ? qxb[q] : 0.0f, qy = valid ? qyb[q] : 0.0f;
    long long off = valid ? flat[q] : 0;
    const float* t = trial + b * 7;
    float half_t = 0.5f * t[4];
    float shift = 6.283185307f * (qx * t[5] + qy * t[6]);
    float ar = s.x, ai = s.y, aa = s.z;
    float even[15], odd[21];
#pragma unroll
    for (int j = 0; j < 15; ++j) even[j] = 0.0f;
#pragma unroll
    for (int j = 0; j < 21; ++j) odd[j] = 0.0f;
    float depth_sum = 0.0f, tilt_sum = 0.0f;
    if (active) {
        for (int k = k0; k < k1; ++k) {
            float kxv = __ldg(kx + k), kyv = __ldg(ky + k);
            float4 gk = geometry_k[k - k0];
            float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x, inv_ang);
            float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x, inv_ang);
            float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
            if (a1 == 0.0f && a2 == 0.0f) continue;
            float2 g = G[(long long)k * plane + off];
            float chi_k = chis[k - k0];
            float s1, c1, sp, cp_;
            __sincosf(direct_chi_full(kxv - qx, kyv - qy, wl, coef) - chi_k, &s1, &c1);
            __sincosf(direct_chi_full(kxv + qx, kyv + qy, wl, coef) - chi_k, &sp, &cp_);
            float b1 = a1, b2 = a2, d1 = 0.0f, d2 = 0.0f, h1 = 0.0f, h2 = 0.0f;
            if (half_t > 0.0f) {
                float m0 = factor * (gm.x - gk.x), p0 = factor * (gp.x - gk.x);
                float sinc;
                h1 = -m0 - shift;
                d1 = fg_sinc_slope(h1 * half_t, sinc) * a1;
                b1 = a1 * sinc;
                h2 = p0 - shift;
                d2 = fg_sinc_slope(h2 * half_t, sinc) * a2;
                b2 = a2 * sinc;
            }
            float gr = b1 * c1 - b2 * cp_;
            float gi = -b1 * s1 - b2 * sp;
            // R = conj(a) G - |a|^2 gamma
            float rr = ar * g.x + ai * g.y - aa * gr;
            float ri = ar * g.y - ai * g.x - aa * gi;
            float alpha1 = -b1 * (c1 * ri + s1 * rr);
            float alpha2 = b2 * (sp * rr - cp_ * ri);
            float beta1 = c1 * rr - s1 * ri;
            float beta2 = -(cp_ * rr + sp * ri);
            tilt_sum += beta1 * d1 + beta2 * d2;
            depth_sum += beta1 * d1 * h1 + beta2 * d2 * h2;
            float e = alpha1 + alpha2, o = alpha2 - alpha1;
            const float* mp = mono + (k - k0) * 21;
#pragma unroll
            for (int j = 0; j < 15; ++j) even[j] = fmaf(e, mp[j], even[j]);
#pragma unroll
            for (int j = 0; j < 21; ++j) odd[j] = fmaf(o, mp[j], odd[j]);
        }
    }
    // T_a = sum_{a' < a} C(a, a') q^(a - a') (odd or even moment of a' by the parity of |a - a'|).
    float qxp[7], qyp[7];
    qxp[0] = 1.0f; qyp[0] = 1.0f;
#pragma unroll
    for (int i = 1; i < 7; ++i) { qxp[i] = qxp[i - 1] * qx; qyp[i] = qyp[i - 1] * qy; }
    float w2 = active ? s.w : 0.0f;
    float out[28];
    out[0] = w2 * 0.5f * depth_sum;
    out[1] = w2 * (-3.141592653589793f) * (2.0f * half_t) * qx * tilt_sum;
    out[2] = w2 * (-3.141592653589793f) * (2.0f * half_t) * qy * tilt_sum;
#pragma unroll
    for (int d = 2; d <= 6; ++d) {
#pragma unroll
        for (int cc = 0; cc <= d; ++cc) {
            int a = d - cc;
            float total = 0.0f;
#pragma unroll
            for (int a2 = 0; a2 <= a; ++a2) {
#pragma unroll
                for (int c2 = 0; c2 <= cc; ++c2) {
                    int order = (a - a2) + (cc - c2);
                    if (order == 0) continue;
                    int index = (a2 + c2) * (a2 + c2 + 1) / 2 + c2;
                    float moment = (order & 1) ? odd[index] : even[index < 15 ? index : 0];
                    float weight = FG_BINOMIAL[a][a2] * FG_BINOMIAL[cc][c2];
                    total = fmaf(weight * qxp[a - a2] * qyp[cc - c2], moment, total);
                }
            }
            out[d * (d + 1) / 2 + cc] = w2 * total;
        }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < 28; ++j) {
        float v = out[j];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sums[warp * 28 + j] = v;
    }
    __syncthreads();
    if (threadIdx.x < 28) {
        float total = 0.0f;
        for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) total += sums[i * 28 + threadIdx.x];
        long long block = ((long long)b * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        partial[block * 28 + threadIdx.x] = total;
    }
}
"""

_fifth_module = None


def _fifth_kernels():
    global _fifth_module
    if _fifth_module is None:
        _fifth_module = cp.RawModule(code=_FIFTH_SOURCE, options=("--std=c++17",))
    return _fifth_module.get_function("fifth_chi_k"), _fifth_module.get_function("fifth_gradient")


class FifthOrderGradients(_BatchObjective):
    """The fifth-order agreement with exact gradients in the 28 dimensionless coordinates z.

    ``values`` returns the agreements of rows of z with the per-frequency powers summed in float64 (the production
    kernel's numerators and denominators); ``gradients`` the exact d agreement / dz at the points of the latest
    ``values`` call of the same stride, adding only the gradient pass. ``stride`` n uses every n-th BF pixel.
    """

    def __init__(self, engine, partial_bytes=64 << 20):
        super().__init__(engine, partial_bytes)
        self._pixels_by_stride = {}
        self._last = None
        self._band_factor = None
        self.value_passes = 0
        self.gradient_passes = 0

    @property
    def num_bright_field_pixels(self) -> int:
        return int(self.e._bf_grid()[0])

    def band_frequencies(self, band=(0.2, 0.9)):
        """(q row, q column) of the band frequencies on the stored half plane, 1/A, float32 on the host."""
        _, _, _, qx, qy, _ = self.e.thick._prepare_fit_band(band)
        return cp.asnumpy(qx).ravel(), cp.asnumpy(qy).ravel()

    def release(self) -> None:
        """Free the latest values' sums (candidates x band frequencies): a reconstruction after the fit may need the
        room, on a small device to take its scan back."""
        self._last = None

    def with_band_weights(self, factor):
        """This objective with each band frequency's weight multiplied by ``factor`` (one value per frequency of
        ``band_frequencies``; 0 leaves a frequency out of values and gradients)."""
        weighted = copy.copy(self)
        weighted._band_factor = cp.asarray(factor, dtype=cp.float32)
        weighted._last = None
        return weighted

    def _pixel_arrays(self, stride):
        return self._pixels(stride)[:2]

    def _pixels(self, stride):
        if stride not in self._pixels_by_stride:
            cache = self.e._cache
            kx = cp.ascontiguousarray(cache["kx_bf"][::stride])
            ky = cp.ascontiguousarray(cache["ky_bf"][::stride])
            powers_x = [cp.ones_like(kx)]
            powers_y = [cp.ones_like(ky)]
            for _ in range(5):
                powers_x.append(powers_x[-1] * kx)
                powers_y.append(powers_y[-1] * ky)
            monomials = cp.stack(
                [powers_x[d - c] * powers_y[c] for d in range(6) for c in range(d + 1)], axis=1
            ).astype(cp.float32)
            self._pixels_by_stride[stride] = (kx, ky, int(kx.size), cp.ascontiguousarray(monomials))
        return self._pixels_by_stride[stride]

    def _sums(self, rows, packed, stride, band, factor=None):
        """Per-frequency numerators and denominators of ``score``'s kernels on every ``stride``-th BF pixel, and the
        band weights times ``factor`` (None for none)."""
        e = self.e
        rows = np.atleast_2d(rows)
        packed = np.asarray(packed, dtype=np.float32).reshape(len(rows), 28)
        _, _, flat, qx, qy, weight = e.thick._prepare_fit_band(band)
        if factor is not None:
            weight = weight * factor
        trial = cp.asarray(
            np.stack(
                [rows[:, 0], rows[:, 1], np.cos(2 * rows[:, 2]), np.sin(2 * rows[:, 2]), rows[:, 5],
                 rows[:, 3] * 1e-3, rows[:, 4] * 1e-3],
                axis=1,
            ).astype(np.float32)
        )
        coefficients = cp.asarray(packed)
        # each batch's sums go straight to their rows: a list joined at the end held them twice, and a shared round
        # of 60 candidates on a 79,441-frequency band ran a small device out of memory (2026-11-13)
        numerators = cp.empty((len(rows), flat.size), cp.complex64)
        denominators = cp.empty((len(rows), flat.size), cp.float32)
        for start, batch, numerator, denominator in self._batches(rows, trial, coefficients, flat, qx, qy, stride):
            numerators[start:start + batch], denominators[start:start + batch] = numerator, denominator
        return numerators, denominators, weight, trial, coefficients

    @staticmethod
    def _agreement64(numerators, denominators, weight):
        return _power_sum(numerators, denominators, weight)

    def values(self, z, stride=1, band=(0.2, 0.9)):
        """Agreements of rows of z (fifth order) on every ``stride``-th BF pixel, per-frequency powers summed in float64."""
        (values, self._last), = self.values_many([(z, stride, self._band_factor)], band)
        return values

    def gradients(self, z=None, stride=1, band=(0.2, 0.9)):
        """d agreement / dz at the points of the latest ``values`` call (which must match ``z`` and ``stride``)."""
        if self._last is None:
            raise RuntimeError("Evaluate the values before their gradients.")
        last_z, last_stride, last_band = self._last[:3]
        if (z is not None and not np.array_equal(np.atleast_2d(z), last_z)) or stride != last_stride\
                or band != last_band:
            raise RuntimeError("The gradient pass needs the value pass of the same points, stride and band.")
        return self.gradients_many([self._last])[0]

    def values_many(self, requests, band=(0.2, 0.9)):
        """Several callers' agreements, one pass over G per stride: requests [(z rows, stride, band factor)].

        Returns [(values, state)]: each request's agreements (its band weights times its factor, None for none) and
        the state its gradient pass needs (``gradients_many``). A candidate's sums do not depend on the others it
        shares a pass with, so this is the values of each request alone, read from G once.
        """
        results = [None] * len(requests)
        for stride in sorted({request[1] for request in requests}):
            members = [index for index, request in enumerate(requests) if request[1] == stride]
            blocks = [np.atleast_2d(np.asarray(requests[index][0], dtype=np.float64)) for index in members]
            z = np.vstack(blocks)
            rows, packed = self.encode(z, 5)
            numerators, denominators, weight, trial, coefficients = self._sums(rows, packed, stride, band, factor=None)
            first = 0
            for index, block in zip(members, blocks):
                part = slice(first, first + len(block))
                first += len(block)
                factor = requests[index][2]
                member_weight = weight if factor is None else weight * factor
                state = (block.copy(), stride, band, numerators[part], denominators[part], member_weight, trial[part],
                         coefficients[part])
                results[index] = (cp.asnumpy(self._agreement64(numerators[part], denominators[part], member_weight)),
                                  state)
            self.value_passes += len(z)
            self.calls += len(z)
        return results

    def gradients_many(self, states):
        """d agreement / dz for several ``values_many`` states, one pass over G per stride."""
        results = [None] * len(states)
        for stride in sorted({state[1] for state in states}):
            members = [index for index, state in enumerate(states) if state[1] == stride]
            band = states[members[0]][2]
            # each state writes its own rows: candidates x frequencies arrays are the fit's largest, and a shared
            # pass of four refinements ran a 4 GB card out of memory concatenating them (2026-11-13)
            scale = cp.empty((sum(len(states[index][3]) for index in members), states[members[0]][3].shape[1], 4),
                             cp.float32)
            first = 0
            for index in members:
                numerators, denominators, weight = states[index][3:6]
                _gradient_scale(numerators, denominators, weight[None], scale[first:first + len(numerators)].ravel())
                first += len(numerators)
            trial = cp.ascontiguousarray(cp.concatenate([states[index][6] for index in members]))
            coefficients = cp.ascontiguousarray(cp.concatenate([states[index][7] for index in members]))
            raw = self._gradient_sums(scale, trial, coefficients, stride, band)
            first = 0
            for index in members:
                count = len(states[index][0])
                results[index] = self.gradient(raw[first:first + count])
                first += count
        return results

    def _gradient_sums(self, scale, trial, coefficients, stride, band):
        """The gradient kernel's 28 sums per candidate (rows of ``scale``, see _gradient_scale), summed over the
        slabs of G."""
        e = self.e
        cache = e._cache
        _, _, flat, qx, qy, _ = e.thick._prepare_fit_band(band)
        kx, ky, nk, monomials = self._pixels(stride)
        nb = int(flat.size)
        count = len(scale)
        chi_kernel, gradient_kernel = _fifth_kernels()
        chi = cp.empty((count, nk), cp.float32)
        chi_kernel(((nk + 255) // 256, count), (256,),
                   (kx, ky, coefficients, chi, np.int32(nk), np.int32(count), np.float32(e.wavelength)))
        blocks = (nb + 255) // 256
        raw = None
        # the gradient is a sum over bright-field pixels: one launch a slab of G (DirectEngine.band_slabs)
        for first, stop, slab, step in e.band_slabs(stride):
            pixels = slice(first // stride, first // stride + len(range(first, stop, stride)))
            count_k = pixels.stop - pixels.start
            if count_k == 0:
                continue
            chunks = (count_k + 255) // 256
            partial = cp.empty((count, chunks, blocks, 28), cp.float32)
            gradient_kernel(
                (blocks, chunks, count), (256,),
                (slab, flat, qx.ravel(), qy.ravel(), kx[pixels], ky[pixels], monomials[pixels],
                 cp.ascontiguousarray(chi[:, pixels]), trial, coefficients, scale, partial, np.int32(count_k),
                 np.int64(int(slab.shape[1] * slab.shape[2]) * step), np.int32(nb),
                 np.float32(e.wavelength), np.float32(cache["semiangle_rad"]), np.float32(cache["ang_y_rad"]),
                 np.float32(cache["ang_x_rad"]), np.float32(e._factor)),
            )
            slab_sum = partial.sum(axis=(1, 2), dtype=cp.float64)
            raw = slab_sum if raw is None else raw + slab_sum
        self.gradient_passes += count
        return cp.asnumpy(raw)

    def values_and_gradients(self, z, stride=1, band=(0.2, 0.9)):
        values = self.values(z, stride, band)
        return values, self.gradients(z, stride, band)


class SharedPasses:
    """Independent refinements of one ``FifthOrderGradients`` objective, their evaluations merged into shared passes.

    Each job (a callable) runs in its own thread with its own ``member`` objectives. A member's ``values`` or
    ``gradients`` call waits until every unfinished job is waiting too; then one ``values_many`` and one
    ``gradients_many`` per stride serve all of them, members in the order they were made. A candidate's sums do not
    depend on the others in its pass, so each job sees what it would see alone, and G is read once a round instead of
    once a job: on a band spectrum streamed from host memory that is the time of the fit.
    """

    def __init__(self, objective):
        import threading

        self.objective = objective
        self.condition = threading.Condition()
        self.pending, self.answers = {}, {}
        self.active = 0
        self.order = itertools.count()
        self.members = []

    def member(self, factor=None):
        """An objective for one job: the shared objective, its band weights times ``factor`` (None for none)."""
        self.members.append(_SharedMember(self, next(self.order), factor))
        return self.members[-1]

    def run(self, jobs):
        """Run the callables ``jobs`` together; their results in order, or the first job's error."""
        import contextvars
        import threading

        results, errors = [None] * len(jobs), [None] * len(jobs)
        self.active = len(jobs)

        def work(index, job):
            try:
                results[index] = job()
            except BaseException as error:  # noqa: BLE001  handed to the caller below, after every job has stopped
                errors[index] = error
            finally:
                with self.condition:
                    self.active -= 1
                    self._serve()

        threads = [threading.Thread(target=contextvars.copy_context().run, args=(work, index, job), daemon=True)
                   for index, job in enumerate(jobs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for member in self.members:
            member.state = None  # each member's last values sums, candidates x band frequencies, no longer needed
        for error in errors:
            if error is not None:
                raise error
        return results

    def request(self, member, kind, z, stride):
        ticket = object()
        with self.condition:
            self.pending[ticket] = (member, kind, z, stride)
            self._serve()
            while ticket not in self.answers:
                self.condition.wait()
            answer = self.answers.pop(ticket)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def _serve(self):
        """With the lock held: once every unfinished job waits, evaluate all waiting requests in shared passes."""
        if not self.pending or len(self.pending) < self.active:
            return
        waiting = sorted(self.pending.items(), key=lambda item: item[1][0].order)
        self.pending = {}
        try:
            values = [(ticket, request) for ticket, request in waiting if request[1] == "values"]
            if values:
                answers = self.objective.values_many([(z, stride, member.factor) for _, (member, _, z, stride) in values])
                for (ticket, (member, _, _, _)), (result, state) in zip(values, answers):
                    member.state = state
                    self.answers[ticket] = result
            gradients = [(ticket, request) for ticket, request in waiting if request[1] == "gradients"]
            if gradients:
                states = []
                for _, (member, _, z, stride) in gradients:
                    if member.state is None or member.state[1] != stride or (
                            z is not None and not np.array_equal(np.atleast_2d(z), member.state[0])):
                        raise RuntimeError("The gradient pass needs the value pass of the same points and stride.")
                    states.append(member.state)
                for (ticket, _), result in zip(gradients, self.objective.gradients_many(states)):
                    self.answers[ticket] = result
        except Exception as error:  # noqa: BLE001  every waiting job raises it (request)
            for ticket, _ in waiting:
                self.answers.setdefault(ticket, error)
        self.condition.notify_all()


class _SharedMember:
    """One job's view of a ``SharedPasses`` objective: the calls ``_gradient_refinement`` and the half-band check make."""

    def __init__(self, shared, order, factor):
        self.shared, self.order, self.factor, self.state = shared, order, factor, None

    @property
    def p(self):
        return self.shared.objective.p

    @property
    def num_bright_field_pixels(self):
        return self.shared.objective.num_bright_field_pixels

    def band_frequencies(self, band=(0.2, 0.9)):
        return self.shared.objective.band_frequencies(band)

    def with_band_weights(self, factor):
        factor = cp.asarray(factor, dtype=cp.float32)
        return self.shared.member(factor if self.factor is None else self.factor * factor)

    def values(self, z, stride=1, band=(0.2, 0.9)):
        return self.shared.request(self, "values", z, stride)

    def gradients(self, z=None, stride=1, band=(0.2, 0.9)):
        return self.shared.request(self, "gradients", z, stride)

# w |N|^2 / D per band frequency in float64 (0 where D = 0), summed over the band: the agreements of the
# production numerators and denominators with the last reduction in float64.
_power_terms = cp.ElementwiseKernel(
    "complex64 numerator, float32 denominator, float32 weight", "float64 power",
    """
    double re = (double)numerator.real(), im = (double)numerator.imag();
    power = denominator > 0.0f ? (double)weight * ((re * re + im * im) / (double)denominator) : 0.0;
    """,
    "direct_exact_power_terms",
)


def _power_sum(numerators, denominators, weight):
    return _power_terms(numerators, denominators, weight[None]).sum(axis=1)


# The gradient kernel's terms per candidate and band frequency, [a_r, a_i, |a|^2, 2 w] with a = N / D (all 0 where
# D = 0), written into a flat float32 output four at a time. One pass with no temporaries; the explicit roundings
# keep |a|^2 from contracting into a fused multiply-add, so it equals the separate array operations bit for bit.
_gradient_scale = cp.ElementwiseKernel(
    "complex64 numerator, float32 denominator, float32 weight", "raw float32 scale",
    """
    float ar = 0.0f, ai = 0.0f, w = 0.0f;
    if (denominator > 0.0f) {
        ar = __fdiv_rn(numerator.real(), denominator);
        ai = __fdiv_rn(numerator.imag(), denominator);
        w = 2.0f * weight;
    }
    scale[4 * i] = ar;
    scale[4 * i + 1] = ai;
    scale[4 * i + 2] = __fadd_rn(__fmul_rn(ar, ar), __fmul_rn(ai, ai));
    scale[4 * i + 3] = w;
    """,
    "direct_exact_gradient_scale",
)


_LOW_SOURCE = _GEOMETRY_SOURCE + r"""
// sinc and its first two derivatives (sinc as tf_sinc of the objective): series below |x| = 1 (g = j1(x) / x,
// sinc' = -x g, sinc'' = 2 g - sinc), above it from sin x, cos x.
__device__ __forceinline__ void ld_sinc(float x, float &f0, float &f1, float &f2) {
    f0 = tf_sinc(x);
    float x2 = x * x;
    if (fabsf(x) < 1.0f) {
        float g = 1.0f / 3.0f + x2 * (-1.0f / 30.0f + x2 * (1.0f / 840.0f + x2 * (-1.0f / 45360.0f
                + x2 * (1.0f / 3991680.0f))));
        f1 = -x * g;
        f2 = 2.0f * g - f0;
        return;
    }
    float s, c;
    __sincosf(x, &s, &c);
    float inv = 1.0f / x;
    f1 = (c - f0) * inv;
    f2 = -(f0 + 2.0f * f1 * inv);
}

// Per band frequency q (thread) and chunk of 256 BF pixels (block row) of one candidate (block plane):
// nd: N_i = sum_k G conj(gamma_i) (6 complex) and D_i = sum_k 2 Re(conj(gamma) gamma_i) (6), 18 floats per q;
// linear: the block's sum over its q of w [2 Re(gamma_i conj R)] (6) and w [2 Re(gamma_ij conj R)
// - 2 |a|^2 Re(gamma_i conj gamma_j)] (21, upper triangle row by row).
// z = (C10, a, b [A], tilt row, tilt column [mrad], depth spread [A]); trial rows as thick_fit_batch.
// scale: (Re a_q, Im a_q, |a_q|^2, w_q), w_q = 0 where D_q = 0.
extern "C" __global__ void __launch_bounds__(256) low_order_derivatives(
    const float2* __restrict__ G, const long long* __restrict__ flat, const float* __restrict__ qxb,
    const float* __restrict__ qyb, const float* __restrict__ kx, const float* __restrict__ ky,
    const float* __restrict__ trial, const float4* __restrict__ scale, float* __restrict__ nd,
    float* __restrict__ linear, int num_bf, long long plane, int n_band, float wl, float semiangle, float ang_y,
    float ang_x, float factor)
{
    const int b = blockIdx.z;
    __shared__ float4 geometry_k[256];
    __shared__ float sums[8 * 27];
    float inv_ang = ang_y == ang_x ? 1.0f / ang_x : 0.0f;
    int k0 = blockIdx.y * 256, k1 = min(num_bf, k0 + 256);
    for (int offset = threadIdx.x; offset < k1 - k0; offset += blockDim.x) {
        int k = k0 + offset;
        geometry_k[offset] = tf_geometry(kx[k], ky[k], wl, semiangle, ang_y, ang_x, inv_ang);
    }
    __syncthreads();
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = q < n_band;
    float4 s = valid ? scale[(long long)b * n_band + q] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool active = valid && s.w != 0.0f;
    float qx = valid ? qxb[q] : 0.0f, qy = valid ? qyb[q] : 0.0f;
    long long off = valid ? flat[q] : 0;
    const float* t = trial + b * 7;
    float c10 = t[0], ca = t[1] * t[2], cb = t[1] * t[3], half_t = 0.5f * t[4];
    float shift = 6.283185307f * (qx * t[5] + qy * t[6]);
    float ar = s.x, ai = s.y, aa = s.z;
    // d x / d (tilt row, tilt column) [mrad], the same for both sidebands, and d2 x / d tilt d depth
    float xr = -3.141592653589793e-3f * qx * (2.0f * half_t), xc = -3.141592653589793e-3f * qy * (2.0f * half_t);
    float xrt = -3.141592653589793e-3f * qx, xct = -3.141592653589793e-3f * qy;
    float nr[6], ni[6], dn[6], lg[6], lh[21];
#pragma unroll
    for (int i = 0; i < 6; ++i) { nr[i] = 0.0f; ni[i] = 0.0f; dn[i] = 0.0f; lg[i] = 0.0f; }
#pragma unroll
    for (int i = 0; i < 21; ++i) lh[i] = 0.0f;
    if (active) {
        for (int k = k0; k < k1; ++k) {
            float kxv = __ldg(kx + k), kyv = __ldg(ky + k);
            float4 gk = geometry_k[k - k0];
            float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x, inv_ang);
            float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x, inv_ang);
            float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
            if (a1 == 0.0f && a2 == 0.0f) continue;
            float2 g = G[(long long)k * plane + off];
            float m[3] = {factor * (gm.x - gk.x), factor * (gm.y - gk.y), factor * (gm.z - gk.z)};
            float p[3] = {factor * (gp.x - gk.x), factor * (gp.y - gk.y), factor * (gp.z - gk.z)};
            float s1, c1, sp, cp_;
            __sincosf(fmaf(m[2], cb, fmaf(m[1], ca, m[0] * c10)), &s1, &c1);
            __sincosf(fmaf(p[2], cb, fmaf(p[1], ca, p[0] * c10)), &sp, &cp_);
            // sideband weights b = a sinc(x) and their derivatives in (tilt row, tilt column, depth)
            float h1 = -m[0] - shift, h2 = p[0] - shift;
            float f10, f11, f12, f20, f21, f22;
            ld_sinc(h1 * half_t, f10, f11, f12);
            ld_sinc(h2 * half_t, f20, f21, f22);
            float b1 = a1 * f10, b2 = a2 * f20;
            float x1[3] = {xr, xc, 0.5f * h1}, x2[3] = {xr, xc, 0.5f * h2};
            float x12[3] = {0.0f, xrt, xct};   // (r, t) and (c, t); (r, c), (r, r), (c, c), (t, t) vanish
            float db1[3], db2[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) { db1[i] = a1 * f11 * x1[i]; db2[i] = a2 * f21 * x2[i]; }
            // e1 = exp(-i phi1), e2 = exp(i phi2), u1 = b1 e1, u2 = b2 e2, gamma = u1 - u2
            float u1r = b1 * c1, u1i = -b1 * s1, u2r = b2 * cp_, u2i = b2 * sp;
            float gr = u1r - u2r, gi = u1i - u2i;
            float rr = ar * g.x + ai * g.y - aa * gr;
            float ri = ar * g.y - ai * g.x - aa * gi;
            float dr[6], di[6];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                // -i (u1 m_i + u2 p_i)
                float ur = u1r * m[i] + u2r * p[i], ui = u1i * m[i] + u2i * p[i];
                dr[i] = ui; di[i] = -ur;
                // db1 e1 - db2 e2
                dr[3 + i] = db1[i] * c1 - db2[i] * cp_;
                di[3 + i] = -db1[i] * s1 - db2[i] * sp;
            }
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                nr[i] += g.x * dr[i] + g.y * di[i];
                ni[i] += g.y * dr[i] - g.x * di[i];
                dn[i] += 2.0f * (gr * dr[i] + gi * di[i]);
                lg[i] += 2.0f * (dr[i] * rr + di[i] * ri);
            }
            int index = 0;
#pragma unroll
            for (int i = 0; i < 6; ++i) {
#pragma unroll
                for (int j = i; j < 6; ++j) {
                    float hr, hi;
                    if (j < 3) {
                        // -u1 m_i m_j + u2 p_i p_j
                        float mm = m[i] * m[j], pp = p[i] * p[j];
                        hr = -u1r * mm + u2r * pp;
                        hi = -u1i * mm + u2i * pp;
                    } else if (i < 3) {
                        // -i (db1_s e1 m_i + db2_s e2 p_i), s = j - 3
                        float vr = db1[j - 3] * c1 * m[i] + db2[j - 3] * cp_ * p[i];
                        float vi = -db1[j - 3] * s1 * m[i] + db2[j - 3] * sp * p[i];
                        hr = vi; hi = -vr;
                    } else {
                        // d2b1 e1 - d2b2 e2, d2b = a (sinc'' x_s x_s' + sinc' x_ss')
                        int u = i - 3, v = j - 3;
                        float mixed = (u == 0 && v == 2) ? x12[1] : (u == 1 && v == 2) ? x12[2] : 0.0f;
                        float e1 = a1 * (f12 * x1[u] * x1[v] + f11 * mixed);
                        float e2 = a2 * (f22 * x2[u] * x2[v] + f21 * mixed);
                        hr = e1 * c1 - e2 * cp_;
                        hi = -e1 * s1 - e2 * sp;
                    }
                    lh[index] += 2.0f * (hr * rr + hi * ri) - 2.0f * aa * (dr[i] * dr[j] + di[i] * di[j]);
                    ++index;
                }
            }
        }
    }
    if (valid) {
        long long base = ((long long)b * gridDim.y + blockIdx.y) * 18;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            nd[(base + i) * n_band + q] = nr[i];
            nd[(base + 6 + i) * n_band + q] = ni[i];
            nd[(base + 12 + i) * n_band + q] = dn[i];
        }
    }
    float w = active ? s.w : 0.0f;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < 27; ++j) {
        float v = w * (j < 6 ? lg[j] : lh[j - 6]);
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sums[warp * 27 + j] = v;
    }
    __syncthreads();
    if (threadIdx.x < 27) {
        float total = 0.0f;
        for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) total += sums[i * 27 + threadIdx.x];
        long long block = ((long long)b * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        linear[block * 27 + threadIdx.x] = total;
    }
}
"""

_low_module = None


def _low_kernel():
    global _low_module
    if _low_module is None:
        _low_module = cp.RawModule(code=_LOW_SOURCE, options=("--std=c++17",))
    return _low_module.get_function("low_order_derivatives")


# Largest partial-sum array one launch of the low-order derivative kernel may take (LowOrderDerivatives.derivatives).
DERIVATIVE_PARTIAL_BYTES = 64 << 20


class LowOrderDerivatives:
    """The low-order joint agreement (``ThickSample.fit_resident``'s values) with exact gradient and Hessian.

    ``values(z)`` evaluates rows of z = (C10, a = C12 cos 2 phi12, b = C12 sin 2 phi12, tilt row, tilt column,
    depth spread) and keeps their numerators; ``derivatives(indices)`` returns the value, gradient and Hessian in z
    at rows of the latest ``values`` call (the Newton of ``_exact_newton``).
    """

    def __init__(self, engine, band_inv_A=(0.2, 0.9)):
        self.e = engine
        self.band = tuple(band_inv_A)
        self._last = None
        self.value_passes = 0
        self.derivative_passes = 0

    def _sums(self, trial):
        e = self.e
        cache = e._cache
        _, _, flat, qx, qy, weight = e.thick._prepare_fit_band(self.band)
        nb = int(flat.size)
        blocks = (nb + 255) // 256
        numerators = cp.empty((len(trial), nb), cp.complex64)
        denominators = cp.empty((len(trial), nb), cp.float32)
        for start in range(0, len(trial), THICK_FIT_MAX_BATCH):
            rows = trial[start:start + THICK_FIT_MAX_BATCH]
            batch = len(rows)
            sums = SlabSums()
            for first, stop, slab, _ in e.band_slabs():
                k_blocks = (stop - first + _K_CHUNK - 1) // _K_CHUNK
                partial_numerator = cp.empty((k_blocks, batch, nb), dtype=cp.complex64)
                partial_denominator = cp.empty((k_blocks, batch, nb), dtype=cp.float32)
                numerator = cp.empty((batch, nb), cp.complex64)
                denominator = cp.empty((batch, nb), cp.float32)
                thick_fit_batch_kernel(
                    (blocks, k_blocks), (256,),
                    (slab, flat, qx.ravel(), qy.ravel(), cache["kx_bf"][first:stop], cache["ky_bf"][first:stop],
                     rows.ravel(), partial_numerator, partial_denominator, cp.zeros(blocks, dtype=cp.uint32),
                     numerator, denominator, np.int32(stop - first), np.int64(slab.shape[1] * slab.shape[2]),
                     np.int32(nb), np.int32(batch), np.float32(e.wavelength), np.float32(cache["semiangle_rad"]),
                     np.float32(cache["ang_y_rad"]), np.float32(cache["ang_x_rad"]), np.float32(e._factor),
                     np.int32(_K_CHUNK)),
                )
                sums.add(numerator, denominator)
            numerators[start:start + batch], denominators[start:start + batch] = sums.result()
        return numerators, denominators, weight

    def values(self, z, all_derivatives=False):
        """Agreements at rows of z: ``ThickSample.fit_resident``'s float64 sums of the same kernel."""
        trial = cp.asarray(joint_trial(z))
        numerators, denominators, weight = self._sums(trial)
        values = _power_sum(numerators, denominators, weight)
        self._last = (trial, numerators, denominators, weight, values)
        self.value_passes += len(trial)
        return cp.asnumpy(values)

    def derivatives(self, indices):
        """(value, gradient (6), Hessian (6, 6)) in z at rows ``indices`` of the latest ``values`` call."""
        trial, numerators, denominators, weight, values = self._last
        indices = list(indices)
        if not indices:
            return []
        e = self.e
        cache = e._cache
        _, _, flat, qx, qy, _ = e.thick._prepare_fit_band(self.band)
        nb = int(flat.size)
        chosen = cp.asarray(np.asarray(indices, dtype=np.int64))
        trial = trial[chosen]
        numerators = numerators[chosen]
        denominators = denominators[chosen]
        count = len(indices)
        covered = denominators > 0
        safe = cp.where(covered, denominators, 1)
        ar = cp.where(covered, numerators.real / safe, 0).astype(cp.float32)
        ai = cp.where(covered, numerators.imag / safe, 0).astype(cp.float32)
        scale = cp.stack([ar, ai, ar * ar + ai * ai, cp.where(covered, weight[None], 0).astype(cp.float32)],
                         axis=-1).astype(cp.float32)
        blocks = (nb + 255) // 256
        lin = sums = None
        trial = cp.ascontiguousarray(trial)
        # the derivative sums run over bright-field pixels: one launch a slab of G (DirectEngine.band_slabs)
        for first, stop, slab, _ in e.band_slabs():
            chunks = (stop - first + 255) // 256
            # positions in groups whose partial sums fit DERIVATIVE_PARTIAL_BYTES: each position's sums are its own,
            # and three positions on a 23,448-frequency band held whole took 174 MiB (2026-11-14)
            group = max(1, DERIVATIVE_PARTIAL_BYTES // (chunks * 18 * nb * 4))
            parts = []
            for start in range(0, count, group):
                rows = slice(start, min(start + group, count))
                size = rows.stop - rows.start
                nd = cp.empty((size, chunks, 18, nb), cp.float32)
                linear = cp.empty((size, chunks, blocks, 27), cp.float32)
                _low_kernel()(
                    (blocks, chunks, size), (256,),
                    (slab, flat, qx.ravel(), qy.ravel(), cache["kx_bf"][first:stop], cache["ky_bf"][first:stop],
                     trial[rows], scale[rows], nd, linear, np.int32(stop - first),
                     np.int64(slab.shape[1] * slab.shape[2]), np.int32(nb), np.float32(e.wavelength),
                     np.float32(cache["semiangle_rad"]), np.float32(cache["ang_y_rad"]), np.float32(cache["ang_x_rad"]),
                     np.float32(e._factor)),
                )
                parts.append((linear.sum(axis=(1, 2), dtype=cp.float64), nd.sum(axis=1, dtype=cp.float64)))
                del nd, linear
            slab_lin, slab_sums = (parts[0] if len(parts) == 1 else
                                   (cp.concatenate([part[0] for part in parts]), cp.concatenate([part[1] for part in parts])))
            lin, sums = (slab_lin, slab_sums) if lin is None else (lin + slab_lin, sums + slab_sums)
        # sums: (count, 18, nb)
        den = safe.astype(cp.float64)[:, None, :]
        ar64 = numerators.real.astype(cp.float64)[:, None, :] / den
        ai64 = numerators.imag.astype(cp.float64)[:, None, :] / den
        # V_i = N_i - a D_i and the quadratic Hessian term sum_q (2 w_q / D_q) Re(conj(V_j) V_i)
        vr = sums[:, 0:6] - ar64 * sums[:, 12:18]
        vi = sums[:, 6:12] - ai64 * sums[:, 12:18]
        factor = cp.where(covered, 2 * weight[None].astype(cp.float64) / den[:, 0, :], 0.0)[:, None, :]
        quadratic = cp.matmul(vr * factor, vr.transpose(0, 2, 1)) + cp.matmul(vi * factor, vi.transpose(0, 2, 1))
        lin = cp.asnumpy(lin)
        quadratic = cp.asnumpy(quadratic)
        host_values = cp.asnumpy(values[chosen])
        self.derivative_passes += count
        out = []
        upper = np.triu_indices(6)
        for row in range(count):
            hessian = np.zeros((6, 6))
            hessian[upper] = lin[row, 6:]
            hessian = hessian + np.triu(hessian, 1).T + quadratic[row]
            out.append((float(host_values[row]), lin[row, :6].copy(), hessian))
        return out
