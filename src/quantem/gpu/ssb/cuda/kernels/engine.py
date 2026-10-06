"""CUDA kernels of the SSB engine: phase reductions, the probe at each bright-field pixel, and the thick-sample model.

The size-specific inverse-FFT kernels live in the ``fft{128,256,512,1024}`` modules; these are the size-independent
kernels ``SSBEngine`` and ``ThickSample`` launch around them.
"""

import cupy as cp

# The exact 512 path emits one or two 32-BF partial planes per chunk.
# Merge both moments together without temporary reduction arrays or atomics.
# Keep the pair addition separate from accumulation, as in sum(axis=0) then +=.
accumulate_phase_moments = cp.ElementwiseKernel(
    "raw float32 sums, raw float32 squares, int32 groups, int32 plane, "
    "float32 prior_sum, float32 prior_square",
    "float32 total_sum, float32 total_square",
    """
    float s = sums[i];
    float q = squares[i];
    if (groups == 2) {
        s += sums[plane + i];
        q += squares[plane + i];
    }
    total_sum = prior_sum + s;
    total_square = prior_square + q;
    """,
    "ssb_accumulate_phase_moments",
)

# Mean phase kernel: avoids materializing a full phase buffer.
mean_phase_kernel = cp.RawKernel(r'''
extern "C" __global__
void mean_phase(
    const float2* __restrict__ corrected,
    float* __restrict__ out,
    int num_bf,
    int ny,
    int nx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny * nx;
    if (idx >= total) return;
    int y = idx / nx;
    int x = idx - y * nx;
    size_t base = (size_t)y * (size_t)nx + (size_t)x;
    size_t plane = (size_t)ny * (size_t)nx;
    float sum = 0.0f;
    for (int b = 0; b < num_bf; ++b) {
        size_t off = (size_t)b * plane + base;
        float2 v = corrected[off];
        sum += atan2f(v.y, v.x);
    }
    out[base] = sum / (float)num_bf;
}
''', 'mean_phase')

# Fused kernel: sum + sumsq of per-BF angles in a single pass.
# Used by reconstruct_with_loss to avoid running the correction pipeline twice.
sum_sumsq_phase_kernel = cp.RawKernel(r'''
extern "C" __global__
void sum_sumsq_phase(
    const float2* __restrict__ corrected,
    float* __restrict__ sum_out,
    float* __restrict__ sumsq_out,
    int num_bf,
    int ny,
    int nx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny * nx;
    if (idx >= total) return;
    size_t plane = (size_t)ny * (size_t)nx;
    float s = 0.0f, sq = 0.0f;
    for (int b = 0; b < num_bf; ++b) {
        float2 v = corrected[(size_t)b * plane + (size_t)idx];
        float a = atan2f(v.y, v.x);
        s += a;
        sq += a * a;
    }
    sum_out[idx] = s;
    sumsq_out[idx] = sq;
}
''', 'sum_sumsq_phase')

# Thick-sample SSB correction (sample tilt + thickness), for ThickSample.reconstruct.
# Standard SSB treats the sample as one plane at the probe defocus. For a crystal of thickness t tilted by theta (straight
# columns), the slice at depth z (from mid-depth) sees the defocus C10 + z and sits shifted by z theta. Averaging the two SSB
# terms over depth gives each its own real weight (sinc of the depth-phase rate times t / 2):
#   t1 = P(q - k) conj(P(k)),   rate1 = -pi lambda (|q - k|^2 - |k|^2) - 2 pi q . theta
#   t2 = conj(P(q + k)) P(k),   rate2 = +pi lambda (|q + k|^2 - |k|^2) - 2 pi q . theta
#   gamma = w1 t1 - w2 t2,  w = sinc(rate t / 2)
# with P = aperture exp(-i chi) at the mid-depth aberrations (chi = factor alpha^2 (C10 + C12 cos 2(phi - phi12)), the same
# geometry and soft aperture as compute_geometry). corrected = G conj(gamma / |gamma|); t = 0 gives w = 1 and the standard
# correction exactly. Units: q, k in 1/A; C10, C12, thickness in the engine's C10 unit (A: chi uses factor = pi / lambda[A]).
# Probe geometry (alpha^2, cos 2 phi, sin 2 phi, soft aperture) and the sinc depth weight, shared by the thick kernels.
_THICK_GEOMETRY_SOURCE = r"""
    __device__ float4 thick_geometry(float dx, float dy, float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad) {
        float r2 = dx * dx + dy * dy;
        float alpha2 = r2 * wavelength * wavelength;
        float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
        float cos2phi = (dx * dx - dy * dy) * inv_r2;
        float sin2phi = 2.0f * dx * dy * inv_r2;
        float r = sqrtf(r2);
        float alpha = r * wavelength;
        float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
        float denom = sqrtf(dx * ang_y_rad * dx * ang_y_rad + dy * ang_x_rad * dy * ang_x_rad) * inv_r;
        float edge = (denom > 1e-15f) ? ((semiangle_rad - alpha) / denom + 0.5f) : 1.0f;
        return make_float4(alpha2, cos2phi, sin2phi, fminf(fmaxf(edge, 0.0f), 1.0f));
    }
    __device__ float thick_sinc(float x) { return (fabsf(x) < 1e-6f) ? 1.0f : sinf(x) / x; }
"""

_THICK_CORRECTION_SOURCE = _THICK_GEOMETRY_SOURCE + r"""
__device__ float2 thick_correct(float2 G, float qx, float qy, float kx, float ky,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10, float C12, float cos2phi12, float sin2phi12, float factor,
    float thickness, float theta_r, float theta_c) {

        float4 g_k = thick_geometry(kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float4 g_m = thick_geometry(qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float4 g_p = thick_geometry(qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float chi_k = factor * g_k.x * fmaf(C12, fmaf(g_k.y, cos2phi12, g_k.z * sin2phi12), C10);
        float chi_m = factor * g_m.x * fmaf(C12, fmaf(g_m.y, cos2phi12, g_m.z * sin2phi12), C10);
        float chi_p = factor * g_p.x * fmaf(C12, fmaf(g_p.y, cos2phi12, g_p.z * sin2phi12), C10);
        float w1 = 1.0f, w2 = 1.0f;
        if (thickness > 0.0f) {
            float shift = 6.283185307f * (qx * theta_r + qy * theta_c);
            float rate1 = -factor * (g_m.x - g_k.x) - shift;
            float rate2 = factor * (g_p.x - g_k.x) - shift;
            w1 = thick_sinc(0.5f * rate1 * thickness);
            w2 = thick_sinc(0.5f * rate2 * thickness);
        }
        // t1 = P(m) conj(P(k)) = a_m a_k exp(-i (chi_m - chi_k)); t2 = conj(P(p)) P(k) = a_p a_k exp(i (chi_p - chi_k))
        float a1 = w1 * g_m.w * g_k.w, a2 = w2 * g_p.w * g_k.w;
        float s1, c1, s2, c2;
        __sincosf(chi_m - chi_k, &s1, &c1);
        __sincosf(chi_p - chi_k, &s2, &c2);
        float g_re = a1 * c1 - a2 * c2;
        float g_im = -a1 * s1 - a2 * s2;
        float mag_sq = g_re * g_re + g_im * g_im;
        float inv_mag = (mag_sq > 1e-16f) ? rsqrtf(mag_sq) : 1e8f;
        g_re *= inv_mag; g_im *= inv_mag;
        float Gr = G.x, Gi = G.y;
        return make_float2(Gr * g_re + Gi * g_im, Gi * g_re - Gr * g_im);

}
"""

thick_correct_kernel = cp.ElementwiseKernel(
    in_params="""
        complex64 G, float32 qx, float32 qy, float32 kx, float32 ky,
        float32 wavelength, float32 semiangle_rad, float32 ang_y_rad, float32 ang_x_rad,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor,
        float32 thickness, float32 theta_r, float32 theta_c
        """,
    out_params="complex64 corrected",
    preamble=_THICK_CORRECTION_SOURCE,
    operation="""
        float2 value = thick_correct(make_float2(G.real(), G.imag()), qx, qy, kx, ky,
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad, C10, C12, cos2phi12, sin2phi12, factor, thickness, theta_r, theta_c);
        corrected = thrust::complex<float>(value.x, value.y);
    """,
    name="thick_correct_kernel",
)

# Sum corrected spectra in bounded BF groups. Read aliases from the native
# spectrum directly, avoiding an expanded (detector, row, col) output volume.
thick_wave_sum_kernel = cp.RawKernel(_THICK_CORRECTION_SOURCE + r"""
extern "C" __global__ void thick_wave_sum(
    const float2* source, const float* qrow, const float* qcol,
    const float* krow, const float* kcol, double2* partial,
    int num_bf, int native_rows, int native_cols, int stored_cols,
    int rows, int cols, int first_bf, int group_size, float2 dc,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10, float C12, float cos2phi12, float sin2phi12, float factor,
    float thickness, float theta_r, float theta_c) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = rows * cols;
    if (q >= plane) return;
    int start = first_bf + blockIdx.y * group_size;
    int stop = min(start + group_size, num_bf);
    double2 sum = make_double2(0.0, 0.0);
    if (q == 0) {
        sum = make_double2((double)dc.x * (stop-start), (double)dc.y * (stop-start));
    } else {
        int row = q / cols, col = q % cols;
        float qr = qrow[row], qc = qcol[col];
        // No overlap of soft probe apertures beyond this radius.
        float support = 2.0f * (semiangle_rad + 0.5f * fmaxf(ang_y_rad, ang_x_rad)) / wavelength;
        if (qr*qr + qc*qc <= support*support) {
            int sr = row % native_rows, sc = col % native_cols;
            bool conjugate = stored_cols != native_cols && sc >= stored_cols;
            if (conjugate) { sr = (native_rows-sr) % native_rows; sc = native_cols-sc; }
            size_t offset = (size_t)sr * stored_cols + sc;
            size_t stride = (size_t)native_rows * stored_cols;
            for (int b=start; b<stop; ++b) {
                float2 g = source[(size_t)b * stride + offset];
                if (conjugate) g.y = -g.y;
                float2 v = thick_correct(g, qr, qc, krow[b], kcol[b],
                    wavelength, semiangle_rad, ang_y_rad, ang_x_rad, C10, C12, cos2phi12, sin2phi12, factor, thickness, theta_r, theta_c);
                sum.x += v.x; sum.y += v.y;
            }
        }
    }
    partial[(size_t)blockIdx.y * plane + q] = sum;
}
""", "thick_wave_sum")

# Least-squares fit of the SSB model G(q, k) = Psi(q) gamma(q, k) (unnormalised gamma of thick_correct_kernel): per q the best
# Psi explains |sum_k G conj(gamma)|^2 / sum_k |gamma|^2 of the data, so the sum of that over a spatial-frequency band measures how
# much of G the model with these aberrations, tilt and thickness accounts for. Unlike the phase-variance loss (phase-only
# correction, pixels weighted equally) it uses how strongly each pixel carries the signal, which is what tilt changes.
thick_fit_kernel = cp.ElementwiseKernel(
    in_params="""
        complex64 G, float32 qx, float32 qy, float32 kx, float32 ky,
        float32 wavelength, float32 semiangle_rad, float32 ang_y_rad, float32 ang_x_rad,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor,
        float32 thickness, float32 theta_r, float32 theta_c
        """,
    out_params="complex64 projected, float32 weight2",
    preamble=_THICK_GEOMETRY_SOURCE + "    ",
    operation="""
        float4 g_k = thick_geometry(kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float4 g_m = thick_geometry(qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float4 g_p = thick_geometry(qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float chi_k = factor * g_k.x * fmaf(C12, fmaf(g_k.y, cos2phi12, g_k.z * sin2phi12), C10);
        float chi_m = factor * g_m.x * fmaf(C12, fmaf(g_m.y, cos2phi12, g_m.z * sin2phi12), C10);
        float chi_p = factor * g_p.x * fmaf(C12, fmaf(g_p.y, cos2phi12, g_p.z * sin2phi12), C10);
        float w1 = 1.0f, w2 = 1.0f;
        if (thickness > 0.0f) {
            float shift = 6.283185307f * (qx * theta_r + qy * theta_c);
            float rate1 = -factor * (g_m.x - g_k.x) - shift;
            float rate2 = factor * (g_p.x - g_k.x) - shift;
            w1 = thick_sinc(0.5f * rate1 * thickness);
            w2 = thick_sinc(0.5f * rate2 * thickness);
        }
        // t1 = P(m) conj(P(k)) = a_m a_k exp(-i (chi_m - chi_k)); t2 = conj(P(p)) P(k) = a_p a_k exp(i (chi_p - chi_k))
        float a1 = w1 * g_m.w * g_k.w, a2 = w2 * g_p.w * g_k.w;
        float s1, c1, s2, c2;
        __sincosf(chi_m - chi_k, &s1, &c1);
        __sincosf(chi_p - chi_k, &s2, &c2);
        float g_re = a1 * c1 - a2 * c2;
        float g_im = -a1 * s1 - a2 * s2;
        float Gr = G.real(), Gi = G.imag();
        projected = thrust::complex<float>(Gr * g_re + Gi * g_im, Gi * g_re - Gr * g_im);
        weight2 = g_re * g_re + g_im * g_im;
    """,
    name="thick_fit_kernel",
)


# Batched thick-sample fit (fast path for ThickSample.fit_batch): one thread per band q on the stored half-plane, loop over
# bright-field pixels k, up to THICK_FIT_MAX_BATCH parameter sets per pass so G is read once per batch. Geometry of q-k, q+k, k
# (apertures, alpha^2, 2phi) is computed once per (k, q) and shared by every parameter set; pairs with no overlap are skipped.
# Same model as thick_fit_kernel (and tests/parity/torch_ssb.py): gamma = w1 P(q-k) conj P(k) - w2 conj P(q+k) P(k).
THICK_FIT_MAX_BATCH = 8
thick_fit_batch_kernel = cp.RawKernel(r"""
#define MAXB 8
__device__ __forceinline__ float4 tf_geometry(float dx, float dy, float wl, float semiangle, float ang_y, float ang_x) {
    float r2 = dx * dx + dy * dy;
    float alpha2 = r2 * wl * wl;
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2 = (dx * dx - dy * dy) * inv_r2;
    float sin2 = 2.0f * dx * dy * inv_r2;
    float r = sqrtf(r2);
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = sqrtf(dx * ang_y * dx * ang_y + dy * ang_x * dy * ang_x) * inv_r;
    float edge = (denom > 1e-15f) ? ((semiangle - r * wl) / denom + 0.5f) : 1.0f;
    return make_float4(alpha2, cos2, sin2, fminf(fmaxf(edge, 0.0f), 1.0f));
}
__device__ __forceinline__ float tf_sinc(float x) { return (fabsf(x) < 1e-6f) ? 1.0f : __sinf(x) / x; }
extern "C" __global__ void thick_fit_batch(
    const float2* __restrict__ G, const long long* __restrict__ flat, const float* __restrict__ qxb, const float* __restrict__ qyb,
    const float* __restrict__ kx, const float* __restrict__ ky, const float* __restrict__ trial,
    float2* __restrict__ numer, float* __restrict__ denom,
    int num_bf, long long plane, int n_band, int B, float wl, float semiangle, float ang_y, float ang_x, float factor,
    int k_chunk)
{
    __shared__ float tp[MAXB * 7];   // C10, C12, cos2phi12, sin2phi12, thickness, theta_r, theta_c (rad)
    for (int i = threadIdx.x; i < B * 7; i += blockDim.x) tp[i] = trial[i];
    // Probe geometry depends only on k. Share it across all q threads in
    // this block, keeping the arithmetic and per-thread accumulation order.
    __shared__ float4 geometry_k[256];  // matches thick_fit_batch's k_chunk
    int first_k = blockIdx.y * k_chunk;
    for (int offset = threadIdx.x; offset < k_chunk; offset += blockDim.x) {
        int k = first_k + offset;
        if (k < num_bf) {
            geometry_k[offset] = tf_geometry(kx[k], ky[k], wl, semiangle, ang_y, ang_x);
        }
    }
    __syncthreads();
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_band) return;
    float qx = qxb[q], qy = qyb[q];
    long long off = flat[q];
    float nr[MAXB], ni[MAXB], dd[MAXB];
    for (int b = 0; b < MAXB; ++b) { nr[b] = 0.0f; ni[b] = 0.0f; dd[b] = 0.0f; }
    // grid.y splits the bright-field pixels: each block sums its slice and adds it atomically (outputs zeroed by the host)
    int k0 = blockIdx.y * k_chunk, k1 = min(num_bf, k0 + k_chunk);
    for (int k = k0; k < k1; ++k) {
        float kxv = __ldg(kx + k), kyv = __ldg(ky + k);
        float4 gk = geometry_k[k - k0];
        float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x);
        float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;             // no double overlap for this (k, q)
        float2 g = G[(long long)k * plane + off];
        for (int b = 0; b < B; ++b) {
            float C10 = tp[b * 7 + 0], C12 = tp[b * 7 + 1], c2 = tp[b * 7 + 2], s2 = tp[b * 7 + 3];
            float t = tp[b * 7 + 4], thr = tp[b * 7 + 5], thc = tp[b * 7 + 6];
            float chi_k = factor * gk.x * fmaf(C12, fmaf(gk.y, c2, gk.z * s2), C10);
            float chi_m = factor * gm.x * fmaf(C12, fmaf(gm.y, c2, gm.z * s2), C10);
            float chi_p = factor * gp.x * fmaf(C12, fmaf(gp.y, c2, gp.z * s2), C10);
            float w1 = 1.0f, w2 = 1.0f;
            if (t > 0.0f) {
                float shift = 6.283185307f * (qx * thr + qy * thc);
                w1 = tf_sinc(0.5f * (-factor * (gm.x - gk.x) - shift) * t);
                w2 = tf_sinc(0.5f * (factor * (gp.x - gk.x) - shift) * t);
            }
            float s1, c1, sp, cp_;
            __sincosf(chi_m - chi_k, &s1, &c1);
            __sincosf(chi_p - chi_k, &sp, &cp_);
            float b1 = w1 * a1, b2 = w2 * a2;
            float gr = b1 * c1 - b2 * cp_;          // gamma = b1 exp(-i(chi_m - chi_k)) - b2 exp(i(chi_p - chi_k))
            float gi = -b1 * s1 - b2 * sp;
            nr[b] += g.x * gr + g.y * gi;          // G conj(gamma)
            ni[b] += g.y * gr - g.x * gi;
            dd[b] += gr * gr + gi * gi;
        }
    }
    for (int b = 0; b < B; ++b) {
        float* nq = reinterpret_cast<float*>(numer + (long long)b * n_band + q);
        atomicAdd(nq, nr[b]); atomicAdd(nq + 1, ni[b]);
        atomicAdd(denom + (long long)b * n_band + q, dd[b]);
    }
}
""", "thick_fit_batch")


# Probe at every bright-field pixel, P(k) = A(k) exp(-i chi(k)), for the fused FFT paths.
pk_kernel = cp.ElementwiseKernel(
    in_params="""
        float32 alpha_k2, float32 cos2phi_k, float32 sin2phi_k, float32 aperture_k,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor
        """,
    out_params="complex64 pk",
    operation="""
        float cos_term_k = __fmaf_rn(cos2phi_k, cos2phi12, sin2phi_k * sin2phi12);
        float chi_k = factor * alpha_k2 * __fmaf_rn(C12, cos_term_k, C10);
        float sin_k, cos_k;
        __sincosf(chi_k, &sin_k, &cos_k);
        pk = thrust::complex<float>(aperture_k * cos_k, -aperture_k * sin_k);
    """,
    name="pk_kernel",
)

# Full-aberration pk kernel.  Computes pk = aperture(k) * exp(-i·χ(k)) for all
# 14 Krivanek aberrations up to 5th order.  Used by SSBEngine.reconstruct_full
# for explorer manual higher-order slider drag.  The three-parameter pk_kernel above is
# NOT modified and continues to serve the optimization hot path.
#
# mags[14] and angles[14] follow the Krivanek order:
#   0: C10, 1: C12/phi12,  2: C21/phi21, 3: C23/phi23,
#   4: C30, 5: C32/phi32,  6: C34/phi34,
#   7: C41/phi41, 8: C43/phi43, 9: C45/phi45,
#   10: C50, 11: C52/phi52, 12: C54/phi54, 13: C56/phi56
#
# Input kx, ky are per-BF-pixel reciprocal-space coordinates (1/m); aperture_k
# is the precomputed soft-edge mask; wavelength and kfactor=2π/wavelength are
# passed in to avoid recomputation.
# Fast variant using host-precomputed Chebyshev-ready arrays (see
# kernels.common.pack_aberration_coefs).  Replaces 14 cosf per BF pixel with
# ~20 FMAs via direct cos(φ)/sin(φ) from (kx, ky) + Chebyshev recurrence.
pk_kernel_full = cp.ElementwiseKernel(
    in_params="""
        float32 kx, float32 ky, float32 aperture_k,
        float32 wavelength, float32 kfactor,
        raw float32 abr_mag_scaled, raw float32 abr_cm, raw float32 abr_sm
        """,
    out_params="complex64 pk",
    operation="""
        float r2 = kx * kx + ky * ky;
        float inv_r = (r2 > 1e-30f) ? rsqrtf(r2) : 0.0f;
        float r = r2 * inv_r;
        float alpha = wavelength * r;
        float c1 = kx * inv_r;
        float s1 = ky * inv_r;
        float c2 = fmaf(2.0f * c1, c1, -1.0f);
        float s2 = 2.0f * s1 * c1;
        float c3 = fmaf(c1, c2, -s1 * s2);
        float s3 = fmaf(s1, c2,  c1 * s2);
        float c4 = fmaf(c2, c2, -s2 * s2);
        float s4 = 2.0f * s2 * c2;
        float c5 = fmaf(c1, c4, -s1 * s4);
        float s5 = fmaf(s1, c4,  c1 * s4);
        float c6 = fmaf(c2, c4, -s2 * s4);
        float s6 = fmaf(s2, c4,  c2 * s4);

        float a2 = alpha * alpha;
        float a3 = a2 * alpha;
        float a4 = a2 * a2;
        float a5 = a4 * alpha;
        float a6 = a3 * a3;

        float chi = 0.0f;
        chi = fmaf(a2 * abr_mag_scaled[0], 1.0f,
              fmaf(a2 * abr_mag_scaled[1], fmaf(c2, abr_cm[1],  s2 * abr_sm[1]),  chi));
        chi = fmaf(a3 * abr_mag_scaled[2], fmaf(c1, abr_cm[2],  s1 * abr_sm[2]),
              fmaf(a3 * abr_mag_scaled[3], fmaf(c3, abr_cm[3],  s3 * abr_sm[3]),  chi));
        chi = fmaf(a4 * abr_mag_scaled[4], 1.0f,
              fmaf(a4 * abr_mag_scaled[5], fmaf(c2, abr_cm[5],  s2 * abr_sm[5]),
              fmaf(a4 * abr_mag_scaled[6], fmaf(c4, abr_cm[6],  s4 * abr_sm[6]),  chi)));
        chi = fmaf(a5 * abr_mag_scaled[7], fmaf(c1, abr_cm[7],  s1 * abr_sm[7]),
              fmaf(a5 * abr_mag_scaled[8], fmaf(c3, abr_cm[8],  s3 * abr_sm[8]),
              fmaf(a5 * abr_mag_scaled[9], fmaf(c5, abr_cm[9],  s5 * abr_sm[9]),  chi)));
        chi = fmaf(a6 * abr_mag_scaled[10], 1.0f,
              fmaf(a6 * abr_mag_scaled[11], fmaf(c2, abr_cm[11], s2 * abr_sm[11]),
              fmaf(a6 * abr_mag_scaled[12], fmaf(c4, abr_cm[12], s4 * abr_sm[12]),
              fmaf(a6 * abr_mag_scaled[13], fmaf(c6, abr_cm[13], s6 * abr_sm[13]), chi))));
        chi *= kfactor;

        float sin_k, cos_k;
        __sincosf(chi, &sin_k, &cos_k);
        pk = thrust::complex<float>(aperture_k * cos_k, -aperture_k * sin_k);
    """,
    name="pk_kernel_full",
)
