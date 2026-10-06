"""Shared CUDA device functions and base class for SSB FFT kernels.

The CUDA device functions (complex arithmetic, geometry, gamma multiplication)
are identical across all scan sizes. The Python base class provides the shared
dispatch logic for ifft2_inplace_fused_pk and the column-accumulate paths.

Size-specific kernels live in fft256.py and fft512.py.
"""

import math

import cupy as cp
import numpy as np

# =========================================================================
#  Shared CUDA device functions
# =========================================================================

_DEVICE_FUNCTIONS_CUDA = r'''
__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cmul_i(float2 a) {
    return make_float2(-a.y, a.x);
}

__device__ __forceinline__ float2 ld_float2(const float2* ptr, size_t idx) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr + idx);
#else
    return ptr[idx];
#endif
}

__device__ __forceinline__ float2 ld_gqk_maybe_herm(
    const float2* ptr,
    unsigned long long bf,
    unsigned int row,
    unsigned int col,
    unsigned int n,
    unsigned int stored_cols
) {
    unsigned long long base = bf * (unsigned long long)n * (unsigned long long)stored_cols;
    if (stored_cols == n || col <= (n >> 1)) {
        return ld_float2(ptr, base + (unsigned long long)row * stored_cols + col);
    }
    unsigned int mirror_row = row == 0u ? 0u : n - row;
    unsigned int mirror_col = n - col;
    float2 z = ld_float2(
        ptr,
        base + (unsigned long long)mirror_row * stored_cols + mirror_col
    );
    return make_float2(z.x, -z.y);
}

__device__ __forceinline__ unsigned int bit_reverse4_8(unsigned int x) {
    return ((x & 0x03u) << 6) | ((x & 0x0Cu) << 2) | ((x & 0x30u) >> 2) | ((x & 0xC0u) >> 6);
}

// Compute geometry (alpha^2, cos2phi, sin2phi, aperture) for a displacement vector.
// Uses algebraic identities to avoid atan2/cos/sin:
//   cos(2phi) = (dx^2 - dy^2) / (dx^2 + dy^2)
//   sin(2phi) = 2*dx*dy / (dx^2 + dy^2)
__device__ __forceinline__ float4 compute_geometry(
    float dx, float dy,
    float wavelength, float semiangle_rad,
    float ang_y_rad, float ang_x_rad
) {
    float dx2 = dx * dx;
    float dy2 = dy * dy;
    float r2 = dx2 + dy2;
    float alpha2 = (r2 * wavelength) * wavelength;

    // cos2phi, sin2phi via algebraic identity (no atan2)
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2phi = (dx2 - dy2) * inv_r2;
    float sin2phi = 2.0f * dx * dy * inv_r2;

    // Soft aperture: clip((semiangle - alpha) / denom + 0.5, 0, 1)
    // denom = sqrt((cos_phi * ang_y)^2 + (sin_phi * ang_x)^2)
    //       = (1/r) * sqrt((dx * ang_y)^2 + (dy * ang_x)^2)
    //
    // Most SSB live-control points are strictly inside the soft aperture.
    // Since denom <= max(ang_y, ang_x), alpha <= semiangle - 0.5*max_ang
    // proves edge >= 1 and therefore aperture == 1 exactly.  Taking this
    // branch skips the second sqrt/div path while falling back to the complete
    // edge expression so the scientific objective is unchanged.
    float max_ang = fmaxf(ang_y_rad, ang_x_rad);
    float inner = (semiangle_rad - 0.5f * max_ang) / wavelength;
    float aperture = 1.0f;
    if (inner <= 0.0f || r2 > inner * inner) {
        float r = sqrtf(r2);
        float alpha = r * wavelength;
        float denom_num2 = fmaf(dx * ang_y_rad, dx * ang_y_rad, dy * ang_x_rad * dy * ang_x_rad);
        float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
        float denom = sqrtf(denom_num2) * inv_r;
        float edge = (denom > 1e-15f) ? ((semiangle_rad - alpha) / denom + 0.5f) : 1.0f;
        aperture = fminf(fmaxf(edge, 0.0f), 1.0f);
    }

    return make_float4(alpha2, cos2phi, sin2phi, aperture);
}

__device__ __forceinline__ float compute_aperture_from_r2(
    float dx, float dy, float r2,
    float wavelength, float semiangle_rad,
    float ang_y_rad, float ang_x_rad,
    float inner2
) {
    if (inner2 > 0.0f && r2 <= inner2) {
        return 1.0f;
    }
    float r = sqrtf(r2);
    float alpha = r * wavelength;
    float denom_num2 = fmaf(dx * ang_y_rad, dx * ang_y_rad, dy * ang_x_rad * dy * ang_x_rad);
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = sqrtf(denom_num2) * inv_r;
    float edge = (denom > 1e-15f) ? ((semiangle_rad - alpha) / denom + 0.5f) : 1.0f;
    return fminf(fmaxf(edge, 0.0f), 1.0f);
}

__device__ __forceinline__ float2 gamma_mul_pk_onthefly(
    float qx, float qy,
    float kx, float ky,
    float wavelength, float semiangle_rad,
    float ang_y_rad, float ang_x_rad,
    float C10,
    float C12,
    float cos2phi12,
    float sin2phi12,
    float factor,
    float pk_re,
    float pk_im,
    float2 G
) {
    // q-k vector
    float4 m = compute_geometry(qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    // q+k vector
    float4 p = compute_geometry(qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);

    float alpha_m2 = m.x;
    float cos2phi_m = m.y;
    float sin2phi_m = m.z;
    float aperture_m = m.w;
    float alpha_p2 = p.x;
    float cos2phi_p = p.y;
    float sin2phi_p = p.z;
    float aperture_p = p.w;

    float cos_term_m = fmaf(cos2phi_m, cos2phi12, sin2phi_m * sin2phi12);
    float cos_term_p = fmaf(cos2phi_p, cos2phi12, sin2phi_p * sin2phi12);

    float chi_m = factor * alpha_m2 * fmaf(C12, cos_term_m, C10);
    float chi_p = factor * alpha_p2 * fmaf(C12, cos_term_p, C10);

    float sin_m, cos_m, sin_p, cos_p;
    __sincosf(chi_m, &sin_m, &cos_m);
    __sincosf(chi_p, &sin_p, &cos_p);

    float pm_re = aperture_m * cos_m;
    float pm_im = -aperture_m * sin_m;
    float pp_re = aperture_p * cos_p;
    float pp_im = -aperture_p * sin_p;

    float pk_conj_im = -pk_im;
    float t1_re = fmaf(pm_re, pk_re, -pm_im * pk_conj_im);
    float t1_im = fmaf(pm_re, pk_conj_im, pm_im * pk_re);

    float pp_conj_im = -pp_im;
    float t2_re = fmaf(pp_re, pk_re, -pp_conj_im * pk_im);
    float t2_im = fmaf(pp_re, pk_im, pp_conj_im * pk_re);

    float g_re = t1_re - t2_re;
    float g_im = t1_im - t2_im;

    float mag_sq = fmaf(g_re, g_re, g_im * g_im);
    float inv_mag = (mag_sq > 1e-16f) ? rsqrtf(mag_sq) : 1e8f;
    g_re *= inv_mag;
    g_im *= inv_mag;

    return make_float2(
        fmaf(G.x, g_re, G.y * g_im),
        fmaf(G.y, g_re, -G.x * g_im)
    );
}

__device__ __forceinline__ float2 gamma_mul_pk_cartesian_onthefly(
    float qx, float qy,
    float kx, float ky,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10,
    float C12,
    float cos2phi12,
    float sin2phi12,
    float factor,
    float phase_scale,
    float inner2,
    float pk_re,
    float pk_im,
    float2 G
) {
    float dx_m = qx - kx;
    float dy_m = qy - ky;
    float dx_p = qx + kx;
    float dy_p = qy + ky;
    float dx2_m = dx_m * dx_m;
    float dy2_m = dy_m * dy_m;
    float dx2_p = dx_p * dx_p;
    float dy2_p = dy_p * dy_p;
    float r2_m = dx2_m + dy2_m;
    float r2_p = dx2_p + dy2_p;
    float aperture_m = compute_aperture_from_r2(
        dx_m, dy_m, r2_m, wavelength, semiangle_rad, ang_y_rad, ang_x_rad, inner2);
    float aperture_p = compute_aperture_from_r2(
        dx_p, dy_p, r2_p, wavelength, semiangle_rad, ang_y_rad, ang_x_rad, inner2);
    float quad_m = fmaf(dx2_m - dy2_m, cos2phi12, (2.0f * dx_m * dy_m) * sin2phi12);
    float quad_p = fmaf(dx2_p - dy2_p, cos2phi12, (2.0f * dx_p * dy_p) * sin2phi12);
    float chi_m = phase_scale * fmaf(C10, r2_m, C12 * quad_m);
    float chi_p = phase_scale * fmaf(C10, r2_p, C12 * quad_p);

    float sin_m, cos_m, sin_p, cos_p;
    __sincosf(chi_m, &sin_m, &cos_m);
    __sincosf(chi_p, &sin_p, &cos_p);

    float pm_re = aperture_m * cos_m;
    float pm_im = -aperture_m * sin_m;
    float pp_re = aperture_p * cos_p;
    float pp_im = -aperture_p * sin_p;

    float pk_conj_im = -pk_im;
    float t1_re = fmaf(pm_re, pk_re, -pm_im * pk_conj_im);
    float t1_im = fmaf(pm_re, pk_conj_im, pm_im * pk_re);

    float pp_conj_im = -pp_im;
    float t2_re = fmaf(pp_re, pk_re, -pp_conj_im * pk_im);
    float t2_im = fmaf(pp_re, pk_im, pp_conj_im * pk_re);

    float g_re = t1_re - t2_re;
    float g_im = t1_im - t2_im;

    float mag_sq = fmaf(g_re, g_re, g_im * g_im);
    float inv_mag = (mag_sq > 1e-16f) ? rsqrtf(mag_sq) : 1e8f;
    g_re *= inv_mag;
    g_im *= inv_mag;

    return make_float2(
        fmaf(G.x, g_re, G.y * g_im),
        fmaf(G.y, g_re, -G.x * g_im)
    );
}

// Evaluate χ including all 14 Krivanek aberrations at a single (dx, dy)
// vector in reciprocal space.
//
// ---- Arithmetic formulation ----
//
// The Krivanek polar expansion is
//
//     χ = (2π/λ) · Σ_{n,m} 1/(n+1) · α^(n+1) · C_{n,m} · cos(m(φ - φ_{n,m})).
//
// Using the identity cos(m(φ - φ_{n,m})) = cos(mφ)·cos(mφ_{n,m}) + sin(mφ)·sin(mφ_{n,m}),
// the per-aberration factor cos(mφ_{n,m}) and sin(mφ_{n,m}) depend ONLY on
// the 14 orientation angles - constant across the whole scan, so we
// precompute them on the host (once per variance_loss_full call).
//
// Per-pixel the kernel computes cos(mφ) / sin(mφ) for m=1..6 directly from
// (dx, dy) via cos(φ)=dx/r, sin(φ)=dy/r and a Chebyshev recurrence.
// That replaces 14 cosf calls per chi evaluation with ~20 FMAs - one of
// the two large wins of this kernel vs the naive implementation.
//
// ---- Input array layout (all length 14) ----
//
//     abr_mag_scaled[i] = mags_m[i] / (n_i + 1)       - scale baked in
//     abr_cm[i]         = cos(m_i · angles_rad[i])    - host-precomputed
//     abr_sm[i]         = sin(m_i · angles_rad[i])    - host-precomputed
//
// in Krivanek order C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50,
// C52, C54, C56.
//
// For m_i = 0 (C10/C30/C50) the formula collapses to
// chi += α^(n+1) · abr_mag_scaled[i] (and we set abr_cm[i]=1, abr_sm[i]=0
// on the host so a uniform summation loop works).
//
__device__ __forceinline__ float chi_full(
    float dx, float dy,
    float wavelength,
    const float* __restrict__ abr_mag_scaled,
    const float* __restrict__ abr_cm,
    const float* __restrict__ abr_sm
) {
    float r2 = dx * dx + dy * dy;
    float inv_r = (r2 > 1e-30f) ? rsqrtf(r2) : 0.0f;
    float r = r2 * inv_r;                  // r = sqrt(r2)  (no extra sqrt)
    float alpha = r * wavelength;
    // Direct cos(φ) / sin(φ) from geometry - cheaper than atan2f + sincosf.
    float c1 = dx * inv_r;                 // cos(φ)
    float s1 = dy * inv_r;                 // sin(φ)

    // Chebyshev recurrence for cos(mφ), sin(mφ), m = 2..6.
    // Uses the identities
    //   cos((a+b)φ) = cos(aφ)cos(bφ) - sin(aφ)sin(bφ)
    //   sin((a+b)φ) = sin(aφ)cos(bφ) + cos(aφ)sin(bφ).
    float c2 = fmaf(2.0f * c1, c1, -1.0f);            // cos(2φ)
    float s2 = 2.0f * s1 * c1;                         // sin(2φ)
    float c3 = fmaf(c1, c2, -s1 * s2);                 // cos(3φ)
    float s3 = fmaf(s1, c2,  c1 * s2);                 // sin(3φ)
    float c4 = fmaf(c2, c2, -s2 * s2);                 // cos(4φ)
    float s4 = 2.0f * s2 * c2;                         // sin(4φ)
    float c5 = fmaf(c1, c4, -s1 * s4);                 // cos(5φ)
    float s5 = fmaf(s1, c4,  c1 * s4);                 // sin(5φ)
    float c6 = fmaf(c2, c4, -s2 * s4);                 // cos(6φ)
    float s6 = fmaf(s2, c4,  c2 * s4);                 // sin(6φ)

    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a4 = a2 * a2;
    float a5 = a4 * alpha;
    float a6 = a3 * a3;

    // Each contribution: α^(n+1) · abr_mag_scaled[i] · (c_m · abr_cm[i] + s_m · abr_sm[i]).
    // For m=0 aberrations (i = 0, 4, 10) we seed abr_cm=1, abr_sm=0, so the
    // bracket is literally 1.0 - no special-casing needed.
    float chi = 0.0f;
    // n = 1: C10 (m=0), C12 (m=2)
    chi = fmaf(a2 * abr_mag_scaled[0], 1.0f,
          fmaf(a2 * abr_mag_scaled[1], fmaf(c2, abr_cm[1],  s2 * abr_sm[1]),  chi));
    // n = 2: C21 (m=1), C23 (m=3)
    chi = fmaf(a3 * abr_mag_scaled[2], fmaf(c1, abr_cm[2],  s1 * abr_sm[2]),
          fmaf(a3 * abr_mag_scaled[3], fmaf(c3, abr_cm[3],  s3 * abr_sm[3]),  chi));
    // n = 3: C30 (m=0), C32 (m=2), C34 (m=4)
    chi = fmaf(a4 * abr_mag_scaled[4], 1.0f,
          fmaf(a4 * abr_mag_scaled[5], fmaf(c2, abr_cm[5],  s2 * abr_sm[5]),
          fmaf(a4 * abr_mag_scaled[6], fmaf(c4, abr_cm[6],  s4 * abr_sm[6]),  chi)));
    // n = 4: C41 (m=1), C43 (m=3), C45 (m=5)
    chi = fmaf(a5 * abr_mag_scaled[7], fmaf(c1, abr_cm[7],  s1 * abr_sm[7]),
          fmaf(a5 * abr_mag_scaled[8], fmaf(c3, abr_cm[8],  s3 * abr_sm[8]),
          fmaf(a5 * abr_mag_scaled[9], fmaf(c5, abr_cm[9],  s5 * abr_sm[9]),  chi)));
    // n = 5: C50 (m=0), C52 (m=2), C54 (m=4), C56 (m=6)
    chi = fmaf(a6 * abr_mag_scaled[10], 1.0f,
          fmaf(a6 * abr_mag_scaled[11], fmaf(c2, abr_cm[11], s2 * abr_sm[11]),
          fmaf(a6 * abr_mag_scaled[12], fmaf(c4, abr_cm[12], s4 * abr_sm[12]),
          fmaf(a6 * abr_mag_scaled[13], fmaf(c6, abr_cm[13], s6 * abr_sm[13]), chi))));

    return (6.2831853071795864f / wavelength) * chi;  // 2π/λ
}

// Full-aberration gamma multiplication.  Takes the same G_qk + pk inputs as
// the three-parameter gamma_mul_pk_onthefly but replaces the 2-term inline χ with
// chi_full above.  Called by the "full" variants of the fused FFT kernels.
__device__ __forceinline__ float2 gamma_mul_pk_onthefly_full(
    float qx, float qy,
    float kx, float ky,
    float wavelength, float semiangle_rad,
    float ang_y_rad, float ang_x_rad,
    const float* __restrict__ abr_mag_scaled,
    const float* __restrict__ abr_cm,
    const float* __restrict__ abr_sm,
    float pk_re,
    float pk_im,
    float2 G
) {
    // q-k and q+k vectors
    float dmx = qx - kx, dmy = qy - ky;
    float dpx = qx + kx, dpy = qy + ky;

    // Aperture (soft edge) - reuse the existing compute_geometry, we only
    // need the aperture from it.  alpha² / cos2phi / sin2phi are ignored.
    float4 m = compute_geometry(dmx, dmy, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p = compute_geometry(dpx, dpy, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float aperture_m = m.w;
    float aperture_p = p.w;

    // Full-polynomial χ at both shifted vectors using the fast Chebyshev path
    float chi_m = chi_full(dmx, dmy, wavelength, abr_mag_scaled, abr_cm, abr_sm);
    float chi_p = chi_full(dpx, dpy, wavelength, abr_mag_scaled, abr_cm, abr_sm);

    float sin_m, cos_m, sin_p, cos_p;
    __sincosf(chi_m, &sin_m, &cos_m);
    __sincosf(chi_p, &sin_p, &cos_p);

    float pm_re = aperture_m * cos_m;
    float pm_im = -aperture_m * sin_m;
    float pp_re = aperture_p * cos_p;
    float pp_im = -aperture_p * sin_p;

    float pk_conj_im = -pk_im;
    float t1_re = fmaf(pm_re, pk_re, -pm_im * pk_conj_im);
    float t1_im = fmaf(pm_re, pk_conj_im, pm_im * pk_re);

    float pp_conj_im = -pp_im;
    float t2_re = fmaf(pp_re, pk_re, -pp_conj_im * pk_im);
    float t2_im = fmaf(pp_re, pk_im, pp_conj_im * pk_re);

    float g_re = t1_re - t2_re;
    float g_im = t1_im - t2_im;

    float mag_sq = fmaf(g_re, g_re, g_im * g_im);
    float inv_mag = (mag_sq > 1e-16f) ? rsqrtf(mag_sq) : 1e8f;
    g_re *= inv_mag;
    g_im *= inv_mag;

    return make_float2(
        fmaf(G.x, g_re, G.y * g_im),
        fmaf(G.y, g_re, -G.x * g_im)
    );
}

__device__ __forceinline__ float atan2f_ssb_poly(float y, float x) {
    float ax = fabsf(x);
    float ay = fabsf(y);
    float hi = fmaxf(ax, ay);
    if (hi == 0.0f) {
        return 0.0f;
    }
    float a = fminf(ax, ay) / hi;
    float s = a * a;
    float p = 0.00773044f;
    p = fmaf(p, s, -0.03645544f);
    p = fmaf(p, s, 0.08302083f);
    p = fmaf(p, s, -0.13427427f);
    p = fmaf(p, s, 0.19861783f);
    p = fmaf(p, s, -0.33323847f);
    p = fmaf(p, s, 0.9999984f);
    float r = a * p;
    if (ay > ax) {
        r = 1.5707963267948966f - r;
    }
    if (x < 0.0f) {
        r = 3.1415926535897932f - r;
    }
    return (y < 0.0f) ? -r : r;
}
'''


# The corrected Fourier-sum kernel is the same for every scan size except the side length, so it is written once and
# spliced into each size module's source at the place it always had: the compiled modules stay byte-identical.
_CORRECTED_FOURIER_SUM = r'''__global__ void ssbSIZE_corrected_fourier_sum_t256(
    const float* __restrict__ kx_bf,
    const float* __restrict__ ky_bf,
    const float* __restrict__ qx_1d,
    const float* __restrict__ qy_1d,
    float wavelength,
    float semiangle_rad,
    float ang_y_rad,
    float ang_x_rad,
    float C10,
    float C12,
    float cos2phi12,
    float sin2phi12,
    float factor,
    const float2* __restrict__ pk,
    const float2* __restrict__ G_qk,
    float2* __restrict__ partial_sum,
    float dc_real,
    float dc_imag,
    int num_bf,
    int k_bf,
    int gqk_cols
) {
    unsigned long long linear = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long plane = SIZEull * SIZEull;
    int groups = (num_bf + k_bf - 1) / k_bf;
    unsigned long long total = (unsigned long long)groups * plane;
    if (linear >= total) return;

    int group = (int)(linear / plane);
    unsigned int idx = (unsigned int)(linear - (unsigned long long)group * plane);
    int row = idx / SIZEu;
    int col = idx - (unsigned int)row * SIZEu;
    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    if (idx == 0u) {
        float count = (float)(bf_end - bf_start);
        partial_sum[(unsigned long long)group * plane] = make_float2(count * dc_real, count * dc_imag);
        return;
    }

    float qx = __ldg(&qx_1d[row]);
    float qy = __ldg(&qy_1d[col]);
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    for (int bf = bf_start; bf < bf_end; ++bf) {
        float2 pkv = pk[bf];
        float2 v = gamma_mul_pk_onthefly(
            qx, qy,
            __ldg(&kx_bf[bf]), __ldg(&ky_bf[bf]),
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
            C10, C12, cos2phi12, sin2phi12, factor,
            pkv.x, pkv.y,
            ld_gqk_maybe_herm(
                G_qk, (unsigned long long)bf, (unsigned int)row,
                (unsigned int)col, SIZEu, (unsigned int)gqk_cols
            ));
        sum_re += v.x;
        sum_im += v.y;
    }
    partial_sum[(unsigned long long)group * plane + idx] = make_float2(sum_re, sum_im);
}'''


def corrected_fourier_sum_source(size: int) -> str:
    """CUDA source of ``ssb{size}_corrected_fourier_sum_t256`` for one square scan size.

    Each thread owns one scan-frequency pixel of one group of ``k_bf`` bright-field pixels and sums the
    aberration-corrected ``G_qk * conj(probe)`` over that group (the DC pixel holds the group size times the DC value),
    so the object-mode reconstruction reduces whole groups without materializing per-BF planes
    (``CustomFFTBase.corrected_fourier_partial_sum``).
    """
    return _CORRECTED_FOURIER_SUM.replace("SIZE", str(size))


def build_cuda_code(size: int, twiddle_decl: str, kernel_code: str) -> str:
    """Assemble full CUDA module: twiddle constant + device functions + kernels."""
    return f'extern "C" {{\n{twiddle_decl}\n{_DEVICE_FUNCTIONS_CUDA}\n{kernel_code}\n}}'


# Aberration index → (m, 1/(n+1)) for the Chebyshev chi_full kernel.
# Krivanek order C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50, C52,
# C54, C56, as in the chi_full kernel.
_ABR_N_PLUS_ONE_INV = np.asarray(
    [1/2, 1/2,   # n=1: C10, C12
     1/3, 1/3,   # n=2: C21, C23
     1/4, 1/4, 1/4,   # n=3: C30, C32, C34
     1/5, 1/5, 1/5,   # n=4: C41, C43, C45
     1/6, 1/6, 1/6, 1/6],  # n=5: C50, C52, C54, C56
    dtype=np.float32,
)
_ABR_M_VALUES = np.asarray(
    [0, 2,       # n=1
     1, 3,       # n=2
     0, 2, 4,    # n=3
     1, 3, 5,    # n=4
     0, 2, 4, 6],  # n=5
    dtype=np.float32,
)


def pack_aberration_coefs(
    mags_m: cp.ndarray, angles_rad: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Precompute Chebyshev-ready aberration coefficient arrays.

    Given raw ``mags_m[14]`` and ``angles_rad[14]`` produces three (14,)
    float32 CuPy arrays that the CUDA ``chi_full`` device function
    consumes directly:

    - ``abr_mag_scaled[i] = mags_m[i] / (n_i + 1)``
    - ``abr_cm[i]         = cos(m_i · angles_rad[i])``   (=1.0 for m=0)
    - ``abr_sm[i]         = sin(m_i · angles_rad[i])``   (=0.0 for m=0)

    Cost: 14 cos + 14 sin on the host plus a 168-byte DtoH copy - negligible
    compared to the per-pixel per-BF work inside the fused FFT kernel.
    """
    if mags_m.shape != (14,) or angles_rad.shape != (14,):
        raise ValueError("mags_m and angles_rad must have shape (14,)")
    mags_cpu = cp.asnumpy(mags_m).astype(np.float32, copy=False)
    angs_cpu = cp.asnumpy(angles_rad).astype(np.float32, copy=False)
    mag_scaled = mags_cpu * _ABR_N_PLUS_ONE_INV
    theta = _ABR_M_VALUES * angs_cpu
    cm = np.cos(theta, dtype=np.float32)
    sm = np.sin(theta, dtype=np.float32)
    return (
        cp.asarray(mag_scaled, dtype=cp.float32),
        cp.asarray(cm, dtype=cp.float32),
        cp.asarray(sm, dtype=cp.float32),
    )


# =========================================================================
#  Base Python class
# =========================================================================

class CustomFFTBase:
    """Base class for size-specific custom IFFT kernels.

    Subclasses provide CUDA kernel code and block/grid configuration.
    This class provides the shared dispatch logic.
    """

    def __init__(
        self,
        *,
        size: int,
        cuda_code: str,
        kernel_names: tuple[str, ...],
        twiddle_name: str,
        rows_block: tuple[int, int, int],
        rows_grid_y: int,
        cols_block: tuple[int, int, int],
        cols_grid_y: int,
    ) -> None:
        """``kernel_names``: the fused row IFFT, the column IFFT, then optionally the column-accumulate and the
        full-aberration row kernels; a subclass fetches any further kernel it names."""
        self._size = size
        options = ("--std=c++11", "--use_fast_math", "--maxrregcount=96", "-Xptxas=-dlcm=ca")
        self._module = cp.RawModule(
            code=cuda_code,
            options=options,
            name_expressions=kernel_names,
        )
        self._fourier_sum = self._module.get_function(f"ssb{size}_corrected_fourier_sum_t256")
        self._rows_fused_pk = self._module.get_function(kernel_names[0])
        self._cols = self._module.get_function(kernel_names[1])
        self._cols_accumulate = (
            self._module.get_function(kernel_names[2]) if len(kernel_names) > 2 else None
        )
        # Used by ifft2_inplace_fused_pk_full when a subclass names it.
        self._rows_fused_pk_full = (
            self._module.get_function(kernel_names[3]) if len(kernel_names) > 3 else None
        )
        self._colvar_group = 32
        self._rows_fused_pk_block = rows_block
        self._rows_fused_pk_grid_y = rows_grid_y
        self._cols_block = cols_block
        self._cols_grid_y = cols_grid_y
        self._twiddle_name = twiddle_name
        self._init_twiddles()

    def _init_twiddles(self) -> None:
        N = self._size
        w = np.exp(2j * math.pi * np.arange(N) / N).astype(np.complex64)
        memptr = self._module.get_global(self._twiddle_name)
        twiddle = cp.ndarray((N,), cp.complex64, memptr)
        twiddle.set(w)

    @staticmethod
    def _require_geometry(cache: dict) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray,
                                                 float, float, float, float]:
        """Extract on-the-fly geometry arrays and scalars from cache.

        Returns (kx_bf, ky_bf, qx_1d, qy_1d, wavelength, semiangle_rad, ang_y_rad, ang_x_rad).
        """
        kx_bf = cache.get("kx_bf")
        ky_bf = cache.get("ky_bf")
        qx_1d = cache.get("qx_1d")
        qy_1d = cache.get("qy_1d")
        if kx_bf is None or ky_bf is None or qx_1d is None or qy_1d is None:
            raise ValueError("Geometry arrays (kx_bf, ky_bf, qx_1d, qy_1d) missing from cache")
        return (kx_bf, ky_bf, qx_1d, qy_1d,
                cache["wavelength"], cache["semiangle_rad"],
                cache["ang_y_rad"], cache["ang_x_rad"])

    def ifft2_inplace_fused_pk(
        self,
        data: cp.ndarray,
        G_qk: cp.ndarray,
        cache: dict,
        pk: cp.ndarray,
        C10: float,
        C12: float,
        cos2phi12: float,
        sin2phi12: float,
        factor: float,
        dc_value: complex,
    ) -> None:
        """Fused gamma multiply + IFFT with pk."""
        N = self._size
        if data.dtype != cp.complex64 or G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 input")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bf = int(data.shape[0])
        if G_qk.ndim != 3 or G_qk.shape[0] != num_bf or G_qk.shape[1] != N:
            raise ValueError(f"G_qk must have shape (num_bf, {N}, {N}) or Hermitian")
        if G_qk.shape[2] not in (N, N // 2 + 1):
            raise ValueError(
                f"G_qk must have {N} columns or Hermitian {N // 2 + 1} columns"
            )
        gqk_cols = int(G_qk.shape[2])
        if pk.shape != (num_bf,):
            raise ValueError("pk must have shape (num_bf,)")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        grid_rows = (1, self._rows_fused_pk_grid_y, num_bf)
        self._rows_fused_pk(
            grid_rows,
            self._rows_fused_pk_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(gqk_cols),
            ),
        )
        scale = np.float32(1.0 / (N * N))
        grid_cols = (1, self._cols_grid_y, num_bf)
        self._cols(grid_cols, self._cols_block, (data, np.int32(num_bf), scale))

    def ifft2_inplace_fused_pk_full(
        self,
        data: cp.ndarray,
        G_qk: cp.ndarray,
        cache: dict,
        pk: cp.ndarray,
        mags_m: cp.ndarray,
        angles_rad: cp.ndarray,
        dc_value: complex,
    ) -> None:
        """Full-aberration version of ``ifft2_inplace_fused_pk``.

        Takes the raw ``mags_m``, ``angles_rad`` arrays (shape (14,)) from
        the caller - same public API as before.  Internally precomputes the
        Chebyshev-ready packed arrays and hands them to the `_full` row
        kernel, which replaces 14 ``cosf`` calls per chi evaluation with
        ~20 FMAs via ``chi_full``'s Chebyshev recurrence.
        """
        if self._rows_fused_pk_full is None:
            raise RuntimeError(
                "Full-aberration row kernel not registered for this FFT size. "
                "Rebuild CustomFFT subclass with the `_full` variant kernel."
            )
        N = self._size
        if data.dtype != cp.complex64 or G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 input")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bf = int(data.shape[0])
        if G_qk.ndim != 3 or G_qk.shape[0] != num_bf or G_qk.shape[1] != N:
            raise ValueError(f"G_qk must have shape (num_bf, {N}, {N}) or Hermitian")
        if G_qk.shape[2] not in (N, N // 2 + 1):
            raise ValueError(
                f"G_qk must have {N} columns or Hermitian {N // 2 + 1} columns"
            )
        gqk_cols = int(G_qk.shape[2])
        if pk.shape != (num_bf,):
            raise ValueError("pk must have shape (num_bf,)")
        if mags_m.dtype != cp.float32 or angles_rad.dtype != cp.float32:
            raise ValueError("mags_m and angles_rad must be float32 CuPy arrays")
        if mags_m.shape != (14,) or angles_rad.shape != (14,):
            raise ValueError("mags_m and angles_rad must have shape (14,)")
        abr_mag_scaled, abr_cm, abr_sm = pack_aberration_coefs(mags_m, angles_rad)
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        grid_rows = (1, self._rows_fused_pk_grid_y, num_bf)
        self._rows_fused_pk_full(
            grid_rows,
            self._rows_fused_pk_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                abr_mag_scaled, abr_cm, abr_sm,
                pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(gqk_cols),
            ),
        )
        scale = np.float32(1.0 / (N * N))
        grid_cols = (1, self._cols_grid_y, num_bf)
        self._cols(grid_cols, self._cols_block, (data, np.int32(num_bf), scale))

    def ifft2_fused_pk_col_accumulate(
        self,
        data: cp.ndarray,
        G_qk: cp.ndarray,
        cache: dict,
        pk: cp.ndarray,
        C10: float,
        C12: float,
        cos2phi12: float,
        sin2phi12: float,
        factor: float,
        dc_value: complex,
        partial_sum: cp.ndarray,
        partial_sumsq: cp.ndarray,
        k_bf: int = 32,
    ) -> None:
        """Row FFT + fused col-FFT with phase accumulation.

        Writes partial sum/sumsq planes instead of the complex result.
        """
        N = self._size
        if self._cols_accumulate is None:
            raise RuntimeError("col_accumulate kernel not available")
        if data.dtype != cp.complex64 or G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 input")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bf = int(data.shape[0])
        if G_qk.ndim != 3 or G_qk.shape[0] != num_bf or G_qk.shape[1] != N:
            raise ValueError(f"G_qk must have shape (num_bf, {N}, {N}) or Hermitian")
        if G_qk.shape[2] not in (N, N // 2 + 1):
            raise ValueError(
                f"G_qk must have {N} columns or Hermitian {N // 2 + 1} columns"
            )
        gqk_cols = int(G_qk.shape[2])
        if pk.shape != (num_bf,):
            raise ValueError("pk must have shape (num_bf,)")
        n_groups = (num_bf + k_bf - 1) // k_bf
        if partial_sum.shape != (n_groups, N, N) or partial_sumsq.shape != (n_groups, N, N):
            raise ValueError(f"partial buffers must have shape ({n_groups}, {N}, {N})")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        # Row FFT (writes to data)
        grid_rows = (1, self._rows_fused_pk_grid_y, num_bf)
        self._rows_fused_pk(
            grid_rows,
            self._rows_fused_pk_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(gqk_cols),
            ),
        )
        # Fused col-FFT + accumulate (reads data, writes partial buffers)
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate(
            grid_cols, self._cols_block,
            (data, partial_sum, partial_sumsq,
             np.int32(num_bf), np.int32(k_bf)),
        )

    def corrected_fourier_partial_sum(
        self,
        partial_sum: cp.ndarray,
        G_qk: cp.ndarray,
        cache: dict,
        pk: cp.ndarray,
        C10: float,
        C12: float,
        cos2phi12: float,
        sin2phi12: float,
        factor: float,
        dc_value: complex,
        k_bf: int = 32,
    ) -> None:
        """Accumulate corrected Fourier terms in BF groups.

        This is the exact linear path for ``reconstruct_object``:
        ``mean_bf(ifft2(corrected_bf)) == ifft2(mean_bf(corrected_bf))``.
        """
        N = self._size
        if G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 G_qk and pk")
        if G_qk.ndim != 3 or G_qk.shape[1] != N or G_qk.shape[2] not in (N, N // 2 + 1):
            raise ValueError(
                f"Expects G_qk shape (num_bf, {N}, {N}) or Hermitian "
                f"(num_bf, {N}, {N // 2 + 1})"
            )
        num_bf = int(G_qk.shape[0])
        gqk_cols = int(G_qk.shape[2])
        if pk.shape != (num_bf,):
            raise ValueError("pk must have shape (num_bf,)")
        n_groups = (num_bf + k_bf - 1) // k_bf
        if partial_sum.shape != (n_groups, N, N) or partial_sum.dtype != cp.complex64:
            raise ValueError(f"partial_sum must have complex64 shape ({n_groups}, {N}, {N})")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        total = n_groups * N * N
        block = (256,)
        grid = ((total + block[0] - 1) // block[0],)
        self._fourier_sum(
            grid,
            block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), pk, G_qk, partial_sum,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(k_bf), np.int32(gqk_cols),
            ),
        )
