"""Shared CUDA device functions and base class for SSB FFT kernels.

The CUDA device functions (complex arithmetic, geometry, gamma multiplication)
are identical across all scan sizes. The Python base class provides the shared
dispatch logic for ifft2_inplace_fused_pk and ifft2_inplace_batch_fused_pk_variance.

Size-specific kernels live in fft256.py and fft512.py.
"""

import math

import numpy as np
import cupy as cp

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

__device__ __forceinline__ float4 cadd2(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 csub2(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ __forceinline__ float4 cmul2(float2 w, float4 b) {
    return make_float4(
        w.x * b.x - w.y * b.y,
        w.x * b.y + w.y * b.x,
        w.x * b.z - w.y * b.w,
        w.x * b.w + w.y * b.z
    );
}

__device__ __forceinline__ float2 cmul_i(float2 a) {
    return make_float2(-a.y, a.x);
}

__device__ __forceinline__ float4 cmul_i2(float4 a) {
    return make_float4(-a.y, a.x, -a.w, a.z);
}

__device__ __forceinline__ float2 ld_float2(const float2* ptr, size_t idx) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr + idx);
#else
    return ptr[idx];
#endif
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
    float r = sqrtf(r2);
    float alpha = r * wavelength;
    float alpha2 = alpha * alpha;

    // cos2phi, sin2phi via algebraic identity (no atan2)
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2phi = (dx2 - dy2) * inv_r2;
    float sin2phi = 2.0f * dx * dy * inv_r2;

    // Soft aperture: clip((semiangle - alpha) / denom + 0.5, 0, 1)
    // denom = sqrt((cos_phi * ang_y)^2 + (sin_phi * ang_x)^2)
    //       = (1/r) * sqrt((dx * ang_y)^2 + (dy * ang_x)^2)
    float denom_num2 = fmaf(dx * ang_y_rad, dx * ang_y_rad, dy * ang_x_rad * dy * ang_x_rad);
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = sqrtf(denom_num2) * inv_r;
    float edge = (denom > 1e-15f) ? ((semiangle_rad - alpha) / denom + 0.5f) : 1.0f;
    float aperture = fminf(fmaxf(edge, 0.0f), 1.0f);

    return make_float4(alpha2, cos2phi, sin2phi, aperture);
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
// with the aberration index layout matching aberration.ABERRATION_INDICES.
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
// the legacy gamma_mul_pk_onthefly but replaces the 2-term inline χ with
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

__device__ __forceinline__ float2 gamma_mul_pk_packed_vals(
    float4 m,
    float4 p,
    float2 G,
    float C10,
    float C12,
    float cos2phi12,
    float sin2phi12,
    float factor,
    float pk_re,
    float pk_im
) {
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
'''


def build_cuda_code(size: int, twiddle_decl: str, kernel_code: str) -> str:
    """Assemble full CUDA module: twiddle constant + device functions + kernels."""
    return f'extern "C" {{\n{twiddle_decl}\n{_DEVICE_FUNCTIONS_CUDA}\n{kernel_code}\n}}'


# Aberration index → (m, 1/(n+1)) for the Chebyshev chi_full kernel.
# This layout matches aberration.ABERRATION_INDICES; kept in-module so the
# CUDA FFT dispatch can pack coefs without pulling aberration.py.
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
) -> "tuple[cp.ndarray, cp.ndarray, cp.ndarray]":
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
        batch_block: tuple[int, int, int],
        batch_grid_y: int,
        var_block: tuple[int, int, int],
        var_grid_y: int,
        cols_block: tuple[int, int, int],
        cols_grid_y: int,
        batch_shared_mem: int = 0,
    ) -> None:
        self._size = size
        options = ("--std=c++11", "--maxrregcount=96", "-Xptxas=-dlcm=cg")
        self._module = cp.RawModule(
            code=cuda_code,
            options=options,
            name_expressions=kernel_names,
        )
        self._rows_fused_pk = self._module.get_function(kernel_names[0])
        self._rows_fused_pk_batch_quad_transpose = self._module.get_function(kernel_names[1])
        self._rows_fused_pk_batch_shared_mem = int(batch_shared_mem)
        if self._rows_fused_pk_batch_shared_mem:
            self._rows_fused_pk_batch_quad_transpose.max_dynamic_shared_size_bytes = (
                self._rows_fused_pk_batch_shared_mem
            )
        self._cols = self._module.get_function(kernel_names[2])
        self._rows_var_batch = self._module.get_function(kernel_names[3])
        self._cols_accumulate = (
            self._module.get_function(kernel_names[4]) if len(kernel_names) > 4 else None
        )
        # Full-aberration row kernel.  Optional: subclasses register it as the
        # 6th kernel name when available.  Used by ifft2_inplace_fused_pk_full.
        self._rows_fused_pk_full = (
            self._module.get_function(kernel_names[5]) if len(kernel_names) > 5 else None
        )
        self.supports_batch_fused = True
        self._colvar_group = 32
        self._rows_fused_pk_block = rows_block
        self._rows_fused_pk_grid_y = rows_grid_y
        self._rows_fused_pk_block_quad = batch_block
        self._rows_fused_pk_grid_y_quad = batch_grid_y
        self._rows_var_block = var_block
        self._rows_var_grid_y = var_grid_y
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
        if data.shape != G_qk.shape:
            raise ValueError("data and G_qk must have the same shape")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bf = int(data.shape[0])
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
                np.int32(num_bf),
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
        if data.shape != G_qk.shape:
            raise ValueError("data and G_qk must have the same shape")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bf = int(data.shape[0])
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
                np.int32(num_bf),
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
        if data.shape != G_qk.shape:
            raise ValueError("data and G_qk must have the same shape")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bf = int(data.shape[0])
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
                np.int32(num_bf),
            ),
        )
        # Fused col-FFT + accumulate (reads data, writes partial buffers)
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate(
            grid_cols, self._cols_block,
            (data, partial_sum, partial_sumsq,
             np.int32(num_bf), np.int32(k_bf)),
        )

    def ifft2_inplace_batch_fused_pk_variance(
        self,
        data: cp.ndarray,
        G_qk: cp.ndarray,
        cache: dict,
        pk: cp.ndarray,
        C10: cp.ndarray,
        C12: cp.ndarray,
        cos2phi12: cp.ndarray,
        sin2phi12: cp.ndarray,
        factor: float,
        dc_value: complex,
        sum_buf: cp.ndarray,
        sumsq_buf: cp.ndarray,
        stream_bf: int = 0,
    ) -> None:
        """Fused gamma multiply + pk + IFFT with variance accumulation.

        When stream_bf > 0, processes BF pixels in small groups that fit in L2
        cache, reducing GMEM traffic for the staging buffer.
        """
        N = self._size
        if data.dtype != cp.complex64 or G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 input")
        batch = int(C10.size)
        num_bf = int(G_qk.shape[0])
        if stream_bf > 0 and stream_bf < num_bf:
            if batch < 4:
                raise ValueError("Batch size must be >= 4; caller should pad")
            (kx_bf, ky_bf, qx_1d, qy_1d,
             wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
            scale = np.float32(1.0 / (N * N))
            quads = (batch + 3) // 4
            dc_re = np.float32(dc_value.real)
            dc_im = np.float32(dc_value.imag)
            factor_f32 = np.float32(factor)
            wl_f32 = np.float32(wavelength)
            semi_f32 = np.float32(semiangle_rad)
            angy_f32 = np.float32(ang_y_rad)
            angx_f32 = np.float32(ang_x_rad)
            batch_i32 = np.int32(batch)
            partial_i32 = np.int32(0)
            pk_staging = cp.empty((batch, stream_bf), dtype=cp.complex64)
            tail = num_bf % stream_bf
            data_tail = cp.empty((batch, tail, N, N), dtype=cp.complex64) if tail else None
            for bf_start in range(0, num_bf, stream_bf):
                bf_end = min(num_bf, bf_start + stream_bf)
                chunk = bf_end - bf_start
                chunk_i32 = np.int32(chunk)
                staging = data if chunk == stream_bf else data_tail
                kx_group = kx_bf[bf_start:bf_end]
                ky_group = ky_bf[bf_start:bf_end]
                G_group = G_qk[bf_start:bf_end]
                if chunk == stream_bf:
                    pk_staging[:] = pk[:, bf_start:bf_end]
                    pk_group = pk_staging
                else:
                    pk_group = cp.ascontiguousarray(pk[:, bf_start:bf_end])
                grid_rows = (1, self._rows_fused_pk_grid_y_quad, quads * chunk)
                self._rows_fused_pk_batch_quad_transpose(
                    grid_rows, self._rows_fused_pk_block_quad,
                    (kx_group, ky_group, qx_1d, qy_1d,
                     wl_f32, semi_f32, angy_f32, angx_f32,
                     C10, C12, cos2phi12, sin2phi12,
                     factor_f32, pk_group, G_group,
                     staging, dc_re, dc_im,
                     chunk_i32, batch_i32),
                    shared_mem=self._rows_fused_pk_batch_shared_mem)
                groups_var = (chunk + self._colvar_group - 1) // self._colvar_group
                grid_var = (1, self._rows_var_grid_y, groups_var * batch)
                self._rows_var_batch(
                    grid_var, self._rows_var_block,
                    (staging, sum_buf, sumsq_buf,
                     chunk_i32, batch_i32, scale, partial_i32))
            return
        # Non-streaming path
        if data.ndim != 4 or data.shape[2] != N or data.shape[3] != N:
            raise ValueError(f"Expects shape (batch, num_bf, {N}, {N})")
        if C10.ndim != 1 or C12.ndim != 1 or cos2phi12.ndim != 1 or sin2phi12.ndim != 1:
            raise ValueError("C10/C12/phi12 arrays must be 1D")
        if not (C10.size == C12.size == cos2phi12.size == sin2phi12.size):
            raise ValueError("C10/C12/phi12 arrays must have matching lengths")
        if data.shape[0] != batch:
            raise ValueError("data batch dimension must match parameter arrays")
        if data.shape[1] != num_bf:
            raise ValueError("data BF dimension must match G_qk")
        if pk.shape != (batch, num_bf):
            raise ValueError("pk must have shape (batch, num_bf)")
        groups = (num_bf + self._colvar_group - 1) // self._colvar_group
        expected_full = (batch, N, N)
        expected_partial = (batch * groups, N, N)
        if (
            sum_buf.shape not in (expected_full, expected_partial)
            or sumsq_buf.shape not in (expected_full, expected_partial)
        ):
            raise ValueError(f"sum buffers must have shape (batch, {N}, {N}) or (batch*groups, {N}, {N})")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        if batch < 4:
            raise ValueError("Batch size must be >= 4; caller should pad")
        quads = (batch + 3) // 4
        grid_rows = (1, self._rows_fused_pk_grid_y_quad, quads * num_bf)
        self._rows_fused_pk_batch_quad_transpose(
            grid_rows, self._rows_fused_pk_block_quad,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                C10, C12, cos2phi12, sin2phi12,
                np.float32(factor), pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(batch),
            ),
            shared_mem=self._rows_fused_pk_batch_shared_mem,
        )
        scale = np.float32(1.0 / (N * N))
        groups = (num_bf + self._colvar_group - 1) // self._colvar_group
        use_partial = (
            sum_buf.ndim == 3
            and sum_buf.shape[0] == batch * groups
        )
        grid_rows_var = (1, self._rows_var_grid_y, groups * batch)
        self._rows_var_batch(
            grid_rows_var, self._rows_var_block,
            (data, sum_buf, sumsq_buf, np.int32(num_bf), np.int32(batch), scale,
             np.int32(1 if use_partial else 0)),
        )
