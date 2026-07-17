"""Custom fixed-size CUDA FFT kernels for SSB (512x512).

512 = 4^4 × 2: uses 4 radix-4 stages (m=4,16,64,256) plus a final
radix-2 stage. 128 threads per row, each handling 4 elements.
"""

from functools import lru_cache

import cupy as cp
import numpy as np

from .fft_common import CustomFFTBase, build_cuda_code

_TWIDDLE_DECL = '__constant__ float2 TWIDDLE_512[512];'

_FFT512_KERNELS = r'''
__device__ __forceinline__ unsigned int digit_reverse_512(unsigned int x) {
    return ((x & 1u) << 8) | bit_reverse4_8(x >> 1);
}

// 512 = 8^3 octal digit reversal: maps n = d2*64 + d1*8 + d0 to
// d0*64 + d1*8 + d2. Used by the radix-8 variance kernel.
__device__ __forceinline__ unsigned int octal_reverse_512(unsigned int n) {
    unsigned int d0 = n & 7u;
    unsigned int d1 = (n >> 3) & 7u;
    unsigned int d2 = n >> 6;
    return (d0 << 6) | (d1 << 3) | d2;
}

#define SQ2_INV_F 0.70710678118654752f

// Multiply by W8^1 = (1 + i) / sqrt(2) -- IDFT convention.
__device__ __forceinline__ float2 cmul_w8_1(float2 a) {
    return make_float2(SQ2_INV_F * (a.x - a.y),
                       SQ2_INV_F * (a.x + a.y));
}

// Multiply by W8^3 = (-1 + i) / sqrt(2) -- IDFT convention.
// (a + bi)(c + di) = (ac - bd) + (ad + bc)i with a = -1/sqrt2, b = 1/sqrt2.
__device__ __forceinline__ float2 cmul_w8_3(float2 a) {
    return make_float2(SQ2_INV_F * (-a.x - a.y),
                       SQ2_INV_F * ( a.x - a.y));
}

// 8-point IDFT butterfly via radix-2 * radix-4 decomposition.
// Inputs in natural order; internal 3-bit reversal is applied.
// Twiddle convention: w = exp(+2*pi*i/8) (IDFT, unnormalized).
// Cost: 2 non-trivial cmul (W8^1, W8^3) + 3 cmul_i (free) + 24 cadd/csub.
__device__ __forceinline__ void radix8_butterfly(
    float2 &x0, float2 &x1, float2 &x2, float2 &x3,
    float2 &x4, float2 &x5, float2 &x6, float2 &x7)
{
    // Internal 3-bit reversal: [0,4,2,6,1,5,3,7]
    float2 a0 = x0, a1 = x4, a2 = x2, a3 = x6;
    float2 a4 = x1, a5 = x5, a6 = x3, a7 = x7;
    // Stage A: 4 radix-2 butterflies, twiddle = 1
    float2 t0 = cadd(a0, a1), t1 = csub(a0, a1);
    float2 t2 = cadd(a2, a3), t3 = csub(a2, a3);
    float2 t4 = cadd(a4, a5), t5 = csub(a4, a5);
    float2 t6 = cadd(a6, a7), t7 = csub(a6, a7);
    // Stage B: 4 radix-2 butterflies, twiddles {1, i}
    float2 u0 = cadd(t0, t2), u2 = csub(t0, t2);
    float2 it3 = cmul_i(t3);
    float2 u1 = cadd(t1, it3), u3 = csub(t1, it3);
    float2 u4 = cadd(t4, t6), u6 = csub(t4, t6);
    float2 it7 = cmul_i(t7);
    float2 u5 = cadd(t5, it7), u7 = csub(t5, it7);
    // Stage C: 4 radix-2 butterflies, twiddles {1, W8^1, i, W8^3}
    float2 w1u5 = cmul_w8_1(u5);
    float2 w3u7 = cmul_w8_3(u7);
    float2 iu6  = cmul_i(u6);
    x0 = cadd(u0, u4);    x4 = csub(u0, u4);
    x1 = cadd(u1, w1u5);  x5 = csub(u1, w1u5);
    x2 = cadd(u2, iu6);   x6 = csub(u2, iu6);
    x3 = cadd(u3, w3u7);  x7 = csub(u3, w3u7);
}

// Compute geometry (alpha^2, cos2phi, sin2phi, aperture) for a displacement vector.
// Uses algebraic identities to avoid atan2/cos/sin:
//   cos(2phi) = (dx^2 - dy^2) / (dx^2 + dy^2)
//   sin(2phi) = 2*dx*dy / (dx^2 + dy^2)

__global__ void ifft512_rows_fused_pk_t128_mr4_packed(
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
	    float2* __restrict__ out,
	    float dc_real,
	    float dc_imag,
	    int num_bf,
	    int gqk_cols
	) {
    int bf = blockIdx.z;
    int row = blockIdx.y * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 512 || tid >= 128) {
        return;
    }
    size_t base = ((size_t)bf * 512 + row) * 512;
    int pos0 = tid;
    int pos1 = tid + 128;
    int pos2 = tid + 256;
    int pos3 = tid + 384;
    size_t idx0 = base + pos0;
    size_t idx1 = base + pos1;
    size_t idx2 = base + pos2;
    size_t idx3 = base + pos3;
    float2 pkv = pk[bf];
    float pk_re = pkv.x;
    float pk_im = pkv.y;

    // Load BF pixel coords (same for all pixels in this bf)
    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    // Load q-space coords
    float qx = __ldg(&qx_1d[row]);

	    float2 res0 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos0]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos0, 512u, (unsigned int)gqk_cols));
	    float2 res1 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 512u, (unsigned int)gqk_cols));
	    float2 res2 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 512u, (unsigned int)gqk_cols));
	    float2 res3 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 512u, (unsigned int)gqk_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[4][512];
    float2* srow = s[threadIdx.y];
    srow[digit_reverse_512((unsigned int)pos0)] = res0;
    srow[digit_reverse_512((unsigned int)pos1)] = res1;
    srow[digit_reverse_512((unsigned int)pos2)] = res2;
    srow[digit_reverse_512((unsigned int)pos3)] = res3;
    __syncthreads();

    for (int m = 4; m <= 256; m <<= 2) {
        int quarter = m >> 2;
        int butterfly = tid;
        int j = butterfly % quarter;
        int k = butterfly / quarter;
        int idx0s = k * m + j;
        int idx1s = idx0s + quarter;
        int idx2s = idx1s + quarter;
        int idx3s = idx2s + quarter;
        int tw = j * (512 / m);
        float2 x0 = srow[idx0s];
        float2 x1 = cmul(TWIDDLE_512[tw], srow[idx1s]);
        float2 x2 = cmul(TWIDDLE_512[tw * 2], srow[idx2s]);
        float2 x3 = cmul(TWIDDLE_512[tw * 3], srow[idx3s]);

        float2 t0 = cadd(x0, x2);
        float2 t1 = csub(x0, x2);
        float2 t2 = cadd(x1, x3);
        float2 t3 = csub(x1, x3);
        float2 y0 = cadd(t0, t2);
        float2 y2 = csub(t0, t2);
        float2 it3 = cmul_i(t3);
        float2 y1 = cadd(t1, it3);
        float2 y3 = csub(t1, it3);

        srow[idx0s] = y0;
        srow[idx1s] = y1;
        srow[idx2s] = y2;
        srow[idx3s] = y3;
        __syncthreads();
    }

    // Radix-2 final stage (512 = 4^4 x 2)
    {
        int j0 = tid;
        int j1 = tid + 128;
        float2 w0 = TWIDDLE_512[j0];
        float2 w1 = TWIDDLE_512[j1];
        float2 a0 = srow[j0], b0 = cmul(w0, srow[j0 + 256]);
        float2 a1 = srow[j1], b1 = cmul(w1, srow[j1 + 256]);
        srow[j0] = cadd(a0, b0);
        srow[j0 + 256] = csub(a0, b0);
        srow[j1] = cadd(a1, b1);
        srow[j1 + 256] = csub(a1, b1);
        __syncthreads();
    }

    out[idx0] = srow[pos0];
    out[idx1] = srow[pos1];
    out[idx2] = srow[pos2];
    out[idx3] = srow[pos3];
}

// Full-aberration variant of ifft512_rows_fused_pk_t128_mr4_packed.
// Structurally identical but calls gamma_mul_pk_onthefly_full with the
// host-precomputed Chebyshev-ready coefficient arrays (see chi_full docs
// in fft_common.py) instead of raw (mags, angles).  Legacy kernel is
// preserved for the Optuna 3-param hot path.
__global__ void ifft512_rows_fused_pk_full_t128_mr4_packed(
    const float* __restrict__ kx_bf,
    const float* __restrict__ ky_bf,
    const float* __restrict__ qx_1d,
    const float* __restrict__ qy_1d,
    float wavelength,
    float semiangle_rad,
    float ang_y_rad,
    float ang_x_rad,
    const float* __restrict__ abr_mag_scaled,
    const float* __restrict__ abr_cm,
    const float* __restrict__ abr_sm,
    const float2* __restrict__ pk,
    const float2* __restrict__ G_qk,
	    float2* __restrict__ out,
	    float dc_real,
	    float dc_imag,
	    int num_bf,
	    int gqk_cols
	) {
    int bf = blockIdx.z;
    int row = blockIdx.y * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 512 || tid >= 128) {
        return;
    }
    size_t base = ((size_t)bf * 512 + row) * 512;
    int pos0 = tid;
    int pos1 = tid + 128;
    int pos2 = tid + 256;
    int pos3 = tid + 384;
    size_t idx0 = base + pos0;
    size_t idx1 = base + pos1;
    size_t idx2 = base + pos2;
    size_t idx3 = base + pos3;
    float2 pkv = pk[bf];
    float pk_re = pkv.x;
    float pk_im = pkv.y;

    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    float qx = __ldg(&qx_1d[row]);

	    float2 res0 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos0]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos0, 512u, (unsigned int)gqk_cols));
	    float2 res1 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 512u, (unsigned int)gqk_cols));
	    float2 res2 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 512u, (unsigned int)gqk_cols));
	    float2 res3 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 512u, (unsigned int)gqk_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[4][512];
    float2* srow = s[threadIdx.y];
    srow[digit_reverse_512((unsigned int)pos0)] = res0;
    srow[digit_reverse_512((unsigned int)pos1)] = res1;
    srow[digit_reverse_512((unsigned int)pos2)] = res2;
    srow[digit_reverse_512((unsigned int)pos3)] = res3;
    __syncthreads();

    for (int m = 4; m <= 256; m <<= 2) {
        int quarter = m >> 2;
        int butterfly = tid;
        int j = butterfly % quarter;
        int k = butterfly / quarter;
        int idx0s = k * m + j;
        int idx1s = idx0s + quarter;
        int idx2s = idx1s + quarter;
        int idx3s = idx2s + quarter;
        int tw = j * (512 / m);
        float2 x0 = srow[idx0s];
        float2 x1 = cmul(TWIDDLE_512[tw], srow[idx1s]);
        float2 x2 = cmul(TWIDDLE_512[tw * 2], srow[idx2s]);
        float2 x3 = cmul(TWIDDLE_512[tw * 3], srow[idx3s]);

        float2 t0 = cadd(x0, x2);
        float2 t1 = csub(x0, x2);
        float2 t2 = cadd(x1, x3);
        float2 t3 = csub(x1, x3);
        float2 y0 = cadd(t0, t2);
        float2 y2 = csub(t0, t2);
        float2 it3 = cmul_i(t3);
        float2 y1 = cadd(t1, it3);
        float2 y3 = csub(t1, it3);

        srow[idx0s] = y0;
        srow[idx1s] = y1;
        srow[idx2s] = y2;
        srow[idx3s] = y3;
        __syncthreads();
    }

    // Radix-2 final stage (512 = 4^4 x 2)
    {
        int j0 = tid;
        int j1 = tid + 128;
        float2 w0 = TWIDDLE_512[j0];
        float2 w1 = TWIDDLE_512[j1];
        float2 a0 = srow[j0], b0 = cmul(w0, srow[j0 + 256]);
        float2 a1 = srow[j1], b1 = cmul(w1, srow[j1 + 256]);
        srow[j0] = cadd(a0, b0);
        srow[j0 + 256] = csub(a0, b0);
        srow[j1] = cadd(a1, b1);
        srow[j1 + 256] = csub(a1, b1);
        __syncthreads();
    }

    out[idx0] = srow[pos0];
    out[idx1] = srow[pos1];
    out[idx2] = srow[pos2];
    out[idx3] = srow[pos3];
}

__device__ __forceinline__ void ifft512_radix8_apply_t64(
    float2 &r0, float2 &r1, float2 &r2, float2 &r3,
    float2 &r4, float2 &r5, float2 &r6, float2 &r7,
    int tid,
    float2* __restrict__ sbuf
) {
    int s2_pre = tid & 7;
    float2 tw2_1 = TWIDDLE_512[(s2_pre * 1 * 8) & 511];
    float2 tw2_2 = TWIDDLE_512[(s2_pre * 2 * 8) & 511];
    float2 tw2_3 = TWIDDLE_512[(s2_pre * 3 * 8) & 511];
    float2 tw2_4 = TWIDDLE_512[(s2_pre * 4 * 8) & 511];
    float2 tw2_5 = TWIDDLE_512[(s2_pre * 5 * 8) & 511];
    float2 tw2_6 = TWIDDLE_512[(s2_pre * 6 * 8) & 511];
    float2 tw2_7 = TWIDDLE_512[(s2_pre * 7 * 8) & 511];
    float2 tw3_1 = TWIDDLE_512[(tid * 1) & 511];
    float2 tw3_2 = TWIDDLE_512[(tid * 2) & 511];
    float2 tw3_3 = TWIDDLE_512[(tid * 3) & 511];
    float2 tw3_4 = TWIDDLE_512[(tid * 4) & 511];
    float2 tw3_5 = TWIDDLE_512[(tid * 5) & 511];
    float2 tw3_6 = TWIDDLE_512[(tid * 6) & 511];
    float2 tw3_7 = TWIDDLE_512[(tid * 7) & 511];

    radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

    #define SHFL_XOR_F2_LOCAL(val, mask) make_float2( \
        __shfl_xor_sync(0xffffffff, (val).x, (mask)), \
        __shfl_xor_sync(0xffffffff, (val).y, (mask)))
    {
        float2 sent, got;
        sent = (tid & 1) ? r0 : r1;
        got = SHFL_XOR_F2_LOCAL(sent, 1);
        if (tid & 1) r0 = got; else r1 = got;
        sent = (tid & 1) ? r2 : r3;
        got = SHFL_XOR_F2_LOCAL(sent, 1);
        if (tid & 1) r2 = got; else r3 = got;
        sent = (tid & 1) ? r4 : r5;
        got = SHFL_XOR_F2_LOCAL(sent, 1);
        if (tid & 1) r4 = got; else r5 = got;
        sent = (tid & 1) ? r6 : r7;
        got = SHFL_XOR_F2_LOCAL(sent, 1);
        if (tid & 1) r6 = got; else r7 = got;
        sent = (tid & 2) ? r0 : r2;
        got = SHFL_XOR_F2_LOCAL(sent, 2);
        if (tid & 2) r0 = got; else r2 = got;
        sent = (tid & 2) ? r1 : r3;
        got = SHFL_XOR_F2_LOCAL(sent, 2);
        if (tid & 2) r1 = got; else r3 = got;
        sent = (tid & 2) ? r4 : r6;
        got = SHFL_XOR_F2_LOCAL(sent, 2);
        if (tid & 2) r4 = got; else r6 = got;
        sent = (tid & 2) ? r5 : r7;
        got = SHFL_XOR_F2_LOCAL(sent, 2);
        if (tid & 2) r5 = got; else r7 = got;
        sent = (tid & 4) ? r0 : r4;
        got = SHFL_XOR_F2_LOCAL(sent, 4);
        if (tid & 4) r0 = got; else r4 = got;
        sent = (tid & 4) ? r1 : r5;
        got = SHFL_XOR_F2_LOCAL(sent, 4);
        if (tid & 4) r1 = got; else r5 = got;
        sent = (tid & 4) ? r2 : r6;
        got = SHFL_XOR_F2_LOCAL(sent, 4);
        if (tid & 4) r2 = got; else r6 = got;
        sent = (tid & 4) ? r3 : r7;
        got = SHFL_XOR_F2_LOCAL(sent, 4);
        if (tid & 4) r3 = got; else r7 = got;
    }
    #undef SHFL_XOR_F2_LOCAL

    r1 = cmul(tw2_1, r1);
    r2 = cmul(tw2_2, r2);
    r3 = cmul(tw2_3, r3);
    r4 = cmul(tw2_4, r4);
    r5 = cmul(tw2_5, r5);
    r6 = cmul(tw2_6, r6);
    r7 = cmul(tw2_7, r7);

    radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

    int g_outer = tid >> 3;
    int base2 = g_outer * 64 + s2_pre;
    sbuf[base2 +  0] = r0;
    sbuf[base2 +  8] = r1;
    sbuf[base2 + 16] = r2;
    sbuf[base2 + 24] = r3;
    sbuf[base2 + 32] = r4;
    sbuf[base2 + 40] = r5;
    sbuf[base2 + 48] = r6;
    sbuf[base2 + 56] = r7;
    __syncthreads();

    r0 = sbuf[tid +   0];
    r1 = cmul(tw3_1, sbuf[tid +  64]);
    r2 = cmul(tw3_2, sbuf[tid + 128]);
    r3 = cmul(tw3_3, sbuf[tid + 192]);
    r4 = cmul(tw3_4, sbuf[tid + 256]);
    r5 = cmul(tw3_5, sbuf[tid + 320]);
    r6 = cmul(tw3_6, sbuf[tid + 384]);
    r7 = cmul(tw3_7, sbuf[tid + 448]);

    radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);
}

__global__ __launch_bounds__(64, 10)
void ifft512_rows_fused_pk_radix8_t64_packed(
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
    float phase_scale,
    float inner2,
    const float2* __restrict__ pk,
    const float2* __restrict__ G_qk,
    float2* __restrict__ out,
    float dc_real,
    float dc_imag,
    int num_bf,
    int gqk_cols
) {
    int bf = blockIdx.z;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 512 || tid >= 64) return;

    int src0 = (int)octal_reverse_512((unsigned int)(tid*8 + 0));
    int src1 = (int)octal_reverse_512((unsigned int)(tid*8 + 1));
    int src2 = (int)octal_reverse_512((unsigned int)(tid*8 + 2));
    int src3 = (int)octal_reverse_512((unsigned int)(tid*8 + 3));
    int src4 = (int)octal_reverse_512((unsigned int)(tid*8 + 4));
    int src5 = (int)octal_reverse_512((unsigned int)(tid*8 + 5));
    int src6 = (int)octal_reverse_512((unsigned int)(tid*8 + 6));
    int src7 = (int)octal_reverse_512((unsigned int)(tid*8 + 7));

    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    float qx = __ldg(&qx_1d[row]);
    float2 pkv = pk[bf];
    float pk_re = pkv.x;
    float pk_im = pkv.y;

    float2 r0 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src0]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src0, 512u, (unsigned int)gqk_cols));
    float2 r1 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src1]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src1, 512u, (unsigned int)gqk_cols));
    float2 r2 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src2]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src2, 512u, (unsigned int)gqk_cols));
    float2 r3 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src3]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src3, 512u, (unsigned int)gqk_cols));
    float2 r4 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src4]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src4, 512u, (unsigned int)gqk_cols));
    float2 r5 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src5]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src5, 512u, (unsigned int)gqk_cols));
    float2 r6 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src6]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src6, 512u, (unsigned int)gqk_cols));
    float2 r7 = gamma_mul_pk_cartesian_onthefly(
        qx, __ldg(&qy_1d[src7]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        phase_scale, inner2,
        pk_re, pk_im,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)bf, (unsigned int)row,
                          (unsigned int)src7, 512u, (unsigned int)gqk_cols));

    if (row == 0) {
        if (src0 == 0) r0 = make_float2(dc_real, dc_imag);
        if (src1 == 0) r1 = make_float2(dc_real, dc_imag);
        if (src2 == 0) r2 = make_float2(dc_real, dc_imag);
        if (src3 == 0) r3 = make_float2(dc_real, dc_imag);
        if (src4 == 0) r4 = make_float2(dc_real, dc_imag);
        if (src5 == 0) r5 = make_float2(dc_real, dc_imag);
        if (src6 == 0) r6 = make_float2(dc_real, dc_imag);
        if (src7 == 0) r7 = make_float2(dc_real, dc_imag);
    }

    int s2_pre = tid & 7;
    float2 tw2_1 = TWIDDLE_512[(s2_pre * 1 * 8) & 511];
    float2 tw2_2 = TWIDDLE_512[(s2_pre * 2 * 8) & 511];
    float2 tw2_3 = TWIDDLE_512[(s2_pre * 3 * 8) & 511];
    float2 tw2_4 = TWIDDLE_512[(s2_pre * 4 * 8) & 511];
    float2 tw2_5 = TWIDDLE_512[(s2_pre * 5 * 8) & 511];
    float2 tw2_6 = TWIDDLE_512[(s2_pre * 6 * 8) & 511];
    float2 tw2_7 = TWIDDLE_512[(s2_pre * 7 * 8) & 511];
    float2 tw3_1 = TWIDDLE_512[(tid * 1) & 511];
    float2 tw3_2 = TWIDDLE_512[(tid * 2) & 511];
    float2 tw3_3 = TWIDDLE_512[(tid * 3) & 511];
    float2 tw3_4 = TWIDDLE_512[(tid * 4) & 511];
    float2 tw3_5 = TWIDDLE_512[(tid * 5) & 511];
    float2 tw3_6 = TWIDDLE_512[(tid * 6) & 511];
    float2 tw3_7 = TWIDDLE_512[(tid * 7) & 511];

    radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

    #define SHFL_XOR_F2(val, mask) make_float2( \
        __shfl_xor_sync(0xffffffff, (val).x, (mask)), \
        __shfl_xor_sync(0xffffffff, (val).y, (mask)))
    {
        float2 sent, got;
        sent = (tid & 1) ? r0 : r1;
        got = SHFL_XOR_F2(sent, 1);
        if (tid & 1) r0 = got; else r1 = got;
        sent = (tid & 1) ? r2 : r3;
        got = SHFL_XOR_F2(sent, 1);
        if (tid & 1) r2 = got; else r3 = got;
        sent = (tid & 1) ? r4 : r5;
        got = SHFL_XOR_F2(sent, 1);
        if (tid & 1) r4 = got; else r5 = got;
        sent = (tid & 1) ? r6 : r7;
        got = SHFL_XOR_F2(sent, 1);
        if (tid & 1) r6 = got; else r7 = got;
        sent = (tid & 2) ? r0 : r2;
        got = SHFL_XOR_F2(sent, 2);
        if (tid & 2) r0 = got; else r2 = got;
        sent = (tid & 2) ? r1 : r3;
        got = SHFL_XOR_F2(sent, 2);
        if (tid & 2) r1 = got; else r3 = got;
        sent = (tid & 2) ? r4 : r6;
        got = SHFL_XOR_F2(sent, 2);
        if (tid & 2) r4 = got; else r6 = got;
        sent = (tid & 2) ? r5 : r7;
        got = SHFL_XOR_F2(sent, 2);
        if (tid & 2) r5 = got; else r7 = got;
        sent = (tid & 4) ? r0 : r4;
        got = SHFL_XOR_F2(sent, 4);
        if (tid & 4) r0 = got; else r4 = got;
        sent = (tid & 4) ? r1 : r5;
        got = SHFL_XOR_F2(sent, 4);
        if (tid & 4) r1 = got; else r5 = got;
        sent = (tid & 4) ? r2 : r6;
        got = SHFL_XOR_F2(sent, 4);
        if (tid & 4) r2 = got; else r6 = got;
        sent = (tid & 4) ? r3 : r7;
        got = SHFL_XOR_F2(sent, 4);
        if (tid & 4) r3 = got; else r7 = got;
    }
    #undef SHFL_XOR_F2

    r1 = cmul(tw2_1, r1);
    r2 = cmul(tw2_2, r2);
    r3 = cmul(tw2_3, r3);
    r4 = cmul(tw2_4, r4);
    r5 = cmul(tw2_5, r5);
    r6 = cmul(tw2_6, r6);
    r7 = cmul(tw2_7, r7);

    radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

    __shared__ float2 sbuf[512];
    int g_outer = tid >> 3;
    int base2 = g_outer * 64 + s2_pre;
    sbuf[base2 +  0] = r0;
    sbuf[base2 +  8] = r1;
    sbuf[base2 + 16] = r2;
    sbuf[base2 + 24] = r3;
    sbuf[base2 + 32] = r4;
    sbuf[base2 + 40] = r5;
    sbuf[base2 + 48] = r6;
    sbuf[base2 + 56] = r7;
    __syncthreads();

    r0 = sbuf[tid +   0];
    r1 = cmul(tw3_1, sbuf[tid +  64]);
    r2 = cmul(tw3_2, sbuf[tid + 128]);
    r3 = cmul(tw3_3, sbuf[tid + 192]);
    r4 = cmul(tw3_4, sbuf[tid + 256]);
    r5 = cmul(tw3_5, sbuf[tid + 320]);
    r6 = cmul(tw3_6, sbuf[tid + 384]);
    r7 = cmul(tw3_7, sbuf[tid + 448]);

    radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

    size_t out_base = (size_t)bf * 512u * 512u + (size_t)row;
    out[out_base + (size_t)(tid +   0) * 512u] = r0;
    out[out_base + (size_t)(tid +  64) * 512u] = r1;
    out[out_base + (size_t)(tid + 128) * 512u] = r2;
    out[out_base + (size_t)(tid + 192) * 512u] = r3;
    out[out_base + (size_t)(tid + 256) * 512u] = r4;
    out[out_base + (size_t)(tid + 320) * 512u] = r5;
    out[out_base + (size_t)(tid + 384) * 512u] = r6;
    out[out_base + (size_t)(tid + 448) * 512u] = r7;
}

__global__ __launch_bounds__(256, 3)
void ifft512_rows_fused_pk_pair_radix8_t64_packed(
    const int* __restrict__ pair_a,
    const int* __restrict__ pair_b,
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
    float phase_scale,
    float inner2,
    const float2* __restrict__ pk,
    const float2* __restrict__ G_qk,
    float2* __restrict__ out,
    float dc_real,
    float dc_imag,
    int num_pairs,
    int gqk_cols
) {
    int pair = blockIdx.z;
    int row = blockIdx.y * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (pair >= num_pairs || row >= 512 || tid >= 64) return;

    int idx_a = __ldg(&pair_a[pair]);
    int idx_b = __ldg(&pair_b[pair]);
    int src0 = (int)octal_reverse_512((unsigned int)(tid*8 + 0));
    int src1 = (int)octal_reverse_512((unsigned int)(tid*8 + 1));
    int src2 = (int)octal_reverse_512((unsigned int)(tid*8 + 2));
    int src3 = (int)octal_reverse_512((unsigned int)(tid*8 + 3));
    int src4 = (int)octal_reverse_512((unsigned int)(tid*8 + 4));
    int src5 = (int)octal_reverse_512((unsigned int)(tid*8 + 5));
    int src6 = (int)octal_reverse_512((unsigned int)(tid*8 + 6));
    int src7 = (int)octal_reverse_512((unsigned int)(tid*8 + 7));

    float kx = __ldg(&kx_bf[idx_a]);
    float ky = __ldg(&ky_bf[idx_a]);
    float qx = __ldg(&qx_1d[row]);
    float2 pka = pk[idx_a];

    float2 a0, a1, a2, a3, a4, a5, a6, a7;
    float2 b0, b1, b2, b3, b4, b5, b6, b7;
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src0]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src0, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src0, 512u, (unsigned int)gqk_cols),
        a0, b0);
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src1]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src1, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src1, 512u, (unsigned int)gqk_cols),
        a1, b1);
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src2]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src2, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src2, 512u, (unsigned int)gqk_cols),
        a2, b2);
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src3]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src3, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src3, 512u, (unsigned int)gqk_cols),
        a3, b3);
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src4]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src4, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src4, 512u, (unsigned int)gqk_cols),
        a4, b4);
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src5]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src5, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src5, 512u, (unsigned int)gqk_cols),
        a5, b5);
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src6]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src6, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src6, 512u, (unsigned int)gqk_cols),
        a6, b6);
    gamma_mul_pk_pair_onthefly(
        qx, __ldg(&qy_1d[src7]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2,
        pka.x, pka.y,
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row,
                          (unsigned int)src7, 512u, (unsigned int)gqk_cols),
        ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row,
                          (unsigned int)src7, 512u, (unsigned int)gqk_cols),
        a7, b7);

    if (row == 0) {
        if (src0 == 0) { a0 = make_float2(dc_real, dc_imag); b0 = make_float2(dc_real, dc_imag); }
        if (src1 == 0) { a1 = make_float2(dc_real, dc_imag); b1 = make_float2(dc_real, dc_imag); }
        if (src2 == 0) { a2 = make_float2(dc_real, dc_imag); b2 = make_float2(dc_real, dc_imag); }
        if (src3 == 0) { a3 = make_float2(dc_real, dc_imag); b3 = make_float2(dc_real, dc_imag); }
        if (src4 == 0) { a4 = make_float2(dc_real, dc_imag); b4 = make_float2(dc_real, dc_imag); }
        if (src5 == 0) { a5 = make_float2(dc_real, dc_imag); b5 = make_float2(dc_real, dc_imag); }
        if (src6 == 0) { a6 = make_float2(dc_real, dc_imag); b6 = make_float2(dc_real, dc_imag); }
        if (src7 == 0) { a7 = make_float2(dc_real, dc_imag); b7 = make_float2(dc_real, dc_imag); }
    }

    __shared__ float2 sbuf_all[4][1024];
    float2* sbuf = sbuf_all[threadIdx.y];
    ifft512_radix8_apply_t64(a0, a1, a2, a3, a4, a5, a6, a7, tid, sbuf);
    ifft512_radix8_apply_t64(b0, b1, b2, b3, b4, b5, b6, b7, tid, sbuf + 512);

    size_t slot_a = (size_t)pair * 2u;
    sbuf[tid +   0] = a0;
    sbuf[tid +  64] = a1;
    sbuf[tid + 128] = a2;
    sbuf[tid + 192] = a3;
    sbuf[tid + 256] = a4;
    sbuf[tid + 320] = a5;
    sbuf[tid + 384] = a6;
    sbuf[tid + 448] = a7;
    sbuf[512 + tid +   0] = b0;
    sbuf[512 + tid +  64] = b1;
    sbuf[512 + tid + 128] = b2;
    sbuf[512 + tid + 192] = b3;
    sbuf[512 + tid + 256] = b4;
    sbuf[512 + tid + 320] = b5;
    sbuf[512 + tid + 384] = b6;
    sbuf[512 + tid + 448] = b7;
    __syncthreads();

    int row_base = blockIdx.y * 4;
    int linear = threadIdx.y * 64 + tid;
    #pragma unroll
    for (int t = linear; t < 4096; t += 256) {
        int slot_offset = t >> 11;
        int rem = t & 2047;
        int col = rem >> 2;
        int local_row = rem & 3;
        float2 v = sbuf_all[local_row][(slot_offset << 9) + col];
        size_t out_idx = (slot_a + (size_t)slot_offset) * 512u * 512u
                       + (size_t)col * 512u
                       + (size_t)(row_base + local_row);
        out[out_idx] = v;
    }
}

__global__ __launch_bounds__(256, 4)
void ifft512_rows_fused_pk_dual_radix8_t64_packed(
    const int* __restrict__ pair_a,
    const int* __restrict__ pair_b,
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
    float phase_scale,
    float inner2,
    const float2* __restrict__ pk,
    const float2* __restrict__ G_qk,
    float2* __restrict__ out,
    float dc_real,
    float dc_imag,
    int num_pairs,
    int gqk_cols
) {
    int pair = blockIdx.z;
    int row = blockIdx.y * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (pair >= num_pairs || row >= 512 || tid >= 64) return;

    int idx_a = __ldg(&pair_a[pair]);
    int idx_b = __ldg(&pair_b[pair]);
    int src0 = (int)octal_reverse_512((unsigned int)(tid*8 + 0));
    int src1 = (int)octal_reverse_512((unsigned int)(tid*8 + 1));
    int src2 = (int)octal_reverse_512((unsigned int)(tid*8 + 2));
    int src3 = (int)octal_reverse_512((unsigned int)(tid*8 + 3));
    int src4 = (int)octal_reverse_512((unsigned int)(tid*8 + 4));
    int src5 = (int)octal_reverse_512((unsigned int)(tid*8 + 5));
    int src6 = (int)octal_reverse_512((unsigned int)(tid*8 + 6));
    int src7 = (int)octal_reverse_512((unsigned int)(tid*8 + 7));

    float kx_a = __ldg(&kx_bf[idx_a]);
    float ky_a = __ldg(&ky_bf[idx_a]);
    float kx_b = __ldg(&kx_bf[idx_b]);
    float ky_b = __ldg(&ky_bf[idx_b]);
    float qx = __ldg(&qx_1d[row]);
    float2 pka = pk[idx_a];
    float2 pkb = pk[idx_b];

    float2 a0, a1, a2, a3, a4, a5, a6, a7;
    float2 b0, b1, b2, b3, b4, b5, b6, b7;
    if (gqk_cols == 257) {
#define DUAL_EVAL(slot, src) \
        a##slot = gamma_mul_pk_cartesian_onthefly( \
            qx, __ldg(&qy_1d[src]), kx_a, ky_a, \
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad, \
            C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2, \
            pka.x, pka.y, \
            ld_gqk_herm_512(G_qk, (unsigned long long)idx_a, (unsigned int)row, \
                            (unsigned int)src)); \
        b##slot = gamma_mul_pk_cartesian_onthefly( \
            qx, __ldg(&qy_1d[src]), kx_b, ky_b, \
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad, \
            C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2, \
            pkb.x, pkb.y, \
            ld_gqk_herm_512(G_qk, (unsigned long long)idx_b, (unsigned int)row, \
                            (unsigned int)src))
        DUAL_EVAL(0, src0);
        DUAL_EVAL(1, src1);
        DUAL_EVAL(2, src2);
        DUAL_EVAL(3, src3);
        DUAL_EVAL(4, src4);
        DUAL_EVAL(5, src5);
        DUAL_EVAL(6, src6);
        DUAL_EVAL(7, src7);
#undef DUAL_EVAL
    } else {
#define DUAL_EVAL(slot, src) \
        a##slot = gamma_mul_pk_cartesian_onthefly( \
            qx, __ldg(&qy_1d[src]), kx_a, ky_a, \
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad, \
            C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2, \
            pka.x, pka.y, \
            ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_a, (unsigned int)row, \
                              (unsigned int)src, 512u, (unsigned int)gqk_cols)); \
        b##slot = gamma_mul_pk_cartesian_onthefly( \
            qx, __ldg(&qy_1d[src]), kx_b, ky_b, \
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad, \
            C10, C12, cos2phi12, sin2phi12, factor, phase_scale, inner2, \
            pkb.x, pkb.y, \
            ld_gqk_maybe_herm(G_qk, (unsigned long long)idx_b, (unsigned int)row, \
                              (unsigned int)src, 512u, (unsigned int)gqk_cols))
        DUAL_EVAL(0, src0);
        DUAL_EVAL(1, src1);
        DUAL_EVAL(2, src2);
        DUAL_EVAL(3, src3);
        DUAL_EVAL(4, src4);
        DUAL_EVAL(5, src5);
        DUAL_EVAL(6, src6);
        DUAL_EVAL(7, src7);
#undef DUAL_EVAL
    }

    if (row == 0) {
        if (src0 == 0) { a0 = make_float2(dc_real, dc_imag); b0 = make_float2(dc_real, dc_imag); }
        if (src1 == 0) { a1 = make_float2(dc_real, dc_imag); b1 = make_float2(dc_real, dc_imag); }
        if (src2 == 0) { a2 = make_float2(dc_real, dc_imag); b2 = make_float2(dc_real, dc_imag); }
        if (src3 == 0) { a3 = make_float2(dc_real, dc_imag); b3 = make_float2(dc_real, dc_imag); }
        if (src4 == 0) { a4 = make_float2(dc_real, dc_imag); b4 = make_float2(dc_real, dc_imag); }
        if (src5 == 0) { a5 = make_float2(dc_real, dc_imag); b5 = make_float2(dc_real, dc_imag); }
        if (src6 == 0) { a6 = make_float2(dc_real, dc_imag); b6 = make_float2(dc_real, dc_imag); }
        if (src7 == 0) { a7 = make_float2(dc_real, dc_imag); b7 = make_float2(dc_real, dc_imag); }
    }

    __shared__ float2 sbuf_all[4][1024];
    float2* sbuf = sbuf_all[threadIdx.y];
    ifft512_radix8_apply_t64(a0, a1, a2, a3, a4, a5, a6, a7, tid, sbuf);
    ifft512_radix8_apply_t64(b0, b1, b2, b3, b4, b5, b6, b7, tid, sbuf + 512);

    size_t slot_a = (size_t)pair * 2u;
    sbuf[tid +   0] = a0;
    sbuf[tid +  64] = a1;
    sbuf[tid + 128] = a2;
    sbuf[tid + 192] = a3;
    sbuf[tid + 256] = a4;
    sbuf[tid + 320] = a5;
    sbuf[tid + 384] = a6;
    sbuf[tid + 448] = a7;
    sbuf[512 + tid +   0] = b0;
    sbuf[512 + tid +  64] = b1;
    sbuf[512 + tid + 128] = b2;
    sbuf[512 + tid + 192] = b3;
    sbuf[512 + tid + 256] = b4;
    sbuf[512 + tid + 320] = b5;
    sbuf[512 + tid + 384] = b6;
    sbuf[512 + tid + 448] = b7;
    __syncthreads();

    int row_base = blockIdx.y * 4;
    int linear = threadIdx.y * 64 + tid;
    #pragma unroll
    for (int t = linear; t < 4096; t += 256) {
        int slot_offset = t >> 11;
        int rem = t & 2047;
        int col = rem >> 2;
        int local_row = rem & 3;
        float2 v = sbuf_all[local_row][(slot_offset << 9) + col];
        size_t out_idx = (slot_a + (size_t)slot_offset) * 512u * 512u
                       + (size_t)col * 512u
                       + (size_t)(row_base + local_row);
        out[out_idx] = v;
    }
}

__global__ __launch_bounds__(256, 4)
void ifft512_rows_fused_pk_batch_t128_mr2_transpose_packed_b4(
    const float* __restrict__ kx_bf,
    const float* __restrict__ ky_bf,
    const float* __restrict__ qx_1d,
    const float* __restrict__ qy_1d,
    float wavelength,
    float semiangle_rad,
    float ang_y_rad,
    float ang_x_rad,
    const float* __restrict__ C10,
    const float* __restrict__ C12,
    const float* __restrict__ cos2phi12,
    const float* __restrict__ sin2phi12,
    float factor,
    const float2* __restrict__ pk,
    const float2* __restrict__ G_qk,
    float2* __restrict__ out,
	    float dc_real,
	    float dc_imag,
	    int num_bf,
	    int batch,
	    int gqk_cols
	) {
    int idx = blockIdx.z;
    int bf = idx % num_bf;
    int quad = idx / num_bf;
    int cand0 = quad * 4;
    int cand1 = cand0 + 1;
    int cand2 = cand0 + 2;
    int cand3 = cand0 + 3;
    // CRITICAL: stride must be *8. block=(128,2,1) with grid_y=64
    // gives rows 0-1, 8-9, 16-17, ... (128 of 512 rows, interleaved).
    // mr2 (not mr4) because float4 s0[4][512]+s1[4][512]=64KB exceeds
    // the 48KB per-block shared memory limit. mr2 uses 32KB.
    int row = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (cand0 >= batch || bf >= num_bf || row >= 512 || tid >= 128) {
        return;
    }
    bool has1 = cand1 < batch;
    bool has2 = cand2 < batch;
    bool has3 = cand3 < batch;
    float C10v0 = C10[cand0];
    float C12v0 = C12[cand0];
    float cos2phi12v0 = cos2phi12[cand0];
    float sin2phi12v0 = sin2phi12[cand0];
    float C10v1 = has1 ? C10[cand1] : 0.0f;
    float C12v1 = has1 ? C12[cand1] : 0.0f;
    float cos2phi12v1 = has1 ? cos2phi12[cand1] : 0.0f;
    float sin2phi12v1 = has1 ? sin2phi12[cand1] : 0.0f;
    float C10v2 = has2 ? C10[cand2] : 0.0f;
    float C12v2 = has2 ? C12[cand2] : 0.0f;
    float cos2phi12v2 = has2 ? cos2phi12[cand2] : 0.0f;
    float sin2phi12v2 = has2 ? sin2phi12[cand2] : 0.0f;
    float C10v3 = has3 ? C10[cand3] : 0.0f;
    float C12v3 = has3 ? C12[cand3] : 0.0f;
    float cos2phi12v3 = has3 ? cos2phi12[cand3] : 0.0f;
    float sin2phi12v3 = has3 ? sin2phi12[cand3] : 0.0f;
    size_t base_cache = ((size_t)bf * 512 + row) * 512;
    int pos0 = tid;
    int pos1 = tid + 128;
    int pos2 = tid + 256;
    int pos3 = tid + 384;
    size_t pk_idx0 = cand0 * num_bf + bf;
    size_t pk_idx1 = cand1 * num_bf + bf;
    size_t pk_idx2 = cand2 * num_bf + bf;
    size_t pk_idx3 = cand3 * num_bf + bf;
    float2 pkv0 = pk[pk_idx0];
    float2 pkv1 = has1 ? pk[pk_idx1] : make_float2(0.0f, 0.0f);
    float2 pkv2 = has2 ? pk[pk_idx2] : make_float2(0.0f, 0.0f);
    float2 pkv3 = has3 ? pk[pk_idx3] : make_float2(0.0f, 0.0f);
    float pk0_re = pkv0.x;
    float pk0_im = pkv0.y;
    float pk1_re = pkv1.x;
    float pk1_im = pkv1.y;
    float pk2_re = pkv2.x;
    float pk2_im = pkv2.y;
    float pk3_re = pkv3.x;
    float pk3_im = pkv3.y;

    // Load BF pixel coords (same for all pixels in this bf)
    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    // Load q-space row coord
    float qx = __ldg(&qx_1d[row]);

    // Pixel 0: compute geometry once, reuse for 4 candidates
    float qy0 = __ldg(&qy_1d[pos0]);
    float4 m0 = compute_geometry(qx - kx, qy0 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p0 = compute_geometry(qx + kx, qy0 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    size_t idx0 = base_cache + pos0;
	    float2 G0 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos0, 512u, (unsigned int)gqk_cols);
    float2 r00 = gamma_mul_pk_packed_vals(
        m0, p0, G0, C10v0, C12v0, cos2phi12v0, sin2phi12v0, factor, pk0_re, pk0_im);
    float2 r01 = has1 ? gamma_mul_pk_packed_vals(
        m0, p0, G0, C10v1, C12v1, cos2phi12v1, sin2phi12v1, factor, pk1_re, pk1_im)
        : make_float2(0.0f, 0.0f);
    float2 r02 = has2 ? gamma_mul_pk_packed_vals(
        m0, p0, G0, C10v2, C12v2, cos2phi12v2, sin2phi12v2, factor, pk2_re, pk2_im)
        : make_float2(0.0f, 0.0f);
    float2 r03 = has3 ? gamma_mul_pk_packed_vals(
        m0, p0, G0, C10v3, C12v3, cos2phi12v3, sin2phi12v3, factor, pk3_re, pk3_im)
        : make_float2(0.0f, 0.0f);

    // Pixel 1
    float qy1 = __ldg(&qy_1d[pos1]);
    float4 m1 = compute_geometry(qx - kx, qy1 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p1 = compute_geometry(qx + kx, qy1 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    size_t idx1 = base_cache + pos1;
	    float2 G1 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos1, 512u, (unsigned int)gqk_cols);
    float2 r10 = gamma_mul_pk_packed_vals(
        m1, p1, G1, C10v0, C12v0, cos2phi12v0, sin2phi12v0, factor, pk0_re, pk0_im);
    float2 r11 = has1 ? gamma_mul_pk_packed_vals(
        m1, p1, G1, C10v1, C12v1, cos2phi12v1, sin2phi12v1, factor, pk1_re, pk1_im)
        : make_float2(0.0f, 0.0f);
    float2 r12 = has2 ? gamma_mul_pk_packed_vals(
        m1, p1, G1, C10v2, C12v2, cos2phi12v2, sin2phi12v2, factor, pk2_re, pk2_im)
        : make_float2(0.0f, 0.0f);
    float2 r13 = has3 ? gamma_mul_pk_packed_vals(
        m1, p1, G1, C10v3, C12v3, cos2phi12v3, sin2phi12v3, factor, pk3_re, pk3_im)
        : make_float2(0.0f, 0.0f);

    // Pixel 2
    float qy2 = __ldg(&qy_1d[pos2]);
    float4 m2 = compute_geometry(qx - kx, qy2 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p2 = compute_geometry(qx + kx, qy2 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    size_t idx2 = base_cache + pos2;
	    float2 G2 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos2, 512u, (unsigned int)gqk_cols);
    float2 r20 = gamma_mul_pk_packed_vals(
        m2, p2, G2, C10v0, C12v0, cos2phi12v0, sin2phi12v0, factor, pk0_re, pk0_im);
    float2 r21 = has1 ? gamma_mul_pk_packed_vals(
        m2, p2, G2, C10v1, C12v1, cos2phi12v1, sin2phi12v1, factor, pk1_re, pk1_im)
        : make_float2(0.0f, 0.0f);
    float2 r22 = has2 ? gamma_mul_pk_packed_vals(
        m2, p2, G2, C10v2, C12v2, cos2phi12v2, sin2phi12v2, factor, pk2_re, pk2_im)
        : make_float2(0.0f, 0.0f);
    float2 r23 = has3 ? gamma_mul_pk_packed_vals(
        m2, p2, G2, C10v3, C12v3, cos2phi12v3, sin2phi12v3, factor, pk3_re, pk3_im)
        : make_float2(0.0f, 0.0f);

    // Pixel 3
    float qy3 = __ldg(&qy_1d[pos3]);
    float4 m3 = compute_geometry(qx - kx, qy3 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p3 = compute_geometry(qx + kx, qy3 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    size_t idx3 = base_cache + pos3;
	    float2 G3 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos3, 512u, (unsigned int)gqk_cols);
    float2 r30 = gamma_mul_pk_packed_vals(
        m3, p3, G3, C10v0, C12v0, cos2phi12v0, sin2phi12v0, factor, pk0_re, pk0_im);
    float2 r31 = has1 ? gamma_mul_pk_packed_vals(
        m3, p3, G3, C10v1, C12v1, cos2phi12v1, sin2phi12v1, factor, pk1_re, pk1_im)
        : make_float2(0.0f, 0.0f);
    float2 r32 = has2 ? gamma_mul_pk_packed_vals(
        m3, p3, G3, C10v2, C12v2, cos2phi12v2, sin2phi12v2, factor, pk2_re, pk2_im)
        : make_float2(0.0f, 0.0f);
    float2 r33 = has3 ? gamma_mul_pk_packed_vals(
        m3, p3, G3, C10v3, C12v3, cos2phi12v3, sin2phi12v3, factor, pk3_re, pk3_im)
        : make_float2(0.0f, 0.0f);

    float4 res0a = make_float4(r00.x, r00.y, r01.x, r01.y);
    float4 res1a = make_float4(r10.x, r10.y, r11.x, r11.y);
    float4 res2a = make_float4(r20.x, r20.y, r21.x, r21.y);
    float4 res3a = make_float4(r30.x, r30.y, r31.x, r31.y);
    float4 res0b = make_float4(r02.x, r02.y, r03.x, r03.y);
    float4 res1b = make_float4(r12.x, r12.y, r13.x, r13.y);
    float4 res2b = make_float4(r22.x, r22.y, r23.x, r23.y);
    float4 res3b = make_float4(r32.x, r32.y, r33.x, r33.y);

    if (row == 0 && tid == 0) {
        res0a.x = dc_real;
        res0a.y = dc_imag;
        if (has1) {
            res0a.z = dc_real;
            res0a.w = dc_imag;
        }
        if (has2) {
            res0b.x = dc_real;
            res0b.y = dc_imag;
        }
        if (has3) {
            res0b.z = dc_real;
            res0b.w = dc_imag;
        }
    }

    __shared__ float4 s0[2][512];
    __shared__ float4 s1[2][512];
    float4* srow0 = s0[threadIdx.y];
    float4* srow1 = s1[threadIdx.y];
    srow0[digit_reverse_512((unsigned int)pos0)] = res0a;
    srow0[digit_reverse_512((unsigned int)pos1)] = res1a;
    srow0[digit_reverse_512((unsigned int)pos2)] = res2a;
    srow0[digit_reverse_512((unsigned int)pos3)] = res3a;
    srow1[digit_reverse_512((unsigned int)pos0)] = res0b;
    srow1[digit_reverse_512((unsigned int)pos1)] = res1b;
    srow1[digit_reverse_512((unsigned int)pos2)] = res2b;
    srow1[digit_reverse_512((unsigned int)pos3)] = res3b;
    __syncthreads();

    for (int m = 4; m <= 256; m <<= 2) {
        int quarter = m >> 2;
        int butterfly = tid;
        int j = butterfly % quarter;
        int k = butterfly / quarter;
        int idx0s = k * m + j;
        int idx1s = idx0s + quarter;
        int idx2s = idx1s + quarter;
        int idx3s = idx2s + quarter;
        int tw = j * (512 / m);
        float2 w0 = TWIDDLE_512[tw];
        float2 w1 = TWIDDLE_512[tw * 2];
        float2 w2 = TWIDDLE_512[tw * 3];

        float4 x0 = srow0[idx0s];
        float4 x1 = cmul2(w0, srow0[idx1s]);
        float4 x2 = cmul2(w1, srow0[idx2s]);
        float4 x3 = cmul2(w2, srow0[idx3s]);

        float4 t0 = cadd2(x0, x2);
        float4 t1 = csub2(x0, x2);
        float4 t2 = cadd2(x1, x3);
        float4 t3 = csub2(x1, x3);
        float4 y0 = cadd2(t0, t2);
        float4 y2 = csub2(t0, t2);
        float4 it3 = cmul_i2(t3);
        float4 y1 = cadd2(t1, it3);
        float4 y3 = csub2(t1, it3);

        srow0[idx0s] = y0;
        srow0[idx1s] = y1;
        srow0[idx2s] = y2;
        srow0[idx3s] = y3;

        float4 xa0 = srow1[idx0s];
        float4 xa1 = cmul2(w0, srow1[idx1s]);
        float4 xa2 = cmul2(w1, srow1[idx2s]);
        float4 xa3 = cmul2(w2, srow1[idx3s]);

        float4 ta0 = cadd2(xa0, xa2);
        float4 ta1 = csub2(xa0, xa2);
        float4 ta2 = cadd2(xa1, xa3);
        float4 ta3 = csub2(xa1, xa3);
        float4 ya0 = cadd2(ta0, ta2);
        float4 ya2 = csub2(ta0, ta2);
        float4 ita3 = cmul_i2(ta3);
        float4 ya1 = cadd2(ta1, ita3);
        float4 ya3 = csub2(ta1, ita3);

        srow1[idx0s] = ya0;
        srow1[idx1s] = ya1;
        srow1[idx2s] = ya2;
        srow1[idx3s] = ya3;
        __syncthreads();
    }

    // Radix-2 final stage (512 = 4^4 x 2) - apply to both srow0 and srow1
    {
        int j0 = tid;
        int j1 = tid + 128;
        float2 w0 = TWIDDLE_512[j0];
        float2 w1 = TWIDDLE_512[j1];

        float4 a0 = srow0[j0], b0 = cmul2(w0, srow0[j0 + 256]);
        float4 a1 = srow0[j1], b1 = cmul2(w1, srow0[j1 + 256]);
        srow0[j0] = cadd2(a0, b0);
        srow0[j0 + 256] = csub2(a0, b0);
        srow0[j1] = cadd2(a1, b1);
        srow0[j1 + 256] = csub2(a1, b1);

        float4 c0 = srow1[j0], d0 = cmul2(w0, srow1[j0 + 256]);
        float4 c1 = srow1[j1], d1 = cmul2(w1, srow1[j1 + 256]);
        srow1[j0] = cadd2(c0, d0);
        srow1[j0 + 256] = csub2(c0, d0);
        srow1[j1] = cadd2(c1, d1);
        srow1[j1 + 256] = csub2(c1, d1);
        __syncthreads();
    }

    float4 out0a = srow0[pos0];
    float4 out1a = srow0[pos1];
    float4 out2a = srow0[pos2];
    float4 out3a = srow0[pos3];
    size_t out_idx00 = (((size_t)cand0 * (size_t)num_bf + bf) * 512 + pos0) * 512 + row;
    size_t out_idx01 = (((size_t)cand0 * (size_t)num_bf + bf) * 512 + pos1) * 512 + row;
    size_t out_idx02 = (((size_t)cand0 * (size_t)num_bf + bf) * 512 + pos2) * 512 + row;
    size_t out_idx03 = (((size_t)cand0 * (size_t)num_bf + bf) * 512 + pos3) * 512 + row;
    out[out_idx00] = make_float2(out0a.x, out0a.y);
    out[out_idx01] = make_float2(out1a.x, out1a.y);
    out[out_idx02] = make_float2(out2a.x, out2a.y);
    out[out_idx03] = make_float2(out3a.x, out3a.y);
    if (has1) {
        size_t out_idx10 = (((size_t)cand1 * (size_t)num_bf + bf) * 512 + pos0) * 512 + row;
        size_t out_idx11 = (((size_t)cand1 * (size_t)num_bf + bf) * 512 + pos1) * 512 + row;
        size_t out_idx12 = (((size_t)cand1 * (size_t)num_bf + bf) * 512 + pos2) * 512 + row;
        size_t out_idx13 = (((size_t)cand1 * (size_t)num_bf + bf) * 512 + pos3) * 512 + row;
        out[out_idx10] = make_float2(out0a.z, out0a.w);
        out[out_idx11] = make_float2(out1a.z, out1a.w);
        out[out_idx12] = make_float2(out2a.z, out2a.w);
        out[out_idx13] = make_float2(out3a.z, out3a.w);
    }
    if (has2 || has3) {
        float4 out0b = srow1[pos0];
        float4 out1b = srow1[pos1];
        float4 out2b = srow1[pos2];
        float4 out3b = srow1[pos3];
        if (has2) {
            size_t out_idx20 = (((size_t)cand2 * (size_t)num_bf + bf) * 512 + pos0) * 512 + row;
            size_t out_idx21 = (((size_t)cand2 * (size_t)num_bf + bf) * 512 + pos1) * 512 + row;
            size_t out_idx22 = (((size_t)cand2 * (size_t)num_bf + bf) * 512 + pos2) * 512 + row;
            size_t out_idx23 = (((size_t)cand2 * (size_t)num_bf + bf) * 512 + pos3) * 512 + row;
            out[out_idx20] = make_float2(out0b.x, out0b.y);
            out[out_idx21] = make_float2(out1b.x, out1b.y);
            out[out_idx22] = make_float2(out2b.x, out2b.y);
            out[out_idx23] = make_float2(out3b.x, out3b.y);
        }
        if (has3) {
            size_t out_idx30 = (((size_t)cand3 * (size_t)num_bf + bf) * 512 + pos0) * 512 + row;
            size_t out_idx31 = (((size_t)cand3 * (size_t)num_bf + bf) * 512 + pos1) * 512 + row;
            size_t out_idx32 = (((size_t)cand3 * (size_t)num_bf + bf) * 512 + pos2) * 512 + row;
            size_t out_idx33 = (((size_t)cand3 * (size_t)num_bf + bf) * 512 + pos3) * 512 + row;
            out[out_idx30] = make_float2(out0b.z, out0b.w);
            out[out_idx31] = make_float2(out1b.z, out1b.w);
            out[out_idx32] = make_float2(out2b.z, out2b.w);
            out[out_idx33] = make_float2(out3b.z, out3b.w);
        }
    }
}

__global__ void ifft512_cols_t128_mr4(float2* __restrict__ data,
                                      int num_bf,
                                      float scale) {
    int bf = blockIdx.z;
    int col = blockIdx.y * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || col >= 512 || tid >= 128) {
        return;
    }
    size_t base = (size_t)bf * 512 * 512 + col;
    __shared__ float2 s[4][512];
    float2* srow = s[threadIdx.y];
    int pos0 = tid;
    int pos1 = tid + 128;
    int pos2 = tid + 256;
    int pos3 = tid + 384;
    srow[digit_reverse_512((unsigned int)pos0)] = data[base + pos0 * 512];
    srow[digit_reverse_512((unsigned int)pos1)] = data[base + pos1 * 512];
    srow[digit_reverse_512((unsigned int)pos2)] = data[base + pos2 * 512];
    srow[digit_reverse_512((unsigned int)pos3)] = data[base + pos3 * 512];
    __syncthreads();

    for (int m = 4; m <= 256; m <<= 2) {
        int quarter = m >> 2;
        int butterfly = tid;
        int j = butterfly % quarter;
        int k = butterfly / quarter;
        int idx0 = k * m + j;
        int idx1 = idx0 + quarter;
        int idx2 = idx1 + quarter;
        int idx3 = idx2 + quarter;
        int tw = j * (512 / m);
        float2 x0 = srow[idx0];
        float2 x1 = cmul(TWIDDLE_512[tw], srow[idx1]);
        float2 x2 = cmul(TWIDDLE_512[tw * 2], srow[idx2]);
        float2 x3 = cmul(TWIDDLE_512[tw * 3], srow[idx3]);

        float2 t0 = cadd(x0, x2);
        float2 t1 = csub(x0, x2);
        float2 t2 = cadd(x1, x3);
        float2 t3 = csub(x1, x3);
        float2 y0 = cadd(t0, t2);
        float2 y2 = csub(t0, t2);
        float2 it3 = cmul_i(t3);
        float2 y1 = cadd(t1, it3);
        float2 y3 = csub(t1, it3);

        srow[idx0] = y0;
        srow[idx1] = y1;
        srow[idx2] = y2;
        srow[idx3] = y3;
        __syncthreads();
    }

    // Radix-2 final stage (512 = 4^4 x 2)
    {
        int j0 = tid;
        int j1 = tid + 128;
        float2 w0 = TWIDDLE_512[j0];
        float2 w1 = TWIDDLE_512[j1];
        float2 a0 = srow[j0], b0 = cmul(w0, srow[j0 + 256]);
        float2 a1 = srow[j1], b1 = cmul(w1, srow[j1 + 256]);
        srow[j0] = cadd(a0, b0);
        srow[j0 + 256] = csub(a0, b0);
        srow[j1] = cadd(a1, b1);
        srow[j1 + 256] = csub(a1, b1);
        __syncthreads();
    }

    float2 out0 = srow[pos0];
    float2 out1 = srow[pos1];
    float2 out2 = srow[pos2];
    float2 out3 = srow[pos3];
    out0.x *= scale;
    out0.y *= scale;
    out1.x *= scale;
    out1.y *= scale;
    out2.x *= scale;
    out2.y *= scale;
    out3.x *= scale;
    out3.y *= scale;
    data[base + pos0 * 512] = out0;
    data[base + pos1 * 512] = out1;
    data[base + pos2 * 512] = out2;
    data[base + pos3 * 512] = out3;
}

// Fused col-FFT + phase accumulate.  Each block processes k_bf BF pixels
// for 4 columns, accumulating atan2(phase) in registers.  Eliminates the
// col write-back and separate reduction kernel.
__global__ void ifft512_cols_accumulate_t128_mr4(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sumsq,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    int col = blockIdx.y * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (col >= 512 || tid >= 128) return;

    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    __shared__ float2 s[4][512];
    float2* srow = s[threadIdx.y];

    int pos0 = tid;
    int pos1 = tid + 128;
    int pos2 = tid + 256;
    int pos3 = tid + 384;
    int rev0 = digit_reverse_512((unsigned int)pos0);
    int rev1 = digit_reverse_512((unsigned int)pos1);
    int rev2 = digit_reverse_512((unsigned int)pos2);
    int rev3 = digit_reverse_512((unsigned int)pos3);

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    float q0 = 0, q1 = 0, q2 = 0, q3 = 0;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * 512 * 512 + col;
        srow[rev0] = data[base + pos0 * 512];
        srow[rev1] = data[base + pos1 * 512];
        srow[rev2] = data[base + pos2 * 512];
        srow[rev3] = data[base + pos3 * 512];
        __syncthreads();

        for (int m = 4; m <= 256; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter;
            int k = tid / quarter;
            int idx0 = k * m + j;
            int idx1 = idx0 + quarter;
            int idx2 = idx1 + quarter;
            int idx3 = idx2 + quarter;
            int tw = j * (512 / m);
            float2 x0 = srow[idx0];
            float2 x1 = cmul(TWIDDLE_512[tw], srow[idx1]);
            float2 x2 = cmul(TWIDDLE_512[tw * 2], srow[idx2]);
            float2 x3 = cmul(TWIDDLE_512[tw * 3], srow[idx3]);

            float2 t0 = cadd(x0, x2);
            float2 t1 = csub(x0, x2);
            float2 t2 = cadd(x1, x3);
            float2 t3 = csub(x1, x3);
            float2 it3 = cmul_i(t3);
            srow[idx0] = cadd(t0, t2);
            srow[idx1] = cadd(t1, it3);
            srow[idx2] = csub(t0, t2);
            srow[idx3] = csub(t1, it3);
            __syncthreads();
        }

        {
            int j0 = tid, j1 = tid + 128;
            float2 w0 = TWIDDLE_512[j0];
            float2 w1 = TWIDDLE_512[j1];
            float2 a0 = srow[j0], b0 = cmul(w0, srow[j0 + 256]);
            float2 a1 = srow[j1], b1 = cmul(w1, srow[j1 + 256]);
            srow[j0] = cadd(a0, b0);
            srow[j0 + 256] = csub(a0, b0);
            srow[j1] = cadd(a1, b1);
            srow[j1 + 256] = csub(a1, b1);
            __syncthreads();
        }

        float2 o0 = srow[pos0], o1 = srow[pos1];
        float2 o2 = srow[pos2], o3 = srow[pos3];
        float p0 = atan2f(o0.y, o0.x);
        float p1 = atan2f(o1.y, o1.x);
        float p2 = atan2f(o2.y, o2.x);
        float p3 = atan2f(o3.y, o3.x);
        s0 += p0; s1 += p1; s2 += p2; s3 += p3;
        q0 += p0*p0; q1 += p1*p1; q2 += p2*p2; q3 += p3*p3;

        __syncthreads();
    }

    size_t plane = 512u * 512u;
    size_t out_base = (size_t)group * plane;
    size_t o0 = out_base + (size_t)pos0 * 512 + col;
    size_t o1 = out_base + (size_t)pos1 * 512 + col;
    size_t o2 = out_base + (size_t)pos2 * 512 + col;
    size_t o3 = out_base + (size_t)pos3 * 512 + col;
    partial_sum[o0] = s0; partial_sumsq[o0] = q0;
    partial_sum[o1] = s1; partial_sumsq[o1] = q1;
    partial_sum[o2] = s2; partial_sumsq[o2] = q2;
    partial_sum[o3] = s3; partial_sumsq[o3] = q3;
}

__global__ void ifft512_cols_accumulate_sum_t128_mr4(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    int col = blockIdx.y * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (col >= 512 || tid >= 128) return;

    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    __shared__ float2 s[4][512];
    float2* srow = s[threadIdx.y];

    int pos0 = tid;
    int pos1 = tid + 128;
    int pos2 = tid + 256;
    int pos3 = tid + 384;
    int rev0 = digit_reverse_512((unsigned int)pos0);
    int rev1 = digit_reverse_512((unsigned int)pos1);
    int rev2 = digit_reverse_512((unsigned int)pos2);
    int rev3 = digit_reverse_512((unsigned int)pos3);

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * 512u * 512u + (size_t)col;
        srow[rev0] = data[base + (size_t)pos0 * 512u];
        srow[rev1] = data[base + (size_t)pos1 * 512u];
        srow[rev2] = data[base + (size_t)pos2 * 512u];
        srow[rev3] = data[base + (size_t)pos3 * 512u];
        __syncthreads();

        for (int m = 4; m <= 256; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter;
            int k = tid / quarter;
            int idx0 = k * m + j;
            int idx1 = idx0 + quarter;
            int idx2 = idx1 + quarter;
            int idx3 = idx2 + quarter;
            int tw = j * (512 / m);
            float2 x0 = srow[idx0];
            float2 x1 = cmul(TWIDDLE_512[tw], srow[idx1]);
            float2 x2 = cmul(TWIDDLE_512[tw * 2], srow[idx2]);
            float2 x3 = cmul(TWIDDLE_512[tw * 3], srow[idx3]);

            float2 t0 = cadd(x0, x2);
            float2 t1 = csub(x0, x2);
            float2 t2 = cadd(x1, x3);
            float2 t3 = csub(x1, x3);
            float2 it3 = cmul_i(t3);
            srow[idx0] = cadd(t0, t2);
            srow[idx1] = cadd(t1, it3);
            srow[idx2] = csub(t0, t2);
            srow[idx3] = csub(t1, it3);
            __syncthreads();
        }

        {
            int j0 = tid, j1 = tid + 128;
            float2 w0 = TWIDDLE_512[j0];
            float2 w1 = TWIDDLE_512[j1];
            float2 a0 = srow[j0], b0 = cmul(w0, srow[j0 + 256]);
            float2 a1 = srow[j1], b1 = cmul(w1, srow[j1 + 256]);
            srow[j0] = cadd(a0, b0);
            srow[j0 + 256] = csub(a0, b0);
            srow[j1] = cadd(a1, b1);
            srow[j1 + 256] = csub(a1, b1);
            __syncthreads();
        }

        float2 o0 = srow[pos0], o1 = srow[pos1];
        float2 o2 = srow[pos2], o3 = srow[pos3];
        s0 += atan2f(o0.y, o0.x);
        s1 += atan2f(o1.y, o1.x);
        s2 += atan2f(o2.y, o2.x);
        s3 += atan2f(o3.y, o3.x);

        __syncthreads();
    }

    size_t plane = 512u * 512u;
    size_t out_base = (size_t)group * plane;
    partial_sum[out_base + (size_t)pos0 * 512u + (size_t)col] = s0;
    partial_sum[out_base + (size_t)pos1 * 512u + (size_t)col] = s1;
    partial_sum[out_base + (size_t)pos2 * 512u + (size_t)col] = s2;
    partial_sum[out_base + (size_t)pos3 * 512u + (size_t)col] = s3;
}

__global__ void ssb512_corrected_fourier_sum_t256(
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
    const unsigned long long plane = 512ull * 512ull;
    int groups = (num_bf + k_bf - 1) / k_bf;
    unsigned long long total = (unsigned long long)groups * plane;
    if (linear >= total) return;

    int group = (int)(linear / plane);
    unsigned int idx = (unsigned int)(linear - (unsigned long long)group * plane);
    int row = idx / 512u;
    int col = idx - (unsigned int)row * 512u;
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
                (unsigned int)col, 512u, (unsigned int)gqk_cols
            ));
        sum_re += v.x;
        sum_im += v.y;
    }
    partial_sum[(unsigned long long)group * plane + idx] = make_float2(sum_re, sum_im);
}

// Radix-8 variance kernel: 64 threads × 8 elements/thread. 3 FFT stages
// (m=8 in-register, m=64 smem, m=512 smem) instead of 5 radix-4 stages.
// Each block processes one column × 32 BFs sequentially. Sum/sumsq are
// accumulated across BFs per output position.
__global__ __launch_bounds__(64, 8)
void ifft512_rows_var_radix8_t64(const float2* __restrict__ data,
                                 float* __restrict__ sum,
                                 float* __restrict__ sumsq,
                                 int num_bf,
                                 int batch,
                                 float scale,
                                 int use_partial) {
    int col = blockIdx.y;
    int tid = threadIdx.x;
    int groups = (num_bf + 31) >> 5;
    int group = blockIdx.z % groups;
    int cand = blockIdx.z / groups;
    if (cand >= batch) return;
    int bf0 = group * 32;

    // Direct-load positions: compute the octal-reversed row for each of
    // the thread's 8 elements. Loading gmem in this order gives us stage 1
    // inputs directly, eliminating the smem scatter+sync+read round-trip.
    int src0 = (int)octal_reverse_512((unsigned int)(tid*8 + 0));
    int src1 = (int)octal_reverse_512((unsigned int)(tid*8 + 1));
    int src2 = (int)octal_reverse_512((unsigned int)(tid*8 + 2));
    int src3 = (int)octal_reverse_512((unsigned int)(tid*8 + 3));
    int src4 = (int)octal_reverse_512((unsigned int)(tid*8 + 4));
    int src5 = (int)octal_reverse_512((unsigned int)(tid*8 + 5));
    int src6 = (int)octal_reverse_512((unsigned int)(tid*8 + 6));
    int src7 = (int)octal_reverse_512((unsigned int)(tid*8 + 7));

    float sum0=0, sum1=0, sum2=0, sum3=0;
    float sum4=0, sum5=0, sum6=0, sum7=0;
    float sumsq0=0, sumsq1=0, sumsq2=0, sumsq3=0;
    float sumsq4=0, sumsq5=0, sumsq6=0, sumsq7=0;

    // Precompute twiddles outside the BF loop. Stage 2 uses W_64^(s2*k) and
    // stage 3 uses W_512^(tid*k). Both only depend on thread index, not BF.
    int s2_pre = tid & 7;
    float2 tw2_1 = TWIDDLE_512[(s2_pre * 1 * 8) & 511];
    float2 tw2_2 = TWIDDLE_512[(s2_pre * 2 * 8) & 511];
    float2 tw2_3 = TWIDDLE_512[(s2_pre * 3 * 8) & 511];
    float2 tw2_4 = TWIDDLE_512[(s2_pre * 4 * 8) & 511];
    float2 tw2_5 = TWIDDLE_512[(s2_pre * 5 * 8) & 511];
    float2 tw2_6 = TWIDDLE_512[(s2_pre * 6 * 8) & 511];
    float2 tw2_7 = TWIDDLE_512[(s2_pre * 7 * 8) & 511];
    float2 tw3_1 = TWIDDLE_512[(tid * 1) & 511];
    float2 tw3_2 = TWIDDLE_512[(tid * 2) & 511];
    float2 tw3_3 = TWIDDLE_512[(tid * 3) & 511];
    float2 tw3_4 = TWIDDLE_512[(tid * 4) & 511];
    float2 tw3_5 = TWIDDLE_512[(tid * 5) & 511];
    float2 tw3_6 = TWIDDLE_512[(tid * 6) & 511];
    float2 tw3_7 = TWIDDLE_512[(tid * 7) & 511];

    // Double buffered: buf 0 = sbuf_all[0..511], buf 1 = sbuf_all[512..1023].
    // Even iters use buf 0, odd iters use buf 1. Writing to one buf while
    // another warp still reads the other buf avoids the RAW hazard that
    // forced an end-of-iter __syncthreads in the single-buffer layout.
    __shared__ float2 sbuf_all[1024];

    size_t cand_base = ((size_t)cand * (size_t)num_bf) * (512u * 512u) + (size_t)col * 512u;
    size_t sum_base = (use_partial == 1)
        ? ((size_t)cand * (size_t)groups + (size_t)group) * (512u * 512u)
        : (size_t)cand * (512u * 512u);

    // Preload first BF before the loop so the inner loop can pipeline
    // gmem loads with smem compute.
    float2 r0, r1, r2, r3, r4, r5, r6, r7;
    if (bf0 < num_bf) {
        int delta = bf0 * (512 * 512);
        r0 = data[cand_base + delta + src0];
        r1 = data[cand_base + delta + src1];
        r2 = data[cand_base + delta + src2];
        r3 = data[cand_base + delta + src3];
        r4 = data[cand_base + delta + src4];
        r5 = data[cand_base + delta + src5];
        r6 = data[cand_base + delta + src6];
        r7 = data[cand_base + delta + src7];
    } else {
        r0 = make_float2(0,0); r1 = make_float2(0,0);
        r2 = make_float2(0,0); r3 = make_float2(0,0);
        r4 = make_float2(0,0); r5 = make_float2(0,0);
        r6 = make_float2(0,0); r7 = make_float2(0,0);
    }

    // Unroll enough to expose memory/SMEM latency without forcing excessive
    // register pressure in the column phase/loss accumulator.
    #pragma unroll 2
    for (int i = 0; i < 32; ++i) {
        int bf = bf0 + i;
        int valid = bf < num_bf;
        int bf_next = bf + 1;
        int has_next = (i < 31) && (bf_next < num_bf);

        // Pick the active buffer for this iter. Toggling buffers lets the
        // next iter's stage 1 writes run in parallel with this iter's
        // stage 3 reads (they hit disjoint smem regions). The
        // end-of-iter __syncthreads is only needed every 2 iters to
        // let buf 0 become safe to overwrite again.
        float2 *sbuf = sbuf_all + ((i & 1) ? 512 : 0);

        // Issue prefetch loads for iter i+1 into separate registers. The
        // compiler can schedule these early to overlap gmem latency with
        // the subsequent smem butterflies.
        float2 n0 = make_float2(0,0), n1 = make_float2(0,0);
        float2 n2 = make_float2(0,0), n3 = make_float2(0,0);
        float2 n4 = make_float2(0,0), n5 = make_float2(0,0);
        float2 n6 = make_float2(0,0), n7 = make_float2(0,0);
        if (has_next) {
            int delta_next = bf_next * (512 * 512);
            n0 = data[cand_base + delta_next + src0];
            n1 = data[cand_base + delta_next + src1];
            n2 = data[cand_base + delta_next + src2];
            n3 = data[cand_base + delta_next + src3];
            n4 = data[cand_base + delta_next + src4];
            n5 = data[cand_base + delta_next + src5];
            n6 = data[cand_base + delta_next + src6];
            n7 = __ldg(&data[cand_base + delta_next + src7]);
        }

        // Stage 1 (m=8): in-register radix-8, no external twiddles.
        radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

        // 8x8 register transpose within each 8-lane group via XOR-butterfly
        // warp shuffles. After this, lane L = (g*8 + s2) holds the r[s2] of
        // each of the 8 source lanes (g*8 + 0 ... g*8 + 7) in the same
        // g_outer group. That's exactly what stage 2 needs to consume - so
        // we skip the stage 1 smem round-trip entirely and feed stage 2
        // directly from registers.
        //
        // The transpose works in 3 stages, each XORing one bit of the
        // lane/elem index. At stage bit b, pairs (L, L^(1<<b)) exchange
        // their "off-diagonal" registers (those whose bit b of the reg
        // index differs from bit b of the lane index).
        // Helper: shuffle a float2 via two separate float shuffles.
        #define SHFL_XOR_F2(val, mask) make_float2( \
            __shfl_xor_sync(0xffffffff, (val).x, (mask)), \
            __shfl_xor_sync(0xffffffff, (val).y, (mask)))
        {
            float2 sent, got;
            // Stage b=0: swap r1↔r0, r3↔r2, r5↔r4, r7↔r6 across lane-XOR 1.
            sent = (tid & 1) ? r0 : r1;
            got = SHFL_XOR_F2(sent, 1);
            if (tid & 1) r0 = got; else r1 = got;
            sent = (tid & 1) ? r2 : r3;
            got = SHFL_XOR_F2(sent, 1);
            if (tid & 1) r2 = got; else r3 = got;
            sent = (tid & 1) ? r4 : r5;
            got = SHFL_XOR_F2(sent, 1);
            if (tid & 1) r4 = got; else r5 = got;
            sent = (tid & 1) ? r6 : r7;
            got = SHFL_XOR_F2(sent, 1);
            if (tid & 1) r6 = got; else r7 = got;
            // Stage b=1: swap r2↔r0, r3↔r1, r6↔r4, r7↔r5 across lane-XOR 2.
            sent = (tid & 2) ? r0 : r2;
            got = SHFL_XOR_F2(sent, 2);
            if (tid & 2) r0 = got; else r2 = got;
            sent = (tid & 2) ? r1 : r3;
            got = SHFL_XOR_F2(sent, 2);
            if (tid & 2) r1 = got; else r3 = got;
            sent = (tid & 2) ? r4 : r6;
            got = SHFL_XOR_F2(sent, 2);
            if (tid & 2) r4 = got; else r6 = got;
            sent = (tid & 2) ? r5 : r7;
            got = SHFL_XOR_F2(sent, 2);
            if (tid & 2) r5 = got; else r7 = got;
            // Stage b=2: swap r4↔r0, r5↔r1, r6↔r2, r7↔r3 across lane-XOR 4.
            sent = (tid & 4) ? r0 : r4;
            got = SHFL_XOR_F2(sent, 4);
            if (tid & 4) r0 = got; else r4 = got;
            sent = (tid & 4) ? r1 : r5;
            got = SHFL_XOR_F2(sent, 4);
            if (tid & 4) r1 = got; else r5 = got;
            sent = (tid & 4) ? r2 : r6;
            got = SHFL_XOR_F2(sent, 4);
            if (tid & 4) r2 = got; else r6 = got;
            sent = (tid & 4) ? r3 : r7;
            got = SHFL_XOR_F2(sent, 4);
            if (tid & 4) r3 = got; else r7 = got;
        }
        #undef SHFL_XOR_F2

        // Apply stage 2 twiddles (all precomputed outside the loop) - no
        // smem read needed, values came from the shuffle transpose above.
        r1 = cmul(tw2_1, r1);
        r2 = cmul(tw2_2, r2);
        r3 = cmul(tw2_3, r3);
        r4 = cmul(tw2_4, r4);
        r5 = cmul(tw2_5, r5);
        r6 = cmul(tw2_6, r6);
        r7 = cmul(tw2_7, r7);

        radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

        // Stage 2 output position for writing back - same stride-8 pattern
        // as the old stage 2 load, but now the load is replaced by shuffles.
        int g_outer = tid >> 3;
        int base2   = g_outer * 64 + s2_pre;

        sbuf[base2 +  0] = r0;
        sbuf[base2 +  8] = r1;
        sbuf[base2 + 16] = r2;
        sbuf[base2 + 24] = r3;
        sbuf[base2 + 32] = r4;
        sbuf[base2 + 40] = r5;
        sbuf[base2 + 48] = r6;
        sbuf[base2 + 56] = r7;
        __syncthreads();

        // Stage 3 (m=512): read stride-64 elements, apply precomputed twiddles.
        r0 = sbuf[tid +   0];
        r1 = cmul(tw3_1, sbuf[tid +  64]);
        r2 = cmul(tw3_2, sbuf[tid + 128]);
        r3 = cmul(tw3_3, sbuf[tid + 192]);
        r4 = cmul(tw3_4, sbuf[tid + 256]);
        r5 = cmul(tw3_5, sbuf[tid + 320]);
        r6 = cmul(tw3_6, sbuf[tid + 384]);
        r7 = cmul(tw3_7, sbuf[tid + 448]);

        radix8_butterfly(r0, r1, r2, r3, r4, r5, r6, r7);

        // Phase extract + accumulate per-position.
        if (valid) {
            float p0 = atan2f_ssb_poly(r0.y, r0.x);
            float p1 = atan2f_ssb_poly(r1.y, r1.x);
            float p2 = atan2f_ssb_poly(r2.y, r2.x);
            float p3 = atan2f_ssb_poly(r3.y, r3.x);
            float p4 = atan2f_ssb_poly(r4.y, r4.x);
            float p5 = atan2f_ssb_poly(r5.y, r5.x);
            float p6 = atan2f_ssb_poly(r6.y, r6.x);
            float p7 = atan2f_ssb_poly(r7.y, r7.x);
            sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
            sum4 += p4; sum5 += p5; sum6 += p6; sum7 += p7;
            sumsq0 += p0*p0; sumsq1 += p1*p1; sumsq2 += p2*p2; sumsq3 += p3*p3;
            sumsq4 += p4*p4; sumsq5 += p5*p5; sumsq6 += p6*p6; sumsq7 += p7*p7;
        }

        // No end-of-iter barrier: double-buffered smem means iter i+1's
        // stage-1 writes go to the OTHER buffer half from iter i's stage-3
        // reads, so there's no RAW hazard. The stage-2 → stage-3 barrier
        // inside each iter already aligns both warps, so by the time any
        // warp reaches iter i+2's stage-1 writes, its own iter i's stage-3
        // reads are done (warp-local linear execution) AND the other warp's
        // iter i stage-3 reads were done before the iter i+1 stage-2→3
        // barrier let it proceed.

        // Swap in the prefetched values for the next iteration.
        r0 = n0; r1 = n1; r2 = n2; r3 = n3;
        r4 = n4; r5 = n5; r6 = n6; r7 = n7;
    }

    // Write sum/sumsq to gmem. Thread tid owns positions
    // [tid + 0, tid + 64, tid + 128, ..., tid + 448] after stage 3.
    size_t base_out = sum_base + (size_t)col * 512;
    size_t o0 = base_out + (size_t)(tid +   0);
    size_t o1 = base_out + (size_t)(tid +  64);
    size_t o2 = base_out + (size_t)(tid + 128);
    size_t o3 = base_out + (size_t)(tid + 192);
    size_t o4 = base_out + (size_t)(tid + 256);
    size_t o5 = base_out + (size_t)(tid + 320);
    size_t o6 = base_out + (size_t)(tid + 384);
    size_t o7 = base_out + (size_t)(tid + 448);
    if (use_partial == 1) {
        sum[o0]=sum0; sumsq[o0]=sumsq0;
        sum[o1]=sum1; sumsq[o1]=sumsq1;
        sum[o2]=sum2; sumsq[o2]=sumsq2;
        sum[o3]=sum3; sumsq[o3]=sumsq3;
        sum[o4]=sum4; sumsq[o4]=sumsq4;
        sum[o5]=sum5; sumsq[o5]=sumsq5;
        sum[o6]=sum6; sumsq[o6]=sumsq6;
        sum[o7]=sum7; sumsq[o7]=sumsq7;
    } else if (use_partial == 2) {
        atomicAdd(&sum[o0], sum0);
        atomicAdd(&sum[o1], sum1);
        atomicAdd(&sum[o2], sum2);
        atomicAdd(&sum[o3], sum3);
        atomicAdd(&sum[o4], sum4);
        atomicAdd(&sum[o5], sum5);
        atomicAdd(&sum[o6], sum6);
        atomicAdd(&sum[o7], sum7);

        float local_sumsq = sumsq0 + sumsq1 + sumsq2 + sumsq3
                          + sumsq4 + sumsq5 + sumsq6 + sumsq7;
        __shared__ float block_sumsq[64];
        block_sumsq[tid] = local_sumsq;
        __syncthreads();
        for (int stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                block_sumsq[tid] += block_sumsq[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(&sumsq[cand], block_sumsq[0]);
        }
    } else if (use_partial == 4) {
        atomicAdd(&sum[o0], sum0);
        atomicAdd(&sum[o1], sum1);
        atomicAdd(&sum[o2], sum2);
        atomicAdd(&sum[o3], sum3);
        atomicAdd(&sum[o4], sum4);
        atomicAdd(&sum[o5], sum5);
        atomicAdd(&sum[o6], sum6);
        atomicAdd(&sum[o7], sum7);
    } else {
        atomicAdd(&sum[o0], sum0); atomicAdd(&sumsq[o0], sumsq0);
        atomicAdd(&sum[o1], sum1); atomicAdd(&sumsq[o1], sumsq1);
        atomicAdd(&sum[o2], sum2); atomicAdd(&sumsq[o2], sumsq2);
        atomicAdd(&sum[o3], sum3); atomicAdd(&sumsq[o3], sumsq3);
        atomicAdd(&sum[o4], sum4); atomicAdd(&sumsq[o4], sumsq4);
        atomicAdd(&sum[o5], sum5); atomicAdd(&sumsq[o5], sumsq5);
        atomicAdd(&sum[o6], sum6); atomicAdd(&sumsq[o6], sumsq6);
        atomicAdd(&sum[o7], sum7); atomicAdd(&sumsq[o7], sumsq7);
    }
}

'''


class CustomFFT512(CustomFFTBase):
    """Custom 512x512 IFFT kernels for SSB."""

    def __init__(self) -> None:
        super().__init__(
            size=512,
            cuda_code=build_cuda_code(512, _TWIDDLE_DECL, _FFT512_KERNELS),
            kernel_names=(
                "ifft512_rows_fused_pk_t128_mr4_packed",
                "ifft512_rows_fused_pk_batch_t128_mr2_transpose_packed_b4",
                "ifft512_cols_t128_mr4",
                "ifft512_rows_var_radix8_t64",
                "ifft512_cols_accumulate_t128_mr4",
                "ifft512_rows_fused_pk_full_t128_mr4_packed",
                "ifft512_cols_accumulate_sum_t128_mr4",
                "ssb512_corrected_fourier_sum_t256",
                "ifft512_rows_fused_pk_pair_radix8_t64_packed",
                "ifft512_rows_fused_pk_dual_radix8_t64_packed",
            ),
            twiddle_name="TWIDDLE_512",
            rows_block=(128, 4, 1),
            rows_grid_y=128,
            batch_block=(128, 2, 1),
            batch_grid_y=64,
            var_block=(64, 1, 1),
            var_grid_y=512,
            cols_block=(128, 4, 1),
            cols_grid_y=128,
        )
        self._cols_accumulate_sum = self._module.get_function(
            "ifft512_cols_accumulate_sum_t128_mr4"
        )
        self._rows_fused_pk_r8 = self._module.get_function(
            "ifft512_rows_fused_pk_radix8_t64_packed"
        )
        self._fourier_sum = self._module.get_function("ssb512_corrected_fourier_sum_t256")
        self._rows_fused_pk_pair_r8 = self._module.get_function(
            "ifft512_rows_fused_pk_pair_radix8_t64_packed"
        )
        self._rows_fused_pk_dual_r8 = self._module.get_function(
            "ifft512_rows_fused_pk_dual_radix8_t64_packed"
        )
        self._rows_pair_block = (64, 4, 1)
        self._rows_pair_grid_y = 128
        self._rows_dual_block = (64, 4, 1)
        self._rows_dual_grid_y = 128
        self._phase_sum_dummy_sumsq = None
        self._direct_dummy_sumsq = None
        self._colvar_group = 32

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
        """Row FFT + radix-8 column IFFT with phase sum/sumsq accumulation."""
        if k_bf != self._colvar_group:
            return super().ifft2_fused_pk_col_accumulate(
                data,
                G_qk,
                cache,
                pk,
                C10,
                C12,
                cos2phi12,
                sin2phi12,
                factor,
                dc_value,
                partial_sum,
                partial_sumsq,
                k_bf,
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
        n_groups = (num_bf + k_bf - 1) // k_bf
        if partial_sum.shape != (n_groups, N, N) or partial_sumsq.shape != (n_groups, N, N):
            raise ValueError(f"partial buffers must have shape ({n_groups}, {N}, {N})")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        phase_scale = np.float32(factor * wavelength * wavelength)
        max_ang = max(float(ang_y_rad), float(ang_x_rad))
        inner = (float(semiangle_rad) - 0.5 * max_ang) / float(wavelength)
        inner2 = np.float32(inner * inner if inner > 0.0 else -1.0)

        grid_rows = (1, N, num_bf)
        self._rows_fused_pk_r8(
            grid_rows,
            self._rows_var_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), phase_scale, inner2, pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(gqk_cols),
            ),
        )
        grid_cols = (1, self._rows_var_grid_y, n_groups)
        self._rows_var_batch(
            grid_cols,
            self._rows_var_block,
            (
                data,
                partial_sum,
                partial_sumsq,
                np.int32(num_bf),
                np.int32(1),
                np.float32(1.0 / (N * N)),
                np.int32(1),
            ),
        )

    def ifft2_fused_pk_col_accumulate_sum(
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
        k_bf: int = 64,
    ) -> None:
        """Row FFT + fused column IFFT with phase-sum accumulation only."""
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
        n_groups = (num_bf + k_bf - 1) // k_bf
        if partial_sum.shape != (n_groups, N, N):
            raise ValueError(f"partial_sum must have shape ({n_groups}, {N}, {N})")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        phase_scale = np.float32(factor * wavelength * wavelength)
        max_ang = max(float(ang_y_rad), float(ang_x_rad))
        inner = (float(semiangle_rad) - 0.5 * max_ang) / float(wavelength)
        inner2 = np.float32(inner * inner if inner > 0.0 else -1.0)

        grid_rows = (1, N, num_bf)
        self._rows_fused_pk_r8(
            grid_rows,
            self._rows_var_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), phase_scale, inner2, pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(gqk_cols),
            ),
        )
        if k_bf == self._colvar_group:
            if (
                self._phase_sum_dummy_sumsq is None
                or self._phase_sum_dummy_sumsq.shape != partial_sum.shape
            ):
                self._phase_sum_dummy_sumsq = cp.empty_like(partial_sum)
            grid_cols = (1, self._rows_var_grid_y, n_groups)
            self._rows_var_batch(
                grid_cols,
                self._rows_var_block,
                (
                    data,
                    partial_sum,
                    self._phase_sum_dummy_sumsq[:n_groups],
                    np.int32(num_bf),
                    np.int32(1),
                    np.float32(1.0 / (N * N)),
                    np.int32(1),
                ),
            )
            return
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate_sum(
            grid_cols,
            self._cols_block,
            (data, partial_sum, np.int32(num_bf), np.int32(k_bf)),
        )

    def ifft2_fused_pk_col_accumulate_direct(
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
        phase_sum: cp.ndarray,
        phase_sumsq: cp.ndarray | None,
        k_bf: int = 64,
    ) -> None:
        """Row FFT + column IFFT, atomically accumulating into final planes."""
        N = self._size
        if k_bf != self._colvar_group:
            raise ValueError(f"direct accumulate requires k_bf={self._colvar_group}")
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
        if phase_sum.shape != (N, N):
            raise ValueError(f"phase_sum must have shape ({N}, {N})")
        sum_only = phase_sumsq is None
        scalar_sumsq = phase_sumsq is not None and phase_sumsq.shape == (1,)
        if phase_sumsq is None:
            if self._direct_dummy_sumsq is None or self._direct_dummy_sumsq.shape != (N, N):
                self._direct_dummy_sumsq = cp.empty_like(phase_sum)
            phase_sumsq = self._direct_dummy_sumsq
        elif not scalar_sumsq and phase_sumsq.shape != (N, N):
            raise ValueError(f"phase_sumsq must have shape ({N}, {N}) or (1,)")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        phase_scale = np.float32(factor * wavelength * wavelength)
        max_ang = max(float(ang_y_rad), float(ang_x_rad))
        inner = (float(semiangle_rad) - 0.5 * max_ang) / float(wavelength)
        inner2 = np.float32(inner * inner if inner > 0.0 else -1.0)

        self._rows_fused_pk_r8(
            (1, N, num_bf),
            self._rows_var_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), phase_scale, inner2, pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(gqk_cols),
            ),
        )
        n_groups = (num_bf + k_bf - 1) // k_bf
        self._rows_var_batch(
            (1, self._rows_var_grid_y, n_groups),
            self._rows_var_block,
            (
                data,
                phase_sum,
                phase_sumsq,
                np.int32(num_bf),
                np.int32(1),
                np.float32(1.0 / (N * N)),
                np.int32(4 if sum_only else 0),
            ),
        )

    def ifft2_fused_pk_pair_col_accumulate_direct(
        self,
        data: cp.ndarray,
        G_qk: cp.ndarray,
        cache: dict,
        pk: cp.ndarray,
        pair_a: cp.ndarray,
        pair_b: cp.ndarray,
        C10: float,
        C12: float,
        cos2phi12: float,
        sin2phi12: float,
        factor: float,
        dc_value: complex,
        phase_sum: cp.ndarray,
        phase_sumsq: cp.ndarray | None,
        k_bf: int = 64,
    ) -> int:
        """Pair +/- BF rows, then run the existing column phase accumulator."""
        N = self._size
        if k_bf != self._colvar_group:
            raise ValueError(f"paired direct accumulate requires k_bf={self._colvar_group}")
        num_pairs = int(pair_a.shape[0])
        out_bf = num_pairs * 2
        if out_bf == 0:
            return 0
        if data.dtype != cp.complex64 or G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 input")
        if data.ndim != 3 or data.shape[0] < out_bf or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"data must have at least ({out_bf}, {N}, {N})")
        if G_qk.ndim != 3 or G_qk.shape[1] != N or G_qk.shape[2] not in (N, N // 2 + 1):
            raise ValueError(
                f"G_qk must have shape (num_bf, {N}, {N}) or Hermitian"
            )
        gqk_cols = int(G_qk.shape[2])
        if pk.shape != (G_qk.shape[0],):
            raise ValueError("pk must have shape (num_bf,)")
        if phase_sum.shape != (N, N):
            raise ValueError(f"phase_sum must have shape ({N}, {N})")
        sum_only = phase_sumsq is None
        scalar_sumsq = phase_sumsq is not None and phase_sumsq.shape == (1,)
        if phase_sumsq is None:
            if self._direct_dummy_sumsq is None or self._direct_dummy_sumsq.shape != (N, N):
                self._direct_dummy_sumsq = cp.empty_like(phase_sum)
            phase_sumsq = self._direct_dummy_sumsq
        elif not scalar_sumsq and phase_sumsq.shape != (N, N):
            raise ValueError(f"phase_sumsq must have shape ({N}, {N}) or (1,)")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        phase_scale = np.float32(factor * wavelength * wavelength)
        max_ang = max(float(ang_y_rad), float(ang_x_rad))
        inner = (float(semiangle_rad) - 0.5 * max_ang) / float(wavelength)
        inner2 = np.float32(inner * inner if inner > 0.0 else -1.0)

        out = data[:out_bf]
        self._rows_fused_pk_pair_r8(
            (1, self._rows_pair_grid_y, num_pairs),
            self._rows_pair_block,
            (
                pair_a, pair_b,
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), phase_scale, inner2, pk, G_qk, out,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_pairs), np.int32(gqk_cols),
            ),
        )
        n_groups = (out_bf + k_bf - 1) // k_bf
        self._rows_var_batch(
            (1, self._rows_var_grid_y, n_groups),
            self._rows_var_block,
            (
                out,
                phase_sum,
                phase_sumsq,
                np.int32(out_bf),
                np.int32(1),
                np.float32(1.0 / (N * N)),
                np.int32(4 if sum_only else (2 if scalar_sumsq else 0)),
            ),
        )
        return out_bf

    def ifft2_fused_pk_dual_col_accumulate_direct(
        self,
        data: cp.ndarray,
        G_qk: cp.ndarray,
        cache: dict,
        pk: cp.ndarray,
        pair_a: cp.ndarray,
        pair_b: cp.ndarray,
        C10: float,
        C12: float,
        cos2phi12: float,
        sin2phi12: float,
        factor: float,
        dc_value: complex,
        phase_sum: cp.ndarray,
        phase_sumsq: cp.ndarray | None,
        k_bf: int = 64,
    ) -> int:
        """Process two arbitrary BF pixels per row block, preserving exact gamma."""
        N = self._size
        if k_bf != self._colvar_group:
            raise ValueError(f"dual direct accumulate requires k_bf={self._colvar_group}")
        num_pairs = int(pair_a.shape[0])
        out_bf = num_pairs * 2
        if out_bf == 0:
            return 0
        if data.dtype != cp.complex64 or G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 input")
        if data.ndim != 3 or data.shape[0] < out_bf or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"data must have at least ({out_bf}, {N}, {N})")
        if G_qk.ndim != 3 or G_qk.shape[1] != N or G_qk.shape[2] not in (N, N // 2 + 1):
            raise ValueError(
                f"G_qk must have shape (num_bf, {N}, {N}) or Hermitian"
            )
        gqk_cols = int(G_qk.shape[2])
        if pk.shape != (G_qk.shape[0],):
            raise ValueError("pk must have shape (num_bf,)")
        if phase_sum.shape != (N, N):
            raise ValueError(f"phase_sum must have shape ({N}, {N})")
        sum_only = phase_sumsq is None
        scalar_sumsq = phase_sumsq is not None and phase_sumsq.shape == (1,)
        if phase_sumsq is None:
            if self._direct_dummy_sumsq is None or self._direct_dummy_sumsq.shape != (N, N):
                self._direct_dummy_sumsq = cp.empty_like(phase_sum)
            phase_sumsq = self._direct_dummy_sumsq
        elif not scalar_sumsq and phase_sumsq.shape != (N, N):
            raise ValueError(f"phase_sumsq must have shape ({N}, {N}) or (1,)")
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        phase_scale = np.float32(factor * wavelength * wavelength)
        max_ang = max(float(ang_y_rad), float(ang_x_rad))
        inner = (float(semiangle_rad) - 0.5 * max_ang) / float(wavelength)
        inner2 = np.float32(inner * inner if inner > 0.0 else -1.0)

        out = data[:out_bf]
        self._rows_fused_pk_dual_r8(
            (1, self._rows_dual_grid_y, num_pairs),
            self._rows_dual_block,
            (
                pair_a, pair_b,
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), phase_scale, inner2, pk, G_qk, out,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_pairs), np.int32(gqk_cols),
            ),
        )
        n_groups = (out_bf + k_bf - 1) // k_bf
        self._rows_var_batch(
            (1, self._rows_var_grid_y, n_groups),
            self._rows_var_block,
            (
                out,
                phase_sum,
                phase_sumsq,
                np.int32(out_bf),
                np.int32(1),
                np.float32(1.0 / (N * N)),
                np.int32(4 if sum_only else (2 if scalar_sumsq else 0)),
            ),
        )
        return out_bf


@lru_cache(maxsize=1)
def get_custom_fft_512() -> CustomFFT512:
    return CustomFFT512()
