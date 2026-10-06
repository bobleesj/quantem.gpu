"""Custom fixed-size CUDA FFT kernels for SSB (1024x1024).

1024 = 4^5.  This backend provides the full reconstruct path and the
row-subsampled batched optimizer objective used by the 256/512 CUDA paths:
fused gamma multiplication, row IFFT, column IFFT, and fused column phase
accumulation.
"""

from functools import lru_cache

import cupy as cp
import numpy as np

from quantem.gpu.ssb.cuda.kernels.common import (
    CustomFFTBase,
    build_cuda_code,
    corrected_fourier_sum_source,
)

_TWIDDLE_DECL = '__constant__ float2 TWIDDLE_1024[1024];'

_FFT1024_KERNELS = r'''
__device__ __forceinline__ unsigned int digit_reverse_1024(unsigned int x) {
    return ((x & 0x003u) << 8) | ((x & 0x00Cu) << 4) |
           (x & 0x030u) |
           ((x & 0x0C0u) >> 4) | ((x & 0x300u) >> 8);
}

__device__ __forceinline__ unsigned int octal_reverse_512_for_1024(unsigned int n) {
    unsigned int d0 = n & 7u;
    unsigned int d1 = (n >> 3) & 7u;
    unsigned int d2 = n >> 6;
    return (d0 << 6) | (d1 << 3) | d2;
}

#define SQ2_INV_1024_F 0.70710678118654752f

__device__ __forceinline__ float2 cmul_w8_1_1024(float2 a) {
    return make_float2(SQ2_INV_1024_F * (a.x - a.y),
                       SQ2_INV_1024_F * (a.x + a.y));
}

__device__ __forceinline__ float2 cmul_w8_3_1024(float2 a) {
    return make_float2(SQ2_INV_1024_F * (-a.x - a.y),
                       SQ2_INV_1024_F * ( a.x - a.y));
}

__device__ __forceinline__ void radix8_butterfly_1024(
    float2 &x0, float2 &x1, float2 &x2, float2 &x3,
    float2 &x4, float2 &x5, float2 &x6, float2 &x7)
{
    float2 a0 = x0, a1 = x4, a2 = x2, a3 = x6;
    float2 a4 = x1, a5 = x5, a6 = x3, a7 = x7;
    float2 t0 = cadd(a0, a1), t1 = csub(a0, a1);
    float2 t2 = cadd(a2, a3), t3 = csub(a2, a3);
    float2 t4 = cadd(a4, a5), t5 = csub(a4, a5);
    float2 t6 = cadd(a6, a7), t7 = csub(a6, a7);
    float2 u0 = cadd(t0, t2), u2 = csub(t0, t2);
    float2 it3 = cmul_i(t3);
    float2 u1 = cadd(t1, it3), u3 = csub(t1, it3);
    float2 u4 = cadd(t4, t6), u6 = csub(t4, t6);
    float2 it7 = cmul_i(t7);
    float2 u5 = cadd(t5, it7), u7 = csub(t5, it7);
    float2 w1u5 = cmul_w8_1_1024(u5);
    float2 w3u7 = cmul_w8_3_1024(u7);
    float2 iu6  = cmul_i(u6);
    x0 = cadd(u0, u4);    x4 = csub(u0, u4);
    x1 = cadd(u1, w1u5);  x5 = csub(u1, w1u5);
    x2 = cadd(u2, iu6);   x6 = csub(u2, iu6);
    x3 = cadd(u3, w3u7);  x7 = csub(u3, w3u7);
}

__device__ __forceinline__ void ifft512_radix8_apply_t64_1024tw(
    float2 &r0, float2 &r1, float2 &r2, float2 &r3,
    float2 &r4, float2 &r5, float2 &r6, float2 &r7,
    int tid,
    float2* __restrict__ sbuf
) {
    int s2_pre = tid & 7;
    float2 tw2_1 = TWIDDLE_1024[(s2_pre * 16) & 1023];
    float2 tw2_2 = TWIDDLE_1024[(s2_pre * 32) & 1023];
    float2 tw2_3 = TWIDDLE_1024[(s2_pre * 48) & 1023];
    float2 tw2_4 = TWIDDLE_1024[(s2_pre * 64) & 1023];
    float2 tw2_5 = TWIDDLE_1024[(s2_pre * 80) & 1023];
    float2 tw2_6 = TWIDDLE_1024[(s2_pre * 96) & 1023];
    float2 tw2_7 = TWIDDLE_1024[(s2_pre * 112) & 1023];
    float2 tw3_1 = TWIDDLE_1024[(tid * 2) & 1023];
    float2 tw3_2 = TWIDDLE_1024[(tid * 4) & 1023];
    float2 tw3_3 = TWIDDLE_1024[(tid * 6) & 1023];
    float2 tw3_4 = TWIDDLE_1024[(tid * 8) & 1023];
    float2 tw3_5 = TWIDDLE_1024[(tid * 10) & 1023];
    float2 tw3_6 = TWIDDLE_1024[(tid * 12) & 1023];
    float2 tw3_7 = TWIDDLE_1024[(tid * 14) & 1023];

    radix8_butterfly_1024(r0, r1, r2, r3, r4, r5, r6, r7);

    #define SHFL_XOR_F2_1024(val, mask) make_float2( \
        __shfl_xor_sync(0xffffffff, (val).x, (mask)), \
        __shfl_xor_sync(0xffffffff, (val).y, (mask)))
    {
        float2 sent, got;
        sent = (tid & 1) ? r0 : r1;
        got = SHFL_XOR_F2_1024(sent, 1);
        if (tid & 1) r0 = got; else r1 = got;
        sent = (tid & 1) ? r2 : r3;
        got = SHFL_XOR_F2_1024(sent, 1);
        if (tid & 1) r2 = got; else r3 = got;
        sent = (tid & 1) ? r4 : r5;
        got = SHFL_XOR_F2_1024(sent, 1);
        if (tid & 1) r4 = got; else r5 = got;
        sent = (tid & 1) ? r6 : r7;
        got = SHFL_XOR_F2_1024(sent, 1);
        if (tid & 1) r6 = got; else r7 = got;
        sent = (tid & 2) ? r0 : r2;
        got = SHFL_XOR_F2_1024(sent, 2);
        if (tid & 2) r0 = got; else r2 = got;
        sent = (tid & 2) ? r1 : r3;
        got = SHFL_XOR_F2_1024(sent, 2);
        if (tid & 2) r1 = got; else r3 = got;
        sent = (tid & 2) ? r4 : r6;
        got = SHFL_XOR_F2_1024(sent, 2);
        if (tid & 2) r4 = got; else r6 = got;
        sent = (tid & 2) ? r5 : r7;
        got = SHFL_XOR_F2_1024(sent, 2);
        if (tid & 2) r5 = got; else r7 = got;
        sent = (tid & 4) ? r0 : r4;
        got = SHFL_XOR_F2_1024(sent, 4);
        if (tid & 4) r0 = got; else r4 = got;
        sent = (tid & 4) ? r1 : r5;
        got = SHFL_XOR_F2_1024(sent, 4);
        if (tid & 4) r1 = got; else r5 = got;
        sent = (tid & 4) ? r2 : r6;
        got = SHFL_XOR_F2_1024(sent, 4);
        if (tid & 4) r2 = got; else r6 = got;
        sent = (tid & 4) ? r3 : r7;
        got = SHFL_XOR_F2_1024(sent, 4);
        if (tid & 4) r3 = got; else r7 = got;
    }
    #undef SHFL_XOR_F2_1024

    r1 = cmul(tw2_1, r1);
    r2 = cmul(tw2_2, r2);
    r3 = cmul(tw2_3, r3);
    r4 = cmul(tw2_4, r4);
    r5 = cmul(tw2_5, r5);
    r6 = cmul(tw2_6, r6);
    r7 = cmul(tw2_7, r7);

    radix8_butterfly_1024(r0, r1, r2, r3, r4, r5, r6, r7);

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

    radix8_butterfly_1024(r0, r1, r2, r3, r4, r5, r6, r7);
}

__global__ void ifft1024_rows_fused_pk_t256_mr2_packed(
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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 1024 || tid >= 256) return;

    bool transpose_out = gqk_cols < 0;
    unsigned int stored_cols = (unsigned int)(transpose_out ? -gqk_cols : gqk_cols);
    size_t base = ((size_t)bf * 1024 + row) * 1024;
    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;
    float2 pkv = pk[bf];
    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    float qx = __ldg(&qx_1d[row]);

	    float2 res0 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos0]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos0, 1024u, stored_cols));
	    float2 res1 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 1024u, stored_cols));
	    float2 res2 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 1024u, stored_cols));
	    float2 res3 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 1024u, stored_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[1024];
    float2* srow = s;
    srow[digit_reverse_1024((unsigned int)pos0)] = res0;
    srow[digit_reverse_1024((unsigned int)pos1)] = res1;
    srow[digit_reverse_1024((unsigned int)pos2)] = res2;
    srow[digit_reverse_1024((unsigned int)pos3)] = res3;
    __syncthreads();

    for (int m = 4; m <= 1024; m <<= 2) {
        int quarter = m >> 2;
        int j = tid % quarter;
        int k = tid / quarter;
        int idx0 = k * m + j;
        int idx1 = idx0 + quarter;
        int idx2 = idx1 + quarter;
        int idx3 = idx2 + quarter;
        int tw = j * (1024 / m);
        float2 x0 = srow[idx0];
        float2 x1 = cmul(TWIDDLE_1024[tw], srow[idx1]);
        float2 x2 = cmul(TWIDDLE_1024[tw * 2], srow[idx2]);
        float2 x3 = cmul(TWIDDLE_1024[tw * 3], srow[idx3]);

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

    if (transpose_out) {
        size_t tbase = (size_t)bf * 1024u * 1024u + (size_t)row;
        out[tbase + (size_t)pos0 * 1024u] = srow[pos0];
        out[tbase + (size_t)pos1 * 1024u] = srow[pos1];
        out[tbase + (size_t)pos2 * 1024u] = srow[pos2];
        out[tbase + (size_t)pos3 * 1024u] = srow[pos3];
    } else {
        out[base + pos0] = srow[pos0];
        out[base + pos1] = srow[pos1];
        out[base + pos2] = srow[pos2];
        out[base + pos3] = srow[pos3];
    }
}

__global__ __launch_bounds__(64, 8)
void ifft1024_rows_fused_pk_split512_t64_packed(
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
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 1024 || tid >= 64) return;

    unsigned int stored_cols = (unsigned int)(gqk_cols < 0 ? -gqk_cols : gqk_cols);
    int src0 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 0));
    int src1 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 1));
    int src2 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 2));
    int src3 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 3));
    int src4 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 4));
    int src5 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 5));
    int src6 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 6));
    int src7 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 7));
    int e0c = src0 * 2, e1c = src1 * 2, e2c = src2 * 2, e3c = src3 * 2;
    int e4c = src4 * 2, e5c = src5 * 2, e6c = src6 * 2, e7c = src7 * 2;
    int o0c = e0c + 1, o1c = e1c + 1, o2c = e2c + 1, o3c = e3c + 1;
    int o4c = e4c + 1, o5c = e5c + 1, o6c = e6c + 1, o7c = e7c + 1;

    float2 pkv = pk[bf];
    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    float qx = __ldg(&qx_1d[row]);

#define ROW_GAMMA1024(COL) \
    gamma_mul_pk_onthefly( \
        qx, __ldg(&qy_1d[(COL)]), kx, ky, \
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad, \
        C10, C12, cos2phi12, sin2phi12, factor, \
        pkv.x, pkv.y, ld_gqk_maybe_herm( \
            G_qk, (unsigned long long)bf, (unsigned int)row, \
            (unsigned int)(COL), 1024u, stored_cols))

    float2 e0 = ROW_GAMMA1024(e0c);
    float2 e1 = ROW_GAMMA1024(e1c);
    float2 e2 = ROW_GAMMA1024(e2c);
    float2 e3 = ROW_GAMMA1024(e3c);
    float2 e4 = ROW_GAMMA1024(e4c);
    float2 e5 = ROW_GAMMA1024(e5c);
    float2 e6 = ROW_GAMMA1024(e6c);
    float2 e7 = ROW_GAMMA1024(e7c);
    float2 o0 = ROW_GAMMA1024(o0c);
    float2 o1 = ROW_GAMMA1024(o1c);
    float2 o2 = ROW_GAMMA1024(o2c);
    float2 o3 = ROW_GAMMA1024(o3c);
    float2 o4 = ROW_GAMMA1024(o4c);
    float2 o5 = ROW_GAMMA1024(o5c);
    float2 o6 = ROW_GAMMA1024(o6c);
    float2 o7 = ROW_GAMMA1024(o7c);
#undef ROW_GAMMA1024

    if (row == 0) {
        if (e0c == 0) e0 = make_float2(dc_real, dc_imag);
        if (e1c == 0) e1 = make_float2(dc_real, dc_imag);
        if (e2c == 0) e2 = make_float2(dc_real, dc_imag);
        if (e3c == 0) e3 = make_float2(dc_real, dc_imag);
        if (e4c == 0) e4 = make_float2(dc_real, dc_imag);
        if (e5c == 0) e5 = make_float2(dc_real, dc_imag);
        if (e6c == 0) e6 = make_float2(dc_real, dc_imag);
        if (e7c == 0) e7 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 se[512];
    __shared__ float2 so[512];
    ifft512_radix8_apply_t64_1024tw(e0, e1, e2, e3, e4, e5, e6, e7, tid, se);
    ifft512_radix8_apply_t64_1024tw(o0, o1, o2, o3, o4, o5, o6, o7, tid, so);

    int j0 = tid;
    int j1 = tid + 64;
    int j2 = tid + 128;
    int j3 = tid + 192;
    int j4 = tid + 256;
    int j5 = tid + 320;
    int j6 = tid + 384;
    int j7 = tid + 448;
    size_t base = (size_t)bf * 1024u * 1024u + (size_t)row;

#define STORE_SPLIT512_ROW(J, E, O) \
    { \
        float2 tw = TWIDDLE_1024[(J) & 1023]; \
        float2 to = cmul(tw, (O)); \
        out[base + (size_t)(J) * 1024u] = cadd((E), to); \
        out[base + (size_t)((J) + 512) * 1024u] = csub((E), to); \
    }
    STORE_SPLIT512_ROW(j0, e0, o0);
    STORE_SPLIT512_ROW(j1, e1, o1);
    STORE_SPLIT512_ROW(j2, e2, o2);
    STORE_SPLIT512_ROW(j3, e3, o3);
    STORE_SPLIT512_ROW(j4, e4, o4);
    STORE_SPLIT512_ROW(j5, e5, o5);
    STORE_SPLIT512_ROW(j6, e6, o6);
    STORE_SPLIT512_ROW(j7, e7, o7);
#undef STORE_SPLIT512_ROW
}

__global__ void ifft1024_rows_fused_pk_full_t256_mr2_packed(
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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 1024 || tid >= 256) return;

    size_t base = ((size_t)bf * 1024 + row) * 1024;
    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;
    float2 pkv = pk[bf];
    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    float qx = __ldg(&qx_1d[row]);

	    float2 res0 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos0]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos0, 1024u, (unsigned int)gqk_cols));
	    float2 res1 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 1024u, (unsigned int)gqk_cols));
	    float2 res2 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 1024u, (unsigned int)gqk_cols));
	    float2 res3 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 1024u, (unsigned int)gqk_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[1024];
    float2* srow = s;
    srow[digit_reverse_1024((unsigned int)pos0)] = res0;
    srow[digit_reverse_1024((unsigned int)pos1)] = res1;
    srow[digit_reverse_1024((unsigned int)pos2)] = res2;
    srow[digit_reverse_1024((unsigned int)pos3)] = res3;
    __syncthreads();

    for (int m = 4; m <= 1024; m <<= 2) {
        int quarter = m >> 2;
        int j = tid % quarter;
        int k = tid / quarter;
        int idx0 = k * m + j;
        int idx1 = idx0 + quarter;
        int idx2 = idx1 + quarter;
        int idx3 = idx2 + quarter;
        int tw = j * (1024 / m);
        float2 x0 = srow[idx0];
        float2 x1 = cmul(TWIDDLE_1024[tw], srow[idx1]);
        float2 x2 = cmul(TWIDDLE_1024[tw * 2], srow[idx2]);
        float2 x3 = cmul(TWIDDLE_1024[tw * 3], srow[idx3]);

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

    out[base + pos0] = srow[pos0];
    out[base + pos1] = srow[pos1];
    out[base + pos2] = srow[pos2];
    out[base + pos3] = srow[pos3];
}

__global__ void ifft1024_cols_t256_mr2(
    float2* __restrict__ data,
    int num_bf,
    float scale
) {
    int bf = blockIdx.z;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || col >= 1024 || tid >= 256) return;

    size_t base = (size_t)bf * 1024u * 1024u + col;
    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;

    __shared__ float2 s[1024];
    float2* srow = s;
    srow[digit_reverse_1024((unsigned int)pos0)] = data[base + (size_t)pos0 * 1024u];
    srow[digit_reverse_1024((unsigned int)pos1)] = data[base + (size_t)pos1 * 1024u];
    srow[digit_reverse_1024((unsigned int)pos2)] = data[base + (size_t)pos2 * 1024u];
    srow[digit_reverse_1024((unsigned int)pos3)] = data[base + (size_t)pos3 * 1024u];
    __syncthreads();

    for (int m = 4; m <= 1024; m <<= 2) {
        int quarter = m >> 2;
        int j = tid % quarter;
        int k = tid / quarter;
        int idx0 = k * m + j;
        int idx1 = idx0 + quarter;
        int idx2 = idx1 + quarter;
        int idx3 = idx2 + quarter;
        int tw = j * (1024 / m);
        float2 x0 = srow[idx0];
        float2 x1 = cmul(TWIDDLE_1024[tw], srow[idx1]);
        float2 x2 = cmul(TWIDDLE_1024[tw * 2], srow[idx2]);
        float2 x3 = cmul(TWIDDLE_1024[tw * 3], srow[idx3]);

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

    float2 o0 = srow[pos0];
    float2 o1 = srow[pos1];
    float2 o2 = srow[pos2];
    float2 o3 = srow[pos3];
    o0.x *= scale; o0.y *= scale;
    o1.x *= scale; o1.y *= scale;
    o2.x *= scale; o2.y *= scale;
    o3.x *= scale; o3.y *= scale;
    data[base + (size_t)pos0 * 1024u] = o0;
    data[base + (size_t)pos1 * 1024u] = o1;
    data[base + (size_t)pos2 * 1024u] = o2;
    data[base + (size_t)pos3 * 1024u] = o3;
}

__global__ void ifft1024_cols_accumulate_t256_mr2(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sumsq,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    bool transposed_in = k_bf < 0;
    if (transposed_in) k_bf = -k_bf;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    if (col >= 1024 || tid >= 256) return;

    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;
    int rev0 = digit_reverse_1024((unsigned int)pos0);
    int rev1 = digit_reverse_1024((unsigned int)pos1);
    int rev2 = digit_reverse_1024((unsigned int)pos2);
    int rev3 = digit_reverse_1024((unsigned int)pos3);

    __shared__ float2 s[1024];
    float2* srow = s;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * 1024u * 1024u
                    + (transposed_in ? (size_t)col * 1024u : (size_t)col);
        size_t stride = transposed_in ? 1u : 1024u;
        srow[rev0] = data[base + (size_t)pos0 * stride];
        srow[rev1] = data[base + (size_t)pos1 * stride];
        srow[rev2] = data[base + (size_t)pos2 * stride];
        srow[rev3] = data[base + (size_t)pos3 * stride];
        __syncthreads();

        for (int m = 4; m <= 1024; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter;
            int k = tid / quarter;
            int idx0 = k * m + j;
            int idx1 = idx0 + quarter;
            int idx2 = idx1 + quarter;
            int idx3 = idx2 + quarter;
            int tw = j * (1024 / m);
            float2 x0 = srow[idx0];
            float2 x1 = cmul(TWIDDLE_1024[tw], srow[idx1]);
            float2 x2 = cmul(TWIDDLE_1024[tw * 2], srow[idx2]);
            float2 x3 = cmul(TWIDDLE_1024[tw * 3], srow[idx3]);

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

        float2 o0 = srow[pos0];
        float2 o1 = srow[pos1];
        float2 o2 = srow[pos2];
        float2 o3 = srow[pos3];
        float p0 = atan2f(o0.y, o0.x);
        float p1 = atan2f(o1.y, o1.x);
        float p2 = atan2f(o2.y, o2.x);
        float p3 = atan2f(o3.y, o3.x);
        sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
        sq0 += p0 * p0; sq1 += p1 * p1; sq2 += p2 * p2; sq3 += p3 * p3;
        __syncthreads();
    }

    size_t plane = 1024u * 1024u;
    size_t out_base = (size_t)group * plane;
    size_t o0 = out_base + (size_t)pos0 * 1024u + col;
    size_t o1 = out_base + (size_t)pos1 * 1024u + col;
    size_t o2 = out_base + (size_t)pos2 * 1024u + col;
    size_t o3 = out_base + (size_t)pos3 * 1024u + col;
    partial_sum[o0] = sum0; partial_sumsq[o0] = sq0;
    partial_sum[o1] = sum1; partial_sumsq[o1] = sq1;
    partial_sum[o2] = sum2; partial_sumsq[o2] = sq2;
    partial_sum[o3] = sum3; partial_sumsq[o3] = sq3;
}

__global__ void ifft1024_cols_accumulate_sum_t256_mr2(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    bool transposed_in = k_bf < 0;
    if (transposed_in) k_bf = -k_bf;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    if (col >= 1024 || tid >= 256) return;

    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;
    int rev0 = digit_reverse_1024((unsigned int)pos0);
    int rev1 = digit_reverse_1024((unsigned int)pos1);
    int rev2 = digit_reverse_1024((unsigned int)pos2);
    int rev3 = digit_reverse_1024((unsigned int)pos3);

    __shared__ float2 s[1024];
    float2* srow = s;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * 1024u * 1024u
                    + (transposed_in ? (size_t)col * 1024u : (size_t)col);
        size_t stride = transposed_in ? 1u : 1024u;
        srow[rev0] = data[base + (size_t)pos0 * stride];
        srow[rev1] = data[base + (size_t)pos1 * stride];
        srow[rev2] = data[base + (size_t)pos2 * stride];
        srow[rev3] = data[base + (size_t)pos3 * stride];
        __syncthreads();

        for (int m = 4; m <= 1024; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter;
            int k = tid / quarter;
            int idx0 = k * m + j;
            int idx1 = idx0 + quarter;
            int idx2 = idx1 + quarter;
            int idx3 = idx2 + quarter;
            int tw = j * (1024 / m);
            float2 x0 = srow[idx0];
            float2 x1 = cmul(TWIDDLE_1024[tw], srow[idx1]);
            float2 x2 = cmul(TWIDDLE_1024[tw * 2], srow[idx2]);
            float2 x3 = cmul(TWIDDLE_1024[tw * 3], srow[idx3]);

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

        float2 o0 = srow[pos0];
        float2 o1 = srow[pos1];
        float2 o2 = srow[pos2];
        float2 o3 = srow[pos3];
        sum0 += atan2f(o0.y, o0.x);
        sum1 += atan2f(o1.y, o1.x);
        sum2 += atan2f(o2.y, o2.x);
        sum3 += atan2f(o3.y, o3.x);
        __syncthreads();
    }

    size_t plane = 1024u * 1024u;
    size_t out_base = (size_t)group * plane;
    partial_sum[out_base + (size_t)pos0 * 1024u + col] = sum0;
    partial_sum[out_base + (size_t)pos1 * 1024u + col] = sum1;
    partial_sum[out_base + (size_t)pos2 * 1024u + col] = sum2;
    partial_sum[out_base + (size_t)pos3 * 1024u + col] = sum3;
}

__global__ __launch_bounds__(64, 8)
void ifft1024_cols_accumulate_split512_t64(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sumsq,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    if (k_bf < 0) k_bf = -k_bf;
    int col = blockIdx.y;
    int tid = threadIdx.x;
    if (col >= 1024 || tid >= 64) return;

    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    int src0 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 0));
    int src1 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 1));
    int src2 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 2));
    int src3 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 3));
    int src4 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 4));
    int src5 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 5));
    int src6 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 6));
    int src7 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 7));

    int j0 = tid;
    int j1 = tid + 64;
    int j2 = tid + 128;
    int j3 = tid + 192;
    int j4 = tid + 256;
    int j5 = tid + 320;
    int j6 = tid + 384;
    int j7 = tid + 448;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    float sum8 = 0.0f, sum9 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
    float sum12 = 0.0f, sum13 = 0.0f, sum14 = 0.0f, sum15 = 0.0f;
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
    float sq4 = 0.0f, sq5 = 0.0f, sq6 = 0.0f, sq7 = 0.0f;
    float sq8 = 0.0f, sq9 = 0.0f, sq10 = 0.0f, sq11 = 0.0f;
    float sq12 = 0.0f, sq13 = 0.0f, sq14 = 0.0f, sq15 = 0.0f;

    __shared__ float2 se[512];
    __shared__ float2 so[512];
    const size_t plane = 1024u * 1024u;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * plane + (size_t)col * 1024u;
        float2 e0 = data[base + (size_t)src0 * 2u];
        float2 e1 = data[base + (size_t)src1 * 2u];
        float2 e2 = data[base + (size_t)src2 * 2u];
        float2 e3 = data[base + (size_t)src3 * 2u];
        float2 e4 = data[base + (size_t)src4 * 2u];
        float2 e5 = data[base + (size_t)src5 * 2u];
        float2 e6 = data[base + (size_t)src6 * 2u];
        float2 e7 = data[base + (size_t)src7 * 2u];
        float2 o0 = data[base + (size_t)src0 * 2u + 1u];
        float2 o1 = data[base + (size_t)src1 * 2u + 1u];
        float2 o2 = data[base + (size_t)src2 * 2u + 1u];
        float2 o3 = data[base + (size_t)src3 * 2u + 1u];
        float2 o4 = data[base + (size_t)src4 * 2u + 1u];
        float2 o5 = data[base + (size_t)src5 * 2u + 1u];
        float2 o6 = data[base + (size_t)src6 * 2u + 1u];
        float2 o7 = data[base + (size_t)src7 * 2u + 1u];

        ifft512_radix8_apply_t64_1024tw(e0, e1, e2, e3, e4, e5, e6, e7, tid, se);
        ifft512_radix8_apply_t64_1024tw(o0, o1, o2, o3, o4, o5, o6, o7, tid, so);

#define ACCUM_SPLIT512(J, E, O, SUM_LO, SQ_LO, SUM_HI, SQ_HI) \
        { \
            float2 tw = TWIDDLE_1024[(J) & 1023]; \
            float2 to = cmul(tw, (O)); \
            float2 lo = cadd((E), to); \
            float2 hi = csub((E), to); \
            float plo = atan2f(lo.y, lo.x); \
            float phi = atan2f(hi.y, hi.x); \
            SUM_LO += plo; \
            SQ_LO += plo * plo; \
            SUM_HI += phi; \
            SQ_HI += phi * phi; \
        }
        ACCUM_SPLIT512(j0, e0, o0, sum0, sq0, sum8, sq8);
        ACCUM_SPLIT512(j1, e1, o1, sum1, sq1, sum9, sq9);
        ACCUM_SPLIT512(j2, e2, o2, sum2, sq2, sum10, sq10);
        ACCUM_SPLIT512(j3, e3, o3, sum3, sq3, sum11, sq11);
        ACCUM_SPLIT512(j4, e4, o4, sum4, sq4, sum12, sq12);
        ACCUM_SPLIT512(j5, e5, o5, sum5, sq5, sum13, sq13);
        ACCUM_SPLIT512(j6, e6, o6, sum6, sq6, sum14, sq14);
        ACCUM_SPLIT512(j7, e7, o7, sum7, sq7, sum15, sq15);
#undef ACCUM_SPLIT512
        __syncthreads();
    }

    size_t out_base = (size_t)group * plane;
    partial_sum[out_base + (size_t)j0 * 1024u + col] = sum0; partial_sumsq[out_base + (size_t)j0 * 1024u + col] = sq0;
    partial_sum[out_base + (size_t)j1 * 1024u + col] = sum1; partial_sumsq[out_base + (size_t)j1 * 1024u + col] = sq1;
    partial_sum[out_base + (size_t)j2 * 1024u + col] = sum2; partial_sumsq[out_base + (size_t)j2 * 1024u + col] = sq2;
    partial_sum[out_base + (size_t)j3 * 1024u + col] = sum3; partial_sumsq[out_base + (size_t)j3 * 1024u + col] = sq3;
    partial_sum[out_base + (size_t)j4 * 1024u + col] = sum4; partial_sumsq[out_base + (size_t)j4 * 1024u + col] = sq4;
    partial_sum[out_base + (size_t)j5 * 1024u + col] = sum5; partial_sumsq[out_base + (size_t)j5 * 1024u + col] = sq5;
    partial_sum[out_base + (size_t)j6 * 1024u + col] = sum6; partial_sumsq[out_base + (size_t)j6 * 1024u + col] = sq6;
    partial_sum[out_base + (size_t)j7 * 1024u + col] = sum7; partial_sumsq[out_base + (size_t)j7 * 1024u + col] = sq7;
    partial_sum[out_base + (size_t)(j0 + 512) * 1024u + col] = sum8; partial_sumsq[out_base + (size_t)(j0 + 512) * 1024u + col] = sq8;
    partial_sum[out_base + (size_t)(j1 + 512) * 1024u + col] = sum9; partial_sumsq[out_base + (size_t)(j1 + 512) * 1024u + col] = sq9;
    partial_sum[out_base + (size_t)(j2 + 512) * 1024u + col] = sum10; partial_sumsq[out_base + (size_t)(j2 + 512) * 1024u + col] = sq10;
    partial_sum[out_base + (size_t)(j3 + 512) * 1024u + col] = sum11; partial_sumsq[out_base + (size_t)(j3 + 512) * 1024u + col] = sq11;
    partial_sum[out_base + (size_t)(j4 + 512) * 1024u + col] = sum12; partial_sumsq[out_base + (size_t)(j4 + 512) * 1024u + col] = sq12;
    partial_sum[out_base + (size_t)(j5 + 512) * 1024u + col] = sum13; partial_sumsq[out_base + (size_t)(j5 + 512) * 1024u + col] = sq13;
    partial_sum[out_base + (size_t)(j6 + 512) * 1024u + col] = sum14; partial_sumsq[out_base + (size_t)(j6 + 512) * 1024u + col] = sq14;
    partial_sum[out_base + (size_t)(j7 + 512) * 1024u + col] = sum15; partial_sumsq[out_base + (size_t)(j7 + 512) * 1024u + col] = sq15;
}

__global__ __launch_bounds__(64, 8)
void ifft1024_cols_accumulate_sum_split512_t64(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    if (k_bf < 0) k_bf = -k_bf;
    int col = blockIdx.y;
    int tid = threadIdx.x;
    if (col >= 1024 || tid >= 64) return;

    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    int src0 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 0));
    int src1 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 1));
    int src2 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 2));
    int src3 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 3));
    int src4 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 4));
    int src5 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 5));
    int src6 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 6));
    int src7 = (int)octal_reverse_512_for_1024((unsigned int)(tid * 8 + 7));

    int j0 = tid;
    int j1 = tid + 64;
    int j2 = tid + 128;
    int j3 = tid + 192;
    int j4 = tid + 256;
    int j5 = tid + 320;
    int j6 = tid + 384;
    int j7 = tid + 448;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    float sum8 = 0.0f, sum9 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
    float sum12 = 0.0f, sum13 = 0.0f, sum14 = 0.0f, sum15 = 0.0f;

    __shared__ float2 se[512];
    __shared__ float2 so[512];
    const size_t plane = 1024u * 1024u;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * plane + (size_t)col * 1024u;
        float2 e0 = data[base + (size_t)src0 * 2u];
        float2 e1 = data[base + (size_t)src1 * 2u];
        float2 e2 = data[base + (size_t)src2 * 2u];
        float2 e3 = data[base + (size_t)src3 * 2u];
        float2 e4 = data[base + (size_t)src4 * 2u];
        float2 e5 = data[base + (size_t)src5 * 2u];
        float2 e6 = data[base + (size_t)src6 * 2u];
        float2 e7 = data[base + (size_t)src7 * 2u];
        float2 o0 = data[base + (size_t)src0 * 2u + 1u];
        float2 o1 = data[base + (size_t)src1 * 2u + 1u];
        float2 o2 = data[base + (size_t)src2 * 2u + 1u];
        float2 o3 = data[base + (size_t)src3 * 2u + 1u];
        float2 o4 = data[base + (size_t)src4 * 2u + 1u];
        float2 o5 = data[base + (size_t)src5 * 2u + 1u];
        float2 o6 = data[base + (size_t)src6 * 2u + 1u];
        float2 o7 = data[base + (size_t)src7 * 2u + 1u];

        ifft512_radix8_apply_t64_1024tw(e0, e1, e2, e3, e4, e5, e6, e7, tid, se);
        ifft512_radix8_apply_t64_1024tw(o0, o1, o2, o3, o4, o5, o6, o7, tid, so);

#define ACCUM_SUM_SPLIT512(J, E, O, SUM_LO, SUM_HI) \
        { \
            float2 tw = TWIDDLE_1024[(J) & 1023]; \
            float2 to = cmul(tw, (O)); \
            float2 lo = cadd((E), to); \
            float2 hi = csub((E), to); \
            SUM_LO += atan2f(lo.y, lo.x); \
            SUM_HI += atan2f(hi.y, hi.x); \
        }
        ACCUM_SUM_SPLIT512(j0, e0, o0, sum0, sum8);
        ACCUM_SUM_SPLIT512(j1, e1, o1, sum1, sum9);
        ACCUM_SUM_SPLIT512(j2, e2, o2, sum2, sum10);
        ACCUM_SUM_SPLIT512(j3, e3, o3, sum3, sum11);
        ACCUM_SUM_SPLIT512(j4, e4, o4, sum4, sum12);
        ACCUM_SUM_SPLIT512(j5, e5, o5, sum5, sum13);
        ACCUM_SUM_SPLIT512(j6, e6, o6, sum6, sum14);
        ACCUM_SUM_SPLIT512(j7, e7, o7, sum7, sum15);
#undef ACCUM_SUM_SPLIT512
        __syncthreads();
    }

    size_t out_base = (size_t)group * plane;
    partial_sum[out_base + (size_t)j0 * 1024u + col] = sum0;
    partial_sum[out_base + (size_t)j1 * 1024u + col] = sum1;
    partial_sum[out_base + (size_t)j2 * 1024u + col] = sum2;
    partial_sum[out_base + (size_t)j3 * 1024u + col] = sum3;
    partial_sum[out_base + (size_t)j4 * 1024u + col] = sum4;
    partial_sum[out_base + (size_t)j5 * 1024u + col] = sum5;
    partial_sum[out_base + (size_t)j6 * 1024u + col] = sum6;
    partial_sum[out_base + (size_t)j7 * 1024u + col] = sum7;
    partial_sum[out_base + (size_t)(j0 + 512) * 1024u + col] = sum8;
    partial_sum[out_base + (size_t)(j1 + 512) * 1024u + col] = sum9;
    partial_sum[out_base + (size_t)(j2 + 512) * 1024u + col] = sum10;
    partial_sum[out_base + (size_t)(j3 + 512) * 1024u + col] = sum11;
    partial_sum[out_base + (size_t)(j4 + 512) * 1024u + col] = sum12;
    partial_sum[out_base + (size_t)(j5 + 512) * 1024u + col] = sum13;
    partial_sum[out_base + (size_t)(j6 + 512) * 1024u + col] = sum14;
    partial_sum[out_base + (size_t)(j7 + 512) * 1024u + col] = sum15;
}

''' + corrected_fourier_sum_source(1024) + r'''

'''


class CustomFFT1024(CustomFFTBase):
    """Custom 1024x1024 IFFT kernels for SSB reconstruction and optimization."""

    def __init__(self) -> None:
        super().__init__(
            size=1024,
            cuda_code=build_cuda_code(1024, _TWIDDLE_DECL, _FFT1024_KERNELS),
            kernel_names=(
                "ifft1024_rows_fused_pk_t256_mr2_packed",
                "ifft1024_cols_t256_mr2",
                "ifft1024_cols_accumulate_t256_mr2",
                "ifft1024_rows_fused_pk_full_t256_mr2_packed",
                "ifft1024_cols_accumulate_sum_t256_mr2",
                "ifft1024_cols_accumulate_split512_t64",
                "ifft1024_cols_accumulate_sum_split512_t64",
                "ifft1024_rows_fused_pk_split512_t64_packed",
            ),
            twiddle_name="TWIDDLE_1024",
            rows_block=(256, 1, 1),
            rows_grid_y=1024,
            cols_block=(256, 1, 1),
            cols_grid_y=1024,
        )
        self._cols_accumulate_sum = self._module.get_function(
            "ifft1024_cols_accumulate_sum_t256_mr2"
        )
        self._cols_accumulate_split512 = self._module.get_function(
            "ifft1024_cols_accumulate_split512_t64"
        )
        self._cols_accumulate_sum_split512 = self._module.get_function(
            "ifft1024_cols_accumulate_sum_split512_t64"
        )
        self._rows_fused_pk_split512 = self._module.get_function(
            "ifft1024_rows_fused_pk_split512_t64_packed"
        )
        self._cols_split512_block = (64, 1, 1)
        self._rows_split512_block = (64, 1, 1)

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
        """Row FFT + fused column IFFT with transposed scratch accumulation."""
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

        grid_rows = (1, self._rows_fused_pk_grid_y, num_bf)
        self._rows_fused_pk_split512(
            grid_rows,
            self._rows_split512_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(-gqk_cols),
            ),
        )
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate_split512(
            grid_cols,
            self._cols_split512_block,
            (data, partial_sum, partial_sumsq, np.int32(num_bf), np.int32(-k_bf)),
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
        k_bf: int = 32,
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

        grid_rows = (1, self._rows_fused_pk_grid_y, num_bf)
        self._rows_fused_pk_split512(
            grid_rows,
            self._rows_split512_block,
            (
                kx_bf, ky_bf, qx_1d, qy_1d,
                np.float32(wavelength), np.float32(semiangle_rad),
                np.float32(ang_y_rad), np.float32(ang_x_rad),
                np.float32(C10), np.float32(C12),
                np.float32(cos2phi12), np.float32(sin2phi12),
                np.float32(factor), pk, G_qk, data,
                np.float32(dc_value.real), np.float32(dc_value.imag),
                np.int32(num_bf), np.int32(-gqk_cols),
            ),
        )
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate_sum_split512(
            grid_cols,
            self._cols_split512_block,
            (data, partial_sum, np.int32(num_bf), np.int32(-k_bf)),
        )


@lru_cache(maxsize=1)
def get_custom_fft_1024() -> CustomFFT1024:
    return CustomFFT1024()
