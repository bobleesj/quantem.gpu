"""Custom fixed-size CUDA FFT kernels for SSB (1024x1024).

1024 = 4^5.  This backend provides the full reconstruct path and the
row-subsampled batched optimizer objective used by the 256/512 CUDA paths:
fused gamma multiplication, row IFFT, column IFFT, and fused column phase
accumulation.
"""

from functools import lru_cache

import cupy as cp
import numpy as np

from .fft_common import CustomFFTBase, build_cuda_code

_TWIDDLE_DECL = '__constant__ float2 TWIDDLE_1024[1024];'

_FFT1024_KERNELS = r'''
__device__ __forceinline__ unsigned int digit_reverse_1024(unsigned int x) {
    return ((x & 0x003u) << 8) | ((x & 0x00Cu) << 4) |
           (x & 0x030u) |
           ((x & 0x0C0u) >> 4) | ((x & 0x300u) >> 8);
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
    int row = blockIdx.y * 2 + threadIdx.y;
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

	    float2 res0 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos0]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos0, 1024u, (unsigned int)gqk_cols));
	    float2 res1 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 1024u, (unsigned int)gqk_cols));
	    float2 res2 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 1024u, (unsigned int)gqk_cols));
	    float2 res3 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pkv.x, pkv.y, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 1024u, (unsigned int)gqk_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[2][1024];
    float2* srow = s[threadIdx.y];
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
    int row = blockIdx.y * 2 + threadIdx.y;
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

    __shared__ float2 s[2][1024];
    float2* srow = s[threadIdx.y];
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
    int col = blockIdx.y * 2 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || col >= 1024 || tid >= 256) return;

    size_t base = (size_t)bf * 1024u * 1024u + col;
    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;

    __shared__ float2 s[2][1024];
    float2* srow = s[threadIdx.y];
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
    int col = blockIdx.y * 2 + threadIdx.y;
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

    __shared__ float2 s[2][1024];
    float2* srow = s[threadIdx.y];

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * 1024u * 1024u + col;
        srow[rev0] = data[base + (size_t)pos0 * 1024u];
        srow[rev1] = data[base + (size_t)pos1 * 1024u];
        srow[rev2] = data[base + (size_t)pos2 * 1024u];
        srow[rev3] = data[base + (size_t)pos3 * 1024u];
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
    int col = blockIdx.y * 2 + threadIdx.y;
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

    __shared__ float2 s[2][1024];
    float2* srow = s[threadIdx.y];

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        size_t base = (size_t)bf * 1024u * 1024u + col;
        srow[rev0] = data[base + (size_t)pos0 * 1024u];
        srow[rev1] = data[base + (size_t)pos1 * 1024u];
        srow[rev2] = data[base + (size_t)pos2 * 1024u];
        srow[rev3] = data[base + (size_t)pos3 * 1024u];
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

__global__ void ssb1024_corrected_fourier_sum_t256(
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
    const unsigned long long plane = 1024ull * 1024ull;
    int groups = (num_bf + k_bf - 1) / k_bf;
    unsigned long long total = (unsigned long long)groups * plane;
    if (linear >= total) return;

    int group = (int)(linear / plane);
    unsigned int idx = (unsigned int)(linear - (unsigned long long)group * plane);
    int row = idx / 1024u;
    int col = idx - (unsigned int)row * 1024u;
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
                (unsigned int)col, 1024u, (unsigned int)gqk_cols
            ));
        sum_re += v.x;
        sum_im += v.y;
    }
    partial_sum[(unsigned long long)group * plane + idx] = make_float2(sum_re, sum_im);
}

__global__ __launch_bounds__(256, 4)
void ifft1024_rows_fused_pk_batch_t256_mr1_transpose_packed_b4(
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
    // Match the sparse-row optimizer pattern used by the 256/512 kernels:
    // sample 128 interleaved qx rows rather than all 1024 rows.
    int row = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (cand0 >= batch || bf >= num_bf || row >= 1024 || tid >= 256) return;

    bool has1 = cand1 < batch;
    bool has2 = cand2 < batch;
    bool has3 = cand3 < batch;
    float C10v0 = C10[cand0], C12v0 = C12[cand0];
    float cos2v0 = cos2phi12[cand0], sin2v0 = sin2phi12[cand0];
    float C10v1 = has1 ? C10[cand1] : 0.0f, C12v1 = has1 ? C12[cand1] : 0.0f;
    float cos2v1 = has1 ? cos2phi12[cand1] : 0.0f, sin2v1 = has1 ? sin2phi12[cand1] : 0.0f;
    float C10v2 = has2 ? C10[cand2] : 0.0f, C12v2 = has2 ? C12[cand2] : 0.0f;
    float cos2v2 = has2 ? cos2phi12[cand2] : 0.0f, sin2v2 = has2 ? sin2phi12[cand2] : 0.0f;
    float C10v3 = has3 ? C10[cand3] : 0.0f, C12v3 = has3 ? C12[cand3] : 0.0f;
    float cos2v3 = has3 ? cos2phi12[cand3] : 0.0f, sin2v3 = has3 ? sin2phi12[cand3] : 0.0f;

    size_t base_cache = ((size_t)bf * 1024u + (size_t)row) * 1024u;
    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;
    float2 pkv0 = pk[(size_t)cand0 * (size_t)num_bf + (size_t)bf];
    float2 pkv1 = has1 ? pk[(size_t)cand1 * (size_t)num_bf + (size_t)bf] : make_float2(0.0f, 0.0f);
    float2 pkv2 = has2 ? pk[(size_t)cand2 * (size_t)num_bf + (size_t)bf] : make_float2(0.0f, 0.0f);
    float2 pkv3 = has3 ? pk[(size_t)cand3 * (size_t)num_bf + (size_t)bf] : make_float2(0.0f, 0.0f);
    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    float qx = __ldg(&qx_1d[row]);

    float qy0 = __ldg(&qy_1d[pos0]);
    float qy1 = __ldg(&qy_1d[pos1]);
    float qy2 = __ldg(&qy_1d[pos2]);
    float qy3 = __ldg(&qy_1d[pos3]);
    float4 m0 = compute_geometry(qx - kx, qy0 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p0 = compute_geometry(qx + kx, qy0 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 m1 = compute_geometry(qx - kx, qy1 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p1 = compute_geometry(qx + kx, qy1 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 m2 = compute_geometry(qx - kx, qy2 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p2 = compute_geometry(qx + kx, qy2 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 m3 = compute_geometry(qx - kx, qy3 - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
    float4 p3 = compute_geometry(qx + kx, qy3 + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
	    float2 G0 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos0, 1024u, (unsigned int)gqk_cols);
	    float2 G1 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos1, 1024u, (unsigned int)gqk_cols);
	    float2 G2 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos2, 1024u, (unsigned int)gqk_cols);
	    float2 G3 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos3, 1024u, (unsigned int)gqk_cols);

#define GM1024(m,p,G,C10v,C12v,cosv,sinv,pkv) \
    gamma_mul_pk_packed_vals(m,p,G,C10v,C12v,cosv,sinv,factor,pkv.x,pkv.y)
    float2 r00 = GM1024(m0, p0, G0, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r10 = GM1024(m1, p1, G1, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r20 = GM1024(m2, p2, G2, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r30 = GM1024(m3, p3, G3, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r01 = has1 ? GM1024(m0, p0, G0, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r11 = has1 ? GM1024(m1, p1, G1, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r21 = has1 ? GM1024(m2, p2, G2, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r31 = has1 ? GM1024(m3, p3, G3, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r02 = has2 ? GM1024(m0, p0, G0, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r12 = has2 ? GM1024(m1, p1, G1, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r22 = has2 ? GM1024(m2, p2, G2, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r32 = has2 ? GM1024(m3, p3, G3, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r03 = has3 ? GM1024(m0, p0, G0, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
    float2 r13 = has3 ? GM1024(m1, p1, G1, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
    float2 r23 = has3 ? GM1024(m2, p2, G2, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
    float2 r33 = has3 ? GM1024(m3, p3, G3, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
#undef GM1024

    float4 res0a = make_float4(r00.x, r00.y, r01.x, r01.y);
    float4 res1a = make_float4(r10.x, r10.y, r11.x, r11.y);
    float4 res2a = make_float4(r20.x, r20.y, r21.x, r21.y);
    float4 res3a = make_float4(r30.x, r30.y, r31.x, r31.y);
    float4 res0b = make_float4(r02.x, r02.y, r03.x, r03.y);
    float4 res1b = make_float4(r12.x, r12.y, r13.x, r13.y);
    float4 res2b = make_float4(r22.x, r22.y, r23.x, r23.y);
    float4 res3b = make_float4(r32.x, r32.y, r33.x, r33.y);
    if (row == 0 && tid == 0) {
        res0a.x = dc_real; res0a.y = dc_imag;
        if (has1) { res0a.z = dc_real; res0a.w = dc_imag; }
        if (has2) { res0b.x = dc_real; res0b.y = dc_imag; }
        if (has3) { res0b.z = dc_real; res0b.w = dc_imag; }
    }

    __shared__ float4 s0[1024];
    __shared__ float4 s1[1024];
    s0[digit_reverse_1024((unsigned int)pos0)] = res0a;
    s0[digit_reverse_1024((unsigned int)pos1)] = res1a;
    s0[digit_reverse_1024((unsigned int)pos2)] = res2a;
    s0[digit_reverse_1024((unsigned int)pos3)] = res3a;
    s1[digit_reverse_1024((unsigned int)pos0)] = res0b;
    s1[digit_reverse_1024((unsigned int)pos1)] = res1b;
    s1[digit_reverse_1024((unsigned int)pos2)] = res2b;
    s1[digit_reverse_1024((unsigned int)pos3)] = res3b;
    __syncthreads();

    for (int m = 4; m <= 1024; m <<= 2) {
        int quarter = m >> 2;
        int j = tid % quarter;
        int k = tid / quarter;
        int idx0s = k * m + j;
        int idx1s = idx0s + quarter;
        int idx2s = idx1s + quarter;
        int idx3s = idx2s + quarter;
        int tw = j * (1024 / m);
        float2 w0 = TWIDDLE_1024[tw];
        float2 w1 = TWIDDLE_1024[tw * 2];
        float2 w2 = TWIDDLE_1024[tw * 3];

        float4 x0 = s0[idx0s], x1 = cmul2(w0, s0[idx1s]);
        float4 x2 = cmul2(w1, s0[idx2s]), x3 = cmul2(w2, s0[idx3s]);
        float4 t0 = cadd2(x0, x2), t1 = csub2(x0, x2);
        float4 t2 = cadd2(x1, x3), t3 = csub2(x1, x3);
        float4 it3 = cmul_i2(t3);
        s0[idx0s] = cadd2(t0, t2);
        s0[idx1s] = cadd2(t1, it3);
        s0[idx2s] = csub2(t0, t2);
        s0[idx3s] = csub2(t1, it3);

        float4 y0 = s1[idx0s], y1 = cmul2(w0, s1[idx1s]);
        float4 y2 = cmul2(w1, s1[idx2s]), y3 = cmul2(w2, s1[idx3s]);
        float4 u0 = cadd2(y0, y2), u1 = csub2(y0, y2);
        float4 u2 = cadd2(y1, y3), u3 = csub2(y1, y3);
        float4 iu3 = cmul_i2(u3);
        s1[idx0s] = cadd2(u0, u2);
        s1[idx1s] = cadd2(u1, iu3);
        s1[idx2s] = csub2(u0, u2);
        s1[idx3s] = csub2(u1, iu3);
        __syncthreads();
    }

    float4 out0a = s0[pos0], out1a = s0[pos1], out2a = s0[pos2], out3a = s0[pos3];
    size_t out_idx00 = (((size_t)cand0 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos0;
    size_t out_idx01 = (((size_t)cand0 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos1;
    size_t out_idx02 = (((size_t)cand0 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos2;
    size_t out_idx03 = (((size_t)cand0 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos3;
    out[out_idx00] = make_float2(out0a.x, out0a.y);
    out[out_idx01] = make_float2(out1a.x, out1a.y);
    out[out_idx02] = make_float2(out2a.x, out2a.y);
    out[out_idx03] = make_float2(out3a.x, out3a.y);
    if (has1) {
        size_t out_idx10 = (((size_t)cand1 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos0;
        size_t out_idx11 = (((size_t)cand1 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos1;
        size_t out_idx12 = (((size_t)cand1 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos2;
        size_t out_idx13 = (((size_t)cand1 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos3;
        out[out_idx10] = make_float2(out0a.z, out0a.w);
        out[out_idx11] = make_float2(out1a.z, out1a.w);
        out[out_idx12] = make_float2(out2a.z, out2a.w);
        out[out_idx13] = make_float2(out3a.z, out3a.w);
    }
    if (has2 || has3) {
        float4 out0b = s1[pos0], out1b = s1[pos1], out2b = s1[pos2], out3b = s1[pos3];
        if (has2) {
            size_t out_idx20 = (((size_t)cand2 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos0;
            size_t out_idx21 = (((size_t)cand2 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos1;
            size_t out_idx22 = (((size_t)cand2 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos2;
            size_t out_idx23 = (((size_t)cand2 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos3;
            out[out_idx20] = make_float2(out0b.x, out0b.y);
            out[out_idx21] = make_float2(out1b.x, out1b.y);
            out[out_idx22] = make_float2(out2b.x, out2b.y);
            out[out_idx23] = make_float2(out3b.x, out3b.y);
        }
        if (has3) {
            size_t out_idx30 = (((size_t)cand3 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos0;
            size_t out_idx31 = (((size_t)cand3 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos1;
            size_t out_idx32 = (((size_t)cand3 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos2;
            size_t out_idx33 = (((size_t)cand3 * (size_t)num_bf + (size_t)bf) * 1024u + (size_t)row) * 1024u + (size_t)pos3;
            out[out_idx30] = make_float2(out0b.z, out0b.w);
            out[out_idx31] = make_float2(out1b.z, out1b.w);
            out[out_idx32] = make_float2(out2b.z, out2b.w);
            out[out_idx33] = make_float2(out3b.z, out3b.w);
        }
    }
}

__global__ __launch_bounds__(256, 4)
void ifft1024_rows_var_t256_mr4_batch(
    const float2* __restrict__ data,
    float* __restrict__ sum,
    float* __restrict__ sumsq,
    int num_bf,
    int batch,
    float scale,
    int use_partial
) {
    (void)scale;
    int col = blockIdx.y;
    int tid = threadIdx.x;
    if (col >= 1024 || tid >= 256) return;
    int groups = (num_bf + 31) >> 5;
    int group = blockIdx.z % groups;
    int cand = blockIdx.z / groups;
    if (cand >= batch) return;
    int bf0 = group * 32;

    int pos0 = tid;
    int pos1 = tid + 256;
    int pos2 = tid + 512;
    int pos3 = tid + 768;
    int rev0 = digit_reverse_1024((unsigned int)pos0);
    int rev1 = digit_reverse_1024((unsigned int)pos1);
    int rev2 = digit_reverse_1024((unsigned int)pos2);
    int rev3 = digit_reverse_1024((unsigned int)pos3);

    __shared__ float2 s[1024];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
    const size_t plane = 1024u * 1024u;
    size_t cand_base = ((size_t)cand * (size_t)num_bf) * plane + (size_t)col;
    size_t sum_base = use_partial
        ? ((size_t)cand * (size_t)groups + (size_t)group) * plane
        : (size_t)cand * plane;

#pragma unroll 1
    for (int i = 0; i < 32; ++i) {
        int bf = bf0 + i;
        float2 in0 = make_float2(0.0f, 0.0f);
        float2 in1 = make_float2(0.0f, 0.0f);
        float2 in2 = make_float2(0.0f, 0.0f);
        float2 in3 = make_float2(0.0f, 0.0f);
        if (bf < num_bf) {
            size_t bf_base = cand_base + (size_t)bf * plane;
            in0 = data[bf_base + (size_t)pos0 * 1024u];
            in1 = data[bf_base + (size_t)pos1 * 1024u];
            in2 = data[bf_base + (size_t)pos2 * 1024u];
            in3 = data[bf_base + (size_t)pos3 * 1024u];
        }
        s[rev0] = in0;
        s[rev1] = in1;
        s[rev2] = in2;
        s[rev3] = in3;
        __syncthreads();

        for (int m = 4; m <= 1024; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter;
            int k = tid / quarter;
            int idx0s = k * m + j;
            int idx1s = idx0s + quarter;
            int idx2s = idx1s + quarter;
            int idx3s = idx2s + quarter;
            int tw = j * (1024 / m);
            float2 x0 = s[idx0s];
            float2 x1 = cmul(TWIDDLE_1024[tw], s[idx1s]);
            float2 x2 = cmul(TWIDDLE_1024[tw * 2], s[idx2s]);
            float2 x3 = cmul(TWIDDLE_1024[tw * 3], s[idx3s]);

            float2 t0 = cadd(x0, x2);
            float2 t1 = csub(x0, x2);
            float2 t2 = cadd(x1, x3);
            float2 t3 = csub(x1, x3);
            float2 it3 = cmul_i(t3);
            s[idx0s] = cadd(t0, t2);
            s[idx1s] = cadd(t1, it3);
            s[idx2s] = csub(t0, t2);
            s[idx3s] = csub(t1, it3);
            __syncthreads();
        }

        if (bf < num_bf) {
            float2 o0 = s[pos0];
            float2 o1 = s[pos1];
            float2 o2 = s[pos2];
            float2 o3 = s[pos3];
            float p0 = atan2f(o0.y, o0.x);
            float p1 = atan2f(o1.y, o1.x);
            float p2 = atan2f(o2.y, o2.x);
            float p3 = atan2f(o3.y, o3.x);
            sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
            sq0 += p0 * p0; sq1 += p1 * p1; sq2 += p2 * p2; sq3 += p3 * p3;
        }
        __syncthreads();
    }

    size_t o0 = sum_base + (size_t)pos0 * 1024u + (size_t)col;
    size_t o1 = sum_base + (size_t)pos1 * 1024u + (size_t)col;
    size_t o2 = sum_base + (size_t)pos2 * 1024u + (size_t)col;
    size_t o3 = sum_base + (size_t)pos3 * 1024u + (size_t)col;
    if (use_partial) {
        sum[o0] = sum0; sumsq[o0] = sq0;
        sum[o1] = sum1; sumsq[o1] = sq1;
        sum[o2] = sum2; sumsq[o2] = sq2;
        sum[o3] = sum3; sumsq[o3] = sq3;
    } else {
        atomicAdd(&sum[o0], sum0); atomicAdd(&sumsq[o0], sq0);
        atomicAdd(&sum[o1], sum1); atomicAdd(&sumsq[o1], sq1);
        atomicAdd(&sum[o2], sum2); atomicAdd(&sumsq[o2], sq2);
        atomicAdd(&sum[o3], sum3); atomicAdd(&sumsq[o3], sq3);
    }
}

'''


class CustomFFT1024(CustomFFTBase):
    """Custom 1024x1024 IFFT kernels for SSB reconstruction and optimization."""

    def __init__(self) -> None:
        super().__init__(
            size=1024,
            cuda_code=build_cuda_code(1024, _TWIDDLE_DECL, _FFT1024_KERNELS),
            kernel_names=(
                "ifft1024_rows_fused_pk_t256_mr2_packed",
                "ifft1024_rows_fused_pk_batch_t256_mr1_transpose_packed_b4",
                "ifft1024_cols_t256_mr2",
                "ifft1024_rows_var_t256_mr4_batch",
                "ifft1024_cols_accumulate_t256_mr2",
                "ifft1024_rows_fused_pk_full_t256_mr2_packed",
                "ifft1024_cols_accumulate_sum_t256_mr2",
                "ssb1024_corrected_fourier_sum_t256",
            ),
            twiddle_name="TWIDDLE_1024",
            rows_block=(256, 2, 1),
            rows_grid_y=512,
            batch_block=(256, 1, 1),
            batch_grid_y=128,
            var_block=(256, 1, 1),
            var_grid_y=1024,
            cols_block=(256, 2, 1),
            cols_grid_y=512,
        )
        self._cols_accumulate_sum = self._module.get_function(
            "ifft1024_cols_accumulate_sum_t256_mr2"
        )
        self._fourier_sum = self._module.get_function("ssb1024_corrected_fourier_sum_t256")

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
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate_sum(
            grid_cols,
            self._cols_block,
            (data, partial_sum, np.int32(num_bf), np.int32(k_bf)),
        )


@lru_cache(maxsize=1)
def get_custom_fft_1024() -> CustomFFT1024:
    return CustomFFT1024()
