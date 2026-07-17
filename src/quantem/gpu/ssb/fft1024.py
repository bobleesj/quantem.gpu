"""Custom fixed-size CUDA FFT kernels for SSB (1024x1024).

1024 = 4^5.  This backend provides the full reconstruct path first:
fused gamma multiplication, row IFFT, column IFFT, and fused column phase
accumulation.  Batched optimizer kernels remain intentionally disabled until
the 1024 variance kernel is specialized separately.
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
    int num_bf
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
        pkv.x, pkv.y, ld_float2(G_qk, base + pos0));
    float2 res1 = gamma_mul_pk_onthefly(
        qx, __ldg(&qy_1d[pos1]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        pkv.x, pkv.y, ld_float2(G_qk, base + pos1));
    float2 res2 = gamma_mul_pk_onthefly(
        qx, __ldg(&qy_1d[pos2]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        pkv.x, pkv.y, ld_float2(G_qk, base + pos2));
    float2 res3 = gamma_mul_pk_onthefly(
        qx, __ldg(&qy_1d[pos3]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        pkv.x, pkv.y, ld_float2(G_qk, base + pos3));

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
    int num_bf
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
        pkv.x, pkv.y, ld_float2(G_qk, base + pos0));
    float2 res1 = gamma_mul_pk_onthefly_full(
        qx, __ldg(&qy_1d[pos1]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        abr_mag_scaled, abr_cm, abr_sm,
        pkv.x, pkv.y, ld_float2(G_qk, base + pos1));
    float2 res2 = gamma_mul_pk_onthefly_full(
        qx, __ldg(&qy_1d[pos2]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        abr_mag_scaled, abr_cm, abr_sm,
        pkv.x, pkv.y, ld_float2(G_qk, base + pos2));
    float2 res3 = gamma_mul_pk_onthefly_full(
        qx, __ldg(&qy_1d[pos3]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        abr_mag_scaled, abr_cm, abr_sm,
        pkv.x, pkv.y, ld_float2(G_qk, base + pos3));

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
    int k_bf
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
            ld_float2(G_qk, (unsigned long long)bf * plane + idx));
        sum_re += v.x;
        sum_im += v.y;
    }
    partial_sum[(unsigned long long)group * plane + idx] = make_float2(sum_re, sum_im);
}

__global__ void ifft1024_rows_fused_pk_batch_dummy(
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
    int batch
) {}

__global__ void ifft1024_rows_var_dummy(
    const float2* __restrict__ data,
    float* __restrict__ sum,
    float* __restrict__ sumsq,
    int num_bf,
    int batch,
    float scale,
    int use_partial
) {}
'''


class CustomFFT1024(CustomFFTBase):
    """Custom 1024x1024 IFFT kernels for SSB reconstruction."""

    def __init__(self) -> None:
        super().__init__(
            size=1024,
            cuda_code=build_cuda_code(1024, _TWIDDLE_DECL, _FFT1024_KERNELS),
            kernel_names=(
                "ifft1024_rows_fused_pk_t256_mr2_packed",
                "ifft1024_rows_fused_pk_batch_dummy",
                "ifft1024_cols_t256_mr2",
                "ifft1024_rows_var_dummy",
                "ifft1024_cols_accumulate_t256_mr2",
                "ifft1024_rows_fused_pk_full_t256_mr2_packed",
                "ifft1024_cols_accumulate_sum_t256_mr2",
                "ssb1024_corrected_fourier_sum_t256",
            ),
            twiddle_name="TWIDDLE_1024",
            rows_block=(256, 2, 1),
            rows_grid_y=512,
            batch_block=(256, 1, 1),
            batch_grid_y=512,
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
        if data.shape != G_qk.shape:
            raise ValueError("data and G_qk must have the same shape")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bf = int(data.shape[0])
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
                np.int32(num_bf),
            ),
        )
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate_sum(
            grid_cols,
            self._cols_block,
            (data, partial_sum, np.int32(num_bf), np.int32(k_bf)),
        )

    def ifft2_inplace_batch_fused_pk_variance(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "1024x1024 CUDA SSB currently supports reconstruction and "
            "reconstruct_with_loss; the batched optimizer variance kernel "
            "needs a dedicated 1024 specialization."
        )


@lru_cache(maxsize=1)
def get_custom_fft_1024() -> CustomFFT1024:
    return CustomFFT1024()
