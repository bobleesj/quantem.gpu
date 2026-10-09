"""Custom fixed-size CUDA FFT kernels for SSB (128x128).

128 = 4^3 x 2: three radix-4 stages plus a final radix-2 stage.
"""

from functools import lru_cache

from quantem.gpu.ssb.cuda.kernels.common import (
    CustomFFTBase,
    build_cuda_code,
    corrected_fourier_sum_source,
)

_TWIDDLE_DECL = '__constant__ float2 TWIDDLE_128[128];'

_FFT128_KERNELS = r'''
__device__ __forceinline__ unsigned int bit_reverse4_6(unsigned int x) {
    return ((x & 0x03u) << 4) | (x & 0x0Cu) | ((x & 0x30u) >> 4);
}

__device__ __forceinline__ unsigned int digit_reverse_128(unsigned int x) {
    return ((x & 1u) << 6) | bit_reverse4_6(x >> 1);
}

#define FFT128_SYNC() __syncwarp()

__global__ void ifft128_rows_fused_pk_t32_mr8_packed(
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
    int row = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 128 || tid >= 32) return;

    int base = (bf * 128 + row) * 128;
    int pos0 = tid;
    int pos1 = tid + 32;
    int pos2 = tid + 64;
    int pos3 = tid + 96;
    int idx0 = base + pos0;
    int idx1 = base + pos1;
    int idx2 = base + pos2;
    int idx3 = base + pos3;
    float2 pkv = pk[bf];
    float pk_re = pkv.x;
    float pk_im = pkv.y;
    float kx = __ldg(&kx_bf[bf]);
    float ky = __ldg(&ky_bf[bf]);
    float qx = __ldg(&qx_1d[row]);

	    float2 res0 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos0]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos0, 128u, (unsigned int)gqk_cols));
	    float2 res1 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 128u, (unsigned int)gqk_cols));
	    float2 res2 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 128u, (unsigned int)gqk_cols));
	    float2 res3 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 128u, (unsigned int)gqk_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[8][128];
    float2* srow = s[threadIdx.y];
    srow[digit_reverse_128((unsigned int)pos0)] = res0;
    srow[digit_reverse_128((unsigned int)pos1)] = res1;
    srow[digit_reverse_128((unsigned int)pos2)] = res2;
    srow[digit_reverse_128((unsigned int)pos3)] = res3;
    FFT128_SYNC();

    for (int m = 4; m <= 64; m <<= 2) {
        int quarter = m >> 2;
        int butterfly = tid;
        int j = butterfly % quarter;
        int k = butterfly / quarter;
        int idx0s = k * m + j;
        int idx1s = idx0s + quarter;
        int idx2s = idx1s + quarter;
        int idx3s = idx2s + quarter;
        int tw = j * (128 / m);
        float2 x0 = srow[idx0s];
        float2 x1 = cmul(TWIDDLE_128[tw], srow[idx1s]);
        float2 x2 = cmul(TWIDDLE_128[tw * 2], srow[idx2s]);
        float2 x3 = cmul(TWIDDLE_128[tw * 3], srow[idx3s]);

        float2 t0 = cadd(x0, x2);
        float2 t1 = csub(x0, x2);
        float2 t2 = cadd(x1, x3);
        float2 t3 = csub(x1, x3);
        float2 it3 = cmul_i(t3);
        srow[idx0s] = cadd(t0, t2);
        srow[idx1s] = cadd(t1, it3);
        srow[idx2s] = csub(t0, t2);
        srow[idx3s] = csub(t1, it3);
        FFT128_SYNC();
    }

    int j0 = tid;
    int j1 = tid + 32;
    float2 a0 = srow[j0], b0 = cmul(TWIDDLE_128[j0], srow[j0 + 64]);
    float2 a1 = srow[j1], b1 = cmul(TWIDDLE_128[j1], srow[j1 + 64]);
    srow[j0] = cadd(a0, b0);
    srow[j0 + 64] = csub(a0, b0);
    srow[j1] = cadd(a1, b1);
    srow[j1 + 64] = csub(a1, b1);
    FFT128_SYNC();

    out[idx0] = srow[pos0];
    out[idx1] = srow[pos1];
    out[idx2] = srow[pos2];
    out[idx3] = srow[pos3];
}

__global__ void ifft128_cols_t32_mr8(float2* __restrict__ data,
                                     int num_bf,
                                     float scale) {
    int bf = blockIdx.z;
    int col = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || col >= 128 || tid >= 32) return;
    int base = bf * 128 * 128 + col;
    __shared__ float2 s[8][128];
    float2* srow = s[threadIdx.y];
    int pos0 = tid, pos1 = tid + 32, pos2 = tid + 64, pos3 = tid + 96;
    srow[digit_reverse_128((unsigned int)pos0)] = data[base + pos0 * 128];
    srow[digit_reverse_128((unsigned int)pos1)] = data[base + pos1 * 128];
    srow[digit_reverse_128((unsigned int)pos2)] = data[base + pos2 * 128];
    srow[digit_reverse_128((unsigned int)pos3)] = data[base + pos3 * 128];
    FFT128_SYNC();
    for (int m = 4; m <= 64; m <<= 2) {
        int quarter = m >> 2;
        int j = tid % quarter;
        int k = tid / quarter;
        int idx0 = k * m + j, idx1 = idx0 + quarter, idx2 = idx1 + quarter, idx3 = idx2 + quarter;
        int tw = j * (128 / m);
        float2 x0 = srow[idx0], x1 = cmul(TWIDDLE_128[tw], srow[idx1]);
        float2 x2 = cmul(TWIDDLE_128[tw * 2], srow[idx2]), x3 = cmul(TWIDDLE_128[tw * 3], srow[idx3]);
        float2 t0 = cadd(x0, x2), t1 = csub(x0, x2);
        float2 t2 = cadd(x1, x3), t3 = csub(x1, x3);
        float2 it3 = cmul_i(t3);
        srow[idx0] = cadd(t0, t2); srow[idx1] = cadd(t1, it3);
        srow[idx2] = csub(t0, t2); srow[idx3] = csub(t1, it3);
        FFT128_SYNC();
    }
    int j0 = tid, j1 = tid + 32;
    float2 a0 = srow[j0], b0 = cmul(TWIDDLE_128[j0], srow[j0 + 64]);
    float2 a1 = srow[j1], b1 = cmul(TWIDDLE_128[j1], srow[j1 + 64]);
    srow[j0] = cadd(a0, b0); srow[j0 + 64] = csub(a0, b0);
    srow[j1] = cadd(a1, b1); srow[j1 + 64] = csub(a1, b1);
    FFT128_SYNC();
    float2 o0 = srow[pos0], o1 = srow[pos1], o2 = srow[pos2], o3 = srow[pos3];
    o0.x *= scale; o0.y *= scale; o1.x *= scale; o1.y *= scale;
    o2.x *= scale; o2.y *= scale; o3.x *= scale; o3.y *= scale;
    data[base + pos0 * 128] = o0;
    data[base + pos1 * 128] = o1;
    data[base + pos2 * 128] = o2;
    data[base + pos3 * 128] = o3;
}

__global__ void ifft128_cols_accumulate_t32_mr8(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sumsq,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    int col = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (col >= 128 || tid >= 32) return;
    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;
    __shared__ float2 s[8][128];
    float2* srow = s[threadIdx.y];
    int pos0 = tid, pos1 = tid + 32, pos2 = tid + 64, pos3 = tid + 96;
    int rev0 = digit_reverse_128((unsigned int)pos0);
    int rev1 = digit_reverse_128((unsigned int)pos1);
    int rev2 = digit_reverse_128((unsigned int)pos2);
    int rev3 = digit_reverse_128((unsigned int)pos3);
    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    float sq0 = 0, sq1 = 0, sq2 = 0, sq3 = 0;
    for (int bf = bf_start; bf < bf_end; ++bf) {
        int base = bf * 128 * 128 + col;
        srow[rev0] = data[base + pos0 * 128];
        srow[rev1] = data[base + pos1 * 128];
        srow[rev2] = data[base + pos2 * 128];
        srow[rev3] = data[base + pos3 * 128];
        FFT128_SYNC();
        for (int m = 4; m <= 64; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter, k = tid / quarter;
            int idx0 = k * m + j, idx1 = idx0 + quarter, idx2 = idx1 + quarter, idx3 = idx2 + quarter;
            int tw = j * (128 / m);
            float2 x0 = srow[idx0], x1 = cmul(TWIDDLE_128[tw], srow[idx1]);
            float2 x2 = cmul(TWIDDLE_128[tw * 2], srow[idx2]), x3 = cmul(TWIDDLE_128[tw * 3], srow[idx3]);
            float2 t0 = cadd(x0, x2), t1 = csub(x0, x2);
            float2 t2 = cadd(x1, x3), t3 = csub(x1, x3);
            float2 it3 = cmul_i(t3);
            srow[idx0] = cadd(t0, t2); srow[idx1] = cadd(t1, it3);
            srow[idx2] = csub(t0, t2); srow[idx3] = csub(t1, it3);
            FFT128_SYNC();
        }
        int j0 = tid, j1 = tid + 32;
        float2 a0 = srow[j0], b0 = cmul(TWIDDLE_128[j0], srow[j0 + 64]);
        float2 a1 = srow[j1], b1 = cmul(TWIDDLE_128[j1], srow[j1 + 64]);
        srow[j0] = cadd(a0, b0); srow[j0 + 64] = csub(a0, b0);
        srow[j1] = cadd(a1, b1); srow[j1 + 64] = csub(a1, b1);
        FFT128_SYNC();
        float2 o0 = srow[pos0], o1 = srow[pos1], o2 = srow[pos2], o3 = srow[pos3];
        float p0 = atan2f(o0.y, o0.x), p1 = atan2f(o1.y, o1.x);
        float p2 = atan2f(o2.y, o2.x), p3 = atan2f(o3.y, o3.x);
        sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
        sq0 += p0 * p0; sq1 += p1 * p1; sq2 += p2 * p2; sq3 += p3 * p3;
    }
    size_t out_base = (size_t)group * 128u * 128u;
    partial_sum[out_base + (size_t)pos0 * 128 + col] = sum0;
    partial_sumsq[out_base + (size_t)pos0 * 128 + col] = sq0;
    partial_sum[out_base + (size_t)pos1 * 128 + col] = sum1;
    partial_sumsq[out_base + (size_t)pos1 * 128 + col] = sq1;
    partial_sum[out_base + (size_t)pos2 * 128 + col] = sum2;
    partial_sumsq[out_base + (size_t)pos2 * 128 + col] = sq2;
    partial_sum[out_base + (size_t)pos3 * 128 + col] = sum3;
    partial_sumsq[out_base + (size_t)pos3 * 128 + col] = sq3;
}

''' + corrected_fourier_sum_source(128) + "\n"


class CustomFFT128(CustomFFTBase):
    """Custom 128x128 IFFT kernels for resident ROI SSB calibration."""

    def __init__(self) -> None:
        super().__init__(
            size=128,
            cuda_code=build_cuda_code(128, _TWIDDLE_DECL, _FFT128_KERNELS),
            kernel_names=(
                "ifft128_rows_fused_pk_t32_mr8_packed",
                "ifft128_cols_t32_mr8",
                "ifft128_cols_accumulate_t32_mr8",
            ),
            twiddle_name="TWIDDLE_128",
            rows_block=(32, 8, 1),
            rows_grid_y=16,
            cols_block=(32, 8, 1),
            cols_grid_y=16,
        )


@lru_cache(maxsize=1)
def get_custom_fft_128() -> CustomFFT128:
    return CustomFFT128()
