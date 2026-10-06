"""Custom fixed-size CUDA FFT kernels for SSB (256x256).

Geometry (alpha², cos2phi, sin2phi, aperture) for q-k and q+k vectors is
computed on-the-fly from small 1D arrays (kx_bf, ky_bf, qx_1d, qy_1d) plus
scalars (wavelength, semiangle_rad, ang_y_rad, ang_x_rad). This eliminates
the ~14 GB packed_m/packed_p cache that previously stored precomputed values.
"""

from functools import lru_cache

from quantem.gpu.ssb.cuda.kernels.common import (
    CustomFFTBase,
    build_cuda_code,
    corrected_fourier_sum_source,
)

_TWIDDLE_DECL = '__constant__ float2 TWIDDLE_256[256];'

_FFT256_KERNELS = r'''
__global__ void ifft256_rows_fused_pk_t64_mr8_packed(
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
    if (bf >= num_bf || row >= 256 || tid >= 64) {
        return;
    }
    int base = (bf * 256 + row) * 256;
    int pos0 = tid;
    int pos1 = tid + 64;
    int pos2 = tid + 128;
    int pos3 = tid + 192;
    int idx0 = base + pos0;
    int idx1 = base + pos1;
    int idx2 = base + pos2;
    int idx3 = base + pos3;
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
	            (unsigned int)pos0, 256u, (unsigned int)gqk_cols));
	    float2 res1 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 256u, (unsigned int)gqk_cols));
	    float2 res2 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 256u, (unsigned int)gqk_cols));
	    float2 res3 = gamma_mul_pk_onthefly(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        C10, C12, cos2phi12, sin2phi12, factor,
	        pk_re, pk_im, ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 256u, (unsigned int)gqk_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[8][256];
    float2* srow = s[threadIdx.y];
    srow[bit_reverse4_8((unsigned int)pos0)] = res0;
    srow[bit_reverse4_8((unsigned int)pos1)] = res1;
    srow[bit_reverse4_8((unsigned int)pos2)] = res2;
    srow[bit_reverse4_8((unsigned int)pos3)] = res3;
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
        int tw = j * (256 / m);
        float2 x0 = srow[idx0s];
        float2 x1 = cmul(TWIDDLE_256[tw], srow[idx1s]);
        float2 x2 = cmul(TWIDDLE_256[tw * 2], srow[idx2s]);
        float2 x3 = cmul(TWIDDLE_256[tw * 3], srow[idx3s]);

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

    out[idx0] = srow[pos0];
    out[idx1] = srow[pos1];
    out[idx2] = srow[pos2];
    out[idx3] = srow[pos3];
}

// Full-aberration variant of ifft256_rows_fused_pk_t64_mr8_packed. Same
// structure; gamma_mul_pk_onthefly → gamma_mul_pk_onthefly_full, with
// the host-precomputed Chebyshev coefficient arrays instead of raw
// (mags, angles).  See chi_full docstring in fft_common.py.
__global__ void ifft256_rows_fused_pk_full_t64_mr8_packed(
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
    int row = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || row >= 256 || tid >= 64) {
        return;
    }
    int base = (bf * 256 + row) * 256;
    int pos0 = tid;
    int pos1 = tid + 64;
    int pos2 = tid + 128;
    int pos3 = tid + 192;
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

	    float2 res0 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos0]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos0, 256u, (unsigned int)gqk_cols));
	    float2 res1 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos1]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos1, 256u, (unsigned int)gqk_cols));
	    float2 res2 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos2]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos2, 256u, (unsigned int)gqk_cols));
	    float2 res3 = gamma_mul_pk_onthefly_full(
	        qx, __ldg(&qy_1d[pos3]), kx, ky,
	        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
	        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im,
	        ld_gqk_maybe_herm(
	            G_qk, (unsigned long long)bf, (unsigned int)row,
	            (unsigned int)pos3, 256u, (unsigned int)gqk_cols));

    if (row == 0 && tid == 0) {
        res0 = make_float2(dc_real, dc_imag);
    }

    __shared__ float2 s[8][256];
    float2* srow = s[threadIdx.y];
    srow[bit_reverse4_8((unsigned int)pos0)] = res0;
    srow[bit_reverse4_8((unsigned int)pos1)] = res1;
    srow[bit_reverse4_8((unsigned int)pos2)] = res2;
    srow[bit_reverse4_8((unsigned int)pos3)] = res3;
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
        int tw = j * (256 / m);
        float2 x0 = srow[idx0s];
        float2 x1 = cmul(TWIDDLE_256[tw], srow[idx1s]);
        float2 x2 = cmul(TWIDDLE_256[tw * 2], srow[idx2s]);
        float2 x3 = cmul(TWIDDLE_256[tw * 3], srow[idx3s]);

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

    out[idx0] = srow[pos0];
    out[idx1] = srow[pos1];
    out[idx2] = srow[pos2];
    out[idx3] = srow[pos3];
}

__global__ void ifft256_cols_t64_mr8(float2* __restrict__ data,
                                     int num_bf,
                                     float scale) {
    int bf = blockIdx.z;
    int col = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (bf >= num_bf || col >= 256 || tid >= 64) {
        return;
    }
    int base = bf * 256 * 256 + col;
    __shared__ float2 s[8][256];
    float2* srow = s[threadIdx.y];
    int pos0 = tid;
    int pos1 = tid + 64;
    int pos2 = tid + 128;
    int pos3 = tid + 192;
    srow[bit_reverse4_8((unsigned int)pos0)] = data[base + pos0 * 256];
    srow[bit_reverse4_8((unsigned int)pos1)] = data[base + pos1 * 256];
    srow[bit_reverse4_8((unsigned int)pos2)] = data[base + pos2 * 256];
    srow[bit_reverse4_8((unsigned int)pos3)] = data[base + pos3 * 256];
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
        int tw = j * (256 / m);
        float2 x0 = srow[idx0];
        float2 x1 = cmul(TWIDDLE_256[tw], srow[idx1]);
        float2 x2 = cmul(TWIDDLE_256[tw * 2], srow[idx2]);
        float2 x3 = cmul(TWIDDLE_256[tw * 3], srow[idx3]);

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
    data[base + pos0 * 256] = out0;
    data[base + pos1 * 256] = out1;
    data[base + pos2 * 256] = out2;
    data[base + pos3 * 256] = out3;
}

// Fused col-FFT + phase accumulate for 256.  Same pattern as 512 variant.
__global__ void ifft256_cols_accumulate_t64_mr8(
    const float2* __restrict__ data,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sumsq,
    int num_bf,
    int k_bf
) {
    int group = blockIdx.z;
    int col = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (col >= 256 || tid >= 64) return;

    int bf_start = group * k_bf;
    int bf_end = bf_start + k_bf;
    if (bf_end > num_bf) bf_end = num_bf;

    __shared__ float2 s[8][256];
    float2* srow = s[threadIdx.y];

    int pos0 = tid;
    int pos1 = tid + 64;
    int pos2 = tid + 128;
    int pos3 = tid + 192;
    int rev0 = bit_reverse4_8((unsigned int)pos0);
    int rev1 = bit_reverse4_8((unsigned int)pos1);
    int rev2 = bit_reverse4_8((unsigned int)pos2);
    int rev3 = bit_reverse4_8((unsigned int)pos3);

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    float q0 = 0, q1 = 0, q2 = 0, q3 = 0;

    for (int bf = bf_start; bf < bf_end; ++bf) {
        int base = bf * 256 * 256 + col;
        srow[rev0] = data[base + pos0 * 256];
        srow[rev1] = data[base + pos1 * 256];
        srow[rev2] = data[base + pos2 * 256];
        srow[rev3] = data[base + pos3 * 256];
        __syncthreads();

        for (int m = 4; m <= 256; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter;
            int k = tid / quarter;
            int idx0 = k * m + j;
            int idx1 = idx0 + quarter;
            int idx2 = idx1 + quarter;
            int idx3 = idx2 + quarter;
            int tw = j * (256 / m);
            float2 x0 = srow[idx0];
            float2 x1 = cmul(TWIDDLE_256[tw], srow[idx1]);
            float2 x2 = cmul(TWIDDLE_256[tw * 2], srow[idx2]);
            float2 x3 = cmul(TWIDDLE_256[tw * 3], srow[idx3]);

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

    size_t plane = 256u * 256u;
    size_t out_base = (size_t)group * plane;
    size_t o0 = out_base + (size_t)pos0 * 256 + col;
    size_t o1 = out_base + (size_t)pos1 * 256 + col;
    size_t o2 = out_base + (size_t)pos2 * 256 + col;
    size_t o3 = out_base + (size_t)pos3 * 256 + col;
    partial_sum[o0] = s0; partial_sumsq[o0] = q0;
    partial_sum[o1] = s1; partial_sumsq[o1] = q1;
    partial_sum[o2] = s2; partial_sumsq[o2] = q2;
    partial_sum[o3] = s3; partial_sumsq[o3] = q3;
}

''' + corrected_fourier_sum_source(256) + "\n"


class CustomFFT256(CustomFFTBase):
    """Custom 256x256 IFFT kernels tuned for the fastest SSB path."""

    def __init__(self) -> None:
        super().__init__(
            size=256,
            cuda_code=build_cuda_code(256, _TWIDDLE_DECL, _FFT256_KERNELS),
            kernel_names=(
                "ifft256_rows_fused_pk_t64_mr8_packed",
                "ifft256_cols_t64_mr8",
                "ifft256_cols_accumulate_t64_mr8",
                "ifft256_rows_fused_pk_full_t64_mr8_packed",
            ),
            twiddle_name="TWIDDLE_256",
            rows_block=(64, 8, 1),
            rows_grid_y=32,
            cols_block=(64, 8, 1),
            cols_grid_y=32,
        )


@lru_cache(maxsize=1)
def get_custom_fft_256() -> CustomFFT256:
    return CustomFFT256()
