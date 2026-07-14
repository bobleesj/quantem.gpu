"""Custom fixed-size CUDA FFT kernels for SSB (256x256).

Geometry (alpha², cos2phi, sin2phi, aperture) for q-k and q+k vectors is
computed on-the-fly from small 1D arrays (kx_bf, ky_bf, qx_1d, qy_1d) plus
scalars (wavelength, semiangle_rad, ang_y_rad, ang_x_rad). This eliminates
the ~14 GB packed_m/packed_p cache that previously stored precomputed values.
"""

from functools import lru_cache

from .fft_common import CustomFFTBase, build_cuda_code

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
    int num_bf
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
        pk_re, pk_im, ld_float2(G_qk, idx0));
    float2 res1 = gamma_mul_pk_onthefly(
        qx, __ldg(&qy_1d[pos1]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        pk_re, pk_im, ld_float2(G_qk, idx1));
    float2 res2 = gamma_mul_pk_onthefly(
        qx, __ldg(&qy_1d[pos2]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        pk_re, pk_im, ld_float2(G_qk, idx2));
    float2 res3 = gamma_mul_pk_onthefly(
        qx, __ldg(&qy_1d[pos3]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        C10, C12, cos2phi12, sin2phi12, factor,
        pk_re, pk_im, ld_float2(G_qk, idx3));

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
    int num_bf
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
        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im, ld_float2(G_qk, idx0));
    float2 res1 = gamma_mul_pk_onthefly_full(
        qx, __ldg(&qy_1d[pos1]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im, ld_float2(G_qk, idx1));
    float2 res2 = gamma_mul_pk_onthefly_full(
        qx, __ldg(&qy_1d[pos2]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im, ld_float2(G_qk, idx2));
    float2 res3 = gamma_mul_pk_onthefly_full(
        qx, __ldg(&qy_1d[pos3]), kx, ky,
        wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        abr_mag_scaled, abr_cm, abr_sm, pk_re, pk_im, ld_float2(G_qk, idx3));

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

__global__ void ifft256_rows_fused_pk_batch_t64_mr8_transpose_packed_b4(
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
) {
    int idx = blockIdx.z;
    int bf = idx % num_bf;
    int quad = idx / num_bf;
    int cand0 = quad * 4;
    int cand1 = cand0 + 1;
    int cand2 = cand0 + 2;
    int cand3 = cand0 + 3;
    // CRITICAL: stride must be *8, NOT *4. block=(64,4,1) with grid_y=64
    // gives rows 0-3, 8-11, 16-19, ... (128 of 256 rows, interleaved).
    // This interleaved row sampling matches the original precomputed-geometry
    // path. Changing to *4 processes all 256 rows but produces a DIFFERENT
    // variance loss landscape that breaks Optuna convergence (loss doubles
    // from 0.039 to 0.082 on Steph's data, wrong C10 by 150nm).
    // The col+var kernel reads the transposed output including the unwritten
    // rows, and the variance was validated against this specific pattern.
    int row = blockIdx.y * 8 + threadIdx.y;
    int tid = threadIdx.x;
    if (cand0 >= batch || bf >= num_bf || row >= 256 || tid >= 64) {
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
    size_t base_cache = ((size_t)bf * 256 + row) * 256;
    int pos0 = tid;
    int pos1 = tid + 64;
    int pos2 = tid + 128;
    int pos3 = tid + 192;
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
    float2 G0 = ld_float2(G_qk, idx0);
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
    float2 G1 = ld_float2(G_qk, idx1);
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
    float2 G2 = ld_float2(G_qk, idx2);
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
    float2 G3 = ld_float2(G_qk, idx3);
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

    __shared__ float4 s0[4][256];
    __shared__ float4 s1[4][256];
    float4* srow0 = s0[threadIdx.y];
    float4* srow1 = s1[threadIdx.y];
    srow0[bit_reverse4_8((unsigned int)pos0)] = res0a;
    srow0[bit_reverse4_8((unsigned int)pos1)] = res1a;
    srow0[bit_reverse4_8((unsigned int)pos2)] = res2a;
    srow0[bit_reverse4_8((unsigned int)pos3)] = res3a;
    srow1[bit_reverse4_8((unsigned int)pos0)] = res0b;
    srow1[bit_reverse4_8((unsigned int)pos1)] = res1b;
    srow1[bit_reverse4_8((unsigned int)pos2)] = res2b;
    srow1[bit_reverse4_8((unsigned int)pos3)] = res3b;
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
        float2 w0 = TWIDDLE_256[tw];
        float2 w1 = TWIDDLE_256[tw * 2];
        float2 w2 = TWIDDLE_256[tw * 3];

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

    float4 out0a = srow0[pos0];
    float4 out1a = srow0[pos1];
    float4 out2a = srow0[pos2];
    float4 out3a = srow0[pos3];
    size_t out_idx00 = (((size_t)cand0 * (size_t)num_bf + bf) * 256 + pos0) * 256 + row;
    size_t out_idx01 = (((size_t)cand0 * (size_t)num_bf + bf) * 256 + pos1) * 256 + row;
    size_t out_idx02 = (((size_t)cand0 * (size_t)num_bf + bf) * 256 + pos2) * 256 + row;
    size_t out_idx03 = (((size_t)cand0 * (size_t)num_bf + bf) * 256 + pos3) * 256 + row;
    out[out_idx00] = make_float2(out0a.x, out0a.y);
    out[out_idx01] = make_float2(out1a.x, out1a.y);
    out[out_idx02] = make_float2(out2a.x, out2a.y);
    out[out_idx03] = make_float2(out3a.x, out3a.y);
    if (has1) {
        size_t out_idx10 = (((size_t)cand1 * (size_t)num_bf + bf) * 256 + pos0) * 256 + row;
        size_t out_idx11 = (((size_t)cand1 * (size_t)num_bf + bf) * 256 + pos1) * 256 + row;
        size_t out_idx12 = (((size_t)cand1 * (size_t)num_bf + bf) * 256 + pos2) * 256 + row;
        size_t out_idx13 = (((size_t)cand1 * (size_t)num_bf + bf) * 256 + pos3) * 256 + row;
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
            size_t out_idx20 = (((size_t)cand2 * (size_t)num_bf + bf) * 256 + pos0) * 256 + row;
            size_t out_idx21 = (((size_t)cand2 * (size_t)num_bf + bf) * 256 + pos1) * 256 + row;
            size_t out_idx22 = (((size_t)cand2 * (size_t)num_bf + bf) * 256 + pos2) * 256 + row;
            size_t out_idx23 = (((size_t)cand2 * (size_t)num_bf + bf) * 256 + pos3) * 256 + row;
            out[out_idx20] = make_float2(out0b.x, out0b.y);
            out[out_idx21] = make_float2(out1b.x, out1b.y);
            out[out_idx22] = make_float2(out2b.x, out2b.y);
            out[out_idx23] = make_float2(out3b.x, out3b.y);
        }
        if (has3) {
            size_t out_idx30 = (((size_t)cand3 * (size_t)num_bf + bf) * 256 + pos0) * 256 + row;
            size_t out_idx31 = (((size_t)cand3 * (size_t)num_bf + bf) * 256 + pos1) * 256 + row;
            size_t out_idx32 = (((size_t)cand3 * (size_t)num_bf + bf) * 256 + pos2) * 256 + row;
            size_t out_idx33 = (((size_t)cand3 * (size_t)num_bf + bf) * 256 + pos3) * 256 + row;
            out[out_idx30] = make_float2(out0b.z, out0b.w);
            out[out_idx31] = make_float2(out1b.z, out1b.w);
            out[out_idx32] = make_float2(out2b.z, out2b.w);
            out[out_idx33] = make_float2(out3b.z, out3b.w);
        }
    }
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

__global__ void ifft256_rows_var_g32_t64_mr1_batch(const float2* __restrict__ data,
                                                  float* __restrict__ sum,
                                                  float* __restrict__ sumsq,
                                                  int num_bf,
                                                  int batch,
                                                  float scale,
                                                  int use_partial) {
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (row >= 256 || tid >= 64) return;
    int groups = (num_bf + 31) >> 5;
    int group = blockIdx.z % groups;
    int cand = blockIdx.z / groups;
    if (cand >= batch) return;
    int bf0 = group * 32;
    int pos0 = tid, pos1 = tid + 64, pos2 = tid + 128, pos3 = tid + 192;
    int rev0 = bit_reverse4_8((unsigned int)pos0);
    int rev1 = bit_reverse4_8((unsigned int)pos1);
    int rev2 = bit_reverse4_8((unsigned int)pos2);
    int rev3 = bit_reverse4_8((unsigned int)pos3);
    float sum0=0, sum1=0, sum2=0, sum3=0;
    float sumsq0=0, sumsq1=0, sumsq2=0, sumsq3=0;
    __shared__ float2 s[256];
    size_t plane = 256u * 256u;
    size_t cand_offset = (size_t)cand * (size_t)num_bf * plane;
    size_t sum_base = use_partial
        ? ((size_t)cand * (size_t)groups + (size_t)group) * plane
        : (size_t)cand * plane;

    for (int i = 0; i < 32; ++i) {
        int bf = bf0 + i;
        size_t base = cand_offset + (size_t)bf * plane + (size_t)row * 256;
        float2 in0={0,0}, in1={0,0}, in2={0,0}, in3={0,0};
        if (bf < num_bf) {
            in0 = data[base + (size_t)pos0];
            in1 = data[base + (size_t)pos1];
            in2 = data[base + (size_t)pos2];
            in3 = data[base + (size_t)pos3];
        }
        s[rev0] = in0; s[rev1] = in1; s[rev2] = in2; s[rev3] = in3;
        __syncthreads();
        for (int m = 4; m <= 256; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter, k = tid / quarter;
            int idx0 = k*m+j, idx1 = idx0+quarter, idx2 = idx1+quarter, idx3 = idx2+quarter;
            int tw = j * (256 / m);
            float2 x0=s[idx0], x1=cmul(TWIDDLE_256[tw],s[idx1]);
            float2 x2=cmul(TWIDDLE_256[tw*2],s[idx2]), x3=cmul(TWIDDLE_256[tw*3],s[idx3]);
            float2 t0=cadd(x0,x2), t1=csub(x0,x2), t2=cadd(x1,x3), t3=csub(x1,x3);
            float2 it3=cmul_i(t3);
            s[idx0]=cadd(t0,t2); s[idx1]=cadd(t1,it3);
            s[idx2]=csub(t0,t2); s[idx3]=csub(t1,it3);
            __syncthreads();
        }
        if (bf < num_bf) {
            float2 o0=s[pos0],o1=s[pos1],o2=s[pos2],o3=s[pos3];
            float p0=atan2f(o0.y,o0.x),p1=atan2f(o1.y,o1.x);
            float p2=atan2f(o2.y,o2.x),p3=atan2f(o3.y,o3.x);
            sum0+=p0;sum1+=p1;sum2+=p2;sum3+=p3;
            sumsq0+=p0*p0;sumsq1+=p1*p1;sumsq2+=p2*p2;sumsq3+=p3*p3;
        }
        __syncthreads();
    }
    size_t o0=sum_base+(size_t)pos0*256+row;
    size_t o1=sum_base+(size_t)pos1*256+row;
    size_t o2=sum_base+(size_t)pos2*256+row;
    size_t o3=sum_base+(size_t)pos3*256+row;
    if (use_partial) {
        sum[o0]=sum0;sumsq[o0]=sumsq0;
        sum[o1]=sum1;sumsq[o1]=sumsq1;
        sum[o2]=sum2;sumsq[o2]=sumsq2;
        sum[o3]=sum3;sumsq[o3]=sumsq3;
    } else {
        atomicAdd(&sum[o0],sum0);atomicAdd(&sumsq[o0],sumsq0);
        atomicAdd(&sum[o1],sum1);atomicAdd(&sumsq[o1],sumsq1);
        atomicAdd(&sum[o2],sum2);atomicAdd(&sumsq[o2],sumsq2);
        atomicAdd(&sum[o3],sum3);atomicAdd(&sumsq[o3],sumsq3);
    }
}
'''


class CustomFFT256(CustomFFTBase):
    """Custom 256x256 IFFT kernels tuned for the fastest SSB path."""

    def __init__(self) -> None:
        super().__init__(
            size=256,
            cuda_code=build_cuda_code(256, _TWIDDLE_DECL, _FFT256_KERNELS),
            kernel_names=(
                "ifft256_rows_fused_pk_t64_mr8_packed",
                "ifft256_rows_fused_pk_batch_t64_mr8_transpose_packed_b4",
                "ifft256_cols_t64_mr8",
                "ifft256_rows_var_g32_t64_mr1_batch",
                "ifft256_cols_accumulate_t64_mr8",
                "ifft256_rows_fused_pk_full_t64_mr8_packed",
            ),
            twiddle_name="TWIDDLE_256",
            rows_block=(64, 8, 1),
            rows_grid_y=32,
            batch_block=(64, 4, 1),
            batch_grid_y=64,
            var_block=(64, 1, 1),
            var_grid_y=256,
            cols_block=(64, 8, 1),
            cols_grid_y=32,
        )


@lru_cache(maxsize=1)
def get_custom_fft_256() -> CustomFFT256:
    return CustomFFT256()
