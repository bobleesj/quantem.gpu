"""Custom fixed-size CUDA FFT kernels for SSB (128x128).

128 = 4^3 x 2: three radix-4 stages plus a final radix-2 stage.  The
optimizer row-sparse path evaluates every row at 128x128, matching the
CUDA-parity row mask pinned for the MPS reference path.
"""

from functools import lru_cache

from .fft_common import CustomFFTBase, build_cuda_code

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

__global__ __launch_bounds__(512, 1)
void ifft128_rows_fused_pk_batch_t32_mr8_transpose_packed_b4(
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
    int row = blockIdx.y * 16 + threadIdx.y;
    int tid = threadIdx.x;
    if (cand0 >= batch || bf >= num_bf || row >= 128 || tid >= 32) return;

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

    size_t base_cache = ((size_t)bf * 128 + row) * 128;
    int pos0 = tid, pos1 = tid + 32, pos2 = tid + 64, pos3 = tid + 96;
    float2 pkv0 = pk[(size_t)cand0 * num_bf + bf];
    float2 pkv1 = has1 ? pk[(size_t)cand1 * num_bf + bf] : make_float2(0.0f, 0.0f);
    float2 pkv2 = has2 ? pk[(size_t)cand2 * num_bf + bf] : make_float2(0.0f, 0.0f);
    float2 pkv3 = has3 ? pk[(size_t)cand3 * num_bf + bf] : make_float2(0.0f, 0.0f);
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
	        (unsigned int)pos0, 128u, (unsigned int)gqk_cols);
	    float2 G1 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos1, 128u, (unsigned int)gqk_cols);
	    float2 G2 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos2, 128u, (unsigned int)gqk_cols);
	    float2 G3 = ld_gqk_maybe_herm(
	        G_qk, (unsigned long long)bf, (unsigned int)row,
	        (unsigned int)pos3, 128u, (unsigned int)gqk_cols);

#define GM(m,p,G,C10v,C12v,cosv,sinv,pkv) gamma_mul_pk_packed_vals(m,p,G,C10v,C12v,cosv,sinv,factor,pkv.x,pkv.y)
    float2 r00 = GM(m0, p0, G0, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r10 = GM(m1, p1, G1, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r20 = GM(m2, p2, G2, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r30 = GM(m3, p3, G3, C10v0, C12v0, cos2v0, sin2v0, pkv0);
    float2 r01 = has1 ? GM(m0, p0, G0, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r11 = has1 ? GM(m1, p1, G1, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r21 = has1 ? GM(m2, p2, G2, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r31 = has1 ? GM(m3, p3, G3, C10v1, C12v1, cos2v1, sin2v1, pkv1) : make_float2(0.0f, 0.0f);
    float2 r02 = has2 ? GM(m0, p0, G0, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r12 = has2 ? GM(m1, p1, G1, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r22 = has2 ? GM(m2, p2, G2, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r32 = has2 ? GM(m3, p3, G3, C10v2, C12v2, cos2v2, sin2v2, pkv2) : make_float2(0.0f, 0.0f);
    float2 r03 = has3 ? GM(m0, p0, G0, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
    float2 r13 = has3 ? GM(m1, p1, G1, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
    float2 r23 = has3 ? GM(m2, p2, G2, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
    float2 r33 = has3 ? GM(m3, p3, G3, C10v3, C12v3, cos2v3, sin2v3, pkv3) : make_float2(0.0f, 0.0f);
#undef GM

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

    extern __shared__ float4 row_tile[];
    float4 (*s0)[128] = (float4 (*)[128])row_tile;
    float4 (*s1)[128] = (float4 (*)[128])(row_tile + 16 * 128);
    float4* srow0 = s0[threadIdx.y];
    float4* srow1 = s1[threadIdx.y];
    srow0[digit_reverse_128((unsigned int)pos0)] = res0a;
    srow0[digit_reverse_128((unsigned int)pos1)] = res1a;
    srow0[digit_reverse_128((unsigned int)pos2)] = res2a;
    srow0[digit_reverse_128((unsigned int)pos3)] = res3a;
    srow1[digit_reverse_128((unsigned int)pos0)] = res0b;
    srow1[digit_reverse_128((unsigned int)pos1)] = res1b;
    srow1[digit_reverse_128((unsigned int)pos2)] = res2b;
    srow1[digit_reverse_128((unsigned int)pos3)] = res3b;
    FFT128_SYNC();

    for (int m = 4; m <= 64; m <<= 2) {
        int quarter = m >> 2;
        int j = tid % quarter;
        int k = tid / quarter;
        int idx0s = k * m + j;
        int idx1s = idx0s + quarter;
        int idx2s = idx1s + quarter;
        int idx3s = idx2s + quarter;
        int tw = j * (128 / m);
        float2 w0 = TWIDDLE_128[tw];
        float2 w1 = TWIDDLE_128[tw * 2];
        float2 w2 = TWIDDLE_128[tw * 3];

        float4 x0 = srow0[idx0s], x1 = cmul2(w0, srow0[idx1s]);
        float4 x2 = cmul2(w1, srow0[idx2s]), x3 = cmul2(w2, srow0[idx3s]);
        float4 t0 = cadd2(x0, x2), t1 = csub2(x0, x2);
        float4 t2 = cadd2(x1, x3), t3 = csub2(x1, x3);
        float4 it3 = cmul_i2(t3);
        srow0[idx0s] = cadd2(t0, t2); srow0[idx1s] = cadd2(t1, it3);
        srow0[idx2s] = csub2(t0, t2); srow0[idx3s] = csub2(t1, it3);

        float4 y0 = srow1[idx0s], y1 = cmul2(w0, srow1[idx1s]);
        float4 y2 = cmul2(w1, srow1[idx2s]), y3 = cmul2(w2, srow1[idx3s]);
        float4 u0 = cadd2(y0, y2), u1 = csub2(y0, y2);
        float4 u2 = cadd2(y1, y3), u3 = csub2(y1, y3);
        float4 iu3 = cmul_i2(u3);
        srow1[idx0s] = cadd2(u0, u2); srow1[idx1s] = cadd2(u1, iu3);
        srow1[idx2s] = csub2(u0, u2); srow1[idx3s] = csub2(u1, iu3);
        FFT128_SYNC();
    }

#define FINAL2(SROW) do { \
    int j0 = tid; int j1 = tid + 32; \
    float4 a0 = SROW[j0], b0 = cmul2(TWIDDLE_128[j0], SROW[j0 + 64]); \
    float4 a1 = SROW[j1], b1 = cmul2(TWIDDLE_128[j1], SROW[j1 + 64]); \
    SROW[j0] = cadd2(a0, b0); SROW[j0 + 64] = csub2(a0, b0); \
    SROW[j1] = cadd2(a1, b1); SROW[j1 + 64] = csub2(a1, b1); \
} while (0)
    FINAL2(srow0);
    FINAL2(srow1);
#undef FINAL2
    FFT128_SYNC();
    __syncthreads();

    int linear = threadIdx.y * 32 + threadIdx.x;
    int row_base = blockIdx.y * 16;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        int elem = linear + n * 512;
        int dst_pos = elem >> 4;
        int dst_row = row_base + (elem & 15);
        float4 va = s0[elem & 15][dst_pos];
        float4 vb = s1[elem & 15][dst_pos];
        size_t base0 = (((size_t)cand0 * num_bf + bf) * 128 + dst_pos) * 128 + dst_row;
        out[base0] = make_float2(va.x, va.y);
        if (has1) {
            size_t base1 = (((size_t)cand1 * num_bf + bf) * 128 + dst_pos) * 128 + dst_row;
            out[base1] = make_float2(va.z, va.w);
        }
        if (has2) {
            size_t base2 = (((size_t)cand2 * num_bf + bf) * 128 + dst_pos) * 128 + dst_row;
            out[base2] = make_float2(vb.x, vb.y);
        }
        if (has3) {
            size_t base3 = (((size_t)cand3 * num_bf + bf) * 128 + dst_pos) * 128 + dst_row;
            out[base3] = make_float2(vb.z, vb.w);
        }
    }
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

__global__ void ifft128_rows_var_g32_t32_mr1_batch(const float2* __restrict__ data,
                                                  float* __restrict__ sum,
                                                  float* __restrict__ sumsq,
                                                  int num_bf,
                                                  int batch,
                                                  float scale,
                                                  int use_partial) {
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (row >= 128 || tid >= 32) return;
    int groups = (num_bf + 31) >> 5;
    int group = blockIdx.z % groups;
    int cand = blockIdx.z / groups;
    if (cand >= batch) return;
    int bf0 = group * 32;
    int pos0 = tid, pos1 = tid + 32, pos2 = tid + 64, pos3 = tid + 96;
    int rev0 = digit_reverse_128((unsigned int)pos0);
    int rev1 = digit_reverse_128((unsigned int)pos1);
    int rev2 = digit_reverse_128((unsigned int)pos2);
    int rev3 = digit_reverse_128((unsigned int)pos3);
    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    float sq0 = 0, sq1 = 0, sq2 = 0, sq3 = 0;
    __shared__ float2 s[128];
    size_t plane = 128u * 128u;
    size_t cand_offset = (size_t)cand * (size_t)num_bf * plane;
    size_t sum_base = use_partial
        ? ((size_t)cand * (size_t)groups + (size_t)group) * plane
        : (size_t)cand * plane;
    for (int i = 0; i < 32; ++i) {
        int bf = bf0 + i;
        size_t base = cand_offset + (size_t)bf * plane + (size_t)row * 128;
        float2 in0 = make_float2(0,0), in1 = make_float2(0,0), in2 = make_float2(0,0), in3 = make_float2(0,0);
        if (bf < num_bf) {
            in0 = data[base + pos0]; in1 = data[base + pos1];
            in2 = data[base + pos2]; in3 = data[base + pos3];
        }
        s[rev0] = in0; s[rev1] = in1; s[rev2] = in2; s[rev3] = in3;
        FFT128_SYNC();
        for (int m = 4; m <= 64; m <<= 2) {
            int quarter = m >> 2;
            int j = tid % quarter, k = tid / quarter;
            int idx0 = k * m + j, idx1 = idx0 + quarter, idx2 = idx1 + quarter, idx3 = idx2 + quarter;
            int tw = j * (128 / m);
            float2 x0 = s[idx0], x1 = cmul(TWIDDLE_128[tw], s[idx1]);
            float2 x2 = cmul(TWIDDLE_128[tw * 2], s[idx2]), x3 = cmul(TWIDDLE_128[tw * 3], s[idx3]);
            float2 t0 = cadd(x0, x2), t1 = csub(x0, x2);
            float2 t2 = cadd(x1, x3), t3 = csub(x1, x3);
            float2 it3 = cmul_i(t3);
            s[idx0] = cadd(t0, t2); s[idx1] = cadd(t1, it3);
            s[idx2] = csub(t0, t2); s[idx3] = csub(t1, it3);
            FFT128_SYNC();
        }
        int j0 = tid, j1 = tid + 32;
        float2 a0 = s[j0], b0 = cmul(TWIDDLE_128[j0], s[j0 + 64]);
        float2 a1 = s[j1], b1 = cmul(TWIDDLE_128[j1], s[j1 + 64]);
        s[j0] = cadd(a0, b0); s[j0 + 64] = csub(a0, b0);
        s[j1] = cadd(a1, b1); s[j1 + 64] = csub(a1, b1);
        FFT128_SYNC();
        if (bf < num_bf) {
            float2 o0 = s[pos0], o1 = s[pos1], o2 = s[pos2], o3 = s[pos3];
            o0.x *= scale; o0.y *= scale; o1.x *= scale; o1.y *= scale;
            o2.x *= scale; o2.y *= scale; o3.x *= scale; o3.y *= scale;
            float p0 = atan2f(o0.y, o0.x), p1 = atan2f(o1.y, o1.x);
            float p2 = atan2f(o2.y, o2.x), p3 = atan2f(o3.y, o3.x);
            sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
            sq0 += p0 * p0; sq1 += p1 * p1; sq2 += p2 * p2; sq3 += p3 * p3;
        }
    }
    size_t o0 = sum_base + (size_t)pos0 * 128 + row;
    size_t o1 = sum_base + (size_t)pos1 * 128 + row;
    size_t o2 = sum_base + (size_t)pos2 * 128 + row;
    size_t o3 = sum_base + (size_t)pos3 * 128 + row;
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

__global__ void ssb128_corrected_fourier_sum_t256(
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
    const unsigned long long plane = 128ull * 128ull;
    int groups = (num_bf + k_bf - 1) / k_bf;
    unsigned long long total = (unsigned long long)groups * plane;
    if (linear >= total) return;

    int group = (int)(linear / plane);
    unsigned int idx = (unsigned int)(linear - (unsigned long long)group * plane);
    int row = idx / 128u;
    int col = idx - (unsigned int)row * 128u;
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
                (unsigned int)col, 128u, (unsigned int)gqk_cols
            ));
        sum_re += v.x;
        sum_im += v.y;
    }
    partial_sum[(unsigned long long)group * plane + idx] = make_float2(sum_re, sum_im);
}
'''


class CustomFFT128(CustomFFTBase):
    """Custom 128x128 IFFT kernels for resident ROI SSB calibration."""

    def __init__(self) -> None:
        super().__init__(
            size=128,
            cuda_code=build_cuda_code(128, _TWIDDLE_DECL, _FFT128_KERNELS),
            kernel_names=(
                "ifft128_rows_fused_pk_t32_mr8_packed",
                "ifft128_rows_fused_pk_batch_t32_mr8_transpose_packed_b4",
                "ifft128_cols_t32_mr8",
                "ifft128_rows_var_g32_t32_mr1_batch",
                "ifft128_cols_accumulate_t32_mr8",
                "ssb128_corrected_fourier_sum_t256",
            ),
            twiddle_name="TWIDDLE_128",
            rows_block=(32, 8, 1),
            rows_grid_y=16,
            batch_block=(32, 16, 1),
            batch_grid_y=8,
            var_block=(32, 1, 1),
            var_grid_y=128,
            cols_block=(32, 8, 1),
            cols_grid_y=16,
            batch_shared_mem=16 * 128 * 2 * 16,
        )
        self._fourier_sum = self._module.get_function("ssb128_corrected_fourier_sum_t256")


@lru_cache(maxsize=1)
def get_custom_fft_128() -> CustomFFT128:
    return CustomFFT128()
