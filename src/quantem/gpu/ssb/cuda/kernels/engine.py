"""CUDA kernels of the SSB engine: phase reductions, the probe at each bright-field pixel, and the thick-sample model.

The size-specific inverse-FFT kernels live in the ``fft{128,256,512,1024}`` modules; these are the size-independent
kernels ``SSBEngine`` and ``ThickSample`` launch around them.
"""

import cupy as cp

from quantem.gpu.ssb.higher_order import CHI_SOURCE

# The exact 512 path emits one or two 32-BF partial planes per chunk.
# Merge both moments together without temporary reduction arrays or atomics.
# Keep the pair addition separate from accumulation, as in sum(axis=0) then +=.
accumulate_phase_moments = cp.ElementwiseKernel(
    "raw float32 sums, raw float32 squares, int32 groups, int32 plane, "
    "float32 prior_sum, float32 prior_square",
    "float32 total_sum, float32 total_square",
    """
    float s = sums[i];
    float q = squares[i];
    if (groups == 2) {
        s += sums[plane + i];
        q += squares[plane + i];
    }
    total_sum = prior_sum + s;
    total_square = prior_square + q;
    """,
    "ssb_accumulate_phase_moments",
)

# Mean phase kernel: avoids materializing a full phase buffer.
mean_phase_kernel = cp.RawKernel(r'''
extern "C" __global__
void mean_phase(
    const float2* __restrict__ corrected,
    float* __restrict__ out,
    int num_bf,
    int ny,
    int nx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny * nx;
    if (idx >= total) return;
    int y = idx / nx;
    int x = idx - y * nx;
    size_t base = (size_t)y * (size_t)nx + (size_t)x;
    size_t plane = (size_t)ny * (size_t)nx;
    float sum = 0.0f;
    for (int b = 0; b < num_bf; ++b) {
        size_t off = (size_t)b * plane + base;
        float2 v = corrected[off];
        sum += atan2f(v.y, v.x);
    }
    out[base] = sum / (float)num_bf;
}
''', 'mean_phase')

# Fused kernel: sum + sumsq of per-BF angles in a single pass.
# Used by reconstruct_with_loss to avoid running the correction pipeline twice.
sum_sumsq_phase_kernel = cp.RawKernel(r'''
extern "C" __global__
void sum_sumsq_phase(
    const float2* __restrict__ corrected,
    float* __restrict__ sum_out,
    float* __restrict__ sumsq_out,
    int num_bf,
    int ny,
    int nx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny * nx;
    if (idx >= total) return;
    size_t plane = (size_t)ny * (size_t)nx;
    float s = 0.0f, sq = 0.0f;
    for (int b = 0; b < num_bf; ++b) {
        float2 v = corrected[(size_t)b * plane + (size_t)idx];
        float a = atan2f(v.y, v.x);
        s += a;
        sq += a * a;
    }
    sum_out[idx] = s;
    sumsq_out[idx] = sq;
}
''', 'sum_sumsq_phase')

# Thick-sample SSB correction (sample tilt + thickness), for ThickSample.reconstruct.
# Standard SSB treats the sample as one plane at the probe defocus. For a crystal of thickness t tilted by theta (straight
# columns), the slice at depth z (from mid-depth) sees the defocus C10 + z and sits shifted by z theta. Averaging the two SSB
# terms over depth gives each its own real weight (sinc of the depth-phase rate times t / 2):
#   t1 = P(q - k) conj(P(k)),   rate1 = -pi lambda (|q - k|^2 - |k|^2) - 2 pi q . theta
#   t2 = conj(P(q + k)) P(k),   rate2 = +pi lambda (|q + k|^2 - |k|^2) - 2 pi q . theta
#   gamma = w1 t1 - w2 t2,  w = sinc(rate t / 2)
# with P = aperture exp(-i chi) at the mid-depth aberrations (chi = factor alpha^2 (C10 + C12 cos 2(phi - phi12)), the same
# geometry and soft aperture as compute_geometry). corrected = G conj(gamma / |gamma|); t = 0 gives w = 1 and the standard
# correction exactly. Units: q, k in 1/A; C10, C12, thickness in the engine's C10 unit (A: chi uses factor = pi / lambda[A]).
# Probe geometry (alpha^2, cos 2 phi, sin 2 phi, soft aperture) and the sinc depth weight, shared by the thick kernels.
_THICK_GEOMETRY_SOURCE = r"""
    __device__ float4 thick_geometry(float dx, float dy, float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad) {
        float r2 = dx * dx + dy * dy;
        float alpha2 = r2 * wavelength * wavelength;
        float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
        float cos2phi = (dx * dx - dy * dy) * inv_r2;
        float sin2phi = 2.0f * dx * dy * inv_r2;
        float r = sqrtf(r2);
        float alpha = r * wavelength;
        float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
        float denom = sqrtf(dx * ang_y_rad * dx * ang_y_rad + dy * ang_x_rad * dy * ang_x_rad) * inv_r;
        float edge = (denom > 1e-15f) ? ((semiangle_rad - alpha) / denom + 0.5f) : 1.0f;
        return make_float4(alpha2, cos2phi, sin2phi, fminf(fmaxf(edge, 0.0f), 1.0f));
    }
    __device__ float thick_sinc(float x) { return (fabsf(x) < 1e-6f) ? 1.0f : sinf(x) / x; }
    // thick_geometry with the soft aperture's denominator taken as the constant ang on square detector pixels
    // (ang_y == ang_x, where |(cos phi ang_y, sin phi ang_x)| = ang), which saves a square root and two divisions per
    // probe tilt; alpha^2, cos 2 phi and sin 2 phi are thick_geometry's expressions, so chi rounds the same. The
    // corrections evaluate it per (q, k) pair.
    __device__ __forceinline__ float4 thick_probe_geometry(float dx, float dy, float wavelength, float semiangle_rad,
                                                           float ang_y_rad, float ang_x_rad) {
        if (ang_y_rad != ang_x_rad) return thick_geometry(dx, dy, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float r2 = dx * dx + dy * dy;
        float alpha2 = r2 * wavelength * wavelength;
        float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
        float cos2phi = (dx * dx - dy * dy) * inv_r2;
        float sin2phi = 2.0f * dx * dy * inv_r2;
        float edge = (r2 > 1e-30f) ? fmaf(semiangle_rad - sqrtf(r2) * wavelength, 1.0f / ang_x_rad, 0.5f) : 1.0f;
        return make_float4(alpha2, cos2phi, sin2phi, fminf(fmaxf(edge, 0.0f), 1.0f));
    }
"""

_THICK_CORRECTION_SOURCE = _THICK_GEOMETRY_SOURCE + r"""
// The BF pixel's own probe geometry and chi(k), which depend on k only: the wave sum computes them once per BF pixel
// and block (thick_probe_k) and passes them to thick_correct_k for every frequency.
__device__ float4 thick_probe_k(float kx, float ky, float wavelength, float semiangle_rad, float ang_y_rad,
    float ang_x_rad, float C10, float C12, float cos2phi12, float sin2phi12, float factor, float *chi_k
#ifdef SSB_HIGHER_ORDER
    , const float* coefficients
#endif
    ) {
        float4 g_k = thick_probe_geometry(kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        *chi_k = factor * g_k.x * fmaf(C12, fmaf(g_k.y, cos2phi12, g_k.z * sin2phi12), C10);
#ifdef SSB_HIGHER_ORDER
        *chi_k = ssb_chi_full(kx, ky, wavelength, coefficients);
#endif
        return g_k;
}
__device__ float2 thick_correct_k(float2 G, float qx, float qy, float kx, float ky, float4 g_k, float chi_k,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10, float C12, float cos2phi12, float sin2phi12, float factor,
    float thickness, float theta_r, float theta_c
#ifdef SSB_HIGHER_ORDER
    , const float* coefficients
#endif
    ) {

        float4 g_m = thick_probe_geometry(qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float4 g_p = thick_probe_geometry(qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
#ifdef SSB_WAVE
        // Closed apertures have exactly zero weight. Keep every BF pixel and
        // the soft aperture edge, but omit phases and depth weights at zero.
        if (g_k.w == 0.0f || (g_m.w == 0.0f && g_p.w == 0.0f)) return make_float2(0.0f, 0.0f);
#endif
        float chi_m = factor * g_m.x * fmaf(C12, fmaf(g_m.y, cos2phi12, g_m.z * sin2phi12), C10);
        float chi_p = factor * g_p.x * fmaf(C12, fmaf(g_p.y, cos2phi12, g_p.z * sin2phi12), C10);
#ifdef SSB_HIGHER_ORDER
        // chi(k - q): G(q, k) ~ P(k - q) conj P(k) - P(k) conj P(k + q); odd orders change sign under q - k
        chi_m = ssb_chi_full(kx - qx, ky - qy, wavelength, coefficients);
        chi_p = ssb_chi_full(qx + kx, qy + ky, wavelength, coefficients);
#endif
        float w1 = 1.0f, w2 = 1.0f;
        if (thickness > 0.0f) {
            float shift = 6.283185307f * (qx * theta_r + qy * theta_c);
            float rate1 = -factor * (g_m.x - g_k.x) - shift;
            float rate2 = factor * (g_p.x - g_k.x) - shift;
            w1 = thick_sinc(0.5f * rate1 * thickness);
            w2 = thick_sinc(0.5f * rate2 * thickness);
        }
        // t1 = P(m) conj(P(k)) = a_m a_k exp(-i (chi_m - chi_k)); t2 = conj(P(p)) P(k) = a_p a_k exp(i (chi_p - chi_k))
        float a1 = w1 * g_m.w * g_k.w, a2 = w2 * g_p.w * g_k.w;
        float s1, c1, s2, c2;
        __sincosf(chi_m - chi_k, &s1, &c1);
        __sincosf(chi_p - chi_k, &s2, &c2);
        float g_re = a1 * c1 - a2 * c2;
        float g_im = -a1 * s1 - a2 * s2;
        float mag_sq = g_re * g_re + g_im * g_im;
        float inv_mag = (mag_sq > 1e-16f) ? rsqrtf(mag_sq) : 1e8f;
        g_re *= inv_mag; g_im *= inv_mag;
        float Gr = G.x, Gi = G.y;
        return make_float2(Gr * g_re + Gi * g_im, Gi * g_re - Gr * g_im);

}
__device__ float2 thick_correct(float2 G, float qx, float qy, float kx, float ky,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10, float C12, float cos2phi12, float sin2phi12, float factor,
    float thickness, float theta_r, float theta_c
#ifdef SSB_HIGHER_ORDER
    , const float* coefficients
#endif
    ) {
        float chi_k;
        float4 g_k = thick_probe_k(kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad, C10, C12, cos2phi12,
                                   sin2phi12, factor, &chi_k
#ifdef SSB_HIGHER_ORDER
                                   , coefficients
#endif
                                   );
        return thick_correct_k(G, qx, qy, kx, ky, g_k, chi_k, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
                               C10, C12, cos2phi12, sin2phi12, factor, thickness, theta_r, theta_c
#ifdef SSB_HIGHER_ORDER
                               , coefficients
#endif
                               );
}
"""

thick_correct_kernel = cp.ElementwiseKernel(
    in_params="""
        complex64 G, float32 qx, float32 qy, float32 kx, float32 ky,
        float32 wavelength, float32 semiangle_rad, float32 ang_y_rad, float32 ang_x_rad,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor,
        float32 thickness, float32 theta_r, float32 theta_c
        """,
    out_params="complex64 corrected",
    preamble=_THICK_CORRECTION_SOURCE,
    operation="""
        float2 value = thick_correct(make_float2(G.real(), G.imag()), qx, qy, kx, ky,
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad, C10, C12, cos2phi12, sin2phi12, factor, thickness, theta_r, theta_c);
        corrected = thrust::complex<float>(value.x, value.y);
    """,
    name="thick_correct_kernel",
)

# Serves: every SSB redraw (sliders, previews, run) on the fine grid; tuned on 3x of a 512 x 512 scan with
# about 9300 BF pixels (30 ms with a 14-term fit, 22.5 ms with C10/C12/phi12 only), where thick.py pairs q, -q.
# Sum corrected spectra in bounded BF groups. Read aliases from the native
# spectrum directly, avoiding an expanded (detector, row, col) output volume.
# The float32 corrections are summed with an error-free float32 two-sum (a high word and its rounding error), close to
# float64 accuracy, and written as float64 once per thread: a float64 add per term ran on the FP64 pipe, which these
# GPUs execute at 1/64 of the float32 rate, and bounded the kernel.
_THICK_WAVE_SUM_SOURCE = r"""
#define WAVE_TILE 128   // BF pixels whose own geometry and chi(k) a block holds at a time
__device__ __forceinline__ void two_sum_add(float &high, float &low, float value) {
    float sum = high + value;
    float rounded = sum - high;
    low += (high - (sum - rounded)) + (value - rounded);
    high = sum;
}
extern "C" __global__ void thick_wave_sum(
    const float2* source, const float* qrow, const float* qcol,
    const float* krow, const float* kcol, double2* partial,
    const float* bounds,
    int num_bf, int native_rows, int native_cols, int stored_cols,
    int rows, int cols, int output_cols, int first_bf, int group_size, float2 dc,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10, float C12, float cos2phi12, float sin2phi12, float factor,
    float thickness, float theta_r, float theta_c
#ifdef SSB_HIGHER_ORDER
    , const float* coefficients
#endif
    ) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = rows * output_cols;
    int start = first_bf + blockIdx.y * group_size;
    int stop = min(start + group_size, num_bf);
    double2 sum = make_double2(0.0, 0.0);
    float high_re = 0.0f, low_re = 0.0f, high_im = 0.0f, low_im = 0.0f;
    bool contributes = false;
    float qr = 0.0f, qc = 0.0f;
    size_t offset = 0, stride = (size_t)native_rows * stored_cols;
    bool conjugate = false;
    if (q == 0) {
        sum = make_double2((double)dc.x * (stop-start), (double)dc.y * (stop-start));
    } else if (q < plane) {
        int row = q / output_cols, col = q % output_cols;
        qr = qrow[row]; qc = qcol[col];
        // No overlap of soft probe apertures beyond this radius.
        float support = 2.0f * (semiangle_rad + 0.5f * fmaxf(ang_y_rad, ang_x_rad)) / wavelength;
        // Replicas of the scan mean contain no spatial contrast. The
        // true origin above retains the common real normalization.
        bool dc_alias = row % native_rows == 0 && col % native_cols == 0;
        // If neither shifted soft aperture can reach this BF group's box,
        // every correction in it is exactly zero. Preserve the soft edge with
        // an outward rounding margin; the nonzero arithmetic stays unchanged.
        bool overlap = true;
#ifdef SSB_WAVE_BOUNDS
        int group = first_bf / group_size + blockIdx.y;
        float row_min = bounds[group*4], row_max = bounds[group*4+1];
        float col_min = bounds[group*4+2], col_max = bounds[group*4+3];
        float minus_row = fmaxf(fmaxf(row_min-qr, qr-row_max), 0.0f);
        float minus_col = fmaxf(fmaxf(col_min-qc, qc-col_max), 0.0f);
        float plus_row = fmaxf(fmaxf(row_min+qr, -qr-row_max), 0.0f);
        float plus_col = fmaxf(fmaxf(col_min+qc, -qc-col_max), 0.0f);
        float radius = support * 0.5f * (1.0f + 8.0f * 1.1920928955078125e-7f);
        overlap = fminf(minus_row*minus_row + minus_col*minus_col,
                        plus_row*plus_row + plus_col*plus_col) <= radius*radius;
#endif
        contributes = !dc_alias && qr*qr + qc*qc <= support*support && overlap;
        int sr = row % native_rows, sc = col % native_cols;
        conjugate = stored_cols != native_cols && sc >= stored_cols;
        if (conjugate) { sr = (native_rows-sr) % native_rows; sc = native_cols-sc; }
        offset = (size_t)sr * stored_cols + sc;
    }
    // The BF pixel's own geometry and chi(k) are the same for every frequency: computed once per block and tile.
    __shared__ float4 probe_k[WAVE_TILE];
    __shared__ float chi_k[WAVE_TILE];
    for (int tile = start; tile < stop; tile += WAVE_TILE) {
        int tile_stop = min(tile + WAVE_TILE, stop);
        __syncthreads();
        for (int b = tile + threadIdx.x; b < tile_stop; b += blockDim.x) {
            probe_k[b - tile] = thick_probe_k(krow[b], kcol[b], wavelength, semiangle_rad, ang_y_rad, ang_x_rad, C10,
                                              C12, cos2phi12, sin2phi12, factor, &chi_k[b - tile]
#ifdef SSB_HIGHER_ORDER
                                              , coefficients
#endif
                                              );
        }
        __syncthreads();
        if (!contributes) continue;
        for (int b = tile; b < tile_stop; ++b) {
            float2 g = source[(size_t)b * stride + offset];
            if (conjugate) g.y = -g.y;
            float2 v = thick_correct_k(g, qr, qc, krow[b], kcol[b], probe_k[b - tile], chi_k[b - tile],
                wavelength, semiangle_rad, ang_y_rad, ang_x_rad, C10, C12, cos2phi12, sin2phi12, factor, thickness, theta_r, theta_c
#ifdef SSB_HIGHER_ORDER
                , coefficients
#endif
                );
            two_sum_add(high_re, low_re, v.x);
            two_sum_add(high_im, low_im, v.y);
        }
    }
    if (q >= plane) return;
    sum.x += (double)high_re + (double)low_re;
    sum.y += (double)high_im + (double)low_im;
    partial[(size_t)blockIdx.y * plane + q] = sum;
}
"""
thick_wave_sum_kernel = cp.RawKernel("#define SSB_WAVE\n" + _THICK_CORRECTION_SOURCE + _THICK_WAVE_SUM_SOURCE, "thick_wave_sum")
thick_wave_sum_bounded_kernel = cp.RawKernel(
    "#define SSB_WAVE\n#define SSB_WAVE_BOUNDS\n" + _THICK_CORRECTION_SOURCE + _THICK_WAVE_SUM_SOURCE,
    "thick_wave_sum",
)

# Corrected full-plane spectra of a group of BF pixels for ThickSample.reconstruct (mean_phase, native grid): the
# element-wise thick_correct of each stored value, with the half plane's mirror read in place, G(-q) = conj G(q), and the
# DC replaced by the session's dc value. One launch replaces the full-plane copy (two gathers, a conjugate and a copy)
# and the element-wise correction; the values are the same.
thick_correct_planes_kernel = cp.RawKernel(_THICK_CORRECTION_SOURCE + r"""
extern "C" __global__ void thick_correct_planes(
    const float2* __restrict__ source, const float* __restrict__ qrow, const float* __restrict__ qcol,
    const float* __restrict__ krow, const float* __restrict__ kcol, float2* __restrict__ corrected,
    int rows, int cols, int stored_cols, int first_bf, float2 dc,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10, float C12, float cos2phi12, float sin2phi12, float factor,
    float thickness, float theta_r, float theta_c)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = rows * cols;
    if (q >= plane) return;
    int b = first_bf + blockIdx.y;
    size_t out = (size_t)blockIdx.y * plane + q;
    if (q == 0) { corrected[out] = dc; return; }
    int row = q / cols, col = q - row * cols;
    int sr = row, sc = col;
    bool conjugate = stored_cols != cols && col >= stored_cols;
    if (conjugate) { sr = (rows - row) % rows; sc = cols - col; }
    float2 g = source[(size_t)b * rows * stored_cols + (size_t)sr * stored_cols + sc];
    if (conjugate) g.y = -g.y;
    corrected[out] = thick_correct(g, qrow[row], qcol[col], krow[b], kcol[b], wavelength, semiangle_rad, ang_y_rad,
                                   ang_x_rad, C10, C12, cos2phi12, sin2phi12, factor, thickness, theta_r, theta_c);
}
""", "thick_correct_planes")

# Per-pixel sum (and sum of squares) of the phase of a group of BF images, added to the running totals in BF order:
# the angle images are never stored.
thick_phase_moments_kernel = cp.RawKernel(r"""
extern "C" __global__ void thick_phase_moments(const float2* __restrict__ images, float* __restrict__ phase_sum,
                                                float* __restrict__ phase_sumsq, int count, int plane, int squares)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= plane) return;
    float total = 0.0f, total_sq = 0.0f;
    for (int b = 0; b < count; ++b) {
        float2 v = images[(size_t)b * plane + q];
        float angle = atan2f(v.y, v.x);
        total += angle;
        total_sq += angle * angle;
    }
    phase_sum[q] += total;
    if (squares) phase_sumsq[q] += total_sq;
}
""", "thick_phase_moments")


# The same polynomial as Metal. No higher-order coefficients enter fitting.
_HIGHER_CORRECTION_SOURCE = (
    "#define SSB_HIGHER_ORDER\n#define SSB_DEVICE __device__ __forceinline__\n"
    "#define SSB_COEFFICIENTS const float*\n" + CHI_SOURCE + _THICK_CORRECTION_SOURCE
)
thick_wave_sum_full_kernel = cp.RawKernel("#define SSB_WAVE\n" + _HIGHER_CORRECTION_SOURCE + _THICK_WAVE_SUM_SOURCE, "thick_wave_sum")
thick_wave_sum_bounded_full_kernel = cp.RawKernel(
    "#define SSB_WAVE\n#define SSB_WAVE_BOUNDS\n" + _HIGHER_CORRECTION_SOURCE + _THICK_WAVE_SUM_SOURCE,
    "thick_wave_sum",
)
thick_correct_full_kernel = cp.ElementwiseKernel(
    in_params="""
        complex64 G, float32 qx, float32 qy, float32 kx, float32 ky,
        float32 wavelength, float32 semiangle_rad, float32 ang_y_rad, float32 ang_x_rad,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor,
        float32 thickness, float32 theta_r, float32 theta_c, raw float32 coefficients
        """,
    out_params="complex64 corrected",
    preamble=_HIGHER_CORRECTION_SOURCE,
    operation="""
        float2 value = thick_correct(make_float2(G.real(), G.imag()), qx, qy, kx, ky,
            wavelength, semiangle_rad, ang_y_rad, ang_x_rad, C10, C12, cos2phi12, sin2phi12,
            factor, thickness, theta_r, theta_c, &coefficients[0]);
        corrected = thrust::complex<float>(value.x, value.y);
    """,
    name="thick_correct_full_kernel",
)

# Least-squares fit of the SSB model G(q, k) = Psi(q) gamma(q, k) (unnormalised gamma of thick_correct_kernel): per q the best
# Psi explains |sum_k G conj(gamma)|^2 / sum_k |gamma|^2 of the data, so the sum of that over a spatial-frequency band measures how
# much of G the model with these aberrations, tilt and thickness accounts for. Unlike the phase-variance loss (phase-only
# correction, pixels weighted equally) it uses how strongly each pixel carries the signal, which is what tilt changes.
thick_fit_kernel = cp.ElementwiseKernel(
    in_params="""
        complex64 G, float32 qx, float32 qy, float32 kx, float32 ky,
        float32 wavelength, float32 semiangle_rad, float32 ang_y_rad, float32 ang_x_rad,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor,
        float32 thickness, float32 theta_r, float32 theta_c
        """,
    out_params="complex64 projected, float32 weight2",
    preamble=_THICK_GEOMETRY_SOURCE + "    ",
    operation="""
        float4 g_k = thick_geometry(kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float4 g_m = thick_geometry(qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float4 g_p = thick_geometry(qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad);
        float chi_k = factor * g_k.x * fmaf(C12, fmaf(g_k.y, cos2phi12, g_k.z * sin2phi12), C10);
        float chi_m = factor * g_m.x * fmaf(C12, fmaf(g_m.y, cos2phi12, g_m.z * sin2phi12), C10);
        float chi_p = factor * g_p.x * fmaf(C12, fmaf(g_p.y, cos2phi12, g_p.z * sin2phi12), C10);
        float w1 = 1.0f, w2 = 1.0f;
        if (thickness > 0.0f) {
            float shift = 6.283185307f * (qx * theta_r + qy * theta_c);
            float rate1 = -factor * (g_m.x - g_k.x) - shift;
            float rate2 = factor * (g_p.x - g_k.x) - shift;
            w1 = thick_sinc(0.5f * rate1 * thickness);
            w2 = thick_sinc(0.5f * rate2 * thickness);
        }
        // t1 = P(m) conj(P(k)) = a_m a_k exp(-i (chi_m - chi_k)); t2 = conj(P(p)) P(k) = a_p a_k exp(i (chi_p - chi_k))
        float a1 = w1 * g_m.w * g_k.w, a2 = w2 * g_p.w * g_k.w;
        float s1, c1, s2, c2;
        __sincosf(chi_m - chi_k, &s1, &c1);
        __sincosf(chi_p - chi_k, &s2, &c2);
        float g_re = a1 * c1 - a2 * c2;
        float g_im = -a1 * s1 - a2 * s2;
        float Gr = G.real(), Gi = G.imag();
        projected = thrust::complex<float>(Gr * g_re + Gi * g_im, Gi * g_re - Gr * g_im);
        weight2 = g_re * g_re + g_im * g_im;
    """,
    name="thick_fit_kernel",
)


# Batched thick-sample fit (fast path for ThickSample.fit_batch): one thread per band q on the stored half-plane, loop over
# bright-field pixels k, up to THICK_FIT_MAX_BATCH parameter sets per pass so G is read once per batch. Geometry of q-k, q+k, k
# (apertures, alpha^2, 2phi) is computed once per (k, q) and shared by every parameter set; pairs with no overlap are skipped.
# Same model as thick_fit_kernel (and tests/parity/torch_ssb.py): gamma = w1 P(q-k) conj P(k) - w2 conj P(q+k) P(k).
# The kernel is arithmetic-bound (its time grows with the number of parameter sets, not with the G reads), so the work per
# (k, q, parameter set) is kept small: chi is linear in (C10, C12 cos 2 phi12, C12 sin 2 phi12), so chi(q -+ k) - chi(k) is
# a three-term dot product of per-(k, q) geometry differences with per-set coefficients, and the tilt shift 2 pi q.theta
# depends on q and the set only, so it is computed once per thread instead of once per pixel.
THICK_FIT_MAX_BATCH = 8
thick_fit_batch_kernel = cp.RawKernel(r"""
#define MAXB 8
// The chi coefficients of a probe tilt v = (dx, dy): alpha^2 (1, cos 2 phi_v, sin 2 phi_v) = lambda^2 (|v|^2, dx^2 - dy^2,
// 2 dx dy), which needs no angle and no division, and the soft aperture clip((semiangle - alpha) / denom + 0.5, 0, 1),
// denom = |(cos phi_v ang_y, sin phi_v ang_x)| (ang_y, ang_x: detector sampling in rad), which is the constant ang on
// square detector pixels. Same model as thick_geometry; only the rounding differs.
__device__ __forceinline__ float4 tf_geometry(float dx, float dy, float wl, float semiangle, float ang_y, float ang_x,
                                              float inv_ang) {
    float r2 = dx * dx + dy * dy;
    float wl2 = wl * wl;
    float edge = 1.0f;
    if (r2 > 1e-30f) {
        // a correctly rounded |v| (rsqrt's error, times lambda / ang, moves the soft edge)
        float r = sqrtf(r2);
        float excess = semiangle - r * wl;
        edge = inv_ang > 0.0f ? fmaf(excess, inv_ang, 0.5f)
                              : excess * r / sqrtf(dx * ang_y * dx * ang_y + dy * ang_x * dy * ang_x) + 0.5f;
    }
    return make_float4(r2 * wl2, (dx * dx - dy * dy) * wl2, 2.0f * dx * dy * wl2, fminf(fmaxf(edge, 0.0f), 1.0f));
}
__device__ __forceinline__ float tf_sinc(float x) { return (fabsf(x) < 1e-6f) ? 1.0f : __fdividef(__sinf(x), x); }
extern "C" __global__ void __launch_bounds__(256, 4) thick_fit_batch(
    const float2* __restrict__ G, const long long* __restrict__ flat, const float* __restrict__ qxb, const float* __restrict__ qyb,
    const float* __restrict__ kx, const float* __restrict__ ky, const float* __restrict__ trial,
    float2* __restrict__ partial_numer, float* __restrict__ partial_denom, unsigned int* __restrict__ arrivals,
    float2* __restrict__ numer, float* __restrict__ denom,
    int num_bf, long long plane, int n_band, int B, float wl, float semiangle, float ang_y, float ang_x, float factor,
    int k_chunk)
{
    // per parameter set: C10, C12 cos 2 phi12, C12 sin 2 phi12 (the chi coefficients), thickness / 2, theta_r, theta_c
    __shared__ float tp[MAXB * 6];
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float C12 = trial[b * 7 + 1];
        tp[b * 6 + 0] = trial[b * 7 + 0];
        tp[b * 6 + 1] = C12 * trial[b * 7 + 2];
        tp[b * 6 + 2] = C12 * trial[b * 7 + 3];
        tp[b * 6 + 3] = 0.5f * trial[b * 7 + 4];
        tp[b * 6 + 4] = trial[b * 7 + 5];
        tp[b * 6 + 5] = trial[b * 7 + 6];
    }
    // Probe geometry depends only on k. Share it across all q threads in
    // this block, keeping the arithmetic and per-thread accumulation order.
    __shared__ float4 geometry_k[256];  // matches thick_fit_batch's k_chunk
    float inv_ang = ang_y == ang_x ? 1.0f / ang_x : 0.0f;   // 0: rectangular detector pixels, the general aperture
    int first_k = blockIdx.y * k_chunk;
    for (int offset = threadIdx.x; offset < k_chunk; offset += blockDim.x) {
        int k = first_k + offset;
        if (k < num_bf) {
            geometry_k[offset] = tf_geometry(kx[k], ky[k], wl, semiangle, ang_y, ang_x, inv_ang);
        }
    }
    __syncthreads();
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = q < n_band;
    float qx = active ? qxb[q] : 0.0f, qy = active ? qyb[q] : 0.0f;
    long long off = active ? flat[q] : 0;
    // Every per-set array is indexed only inside loops unrolled over the compile-time MAXB (B is uniform across the
    // grid, so the b < B guards never diverge): the arrays stay in registers. A loop to the runtime B indexed them
    // dynamically, which put them in local memory and made the kernel bound by local-memory traffic.
    float nr[MAXB], ni[MAXB], dd[MAXB], shift[MAXB], c10[MAXB], c12c[MAXB], c12s[MAXB], half_t[MAXB];
#pragma unroll
    for (int b = 0; b < MAXB; ++b) {
        nr[b] = 0.0f; ni[b] = 0.0f; dd[b] = 0.0f;
        bool used = b < B;
        c10[b] = used ? tp[b * 6 + 0] : 0.0f; c12c[b] = used ? tp[b * 6 + 1] : 0.0f;
        c12s[b] = used ? tp[b * 6 + 2] : 0.0f; half_t[b] = used ? tp[b * 6 + 3] : 0.0f;
        shift[b] = used ? 6.283185307f * (qx * tp[b * 6 + 4] + qy * tp[b * 6 + 5]) : 0.0f;
    }
    // grid.y splits the bright-field pixels: each block sums its slice into its own partial, and the last block of a q
    // group to finish adds the slices in slice order (atomic additions made the sum, and the tilt fit, change from run
    // to run)
    int k0 = blockIdx.y * k_chunk, k1 = active ? min(num_bf, k0 + k_chunk) : k0;
    for (int k = k0; k < k1; ++k) {
        float kxv = __ldg(kx + k), kyv = __ldg(ky + k);
        float4 gk = geometry_k[k - k0];
        float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x, inv_ang);
        float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x, inv_ang);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;             // no double overlap for this (k, q)
        float2 g = G[(long long)k * plane + off];
        // chi(v) - chi(k) = factor (d0 C10 + d1 C12 cos 2 phi12 + d2 C12 sin 2 phi12), d = alpha^2 (1, cos 2 phi, sin 2 phi)
        float m0 = factor * (gm.x - gk.x), m1 = factor * (gm.y - gk.y), m2 = factor * (gm.z - gk.z);
        float p0 = factor * (gp.x - gk.x), p1 = factor * (gp.y - gk.y), p2 = factor * (gp.z - gk.z);
#pragma unroll
        for (int b = 0; b < MAXB; ++b) {
            if (b >= B) break;
            float s1, c1, sp, cp_;
            __sincosf(fmaf(m2, c12s[b], fmaf(m1, c12c[b], m0 * c10[b])), &s1, &c1);
            __sincosf(fmaf(p2, c12s[b], fmaf(p1, c12c[b], p0 * c10[b])), &sp, &cp_);
            float b1 = a1, b2 = a2;
            if (half_t[b] > 0.0f) {
                b1 *= tf_sinc((-m0 - shift[b]) * half_t[b]);
                b2 *= tf_sinc((p0 - shift[b]) * half_t[b]);
            }
            float gr = b1 * c1 - b2 * cp_;          // gamma = b1 exp(-i(chi_m - chi_k)) - b2 exp(i(chi_p - chi_k))
            float gi = -b1 * s1 - b2 * sp;
            nr[b] += g.x * gr + g.y * gi;          // G conj(gamma)
            ni[b] += g.y * gr - g.x * gi;
            dd[b] += gr * gr + gi * gi;
        }
    }
    if (active) {
#pragma unroll
        for (int b = 0; b < MAXB; ++b) {
            if (b >= B) break;
            long long slot = ((long long)blockIdx.y * B + b) * n_band + q;
            partial_numer[slot] = make_float2(nr[b], ni[b]);
            partial_denom[slot] = dd[b];
        }
    }
    __threadfence();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(arrivals + blockIdx.x, 1u) == gridDim.y - 1;
    __syncthreads();
    if (!last || !active) return;
    for (int b = 0; b < B; ++b) {
        float2 n = make_float2(0.0f, 0.0f);
        float d = 0.0f;
        for (int slice = 0; slice < (int)gridDim.y; ++slice) {
            long long slot = ((long long)slice * B + b) * n_band + q;
            float2 value = __ldcg(partial_numer + slot);
            n.x += value.x; n.y += value.y;
            d += __ldcg(partial_denom + slot);
        }
        numer[(long long)b * n_band + q] = n;
        denom[(long long)b * n_band + q] = d;
    }
}
""", "thick_fit_batch")


# Probe at every bright-field pixel, P(k) = A(k) exp(-i chi(k)), for the fused FFT paths.
pk_kernel = cp.ElementwiseKernel(
    in_params="""
        float32 alpha_k2, float32 cos2phi_k, float32 sin2phi_k, float32 aperture_k,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor
        """,
    out_params="complex64 pk",
    operation="""
        float cos_term_k = __fmaf_rn(cos2phi_k, cos2phi12, sin2phi_k * sin2phi12);
        float chi_k = factor * alpha_k2 * __fmaf_rn(C12, cos_term_k, C10);
        float sin_k, cos_k;
        __sincosf(chi_k, &sin_k, &cos_k);
        pk = thrust::complex<float>(aperture_k * cos_k, -aperture_k * sin_k);
    """,
    name="pk_kernel",
)

# Full-aberration pk kernel.  Computes pk = aperture(k) * exp(-i·χ(k)) for all
# 14 Krivanek aberrations up to 5th order.  Used by SSBEngine.reconstruct_full
# for explorer manual higher-order slider drag.  The three-parameter pk_kernel above is
# NOT modified and continues to serve the optimization hot path.
#
# mags[14] and angles[14] follow the Krivanek order:
#   0: C10, 1: C12/phi12,  2: C21/phi21, 3: C23/phi23,
#   4: C30, 5: C32/phi32,  6: C34/phi34,
#   7: C41/phi41, 8: C43/phi43, 9: C45/phi45,
#   10: C50, 11: C52/phi52, 12: C54/phi54, 13: C56/phi56
#
# Input kx, ky are per-BF-pixel reciprocal-space coordinates (1/Angstrom); aperture_k
# is the precomputed soft-edge mask; wavelength and kfactor=2π/wavelength are
# passed in to avoid recomputation.
# Fast variant using host-precomputed Chebyshev-ready arrays (see
# kernels.common.pack_aberration_coefs).  Replaces 14 cosf per BF pixel with
# ~20 FMAs via direct cos(φ)/sin(φ) from (kx, ky) + Chebyshev recurrence.
pk_kernel_full = cp.ElementwiseKernel(
    in_params="""
        float32 kx, float32 ky, float32 aperture_k,
        float32 wavelength, float32 kfactor,
        raw float32 abr_mag_scaled, raw float32 abr_cm, raw float32 abr_sm
        """,
    out_params="complex64 pk",
    operation="""
        float r2 = kx * kx + ky * ky;
        float inv_r = (r2 > 1e-30f) ? rsqrtf(r2) : 0.0f;
        float r = r2 * inv_r;
        float alpha = wavelength * r;
        float c1 = kx * inv_r;
        float s1 = ky * inv_r;
        float c2 = fmaf(2.0f * c1, c1, -1.0f);
        float s2 = 2.0f * s1 * c1;
        float c3 = fmaf(c1, c2, -s1 * s2);
        float s3 = fmaf(s1, c2,  c1 * s2);
        float c4 = fmaf(c2, c2, -s2 * s2);
        float s4 = 2.0f * s2 * c2;
        float c5 = fmaf(c1, c4, -s1 * s4);
        float s5 = fmaf(s1, c4,  c1 * s4);
        float c6 = fmaf(c2, c4, -s2 * s4);
        float s6 = fmaf(s2, c4,  c2 * s4);

        float a2 = alpha * alpha;
        float a3 = a2 * alpha;
        float a4 = a2 * a2;
        float a5 = a4 * alpha;
        float a6 = a3 * a3;

        float chi = 0.0f;
        chi = fmaf(a2 * abr_mag_scaled[0], 1.0f,
              fmaf(a2 * abr_mag_scaled[1], fmaf(c2, abr_cm[1],  s2 * abr_sm[1]),  chi));
        chi = fmaf(a3 * abr_mag_scaled[2], fmaf(c1, abr_cm[2],  s1 * abr_sm[2]),
              fmaf(a3 * abr_mag_scaled[3], fmaf(c3, abr_cm[3],  s3 * abr_sm[3]),  chi));
        chi = fmaf(a4 * abr_mag_scaled[4], 1.0f,
              fmaf(a4 * abr_mag_scaled[5], fmaf(c2, abr_cm[5],  s2 * abr_sm[5]),
              fmaf(a4 * abr_mag_scaled[6], fmaf(c4, abr_cm[6],  s4 * abr_sm[6]),  chi)));
        chi = fmaf(a5 * abr_mag_scaled[7], fmaf(c1, abr_cm[7],  s1 * abr_sm[7]),
              fmaf(a5 * abr_mag_scaled[8], fmaf(c3, abr_cm[8],  s3 * abr_sm[8]),
              fmaf(a5 * abr_mag_scaled[9], fmaf(c5, abr_cm[9],  s5 * abr_sm[9]),  chi)));
        chi = fmaf(a6 * abr_mag_scaled[10], 1.0f,
              fmaf(a6 * abr_mag_scaled[11], fmaf(c2, abr_cm[11], s2 * abr_sm[11]),
              fmaf(a6 * abr_mag_scaled[12], fmaf(c4, abr_cm[12], s4 * abr_sm[12]),
              fmaf(a6 * abr_mag_scaled[13], fmaf(c6, abr_cm[13], s6 * abr_sm[13]), chi))));
        chi *= kfactor;

        float sin_k, cos_k;
        __sincosf(chi, &sin_k, &cos_k);
        pk = thrust::complex<float>(aperture_k * cos_k, -aperture_k * sin_k);
    """,
    name="pk_kernel_full",
)
