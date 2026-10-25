"""Shared CUDA device functions and base class for SSB FFT kernels.

The CUDA device functions (complex arithmetic, geometry, gamma multiplication)
are identical across all scan sizes. The Python base class provides the shared
dispatch logic for ifft2_inplace_fused_pk and the column-accumulate paths.

Size-specific kernels live in fft128.py, fft256.py, fft512.py and fft1024.py.
"""

import math

import cupy as cp
import numpy as np

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

__device__ __forceinline__ float2 cmul_i(float2 a) {
    return make_float2(-a.y, a.x);
}

__device__ __forceinline__ float2 ld_float2(const float2* ptr, size_t idx) {
    return __ldg(ptr + idx);
}

__device__ __forceinline__ float2 ld_gqk_maybe_herm(
    const float2* ptr,
    unsigned long long bf,
    unsigned int row,
    unsigned int col,
    unsigned int n,
    unsigned int stored_cols
) {
    unsigned long long base = bf * (unsigned long long)n * (unsigned long long)stored_cols;
    if (stored_cols == n || col <= (n >> 1)) {
        return ld_float2(ptr, base + (unsigned long long)row * stored_cols + col);
    }
    unsigned int mirror_row = row == 0u ? 0u : n - row;
    unsigned int mirror_col = n - col;
    float2 z = ld_float2(
        ptr,
        base + (unsigned long long)mirror_row * stored_cols + mirror_col
    );
    return make_float2(z.x, -z.y);
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
    float alpha2 = (r2 * wavelength) * wavelength;

    // cos2phi, sin2phi via algebraic identity (no atan2)
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2phi = (dx2 - dy2) * inv_r2;
    float sin2phi = 2.0f * dx * dy * inv_r2;

    // Soft aperture: clip((semiangle - alpha) / denom + 0.5, 0, 1)
    // denom = sqrt((cos_phi * ang_y)^2 + (sin_phi * ang_x)^2)
    //       = (1/r) * sqrt((dx * ang_y)^2 + (dy * ang_x)^2)
    //
    // Most SSB live-control points are strictly inside the soft aperture.
    // Since denom <= max(ang_y, ang_x), alpha <= semiangle - 0.5*max_ang
    // proves edge >= 1 and therefore aperture == 1 exactly.  Taking this
    // branch skips the second sqrt/div path while falling back to the complete
    // edge expression so the scientific objective is unchanged.
    float max_ang = fmaxf(ang_y_rad, ang_x_rad);
    float inner = (semiangle_rad - 0.5f * max_ang) / wavelength;
    float aperture = 1.0f;
    if (inner <= 0.0f || r2 > inner * inner) {
        float r = sqrtf(r2);
        float alpha = r * wavelength;
        float denom_num2 = fmaf(dx * ang_y_rad, dx * ang_y_rad, dy * ang_x_rad * dy * ang_x_rad);
        float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
        float denom = sqrtf(denom_num2) * inv_r;
        float edge = (denom > 1e-15f) ? ((semiangle_rad - alpha) / denom + 0.5f) : 1.0f;
        aperture = fminf(fmaxf(edge, 0.0f), 1.0f);
    }

    return make_float4(alpha2, cos2phi, sin2phi, aperture);
}

__device__ __forceinline__ float compute_aperture_from_r2(
    float dx, float dy, float r2,
    float wavelength, float semiangle_rad,
    float ang_y_rad, float ang_x_rad,
    float inner2
) {
    if (inner2 > 0.0f && r2 <= inner2) {
        return 1.0f;
    }
    float r = sqrtf(r2);
    float alpha = r * wavelength;
    float denom_num2 = fmaf(dx * ang_y_rad, dx * ang_y_rad, dy * ang_x_rad * dy * ang_x_rad);
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = sqrtf(denom_num2) * inv_r;
    float edge = (denom > 1e-15f) ? ((semiangle_rad - alpha) / denom + 0.5f) : 1.0f;
    return fminf(fmaxf(edge, 0.0f), 1.0f);
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

__device__ __forceinline__ float2 gamma_mul_pk_cartesian_onthefly(
    float qx, float qy,
    float kx, float ky,
    float wavelength, float semiangle_rad, float ang_y_rad, float ang_x_rad,
    float C10,
    float C12,
    float cos2phi12,
    float sin2phi12,
    float factor,
    float phase_scale,
    float inner2,
    float pk_re,
    float pk_im,
    float2 G
) {
    float dx_m = qx - kx;
    float dy_m = qy - ky;
    float dx_p = qx + kx;
    float dy_p = qy + ky;
    float dx2_m = dx_m * dx_m;
    float dy2_m = dy_m * dy_m;
    float dx2_p = dx_p * dx_p;
    float dy2_p = dy_p * dy_p;
    float r2_m = dx2_m + dy2_m;
    float r2_p = dx2_p + dy2_p;
    float aperture_m = compute_aperture_from_r2(
        dx_m, dy_m, r2_m, wavelength, semiangle_rad, ang_y_rad, ang_x_rad, inner2);
    float aperture_p = compute_aperture_from_r2(
        dx_p, dy_p, r2_p, wavelength, semiangle_rad, ang_y_rad, ang_x_rad, inner2);
    float quad_m = fmaf(dx2_m - dy2_m, cos2phi12, (2.0f * dx_m * dy_m) * sin2phi12);
    float quad_p = fmaf(dx2_p - dy2_p, cos2phi12, (2.0f * dx_p * dy_p) * sin2phi12);
    float chi_m = phase_scale * fmaf(C10, r2_m, C12 * quad_m);
    float chi_p = phase_scale * fmaf(C10, r2_p, C12 * quad_p);

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

__device__ __forceinline__ float atan2f_ssb_poly(float y, float x) {
    float ax = fabsf(x);
    float ay = fabsf(y);
    float hi = fmaxf(ax, ay);
    if (hi == 0.0f) {
        return 0.0f;
    }
    float a = fminf(ax, ay) / hi;
    float s = a * a;
    float p = 0.00773044f;
    p = fmaf(p, s, -0.03645544f);
    p = fmaf(p, s, 0.08302083f);
    p = fmaf(p, s, -0.13427427f);
    p = fmaf(p, s, 0.19861783f);
    p = fmaf(p, s, -0.33323847f);
    p = fmaf(p, s, 0.9999984f);
    float r = a * p;
    if (ay > ax) {
        r = 1.5707963267948966f - r;
    }
    if (x < 0.0f) {
        r = 3.1415926535897932f - r;
    }
    return (y < 0.0f) ? -r : r;
}
'''


def build_cuda_code(twiddle_decl: str, kernel_code: str) -> str:
    """Assemble full CUDA module: twiddle constant + device functions + kernels."""
    return f'extern "C" {{\n{twiddle_decl}\n{_DEVICE_FUNCTIONS_CUDA}\n{kernel_code}\n}}'


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
        rows: str,
        cols: str,
        cols_accumulate: str | None,
        twiddle_name: str,
        rows_block: tuple[int, int, int],
        rows_grid_y: int,
        cols_block: tuple[int, int, int],
        cols_grid_y: int,
    ) -> None:
        """Compile one size's module and fetch its kernels: the fused row IFFT, the column IFFT and the
        column-accumulate kernel of ``ifft2_fused_pk_col_accumulate`` (None where a subclass replaces that method); a
        subclass fetches any further kernel it uses."""
        self._size = size
        options = ("--std=c++11", "--use_fast_math", "--maxrregcount=96", "-Xptxas=-dlcm=ca")
        self._module = cp.RawModule(
            code=cuda_code,
            options=options,
            name_expressions=tuple(name for name in (rows, cols, cols_accumulate) if name is not None),
        )
        self._rows_fused_pk = self._module.get_function(rows)
        self._cols = self._module.get_function(cols)
        self._cols_accumulate = None if cols_accumulate is None else self._module.get_function(cols_accumulate)
        self._colvar_group = 32
        self._rows_fused_pk_block = rows_block
        self._rows_fused_pk_grid_y = rows_grid_y
        self._cols_block = cols_block
        self._cols_grid_y = cols_grid_y
        # every radix stage reads exp(2 pi i j / size) from the module's constant table, filled once here
        twiddles = np.exp(2j * math.pi * np.arange(size) / size).astype(np.complex64)
        cp.ndarray((size,), cp.complex64, self._module.get_global(twiddle_name)).set(twiddles)

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

    def _plane_counts(self, data: cp.ndarray, G_qk: cp.ndarray, pk: cp.ndarray) -> tuple[int, int]:
        """Check the corrected-plane stack, ``G_qk`` and the probe against this kernel size; return (num_bright_field_pixels, gqk_cols).

        The raw kernels index fixed ``size x size`` planes without bounds checks, so a wrong dtype or shape would
        read or write outside the arrays instead of failing. ``gqk_cols`` is ``size`` or the Hermitian ``size / 2 + 1``.
        """
        N = self._size
        if data.dtype != cp.complex64 or G_qk.dtype != cp.complex64 or pk.dtype != cp.complex64:
            raise ValueError("Requires complex64 input")
        if data.ndim != 3 or data.shape[1] != N or data.shape[2] != N:
            raise ValueError(f"Expects shape (num_bf, {N}, {N})")
        num_bright_field_pixels = int(data.shape[0])
        if G_qk.ndim != 3 or G_qk.shape[0] != num_bright_field_pixels or G_qk.shape[1] != N:
            raise ValueError(f"G_qk must have shape (num_bf, {N}, {N}) or Hermitian")
        if G_qk.shape[2] not in (N, N // 2 + 1):
            raise ValueError(
                f"G_qk must have {N} columns or Hermitian {N // 2 + 1} columns"
            )
        gqk_cols = int(G_qk.shape[2])
        if pk.shape != (num_bright_field_pixels,):
            raise ValueError("pk must have shape (num_bf,)")
        return num_bright_field_pixels, gqk_cols

    def _rows_two_term(
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
        num_bright_field_pixels: int,
        gqk_cols: int,
    ) -> None:
        """Row pass shared by the full IFFT and the column-accumulate path: correct G_qk for C10/C12, IFFT each row."""
        (kx_bf, ky_bf, qx_1d, qy_1d,
         wavelength, semiangle_rad, ang_y_rad, ang_x_rad) = self._require_geometry(cache)
        grid_rows = (1, self._rows_fused_pk_grid_y, num_bright_field_pixels)
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
                np.int32(num_bright_field_pixels), np.int32(gqk_cols),
            ),
        )

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
        num_bright_field_pixels, gqk_cols = self._plane_counts(data, G_qk, pk)
        self._rows_two_term(data, G_qk, cache, pk, C10, C12, cos2phi12, sin2phi12, factor, dc_value, num_bright_field_pixels, gqk_cols)
        scale = np.float32(1.0 / (N * N))
        grid_cols = (1, self._cols_grid_y, num_bright_field_pixels)
        self._cols(grid_cols, self._cols_block, (data, np.int32(num_bright_field_pixels), scale))

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
        num_bright_field_pixels, gqk_cols = self._plane_counts(data, G_qk, pk)
        n_groups = (num_bright_field_pixels + k_bf - 1) // k_bf
        if partial_sum.shape != (n_groups, N, N) or partial_sumsq.shape != (n_groups, N, N):
            raise ValueError(f"partial buffers must have shape ({n_groups}, {N}, {N})")
        self._rows_two_term(data, G_qk, cache, pk, C10, C12, cos2phi12, sin2phi12, factor, dc_value, num_bright_field_pixels, gqk_cols)
        # Fused col-FFT + accumulate (reads data, writes partial buffers)
        grid_cols = (1, self._cols_grid_y, n_groups)
        self._cols_accumulate(
            grid_cols, self._cols_block,
            (data, partial_sum, partial_sumsq,
             np.int32(num_bright_field_pixels), np.int32(k_bf)),
        )
