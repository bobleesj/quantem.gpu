"""The complex SSB object on Metal: corrected spectra summed over bright-field pixels, then one inverse FFT.

The inverse FFT is linear, so the mean over bright-field pixels of the per-pixel object equals the inverse FFT of the
mean corrected spectrum; summing in Fourier space avoids one inverse FFT per pixel.
"""

import math
from functools import lru_cache

import numpy as np

from quantem.gpu.ssb.mps.hardware import default_object_redraw_threadgroup, require_mlx
from quantem.gpu.ssb.mps.prepared import PreparedMpsSSB, pk_from_prepared


@lru_cache(maxsize=16)
def _object_fourier_sum_dynamic_kernel(
    num_bf: int,
    logical_num_bf: int,
    chunk_bf: int,
    ny: int,
    nx: int,
    gqk_cols: int,
    sparse_storage: bool,
):
    mx = require_mlx()
    groups = (int(num_bf) + int(chunk_bf) - 1) // int(chunk_bf)
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint LOGICAL_NUM_BF = {int(logical_num_bf)};
        constexpr uint CHUNK = {int(chunk_bf)};
        constexpr uint GROUPS = {int(groups)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = NY * GQK_COLS;
        constexpr bool SPARSE_STORAGE = {str(bool(sparse_storage)).lower()};
        uint total = GROUPS * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint group = elem / PLANE;
        uint pixel = elem - group * PLANE;
        uint row = pixel / NX;
        uint col = pixel - row * NX;

        float c10v = params[0];
        float c12v = params[1];
        float cos2v = params[2];
        float sin2v = params[3];
        float factor = params[4];
        float dc_r = params[5];
        float dc_i = params[6];
        float wavelength = params[7];
        float semiangle = params[8];
        float ang_y = params[9];
        float ang_x = params[10];
        float qxv = q_row[row];
        float qyv = q_col[col];
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        uint group_start = group * CHUNK;

        if (pixel == 0) {{
            if (LOGICAL_NUM_BF == NUM_BF) {{
                uint remaining = NUM_BF > group_start ? NUM_BF - group_start : 0;
                uint valid = remaining < CHUNK ? remaining : CHUNK;
                partial[elem].real = dc_r * float(valid);
                partial[elem].imag = dc_i * float(valid);
            }} else {{
                partial[elem].real = group == 0u ? dc_r * float(LOGICAL_NUM_BF) : 0.0f;
                partial[elem].imag = group == 0u ? dc_i * float(LOGICAL_NUM_BF) : 0.0f;
            }}
            return;
        }}

        for (uint local = 0; local < CHUNK; ++local) {{
            uint bf = group_start + local;
            if (bf >= NUM_BF) {{
                continue;
            }}

            uint stored_bf = bf;
            if (SPARSE_STORAGE) {{
                int mapped_bf = storage_map[bf];
                if (mapped_bf < 0) {{
                    continue;
                }}
                stored_bf = uint(mapped_bf);
            }}

            float kxv = kx[stored_bf];
            float kyv = ky[stored_bf];

            float dx = qxv - kxv;
            float dy = qyv - kyv;
            float dx2 = dx * dx;
            float dy2 = dy * dy;
            float r2 = dx2 + dy2;
            float r = metal::sqrt(r2);
            float alpha = r * wavelength;
            float alpha2_m = alpha * alpha;
            float inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
            float cos2_m = (dx2 - dy2) * inv_r2;
            float sin2_m = 2.0f * dx * dy * inv_r2;
            float denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
            float inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
            float denom = metal::sqrt(denom_num2) * inv_r;
            float edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
            float ap_m = metal::clamp(edge, 0.0f, 1.0f);

            dx = qxv + kxv;
            dy = qyv + kyv;
            dx2 = dx * dx;
            dy2 = dy * dy;
            r2 = dx2 + dy2;
            r = metal::sqrt(r2);
            alpha = r * wavelength;
            float alpha2_p = alpha * alpha;
            inv_r2 = r2 > 1.0e-30f ? 1.0f / r2 : 0.0f;
            float cos2_p = (dx2 - dy2) * inv_r2;
            float sin2_p = 2.0f * dx * dy * inv_r2;
            denom_num2 = (dx * ang_y) * (dx * ang_y) + (dy * ang_x) * (dy * ang_x);
            inv_r = r > 1.0e-15f ? 1.0f / r : 0.0f;
            denom = metal::sqrt(denom_num2) * inv_r;
            edge = denom > 1.0e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
            float ap_p = metal::clamp(edge, 0.0f, 1.0f);

            if (ap_m <= 0.0f && ap_p <= 0.0f) {{
                continue;
            }}

            auto pkz = pk[stored_bf];
            float pkr = pkz.real;
            float pki = pkz.imag;

            float chi_m = factor * alpha2_m * (c12v * (cos2_m * cos2v + sin2_m * sin2v) + c10v);
            float cos_chi_m;
            float sin_chi_m = metal::fast::sincos(chi_m, cos_chi_m);
            float pmr = ap_m * cos_chi_m;
            float pmi = -ap_m * sin_chi_m;
            float chi_p = factor * alpha2_p * (c12v * (cos2_p * cos2v + sin2_p * sin2v) + c10v);
            float cos_chi_p;
            float sin_chi_p = metal::fast::sincos(chi_p, cos_chi_p);
            float ppr = ap_p * cos_chi_p;
            float ppi = -ap_p * sin_chi_p;

            float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
            float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
            float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
            float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
            float conj_gamma_r = gamma_r * inv_mag;
            float conj_gamma_i = -gamma_i * inv_mag;

            size_t g_idx;
            bool mirror = false;
            if (GQK_COLS == NX) {{
                g_idx = (size_t)stored_bf * (size_t)PLANE + (size_t)pixel;
            }} else if (col <= NX / 2) {{
                g_idx = (size_t)stored_bf * (size_t)GQK_PLANE
                    + (size_t)row * (size_t)GQK_COLS
                    + (size_t)col;
            }} else {{
                uint mirror_row = row == 0 ? 0 : NY - row;
                uint mirror_col = NX - col;
                g_idx = (size_t)stored_bf * (size_t)GQK_PLANE
                    + (size_t)mirror_row * (size_t)GQK_COLS
                    + (size_t)mirror_col;
                mirror = true;
            }}
            auto gz = g[g_idx];
            if (mirror) {{
                gz.imag = -gz.imag;
            }}
            sum_r += gz.real * conj_gamma_r - gz.imag * conj_gamma_i;
            sum_i += gz.real * conj_gamma_i + gz.imag * conj_gamma_r;
        }}
        partial[elem].real = sum_r;
        partial[elem].imag = sum_i;
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_object_fourier_sum_dyn_fast_sincos_b{int(chunk_bf)}_n{int(num_bf)}_"
            f"logical{int(logical_num_bf)}_"
            f"sparse{int(bool(sparse_storage))}_"
            f"{int(ny)}_{int(nx)}_g{int(gqk_cols)}"
        ),
        input_names=[
            "g", "q_row", "q_col", "kx", "ky", "pk", "params", "storage_map"
        ],
        output_names=["partial"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def object_fourier_sum_dynamic(
    prepared: PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    chunk_bf: int,
    threadgroup_size: int | None = None,
):
    """Exact object wave using BF-summed Fourier-domain correction on MPS."""
    mx = prepared.mx
    ny, nx = prepared.scan_shape
    chunk_bf = max(1, int(chunk_bf))
    if threadgroup_size is None:
        threadgroup_size = default_object_redraw_threadgroup(prepared.scan_shape)
    threadgroup_size = max(1, int(threadgroup_size))
    kernel_num_bf, sparse_storage = _object_redraw_storage_topology(prepared)
    groups = (kernel_num_bf + chunk_bf - 1) // chunk_bf
    kernel = _object_fourier_sum_dynamic_kernel(
        kernel_num_bf,
        int(prepared.num_bf),
        chunk_bf,
        int(ny),
        int(nx),
        int(prepared.g_qk.shape[-1]),
        sparse_storage,
    )
    params = mx.array(
        [
            float(C10),
            float(C12),
            math.cos(2.0 * float(phi12)),
            math.sin(2.0 * float(phi12)),
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    pk = pk_from_prepared(prepared, C10=C10, C12=C12, phi12=phi12)
    if sparse_storage:
        storage_map_np = np.full(prepared.num_bf, -1, dtype=np.int32)
        storage_map_np[prepared.bf_storage_indices_np] = np.arange(
            int(prepared.g_qk.shape[0]), dtype=np.int32
        )
        storage_map = mx.array(storage_map_np)
    else:
        storage_map = mx.zeros((1,), dtype=mx.int32)
    partial = kernel(
        inputs=[
            prepared.g_qk,
            prepared.q_row,
            prepared.q_col,
            prepared.kx,
            prepared.ky,
            pk,
            params,
            storage_map,
        ],
        template=[],
        grid=(groups * int(ny) * int(nx), 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(groups, int(ny), int(nx))],
        output_dtypes=[mx.complex64],
    )[0]
    fourier_sum = mx.sum(partial, axis=0) / prepared.num_bf
    object_wave = mx.fft.ifft2(fourier_sum)
    mx.eval(object_wave)
    return object_wave


def _object_redraw_storage_topology(
    prepared: PreparedMpsSSB,
) -> tuple[int, bool]:
    """Use compact active rows directly when preparation already packed them."""
    stored_num_bf = int(prepared.g_qk.shape[0])
    storage_indices = prepared.bf_storage_indices_np
    compact_active = (
        storage_indices is not None
        and stored_num_bf == int(storage_indices.size)
        and stored_num_bf < int(prepared.num_bf)
    )
    if compact_active:
        return stored_num_bf, False
    if storage_indices is not None:
        return int(prepared.num_bf), True
    return stored_num_bf, False
