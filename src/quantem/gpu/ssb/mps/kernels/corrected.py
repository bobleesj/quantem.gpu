"""Aberration-corrected spectra G(q, k) conj(gamma) on Metal, from cached or on-the-fly probe geometry."""

from functools import lru_cache

from quantem.gpu.ssb.mps.hardware import require_mlx
from quantem.gpu.ssb.mps.prepared import PreparedMpsSSB, pk_batch_from_prepared


@lru_cache(maxsize=16)
def _corrected_kernel(batch: int, chunk: int, ny: int, nx: int, gqk_cols: int):
    mx = require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = NY * GQK_COLS;
        uint total = BATCH * CHUNK * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / (CHUNK * PLANE);
        uint rem = elem - batch * CHUNK * PLANE;
        uint bf = rem / PLANE;
        uint pixel = rem - bf * PLANE;
        uint geom_idx = bf * PLANE + pixel;
        uint row = pixel / NX;
        uint col = pixel - row * NX;

        if (pixel == 0) {{
            corrected[elem].real = scalars[1];
            corrected[elem].imag = scalars[2];
            return;
        }}

        float c10v = c10[batch];
        float c12v = c12[batch];
        float cos2v = cos2phi12[batch];
        float sin2v = sin2phi12[batch];
        float factor = scalars[0];

        float cos_term_k = cos2_k[bf] * cos2v + sin2_k[bf] * sin2v;
        float chi_k = factor * alpha_k2[bf] * (c12v * cos_term_k + c10v);
        float pk_amp = aperture_k[bf];
        float cos_chi_k;
        float sin_chi_k = metal::fast::sincos(chi_k, cos_chi_k);
        float pkr = pk_amp * cos_chi_k;
        float pki = -pk_amp * sin_chi_k;

        float cos_term_m = cos2_m[geom_idx] * cos2v + sin2_m[geom_idx] * sin2v;
        float chi_m = factor * alpha_m2[geom_idx] * (c12v * cos_term_m + c10v);
        float pm_amp = ap_m[geom_idx];
        float cos_chi_m;
        float sin_chi_m = metal::fast::sincos(chi_m, cos_chi_m);
        float pmr = pm_amp * cos_chi_m;
        float pmi = -pm_amp * sin_chi_m;

        float cos_term_p = cos2_p[geom_idx] * cos2v + sin2_p[geom_idx] * sin2v;
        float chi_p = factor * alpha_p2[geom_idx] * (c12v * cos_term_p + c10v);
        float pp_amp = ap_p[geom_idx];
        float cos_chi_p;
        float sin_chi_p = metal::fast::sincos(chi_p, cos_chi_p);
        float ppr = pp_amp * cos_chi_p;
        float ppi = -pp_amp * sin_chi_p;

        float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
        float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
        float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
        float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
        float conj_gamma_r = gamma_r * inv_mag;
        float conj_gamma_i = -gamma_i * inv_mag;

        size_t g_idx;
        bool mirror = false;
        if (GQK_COLS == NX) {{
            g_idx = (size_t)bf * (size_t)PLANE + (size_t)pixel;
        }} else if (col <= NX / 2) {{
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)row * (size_t)GQK_COLS
                + (size_t)col;
        }} else {{
            uint mirror_row = row == 0 ? 0 : NY - row;
            uint mirror_col = NX - col;
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)mirror_row * (size_t)GQK_COLS
                + (size_t)mirror_col;
            mirror = true;
        }}
        auto gz = g[g_idx];
        if (mirror) {{
            gz.imag = -gz.imag;
        }}
        corrected[elem].real = gz.real * conj_gamma_r - gz.imag * conj_gamma_i;
        corrected[elem].imag = gz.real * conj_gamma_i + gz.imag * conj_gamma_r;
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_corrected_fast_sincos_n{int(batch)}_b{int(chunk)}_"
            f"{int(ny)}_{int(nx)}_g{int(gqk_cols)}"
        ),
        input_names=[
            "g",
            "alpha_k2",
            "cos2_k",
            "sin2_k",
            "aperture_k",
            "alpha_m2",
            "cos2_m",
            "sin2_m",
            "ap_m",
            "alpha_p2",
            "cos2_p",
            "sin2_p",
            "ap_p",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
        ],
        output_names=["corrected"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def corrected_from_cached_geometry(
    prepared: PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Fused Metal correction for cached-geometry MPS sparse objectives."""
    mx = prepared.mx
    batch = int(c10.shape[0])
    chunk = int(stop) - int(start)
    ny, nx = prepared.scan_shape
    gqk_cols = int(prepared.g_qk.shape[-1])
    kernel = _corrected_kernel(batch, chunk, int(ny), int(nx), gqk_cols)
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
        ],
        dtype=mx.float32,
    )
    outputs = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.alpha_k2[start:stop],
            prepared.cos2_k[start:stop],
            prepared.sin2_k[start:stop],
            prepared.aperture_k[start:stop],
            prepared.alpha_m2[start:stop],
            prepared.cos2_m[start:stop],
            prepared.sin2_m[start:stop],
            prepared.ap_m[start:stop],
            prepared.alpha_p2[start:stop],
            prepared.cos2_p[start:stop],
            prepared.sin2_p[start:stop],
            prepared.ap_p[start:stop],
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
        ],
        template=[],
        grid=(batch * chunk * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, chunk, int(ny), int(nx))],
        output_dtypes=[mx.complex64],
    )
    return outputs[0]


@lru_cache(maxsize=16)
def _corrected_dynamic_kernel(batch: int, chunk: int, ny: int, nx: int, gqk_cols: int):
    mx = require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = NY * GQK_COLS;
        uint total = BATCH * CHUNK * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / (CHUNK * PLANE);
        uint rem = elem - batch * CHUNK * PLANE;
        uint bf = rem / PLANE;
        uint pixel = rem - bf * PLANE;
        uint row = pixel / NX;
        uint col = pixel - row * NX;

        if (pixel == 0) {{
            corrected[elem].real = scalars[1];
            corrected[elem].imag = scalars[2];
            return;
        }}

        float factor = scalars[0];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float c10v = c10[batch];
        float c12v = c12[batch];
        float cos2v = cos2phi12[batch];
        float sin2v = sin2phi12[batch];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];
        float qyv = q_col[col];
        auto pkz = pk[batch * CHUNK + bf];
        float pkr = pkz.real;
        float pki = pkz.imag;

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
            g_idx = (size_t)bf * (size_t)PLANE + (size_t)pixel;
        }} else if (col <= NX / 2) {{
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)row * (size_t)GQK_COLS
                + (size_t)col;
        }} else {{
            uint mirror_row = row == 0 ? 0 : NY - row;
            uint mirror_col = NX - col;
            g_idx = (size_t)bf * (size_t)GQK_PLANE
                + (size_t)mirror_row * (size_t)GQK_COLS
                + (size_t)mirror_col;
            mirror = true;
        }}
        auto gz = g[g_idx];
        if (mirror) {{
            gz.imag = -gz.imag;
        }}
        corrected[elem].real = gz.real * conj_gamma_r - gz.imag * conj_gamma_i;
        corrected[elem].imag = gz.real * conj_gamma_i + gz.imag * conj_gamma_r;
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_corrected_dyn_pk_fast_sincos_n{int(batch)}_b{int(chunk)}_"
            f"{int(ny)}_{int(nx)}_g{int(gqk_cols)}"
        ),
        input_names=[
            "g",
            "q_row",
            "q_col",
            "kx",
            "ky",
            "pk",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
        ],
        output_names=["corrected"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def corrected_from_dynamic_geometry(
    prepared: PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Fused Metal correction for large-BF MPS paths without cached geometry."""
    mx = prepared.mx
    batch = int(c10.shape[0])
    chunk = int(stop) - int(start)
    ny, nx = prepared.scan_shape
    gqk_cols = int(prepared.g_qk.shape[-1])
    kernel = _corrected_dynamic_kernel(batch, chunk, int(ny), int(nx), gqk_cols)
    scalars = mx.array(
        [
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
    outputs = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk_batch_from_prepared(
                prepared,
                start=start,
                stop=stop,
                c10=c10,
                c12=c12,
                cos2phi12=cos2phi12,
                sin2phi12=sin2phi12,
            ),
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
        ],
        template=[],
        grid=(batch * chunk * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, chunk, int(ny), int(nx))],
        output_dtypes=[mx.complex64],
    )
    return outputs[0]
