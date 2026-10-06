"""Row inverse FFT of the aberration-corrected spectra, with the probe geometry evaluated on the fly.

Each kernel corrects G(q, k) for the probe at q - k, q + k and k and runs the row transform in threadgroup memory; the
column stage follows in ``phase_columns`` or ``phase_columns_512``.
"""

from functools import lru_cache

from quantem.gpu.ssb.mps.hardware import require_mlx
from quantem.gpu.ssb.mps.kernels.radix import small_fft_macros, twiddle_512, twiddle_n
from quantem.gpu.ssb.mps.prepared import PreparedMpsSSB, pk_batch_from_prepared


@lru_cache(maxsize=32)
def _row_ifft512_dynamic_kernel(
    batch: int,
    chunk: int,
    gqk_cols: int,
    tiled_output: bool = False,
    storage_chunk: int | None = None,
):
    mx = require_mlx()
    storage_chunk = int(chunk if storage_chunk is None else storage_chunk)
    storage_define = ""
    storage_name = ""
    row_ifft_store = """
        size_t base = (
            ((size_t)output_batch * (size_t)CHUNK + (size_t)bf) * (size_t)PLANE
            + (size_t)row * 512u
        );
        row_ifft[base + tid].real = srow[tid].x;
        row_ifft[base + tid].imag = srow[tid].y;
        row_ifft[base + tid + 64u].real = srow[tid + 64u].x;
        row_ifft[base + tid + 64u].imag = srow[tid + 64u].y;
        row_ifft[base + tid + 128u].real = srow[tid + 128u].x;
        row_ifft[base + tid + 128u].imag = srow[tid + 128u].y;
        row_ifft[base + tid + 192u].real = srow[tid + 192u].x;
        row_ifft[base + tid + 192u].imag = srow[tid + 192u].y;
        row_ifft[base + tid + 256u].real = srow[tid + 256u].x;
        row_ifft[base + tid + 256u].imag = srow[tid + 256u].y;
        row_ifft[base + tid + 320u].real = srow[tid + 320u].x;
        row_ifft[base + tid + 320u].imag = srow[tid + 320u].y;
        row_ifft[base + tid + 384u].real = srow[tid + 384u].x;
        row_ifft[base + tid + 384u].imag = srow[tid + 384u].y;
        row_ifft[base + tid + 448u].real = srow[tid + 448u].x;
        row_ifft[base + tid + 448u].imag = srow[tid + 448u].y;
        """
    if tiled_output:
        row_ifft_store = """
        size_t base = (
            ((size_t)output_batch * (size_t)CHUNK + (size_t)bf)
            * (size_t)PLANE
            + (size_t)(tid >> 3) * 4096u
            + (size_t)row * 8u
            + (tid & 7u)
        );
        row_ifft[base].real = r0.x;
        row_ifft[base].imag = r0.y;
        row_ifft[base + 32768u].real = r1.x;
        row_ifft[base + 32768u].imag = r1.y;
        row_ifft[base + 65536u].real = r2.x;
        row_ifft[base + 65536u].imag = r2.y;
        row_ifft[base + 98304u].real = r3.x;
        row_ifft[base + 98304u].imag = r3.y;
        row_ifft[base + 131072u].real = r4.x;
        row_ifft[base + 131072u].imag = r4.y;
        row_ifft[base + 163840u].real = r5.x;
        row_ifft[base + 163840u].imag = r5.y;
        row_ifft[base + 196608u].real = r6.x;
        row_ifft[base + 196608u].imag = r6.y;
        row_ifft[base + 229376u].real = r7.x;
        row_ifft[base + 229376u].imag = r7.y;
        """
    if storage_chunk != int(chunk):
        row_ifft_store = row_ifft_store.replace("CHUNK", "STORAGE_CHUNK")
        storage_define = f"constexpr uint STORAGE_CHUNK = {storage_chunk}u;"
        storage_name = f"_s{storage_chunk}"
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        #define OCTREV512(n) ((((n) & 7u) << 6) | (((n) & 56u)) | ((n) >> 6))
        #define W8_1(a) float2(0.70710678118654752f * ((a).x - (a).y), 0.70710678118654752f * ((a).x + (a).y))
        #define W8_3(a) float2(0.70710678118654752f * (-(a).x - (a).y), 0.70710678118654752f * ((a).x - (a).y))
        #define RADIX8(x0,x1,x2,x3,x4,x5,x6,x7) {{ \
            float2 a0=(x0), a1=(x4), a2=(x2), a3=(x6); \
            float2 a4=(x1), a5=(x5), a6=(x3), a7=(x7); \
            float2 t0=CADD(a0,a1), t1=CSUB(a0,a1); \
            float2 t2=CADD(a2,a3), t3=CSUB(a2,a3); \
            float2 t4=CADD(a4,a5), t5=CSUB(a4,a5); \
            float2 t6=CADD(a6,a7), t7=CSUB(a6,a7); \
            float2 u0=CADD(t0,t2), u2=CSUB(t0,t2); \
            float2 it3=CMULI(t3), u1=CADD(t1,it3), u3=CSUB(t1,it3); \
            float2 u4=CADD(t4,t6), u6=CSUB(t4,t6); \
            float2 it7=CMULI(t7), u5=CADD(t5,it7), u7=CSUB(t5,it7); \
            float2 w1u5=W8_1(u5), w3u7=W8_3(u7), iu6=CMULI(u6); \
            (x0)=CADD(u0,u4); (x4)=CSUB(u0,u4); \
            (x1)=CADD(u1,w1u5); (x5)=CSUB(u1,w1u5); \
            (x2)=CADD(u2,iu6); (x6)=CSUB(u2,iu6); \
            (x3)=CADD(u3,w3u7); (x7)=CSUB(u3,w3u7); \
        }}

        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        {storage_define}
        constexpr uint NX = 512u;
        constexpr uint PLANE = 512u * 512u;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = 512u * GQK_COLS;
        constexpr uint FUSED_CANDIDATES = BATCH == 4u ? 4u : (BATCH == 2u ? 2u : 1u);
        constexpr uint ROWS_PER_GROUP = BATCH == 2u ? 4u : (BATCH >= 4u ? 2u : 4u);
        constexpr bool FUSE_CANDIDATES = FUSED_CANDIDATES > 1u;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        uint batch = FUSE_CANDIDATES ? 0u : z / CHUNK;
        uint bf = FUSE_CANDIDATES ? z : z - batch * CHUNK;
        if (tid >= 64u || local_row >= ROWS_PER_GROUP || row >= 512u || batch >= BATCH || bf >= CHUNK) {{
            return;
        }}

        threadgroup float2 shared_rows[ROWS_PER_GROUP][FUSED_CANDIDATES][512];

        float factor = scalars[0];
        float dc_r = scalars[1];
        float dc_i = scalars[2];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];

        uint candidate_begin = FUSE_CANDIDATES ? 0u : batch;
        uint candidate_end = FUSE_CANDIDATES ? FUSED_CANDIDATES : batch + 1u;
        auto first_pk = pk[(size_t)candidate_begin * (size_t)CHUNK + bf];
        if (first_pk.real == 0.0f && first_pk.imag == 0.0f) {{
            return;
        }}
        for (uint lane = 0u; lane < 8u; ++lane) {{
            uint col = tid + lane * 64u;
            if (row == 0u && col == 0u) {{
                for (uint candidate=candidate_begin; candidate<candidate_end; ++candidate) {{
                    uint slot = FUSE_CANDIDATES ? candidate : 0u;
                    shared_rows[local_row][slot][OCTREV512(col)] = float2(dc_r, dc_i);
                }}
            }} else {{
                float qyv = q_col[col];

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

                if (ap_m == 0.0f && ap_p == 0.0f) {{
                    for (uint candidate=candidate_begin; candidate<candidate_end; ++candidate) {{
                        uint slot = FUSE_CANDIDATES ? candidate : 0u;
                        shared_rows[local_row][slot][OCTREV512(col)] = float2(0.0f);
                    }}
                }} else {{
                size_t g_idx;
                bool mirror = false;
                if (GQK_COLS == NX) {{
                    g_idx = (size_t)bf * (size_t)PLANE + (size_t)row * (size_t)NX + (size_t)col;
                }} else if (col <= NX / 2u) {{
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)row * (size_t)GQK_COLS
                        + (size_t)col;
                }} else {{
                    uint mirror_row = row == 0u ? 0u : 512u - row;
                    uint mirror_col = NX - col;
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)mirror_row * (size_t)GQK_COLS
                        + (size_t)mirror_col;
                    mirror = true;
                }}
                auto gz = g[g_idx];
                float gr = gz.real;
                float gi = mirror ? -gz.imag : gz.imag;

                for (uint candidate=candidate_begin; candidate<candidate_end; ++candidate) {{
                    float c10v = c10[candidate];
                    float c12v = c12[candidate];
                    float cos2v = cos2phi12[candidate];
                    float sin2v = sin2phi12[candidate];
                    auto pkz = pk[(size_t)candidate * (size_t)CHUNK + bf];
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

                    float2 corrected = float2(
                        gr * conj_gamma_r - gi * conj_gamma_i,
                        gr * conj_gamma_i + gi * conj_gamma_r
                    );
                    uint slot = FUSE_CANDIDATES ? candidate : 0u;
                    shared_rows[local_row][slot][OCTREV512(col)] = corrected;
                }}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint output_batch=candidate_begin; output_batch<candidate_end; ++output_batch) {{
        uint slot = FUSE_CANDIDATES ? output_batch : 0u;
            threadgroup float2* srow = &shared_rows[local_row][slot][0];
        uint logical=tid*8u;
        float2 r0=srow[logical], r1=srow[logical+1u];
        float2 r2=srow[logical+2u], r3=srow[logical+3u];
        float2 r4=srow[logical+4u], r5=srow[logical+5u];
        float2 r6=srow[logical+6u], r7=srow[logical+7u];
        RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
        srow[logical]=r0; srow[logical+1u]=r1; srow[logical+2u]=r2; srow[logical+3u]=r3;
        srow[logical+4u]=r4; srow[logical+5u]=r5; srow[logical+6u]=r6; srow[logical+7u]=r7;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint s2=tid&7u;
        uint base2=(tid>>3)*64u+s2;
        r0=srow[base2]; r1=CMUL(TW(s2*8u),srow[base2+8u]);
        r2=CMUL(TW(s2*16u),srow[base2+16u]); r3=CMUL(TW(s2*24u),srow[base2+24u]);
        r4=CMUL(TW(s2*32u),srow[base2+32u]); r5=CMUL(TW(s2*40u),srow[base2+40u]);
        r6=CMUL(TW(s2*48u),srow[base2+48u]); r7=CMUL(TW(s2*56u),srow[base2+56u]);
        RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
        srow[base2]=r0; srow[base2+8u]=r1; srow[base2+16u]=r2; srow[base2+24u]=r3;
        srow[base2+32u]=r4; srow[base2+40u]=r5; srow[base2+48u]=r6; srow[base2+56u]=r7;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        r0=srow[tid]; r1=CMUL(TW(tid),srow[tid+64u]);
        r2=CMUL(TW(tid*2u),srow[tid+128u]); r3=CMUL(TW(tid*3u),srow[tid+192u]);
        r4=CMUL(TW(tid*4u),srow[tid+256u]); r5=CMUL(TW(tid*5u),srow[tid+320u]);
        r6=CMUL(TW(tid*6u),srow[tid+384u]); r7=CMUL(TW(tid*7u),srow[tid+448u]);
        RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
        srow[tid]=r0; srow[tid+64u]=r1; srow[tid+128u]=r2; srow[tid+192u]=r3;
        srow[tid+256u]=r4; srow[tid+320u]=r5; srow[tid+384u]=r6; srow[tid+448u]=r7;

        {row_ifft_store}
        }}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef OCTREV512
        #undef W8_1
        #undef W8_3
        #undef RADIX8
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_row_ifft512_dyn_n{int(batch)}_b{int(chunk)}_"
            f"g{int(gqk_cols)}_t{int(tiled_output)}{storage_name}"
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
            "twiddle",
        ],
        output_names=["row_ifft"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def row_ifft512_from_dynamic_geometry(
    prepared: PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
    return_active: bool = False,
    tiled_output: bool = False,
):
    """Fused dynamic correction + 512 row IFFT for exact MPS phase/loss."""
    mx = prepared.mx
    if prepared.scan_shape != (512, 512):
        raise ValueError("Fused MPS row IFFT currently supports only 512x512.")
    if int(c10.shape[0]) != 1:
        raise ValueError("Fused MPS row IFFT currently supports one candidate.")
    chunk = int(stop) - int(start)
    kernel = _row_ifft512_dynamic_kernel(
        1,
        chunk,
        int(prepared.g_qk.shape[-1]),
        bool(tiled_output),
    )
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
    pk = pk_batch_from_prepared(
        prepared,
        start=start,
        stop=stop,
        c10=c10,
        c12=c12,
        cos2phi12=cos2phi12,
        sin2phi12=sin2phi12,
    )
    row_ifft_batch = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            twiddle_512(mx),
        ],
        template=[],
        grid=(64, 512, chunk),
        threadgroup=(64, 4, 1),
        output_shapes=[(1, chunk, 512, 512)],
        output_dtypes=[mx.complex64],
    )[0]
    row_ifft = mx.reshape(row_ifft_batch, (chunk, 512, 512))
    if return_active:
        active_bf = (mx.abs(pk[0]) > 0.0).astype(mx.uint8)
        return row_ifft, active_bf
    return row_ifft


def row_ifft512_batch_from_dynamic_geometry(
    prepared: PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
    return_active: bool = False,
    pk_override=None,
    storage_bf: int | None = None,
):
    """Fused dynamic correction + 512 row IFFT for batched exact MPS loss."""
    mx = prepared.mx
    if prepared.scan_shape != (512, 512):
        raise ValueError("Batched fused MPS row IFFT currently supports only 512x512.")
    batch = int(c10.shape[0])
    chunk = int(stop) - int(start)
    storage_bf = chunk if storage_bf is None else max(chunk, int(storage_bf))
    kernel = _row_ifft512_dynamic_kernel(
        batch,
        chunk,
        int(prepared.g_qk.shape[-1]),
        batch <= 2,
        storage_bf,
    )
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
    pk = pk_override
    if pk is None:
        pk = pk_batch_from_prepared(
            prepared,
            start=start,
            stop=stop,
            c10=c10,
            c12=c12,
            cos2phi12=cos2phi12,
            sin2phi12=sin2phi12,
        )
    row_ifft = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            twiddle_512(mx),
        ],
        template=[],
        grid=(64, 512, chunk if batch in (2, 4) else batch * chunk),
        threadgroup=(64, 4 if batch == 2 else (2 if batch >= 4 else 4), 1),
        output_shapes=[(batch, storage_bf, 512, 512)],
        output_dtypes=[mx.complex64],
    )[0]
    if return_active:
        active_bf = (mx.abs(pk[0]) > 0.0).astype(mx.uint8)
        return row_ifft, active_bf
    return row_ifft


@lru_cache(maxsize=48)
def _row_ifft_small_dynamic_kernel(
    n: int,
    chunk: int,
    gqk_cols: int,
    batch: int = 1,
    rows_per_group: int = 4,
    thick: bool = False,
):
    """Fused SSB correction + row IFFT for 128/256/1024 scans.

    ``thick=True`` builds gamma = w1 t1 - w2 t2 with the thick-sample depth weights w = sinc(rate t / 2) of
    ``thick_sample`` from one extra input ``thick_params`` = (thickness, tilt_row_rad, tilt_col_rad). The thick
    variant forms t1, t2 from the phase differences chi(q -/+ k) - chi(k) with precise sincos (as the reference model
    and ``thick_fit_batch`` do): the depth weights can cancel the two terms, and the normalisation gamma / |gamma| then
    amplifies the ~1e-5 rad error of fast sincos at large defocus. chi(k) is evaluated here with the same expression
    as chi(q -/+ k) so their difference cancels consistently. With ``thick=False`` the generated Metal code is the
    standard kernel's.
    """
    mx = require_mlx()
    n = int(n)
    t = n // 4
    half = n // 2
    define_rev, undef_rev, radix4_max, has_final = small_fft_macros(n)
    # geometry of k once per thread, same formula as prepared.compute_geometry
    thick_load = (
        "float thickness = thick_params[0]; float theta_row = thick_params[1]; float theta_col = thick_params[2]; "
        "float k_r2 = kxv * kxv + kyv * kyv; float k_r = metal::sqrt(k_r2); float k_alpha = k_r * wavelength; "
        "float alpha_kv = k_alpha * k_alpha; float k_inv_r2 = k_r2 > 1.0e-30f ? 1.0f / k_r2 : 0.0f; "
        "float cos2_kv = (kxv * kxv - kyv * kyv) * k_inv_r2; float sin2_kv = 2.0f * kxv * kyv * k_inv_r2; "
        "float k_inv_r = k_r > 1.0e-15f ? 1.0f / k_r : 0.0f; "
        "float k_denom = metal::sqrt((kxv * ang_y) * (kxv * ang_y) + (kyv * ang_x) * (kyv * ang_x)) * k_inv_r; "
        "float k_edge = k_denom > 1.0e-15f ? (semiangle - k_alpha) / k_denom + 0.5f : 1.0f; "
        "float ap_kv = metal::clamp(k_edge, 0.0f, 1.0f);"
        if thick else ""
    )
    # depth weights of the two SSB terms; |x| < 1e-6 gives w = 1 exactly (standard SSB), as in thick_sample
    thick_weights = (
        "float shift = 6.283185307179586f * (qxv * theta_row + qyv * theta_col); "
        "float x1 = 0.5f * thickness * (-factor * (alpha2_m - alpha_kv) - shift); "
        "float x2 = 0.5f * thickness * (factor * (alpha2_p - alpha_kv) - shift); "
        "float w1 = metal::abs(x1) < 1.0e-6f ? 1.0f : metal::precise::sin(x1) / x1; "
        "float w2 = metal::abs(x2) < 1.0e-6f ? 1.0f : metal::precise::sin(x2) / x2;"
        if thick else ""
    )
    if thick:
        gamma_lines = (
            "float chi_k = factor * alpha_kv * (c12v * (cos2_kv * cos2v + sin2_kv * sin2v) + c10v); "
            "float d1 = chi_m - chi_k; float d2 = chi_p - chi_k; "
            "float c1; float s1 = metal::precise::sincos(d1, c1); "
            "float cp; float sp = metal::precise::sincos(d2, cp); "
            "float b1 = w1 * ap_m * ap_kv; float b2 = w2 * ap_p * ap_kv; "
            "float gamma_r = b1 * c1 - b2 * cp; float gamma_i = -b1 * s1 - b2 * sp;"
        )
    else:
        gamma_lines = (
            "float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);\n"
            "                    float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);"
        )
    final_stage = ""
    if has_final:
        final_stage = f"""
        for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
            threadgroup float2* frow =
                &shared_rows[local_row][candidate][0];
            uint j0 = tid;
            uint j1 = tid + {t}u;
            float2 a0 = frow[j0];
            float2 b0 = CMUL(TW(j0), frow[j0 + {half}u]);
            float2 a1 = frow[j1];
            float2 b1 = CMUL(TW(j1), frow[j1 + {half}u]);
            frow[j0] = CADD(a0, b0);
            frow[j0 + {half}u] = CSUB(a0, b0);
            frow[j1] = CADD(a1, b1);
            frow[j1 + {half}u] = CSUB(a1, b1);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        {define_rev}

        constexpr uint CHUNK = {int(chunk)};
        constexpr uint BATCH = {int(batch)};
        constexpr uint ROWS_PER_GROUP = {int(rows_per_group)};
        constexpr uint N = {n}u;
        constexpr uint T = {t}u;
        constexpr uint PLANE = N * N;
        constexpr uint GQK_COLS = {int(gqk_cols)};
        constexpr uint GQK_PLANE = N * GQK_COLS;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint bf = thread_position_in_grid.z;
        if (tid >= T || local_row >= ROWS_PER_GROUP || row >= N || bf >= CHUNK) {{
            return;
        }}

        threadgroup float2 shared_rows[ROWS_PER_GROUP][BATCH][{n}];

        float factor = scalars[0];
        float dc_r = scalars[1];
        float dc_i = scalars[2];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];
        {thick_load}

        for (uint lane = 0u; lane < 4u; ++lane) {{
            uint col = tid + lane * T;
            if (row == 0u && col == 0u) {{
                for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
                    shared_rows[local_row][candidate][DIGITREVN(col)] =
                        float2(dc_r, dc_i);
                }}
            }} else {{
                float qyv = q_col[col];

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
                {thick_weights}

                size_t g_idx;
                bool mirror = false;
                if (GQK_COLS == N) {{
                    g_idx = (size_t)bf * (size_t)PLANE + (size_t)row * (size_t)N + (size_t)col;
                }} else if (col <= N / 2u) {{
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)row * (size_t)GQK_COLS
                        + (size_t)col;
                }} else {{
                    uint mirror_row = row == 0u ? 0u : N - row;
                    uint mirror_col = N - col;
                    g_idx = (size_t)bf * (size_t)GQK_PLANE
                        + (size_t)mirror_row * (size_t)GQK_COLS
                        + (size_t)mirror_col;
                    mirror = true;
                }}
                auto gz = g[g_idx];
                float gr = gz.real;
                float gi = mirror ? -gz.imag : gz.imag;
                for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
                    float c10v = c10[candidate];
                    float c12v = c12[candidate];
                    float cos2v = cos2phi12[candidate];
                    float sin2v = sin2phi12[candidate];
                    auto pkz = pk[(size_t)candidate * (size_t)CHUNK + bf];
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

                    {gamma_lines}
                    float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
                    float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
                    float conj_gamma_r = gamma_r * inv_mag;
                    float conj_gamma_i = -gamma_i * inv_mag;
                    shared_rows[local_row][candidate][DIGITREVN(col)] = float2(
                        gr * conj_gamma_r - gi * conj_gamma_i,
                        gr * conj_gamma_i + gi * conj_gamma_r
                    );
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint m = 4u; m <= {radix4_max}u; m <<= 2) {{
            for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
                threadgroup float2* srow =
                    &shared_rows[local_row][candidate][0];
                uint quarter = m >> 2;
                uint j = tid % quarter;
                uint k = tid / quarter;
                uint idx0 = k * m + j;
                uint idx1 = idx0 + quarter;
                uint idx2 = idx1 + quarter;
                uint idx3 = idx2 + quarter;
                uint tw = j * (N / m);
                float2 x0 = srow[idx0];
                float2 x1 = CMUL(TW(tw), srow[idx1]);
                float2 x2 = CMUL(TW(tw * 2u), srow[idx2]);
                float2 x3 = CMUL(TW(tw * 3u), srow[idx3]);
                float2 t0 = CADD(x0, x2);
                float2 t1 = CSUB(x0, x2);
                float2 t2 = CADD(x1, x3);
                float2 t3 = CSUB(x1, x3);
                float2 it3 = CMULI(t3);
                srow[idx0] = CADD(t0, t2);
                srow[idx1] = CADD(t1, it3);
                srow[idx2] = CSUB(t0, t2);
                srow[idx3] = CSUB(t1, it3);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        {final_stage}

        for (uint candidate = 0u; candidate < BATCH; ++candidate) {{
        threadgroup float2* srow = &shared_rows[local_row][candidate][0];
        size_t base = ((size_t)candidate * (size_t)CHUNK + (size_t)bf)
            * (size_t)PLANE + (size_t)row * (size_t)N;
        row_ifft[base + tid].real = srow[tid].x;
        row_ifft[base + tid].imag = srow[tid].y;
        row_ifft[base + tid + T].real = srow[tid + T].x;
        row_ifft[base + tid + T].imag = srow[tid + T].y;
        row_ifft[base + tid + 2u * T].real = srow[tid + 2u * T].x;
        row_ifft[base + tid + 2u * T].imag = srow[tid + 2u * T].y;
        row_ifft[base + tid + 3u * T].real = srow[tid + 3u * T].x;
        row_ifft[base + tid + 3u * T].imag = srow[tid + 3u * T].y;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        {undef_rev}
    """
    if thick:
        # the depth phase x = t factor (alpha^2(q -/+ k) - alpha^2(k)) / 2 multiplies the geometry's rounding by the
        # thickness; fast sqrt put ~6e-7 rad of noise on a 15 nm tilted preview (precise: ~1e-8, as the MLX reference)
        source = source.replace("metal::sqrt(", "metal::precise::sqrt(")
    return mx.fast.metal_kernel(
        name=(
            f"ssb_row_ifft{n}_dyn_n{int(batch)}_b{int(chunk)}_"
            f"g{int(gqk_cols)}_r{int(rows_per_group)}_sb1"
            + ("_thick" if thick else "")
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
            "twiddle",
        ] + (["thick_params"] if thick else []),
        output_names=["row_ifft"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def row_ifft_small_from_dynamic_geometry(
    prepared: PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
    thick=None,
):
    """Fused dynamic correction + 128/256/1024 row IFFT for exact MPS phase/loss.

    ``thick`` = (thickness, tilt_row_rad, tilt_col_rad) switches on the thick-sample depth weights
    (``_row_ifft_small_dynamic_kernel(thick=True)``); None is standard SSB.
    """
    mx = prepared.mx
    if prepared.scan_shape not in ((128, 128), (256, 256), (1024, 1024)):
        raise ValueError(
            "Fused MPS row IFFT supports only 128x128, 256x256, or 1024x1024."
        )
    if int(c10.shape[0]) != 1:
        raise ValueError("Fused MPS small row IFFT currently supports one candidate.")
    n = int(prepared.scan_shape[0])
    chunk = int(stop) - int(start)
    kernel = _row_ifft_small_dynamic_kernel(
        n,
        chunk,
        int(prepared.g_qk.shape[-1]),
        thick=thick is not None,
    )
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
    pk = pk_batch_from_prepared(
        prepared,
        start=start,
        stop=stop,
        c10=c10,
        c12=c12,
        cos2phi12=cos2phi12,
        sin2phi12=sin2phi12,
    )[0]
    extra = [] if thick is None else [mx.array([float(v) for v in thick], dtype=mx.float32)]
    t = n // 4
    return kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            twiddle_n(mx, n),
        ] + extra,
        template=[],
        grid=(t, n, chunk),
        threadgroup=(t, 4, 1),
        output_shapes=[(chunk, n, n)],
        output_dtypes=[mx.complex64],
    )[0]


def row_ifft_small_batch_from_dynamic_geometry(
    prepared: PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
    pk_override=None,
    rows_per_group: int = 4,
):
    """Share small-scan geometry and G reads across an exact candidate pair."""
    mx = prepared.mx
    if prepared.scan_shape not in ((128, 128), (256, 256), (1024, 1024)):
        raise ValueError(
            "Batched fused MPS row IFFT supports 128x128, 256x256, "
            "or 1024x1024."
        )
    batch = int(c10.shape[0])
    if batch not in (1, 2):
        raise ValueError("Batched fused small MPS row IFFT supports at most a pair.")
    n = int(prepared.scan_shape[0])
    chunk = int(stop) - int(start)
    kernel = _row_ifft_small_dynamic_kernel(
        n,
        chunk,
        int(prepared.g_qk.shape[-1]),
        batch,
        rows_per_group,
    )
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
    pk = pk_override
    if pk is None:
        pk = pk_batch_from_prepared(
            prepared,
            start=start,
            stop=stop,
            c10=c10,
            c12=c12,
            cos2phi12=cos2phi12,
            sin2phi12=sin2phi12,
        )
    t = n // 4
    return kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
            twiddle_n(mx, n),
        ],
        template=[],
        grid=(t, n, chunk),
        threadgroup=(t, rows_per_group, 1),
        output_shapes=[(batch, chunk, n, n)],
        output_dtypes=[mx.complex64],
    )[0]
