"""Fused column inverse FFT and phase accumulation for 128, 256 and 1024 scans.

The row inverse FFT (``row_ifft``) leaves one row-transformed plane per bright-field pixel; these kernels finish the
column transform in threadgroup memory and accumulate the phase sum (and the loss terms) without writing the object
planes.
"""

from functools import lru_cache

from quantem.gpu.ssb.mps.hardware import require_mlx
from quantem.gpu.ssb.mps.kernels.radix import small_fft_macros, twiddle_n


@lru_cache(maxsize=32)
def _phase_cols_small_reduced_kernel(
    n: int,
    num_bf: int,
    k_bf: int,
    compute_loss: bool,
    cols_per_group: int = 4,
    batch: int = 1,
):
    mx = require_mlx()
    n = int(n)
    t = n // 4
    half = n // 2
    cols_per_group = max(1, int(cols_per_group))
    define_rev, undef_rev, radix4_max, has_final = small_fft_macros(n)
    loss_decl = (
        "float sq0 = 0.0f; float sq1 = 0.0f; "
        "float sq2 = 0.0f; float sq3 = 0.0f;"
        if compute_loss else ""
    )
    loss_accum = (
        "sq0 += p0 * p0; sq1 += p1 * p1; "
        "sq2 += p2 * p2; sq3 += p3 * p3;"
        if compute_loss else ""
    )
    loss_output = (
        f"sumsq_tile[(((size_t)candidate * (size_t)GROUPS + (size_t)group) * {n}u + (size_t)col) * {t}u + tid] = "
        "sq0 + sq1 + sq2 + sq3;"
        if compute_loss else ""
    )
    first_stage = """
            srow[rev0] = float2(z0.real, z0.imag);
            srow[rev1] = float2(z1.real, z1.imag);
            srow[rev2] = float2(z2.real, z2.imag);
            srow[rev3] = float2(z3.real, z3.imag);
            threadgroup_barrier(mem_flags::mem_threadgroup);
    """
    radix4_start = 4
    register_first_stage = n == 1024
    if register_first_stage:
        # At 1024, the four values loaded by one thread map to one first-stage
        # butterfly after digit reversal. Compute that unchanged butterfly in
        # registers, then publish its outputs for stage two.
        first_stage = """
            float2 x0 = float2(z0.real, z0.imag);
            float2 x1 = CMUL(TW(0u), float2(z1.real, z1.imag));
            float2 x2 = CMUL(TW(0u), float2(z2.real, z2.imag));
            float2 x3 = CMUL(TW(0u), float2(z3.real, z3.imag));
            float2 t0 = CADD(x0, x2);
            float2 t1 = CSUB(x0, x2);
            float2 t2 = CADD(x1, x3);
            float2 t3 = CSUB(x1, x3);
            float2 it3 = CMULI(t3);
            srow[rev0] = CADD(t0, t2);
            srow[rev1] = CADD(t1, it3);
            srow[rev2] = CSUB(t0, t2);
            srow[rev3] = CSUB(t1, it3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        radix4_start = 16
    final_stage = ""
    if has_final:
        final_stage = f"""
            uint j0 = tid;
            uint j1 = tid + {t}u;
            float2 a0 = srow[j0];
            float2 b0 = CMUL(TW(j0), srow[j0 + {half}u]);
            float2 a1 = srow[j1];
            float2 b1 = CMUL(TW(j1), srow[j1 + {half}u]);
            srow[j0] = CADD(a0, b0);
            srow[j0 + {half}u] = CSUB(a0, b0);
            srow[j1] = CADD(a1, b1);
            srow[j1 + {half}u] = CSUB(a1, b1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    output_names = ["sum_out", "sumsq_tile"] if compute_loss else ["sum_out"]
    name_suffix = "scalar" if compute_loss else "sum"
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        {define_rev}

        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint BATCH = {int(batch)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint N = {n}u;
        constexpr uint T = {t}u;
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        uint candidate = z / GROUPS;
        uint group = z - candidate * GROUPS;
        if (tid >= T || local_col >= {cols_per_group}u || col >= N || group >= GROUPS || candidate >= BATCH) {{
            return;
        }}

        threadgroup float2 shared_cols[{cols_per_group}][{n}];
        threadgroup float2* srow = &shared_cols[local_col][0];

        uint pos0 = tid;
        uint pos1 = tid + T;
        uint pos2 = tid + 2u * T;
        uint pos3 = tid + 3u * T;
        uint rev0 = DIGITREVN(pos0);
        uint rev1 = DIGITREVN(pos1);
        uint rev2 = DIGITREVN(pos2);
        uint rev3 = DIGITREVN(pos3);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        {loss_decl}

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            size_t base = (((size_t)candidate * (size_t)NUM_BF + (size_t)bf)
                * (size_t)N * (size_t)N) + (size_t)col;
            auto z0 = row_ifft[base + (size_t)pos0 * (size_t)N];
            auto z1 = row_ifft[base + (size_t)pos1 * (size_t)N];
            auto z2 = row_ifft[base + (size_t)pos2 * (size_t)N];
            auto z3 = row_ifft[base + (size_t)pos3 * (size_t)N];
            {first_stage}

            for (uint m = {radix4_start}u; m <= {radix4_max}u; m <<= 2) {{
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
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            {final_stage}

            float2 o0 = srow[pos0];
            float2 o1 = srow[pos1];
            float2 o2 = srow[pos2];
            float2 o3 = srow[pos3];
            float p0 = metal::atan2(o0.y, o0.x);
            float p1 = metal::atan2(o1.y, o1.x);
            float p2 = metal::atan2(o2.y, o2.x);
            float p3 = metal::atan2(o3.y, o3.x);
            sum0 += p0;
            sum1 += p1;
            sum2 += p2;
            sum3 += p3;
            {loss_accum}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        size_t out_base = ((size_t)candidate * (size_t)GROUPS + (size_t)group)
            * (size_t)N * (size_t)N;
        sum_out[out_base + (size_t)pos0 * (size_t)N + (size_t)col] = sum0;
        sum_out[out_base + (size_t)pos1 * (size_t)N + (size_t)col] = sum1;
        sum_out[out_base + (size_t)pos2 * (size_t)N + (size_t)col] = sum2;
        sum_out[out_base + (size_t)pos3 * (size_t)N + (size_t)col] = sum3;
        {loss_output}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        {undef_rev}
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_phase_cols{n}_{name_suffix}_n{int(num_bf)}_"
            f"k{int(k_bf)}_c{int(cols_per_group)}_b{int(batch)}_"
            f"f1{int(register_first_stage)}"
        ),
        input_names=["row_ifft", "twiddle"],
        output_names=output_names,
        source=source,
        compile_options={"math_mode": "fast"},
    )


@lru_cache(maxsize=16)
def _phase_cols_small_scalar_pair_kernel(
    n: int,
    num_bf: int,
    k_bf: int,
    cols_per_group: int = 4,
):
    """Share a column threadgroup while retaining scalar candidate FFTs."""
    mx = require_mlx()
    n = int(n)
    t = n // 4
    half = n // 2
    cols_per_group = max(1, int(cols_per_group))
    define_rev, undef_rev, radix4_max, has_final = small_fft_macros(n)
    final_stage = ""
    if has_final:
        final_stage = f"""
            for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                threadgroup float2* frow =
                    &shared_cols[local_col][candidate][0];
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

        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint N = {n}u;
        constexpr uint T = {t}u;
        constexpr uint PLANE = N * N;
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        if (tid >= T || local_col >= {cols_per_group}u || col >= N || group >= GROUPS) {{
            return;
        }}

        threadgroup float2 shared_cols[{cols_per_group}][2][{n}];
        uint pos0 = tid;
        uint pos1 = tid + T;
        uint pos2 = tid + 2u * T;
        uint pos3 = tid + 3u * T;
        uint rev0 = DIGITREVN(pos0);
        uint rev1 = DIGITREVN(pos1);
        uint rev2 = DIGITREVN(pos2);
        uint rev3 = DIGITREVN(pos3);
        float sum0[2] = {{0.0f, 0.0f}};
        float sum1[2] = {{0.0f, 0.0f}};
        float sum2[2] = {{0.0f, 0.0f}};
        float sum3[2] = {{0.0f, 0.0f}};
        float sq0[2] = {{0.0f, 0.0f}};
        float sq1[2] = {{0.0f, 0.0f}};
        float sq2[2] = {{0.0f, 0.0f}};
        float sq3[2] = {{0.0f, 0.0f}};

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                threadgroup float2* srow =
                    &shared_cols[local_col][candidate][0];
                size_t base = (((size_t)candidate * (size_t)NUM_BF + bf)
                    * (size_t)PLANE) + (size_t)col;
                auto z0 = row_ifft[base + (size_t)pos0 * N];
                auto z1 = row_ifft[base + (size_t)pos1 * N];
                auto z2 = row_ifft[base + (size_t)pos2 * N];
                auto z3 = row_ifft[base + (size_t)pos3 * N];
                srow[rev0] = float2(z0.real, z0.imag);
                srow[rev1] = float2(z1.real, z1.imag);
                srow[rev2] = float2(z2.real, z2.imag);
                srow[rev3] = float2(z3.real, z3.imag);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint m = 4u; m <= {radix4_max}u; m <<= 2) {{
                for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                    threadgroup float2* srow =
                        &shared_cols[local_col][candidate][0];
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

            for (uint candidate = 0u; candidate < 2u; ++candidate) {{
                threadgroup float2* srow =
                    &shared_cols[local_col][candidate][0];
                float2 o0 = srow[pos0];
                float2 o1 = srow[pos1];
                float2 o2 = srow[pos2];
                float2 o3 = srow[pos3];
                float p0 = metal::atan2(o0.y, o0.x);
                float p1 = metal::atan2(o1.y, o1.x);
                float p2 = metal::atan2(o2.y, o2.x);
                float p3 = metal::atan2(o3.y, o3.x);
                sum0[candidate] += p0; sum1[candidate] += p1;
                sum2[candidate] += p2; sum3[candidate] += p3;
                sq0[candidate] += p0 * p0; sq1[candidate] += p1 * p1;
                sq2[candidate] += p2 * p2; sq3[candidate] += p3 * p3;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        for (uint candidate = 0u; candidate < 2u; ++candidate) {{
            size_t out_base = ((size_t)candidate * GROUPS + group) * PLANE;
            sum_out[out_base + (size_t)pos0 * N + col] = sum0[candidate];
            sum_out[out_base + (size_t)pos1 * N + col] = sum1[candidate];
            sum_out[out_base + (size_t)pos2 * N + col] = sum2[candidate];
            sum_out[out_base + (size_t)pos3 * N + col] = sum3[candidate];
            size_t sq_base = (((size_t)candidate * GROUPS + group) * N + col)
                * T + tid;
            sumsq_tile[sq_base] = sq0[candidate] + sq1[candidate]
                + sq2[candidate] + sq3[candidate];
        }}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        {undef_rev}
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_phase_cols{n}_scalar_pair_n{int(num_bf)}_"
            f"k{int(k_bf)}_c{int(cols_per_group)}"
        ),
        input_names=["row_ifft", "twiddle"],
        output_names=["sum_out", "sumsq_tile"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def phase_cols_small_sum_from_row_ifft(mx, row_ifft, *, k_bf: int = 32):
    """Fuse 128/256/1024-column IFFT and phase accumulation without loss work."""
    shape = tuple(int(size) for size in row_ifft.shape)
    if (
        len(shape) != 3
        or shape[-2] != shape[-1]
        or shape[-1] not in (128, 256, 1024)
    ):
        raise ValueError(
            "Expected row-IFFT chunk shape "
            f"(BF, 128/256/1024, 128/256/1024), got {shape}."
        )
    num_bf = int(shape[0])
    n = int(shape[-1])
    t = n // 4
    k_bf = max(1, int(k_bf))
    cols_per_group = 8 if n <= 256 else 4
    groups = (num_bf + k_bf - 1) // k_bf
    kernel = _phase_cols_small_reduced_kernel(
        n,
        num_bf,
        k_bf,
        False,
        cols_per_group,
    )
    partial_sum = kernel(
        inputs=[row_ifft, twiddle_n(mx, n)],
        template=[],
        grid=(t, n, groups),
        threadgroup=(t, cols_per_group, 1),
        output_shapes=[(groups, n, n)],
        output_dtypes=[mx.float32],
    )[0]
    if groups == 1:
        return partial_sum[0]
    return mx.sum(partial_sum, axis=0)


def phase_cols_small_scalar_loss_from_row_ifft(mx, row_ifft, *, k_bf: int = 32):
    """Fuse 128/256/1024-column IFFT, phase sum, and scalar phase-squared loss."""
    shape = tuple(int(size) for size in row_ifft.shape)
    if (
        len(shape) != 3
        or shape[-2] != shape[-1]
        or shape[-1] not in (128, 256, 1024)
    ):
        raise ValueError(
            "Expected row-IFFT chunk shape "
            f"(BF, 128/256/1024, 128/256/1024), got {shape}."
        )
    num_bf = int(shape[0])
    n = int(shape[-1])
    t = n // 4
    k_bf = max(1, int(k_bf))
    cols_per_group = 4
    groups = (num_bf + k_bf - 1) // k_bf
    kernel = _phase_cols_small_reduced_kernel(
        n,
        num_bf,
        k_bf,
        True,
        cols_per_group,
    )
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, twiddle_n(mx, n)],
        template=[],
        grid=(t, n, groups),
        threadgroup=(t, cols_per_group, 1),
        output_shapes=[(groups, n, n), (groups, n, t)],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sum = partial_sum[0] if groups == 1 else mx.sum(partial_sum, axis=0)
    return phase_sum, mx.sum(partial_sumsq_tile)


def phase_cols_small_scalar_loss_batch_from_row_ifft(
    mx,
    row_ifft,
    *,
    k_bf: int = 32,
    packed_pair: bool = False,
):
    """Run exact small-scan column FFT and reductions for candidate batches."""
    shape = tuple(int(size) for size in row_ifft.shape)
    if (
        len(shape) != 4
        or shape[-2] != shape[-1]
        or shape[-1] not in (128, 256, 1024)
    ):
        raise ValueError(
            "Expected batched row-IFFT shape "
            "(batch, BF, 128/256/1024, 128/256/1024), "
            f"got {shape}."
        )
    batch, num_bf, n, _ = shape
    t = n // 4
    k_bf = max(1, int(k_bf))
    cols_per_group = 4
    groups = (num_bf + k_bf - 1) // k_bf
    if packed_pair:
        if batch != 2:
            raise ValueError("Packed small MPS columns require two candidates.")
        kernel = _phase_cols_small_scalar_pair_kernel(
            n,
            num_bf,
            k_bf,
            cols_per_group,
        )
        grid_z = groups
    else:
        kernel = _phase_cols_small_reduced_kernel(
            n,
            num_bf,
            k_bf,
            True,
            cols_per_group,
            batch,
        )
        grid_z = batch * groups
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, twiddle_n(mx, n)],
        template=[],
        grid=(t, n, grid_z),
        threadgroup=(t, cols_per_group, 1),
        output_shapes=[
            (batch, groups, n, n),
            (batch, groups, n, t),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    # Keep each candidate's reduction shape identical to the scalar exact path.
    # A batched multi-axis MLX reduction changes the float32 association by an
    # ULP for some 256x256 inputs even though the Metal column outputs match.
    phase_sum = mx.stack(
        [
            partial_sum[candidate, 0]
            if groups == 1
            else mx.sum(partial_sum[candidate], axis=0)
            for candidate in range(batch)
        ]
    )
    phase_sumsq = mx.stack(
        [mx.sum(partial_sumsq_tile[candidate]) for candidate in range(batch)]
    )
    return phase_sum, phase_sumsq
