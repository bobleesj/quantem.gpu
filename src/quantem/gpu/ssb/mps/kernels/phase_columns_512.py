"""Fused column inverse FFT and phase accumulation for 512 x 512 scans (radix-8, paired candidates)."""

from functools import lru_cache

from quantem.gpu.ssb.mps.hardware import require_mlx, use_simd_radix8_col_stage_512
from quantem.gpu.ssb.mps.kernels.radix import twiddle_512, twiddle_512_metal_header


@lru_cache(maxsize=32)
def _phase_cols512_reduced_kernel(num_bf: int, k_bf: int, compute_loss: bool):
    mx = require_mlx()
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
        "sumsq_tile[((size_t)group * 512u + (size_t)col) * 128u + tid] = "
        "sq0 + sq1 + sq2 + sq3;"
        if compute_loss else ""
    )
    output_names = ["sum_out", "sumsq_tile"] if compute_loss else ["sum_out"]
    name_suffix = "scalar" if compute_loss else "sum"
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) float2(twiddle[(i)].real, twiddle[(i)].imag)
        #define BITREV4_8(x) ((((x) & 0x03u) << 6) | (((x) & 0x0Cu) << 2) | (((x) & 0x30u) >> 2) | (((x) & 0xC0u) >> 6))
        #define DIGITREV512(x) ((((x) & 1u) << 8) | BITREV4_8((x) >> 1))

        constexpr uint NUM_BF = {int(num_bf)};
        constexpr uint K_BF = {int(k_bf)};
        constexpr uint GROUPS = (NUM_BF + K_BF - 1u) / K_BF;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint group = thread_position_in_grid.z;
        if (tid >= 128u || local_col >= 4u || col >= 512u || group >= GROUPS) {{
            return;
        }}

        threadgroup float2 shared_cols[4][512];
        threadgroup float2* srow = &shared_cols[local_col][0];

        uint pos0 = tid;
        uint pos1 = tid + 128u;
        uint pos2 = tid + 256u;
        uint pos3 = tid + 384u;
        uint rev0 = DIGITREV512(pos0);
        uint rev1 = DIGITREV512(pos1);
        uint rev2 = DIGITREV512(pos2);
        uint rev3 = DIGITREV512(pos3);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        {loss_decl}

        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
        for (uint bf = bf_start; bf < bf_end; ++bf) {{
            size_t base = ((size_t)bf * 512u * 512u) + (size_t)col;
            auto z0 = row_ifft[base + (size_t)pos0 * 512u];
            auto z1 = row_ifft[base + (size_t)pos1 * 512u];
            auto z2 = row_ifft[base + (size_t)pos2 * 512u];
            auto z3 = row_ifft[base + (size_t)pos3 * 512u];
            srow[rev0] = float2(z0.real, z0.imag);
            srow[rev1] = float2(z1.real, z1.imag);
            srow[rev2] = float2(z2.real, z2.imag);
            srow[rev3] = float2(z3.real, z3.imag);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint m = 4u; m <= 256u; m <<= 2) {{
                uint quarter = m >> 2;
                uint j = tid % quarter;
                uint k = tid / quarter;
                uint idx0 = k * m + j;
                uint idx1 = idx0 + quarter;
                uint idx2 = idx1 + quarter;
                uint idx3 = idx2 + quarter;
                uint tw = j * (512u / m);
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

            uint j0 = tid;
            uint j1 = tid + 128u;
            float2 a0 = srow[j0];
            float2 b0 = CMUL(TW(j0), srow[j0 + 256u]);
            float2 a1 = srow[j1];
            float2 b1 = CMUL(TW(j1), srow[j1 + 256u]);
            srow[j0] = CADD(a0, b0);
            srow[j0 + 256u] = CSUB(a0, b0);
            srow[j1] = CADD(a1, b1);
            srow[j1 + 256u] = CSUB(a1, b1);
            threadgroup_barrier(mem_flags::mem_threadgroup);

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

        size_t out_base = (size_t)group * 512u * 512u;
        sum_out[out_base + (size_t)pos0 * 512u + (size_t)col] = sum0;
        sum_out[out_base + (size_t)pos1 * 512u + (size_t)col] = sum1;
        sum_out[out_base + (size_t)pos2 * 512u + (size_t)col] = sum2;
        sum_out[out_base + (size_t)pos3 * 512u + (size_t)col] = sum3;
        {loss_output}

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef BITREV4_8
        #undef DIGITREV512
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_cols512_{name_suffix}_n{int(num_bf)}_k{int(k_bf)}",
        input_names=["row_ifft", "twiddle"],
        output_names=output_names,
        source=source,
        compile_options={"math_mode": "fast"},
    )


def phase_cols512_sum_from_row_ifft(
    mx,
    row_ifft,
    *,
    k_bf: int = 32,
    active_bf=None,
    tiled_input: bool = False,
):
    """Fuse masked radix-8 column IFFT and phase accumulation."""
    shape = tuple(int(size) for size in row_ifft.shape)
    if len(shape) != 3 or shape[-2:] != (512, 512):
        raise ValueError(f"Expected row-IFFT chunk shape (BF, 512, 512), got {shape}.")
    batch_sum, _batch_sumsq = phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        row_ifft[None, ...],
        k_bf=k_bf,
        active_bf=active_bf,
        tiled_input=tiled_input,
    )
    return batch_sum[0]


def phase_cols512_scalar_loss_from_row_ifft(mx, row_ifft, *, k_bf: int = 32):
    """Fuse 512-column IFFT, phase sum, and scalar phase-squared loss."""
    shape = tuple(int(size) for size in row_ifft.shape)
    if len(shape) != 3 or shape[-2:] != (512, 512):
        raise ValueError(f"Expected row-IFFT chunk shape (BF, 512, 512), got {shape}.")
    num_bf = int(shape[0])
    k_bf = max(1, int(k_bf))
    groups = (num_bf + k_bf - 1) // k_bf
    kernel = _phase_cols512_reduced_kernel(num_bf, k_bf, True)
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, twiddle_512(mx)],
        template=[],
        grid=(128, 512, groups),
        threadgroup=(128, 4, 1),
        output_shapes=[(groups, 512, 512), (groups, 512, 128)],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sum = partial_sum[0] if groups == 1 else mx.sum(partial_sum, axis=0)
    return phase_sum, mx.sum(partial_sumsq_tile)


@lru_cache(maxsize=32)
def _phase_cols512_radix8_batch_kernel(
    batch: int,
    num_bf: int,
    k_bf: int,
    tiled_input: bool,
    storage_num_bf: int | None = None,
    bf_offset: int = 0,
    bf_ranges: tuple[tuple[int, int], ...] | None = None,
):
    """Build the 64-thread radix-8 exact phase/loss column kernel."""
    mx = require_mlx()
    simd_radix8 = use_simd_radix8_col_stage_512()
    storage_num_bf = int(num_bf if storage_num_bf is None else storage_num_bf)
    bf_offset = int(bf_offset)
    range_setup = """
        uint cand = z / GROUPS;
        uint group = z - cand * GROUPS;
        if (tid >= 64u || local_col >= 8u || col >= 512u || cand >= BATCH || group >= GROUPS) {
            return;
        }
        uint bf_start = group * K_BF;
        uint bf_end = metal::min(bf_start + K_BF, NUM_BF);
    """
    range_name = ""
    if bf_ranges is not None:
        bf_ranges = tuple((int(start), int(stop)) for start, stop in bf_ranges)
        invalid_bounds = any(
            start < 0 or stop <= start or stop > storage_num_bf
            for start, stop in bf_ranges
        )
        overlapping = any(
            start < previous_stop
            for (_previous_start, previous_stop), (start, _stop) in zip(
                bf_ranges, bf_ranges[1:]
            )
        )
        if not bf_ranges or invalid_bounds or overlapping:
            raise ValueError(
                "Packed column BF ranges must be non-empty, ordered, "
                f"non-overlapping, and within [0, {storage_num_bf})."
            )
        start_expr = f"{bf_ranges[-1][0]}u"
        stop_expr = f"{bf_ranges[-1][1]}u"
        for index in range(len(bf_ranges) - 2, -1, -1):
            start_expr = (
                f"group == {index}u ? {bf_ranges[index][0]}u : ({start_expr})"
            )
            stop_expr = (
                f"group == {index}u ? {bf_ranges[index][1]}u : ({stop_expr})"
            )
        range_setup = f"""
        uint cand = z / GROUPS;
        uint group = z - cand * GROUPS;
        if (tid >= 64u || local_col >= 8u || col >= 512u || cand >= BATCH || group >= GROUPS) {{
            return;
        }}
        uint bf_start = {start_expr};
        uint bf_end = {stop_expr};
        """
        range_name = "_r" + "_".join(
            f"{start}x{stop}" for start, stop in bf_ranges
        )
    groups_define = (
        str(len(bf_ranges))
        if bf_ranges is not None
        else "(NUM_BF + K_BF - 1u) / K_BF"
    )
    row_ifft_load = """
            size_t base = (
                ((size_t)cand * (size_t)STORAGE_NUM_BF
                    + (size_t)BF_OFFSET + (size_t)bf)
                * 512u * 512u
                + (size_t)col
            );
            float2 r0=float2(row_ifft[base+(size_t)src0*512u].real,row_ifft[base+(size_t)src0*512u].imag);
            float2 r1=float2(row_ifft[base+(size_t)src1*512u].real,row_ifft[base+(size_t)src1*512u].imag);
            float2 r2=float2(row_ifft[base+(size_t)src2*512u].real,row_ifft[base+(size_t)src2*512u].imag);
            float2 r3=float2(row_ifft[base+(size_t)src3*512u].real,row_ifft[base+(size_t)src3*512u].imag);
            float2 r4=float2(row_ifft[base+(size_t)src4*512u].real,row_ifft[base+(size_t)src4*512u].imag);
            float2 r5=float2(row_ifft[base+(size_t)src5*512u].real,row_ifft[base+(size_t)src5*512u].imag);
            float2 r6=float2(row_ifft[base+(size_t)src6*512u].real,row_ifft[base+(size_t)src6*512u].imag);
            float2 r7=float2(row_ifft[base+(size_t)src7*512u].real,row_ifft[base+(size_t)src7*512u].imag);
    """
    if tiled_input:
        row_ifft_load = """
            size_t plane = (
                ((size_t)cand * (size_t)STORAGE_NUM_BF
                    + (size_t)BF_OFFSET + (size_t)bf)
                * 512u * 512u
            );
            size_t tile = plane + (size_t)(col >> 3) * 4096u + (col & 7u);
            size_t i0=tile+(size_t)src0*8u;
            size_t i1=tile+(size_t)src1*8u;
            size_t i2=tile+(size_t)src2*8u;
            size_t i3=tile+(size_t)src3*8u;
            size_t i4=tile+(size_t)src4*8u;
            size_t i5=tile+(size_t)src5*8u;
            size_t i6=tile+(size_t)src6*8u;
            size_t i7=tile+(size_t)src7*8u;
            float2 r0=float2(row_ifft[i0].real,row_ifft[i0].imag);
            float2 r1=float2(row_ifft[i1].real,row_ifft[i1].imag);
            float2 r2=float2(row_ifft[i2].real,row_ifft[i2].imag);
            float2 r3=float2(row_ifft[i3].real,row_ifft[i3].imag);
            float2 r4=float2(row_ifft[i4].real,row_ifft[i4].imag);
            float2 r5=float2(row_ifft[i5].real,row_ifft[i5].imag);
            float2 r6=float2(row_ifft[i6].real,row_ifft[i6].imag);
            float2 r7=float2(row_ifft[i7].real,row_ifft[i7].imag);
        """
    simd_macros = ""
    simd_undefs = ""
    radix_stage12 = """
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            s[logical]=r0; s[logical+1u]=r1; s[logical+2u]=r2; s[logical+3u]=r3;
            s[logical+4u]=r4; s[logical+5u]=r5; s[logical+6u]=r6; s[logical+7u]=r7;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            r0=s[base2]; r1=CMUL(TW(s2*8u),s[base2+8u]);
            r2=CMUL(TW(s2*16u),s[base2+16u]); r3=CMUL(TW(s2*24u),s[base2+24u]);
            r4=CMUL(TW(s2*32u),s[base2+32u]); r5=CMUL(TW(s2*40u),s[base2+40u]);
            r6=CMUL(TW(s2*48u),s[base2+48u]); r7=CMUL(TW(s2*56u),s[base2+56u]);
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            s[base2]=r0; s[base2+8u]=r1; s[base2+16u]=r2; s[base2+24u]=r3;
            s[base2+32u]=r4; s[base2+40u]=r5; s[base2+48u]=r6; s[base2+56u]=r7;
            threadgroup_barrier(mem_flags::mem_threadgroup);
    """
    if simd_radix8:
        simd_macros = """
        #define XPOSE_PAIR(a,b,m) { float2 _xa=(a), _xb=(b); float2 _sa=simd_shuffle_xor(_xa,(ushort)(m)); float2 _sb=simd_shuffle_xor(_xb,(ushort)(m)); bool _hi=(tid & (m)) != 0u; (a)=_hi?_sb:_xa; (b)=_hi?_xb:_sa; }
        #define XPOSE8(a0,a1,a2,a3,a4,a5,a6,a7) { XPOSE_PAIR(a0,a1,1u); XPOSE_PAIR(a2,a3,1u); XPOSE_PAIR(a4,a5,1u); XPOSE_PAIR(a6,a7,1u); XPOSE_PAIR(a0,a2,2u); XPOSE_PAIR(a1,a3,2u); XPOSE_PAIR(a4,a6,2u); XPOSE_PAIR(a5,a7,2u); XPOSE_PAIR(a0,a4,4u); XPOSE_PAIR(a1,a5,4u); XPOSE_PAIR(a2,a6,4u); XPOSE_PAIR(a3,a7,4u); }
        """
        simd_undefs = """
        #undef XPOSE_PAIR
        #undef XPOSE8
        """
        radix_stage12 = """
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            XPOSE8(r0,r1,r2,r3,r4,r5,r6,r7);
            r1=CMUL(TW(s2*8u),r1);
            r2=CMUL(TW(s2*16u),r2); r3=CMUL(TW(s2*24u),r3);
            r4=CMUL(TW(s2*32u),r4); r5=CMUL(TW(s2*40u),r5);
            r6=CMUL(TW(s2*48u),r6); r7=CMUL(TW(s2*56u),r7);
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            s[base2]=r0; s[base2+8u]=r1; s[base2+16u]=r2; s[base2+24u]=r3;
            s[base2+32u]=r4; s[base2+40u]=r5; s[base2+48u]=r6; s[base2+56u]=r7;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    source = f"""
        #define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
        #define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
        #define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
        #define CMULI(a) float2(-(a).y, (a).x)
        #define TW(i) as_type<float2>(SSB_TWIDDLE_512[(i) & 511u])
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
        {simd_macros}

        constexpr uint BATCH = {int(batch)}u;
        constexpr uint NUM_BF = {int(num_bf)}u;
        constexpr uint STORAGE_NUM_BF = {storage_num_bf}u;
        constexpr uint BF_OFFSET = {bf_offset}u;
        constexpr uint K_BF = {int(k_bf)}u;
        constexpr uint GROUPS = {groups_define};
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        {range_setup}

        threadgroup float2 shared_cols[8][512];
        threadgroup float2* s = &shared_cols[local_col][0];
        float sum0=0.0f, sum1=0.0f, sum2=0.0f, sum3=0.0f;
        float sum4=0.0f, sum5=0.0f, sum6=0.0f, sum7=0.0f;
        float sq0=0.0f, sq1=0.0f, sq2=0.0f, sq3=0.0f;
        float sq4=0.0f, sq5=0.0f, sq6=0.0f, sq7=0.0f;
        uint logical = tid * 8u;
        uint src0=OCTREV512(logical), src1=OCTREV512(logical+1u);
        uint src2=OCTREV512(logical+2u), src3=OCTREV512(logical+3u);
        uint src4=OCTREV512(logical+4u), src5=OCTREV512(logical+5u);
        uint src6=OCTREV512(logical+6u), src7=OCTREV512(logical+7u);
        uint s2 = tid & 7u;
        uint base2 = (tid >> 3) * 64u + s2;

        for (uint bf=bf_start; bf<bf_end; ++bf) {{
            if (active_bf[BF_OFFSET + bf] == 0u) {{
                continue;
            }}
            {row_ifft_load}
            {radix_stage12}

            r0=s[tid]; r1=CMUL(TW(tid),s[tid+64u]);
            r2=CMUL(TW(tid*2u),s[tid+128u]); r3=CMUL(TW(tid*3u),s[tid+192u]);
            r4=CMUL(TW(tid*4u),s[tid+256u]); r5=CMUL(TW(tid*5u),s[tid+320u]);
            r6=CMUL(TW(tid*6u),s[tid+384u]); r7=CMUL(TW(tid*7u),s[tid+448u]);
            RADIX8(r0,r1,r2,r3,r4,r5,r6,r7);
            float p0=metal::atan2(r0.y,r0.x), p1=metal::atan2(r1.y,r1.x);
            float p2=metal::atan2(r2.y,r2.x), p3=metal::atan2(r3.y,r3.x);
            float p4=metal::atan2(r4.y,r4.x), p5=metal::atan2(r5.y,r5.x);
            float p6=metal::atan2(r6.y,r6.x), p7=metal::atan2(r7.y,r7.x);
            sum0+=p0; sum1+=p1; sum2+=p2; sum3+=p3;
            sum4+=p4; sum5+=p5; sum6+=p6; sum7+=p7;
            sq0+=p0*p0; sq1+=p1*p1; sq2+=p2*p2; sq3+=p3*p3;
            sq4+=p4*p4; sq5+=p5*p5; sq6+=p6*p6; sq7+=p7*p7;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        size_t out_base=((size_t)cand*(size_t)GROUPS+(size_t)group)*512u*512u+(size_t)col;
        sum_out[out_base+(size_t)tid*512u]=sum0;
        sum_out[out_base+(size_t)(tid+64u)*512u]=sum1;
        sum_out[out_base+(size_t)(tid+128u)*512u]=sum2;
        sum_out[out_base+(size_t)(tid+192u)*512u]=sum3;
        sum_out[out_base+(size_t)(tid+256u)*512u]=sum4;
        sum_out[out_base+(size_t)(tid+320u)*512u]=sum5;
        sum_out[out_base+(size_t)(tid+384u)*512u]=sum6;
        sum_out[out_base+(size_t)(tid+448u)*512u]=sum7;
        sumsq_tile[(((size_t)cand*(size_t)GROUPS+(size_t)group)*512u+(size_t)col)*64u+tid]
            =sq0+sq1+sq2+sq3+sq4+sq5+sq6+sq7;

        #undef CADD
        #undef CSUB
        #undef CMUL
        #undef CMULI
        #undef TW
        #undef OCTREV512
        #undef W8_1
        #undef W8_3
        #undef RADIX8
        {simd_undefs}
    """
    return mx.fast.metal_kernel(
        name=(
            f"ssb_phase_cols512_radix8_batch_n{batch}_bf{num_bf}_"
            f"k{k_bf}_t{int(tiled_input)}_s{storage_num_bf}_o{bf_offset}"
            f"_x{int(simd_radix8)}{range_name}"
        ),
        input_names=["row_ifft", "active_bf", "twiddle"],
        output_names=["sum_out", "sumsq_tile"],
        source=source,
        header=twiddle_512_metal_header(),
        compile_options={"math_mode": "fast"},
    )


def phase_cols512_scalar_loss_batch_from_row_ifft(
    mx,
    row_ifft,
    *,
    k_bf: int = 32,
    active_bf=None,
    tiled_input: bool = False,
    bf_start: int = 0,
    bf_stop: int | None = None,
):
    """Fuse 512-column IFFT and scalar loss for candidate-batched row IFFT."""
    shape = tuple(int(size) for size in row_ifft.shape)
    if len(shape) != 4 or shape[-2:] != (512, 512):
        raise ValueError(
            "Expected row-IFFT chunk shape (batch, BF, 512, 512), "
            f"got {shape}."
        )
    batch, storage_num_bf = int(shape[0]), int(shape[1])
    bf_start = max(0, int(bf_start))
    bf_stop = storage_num_bf if bf_stop is None else int(bf_stop)
    if bf_stop <= bf_start or bf_stop > storage_num_bf:
        raise ValueError(
            f"Invalid BF subrange [{bf_start}, {bf_stop}) for "
            f"storage length {storage_num_bf}."
        )
    num_bf = bf_stop - bf_start
    k_bf = max(1, int(k_bf))
    groups = (num_bf + k_bf - 1) // k_bf
    if active_bf is None:
        active_bf = mx.ones((storage_num_bf,), dtype=mx.uint8)
    kernel = _phase_cols512_radix8_batch_kernel(
        batch,
        num_bf,
        k_bf,
        tiled_input,
        storage_num_bf,
        bf_start,
    )
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, active_bf, twiddle_512(mx)],
        template=[],
        grid=(64, 512, batch * groups),
        threadgroup=(64, 8, 1),
        output_shapes=[(batch, groups, 512, 512), (batch, groups, 512, 64)],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sum = partial_sum[:, 0, :, :] if groups == 1 else mx.sum(partial_sum, axis=1)
    phase_sumsq = mx.sum(mx.sum(mx.sum(partial_sumsq_tile, axis=3), axis=2), axis=1)
    return phase_sum, phase_sumsq


def phase_cols512_pack_loss_batch_from_row_ifft(
    mx,
    row_ifft,
    *,
    bf_ranges: tuple[tuple[int, int], ...],
    active_bf,
):
    """Evaluate original BF boundaries together while keeping separate sums."""
    shape = tuple(int(size) for size in row_ifft.shape)
    if len(shape) != 4 or shape[-2:] != (512, 512):
        raise ValueError(
            "Expected row-IFFT pack shape (batch, BF, 512, 512), "
            f"got {shape}."
        )
    batch, storage_num_bf = int(shape[0]), int(shape[1])
    ranges = tuple((int(start), int(stop)) for start, stop in bf_ranges)
    kernel = _phase_cols512_radix8_batch_kernel(
        batch,
        storage_num_bf,
        4096,
        True,
        storage_num_bf,
        0,
        ranges,
    )
    partial_sum, partial_sumsq_tile = kernel(
        inputs=[row_ifft, active_bf, twiddle_512(mx)],
        template=[],
        grid=(64, 512, batch * len(ranges)),
        threadgroup=(64, 8, 1),
        output_shapes=[
            (batch, len(ranges), 512, 512),
            (batch, len(ranges), 512, 64),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    phase_sums = [partial_sum[:, index] for index in range(len(ranges))]
    phase_sumsqs = [
        mx.sum(
            mx.sum(
                mx.sum(partial_sumsq_tile[:, index : index + 1], axis=3),
                axis=2,
            ),
            axis=1,
        )
        for index in range(len(ranges))
    ]
    return phase_sums, phase_sumsqs
