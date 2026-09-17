"""ALU roofline for the row stage's correction block, measured in-session.

The row stage's cost was previously described as "~75% non-device_memory", but the only
ALU ablations on record remove sqrt+sincos (8% of the stage) and atan2 (5% of
the column stage), which cannot add up to that residual.  This probe replaces
the inference with a measurement:

  1. peak throughput for the four special ops the correction uses
     (fma, sqrt, reciprocal, fast::sincos), with 8 independent chains per
     thread so the number is a throughput and not a latency,
  2. a no-memory replica of the exact correction block from
     backends/mps/engine.py:2740-2795, driven over the same element count as a
     real 512-BF pack (512 bf x 512 rows x 512 cols = 134,217,728),
  3. the real row stage in the same session, so the ALU share is a difference
     of same-session numbers rather than a cross-session ratio.

Everything here is a measurement of the shipped code; nothing is landed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent.parent / "20260916-ssb-mps-hotpath"))

from profile_mps import FIXTURE, bandwidth_control, open_fixture, prepared_of  # noqa: E402
from row_occupancy import (  # noqa: E402
    CHUNK, GQK_COLS, _capture_source, _owning_module, launch, stat, variant_kernel,
)

REPEATS = 5
WARMUP = 2
ELEMENTS = 512 * 512 * 512  # (bf, row, col) evaluations in one 512-BF pack

_PRELUDE = """
#define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
#define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
#define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
#define CMULI(a) float2(-(a).y, (a).x)
#define W8_1(a) float2(0.70710678118654752f * ((a).x - (a).y), 0.70710678118654752f * ((a).x + (a).y))
#define W8_3(a) float2(0.70710678118654752f * (-(a).x - (a).y), 0.70710678118654752f * ((a).x - (a).y))
"""


def op_kernel(mx, op: str, iters: int, chains: int = 8):
    body = {
        "fma": "a{c} = metal::fma(a{c}, k1, k2);",
        "sqrt": "a{c} = metal::sqrt(a{c} + k2);",
        "rcp": "a{c} = 1.0f / (a{c} + k2);",
        "sincos": "a{c} = metal::fast::sincos(a{c} + k2, sc{c});",
    }[op]
    decl, init = [], []
    for c in range(chains):
        decl.append(f"float a{c} = seed + {c + 1}.0f * 1.0e-3f;")
        if op == "sincos":
            decl.append(f"float sc{c} = 0.0f;")
        init.append(body.format(c=c))
    loop = "\n".join(f"        {line}" for line in init)
    tail = " + ".join(f"a{c}" for c in range(chains))
    source = f"""
        {_PRELUDE}
        constexpr uint ITERS = {int(iters)}u;
        constexpr uint THREADS = 64u;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint bf = thread_position_in_grid.z;
        float seed = (float)(row * 131u + bf * 7u + tid) * 1.0e-6f;
        float k1 = 0.9999999f;
        float k2 = 1.0e-7f;
{chr(10).join('        ' + d for d in decl)}
        for (uint i = 0u; i < ITERS; ++i) {{
{loop}
        }}
        float acc = {tail};
        threadgroup float2 probe_shared[ROWS_PER_GROUP_PROBE][512];
        probe_shared[local_row][tid] = float2(acc, seed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u && local_row == 0u && bf == 0u) {{
            out[0].real = probe_shared[0][0].x;
            out[0].imag = probe_shared[0][0].y;
        }}
    """.replace("ROWS_PER_GROUP_PROBE", "1")
    name = f"ssb_alu_probe_{op}_{iters}_{chains}"
    return mx.fast.metal_kernel(
        name=name, input_names=[], output_names=["out"], source=source,
        compile_options={"math_mode": "fast"},
    )


def correction_replica(mx, engine):
    """The exact correction block from the row kernel, with no memory traffic."""
    src = _capture_source(engine, 2, CHUNK, CHUNK)
    start = src.index("                float dx = qxv - kxv;")
    end = src.index("                if (ap_m == 0.0f && ap_p == 0.0f)")
    body = src[start:end]
    source = f"""
        {_PRELUDE}
        constexpr uint CHUNK = {CHUNK}u;
        constexpr uint GQK_COLS = {GQK_COLS}u;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint bf = thread_position_in_grid.z;
        float factor = scalars[0];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];
        float acc = 0.0f;
        for (uint lane = 0u; lane < 8u; ++lane) {{
            uint col = tid + lane * 64u;
            float qyv = q_col[col];
{body}
            acc += ap_m + ap_p;
        }}
        threadgroup float2 probe_shared[1][512];
        probe_shared[local_row][tid] = float2(acc, 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u && local_row == 0u && bf == 0u) {{
            out[0].real = probe_shared[0][0].x;
            out[0].imag = probe_shared[0][0].y;
        }}
    """
    return mx.fast.metal_kernel(
        name="ssb_row_correction_replica",
        input_names=["q_row", "q_col", "kx", "ky", "scalars"],
        output_names=["out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def correction_plus_candidate(mx, engine):
    """Correction block plus the per-candidate sincos/gamma arithmetic."""
    src = _capture_source(engine, 2, CHUNK, CHUNK)
    start = src.index("                float dx = qxv - kxv;")
    end = src.index("                if (ap_m == 0.0f && ap_p == 0.0f)")
    body = src[start:end]
    cand_start = src.index("                    float c10v = c10[candidate];")
    # "uint slot" also appears in the DC and aperture early-out branches, which
    # precede the candidate loop, so search forward from cand_start.
    cand_end = src.index("                    uint slot = FUSE_CANDIDATES", cand_start)
    cand = src[cand_start:cand_end].replace("candidate", "cc")
    body = body.replace("candidate_begin", "0u").replace("candidate_end", "2u")
    source = f"""
        {_PRELUDE}
        constexpr uint CHUNK = {CHUNK}u;
        constexpr uint GQK_COLS = {GQK_COLS}u;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint bf = thread_position_in_grid.z;
        float factor = scalars[0];
        float wavelength = scalars[3];
        float semiangle = scalars[4];
        float ang_y = scalars[5];
        float ang_x = scalars[6];
        float kxv = kx[bf];
        float kyv = ky[bf];
        float qxv = q_row[row];
        float gr = qxv * 1.0e-5f;
        float gi = kyv * 1.0e-5f;
        float acc = 0.0f;
        for (uint lane = 0u; lane < 8u; ++lane) {{
            uint col = tid + lane * 64u;
            float qyv = q_col[col];
{body}
            if (ap_m != 0.0f || ap_p != 0.0f) {{
                for (uint cc = 0u; cc < 2u; ++cc) {{
{cand}
                    acc += corrected.x + corrected.y;
                }}
            }}
        }}
        threadgroup float2 probe_shared[1][512];
        probe_shared[local_row][tid] = float2(acc, 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u && local_row == 0u && bf == 0u) {{
            out[0].real = probe_shared[0][0].x;
            out[0].imag = probe_shared[0][0].y;
        }}
    """
    return mx.fast.metal_kernel(
        name="ssb_row_correction_plus_candidate",
        input_names=[
            "q_row", "q_col", "kx", "ky", "pk",
            "c10", "c12", "cos2phi12", "sin2phi12", "scalars",
        ],
        output_names=["out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--iters", type=int, default=256)
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine

    rec = {
        "probe": "alu-roofline",
        "label": os.environ.get("GPU_RUN_LABEL", "unset"),
        "host": platform.node(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elements_per_pack": ELEMENTS,
    }

    opened, _ = open_fixture(FIXTURE)
    backend, prepared, _ = prepared_of(opened)
    rec["bandwidth_control_gbs"] = round(bandwidth_control(mx), 1)

    # ---- 1. per-op peak throughput ----------------------------------------
    threads_total = 64 * 4 * 512 * 512  # same grid as the shipped row kernel
    ops = {}
    for op in ("fma", "sqrt", "rcp", "sincos"):
        k = op_kernel(mx, op, args.iters, 8)
        grid = (64, 512, 512)
        tg = (64, 4, 1)
        for _ in range(WARMUP):
            mx.eval(k(inputs=[], template=[], grid=grid, threadgroup=tg,
                      output_shapes=[(1,)], output_dtypes=[mx.complex64])[0])
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            mx.eval(k(inputs=[], template=[], grid=grid, threadgroup=tg,
                      output_shapes=[(1,)], output_dtypes=[mx.complex64])[0])
            ts.append(time.perf_counter() - t0)
        total_ops = threads_total * args.iters * 8
        ops[op] = stat(f"alu_{op}", ts, None, {
            "total_ops": int(total_ops),
            "ops_per_s": float(total_ops / float(np.median(ts))),
            "chains": 8,
            "iters": args.iters,
        })
    rec["op_throughput"] = ops

    # ---- 2. correction replica --------------------------------------------
    scalars = mx.array(
        [float(prepared.factor), float(prepared.dc_value.real),
         float(prepared.dc_value.imag), float(prepared.wavelength),
         float(prepared.semiangle_rad), float(prepared.ang_y_rad),
         float(prepared.ang_x_rad)], dtype=mx.float32)

    grid = (64, 512, CHUNK)
    tg = (64, 4, 1)

    k_corr = correction_replica(mx, engine)
    for _ in range(WARMUP):
        mx.eval(k_corr(inputs=[prepared.q_row, prepared.q_col,
                               prepared.kx[:CHUNK], prepared.ky[:CHUNK], scalars],
                       template=[], grid=grid, threadgroup=tg,
                       output_shapes=[(1,)], output_dtypes=[mx.complex64])[0])
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        mx.eval(k_corr(inputs=[prepared.q_row, prepared.q_col,
                               prepared.kx[:CHUNK], prepared.ky[:CHUNK], scalars],
                       template=[], grid=grid, threadgroup=tg,
                       output_shapes=[(1,)], output_dtypes=[mx.complex64])[0])
        ts.append(time.perf_counter() - t0)
    rec["correction_only"] = stat("correction_only", ts, None,
                                  {"elements": ELEMENTS})

    # ---- 3. correction + per-candidate arithmetic -------------------------
    c10 = mx.array(np.asarray([7.017120839737006, -50.0], dtype=np.float32))
    c12 = mx.array(np.asarray([0.0, 35.0], dtype=np.float32))
    phi = np.asarray([-0.15393969519675116, -0.2], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))
    pk = engine._pk_batch_from_prepared(
        prepared, start=0, stop=CHUNK, c10=c10, c12=c12,
        cos2phi12=cos2, sin2phi12=sin2)

    k_corr2 = correction_plus_candidate(mx, engine)
    for _ in range(WARMUP):
        mx.eval(k_corr2(inputs=[prepared.q_row, prepared.q_col,
                                prepared.kx[:CHUNK], prepared.ky[:CHUNK], pk,
                                c10, c12, cos2, sin2, scalars],
                        template=[], grid=grid, threadgroup=tg,
                        output_shapes=[(1,)], output_dtypes=[mx.complex64])[0])
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        mx.eval(k_corr2(inputs=[prepared.q_row, prepared.q_col,
                                prepared.kx[:CHUNK], prepared.ky[:CHUNK], pk,
                                c10, c12, cos2, sin2, scalars],
                        template=[], grid=grid, threadgroup=tg,
                        output_shapes=[(1,)], output_dtypes=[mx.complex64])[0])
        ts.append(time.perf_counter() - t0)
    rec["correction_plus_candidate"] = stat(
        "correction_plus_candidate", ts, None, {"elements": ELEMENTS})

    # ---- 4. same-session row stage reference ------------------------------
    kernels = variant_kernel(mx, engine, 4)
    for _ in range(WARMUP):
        mx.eval(launch(mx, engine, kernels, prepared, 4, c10, c12, cos2, sin2, 0, CHUNK))
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        mx.eval(launch(mx, engine, kernels, prepared, 4, c10, c12, cos2, sin2, 0, CHUNK))
        ts.append(time.perf_counter() - t0)
    moved = 2 * CHUNK * 2097152 + CHUNK * 512 * GQK_COLS * 8
    rec["row_stage_reference"] = stat("row_stage_rpg4_ref", ts, moved)

    peak = mx.get_active_memory() if hasattr(mx, "get_active_memory") else None
    rec["peak_active_bytes"] = int(peak) if peak is not None else None
    rec["load_average_end"] = list(os.getloadavg())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
