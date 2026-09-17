"""Column-stage ALU roofline, plus an order-balanced re-test of the row sweep.

The column stage (backends/mps/engine.py:2426-2500) does, per (bf, column):
a 512-point radix-8 transform, eight ``metal::atan2``, and eight sum/sumsq
accumulations.  This probe measures the atan2 throughput of this host and a
no-memory replica of the post-butterfly work over the same element count as a
512-BF pack (2 candidates x 512 bf x 512 col x 512 = 5.37e8 evaluations,
2.68e8 atan2), then times the real stage in the same session.

The row-stage ROWS_PER_GROUP sweep is repeated here in an order-balanced
sequence (4,2,1,1,2,4) so the negative result cannot be an ordering artefact.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent.parent / "20260916-ssb-mps-hotpath"))

from profile_mps import FIXTURE, bandwidth_control, open_fixture, prepared_of  # noqa: E402
from row_occupancy import CHUNK, GQK_COLS, launch, stat, variant_kernel  # noqa: E402

REPEATS = 5
WARMUP = 2
# 2 candidates x 512 bf x 512 col x 512 rows
COL_EVALS = 2 * 512 * 512 * 512

_PRELUDE = """
#define CADD(a, b) float2((a).x + (b).x, (a).y + (b).y)
#define CSUB(a, b) float2((a).x - (b).x, (a).y - (b).y)
#define CMUL(a, b) float2((a).x * (b).x - (a).y * (b).y, (a).x * (b).y + (a).y * (b).x)
#define CMULI(a) float2(-(a).y, (a).x)
#define W8_1(a) float2(0.70710678118654752f * ((a).x - (a).y), 0.70710678118654752f * ((a).x + (a).y))
#define W8_3(a) float2(0.70710678118654752f * (-(a).x - (a).y), 0.70710678118654752f * ((a).x - (a).y))
"""


def atan2_kernel(mx, iters: int, chains: int = 8):
    decl = "\n".join(
        f"        float a{c} = seed + {c + 1}.0f * 1.0e-3f;" for c in range(chains)
    )
    loop = "\n".join(
        f"        a{c} = metal::atan2(a{c} + k2, k1);" for c in range(chains)
    )
    tail = " + ".join(f"a{c}" for c in range(chains))
    source = f"""
        constexpr uint ITERS = {int(iters)}u;
        uint tid = thread_position_in_threadgroup.x;
        uint local_row = thread_position_in_threadgroup.y;
        uint row = thread_position_in_grid.y;
        uint bf = thread_position_in_grid.z;
        float seed = (float)(row * 131u + bf * 7u + tid) * 1.0e-6f;
        float k1 = 1.0000001f;
        float k2 = 1.0e-7f;
{decl}
        for (uint i = 0u; i < ITERS; ++i) {{
{loop}
        }}
        float acc = {tail};
        threadgroup float2 probe_shared[1][512];
        probe_shared[local_row][tid] = float2(acc, seed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u && local_row == 0u && bf == 0u) {{
            out[0].real = probe_shared[0][0].x;
            out[0].imag = probe_shared[0][0].y;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"ssb_atan2_probe_{iters}_{chains}",
        input_names=[], output_names=["out"], source=source,
        compile_options={"math_mode": "fast"},
    )


def column_alu_replica(mx, k_bf: int = 64):
    """The column stage's post-butterfly work: 8 atan2 + 16 accumulates/thread."""
    source = f"""
        constexpr uint K_BF = {int(k_bf)}u;
        uint tid = thread_position_in_threadgroup.x;
        uint local_col = thread_position_in_threadgroup.y;
        uint col = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        float sum0=0.0f, sum1=0.0f, sum2=0.0f, sum3=0.0f;
        float sq0=0.0f, sq1=0.0f, sq2=0.0f, sq3=0.0f;
        float seed = (float)(col * 13u + tid) * 1.0e-4f + (float)z;
        for (uint i = 0u; i < K_BF; ++i) {{
            float x0 = seed + (float)i * 1.0e-3f + (float)tid;
            float x1 = x0 + 1.0f; float x2 = x0 + 2.0f; float x3 = x0 + 3.0f;
            float p0 = metal::atan2(x0, x0 + 0.5f);
            float p1 = metal::atan2(x1, x1 + 0.5f);
            float p2 = metal::atan2(x2, x2 + 0.5f);
            float p3 = metal::atan2(x3, x3 + 0.5f);
            sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
            sq0 += p0 * p0; sq1 += p1 * p1; sq2 += p2 * p2; sq3 += p3 * p3;
        }}
        threadgroup float2 probe_shared[1][512];
        probe_shared[local_col][tid] = float2(sum0 + sum1 + sum2 + sum3,
                                             sq0 + sq1 + sq2 + sq3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u && local_col == 0u && z == 0u) {{
            out[0].real = probe_shared[0][0].x;
            out[0].imag = probe_shared[0][0].y;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"ssb_column_alu_replica_k{k_bf}",
        input_names=[], output_names=["out"], source=source,
        compile_options={"math_mode": "fast"},
    )


def timed(mx, thunk, repeats):
    for _ in range(WARMUP):
        mx.eval(thunk())
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mx.eval(thunk())
        ts.append(time.perf_counter() - t0)
    return ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--iters", type=int, default=256)
    ap.add_argument("--k-bf", type=int, default=64)
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine

    rec = {
        "probe": "column-roofline",
        "label": os.environ.get("GPU_RUN_LABEL", "unset"),
        "host": platform.node(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    opened, _ = open_fixture(FIXTURE)
    backend, prepared, _ = prepared_of(opened)
    rec["bandwidth_control_gbs"] = round(bandwidth_control(mx), 1)

    # ---- atan2 peak --------------------------------------------------------
    grid, tg = (64, 512, 512), (64, 4, 1)
    k_atan = atan2_kernel(mx, args.iters, 8)
    ts = timed(mx, lambda: k_atan(inputs=[], template=[], grid=grid, threadgroup=tg,
                                  output_shapes=[(1,)], output_dtypes=[mx.complex64])[0],
               args.repeats)
    total = 64 * 4 * 512 * 512 * args.iters * 8
    rec["atan2_throughput"] = stat("alu_atan2", ts, None, {
        "total_ops": int(total), "ops_per_s": float(total / float(np.median(ts)))})

    # ---- column ALU replica ------------------------------------------------
    k_col = column_alu_replica(mx, args.k_bf)
    cgrid, ctg = (64, 512, 16), (64, 8, 1)
    ts = timed(mx, lambda: k_col(inputs=[], template=[], grid=cgrid, threadgroup=ctg,
                                 output_shapes=[(1,)], output_dtypes=[mx.complex64])[0],
               args.repeats)
    n_atan = 64 * 8 * 512 * 16 * args.k_bf * 4
    rec["column_alu_replica"] = stat("column_alu_replica", ts, None,
                                     {"atan2_ops": int(n_atan), "k_bf": args.k_bf})

    # ---- row sweep, order balanced ----------------------------------------
    c10 = mx.array(np.asarray([7.017120839737006, -50.0], dtype=np.float32))
    c12 = mx.array(np.asarray([0.0, 35.0], dtype=np.float32))
    phi = np.asarray([-0.15393969519675116, -0.2], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))

    kernels = {r: variant_kernel(mx, engine, r) for r in (1, 2, 4)}
    moved = 2 * CHUNK * 2097152 + CHUNK * 512 * GQK_COLS * 8
    order = [4, 2, 1, 1, 2, 4]
    samples = {r: [] for r in kernels}
    for r in order:
        for _ in range(WARMUP):
            mx.eval(launch(mx, engine, kernels[r], prepared, r, c10, c12, cos2, sin2, 0, CHUNK))
        t0 = time.perf_counter()
        mx.eval(launch(mx, engine, kernels[r], prepared, r, c10, c12, cos2, sin2, 0, CHUNK))
        samples[r].append(time.perf_counter() - t0)
    rec["row_sweep_order_balanced"] = {
        str(r): stat(f"row_rpg{r}", samples[r], moved,
                     {"threadgroup_bytes": r * 2 * 512 * 8, "order": order})
        for r in kernels
    }

    # ---- column stage reference (real) ------------------------------------
    row_ifft = launch(mx, engine, kernels[4], prepared, 4, c10, c12, cos2, sin2, 0, CHUNK)
    active = mx.ones((CHUNK,), dtype=mx.uint8)

    def col_stage():
        return engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, row_ifft, k_bf=args.k_bf, active_bf=active,
            tiled_input=True, bf_start=0, bf_stop=CHUNK,
        )

    for _ in range(WARMUP):
        mx.eval(*col_stage())
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        mx.eval(*col_stage())
        ts.append(time.perf_counter() - t0)
    rec["column_stage_reference"] = stat(
        "column_stage_ref", ts, 2 * CHUNK * 2097152, {"k_bf": args.k_bf})

    peak = mx.get_active_memory() if hasattr(mx, "get_active_memory") else None
    rec["peak_active_bytes"] = int(peak) if peak is not None else None
    rec["load_average_end"] = list(os.getloadavg())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
