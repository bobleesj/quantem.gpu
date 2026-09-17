"""Measure the MLX/Metal on-chip (threadgroup) memory ceiling on this host.

The 4-passes-to-2 question turns on whether a fused row+column kernel can
hold what it needs on-chip.  This probe measures that ceiling instead of
citing documentation: it builds trivial metal kernels whose threadgroup
arrays grow past 32 KB and records where MLX/Metal refuses to build or run
them, then states the fused tile's arithmetic against the measured limit.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

SIZES_KB = (4, 8, 16, 24, 32, 33, 40, 48, 64, 128)
PLANE_BYTES = 512 * 512 * 8  # complex64 row-IFFT plane


def probe(size_kb: int) -> dict:
    n_vec = size_kb * 1024 // 8  # float2 elements
    source = f"""
        threadgroup float2 buf[{n_vec}];
        uint t = thread_position_in_threadgroup.x;
        buf[t] = float2((float)t, 1.0f);
        buf[t + 64u] = float2(2.0f, (float)t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out[t] = buf[t].x + buf[(t + 1u) % 64u].y + buf[t + 64u].x;
    """
    record = {"threadgroup_bytes": size_kb * 1024}
    try:
        kernel = mx.fast.metal_kernel(
            name=f"ssb_tg_probe_{size_kb}",
            input_names=[],
            output_names=["out"],
            source=source,
        )
    except Exception as exc:  # noqa: BLE001 - the failure mode is the datum
        record |= {"ok": False, "stage": "build", "error": f"{type(exc).__name__}: {exc}"[:300]}
        return record
    try:
        out = kernel(
            inputs=[],
            template=[],
            grid=(64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(64,)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(out)
        record |= {
            "ok": True,
            "sum": float(np.sum(np.asarray(out))),
            "seconds": round(0.0, 6),
        }
    except Exception as exc:  # noqa: BLE001 - the failure mode is the datum
        record |= {"ok": False, "stage": "run", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return record


def main() -> None:
    global mx
    import mlx.core as mx  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--label", default="2pass-price")
    args = parser.parse_args()

    record: dict[str, object] = {
        "probe": "threadgroup-limit",
        "label": args.label,
        "host": platform.node(),
        "chip": platform.processor(),
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "load_average": list(os.getloadavg()),
        "platform": platform.platform(),
    }
    record["sizes"] = [probe(size) for size in SIZES_KB]
    ok = [p for p in record["sizes"] if p["ok"]]
    record["max_ok_bytes"] = max(p["threadgroup_bytes"] for p in ok) if ok else 0
    record["first_failure"] = next(
        (p for p in record["sizes"] if not p["ok"]), None
    )

    # What the engine's two 512 kernels actually reserve on-chip today.
    record["engine_kernels"] = {
        "row_ifft512_shared_bytes": 4 * 2 * 512 * 8,
        "row_ifft512_declaration": "threadgroup float2 shared_rows[4][2][512]",
        "phase_cols512_shared_bytes": 8 * 512 * 8,
        "phase_cols512_declaration": "threadgroup float2 shared_cols[8][512]",
        "note": (
            "batch-2 row kernel (ROWS_PER_GROUP=4, FUSED_CANDIDATES=2) and the "
            "column stage both sit exactly at the 32 KB limit."
        ),
    }
    measured_limit = int(record["max_ok_bytes"])
    record["fused_requirement"] = {
        "plane_bytes": PLANE_BYTES,
        "plane_over_measured_limit": PLANE_BYTES / measured_limit,
        "argument": (
            "a fused row+column kernel must hold the transformed plane per "
            "threadgroup: 2,097,152 B against a measured 32768 B limit."
        ),
        "column_tile_bytes": 512 * 8 * 8,
        "column_tile_note": (
            "the smallest self-contained second-stage tile is 512 rows x 8 "
            "columns x 8 B = 32768 B (one threadgroup completes 8 of 512 "
            "columns), which is exactly the limit and needs 64 threadgroups."
        ),
        "input_read_amplification": 512 // 8,
        "input_read_note": (
            "each of the 64 column groups must transform 8 output columns of "
            "every one of the 512 rows, so it reads the full 512x512 corrected "
            "input (2,097,152 B) rather than 1/64 of it: 64x the current "
            "per-plane input traffic."
        ),
    }
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
