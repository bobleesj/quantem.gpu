"""Measure the cost of the per-chunk mx.eval sync in the scalar 512 loss path.

The eval in `_reconstruct_prepared`'s chunk loop only synchronizes; it does
not change arithmetic. Suppressing every k-th eval is therefore bit-exact and
isolates the cost of the sync against the cost of the extra live buffers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import (  # noqa: E402
    FIXTURE,
    bandwidth_control,
    open_fixture,
    prepared_of,
    sha256_array,
)

C10, C12, PHI = 7.017120839737006, 0.0, -0.15393969519675116
REPEATS = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="eval-depth")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine

    opened, _ = open_fixture(FIXTURE)
    _backend, prepared, prepare_seconds = prepared_of(opened)
    record = {"schema": "quantem.ssb.mps.evaldepth.v1", "label": args.label,
              "prepare_seconds": round(prepare_seconds, 3),
              "bw_control_start_gbs": round(bandwidth_control(mx), 1)}

    def run(k: int):
        real_eval = mx.eval
        calls = {"outer": 0, "inner": 0}

        def wrapper(*arrays):
            calls["inner"] += 1
            if calls["inner"] % k == 0:
                return real_eval(*arrays)
            return None

        mx.eval = wrapper
        try:
            _obj, loss, phase = engine._reconstruct_prepared(
                prepared, C10=C10, C12=C12, phi12=PHI, chunk_bf=512,
                compute_loss=True, compute_object=False, return_phase=True)
        finally:
            mx.eval = real_eval
        return loss, phase, calls["inner"]

    record["ks"] = {}
    for k in (1, 2, 3, 4):
        mx.reset_peak_memory()
        loss, phase, inner = run(k)
        peak = int(mx.get_peak_memory())
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            run(k)
            times.append(time.perf_counter() - t0)
        record["ks"][str(k)] = {
            "eval_calls_per_call": inner,
            "p50_ms": round(float(np.median(times)) * 1e3, 3),
            "min_ms": round(float(min(times)) * 1e3, 3),
            "peak_active_bytes": peak,
            "loss_hex": float(loss).hex(),
            "phase_sha256": sha256_array(phase),
            "seconds": [round(float(v), 6) for v in times],
        }
        print(f"k={k} p50={record['ks'][str(k)]['p50_ms']} evals={inner} "
              f"peak={peak/1e9:.3f}GB", flush=True)

    record["bw_control_end_gbs"] = round(bandwidth_control(mx), 1)
    line = json.dumps(record, sort_keys=True, default=str)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
