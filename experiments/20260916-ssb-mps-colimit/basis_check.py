"""Put the frozen objective and the two isolated stages on one clock.

The row and column stage timings that priced the intermediate were taken in
sessions where the objective itself was not measured, so their sum
(39.2 ms/pack in one session) can exceed the in-situ per-pack objective time
(35.7 ms/pack from the frozen run).  This probe removes that ambiguity: it
times the frozen exact-pair objective and both stages in the same locked
session, so every share is a same-session difference and the isolated-stage
overhead is visible rather than assumed.
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
PINNED = (7.017120839737006, 0.0, -0.15393969519675116)
SECOND = (-50.0, 35.0, -0.2)
PACKS = 18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--k-bf", type=int, default=64)
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine

    rec = {
        "probe": "basis-check",
        "label": os.environ.get("GPU_RUN_LABEL", "unset"),
        "host": platform.node(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packs": PACKS,
    }

    opened, _ = open_fixture(FIXTURE)
    backend, prepared, _ = prepared_of(opened)
    rec["bandwidth_control_gbs"] = round(bandwidth_control(mx), 1)

    plane_bytes = 512 * 512 * 8
    half_plane_bytes = 512 * 257 * 8

    # ---- frozen objective --------------------------------------------------
    mx.reset_peak_memory()

    def pair_call():
        return engine._reconstruct_prepared_batch_exact_loss(
            prepared,
            C10=np.asarray([PINNED[0], SECOND[0]], dtype=np.float32),
            C12=np.asarray([PINNED[1], SECOND[1]], dtype=np.float32),
            phi12=np.asarray([PINNED[2], SECOND[2]], dtype=np.float32),
            chunk_bf=512,
        )

    for _ in range(WARMUP):
        mx.eval(pair_call())
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        mx.eval(pair_call())
        ts.append(time.perf_counter() - t0)
    losses = np.asarray(pair_call()).copy()
    modelled = PACKS * (2 * 512 * plane_bytes * 2 + 512 * half_plane_bytes)
    rec["pair_objective"] = stat("pair_objective", ts, modelled)
    rec["pair_objective"]["loss"] = [float(v) for v in losses]
    rec["pair_objective"]["loss_pinned_match"] = bool(
        float(losses[0]) == 0.13769753277301788)
    rec["pair_objective"]["peak_active_bytes"] = int(mx.get_peak_memory())
    rec["pair_objective"]["per_pack_ms"] = round(float(np.median(ts)) * 1e3 / PACKS, 3)

    # ---- isolated stages, same session ------------------------------------
    c10 = mx.array(np.asarray([PINNED[0], SECOND[0]], dtype=np.float32))
    c12 = mx.array(np.asarray([PINNED[1], SECOND[1]], dtype=np.float32))
    phi = np.asarray([PINNED[2], SECOND[2]], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))

    kernels = variant_kernel(mx, engine, 4)
    row_moved = 2 * CHUNK * plane_bytes + CHUNK * half_plane_bytes

    def row_call():
        return launch(mx, engine, kernels, prepared, 4, c10, c12, cos2, sin2, 0, CHUNK)

    for _ in range(WARMUP):
        mx.eval(row_call())
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        mx.eval(row_call())
        ts.append(time.perf_counter() - t0)
    rec["row_stage"] = stat("row_stage_isolated", ts, row_moved)

    row_ifft = row_call()
    mx.eval(row_ifft)
    active = mx.ones((CHUNK,), dtype=mx.uint8)

    def col_call():
        return engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, row_ifft, k_bf=args.k_bf, active_bf=active,
            tiled_input=True, bf_start=0, bf_stop=CHUNK)

    for _ in range(WARMUP):
        mx.eval(*col_call())
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        mx.eval(*col_call())
        ts.append(time.perf_counter() - t0)
    rec["column_stage"] = stat("column_stage_isolated", ts, 2 * CHUNK * plane_bytes,
                               {"k_bf": args.k_bf})

    iso_row = rec["row_stage"]["p50_ms"]
    iso_col = rec["column_stage"]["p50_ms"]
    per_pack = rec["pair_objective"]["per_pack_ms"]
    rec["accounting"] = {
        "isolated_row_plus_column_ms": round(iso_row + iso_col, 3),
        "insitu_objective_per_pack_ms": per_pack,
        "isolated_overhead_ms": round(iso_row + iso_col - per_pack, 3),
        "note": ("Isolated stage timings each pay a full launch + eval round "
                 "trip; the objective pipelines 18 packs inside one eval, so "
                 "the isolated sum is an upper bound on the in-situ stage cost."),
    }
    rec["load_average_end"] = list(os.getloadavg())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
