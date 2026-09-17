"""Compare the non-tiled scalar 512 path with the tiled batched layout."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import open_fixture, prepared_of, bandwidth_control
import mlx.core as mx
from quantem.gpu.ssb.compute.mps import engine

REPEATS = 7

opened, _ = open_fixture("/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a")
backend, prepared, prep_s = prepared_of(opened)
print(f"prepare={prep_s:.2f}s bw_control={bandwidth_control(mx):.1f} GB/s", flush=True)

C10, C12, PHI = -50.0, 35.0, -0.2
k_bf = engine._default_phase_col_k_bf((512, 512))
c10s = mx.array([C10], dtype=mx.float32)
c12s = mx.array([C12], dtype=mx.float32)
cos2s = mx.array([np.cos(2.0 * PHI)], dtype=mx.float32)
sin2s = mx.array([np.sin(2.0 * PHI)], dtype=mx.float32)
active_full = mx.abs(engine._pk_batch_from_prepared(
    prepared, start=0, stop=8937, c10=c10s, c12=c12s,
    cos2phi12=cos2s, sin2phi12=sin2s)[0]) > 0.0
active_full = active_full.astype(mx.uint8)
mx.eval(active_full)
pk_all = engine._pk_batch_from_prepared(
    prepared, start=0, stop=8937, c10=c10s, c12=c12s,
    cos2phi12=cos2s, sin2phi12=sin2s)
mx.eval(pk_all)

START, STOP = 0, 512


def scalar_stage():
    row = engine._row_ifft512_from_dynamic_geometry(
        prepared, start=START, stop=STOP, c10=c10s, c12=c12s,
        cos2phi12=cos2s, sin2phi12=sin2s)
    s, sq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, row[None, ...], k_bf=k_bf, active_bf=active_full[START:STOP],
        tiled_input=False)
    mx.eval(s, sq)
    return s, sq


def tiled_stage():
    row = engine._row_ifft512_batch_from_dynamic_geometry(
        prepared, start=START, stop=STOP, c10=c10s, c12=c12s,
        cos2phi12=cos2s, sin2phi12=sin2s,
        pk_override=pk_all[:, START:STOP], storage_bf=STOP - START)
    s, sq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, row, k_bf=k_bf, active_bf=active_full[START:STOP],
        tiled_input=True, bf_start=0, bf_stop=STOP - START)
    mx.eval(s, sq)
    return s, sq


scalar_stage(); tiled_stage()
a_s, a_sq = scalar_stage()
b_s, b_sq = tiled_stage()
np.testing.assert_array_equal(np.asarray(a_s), np.asarray(b_s))
np.testing.assert_array_equal(np.asarray(a_sq), np.asarray(b_sq))
print("bit-exact: tiled scalar stage == non-tiled scalar stage", flush=True)

for name, fn in (("scalar_non_tiled", scalar_stage), ("batched_tiled", tiled_stage)):
    fn()
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    p50 = float(np.median(ts))
    print(json.dumps({
        "stage": name,
        "p50_ms": round(p50 * 1e3, 3),
        "p95_ms": round(float(np.percentile(ts, 95)) * 1e3, 3),
        "gb_moved": round((538 + 1074 + 1074) / 1e3, 3),
        "gb_per_s": round(2.686 / p50, 1),
    }), flush=True)

# whole 8937-plane phase/loss via both routes
def full_scalar():
    _o, loss, _p = engine._reconstruct_prepared(
        prepared, C10=C10, C12=C12, phi12=PHI, chunk_bf=512,
        compute_loss=True, compute_object=False, return_phase=False)
    return loss

full_scalar()
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    full_scalar()
    ts.append(time.perf_counter() - t0)
print(json.dumps({"stage": "full_phase_loss_scalar", "p50_ms": round(float(np.median(ts)) * 1e3, 3)}), flush=True)
