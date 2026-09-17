"""Split one exact 512 objective into its real component costs."""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import open_fixture, prepared_of, report_times

import mlx.core as mx
from quantem.gpu.ssb.compute.mps import engine

FIXTURE = "/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a"
REPEATS = 7

opened, open_s = open_fixture(FIXTURE)
backend, prepared, prep_s = prepared_of(opened)
print(f"open={open_s:.3f}s prepare={prep_s:.3f}s stored_bf={prepared.g_qk.shape[0]}", flush=True)

c10 = mx.array([-50.0], dtype=mx.float32)
c12 = mx.array([35.0], dtype=mx.float32)
cos2phi = mx.array([np.cos(-0.4)], dtype=mx.float32)
sin2phi = mx.array([np.sin(-0.4)], dtype=mx.float32)
phi = np.asarray([-0.2], dtype=np.float32)

# --- component: pk over all 8937 planes
t0 = time.perf_counter()
pk_all = engine._pk_batch_from_prepared(
    prepared, start=0, stop=8937, c10=c10, c12=c12,
    cos2phi12=cos2phi, sin2phi12=sin2phi)
mx.eval(pk_all)
print(f"pk_all (8937 planes, batched trig) first: {time.perf_counter() - t0:.4f}s", flush=True)
mx.eval(pk_all)
ts = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    pk_all = engine._pk_batch_from_prepared(
        prepared, start=0, stop=8937, c10=c10, c12=c12,
        cos2phi12=cos2phi, sin2phi12=sin2phi)
    mx.eval(pk_all)
    ts.append(time.perf_counter() - t0)
print("pk_all warm p50", round(float(np.median(ts)) * 1e3, 3), "ms", flush=True)

# --- component: one 512-plane row IFFT
packs = list(engine._bf_storage_chunk_packs(
    prepared, 512, max_storage_bf=engine._exact_pair_row_policy_512(1)[0]))
print("pack count", len(packs), "sizes", [len(p) for p in packs][:5], "first", packs[0], flush=True)
start, stop = packs[0][0]
row = engine._row_ifft512_batch_from_dynamic_geometry(
    prepared, start=start, stop=stop, c10=c10, c12=c12,
    cos2phi12=cos2phi, sin2phi12=sin2phi,
    pk_override=pk_all[:, start:stop], storage_bf=stop - start)
mx.eval(row)
ts = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    row = engine._row_ifft512_batch_from_dynamic_geometry(
        prepared, start=start, stop=stop, c10=c10, c12=c12,
        cos2phi12=cos2phi, sin2phi12=sin2phi,
        pk_override=pk_all[:, start:stop], storage_bf=stop - start)
    mx.eval(row)
    ts.append(time.perf_counter() - t0)
row_p50 = float(np.median(ts))
row_bytes = (int(prepared.g_qk[start:stop].nbytes) + int(row.nbytes))
print(f"row_ifft 512-bf p50 {row_p50 * 1e3:.2f} ms  out={tuple(row.shape)} "
      f"bytes={row_bytes / 1e9:.3f} GB  {row_bytes / row_p50 / 1e9:.1f} GB/s", flush=True)

# --- component: one 512-plane column loss stage
active = (mx.abs(pk_all[0]) > 0.0).astype(mx.uint8)
mx.eval(active)
ts = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    s, sq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, row, k_bf=engine._default_phase_col_k_bf((512, 512)),
        active_bf=active, tiled_input=True, bf_start=0, bf_stop=stop - start)
    mx.eval(s, sq)
    ts.append(time.perf_counter() - t0)
col_p50 = float(np.median(ts))
col_bytes = int(row.nbytes)
print(f"phase_cols 512-bf p50 {col_p50 * 1e3:.2f} ms  "
      f"read={col_bytes / 1e9:.3f} GB  {col_bytes / col_p50 / 1e9:.1f} GB/s", flush=True)

# --- full objective
call = lambda: engine._reconstruct_prepared_batch_exact_loss(
    prepared, C10=phi * 0 + np.asarray([-50.0], np.float32),
    C12=np.asarray([35.0], np.float32), phi12=phi, chunk_bf=512)
mx.eval(call())
ts = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    out = call()
    ts.append(time.perf_counter() - t0)
full_p50 = float(np.median(ts))
print(f"full objective p50 {full_p50 * 1e3:.2f} ms", flush=True)
print(
    f"accounted: {len(packs)} packs x (row {row_p50 * 1e3:.2f} + col {col_p50 * 1e3:.2f}) "
    f"= {len(packs) * (row_p50 + col_p50) * 1e3:.1f} ms  "
    f"({100 * len(packs) * (row_p50 + col_p50) / full_p50:.1f}% of full)",
    flush=True,
)
print(f"peak mlx active {mx.get_peak_memory() / 1e9:.3f} GB", flush=True)
