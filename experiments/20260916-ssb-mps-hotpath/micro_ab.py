"""A/B the object-redraw chunk size and re-confirm the tiled scalar stage."""
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
print(f"prepare={prep_s:.1f}s bw={bandwidth_control(mx):.1f} GB/s g_qk={prepared.g_qk.nbytes/1e9:.3f}GB", flush=True)

print("--- object redraw chunk size (reads the whole 9.41 GB G cache) ---", flush=True)
ref = None
for chunk in (64, 128, 256, 512, 1024, 8937):
    fn = lambda: engine._object_fourier_sum_dynamic(
        prepared, C10=-50.0, C12=35.0, phi12=-0.2, chunk_bf=chunk)
    obj = fn(); mx.eval(obj)
    got = np.asarray(obj)
    dm = 0.0
    if ref is None:
        ref = got.copy()
    else:
        dm = float(np.max(np.abs(got - ref)))
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        o = fn(); mx.eval(o)
        ts.append(time.perf_counter() - t0)
    p50 = float(np.median(ts))
    print(json.dumps({
        "redraw_chunk_bf": chunk,
        "groups": (8937 + chunk - 1) // chunk,
        "p50_ms": round(p50 * 1e3, 2),
        "gb_per_s": round(prepared.g_qk.nbytes / p50 / 1e9, 1),
        "max_abs_diff_vs_chunk64": dm,
    }), flush=True)

print("--- tiled vs non-tiled scalar 512 phase/loss stage (512 planes) ---", flush=True)
k_bf = engine._default_phase_col_k_bf((512, 512))
c10s = mx.array([-50.0], dtype=mx.float32)
c12s = mx.array([35.0], dtype=mx.float32)
cos2s = mx.array([np.cos(-0.4)], dtype=mx.float32)
sin2s = mx.array([np.sin(-0.4)], dtype=mx.float32)
pk_all = engine._pk_batch_from_prepared(
    prepared, start=0, stop=8937, c10=c10s, c12=c12s,
    cos2phi12=cos2s, sin2phi12=sin2s)
mx.eval(pk_all)
active = (mx.abs(pk_all[0]) > 0.0).astype(mx.uint8)
mx.eval(active)

def scalar():
    row = engine._row_ifft512_from_dynamic_geometry(
        prepared, start=0, stop=512, c10=c10s, c12=c12s,
        cos2phi12=cos2s, sin2phi12=sin2s)
    s, sq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, row[None, ...], k_bf=k_bf, active_bf=active[:512], tiled_input=False)
    mx.eval(s, sq); return s, sq

def tiled():
    row = engine._row_ifft512_batch_from_dynamic_geometry(
        prepared, start=0, stop=512, c10=c10s, c12=c12s,
        cos2phi12=cos2s, sin2phi12=sin2s, pk_override=pk_all[:, :512],
        storage_bf=512)
    s, sq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, row, k_bf=k_bf, active_bf=active[:512], tiled_input=True,
        bf_start=0, bf_stop=512)
    mx.eval(s, sq); return s, sq

a = scalar(); b = tiled()
np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
out = {}
for name, fn in (("scalar_non_tiled", scalar), ("batched_tiled", tiled)):
    fn()
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    p50 = float(np.median(ts))
    out[name] = p50
    print(json.dumps({"stage": name, "p50_ms": round(p50 * 1e3, 2)}), flush=True)
print(json.dumps({"tiled_speedup": round(out["scalar_non_tiled"] / out["batched_tiled"], 3)}), flush=True)
print("bw_end", round(bandwidth_control(mx), 1), flush=True)
