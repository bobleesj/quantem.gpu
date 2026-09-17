"""A/B the exact scalar submission depth vs steady-state time and peak memory."""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import open_fixture, prepared_of, bandwidth_control
import mlx.core as mx
from quantem.gpu.ssb.compute.mps import engine

REPEATS = 7
DEPTHS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 4, 3, 2, 1]

opened, _ = open_fixture("/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a")
backend, prepared, prep_s = prepared_of(opened)
print(f"prepare={prep_s:.2f}s bw_control={bandwidth_control(mx):.1f} GB/s", flush=True)

c10 = np.asarray([-50.0], np.float32)
c12 = np.asarray([35.0], np.float32)
phi = np.asarray([-0.2], np.float32)
call = lambda: engine._reconstruct_prepared_batch_exact_loss(
    prepared, C10=c10, C12=c12, phi12=phi, chunk_bf=512)
b2 = lambda: engine._reconstruct_prepared_batch_exact_loss(
    prepared, C10=np.asarray([-50.0, 10.0], np.float32),
    C12=np.asarray([35.0, 20.0], np.float32),
    phi12=np.asarray([-0.2, 0.1], np.float32), chunk_bf=512)

reference = np.asarray(call()).copy()
print("reference loss", float(reference[0]), flush=True)

rows = []
for depth in DEPTHS:
    engine._EXACT_SCALAR_ROW_PACK_DEPTH_512 = depth
    mx.clear_cache()
    gc.collect()
    call(); call()  # warm
    mx.reset_peak_memory()
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = np.asarray(call()).copy()
        ts.append(time.perf_counter() - t0)
    np.testing.assert_array_equal(out, reference)
    peak1 = int(mx.get_peak_memory())

    mx.clear_cache()
    b2(); b2()
    mx.reset_peak_memory()
    ts2 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out2 = np.asarray(b2()).copy()
        ts2.append(time.perf_counter() - t0)
    peak2 = int(mx.get_peak_memory())
    res = {
        "depth": depth,
        "batch1_p50_ms": round(float(np.median(ts)) * 1e3, 3),
        "batch1_p95_ms": round(float(np.percentile(ts, 95)) * 1e3, 3),
        "batch1_peak_gb": round(peak1 / 1e9, 3),
        "batch2_p50_ms": round(float(np.median(ts2)) * 1e3, 3),
        "batch2_peak_gb": round(peak2 / 1e9, 3),
    }
    print(json.dumps(res), flush=True)
    rows.append(res)

engine._EXACT_SCALAR_ROW_PACK_DEPTH_512 = 5
print("bw_control_end", round(bandwidth_control(mx), 1), flush=True)
Path("/path/to/local/perf-lab/ssb-audit/mps-runs/20260916-ssb-mps-hotpath/depth_ab.json").write_text(json.dumps(rows, indent=1))
