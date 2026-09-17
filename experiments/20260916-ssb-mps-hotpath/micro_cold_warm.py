"""Time the first exact objective call versus steady state, in one process."""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import open_fixture, prepared_of, DET_SAMPLING_MRAD

import mlx.core as mx
from quantem.gpu.ssb.compute.mps import engine

opened, open_s = open_fixture("/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a")
backend, prepared, prep_s = prepared_of(opened)
print(f"open={open_s:.3f}s prepare={prep_s:.3f}s", flush=True)

c10 = np.asarray([-50.0], dtype=np.float32)
c12 = np.asarray([35.0], dtype=np.float32)
phi = np.asarray([-0.2], dtype=np.float32)
call = lambda: engine._reconstruct_prepared_batch_exact_loss(
    prepared, C10=c10, C12=c12, phi12=phi, chunk_bf=512)

for i in range(6):
    t0 = time.perf_counter()
    out = call()
    dt = time.perf_counter() - t0
    print(f"call {i}: {dt:.4f}s loss={float(np.asarray(out)[0]):.9f}", flush=True)

# same call, second rotation-independent repeat after a cache clear
mx.clear_cache()
t0 = time.perf_counter()
call()
print(f"after clear_cache: {time.perf_counter() - t0:.4f}s", flush=True)
