"""Correlate the first objective call's cost with allocation/system state."""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import open_fixture, prepared_of
import mlx.core as mx
from quantem.gpu.ssb.compute.mps import engine


def snap() -> str:
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    vals = {}
    for line in vm.splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            vals[k.strip()] = int(v.strip().rstrip(".").lstrip() or 0)
    page = 16384
    free = (vals.get("Pages free", 0) + vals.get("Pages inactive", 0)) * page / 1e9
    return f"free+inactive={free:.2f}GB swap=[{swap}]"


opened, _ = open_fixture("/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a")
backend, prepared, prep_s = prepared_of(opened)
print(f"prepare={prep_s:.3f}s {snap()}", flush=True)

c10 = np.asarray([-50.0], np.float32)
c12 = np.asarray([35.0], np.float32)
phi = np.asarray([-0.2], np.float32)
call = lambda: engine._reconstruct_prepared_batch_exact_loss(
    prepared, C10=c10, C12=c12, phi12=phi, chunk_bf=512)

mx.reset_peak_memory()
for i in range(4):
    t0 = time.perf_counter()
    out = call()
    dt = time.perf_counter() - t0
    print(f"call {i}: {dt:.4f}s peak={mx.get_peak_memory() / 1e9:.3f}GB {snap()}", flush=True)
