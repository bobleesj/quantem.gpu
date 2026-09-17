from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quantem.gpu import SSB

SRC = "/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a"
t0 = time.perf_counter()
opened = SSB.open(
    SRC, backend="mps", calibration=SRC,
    voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.264,
    rotation_angle_deg=158.88268568029937, det_sampling=1.0,
)
t_open = time.perf_counter() - t0
backend = opened._backend_protocol
print(f"open {t_open:.3f}s kind={opened.source_kind}")
sel = backend._selection
print("scan", backend.scan_shape, "num_bf", sel.size, "center", sel.center_row_col)
t1 = time.perf_counter()
backend.cache_rotation(np.radians(158.88268568029937))
t_prep = time.perf_counter() - t1
p = backend._prepared
print(f"prepare {t_prep:.3f}s stored_bf={int(p.g_qk.shape[0])} logical_bf={p.num_bf} "
      f"g_qk={tuple(p.g_qk.shape)} scan={p.scan_shape}")
import mlx.core as mx
mx.eval(p.g_qk)
print("g_qk bytes GiB", p.g_qk.nbytes / 1024**3)
print("active aperture terms", int(np.count_nonzero(np.asarray(p.aperture_k_1d > 0))) if p.aperture_k_1d is not None else None)
