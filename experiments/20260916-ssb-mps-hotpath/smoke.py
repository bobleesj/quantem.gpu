"""Smoke: open the real 512x512 acquisition on MPS and report BF selection."""
from __future__ import annotations

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quantem.gpu import SSB

SRC = "/path/to/local/data/Live4DSTEM Testing/ARINA/arina-fixture-a_master.h5"

t0 = time.perf_counter()
opened = SSB.open(
    SRC,
    backend="mps",
    voltage_kV=300.0,
    semiangle_mrad=30.0,
    scan_sampling_A=0.264,
    bf_radius=53.35992814757164,
    rotation_angle_deg=158.88268568029937,
)
print("open seconds", round(time.perf_counter() - t0, 3), flush=True)
backend = opened._backend_protocol
sel = backend._selection
print("scan_shape", backend.scan_shape)
print("det_shape", backend.detector_shape)
print("num_bf", sel.size)
print("center", sel.center_row_col)
print("radius", sel.radius_px, "detected", sel.detected_radius_px)
print("has 78,74:", bool(np.any((sel.rows == 78) & (sel.cols == 74))))
