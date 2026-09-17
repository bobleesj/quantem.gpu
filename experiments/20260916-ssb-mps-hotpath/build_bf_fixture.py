"""One-time export of the exact BF columns of the real acquisition.

Writes the ShowPtycho BF-column companion (manifest.json + snapshots/cal.json
+ source/*.u16) so every later run opens the exact same evidence without a
19 GB dense decode. No bin, crop, or count change: the exported integers are
the exact selected detector columns for the calibrated Metal-probe selection.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SRC = Path(
    "/path/to/local/data/Live4DSTEM Testing/ARINA/"
    "arina-fixture-a_master.h5"
)
OUT = Path("/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a")
CENTER_ROW = 94.88451385498047
CENTER_COL = 96.35952758789062
BF_RADIUS = 53.35992814757164
EXCLUDED_PIXEL = 78 * 192 + 74
ROTATION_DEG = 158.88268568029937
SCAN_SAMPLING_A = 0.264
DET_SAMPLING_MRAD = 1.0
VOLTAGE_KV = 300.0
SEMIANGLE_MRAD = 30.0

_stop = threading.Event()


def _watch() -> None:
    peak = 0
    while not _stop.is_set():
        import subprocess

        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        try:
            rss = int(out or 0)
        except ValueError:
            rss = 0
        peak = max(peak, rss)
        _stop.wait(0.5)
    print(f"[watch] peak process RSS {peak / 1024**2:.2f} GiB", flush=True)


def calibrated_selection(detector_shape: tuple[int, int]):
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk

    rows, cols = np.mgrid[0 : detector_shape[0], 0 : detector_shape[1]]
    rows = rows.reshape(-1).astype(np.int32)
    cols = cols.reshape(-1).astype(np.int32)
    dist2 = (rows.astype(np.float32) - CENTER_ROW) ** 2 + (
        cols.astype(np.float32) - CENTER_COL
    ) ** 2
    keep = dist2 <= np.float32(BF_RADIUS) ** 2
    keep[EXCLUDED_PIXEL] = False
    return BrightfieldDisk(
        rows=rows[keep],
        cols=cols[keep],
        center_row_col=(CENTER_ROW, CENTER_COL),
        radius_px=BF_RADIUS,
        detected_radius_px=BF_RADIUS,
        detector_shape=detector_shape,
    )


def main() -> None:
    from quantem.gpu import SSB

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source").mkdir(exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    started = time.perf_counter()
    opened = SSB.open(
        str(SRC),
        backend="mps",
        voltage_kV=VOLTAGE_KV,
        semiangle_mrad=SEMIANGLE_MRAD,
        scan_sampling_A=SCAN_SAMPLING_A,
        det_sampling=DET_SAMPLING_MRAD,
        bf_radius=BF_RADIUS,
        rotation_angle_deg=ROTATION_DEG,
    )
    print(f"[open] {time.perf_counter() - started:.2f}s", flush=True)
    backend = opened._backend_protocol
    det_shape = tuple(int(v) for v in backend.detector_shape)
    selection = calibrated_selection(det_shape)
    print(
        f"[select] num_bf={selection.size} center={selection.center_row_col} "
        f"radius={selection.radius_px}",
        flush=True,
    )

    backend._selection = selection
    rows = selection.rows
    cols = selection.cols
    if backend._detector_sum is not None:
        selected_dc = backend._detector_sum[rows, cols]
        backend._dc_value_override = complex(
            np.complex64(np.asarray(selected_dc, dtype=np.float64).mean())
        )
    else:
        backend._dc_value_override = None

    stem = OUT / "source" / "bf_columns"
    export_started = time.perf_counter()
    written = backend.export_brightfield(backend._source_data, stem)
    print(f"[export] {time.perf_counter() - export_started:.2f}s -> {written}", flush=True)
    if written is None:
        raise SystemExit("export_brightfield declined this source")

    bf_path = Path(written[0])
    dtype = np.dtype(np.uint16) if bf_path.suffix == ".u16" else np.dtype(np.uint8)
    scan_shape = [int(backend.scan_shape[0]), int(backend.scan_shape[1])]
    det_list = [int(det_shape[0]), int(det_shape[1])]
    maximum = int(backend._bf_source_max_value)
    dc = backend._dc_value_override

    cal = {
        "detector_shape": det_list,
        "working_shape": det_list,
        "scan_region": {"shape": scan_shape},
        "bf_rows": [int(v) for v in rows],
        "bf_cols": [int(v) for v in cols],
        "bf_center": [CENTER_ROW, CENTER_COL],
        "bf_radius_px": BF_RADIUS,
        "detector_bin": 1,
        "bf_column_companion": True,
    }
    if dc is not None:
        cal["dc_value"] = [float(dc.real), float(dc.imag)]
    (OUT / "snapshots" / "cal.json").write_text(json.dumps(cal), encoding="utf-8")

    manifest = {
        "calibration": "snapshots/cal.json",
        "source": {
            "bf_columns": {
                "kind": "bf_columns",
                "order": "bf,scan",
                "path": f"source/{bf_path.name}",
                "encoding": "u16" if dtype == np.dtype(np.uint16) else "u8",
                "dtype": str(dtype),
                "shape": [int(rows.size), int(np.prod(scan_shape))],
                "scan_shape": scan_shape,
                "detector_shape": det_list,
                "detector_bin": 1,
                "bytes": int(bf_path.stat().st_size),
                "bytes_per_bf": int(np.prod(scan_shape)) * dtype.itemsize,
                "bits_per_value": int(dtype.itemsize * 8),
                "max_value": maximum,
            }
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(
        f"[done] {bf_path} {bf_path.stat().st_size / 1e9:.3f} GB max={maximum} "
        f"dc={dc}",
        flush=True,
    )
    _stop.set()
    watcher.join(timeout=2)


if __name__ == "__main__":
    main()
