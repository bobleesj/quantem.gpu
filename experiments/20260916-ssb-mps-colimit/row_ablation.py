"""Timing-only ablations that finish the row stage's decomposition.

The ALU replica (alu_roofline.py) prices the row stage's per-element arithmetic
at 8.005 ms of a 20.52 ms stage.  The store ablation prices the plane write.
This probe closes the remaining buckets by removing, in isolation and in
combination, the G read and the correction block, so that everything left is
butterfly arithmetic, address math and barrier time.

Every variant produces wrong values by design; only the clock is meaningful.
The rewritten sites are asserted, so a source change cannot silently turn an
ablation into a no-op.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent.parent / "20260916-ssb-mps-hotpath"))

from profile_mps import FIXTURE, bandwidth_control, open_fixture, prepared_of  # noqa: E402
from row_occupancy import (  # noqa: E402
    CHUNK, GQK_COLS, INPUT_NAMES, _capture_source, launch, stat,
)

WARMUP = 2
PINNED = (7.017120839737006, 0.0, -0.15393969519675116)
SECOND = (-50.0, 35.0, -0.2)

CORR_START = "                float dx = qxv - kxv;"
CORR_END = "                if (ap_m == 0.0f && ap_p == 0.0f)"
G_START = "                size_t g_idx;"
G_END = "                float gi = mirror ? -gz.imag : gz.imag;"
STORE_NEEDLE = "(size_t)output_batch * (size_t)CHUNK + (size_t)bf)"

CORR_CONST = """                float ap_m = 1.0f;
                float ap_p = 1.0f;
                float alpha2_m = 0.0f;
                float alpha2_p = 0.0f;
                float cos2_m = 0.0f;
                float sin2_m = 0.0f;
                float cos2_p = 0.0f;
                float sin2_p = 0.0f;
"""
G_CONST = """                float gr = 1.0f;
                float gi = 0.0f;
"""


def build_variant(mx, engine, name: str, *, drop_corr: bool, drop_g: bool,
                  fold_store: bool):
    src = _capture_source(engine, 2, CHUNK, CHUNK)
    applied = []
    if drop_corr:
        i, j = src.index(CORR_START), src.index(CORR_END)
        src = src[:i] + CORR_CONST + src[j:]
        applied.append("drop_corr")
    if drop_g:
        i, j = src.index(G_START), src.index(G_END) + len(G_END)
        src = src[:i] + G_CONST + src[j:]
        applied.append("drop_g")
    if fold_store:
        if STORE_NEEDLE not in src:
            raise RuntimeError("store pattern changed; fold_store not built")
        src = src.replace(
            STORE_NEEDLE, "(size_t)output_batch * 4u + (size_t)(bf & 3u))")
        applied.append("fold_store")
    if not applied:
        applied = ["baseline"]
    return mx.fast.metal_kernel(
        name=f"ssb_row_abl_{name}",
        input_names=list(INPUT_NAMES),
        output_names=["row_ifft"],
        source=src,
        compile_options={"math_mode": "fast"},
    ), applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine

    rec = {
        "probe": "row-ablation",
        "label": os.environ.get("GPU_RUN_LABEL", "unset"),
        "host": platform.node(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    opened, _ = open_fixture(FIXTURE)
    backend, prepared, _ = prepared_of(opened)
    rec["bandwidth_control_gbs"] = round(bandwidth_control(mx), 1)

    c10 = mx.array(np.asarray([PINNED[0], SECOND[0]], dtype=np.float32))
    c12 = mx.array(np.asarray([PINNED[1], SECOND[1]], dtype=np.float32))
    phi = np.asarray([PINNED[2], SECOND[2]], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))

    specs = {
        "baseline": dict(drop_corr=False, drop_g=False, fold_store=False),
        "no_corr": dict(drop_corr=True, drop_g=False, fold_store=False),
        "no_g": dict(drop_corr=False, drop_g=True, fold_store=False),
        "no_g_no_corr": dict(drop_corr=True, drop_g=True, fold_store=False),
        "no_corr_fold_store": dict(drop_corr=True, drop_g=False, fold_store=True),
    }
    kernels, applied = {}, {}
    for name, spec in specs.items():
        kernels[name], applied[name] = build_variant(mx, engine, name, **spec)
    rec["variants_applied"] = applied

    moved = 2 * CHUNK * 2097152 + CHUNK * 512 * GQK_COLS * 8
    order = ["baseline", "no_corr", "no_g", "no_g_no_corr", "no_corr_fold_store",
             "baseline"]
    samples = {name: [] for name in specs}
    for name in order:
        for _ in range(WARMUP):
            mx.eval(launch(mx, engine, kernels[name], prepared, 4, c10, c12, cos2, sin2, 0, CHUNK))
        t0 = time.perf_counter()
        mx.eval(launch(mx, engine, kernels[name], prepared, 4, c10, c12, cos2, sin2, 0, CHUNK))
        samples[name].append(time.perf_counter() - t0)

    base = float(np.median(samples["baseline"]))
    rec["ablations"] = {}
    for name in specs:
        s = stat(f"row_{name}", samples[name], moved)
        s["delta_ms_vs_baseline"] = round(
            base * 1e3 - float(np.median(samples[name])) * 1e3, 3)
        s["share_of_baseline"] = round(
            (base - float(np.median(samples[name]))) / base, 4)
        s["order"] = order
        rec["ablations"][name] = s

    peak = mx.get_active_memory() if hasattr(mx, "get_active_memory") else None
    rec["peak_active_bytes"] = int(peak) if peak is not None else None
    rec["load_average_end"] = list(os.getloadavg())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
