"""Count which reconstruction path the 8937-BF fit actually executes.

The exact optimizer has two 512 entry points: the batched row IFFT used by the
200-trial and Nelder-Mead objectives, and the scalar one reached only through
``_reconstruct_prepared`` (final phase/loss and redraw).  The accepted change
touches the scalar function, so this probe counts calls and time per entry
point during a small fit to show which of them the fit wall is made of.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import (  # noqa: E402
    FIXTURE,
    bandwidth_control,
    objective_timer,
    open_fixture,
    optimizer_patch,
    prepared_of,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--label", default="path")
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine as E
    from quantem.gpu.ssb.backends.mps import optimizer

    record: dict = {
        "schema": "quantem.ssb.mps.hotpath.v1",
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label,
        "trials": int(args.trials),
        "fixture": args.fixture,
        "precision": {"real": "float32", "complex": "complex64"},
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
    }

    stats: dict[str, dict] = {}
    originals = {
        "scalar_row_ifft": E._row_ifft512_from_dynamic_geometry,
        "batch_row_ifft": E._row_ifft512_batch_from_dynamic_geometry,
        "reconstruct_prepared": E._reconstruct_prepared,
        "batch_exact_loss": E._reconstruct_prepared_batch_exact_loss,
    }

    def wrap(name, fn):
        bucket = stats.setdefault(name, {"calls": 0, "seconds": 0.0})
        if getattr(fn, "__wrapped_by_probe__", False):
            return fn

        def counted(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                bucket["calls"] += 1
                bucket["seconds"] += time.perf_counter() - t0

        counted.__wrapped_by_probe__ = True
        return counted

    opened, open_seconds = open_fixture(args.fixture)
    backend, prepared, prepare_seconds = prepared_of(opened)
    selection = backend._selection
    record["open_seconds"] = open_seconds
    record["prepare_seconds"] = prepare_seconds
    record["g_qk_shape"] = list(prepared.g_qk.shape)
    record["logical_bf"] = int(prepared.num_bf)

    E._row_ifft512_from_dynamic_geometry = wrap("scalar_row_ifft", originals["scalar_row_ifft"])
    E._row_ifft512_batch_from_dynamic_geometry = wrap("batch_row_ifft", originals["batch_row_ifft"])
    E._reconstruct_prepared = wrap("reconstruct_prepared", originals["reconstruct_prepared"])
    E._reconstruct_prepared_batch_exact_loss = wrap(
        "batch_exact_loss", originals["batch_exact_loss"]
    )
    optimizer._reconstruct_prepared_batch_exact_loss = E._reconstruct_prepared_batch_exact_loss

    calls: list = []
    try:
        with optimizer_patch(prepared, selection), objective_timer(calls):
            t0 = time.perf_counter()
            result = optimizer.optimize(
                object(),
                voltage_kV=300.0,
                semiangle_mrad=30.0,
                scan_sampling_A=0.264,
                det_sampling=0.5622196476170719,
                aberrations=None,
                n_trials=int(args.trials),
                refine="nelder-mead",
                seed=42,
                verbose=False,
            )
            wall = time.perf_counter() - t0
    finally:
        for name, fn in originals.items():
            setattr(E, name, fn)
        optimizer._reconstruct_prepared_batch_exact_loss = originals["batch_exact_loss"]

    record["fit_wall_seconds"] = wall
    record["result_loss"] = float(result.loss)
    record["refine_nfev"] = int(result.refine_nfev)
    record["timings"] = {k: float(v) for k, v in result.timings.items()}
    record["path_calls"] = stats
    batch1 = [c for c in calls if c["batch"] == 1]
    batch2 = [c for c in calls if c["batch"] == 2]
    record["objective_calls"] = {
        "n": len(calls),
        "n_batch1": len(batch1),
        "n_batch2": len(batch2),
        "total_seconds": float(sum(c["seconds"] for c in calls)),
        "batch1_p50_ms": float(1000 * np.median([c["seconds"] for c in batch1])) if batch1 else None,
        "batch2_p50_ms": float(1000 * np.median([c["seconds"] for c in batch2])) if batch2 else None,
    }
    record["bandwidth_control_gbs"] = bandwidth_control(mx)

    line = json.dumps(record, sort_keys=True)
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
