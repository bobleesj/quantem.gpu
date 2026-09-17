"""Stage-isolated profiler for the MLX/MPS SSB path on the real acquisition.

Separates source prepare/upload, the fixed-pair objective, the phase-variance
loss, the 200-trial + Nelder-Mead fit, and repeated object redraws. Counts the
Python-level ``mx.eval`` sync points per stage so chunk-level serialization is
measurable rather than assumed. Writes one JSON record per stage.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", str(ROOT / "src")))
sys.path.insert(0, str(SRC))

FIXTURE = "/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a"
DET_SAMPLING_MRAD = 0.5622196476170719
ROTATION_DEG = 158.88268568029937
VOLTAGE_KV = 300.0
SEMIANGLE_MRAD = 30.0
SCAN_SAMPLING_A = 0.264


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", default=FIXTURE)
    p.add_argument("--stage", default="all",
                   choices=["prepare", "pair", "redraw", "fit", "all", "fast"])
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--label", default="baseline")
    p.add_argument("--reference-engine", type=Path)
    return p.parse_args()


class SyncCounter:
    """Count Python-level mlx sync entry points without changing behaviour."""

    def __init__(self) -> None:
        self.eval_calls = 0
        self.item_calls = 0
        self._saved = None

    def __enter__(self):
        import mlx.core as mx

        self._mx = mx
        self._saved_eval = mx.eval
        outer = self

        def counted_eval(*args):
            outer.eval_calls += 1
            return outer._saved_eval(*args)

        mx.eval = counted_eval
        return self

    def __exit__(self, *exc):
        self._mx.eval = self._saved_eval
        return False


def bandwidth_control(mx) -> float:
    """Streaming ceiling in GB/s, used to flag concurrent GPU contention."""
    n = 512 * 1024 * 1024 // 4
    a = mx.arange(n, dtype=mx.float32) % 7.0
    mx.eval(a)
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        mx.eval(mx.array(a))
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    del a
    return float(2 * n * 4 / best / 1e9)


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def open_fixture(fixture: str, det_sampling: float = DET_SAMPLING_MRAD):
    from quantem.gpu import SSB

    t0 = time.perf_counter()
    opened = SSB.open(
        fixture,
        backend="mps",
        calibration=fixture,
        voltage_kV=VOLTAGE_KV,
        semiangle_mrad=SEMIANGLE_MRAD,
        scan_sampling_A=SCAN_SAMPLING_A,
        det_sampling=det_sampling,
        rotation_angle_deg=ROTATION_DEG,
    )
    return opened, time.perf_counter() - t0


def prepared_of(opened):
    backend = opened._backend_protocol
    t0 = time.perf_counter()
    backend.cache_rotation(math.radians(ROTATION_DEG))
    return backend, backend._prepared, time.perf_counter() - t0


def report_times(name: str, times: list[float]) -> dict:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "stage": name,
        "n": int(arr.size),
        "p50_seconds": float(np.percentile(arr, 50)),
        "p95_seconds": float(np.percentile(arr, 95)),
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
        "mean_seconds": float(arr.mean()),
        "seconds": [float(v) for v in arr],
    }


def main() -> None:
    args = parse_args()
    import mlx.core as mx
    from quantem.gpu.ssb.compute.mps import engine, optimizer

    record: dict = {
        "schema": "quantem.ssb.mps.hotpath.v1",
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label,
        "stage_requested": args.stage,
        "trials": args.trials,
        "seed": args.seed,
        "repeats": args.repeats,
        "fixture": args.fixture,
        "det_sampling_mrad": DET_SAMPLING_MRAD,
        "rotation_deg": ROTATION_DEG,
        "precision": {"real": "float32", "complex": "complex64"},
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }

    opened, open_seconds = open_fixture(args.fixture)
    backend, prepared, prepare_seconds = prepared_of(opened)
    record["open_seconds"] = open_seconds
    record["prepare_seconds"] = prepare_seconds
    record["scan_shape"] = list(prepared.scan_shape)
    record["logical_bf"] = int(prepared.num_bf)
    record["stored_bf"] = int(prepared.g_qk.shape[0])
    record["g_qk_bytes"] = int(prepared.g_qk.nbytes)
    record["g_qk_shape"] = list(prepared.g_qk.shape)
    record["source_kind"] = opened.source_kind
    selection = backend._selection
    active = (
        int(np.count_nonzero(np.asarray(prepared.aperture_k_1d) > 0))
        if prepared.aperture_k_1d is not None
        else None
    )
    record["bf_center"] = list(selection.center_row_col)
    record["bf_radius_px"] = float(selection.radius_px)
    record["selected_bf"] = int(selection.size)
    record["active_bf"] = active

    stages: dict[str, object] = {}

    if args.stage in ("pair", "all", "fast"):
        c10 = np.asarray([-50.0], dtype=np.float32)
        c12 = np.asarray([35.0], dtype=np.float32)
        phi = np.asarray([-0.2], dtype=np.float32)

        def pair_call():
            return engine._reconstruct_prepared_batch_exact_loss(
                prepared, C10=c10, C12=c12, phi12=phi, chunk_bf=512
            )

        with SyncCounter() as counter:
            pair_call()
        first = np.asarray(pair_call()).copy()
        for _ in range(args.warmup):
            pair_call()
        times = []
        mx.reset_peak_memory()
        with SyncCounter() as counter:
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = np.asarray(pair_call()).copy()
                times.append(time.perf_counter() - t0)
                np.testing.assert_array_equal(out, first)
        pair_report = report_times("pair", times)
        pair_report["evals_after_first"] = counter.eval_calls
        pair_report["loss"] = float(first[0])
        pair_report["peak_active_bytes"] = int(mx.get_peak_memory())
        stages["pair"] = pair_report

        with SyncCounter() as counter:
            pair_call()
        stages["pair"]["evals_per_call"] = counter.eval_calls

    if args.stage in ("redraw", "all", "fast"):
        times = []
        with SyncCounter() as counter:
            for _ in range(args.warmup):
                obj = engine._object_fourier_sum_dynamic(
                    prepared,
                    C10=-50.0,
                    C12=35.0,
                    phi12=-0.2,
                    chunk_bf=engine._default_object_redraw_chunk_bf(),
                )
                mx.eval(obj)
            first_obj = np.asarray(
                engine._object_fourier_sum_dynamic(
                    prepared, C10=-50.0, C12=35.0, phi12=-0.2,
                    chunk_bf=engine._default_object_redraw_chunk_bf(),
                )
            ).copy()
            counter.eval_calls = 0
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                obj = engine._object_fourier_sum_dynamic(
                    prepared, C10=-50.0, C12=35.0, phi12=-0.2,
                    chunk_bf=engine._default_object_redraw_chunk_bf(),
                )
                mx.eval(obj)
                out = np.asarray(obj)
                times.append(time.perf_counter() - t0)
                np.testing.assert_array_equal(out, first_obj)
        redraw = report_times("redraw", times)
        redraw["evals_total"] = counter.eval_calls
        redraw["object_sha256"] = sha256_array(first_obj)
        stages["redraw"] = redraw

    if args.stage in ("fit", "all"):
        calls: list = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(optimizer_patch(prepared, selection))
            counter = stack.enter_context(SyncCounter())
            if args.stage in ("fit", "all"):
                stack.enter_context(objective_timer(calls))
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            result = optimizer.optimize(
                object(),
                voltage_kV=VOLTAGE_KV,
                semiangle_mrad=SEMIANGLE_MRAD,
                scan_sampling_A=SCAN_SAMPLING_A,
                det_sampling=DET_SAMPLING_MRAD,
                aberrations=None,
                n_trials=args.trials,
                refine="nelder-mead",
                seed=args.seed,
                verbose=False,
            )
            wall = time.perf_counter() - t0
        stages["fit"] = {
            "stage": "fit",
            "wall_seconds": wall,
            "elapsed_seconds": float(result.elapsed),
            "timings": {k: float(v) for k, v in result.timings.items()},
            "records": len(result.optuna_trials),
            "refine_nfev": int(result.refine_nfev),
            "aberrations": {k: float(v) for k, v in result.aberrations.items()},
            "loss": float(result.loss),
            "phase_sha256": sha256_array(result.phase),
            "object_sha256": sha256_array(result.object_wave),
            "peak_active_bytes": int(mx.get_peak_memory()),
            "evals_total": counter.eval_calls,
            "evals_per_objective": (
                counter.eval_calls / max(1, len(result.optuna_trials) + int(result.refine_nfev))
            ),
            "objective_calls": {
                "n": len(calls),
                "n_batch1": sum(1 for c in calls if c["batch"] == 1),
                "n_batch2": sum(1 for c in calls if c["batch"] == 2),
                "total_seconds": float(sum(c["seconds"] for c in calls)),
                "first_seconds": float(calls[0]["seconds"]) if calls else None,
                "batch1_seconds": [round(c["seconds"], 6) for c in calls if c["batch"] == 1],
                "batch2_seconds": [round(c["seconds"], 6) for c in calls if c["batch"] == 2],
                "p50_seconds": float(np.median([c["seconds"] for c in calls])) if calls else None,
            },
        }

    record["bandwidth_control_gbs"] = bandwidth_control(mx)
    record["stages"] = stages
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


@contextlib.contextmanager
def objective_timer(store: list):
    """Record every exact objective call the fit makes, with its batch size."""
    from quantem.gpu.ssb.backends.mps import engine as _engine
    from quantem.gpu.ssb.backends.mps import optimizer as _optimizer

    original = _engine._reconstruct_prepared_batch_exact_loss

    def timed(prepared, *, C10, C12, phi12, chunk_bf):
        batch = int(np.asarray(C10).size)
        t0 = time.perf_counter()
        out = original(prepared, C10=C10, C12=C12, phi12=phi12, chunk_bf=chunk_bf)
        store.append(
            {
                "batch": batch,
                "chunk_bf": int(chunk_bf),
                "seconds": time.perf_counter() - t0,
            }
        )
        return out

    _engine._reconstruct_prepared_batch_exact_loss = timed
    _optimizer._reconstruct_prepared_batch_exact_loss = timed
    try:
        yield
    finally:
        _engine._reconstruct_prepared_batch_exact_loss = original
        _optimizer._reconstruct_prepared_batch_exact_loss = original


@contextlib.contextmanager
def optimizer_patch(prepared, selection):
    """Drive the real optimizer against already-prepared evidence."""
    from quantem.gpu.ssb.backends.mps import optimizer

    patched = {
        "_as_chunked_frames": optimizer._as_chunked_frames,
        "_scan_shape": optimizer._scan_shape,
        "_resolve_bf_selection": optimizer._resolve_bf_selection,
        "_prepare_selection": optimizer._prepare_selection,
        "mean_dp": optimizer.mean_dp,
        "_default_object_redraw_chunk_bf": optimizer._default_object_redraw_chunk_bf,
    }
    optimizer._as_chunked_frames = lambda data: data
    optimizer._scan_shape = lambda frames: prepared.scan_shape
    optimizer._resolve_bf_selection = lambda *a, **k: selection
    optimizer._prepare_selection = lambda *a, **k: prepared
    optimizer.mean_dp = lambda frames: np.zeros(selection.detector_shape, np.float32)
    optimizer._default_object_redraw_chunk_bf = lambda: 128
    try:
        yield
    finally:
        for name, value in patched.items():
            setattr(optimizer, name, value)


if __name__ == "__main__":
    main()
