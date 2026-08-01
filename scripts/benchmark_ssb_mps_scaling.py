#!/usr/bin/env python
"""Benchmark exact MPS SSB scaling with reproducible JSON output.

The native source size is the scientist-workflow measurement. Other requested
sizes are optional Fourier-resized kernel microscopes and are labeled as such.
They preserve BF evidence and float32/complex64 computation, but they are not
substitutes for native acquisitions at those scan sizes.

Each record separates the prepared-evidence fit from warm source-open through
fit wall time. Repeated pair and parity probes run after both measurements so
they cannot thermally bias the primary workflow timing.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from quantem.gpu import SSB
from quantem.gpu.ssb.compute.mps import engine, optimizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact MPS SSB pair and complete-fit scaling."
    )
    parser.add_argument("source", type=Path, help="Native 4D-STEM source.")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--voltage-kv", type=float, default=300.0)
    parser.add_argument("--semiangle-mrad", type=float, default=30.0)
    parser.add_argument("--scan-sampling-a", type=float, required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-warmup", type=int, default=2)
    parser.add_argument("--pair-iters", type=int, default=12)
    parser.add_argument("--parity-pairs", type=int, default=20)
    parser.add_argument(
        "--reference-engine",
        type=Path,
        help="Frozen engine.py used for randomized bit-exact A/B parity.",
    )
    parser.add_argument(
        "--allow-fourier-resize",
        action="store_true",
        help="Permit non-native kernel microscopes by Fourier crop/zero-pad.",
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Run fixed pair/parity measurements without the adaptive fit.",
    )
    parser.add_argument("--jsonl-out", type=Path)
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help=(
            "Include local source/calibration paths and the calibration hash; "
            "omit this flag for shareable output."
        ),
    )
    return parser.parse_args()


def _sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_revision() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _resize_half_spectrum(g_qk, size: int):
    old_size = int(g_qk.shape[1])
    old_cols = int(g_qk.shape[2])
    new_cols = size // 2 + 1
    if size == old_size:
        return g_qk
    if size < old_size:
        half = size // 2
        resized = mx.concatenate(
            [g_qk[:, :half, :new_cols], g_qk[:, -half:, :new_cols]],
            axis=1,
        )
    else:
        half = old_size // 2
        resized = mx.concatenate(
            [
                g_qk[:, :half],
                mx.zeros(
                    (int(g_qk.shape[0]), size - old_size, old_cols),
                    dtype=mx.complex64,
                ),
                g_qk[:, -half:],
            ],
            axis=1,
        )
        resized = mx.concatenate(
            [
                resized,
                mx.zeros(
                    (int(g_qk.shape[0]), size, new_cols - old_cols),
                    dtype=mx.complex64,
                ),
            ],
            axis=2,
        )
    resized = mx.contiguous(resized)
    mx.eval(resized)
    return resized


def _resized_prepared(prepared, size: int, scan_sampling_a: float):
    native_size = int(prepared.scan_shape[0])
    if size == native_size:
        return prepared
    q_row_np, q_col_np = engine._spatial_frequencies(
        (size, size),
        (scan_sampling_a, scan_sampling_a),
    )
    q_row = mx.array(q_row_np, dtype=mx.float32)
    q_col = mx.array(q_col_np, dtype=mx.float32)
    dc_mask = np.zeros((size, size), dtype=bool)
    dc_mask[0, 0] = True
    resized = dataclasses.replace(
        prepared,
        g_qk=_resize_half_spectrum(prepared.g_qk, size),
        qx=q_row[None, :, None],
        qy=q_col[None, None, :],
        q_row=q_row,
        q_col=q_col,
        scan_shape=(size, size),
        dc_mask=mx.array(dc_mask),
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )
    mx.eval(resized.g_qk, resized.q_row, resized.q_col, resized.dc_mask)
    return resized


@contextlib.contextmanager
def _prepared_optimizer(prepared, selection) -> Iterator[None]:
    patched = {
        "_as_chunked_frames": optimizer._as_chunked_frames,
        "_scan_shape": optimizer._scan_shape,
        "_resolve_bf_selection": optimizer._resolve_bf_selection,
        "_prepare_selection": optimizer._prepare_selection,
        "mean_dp": optimizer.mean_dp,
    }
    optimizer._as_chunked_frames = lambda data: data
    optimizer._scan_shape = lambda frames: prepared.scan_shape
    optimizer._resolve_bf_selection = lambda *args, **kwargs: selection
    optimizer._prepare_selection = lambda *args, **kwargs: prepared
    optimizer.mean_dp = lambda frames: np.zeros(selection.detector_shape, np.float32)
    try:
        yield
    finally:
        for name, value in patched.items():
            setattr(optimizer, name, value)


def _load_reference_engine(path: Path | None):
    if path is None:
        return None
    name = "quantem.gpu.ssb.compute.mps.engine_benchmark_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reference engine from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pair(module, prepared, c10: np.ndarray, c12: np.ndarray, phi12: np.ndarray):
    return module._reconstruct_prepared_batch_exact_loss(
        prepared,
        C10=c10,
        C12=c12,
        phi12=phi12,
        chunk_bf=512,
    )


def _pair_report(
    prepared,
    *,
    warmup: int,
    iterations: int,
    parity_pairs: int,
    reference_engine,
):
    c10 = np.asarray([-50.0, 35.0], dtype=np.float32)
    c12 = np.asarray([20.0, 45.0], dtype=np.float32)
    phi12 = np.asarray([-0.2, 0.3], dtype=np.float32)
    fixed = np.asarray(_pair(engine, prepared, c10, c12, phi12)).copy()
    for _ in range(max(0, warmup)):
        _pair(engine, prepared, c10, c12, phi12)
    timings = []
    mx.reset_peak_memory()
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        actual = np.asarray(_pair(engine, prepared, c10, c12, phi12)).copy()
        timings.append(time.perf_counter() - started)
        np.testing.assert_array_equal(actual, fixed)

    checked_parity_pairs = None
    if reference_engine is not None:
        rng = np.random.default_rng(20260801 + int(prepared.scan_shape[0]))
        for _ in range(max(0, parity_pairs)):
            random_c10 = rng.uniform(-150.0, 150.0, size=2).astype(np.float32)
            random_c12 = rng.uniform(0.0, 100.0, size=2).astype(np.float32)
            random_phi = rng.uniform(-np.pi, np.pi, size=2).astype(np.float32)
            actual = np.asarray(
                _pair(engine, prepared, random_c10, random_c12, random_phi)
            )
            expected = np.asarray(
                _pair(
                    reference_engine,
                    prepared,
                    random_c10,
                    random_c12,
                    random_phi,
                )
            )
            np.testing.assert_array_equal(actual, expected)
        checked_parity_pairs = max(0, parity_pairs)

    return {
        "fixed_losses": fixed.tolist(),
        "iterations": len(timings),
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "p95_seconds": float(np.percentile(timings, 95)),
        "parity_pairs_bit_exact": checked_parity_pairs,
        "peak_active_bytes": int(mx.get_peak_memory()),
    }


def _fit_report(prepared, selection, args: argparse.Namespace):
    mx.reset_peak_memory()
    started = time.perf_counter()
    with _prepared_optimizer(prepared, selection):
        result = optimizer.optimize(
            object(),
            voltage_kV=args.voltage_kv,
            semiangle_mrad=args.semiangle_mrad,
            scan_sampling_A=args.scan_sampling_a,
            det_sampling=None,
            aberrations=None,
            n_trials=args.trials,
            refine="nelder-mead",
            seed=args.seed,
            verbose=False,
        )
    return {
        "wall_seconds": time.perf_counter() - started,
        "elapsed_seconds": result.elapsed,
        "timings": result.timings,
        "records": len(result.optuna_trials),
        "refine_nfev": result.refine_nfev,
        "aberrations": result.aberrations,
        "loss": result.loss,
        "phase_sha256": _sha256(result.phase),
        "object_sha256": _sha256(result.object_wave),
        "peak_active_bytes": int(mx.get_peak_memory()),
    }


def main() -> None:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    calibration = (
        args.calibration.expanduser().resolve() if args.calibration else None
    )
    if not source.exists():
        raise SystemExit("source does not exist")
    if calibration is not None and not calibration.exists():
        raise SystemExit("calibration does not exist")
    reference_path = (
        args.reference_engine.expanduser().resolve()
        if args.reference_engine
        else None
    )
    if reference_path is not None and not reference_path.exists():
        raise SystemExit("reference engine does not exist")
    reference_engine = _load_reference_engine(reference_path)

    opened = SSB.open(
        source,
        backend="mps",
        calibration=str(calibration) if calibration else None,
        voltage_kV=args.voltage_kv,
        semiangle_mrad=args.semiangle_mrad,
        scan_sampling_A=args.scan_sampling_a,
    )
    backend = opened._backend_protocol
    backend.cache_rotation(0.0)
    native_prepared = backend._prepared
    selection = backend._selection
    native_size = int(native_prepared.scan_shape[0])
    requested_sizes = list(dict.fromkeys(args.sizes))
    if any(size != native_size for size in requested_sizes) and not args.allow_fourier_resize:
        raise SystemExit(
            "non-native sizes require --allow-fourier-resize so their "
            "kernel-microscope status is explicit"
        )

    machine = {
        "platform": platform.platform(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "physical_memory_bytes": (
            int(value) if (value := _sysctl("hw.memsize")) else None
        ),
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "git": _git_revision(),
    }
    provenance = {
        "source": str(source) if args.show_paths else None,
        "source_size_bytes": source.stat().st_size,
        "calibration": (
            str(calibration) if args.show_paths and calibration else None
        ),
        "calibration_sha256": (
            _file_sha256(calibration) if args.show_paths else None
        ),
        "native_scan_shape": list(native_prepared.scan_shape),
        "logical_bf": int(native_prepared.num_bf),
        "stored_bf": int(native_prepared.g_qk.shape[0]),
    }
    backend._prepared = None
    opened = None
    backend = None
    selection = None
    native_prepared = None
    gc.collect()
    mx.clear_cache()

    for size in requested_sizes:
        workflow_started = time.perf_counter()
        opened = SSB.open(
            source,
            backend="mps",
            calibration=str(calibration) if calibration else None,
            voltage_kV=args.voltage_kv,
            semiangle_mrad=args.semiangle_mrad,
            scan_sampling_A=args.scan_sampling_a,
        )
        backend = opened._backend_protocol
        backend.cache_rotation(0.0)
        native_prepared = backend._prepared
        selection = backend._selection
        prepared = _resized_prepared(native_prepared, size, args.scan_sampling_a)
        evidence = "native" if size == native_size else "fourier_resized_microscope"
        if size != native_size:
            backend._prepared = None
            opened = None
            backend = None
            native_prepared = None
            gc.collect()
            mx.clear_cache()
        setup_wall_seconds = time.perf_counter() - workflow_started
        # The complete scientist workflow is the primary benchmark. Run it
        # before the repeated pair/parity probes so those probes do not
        # thermally bias the end-to-end timing, especially at 1024x1024.
        fit_report = (
            None if args.skip_fit else _fit_report(prepared, selection, args)
        )
        workflow_wall_seconds = time.perf_counter() - workflow_started
        pair_report = _pair_report(
            prepared,
            warmup=args.pair_warmup,
            iterations=args.pair_iters,
            parity_pairs=args.parity_pairs,
            reference_engine=reference_engine,
        )
        record = {
            "schema": "quantem.ssb.mps.scaling.v1",
            "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "size": size,
            "evidence": evidence,
            "timing_scope": {
                "fit": "prepared_evidence_optimization",
                "workflow_wall": "warm_source_open_through_fit",
                "pair": "post_fit_sustained",
                "order": ["setup", "fit", "pair", "parity"],
            },
            "setup_wall_seconds": setup_wall_seconds,
            "workflow_wall_seconds": workflow_wall_seconds,
            "precision": {"real": "float32", "complex": "complex64"},
            "reference_engine_sha256": _file_sha256(reference_path),
            "optimizer": {
                "trials": args.trials,
                "seed": args.seed,
                "refine": "nelder-mead",
            },
            "machine": machine,
            "provenance": provenance,
            "pair": pair_report,
            "fit": fit_report,
        }
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if args.jsonl_out:
            args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
            with args.jsonl_out.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

        if size == native_size:
            backend._prepared = None
        opened = None
        backend = None
        native_prepared = None
        prepared = None
        selection = None
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
