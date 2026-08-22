#!/usr/bin/env python
"""Benchmark QuantEM HDF5 load/decompress paths.

Maintainer tool for comparing CUDA and MPS load/decompression against browser
WebGPU profiles. Paths are anonymized in the output by default; pass
``--show-paths`` only for local provenance reports.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from _benchmark_support import (
    _clear_backend,
    _git_state,
    _host_info,
    _memory_snapshot,
    _nearest_rank_summary,
    _release_array,
    _sync_backend,
)


def _parse_region(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 4:
        raise SystemExit("--scan-region must be r0,r1,c0,c1")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("masters", nargs="+", help="HDF5 master files.")
    parser.add_argument(
        "--backend", default="cuda", choices=("cuda", "mps", "cpu", "auto")
    )
    parser.add_argument("--dtype", default=None, help="Optional browse dtype, e.g. u8.")
    parser.add_argument("--det-bin", type=int, default=1)
    parser.add_argument("--scan-region", type=_parse_region)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--cache-state",
        default="unspecified",
        help="Observed or controlled source/cache state; this script does not infer it.",
    )
    parser.add_argument("--fixture-id", default="anonymous-fixture")
    parser.add_argument("--source-sha256")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--show-paths", action="store_true")
    parser.add_argument("--skip-mps-memory-check", action="store_true")
    return parser.parse_args()


def _nbytes(data: Any) -> int | None:
    if hasattr(data, "nbytes"):
        return int(data.nbytes)
    chunks = getattr(data, "chunks", None)
    if chunks is not None:
        return int(sum(int(getattr(chunk, "nbytes", 0)) for chunk in chunks))
    return None


def _shape(data: Any) -> list[int] | None:
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    return [int(value) for value in shape]


def _load_once(master: Path, args: argparse.Namespace):
    from quantem.gpu.io import load

    kwargs: dict[str, Any] = {
        "backend": args.backend,
        "det_bin": args.det_bin,
        "verbose": False,
    }
    if args.dtype is not None:
        kwargs["dtype"] = args.dtype
    if args.scan_region is not None:
        kwargs["scan_region"] = args.scan_region
    override_key = "QUANTEM_GPU_MPS_SKIP_MEMORY_CHECK"
    override = args.skip_mps_memory_check and args.scan_region is None
    previous = os.environ.get(override_key)
    if override:
        os.environ[override_key] = "1"
    try:
        return load(str(master), **kwargs)
    finally:
        if override and previous is None:
            os.environ.pop(override_key, None)
        elif override:
            os.environ[override_key] = previous


def main() -> None:
    args = _parse_args()
    if args.reps < 1 or args.warmup < 0:
        raise SystemExit(
            f"reps must be positive and warmup nonnegative; got {args.reps}, {args.warmup}"
        )
    if args.source_sha256 is not None:
        value = args.source_sha256.lower()
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise SystemExit(
                "source-sha256 must contain exactly 64 lowercase hex characters"
            )
    masters = [Path(path).expanduser() for path in args.masters]
    rows: list[dict[str, Any]] = []

    for index, master in enumerate(masters):
        if not master.exists():
            raise SystemExit(f"missing master #{index + 1}")
        label = str(master) if args.show_paths else f"master-{index + 1}"
        for _ in range(max(0, args.warmup)):
            result = _load_once(master, args)
            _sync_backend(args.backend)
            _release_array(result.data)
            del result
            _clear_backend(args.backend)

        times: list[float] = []
        run_records: list[dict[str, Any]] = []
        last = None
        for trial in range(1, args.reps + 1):
            _clear_backend(args.backend)
            memory_before = _memory_snapshot(args.backend)
            t0 = time.perf_counter()
            result = _load_once(master, args)
            _sync_backend(args.backend)
            elapsed = time.perf_counter() - t0
            memory_after = _memory_snapshot(args.backend)
            shape = _shape(result.data)
            resident_bytes = _nbytes(result.data)
            release_method = _release_array(result.data)
            del result
            _clear_backend(args.backend)
            memory_after_release = _memory_snapshot(args.backend)
            times.append(elapsed)
            run_records.append(
                {
                    "trial": trial,
                    "wall_seconds": elapsed,
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                    "release_method": release_method,
                    "memory_after_release": memory_after_release,
                }
            )
            last = {
                "label": label,
                "backend": args.backend,
                "dtype": args.dtype or "native",
                "det_bin": args.det_bin,
                "scan_region": list(args.scan_region) if args.scan_region else None,
                "shape": shape,
                "resident_gb": None if resident_bytes is None else resident_bytes / 1e9,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "release_method": release_method,
                "memory_after_release": memory_after_release,
            }

        assert last is not None
        summary = _nearest_rank_summary(times)
        last.update(
            {
                "reps": len(times),
                "median_s": summary["p50_seconds"],
                "p50_s": summary["p50_seconds"],
                "p95_s": summary["p95_seconds"],
                "min_s": min(times),
                "max_s": summary["max_seconds"],
                "runs": run_records,
            }
        )
        rows.append(last)

    report = {
        "schema": "quantem-gpu-hdf5-load-benchmark/v3",
        "benchmark_definition": {
            "timing_boundary": "public io.load return after backend synchronization",
            "cache_state": args.cache_state,
            "warmup_count": args.warmup,
            "repetitions": args.reps,
        },
        "code": _git_state(),
        "host": _host_info(),
        "fixture_id": args.fixture_id,
        "source_sha256": args.source_sha256,
        "source_paths_included": bool(args.show_paths),
        "rows": rows,
        "limitations": [
            "The caller-declared cache state is recorded but not controlled by this process.",
            "Output parity and application wall time require separate retained artifacts.",
        ],
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
