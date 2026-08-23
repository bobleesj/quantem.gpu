#!/usr/bin/env python
"""Benchmark exact screening construction and prepared-result reopen.

This entry point keeps source construction and saved-result reopen as separate
timing distributions. It writes no raw-data path into the JSON artifact and
requires a new empty benchmark cache directory so existing user results cannot
be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from _benchmark_support import (
    _clear_backend,
    _git_state,
    _host_info,
    _memory_snapshot,
    _nearest_rank_summary,
    _stable_json_sha256,
    _sync_backend,
)

PRODUCT_FIELDS = (
    "mean_dp",
    "bright_field",
    "dark_field",
    "dpc_phase",
    "com_row",
    "com_col",
)
EXACT_INTEGER_PRODUCT_FIELDS = (
    "total_intensity",
    "annular_bright_field",
    "annular_dark_field",
)
FULL_PRODUCT_FIELDS = (
    "mean_dp",
    "total_intensity",
    "bright_field",
    "annular_bright_field",
    "annular_dark_field",
    "dark_field",
    "dpc_phase",
    "com_row",
    "com_col",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="One HDF5 master file.")
    parser.add_argument("--backend", required=True, choices=("cuda", "mps"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-state", default="unspecified")
    parser.add_argument("--fixture-id", default="anonymous-fixture")
    parser.add_argument("--source-sha256")
    parser.add_argument("--memory-budget-gb", type=float, required=True)
    parser.add_argument("--chunk-rows", type=int)
    parser.add_argument("--sample-positions", type=int, default=0)
    parser.add_argument("--rotation-steps", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--reopen-reps", type=int, default=7)
    parser.add_argument(
        "--require-exact-full-suite",
        action="store_true",
        help=(
            "Require and hash exact uint64 total/ABF/ADF in addition to the "
            "historical six screening products."
        ),
    )
    parser.add_argument("--reference-hashes-json", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise SystemExit(f"source must be an existing HDF5 master; got {args.source}")
    if args.reps < 1 or args.reopen_reps < 1 or args.warmup < 0:
        raise SystemExit(
            "reps and reopen-reps must be positive and warmup must be nonnegative; "
            f"got reps={args.reps}, reopen_reps={args.reopen_reps}, warmup={args.warmup}"
        )
    if args.memory_budget_gb <= 0:
        raise SystemExit(
            f"memory-budget-gb must be positive; got {args.memory_budget_gb}"
        )
    if args.chunk_rows is not None and args.chunk_rows <= 0:
        raise SystemExit(f"chunk-rows must be positive; got {args.chunk_rows}")
    if args.source_sha256 is not None:
        value = args.source_sha256.lower()
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise SystemExit(
                "source-sha256 must contain exactly 64 lowercase hex characters"
            )
    if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
        raise SystemExit(
            "cache-dir must be new or empty so benchmark output cannot overwrite "
            f"existing saved results; got {args.cache_dir}"
        )
    args.cache_dir.mkdir(parents=True, exist_ok=True)


def _product_arrays(
    result: Any,
    product_fields: tuple[str, ...] = FULL_PRODUCT_FIELDS,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in product_fields:
        value = getattr(result, name, None)
        if value is None:
            raise RuntimeError(
                "The requested screening path did not publish the full exact "
                f"product suite; missing {name}. Use the default exact uint16 "
                "CUDA preparation path."
            )
        arrays[name] = np.ascontiguousarray(value)
    expected_scan_shape = arrays["bright_field"].shape
    for name in product_fields:
        if name == "mean_dp":
            continue
        if arrays[name].shape != expected_scan_shape:
            raise RuntimeError(
                f"{name} has shape {arrays[name].shape}, expected scan shape "
                f"{expected_scan_shape}."
            )
    for name in EXACT_INTEGER_PRODUCT_FIELDS:
        if name not in arrays:
            continue
        if arrays[name].dtype != np.dtype(np.uint64):
            raise RuntimeError(
                f"{name} must preserve exact uint64 detector counts; "
                f"got {arrays[name].dtype}."
            )
    return arrays


def _product_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, array in arrays.items():
        hashes[name] = hashlib.sha256(array.view(np.uint8)).hexdigest()
    return hashes


def _product_specs(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
        }
        for name, array in arrays.items()
    }


def _reference_hashes(
    path: Path | None,
    product_fields: tuple[str, ...] = PRODUCT_FIELDS,
) -> dict[str, str] | None:
    if path is None:
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    missing = set(product_fields) - set(values)
    if missing:
        raise SystemExit(
            "reference-hashes-json must contain all screening products; missing "
            + ", ".join(sorted(missing))
        )
    return {name: str(values[name]) for name in product_fields}


def _cache_bytes(cache_dir: Path) -> int:
    return sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())


def _run_once(args: argparse.Namespace, *, refresh: bool) -> tuple[Any, dict[str, Any]]:
    from quantem.gpu.screening import prepare

    _clear_backend(args.backend)
    before = _memory_snapshot(args.backend)
    started = time.perf_counter()
    result = prepare(
        args.source,
        backend=args.backend,
        cache=True,
        cache_dir=args.cache_dir,
        refresh=refresh,
        memory_budget_gb=args.memory_budget_gb,
        chunk_rows=args.chunk_rows,
        sample_positions=args.sample_positions,
        rotation_steps=args.rotation_steps,
        verbose=False,
    )
    _sync_backend(args.backend)
    wall_seconds = time.perf_counter() - started
    after = _memory_snapshot(args.backend)
    product_fields = (
        FULL_PRODUCT_FIELDS if args.require_exact_full_suite else PRODUCT_FIELDS
    )
    arrays = _product_arrays(result, product_fields)
    hashes = _product_hashes(arrays)
    metadata = result.metadata
    record = {
        "wall_seconds": wall_seconds,
        "from_saved_result": bool(result.from_cache),
        "product_hashes": hashes,
        "product_specs": _product_specs(arrays),
        "product_hash_set_sha256": _stable_json_sha256(hashes),
        "probe_center_row": float(result.probe_center[0]),
        "probe_center_column": float(result.probe_center[1]),
        "probe_radius_pixels": float(result.probe_radius),
        "rotation_degrees": float(result.rotation_deg),
        "transposed": bool(result.transposed),
        "parameters": metadata.get("parameters"),
        "stage_timing": metadata.get("timing"),
        "memory_plan": metadata.get("memory"),
        "memory_before": before,
        "memory_after": after,
    }
    return result, record


def _parity_record(
    records: list[dict[str, Any]],
    reference: dict[str, str] | None,
) -> dict[str, Any]:
    observed = [record["product_hashes"] for record in records]
    repeat_exact = all(item == observed[0] for item in observed[1:])
    reference_exact = None
    if reference is not None:
        reference_exact = all(item == reference for item in observed)
    return {
        "repeat_hashes_exact": repeat_exact,
        "reference_hashes_supplied": reference is not None,
        "reference_hashes_exact": reference_exact,
        "accepted": bool(repeat_exact and reference_exact),
    }


def main() -> None:
    """Run screening build and saved-result reopen distributions.

    Examples
    --------
    Run a CUDA screening profile with a dedicated empty cache directory::

        python scripts/benchmark_screening.py master.h5 --backend cuda \
            --cache-dir run-cache --cache-state warm-source \
            --memory-budget-gb 6 --json-out screening.json
    """

    args = _parse_args()
    _validate_args(args)
    product_fields = (
        FULL_PRODUCT_FIELDS if args.require_exact_full_suite else PRODUCT_FIELDS
    )
    reference = _reference_hashes(args.reference_hashes_json, product_fields)

    for _ in range(args.warmup):
        result, _ = _run_once(args, refresh=True)
        del result
        _clear_backend(args.backend)

    build_records: list[dict[str, Any]] = []
    for trial in range(1, args.reps + 1):
        result, record = _run_once(args, refresh=True)
        record["trial"] = trial
        build_records.append(record)
        del result
        _clear_backend(args.backend)

    reopen_records: list[dict[str, Any]] = []
    for trial in range(1, args.reopen_reps + 1):
        result, record = _run_once(args, refresh=False)
        record["trial"] = trial
        reopen_records.append(record)
        del result
        _clear_backend(args.backend)

    report = {
        "schema": "quantem-gpu-screening-benchmark/v1",
        "benchmark_definition": {
            "build": "quantem.gpu.screening.prepare refresh through complete ScreeningResult publication",
            "reopen": "validated prepared ScreeningResult reopen through complete array materialization",
            "cache_state": args.cache_state,
            "warmup_count": args.warmup,
            "build_repetitions": args.reps,
            "reopen_repetitions": args.reopen_reps,
            "product_suite": (
                "exact-full"
                if args.require_exact_full_suite
                else "historical-six-product"
            ),
        },
        "code": _git_state(),
        "host": _host_info(),
        "backend": args.backend,
        "fixture_id": args.fixture_id,
        "source_sha256": args.source_sha256,
        "source_path_included": False,
        "parameters": {
            "memory_budget_gb": args.memory_budget_gb,
            "chunk_rows": args.chunk_rows,
            "sample_positions": args.sample_positions,
            "rotation_steps": args.rotation_steps,
        },
        "build": {
            "summary": _nearest_rank_summary(
                [record["wall_seconds"] for record in build_records]
            ),
            "parity": _parity_record(build_records, reference),
            "runs": build_records,
        },
        "prepared_reopen": {
            "summary": _nearest_rank_summary(
                [record["wall_seconds"] for record in reopen_records]
            ),
            "parity": _parity_record(reopen_records, reference),
            "runs": reopen_records,
        },
        "saved_result_bytes": _cache_bytes(args.cache_dir),
        "limitations": [
            "The caller-declared cache state is recorded but not controlled by this process.",
            "Build timing and prepared-result reopen are separate distributions.",
            "Reference parity is not accepted unless reference-hashes-json is supplied.",
            "Application loading, presentation, and interaction are excluded.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
