#!/usr/bin/env python
"""Benchmark evidence-selective CUDA or MPS scan-position loading.

Selectors are read from a JSON file so row/column order, duplicates, expected
shape, and an optional full-output reference hash remain immutable across
backends. The timed boundary ends after synchronized exact resident output;
full-output hashing follows the boundary and is reported separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _benchmark_support import (
    _array_description,
    _array_sha256,
    _clear_backend,
    _git_state,
    _host_info,
    _memory_snapshot,
    _nearest_rank_summary,
    _release_array,
    _stable_json_sha256,
    _sync_backend,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="One HDF5 master file.")
    parser.add_argument("--backend", required=True, choices=("cuda", "mps"))
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--det-bin", type=int, default=1)
    parser.add_argument("--cache-state", default="unspecified")
    parser.add_argument("--fixture-id", default="anonymous-fixture")
    parser.add_argument("--source-sha256")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _load_selector(path: Path) -> dict[str, Any]:
    selector = json.loads(path.read_text(encoding="utf-8"))
    kind = selector.get("kind")
    if kind == "scan_region":
        region = selector.get("scan_region")
        if not isinstance(region, list) or len(region) != 4:
            raise SystemExit(
                "scan_region selector must contain [row_start, row_stop, "
                f"column_start, column_stop]; got {region!r}"
            )
        row_start, row_stop, column_start, column_stop = map(int, region)
        if (
            row_start < 0
            or column_start < 0
            or row_stop <= row_start
            or column_stop <= column_start
        ):
            raise SystemExit(
                f"scan_region bounds must be ordered and nonnegative; got {region!r}"
            )
        selector["scan_region"] = [row_start, row_stop, column_start, column_stop]
    elif kind == "scan_indices":
        positions = selector.get("scan_indices")
        if not isinstance(positions, list) or not positions:
            raise SystemExit("scan_indices selector must contain at least one position")
        normalized: list[int | list[int]] = []
        for position in positions:
            if isinstance(position, int):
                normalized.append(position)
            elif isinstance(position, list) and len(position) == 2:
                normalized.append([int(position[0]), int(position[1])])
            else:
                raise SystemExit(
                    "scan_indices positions must be flat integers or [row, column] "
                    f"pairs; got {position!r}"
                )
        selector["scan_indices"] = normalized
    else:
        raise SystemExit(
            f"selector kind must be 'scan_region' or 'scan_indices'; got {kind!r}"
        )
    expected_hash = selector.get("expected_output_sha256")
    if expected_hash is not None and (
        len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise SystemExit(
            "expected_output_sha256 must contain exactly 64 lowercase hex characters"
        )
    return selector


def _validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise SystemExit(f"source must be an existing HDF5 master; got {args.source}")
    if not args.selector.is_file():
        raise SystemExit(f"selector must be an existing JSON file; got {args.selector}")
    if args.det_bin < 1:
        raise SystemExit(f"det-bin must be positive; got {args.det_bin}")
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


def _selector_kwargs(selector: dict[str, Any]) -> dict[str, Any]:
    if selector["kind"] == "scan_region":
        return {"scan_region": tuple(selector["scan_region"])}
    return {"scan_indices": selector["scan_indices"]}


def _metadata_record(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "backend",
        "full_scan_shape",
        "scan_shape",
        "scan_region",
        "scan_order",
        "n_frames",
        "scan_indices",
        "scan_positions",
        "unique_frame_count",
        "duplicate_frame_count",
        "read_order",
        "det_bin",
        "decoded_detector_shape",
        "output_detector_shape",
        "prepare_seconds",
        "decode_seconds",
        "load_seconds",
        "total_compressed_bytes",
        "read_span_count",
        "read_gap_bytes",
        "prepare_timing_s",
    )
    return {key: metadata.get(key) for key in allowed if key in metadata}


def _run_once(
    args: argparse.Namespace,
    selector: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from quantem.gpu.io import load

    _clear_backend(args.backend)
    memory_before = _memory_snapshot(args.backend)
    started = time.perf_counter()
    result = load(
        args.source,
        backend=args.backend,
        det_bin=args.det_bin,
        verbose=False,
        **_selector_kwargs(selector),
    )
    _sync_backend(args.backend)
    wall_seconds = time.perf_counter() - started
    memory_after = _memory_snapshot(args.backend)

    hash_started = time.perf_counter()
    output_sha256 = _array_sha256(result.data)
    hash_seconds = time.perf_counter() - hash_started
    expected_hash = selector.get("expected_output_sha256")
    expected_shape = selector.get("expected_output_shape")
    expected_dtype = selector.get("expected_output_dtype")
    description = _array_description(result.data)
    record = {
        "wall_seconds": wall_seconds,
        "hash_seconds_after_timed_boundary": hash_seconds,
        "output_sha256": output_sha256,
        "output": description,
        "metadata": _metadata_record(result.metadata),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "parity": {
            "reference_hash_supplied": expected_hash is not None,
            "reference_hash_exact": None
            if expected_hash is None
            else output_sha256 == expected_hash,
            "expected_shape": expected_shape,
            "shape_exact": None
            if expected_shape is None
            else description["shape"] == expected_shape,
            "expected_dtype": expected_dtype,
            "dtype_exact": None
            if expected_dtype is None
            else description["dtype"] == expected_dtype,
        },
    }
    return result, record


def _accepted(records: list[dict[str, Any]]) -> bool:
    hashes = [record["output_sha256"] for record in records]
    repeat_exact = all(value == hashes[0] for value in hashes[1:])
    for record in records:
        parity = record["parity"]
        if parity["reference_hash_exact"] is not True:
            return False
        if parity["shape_exact"] is False or parity["dtype_exact"] is False:
            return False
    return repeat_exact


def main() -> None:
    """Run one immutable selector repeatedly on a named physical backend.

    Examples
    --------
    Benchmark an exact rectangular MPS selection::

        python scripts/benchmark_selective_load.py master.h5 --backend mps \
            --selector selectors/region-64.json --det-bin 1 \
            --json-out selective-mps.json
    """

    args = _parse_args()
    _validate_args(args)
    selector = _load_selector(args.selector)

    for _ in range(args.warmup):
        result, _ = _run_once(args, selector)
        _release_array(result.data)
        del result
        _clear_backend(args.backend)

    records: list[dict[str, Any]] = []
    for trial in range(1, args.reps + 1):
        result, record = _run_once(args, selector)
        record["trial"] = trial
        record["release_method"] = _release_array(result.data)
        del result
        _clear_backend(args.backend)
        record["memory_after_release"] = _memory_snapshot(args.backend)
        records.append(record)

    hashes = [record["output_sha256"] for record in records]
    report = {
        "schema": "quantem-gpu-selective-load-benchmark/v2",
        "benchmark_definition": {
            "cache_state": args.cache_state,
            "timing_boundary": "public load return after synchronized exact selected resident output",
            "hash_boundary": "full-output hash after timed boundary",
            "warmup_count": args.warmup,
            "repetitions": args.reps,
        },
        "code": _git_state(),
        "host": _host_info(),
        "backend": args.backend,
        "fixture_id": args.fixture_id,
        "source_sha256": args.source_sha256,
        "source_path_included": False,
        "selector": selector,
        "selector_sha256": _stable_json_sha256(selector),
        "detector_bin": args.det_bin,
        "summary": _nearest_rank_summary(
            [record["wall_seconds"] for record in records]
        ),
        "repeat_hashes_exact": all(value == hashes[0] for value in hashes[1:]),
        "accepted": _accepted(records),
        "runs": records,
        "limitations": [
            "The caller-declared cache state is recorded but not controlled by this process.",
            "A result is accepted only when the selector supplies an exact full-output reference hash.",
            "Full-output hashing occurs after the timed boundary and its cost is reported separately.",
            "Application loading, presentation, and interaction are excluded.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
