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
    MemorySampler,
    _array_description,
    _array_sha256,
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


def _parse_shape(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(","))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "expected-output-shape must contain positive comma-separated dimensions"
        )
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
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--expected-output-dtype")
    parser.add_argument("--expected-output-shape", type=_parse_shape)
    parser.add_argument("--expected-scan-shape", type=_parse_shape)
    parser.add_argument("--expected-source-detector-shape", type=_parse_shape)
    parser.add_argument("--expected-working-detector-shape", type=_parse_shape)
    parser.add_argument("--expected-source-dtype")
    parser.add_argument(
        "--require-full-output-parity",
        action="store_true",
        help=(
            "Require and verify an expected full-volume SHA-256, observed dtype, "
            "and observed logical shape after every timed load. Hashing is "
            "recorded separately and excluded from load wall time."
        ),
    )
    parser.add_argument(
        "--memory-sample-ms",
        type=float,
        default=10.0,
        help="Sampling interval for accelerator and process peak memory.",
    )
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


def _validate_sha256(value: str | None, option: str) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SystemExit(f"{option} must contain exactly 64 hexadecimal characters")
    return normalized


def _output_parity(data: Any, args: argparse.Namespace) -> dict[str, Any]:
    description = _array_description(data)
    output_sha256 = None
    validation_seconds = 0.0
    if args.expected_output_sha256 is not None or args.require_full_output_parity:
        t0 = time.perf_counter()
        output_sha256 = _array_sha256(data)
        validation_seconds = time.perf_counter() - t0
    errors: list[str] = []
    if (
        args.expected_output_sha256 is not None
        and output_sha256 != args.expected_output_sha256
    ):
        errors.append(
            "full-volume SHA-256 mismatch: "
            f"expected {args.expected_output_sha256}, got {output_sha256}"
        )
    if (
        args.expected_output_dtype is not None
        and description["dtype"] != args.expected_output_dtype
    ):
        errors.append(
            "output dtype mismatch: "
            f"expected {args.expected_output_dtype}, got {description['dtype']}"
        )
    expected_shape = (
        None if args.expected_output_shape is None else list(args.expected_output_shape)
    )
    if expected_shape is not None and description["shape"] != expected_shape:
        errors.append(
            "output shape mismatch: "
            f"expected {expected_shape}, got {description['shape']}"
        )
    return {
        **description,
        "full_volume_sha256": output_sha256,
        "validation_seconds": validation_seconds,
        "passed": not errors,
        "errors": errors,
    }


def _shape_field(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def _scientific_metadata(result: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Validate source and working geometry carried by one load result."""

    metadata = dict(getattr(result, "metadata", {}) or {})
    data = result.data
    scan_shape = getattr(data, "scan_shape", None) or metadata.get("scan_shape")
    working_detector_shape = getattr(data, "detector_shape", None)
    if working_detector_shape is None:
        working_detector_shape = metadata.get("detector_shape")
    source_detector_shape = metadata.get("raw_detector_shape")
    if source_detector_shape is None:
        source_detector_shape = metadata.get("source_detector_shape")
    if source_detector_shape is None and int(args.det_bin) == 1:
        source_detector_shape = working_detector_shape
    source_dtype = metadata.get("source_dtype")
    working_dtype = metadata.get("dtype")
    if working_dtype is None:
        working_dtype = str(getattr(data, "dtype", "")) or None
    detector_bin = metadata.get("det_bin")
    if detector_bin is None:
        detector_bin = metadata.get("detector_bin")

    observed = {
        "scan_shape": _shape_field(scan_shape),
        "source_detector_shape": _shape_field(source_detector_shape),
        "working_detector_shape": _shape_field(working_detector_shape),
        "source_dtype": None if source_dtype is None else str(source_dtype),
        "working_dtype": None if working_dtype is None else str(working_dtype),
        "detector_bin": None if detector_bin is None else int(detector_bin),
    }
    expected = {
        "scan_shape": _shape_field(args.expected_scan_shape),
        "source_detector_shape": _shape_field(args.expected_source_detector_shape),
        "working_detector_shape": _shape_field(args.expected_working_detector_shape),
        "source_dtype": args.expected_source_dtype,
        "working_dtype": args.expected_output_dtype,
        "detector_bin": int(args.det_bin),
    }
    errors = [
        f"metadata {field} mismatch: expected {expected_value}, got {observed[field]}"
        for field, expected_value in expected.items()
        if expected_value is not None and observed[field] != expected_value
    ]
    return {
        "expected": expected,
        "observed": observed,
        "passed": not errors,
        "errors": errors,
    }


def _write_report(report: dict[str, Any], path: Path | None) -> None:
    """Print a report and atomically persist it when a destination is set."""

    rendered = json.dumps(report, indent=2) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(rendered, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print(rendered, end="")


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


def _release_result(result: Any, backend: str) -> tuple[str | None, str | None]:
    """Release a backend result and report cleanup failures without hiding them."""

    release_method = None
    release_error = None
    try:
        if result is not None:
            release_method = _release_array(result.data)
    except Exception as exc:  # noqa: BLE001 - retain cleanup failure evidence
        release_error = f"{type(exc).__name__}: {exc}"
    finally:
        _clear_backend(backend)
    return release_method, release_error


def _run_warmup(master: Path, args: argparse.Namespace) -> str | None:
    """Run and deterministically release one untimed warmup."""

    result = None
    failure = None
    try:
        result = _load_once(master, args)
        _sync_backend(args.backend)
    except Exception as exc:  # noqa: BLE001 - retain warmup failure evidence
        failure = f"warmup {type(exc).__name__}: {exc}"
    release_method, release_error = _release_result(result, args.backend)
    del release_method
    if release_error is not None:
        failure = f"{failure}; cleanup {release_error}" if failure else release_error
    return failure


def _run_timed_trial(
    master: Path, args: argparse.Namespace, trial: int
) -> tuple[dict[str, Any], str | None]:
    """Run one load, retain pass/fail evidence, and always release its output."""

    result = None
    elapsed = None
    output_parity = None
    scientific_metadata = None
    failure = None
    failure_stage = None
    memory_before = _memory_snapshot(args.backend)
    memory_sampler = MemorySampler(args.backend, args.memory_sample_ms)
    sampled_memory = None
    try:
        memory_sampler.start()
        t0 = time.perf_counter()
        result = _load_once(master, args)
        _sync_backend(args.backend)
        elapsed = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 - retain backend failure evidence
        failure_stage = "load-or-synchronize"
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            sampled_memory = memory_sampler.stop()
        except Exception as exc:  # noqa: BLE001 - retain sampler failure evidence
            sampler_failure = f"{type(exc).__name__}: {exc}"
            if failure is None:
                failure_stage = "memory-sampler"
                failure = sampler_failure
            else:
                failure = f"{failure}; memory sampler {sampler_failure}"

    memory_after = _memory_snapshot(args.backend)
    if result is not None and failure is None:
        try:
            output_parity = _output_parity(result.data, args)
            scientific_metadata = _scientific_metadata(result, args)
            parity_errors = [
                *output_parity["errors"],
                *scientific_metadata["errors"],
            ]
            if parity_errors:
                failure_stage = "parity"
                failure = "; ".join(parity_errors)
        except Exception as exc:  # noqa: BLE001 - retain parity failure evidence
            failure_stage = "parity-evaluation"
            failure = f"{type(exc).__name__}: {exc}"

    shape = None if result is None else _shape(result.data)
    resident_bytes = None if result is None else _nbytes(result.data)
    release_method, release_error = _release_result(result, args.backend)
    result = None
    memory_after_release = _memory_snapshot(args.backend)
    if release_error is not None:
        if failure is None:
            failure_stage = "release"
            failure = release_error
        else:
            failure = f"{failure}; cleanup {release_error}"

    record = {
        "trial": trial,
        "status": "passed" if failure is None else "failed",
        "failure_stage": failure_stage,
        "error": failure,
        "wall_seconds": elapsed,
        "shape": shape,
        "resident_bytes": resident_bytes,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "sampled_memory": sampled_memory,
        "output_parity": output_parity,
        "scientific_metadata": scientific_metadata,
        "release_method": release_method,
        "release_error": release_error,
        "memory_after_release": memory_after_release,
    }
    return record, failure


def main() -> None:
    args = _parse_args()
    if args.reps < 1 or args.warmup < 0:
        raise SystemExit(
            f"reps must be positive and warmup nonnegative; got {args.reps}, {args.warmup}"
        )
    args.source_sha256 = _validate_sha256(args.source_sha256, "source-sha256")
    args.expected_output_sha256 = _validate_sha256(
        args.expected_output_sha256, "expected-output-sha256"
    )
    if args.memory_sample_ms <= 0:
        raise SystemExit(
            f"memory-sample-ms must be positive; got {args.memory_sample_ms}"
        )
    if args.require_full_output_parity:
        missing = [
            option
            for option, value in (
                ("--expected-output-sha256", args.expected_output_sha256),
                ("--expected-output-dtype", args.expected_output_dtype),
                ("--expected-output-shape", args.expected_output_shape),
                ("--expected-scan-shape", args.expected_scan_shape),
                (
                    "--expected-source-detector-shape",
                    args.expected_source_detector_shape,
                ),
                (
                    "--expected-working-detector-shape",
                    args.expected_working_detector_shape,
                ),
                ("--expected-source-dtype", args.expected_source_dtype),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "require-full-output-parity also requires " + ", ".join(missing)
            )
    report = {
        "schema": "quantem-gpu-hdf5-load-benchmark/v4",
        "status": "running",
        "benchmark_definition": {
            "timing_boundary": "public io.load return after backend synchronization",
            "cache_state": args.cache_state,
            "warmup_count": args.warmup,
            "repetitions": args.reps,
            "full_volume_hash_excluded_from_wall_time": True,
            "memory_sample_interval_ms": args.memory_sample_ms,
            "memory_peak_semantics": (
                "Observed sampled lower bounds at the requested interval; "
                "process_peak_rss_bytes is a separate process-lifetime high-water."
            ),
        },
        "reference_contract": {
            "full_volume_sha256": args.expected_output_sha256,
            "storage_dtype": args.expected_output_dtype,
            "storage_shape": _shape_field(args.expected_output_shape),
            "scan_shape": _shape_field(args.expected_scan_shape),
            "source_detector_shape": _shape_field(args.expected_source_detector_shape),
            "working_detector_shape": _shape_field(
                args.expected_working_detector_shape
            ),
            "source_dtype": args.expected_source_dtype,
            "detector_bin": args.det_bin,
            "required": bool(args.require_full_output_parity),
        },
        "code": _git_state(),
        "host": _host_info(),
        "fixture_id": args.fixture_id,
        "source_sha256": args.source_sha256,
        "source_paths_included": bool(args.show_paths),
        "rows": [],
        "failures": [],
        "limitations": [
            "The caller-declared cache state is recorded but not controlled by this process.",
            "Application wall time requires a separate retained artifact.",
            "Source SHA-256 is caller-declared provenance; full output bytes, dtype, and shape are independently checked when --require-full-output-parity is used.",
            "Sampled memory peaks are interval-dependent lower bounds; external process-tree, compressed-memory, pageout, and device telemetry remain separate required artifacts where the runbook requests them.",
        ],
    }
    masters = [Path(path).expanduser() for path in args.masters]
    terminal_failure = None
    for index, master in enumerate(masters):
        if not master.exists():
            terminal_failure = f"missing master #{index + 1}"
            report["failures"].append(
                {"master": f"master-{index + 1}", "error": terminal_failure}
            )
            break
        label = str(master) if args.show_paths else f"master-{index + 1}"
        row: dict[str, Any] = {
            "label": label,
            "backend": args.backend,
            "dtype": args.dtype or "native",
            "det_bin": args.det_bin,
            "scan_region": list(args.scan_region) if args.scan_region else None,
            "status": "running",
            "runs": [],
        }
        report["rows"].append(row)
        for warmup in range(1, args.warmup + 1):
            warmup_failure = _run_warmup(master, args)
            if warmup_failure is not None:
                terminal_failure = warmup_failure
                row["status"] = "failed"
                row["failed_warmup"] = warmup
                row["error"] = warmup_failure
                report["failures"].append(
                    {"master": label, "warmup": warmup, "error": warmup_failure}
                )
                break
        if terminal_failure is not None:
            break

        times: list[float] = []
        for trial in range(1, args.reps + 1):
            _clear_backend(args.backend)
            record, failure = _run_timed_trial(master, args, trial)
            row["runs"].append(record)
            if record["wall_seconds"] is not None:
                times.append(float(record["wall_seconds"]))
            if record["output_parity"] is not None:
                parity = record["output_parity"]
                scientific = record["scientific_metadata"]
                row.update(
                    {
                        "observed_dtype": parity["dtype"],
                        "shape": record["shape"],
                        "resident_gb": (
                            None
                            if record["resident_bytes"] is None
                            else record["resident_bytes"] / 1e9
                        ),
                        "full_volume_sha256": parity["full_volume_sha256"],
                        "full_output_parity_passed": bool(
                            parity["passed"] and scientific["passed"]
                        ),
                        "scientific_metadata": scientific,
                    }
                )
            if failure is not None:
                terminal_failure = failure
                row["status"] = "failed"
                row["error"] = failure
                report["failures"].append(
                    {"master": label, "trial": trial, "error": failure}
                )
                break
        if times:
            summary = _nearest_rank_summary(times)
            row.update(
                {
                    "reps_completed": len(times),
                    "median_s": summary["p50_seconds"],
                    "p50_s": summary["p50_seconds"],
                    "p95_s": summary["p95_seconds"],
                    "min_s": min(times),
                    "max_s": summary["max_seconds"],
                }
            )
        if terminal_failure is not None:
            break
        row["status"] = "passed"

    report["status"] = "failed" if terminal_failure is not None else "passed"
    _write_report(report, args.json_out)
    if terminal_failure is not None:
        raise SystemExit(f"benchmark failed closed: {terminal_failure}")


if __name__ == "__main__":
    main()
