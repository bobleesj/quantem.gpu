#!/usr/bin/env python
"""Benchmark exact resident detector reductions through public QuantEM.GPU APIs.

The benchmark loads one full-scan detector volume once, verifies its canonical
logical-pixel hash, and then times repeated detector-product publications. A
separately sealed NumPy reference bundle is mandatory. DPC, iDPC, display, and
FFT remain separate benchmark gates. Load, hash, parity, and product timing
remain separate evidence boundaries.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from _benchmark_support import (
    MemorySampler,
    _array_description,
    _array_sha256,
    _file_sha256,
    _git_state,
    _host_info,
    _memory_snapshot,
    _nearest_rank_summary,
    _numeric_delta,
    _stable_json_sha256,
    _sync_backend,
    _system_memory_snapshot,
    _validate_sha256,
    _write_json_report,
)
from benchmark_hdf5_load import (
    _load_once,
    _output_parity,
    _release_result,
    _scientific_metadata,
)

REFERENCE_SCHEMA = "quantem.gpu.detector-product-reference/v1"
REPORT_SCHEMA = "quantem.gpu.detector-product-benchmark/v1"
EXACT_PRODUCT_NAMES = (
    "total_intensity",
    "bright_field",
    "annular_bright_field",
    "annular_dark_field",
    "dark_field",
)
ARRAY_PRODUCT_NAMES = ("mean_dp", *EXACT_PRODUCT_NAMES)


def _parse_shape(value: str) -> tuple[int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 2 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "shape must contain two positive comma-separated values in row,column order"
        )
    return parts


def _parse_dimensions(value: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if not parts or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "shape must contain positive comma-separated dimensions"
        )
    return parts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="One complete HDF5 master.")
    parser.add_argument("--backend", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--computer", required=True, help="Public hardware label.")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-volume-sha256", required=True)
    parser.add_argument(
        "--expected-storage-shape",
        type=_parse_dimensions,
        required=True,
    )
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument(
        "--reference-sha256",
        required=True,
        help="SHA-256 of the independently sealed reference NPZ.",
    )
    parser.add_argument("--scan-shape", type=_parse_shape, required=True)
    parser.add_argument("--source-detector-shape", type=_parse_shape, required=True)
    parser.add_argument("--working-detector-shape", type=_parse_shape, required=True)
    parser.add_argument("--source-dtype", required=True)
    parser.add_argument("--working-dtype", required=True)
    parser.add_argument("--source-value-maximum", type=int, required=True)
    parser.add_argument(
        "--source-value-maximum-basis",
        required=True,
        help="SHA-256 of the sealed source-range audit.",
    )
    parser.add_argument("--det-bin", type=int, required=True)
    parser.add_argument("--center-row", type=float, required=True)
    parser.add_argument("--center-column", type=float, required=True)
    parser.add_argument("--bf-radius-pixels", type=float, required=True)
    parser.add_argument(
        "--pixel-mask",
        required=True,
        choices=("apply", "preserve"),
        help="Apply the HDF5 bad-pixel mask or preserve source counts unchanged.",
    )
    parser.add_argument("--cache-state", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--memory-sample-ms", type=float, default=10.0)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _source_evidence(args: argparse.Namespace) -> dict[str, Any]:
    from quantem.gpu.io import inspect

    if not args.source.is_file():
        raise ValueError(f"source must be an existing HDF5 master; got {args.source}")
    started = time.perf_counter()
    observed_master_sha256 = _file_sha256(args.source)
    master_hash_seconds = time.perf_counter() - started
    if observed_master_sha256 != args.source_sha256:
        raise ValueError(
            "source master SHA-256 mismatch: "
            f"expected {args.source_sha256}, got {observed_master_sha256}"
        )
    inspection = inspect(str(args.source), scan_shape=args.scan_shape)
    if not inspection.ready:
        raise ValueError(
            f"source readiness failed: {inspection.reason} {inspection.action}"
        )
    errors = []
    if inspection.scan_shape != args.scan_shape:
        errors.append(
            f"scan shape expected {args.scan_shape}, got {inspection.scan_shape}"
        )
    if inspection.detector_shape != args.source_detector_shape:
        errors.append(
            "source detector shape expected "
            f"{args.source_detector_shape}, got {inspection.detector_shape}"
        )
    if inspection.dtype != np.dtype(args.source_dtype).name:
        errors.append(
            f"source dtype expected {np.dtype(args.source_dtype).name}, "
            f"got {inspection.dtype}"
        )
    expected_frames = int(np.prod(args.scan_shape, dtype=np.int64))
    if inspection.actual_frames != expected_frames:
        errors.append(
            f"source frame count expected {expected_frames}, got {inspection.actual_frames}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    files = inspection.source_signature.get("files", [])
    source_bytes = sum(
        int(item.get("size", 0)) for item in files if isinstance(item, dict)
    )
    return {
        "fixture_id": args.fixture_id,
        "master_sha256": observed_master_sha256,
        "master_hash_seconds": master_hash_seconds,
        "inspection_signature_sha256": _stable_json_sha256(inspection.source_signature),
        "source_file_count": len(files),
        "source_bytes": source_bytes,
        "source_kind": inspection.source_kind,
        "actual_frames": inspection.actual_frames,
        "scan_shape": list(args.scan_shape),
        "source_detector_shape": list(args.source_detector_shape),
        "source_dtype": np.dtype(args.source_dtype).name,
        "paths_included": False,
    }


def _build_masks(args: argparse.Namespace) -> dict[str, np.ndarray]:
    from quantem.gpu.detector import detector_mask

    center = (args.center_row, args.center_column)
    shape = args.working_detector_shape
    radius = args.bf_radius_pixels
    return {
        "total_intensity": np.ones(shape, dtype=bool),
        "bright_field": detector_mask(center, 0.0, radius, shape),
        "annular_bright_field": detector_mask(center, 0.5 * radius, radius, shape),
        "annular_dark_field": detector_mask(center, radius, 2.0 * radius, shape),
        "dark_field": detector_mask(center, radius, np.inf, shape),
    }


def _overflow_proof(
    args: argparse.Namespace, masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    source_dtype = np.dtype(args.source_dtype)
    working_dtype = np.dtype(args.working_dtype)
    if source_dtype.kind != "u" or working_dtype.kind != "u":
        raise ValueError(
            "exact count benchmarking requires unsigned integer source and working dtypes; "
            f"got {source_dtype} and {working_dtype}"
        )
    source_limit = int(np.iinfo(source_dtype).max)
    working_limit = int(np.iinfo(working_dtype).max)
    if not 0 <= args.source_value_maximum <= source_limit:
        raise ValueError(
            "source-value-maximum must fit the source dtype; "
            f"got {args.source_value_maximum} for {source_dtype}"
        )
    maximum_working_value = args.source_value_maximum * args.det_bin**2
    if maximum_working_value > working_limit:
        raise OverflowError(
            "exact detector binning can overflow the requested working dtype: "
            f"{args.source_value_maximum} * {args.det_bin}^2 = "
            f"{maximum_working_value}, but {working_dtype} holds at most "
            f"{working_limit}"
        )
    scan_positions = int(np.prod(args.scan_shape, dtype=np.int64))
    mean_limit = int(np.iinfo(np.uint64).max)
    product_dtype = (
        np.dtype(np.int32)
        if args.backend == "mps" and working_dtype.itemsize < 4
        else np.dtype(np.uint64)
    )
    product_limit = int(np.iinfo(product_dtype).max)
    accumulator_bounds = {
        "mean_dp_per_pixel": maximum_working_value * scan_positions,
        **{
            f"{name}_per_frame": maximum_working_value * int(mask.sum())
            for name, mask in masks.items()
        },
    }
    unsafe = {}
    if accumulator_bounds["mean_dp_per_pixel"] > mean_limit:
        unsafe["mean_dp_per_pixel"] = accumulator_bounds["mean_dp_per_pixel"]
    unsafe.update(
        {
            name: value
            for name, value in accumulator_bounds.items()
            if name != "mean_dp_per_pixel" and value > product_limit
        }
    )
    if unsafe:
        detail = ", ".join(f"{name}={value}" for name, value in unsafe.items())
        raise OverflowError(
            f"the public exact reduction path cannot prove accumulator safety: {detail}"
        )
    return {
        "source_value_maximum": args.source_value_maximum,
        "source_value_maximum_basis": args.source_value_maximum_basis,
        "detector_bin_area": args.det_bin**2,
        "maximum_working_value": maximum_working_value,
        "working_dtype_maximum": working_limit,
        "mean_dp_accumulator_dtype": "uint64",
        "mean_dp_accumulator_maximum": mean_limit,
        "exact_product_accumulator_dtype": product_dtype.name,
        "exact_product_accumulator_maximum": product_limit,
        "accumulator_bounds": accumulator_bounds,
        "passed": True,
    }


def _reference_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Return the complete backend-neutral reference identity."""

    return {
        "schema": REFERENCE_SCHEMA,
        "source_sha256": args.source_sha256,
        "logical_volume_sha256": args.expected_volume_sha256,
        "scan_shape": list(args.scan_shape),
        "source_detector_shape": list(args.source_detector_shape),
        "working_detector_shape": list(args.working_detector_shape),
        "source_dtype": np.dtype(args.source_dtype).name,
        "working_dtype": np.dtype(args.working_dtype).name,
        "detector_bin": args.det_bin,
        "source_value_maximum": args.source_value_maximum,
        "source_value_maximum_basis": args.source_value_maximum_basis,
        "pixel_mask_applied": args.pixel_mask == "apply",
        "center_row": args.center_row,
        "center_column": args.center_column,
        "bf_radius_pixels": args.bf_radius_pixels,
    }


def _reference_bundle(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not args.reference_npz.is_file():
        raise ValueError(
            f"reference-npz must be an existing sealed bundle; got {args.reference_npz}"
        )
    observed_bundle_sha256 = _file_sha256(args.reference_npz)
    if observed_bundle_sha256 != args.reference_sha256:
        raise ValueError(
            "reference NPZ SHA-256 mismatch: "
            f"expected {args.reference_sha256}, got {observed_bundle_sha256}"
        )
    with np.load(args.reference_npz, allow_pickle=False) as bundle:
        required = {*ARRAY_PRODUCT_NAMES, "contract_json"}
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(
                "reference bundle is incomplete; missing " + ", ".join(sorted(missing))
            )
        contract_value = np.asarray(bundle["contract_json"])
        if contract_value.shape != ():
            raise ValueError("reference contract_json must be a scalar UTF-8 string")
        metadata = json.loads(str(contract_value.item()))
        if not isinstance(metadata, dict):
            raise TypeError("reference contract_json must decode to an object")
        arrays = {
            name: np.ascontiguousarray(bundle[name]) for name in ARRAY_PRODUCT_NAMES
        }

    expected_contract = _reference_contract(args)
    errors = []
    if metadata != expected_contract:
        changed = sorted(
            key
            for key in set(metadata) | set(expected_contract)
            if metadata.get(key) != expected_contract.get(key)
        )
        errors.append(
            "reference contract does not match the requested source, geometry, "
            "dtype, mask, and product parameters; changed fields: " + ", ".join(changed)
        )
    scan_shape = args.scan_shape
    detector_shape = args.working_detector_shape
    expected_specs = {
        "mean_dp": (detector_shape, np.dtype(np.float32)),
        **{name: (scan_shape, np.dtype(np.uint64)) for name in EXACT_PRODUCT_NAMES},
    }
    for name, (shape, dtype) in expected_specs.items():
        array = arrays[name]
        if array.shape != shape:
            errors.append(f"reference {name} shape expected {shape}, got {array.shape}")
        if array.dtype != dtype:
            errors.append(f"reference {name} dtype expected {dtype}, got {array.dtype}")
        if name not in EXACT_PRODUCT_NAMES and not np.isfinite(array).all():
            errors.append(f"reference {name} contains nonfinite values")
    if errors:
        raise ValueError("; ".join(errors))
    reference = dict(arrays)
    evidence = {
        "schema": REFERENCE_SCHEMA,
        "expected_bundle_sha256": args.reference_sha256,
        "bundle_sha256": observed_bundle_sha256,
        "bundle_sha256_passed": True,
        "bundle_bytes": args.reference_npz.stat().st_size,
        "contract": metadata,
        "contract_sha256": _stable_json_sha256(metadata),
        "array_hashes": {name: _array_sha256(array) for name, array in arrays.items()},
        "array_specs": {
            name: _array_description(array) for name, array in arrays.items()
        },
        "paths_included": False,
    }
    return reference, evidence


def _resident_contract(result: Any, args: argparse.Namespace) -> dict[str, Any]:
    metadata = dict(getattr(result, "metadata", {}) or {})
    output = _output_parity(result.data, args)
    science = _scientific_metadata(result, args)
    errors = [*output["errors"], *science["errors"]]
    for name in ("scan_region", "scan_crop", "detector_crop"):
        if metadata.get(name) is not None:
            errors.append(f"{name} must be absent for a full-scan/no-crop run")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "output": output,
        "scientific_metadata": science,
        "scan_bin": 1,
        "scan_crop": None,
        "detector_crop": None,
        "pixel_mask_applied": args.pixel_mask == "apply",
        "passed": True,
    }


def _timed_call(backend: str, operation: Callable[[], Any]) -> tuple[Any, float]:
    _sync_backend(backend)
    started = time.perf_counter()
    result = operation()
    _sync_backend(backend)
    return result, time.perf_counter() - started


def _run_suite(
    data: Any,
    masks: dict[str, np.ndarray],
    *,
    backend: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    from quantem.gpu import detector

    outputs: dict[str, Any] = {}
    stages: dict[str, float] = {}
    _sync_backend(backend)
    complete_started = time.perf_counter()
    session_started = time.perf_counter()
    session = detector.prepare(data)
    stages["session_prepare_seconds"] = time.perf_counter() - session_started
    try:
        outputs["mean_dp"], stages["mean_dp_seconds"] = _timed_call(
            backend, session.mean_dp
        )
        detector_started = time.perf_counter()
        for name in EXACT_PRODUCT_NAMES:
            outputs[name], stages[f"{name}_seconds"] = _timed_call(
                backend,
                lambda product=name: session.masked_sum_exact(masks[product]),
            )
        stages["exact_detector_products_seconds"] = (
            time.perf_counter() - detector_started
        )
    finally:
        session.close()
    _sync_backend(backend)
    stages["complete_product_suite_seconds"] = time.perf_counter() - complete_started
    return outputs, stages


def _product_parity(
    outputs: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    missing = set(ARRAY_PRODUCT_NAMES) - set(outputs)
    if missing:
        raise ValueError(
            "public product suite is incomplete; missing " + ", ".join(sorted(missing))
        )
    arrays = {name: np.ascontiguousarray(outputs[name]) for name in ARRAY_PRODUCT_NAMES}
    exact: dict[str, Any] = {}
    for name in EXACT_PRODUCT_NAMES:
        array = arrays[name]
        expected = reference[name]
        passed = array.dtype == np.dtype(np.uint64) and np.array_equal(array, expected)
        exact[name] = {
            "passed": bool(passed),
            "dtype": str(array.dtype),
            "sha256": _array_sha256(array),
            "reference_sha256": _array_sha256(expected),
        }
    mean_dp = arrays["mean_dp"]
    mean_passed = mean_dp.dtype == np.dtype(np.float32) and np.array_equal(
        mean_dp, reference["mean_dp"]
    )
    hashes = {name: _array_sha256(array) for name, array in arrays.items()}
    specs = {name: _array_description(array) for name, array in arrays.items()}
    passed = bool(all(item["passed"] for item in exact.values()) and mean_passed)
    return {
        "passed": passed,
        "exact_integer_products": exact,
        "mean_dp": {
            "passed": bool(mean_passed),
            "dtype": str(mean_dp.dtype),
            "sha256": hashes["mean_dp"],
            "reference_sha256": _array_sha256(reference["mean_dp"]),
            "contract": "byte exact float32 after exact integer accumulation",
        },
        "product_hashes": hashes,
        "product_specs": specs,
        "product_hash_set_sha256": _stable_json_sha256(hashes),
    }


def _run_product_trial(
    data: Any,
    masks: dict[str, np.ndarray],
    reference: dict[str, Any],
    args: argparse.Namespace,
    trial: int,
) -> tuple[dict[str, Any], str | None]:
    memory_before = _memory_snapshot(args.backend)
    system_before = _system_memory_snapshot()
    sampler = MemorySampler(args.backend, args.memory_sample_ms)
    outputs = None
    stages = None
    parity = None
    parity_seconds = None
    failure = None
    failure_stage = None
    sampler.start()
    try:
        outputs, stages = _run_suite(
            data,
            masks,
            backend=args.backend,
        )
    except BaseException as exc:  # noqa: BLE001 - retain failed physical evidence
        failure_stage = "product-suite"
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        sampled_memory = sampler.stop()
    memory_after = _memory_snapshot(args.backend)
    system_after = _system_memory_snapshot()
    if outputs is not None and failure is None:
        try:
            parity_started = time.perf_counter()
            parity = _product_parity(outputs, reference)
            parity_seconds = time.perf_counter() - parity_started
            if not parity["passed"]:
                failure_stage = "parity"
                failure = f"trial {trial} failed the frozen product parity contract"
        except BaseException as exc:  # noqa: BLE001 - retain parity failure evidence
            failure_stage = "parity"
            failure = f"{type(exc).__name__}: {exc}"
    record = {
        "trial": trial,
        "status": "passed" if failure is None else "failed",
        "failure_stage": failure_stage,
        "error": failure,
        "stage_timing": stages,
        "parity_seconds_excluded_from_timing": parity_seconds,
        "parity": parity,
        "memory_before": memory_before,
        "sampled_memory": sampled_memory,
        "memory_after": memory_after,
        "system_before": system_before,
        "system_after": system_after,
        "system_delta": _numeric_delta(system_before, system_after),
    }
    return record, failure


def _summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    stage_names = records[0]["stage_timing"]
    return {
        name: _nearest_rank_summary(
            [float(record["stage_timing"][name]) for record in records]
        )
        for name in stage_names
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.det_bin < 1:
        raise ValueError(f"det-bin must be positive; got {args.det_bin}")
    if args.reps < 1 or args.warmup < 0:
        raise ValueError(
            f"reps must be positive and warmup nonnegative; got {args.reps}, {args.warmup}"
        )
    if args.memory_sample_ms <= 0:
        raise ValueError(
            f"memory-sample-ms must be positive; got {args.memory_sample_ms}"
        )
    if args.bf_radius_pixels <= 0:
        raise ValueError(
            f"bf-radius-pixels must be positive; got {args.bf_radius_pixels}"
        )
    expected_source_shape = tuple(
        value * args.det_bin for value in args.working_detector_shape
    )
    if args.source_detector_shape != expected_source_shape:
        raise ValueError(
            "source and working detector geometry disagree with det-bin: "
            f"{args.source_detector_shape} -> {args.working_detector_shape} "
            f"at bin {args.det_bin}"
        )
    allowed_storage_shapes = {
        (*args.scan_shape, *args.working_detector_shape),
        (int(np.prod(args.scan_shape, dtype=np.int64)), *args.working_detector_shape),
    }
    if args.expected_storage_shape not in allowed_storage_shapes:
        raise ValueError(
            "expected-storage-shape must preserve the full scan and working detector; "
            f"got {args.expected_storage_shape}"
        )
    args.source_sha256 = _validate_sha256(args.source_sha256, "source-sha256")
    assert args.source_sha256 is not None
    args.reference_sha256 = _validate_sha256(args.reference_sha256, "reference-sha256")
    assert args.reference_sha256 is not None
    args.source_value_maximum_basis = _validate_sha256(
        args.source_value_maximum_basis, "source-value-maximum-basis"
    )
    assert args.source_value_maximum_basis is not None
    args.expected_volume_sha256 = _validate_sha256(
        args.expected_volume_sha256, "expected-volume-sha256"
    )
    assert args.expected_volume_sha256 is not None
    args.dtype = np.dtype(args.working_dtype).name
    args.expected_output_sha256 = args.expected_volume_sha256
    args.expected_output_dtype = args.dtype
    args.expected_output_shape = args.expected_storage_shape
    args.expected_scan_shape = args.scan_shape
    args.expected_source_detector_shape = args.source_detector_shape
    args.expected_working_detector_shape = args.working_detector_shape
    args.expected_source_dtype = np.dtype(args.source_dtype).name
    args.require_full_output_parity = True
    args.scan_region = None
    args.skip_mps_memory_check = False
    args.apply_mask = args.pixel_mask == "apply"
    args.auto_narrow = False


def _mask_contract(
    args: argparse.Namespace, masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    radius = args.bf_radius_pixels
    return {
        "center_row": args.center_row,
        "center_column": args.center_column,
        "bf_radius_pixels": radius,
        "bands_pixels_inclusive": {
            "total_intensity": [0.0, "infinity"],
            "bright_field": [0.0, radius],
            "annular_bright_field": [0.5 * radius, radius],
            "annular_dark_field": [radius, 2.0 * radius],
            "dark_field": [radius, "infinity"],
        },
        "masks": {
            name: {
                "selected_pixels": int(mask.sum()),
                "sha256": _array_sha256(mask.astype(np.uint8)),
            }
            for name, mask in masks.items()
        },
    }


def _benchmark_runs(
    report: dict[str, Any],
    data: Any,
    masks: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    for warmup in range(1, args.warmup + 1):
        record, failure = _run_product_trial(data, masks, reference, args, trial=warmup)
        record["warmup"] = warmup
        report["warmups"].append(record)
        if failure is not None:
            raise RuntimeError(f"warmup {warmup} failed: {failure}")
    for trial in range(1, args.reps + 1):
        record, failure = _run_product_trial(data, masks, reference, args, trial)
        report["runs"].append(record)
        if failure is not None:
            raise RuntimeError(f"trial {trial} failed: {failure}")
    hashes = [record["parity"]["product_hashes"] for record in report["runs"]]
    if any(value != hashes[0] for value in hashes[1:]):
        raise ValueError("product hashes changed between retained repetitions")
    report["stage_summaries"] = _summaries(report["runs"])
    report["repeat_product_hashes_exact"] = True


def run(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    """Run one resident product benchmark and return its report and failure."""

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "benchmark_definition": {
            "timing_boundary": (
                "public detector.prepare and exact detector reductions "
                "from one verified resident volume through synchronized host publication"
            ),
            "load_boundary": "public io.load return after backend synchronization",
            "cache_state": args.cache_state,
            "warmup_count": args.warmup,
            "repetitions": args.reps,
            "scan_bin": 1,
            "scan_crop": None,
            "detector_crop": None,
            "detector_bin": args.det_bin,
            "related_runbooks": {
                "source_load": "python-load-matrix",
                "com_dpc_idpc": "dpc-parity",
            },
            "full_resident_hash_excluded_from_timing": True,
            "parity_and_product_hashes_excluded_from_timing": True,
            "memory_sample_interval_ms": args.memory_sample_ms,
        },
        "code": {
            **_git_state(),
            "benchmark_entrypoint_sha256": _file_sha256(Path(__file__)),
        },
        "host": {**_host_info(), "computer": args.computer},
        "runtime": {"numpy": np.__version__},
        "backend": args.backend,
        "interaction_sidecar_boundary": {
            "possible_background_build": bool(
                args.backend == "mps" and args.det_bin == 1
            ),
            "scientific_reductions": "masked_sum_exact always uses the declared resident detector grid",
            "timing_isolation": (
                "background sidecar work may overlap this bin1 MPS benchmark"
                if args.backend == "mps" and args.det_bin == 1
                else "no lower-resolution interaction sidecar is eligible"
            ),
        },
        "runs": [],
        "warmups": [],
        "failures": [],
        "limitations": [
            "The caller-declared cache state is recorded but not controlled by this process.",
            "Resident-product timing excludes source load, full-volume hashing, parity, and application presentation.",
            "Sampled memory peaks are interval-dependent lower bounds; process_peak_rss_bytes is a separate process-lifetime high-water.",
            "A reference bundle must be generated and sealed independently of the backend run being adjudicated.",
            "CoM, DPC, iDPC, display, FFT, source-load, and application wall time are separate benchmark gates.",
            "A full-native MPS detector session may build its optional interaction sidecar concurrently; exact reductions bypass it, but bin1 timing is not sidecar-isolated.",
        ],
    }
    result = None
    data = None
    failure = None
    masks: dict[str, np.ndarray] | None = None
    try:
        report["source"] = _source_evidence(args)
        reference, report["reference"] = _reference_bundle(args)
        masks = _build_masks(args)
        report["mask_contract"] = _mask_contract(args, masks)
        report["overflow_proof"] = _overflow_proof(args, masks)

        report["system_before_load"] = _system_memory_snapshot()
        report["memory_before_load"] = _memory_snapshot(args.backend)
        load_sampler = MemorySampler(args.backend, args.memory_sample_ms)
        load_sampler.start()
        try:
            started = time.perf_counter()
            result = _load_once(args.source, args)
            _sync_backend(args.backend)
            report["load_seconds"] = time.perf_counter() - started
        finally:
            report["load_sampled_memory"] = load_sampler.stop()
        data = result.data
        report["memory_after_load"] = _memory_snapshot(args.backend)
        report["system_after_load"] = _system_memory_snapshot()
        report["resident_contract"] = _resident_contract(result, args)

        _benchmark_runs(report, data, masks, reference, args)
        report["status"] = "passed"
    except BaseException as exc:  # noqa: BLE001 - retain first failing evidence
        failure = f"{type(exc).__name__}: {exc}"
        report["status"] = "failed"
        report["failures"].append(failure)
    finally:
        release_method, release_error = _release_result(result, args.backend)
        data = None
        result = None
        if release_error is not None and failure is None:
            failure = release_error
            report["status"] = "failed"
            report["failures"].append(f"release {release_error}")
        report["release"] = {
            "method": release_method,
            "error": release_error,
            "memory_after_release": _memory_snapshot(args.backend),
            "system_after_release": _system_memory_snapshot(),
        }
        if "system_before_load" in report:
            report["system_run_delta"] = _numeric_delta(
                report["system_before_load"], report["release"]["system_after_release"]
            )
    return report, failure


def main() -> None:
    args = _parse_args()
    failure = None
    try:
        _validate_args(args)
        report, failure = run(args)
    except BaseException as exc:  # noqa: BLE001 - write configuration failures too
        failure = f"{type(exc).__name__}: {exc}"
        report = {
            "schema": REPORT_SCHEMA,
            "status": "failed",
            "code": _git_state(),
            "failures": [failure],
        }
    _write_json_report(report, args.json_out)
    if failure is not None:
        raise SystemExit(f"benchmark failed closed: {failure}")


if __name__ == "__main__":
    main()
