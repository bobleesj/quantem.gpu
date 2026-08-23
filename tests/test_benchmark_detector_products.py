from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_detector_products as benchmark


class _Sampler:
    def start(self) -> None:
        pass

    def stop(self) -> dict:
        return {"interval_ms": 10.0, "sample_count": 2, "peak": {}}


class _ManagedArray:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.freed = False

    def free(self) -> None:
        self.freed = True


def _data() -> np.ndarray:
    values = np.arange(3 * 3 * 4 * 4, dtype=np.uint16).reshape(3, 3, 4, 4)
    return (values % 29 + 1).astype(np.uint16)


def _args(tmp_path: Path, data: np.ndarray) -> argparse.Namespace:
    source = tmp_path / "fixture_master.h5"
    source.write_bytes(b"fixture-master")
    return argparse.Namespace(
        source=source,
        backend="cpu",
        computer="Portable CPU test runner",
        fixture_id="synthetic-3x3x4x4-u16",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_volume_sha256=benchmark._array_sha256(data),
        expected_storage_shape=data.shape,
        reference_npz=tmp_path / "reference.npz",
        reference_sha256="0" * 64,
        scan_shape=(3, 3),
        source_detector_shape=(4, 4),
        working_detector_shape=(4, 4),
        source_dtype="uint16",
        working_dtype="uint16",
        source_value_maximum=int(data.max()),
        source_value_maximum_basis=hashlib.sha256(
            b"complete synthetic array maximum"
        ).hexdigest(),
        det_bin=1,
        center_row=1.5,
        center_column=1.5,
        bf_radius_pixels=1.5,
        pixel_mask="preserve",
        cache_state="resident synthetic source",
        warmup=1,
        reps=3,
        memory_sample_ms=10.0,
        json_out=tmp_path / "report.json",
    )


def _reference(data: np.ndarray, args: argparse.Namespace) -> dict:
    masks = benchmark._build_masks(args)
    outputs, _ = benchmark._run_suite(
        data,
        masks,
        backend="cpu",
    )
    return {
        name: np.ascontiguousarray(outputs[name])
        for name in benchmark.ARRAY_PRODUCT_NAMES
    }


def _write_reference(
    path: Path,
    args: argparse.Namespace,
    reference: dict,
) -> None:
    np.savez(
        path,
        contract_json=np.asarray(json.dumps(benchmark._reference_contract(args))),
        **{name: reference[name] for name in benchmark.ARRAY_PRODUCT_NAMES},
    )
    args.reference_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "_memory_snapshot", lambda backend: {})
    monkeypatch.setattr(benchmark, "_system_memory_snapshot", dict)
    monkeypatch.setattr(
        benchmark, "MemorySampler", lambda backend, interval: _Sampler()
    )
    monkeypatch.setattr(benchmark, "_sync_backend", lambda backend: None)


def test_reference_bundle_is_complete_and_product_parity_is_fail_closed(
    tmp_path: Path,
) -> None:
    data = _data()
    args = _args(tmp_path, data)
    reference = _reference(data, args)
    _write_reference(args.reference_npz, args, reference)

    loaded, evidence = benchmark._reference_bundle(args)
    parity = benchmark._product_parity(reference, loaded)

    assert parity["passed"] is True
    assert evidence["schema"] == benchmark.REFERENCE_SCHEMA
    assert evidence["expected_bundle_sha256"] == args.reference_sha256
    assert evidence["bundle_sha256_passed"] is True
    assert (
        evidence["bundle_sha256"]
        == hashlib.sha256(args.reference_npz.read_bytes()).hexdigest()
    )

    changed = dict(reference)
    changed["bright_field"] = reference["bright_field"].copy()
    changed["bright_field"][0, 0] += np.uint64(1)
    failed = benchmark._product_parity(changed, loaded)
    assert failed["passed"] is False
    assert failed["exact_integer_products"]["bright_field"]["passed"] is False


def test_reference_bundle_rejects_hash_mismatch_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    args = _args(tmp_path, data)
    reference = _reference(data, args)
    _write_reference(args.reference_npz, args, reference)
    args.reference_sha256 = "f" * 64

    monkeypatch.setattr(
        benchmark.np,
        "load",
        lambda *unused_args, **unused_kwargs: pytest.fail(
            "np.load must not run before reference SHA-256 validation"
        ),
    )

    with pytest.raises(ValueError, match="reference NPZ SHA-256 mismatch"):
        benchmark._reference_bundle(args)


def test_range_audit_sha_is_validated_and_bound_to_reference(
    tmp_path: Path,
) -> None:
    data = _data()
    args = _args(tmp_path, data)
    reference = _reference(data, args)
    _write_reference(args.reference_npz, args, reference)

    malformed = _args(tmp_path, data)
    malformed.reference_sha256 = "not-a-sha256"
    with pytest.raises(ValueError, match="reference-sha256"):
        benchmark._validate_args(malformed)

    malformed = _args(tmp_path, data)
    malformed.source_value_maximum_basis = "not-a-sha256"
    with pytest.raises(ValueError, match="source-value-maximum-basis"):
        benchmark._validate_args(malformed)

    args.source_value_maximum_basis = "a" * 64
    with pytest.raises(ValueError, match="changed fields: source_value_maximum_basis"):
        benchmark._reference_bundle(args)


def test_overflow_proof_rejects_unsafe_detector_bin(
    tmp_path: Path,
) -> None:
    data = _data()
    args = _args(tmp_path, data)
    args.det_bin = 2
    args.source_value_maximum = 20_000
    masks = benchmark._build_masks(args)

    with pytest.raises(OverflowError, match="overflow the requested working dtype"):
        benchmark._overflow_proof(args, masks)


def test_overflow_proof_uses_mps_int32_exact_product_limit(tmp_path: Path) -> None:
    args = _args(tmp_path, _data())
    args.backend = "mps"
    args.source_detector_shape = (192, 192)
    args.working_detector_shape = (192, 192)
    args.source_value_maximum = int(np.iinfo(np.uint16).max)
    masks = benchmark._build_masks(args)

    with pytest.raises(OverflowError, match="accumulator safety"):
        benchmark._overflow_proof(args, masks)


def test_resident_contract_accepts_full_scan_and_rejects_crop_metadata(
    tmp_path: Path,
) -> None:
    data = _data()
    args = _args(tmp_path, data)
    benchmark._validate_args(args)
    result = SimpleNamespace(
        data=data,
        metadata={
            "scan_shape": args.scan_shape,
            "raw_detector_shape": args.source_detector_shape,
            "detector_shape": args.working_detector_shape,
            "source_dtype": args.source_dtype,
            "det_bin": args.det_bin,
        },
    )

    evidence = benchmark._resident_contract(result, args)
    assert evidence["passed"] is True
    assert evidence["output"]["full_volume_sha256"] == args.expected_volume_sha256

    result.metadata["scan_region"] = (0, 2, 0, 2)
    with pytest.raises(ValueError, match="full-scan/no-crop"):
        benchmark._resident_contract(result, args)


def test_run_emits_distributions_and_exact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    args = _args(tmp_path, data)
    reference = _reference(data, args)
    result = SimpleNamespace(data=data, metadata={})
    monkeypatch.setattr(benchmark, "_source_evidence", lambda current: {"passed": True})
    monkeypatch.setattr(
        benchmark,
        "_reference_bundle",
        lambda current: (reference, {"bundle_sha256": "a" * 64}),
    )
    monkeypatch.setattr(benchmark, "_load_once", lambda master, current: result)
    monkeypatch.setattr(
        benchmark,
        "_resident_contract",
        lambda loaded, current: {"passed": True},
    )
    _patch_telemetry(monkeypatch)

    report, failure = benchmark.run(args)

    assert failure is None
    assert report["status"] == "passed"
    assert len(report["warmups"]) == 1
    assert len(report["runs"]) == 3
    assert report["stage_summaries"]["complete_product_suite_seconds"]["samples"] == 3
    assert report["repeat_product_hashes_exact"] is True
    assert all(row["parity"]["passed"] for row in report["runs"])


def test_run_releases_managed_volume_after_parity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    args = _args(tmp_path, data)
    args.warmup = 0
    args.reps = 1
    reference = _reference(data, args)
    reference["bright_field"] = reference["bright_field"].copy()
    reference["bright_field"][0, 0] += np.uint64(1)
    managed = _ManagedArray(data)
    result = SimpleNamespace(data=managed, metadata={})
    monkeypatch.setattr(benchmark, "_source_evidence", lambda current: {"passed": True})
    monkeypatch.setattr(
        benchmark,
        "_reference_bundle",
        lambda current: (reference, {"bundle_sha256": "b" * 64}),
    )
    monkeypatch.setattr(benchmark, "_load_once", lambda master, current: result)
    monkeypatch.setattr(
        benchmark,
        "_resident_contract",
        lambda loaded, current: {"passed": True},
    )
    _patch_telemetry(monkeypatch)

    report, failure = benchmark.run(args)

    assert report["status"] == "failed"
    assert "failed the frozen product parity contract" in failure
    assert report["runs"][0]["failure_stage"] == "parity"
    assert report["runs"][0]["sampled_memory"]["sample_count"] == 2
    assert managed.freed is True
