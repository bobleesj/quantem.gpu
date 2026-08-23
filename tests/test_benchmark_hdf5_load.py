from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_hdf5_load as benchmark
from _benchmark_support import _array_sha256
from benchmark_hdf5_load import _output_parity, _parse_shape


def _args(array: np.ndarray) -> argparse.Namespace:
    return argparse.Namespace(
        expected_output_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        expected_output_dtype=str(array.dtype),
        expected_output_shape=array.shape,
        expected_scan_shape=(2, 1),
        expected_source_detector_shape=array.shape[-2:],
        expected_working_detector_shape=array.shape[-2:],
        expected_source_dtype=str(array.dtype),
        det_bin=1,
        require_full_output_parity=True,
    )


def test_output_parity_hashes_full_array_and_checks_dtype_and_shape() -> None:
    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

    evidence = _output_parity(array, _args(array))

    assert evidence["passed"] is True
    assert evidence["errors"] == []
    assert evidence["shape"] == [2, 3, 4]
    assert evidence["dtype"] == "uint16"
    assert evidence["logical_resident_bytes"] == 48
    assert evidence["full_volume_sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
    assert evidence["logical_pixel_hash_schema"] == (
        "quantem.gpu.4dstem-logical-pixels/v1"
    )
    assert evidence["logical_pixel_axis_order"] == [
        "scan_row",
        "scan_column",
        "detector_row",
        "detector_column",
    ]
    assert evidence["logical_pixel_byte_order"] == "little_endian"


def test_logical_pixel_hash_matches_cross_language_vector() -> None:
    values = np.arange(2 * 3 * 3 * 5, dtype=np.uint16).reshape(2, 3, 3, 5)

    assert _array_sha256(values) == (
        "cae677456d8bcae7bd20864213c1c380b694f075e1980e2645a707db8301b977"
    )


def test_logical_pixel_hash_normalizes_endianness_and_chunk_boundaries() -> None:
    values = np.arange(90, dtype=np.uint16).reshape(6, 3, 5)

    class Chunked:
        def __init__(self) -> None:
            self.chunks = [values[:2].astype(">u2"), values[2:5], values[5:]]

    assert _array_sha256(Chunked()) == _array_sha256(values)


def test_output_parity_fails_closed_on_changed_bytes() -> None:
    reference = np.arange(8, dtype=np.uint16)
    changed = reference.copy()
    changed[-1] += 1

    evidence = _output_parity(changed, _args(reference))

    assert evidence["passed"] is False
    assert evidence["errors"] == [
        (
            "full-volume SHA-256 mismatch: "
            f"expected {hashlib.sha256(reference.tobytes()).hexdigest()}, "
            f"got {hashlib.sha256(changed.tobytes()).hexdigest()}"
        )
    ]


def test_expected_output_shape_uses_positive_explicit_dimensions() -> None:
    assert _parse_shape("262144,96,96") == (262144, 96, 96)


def test_load_once_reuses_explicit_geometry_mask_and_dtype_contract(
    monkeypatch,
) -> None:
    calls = []

    def fake_load(source, **kwargs):
        calls.append((source, kwargs))
        return object()

    from quantem.gpu import io

    monkeypatch.setattr(io, "load", fake_load)
    args = argparse.Namespace(
        backend="cpu",
        det_bin=4,
        dtype="uint16",
        expected_scan_shape=(2, 3),
        apply_mask=False,
        auto_narrow=False,
        scan_region=None,
        skip_mps_memory_check=False,
    )

    result = benchmark._load_once(Path("fixture.h5"), args)

    assert result is not None
    assert calls == [
        (
            "fixture.h5",
            {
                "backend": "cpu",
                "det_bin": 4,
                "verbose": False,
                "dtype": "uint16",
                "scan_shape": (2, 3),
                "apply_mask": False,
                "auto_narrow": False,
            },
        )
    ]


class _ManagedArray:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.scan_shape = (2, 1)
        self.detector_shape = array.shape[-2:]
        self.freed = False

    def __array__(self) -> np.ndarray:
        return self.array

    def free(self) -> None:
        self.freed = True


class _Result:
    def __init__(self, data: _ManagedArray) -> None:
        self.data = data
        self.metadata = {
            "scan_shape": (2, 1),
            "raw_detector_shape": data.detector_shape,
            "detector_shape": data.detector_shape,
            "source_dtype": "uint16",
            "dtype": "uint16",
            "det_bin": 1,
        }


class _Sampler:
    def start(self) -> None:
        pass

    def stop(self) -> dict:
        return {"interval_ms": 10.0, "sample_count": 2, "peak": {}}


def _trial_args(array: np.ndarray) -> argparse.Namespace:
    args = _args(array)
    args.backend = "mps"
    args.memory_sample_ms = 10.0
    return args


def test_scientific_metadata_proves_scan_and_detector_geometry() -> None:
    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    result = _Result(_ManagedArray(array))

    evidence = benchmark._scientific_metadata(result, _trial_args(array))

    assert evidence["passed"] is True
    assert evidence["observed"] == {
        "scan_shape": [2, 1],
        "source_detector_shape": [3, 4],
        "working_detector_shape": [3, 4],
        "source_dtype": "uint16",
        "working_dtype": "uint16",
        "detector_bin": 1,
    }


def test_trial_releases_result_after_hash_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    managed = _ManagedArray(array)
    result = _Result(managed)
    monkeypatch.setattr(benchmark, "_load_once", lambda master, args: result)
    monkeypatch.setattr(benchmark, "_sync_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_clear_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_memory_snapshot", lambda backend: {})
    monkeypatch.setattr(
        benchmark, "MemorySampler", lambda backend, interval: _Sampler()
    )
    monkeypatch.setattr(
        benchmark,
        "_array_sha256",
        lambda data: (_ for _ in ()).throw(RuntimeError("hash failed")),
    )

    record, failure = benchmark._run_timed_trial(
        Path("unused-master.h5"), _trial_args(array), 1
    )

    assert record["status"] == "failed"
    assert record["failure_stage"] == "parity-evaluation"
    assert "hash failed" in failure
    assert managed.freed is True


def test_trial_retains_mismatch_and_releases_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    changed = reference.copy()
    changed[-1, -1, -1] += 1
    managed = _ManagedArray(changed)
    result = _Result(managed)
    monkeypatch.setattr(benchmark, "_load_once", lambda master, args: result)
    monkeypatch.setattr(benchmark, "_sync_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_clear_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_memory_snapshot", lambda backend: {})
    monkeypatch.setattr(
        benchmark, "MemorySampler", lambda backend, interval: _Sampler()
    )

    record, failure = benchmark._run_timed_trial(
        Path("unused-master.h5"), _trial_args(reference), 1
    )

    assert record["status"] == "failed"
    assert record["failure_stage"] == "parity"
    assert record["output_parity"]["passed"] is False
    assert (
        record["output_parity"]["full_volume_sha256"]
        == hashlib.sha256(changed.tobytes()).hexdigest()
    )
    assert "full-volume SHA-256 mismatch" in failure
    assert managed.freed is True


def test_trial_releases_result_after_shape_evidence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    managed = _ManagedArray(array)
    result = _Result(managed)
    monkeypatch.setattr(benchmark, "_load_once", lambda master, args: result)
    monkeypatch.setattr(benchmark, "_sync_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_clear_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_memory_snapshot", lambda backend: {})
    monkeypatch.setattr(
        benchmark, "MemorySampler", lambda backend, interval: _Sampler()
    )
    monkeypatch.setattr(
        benchmark,
        "_output_parity",
        lambda data, args: {"errors": [], "passed": True},
    )
    monkeypatch.setattr(
        benchmark,
        "_scientific_metadata",
        lambda loaded, args: {"errors": [], "passed": True},
    )
    monkeypatch.setattr(
        benchmark,
        "_shape",
        lambda data: (_ for _ in ()).throw(RuntimeError("shape failed")),
    )

    record, failure = benchmark._run_timed_trial(
        Path("unused-master.h5"), _trial_args(array), 1
    )

    assert record["status"] == "failed"
    assert record["failure_stage"] == "result-metadata"
    assert "shape failed" in failure
    assert managed.freed is True


def test_trial_releases_result_after_resident_byte_evidence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    managed = _ManagedArray(array)
    result = _Result(managed)
    monkeypatch.setattr(benchmark, "_load_once", lambda master, args: result)
    monkeypatch.setattr(benchmark, "_sync_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_clear_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_memory_snapshot", lambda backend: {})
    monkeypatch.setattr(
        benchmark, "MemorySampler", lambda backend, interval: _Sampler()
    )
    monkeypatch.setattr(
        benchmark,
        "_output_parity",
        lambda data, args: {"errors": [], "passed": True},
    )
    monkeypatch.setattr(
        benchmark,
        "_scientific_metadata",
        lambda loaded, args: {"errors": [], "passed": True},
    )
    monkeypatch.setattr(
        benchmark,
        "_nbytes",
        lambda data: (_ for _ in ()).throw(RuntimeError("nbytes failed")),
    )

    record, failure = benchmark._run_timed_trial(
        Path("unused-master.h5"), _trial_args(array), 1
    )

    assert record["status"] == "failed"
    assert record["failure_stage"] == "result-metadata"
    assert "nbytes failed" in failure
    assert managed.freed is True


def test_trial_releases_result_after_post_load_memory_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    managed = _ManagedArray(array)
    result = _Result(managed)
    snapshot_calls = 0

    def memory_snapshot(backend: str) -> dict:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 2:
            raise RuntimeError("telemetry failed")
        return {"snapshot": snapshot_calls}

    monkeypatch.setattr(benchmark, "_load_once", lambda master, args: result)
    monkeypatch.setattr(benchmark, "_sync_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_clear_backend", lambda backend: None)
    monkeypatch.setattr(benchmark, "_memory_snapshot", memory_snapshot)
    monkeypatch.setattr(
        benchmark, "MemorySampler", lambda backend, interval: _Sampler()
    )

    record, failure = benchmark._run_timed_trial(
        Path("unused-master.h5"), _trial_args(array), 1
    )

    assert record["status"] == "failed"
    assert record["failure_stage"] == "post-load-memory"
    assert "telemetry failed" in failure
    assert record["memory_after_release"] == {"snapshot": 3}
    assert managed.freed is True
