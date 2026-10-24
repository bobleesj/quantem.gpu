"""Portable checks for matched IO regression evidence; physical runs stay separate."""

import copy
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_io_loading as benchmark


def _report() -> dict:
    summary = {"n": 5, "median_s": 0.5, "cache_qualified": True,
               "correctness_passed": True}
    return {
        "schema": "test", "backend": "cuda", "device_identity": "physical-device",
        "storage_id": "test-ssd", "boundary": "synchronized loading",
        "index_policy": "fresh first, reused repeats", "allocator_policy": "reuse",
        "product_definition": "fixed mask",
        "passed": True,
        "cases": [{
            "id": "fixture", "reference_identity": "frozen-body", "shape": [2, 2, 4, 4],
            "dtype": "uint16", "source_master_sha256": "frozen-master",
            "source_device": 1, "qem_device": 1,
            "hdf5": {"cold_summary": dict(summary), "warm_summary": dict(summary)},
            "qem": {"cold_summary": dict(summary), "warm_summary": dict(summary)},
        }],
    }


def test_matched_run_flags_slowdown_without_replacing_reference() -> None:
    baseline = _report()
    frozen = copy.deepcopy(baseline)
    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["qem"]["cold_summary"]["median_s"] = 0.7

    result = benchmark._compare_reports(candidate, baseline, 20)

    assert result["status"] == "review"
    assert [row["measurement"] for row in result["comparisons"] if row["review"]] == [
        "fixture/qem/cold"
    ]
    assert baseline == frozen
    assert benchmark._compare_reports(baseline, frozen, 20)["status"] == "within_threshold"


@pytest.mark.parametrize("change", ["gpu", "source", "allocator", "cache", "parity", "samples", "unstable"])
def test_unmatched_or_failed_measurements_cannot_pass_regression_check(change: str) -> None:
    baseline = _report()
    candidate = copy.deepcopy(baseline)
    if change == "gpu":
        candidate["device_identity"] = "another-physical-device"
    elif change == "source":
        candidate["cases"][0]["reference_identity"] = "different-body"
    elif change == "allocator":
        candidate["allocator_policy"] = "reset pools"
    elif change == "unstable":
        candidate["passed"] = False
    else:
        summary = candidate["cases"][0]["hdf5"]["cold_summary"]
        summary[{"cache": "cache_qualified", "parity": "correctness_passed",
                 "samples": "n"}[change]] = 0

    assert benchmark._compare_reports(candidate, baseline, 20)["status"] == "incomparable"


def test_cache_classification_uses_measured_residency() -> None:
    assert benchmark._cache_matches({"available": True, "resident_percent": 0}, "cold")
    assert benchmark._cache_matches({"available": True, "resident_percent": 100}, "warm")
    assert not benchmark._cache_matches({"available": True, "resident_percent": 36}, "cold")
    assert not benchmark._cache_matches({"available": False}, "warm")


def test_loading_schedule_keeps_first_separate_and_preserves_failed_evidence(monkeypatch) -> None:
    modes, snapshots = [], []

    def trial(path, files, backend, expected, mode):
        modes.append(mode)
        return {"mode": mode, "seconds": 1.0,
                "cache_before": {"available": True, "resident_percent": 0 if mode == "cold" else 100},
                "correctness": {"passed": True}}

    monkeypatch.setattr(benchmark, "_trial", trial)
    result = benchmark._measure(Path("test.qem"), [], "cuda", {}, 5,
                                lambda value: snapshots.append(copy.deepcopy(value)))
    assert modes == ["first", *[mode for _ in range(5) for mode in ("cold", "warm")]]
    assert result["cold_summary"]["n"] == result["warm_summary"]["n"] == 5
    assert len(snapshots) == 11

    def failed(*args):
        return {"correctness": {"passed": False}}

    monkeypatch.setattr(benchmark, "_trial", failed)
    snapshots.clear()
    with pytest.raises(ValueError, match="Frozen product mismatch"):
        benchmark._measure(Path("test.qem"), [], "cuda", {}, 5, snapshots.append)
    assert snapshots[0]["first"]["correctness"]["passed"] is False


def test_timer_excludes_product_check_and_releases_loaded_owner(monkeypatch) -> None:
    from quantem.gpu import io

    clock, released = [0.0], []

    class Loaded:
        metadata = {"load_timings": {"ready": 4.0}}
        resident_bytes = 256

        def __enter__(self):
            return self

        def __exit__(self, *args):
            released.append(True)

    def load(path, **kwargs):
        assert kwargs == {"backend": "cuda", "representation": "encoded",
                          "hot_pixel_correction": "none", "verbose": False}
        clock[0] += 3
        return Loaded()

    def sync(backend):
        clock[0] += 1

    def check(data, expected):
        clock[0] += 100
        return {"passed": True, "seconds": 100}

    monkeypatch.setattr(io, "load", load)
    monkeypatch.setattr(benchmark, "_sync_backend", sync)
    monkeypatch.setattr(benchmark, "_product_check", check)
    monkeypatch.setattr(benchmark, "_process_io", lambda: {})
    monkeypatch.setattr(benchmark, "_cache_snapshot", lambda paths: {})
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    result = benchmark._trial(Path("fixture.qem"), [], "cuda", {}, "warm")

    assert result["seconds"] == 4
    assert result["correctness"]["seconds"] == 100
    assert released == [True]


def test_native_hdf5_timer_includes_catalog_and_checks_frozen_frames(monkeypatch) -> None:
    measured = {"shape": [2, 2, 4, 4], "source_dtype": "uint16",
                "sample_frames": [0, 2, 3], "sample_hashes_u32_le": ["a", "b", "c"],
                "ready_seconds_including_preparation": 1.2, "load_seconds": 0.7,
                "physical_read_bytes": 100, "resident_bytes": 200,
                "verification_seconds_excluded": 0.1}
    expected = {"shape": measured["shape"], "dtype": "uint16",
                "sample_frames": measured["sample_frames"],
                "sample_hashes_u32_le": list(measured["sample_hashes_u32_le"])}
    monkeypatch.setenv("QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR", "/test-index")
    monkeypatch.setattr(benchmark, "_cache_snapshot", lambda paths: {})
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout=json.dumps({"acquisitions": [measured]})))
    result = benchmark._native_trial(Path("fixture.h5"), [], expected, "warm", Path("native"))
    assert result["seconds"] == 1.2
    assert result["correctness"]["passed"]
    measured["sample_hashes_u32_le"][1] = "changed"
    result = benchmark._native_trial(Path("fixture.h5"), [], expected, "warm", Path("native"))
    assert not result["correctness"]["passed"]
