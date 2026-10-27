"""The installed benchmark preserves the fit budget and export boundary."""

import json
from types import SimpleNamespace

import pytest
from quantem.gpu import benchmark, cli


def test_report_separates_first_fit_from_warm_measurements():
    report = benchmark._summary([10.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert report["first_s"] == 10.0
    assert report["warm_n"] == 5
    assert report["warm_p50_s"] == 3.0
    assert report["warm_p90_s"] == pytest.approx(4.6)


def test_two_rotation_branches_require_the_full_gpu_trial_budget():
    result = SimpleNamespace(
        optimizer={"device": "mps", "host_numerical_search": False},
        refinement="newton",
        trial_records=[None] * 400,
        aberrations={"C10_nm": 8.0},
        sample_tilt_mrad=(2.0, -3.0),
        depth_spread_nm=12.0,
        scan_detector_rotation_deg=345.0,
        sample_agreement=10.0,
        timings={"coarse_search_s": 0.3},
        source_path="private-acquisition.h5",
    )
    record = benchmark._fit_record(result, "mps", 200, 2)
    assert record["completed_trials"] == 400
    assert "private-acquisition" not in json.dumps(record)
    result.trial_records = result.trial_records[:200]
    with pytest.raises(RuntimeError, match="requested trials"):
        benchmark._fit_record(result, "mps", 200, 2)
    result.trial_records *= 2
    result.optimizer["host_numerical_search"] = True
    with pytest.raises(RuntimeError, match="selected GPU"):
        benchmark._fit_record(result, "mps", 200, 2)


def test_cli_records_portable_report_and_scalar_calibration(tmp_path, monkeypatch):
    received = {}

    def run(source, **kwargs):
        received.update(kwargs)
        return {"measurements": {"full_field_fit": benchmark._summary([2.0, 1.0])}}

    monkeypatch.setattr(benchmark, "run_ssb", run)
    output = tmp_path / "benchmark.json"
    assert (
        cli.main(
            [
                "benchmark-ssb",
                "scan.qem",
                "--out",
                str(output),
                "--runs",
                "2",
                "--scan-sampling-a",
                "0.5",
                "--require-installed",
            ]
        )
        == 0
    )
    assert received["calibration"] == {"scan_sampling_A": 0.5}
    assert received["require_installed"] is True
    assert received["num_trials"] == 200 and received["seed"] == 42
    assert (
        json.loads(output.read_text())["measurements"]["full_field_fit"]["warm_n"] == 1
    )
