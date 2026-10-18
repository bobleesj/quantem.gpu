"""Keep routine SSB verification on frozen GPU results."""

import subprocess
import sys
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


def test_default_parity_uses_the_frozen_gpu_fit(tmp_path):
    """The normal parity command dispatches a GPU fit, not the CPU oracle."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path("scripts/check_ssb_parity.sh").read_text()
    (scripts / "check_ssb_parity.sh").write_text(source)
    (scripts / "check_ssb_fit_trajectory.sh").write_text(
        'printf "gpu-fit %s\\n" "$*"\n'
    )
    completed = subprocess.run(
        ["bash", str(scripts / "check_ssb_parity.sh"), "--build-metal"],
        capture_output=True, text=True, check=True,
    )
    assert "gpu-fit --build-metal" in completed.stdout
    assert "no CPU oracle" in completed.stdout


def test_cpu_oracle_needs_explicit_opt_in(monkeypatch):
    """Neither pytest nor the direct oracle command starts CPU work implicitly."""
    from tests.parity import test_ssb_strict_parity as workflow

    monkeypatch.delenv("QUANTEM_SSB_CPU_ORACLE", raising=False)
    with pytest.raises(pytest.skip.Exception, match="CPU oracle is opt-in"):
        workflow._measure(("arina-128-full-disk",), use_metal=True, tag="unused")
    completed = subprocess.run(
        [sys.executable, "tests/parity/ssb_parity_gate.py", "--metal-only"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "CPU reconstruction is opt-in" in completed.stderr


def test_gpu_report_rejects_nondeterminism_and_impurity():
    """Printed GPU acceptance checks must also fail the command."""
    from scripts.ssb_fit_trajectory_report import production_checks_pass

    report = {
        "productionOptimizeAlwaysMatchesSequential": True,
        "objectivePurity": {
            "repeatSameEngineBitwiseMismatches": 0,
            "freshEngineBitwiseMismatches": 0,
            "float32AliasBitwiseMismatches": 0,
        },
        "sequential": {
            "bestC10Angstroms": 1.0,
            "bestC12Angstroms": 0.0,
            "bestPhi12Radians": 0.0,
            "bestLoss": 0.5,
            "trials": [{
                "c10Angstroms": 1.0, "c12Angstroms": 0.0,
                "phi12Radians": 0.0, "loss": 0.5, "stage": "global",
            }],
        },
    }
    assert production_checks_pass(report, deepcopy(report))
    changed = deepcopy(report)
    changed["sequential"]["trials"][0]["loss"] = 0.6
    assert not production_checks_pass(report, changed)
    changed = deepcopy(report)
    changed["sequential"]["bestLoss"] = 0.6
    assert not production_checks_pass(report, changed)
    for field in report["objectivePurity"]:
        changed = deepcopy(report)
        changed["objectivePurity"][field] = 1
        assert not production_checks_pass(changed)
        assert not production_checks_pass(report, changed)
    changed = deepcopy(report)
    changed["productionOptimizeAlwaysMatchesSequential"] = False
    assert not production_checks_pass(changed)


def test_report_matches_reference_trials_in_engine_units():
    """A recorded reference trial (the MPS search's C10_A, C12_A, phi12_deg) keys like the native point it equals."""
    import math

    from scripts.ssb_fit_trajectory_report import _reference_candidates, float32_key

    native = {"c10Angstroms": 73.25, "c12Angstroms": 14.5, "phi12Radians": 30.0 * math.pi / 180.0}
    entry = {"params": {"C10_A": 73.25, "C12_A": 14.5, "phi12_deg": 30.0}}
    assert _reference_candidates(entry) == (float32_key(native),)


def test_archived_reference_reuses_counts_without_editing_fixture(tmp_path):
    """A saved exact-count fixture opens through a disposable current-schema view."""
    from scripts.ssb_fit_reference_trajectory import _reference_input

    source = tmp_path / "source"
    (source / "snapshots").mkdir(parents=True)
    (source / "bf_columns.u16").write_bytes(b"\x01\x00" * 4)
    meta = {"voltage_kV": 200, "semiangle_mrad": 20,
            "scan_sampling_A": [0.2, 0.3], "det_sampling_mrad": 1.2,
            "rotation_angle_deg": 17, "scan_region": [0, 2, 0, 2],
            "scan_shape": [2, 2]}
    calibration = {"bf_rows": [1], "bf_cols": [2], "bf_center": [1, 2],
                   "bf_radius_px": 1, "detector_shape": [4, 4],
                   "bf_excluded_detector_pixels": []}
    (tmp_path / "case.json").write_text(json.dumps(meta))
    (source / "snapshots/cal.json").write_text(json.dumps(calibration))
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with _reference_input(tmp_path) as (view, loaded, options):
        assert (view / "bf_columns.u16").resolve() == source / "bf_columns.u16"
        assert options["scan_sampling_A"] == (0.2, 0.3)
        assert options["detector_sampling_mrad"] == (1.2, 1.2)
        assert options["scan_detector_rotation_deg"] == 17
        assert loaded == meta
        current = json.loads((view / "cal.json").read_text())
        assert current["bright_field_rows"] == calibration["bf_rows"]
        assert current["scan_shape"] == [2, 2]
    assert not view.exists()
    assert all(path.read_bytes() == content for path, content in original.items())
    (source / "snapshots/cal.json").write_text(json.dumps(current))
    meta["detector_sampling_mrad"] = [1.1, 1.3]
    meta["scan_detector_rotation_deg"] = 19
    (tmp_path / "case.json").write_text(json.dumps(meta))
    with _reference_input(tmp_path) as (view, _, options):
        assert view == source
        assert options["detector_sampling_mrad"] == (1.1, 1.3)
        assert options["scan_detector_rotation_deg"] == 19


def test_full_fit_runner_separates_outputs_and_repeats_both_sizes(tmp_path, monkeypatch):
    """One invocation records both sizes without overwriting fixtures or earlier runs."""
    from scripts import ssb_fit_gate

    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "inputs with spaces"
    case = runs / "strict-parity/arina-128-full-disk/case.json"
    case.parent.mkdir(parents=True)
    case.write_text("{}")
    pin = tmp_path / "tests/parity/fixtures/ssb_fit_trajectory_128.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"case_json_sha256": hashlib.sha256(b"{}").hexdigest()}))
    monkeypatch.setenv("QUANTEM_SSB_PARITY_OUTPUT", str(tmp_path / "results"))
    monkeypatch.setenv("QUANTEM_SSB_PARITY_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(sys, "argv", ["gate", "--runs", str(runs), "--batch-mode", "closure", "--full"])
    commands = []
    monkeypatch.setattr(ssb_fit_gate.subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert ssb_fit_gate.main() == 0
    assert sum(command[0] == "build/ssb-fit-trajectory" for command in commands) == 4
    references = [command for command in commands if "scripts/ssb_fit_reference_trajectory.py" in command]
    assert len(references) == 2
    assert all(str(runs) in command for command in references)
    assert case.read_bytes() == b"{}"
    with pytest.raises(FileExistsError):
        ssb_fit_gate.main()
    case.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="differs from pin"):
        ssb_fit_gate.main()
