"""Keep routine SSB verification on frozen GPU results."""

import subprocess
import sys
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
            "bestC10Nanometers": 1.0,
            "bestC12Nanometers": 0.0,
            "bestPhi12Radians": 0.0,
            "bestLoss": 0.5,
            "trials": [{
                "c10Nanometers": 1.0, "c12Nanometers": 0.0,
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
