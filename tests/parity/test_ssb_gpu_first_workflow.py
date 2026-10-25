"""Keep routine SSB verification on the GPU gates."""

import subprocess
import sys
from pathlib import Path

import pytest


def _stub_gate(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "check_ssb_parity.sh").write_text(Path("scripts/check_ssb_parity.sh").read_text())
    (scripts / "check_ssb_fit_trajectory.sh").write_text('printf "gpu-fit %s\\n" "$*"\n')
    return scripts / "check_ssb_parity.sh"


def test_default_parity_uses_the_gpu_agreement_gate(tmp_path):
    """The normal parity command dispatches the GPU agreement gate, not the CPU oracle."""
    completed = subprocess.run(
        ["bash", str(_stub_gate(tmp_path))], capture_output=True, text=True, check=True,
    )
    assert "gpu-fit" in completed.stdout
    assert "no CPU oracle" in completed.stdout


@pytest.mark.parametrize("flag", ["--full", "--export", "--build-metal", "--metal-only"])
def test_oracle_and_retired_flags_are_refused_without_the_oracle(tmp_path, flag):
    """Oracle flags need --cpu-oracle; the retired native harness flags are unknown."""
    completed = subprocess.run(
        ["bash", str(_stub_gate(tmp_path)), flag], capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "gpu-fit" not in completed.stdout


def test_cpu_oracle_needs_explicit_opt_in(monkeypatch):
    """Neither pytest nor the direct oracle command starts CPU work implicitly."""
    from tests.parity import test_ssb_strict_parity as workflow

    monkeypatch.delenv("QUANTEM_SSB_CPU_ORACLE", raising=False)
    with pytest.raises(pytest.skip.Exception, match="CPU oracle is opt-in"):
        workflow._measure(("arina-128-full-disk",), tag="unused")
    completed = subprocess.run(
        [sys.executable, "tests/parity/ssb_parity_gate.py"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "CPU reconstruction is opt-in" in completed.stderr
