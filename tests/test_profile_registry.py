import json
import subprocess
import sys
from pathlib import Path

PROFILE_MATRIX = Path("benchmarks/profile_matrix.json")


def _cells() -> dict[str, dict]:
    plan = json.loads(PROFILE_MATRIX.read_text(encoding="utf-8"))
    return {cell["id"]: cell for cell in plan["cells"]}


def test_profile_registry_validator_accepts_retained_evidence() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_profile_registry.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "35 platform/module cells" in result.stdout
    assert "12 retained experiments" in result.stdout


def test_profile_matrix_has_one_atomic_cell_per_backend_capability() -> None:
    plan = json.loads(PROFILE_MATRIX.read_text(encoding="utf-8"))
    cells = plan["cells"]

    assert len(cells) == 35
    assert len({cell["id"] for cell in cells}) == len(cells)
    assert all(
        cell["id"] == f"{cell['capability']}::{cell['backend']}" for cell in cells
    )


def test_profile_matrix_keeps_current_gaps_and_unsupported_paths_explicit() -> None:
    cells = _cells()

    assert cells["screening.prepared-products::cuda"]["state"] == "evidence-gap"
    assert cells["dpc.com-rotation-idpc::swift-metal"]["state"] == "evidence-gap"
    assert cells["dpc.com-rotation-idpc::webgpu"]["state"] == "evidence-gap"
    assert cells["ssb.calibration-200-nelder-mead::mps"]["state"] == "evidence-gap"
    assert cells["ssb.object-phase-loss::swift-metal"]["state"] == "ready"
    assert cells["ssb.calibration-200-nelder-mead::swift-metal"]["state"] == "ready"

    for cell in cells.values():
        if cell["support_level"] == "not-implemented":
            assert cell["state"] == "unsupported"
            assert cell["scheduled_profile"] == "none"
            assert cell["release_signoff"] is False
