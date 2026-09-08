import json
import subprocess
import sys
from pathlib import Path

import pytest

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
    assert "70 platform/module cells" in result.stdout
    experiment_count = len(list(Path("experiments").glob("*/manifest.json")))
    assert f"{experiment_count} retained experiments" in result.stdout


def test_profile_matrix_has_one_atomic_cell_per_backend_capability() -> None:
    plan = json.loads(PROFILE_MATRIX.read_text(encoding="utf-8"))
    cells = plan["cells"]

    assert len(cells) == 70
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
    # Historical scientific results are retained, but were not linked as
    # complete cell-scoped release evidence by the original profiling plan.
    assert cells["ssb.object-phase-loss::swift-metal"]["state"] == "evidence-gap"
    assert (
        cells["ssb.calibration-200-nelder-mead::swift-metal"]["state"] == "evidence-gap"
    )
    assert cells["io.selective-scan-loading::cpu-reference"]["state"] == "unsupported"
    assert cells["io.selective-scan-loading::cuda"]["state"] == "evidence-gap"
    assert cells["io.selective-scan-loading::mps"]["state"] == "evidence-gap"
    assert cells["io.selective-scan-loading::swift-metal"]["state"] == "unsupported"
    assert cells["io.selective-scan-loading::webgpu"]["state"] == "evidence-gap"
    assert cells["io.selective-scan-loading::vulkan"]["state"] == "unsupported"
    assert cells["io.decode-bin-provenance::vulkan"]["state"] == "evidence-gap"
    assert (
        cells["display.transform-histogram-color-fft::direct3d"]["state"]
        == "evidence-gap"
    )

    for cell in cells.values():
        if cell["backend"] in {"vulkan", "direct3d"}:
            assert cell["scheduled_profile"] == "none"
            assert cell["release_signoff"] is False

    for cell in cells.values():
        if cell["support_level"] == "not-implemented":
            assert cell["state"] == "unsupported"
            assert cell["scheduled_profile"] == "none"
            assert cell["release_signoff"] is False


@pytest.fixture
def experiment_registry(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import check_profile_registry as registry

    monkeypatch.setattr(registry, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(registry, "RUNS_INDEX", tmp_path / "RUNS.md")
    return registry


def _retained_experiment(registry, *, canonical=True, status=None):
    experiment_id = "20260906-test"
    artifact_id = "bounded-integer-check"
    output = {"path": "local-evidence://check.log", "sha256": "a" * 64}
    if canonical:
        output.update(
            artifact_id=artifact_id,
            size_bytes=12,
            retention="durable",
            consuming_figures=[],
        )
    else:
        output.update(artifact=artifact_id, result="Exact integer checks pass")
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": status or ("succeeded" if canonical else "completed"),
        "question": "Are integer outputs exact?",
        "paper": {},
        "code": {"revision": "b" * 40, "dirty": False},
        "inputs": [{"dataset_id": "bounded-fixture", "sha256": "c" * 64}],
        "parameters": {"artifact_results": {artifact_id: "Exact integer checks pass"}}
        if canonical
        else {},
        "execution": {},
        "outputs": [output],
        "timestamps": {"finished": "2026-09-06T00:00:00Z"},
    }
    directory = registry.EXPERIMENT_ROOT / experiment_id
    directory.mkdir()
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    registry.RUNS_INDEX.write_text(f"| {experiment_id} | integer check | CPU | ok |\n")
    return manifest, path


@pytest.mark.parametrize("canonical", [False, True])
def test_registry_preserves_historical_and_canonical_manifests(
    experiment_registry, canonical
):
    registry = experiment_registry
    _retained_experiment(registry, canonical=canonical)
    errors = []
    count, identifiers = registry._validate_experiments(errors)
    registry._validate_runs_index(identifiers, errors)
    assert count == 1
    assert errors == []


@pytest.mark.parametrize(
    "field", ["artifact_results", "size_bytes", "retention", "consuming_figures"]
)
def test_canonical_manifest_still_requires_claim_and_retention(
    experiment_registry, field
):
    registry = experiment_registry
    manifest, path = _retained_experiment(registry)
    if field == "artifact_results":
        del manifest["parameters"][field]
    else:
        del manifest["outputs"][0][field]
    path.write_text(json.dumps(manifest))
    errors = []
    registry._validate_experiments(errors)
    assert errors


def test_unknown_experiment_status_reports_error_instead_of_crashing(
    experiment_registry,
):
    registry = experiment_registry
    _retained_experiment(registry, status="unknown")
    errors = []
    _, identifiers = registry._validate_experiments(errors)
    registry._validate_runs_index(identifiers, errors)
    assert any("invalid status" in error for error in errors)
