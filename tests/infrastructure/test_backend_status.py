"""The readiness view must not turn source presence into scientific signoff."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import backend_status
from scripts.backend_status import BACKENDS, build_status, render_status


def _matrices() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads(Path(path).read_text())
        for path in (
            "tests/parity/backend_matrix.json",
            "benchmarks/profile_matrix.json",
            "benchmarks/benchmark_registry.json",
        )
    )


def _run_record(root: Path, cell: dict) -> dict:
    result = root / "result.bin"
    result.write_bytes(b"independent regression-test result")
    return {
        "schema_version": 1,
        "protocol_version": "quantem-gpu-cell-evidence/v1",
        "cell_id": cell["id"],
        "backend": cell["backend"],
        "runner": cell["runner"],
        "status": "passed",
        "source_revision": "1" * 40,
        "fixture": {"id": "temporary-fixture-v1", "sha256": "2" * 64},
        "result": {
            "path": result.name,
            "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        },
        "outcomes": {
            cell["pr_gate"]: "passed",
            "scientific-parity": "passed",
            "real-data-e2e": "passed",
        },
    }


def _retain_record(root: Path, cell: dict, run: dict) -> Path:
    artifact = root / "retained-run.json"
    artifact.write_text(json.dumps(run))
    cell["retained_evidence"] = [
        {
            "path": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    ]
    return artifact


def test_readiness_keeps_missing_features_and_measurements_visible() -> None:
    status = build_status(*_matrices())

    assert set(status["backends"]) == set(BACKENDS)
    for backend, entry in status["backends"].items():
        assert len(entry["cells"]) == 10
        assert entry["blocking_cells"] == [
            cell["id"]
            for cell in entry["cells"]
            if cell["release_signoff"] and cell["evidence"] != "ready"
        ]
        if backend in {"vulkan", "direct3d"}:
            assert entry["signoff"] == "not-scheduled"
            assert entry["performance"]["state"] == "not-recorded"
            assert entry["performance"]["measurements"] == []
        else:
            assert entry["signoff"] == "blocked"
    # Existing workload-specific measurements are not erased by correcting
    # unlinked coarse readiness declarations.
    retained = status["backends"]["swift-metal"]["performance"]["measurements"]
    assert retained
    assert all("source_revision" in row and "cache_state" in row for row in retained)


def test_signoff_requires_hashed_retained_output_not_a_test_path(
    tmp_path: Path,
) -> None:
    capabilities, profile, _ = _matrices()
    for cell in profile["cells"]:
        cell["release_signoff"] = False
    selected = profile["cells"][0]
    selected.update(state="ready", release_signoff=True)
    with pytest.raises(ValueError, match="ready requires retained_evidence"):
        build_status(capabilities, profile, {}, root=tmp_path)

    artifact = _retain_record(tmp_path, selected, _run_record(tmp_path, selected))
    status = build_status(capabilities, profile, {}, root=tmp_path)
    assert status["backends"][selected["backend"]]["signoff"] == "ready"
    artifact.write_text('{"result": "different"}\n')
    with pytest.raises(ValueError, match="digest differs"):
        build_status(capabilities, profile, {}, root=tmp_path)


@pytest.mark.parametrize(
    "change, message",
    [
        ("test-source", "structured run-evidence JSON required"),
        ("backend", "backend must be"),
        ("runner", "runner must be"),
        ("cell_id", "cell_id must be"),
        ("schema_version", "schema_version must be"),
        ("status", "status must be"),
        ("source_revision", "full source_revision"),
        ("fixture", "fixture id and SHA-256"),
        ("result", "missing retained evidence"),
        ("failed-parity", "scientific-parity.*passed outcome"),
        ("missing-real-data", "real-data-e2e.*passed outcome"),
        ("missing-pr-gate", "required gate.*passed outcome"),
        ("additional-required-gate", "minimum-memory.*passed outcome"),
    ],
)
def test_ready_claim_requires_matching_executed_evidence(
    tmp_path: Path,
    change: str,
    message: str,
) -> None:
    capabilities, profile, _ = _matrices()
    selected = next(cell for cell in profile["cells"] if cell["backend"] == "cuda")
    selected["state"] = "ready"
    run = _run_record(tmp_path, selected)
    if change == "failed-parity":
        run["outcomes"]["scientific-parity"] = "failed"
    elif change == "missing-real-data":
        run["outcomes"].pop("real-data-e2e")
    elif change == "missing-pr-gate":
        run["outcomes"].pop(selected["pr_gate"])
    elif change == "additional-required-gate":
        profile["required_evidence"]["required_outcomes"].append("minimum-memory")
    elif change == "result":
        run["result"]["path"] = "missing-result.bin"
    elif change != "test-source":
        run[change] = "incorrect"
    artifact = _retain_record(tmp_path, selected, run)
    if change == "test-source":
        # A real Python test source, correctly hashed, is still not a run.
        artifact.write_bytes(Path(__file__).read_bytes())
        selected["retained_evidence"][0]["sha256"] = hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError, match=message):
        build_status(capabilities, profile, {}, root=tmp_path)


@pytest.mark.parametrize(
    "change, message",
    [
        ("missing-cell", "missing profile cell"),
        ("duplicate-cell", "duplicate profile cell"),
        ("missing-backend", "retain every supported contract backend"),
        ("missing-coverage", "coverage must retain every backend"),
        ("missing-platform", "profile platforms must match"),
        ("missing-capability", "required capabilities disappeared"),
        ("duplicate-capability", "duplicate capability"),
        ("support-mismatch", "support level disagrees"),
        ("unknown-performance", "unknown platform"),
    ],
)
def test_conformance_join_fails_closed_when_a_backend_or_claim_drifts(
    change: str,
    message: str,
) -> None:
    capabilities, profile, benchmarks = _matrices()
    if change == "missing-cell":
        profile["cells"].pop()
    elif change == "duplicate-cell":
        profile["cells"].append(profile["cells"][0])
    elif change == "missing-backend":
        capabilities["backends"].pop()
    elif change == "missing-coverage":
        capabilities["capabilities"][0]["coverage"].pop("cuda")
    elif change == "missing-platform":
        profile["platforms"].pop("cuda")
    elif change == "missing-capability":
        missing = capabilities["capabilities"].pop()["id"]
        profile["capabilities"].pop(missing)
        profile["cells"] = [
            cell for cell in profile["cells"] if cell["capability"] != missing
        ]
    elif change == "duplicate-capability":
        capabilities["capabilities"].append(capabilities["capabilities"][0])
    elif change == "support-mismatch":
        profile["cells"][0]["support_level"] = "required"
    else:
        benchmarks["gates"][0]["platform"] = "unknown-accelerator"
    with pytest.raises(ValueError, match=message):
        build_status(capabilities, profile, benchmarks)


def test_cli_and_generated_document_are_the_same_readiness_view() -> None:
    status = build_status(*_matrices())
    document = Path("docs/_generated/backend_readiness.md")
    assert document.read_text() == render_status(status)
    result = subprocess.run(
        [sys.executable, "scripts/backend_status.py", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [sys.executable, "scripts/backend_status.py", "json", "--backend", "vulkan"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == status["backends"]["vulkan"]


def test_cli_rejects_stale_generated_docs_and_duplicate_json_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    for relative in (
        "tests/parity/backend_matrix.json",
        "benchmarks/profile_matrix.json",
        "benchmarks/benchmark_registry.json",
        "docs/_generated/backend_readiness.md",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(Path(relative).read_text())
    monkeypatch.setattr(backend_status, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["backend_status.py", "check"])
    assert backend_status.main() == 0
    document = tmp_path / "docs/_generated/backend_readiness.md"
    document.write_text(document.read_text() + "Unmaintained status.\n")
    assert backend_status.main() == 1
    assert "readiness is stale" in capsys.readouterr().out
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"state": "evidence-gap", "state": "ready"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        backend_status._read(duplicate)
