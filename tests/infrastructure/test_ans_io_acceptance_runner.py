"""Acceptance reports must not turn missing or partial execution into support."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from scripts import check_ans_io as gate


def test_gate_requires_complete_test_execution():
    """Skipped checks and teardown failures remain visible in the final cell."""
    results = gate._Results()

    def observe(node, when="call", failed=False, skipped=False):
        results.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid=node,
                when=when,
                duration=0.01,
                user_properties=[],
                failed=failed,
                skipped=skipped,
                longrepr="fixture unavailable",
            )
        )

    observe("roundtrip")
    assert gate._cell(list(results.tests.values())) == "passed"
    observe("metadata", when="setup", skipped=True)
    assert gate._cell(list(results.tests.values())) == "blocked"
    observe("roundtrip", when="teardown", failed=True)
    assert gate._cell(list(results.tests.values())) == "failed"
    assert gate._cell([]) == "not-run"


def test_reports_only_combine_matching_source_candidates(tmp_path):
    """A shared table cannot accidentally mix different implementation snapshots."""
    reports = []
    for backend in ("mps", "cuda"):
        path = tmp_path / f"{backend}.json"
        path.write_text(
            json.dumps(
                dict(
                    backend=backend,
                    source_fingerprint=backend,
                    status="passed",
                    date="2026-09-21",
                    cells={"NumPy uint8/uint16": "passed"},
                    device={"name": "test fixture"},
                )
            )
        )
        reports.append(path)
    output = tmp_path / "combined.json"
    command = [
        sys.executable,
        str(Path(gate.__file__)),
        "--combine",
        *map(str, reports),
        "--output",
        str(output),
    ]
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0 and "fingerprints differ" in rejected.stderr
    assert not output.exists()
    record = json.loads(reports[1].read_text())
    record["source_fingerprint"] = "mps"
    reports[1].write_text(json.dumps(record))
    accepted = subprocess.run(command, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert (
        "| NumPy uint8/uint16 | passed | passed | not-run |"
        in output.with_suffix(".md").read_text()
    )
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and "never replaced" in repeated.stderr


def test_missing_device_blocks_instead_of_falling_back(tmp_path, monkeypatch):
    """A machine without the requested accelerator cannot certify that backend."""

    def unavailable(backend):
        raise RuntimeError("No physical device")

    monkeypatch.setattr(gate, "_device", unavailable)
    monkeypatch.setattr(gate, "_fingerprint", lambda: "synthetic")
    result = gate._run("cuda", tmp_path / "cuda.json", None)
    assert result["status"] == "blocked"
    assert result["tests"] == []
    assert "passed" not in result["cells"].values()
