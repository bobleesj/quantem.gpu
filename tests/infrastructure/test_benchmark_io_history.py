"""Retain successful and failed measurements without replacing frozen evidence."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark_io_history import _register


def test_history_is_idempotent_and_preserves_earlier_report_bytes(tmp_path: Path) -> None:
    report = {"schema": "quantem-gpu-io-loading/v1", "backend": "mps",
              "device": {"name": "test device"}, "device_identity": "device-1",
              "storage_id": "ssd-1", "source_fingerprint": "source-1",
              "passed": True, "performance": {"status": "no_baseline"},
              "boundary": "synchronized load", "cases": []}
    source = tmp_path / "loading.json"
    source.write_text(json.dumps(report))
    original = source.read_bytes()
    history = tmp_path / "history"
    _register([source], history)
    index = _register([source], history)
    assert len(index["runs"]) == 1
    assert index["runs"][0]["baseline"] is None

    report.update(passed=False, performance={"status": "review"})
    source.write_text(json.dumps(report))
    index = _register([source], history)
    assert len(index["runs"]) == 2
    frozen = history / "records" / (hashlib.sha256(original).hexdigest() + ".json")
    assert frozen.read_bytes() == original
    assert [row["correctness_passed"] for row in index["runs"]] == [True, False]
    assert "review" in (history / "README.md").read_text()
