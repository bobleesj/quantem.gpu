"""Opt-in real-browser evidence gates for held-drag black clipping."""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def evidence() -> Path:
    root = os.environ.get("QUANTEM_WEBGPU_LIVE_RANGE_EVIDENCE")
    if not root:
        pytest.skip("Set QUANTEM_WEBGPU_LIVE_RANGE_EVIDENCE to a seven-tilt browser run")
    return Path(root)


def test_off_disk_held_images_match_release(evidence: Path) -> None:
    """An intensity-drop fixture fails the frozen renderer but passes the fix."""
    results = json.loads((evidence / "range-travel-results.json").read_text())
    baseline = next(row for row in results if "/baseline/" in row["url"])
    candidates = [row for row in results if "/baseline/" not in row["url"]]
    assert len(baseline["panels"]) == 7
    assert all(panel["held_dark_fraction"] == 1 for panel in baseline["panels"])
    assert all(panel["held_settled_mismatches"] > 0 for panel in baseline["panels"])
    assert candidates
    for result in candidates:
        assert result["held_geometry"] == result["settled_geometry"]
        assert result["held_geometry"] == baseline["held_geometry"]
        assert len(result["panels"]) == 7
        for panel in result["panels"]:
            assert panel["held_settled_mismatches"] == panel["return_mismatches"] == 0
            assert panel["held_dark_fraction"] < 0.01


def test_gpu_live_ranges_match_fixed_range_renderer(evidence: Path) -> None:
    """Changing intensity, signed log, invalid values and zeros preserve RGBA."""
    reports = [json.loads(path.read_text()) for path in evidence.glob("audit-*.json")]
    report = next(row for row in reports if row.get("kind") == "live-range-parity")
    assert report["gpuErrors"] == report["deviceLosses"] == []
    assert len(report["live"]) == 5
    for generation in report["live"]:
        assert generation["rendered"] == len(generation["panels"]) == 7
        assert all(panel["mismatches"] == panel["maxError"] == 0
                   for panel in generation["panels"])


def test_scientific_counts_and_held_gpu_ownership(evidence: Path) -> None:
    """The display correction leaves exact source counts and residency intact."""
    parity = json.loads((evidence / "parity-final.json").read_text())
    assert parity["source_count"] == 7 and parity["exact"]
    assert len(parity["comparisons"]) >= 14
    for result in parity["comparisons"]:
        assert result["elements"] == 512 * 512
        assert result["mismatches"] == result["max_abs_error"] == 0
    timings = json.loads((evidence / "timings.json").read_text())
    assert {row["geometry"]["mode"] for row in timings} >= {"circle", "annular"}
    for row in timings:
        assert row["source_identity_count"] == 7
        assert row["range_readback_bytes"] == 0
        assert row["errors"] == row["device_losses"] == []
        for call in ("createBuffer", "createBindGroup", "mapAsync", "transferToImageBitmap"):
            assert row["calls"][call]["count"] == 0
