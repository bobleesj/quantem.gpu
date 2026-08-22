from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REGISTRY = Path("benchmarks/benchmark_registry.json")
GENERATED = Path("docs/_generated/benchmark_coverage.md")


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_every_module_platform_and_unrun_load_plan_stays_visible() -> None:
    registry = _registry()
    coverage: dict[str, set[str]] = defaultdict(set)
    for gate in registry["gates"]:
        coverage[gate["module"]].add(gate["platform"])

    platforms = {
        "CPU reference",
        "CUDA",
        "Python MPS",
        "Native Swift/Metal",
        "WebGPU",
    }
    assert set(coverage) == {
        "I/O and load",
        "Detector products",
        "Screening",
        "CoM, DPC, and iDPC",
        "Display and FFT",
        "Single-sideband ptychography",
        "Selective loading",
    }
    assert all(backends == platforms for backends in coverage.values())

    gates = {gate["id"]: gate for gate in registry["gates"]}
    assert {
        gates[f"io.mps.apple-m5-max-128gb.bin{detector_bin}.cold-original"]["state"]
        for detector_bin in (1, 2, 4, 8)
    } == {"pending"}
    assert gates["io.swift.apple-m5-max-128gb.bin8.contract"]["state"] == "unsupported"
    assert gates["io.swift.apple-m2-air-8gb.bin2.cold-original"]["state"] == "pending"
    assert gates["io.swift.apple-m5-24gb.bin1.prepared"]["state"] == "partial"


def test_registry_validator_and_generated_table_agree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_registry.py", "validate"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "77 gates, 33 measurements, and 14 runbooks" in result.stdout

    rendered = GENERATED.read_text(encoding="utf-8")
    for field in (
        "State",
        "Module",
        "Platform",
        "Computer",
        "Selected scan",
        "Source detector",
        "Detector bin",
        "p50",
        "p95",
        "Maximum",
        "Logical resident",
        "Driver allocated after load",
        "Driver allocated after release",
        "Accelerator peak",
        "Total-device peak",
        "Process/tree peak",
        "Swap delta",
        "Device tested",
        "Date tested",
        "Revision",
    ):
        assert field in rendered
    for state in ("✓ Measured", "◐ Partial", "○ Pending", "× Refuted", "Not supported"):
        assert state in rendered


def test_computer_labels_describe_reproducible_hardware() -> None:
    registry = _registry()
    allowed = {
        "Linux CUDA workstation (dual 96 GB Blackwell GPUs)",
        "MacBook Air (M2, 8 GB)",
        "MacBook Pro (M5, 24 GB)",
        "MacBook Pro (M5 Max, 128 GB)",
        "Portable CI runner",
    }

    assert {gate["computer"] for gate in registry["gates"]} <= allowed
    assert {
        measurement["computer"]
        for measurement in registry["additional_measurements"]
    } <= allowed

    rendered = GENERATED.read_text(encoding="utf-8")
    for label in allowed - {"Portable CI runner"}:
        assert label in rendered

    measurements = rendered.split("## Retained atomic measurements", 1)[1].split(
        "## Reproducible runbooks", 1
    )[0]
    public_prefixes = (
        "`linux-dual-blackwell-96gb-",
        "`macbook-air-m2-8gb-",
        "`macbook-pro-m5-24gb-",
        "`macbook-pro-m5-max-128gb-",
    )
    measurement_rows = [
        line for line in measurements.splitlines() if line.startswith("| ✓")
    ]
    assert measurement_rows
    assert all(
        any(prefix in row for prefix in public_prefixes) for row in measurement_rows
    )

    writing = Path("docs/developer/writing.md").read_text(encoding="utf-8")
    assert "Identify a benchmark computer by reproducible hardware" in writing


def test_agent_can_resolve_the_next_gate_and_real_command() -> None:
    next_result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_registry.py",
            "next",
            "--computer",
            "MacBook Air (M2, 8 GB)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "io.swift.apple-m2-air-8gb.bin2.cold-original" in next_result.stdout
    assert "io.webgpu.apple-m2-air-8gb.bin2.minimum-memory" in next_result.stdout

    command = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_registry.py",
            "command",
            "io.swift.apple-m2-air-8gb.bin2.cold-original",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "swift run -c release metal-4dstem-indexed-load-benchmark" in command
    assert "--detector-bin 2" in command
    assert "Preflight:" in command
    assert "Required artifacts:" in command
    assert "Live4DSTEM policy, UI, cache lifecycle" in command


def test_every_runbook_command_resolves_to_repository_source() -> None:
    registry = _registry()
    commands = [runbook["command"] for runbook in registry["runbooks"].values()]
    script_paths = {
        Path(match)
        for command in commands
        for match in re.findall(r"(scripts/[A-Za-z0-9_-]+\.py)", command)
    }
    assert script_paths
    assert all(path.is_file() for path in script_paths)

    package = Path("Package.swift").read_text(encoding="utf-8")
    for executable in (
        "metal-4dstem-indexed-load-benchmark",
        "metal-ssb-benchmark",
    ):
        assert f'name: "{executable}"' in package

    for script in (
        "scripts/benchmark_hdf5_load.py",
        "scripts/benchmark_screening.py",
        "scripts/benchmark_selective_load.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_registry_keeps_private_paths_out_and_routes_docs_to_one_owner() -> None:
    registry_text = REGISTRY.read_text(encoding="utf-8")
    assert "/Users/" not in registry_text
    assert "/home/" not in registry_text

    coverage = Path("docs/performance/coverage.md").read_text(encoding="utf-8")
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    testing = Path("docs/developer/testing.md").read_text(encoding="utf-8")
    assert "python scripts/benchmark_registry.py next" in coverage
    assert "_generated/benchmark_coverage.md" in coverage
    assert "## Coverage and next runs" in dashboard
    assert "performance/coverage.md" in dashboard
    assert "scripts/benchmark_selective_load.py" in testing
    assert "scripts/benchmark_screening.py" in testing
