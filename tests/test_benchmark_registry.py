from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_registry import _measurement_state, resolved_measurements

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
    assert {
        gate["state"]
        for gate in registry["gates"]
        if gate["id"].startswith("io.webgpu.")
        and any(f".bin{detector_bin}." in gate["id"] for detector_bin in (2, 4, 8))
    } == {"blocked"}
    assert {
        gate["computer"]
        for gate in registry["gates"]
        if gate["platform"] == "WebGPU" and gate["module"] == "I/O and load"
    } == {
        "MacBook Air (M2, 8 GB)",
        "MacBook Pro (M5, 24 GB)",
        "MacBook Pro (M5 Max, 128 GB)",
    }


def test_registry_validator_and_generated_table_agree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_registry.py", "validate"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    registry = _registry()
    measurement_count = len(registry["additional_measurements"])
    for evidence_import in registry["evidence_imports"]:
        evidence = json.loads(Path(evidence_import["path"]).read_text(encoding="utf-8"))
        measurement_count += len(evidence["measurements"])
    assert (
        f"{len(registry['gates'])} gates, {measurement_count} measurements, and "
        f"{len(registry['runbooks'])} runbooks"
    ) in result.stdout

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
    assert "### Platform and computer coverage" in rendered
    assert "| Platform | Computer | Tracked cells |" in rendered
    assert "Master SHA-256" in rendered
    assert "Source identity SHA-256" in rendered


def test_generated_registry_is_ordered_by_platform_then_computer() -> None:
    rendered = GENERATED.read_text(encoding="utf-8")
    required = rendered.split("## Required coverage gates", 1)[1].split(
        "## Retained atomic measurements", 1
    )[0]

    platform_offsets = [
        required.index(f"| {platform} |")
        for platform in (
            "CUDA",
            "Python MPS",
            "Native Swift/Metal",
            "WebGPU",
            "CPU reference",
        )
    ]
    assert platform_offsets == sorted(platform_offsets)

    mps_rows = [line for line in required.splitlines() if "| Python MPS |" in line]
    computers = [row.split("|")[2].strip() for row in mps_rows]
    assert computers == sorted(computers)


def test_load_matrix_tracks_every_compatible_platform_computer_pair() -> None:
    registry = _registry()
    bins_by_pair: dict[tuple[str, str], set[int]] = defaultdict(set)
    for gate in registry["gates"]:
        if gate["module"] != "I/O and load":
            continue
        match = re.search(r"full-native-bin([1248])-u16", gate["configuration"])
        if match:
            bins_by_pair[(gate["platform"], gate["computer"])].add(int(match.group(1)))

    assert bins_by_pair[
        ("CUDA", "Linux CUDA workstation (dual 96 GB Blackwell GPUs)")
    ] == {1, 2, 4, 8}
    for computer in (
        "MacBook Air (M2, 8 GB)",
        "MacBook Pro (M5, 24 GB)",
        "MacBook Pro (M5 Max, 128 GB)",
    ):
        assert bins_by_pair[("Python MPS", computer)] == {1, 2, 4, 8}
        assert bins_by_pair[("WebGPU", computer)] == {1, 2, 4, 8}

    assert bins_by_pair[("Native Swift/Metal", "MacBook Pro (M5 Max, 128 GB)")] == {
        1,
        2,
        4,
        8,
    }
    for computer in ("MacBook Air (M2, 8 GB)", "MacBook Pro (M5, 24 GB)"):
        assert bins_by_pair[("Native Swift/Metal", computer)] == {1, 2, 4}


def test_performance_modules_track_each_compatible_apple_computer() -> None:
    registry = _registry()
    modules_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for gate in registry["gates"]:
        modules_by_pair[(gate["platform"], gate["computer"])].add(gate["module"])

    all_runtime_modules = {
        "I/O and load",
        "Detector products",
        "Screening",
        "CoM, DPC, and iDPC",
        "Display and FFT",
        "Single-sideband ptychography",
        "Selective loading",
    }
    for computer in (
        "MacBook Air (M2, 8 GB)",
        "MacBook Pro (M5, 24 GB)",
        "MacBook Pro (M5 Max, 128 GB)",
    ):
        assert modules_by_pair[("Python MPS", computer)] == all_runtime_modules

    for computer in ("MacBook Air (M2, 8 GB)", "MacBook Pro (M5, 24 GB)"):
        assert {
            "I/O and load",
            "Detector products",
            "CoM, DPC, and iDPC",
            "Display and FFT",
            "Single-sideband ptychography",
        } <= modules_by_pair[("Native Swift/Metal", computer)]
        assert {
            "I/O and load",
            "Detector products",
            "CoM, DPC, and iDPC",
            "Display and FFT",
            "Selective loading",
        } <= modules_by_pair[("WebGPU", computer)]


def test_failed_or_probe_parity_cannot_be_promoted_as_measured() -> None:
    timing = {
        "wall_p50_seconds": 0.1,
        "wall_p95_seconds": 0.2,
        "wall_max_seconds": 0.3,
    }
    assert _measurement_state({**timing, "parity": "full output exact"}) == "measured"
    assert (
        _measurement_state({**timing, "parity": "zero tolerance violations"})
        == "measured"
    )
    assert _measurement_state({**timing, "parity": "phase mismatch"}) == "refuted"
    assert (
        _measurement_state({**timing, "parity": "qualified probes only"}) == "partial"
    )


def test_running_platform_computer_manifest_stays_partial_and_resolves_outputs() -> (
    None
):
    registry = _registry()
    manifest_path = Path("experiments/20260822-platform-computer-profile/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "running"

    output_ids = {output["id"] for output in manifest["outputs"]}
    measurements = {
        measurement["measurement_id"]: measurement
        for measurement in registry["additional_measurements"]
        if measurement.get("evidence") == manifest_path.as_posix()
    }
    assert measurements
    assert all(
        measurement["state"] == "partial" for measurement in measurements.values()
    )
    assert {
        measurement["artifact_id"] for measurement in measurements.values()
    } <= output_ids

    gates = {
        gate["id"]: gate
        for gate in registry["gates"]
        if gate["id"].startswith("io.swift.apple-m5-24gb.bin")
        and gate.get("satisfied_by")
        and any(
            measurement_id in measurements for measurement_id in gate["satisfied_by"]
        )
    }
    assert gates
    assert all(gate["state"] == "partial" for gate in gates.values())


def test_measured_gates_only_name_measured_evidence() -> None:
    registry = _registry()
    measurement_states = {
        row["measurement_id"]: row["state"] for row in resolved_measurements(registry)
    }

    for gate in registry["gates"]:
        if gate["state"] != "measured":
            continue
        assert gate.get("satisfied_by")
        assert {
            measurement_states[measurement_id]
            for measurement_id in gate["satisfied_by"]
        } == {"measured"}


def test_zero_tolerance_violations_render_as_passing_parity() -> None:
    rendered = GENERATED.read_text(encoding="utf-8")
    idpc_row = next(
        line
        for line in rendered.splitlines()
        if "webgpu-resident-idpc-optimized" in line
    )
    assert "| Pass |" in idpc_row
    assert "| Failed |" not in idpc_row


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
        measurement["computer"] for measurement in registry["additional_measurements"]
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
    measurement_rows = []
    for line in measurements.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) > 2 and cells[2] == "✓ Measured":
            measurement_rows.append(line)
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
            "--limit",
            "50",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "io.swift.apple-m2-air-8gb.bin2.cold-original" in next_result.stdout
    assert "io.webgpu.apple-m2-air-8gb.bin2.cold-original" in next_result.stdout
    assert "io.mps.apple-m2-air-8gb.bin2.cold-original" in next_result.stdout

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

    webgpu_command = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_registry.py",
            "command",
            "io.webgpu.apple-m5-max-128gb.bin1.prepared",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--require-integer-resident" in webgpu_command
    assert "--expected-resident-dtype uint16" in webgpu_command
    assert "--require-checksum-parity" in webgpu_command
    assert "--require-full-output-parity" in webgpu_command

    mps_command = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_registry.py",
            "command",
            "io.mps.apple-m5-24gb.bin4.warm-fresh",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--require-full-output-parity" in mps_command
    assert "--expected-output-sha256" in mps_command
    assert "--expected-scan-shape" in mps_command
    assert "--expected-source-detector-shape" in mps_command
    assert "--expected-working-detector-shape" in mps_command
    assert "--memory-sample-ms 10" in mps_command
    assert "--warmup 1" in mps_command

    cold_mps_command = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_registry.py",
            "command",
            "io.mps.apple-m5-24gb.bin4.cold-original",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--warmup 0" in cold_mps_command
    assert "--warmup 1" not in cold_mps_command


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
