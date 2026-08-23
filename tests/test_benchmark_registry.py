from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_registry import _bytes, _measurement_state, resolved_measurements

REGISTRY = Path("benchmarks/benchmark_registry.json")
GENERATED = Path("docs/_generated/benchmark_coverage.md")


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_approximate_memory_is_visibly_qualified() -> None:
    assert _bytes(1 << 30) == "1.000 GiB"
    assert _bytes(1 << 30, approximate=True) == "~1.000 GiB"


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
        "Display kernels",
        "FFT",
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
        "Process physical-footprint peak",
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


def test_current_load_table_has_one_explicit_row_per_configuration() -> None:
    registry = _registry()
    current = [
        row for row in resolved_measurements(registry) if row.get("dashboard_current")
    ]
    assert current
    assert all(row["module"] == "I/O and load" for row in current)
    assert all(row["state"] not in {"refuted", "superseded"} for row in current)

    keys = {
        (
            row["platform"],
            row["computer"],
            row.get("selected_scan_rows"),
            row.get("selected_scan_columns"),
            row.get("source_detector_rows"),
            row.get("source_detector_columns"),
            row.get("detector_bin"),
            row.get("source_dtype"),
            row.get("working_dtype"),
        )
        for row in current
    }
    assert len(keys) == len(current)

    rendered = GENERATED.read_text(encoding="utf-8")
    section = rendered.split("<!-- benchmark-current-load-start -->", 1)[1].split(
        "<!-- benchmark-current-load-end -->", 1
    )[0]
    assert section.index("| Python MPS |") < section.index("| Native Swift/Metal |")
    assert "Process physical-footprint peak" in section
    assert "~0.706 GiB" in section


def test_current_webgpu_full_native_row_requires_complete_output_parity() -> None:
    registry = _registry()
    rows = {
        row["measurement_id"]: row for row in resolved_measurements(registry)
    }
    current_id = "macbook-pro-m5-max-128gb-webgpu-fullnative-exact-r7-d8e6f56"
    historical_ids = (
        "macbook-pro-m5-max-128gb-webgpu-fullnative-load-single",
        "macbook-pro-m5-max-128gb-webgpu-fullnative-exact-smoke-34f0029",
    )

    current = rows[current_id]
    assert current["dashboard_current"] is True
    assert current["state"] == "partial"
    assert current["sample_count"] == 7
    assert current["logical_resident_bytes"] == 19_327_352_832
    assert current["process_tree_peak_bytes"] == 6_979_485_696
    assert current.get("process_tree_peak_is_lower_bound") is not True
    assert current["source_revision"] == (
        "d8e6f562a8ab43086a4ddea9eecfe6fd26b7beea"
    )
    assert "all seven retained runs" in current["parity"]
    assert current["profile_min_seconds"] == 0.975
    assert current["p50_seconds"] == 1.358
    assert current["cold_claim"] is False

    for historical_id in historical_ids:
        historical = rows[historical_id]
        assert historical["state"] == "superseded"
        assert historical["dashboard_current"] is False

    gate = next(
        gate
        for gate in registry["gates"]
        if gate["id"] == "io.webgpu.apple-m5-max-128gb.bin1.prepared"
    )
    assert gate["satisfied_by"] == [current_id]
    assert "Repeated harness wall is not yet at or below one second" in gate[
        "next_gate"
    ]


def test_m5_24gb_mps_product_smoke_keeps_fft_failure_atomic() -> None:
    registry = _registry()
    rows = {
        row["measurement_id"]: row for row in resolved_measurements(registry)
    }
    products_id = (
        "macbook-pro-m5-24gb-python-mps-bin2-detector-products-smoke-68dbe3a"
    )
    dpc_id = "macbook-pro-m5-24gb-python-mps-bin2-dpc-idpc-smoke-68dbe3a"
    fft_id = "macbook-pro-m5-24gb-python-mps-bin2-display-fft-refuted-68dbe3a"

    assert rows[products_id]["state"] == "partial"
    assert "byte exact" in rows[products_id]["parity"]
    assert rows[dpc_id]["state"] == "partial"
    assert "frozen full-precision" in rows[dpc_id]["parity"]
    assert rows[fft_id]["state"] == "refuted"
    assert "0.001680374" in rows[fft_id]["parity"]
    assert all(rows[row_id]["p50_seconds"] is None for row_id in (products_id, dpc_id, fft_id))
    assert all(
        rows[row_id]["pageout_delta_pages"] == 73
        for row_id in (products_id, dpc_id, fft_id)
    )

    gates = {gate["id"]: gate for gate in registry["gates"]}
    assert gates["products.mps.apple-m5-24gb.bin2.full-suite"]["state"] == "partial"
    assert gates["dpc.mps.apple-m5-24gb.full-suite"]["state"] == "partial"
    assert gates["display.mps.apple-m5-24gb.maps"]["state"] == "pending"
    assert gates["display.mps.apple-m5-24gb.maps"].get("satisfied_by") is None
    assert gates["fft.mps.apple-m5-24gb.maps"]["state"] == "refuted"
    assert gates["fft.mps.apple-m5-24gb.maps"]["satisfied_by"] == [fft_id]
    assert rows[fft_id]["module"] == "FFT"
    assert all(
        fft_id not in gate.get("satisfied_by", [])
        for gate_id, gate in gates.items()
        if gate_id != "fft.mps.apple-m5-24gb.maps"
    )


def test_load_matrix_tracks_every_compatible_platform_computer_pair() -> None:
    registry = _registry()
    bins_by_pair: dict[tuple[str, str], set[int]] = defaultdict(set)
    for gate in registry["gates"]:
        if gate["module"] != "I/O and load":
            continue
        configuration = registry["configurations"][gate["configuration"]]
        detector_bin = configuration["detector_bin"]
        if detector_bin in {1, 2, 4, 8}:
            bins_by_pair[(gate["platform"], gate["computer"])].add(detector_bin)

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
        assert bins_by_pair[("CPU reference", computer)] == {1, 2, 4, 8}

    assert bins_by_pair[
        ("CPU reference", "Linux CUDA workstation (dual 96 GB Blackwell GPUs)")
    ] == {1, 2, 4, 8}

    assert bins_by_pair[("Native Swift/Metal", "MacBook Pro (M5 Max, 128 GB)")] == {
        1,
        2,
        4,
        8,
    }
    for computer in ("MacBook Air (M2, 8 GB)", "MacBook Pro (M5, 24 GB)"):
        assert bins_by_pair[("Native Swift/Metal", computer)] == {1, 2, 4}


def test_cuda_binned_load_gates_use_the_general_u32_contract() -> None:
    registry = _registry()
    gates = [
        gate
        for gate in registry["gates"]
        if gate["platform"] == "CUDA"
        and gate["module"] == "I/O and load"
        and registry["configurations"][gate["configuration"]]["detector_bin"] > 1
    ]

    assert len(gates) == 6
    for gate in gates:
        configuration = registry["configurations"][gate["configuration"]]
        detector_bin = configuration["detector_bin"]
        assert gate["configuration"] == f"full-native-bin{detector_bin}-u16-to-u32"
        assert configuration["source_dtype"] == "uint16"
        assert configuration["working_dtype"] == "uint32"

    command = registry["runbooks"]["python-load-matrix"]["command"]
    assert '--dtype "$QGPU_EXPECTED_OUTPUT_DTYPE"' in command
    assert '--expected-output-dtype "$QGPU_EXPECTED_OUTPUT_DTYPE"' in command


def test_cpu_binning_and_selective_loading_do_not_overclaim_general_support() -> None:
    registry = _registry()
    selectors = [
        gate
        for gate in registry["gates"]
        if gate["platform"] == "CPU reference"
        and gate["module"] == "Selective loading"
    ]
    assert len(selectors) == 2
    assert {gate["state"] for gate in selectors} == {"unsupported"}
    assert all("public quantem.gpu.io.load" in gate["reason"] for gate in selectors)
    assert all("CUDA and Python MPS" in gate["reason"] for gate in selectors)

    binned = [
        gate
        for gate in registry["gates"]
        if gate["platform"] == "CPU reference"
        and gate["module"] == "I/O and load"
        and registry["configurations"][gate["configuration"]]["detector_bin"] > 1
    ]
    assert len(binned) == 12
    for gate in binned:
        assert "incomplete detector edges" in gate["reason"]
        assert "overflow" in gate["reason"]
        if gate["computer"] == "MacBook Pro (M5 Max, 128 GB)":
            assert gate["state"] == "partial"
            assert gate["satisfied_by"]
            assert "Fixture C is exact" in gate["reason"]
        else:
            assert gate["state"] == "blocked"
            assert not gate.get("satisfied_by")


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
        "Display kernels",
        "FFT",
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
            "Display kernels",
            "FFT",
            "Single-sideband ptychography",
        } <= modules_by_pair[("Native Swift/Metal", computer)]
        assert {
            "I/O and load",
            "Detector products",
            "CoM, DPC, and iDPC",
            "Display kernels",
            "FFT",
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


def test_platform_computer_manifest_lifecycle_resolves_outputs() -> None:
    registry = _registry()
    manifest_path = Path("experiments/20260822-platform-computer-profile/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] in {"running", "completed"}

    output_ids = {output["id"] for output in manifest["outputs"]}
    measurements = {
        measurement["measurement_id"]: measurement
        for measurement in registry["additional_measurements"]
        if measurement.get("evidence") == manifest_path.as_posix()
    }
    assert measurements
    assert {
        measurement["artifact_id"] for measurement in measurements.values()
    } <= output_ids

    if manifest["status"] == "running":
        assert manifest["timestamps"]["finished"] is None
        assert all(
            measurement["state"] == "partial"
            for measurement in measurements.values()
        )
    else:
        assert manifest["timestamps"]["finished"]
        assert any(
            measurement["state"] == "measured"
            for measurement in measurements.values()
        )

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
    if manifest["status"] == "running":
        assert all(gate["state"] == "partial" for gate in gates.values())
    else:
        assert all(gate["state"] in {"measured", "partial"} for gate in gates.values())


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
