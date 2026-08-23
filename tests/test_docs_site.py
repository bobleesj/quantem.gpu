from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

TOC = Path("docs/_toc.yml")
CONFIG = Path("docs/_config.yml")
EVIDENCE = Path("docs/performance/evidence_manifest.json")


def test_docs_navigation_has_world_class_top_level_sections() -> None:
    toc = TOC.read_text(encoding="utf-8")

    for caption in (
        "Start here",
        "Scientific kernels",
        "Kernel implementations",
        "Remote compute",
        "API contracts",
        "Benchmarks and parity",
        "Contributing",
    ):
        assert f"caption: {caption}" in toc

    for page in (
        "dashboard",
        "concepts/scientific-contract",
        "concepts/kernel-architecture",
        "kernels/index",
        "kernels/data-model",
        "kernels/load-decode-bin",
        "kernels/virtual-detectors",
        "kernels/com-dpc-idpc",
        "kernels/ssb",
        "kernels/scan-regions",
        "kernels/display-export",
        "platforms/index",
        "platforms/cuda",
        "platforms/mps",
        "platforms/swift-metal",
        "platforms/webgpu",
        "platforms/cpu-reference",
        "remote/index",
        "remote/deployment",
        "remote/connect",
        "remote/protocol",
        "remote/admission",
        "maintainer/history/index",
        "maintainer/history/webgpu-gqk-memory-2026-07",
        "maintainer/history/webgpu-frame-coop-u16-clip8-2026-07-25",
        "maintainer/native-metal-ssb-migration",
        "developer/writing",
        "performance/index",
        "performance/coverage",
        "performance/results",
        "performance/changes",
        "performance/methodology",
        "performance/parity",
        "backends",
        "developer/index",
        "developer/kernel-lifecycle",
        "developer/adding-backend",
        "developer/testing",
        "maintainer/index",
        "api/core",
    ):
        assert f"file: {page}" in toc


def test_every_toc_page_exists() -> None:
    missing: list[Path] = []
    for line in TOC.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- file: "):
            continue
        relative = stripped.removeprefix("- file: ").strip()
        candidates = [Path("docs", relative + suffix) for suffix in (".md", ".ipynb")]
        if not any(path.is_file() for path in candidates):
            missing.append(candidates[0])

    assert not missing


def test_numerical_evidence_pages_keep_frozen_fingerprints() -> None:
    manifest = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    source_revision = manifest["source_revision"]
    assert len(source_revision) == 40
    assert all(character in "0123456789abcdef" for character in source_revision)

    changed: list[tuple[str, str, str]] = []
    for entry in manifest["pages"] + manifest["authoritative_inputs"]:
        path = Path(entry["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            changed.append((entry["path"], entry["sha256"], actual))

    assert not changed


def test_performance_landing_keeps_every_evidence_page_navigable() -> None:
    text = Path("docs/performance/index.md").read_text(encoding="utf-8")
    manifest = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    for entry in manifest["pages"]:
        name = Path(entry["path"]).name
        assert name in text or entry["role"] in text


def test_dashboard_is_the_dense_human_overview() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    dashboard_words = " ".join(dashboard.split())
    generated = Path("docs/_generated/benchmark_coverage.md").read_text(
        encoding="utf-8"
    )
    results = Path("docs/performance/results.md").read_text(encoding="utf-8")
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    toc = TOC.read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "caption: Start here" in toc
    assert "file: dashboard" in toc
    assert "implementation overview" in intro
    assert "[Implementation overview](docs/dashboard.md)" in readme

    for heading in (
        "## Platform-first module dashboard",
        "## Speed and memory at a glance",
        "### Current qualified load measurements",
        "### Dtype support and peak memory",
        "### Minimum-device memory gates",
        "### What a 4 or 6 GiB budget can hold",
        "### I/O and first usable product — `quantem.gpu.io`",
        "### Screening and prepared-product caches — `quantem.gpu.screening`",
        "### Virtual images — `quantem.gpu.detector`",
        "### Detector moments and phase contrast — `quantem.gpu.dpc`",
        "### Single-sideband ptychography — `quantem.gpu.SSB`",
        "### Cross-module platform map",
        "## Where an implementer starts",
        "## Dashboard maintenance rule",
        "## Coverage and next runs",
    ):
        assert heading in dashboard

    for runtime in (
        "CUDA",
        "Python MPS",
        "Native Swift/Metal",
        "WebGPU",
        "CPU reference",
    ):
        assert runtime in dashboard

    for evidence_state in ("✓", "Test", "Pending", "Ref", "unsupported or not a target"):
        assert evidence_state in dashboard

    dashboard_surface = dashboard + "\n" + generated
    for device in (
        "NVIDIA RTX PRO 6000 Blackwell",
        "Apple M5 Max 40-core integrated GPU; 128 GB unified memory",
        "Apple M2 MacBook Air (`Mac14,2`, 8 GB)",
    ):
        assert device in dashboard_surface

    for provenance in (
        "measurement date",
        "exact source revision",
        "physical device/runtime",
        "source shape and dtype",
        "cache state",
        "crop/bin/load plan",
        "benchmark definition",
        "peak memory or swap",
        "parity",
    ):
        assert provenance in dashboard_words

    assert "I[R_r,R_c,k_r,k_c]" in dashboard
    assert "not ranked" in dashboard_words
    assert "18.00 GiB" in dashboard
    assert "9.00 GiB" in dashboard
    assert "1.125 GiB" in dashboard
    assert "2.25 GiB" in dashboard
    assert "| 4 GiB | `uint16` | 56 | **1.97 GiB** | 10 | Pending |" in dashboard
    assert "| 6 GiB | `uint16` | 85 | **2.99 GiB** | 7 | Pending |" in dashboard
    assert "Each `512x512 float32` product map is only **1 MiB**" in dashboard
    assert "A calculated payload is never relabeled as a measured peak" in dashboard
    assert "6 GiB of dedicated VRAM for CUDA" in dashboard_words
    assert "8 GB of total laptop RAM for WebGPU" in dashboard_words
    assert "previously made a refuted WebGPU SSB result look accepted" in dashboard
    assert "Reconstruction ✓" not in dashboard
    assert "Refuted diagnostic" in dashboard

    for scan_size in (128, 256, 512, 1024):
        assert f"`{scan_size}x{scan_size}`" in dashboard
    for detector_bin in (1, 2, 4, 8):
        assert f"| {detector_bin} |" in dashboard
    assert "square scan-grid sizes, not detector dimensions" in dashboard

    for dashboard_value, results_value in (
        ("6.711 s", "6.711 s"),
        ("20.803 ms", "20.803 ms"),
        ("11.168 s", "11.168/11.235/11.242 s"),
        ("0.029 s", "0.029 s"),
        ("103.0 ms", "103.0 ms"),
        ("11.389 ms", "11.389 ms"),
    ):
        assert dashboard.count(dashboard_value) == 1
        assert results.count(results_value) == 1

    for historical_load in ("0.386 s", "0.359606 s", "0.824 s"):
        assert historical_load not in dashboard
        assert historical_load in results

    current_load = generated.split("<!-- benchmark-current-load-start -->", 1)[1].split(
        "<!-- benchmark-current-load-end -->", 1
    )[0]
    assert current_load.count("0.577793 s") == 1
    assert dashboard.count("119.040 ms") == 1
    controlled = results.split(
        "### Current controlled native exact resident load", 1
    )[1].split("### Current native exact resident summary", 1)[0]
    assert "0.577793 s" in controlled
    assert "119.040 s" not in controlled
    assert "0.119040 s" in controlled

    for removed_value in (
        "0.338 s",
        "1.695 s",
        "2.027 s",
        "3.451 s",
        "8.096 s",
        "537.58 ms",
        "669.1 ms",
    ):
        assert removed_value not in dashboard
        assert removed_value not in intro
        assert removed_value not in results

    module_headings = (
        "### I/O and first usable product — `quantem.gpu.io`",
        "### Screening and prepared-product caches — `quantem.gpu.screening`",
        "### Virtual images — `quantem.gpu.detector`",
        "### Detector moments and phase contrast — `quantem.gpu.dpc`",
        "### Single-sideband ptychography — `quantem.gpu.SSB`",
        "### Cross-module platform map",
    )
    io_section = dashboard.split(module_headings[0], 1)[1].split(
        module_headings[1], 1
    )[0]
    assert "filterable coverage registry" in io_section
    for index, heading in enumerate(module_headings[1:-1], start=1):
        section = dashboard.split(heading, 1)[1].split(module_headings[index + 1], 1)[0]
        for runtime in ("CUDA", "Python MPS", "Native Swift/Metal", "WebGPU", "CPU reference"):
            assert runtime in section

def test_intro_routes_to_benchmarks_without_copying_them() -> None:
    intro = Path("docs/intro.md").read_text(encoding="utf-8")

    assert "## Implementation and benchmark overview" in intro
    assert "[implementation dashboard](dashboard.md)" in intro
    assert "[verified benchmark results](performance/results.md)" in intro
    assert "[Benchmark methodology](performance/methodology.md)" in intro
    assert "[Optimization ledger](maintainer/backend-optimization-matrix.md)" in intro
    assert "One claim, one owner" in intro

    for copied_field in (
        "Device tested",
        "Date tested",
        "Measured load configurations",
        "Current resident products",
        "Current SSB reconstruction",
    ):
        assert copied_field not in intro
    for copied_time in (
        "0.029 s",
        "0.386 s",
        "0.903 s",
        "0.824 s",
        "0.578 s",
        "119.040 ms",
        "11.168 s",
        "11.389 ms",
        "103.0 ms",
        "8.911 ms",
    ):
        assert copied_time not in intro

    assert "## How loading becomes a usable product" in intro
    assert "START WALL CLOCK" in intro
    assert "FIRST COMPLETE USABLE PRODUCT" in intro
    assert "no automatic real-space crop" in intro
    assert "## The shared coordinate contract" in intro

    load_page = Path("docs/kernels/load-decode-bin.md").read_text(encoding="utf-8")
    assert "## Count-preserving detector binning" in load_page
    assert "exact sum of one" in load_page
    assert "materializing both a full unbinned volume" in load_page

def test_dashboard_ssb_size_matrix_tracks_fixed_size_runtime_registries() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    ssb_section = dashboard.split(
        "### Single-sideband ptychography — `quantem.gpu.SSB`", 1
    )[1].split("### Cross-module platform map", 1)[0]
    cuda = Path("src/quantem/gpu/ssb/compute/cuda/kernels/__init__.py").read_text(
        encoding="utf-8"
    )
    mps = Path("src/quantem/gpu/ssb/compute/mps/kernels/__init__.py").read_text(
        encoding="utf-8"
    )
    webgpu = Path(
        "src/quantem/gpu/ssb/compute/webgpu/kernels/index.ts"
    ).read_text(encoding="utf-8")

    assert "square scan-grid sizes, not detector dimensions" in ssb_section
    header = "| Platform | Computer | Scan grid | Source kind | BF policy | State |"
    assert header in ssb_section
    assert "| Statistic | Time | Device tested | Date tested |" not in ssb_section.split(
        header, 1
    )[1]
    for size in (128, 256, 512, 1024):
        assert f"{size}:" in cuda
        assert f"{size}:" in mps
        for platform in ("CUDA", "Python MPS", "Native Swift/Metal", "WebGPU", "CPU reference"):
            rows = [
                line
                for line in ssb_section.splitlines()
                if line.startswith(f"| **{platform}** |")
                and f"| `{size}x{size}` |" in line
            ]
            assert len(rows) == 1
    assert "SUPPORTED_SSB_SIZES = [128, 256, 512, 1024]" in webgpu
    assert "Incomplete frozen reference" in ssb_section
    assert "Native real acquisition | Full active BF" in ssb_section
    assert "537.58 ms" not in ssb_section
    assert "669.1 ms" not in ssb_section

def test_platform_first_io_tables_expose_current_bins_devices_and_dates() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    generated = Path("docs/_generated/benchmark_coverage.md").read_text(
        encoding="utf-8"
    )
    swift_plan = Path(
        "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Metal4DSTEMLoadPlan.swift"
    ).read_text(encoding="utf-8")

    assert "### Current qualified load measurements" in dashboard
    assert ":start-after: <!-- benchmark-current-load-start -->" in dashboard
    assert ":end-before: <!-- benchmark-current-load-end -->" in dashboard
    assert "supportedDetectorBins = [1, 2, 4]" in swift_plan

    current = generated.split("<!-- benchmark-current-load-start -->", 1)[1].split(
        "<!-- benchmark-current-load-end -->", 1
    )[0]
    lines = [line for line in current.splitlines() if line.startswith("|")]
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines[2:]
    ]
    registry = json.loads(
        Path("benchmarks/benchmark_registry.json").read_text(encoding="utf-8")
    )
    expected_current_rows = sum(
        bool(measurement.get("dashboard_current"))
        for measurement in registry["additional_measurements"]
    )
    assert len(rows) == expected_current_rows
    assert headers[:3] == ["Platform", "Computer", "State"]
    assert all(row[0] and row[1] for row in rows)
    assert all(name not in current.lower() for name in ("phil", "mjgoat", "rodman"))

    platform_order = {
        platform: current.index(f"| {platform} |")
        for platform in ("Python MPS", "Native Swift/Metal", "CPU reference")
    }
    assert list(platform_order.values()) == sorted(platform_order.values())

    bin_index = headers.index("Detector bin")
    p50_index = headers.index("p50")
    p95_index = headers.index("p95")
    maximum_index = headers.index("Maximum")
    resident_index = headers.index("Logical resident")
    accelerator_index = headers.index("Accelerator/driver peak")
    rss_index = headers.index("Process/tree peak")
    footprint_index = headers.index("Process physical-footprint peak")
    swap_index = headers.index("Swap delta")
    revision_index = headers.index("Revision")

    mps_max_rows = [
        row
        for row in rows
        if row[0] == "Python MPS"
        and row[1] == "MacBook Pro (M5 Max, 128 GB)"
    ]
    assert {int(row[bin_index]) for row in mps_max_rows} == {1, 2, 4, 8}
    expected_mps = {
        1: ("0.406624 s", "0.428164 s", "18.000 GiB", "18.442 GiB", "18.692 GiB"),
        2: ("0.477740 s", "1.064425 s", "4.500 GiB", "5.688 GiB", "5.091 GiB"),
        4: ("0.370645 s", "0.939238 s", "1.125 GiB", "2.313 GiB", "1.726 GiB"),
        8: ("0.340210 s", "0.341541 s", "0.281 GiB", "1.470 GiB", "0.855 GiB"),
    }
    for row in mps_max_rows:
        detector_bin = int(row[bin_index])
        expected = expected_mps[detector_bin]
        assert (
            row[p50_index],
            row[p95_index],
            row[resident_index],
            row[accelerator_index],
            row[rss_index],
        ) == expected
        assert row[maximum_index] == row[p95_index]
        assert row[swap_index] == "0 B"

    mps_24 = [
        row
        for row in rows
        if row[0] == "Python MPS" and row[1] == "MacBook Pro (M5, 24 GB)"
    ]
    assert {int(row[bin_index]) for row in mps_24} == {2, 4, 8}
    mps_24_bin2 = next(row for row in mps_24 if int(row[bin_index]) == 2)
    assert (
        mps_24_bin2[p50_index],
        mps_24_bin2[p95_index],
        mps_24_bin2[footprint_index],
    ) == ("1.337644 s", "1.338026 s", "6.044 GiB")
    mps_24_bin4 = next(row for row in mps_24 if int(row[bin_index]) == 4)
    assert (
        mps_24_bin4[p50_index],
        mps_24_bin4[p95_index],
        mps_24_bin4[footprint_index],
    ) == ("1.250958 s", "1.255510 s", "3.791 GiB")
    mps_24_bin8 = next(row for row in mps_24 if int(row[bin_index]) == 8)
    assert (
        mps_24_bin8[p50_index],
        mps_24_bin8[p95_index],
        mps_24_bin8[resident_index],
        mps_24_bin8[accelerator_index],
        mps_24_bin8[rss_index],
        mps_24_bin8[footprint_index],
    ) == (
        "1.220057 s",
        "1.228715 s",
        "0.281 GiB",
        "1.470 GiB",
        "0.937 GiB",
        "2.104 GiB",
    )
    assert all(row[swap_index] == "0 B" for row in mps_24)

    native_max_bin2 = next(
        row
        for row in rows
        if row[0] == "Native Swift/Metal"
        and row[1] == "MacBook Pro (M5 Max, 128 GB)"
        and int(row[bin_index]) == 2
    )
    assert (
        native_max_bin2[p50_index],
        native_max_bin2[p95_index],
        native_max_bin2[footprint_index],
    ) == ("0.225984 s", "0.231192 s", "5.336 GiB")
    assert native_max_bin2[swap_index] == "0 B"

    native_24 = [
        row
        for row in rows
        if row[0] == "Native Swift/Metal"
        and row[1] == "MacBook Pro (M5, 24 GB)"
    ]
    assert {int(row[bin_index]) for row in native_24} == {1, 2, 4}
    bin1 = next(row for row in native_24 if int(row[bin_index]) == 1)
    assert bin1[p50_index] == "3.424108 s"
    assert bin1[footprint_index] == "18.673 GiB"
    assert bin1[swap_index] == "~0.706 GiB"
    assert bin1[2] == "◐ Partial"

    bin2 = next(row for row in native_24 if int(row[bin_index]) == 2)
    assert (bin2[p50_index], bin2[p95_index], bin2[footprint_index]) == (
        "0.678703 s",
        "0.697179 s",
        "5.271 GiB",
    )
    bin4 = next(row for row in native_24 if int(row[bin_index]) == 4)
    assert (bin4[p50_index], bin4[p95_index], bin4[footprint_index]) == (
        "0.640942 s",
        "0.651931 s",
        "2.319 GiB",
    )

    webgpu_full_native = next(
        row
        for row in rows
        if row[0] == "WebGPU"
        and row[1] == "MacBook Pro (M5 Max, 128 GB)"
        and int(row[bin_index]) == 1
    )
    assert webgpu_full_native[2] == "◐ Partial"
    assert (
        webgpu_full_native[p50_index],
        webgpu_full_native[p95_index],
        webgpu_full_native[maximum_index],
        webgpu_full_native[resident_index],
        webgpu_full_native[rss_index],
        webgpu_full_native[swap_index],
    ) == (
        "1.094000 s",
        "1.094000 s",
        "1.094000 s",
        "18.000 GiB",
        "6.604 GiB",
        "0 B",
    )
    assert webgpu_full_native[headers.index("Samples")] == "1"
    assert "warm from the immediately preceding independent CPU reference" in (
        webgpu_full_native[headers.index("Cache/process state")]
    )
    assert "full-volume verification hash" in webgpu_full_native[
        headers.index("Wall boundary")
    ]
    assert all(row[revision_index].startswith("`") for row in rows)

    selective = dashboard.split("#### Selective scan rectangles", 1)[1].split(
        "### Screening and prepared-product caches", 1
    )[0]
    selective_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in selective.splitlines()
        if line.startswith("| **WebGPU**")
    ]
    assert {
        (row[2], row[6], row[7], row[9], row[12], row[13], row[14])
        for row in selective_rows
    } == {
        ("`64x64`", "4 of 27", "488,224,242 B", "**0.147 s**", "301,989,888 B", "1,724,317,696 B", "0 B"),
        ("`256x256`", "14 of 27", "1,705,556,941 B", "**0.381 s**", "4,831,838,208 B", "3,002,875,904 B", "0 B"),
        ("`384x384`", "20 of 27", "2,432,636,897 B", "**0.574 s**", "10,871,635,968 B", "3,896,934,400 B", "0 B"),
    }
    assert "whole-shard-selective" in selective
    assert "does not issue byte-range reads" in " ".join(selective.split())
    assert "read all 27 shards (3.17 GB)" in selective

    assert "Measured load configurations" not in intro
    assert "Device tested" not in intro
def test_load_memory_rows_separate_payload_from_measured_peak() -> None:
    generated = Path("docs/_generated/benchmark_coverage.md").read_text(
        encoding="utf-8"
    )
    current = generated.split("<!-- benchmark-current-load-start -->", 1)[1].split(
        "<!-- benchmark-current-load-end -->", 1
    )[0]
    lines = [line for line in current.splitlines() if line.startswith("|")]
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines[2:]
    ]

    for field in (
        "Logical resident",
        "Accelerator/driver peak",
        "Process/tree peak",
        "Process physical-footprint peak",
        "Swap delta",
    ):
        assert field in headers

    resident_index = headers.index("Logical resident")
    accelerator_index = headers.index("Accelerator/driver peak")
    rss_index = headers.index("Process/tree peak")
    footprint_index = headers.index("Process physical-footprint peak")
    swap_index = headers.index("Swap delta")
    bin_index = headers.index("Detector bin")

    full_native_mps = next(
        row
        for row in rows
        if row[0] == "Python MPS" and int(row[bin_index]) == 1
    )
    assert (
        full_native_mps[resident_index],
        full_native_mps[accelerator_index],
        full_native_mps[rss_index],
        full_native_mps[footprint_index],
        full_native_mps[swap_index],
    ) == ("18.000 GiB", "18.442 GiB", "18.692 GiB", "n/a", "0 B")

    full_native_24gb = next(
        row
        for row in rows
        if row[0] == "Native Swift/Metal"
        and row[1] == "MacBook Pro (M5, 24 GB)"
        and int(row[bin_index]) == 1
    )
    assert (
        full_native_24gb[resident_index],
        full_native_24gb[accelerator_index],
        full_native_24gb[rss_index],
        full_native_24gb[footprint_index],
        full_native_24gb[swap_index],
    ) == ("18.000 GiB", "18.587 GiB", "0.611 GiB", "18.673 GiB", "~0.706 GiB")

    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    dashboard_words = " ".join(dashboard.split())
    assert "are distinct observations and are not additive" in dashboard_words
    assert "758,448,128 B of swapouts" in dashboard
    assert "partial evidence, not a repeatable" in dashboard
def test_dashboard_small_gpu_numbers_match_the_screening_planner() -> None:
    from quantem.gpu.screening.workflow import _memory_plan_for_shapes

    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    four_gib = _memory_plan_for_shapes((512, 512), (192, 192), 2, 4.0)
    six_gib = _memory_plan_for_shapes((512, 512), (192, 192), 2, 6.0)

    assert four_gib.chunk_rows == 56
    assert four_gib.chunk_count == 10
    assert four_gib.chunk_resident_gb / (1 << 30) * 1e9 == pytest.approx(1.96875)
    assert six_gib.chunk_rows == 85
    assert six_gib.chunk_count == 7
    assert six_gib.chunk_resident_gb / (1 << 30) * 1e9 == pytest.approx(2.98828125)

    assert "| 4 GiB | `uint16` | 56 | **1.97 GiB** | 10 | Pending |" in dashboard
    assert "| 6 GiB | `uint16` | 85 | **2.99 GiB** | 7 | Pending |" in dashboard
    assert "`memory_budget_gb=4.0`" in Path("docs/api/core.md").read_text(
        encoding="utf-8"
    )


def test_minimum_device_memory_gates_are_atomic_and_fail_closed() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    methodology = Path("docs/performance/methodology.md").read_text(encoding="utf-8")
    cuda = Path("docs/platforms/cuda.md").read_text(encoding="utf-8")
    webgpu = Path("docs/platforms/webgpu.md").read_text(encoding="utf-8")

    assert "### Minimum-device memory gates" in dashboard
    assert "[implementation dashboard](dashboard.md)" in intro
    assert "### Minimum-device memory gates" not in intro
    assert "## Minimum-device memory gates" in methodology
    assert "### 6 GiB VRAM release floor" in cuda
    assert "### 8 GB laptop release floor" in webgpu

    for detector_bin, payload, gate in (
        (1, "18.00 GiB", "No"),
        (2, "9.00 GiB", "No"),
        (4, "2.25 GiB", "Pending"),
        (8, "0.5625 GiB", "Pending"),
    ):
        matching = next(
            line
            for line in dashboard.splitlines()
            if line.startswith("| [**CUDA**](platforms/cuda.md) |")
            and "| 6 GiB VRAM | `512x512` | Full | `192x192` |" in line
            and f"| {detector_bin} |" in line
        )
        assert f"**{payload}**" in matching
        assert f"**{gate}**" in matching

    for detector_bin, payload, gate in (
        (1, "18.00 GiB", "Blocked"),
        (2, "4.50 GiB", "Blocked"),
        (4, "1.125 GiB", "Blocked"),
        (8, "0.28125 GiB", "Blocked"),
    ):
        row = (
            f"| [**WebGPU**](platforms/webgpu.md) | MacBook Air (M2, 8 GB) | "
            f"8 GB total RAM | `512x512` "
            f"| Full | `192x192` | {detector_bin} |"
        )
        matching = next(line for line in dashboard.splitlines() if line.startswith(row))
        assert f"**{payload}**" in matching
        assert f"**{gate}**" in matching

    native_row = (
        "| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Air "
        "(M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 4 |"
    )
    matching = next(line for line in dashboard.splitlines() if line.startswith(native_row))
    assert "**1.125 GiB**" in matching
    assert "**Historical**" in matching
    assert "Apple M2 MacBook Air (`Mac14,2`, 8 GB)" in matching

    assert "A calculated payload can prove **No**" in " ".join(dashboard.split())
    assert "Do not convert **Pending** or **Test** to ✓" in methodology
    assert "Measurements on a larger Blackwell GPU do not by themselves prove" in cuda
    assert "A real-adapter run on a higher-memory machine cannot receive this ✓" in webgpu

def test_load_dtype_docs_keep_precision_and_peak_memory_distinct() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    generated = Path("docs/_generated/benchmark_coverage.md").read_text(
        encoding="utf-8"
    )
    io_api = Path("docs/api/io.md").read_text(encoding="utf-8")
    load_kernel = Path("docs/kernels/load-decode-bin.md").read_text(encoding="utf-8")
    methodology = Path("docs/performance/methodology.md").read_text(encoding="utf-8")
    mps = Path("docs/platforms/mps.md").read_text(encoding="utf-8")
    native_api = Path("docs/api/native_4dstem_io.md").read_text(encoding="utf-8")
    public_load = Path("src/quantem/gpu/io/load.py").read_text(encoding="utf-8")

    for text in (dashboard, io_api, load_kernel, methodology):
        assert "source" in text.lower()
        assert "working" in text.lower()
        assert "accumulation" in text.lower()
        assert "resident" in text.lower()
        assert "peak" in text.lower()

    assert '`dtype="u8"` requests saturating unsigned 8-bit browse counts' in dashboard
    assert (
        "| Logical resident | Accelerator/driver peak | Process/tree peak | "
        "Process physical-footprint peak |" in generated
    )
    assert "19,327,352,832 bytes" in dashboard
    assert "18.442 GiB" in dashboard
    assert "Process RSS does not include every direct Metal" in dashboard
    assert "maximum-count audit of 53" in dashboard
    assert "8x8 exact" in dashboard
    assert "sum is at most 3,392" in dashboard
    assert "2.273 s" not in dashboard
    assert "values above 255" in io_api
    assert "Do not call a payload size “peak memory.”" in methodology
    assert "Metal-driver allocation after output release" in methodology
    assert "loaded.data.free()" in mps
    assert "19,327,352,832 bytes (18.00 GiB)" in mps
    assert "torch.mps.driver_allocated_memory()" in mps
    assert "maximum possible sums 212, 848, and 3,392" in mps
    assert "unsigned 8-bit and unsigned 16-bit detector" in native_api
    assert '``"u8"`` requests a saturating browse output' in public_load


def test_narrow_tables_scroll_without_compressing_provenance_columns() -> None:
    css = Path("docs/_static/custom.css").read_text(encoding="utf-8")

    assert ".pst-scrollable-table-container" in css
    assert "overflow-x: auto" in css
    assert ".pst-scrollable-table-container > table.table" in css
    assert "min-width: 52rem" in css
    assert "table.table:has(th:nth-child(15))" in css
    assert "min-width: 128rem" in css


def test_platform_tables_have_dependency_free_local_filters() -> None:
    config = Path("docs/_config.yml").read_text(encoding="utf-8")
    script = Path("docs/_static/benchmark-tables.js").read_text(encoding="utf-8")
    css = Path("docs/_static/custom.css").read_text(encoding="utf-8")

    assert "benchmark-tables.js" in config
    assert 'headers.indexOf("Platform")' in script
    assert 'headers.indexOf(columnName)' in script
    assert script.index('"Platform",') < script.index('"Computer",')
    assert script.index('"Computer",') < script.index('"State",')
    assert '"Cache/process state"' in script
    assert 'row.hidden = !(textMatches && selectionsMatch)' in script
    assert "fetch(" not in script
    assert ".qgpu-table-tools" in css
    assert "table tr[hidden]" in css


def test_dashboard_tables_use_device_and_date_while_landing_stays_timing_free() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    writing = Path("docs/developer/writing.md").read_text(encoding="utf-8")

    assert "Device tested" in dashboard
    assert "Date tested" in dashboard
    assert "| Evidence |" not in dashboard
    assert "Evidence / next gap" not in dashboard
    assert "Device tested" not in intro
    assert "Date tested" not in intro

    for block in dashboard.split("\n\n"):
        lines = [line for line in block.splitlines() if line.startswith("|")]
        if len(lines) < 3:
            continue
        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
        if not {"Time", "Device tested", "Date tested"} <= set(headers):
            continue
        time_index = headers.index("Time")
        device_index = headers.index("Device tested")
        date_index = headers.index("Date tested")
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            assert len(cells) == len(headers)
            if cells[time_index] in {"—", "**Pending**"}:
                continue
            assert cells[device_index] != "—"
            assert re.fullmatch(r"20\d{2}-\d{2}-\d{2}", cells[date_index])

    assert "human-facing current-measurement table ends with" in writing
    assert "**Device tested**, **Date tested**, and **Revision**" in writing
    assert "Do not add opaque evidence" in writing

def test_api_guide_maps_every_public_namespace() -> None:
    api = Path("docs/api/index.md").read_text(encoding="utf-8")
    root = Path("src/quantem/gpu/__init__.py").read_text(encoding="utf-8")

    for namespace in (
        "detector",
        "device",
        "dpc",
        "io",
        "movie",
        "optics",
        "parallax",
        "screening",
    ):
        assert f'"{namespace}"' in root
        assert f"`{namespace}`" in api

    core = Path("docs/api/core.md").read_text(encoding="utf-8")
    for entry_point in (
        "device.detect",
        "device.profile",
        "device.resolve",
        "screening.prepare",
        "parallax.run",
        "wavelength_A_from_kV",
        "chi_polar",
        "fit_aberrations",
    ):
        assert entry_point in core


def test_current_benchmarks_have_complete_provenance_rows() -> None:
    text = Path("docs/performance/results.md").read_text(encoding="utf-8")

    for heading in (
        "### Current qualified exact loads",
        "### Historical 2026-08-19 warm load/decode/bin",
        "### Historical consolidated Python MPS resident lifecycle",
        "### Current controlled native exact resident load",
        "### Current native exact resident summary",
        "### Current streamed screening",
        "### Current resident products",
        "### Current SSB reconstruction and calibration",
        "### Current native Swift/Metal boundary",
        "### Current evidence fingerprints",
        "## Historical and rejected results",
    ):
        assert heading in text

    load = text.split("### Historical 2026-08-19 warm load/decode/bin", 1)[1].split(
        "### Current prepared WebGPU shard-selective rectangles", 1
    )[0]
    rows = [
        line
        for line in load.splitlines()
        if line.startswith(
            (
                "| **CUDA**",
                "| **Python MPS**",
                "| **WebGPU**",
                "| **CPU reference**",
            )
        )
    ]
    assert len(rows) == 16
    for row in rows:
        assert row.count("|") == 16
        assert "2026" not in row  # date is profile-level and not duplicated per row
        assert (
            "| Pass |" in row
            or "Exact" in row
            or "adjudicator" in row
            or "Native bin1 reference" in row
        )

    for required in (
        "Date tested:",
        "Baseline revisions:",
        "Fixture C:",
        "Fixture D:",
        "not called cold",
        "controlled uncached source pages",
        "complete `512x512` scan",
        "no scan or detector crop",
        "Current evidence fingerprints",
        "failed scientific gates",
    ):
        assert required.lower() in text.lower()

    for removed in (
        "## Earlier three-host full-scan campaign",
        "## Historical native CUDA and MPS IO",
        "CUDA-512-LOAD",
        "PRODUCT-CACHE-REOPEN",
        "8.096 s",
        "3.451 s",
    ):
        assert removed not in text

def test_docs_build_is_hardware_independent() -> None:
    config = CONFIG.read_text(encoding="utf-8")
    requirements = Path("docs/requirements.txt").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/gpu-docs.yml").read_text(encoding="utf-8")
    pull_request_workflow = Path(".github/workflows/test.yml").read_text(
        encoding="utf-8"
    )

    assert 'execute_notebooks: "off"' in config
    assert "jupyter-book>=1.0,<2" in requirements
    assert "sphinx>=7,<8" in requirements
    assert "sphinx_thebe" not in config
    assert "jupyter-book build docs" in workflow
    assert "check_docs_links.py" in workflow
    assert "check_docs_nav_toggle.py" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "actions/deploy-pages@v5" in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "pull_request:" in pull_request_workflow
    assert "jupyter-book build docs" in pull_request_workflow
    assert "check_docs_links.py --html-root" in pull_request_workflow
    assert "check_docs_nav_toggle.py" in pull_request_workflow

    for forbidden in ("nvidia-smi", "xcodebuild", "swift test", "private HDF5"):
        assert forbidden not in workflow


def test_public_repository_links_and_citation_copy() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    package = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "https://bobleesj.github.io/quantem.gpu/" in readme
    assert "[Contributing](CONTRIBUTING.md)" in readme
    assert Path("CONTRIBUTING.md").is_file()
    assert "## Citing quantem.gpu" in readme
    assert "the quantEM interactive framework" in readme
    assert "https://doi.org/10.1093/mam/ozag053.941" in readme
    assert '"CONTRIBUTING.md"' in package
    assert Path("CITATION.cff").is_file()
    assert '"CITATION.cff"' in package


def test_prerelease_docs_pin_the_exact_declared_candidate() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    install = Path("docs/install.md").read_text(encoding="utf-8")
    package = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = package["project"]["version"]
    exact_pin = f"quantem.gpu=={version}"

    assert version == "0.0.1rc6"
    assert exact_pin in readme
    assert exact_pin in intro
    assert exact_pin in install
    assert "evolving pre-release draft" in install
    assert "candidates are not assumed to be" in install
    assert "unpinned `--pre` install" in install

    requirements = re.findall(r'"(quantem\.gpu(?:\[[^]]+\])?==[^"]+)"', install)
    assert requirements
    assert all(requirement.endswith(f"=={version}") for requirement in requirements)


def test_scientific_writing_convention_is_explicit() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    writing = Path("docs/developer/writing.md").read_text(encoding="utf-8")

    assert "docs/developer/writing.md" in contributing
    assert "ophusgroup/dev" in writing
    assert "NumPy-style docstrings" in writing
    assert "(row, col)" in writing
    assert "200 kV" in writing
    assert "6 GiB" in writing
    assert "``:math:``" in writing
    assert "``.. math::``" in writing
    assert "## Tables: one cell, one value" in writing
    assert "exact configuration or one exact measurement" in writing
    assert "Do not infer a Cartesian product" in writing
    assert "1.199/1.212/1.106 s" in writing


def test_new_public_pages_do_not_leak_private_fixture_paths() -> None:
    paths = [
        Path("docs/intro.md"),
        Path("docs/dashboard.md"),
        Path("docs/concepts/scientific-contract.md"),
        *sorted(Path("docs/kernels").glob("*.md")),
        *sorted(Path("docs/platforms").glob("*.md")),
        *sorted(Path("docs/performance").glob("*.md")),
        *sorted(Path("docs/developer").glob("*.md")),
    ]
    private_markers = ("/home/", "/users/", "ssd/data", "downloads/")

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for marker in private_markers:
            assert marker not in text, f"{marker!r} leaked in {path}"


def test_scientific_kernel_pages_define_coordinates_math_and_optimization() -> None:
    pages = {
        "load": Path("docs/kernels/load-decode-bin.md"),
        "detector": Path("docs/kernels/virtual-detectors.md"),
        "dpc": Path("docs/kernels/com-dpc-idpc.md"),
        "ssb": Path("docs/kernels/ssb.md"),
        "region": Path("docs/kernels/scan-regions.md"),
    }

    for name, path in pages.items():
        text = path.read_text(encoding="utf-8")
        assert "(row, column)" in text, name
        assert "(r, c)" in text, name
        assert "## Optimization model" in text, name

    data_model = Path("docs/kernels/data-model.md").read_text(encoding="utf-8")
    for term in ("R_r", "R_c", "k_r", "k_c", "half-open", "Binning preserves counts"):
        assert term in data_model

    detector = pages["detector"].read_text(encoding="utf-8")
    assert "V_M[R_r,R_c]" in detector
    assert "Mean diffraction" in detector

    dpc = pages["dpc"].read_text(encoding="utf-8")
    for term in ("com_row", "com_col", "mu_r", "mu_c", "iDPC"):
        assert term in dpc
    for reference_function in (
        "masked_counts",
        "center_of_mass_reference",
        "rotate_dpc_reference",
        "curl_score",
        "select_dpc_rotation_reference",
        "integrate_idpc_reference",
    ):
        assert f"def {reference_function}" in dpc
    assert "-0.25j" in dpc
    assert "dim=(-2, -1)" not in dpc
    assert "movedim(-1" not in dpc
    assert ":-" not in dpc
    assert "from __future__ import annotations" not in dpc
    assert "ValueError" not in dpc
    assert "## Process at a glance" in dpc
    assert "Input | Scientific operation | Output and purpose" in dpc
    assert "detector distribution → vector field" in dpc
    for step in range(1, 9):
        assert f"### Step {step}" in dpc

    ssb = pages["ssb"].read_text(encoding="utf-8")
    for term in (
        "\\phi_b[\\mathbf R;\\boldsymbol\\theta]",
        "\\Gamma_b(\\boldsymbol{\\nu};\\boldsymbol\\theta)",
        "torch.fft.fft2",
        "def probe(",
            "gamma_b_nu =",
        "def phase_variance_loss(",
        "torch.argmin",
        "phi_R = torch.angle(object_R)",
    ):
        assert term in ssb
    assert "\\varphi" not in ssb
    assert "\\frac{2\\pi}{\\lambda}" not in ssb
    assert "G_b[\\boldsymbol{\\nu}]P_b" not in ssb
    assert "dim=(-2, -1)" not in ssb
    assert "movedim(-1, 0)" not in ssb
    assert "selected_R_k.permute(2, 0, 1)" in ssb
    assert "fft2 transforms the last two axes by default" in ssb


def test_primary_scientific_docs_use_row_column_component_symbols() -> None:
    paths = [
        Path("README.md"),
        Path("docs/intro.md"),
        *sorted(Path("docs/concepts").glob("*.md")),
        *sorted(Path("docs/kernels").glob("*.md")),
        *sorted(Path("docs/platforms").glob("*.md")),
        *sorted(Path("docs/developer").glob("*.md")),
    ]
    forbidden = (
        "r_y",
        "r_x",
        "q_y",
        "q_x",
        "c_y",
        "c_x",
        "g_y",
        "g_x",
        "k_y",
        "k_x",
        "(y, x)",
        "[y,x]",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{term!r} remains in {path}"


def test_primary_scientific_docs_use_R_q_notation() -> None:
    required_paths = [
        Path("README.md"),
        Path("docs/intro.md"),
        Path("docs/dashboard.md"),
        Path("docs/concepts/scientific-contract.md"),
        Path("docs/kernels/data-model.md"),
        Path("docs/kernels/index.md"),
        Path("docs/kernels/load-decode-bin.md"),
        Path("docs/kernels/ssb.md"),
    ]

    for path in required_paths:
        text = path.read_text(encoding="utf-8")
        assert "I[R_r,R_c,k_r,k_c]" in text, path
        assert "I[s_r,s_c,k_r,k_c]" not in text, path

    data_model = Path("docs/kernels/data-model.md").read_text(encoding="utf-8")
    ssb = Path("docs/kernels/ssb.md").read_text(encoding="utf-8")
    assert "\\mathbf R=(R_r,R_c)" in data_model
    assert "\\mathbf k=(k_r,k_c)" in data_model
    assert "\\mathcal F_{\\mathbf R\\rightarrow\\boldsymbol{\\nu}}" in ssb


def test_public_scientific_notation_uses_k_for_detector_and_nu_for_scan_frequency() -> None:
    public_pages = [
        Path("README.md"),
        Path("docs/intro.md"),
        Path("docs/dashboard.md"),
        *sorted(Path("docs/concepts").glob("*.md")),
        *sorted(Path("docs/kernels").glob("*.md")),
        *sorted(Path("docs/api").glob("*.md")),
        *sorted(Path("docs/platforms").glob("*.md")),
        Path("docs/developer/writing.md"),
    ]

    for path in public_pages:
        text = path.read_text(encoding="utf-8")
        assert "q_r" not in text, path
        assert "q_c" not in text, path
        assert "\\mathbf q" not in text, path

    ssb = Path("docs/kernels/ssb.md").read_text(encoding="utf-8")
    dpc = Path("docs/kernels/com-dpc-idpc.md").read_text(encoding="utf-8")
    assert "\\boldsymbol{\\nu}" in ssb
    assert "\\boldsymbol{\\nu}" in dpc


def test_remote_compute_is_a_deployment_not_a_kernel_runtime() -> None:
    index = Path("docs/platforms/index.md").read_text(encoding="utf-8")
    toc = Path("docs/_toc.yml").read_text(encoding="utf-8")
    service = Path("docs/remote/index.md").read_text(encoding="utf-8")
    service_words = " ".join(service.split())

    assert "caption: Kernel implementations" in toc
    assert "caption: Remote compute" in toc
    assert "[QuantEM.GPU Remote](../remote/index.md)" in index
    assert service.startswith("# QuantEM.GPU Remote")
    assert "not another kernel runtime" in service_words
    assert "`quantem-gpu-remote`" in service


def test_primary_compute_docs_do_not_depend_on_application_frameworks() -> None:
    paths = [
        Path("README.md"),
        Path("docs/intro.md"),
        Path("docs/dashboard.md"),
        Path("docs/install.md"),
        Path("docs/backends.md"),
        *sorted(Path("docs/concepts").glob("*.md")),
        *sorted(Path("docs/kernels").glob("*.md")),
        *sorted(Path("docs/platforms").glob("*.md")),
        *sorted(Path("docs/api").glob("*.md")),
        *sorted(Path("docs/developer").glob("*.md")),
    ]
    forbidden = ("quantem.widget", "Show4DSTEM", "Show2D", "Live4DSTEM", "quantem.live")

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{term!r} leaked into {path}"


def test_backend_pages_support_implementers_not_only_api_users() -> None:
    required_sections = {
        "cuda.md": (
            "## Dispatch and implementation layers",
            "## Execution and memory model",
            "## Build and focused checks",
            "## Profiling",
            "## Acceptance",
        ),
        "mps.md": (
            "## Dispatch and implementation layers",
            "## Execution and memory model",
            "## Build and focused checks",
            "## Profiling",
            "## Acceptance",
        ),
        "webgpu.md": (
            "## Dispatch and implementation layers",
            "## Execution and memory model",
            "## Source and build checks",
            "## Profiling and acceptance",
        ),
        "swift-metal.md": (
            "## Products and sources",
            "## Call and resource path",
            "## Coordinate and buffer contract",
            "## Build, profile, and verify",
        ),
        "cpu-reference.md": (
            "## Dispatch and implementation layers",
            "## Reference design",
            "## Arithmetic and independence",
            "## Focused checks",
        ),
    }

    for name, sections in required_sections.items():
        text = Path("docs/platforms", name).read_text(encoding="utf-8")
        for section in sections:
            assert section in text, f"{section!r} missing from {name}"


def test_ssb_page_explains_the_complete_default_fit() -> None:
    text = Path("docs/kernels/ssb.md").read_text(encoding="utf-8")
    text_words = " ".join(text.split())

    for required in (
        "200 TPE trials",
        "Nelder-Mead refinement",
        "does not average the 200 parameter sets",
        "phase-variance objective",
        "phase variance across $b$",
        "not another average",
        "operatorname*{arg\\,min}",
        "torch.fft.fft2",
        "def select_bright_field(",
        "def scan_fft(",
        "class SSBGeometry(",
        "return torch.fft.ifft2",
        "phi_b_R = torch.angle",
        "def evaluate_candidate(",
        "def best_tpe_candidate(",
        "def final_object(",
        "real PyTorch function definitions, not pseudocode",
        "not the source of the optimized CUDA, Python MPS, native Swift/Metal, or WebGPU runtimes",
        "final complex object wave",
    ):
        assert required in text_words


def test_public_api_pages_are_contracts_not_consumer_ui_guides() -> None:
    required_sections = {
        "images_dpc.md": (
            "## Inputs and outputs",
            "## Shapes, coordinates, dtypes, and units",
            "## Errors and unsupported requests",
            "## Provenance",
            "## Minimal example",
            "## Integration boundary",
        ),
        "ssb.md": (
            "## Inputs and outputs",
            "## Shapes, coordinates, dtypes, and units",
            "## Errors and unsupported requests",
            "## Provenance and exact reuse",
            "## Minimal fit",
            "## Integration boundary",
        ),
        "movie.md": (
            "## Inputs and outputs",
            "## Shapes, coordinates, dtypes, and units",
            "## Errors and unsupported requests",
            "## Provenance",
            "## Integration boundary",
        ),
    }

    for name, sections in required_sections.items():
        text = Path("docs/api", name).read_text(encoding="utf-8")
        for section in sections:
            assert section in text, f"{section!r} missing from {name}"
        assert "quantem.widget" not in text
        assert "Show3D" not in text


def test_ssb_api_keeps_benchmark_numbers_out_of_the_contract() -> None:
    text = Path("docs/api/ssb.md").read_text(encoding="utf-8")
    prose = " ".join(text.split())

    assert "23.058 s" not in text
    assert "24.528 s" not in text
    assert "Performance numbers do not live in this API contract" in prose


def test_scientific_kernel_pages_define_pipeline_and_data_contract() -> None:
    expectations = {
        "virtual-detectors.md": (
            "4D counts → detector geometry/mask",
            "## Coordinate, shape, dtype, unit, and provenance contract",
            "## Optimization model",
            "## Source map and gates",
        ),
        "com-dpc-idpc.md": (
            "4D counts → fused intensity/row-moment/column-moment reduction",
            "## Coordinate, shape, dtype, unit, and provenance contract",
            "## Optimization model",
            "## Source map and gates",
        ),
        "scan-regions.md": (
            "full source geometry → explicit half-open scan region",
            "## Coordinate, shape, dtype, unit, and provenance contract",
            "## Optimization model",
            "## Source map and gates",
        ),
        "display-export.md": (
            "scientific array → finite-value/range transform",
            "## Coordinate, shape, dtype, unit, and provenance contract",
            "## Optimization model",
            "## Source map and gates",
        ),
    }

    for name, required_fragments in expectations.items():
        text = Path("docs/kernels", name).read_text(encoding="utf-8")
        for fragment in required_fragments:
            assert fragment in text, f"{fragment!r} missing from {name}"
