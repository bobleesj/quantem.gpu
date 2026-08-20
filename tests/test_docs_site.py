from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tomllib

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
    for entry in manifest["pages"]:
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
    toc = TOC.read_text(encoding="utf-8")
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "caption: Start here" in toc
    assert "file: dashboard" in toc
    assert "implementation overview" in intro
    assert "[Implementation overview](docs/dashboard.md)" in readme

    for heading in (
        "## Platform-first module dashboard",
        "## Speed and memory at a glance",
        "### Measured load configurations",
        "### Dtype support and peak memory",
        "### Minimum-device memory gates",
        "### What a 4 or 6 GiB budget can hold",
        "### I/O and first usable product — `quantem.gpu.io`",
        "#### Exact configuration gaps",
        "### Screening and prepared-product caches — `quantem.gpu.screening`",
        "### Virtual images — `quantem.gpu.detector`",
        "### Detector moments and phase contrast — `quantem.gpu.dpc`",
        "### Single-sideband ptychography — `quantem.gpu.SSB`",
        "### Cross-module platform map",
        "## Where an implementer starts",
        "## Dashboard maintenance rule",
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

    for evidence_state in (
        "✓",
        "Test",
        "Pending",
        "Ref",
        "unsupported or not a target",
    ):
        assert evidence_state in dashboard

    assert "| Evidence |" not in dashboard
    assert "Evidence / next gap" not in dashboard
    for device in (
        "NVIDIA RTX PRO 6000 Blackwell",
        "Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU)",
        "Apple M2 MacBook Air (`Mac14,2`, 8 GB)",
    ):
        assert device in dashboard

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
    assert "not ranked" in dashboard
    assert "| Process peak | **1.43 GB** |" in dashboard
    assert "18.00 GiB" in dashboard
    assert "9.00 GiB" in dashboard
    assert "1.125 GiB" in dashboard
    assert "2.25 GiB" in dashboard
    assert "| 4 GiB | `uint16` | 56 | **1.97 GiB** | 10 | Pending |" in dashboard
    assert "| 6 GiB | `uint16` | 85 | **2.99 GiB** | 7 | Pending |" in dashboard
    assert "Each `512x512 float32` product map is only **1 MiB**" in dashboard
    assert "ADF has an accelerated" in dashboard
    assert "physical 4/6 GiB signoff Pending" in dashboard
    assert "| `uint8` | `uint8` | `uint8` | Exact native counts | ✓ |" in dashboard
    assert "| `uint16` | `uint16` | `uint32` | General exact detector sum | ✓ |" in dashboard
    assert "| `uint16` | `uint8` | `float32` | Exact detector sum | ✓ |" in dashboard
    assert "A calculated payload is never relabeled as a measured peak" in dashboard
    assert "6 GiB of dedicated VRAM for CUDA" in dashboard_words
    assert "8 GB of total laptop RAM for WebGPU" in dashboard_words
    assert "| 8 GB ✓ at detector bin 4; SSB physical 8 GB Pending |" in dashboard

    for scan_size in (128, 256, 512, 1024):
        assert f"`{scan_size}x{scan_size}`" in dashboard
    for detector_bin in (1, 2, 4, 8):
        assert f"| {detector_bin} |" in dashboard
    assert "square scan-grid sizes, not detector dimensions" in dashboard
    assert "| **Python MPS** | `512x512` | Native real acquisition | Full active BF | ✓ | p50 | **537.58 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-07-28 |" in dashboard

    module_headings = (
        "### I/O and first usable product — `quantem.gpu.io`",
        "### Screening and prepared-product caches — `quantem.gpu.screening`",
        "### Virtual images — `quantem.gpu.detector`",
        "### Detector moments and phase contrast — `quantem.gpu.dpc`",
        "### Single-sideband ptychography — `quantem.gpu.SSB`",
        "### Cross-module platform map",
    )
    for index, heading in enumerate(module_headings[:-1]):
        section_end = module_headings[index + 1]
        section = dashboard.split(heading, 1)[1].split(section_end, 1)[0]
        for runtime in (
            "CUDA",
            "Python MPS",
            "Native Swift/Metal",
            "WebGPU",
            "CPU reference",
        ):
            assert runtime in section


def test_intro_exposes_current_benchmarks_without_erasing_provenance() -> None:
    intro = Path("docs/intro.md").read_text(encoding="utf-8")

    assert "| Evidence |" not in intro
    assert "Evidence / next gap" not in intro
    assert "NVIDIA RTX PRO 6000 Blackwell" in intro
    assert "Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU)" in intro
    assert "Apple M2 MacBook Air (`Mac14,2`, 8 GB)" in intro

    module_sections = (
        "### I/O — `quantem.gpu.io`",
        "### Screening — `quantem.gpu.screening`",
        "### Virtual images — `quantem.gpu.detector`",
        "### Detector moments and phase contrast — `quantem.gpu.dpc`",
        "### Single-sideband ptychography — `quantem.gpu.SSB`",
        "### Other public modules",
    )
    for section in module_sections:
        assert section in intro
    assert [intro.index(section) for section in module_sections] == sorted(
        intro.index(section) for section in module_sections
    )

    for status in (
        "**✓**",
        "**Test**",
        "**Pending**",
        "**Ref**",
        "**—**",
    ):
        assert status in intro

    for table_field in (
        "Platform",
        "Selected scan",
        "Detector bin",
        "Cache state",
        "Fixture",
        "Statistic",
        "Time",
        "Device tested",
        "Date tested",
    ):
        assert table_field in intro

    for headline_time in (
        "0.386 s",
        "0.390 s",
        "0.586 s",
        "1.695 s",
        "2.273 s",
        "0.605 s",
        "2.027 s",
        "1.982 s",
        "2.775 s",
        "1.985 s",
        "2.043 s",
        "0.824 s",
        "1.281 s",
        "1.044 s",
        "0.979 s",
        "12.31 s",
        "6.711 s",
        "20.803 ms",
        "6.8 ms",
        "13.448 ms",
        "5.339 ms",
        "15.622 ms",
        "0.291 ms",
        "0.353 ms",
        "32.2 ms",
        "32.335 ms",
        "497.187 ms",
        "11.168 s",
        "537.58 ms",
        "669.1 ms",
        "1.69 s",
        "1.91 s",
    ):
        assert headline_time in intro

    assert "resident kernels are **not loading times**" in intro
    assert "No empty cell implies support" in intro

    for qualifier in (
        "cache state",
        "bin/crop plan",
        "parity artifact",
    ):
        assert qualifier in intro

    assert intro.index("## Module capabilities and benchmarks") < intro.index(
        "## The shared coordinate contract"
    )
    assert "### Resident scientific products" not in intro
    assert "## How loading becomes a usable product" in intro
    assert "START WALL CLOCK" in intro
    assert "FIRST COMPLETE USABLE PRODUCT" in intro
    assert "no automatic real-space crop" in intro

    load_page = Path("docs/kernels/load-decode-bin.md").read_text(encoding="utf-8")
    assert "## Count-preserving detector binning" in load_page
    assert "exact sum of one" in load_page
    assert "materializing both a full unbinned volume" in load_page


def test_intro_ssb_size_matrix_tracks_fixed_size_runtime_registries() -> None:
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    ssb_section = intro.split(
        "### Single-sideband ptychography — `quantem.gpu.SSB`", 1
    )[1].split("### Other public modules", 1)[0]
    cuda = Path(
        "src/quantem/gpu/ssb/compute/cuda/kernels/__init__.py"
    ).read_text(encoding="utf-8")
    mps = Path(
        "src/quantem/gpu/ssb/compute/mps/kernels/__init__.py"
    ).read_text(encoding="utf-8")
    webgpu = Path(
        "src/quantem/gpu/ssb/compute/webgpu/kernels/index.ts"
    ).read_text(encoding="utf-8")

    assert "scan sizes, not detector sizes" in ssb_section
    assert "| Platform | Scan grid | Source kind | BF policy | State | Statistic | Time | Device tested | Date tested |" in ssb_section
    assert "| Platform | `128x128` | `256x256` |" not in ssb_section
    for size in (128, 256, 512, 1024):
        assert f"`{size}x{size}`" in ssb_section
        assert f"{size}:" in cuda
        assert f"{size}:" in mps
        for platform in (
            "CUDA",
            "Python MPS",
            "Native Swift/Metal",
            "WebGPU",
            "CPU reference",
        ):
            assert ssb_section.count(f"| **{platform}** | `{size}x{size}` |") == 1
    assert "SUPPORTED_SSB_SIZES = [128, 256, 512, 1024]" in webgpu

    assert "Incomplete frozen reference" in ssb_section
    assert "Native real acquisition | Full active BF" in ssb_section
    assert "NVIDIA RTX PRO 6000 Blackwell | 2026-07-19" in ssb_section
    assert "Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-07-28" in ssb_section
    assert "native swift/metal now has a package-owned exact 512×512 implementation" in ssb_section.lower()


def test_platform_first_io_tables_expose_bins_devices_and_dates() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    swift_plan = Path(
        "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/"
        "Metal4DSTEMLoadPlan.swift"
    ).read_text(encoding="utf-8")

    load_section = dashboard.split(
        "### Measured load configurations", 1
    )[1].split(
        "(dtype-support-and-peak-memory)=", 1
    )[0]
    gap_section = dashboard.split(
        "### I/O and first usable product — `quantem.gpu.io`", 1
    )[1].split(
        "### Screening and prepared-product caches — `quantem.gpu.screening`", 1
    )[0]

    assert "| Platform | Selected scan | Scan plan |" in load_section
    assert "| Platform | Selected scan | Scan plan |" in gap_section
    assert "supportedDetectorBins = [1, 2, 4]" in swift_plan

    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in load_section.splitlines()
        if line.startswith("| [**")
    ]
    webgpu_rows = [row for row in rows if "WebGPU" in row[0]]
    observed = {
        (row[1], int(row[4]), row[5], row[7], row[8], row[13])
        for row in webgpu_rows
    }
    assert {
        ("`256x256`", 1, "`192x192`", "`uint8`", "`uint8`", "**0.338 s**"),
        ("`256x256`", 2, "`96x96`", "`uint8`", "`float32`", "**0.774 s**"),
        ("`256x256`", 4, "`48x48`", "`uint8`", "`float32`", "**0.755 s**"),
        ("`256x256`", 8, "`24x24`", "`uint8`", "`float32`", "**0.733 s**"),
        ("`512x512`", 1, "`192x192`", "`uint8`", "`uint8`", "**0.824 s**"),
        ("`512x512`", 2, "`96x96`", "`uint16`", "`float32`", "**1.281 s**"),
        ("`512x512`", 4, "`48x48`", "`uint16`", "`float32`", "**1.044 s**"),
        ("`512x512`", 8, "`24x24`", "`uint16`", "`float32`", "**0.979 s**"),
        ("`512x512`", 2, "`96x96`", "`uint16`", "`float32`", "**2.651 s**"),
    } <= observed

    assert "1.199/1.212/1.106" not in load_section
    assert "0.774/0.755/0.733" not in load_section
    assert "1.985 / 2.043" not in load_section
    assert "Joint crop-plus-bin parity and hardware timing" in gap_section
    assert "| **Native Swift/Metal** | `512x512` | Full | `192x192` | 8 | `24x24` | — | — |" in gap_section
    assert "|  |" not in load_section
    assert "|  |" not in gap_section

    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    intro_io = intro.split("### I/O — `quantem.gpu.io`", 1)[1].split(
        "### Screening — `quantem.gpu.screening`", 1
    )[0]
    intro_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in intro_io.splitlines()
        if line.startswith("| **")
    ]
    platforms = ("CUDA", "Python MPS", "Native Swift/Metal", "WebGPU")
    def platform(cell: str) -> str:
        return next(name for name in platforms if name in cell)

    dashboard_keys = {
        (
            platform(row[0]), row[1], row[2], row[4], row[5], row[6], row[7],
            row[8], row[9], row[11], row[12], row[13], row[17], row[18],
        )
        for row in rows
    }
    intro_keys = {
        (
            platform(row[0]), row[1], row[2], row[3], row[4], row[5], row[6],
            row[7], row[8], row[9], row[10], row[11], row[12], row[13],
        )
        for row in intro_rows
    }
    assert intro_keys == dashboard_keys


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
    assert "### Minimum-memory release gates" in intro
    assert "## Minimum-device memory gates" in methodology
    assert "### 6 GiB VRAM release floor" in cuda
    assert "### 8 GB laptop release floor" in webgpu

    for detector_bin, payload, gate in (
        (1, "18.00 GiB", "No"),
        (2, "9.00 GiB", "No"),
        (4, "2.25 GiB", "Pending"),
        (8, "0.5625 GiB", "Pending"),
    ):
        row = (
            f"| [**CUDA**](platforms/cuda.md) | 6 GiB VRAM | `512x512` | "
            f"Full | `192x192` | {detector_bin} |"
        )
        assert row in dashboard
        matching = next(line for line in dashboard.splitlines() if line.startswith(row))
        assert f"**{payload}**" in matching
        assert f"**{gate}**" in matching

    for detector_bin, payload, gate in (
        (1, "9.00 GiB", "No"),
        (2, "9.00 GiB", "No"),
        (4, "2.25 GiB", "Pending"),
        (8, "0.5625 GiB", "Pending"),
    ):
        row = (
            f"| [**WebGPU**](platforms/webgpu.md) | 8 GB total RAM | `512x512` "
            f"| Full | `192x192` | {detector_bin} |"
        )
        assert row in dashboard
        matching = next(line for line in dashboard.splitlines() if line.startswith(row))
        assert f"**{payload}**" in matching
        assert f"**{gate}**" in matching

    native_row = (
        "| [**Native Swift/Metal**](platforms/swift-metal.md) | 8 GB unified "
        "RAM | `512x512` | Full | `192x192` | 4 |"
    )
    matching = next(line for line in dashboard.splitlines() if line.startswith(native_row))
    assert "**1.125 GiB**" in matching
    assert "**✓**" in matching
    assert "Apple M2 MacBook Air (`Mac14,2`, 8 GB)" in matching

    assert "A calculated payload can prove **No**" in " ".join(dashboard.split())
    assert "Do not convert **Pending** or **Test** to ✓" in methodology
    assert "Measurements on a larger Blackwell GPU do not by themselves prove" in cuda
    assert "A real-adapter run on a higher-memory machine cannot receive this ✓" in webgpu


def test_load_dtype_docs_keep_precision_and_peak_memory_distinct() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    io_api = Path("docs/api/io.md").read_text(encoding="utf-8")
    load_kernel = Path("docs/kernels/load-decode-bin.md").read_text(encoding="utf-8")
    methodology = Path("docs/performance/methodology.md").read_text(encoding="utf-8")
    native_api = Path("docs/api/native_4dstem_io.md").read_text(encoding="utf-8")
    public_load = Path("src/quantem/gpu/io/load.py").read_text(encoding="utf-8")

    for text in (dashboard, io_api, load_kernel, methodology):
        assert "source" in text.lower()
        assert "working" in text.lower()
        assert "accumulation" in text.lower()
        assert "resident" in text.lower()
        assert "peak" in text.lower()

    assert '`dtype="u8"` requests saturating unsigned 8-bit browse counts' in dashboard
    assert "values above 255" in io_api
    assert "Do not call a payload size “peak memory.”" in methodology
    assert "unsigned 8-bit and unsigned 16-bit detector" in native_api
    assert '``"u8"`` requests a saturating browse output' in public_load


def test_narrow_tables_scroll_without_compressing_provenance_columns() -> None:
    css = Path("docs/_static/custom.css").read_text(encoding="utf-8")

    assert ".pst-scrollable-table-container" in css
    assert "overflow-x: auto" in css
    assert ".pst-scrollable-table-container > table.table" in css
    assert "min-width: 52rem" in css


def test_overview_tables_use_device_and_date_not_evidence_columns() -> None:
    dashboard = Path("docs/dashboard.md").read_text(encoding="utf-8")
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    writing = Path("docs/developer/writing.md").read_text(encoding="utf-8")

    for overview in (dashboard, intro):
        assert "Device tested" in overview
        assert "Date tested" in overview
        assert "| Evidence |" not in overview
        assert "Evidence / next gap" not in overview

        for block in overview.split("\n\n"):
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

    assert "human-facing overview table ends with **Device" in writing
    assert "Do not add evidence-ID or source-revision columns" in writing


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
    benchmark_ids = (
        "CUDA-512-LOAD",
        "CUDA-1024-LOAD",
        "CUDA-STOCHASTIC-IO",
        "CUDA-CAL-BUILD",
        "MPS-CAL-BUILD",
        "PRODUCT-CACHE-REOPEN",
        "MPS-1024-LOAD",
        "M2-AIR-BIN4-E2E",
        "WEBGPU-512-FULL",
        "WEBGPU-1024-FULL-REJECTED",
        "WEBGPU-DET-BIN",
        "WEBGPU-256-CROP",
        "WEBGPU-BF-256",
        "WEBGPU-BF-512",
        "WEBGPU-BF-1024",
        "WEBGPU-BF-1024-STRESS",
        "WEBGPU-VISIBLE-512",
        "WEBGPU-DPC-512",
        "CUDA-BF-512",
        "CUDA-ADF-512",
        "CUDA-DF-512",
        "CUDA-COM-512",
        "MPS-SAVE-U16-512",
        "SSB-CUDA-512-FULL",
        "SSB-MPS-512-R30",
        "SSB-MPS-512-FULL",
        "SSB-MPS-1024-SYNTH",
    )

    for benchmark_id in benchmark_ids:
        rows = [line for line in text.splitlines() if line.startswith(f"| {benchmark_id} |")]
        assert len(rows) == 1, benchmark_id
        row = rows[0]
        assert "2026-" in row
        assert "github.com/bobleesj/quantem.gpu/commit/" in row
        assert row.count("|") == 8

    for required in (
        "cache state and benchmark definition",
        "scientific and calibration provenance",
        "first process",
        "saved-result reopen",
        "explicit exact-sum detector bin 4",
        "single visible run, not a median",
        "No frozen parity value",
    ):
        assert required.lower() in text.lower()

    assert "cold `8.90 s`" not in text
    assert (
        "| SSB-CUDA-512-FULL | 2026-07-19,"
        in text
    )
    assert "NVIDIA RTX PRO 6000 Blackwell | Prepared real `512x512` SSB" in text
    assert text.count(
        "Apple M5 MacBook Pro (`Mac17,2`), 10-core GPU, 24 GB unified memory"
    ) == 3


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


def test_public_repository_files_are_linked_from_readme() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    citation = Path("CITATION.cff").read_text(encoding="utf-8")
    package = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "https://bobleesj.github.io/quantem.gpu/" in readme
    assert "[Contributing](CONTRIBUTING.md)" in readme
    assert "[CITATION.cff](CITATION.cff)" in readme
    assert Path("CONTRIBUTING.md").is_file()
    assert Path("CITATION.cff").is_file()
    assert "cff-version: 1.2.0" in citation
    assert "https://doi.org/10.1093/mam/ozag053.941" in readme
    assert '"CONTRIBUTING.md"' in package
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
