from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
        "## Benchmark snapshot",
        "## Scientific kernel and implementation matrix",
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

    for operation in (
        "Load, bitshuffle/LZ4 decode",
        "BF/ABF/ADF/DF",
        "CoM row/column, DPC, rotation, iDPC",
        "SSB object, phase, loss",
        "Display transform, histogram, colormap, and FFT",
    ):
        assert operation in dashboard

    for evidence_id in (
        "CUDA-512-LOAD",
        "M2-AIR-BIN4-E2E",
        "WEBGPU-512-FULL",
        "WEBGPU-DPC-512",
        "CUDA-COM-512",
        "SSB-CUDA-512-FULL",
        "SSB-MPS-512-FULL",
        "PRODUCT-CACHE-REOPEN",
    ):
        assert evidence_id in dashboard

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

    assert "I[R_r,R_c,q_r,q_c]" in dashboard
    assert "not ranked" in dashboard


def test_intro_exposes_current_benchmarks_without_erasing_provenance() -> None:
    intro = Path("docs/intro.md").read_text(encoding="utf-8")

    for evidence_id in (
        "M2-AIR-BIN4-E2E",
        "CUDA-512-LOAD",
        "WEBGPU-512-FULL",
    ):
        assert evidence_id in intro

    for qualifier in (
        "not a platform ranking",
        "cache state",
        "bin/crop plan",
        "parity artifact",
    ):
        assert qualifier in intro


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
    for term in ("R_r", "R_c", "q_r", "q_c", "half-open", "Binning preserves counts"):
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

    ssb = pages["ssb"].read_text(encoding="utf-8")
    for term in (
        "\\phi_b[\\mathbf R;\\boldsymbol\\theta]",
        "\\Gamma_b(\\mathbf k;\\boldsymbol\\theta)",
        "torch.fft.fft2",
        "def probe(",
        "gamma_b_k =",
        "def phase_variance_loss(",
        "torch.argmin",
        "phi_R = torch.angle(object_R)",
    ):
        assert term in ssb
    assert "\\varphi" not in ssb
    assert "\\frac{2\\pi}{\\lambda}" not in ssb
    assert "G_b[\\mathbf k]P_b" not in ssb
    assert "dim=(-2, -1)" not in ssb
    assert "movedim(-1, 0)" not in ssb
    assert "selected_R_q.permute(2, 0, 1)" in ssb
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
        assert "I[R_r,R_c,q_r,q_c]" in text, path
        assert "I[s_r,s_c,q_r,q_c]" not in text, path

    data_model = Path("docs/kernels/data-model.md").read_text(encoding="utf-8")
    ssb = Path("docs/kernels/ssb.md").read_text(encoding="utf-8")
    assert "\\mathbf R=(R_r,R_c)" in data_model
    assert "\\mathbf q=(q_r,q_c)" in data_model
    assert "\\mathcal F_{\\mathbf R\\rightarrow\\mathbf k}" in ssb


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
        "not the source of the optimized CUDA, MPS/Metal, or WebGPU runtimes",
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
