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
        "Getting started",
        "Scientific workflows",
        "Performance & parity",
        "Developers",
    ):
        assert f"caption: {caption}" in toc

    for page in (
        "concepts/scientific-contract",
        "platforms/index",
        "platforms/cuda",
        "platforms/mps",
        "platforms/swift-metal",
        "platforms/webgpu",
        "platforms/cpu-reference",
        "tutorials/workflow_math",
        "maintainer/history/index",
        "maintainer/history/webgpu-gqk-memory-2026-07",
        "maintainer/history/webgpu-frame-coop-u16-clip8-2026-07-25",
        "developer/writing",
        "performance/index",
        "performance/results",
        "performance/methodology",
        "performance/parity",
        "backends",
        "developer/index",
        "developer/adding-backend",
        "developer/testing",
        "maintainer/index",
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
    assert not Path("CITATION.cff").exists()
    assert "CITATION.cff" not in package


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
        Path("docs/concepts/scientific-contract.md"),
        Path("docs/tutorials/workflow_math.md"),
        *sorted(Path("docs/platforms").glob("*.md")),
        *sorted(Path("docs/performance").glob("*.md")),
        *sorted(Path("docs/developer").glob("*.md")),
    ]
    private_markers = ("/home/", "/users/", "ssd/data", "downloads/")

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for marker in private_markers:
            assert marker not in text, f"{marker!r} leaked in {path}"


def test_scientific_workflow_map_covers_coordinates_and_products() -> None:
    text = Path("docs/tutorials/workflow_math.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for term in (
        "I(\\mathbf r,\\mathbf q)",
        "scan coordinate",
        "detector coordinate",
        "BF",
        "DF",
        "ADF",
        "center of mass",
        "iDPC",
        "Single-sideband ptychography",
        "no real-space crop",
    ):
        assert term in normalized
