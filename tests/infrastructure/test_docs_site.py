import re
from pathlib import Path

from tests.infrastructure.private_names import mentions_private_name

TOC = Path("docs/_toc.yml")
CONFIG = Path("docs/_config.yml")
PUBLIC_TEXT = [Path("README.md"), Path("CONTRIBUTING.md"), *sorted(Path("docs").rglob("*.md"))]


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


def test_public_docs_name_hardware_not_hosts_people_or_local_paths() -> None:
    private_paths = ("/home/", "/users/", "ssd/data", "downloads/", "logic_")

    for path in PUBLIC_TEXT:
        text = path.read_text(encoding="utf-8")
        assert not mentions_private_name(text), path
        for marker in private_paths:
            assert marker not in text.lower(), f"{marker!r} leaked in {path}"


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
    intro = Path("docs/intro.md").read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    package = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "https://bobleesj.github.io/quantem.gpu/" in readme
    assert "[Contributing](CONTRIBUTING.md)" in readme
    assert Path("CONTRIBUTING.md").is_file()
    assert "## Citing quantem.gpu" in readme
    assert "https://doi.org/10.1093/mam/ozag053.941" in readme
    assert "Sangjoon Lee et al." in intro
    assert "https://doi.org/10.1093/mam/ozag053.941" in intro
    assert "author: Sangjoon Lee et al." in config
    assert '{name = "Sangjoon Lee"}' in package
    citation = Path("CITATION.cff").read_text(encoding="utf-8")
    assert "preferred-citation:" in citation
    assert "10.1093/mam/ozag053.941" in citation
    for public_source in (readme, intro, config, package):
        assert re.search(r"\bBob\b", public_source) is None


def test_api_guide_maps_every_public_namespace() -> None:
    api = Path("docs/api/index.md").read_text(encoding="utf-8")
    root = Path("src/quantem/gpu/__init__.py").read_text(encoding="utf-8")

    for namespace in ("detector", "device", "dpc", "io", "movie", "optics", "parallax", "screening"):
        assert f'"{namespace}"' in root
        assert f"`{namespace}`" in api


def test_scientific_docs_use_row_column_symbols() -> None:
    paths = [
        Path("README.md"),
        Path("docs/intro.md"),
        *sorted(Path("docs/concepts").glob("*.md")),
        *sorted(Path("docs/kernels").glob("*.md")),
        *sorted(Path("docs/platforms").glob("*.md")),
        *sorted(Path("docs/developer").glob("*.md")),
    ]
    forbidden = ("r_y", "r_x", "q_y", "q_x", "c_y", "c_x", "g_y", "g_x", "k_y", "k_x", "(y, x)", "[y,x]")

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{term!r} remains in {path}"


def test_scientific_docs_use_k_for_detector_and_nu_for_scan_frequency() -> None:
    public_pages = [
        Path("README.md"),
        Path("docs/intro.md"),
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
        assert "I[s_r,s_c,k_r,k_c]" not in text, path

    data_model = Path("docs/kernels/data-model.md").read_text(encoding="utf-8")
    assert "\\mathbf R=(R_r,R_c)" in data_model
    assert "\\mathbf k=(k_r,k_c)" in data_model
    for page in ("ssb.md", "com-dpc-idpc.md"):
        assert "\\boldsymbol{\\nu}" in Path("docs/kernels", page).read_text(encoding="utf-8")


def test_compute_docs_do_not_depend_on_application_frameworks() -> None:
    paths = [
        Path("README.md"),
        Path("docs/intro.md"),
        Path("docs/install.md"),
        Path("docs/backends.md"),
        *sorted(Path("docs/concepts").glob("*.md")),
        *sorted(Path("docs/kernels").glob("*.md")),
        *sorted(Path("docs/platforms").glob("*.md")),
        *sorted(Path("docs/api").glob("*.md")),
        *sorted(Path("docs/developer").glob("*.md")),
    ]
    forbidden = ("quantem.widget", "Show4DSTEM", "Show2D", "Show3D", "Live4DSTEM", "quantem.live")

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{term!r} leaked into {path}"


def test_scientific_kernel_pages_define_coordinates_contract_and_optimization() -> None:
    for name in ("load-decode-bin", "virtual-detectors", "com-dpc-idpc", "ssb", "scan-regions"):
        text = Path("docs/kernels", f"{name}.md").read_text(encoding="utf-8")
        assert "(row, column)" in text, name
        assert "(r, c)" in text, name
        assert "## Optimization model" in text, name
    for name in ("virtual-detectors", "com-dpc-idpc", "scan-regions", "display-export"):
        text = Path("docs/kernels", f"{name}.md").read_text(encoding="utf-8")
        assert "## Coordinate, shape, dtype, unit, and provenance contract" in text, name
        assert "## Source map and gates" in text, name


def test_backend_pages_support_implementers_not_only_api_users() -> None:
    required_sections = {
        "cuda.md": ("## Dispatch and implementation layers", "## Execution and memory model", "## Acceptance"),
        "mps.md": ("## Dispatch and implementation layers", "## Execution and memory model", "## Acceptance"),
        "webgpu.md": ("## Dispatch and implementation layers", "## Execution and memory model"),
        "swift-metal.md": ("## Products and sources", "## Build, profile, and verify"),
        "cpu-reference.md": ("## Dispatch and implementation layers", "## Reference design"),
    }

    for name, sections in required_sections.items():
        text = Path("docs/platforms", name).read_text(encoding="utf-8")
        for section in sections:
            assert section in text, f"{section!r} missing from {name}"


def test_public_api_pages_are_contracts() -> None:
    for name in ("images_dpc.md", "ssb.md", "movie.md"):
        text = Path("docs/api", name).read_text(encoding="utf-8")
        for section in (
            "## Inputs and outputs",
            "## Shapes, coordinates, dtypes, and units",
            "## Errors and unsupported requests",
            "## Integration boundary",
        ):
            assert section in text, f"{section!r} missing from {name}"
