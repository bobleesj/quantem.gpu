"""Keep source-map links valid in both site and packaged backend guides."""

import importlib.util
from pathlib import Path


def load_checker():
    path = Path(__file__).resolve().parents[2] / "scripts/check_docs_links.py"
    spec = importlib.util.spec_from_file_location("docs_link_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backend_readme_links_are_checked(tmp_path):
    (tmp_path / "README.md").write_text("# Package\n")
    (tmp_path / "CONTRIBUTING.md").write_text("# Contributing\n")
    backend = tmp_path / "src/quantem/gpu/io/backends/mps/kernels"
    backend.mkdir(parents=True)
    (backend / "README.md").write_text("[Runtime owner](../dense.py)\n")
    checker = load_checker()

    failures = checker.check_markdown(tmp_path)
    assert len(failures) == 1
    assert "src/quantem/gpu/io/backends/mps/kernels/README.md" in failures[0]
    assert "../dense.py" in failures[0]

    (backend.parent / "dense.py").write_text('"""Runtime owner."""\n')
    assert checker.check_markdown(tmp_path) == []


def test_repository_source_links_resolve():
    root = Path(__file__).resolve().parents[2]
    assert load_checker().check_markdown(root) == []
