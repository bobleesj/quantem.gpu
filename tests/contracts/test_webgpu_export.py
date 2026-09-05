"""Build-time source export must include the complete browser import graph."""

import json
from pathlib import Path
import re

import pytest

from quantem.gpu import webgpu


def test_manifest_is_unique_relative_and_import_complete():
    names = webgpu.source_names()
    assert names == tuple(sorted(set(names)))
    assert "webgpu/index.ts" in names
    for name in names:
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts
        text = webgpu.source_text(name)
        if name.endswith(".json"):
            json.loads(text)
            continue
        for target in re.findall(r"(?:from\s*|import\s*\()\s*[\"']([.][^\"']+)", text):
            # Resolve static relative imports, including directory index exports.
            resolved = Path("/", path.parent, target).resolve().relative_to("/")
            candidates = (str(resolved), str(resolved) + ".ts", str(resolved / "index.ts"))
            assert any(candidate in names for candidate in candidates), (name, target)
        for target in re.findall(r'///\s*<reference\s+path="([^"]+)"', text):
            resolved = Path("/", path.parent, target).resolve().relative_to("/")
            assert str(resolved) in names, (name, target)


def test_export_matches_resources_and_does_not_overwrite(tmp_path):
    root = webgpu.export_sources(tmp_path / "browser")
    assert (root / "io/backends/webgpu/jsfive.d.ts").is_file()
    assert set(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()) == set(webgpu.source_names())
    for name in webgpu.source_names():
        assert (root / name).read_text() == webgpu.source_text(name)
    with pytest.raises(FileExistsError, match="fresh generated directory"):
        webgpu.export_sources(root)
    with pytest.raises(ValueError, match="source_names"):
        webgpu.source_text("../io/load.py")
