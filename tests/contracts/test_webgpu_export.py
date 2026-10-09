"""The browser source manifest must include the complete browser import graph."""

import json
import re
from importlib.resources import files
from pathlib import Path


def test_manifest_is_unique_relative_and_import_complete():
    root = files("quantem.gpu")
    names = tuple(json.loads(root.joinpath("webgpu", "sources.json").read_text()))
    assert names == tuple(sorted(set(names)))
    assert "webgpu/index.ts" in names
    for name in names:
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts
        text = root.joinpath(name).read_text(encoding="utf-8")
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


def test_manifest_lists_no_browser_display():
    """Browser display belongs to quantem.widget's js/display; quantem.gpu lends it
    exact counts through detector/webgpu/borrowed-image.ts and ships no display kernels."""
    names = json.loads(files("quantem.gpu").joinpath("webgpu", "sources.json").read_text())
    assert not [name for name in names if name.startswith("display/")]
    assert "detector/webgpu/borrowed-image.ts" in names
