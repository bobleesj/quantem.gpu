import json
import os
from importlib.resources import files
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_WIDGET_REPO"),
    reason="set QUANTEM_WIDGET_REPO to check widget WebGPU source sync",
)
def test_widget_webgpu_sources_match_quantem_gpu() -> None:
    """The widget bundle copy should match canonical quantem.gpu WebGPU sources."""
    root = files("quantem.gpu")

    widget_repo = Path(os.environ["QUANTEM_WIDGET_REPO"]).expanduser()
    legacy_engine_dir = widget_repo / "js" / "engine"
    if legacy_engine_dir.is_dir():
        assert not list(legacy_engine_dir.glob("*.ts"))

    engine_dir = widget_repo / "js" / ".generated" / "engine"
    if not engine_dir.is_dir():
        pytest.skip("widget generated engine source directory is not available")

    for name in json.loads(root.joinpath("webgpu", "sources.json").read_text()):
        target = engine_dir / name
        assert target.exists(), f"widget generated engine is missing synced {name}"
        assert target.read_text(encoding="utf-8") == root.joinpath(name).read_text(encoding="utf-8")
