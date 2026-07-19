from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_WIDGET_REPO"),
    reason="set QUANTEM_WIDGET_REPO to check widget WebGPU source sync",
)
def test_widget_webgpu_sources_match_quantem_gpu() -> None:
    """The widget bundle copy should match canonical quantem.gpu WebGPU sources."""
    from quantem.gpu import webgpu

    widget_repo = Path(os.environ["QUANTEM_WIDGET_REPO"]).expanduser()
    engine_dir = widget_repo / "js" / "engine"
    if not engine_dir.is_dir():
        pytest.skip("widget engine source directory is not available")

    for name in webgpu.source_names():
        target = engine_dir / name
        assert target.exists(), f"widget engine is missing synced {name}"
        assert target.read_text(encoding="utf-8") == webgpu.source_text(name)
