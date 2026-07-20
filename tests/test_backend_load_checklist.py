from __future__ import annotations

import re
from pathlib import Path


CHECKLIST = Path("docs/maintainer/backend-4dstem-load-checklist.md")
MATRIX = Path("docs/maintainer/backend-optimization-matrix.md")


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    tail = text[start + len(heading):]
    match = re.search(r"^## ", tail, flags=re.MULTILINE)
    return tail[: match.start()] if match else tail


def _rows(section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def _plain(text: str) -> str:
    return re.sub(r"[*`]", "", text).strip()


def test_signed_off_capabilities_have_public_evidence_rows() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    capability_rows = _rows(_section(text, "## Capability Matrix"))
    evidence_rows = _rows(_section(text, "## Signed-Off Evidence Map"))

    evidence: set[tuple[str, str]] = set()
    for row in evidence_rows[1:]:
        capability = _plain(row[0])
        backends = [_plain(part) for part in row[1].split(",")]
        for backend in backends:
            evidence.add((capability, backend))

    missing: list[tuple[str, str]] = []
    for row in capability_rows[1:]:
        capability = _plain(row[1])
        for backend, status in zip(("CUDA", "MPS", "WebGPU"), row[2:5], strict=True):
            if status == "Done" and (capability, backend) not in evidence:
                missing.append((capability, backend))

    assert not missing


def test_1024_rows_remain_repeat_stress_until_real_acquisition_signoff() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    capability_rows = _rows(_section(text, "## Capability Matrix"))
    load_1024 = next(row for row in capability_rows if "1024 scan load" in row[1])

    assert all(status != "Done" for status in load_1024[2:5])
    assert "repeat-stress" in load_1024[5]
    assert "No backend is signed off on a real" in load_1024[5]


def test_checklist_keeps_acceptance_and_privacy_contracts() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")

    for required in (
        "No hidden reduction",
        "Adapter honesty",
        "Stage split",
        "Footprint stated",
        "SwiftShader",
        "must not contain raw local file paths",
    ):
        assert required in text

    private_markers = (
        "/home/",
        "ssd/data",
        "logic_",
    )
    lowered = text.lower()
    for marker in private_markers:
        assert marker.lower() not in lowered


def test_rejected_low6_decoder_is_documented_but_not_shipped() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "Compressed-payload low6 sidecar" in text
    assert "low6 code is not shipped" in text

    for source_path in (
        Path("src/quantem/gpu/webgpu/bslz4.ts"),
        Path("src/quantem/gpu/webgpu/compute.ts"),
    ):
        source = source_path.read_text(encoding="utf-8")
        assert "PACKED6" not in source
        assert "LOW6" not in source
        assert "packed low6" not in source


def test_rejected_subgroup_token_decoder_is_documented_but_not_shipped() -> None:
    checklist = CHECKLIST.read_text(encoding="utf-8")
    matrix = MATRIX.read_text(encoding="utf-8")
    source = Path("src/quantem/gpu/webgpu/bslz4.ts").read_text(encoding="utf-8")

    assert "subgroup-token" in checklist
    assert "subgroup token parser" in matrix
    assert "__BSLZ4_SUBGROUP_LOW8" not in source
    assert "FUSED_FRAME_SUBGROUP_LOW8_WGSL" not in source


def test_product_browser_benchmark_matches_local_api_contract() -> None:
    source = Path("scripts/benchmark_webgpu_h5_product_browser.py").read_text(
        encoding="utf-8"
    )

    assert 'parser.add_argument("--master-name", default="sample_master.h5")' in source
    assert "Selected-block product sidecar template" in source
    assert "instead of native data files" in source
    assert "globalThis.__runProductFirst(input.files).catch" in source
    assert "__QT_LOCAL_H5_DEBUG" in source
