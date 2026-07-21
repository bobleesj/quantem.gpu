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


def test_1024_row_tracks_cuda_mps_real_signoff_only() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    capability_rows = _rows(_section(text, "## Capability Matrix"))
    load_1024 = next(row for row in capability_rows if "1024 scan load" in row[1])

    assert load_1024[2:5] == ["Done", "Done", "Partial"]
    assert "repeat-stress" in load_1024[5]
    assert "CUDA and MPS have true real-acquisition" in load_1024[5]
    assert "WebGPU has true-acquisition product-first BF signoff" in load_1024[5]
    assert "full-stack no-bin browser scan load still needs signoff" in load_1024[5]


def test_detector_bin_row_tracks_webgpu_signoff() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    matrix = MATRIX.read_text(encoding="utf-8")
    capability_rows = _rows(_section(text, "## Capability Matrix"))
    det_bin = next(row for row in capability_rows if "Detector bin" in row[1])

    assert det_bin[2:5] == ["Done", "Done", "Done"]
    assert "full-512" in det_bin[5]
    assert "crop-256" in det_bin[5]
    assert "detBin=2" in det_bin[5]
    assert "exact corrected-frame checksums" in det_bin[5]
    assert "WebGPU detector-bin local-file load" in matrix
    assert "crop-256 20-repeat medians" in matrix
    assert "zero-bad-before-bin reference" in matrix


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
        "/" + "home/",
        "ssd" + "/data",
        "logic" + "_",
    )
    lowered = text.lower()
    for marker in private_markers:
        assert marker.lower() not in lowered


def test_public_release_docs_do_not_name_private_datasets() -> None:
    paths = [
        Path("README.md"),
        Path("CHANGELOG.md"),
        Path("docs/backends.md"),
        Path("docs/maintainer/backend-4dstem-load-checklist.md"),
        Path("docs/maintainer/backend-optimization-matrix.md"),
    ]
    private_markers = (
        "/" + "home/",
        "ssd" + "/data",
        "logic" + "_",
        "sam" + "sung",
    )

    for path in paths:
        lowered = path.read_text(encoding="utf-8").lower()
        for marker in private_markers:
            assert marker not in lowered, f"{marker!r} leaked in {path}"


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
    assert "Fixture file(s) not found" in source
    assert "Browser mounted" in source
    assert "globalThis.__runTrue1024Product(input.files).catch" in source
    assert "globalThis.__runProductFirst(input.files).catch" in source
    assert "globalThis.__sh4d.h5ProductFirstRoi" in source
    assert "reference .f32 file was not selected" in source
    assert "__QT_LOCAL_H5_DEBUG" in source


def test_true_1024_product_signoff_is_documented_without_overclaiming() -> None:
    checklist = CHECKLIST.read_text(encoding="utf-8")
    matrix = MATRIX.read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    for text in (checklist, matrix, readme):
        assert "true real-acquisition `1024" in text or "true `1024" in text
        assert "product-first BF" in text
        assert "full-stack no-bin" in text

    assert "4.92 s" in matrix
    assert "selected compressed payload `6.88 GB`" in matrix
    assert "max/mean abs error `0`" in readme
    assert "This is not full-stack no-bin browse/load signoff" in readme


def test_show4dstem_browser_benchmark_can_reject_url_fallback() -> None:
    source = Path("scripts/benchmark_webgpu_h5_browser.py").read_text(
        encoding="utf-8"
    )

    assert '"--require-local-profile"' in source
    assert 'profile.get("localFiles")' in source
    assert "instead of accepting the URL/fetch fallback" in source
    assert '"--block-index-template"' in source
    assert "metadata-only QH5IDX sidecar template" in source
    assert "args.block_index_template.format(index=index)" in source
    assert '"blockIndex": bool(args.block_index_template)' in source
    assert "Fixture file(s) not found" in source
    assert "Browser mounted" in source
