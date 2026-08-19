from __future__ import annotations

import json
from pathlib import Path


MANIFEST = Path("tests/parity/backend_matrix.json")
EXPECTED_BACKENDS = {
    "cpu-reference",
    "cuda",
    "mps",
    "swift-metal",
    "webgpu",
}
EXPECTED_CAPABILITIES = {
    "io.decode-bin-provenance",
    "detector.integer-products",
    "dpc.com-rotation-idpc",
    "display.transform-histogram-color-fft",
    "ssb.object-phase-loss",
}
ALLOWED_LEVELS = {
    "reference",
    "reference-fixture",
    "required",
    "required-hardware",
    "partial-hardware",
    "not-implemented",
}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_backend_parity_manifest_covers_every_domain_and_backend() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == 1
    assert set(manifest["backends"]) == EXPECTED_BACKENDS
    assert {item["id"] for item in manifest["capabilities"]} == EXPECTED_CAPABILITIES

    for capability in manifest["capabilities"]:
        coverage = capability["coverage"]
        assert set(coverage) == EXPECTED_BACKENDS
        assert capability["parity"].strip()
        for backend, entry in coverage.items():
            assert entry["level"] in ALLOWED_LEVELS, (capability["id"], backend)
            assert entry["gates"], (capability["id"], backend)


def test_backend_parity_manifest_only_names_retained_gates() -> None:
    manifest = _manifest()
    missing: list[tuple[str, str, str]] = []

    for capability in manifest["capabilities"]:
        for backend, entry in capability["coverage"].items():
            for gate in entry["gates"]:
                if not Path(gate).is_file():
                    missing.append((capability["id"], backend, gate))

    assert not missing


def test_backend_parity_manifest_freezes_scientific_policy() -> None:
    contract = _manifest()["contract"]

    assert contract["coordinate_order"] == "row-column"
    assert contract["real_space_crop"] == "explicit-only"
    assert contract["detector_bin"] == "explicit-count-preserving-with-partial-edges"
    assert contract["cpu_fallback"] == "explicit-reference-only"
    assert contract["integer_outputs"] == "byte-exact"

    required = set(contract["required_provenance"])
    assert {
        "source_identity",
        "source_shape",
        "source_dtype",
        "scan_region",
        "detector_region",
        "scan_bin",
        "detector_bin",
        "output_shape",
        "output_dtype",
        "backend",
        "device",
        "source_revision",
    } <= required
