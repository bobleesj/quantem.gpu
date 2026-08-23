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
    "io.selective-scan-loading",
    "detector.integer-products",
    "screening.prepared-products",
    "dpc.com-rotation-idpc",
    "display.transform-histogram-color-fft",
    "ssb.object-phase-loss",
    "ssb.calibration-200-nelder-mead",
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


def test_selective_scan_loading_contract_is_explicit_and_fail_closed() -> None:
    manifest = _manifest()
    selective = next(
        item
        for item in manifest["capabilities"]
        if item["id"] == "io.selective-scan-loading"
    )

    assert selective["contract_version"] == "quantem-gpu-selective-scan/v1"
    assert selective["public_owner"] == "quantem.gpu.io.load"
    selectors = selective["selectors"]
    assert selectors["mutually_exclusive"] == [
        "scan_region",
        "scan_indices",
        "random_positions",
    ]
    assert selectors["scan_region"] == {
        "coordinates": [
            "scan_row_start",
            "scan_row_stop",
            "scan_column_start",
            "scan_column_stop",
        ],
        "interval": "half-open",
        "bounds": "nonempty-and-contained-in-source-scan",
        "output_order": "logical-row-major",
        "output_shape": (
            "(selected_scan_rows, selected_scan_columns, detector_rows, "
            "detector_columns)"
        ),
    }
    assert selectors["scan_indices"]["output_order"] == "request-order-preserved"
    assert selectors["scan_indices"]["duplicates"] == "preserved"
    assert selectors["random_positions"] == {
        "index_space": "logical-row-major-scan",
        "seed": "same-integer-seed-reproduces-selection",
        "replace_default": False,
        "replace_false": "unique-positions",
        "replace_true": "duplicates-allowed",
        "output_contract": "scan_indices",
    }

    detector = selective["detector_region"]
    assert detector["requires_selector"] == "scan_region"
    assert detector["coordinates"] == [
        "detector_row_start",
        "detector_row_stop",
        "detector_column_start",
        "detector_column_stop",
    ]
    assert detector["application_order"] == "after-explicit-detector-bin"

    result = selective["exact_result"]
    assert result["scan_bin"] == 1
    assert result["counts"] == "unchanged-exact-integer-counts"
    assert result["lossy_or_saturating_output"] == "outside-this-parity-contract"
    assert {
        "source_identity",
        "source_shape",
        "source_dtype",
        "scan_selector",
        "selected_scan_positions",
        "scan_region",
        "detector_region",
        "scan_order",
        "scan_bin",
        "detector_bin",
        "output_shape",
        "output_dtype",
        "backend",
        "device",
        "source_revision",
    } == set(result["required_provenance"])

    assert selective["failure_contract"] == {
        "empty_selection": "error-before-decode",
        "out_of_bounds": "error-before-decode",
        "conflicting_selectors": "error-before-source-load",
        "detector_region_without_scan_region": "error-before-source-load",
        "unsupported_backend": "error-without-fallback",
    }


def test_selective_scan_loading_support_matches_retained_sources() -> None:
    selective = next(
        item
        for item in _manifest()["capabilities"]
        if item["id"] == "io.selective-scan-loading"
    )
    coverage = selective["coverage"]

    assert {backend: entry["level"] for backend, entry in coverage.items()} == {
        "cpu-reference": "not-implemented",
        "cuda": "required",
        "mps": "required",
        "swift-metal": "not-implemented",
        "webgpu": "partial-hardware",
    }
    complete_selectors = ["scan_region", "scan_indices", "random_positions"]
    assert coverage["cpu-reference"]["implemented_selectors"] == []
    assert coverage["cuda"]["implemented_selectors"] == complete_selectors
    assert coverage["mps"]["implemented_selectors"] == complete_selectors
    assert coverage["swift-metal"]["implemented_selectors"] == []
    assert coverage["webgpu"]["implemented_selectors"] == ["scan_region"]
    assert "single-position-indexed-accessor" in coverage["swift-metal"][
        "implemented_subset"
    ]
    assert "no-ordered-duplicate-index-loader" in coverage["webgpu"][
        "limitations"
    ]
