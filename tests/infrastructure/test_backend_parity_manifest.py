import hashlib
import json
from pathlib import Path

MANIFEST = Path("tests/parity/backend_matrix.json")
EXPECTED_BACKENDS = {
    "cpu-reference",
    "cuda",
    "mps",
    "swift-metal",
    "webgpu",
    "vulkan",
}
EXPECTED_CAPABILITIES = {
    "io.packed-display-precision",
    "geometry.scan-quarter-turn",
    "io.decode-bin-provenance",
    "io.selective-scan-loading",
    "io.empad-float-resident",
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

    assert selective["contract_version"] == "quantem-gpu-selective-scan/v2"
    assert selective["public_owner"] == "quantem.gpu.io.Dataset4dstemGPU.read"
    selectors = selective["selectors"]
    assert set(selectors) == {"scan_region"}
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

    detector = selective["detector_region"]
    assert "requires_selector" not in detector
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
        "vulkan": "not-implemented",
        "cpu-reference": "not-implemented",
        "cuda": "required",
        "mps": "required",
        "swift-metal": "not-implemented",
        "webgpu": "partial-hardware",
    }
    assert coverage["cpu-reference"]["implemented_selectors"] == []
    assert coverage["cuda"]["implemented_selectors"] == ["scan_region"]
    assert coverage["mps"]["implemented_selectors"] == ["scan_region"]
    assert coverage["swift-metal"]["implemented_selectors"] == []
    assert coverage["webgpu"]["implemented_selectors"] == ["scan_region"]
    assert coverage["vulkan"]["implemented_selectors"] == []
    assert "single-position-indexed-accessor" in coverage["swift-metal"][
        "implemented_subset"
    ]
    assert "bounds-round-and-clamp-instead-of-strict-rejection" in coverage[
        "webgpu"
    ]["limitations"]


def test_backend_parity_manifest_freezes_shared_gold_fixtures() -> None:
    fixtures = _manifest()["gold_fixtures"]

    assert set(fixtures) == {"scan_rotation_v1"}
    for fixture in fixtures.values():
        path = Path(fixture["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"]
