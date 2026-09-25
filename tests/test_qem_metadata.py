"""Portable microscope vocabulary and explicit unknown/calibration semantics."""

import copy

import pytest

from quantem.gpu.io._qem_metadata import acquisition_metadata, validate_header
from quantem.gpu.io._qem_metadata import effective_metadata, microscopy_metadata, SCHEMA
from quantem.gpu.io.representation import DataRepresentation


@pytest.mark.parametrize("value", [True, "300", -1, float("nan"), float("inf")])
def test_microscopy_units_reject_invalid_quantities(value):
    with pytest.raises(ValueError, match="Invalid QEM quantity"):
        microscopy_metadata({
            "schema": SCHEMA,
            "electron_microscope": {
                "electron_source/accelerating_voltage": {"value": value, "unit": "kV"},
            },
        })


def test_ncem_quantities_and_axis_conventions():
    """Recorded NCEM units map to physical quantities without guessing angles."""
    source = {
        "electron_microscope/electron_source/accelerating_voltage": "300000 V",
        "electron_microscope/illumination_system/semi_convergence_angle": "0 mrad",
        "electron_microscope/scan_controller/regular_scan/dwell_time": "50 us",
        "electron_microscope/imaging_system/camera_length": "230 mm",
        "vendor/private_tag": "unchanged",
    }
    metadata = acquisition_metadata(
        (3, 5, 128, 128),
        dict(
            source_metadata=source,
            scan_sampling_A=[2, 3],
            detector_sampling=[0.2, 0.3],
            detector_sampling_unit="mrad",
        ),
    )
    quantities = metadata["electron_microscope"]
    assert quantities["electron_source/accelerating_voltage"]["value"] == 300
    assert quantities["electron_source/accelerating_voltage"]["unit"] == "kV"
    assert "illumination_system/semi_convergence_angle" not in quantities
    assert quantities["scan_controller/regular_scan/dwell_time"][
        "value"
    ] == pytest.approx(50)
    assert quantities["scan_controller/regular_scan/dwell_time"]["unit"] == "us"
    assert quantities["imaging_system/camera_length"]["value"] == pytest.approx(230)
    assert quantities["imaging_system/camera_length"]["unit"] == "mm"
    assert metadata["axes"][0]["sampling"]["value"] == pytest.approx(2)
    assert metadata["axes"][1]["sampling"]["value"] == pytest.approx(3)
    assert metadata["axes"][0]["sampling"]["unit"] == "angstrom"
    assert metadata["axes"][2]["sampling"]["unit"] == "mrad"
    assert metadata["source_metadata"] == source
    assert metadata["source_metadata_coverage"] == "reader-retained"


@pytest.mark.parametrize("value", ["0 V", "nan V", "inf V", "300 unknown", "300"])
def test_invalid_recorded_quantities_remain_unknown(value):
    result = acquisition_metadata(
        (1, 1, 128, 128),
        dict(
            source_metadata={
                "electron_microscope/electron_source/accelerating_voltage": value
            }
        ),
    )
    assert "electron_source/accelerating_voltage" not in result["electron_microscope"]


def test_arina_incident_energy_maps_to_same_vocabulary():
    result = acquisition_metadata(
        (1, 1, 128, 128),
        dict(
            source_metadata={
                "entry/instrument/detector/incident_energy": "300",
                "entry/instrument/detector/incident_energy@units": "keV",
            }
        ),
    )
    assert (
        result["electron_microscope"]["electron_source/accelerating_voltage"]["value"]
        == 300
    )


def test_version_geometry_and_malformed_metadata_rejected():
    shape = [2, 3, 128, 128]
    header = dict(
        container="quantem.qem",
        container_version=1,
        codec="test-codec",
        profile="test-codec",
        shape=shape,
        scientific_metadata=acquisition_metadata(shape, {}),
    )
    validate_header(header)
    for change in (
        {"container_version": 2},
        {"shape": [3, 2, 128, 128]},
        {"codec": "different"},
        {"scientific_metadata": None},
        {"scientific_metadata": {"axes": [None]}},
    ):
        invalid = copy.deepcopy(header)
        invalid.update(change)
        with pytest.raises(ValueError):
            validate_header(invalid)


def test_representation_detected_by_signature_not_extension(tmp_path):
    path = tmp_path / "renamed.data"
    path.write_bytes(b"QEMDATA1")
    assert DataRepresentation.detect_source(path) == DataRepresentation.ENCODED


def test_unsupported_precision_never_writes_another_format_as_qem(tmp_path):
    """An unsupported export must not create mislabeled HDF5 data."""
    import numpy as np
    from quantem.gpu import io

    destination = tmp_path / "precision.qem"
    with pytest.raises(NotImplementedError, match="QEM precision codecs"):
        io.save(destination, np.ones((1, 1, 2, 2), np.float32), dtype="scaled_uint16")
    assert not destination.exists()


def test_saved_user_calibration_restores_without_replacing_recorded_values():
    shape = [2, 3, 128, 128]
    original = dict(scan_sampling_A=[1, 2], voltage_kV=300)
    scientific = acquisition_metadata(shape, original)
    scientific["calibration_overrides"] = {
        path: dict(value=value, unit=unit, provenance="user_override", evidence="measured standard")
        for path, value, unit in (
            ("scan_controller/regular_scan/pixel_size_row", 0.4, "angstrom"),
            ("scan_controller/regular_scan/pixel_size_column", 0.6, "angstrom"),
            ("electron_source/accelerating_voltage", 200, "kV"),
        )
    }
    restored = effective_metadata(original, scientific)
    assert restored["scan_sampling_A"] == pytest.approx([0.4, 0.6])
    assert restored["voltage_kV"] == 200
    assert original == dict(scan_sampling_A=[1, 2], voltage_kV=300)
    restored["scientific_metadata"] = scientific
    exported = acquisition_metadata(shape, restored)
    assert exported == scientific
    assert exported["axes"][0]["sampling"]["value"] == 1
    invalid = copy.deepcopy(scientific)
    invalid["calibration_overrides"]["electron_source/accelerating_voltage"]["unit"] = "V"
    with pytest.raises(ValueError, match="calibration override"):
        effective_metadata(original, invalid)


def test_existing_reference_migrates_units_without_changing_source_metadata():
    """An existing file keeps its physical calibration when saved with readable units."""
    import json
    from pathlib import Path

    manifest = json.loads((Path(__file__).parent / "data/qem-v1/manifest.json").read_text())
    original = manifest["entries"][1]["scientific_metadata"]
    before = copy.deepcopy(original)
    normalized = microscopy_metadata(original)
    assert normalized["schema"] == SCHEMA
    assert normalized["source_metadata"] == original["source_metadata"]
    override = normalized["calibration_overrides"]
    scan = override["scan_controller/regular_scan/pixel_size_row"]
    assert scan["unit"] == "angstrom"
    assert scan["value"] == pytest.approx(0.4)
    assert override["imaging_system/reciprocal_pixel_size_row"]["unit"] == "1/angstrom"
    upgraded = effective_metadata({}, normalized)
    for key, value in effective_metadata({}, original).items():
        assert upgraded[key] == value
    assert microscopy_metadata(normalized) == normalized
    assert original == before


def test_diffraction_units_keep_angular_and_reciprocal_sampling_distinct():
    """Equivalent reciprocal units normalize, while angular sampling stays angular."""
    for source_unit, expected_unit, expected in (
        ("1/nm", "1/angstrom", 0.02),
        ("1/Å", "1/angstrom", 0.2),
        ("rad", "mrad", 200),
    ):
        metadata = acquisition_metadata((2, 3, 16, 16), {
            "detector_sampling": [0.2, 0.3], "detector_sampling_unit": source_unit,
        })
        sampling = metadata["axes"][2]["sampling"]
        assert sampling["unit"] == expected_unit
        assert sampling["value"] == pytest.approx(expected, rel=1e-14)


def test_public_calibration_is_authoritative_for_independent_writers():
    scientific = acquisition_metadata((2, 3, 16, 16), {
        "scan_sampling_A": [1, 2], "voltage_kV": 300,
    })
    for private in ({}, {"scan_sampling_A": [8, 9], "voltage_kV": 80}):
        restored = effective_metadata(private, scientific)
        assert restored["scan_sampling_A"] == [1, 2]
        assert restored["voltage_kV"] == 300
    header = dict(container="quantem.qem", container_version=1,
                  codec="example", profile="example", shape=[2, 3, 16, 16],
                  scientific_metadata=scientific)
    validate_header(header)
    for change in ("conflict", "provenance", "coverage"):
        invalid = copy.deepcopy(header)
        record = invalid["scientific_metadata"]
        if change == "conflict":
            record["electron_microscope"]["scan_controller/regular_scan/pixel_size_row"]["value"] = 99
        elif change == "provenance":
            del record["axes"][0]["sampling"]["provenance"]
        else:
            del record["source_metadata_coverage"]
        with pytest.raises(ValueError):
            validate_header(invalid)


# --- sample (specification 0.0.3) -------------------------------------------------------------------------------------

SAMPLE = {
    "provenance": "dataset.yaml",
    "evidence": "dataset.yaml sha256:0123, files[38]",
    "id": "BTO-STO-01",
    "name": "BaTiO3 film on SrTiO3",
    "geometry": "cross-section",
    "growth_direction": [0, 0, 1],
    "orientation_relationship": "(001)[100] BTO || (001)[100] STO",
    "components": {
        "BTO": {
            "role": "film",
            "chemical_formula": "BaTiO3",
            "zone_axis": [0, 0, 1],
            "cif": {"document": "BaTiO3.cif.json", "sha256": "ab" * 32},
            "thickness_estimates": [
                {"method": "ptychography_multislice", "value": 410.0, "unit": "angstrom", "uncertainty": 40.0,
                 "region": {"rows": [320, 512], "cols": [128, 384]}, "reference": "trials/000", "date": "2026-09-26", "preferred": True},
                {"method": "diffraction_ridge", "value": 430.0, "unit": "angstrom", "range": [390.0, 450.0], "region": "BTO"},
            ],
        },
        "STO": {"role": "substrate", "chemical_formula": "SrTiO3", "zone_axis": [0, 0, 1]},
    },
    "components_in_view": ["BTO", "STO"],
}


def _header_with(sample):
    shape = [2, 3, 16, 16]
    return dict(container="quantem.qem", container_version=1, codec="c", profile="c", shape=shape,
                scientific_metadata=acquisition_metadata(shape, {"sample": sample}))


def test_declared_sample_is_written_and_validated_unchanged():
    """A declared specimen passes through a new copy as written; with none, no sample group appears."""
    header = _header_with(copy.deepcopy(SAMPLE))
    validate_header(header)
    assert header["scientific_metadata"]["sample"] == SAMPLE
    assert "sample" not in acquisition_metadata([2, 3, 16, 16], {})


@pytest.mark.parametrize("path, value", [
    (("provenance",), ""),                                                        # every declared group names its origin
    (("geometry",), "side view"),
    (("components", "BTO", "role"), "layer"),
    (("components", "BTO", "zone_axis"), [0, 0, 0]),
    (("components", "BTO", "zone_axis"), "[001]"),
    (("components", "BTO", "thickness_estimates", 0, "method"), "guess"),
    (("components", "BTO", "thickness_estimates", 0, "unit"), "nm"),           # one canonical length unit
    (("components", "BTO", "thickness_estimates", 0, "value"), -4.0),
    (("components", "BTO", "thickness_estimates", 1, "range"), [450.0, 390.0]),
    (("components", "BTO", "thickness_estimates", 0, "region"), {"rows": [512, 320], "cols": [0, 8]}),
    (("components", "BTO", "thickness_estimates", 1, "preferred"), True),       # at most one preferred estimate
    (("components_in_view",), ["BTO", "LAO"]),
])
def test_invalid_sample_rejected(path, value):
    sample = copy.deepcopy(SAMPLE)
    target = sample
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    header = _header_with(SAMPLE)
    header["scientific_metadata"]["sample"] = sample
    with pytest.raises(ValueError):
        validate_header(header)
