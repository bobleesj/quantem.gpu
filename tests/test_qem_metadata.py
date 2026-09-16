"""Portable microscope vocabulary and explicit unknown/calibration semantics."""

import copy

import pytest

from quantem.gpu.io._qem_metadata import acquisition_metadata, validate_header
from quantem.gpu.io._qem_metadata import effective_metadata
from quantem.gpu.io.representation import DataRepresentation


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
    assert quantities["electron_source/accelerating_voltage"]["value"] == 300000
    assert "illumination_system/semi_convergence_angle" not in quantities
    assert quantities["scan_controller/regular_scan/dwell_time"][
        "value"
    ] == pytest.approx(50e-6)
    assert quantities["imaging_system/camera_length"]["value"] == pytest.approx(0.23)
    assert metadata["axes"][0]["sampling"]["value"] == pytest.approx(2e-10)
    assert metadata["axes"][1]["sampling"]["value"] == pytest.approx(3e-10)
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
        == 300000
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
            ("scan_controller/regular_scan/pixel_size_y", 0.4e-10, "m"),
            ("scan_controller/regular_scan/pixel_size_x", 0.6e-10, "m"),
            ("electron_source/accelerating_voltage", 200000, "V"),
        )
    }
    restored = effective_metadata(original, scientific)
    assert restored["scan_sampling_A"] == pytest.approx([0.4, 0.6])
    assert restored["voltage_kV"] == 200
    assert original == dict(scan_sampling_A=[1, 2], voltage_kV=300)
    restored["scientific_metadata"] = scientific
    exported = acquisition_metadata(shape, restored)
    assert exported == scientific
    assert exported["axes"][0]["sampling"]["value"] == 1e-10
    invalid = copy.deepcopy(scientific)
    invalid["calibration_overrides"]["electron_source/accelerating_voltage"]["unit"] = "kV"
    with pytest.raises(ValueError, match="calibration override"):
        effective_metadata(original, invalid)
