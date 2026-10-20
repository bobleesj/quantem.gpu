"""Read acquisition calibration without materializing diffraction data."""

import pytest

from quantem.gpu.formats.qem.metadata import acquisition_metadata, effective_metadata
from quantem.gpu.io import Dataset4dstemGPU


def test_calibration_properties_follow_effective_qem_overrides():
    """A reopened QEM exposes the saved calibration used by calculations."""
    shape = (16, 24, 128, 128)
    recorded = {
        "working_shape": shape,
        "scan_sampling_A": [0.4, 0.6],
        "detector_sampling_inv_A": [0.02, 0.03],
        "origin": [1.5, -2.0, None, None],
    }
    scientific = acquisition_metadata(shape, recorded)
    scientific["calibration_overrides"] = {
        "scan_controller/regular_scan/pixel_size_row": {
            "value": 0.5, "unit": "angstrom", "provenance": "user_override",
            "evidence": "standard specimen",
        },
        "scan_controller/regular_scan/pixel_size_column": {
            "value": 0.75, "unit": "angstrom", "provenance": "user_override",
            "evidence": "standard specimen",
        },
    }
    # a loader states the shape it holds; the saved record does not carry it
    data = Dataset4dstemGPU(None, {**effective_metadata(recorded, scientific), "working_shape": shape})
    assert data.sampling == pytest.approx((0.5, 0.75, 0.02, 0.03))
    assert data.units == ("angstrom", "angstrom", "1/angstrom", "1/angstrom")
    assert data.origin == (1.5, -2.0, None, None)
    # Reading calibration works with metadata alone and retains no second copy.
    data.metadata["scan_sampling_A"] = [0.8, 0.9]
    assert data.sampling[:2] == (0.8, 0.9)
    assert scientific["axes"][0]["sampling"]["value"] == pytest.approx(0.4)


def test_partial_calibration_does_not_invent_detector_units_or_origin():
    """A calibrated scan can still have an uncalibrated detector."""
    data = Dataset4dstemGPU(None, {
        "working_shape": (16, 24, 128, 128), "scan_sampling_A": [0.4, 0.4],
    })
    assert data.sampling == (0.4, 0.4, None, None)
    assert data.units == ("angstrom", "angstrom", None, None)
    assert data.origin == (None, None, None, None)
    data.metadata["detector_sampling_mrad"] = [0.2, 0.3]
    assert data.sampling == (0.4, 0.4, 0.2, 0.3)
    assert data.units[2:] == ("mrad", "mrad")
    # the scan step is a [row, col] pair, never one number for both axes
    data.metadata["scan_sampling_A"] = 0.4
    with pytest.raises(ValueError, match=r"\[row, col\] pair"):
        data.sampling


def test_explicit_series_axes_keep_their_units_and_coordinates():
    """A series can describe time followed by scan and detector axes."""
    data = Dataset4dstemGPU(None, {
        "working_shape": (3, 16, 24, 128, 128),
        "sampling": [2.0, 0.4, 0.6, 0.02, 0.03],
        "units": ["s", "angstrom", "angstrom", "mrad", "mrad"],
        "origin": [10.0, 0.0, 0.0, None, None],
    })
    assert data.sampling == (2.0, 0.4, 0.6, 0.02, 0.03)
    assert data.units == ("s", "angstrom", "angstrom", "mrad", "mrad")
    assert data.origin == (10.0, 0.0, 0.0, None, None)


@pytest.mark.parametrize("unit,key,origin,expected_unit,expected_spacing,expected_origin", [
    ("rad", "detector_sampling_mrad", -0.128, "mrad", 2.0, -128.0),
    ("1/nm", "detector_sampling_inv_A", -3.0, "1/angstrom", 0.02, -0.3),
])
@pytest.mark.parametrize("scan_units", [None, "nm"])
def test_origin_coordinates_follow_normalized_units(
    unit, key, origin, expected_unit, expected_spacing, expected_origin, scan_units,
):
    """An origin recorded in another unit of the same kind keeps its signed physical coordinate."""
    shape = (16, 24, 128, 128)
    metadata = {
        "working_shape": shape, "scan_sampling_A": [0.4, 0.6],
        key: [expected_spacing, expected_spacing],
        "origin": [None, None, origin, origin],
        "units": [scan_units, scan_units, unit, unit],
    }
    scientific = acquisition_metadata(shape, metadata)
    data = Dataset4dstemGPU(None, {**effective_metadata(metadata, scientific), "working_shape": shape})
    assert data.sampling[2:] == pytest.approx((expected_spacing,) * 2)
    assert data.units[2:] == (expected_unit,) * 2
    assert data.origin[2:] == pytest.approx((expected_origin,) * 2)
    data.metadata["origin"][:2] = [1.0, -2.0]
    data.metadata["units"][:2] = ["nm", "nm"]
    assert data.origin[:2] == (10.0, -20.0)

    # A number under the retired unit-less key is refused, not written as a calibration.
    with pytest.raises(ValueError, match="detector_sampling_mrad or detector_sampling_inv_A"):
        acquisition_metadata(shape, {"working_shape": shape, "detector_sampling": [0.2, 0.2]})
    with pytest.raises(ValueError, match="not both"):
        acquisition_metadata(shape, {"detector_sampling_mrad": [2.0, 2.0], "detector_sampling_inv_A": [0.02, 0.02]})


@pytest.mark.parametrize("unit,spacing,key,origin,expected_unit,expected_spacing,expected_origin", [
    ("rad", 0.002, "detector_sampling_mrad", -0.128, "mrad", 2.0, -128.0),
    ("1/nm", 0.2, "detector_sampling_inv_A", -3.0, "1/angstrom", 0.02, -0.3),
])
@pytest.mark.parametrize("explicit_units", [None, ["nm", "nm", None, None]])
def test_origin_without_detector_units_is_scaled_as_the_recorded_step_was(
    unit, spacing, key, origin, expected_unit, expected_spacing, expected_origin, explicit_units,
):
    """A header that states the supplied detector unit beside the step keeps its origin's physical coordinate."""
    shape = (16, 24, 128, 128)
    # the retained block of a schema-2 file written with the unit-less pair: the axes hold the
    # step in the stored unit, the origin stays in the supplied one and names no unit of its own
    retained = {
        "scan_sampling_A": [0.4, 0.6], "detector_sampling": [spacing, spacing], "detector_sampling_unit": unit,
        "origin": [None, None, origin, origin], "units": explicit_units,
    }
    scientific = acquisition_metadata(shape, {"scan_sampling_A": [0.4, 0.6], key: [expected_spacing, expected_spacing]})
    data = Dataset4dstemGPU(None, {**effective_metadata(retained, scientific), "working_shape": shape})
    assert data.sampling[2:] == pytest.approx((expected_spacing,) * 2)
    assert data.units[2:] == (expected_unit,) * 2
    assert data.origin[2:] == pytest.approx((expected_origin,) * 2)
    data.metadata["origin"][:2] = [1.0, -2.0]
    data.metadata["units"][:2] = ["nm", "nm"]
    assert data.origin[:2] == (10.0, -20.0)

    # A saved number without a detector unit is not a physical calibration.
    unknown = {"detector_sampling": [spacing, spacing]}
    reopened = Dataset4dstemGPU(None, {**effective_metadata(unknown, acquisition_metadata(shape, {})), "working_shape": shape})
    assert reopened.sampling == (None,) * 4
    assert reopened.units == (None,) * 4


def test_name_is_the_recorded_name_else_the_source_file():
    """A loaded acquisition names itself as a quantem core dataset does, without decoding data."""
    shape = {"working_shape": (2, 2, 4, 4)}
    assert Dataset4dstemGPU(None, {**shape, "source_path": "/data/scan_001_master.h5"}).name == "scan_001"
    assert Dataset4dstemGPU(None, {**shape, "source_path": "/data/scan.npy"}).name == "scan"
    assert Dataset4dstemGPU(None, {**shape, "source_path": "/data/scan.npy", "name": "gold"}).name == "gold"
    assert Dataset4dstemGPU(None, shape).name == ""


def test_metadata_table_preserves_saved_overrides_and_missing_rotation():
    """Review calibration before reconstruction without reading detector counts."""
    import copy

    shape = (16, 24, 128, 128)
    recorded = {"scan_sampling_A": [0.4, 0.6], "voltage_kV": 200.0}
    scientific = acquisition_metadata(shape, recorded)
    scientific["calibration_overrides"] = {
        "electron_source/accelerating_voltage": {
            "value": 300.0, "unit": "kV", "provenance": "user_override",
            "evidence": "calibration measurement",
        },
    }
    metadata = {
        **effective_metadata(recorded, scientific), "scientific_metadata": scientific,
        "working_shape": shape, "dtype": "uint16", "source_path": "<script>scan</script>",
    }
    original = copy.deepcopy(metadata)
    # No array owner is needed to review stored metadata.
    data = Dataset4dstemGPU(None, metadata)
    table = data.metadata_table()
    assert table.loc["Accelerating voltage", "Value"] == 300.0
    assert table.loc["Accelerating voltage", "Metadata"] == "Overridden"
    assert table.loc["Detector step (row, col)", "Value"] == "Not recorded"
    assert table.loc["Scan-detector rotation", "Value"] == "Not recorded"
    assert "<script>" not in data._repr_html_()
    assert "Overridden" in repr(data)
    assert metadata == original
    metadata["scan_detector_rotation_deg"] = 0.0
    assert data.metadata_table().loc["Scan-detector rotation", "Value"] == 0.0
