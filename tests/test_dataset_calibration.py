"""Read acquisition calibration without materializing diffraction data."""

import pytest

from quantem.gpu.io import Dataset4dstemGPU
from quantem.gpu.io._qem_metadata import acquisition_metadata, effective_metadata


def test_calibration_properties_follow_effective_qem_overrides():
    """A reopened QEM exposes the saved calibration used by calculations."""
    shape = (16, 24, 128, 128)
    recorded = {
        "working_shape": shape,
        "scan_sampling_A": [0.4, 0.6],
        "detector_sampling": [0.02, 0.03],
        "detector_sampling_unit": "1/angstrom",
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
    data = Dataset4dstemGPU(None, effective_metadata(recorded, scientific))
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
        "working_shape": (16, 24, 128, 128), "scan_sampling_A": 0.4,
    })
    assert data.sampling == (0.4, 0.4, None, None)
    assert data.units == ("angstrom", "angstrom", None, None)
    assert data.origin == (None, None, None, None)
    data.metadata["detector_sampling"] = [0.2, 0.3]
    assert data.units[2:] == (None, None)
    data.metadata["detector_sampling_unit"] = "mrad"
    assert data.sampling == (0.4, 0.4, 0.2, 0.3)
    assert data.units[2:] == ("mrad", "mrad")


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


@pytest.mark.parametrize("unit,spacing,origin,expected_unit,expected_spacing,expected_origin", [
    ("rad", 0.002, -0.128, "mrad", 2.0, -128.0),
    ("1/nm", 0.2, -3.0, "1/angstrom", 0.02, -0.3),
])
@pytest.mark.parametrize("explicit_units", [None, ["nm", "nm", None, None]])
def test_origin_coordinates_follow_normalized_units(
    unit, spacing, origin, expected_unit, expected_spacing, expected_origin, explicit_units,
):
    """QEM unit normalization preserves signed physical coordinates."""
    shape = (16, 24, 128, 128)
    metadata = {
        "working_shape": shape, "scan_sampling_A": [0.4, 0.6],
        "detector_sampling": [spacing, spacing], "detector_sampling_unit": unit,
        "origin": [None, None, origin, origin],
        "units": explicit_units,
    }
    scientific = acquisition_metadata(shape, metadata)
    data = Dataset4dstemGPU(None, effective_metadata(metadata, scientific))
    assert data.sampling[2:] == pytest.approx((expected_spacing,) * 2)
    assert data.units[2:] == (expected_unit,) * 2
    assert data.origin[2:] == pytest.approx((expected_origin,) * 2)
    data.metadata["origin"][:2] = [1.0, -2.0]
    data.metadata["units"][:2] = ["nm", "nm"]
    assert data.origin[:2] == (10.0, -20.0)

    # A saved number without a detector unit is not a physical calibration.
    unknown = {"working_shape": shape, "detector_sampling": [spacing, spacing]}
    scientific = acquisition_metadata(shape, unknown)
    reopened = Dataset4dstemGPU(None, effective_metadata(unknown, scientific))
    assert reopened.sampling == (None,) * 4
    assert reopened.units == (None,) * 4
