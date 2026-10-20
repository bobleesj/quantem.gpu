"""Calibration set by name keeps the recorded values, units and explicit unknowns."""

import copy
import json

import numpy as np
import pytest

from quantem.gpu.formats.qem import metadata
from quantem.gpu.io.dataset import Dataset4dstemGPU


def test_calibration_summary_and_atomic_override():
    data = Dataset4dstemGPU(
        np.zeros((2, 3, 4, 5), np.uint32),
        {
            "voltage_kV": 300,
            "semiangle_mrad": 0,
            "scan_sampling_A": [0.4, 0.5],
            "detector_sampling_mrad": [0.2, 0.3],
            "source_metadata": {"detector/pixel_flags": np.asarray([0, 1, 2], np.uint32)},
        },
    )
    assert data.calibration.table().loc["Accelerating voltage", "Source"] == "Recorded"
    assert data.calibration.table().loc["Convergence semiangle", "Source"] == "Not recorded"
    data.calibration.semiangle_mrad = 30
    data.calibration.detector_sampling_inv_A = (0.02, 0.03)
    assert "detector_sampling_mrad" not in data.metadata
    assert data.sampling == (0.4, 0.5, 0.02, 0.03)
    assert data.calibration.table().loc["Convergence semiangle", "Source"] == "Overridden"
    assert data.calibration.table().loc["Detector step (row, col)", "Unit"] == "1/angstrom per pixel"
    stored = metadata.acquisition_metadata(data.shape, data.metadata)
    assert stored["source_metadata"]["detector/pixel_flags"] == [0, 1, 2]
    assert stored["axes"][2]["sampling"]["value"] == 0.2
    assert stored["axes"][2]["sampling"]["unit"] == "mrad"
    assert (
        stored["calibration_overrides"][metadata.DETECTOR_PIXEL_SIZE[0]]["value"]
        == 0.02
    )
    before = copy.deepcopy(data.metadata)
    with pytest.raises(ValueError):
        data.calibration.scan_sampling_A = (-1, 1)
    assert json.dumps(data.metadata, default=metadata.json_metadata) == json.dumps(before, default=metadata.json_metadata)
    data.calibration.voltage_kV = 200
    assert data.calibration.voltage_kV == 200
    assert data.metadata["semiangle_mrad"] == 30
    assert (
        stored["electron_microscope"]["electron_source/accelerating_voltage"]["value"]
        == 300
    )
    assert "Overridden" in data._repr_html_()


def test_unknown_calibration_does_not_prevent_raw_inspection():
    values = np.ones((2, 3, 4, 5), np.uint16)
    data = Dataset4dstemGPU(values, {})
    assert set(data.calibration.table().Source) == {"Not recorded"}
    assert data.calibration.voltage_kV is None
    np.testing.assert_array_equal(data.read().numpy(), values)
    assert "Not recorded" in data._repr_html_()


def test_calibration_fields_are_named_and_a_scalar_sets_both_axes():
    data = Dataset4dstemGPU(np.zeros((2, 3, 4, 5), np.uint16), {})
    data.calibration.scan_sampling_A = 0.264
    assert data.calibration.scan_sampling_A == (0.264, 0.264)
    with pytest.raises(AttributeError, match="Did you mean 'voltage_kV'"):
        data.calibration.voltage_kv = 300
    with pytest.raises(AttributeError, match="no field"):
        data.calibration.rotation
    assert "Scan step (row, col)" in data.calibration._repr_html_()


def test_every_calibration_table_uses_one_label_set():
    """data.calibration.table(), data.metadata_table() and SSB.calibration_table() name rows and units alike."""
    from quantem.gpu.io import metadata_display as labels

    data = Dataset4dstemGPU(np.zeros((2, 3, 4, 5), np.uint16), {
        "voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": [0.264, 0.264],
        "detector_sampling_mrad": [0.55, 0.55]})
    calibration = data.calibration.table()
    overview = data.metadata_table()
    for name in ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad"):
        label, unit = labels.CALIBRATION_LABELS[name], labels.CALIBRATION_UNITS[name]
        assert calibration.loc[label, "Unit"] == overview.loc[label, "Unit"] == unit
        assert calibration.loc[label, "Source"] == overview.loc[label, "Metadata"] == labels.RECORDED
    assert calibration.loc["Scan step (row, col)", "Value"] == (0.264, 0.264)
    assert overview.loc["Scan-detector rotation", "Metadata"] == labels.NOT_RECORDED


def test_tables_round_for_reading_and_a_confirmed_value_stays_recorded():
    """Displayed values have six significant figures; setting the recorded value again is not an override."""
    from quantem.gpu.io import metadata_display as labels

    assert labels.display_value(0.6035446045631986) == 0.603545
    assert labels.display_value((0.6035446045631986, 2)) == (0.603545, 2)
    assert labels.display_value("Not recorded") == "Not recorded"
    data = Dataset4dstemGPU(np.zeros((2, 3, 4, 5), np.uint16), {
        "voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": [0.264, 0.264],
        "detector_sampling_mrad": [0.6035446045631986, 0.6035446045631986]})
    assert data.calibration.table().loc["Detector step (row, col)", "Value"] == (0.603545, 0.603545)
    assert data.calibration.detector_sampling_mrad == (0.6035446045631986, 0.6035446045631986)
    data.calibration.voltage_kV = 300.0
    data.calibration.semiangle_mrad = 25.0
    table = data.calibration.table()
    assert table.loc["Accelerating voltage", "Source"] == "Recorded"
    assert table.loc["Convergence semiangle", "Source"] == "Overridden"
