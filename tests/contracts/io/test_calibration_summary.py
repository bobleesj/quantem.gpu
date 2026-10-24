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


def _disk_acquisition(metadata: dict) -> Dataset4dstemGPU:
    rows, cols = np.mgrid[:48, :48]
    disk = np.hypot(rows - 23.0, cols - 25.0) <= 12.0
    counts = (np.random.default_rng(1).poisson(1, (4, 5, 48, 48)) + 60 * disk).astype(np.uint16)
    return Dataset4dstemGPU(counts, metadata)


def test_fit_detector_sampling_records_the_disk_and_its_detector_step():
    """The semiangle sits at the disk edge: the fit records the step, the centre and the edge with their source."""
    from quantem.gpu import detector

    data = _disk_acquisition({"voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": [0.5, 0.5]})
    fit = data.calibration.fit_detector_sampling()
    expected = detector.fit_bright_field_disk(detector.mean(data), 30)
    assert fit.bright_field_center_px == expected.bright_field_center_px == (23.0, 25.0)
    assert fit.detector_sampling_mrad == expected.detector_sampling_mrad
    assert data.calibration.detector_sampling_mrad == fit.detector_sampling_mrad
    assert data.calibration.bright_field_center_px == fit.bright_field_center_px
    assert data.calibration.bright_field_edge_radius_px == fit.bright_field_edge_radius_px
    assert data.sampling[2:] == fit.detector_sampling_mrad
    table = data.calibration.table()
    assert table.loc["Detector step (row, col)", "Source"] == "Fitted (bright-field disk)"
    assert table.loc["Bright-field centre (row, col)", "Value"] == (23.0, 25.0)
    assert table.loc["Bright-field edge radius", "Unit"] == "px"
    override = data.metadata["scientific_metadata"]["calibration_overrides"][metadata.DETECTOR_PIXEL_SIZE[0]]
    assert override["provenance"] == metadata.BRIGHT_FIELD_DISK_FIT and "30 mrad" in override["evidence"]
    # the record is what io.save keeps, and a record of another schema is no record
    assert metadata.retained_metadata(data.metadata)["bright_field_disk"]["schema"] == metadata.BRIGHT_FIELD_DISK_SCHEMA
    assert "bright_field_disk" not in metadata.retained_metadata({"bright_field_disk": {"schema": "other/1"}})


def test_a_detector_step_set_by_hand_wins_over_the_fit(capsys):
    data = _disk_acquisition({"voltage_kV": 300, "semiangle_mrad": 30})
    data.calibration.detector_sampling_mrad = 0.7
    fit = data.calibration.fit_detector_sampling()
    assert data.calibration.detector_sampling_mrad == (0.7, 0.7)
    assert data.calibration.table().loc["Detector step (row, col)", "Source"] == "Overridden"
    assert data.calibration.bright_field_center_px == fit.bright_field_center_px
    assert "kept the detector sampling set by hand" in capsys.readouterr().out
    # and a value set by hand after the fit replaces the fitted one
    other = _disk_acquisition({"voltage_kV": 300, "semiangle_mrad": 30})
    other.calibration.fit_detector_sampling()
    other.calibration.detector_sampling_mrad = 0.7
    assert other.calibration.table().loc["Detector step (row, col)", "Source"] == "Overridden"


def test_fit_detector_sampling_needs_the_semiangle_and_its_disk_is_not_set_by_hand():
    data = _disk_acquisition({"voltage_kV": 300})
    with pytest.raises(ValueError, match="semiangle_mrad"):
        data.calibration.fit_detector_sampling()
    assert data.calibration.bright_field_center_px is None
    with pytest.raises(AttributeError, match="bright_field_center_px= to DirectPtycho"):
        data.calibration.bright_field_center_px = (20, 20)
    with pytest.raises(AttributeError, match="bright_field_radius_px= to DirectPtycho"):
        data.calibration.bright_field_edge_radius_px = 10
    # a fitted provenance belongs to the detector step alone
    with pytest.raises(ValueError, match="calibration override"):
        data._record_calibration(voltage_kV=200, provenance=metadata.BRIGHT_FIELD_DISK_FIT, evidence="fit")
