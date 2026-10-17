"""Calibration edits retain recorded evidence, units and explicit unknowns."""

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
    assert data.calibration.loc["Voltage", "Source"] == "Recorded"
    assert data.calibration.loc["Convergence semiangle", "Source"] == "Missing"
    data.set_calibration(
        semiangle_mrad=30,
        detector_sampling_inv_A=(0.02, 0.03),
        evidence="Independent calibration",
    )
    assert "detector_sampling_mrad" not in data.metadata
    assert data.sampling == (0.4, 0.5, 0.02, 0.03)
    assert data.calibration.loc["Convergence semiangle", "Source"] == "Overridden"
    assert data.calibration.loc["Detector sampling (row, col)", "Unit"] == "1/angstrom"
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
        data.set_calibration(
            voltage_kV=200, scan_sampling_A=(-1, 1), evidence="Bad edit"
        )
    assert json.dumps(data.metadata, default=metadata.json_metadata) == json.dumps(before, default=metadata.json_metadata)
    data.set_calibration(voltage_kV=200, evidence="Operator setting")
    assert data.metadata["semiangle_mrad"] == 30
    assert (
        stored["electron_microscope"]["electron_source/accelerating_voltage"]["value"]
        == 300
    )
    assert "Overridden" in data._repr_html_()


def test_unknown_calibration_does_not_prevent_raw_inspection():
    values = np.ones((2, 3, 4, 5), np.uint16)
    data = Dataset4dstemGPU(values, {})
    assert set(data.calibration.Source) == {"Missing"}
    np.testing.assert_array_equal(data.read().numpy(), values)
    assert "Missing" in data._repr_html_()
