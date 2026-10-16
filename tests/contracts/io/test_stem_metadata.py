"""Microscope calibration follows an acquisition through inspection and saving."""

import h5py
import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.formats.hdf5.master import get_metadata
from quantem.gpu.formats.qem.metadata import acquisition_metadata, recorded_metadata


@pytest.fixture
def acquisition(tmp_path):
    master = tmp_path / "acquisition_master.h5"
    with h5py.File(tmp_path / "acquisition_data_000001.h5", "w") as handle:
        data = handle.create_dataset(
            "entry/data/data", data=np.arange(96, dtype=np.uint16).reshape(6, 4, 4),
            chunks=(1, 4, 4),
        )
        data.attrs["scan_shape"] = [2, 3]
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(
            "acquisition_data_000001.h5", "/entry/data/data"
        )
        detector = handle.require_group("entry/instrument/detector")
        detector["description"] = "Dectris ARINA Si"
        detector["detectorSpecific/ntrigger"] = 6
        detector["detectorSpecific/photon_energy"] = 200000.0
    with h5py.File(tmp_path / "acquisition.h5", "w") as handle:
        attrs = handle.create_group("STEM Metadata").attrs
        for key, value in {
            "NumPixelsY": 2, "NumPixelsX": 3, "NumPixelsPrescan": 16,
            "Pixel Size": 0.05, "IsPixelSizeCalibrated": 1,
            "High Tension": 300000.0, "Camera Length": 0.091, "Rotation": 12.0,
        }.items():
            attrs[key] = [value]
    return master


def test_calibrated_acquisition_inspection_load_and_saved_metadata(acquisition):
    info = io.inspect(acquisition)
    assert info.ready
    assert info.scan_shape == (2, 3)
    assert info.metadata["voltage_kV"] == 300
    assert "disagrees" in info.metadata["microscope_metadata_warning"]
    data = io.load(acquisition, backend="cpu", representation="dense", apply_mask=False)
    assert data.sampling == (0.5, 0.5, None, None)
    assert data.units == ("angstrom", "angstrom", None, None)
    assert "rotation_deg" not in data.metadata
    record = acquisition_metadata(data.shape, data.metadata)
    saved = recorded_metadata(record)
    assert saved["scan_sampling_A"] == [0.5, 0.5]
    assert saved["voltage_kV"] == 300
    assert saved["camera_length_mm"] == 91
    assert record["source_metadata"]["STEM Metadata@Rotation"] == 12


def test_nominal_step_remains_visible_without_false_calibration(acquisition):
    before = io.inspect(acquisition).source_signature
    with h5py.File(acquisition.with_name("acquisition.h5"), "a") as handle:
        handle["STEM Metadata"].attrs["IsPixelSizeCalibrated"] = [0]
    assert io.inspect(acquisition).source_signature != before
    data = io.load(acquisition, backend="cpu", representation="dense", apply_mask=False)
    assert data.sampling == (None, None, None, None)
    assert data.metadata["scan_sampling_nominal_A"] == [0.5, 0.5]
    record = acquisition_metadata(data.shape, data.metadata)
    assert "scan_sampling_A" not in recorded_metadata(record)
    assert record["source_metadata"]["scan_sampling_calibrated"] is False
    assert "nominal" in record["source_metadata"]["microscope_metadata_warning"]


def test_wrong_companion_and_invalid_step_are_not_calibration(acquisition):
    companion = acquisition.with_name("acquisition.h5")
    with h5py.File(companion, "a") as handle:
        handle["STEM Metadata"].attrs["NumPixelsY"] = [3]
    metadata = get_metadata(str(acquisition))
    assert "scan_sampling_A" not in metadata
    assert "voltage_kV" not in metadata
    assert "frame count" in metadata["microscope_metadata_warning"]
    with h5py.File(companion, "a") as handle:
        handle["STEM Metadata"].attrs["NumPixelsY"] = [2]
        handle["STEM Metadata"].attrs["Pixel Size"] = [5e-324]
    assert "scan_sampling_A" not in get_metadata(str(acquisition))
    companion.rename(companion.with_name("other.h5"))
    assert "microscope_metadata_source" not in get_metadata(str(acquisition))
