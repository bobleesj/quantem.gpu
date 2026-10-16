"""An ARINA acquisition carries its paired NXem calibration through IO."""

import h5py
import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.formats.qem.metadata import acquisition_metadata, recorded_metadata


@pytest.fixture
def acquisition(tmp_path):
    master = tmp_path / "acquisition_master.h5"
    with h5py.File(tmp_path / "acquisition_data_000001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=np.arange(96, dtype=np.uint16).reshape(6, 4, 4))
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(
            "acquisition_data_000001.h5", "/entry/data/data"
        )
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = 6
    with h5py.File(tmp_path / "acquisition_em_metadata.h5", "w") as handle:
        for path, value, unit in (
            ("scan_controller/regular_scan/n_pixels_y", 2, None),
            ("scan_controller/regular_scan/n_pixels_x", 3, None),
            ("scan_controller/regular_scan/pixel_size_y", 4e-10, "m"),
            ("scan_controller/regular_scan/pixel_size_x", .5, "nm"),
            ("imaging_system/reciprocal_pixel_size_y", .0002, "rad"),
            ("imaging_system/reciprocal_pixel_size_x", .3, "mrad"),
            ("electron_source/accelerating_voltage", 300000, "V"),
            ("illumination_system/semi_convergence_angle", 0, "mrad"),
            ("imaging_system/camera_length", .23, "m"),
            ("scan_controller/rotation", 18, "degrees"),
        ):
            dataset = handle.create_dataset("electron_microscope/" + path, data=value)
            if unit:
                dataset.attrs["units"] = unit
    return master


def test_load_and_saved_metadata_keep_axis_units(acquisition):
    info = io.inspect(acquisition)
    assert info.ready and info.scan_shape == (2, 3)
    data = io.load(acquisition, backend="cpu", representation="dense", apply_mask=False)
    assert data.sampling == (4, 5, .2, .3)
    assert data.units == ("angstrom", "angstrom", "mrad", "mrad")
    assert data.metadata["voltage_kV"] == 300
    assert data.metadata["camera_length_mm"] == 230
    assert "semiangle_mrad" not in data.metadata
    assert "rotation_deg" not in data.metadata
    saved = recorded_metadata(acquisition_metadata(data.shape, data.metadata))
    assert saved["scan_sampling_A"] == [4, 5]
    assert saved["detector_sampling_mrad"] == [.2, .3]
    assert saved["voltage_kV"] == 300


def test_updated_companion_changes_calibration_and_cache_identity(acquisition):
    before = io.inspect(acquisition).source_signature
    companion = acquisition.with_name("acquisition_em_metadata.h5")
    with h5py.File(companion, "a") as handle:
        handle["electron_microscope/scan_controller/regular_scan/pixel_size_y"][()] = 8e-10
        handle["electron_microscope/illumination_system/semi_convergence_angle"][()] = 20
    info = io.inspect(acquisition)
    assert info.source_signature != before
    assert info.metadata["scan_sampling_A"] == [8, 5]
    assert info.metadata["semiangle_mrad"] == 20


def test_wrong_raster_or_missing_units_do_not_supply_calibration(acquisition):
    companion = acquisition.with_name("acquisition_em_metadata.h5")
    with h5py.File(companion, "a") as handle:
        handle["electron_microscope/scan_controller/regular_scan/n_pixels_y"][()] = 3
    info = io.inspect(acquisition)
    assert "scan_sampling_A" not in info.metadata
    assert "detector_sampling_mrad" not in info.metadata
    assert "raster" in info.metadata["microscope_metadata_warning"]
    saved = recorded_metadata(acquisition_metadata((2, 3, 4, 4), info.metadata))
    assert "detector_sampling_mrad" not in saved
    with h5py.File(companion, "a") as handle:
        handle["electron_microscope/scan_controller/regular_scan/n_pixels_y"][()] = 2
        del handle["electron_microscope/scan_controller/regular_scan/pixel_size_y"].attrs["units"]
    assert "scan_sampling_A" not in io.inspect(acquisition).metadata
