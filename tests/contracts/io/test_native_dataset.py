"""Native calibration follows selected measurements through QEM export."""

import numpy as np
import pytest
from quantem.core.datastructures import Dataset2d, Dataset4dstem

from quantem.gpu import io


def test_native_selection_roundtrip_keeps_values_and_calibration(tmp_path):
    values = np.arange(3 * 4 * 8 * 8, dtype=np.uint16).reshape(3, 4, 8, 8)
    native = Dataset4dstem.from_array(
        values,
        sampling=(0.5, 0.6, 0.01, 0.02),
        origin=(1, 2, 3, 4),
        units=("nm", "nm", "1/nm", "1/nm"),
        signal_units="electrons",
    )
    native.metadata["experiment"] = "synthetic"
    full_path = tmp_path / "full.qem"
    io.save(full_path, native, backend="cpu")
    with io.load(
        full_path, backend="cpu", representation="dense", verbose=False
    ) as loaded:
        assert type(loaded) is Dataset4dstem
        np.testing.assert_array_equal(loaded.array, values)
        np.testing.assert_array_equal(loaded.sampling, native.sampling)
        assert loaded.units == native.units
        pattern = loaded[1, 2]
        assert type(pattern) is Dataset2d
        np.testing.assert_array_equal(pattern.sampling, (0.01, 0.02))
        crop = loaded[1:, 2:, 2:6, 2:6]
        crop_path = tmp_path / "crop.qem"
        io.save(crop_path, crop, backend="cpu")
    with io.load(
        crop_path, backend="cpu", representation="dense", verbose=False
    ) as reopened:
        np.testing.assert_array_equal(reopened.array, values[1:, 2:, 2:6, 2:6])
        np.testing.assert_allclose(reopened.origin, (1.5, 3.2, 3.02, 4.04))
        assert reopened.signal_units == "electrons"
        assert reopened.metadata["experiment"] == "synthetic"
        assert [
            axis["size"] for axis in reopened.metadata["scientific_metadata"]["axes"]
        ] == [2, 2, 4, 4]


@pytest.mark.parametrize("direction", [1, -1])
def test_native_metadata_export_uses_current_calibration(tmp_path, direction):
    native = Dataset4dstem.from_array(np.ones((2, 2, 4, 4), dtype=np.uint16))
    native.sampling = (direction * 0.5, 0.6, direction * 0.02, 0.03)
    native.units = ("angstrom", "angstrom", "1/angstrom", "1/angstrom")
    path = tmp_path / "calibrated.qem"
    io.save(path, native, backend="cpu")
    with io.load(path, backend="cpu", representation="dense", verbose=False) as loaded:
        np.testing.assert_array_equal(loaded.sampling, native.sampling)
        assert loaded.metadata["scan_sampling_A"] == pytest.approx([0.5, 0.6])
        assert loaded.metadata["detector_sampling"] == pytest.approx([0.02, 0.03])
        assert loaded.metadata["detector_sampling_unit"] == "1/angstrom"


@pytest.mark.parametrize("step", [1, 2])
def test_calibrated_selection_roundtrip_supersedes_stale_overrides(tmp_path, step):
    from copy import deepcopy

    from quantem.gpu.io._qem_metadata import acquisition_metadata

    values = np.arange(4 * 3 * 8 * 8, dtype=np.uint16).reshape(4, 3, 8, 8)
    scientific = acquisition_metadata(
        values.shape,
        {
            "scan_sampling_A": [1, 2],
            "detector_sampling": [0.01, 0.02],
            "detector_sampling_unit": "1/angstrom",
            "voltage_kV": 300,
            "source_metadata": {"acquisition": "calibrated reference"},
        },
    )
    scientific["calibration_overrides"] = {
        path: {
            "value": value,
            "unit": unit,
            "provenance": "user_override",
            "evidence": "measured standard",
        }
        for path, value, unit in (
            ("scan_controller/regular_scan/pixel_size_row", 0.4, "angstrom"),
            ("scan_controller/regular_scan/pixel_size_column", 0.6, "angstrom"),
            ("imaging_system/reciprocal_pixel_size_row", 0.02, "1/angstrom"),
            ("imaging_system/reciprocal_pixel_size_column", 0.03, "1/angstrom"),
            ("electron_source/accelerating_voltage", 200, "kV"),
        )
    }
    original = deepcopy(scientific)
    source_path = tmp_path / "calibrated.qem"
    selected_path = tmp_path / "selected.qem"
    io.save(
        source_path,
        values,
        metadata={"scientific_metadata": scientific},
        backend="cpu",
    )
    with io.load(
        source_path, backend="cpu", representation="dense", verbose=False
    ) as loaded:
        selected = loaded[::step, :, ::step, :]
        io.save(selected_path, selected, backend="cpu")
        assert loaded.metadata["scientific_metadata"] == original
    with io.load(
        selected_path, backend="cpu", representation="dense", verbose=False
    ) as reopened:
        np.testing.assert_array_equal(reopened.array, values[::step, :, ::step, :])
        np.testing.assert_allclose(
            reopened.sampling, [0.4 * step, 0.6, 0.02 * step, 0.03]
        )
        assert reopened.metadata["scan_sampling_A"] == pytest.approx([0.4 * step, 0.6])
        assert reopened.metadata["detector_sampling"] == pytest.approx([0.02 * step, 0.03])
        assert reopened.metadata["detector_sampling_unit"] in {"1/angstrom", "1/Å"}
        saved = reopened.metadata["scientific_metadata"]
        assert saved["source_metadata"] == original["source_metadata"]
        expected_overrides = original["calibration_overrides"]
        if step != 1:
            expected_overrides = {
                "electron_source/accelerating_voltage": expected_overrides[
                    "electron_source/accelerating_voltage"
                ]
            }
        assert saved["calibration_overrides"] == expected_overrides
        assert reopened.metadata["voltage_kV"] == 200
