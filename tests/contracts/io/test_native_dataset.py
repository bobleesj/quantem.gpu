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
