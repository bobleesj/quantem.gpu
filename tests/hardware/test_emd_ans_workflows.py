"""Preserve calibrated EMD measurements while browsing the compressed resident."""

import os

import h5py
import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io.qem_validation import validate_qem


@pytest.mark.parametrize(
    "metadata_location",
    ["attributes", "array-attributes", "scalar-datasets", "unknown-units"],
)
def test_emd_calibration_and_multiple_acquisitions(tmp_path, metadata_location):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"mps", "cuda"}:
        pytest.skip("Set QEM_TEST_BACKEND=mps or cuda on a physical accelerator.")
    original, saved = tmp_path / "series.emd", tmp_path / "selected.qem"
    values = (np.arange(3 * 5 * 24 * 32).reshape(3, 5, 24, 32) % 101).astype(
        np.float32
    ) / 8
    with h5py.File(original, "w") as handle:
        handle.attrs["version_major"] = 1
        group = handle.create_group("experiment/acquisition")
        group.attrs["emd_group_type"] = "array"
        group.create_dataset("data", data=values, compression="gzip")
        for number, (name, unit, spacing) in enumerate(
            [
                ("scan_row", "nm", 0.2),
                ("scan_column", "nm", 0.3),
                ("detector_row", "1/nm", 0.4),
                ("detector_column", "1/nm", 0.5),
            ],
            1,
        ):
            dim = group.create_dataset(f"dim{number}", data=[0, spacing])
            if metadata_location == "scalar-datasets":
                group[f"dim{number}_name"] = name
                group[f"dim{number}_units"] = unit
            else:
                dim.attrs["name"] = name
                dim.attrs["units"] = (
                    "unknown" if metadata_location == "unknown-units" else unit
                )
                if metadata_location == "array-attributes":
                    dim.attrs["units"] = np.array([unit.encode()])
        handle["experiment/other/data"] = np.zeros((1, 2, 3, 4), np.uint8)
    with pytest.raises(ValueError, match="dataset_path"):
        io.inspect(original)
    selected = "experiment/acquisition/data"
    info = io.inspect(original, dataset_path=selected)
    assert info.ready and info.scan_shape == (3, 5)
    with io.load(
        original, dataset_path=selected, backend=backend, verbose=False
    ) as loaded:
        assert loaded.representation.value == "encoded"
        np.testing.assert_array_equal(
            detector.prepare(loaded).frame(14), values[-1, -1]
        )
        io.save(saved, loaded)
    assert validate_qem(saved)["integrity"] == "verified"
    with io.load(saved, backend=backend, verbose=False) as loaded:
        metadata = loaded.metadata
        source = metadata["scientific_metadata"]["source_metadata"]
        assert source["/experiment/acquisition/dim1"] == [0, 0.2]
        if metadata_location == "unknown-units":
            assert "scan_sampling_A" not in metadata
            assert "detector_sampling" not in metadata
            assert source["/experiment/acquisition/dim1@units"] == "unknown"
        else:
            np.testing.assert_allclose(metadata["scan_sampling_A"], [2, 3])
            np.testing.assert_allclose(metadata["detector_sampling"], [0.04, 0.05])
        np.testing.assert_array_equal(detector.prepare(loaded).frame(0), values[0, 0])
