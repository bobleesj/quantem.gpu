"""Open complete native geometry without guessing rectangular scan dimensions."""

import h5py
import json
import numpy as np
import pytest
from quantem.gpu import io


@pytest.mark.parametrize("shape", [(4, 9, 7, 11), (3, 5, 8, 10)])
def test_explicit_four_dimensions_override_square_scan_inference(tmp_path, shape):
    path = tmp_path / "native.h5"
    with h5py.File(path, "w") as source:
        source["entry/data/data"] = np.zeros(shape, np.uint16)
    info = io.inspect(path)
    assert info.ready
    assert info.scan_shape == shape[:2]
    assert info.detector_shape == shape[2:]
    assert info.metadata["dataset_path"] == "entry/data/data"
    mismatched = io.inspect(path, scan_shape=(1, 1))
    assert not mismatched.ready and mismatched.expected_frames == 1


def test_encoded_inspection_uses_header_and_preserves_validity(tmp_path):
    path = tmp_path / "encoded.no_extension"
    values = np.arange(3 * 5 * 7 * 9, dtype=np.uint16).reshape(3, 5, 7, 9)
    io.save(
        path,
        io.FourDSTEMData(values, {"excluded_detector_pixels": [3, 20]}),
        format="quantem",
        compression="ans",
        backend="cpu",
    )
    info = io.inspect(path)
    assert info.ready and info.source_kind == "ans"
    assert info.scan_shape == (3, 5) and info.detector_shape == (7, 9)
    assert info.metadata["encoded_bytes"] > 0
    np.testing.assert_array_equal(np.flatnonzero(info.pixel_mask), [3, 20])
    assert "unverified" in info.reason
    mismatched = io.inspect(path, scan_shape=(1, 1))
    assert not mismatched.ready and mismatched.expected_frames == 1


def test_prepared_header_checks_expected_scan_dimensions(tmp_path):
    (tmp_path / "checkpoint.json").write_text(json.dumps(dict(
        complete=True,
        layout=dict(complete=True, shape=[2, 4, 9, 7, 11], source_dtype="uint16"),
    )))
    info = io.inspect(tmp_path, scan_shape=(4, 9))
    assert info.ready and info.scan_shape == (4, 9)
    mismatched = io.inspect(tmp_path, scan_shape=(1, 1))
    assert not mismatched.ready and mismatched.reason == "scan_shape_mismatch"
    assert mismatched.expected_frames == 1 and mismatched.actual_frames == 36
