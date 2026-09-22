"""Masked detector sentinels do not prevent exact working-count ingestion."""

import os

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import detector, io


@pytest.mark.parametrize("method", ["zero", "median"])
def test_uint32_masked_sentinels_preserve_working_counts(tmp_path, method):
    """Open a masked uint32 master, inspect native counts, and reopen its QEM."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND to the physical accelerator.")
    values = np.arange(6 * 4 * 6, dtype=np.uint32).reshape(2, 3, 4, 6) * 31
    mask = np.zeros((4, 6), np.uint32)
    mask[0, 0] = 1
    mask[2, 3] = 2
    values[..., mask != 0] = np.iinfo(np.uint32).max
    expected = values.copy()
    for row, column in np.argwhere(mask):
        if method == "zero":
            expected[..., row, column] = 0
        else:
            neighbors = [
                values[..., rr, cc]
                for rr in range(max(0, row - 1), min(4, row + 2))
                for cc in range(max(0, column - 1), min(6, column + 2))
                if mask[rr, cc] == 0
            ]
            expected[..., row, column] = np.median(neighbors, axis=0).astype(np.uint32)
    original, saved = tmp_path / "masked.h5", tmp_path / "copy.qem"
    with h5py.File(original, "w") as handle:
        handle.create_dataset("entry/data/data", data=values.reshape(6, 4, 6),
                              chunks=(1, 4, 6), **hdf5plugin.Bitshuffle(cname="lz4"))
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask
    with io.load(original, backend=backend, scan_shape=(2, 3),
                 auto_narrow=True, hot_pixel_correction=method,
                 verbose=False) as loaded:
        actual = loaded.read().cpu().numpy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[..., mask == 0], values[..., mask == 0])
        assert loaded.metadata["source_dtype"] == "uint32"
        assert loaded.metadata["hot_pixel_correction"]["method"] == method
        assert loaded.metadata["working_counts_exact"]
        assert not loaded.metadata["file_counts_exact"]
        io.save(saved, loaded)
    with io.load(saved, backend=backend, verbose=False) as loaded:
        np.testing.assert_array_equal(detector.prepare(loaded).frame(5), expected[-1, -1])
    with h5py.File(original) as handle:
        np.testing.assert_array_equal(handle["entry/data/data"][:], values.reshape(6, 4, 6))
    # With correction disabled, the sentinel is part of the preserved data.
    with pytest.raises((NotImplementedError, ValueError), match="65535|uint16"):
        io.load(original, backend=backend, scan_shape=(2, 3),
                auto_narrow=True, hot_pixel_correction="none", verbose=False)
    # A high count at a valid detector pixel must never be clipped or ignored.
    with h5py.File(original, "r+") as handle:
        handle["entry/data/data"][0, 0, 1] = 65536
    with pytest.raises((NotImplementedError, ValueError), match="65535|uint16"):
        io.load(original, backend=backend, scan_shape=(2, 3),
                auto_narrow=True, hot_pixel_correction=method, verbose=False)
