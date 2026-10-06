"""Exact original-HDF5 to runtime-ANS Metal workflows."""

import numpy as np
import pytest

from quantem.gpu import io
from tests.hardware.mps.test_mps_buffer_release import _write_bslz4_master


def _decoded(source) -> np.ndarray:
    blocks = []
    for index in range(len(source.chunks)):
        output = source.decode_block_device(index)
        try:
            blocks.append(output.get())
        finally:
            output.release()
    return np.concatenate(blocks).reshape(source.shape)


def test_original_h5_auto_selects_lossless_runtime_ans(tmp_path):
    """Metal opens without hot-pixel correction preserve all native counts and validity metadata."""
    pytest.importorskip("Metal")
    rng = np.random.default_rng(20260912)
    dtype = np.uint16
    counts = rng.integers(0, 33, size=(1024, 4, 8), dtype=dtype)
    counts[0, 0, 0] = np.iinfo(dtype).max
    counts[511, 2, 3] = np.iinfo(dtype).max
    pixel_mask = np.zeros((4, 8), dtype=np.uint8)
    pixel_mask[1, 2] = 1
    master = _write_bslz4_master(
        tmp_path, "runtime-ans", counts, pixel_mask=pixel_mask
    )

    # Loads correct flagged pixels by default (median); "none" keeps the file's counts.
    loaded = io.load(master, backend="mps", apply_mask=False, hot_pixel_correction="none")
    try:
        assert loaded.representation is io.DataRepresentation.ENCODED
        assert loaded.shape == (32, 32, 4, 8)
        assert loaded.dtype == np.dtype(dtype)
        assert loaded.metadata["lossless_exact"] is True
        assert loaded.metadata["file_counts_exact"] is True
        assert loaded.metadata["scan_bin"] == 1
        assert loaded.metadata["detector_bin"] == 1
        assert loaded.metadata["crop"] is None
        np.testing.assert_array_equal(_decoded(loaded.data), counts.reshape(32, 32, 4, 8))

        first = np.zeros((4, 8), dtype=np.uint8)
        first[0:3, 0:4] = 1
        second = first.copy()
        second[0, 4] = 1
        second[2, 1] = 0
        output = loaded.data.detector_delta_device(first)
        try:
            expected_first = (
                counts.reshape(32, 32, 4, 8)
                * (first.astype(bool) & (pixel_mask == 0))
            ).sum(axis=(2, 3), dtype=np.uint32)
            np.testing.assert_array_equal(output.get(), expected_first)
            loaded.data.detector_delta_device(second, first, output)
            expected_second = (
                counts.reshape(32, 32, 4, 8)
                * (second.astype(bool) & (pixel_mask == 0))
            ).sum(axis=(2, 3), dtype=np.uint32)
            np.testing.assert_array_equal(output.get(), expected_second)
        finally:
            output.release()
    finally:
        loaded.close()


def test_original_h5_series_stays_independent_and_batches_diffraction(tmp_path):
    """A folder-style list produces ordered, independent encoded residents."""
    pytest.importorskip("Metal")
    base = np.arange(1024 * 4 * 8, dtype=np.uint16).reshape(1024, 4, 8)
    counts = [base, base + 31, base + 79]
    paths = [
        _write_bslz4_master(tmp_path, f"series-{index}", values)
        for index, values in enumerate(counts)
    ]

    with pytest.raises(ValueError, match="omit stack"):
        io.load(paths, backend="mps", stack=True, apply_mask=False)
    loaded = io.load(paths, backend="mps", apply_mask=False)
    try:
        assert len(loaded) == len(counts)
        assert all(item.representation is io.DataRepresentation.ENCODED for item in loaded)
        scan = 17 * 32 + 9
        for item, values in zip(loaded, counts, strict=True):
            np.testing.assert_array_equal(item[17, 9].cpu().numpy(), values[scan])
    finally:
        for item in loaded:
            item.close()
