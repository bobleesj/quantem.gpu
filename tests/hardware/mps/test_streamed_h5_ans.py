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
            blocks.append(output.to_numpy())
        finally:
            output.release()
    return np.concatenate(blocks).reshape(source.shape)


def test_original_h5_auto_selects_lossless_runtime_ans(tmp_path):
    """Normal Metal opens preserve all native counts and validity metadata."""
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

    loaded = io.load(master, backend="mps", apply_mask=False)
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
            np.testing.assert_array_equal(output.to_numpy(), expected_first)
            loaded.data.detector_delta_device(second, first, output)
            expected_second = (
                counts.reshape(32, 32, 4, 8)
                * (second.astype(bool) & (pixel_mask == 0))
            ).sum(axis=(2, 3), dtype=np.uint32)
            np.testing.assert_array_equal(output.to_numpy(), expected_second)
        finally:
            output.release()
    finally:
        loaded.close()


def test_original_h5_series_stays_independent_and_batches_diffraction(tmp_path):
    """A folder-style list produces ordered residents and one exact DP command."""
    pytest.importorskip("Metal")
    from quantem.gpu.io.backends.mps._streamed import MPSStreamedSeries
    base = np.arange(1024 * 4 * 8, dtype=np.uint16).reshape(1024, 4, 8)
    counts = [base, base + 31, base + 79]
    paths = [
        _write_bslz4_master(tmp_path, f"series-{index}", values)
        for index, values in enumerate(counts)
    ]

    with pytest.raises(ValueError, match="stack=False"):
        io.load(paths, backend="mps", apply_mask=False)
    loaded = io.load(paths, backend="mps", stack=False, apply_mask=False)
    series = None
    try:
        assert len(loaded) == len(counts)
        assert all(item.representation is io.DataRepresentation.ENCODED for item in loaded)
        series = MPSStreamedSeries([item.data for item in loaded])
        assert series.residency_bytes >= sum(item.data.nbytes for item in loaded)
        output = series.extract_diffraction_device(17, 9)
        try:
            scan = 17 * 32 + 9
            np.testing.assert_array_equal(
                output.to_numpy(), np.stack([values[scan] for values in counts])
            )
        finally:
            output.release()
        foreground, submission = series.submit_priority_diffraction_device(
            17, 9, priority_index=2
        )
        try:
            np.testing.assert_array_equal(foreground.to_numpy(), counts[2][scan])
            np.testing.assert_array_equal(
                submission.finish().to_numpy(),
                np.stack([values[scan] for values in counts]),
            )
            assert submission.finish() is submission.output
        finally:
            foreground.release()
            if not submission.output.is_released:
                submission.output.release()

        series.release()
        assert series.residency_bytes == 0
        with pytest.raises(RuntimeError, match="released"):
            series.extract_diffraction_device(0, 0)
    finally:
        if series is not None:
            series.release()
        for item in loaded:
            item.close()
