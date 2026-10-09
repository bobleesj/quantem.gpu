"""Arina uint32 masters with flagged pixels load as the same uint16 counts on CUDA and MPS.

Arina writes uint32 counts and 0xFFFFFFFF at every pixel its mask flags. The
loader replaces the flagged pixels on the GPU (by default with the median of
their valid 3x3 neighbors), then checks that every count fits uint16 before
encoding. Without correction a flagged sentinel is stored as 0 and counted. MPS used to check the raw counts on the host before the correction
and refused these files. h5py with hdf5plugin's bitshuffle filter and a NumPy
median are the oracle.
"""

import os

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import detector, io

SENTINEL = np.iinfo(np.uint32).max


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return backend


def _write_master(folder, counts, pixel_mask):
    """Write an Arina master: two external bitshuffle/LZ4 data files and the stored pixel mask."""
    master = folder / "flagged_master.h5"
    split = len(counts) // 2
    with h5py.File(master, "w") as handle:
        for index, frames in enumerate((counts[:split], counts[split:]), start=1):
            data_path = folder / f"flagged_data_{index:06d}.h5"
            with h5py.File(data_path, "w") as data_file:
                data_file.create_dataset(
                    "entry/data/data", data=frames, chunks=(1, *frames.shape[1:]),
                    **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
                )
            handle[f"entry/data/data_{index:06d}"] = h5py.ExternalLink(str(data_path), "/entry/data/data")
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = np.uint32(len(counts))
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = pixel_mask
    return master


def _flagged_counts(scans, detector_shape):
    """Random counts up to 1199 with uint16's largest count once, and four flagged pixels."""
    counts = np.random.default_rng(32).integers(0, 1200, (scans, *detector_shape), dtype=np.uint32)
    counts[7, 30, 30] = 65535
    pixel_mask = np.zeros(detector_shape, np.uint32)
    pixel_mask[0, 0] = 1  # corner: three neighbors
    pixel_mask[20, 31] = 1  # interior: eight neighbors
    pixel_mask[5, 6:8] = 2  # adjacent pair: neither enters the other's median
    counts[:, pixel_mask != 0] = SENTINEL
    return counts, pixel_mask


def _corrected(counts, pixel_mask, method):
    """Replace each flagged pixel by the median of its valid 3x3 neighbors (rounded down), or by zero."""
    expected = counts.copy()
    height, width = pixel_mask.shape
    for row, col in np.argwhere(pixel_mask != 0):
        neighbors = [
            counts[:, neighbor_row, neighbor_col]
            for neighbor_row in range(row - 1, row + 2)
            for neighbor_col in range(col - 1, col + 2)
            if 0 <= neighbor_row < height and 0 <= neighbor_col < width
            and pixel_mask[neighbor_row, neighbor_col] == 0
        ]
        expected[:, row, col] = 0 if method == "zero" else np.median(np.stack(neighbors, axis=-1), axis=-1)
    return expected.astype(np.uint16)


# 1100 scans span two Metal load batches; a 48 x 48 uint32 frame (9216 bytes)
# ends in a partial bitshuffle block.
@pytest.mark.parametrize("method", ["median", "zero"])
def test_flagged_uint32_master_loads_corrected_uint16_counts(tmp_path, method):
    backend = _backend()
    scan_shape, detector_shape = (25, 44), (48, 48)
    counts, pixel_mask = _flagged_counts(25 * 44, detector_shape)
    master = _write_master(tmp_path, counts, pixel_mask)
    expected = _corrected(counts, pixel_mask, method)

    with io.load(master, backend=backend, scan_shape=scan_shape, hot_pixel_correction=method, verbose=False) as loaded:
        assert loaded.dtype == np.uint16
        assert loaded.metadata["source_dtype"] == "uint32"
        assert loaded.metadata["hot_pixel_correction"]["applied"] is True
        assert loaded.metadata["file_counts_exact"] is False
        np.testing.assert_array_equal(loaded.read().cpu().numpy(), expected.reshape(*scan_shape, *detector_shape))
        session = detector.prepare(loaded)
        try:
            rows, cols = np.indices(detector_shape)
            disk = (rows - 24) ** 2 + (cols - 24) ** 2 <= 12**2
            np.testing.assert_array_equal(
                session.masked_sum_exact(disk),
                (expected * disk).sum((-2, -1), dtype=np.uint64).reshape(scan_shape),
            )
            total = expected.sum(0, dtype=np.uint64)
            np.testing.assert_array_equal(session.detector_total(), total)
            np.testing.assert_array_equal(session.mean_dp(), (total / len(expected)).astype(np.float32))
        finally:
            session.close()
    with h5py.File(master) as handle:
        np.testing.assert_array_equal(handle["entry/data/data_000001"][:3], counts[:3])


def test_flagged_uint32_master_refuses_valid_counts_uint16_cannot_hold(tmp_path, capsys):
    backend = _backend()
    counts, pixel_mask = _flagged_counts(1024, (48, 48))
    counts[700, 10, 10] = 65536  # a genuine count at a valid pixel is never clipped
    (tmp_path / "over-range").mkdir()
    with pytest.raises(ValueError, match="counts above 65535"):
        io.load(_write_master(tmp_path / "over-range", counts, pixel_mask), backend=backend,
                scan_shape=(32, 32), verbose=False)

    # Without correction the flagged sentinels cannot be uint16 counts. They are
    # stored as 0 and counted; the mask keeps their positions and every valid
    # count stays the file's.
    counts[700, 10, 10] = 3
    (tmp_path / "uncorrected").mkdir()
    master = _write_master(tmp_path / "uncorrected", counts, pixel_mask)
    with io.load(master, backend=backend, scan_shape=(32, 32), hot_pixel_correction="none",
                 apply_mask=False, verbose=True) as loaded:
        assert "4 bad pixels kept in the mask, their uint32 marker stored as 0" in capsys.readouterr().out
        stored = loaded.read().cpu().numpy().reshape(counts.shape)
        assert loaded.metadata["flagged_markers_stored_as_zero"] == 1024 * 4
        assert loaded.metadata["detector_mask_policy"] == "flagged-markers-stored-as-zero"
        np.testing.assert_array_equal(np.asarray(loaded.metadata["pixel_mask"]), pixel_mask)
    valid = pixel_mask == 0
    np.testing.assert_array_equal(stored[:, valid], counts[:, valid])
    assert not stored[:, ~valid].any()
