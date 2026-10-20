"""The flagged-pixel median fill is unbiased: half to even, and never 0 inside a cluster.

The CPU reference (``fill_flagged``) is what the Metal and CUDA kernels store; the
hardware test ``tests/hardware/test_hot_pixel_fill_backends.py`` holds them to it.
"""

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.resident.hot_pixels import fill_flagged, hot_pixel_record, median_plan


def test_even_neighbor_count_rounds_half_to_even():
    """An edge pixel beside another flagged one keeps four neighbors: their .5 median rounds to the even count."""
    mask = np.zeros((6, 6), np.uint32)
    mask[0, 2:4] = 1
    frames = np.full((2, 6, 6), 50, np.uint16)
    # (0, 2) keeps (0, 1), (1, 1), (1, 2), (1, 3); medians 2.5 and 3.5
    frames[0, 0, 1], frames[0, 1, 1], frames[0, 1, 2], frames[0, 1, 3] = 1, 2, 3, 9
    frames[1, 0, 1], frames[1, 1, 1], frames[1, 1, 2], frames[1, 1, 3] = 1, 3, 4, 9
    plan = median_plan(mask)
    assert np.diff(plan.offsets).tolist() == [4, 4]
    filled = fill_flagged(frames.copy(), plan, "median")
    assert filled[:, 0, 2].tolist() == [2, 4]  # rounding down gave 2 and 3
    np.testing.assert_array_equal(filled[:, mask == 0], frames[:, mask == 0])


def test_isolated_cluster_fills_from_the_5x5_ring():
    """The centre of a 3 x 3 flagged block has no valid 3 x 3 neighbor: it takes the median of the 5 x 5 ring, not 0."""
    mask = np.zeros((16, 16), np.uint8)
    mask[6:9, 6:9] = 1
    frames = np.random.default_rng(3).integers(100, 200, (3, 16, 16), dtype=np.uint16)
    plan = median_plan(mask)
    centre = list(plan.bad).index(7 * 16 + 7)
    assert plan.windows[centre] == 5 and plan.largest_window_px == 5
    ring = [(r, c) for r in range(5, 10) for c in range(5, 10) if not 6 <= r <= 8 or not 6 <= c <= 8]
    assert sorted(plan.neighbors[plan.offsets[centre]: plan.offsets[centre + 1]].tolist()) == sorted(
        r * 16 + c for r, c in ring)
    filled = fill_flagged(frames.copy(), plan, "median")
    expected = np.round(np.median(np.stack([frames[:, r, c] for r, c in ring], axis=1), axis=1))
    np.testing.assert_array_equal(filled[:, 7, 7], expected)
    assert (filled[:, 6:9, 6:9] > 0).all()
    assert hot_pixel_record(mask, "median", backend="cpu")["largest_window_px"] == 5


def test_every_pixel_flagged_is_refused():
    with pytest.raises(ValueError, match="Every detector pixel is flagged"):
        median_plan(np.ones((4, 4), np.uint8))


def test_median_is_the_default_and_the_cpu_reference_applies_it(tmp_path, capsys):
    """io.load's default fills flagged pixels; the CPU reference stores what fill_flagged gives, and says so."""
    counts = np.random.default_rng(5).integers(0, 1000, (12, 16, 16), dtype=np.uint16)
    mask = np.zeros((16, 16), np.uint32)
    mask[0, 0] = mask[8, 8] = 1
    mask[3:6, 10:13] = 2
    counts[:, mask != 0] = 65535
    master = tmp_path / "flagged_master.h5"
    data = tmp_path / "flagged_data_000001.h5"
    with h5py.File(data, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts, chunks=(1, 16, 16),
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(str(data), "/entry/data/data")
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = np.uint32(len(counts))
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask
    loaded = io.load(master, backend="cpu", representation="dense", scan_shape=(3, 4))
    record = loaded.metadata["hot_pixel_correction"]
    assert (record["method"], record["applied"], record["pixel_count"], record["largest_window_px"]) == ("median", True, 11, 5)
    assert "11 bad pixels replaced by neighbor median, windows up to 5 x 5" in capsys.readouterr().out
    expected = fill_flagged(counts.copy(), median_plan(mask), "median")
    np.testing.assert_array_equal(np.asarray(loaded.data).reshape(counts.shape), expected)
    raw = io.load(master, backend="cpu", representation="dense", scan_shape=(3, 4), hot_pixel_correction="none")
    np.testing.assert_array_equal(np.asarray(raw.data).reshape(counts.shape), counts)
