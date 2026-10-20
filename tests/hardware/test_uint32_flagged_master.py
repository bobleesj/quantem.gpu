"""ARINA uint32 input stays uint32, including flagged markers and large valid counts.

Optional corrections run on the GPU and are recorded as changed measurements.
The uncorrected route preserves all original counts; product masks remain separate.
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
    return expected


# 1100 scans span two Metal load batches; a 48 x 48 uint32 frame (9216 bytes)
# ends in a partial bitshuffle block.
@pytest.mark.parametrize("method", ["median", "zero"])
def test_flagged_uint32_master_preserves_dtype_with_explicit_correction(tmp_path, method):
    backend = _backend()
    scan_shape, detector_shape = (25, 44), (48, 48)
    counts, pixel_mask = _flagged_counts(25 * 44, detector_shape)
    master = _write_master(tmp_path, counts, pixel_mask)
    expected = _corrected(counts, pixel_mask, method)

    with io.load(master, backend=backend, scan_shape=scan_shape, hot_pixel_correction=method, verbose=False) as loaded:
        assert loaded.dtype == np.uint32
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


def test_flagged_uint32_master_retains_all_original_counts(tmp_path, capsys):
    backend = _backend()
    counts, pixel_mask = _flagged_counts(1024, (48, 48))
    counts[700, 10, 10] = 70000
    master = _write_master(tmp_path, counts, pixel_mask)
    with io.load(master, backend=backend, scan_shape=(32, 32), hot_pixel_correction="none",
                 apply_mask=False, verbose=True) as loaded:
        assert '4 bad pixels left as recorded (hot_pixel_correction="median" replaces them' in capsys.readouterr().out
        assert loaded.dtype == np.uint32
        assert loaded.metadata["file_counts_exact"] is True
        assert loaded.metadata["flagged_markers_stored_as_zero"] == 0
        assert loaded.metadata["detector_mask_policy"] == "preserve-stored-counts"
        np.testing.assert_array_equal(loaded.read().cpu().numpy().reshape(counts.shape), counts)
        np.testing.assert_array_equal(np.asarray(loaded.metadata["pixel_mask"]), pixel_mask)
        session = detector.prepare(loaded)
        valid = pixel_mask == 0
        np.testing.assert_array_equal(session.masked_sum_exact(valid), counts[:, valid].sum(1).reshape(32, 32))
        np.testing.assert_array_equal(session.frame(0)[~valid], 0)


def test_failed_uint32_load_releases_the_partial_resident(tmp_path, monkeypatch):
    """A later encoder failure frees earlier chunks even while a traceback survives."""
    from quantem.gpu.resident.bit_ans import BitANSResident

    backend = _backend()
    counts = np.random.default_rng(3).integers(0, 50, (1700, 8, 8), dtype=np.uint32)
    master = _write_master(tmp_path, counts, np.zeros((8, 8), np.uint32))
    append = BitANSResident.append_device
    partial = []

    def fail_later(source, raw):
        if source._lanes.ready_scans:
            partial.append(source)
            raise ValueError("injected encoder failure")
        return append(source, raw)

    monkeypatch.setattr(BitANSResident, "append_device", fail_later)
    with pytest.raises(ValueError, match="injected encoder failure"):
        io.load(str(master), scan_shape=(1700, 1), backend=backend, verbose=False)
    assert partial and all(source.is_released and source._lanes is None for source in partial)


@pytest.mark.parametrize("method", ["median", "none"])
def test_flagged_uint32_dataset_outside_the_arina_layout_loads_on_every_backend(tmp_path, method):
    """A plain 4D uint32 dataset with flagged-pixel markers loads like an Arina master.

    Stored markers remain intact unless correction was explicitly selected.
    """
    backend = _backend()
    counts, pixel_mask = _flagged_counts(6 * 8, (48, 48))
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as handle:
        handle["data"] = counts.reshape(6, 8, 48, 48)
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = pixel_mask
    expected = _corrected(counts, pixel_mask, "median") if method == "median" else counts.copy()
    with io.load(path, backend=backend, hot_pixel_correction=method, apply_mask=False, verbose=False) as loaded:
        assert loaded.dtype == np.uint32
        np.testing.assert_array_equal(loaded.read().cpu().numpy(), expected.reshape(6, 8, 48, 48))
        if method == "none":
            assert loaded.metadata["flagged_markers_stored_as_zero"] == 0


def test_ssb_open_replaces_flagged_pixels_says_so_and_records_it(tmp_path, capsys):
    """SSB.open applies the median replacement (decision D4) and prints one line naming how to keep raw values."""
    from quantem.gpu import SSB

    backend = _backend()
    counts, pixel_mask = _flagged_counts(1024, (48, 48))
    master = _write_master(tmp_path, counts, pixel_mask)
    with SSB.open(master, backend=backend, scan_shape=(32, 32), voltage_kV=300, semiangle_mrad=30,
                  scan_sampling_A=0.5, scan_detector_rotation_deg=0.0) as session:
        correction = session.source_provenance["hot_pixel_correction"]
        assert (correction["method"], correction["pixel_count"], correction["applied"]) == ("median", 4, True)
    out = capsys.readouterr().out
    assert "SSB: 4 flagged detector pixels replaced by their 3 x 3 neighbor median" in out
    assert "io.load(path) and use SSB(data)" in out
