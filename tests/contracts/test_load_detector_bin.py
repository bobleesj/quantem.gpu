"""2026-11-17: io.load(master, detector_bin=2) sums 2 x 2 on the GPU as it encodes: a GPU too small for the full scan
(the ARINA PC's 4 GB A400) loads it binned, every electron kept, the record saying so."""

import h5py
import hdf5plugin
import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _compiles():
    try:
        cp.ElementwiseKernel("T x", "T y", "y = x + 1", "bin_probe")(cp.zeros(1)).item()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _compiles(), reason="needs a CUDA GPU that compiles kernels")

SCAN, DETECTOR, FLAGGED = (6, 6), (16, 16), [(3, 5)]


def master(folder, sampling=None):
    rng = np.random.default_rng(3)
    data = rng.poisson(4.0, size=(SCAN[0] * SCAN[1], *DETECTOR)).astype(np.uint16)
    mask = np.zeros(DETECTOR, np.uint32)
    for row, col in FLAGGED:
        data[:, row, col] = 65535
        mask[row, col] = 1
    path = folder / "scan_master.h5"
    with h5py.File(path, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector["detectorSpecific/pixel_mask"] = mask
        detector["detectorSpecific/ntrigger"] = data.shape[0]
        if sampling is not None:
            handle.attrs["detector_sampling_mrad"] = np.array([sampling, sampling])
        with h5py.File(folder / "scan_data_000001.h5", "w") as part:
            part.create_dataset("entry/data/data", data=data, chunks=(1, *DETECTOR), **hdf5plugin.Bitshuffle(cname="lz4"))
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data_000001.h5", "entry/data/data")
    return path, data


def test_a_binned_load_is_the_2x2_sum_and_says_so(tmp_path):
    from quantem.gpu import io

    path, data = master(tmp_path, sampling=0.5765)
    full = io.load(str(path), hot_pixel_correction="median", verbose=False)
    binned = io.load(str(path), hot_pixel_correction="median", verbose=False, detector_bin=2)
    try:
        assert tuple(binned.shape) == (*SCAN, 8, 8)
        corrected = cp.asnumpy(full.read(output="native")).astype(np.int64)        # the flagged pixel filled first
        expected = corrected.reshape(*SCAN, 8, 2, 8, 2).sum(axis=(3, 5))
        np.testing.assert_array_equal(cp.asnumpy(binned.read(output="native")).astype(np.int64), expected)
        assert int(expected.sum()) == int(corrected.sum())                          # every electron kept
        assert binned.metadata["detector_bin"] == 2
        assert binned.metadata["binned_from_detector_shape"] == [16, 16]
        assert binned.metadata["detector_sampling_mrad"] == [1.153, 1.153]
        assert binned.resident_bytes < full.resident_bytes
    finally:
        full.close()
        binned.close()


def test_detector_bin_is_refused_where_it_would_be_ignored(tmp_path):
    from quantem.gpu import io

    with pytest.raises(ValueError, match="ARINA HDF5 masters"):
        io.load(str(tmp_path / "scan.qem"), detector_bin=2)
    path, _ = master(tmp_path)
    with pytest.raises(ValueError, match="1, 2, 4 or 8"):
        io.load(str(path), detector_bin=3, verbose=False)
