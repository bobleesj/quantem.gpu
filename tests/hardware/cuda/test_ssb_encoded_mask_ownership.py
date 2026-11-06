"""Bright-field masking preserves exact decoded counts and the encoded source."""

import hashlib

import h5py
import numpy as np
import pytest

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_masked_fit_crop_does_not_modify_acquisition(tmp_path, dtype):
    """An invalid detector pixel is excluded only from the fit-owned crop."""
    from quantem.gpu import io
    from quantem.gpu.direct_ptycho.brightfield import crop_bright_field

    source = tmp_path / "scan_master.h5"
    counts = np.zeros((8, 8, 16, 16), dtype=dtype)
    counts[:, :, 3:13, 3:13] = np.arange(64, dtype=dtype).reshape(8, 8, 1, 1) + 1
    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[7, 7] = 1
    counts[:, :, 7, 7] = np.iinfo(dtype).max
    with h5py.File(source, "w") as handle:
        handle["entry/data/data"] = counts
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    with io.load(source, backend="cuda", hot_pixel_correction="none", verbose=False) as data:
        original = data.read().cpu().numpy()
        crop, _, _, _, (row, col), _ = crop_bright_field(data, "cuda", 0.05, None)
        expected = counts[:, :, row:row + crop.shape[2], col:col + crop.shape[3]].copy()
        expected[:, :, 7 - row, 7 - col] = 0
        np.testing.assert_array_equal(cp.asnumpy(crop), expected)
        np.testing.assert_array_equal(data.read().cpu().numpy(), original)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
