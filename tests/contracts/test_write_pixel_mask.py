"""2026-11-17: an ARINA master without a pixel_mask gets one where read_pixel_mask reads it; an existing one is kept."""

import h5py
import numpy as np
import pytest

from quantem.gpu.formats.hdf5.master import read_pixel_mask, write_pixel_mask


def test_a_master_without_a_mask_gets_one_and_an_existing_one_is_kept(tmp_path):
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_group("entry/instrument/detector/detectorSpecific")
    assert read_pixel_mask(master) is None
    mask = np.zeros((4, 5), np.uint32)
    mask[1, 2] = 1
    write_pixel_mask(master, mask)
    np.testing.assert_array_equal(read_pixel_mask(master), mask)
    with pytest.raises(ValueError, match="already has a pixel_mask"):
        write_pixel_mask(master, np.zeros((4, 5)))
