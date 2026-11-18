"""HDF5 masters preserve full scan trajectories as attributes."""

import h5py
import numpy as np
import pytest

from quantem.gpu.io.hdf5.write import _write_master_file


@pytest.mark.parametrize("copy_source", [False, True])
def test_master_preserves_large_scan_positions(tmp_path, copy_source):
    order = np.arange(512 * 512, dtype=np.int32)[::-1]
    positions = np.column_stack(np.divmod(order, 512))
    source = tmp_path / "source_master.h5" if copy_source else None
    if source is not None:
        with h5py.File(source, "w", track_order=True) as handle:
            handle.attrs["original_positions"] = positions

    master = tmp_path / "scan_master.h5"
    _write_master_file(master, ["scan_data_000001.h5"], [(1, len(positions))], (1, len(positions)),
                       (96, 96), np.uint16, {"scan_positions": positions}, source)

    with h5py.File(master, "r") as handle:
        np.testing.assert_array_equal(handle.attrs["scan_positions"], positions)
        assert handle.attrs["scan_positions"].dtype == positions.dtype
        if source is not None:
            np.testing.assert_array_equal(handle.attrs["original_positions"], positions)
