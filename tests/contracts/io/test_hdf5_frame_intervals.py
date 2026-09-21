"""Read flattened scan intervals without repeating storage-chunk reads."""

import h5py
import numpy as np

from quantem.gpu.io._hdf5_array_resident import _read_four_dimensional_frames


def test_rectangular_scan_intervals_use_one_read(tmp_path):
    """Partial rows and multirow selections preserve acquisition order."""
    values = np.arange(5 * 7 * 3 * 4, dtype=np.float32).reshape(5, 7, 3, 4) / 8
    with h5py.File(tmp_path / "scan.h5", "w") as handle:
        data = handle.create_dataset(
            "data", data=values, chunks=(5, 7, 1, 2), compression="gzip"
        )

        class ObservedID:
            calls = 0

            def get_space(self):
                return data.id.get_space()

            def read(self, *args):
                self.calls += 1
                data.id.read(*args)

        class ObservedDataset:
            id = ObservedID()
            shape = data.shape

        observed = ObservedDataset()
        for first, stop in [(0, 1), (0, 35), (1, 34), (6, 22), (8, 10), (28, 35)]:
            result = np.empty((stop - first, 3, 4), np.float32)
            previous = observed.id.calls
            _read_four_dimensional_frames(observed, result, first, stop)
            assert observed.id.calls == previous + 1
            np.testing.assert_array_equal(result, values.reshape(35, 3, 4)[first:stop])
