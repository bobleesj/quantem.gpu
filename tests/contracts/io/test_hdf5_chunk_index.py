"""Chunk metadata must address the same on-disk evidence as h5py."""

from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pytest

from quantem.gpu.io._hdf5_chunk_index import (
    _chunk_locations, _chunk_locations_python,
)


@pytest.mark.parametrize("userblock", [0, 512])
def test_direct_chunk_locations_match_saved_frames(tmp_path, userblock):
    """Index complete and partially acquired scans without changing addresses."""
    path = tmp_path / "frames.h5"
    data = np.arange(11 * 8 * 8, dtype=np.uint16).reshape(11, 8, 8)
    with h5py.File(path, "w", userblock_size=userblock) as file:
        for name, frames in [("complete", range(11)), ("partial", [0, 3, 10])]:
            dataset = file.create_dataset(
                name, shape=data.shape, chunks=(1, 8, 8), dtype=data.dtype,
                compression="gzip",
            )
            for frame in frames:
                dataset[frame] = data[frame]
        file.create_dataset("empty", shape=data.shape, chunks=(1, 8, 8), dtype=data.dtype)
    with h5py.File(path) as file:
        for dataset in file.values():
            expected = _chunk_locations_python(dataset)
            actual = _chunk_locations(dataset)
            np.testing.assert_array_equal(actual, expected)
            # Raw bytes at each returned address must be the actual chunks.
            with path.open("rb") as source:
                allocated = []
                dataset.id.chunk_iter(lambda info: allocated.append(info))
                for info, (offset, size) in zip(allocated, actual, strict=True):
                    source.seek(int(offset))
                    assert source.read(int(size)) == dataset.id.read_direct_chunk(
                        info.chunk_offset
                    )[1]


def test_parallel_acquisition_indexing(tmp_path):
    """Concurrent file loads retain independent chunk tables."""
    path = tmp_path / "two_scans.h5"
    with h5py.File(path, "w") as file:
        for index in range(2):
            file.create_dataset(str(index), data=np.full((41, 8, 8), index),
                                chunks=(1, 8, 8), compression="gzip")
    def read(index):
        with h5py.File(path) as file:
            dataset = file[str(index)]
            np.testing.assert_array_equal(
                _chunk_locations(dataset), _chunk_locations_python(dataset)
            )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(read, [0, 1] * 4))
