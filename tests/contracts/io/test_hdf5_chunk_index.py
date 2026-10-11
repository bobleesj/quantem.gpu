"""Chunk metadata must address the same on-disk evidence as h5py."""

from concurrent.futures import ThreadPoolExecutor

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu.formats.hdf5.frames import (
    _chunk_locations_python,
    chunk_locations,
)
from quantem.gpu.formats.hdf5.reads import FrameReader


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
            actual = chunk_locations(dataset)
            np.testing.assert_array_equal(actual, expected)
            # Raw bytes at each returned address must be the actual chunks.
            with path.open("rb") as source:
                allocated = []
                dataset.id.chunk_iter(allocated.append)
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
                chunk_locations(dataset), _chunk_locations_python(dataset)
            )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(read, [0, 1] * 4))


def _lengthened_first_block(chunk: bytes) -> bytes:
    return chunk[:12] + (int.from_bytes(chunk[12:16], "big") + 1000).to_bytes(4, "big") + chunk[16:]


@pytest.mark.parametrize("damage", [
    lambda chunk: chunk[:-40],             # the last LZ4 block lost its end: a file still being written
    lambda chunk: chunk[:10],              # shorter than the 12-byte chunk header
    lambda chunk: bytes(len(chunk)),       # allocated but never written
    _lengthened_first_block,               # a block length that points past the chunk
], ids=["cut short", "header only", "zeroed", "block length too long"])
def test_truncated_or_damaged_chunk_is_refused_by_the_frame_reader(tmp_path, damage):
    """The block walk of a bitshuffle+LZ4 chunk stops at the chunk's last byte and must end exactly there.

    Each chunk is a 12-byte header and then, per 8192-byte block, a 4-byte length and the block. A chunk that is
    cut short or carries a damaged length would send an unchecked walk past its own bytes (a segfault in the host
    parser, or block starts outside the chunk handed to the GPU decoder), so the reader refuses the frame.
    """
    frames = np.random.default_rng(7).poisson(3.0, (4, 64, 96)).astype(np.uint16)  # one whole and one partial block
    master = tmp_path / "damaged_master.h5"
    with h5py.File(master, "w") as handle:
        dataset = handle.create_dataset("entry/data/data", data=frames, chunks=(1, 64, 96),
                                        **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        filter_mask, chunk = dataset.id.read_direct_chunk((1, 0, 0))
        dataset.id.write_direct_chunk((1, 0, 0), damage(chunk), filter_mask)
    with FrameReader(str(master)) as reader:
        intact = reader.prepare(np.array([0, 2, 3]))
        np.testing.assert_array_equal(intact["block_counts"], [2, 2, 2])
        with pytest.raises(ValueError, match="frame 1 is truncated or corrupt"):
            reader.prepare(np.arange(4))
