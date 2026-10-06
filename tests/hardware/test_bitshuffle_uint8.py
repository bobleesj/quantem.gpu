"""Bitshuffle/LZ4 uint8 masters load exactly on CUDA and MPS.

The GPU decoders had no one-byte unshuffle: CUDA unshuffled uint8 frames with
the four-byte kernel and returned wrong counts without an error, and MPS
refused the file. h5py with hdf5plugin's reference bitshuffle filter is the
oracle for every count.
"""

import os
from functools import partial
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import io


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return backend


def _master(folder: Path, counts: np.ndarray) -> np.ndarray:
    """Write an Arina-style bitshuffle/LZ4 uint8 master; return what h5py decodes from it."""
    path = folder / "uint8_master.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=counts.reshape(-1, *counts.shape[2:]),
            chunks=(1, *counts.shape[2:]),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = (
            counts.shape[0] * counts.shape[1]
        )
    with h5py.File(path, "r") as handle:
        return path, handle["entry/data/data"][...].reshape(counts.shape)


# Frame bytes: one complete 8 KiB block; a partial block only; a complete block
# and a partial one; three complete blocks; four complete blocks and a partial one.
@pytest.mark.parametrize("detector_shape", [(64, 128), (48, 48), (96, 100), (128, 192), (192, 192)])
def test_uint8_master_loads_the_counts_h5py_decodes(tmp_path, detector_shape):
    backend = _backend()
    counts = np.random.default_rng(sum(detector_shape)).integers(
        0, 256, (6, 7, *detector_shape), dtype=np.uint8
    )
    counts[0, 0, 0, 0] = 255
    path, expected = _master(tmp_path, counts)
    np.testing.assert_array_equal(expected, counts)

    with io.load(path, backend=backend, scan_shape=counts.shape[:2],
                  hot_pixel_correction="none", verbose=False) as loaded:
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded.read().cpu().numpy(), expected)


def test_uint8_master_decodes_exactly_across_cuda_upload_batches(tmp_path, monkeypatch):
    """Frames decode in batches of three, so the double-buffered upload path runs."""
    if _backend() != "cuda":
        pytest.skip("Batched compressed uploads are a CUDA decoder path.")
    from quantem.gpu.io import encoded

    counts = np.random.default_rng(3).integers(0, 256, (5, 4, 96, 100), dtype=np.uint8)
    path, expected = _master(tmp_path, counts)
    decode = encoded.decompress_prepared
    monkeypatch.setattr(
        encoded, "decompress_prepared",
        lambda prepared, batch_bytes_target: partial(decode, batch_bytes_target=3 * 96 * 100)(prepared),
    )
    with io.load(path, backend="cuda", scan_shape=counts.shape[:2],
                 hot_pixel_correction="none", verbose=False) as loaded:
        np.testing.assert_array_equal(loaded.read().cpu().numpy(), expected)


def test_uint8_frames_with_an_unshuffled_remainder_are_refused(tmp_path):
    """Bitshuffle stores the last frame_bytes % 8 elements unshuffled; the GPU decoders refuse them."""
    backend = _backend()
    counts = np.random.default_rng(5).integers(0, 256, (2, 2, 33, 33), dtype=np.uint8)
    path, _ = _master(tmp_path, counts)
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        io.load(path, backend=backend, scan_shape=counts.shape[:2],
                hot_pixel_correction="none", verbose=False)
