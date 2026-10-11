"""Exact original-HDF5 to runtime-ANS Metal workflows."""

import subprocess
import sys

import numpy as np
import pytest

import quantem.gpu
from quantem.gpu import io
from tests.hardware.mps.test_mps_buffer_release import _write_bslz4_master


def _decoded(source) -> np.ndarray:
    blocks = []
    for index in range(len(source.chunks)):
        output = source.decode_block_device(index)
        try:
            blocks.append(output.get())
        finally:
            output.release()
    return np.concatenate(blocks).reshape(source.shape)


def test_original_h5_auto_selects_lossless_runtime_ans(tmp_path):
    """Metal opens without hot-pixel correction preserve all native counts and validity metadata."""
    pytest.importorskip("Metal")
    rng = np.random.default_rng(20260912)
    dtype = np.uint16
    counts = rng.integers(0, 33, size=(1024, 4, 8), dtype=dtype)
    counts[0, 0, 0] = np.iinfo(dtype).max
    counts[511, 2, 3] = np.iinfo(dtype).max
    pixel_mask = np.zeros((4, 8), dtype=np.uint8)
    pixel_mask[1, 2] = 1
    master = _write_bslz4_master(
        tmp_path, "runtime-ans", counts, pixel_mask=pixel_mask
    )

    # Loads correct flagged pixels by default (median); "none" keeps the file's counts.
    loaded = io.load(master, backend="mps", apply_mask=False, hot_pixel_correction="none")
    try:
        assert loaded.representation is io.DataRepresentation.ENCODED
        assert loaded.shape == (32, 32, 4, 8)
        assert loaded.dtype == np.dtype(dtype)
        assert loaded.metadata["lossless_exact"] is True
        assert loaded.metadata["file_counts_exact"] is True
        assert loaded.metadata["scan_bin"] == 1
        assert loaded.metadata["detector_bin"] == 1
        assert loaded.metadata["crop"] is None
        np.testing.assert_array_equal(_decoded(loaded.data), counts.reshape(32, 32, 4, 8))

        first = np.zeros((4, 8), dtype=np.uint8)
        first[0:3, 0:4] = 1
        second = first.copy()
        second[0, 4] = 1
        second[2, 1] = 0
        output = loaded.data.detector_delta_device(first)
        try:
            expected_first = (
                counts.reshape(32, 32, 4, 8)
                * (first.astype(bool) & (pixel_mask == 0))
            ).sum(axis=(2, 3), dtype=np.uint32)
            np.testing.assert_array_equal(output.get(), expected_first)
            loaded.data.detector_delta_device(second, first, output)
            expected_second = (
                counts.reshape(32, 32, 4, 8)
                * (second.astype(bool) & (pixel_mask == 0))
            ).sum(axis=(2, 3), dtype=np.uint32)
            np.testing.assert_array_equal(output.get(), expected_second)
        finally:
            output.release()
    finally:
        loaded.close()


def test_original_h5_series_stays_independent_and_batches_diffraction(tmp_path):
    """A folder-style list produces ordered, independent encoded residents."""
    pytest.importorskip("Metal")
    base = np.arange(1024 * 4 * 8, dtype=np.uint16).reshape(1024, 4, 8)
    counts = [base, base + 31, base + 79]
    paths = [
        _write_bslz4_master(tmp_path, f"series-{index}", values)
        for index, values in enumerate(counts)
    ]

    with pytest.raises(ValueError, match="omit stack"):
        io.load(paths, backend="mps", stack=True, apply_mask=False)
    loaded = io.load(paths, backend="mps", apply_mask=False)
    try:
        assert len(loaded) == len(counts)
        assert all(item.representation is io.DataRepresentation.ENCODED for item in loaded)
        scan = 17 * 32 + 9
        for item, values in zip(loaded, counts, strict=True):
            np.testing.assert_array_equal(item[17, 9].cpu().numpy(), values[scan])
    finally:
        for item in loaded:
            item.close()


DAMAGED_LZ4_SCRIPT = r"""
import sys

import h5py
import hdf5plugin
import numpy as np

import quantem.gpu
from quantem.gpu.formats.hdf5.reads import FrameReader
from quantem.gpu.io.hdf5.mps.decode import load_prepared_frames

assert quantem.gpu.__file__ == sys.argv[1], quantem.gpu.__file__
noise = np.random.default_rng(3)
damage = {
    "zeroed": lambda length, block: block.fill(0),  # token 0, match offset 0: a copy from before the block
    "random": lambda length, block: block.__setitem__(slice(None), noise.integers(0, 256, block.size, np.uint8)),
    "half": lambda length, block: block[block.size // 2:].fill(0),
    # one literal, then a match of 10219 bytes that repeats it: every read is valid, the block is 2 KiB too long
    "overlong": lambda length, block: block.__setitem__(slice(0, 45), [0x1F, 0, 1, 0] + [255] * 40 + [0]),
    # the block stops halfway, as in a file still being written
    "short": lambda length, block: length.__setitem__(slice(None), list((block.size // 2).to_bytes(4, "big"))),
}
for master in sys.argv[2:]:
    with h5py.File(master) as file:
        values = file["entry/data/data_000001"][:]
    reader = FrameReader(master)
    for name, change in damage.items():
        prepared = reader.prepare(np.arange(8))
        header = int(prepared["chunk_offsets"][2]) + int(prepared["block_starts"][prepared["block_offsets"][2]])
        size = int.from_bytes(bytes(prepared["read_buffer"][header:header + 4]), "big")
        change(prepared["read_buffer"][header:header + 4], prepared["read_buffer"][header + 4:header + 4 + size])
        try:
            load_prepared_frames(prepared)
        except ValueError as error:
            assert "Frames 2 to 2" in str(error) and "truncated or corrupt" in str(error), error
            print(name, "ValueError")
        else:
            raise SystemExit(f"{name}: damaged LZ4 bytes decoded without an error")
    decoded = load_prepared_frames(reader.prepare(np.arange(8)))
    reader.close()
    assert np.array_equal(decoded, values)
    print("valid batch exact after the damaged ones")
"""


def test_damaged_lz4_bytes_raise_and_the_gpu_stays_usable(tmp_path):
    """LZ4 bytes damaged after the chunk walk raise ValueError naming the frame, and the same process then decodes a valid batch exactly.

    Both Metal LZ4 decoders bound their reads and writes and refuse a block that does not decode to exactly its own
    bytes: 192 x 192 uint16 frames take the fused decode-and-unshuffle kernel, and 100 x 100 frames end in a partial
    block and take the kernel that decodes into device memory. Without the bounds a damaged block decodes to wrong
    counts without an error and copies outside its own bytes. Runs in a subprocess so that a regression cannot
    overwrite this session's Metal buffers.
    """
    pytest.importorskip("Metal")
    rng = np.random.default_rng(5)
    masters = [
        _write_bslz4_master(tmp_path, f"damaged-{side}", rng.poisson(3.0, (8, side, side)).astype(np.uint16))
        for side in (192, 100)
    ]
    result = subprocess.run(
        [sys.executable, "-c", DAMAGED_LZ4_SCRIPT, quantem.gpu.__file__, *map(str, masters)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == 2 * [
        "zeroed ValueError", "random ValueError", "half ValueError", "overlong ValueError", "short ValueError",
        "valid batch exact after the damaged ones",
    ]


def test_failed_decode_command_raises_instead_of_returning_frames(tmp_path, monkeypatch):
    """A Metal decode command that ends with an error raises: the frames it never wrote are not returned."""
    pytest.importorskip("Metal")
    from quantem.gpu.formats.hdf5.reads import FrameReader
    from quantem.gpu.io.hdf5.mps import decode

    class FailedCommand:
        """The two calls the decoder makes on its submitted command, for one the GPU gave up on."""

        def waitUntilCompleted(self):
            pass

        def error(self):
            return "the GPU stopped responding"

    master = _write_bslz4_master(tmp_path, "failed-command", np.arange(4 * 64 * 64, dtype=np.uint16).reshape(4, 64, 64))
    monkeypatch.setattr(decode.MPSDecompressor, "_submit_gpu", lambda self, *arguments: FailedCommand())
    with FrameReader(str(master)) as reader:
        with pytest.raises(RuntimeError, match="the GPU stopped responding"):
            decode.load_prepared_frames(reader.prepare(np.arange(4)))
