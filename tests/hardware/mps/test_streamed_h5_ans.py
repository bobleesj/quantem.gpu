"""Exact original-HDF5 to runtime-ANS Metal workflows."""

import subprocess
import sys

import numpy as np
import pytest

import quantem.gpu
from quantem.gpu import io
from quantem.gpu.formats.qem.snapshot import INTEGER_CODEC
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
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

    # The default keeps every file count; detector queries apply validity separately.
    loaded = io.load(master, backend="mps", apply_mask=False)
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
    finally:
        loaded.close()


def _read_back(loaded) -> np.ndarray:
    output = loaded.read(output="native")
    try:
        return output.get()
    finally:
        output.release()


def _uint32_counts(seed: int) -> np.ndarray:
    """5120 uint32 patterns (two load batches) whose counts span the uint16 range."""
    counts = np.random.default_rng(seed).integers(0, 40, size=(5120, 4, 8), dtype=np.uint32)
    counts[0, 0, 0] = counts[4500, 3, 7] = 65535
    return counts


def test_uint32_counts_that_fit_load_as_exact_uint16(tmp_path, capsys):
    """uint32 counts that all fit 16 bits are stored as uint16 count ANS and read back equal to the file."""
    pytest.importorskip("Metal")
    import h5py
    import hdf5plugin  # noqa: F401

    counts = _uint32_counts(20261020)
    master = _write_bslz4_master(tmp_path, "uint32-fits", counts)
    with io.load(master, backend="mps", scan_shape=(64, 80)) as loaded:
        assert loaded.dtype == np.dtype("uint16")
        assert loaded.metadata["source_dtype"] == "uint32"
        assert loaded.metadata["resident_profile"] == INTEGER_CODEC
        assert isinstance(loaded.data, MPSStreamedCounts)
        assert len(loaded.data.spatial_chunks) == len(loaded.data.chunks) == 2
        values = _read_back(loaded)
    with h5py.File(master) as handle:
        stored = handle["entry/data/data_000001"][:]
    assert stored.dtype == np.uint32
    np.testing.assert_array_equal(values.astype(np.uint32), stored.reshape(64, 80, 4, 8))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Stored as")]
    assert lines == ["Stored as uint16: every uint32 value fits in 16 bits, so every count reads back unchanged."]


def test_uint32_values_above_uint16_keep_uint32_storage(tmp_path, capsys):
    """A count above 65535 in the second batch, or a kept 0xffffffff marker, keeps every value as uint32."""
    pytest.importorskip("Metal")
    import h5py
    import hdf5plugin  # noqa: F401

    late = _uint32_counts(7)
    late[4600, 1, 2] = 70000
    marker = _uint32_counts(8)
    marker[:, 2, 5] = 0xFFFFFFFF
    pixel_mask = np.zeros((4, 8), np.uint8)
    pixel_mask[2, 5] = 1
    for name, counts, mask in (("late", late, None), ("marker", marker, pixel_mask)):
        master = _write_bslz4_master(tmp_path, f"uint32-{name}", counts, pixel_mask=mask)
        with io.load(master, backend="mps", scan_shape=(64, 80)) as loaded:
            assert loaded.dtype == np.dtype("uint32")
            values = _read_back(loaded)
        with h5py.File(master) as handle:
            np.testing.assert_array_equal(values, handle["entry/data/data_000001"][:].reshape(64, 80, 4, 8))
        assert "Stored as" not in capsys.readouterr().out


def test_uint32_counts_narrowed_after_correction_match_uint32_storage(tmp_path):
    """With the flagged marker replaced by the neighbour median, uint16 storage holds what uint32 storage holds."""
    pytest.importorskip("Metal")
    from quantem.gpu.io import encoded
    from quantem.gpu.io.inspect import inspect

    counts = _uint32_counts(9)
    counts[:, 2, 5] = 0xFFFFFFFF
    pixel_mask = np.zeros((4, 8), np.uint8)
    pixel_mask[2, 5] = 1
    master = _write_bslz4_master(tmp_path, "uint32-median", counts, pixel_mask=pixel_mask)
    with io.load(master, backend="mps", scan_shape=(64, 80), hot_pixel_correction="median",
                 verbose=False) as narrowed:
        assert narrowed.dtype == np.dtype("uint16")
        values = _read_back(narrowed)
        sums = narrowed.data.masked_code_sums(np.ones((4, 8), bool))
    info = inspect(master, scan_shape=(64, 80))
    wide = encoded._load_h5_counts_mps(master, info, "median", False, np.dtype("uint32"))
    try:
        assert wide.dtype == np.dtype("uint32")
        np.testing.assert_array_equal(values.astype(np.uint32), _read_back(wide))
        np.testing.assert_array_equal(sums, wide.data.detector_sum_device(np.ones((4, 8), bool)).cpu().numpy().reshape(-1))
    finally:
        wide.close()
    valid = pixel_mask == 0
    np.testing.assert_array_equal(values[..., valid], counts.reshape(64, 80, 4, 8)[..., valid])


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


def test_decode_scratch_handles_changing_compressed_batch_sizes(tmp_path):
    """A sparse batch followed by a larger dense batch stays exact on reload."""
    pytest.importorskip("Metal")
    counts = np.zeros((12288, 4, 8), np.uint16)
    counts[8192:] = np.random.default_rng(13).integers(
        0, 65536, size=counts[8192:].shape, dtype=np.uint16
    )
    master = _write_bslz4_master(tmp_path, "varying-batches", counts)
    for _ in range(2):
        with io.load(master, backend="mps", scan_shape=(96, 128), verbose=False) as loaded:
            assert len(loaded.data.chunks) == 2
            np.testing.assert_array_equal(
                _decoded(loaded.data), counts.reshape(96, 128, 4, 8)
            )


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
    # the length field claims more than two blocks of bytes, which no valid block needs: refused without reading it
    "long": lambda length, block: length.__setitem__(slice(None), list((2 * 8192 + 17).to_bytes(4, "big"))),
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
        "long ValueError", "valid batch exact after the damaged ones",
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
