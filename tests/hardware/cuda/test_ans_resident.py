"""CUDA tests for exact, bounded resident ANS reads."""

from quantem.gpu.io.dataset import Dataset4dstemGPU

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)

from quantem.gpu.resident.cuda.counts import StreamedCounts


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_streamed_ans_decodes_selected_scan_ranges(dtype):
    """Selected ranges remain exact across entropy chunk boundaries."""
    shape = (6, 300, 3, 4)
    maximum = 251 if dtype == np.uint8 else 4093
    values = (
        cp.arange(np.prod(shape), dtype=cp.uint64) % maximum
    ).astype(dtype).reshape(shape)
    source = StreamedCounts(shape, dtype)
    flat = values.reshape(-1, *shape[2:])
    try:
        for first, stop in ((0, 700), (700, 1500), (1500, 1800)):
            source.append(cp.ascontiguousarray(flat[first:stop]))
        for first, stop in ((510, 515), (695, 705), (1490, 1510)):
            observed = source.decode_scan_range_device(first, stop)
            assert bool(cp.all(observed == flat[first:stop]))
    finally:
        source.release()


def test_saved_copy_holds_each_encoded_array_at_its_offset(tmp_path, monkeypatch):
    """Arrays leave the GPU in pinned pieces; pieces far smaller than an array still land in place."""
    from quantem.gpu import io
    from quantem.gpu.formats.qem.snapshot import read_header
    from quantem.gpu.io import qem

    shape = (6, 300, 3, 4)
    values = (cp.arange(np.prod(shape), dtype=cp.uint64) % 4093).astype(cp.uint16).reshape(-1, *shape[2:])
    source = StreamedCounts(shape, np.uint16)
    path = tmp_path / "copy.qem"
    try:
        for first, stop in ((0, 700), (700, 1800)):
            source.append(cp.ascontiguousarray(values[first:stop]))
        with monkeypatch.context() as patch:
            patch.setattr(qem, "BLOCK_BYTES", 1000)
            qem.save_streamed(path, source, {})
        header, start = read_header(path)
        expected = bytearray(header["bytes"])
        for chunk, entry in zip(source.chunks, header["chunks"]):
            for array, span in zip(chunk.arrays, entry["arrays"]):
                expected[span["offset"] : span["offset"] + array.nbytes] = cp.asnumpy(array).tobytes()
        assert path.read_bytes()[start:] == bytes(expected)
    finally:
        source.release()
    with io.load(path, backend="cuda", verbose=False) as reopened:
        assert bool(cp.all(reopened.data.decode_scan_range_device(0, 1800) == values))


def test_resident_read_matches_numpy_region():
    """The public read contract restores requested CUDA rows and columns."""
    import torch

    shape = (5, 6, 4, 7)
    values = cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 251
    valid = np.ones(shape[2:], dtype=bool)
    valid[2, 4] = False
    source = StreamedCounts(shape, np.uint16, valid)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    loaded = Dataset4dstemGPU(
        source,
        {
            "working_shape": shape,
            "working_dtype": "uint16",
            "representation": "encoded",
        },
    )
    try:
        observed = loaded.read(
            scan_region=(1, 4, 2, 6),
            detector_region=(1, 4, 3, 7),
        )
        assert observed.device.type == "cuda"
        assert observed.dtype == torch.uint16
        expected = values.get()[1:4, 2:6, 1:4, 3:7]
        np.testing.assert_array_equal(observed.cpu().numpy(), expected)
    finally:
        loaded.close()


def test_detector_region_read_decodes_only_the_cropped_counts(monkeypatch):
    """A detector crop decodes the cropped pixels only: whole scan rows straight into the result, others in bounded blocks.

    Decoding the whole region before cropping held a full-detector copy of every
    requested scan position (19 GB for a 512 x 512 x 192 x 192 uint16 scan just to
    keep a 110 x 110 bright-field disk).
    """
    from quantem.gpu.io import read

    shape = (7, 5, 6, 8)
    values = cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 251
    source = StreamedCounts(shape, np.uint16)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    frame_bytes = 6 * 8 * 2
    monkeypatch.setattr(read, "_BLOCK_BYTES", 2 * 3 * frame_bytes)
    decoded = []
    decode = source.decode_scan_range_device
    def record_decode(first, stop, *, detector_region=None):
        block = decode(first, stop, detector_region=detector_region)
        decoded.append(block.nbytes)
        return block

    monkeypatch.setattr(source, "decode_scan_range_device", record_decode)
    loaded = Dataset4dstemGPU(
        source,
        {"working_shape": shape, "working_dtype": "uint16", "representation": "encoded"},
    )
    try:
        for scan_region in ((0, 7, 0, 5), (1, 6, 1, 4)):
            decoded.clear()
            observed = loaded.read(scan_region=scan_region, detector_region=(1, 5, 2, 7))
            row0, row1, column0, column1 = scan_region
            expected = values.get()[row0:row1, column0:column1, 1:5, 2:7]
            np.testing.assert_array_equal(observed.cpu().numpy(), expected)
            assert sum(decoded) == expected.nbytes
            if (column0, column1) == (0, shape[1]):
                assert len(decoded) == 1
            else:
                assert len(decoded) > 1
                assert max(decoded) <= read._BLOCK_BYTES
    finally:
        loaded.close()


@pytest.mark.parametrize("layout", ["streamed", "paired"])
def test_reads_of_scans_not_held_raise(layout):
    """Scans not yet appended, freed, or released raise instead of returning stale device memory."""
    from quantem.gpu.resident.cuda.paired import PairedCounts

    shape = (4, 512, 8, 8)
    values = (cp.arange(np.prod(shape), dtype=cp.uint64) % 4093).astype(cp.uint16)
    flat = values.reshape(-1, *shape[2:])
    source = (PairedCounts if layout == "paired" else StreamedCounts)(shape, "uint16")
    for first in (0, 512, 1024):
        source.append(cp.ascontiguousarray(flat[first:first + 512]))
    # Fill the memory pool with a pattern a stale read would return.
    cp.full(flat.shape, 0xABCD, cp.uint16)
    assert bool(cp.all(source.decode_scan_range_device(1000, 1536) == flat[1000:1536]))
    for first, stop in ((1500, 1600), (1536, 2048)):
        with pytest.raises(ValueError):
            source.decode_scan_range_device(first, stop)
    source.release_scans_before(512)
    with pytest.raises(ValueError):
        source.decode_scan_range_device(100, 600)
    assert bool(cp.all(source.decode_scan_range_device(512, 600) == flat[512:600]))
    source.release()
    with pytest.raises(ValueError):
        source.decode_scan_range_device(600, 700)


@pytest.mark.parametrize("layout", ["streamed", "paired"])
def test_detector_total_covers_the_whole_scan_or_raises(layout):
    """The total kept while encoding and the total of chunks that arrive encoded are exact; scans arriving or freed raise."""
    from quantem.gpu.resident.cuda.paired import PairedCounts

    shape = (4, 512, 8, 8)
    flat = (cp.arange(np.prod(shape), dtype=cp.uint64) * 7919 % 65536).astype(cp.uint16).reshape(-1, *shape[2:])
    valid = np.ones(shape[2:], bool)
    valid[2, 3] = False
    expected = (flat.sum(axis=0, dtype=cp.uint64) * cp.asarray(valid)).get()
    owner = PairedCounts if layout == "paired" else StreamedCounts
    source = owner(shape, "uint16", valid)
    for first in range(0, 2048, 512):
        with pytest.raises(ValueError, match="not resident"):
            source.detector_total_device()
        source.append(cp.ascontiguousarray(flat[first:first + 512]))
    np.testing.assert_array_equal(source.detector_total_device().get(), expected)
    # chunks that arrive encoded, as from a file, carry no total: the first request adds their streams
    reopened = owner(shape, "uint16", valid)
    reopened.chunks.extend(source.chunks)
    reopened.ready_scans = source.ready_scans
    np.testing.assert_array_equal(reopened.detector_total_device().get(), expected)
    np.testing.assert_array_equal(reopened.detector_total_device().get(), expected)
    source.release_scans_before(1024)
    with pytest.raises(ValueError, match="released"):
        source.detector_total_device()
