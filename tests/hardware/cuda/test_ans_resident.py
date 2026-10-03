"""CUDA tests for exact, bounded resident ANS reads."""

from quantem.gpu.io.models import create_dataset

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)

from quantem.gpu._compact.streamed import StreamedCounts


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


def test_resident_read_matches_numpy_region():
    """The public read contract restores requested CUDA rows and columns."""
    import torch

    shape = (5, 6, 4, 7)
    values = cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 251
    valid = np.ones(shape[2:], dtype=bool)
    valid[2, 4] = False
    source = StreamedCounts(shape, np.uint16, valid)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    loaded = create_dataset(
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


def test_detector_region_read_decodes_bounded_scan_blocks(monkeypatch):
    """A detector crop never decodes more than one bounded scratch block.

    Decoding the whole region before cropping held a full-detector copy of every
    requested scan position (19 GB for a 512 x 512 x 192 x 192 uint16 scan just to
    keep a 110 x 110 bright-field disk).
    """
    from quantem.gpu.io import _read

    shape = (7, 5, 6, 8)
    values = cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 251
    source = StreamedCounts(shape, np.uint16)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    frame_bytes = 6 * 8 * 2
    monkeypatch.setattr(_read, "_BLOCK_BYTES", 2 * 3 * frame_bytes)
    decoded = []
    decode = source.decode_scan_range_device
    def record_decode(first, stop, *, detector_region=None):
        block = decode(first, stop, detector_region=detector_region)
        decoded.append(block.nbytes)
        return block

    monkeypatch.setattr(source, "decode_scan_range_device", record_decode)
    loaded = create_dataset(
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
            assert len(decoded) > 1
            assert max(decoded) <= _read._BLOCK_BYTES
    finally:
        loaded.close()
