"""CUDA tests for exact, bounded resident ANS reads."""

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)

from quantem.gpu import io
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
    loaded = io.FourDSTEMData(
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
        expected[..., 1, 1] = 0
        np.testing.assert_array_equal(observed.cpu().numpy(), expected)
    finally:
        loaded.close()
