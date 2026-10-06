"""Frozen native-uint16 products through the public raw-Metal session.

This is a dense resident-buffer test, not a compact-v3 uint8 or file-load test.
Only the explicitly requested small products are read back for comparison.
"""

import sys

import numpy as np
import pytest

from quantem.gpu import detector
from tests.parity.resident_integer_assertions import (
    _assert_frozen_products,
    _assert_invalid_requests,
    _assert_large_sum,
    _assert_selected_frame_is_independent,
)
from tests.parity.resident_integer_oracle import _fixture, _working_source


@pytest.mark.parametrize("large_sum", [False, True])
def test_mps_native_uint16_frozen_detector_contract(large_sum) -> None:
    if sys.platform != "darwin":
        pytest.skip("Raw Metal detector products require macOS.")
    pytest.importorskip("Metal")
    from quantem.gpu.device import metal_runtime as mps
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    if mps.metal_module().MTLCreateSystemDefaultDevice() is None:
        pytest.skip("A Metal device is not available.")
    case = _fixture()
    if large_sum:
        large = case["large_sum"]
        source = np.full(large["shape"], large["fill"], dtype=np.uint16)
        for override in large["overrides"]:
            source.reshape(-1)[override["flat_index"]] = override["value"]
    else:
        source = _working_source(case)

    # The test owns the frame buffer; the virtual image releases its own buffers.
    buffer = mps.allocate_shared(source.nbytes, "test frames")
    try:
        chunk = mps.numpy_view(buffer, np.uint16, source.size).reshape(
            source.shape[0] * source.shape[1], *source.shape[2:]
        ).view(mps.SharedArray)
        chunk._mtl = buffer
        chunk[:] = source.reshape(chunk.shape)
        frames = ChunkedFrames([chunk])
        session = detector.prepare(frames)
        assert session._backend.device == "mps"
        assert session._backend.frames is frames
        assert frames.dtype == np.dtype("uint16")
        assert not session._backend._auto_fast
        if large_sum:
            _assert_large_sum(session, large)
        else:
            _assert_frozen_products(session, case)
            _assert_invalid_requests(session, case)
            _assert_selected_frame_is_independent(session, case)
            indices = [row * 3 + col for row, col in case["selected_scan_row_columns"]]
            # Raw Metal now sums selected frames exactly, the frozen value every backend gives.
            np.testing.assert_array_equal(
                session.reduce_frames_exact(indices).reshape(-1), case["selected_scan_sum_u64"]
            )
        np.testing.assert_array_equal(np.asarray(chunk), source.reshape(chunk.shape))
    finally:
        mps.release_buffer(buffer)
