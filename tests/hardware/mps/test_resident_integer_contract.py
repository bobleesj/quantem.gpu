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
def test_mps_native_uint16_frozen_detector_contract(monkeypatch, large_sum) -> None:
    if sys.platform != "darwin":
        pytest.skip("Raw Metal detector products require macOS.")
    pytest.importorskip("Metal")
    from quantem.gpu.detector.backends.mps.kernels import ChunkedFrames
    from quantem.gpu.io.backends.mps import dense as mps

    if mps._device is None:
        pytest.skip("A Metal device is not available.")
    case = _fixture()
    if large_sum:
        large = case["large_sum"]
        source = np.full(large["shape"], large["fill"], dtype=np.uint16)
        for override in large["overrides"]:
            source.reshape(-1)[override["flat_index"]] = override["value"]
    else:
        source = _working_source(case)

    # Record only these small test-owned buffers for explicit teardown. Allocation
    # and every Metal kernel remain real; no compute result is replaced or mocked.
    allocated = []
    allocate = mps._metal_buffer_alloc

    def owned_allocation(size):
        buffer = allocate(size)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(mps, "_metal_buffer_alloc", owned_allocation)
    try:
        buffer = owned_allocation(source.nbytes)
        chunk = mps._mtl_array_from_buffer(
            buffer, np.uint16,
            (source.shape[0] * source.shape[1], *source.shape[2:]),
        )
        chunk[:] = source.reshape(chunk.shape)
        frames = ChunkedFrames([chunk], torch_compat=False)
        session = detector.prepare(frames)
        assert session._backend.device == "mps"
        assert session._backend._cf is frames
        assert frames.dtype == np.dtype("uint16")
        assert not session._backend._auto_fast
        if large_sum:
            _assert_large_sum(session, large)
        else:
            _assert_frozen_products(session, case)
            _assert_invalid_requests(session, case)
            _assert_selected_frame_is_independent(session, case)
            indices = [row * 3 + col for row, col in case["selected_scan_row_columns"]]
            with pytest.raises(NotImplementedError, match="no exact selected-frame"):
                session.reduce_frames_exact(indices)
        np.testing.assert_array_equal(np.asarray(chunk), source.reshape(chunk.shape))
    finally:
        for buffer in reversed(allocated):
            mps._release_metal_buffer(buffer)
