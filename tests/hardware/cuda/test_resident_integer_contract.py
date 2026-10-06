"""Frozen native-uint16 products through the public resident CUDA session."""

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


def test_cuda_native_uint16_frozen_detector_contract() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is not available.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not available: {exc}")

    case = _fixture()
    source = _working_source(case)
    resident = cp.asarray(source)
    assert resident.dtype == cp.uint16
    session = detector.prepare(resident)
    assert session._backend.device == "cuda"
    assert session._backend._data is resident
    _assert_frozen_products(session, case)
    _assert_invalid_requests(session, case)
    _assert_selected_frame_is_independent(session, case)
    indices = [row * 3 + col for row, col in case["selected_scan_row_columns"]]
    np.testing.assert_array_equal(
        session.reduce_frames_exact(indices).reshape(-1),
        case["selected_scan_sum_u64"],
    )
    np.testing.assert_array_equal(resident.get(), source)

    large = case["large_sum"]
    source = np.full(large["shape"], large["fill"], dtype=np.uint16)
    for override in large["overrides"]:
        source.reshape(-1)[override["flat_index"]] = override["value"]
    resident_large = cp.asarray(source)
    assert resident_large.dtype == cp.uint16
    _assert_large_sum(detector.prepare(resident_large), large)
