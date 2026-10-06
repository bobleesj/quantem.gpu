"""Shared assertions over frozen values; no production oracle or runner choice."""

import numpy as np
import pytest


def _assert_frozen_products(session, case: dict) -> None:
    """Check full masks and interleaved selected DPs without recapturing values."""
    assert session.scan_shape == tuple(case["shape"][:2])
    assert session.detector_shape == tuple(case["shape"][2:])
    masks = {request["id"]: request for request in case["detector_masks"]}
    for name in case["request_order"]:
        request = masks[name]
        mask = np.asarray(request["mask_u8"], dtype=np.uint8).reshape(3, 4)
        result = session.masked_sum_exact(mask)
        assert result.dtype == np.dtype("uint64")
        np.testing.assert_array_equal(
            result.reshape(-1), request["expected_sum_u64"]
        )
        for position, expected in zip(
            case["selected_scan_row_columns"],
            case["expected_selected_frames_u16"],
            strict=True,
        ):
            row, column = position
            diffraction = session.frame(row * session.scan_shape[1] + column)
            assert diffraction.dtype == np.dtype("uint16")
            np.testing.assert_array_equal(diffraction.reshape(-1), expected)


def _assert_invalid_requests(session, case: dict) -> None:
    """Invalid requests must fail before replacing the source or cached totals."""
    full = np.ones(case["shape"][2:], dtype=np.uint8)
    expected = next(
        request["expected_sum_u64"]
        for request in case["detector_masks"] if request["id"] == "full"
    )

    def assert_intact() -> None:
        np.testing.assert_array_equal(
            session.masked_sum_exact(full).reshape(-1), expected
        )

    for index in (-1, session.num_frames):
        with pytest.raises(IndexError):
            session.frame(index)
        assert_intact()
    with pytest.raises(ValueError, match="Detector mask shape"):
        session.masked_sum_exact(np.ones((1, 1), dtype=np.uint8))
    assert_intact()
    for malformed in (np.nan, np.inf, 0.5, -1, 2):
        mask = full.astype(np.float64)
        mask[0, 1] = malformed
        with pytest.raises(ValueError, match="binary"):
            session.masked_sum_exact(mask)
        assert_intact()
    for dtype in (np.bool_, np.uint8, np.float32):
        np.testing.assert_array_equal(
            session.masked_sum_exact(full.astype(dtype)).reshape(-1), expected
        )


def _assert_selected_frame_is_independent(session, case: dict) -> None:
    """Editing a returned DP must not alter source counts or cached totals."""
    frame = session.frame(1)
    original = np.asarray(case["expected_working_frames_u16"][1], dtype=np.uint16)
    np.testing.assert_array_equal(frame.reshape(-1), original)
    frame[:] = 0
    np.testing.assert_array_equal(session.frame(1).reshape(-1), original)
    _assert_frozen_products(session, case)


def _assert_large_sum(session, case: dict) -> None:
    """Preserve the frozen odd integer total above float32's exact range."""
    assert session.scan_shape == tuple(case["shape"][:2])
    assert session.detector_shape == tuple(case["shape"][2:])
    result = session.masked_sum_exact(np.ones(case["shape"][2:], dtype=np.uint8))
    assert result.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(result.reshape(-1), case["expected_full_sum_u64"])
    assert result[0, 1] != np.float32(result[0, 1])
