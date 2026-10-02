"""Scientist-facing basic indexing of encoded acquisitions."""
import os

import numpy as np
import pytest

from quantem.gpu import io


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_encoded_indexing(tmp_path, dtype):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    values = (np.arange(4 * 5 * 8 * 10).reshape(4, 5, 8, 10) % 101).astype(dtype)
    if dtype == np.float32:
        values = values / 8 - 3
    path = tmp_path / "input.npy"
    np.save(path, values)
    selections = [
        (1, 2), -1, (1, 2, 3, 4), (), Ellipsis,
        (Ellipsis, -1), (slice(1, 3), slice(2, 5)),
        (1, 2, slice(2, 7), slice(3, 9)),
        (slice(None, None, 2), slice(None), slice(None, None, -2), -1),
        (slice(None, None, -1), slice(4, 0, -2), slice(None), slice(None)),
        (slice(3, 1), slice(None)), (slice(20, 30), 1),
        (np.int64(-1), np.int32(2)),
    ]
    with io.load(path, backend=backend, verbose=False) as data:
        for key in selections:
            actual = data[key]
            assert actual.device.type == backend
            np.testing.assert_array_equal(actual.cpu().numpy(), values[key])
        for key in [4, -5, (0, 0, 0, 0, 0), (Ellipsis, Ellipsis)]:
            with pytest.raises(IndexError):
                data[key]
        for key in [True, [0, 1], None, 1.5]:
            with pytest.raises(TypeError):
                data[key]
        with pytest.raises(ValueError):
            data[::0]
        saved = tmp_path / "saved.qem"
        io.save(saved, data)
    series = io.load([saved, saved], backend=backend, stack=False, verbose=False)
    try:
        np.testing.assert_array_equal(series[1][1, 2].cpu().numpy(), values[1, 2])
        assert isinstance(series[1].metadata, dict)
    finally:
        for data in series:
            data.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_detector_pixels_across_ans_intervals(tmp_path, dtype):
    """Detector crops preserve scans across 512-frame streams and a short tail."""
    if os.environ.get("QEM_TEST_BACKEND") != "cuda":
        pytest.skip("Selective ANS stream decoder is CUDA-specific.")
    values = np.random.default_rng(5).integers(
        0, 200, size=(35, 33, 8, 10), dtype=dtype
    )
    path = tmp_path / "intervals.npy"
    np.save(path, values)
    with io.load(path, backend="cuda", verbose=False) as data:
        for key in [
            (slice(None), slice(None), 3, 6),
            (slice(None), slice(None), slice(2, 7), slice(4, 9)),
            (slice(14, 35), slice(5, 30), slice(2, 7), slice(4, 9)),
            (slice(None, None, -2), slice(None), -1, 0),
        ]:
            np.testing.assert_array_equal(data[key].cpu().numpy(), values[key])
