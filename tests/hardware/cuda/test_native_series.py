"""Exact joint native queries across dense and independently encoded sources."""

import numpy as np
import pytest
from quantem.gpu import io, detector


def test_uint64_series_sum_never_truncates_high_counts():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    data = cp.full((1, 1, 257, 257), 65535, cp.uint16)
    session = detector.prepare([data, data])
    value = session.masked_sum(np.ones((257, 257), bool), output="native")
    assert value.dtype == cp.uint64
    np.testing.assert_array_equal(
        value.get(), np.full((2, 1, 1), 257 * 257 * 65535, np.uint64)
    )


def test_original_h5_joint_queries_preserve_raw_counts(tmp_path):
    import h5py

    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    raw = (np.arange(32 * 6, dtype=np.uint16) % 12).reshape(32, 6)
    raw[:, 2] = 65535
    file = tmp_path / "original.h5"
    with h5py.File(file, "w") as handle:
        handle["entry/data/data"] = raw.reshape(4, 8, 2, 3)
    loaded = io.load(file, backend="cuda", apply_mask=False)
    session = detector.prepare([loaded, loaded])
    for index in (0, 7, 31):
        np.testing.assert_array_equal(
            session.frame(index, output="native").get(),
            np.broadcast_to(raw[index].reshape(2, 3), (2, 2, 3)),
        )
    np.testing.assert_array_equal(
        session.masked_sum(np.ones((2, 3), bool), output="native").get(),
        np.broadcast_to(raw.sum(-1).reshape(4, 8), (2, 4, 8)),
    )
