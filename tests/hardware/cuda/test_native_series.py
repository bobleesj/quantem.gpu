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


def test_packed_h5_joint_queries_restore_excluded_raw_counts(tmp_path):
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path

    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    spec = spec_from_file_location(
        "packed_fixture", Path(__file__).parents[2] / "contracts/io/test_compact_h5.py"
    )
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    raw = (np.arange(32 * 6, dtype=np.uint16) % 12).reshape(32, 6)
    raw[:, 2] = 65535
    file = tmp_path / "packed.h5"
    fixture._write_v3_fixture(
        file, raw, detector_shape=(2, 3), masked_pixels=(2,), scan_shape=(4, 8)
    )
    import hashlib

    loaded = io.load(
        file,
        backend="cuda",
        expected_source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
    )
    session = detector.prepare([loaded, loaded])
    for index in (0, 7, 31):
        np.testing.assert_array_equal(
            session.frame(index, output="native").get(),
            np.broadcast_to(raw[index].reshape(2, 3), (2, 2, 3)),
        )
    np.testing.assert_array_equal(
        session.masked_sum(np.ones((2, 3), bool), output="native").get(),
        np.broadcast_to(np.delete(raw, 2, axis=1).sum(-1).reshape(4, 8), (2, 4, 8)),
    )
