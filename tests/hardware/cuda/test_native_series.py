"""Exact joint native queries across dense and independently encoded sources."""

import numpy as np
import pytest
from quantem.gpu import io, detector


def test_native_series_owns_outputs_and_retains_encoded_storage(tmp_path):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    rng = np.random.default_rng(73)
    raw = rng.poisson(3, (4, 9, 7, 11)).astype(np.uint16)
    raw[0, 0, 0, 0] = 65535
    excluded = [0, 13, 72]
    file = tmp_path / "real.ans"
    io.save(
        file,
        io.FourDSTEMData(raw, {"excluded_detector_pixels": excluded}),
        format="quantem",
        compression="ans",
        backend="cpu",
    )
    ans = io.load(file, backend="cuda")
    packed = io.load(file, backend="cuda", representation="packed")
    dense = io.FourDSTEMData(cp.asarray(raw), {"excluded_detector_pixels": excluded})
    session = detector.prepare([ans, packed, dense])
    assert session.series_shape == (3,) and session.scan_shape == (4, 9)
    assert session.backend_metadata["sum_dtype"] == "uint32"
    rr, cc = np.indices((7, 11), dtype=np.float64)
    valid = np.ones((7, 11), bool)
    valid.reshape(-1)[excluded] = False
    previous = None
    for row, col, inner, outer in [
        (3, 5, 0, 4),
        (1.25, 8.75, 0.4, 6.125),
        (-2.5, 20, 0, 0.5),
        (0, 0, 0, 100),
        (50, 50, 1, 2),
        (3, 5, 4, 4),
    ]:
        d2 = (rr - row) ** 2 + (cc - col) ** 2
        mask = (d2 >= inner**2) & (d2 <= outer**2)
        result = session.masked_sum(mask, output="native")
        expected = np.where(mask & valid, raw, 0).sum((-2, -1), dtype=np.uint64)
        np.testing.assert_array_equal(
            result.get(), np.broadcast_to(expected, (3, 4, 9))
        )
        if previous is not None:
            np.testing.assert_array_equal(previous[0].get(), previous[1])
        previous = result, result.get()
    for position in (0, 1, 15, 35, 2):
        np.testing.assert_array_equal(
            session.frame(position, output="native").get(),
            np.broadcast_to(raw.reshape(-1, 7, 11)[position], (3, 7, 11)),
        )
    assert session.timings["query_launches"] == 1
    with pytest.raises(ValueError, match="overlap"):
        session.masked_sum(
            np.ones((7, 11), bool),
            output="native",
            out=dense.data.ravel().view(cp.uint32)[:108].reshape(3, 4, 9),
        )
    assert ans.metadata["representation"] == "ans"


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
