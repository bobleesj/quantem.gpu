"""Complete H5/ANS workflows, count integrity and translated detector products."""

import h5py
import numpy as np
import pytest

from quantem.gpu import detector, io


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_h5_opens_encoded_and_mixes_with_original_ans(tmp_path, dtype):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("Requires an admitted CUDA device.")
    rng = np.random.default_rng(2026)
    raw = rng.poisson(2, (1, 513, 11, 7)).astype(dtype)
    raw[:, ::3, 0, 0] = np.iinfo(dtype).max
    raw[:, :, 1, 0] = np.iinfo(dtype).max
    raw[:, :, 2, 0] = 0
    h5 = tmp_path / "acquisition.h5"
    with h5py.File(h5, "w") as handle:
        handle["entry/data/data"] = raw
    ans = tmp_path / "acquisition.ans"
    io.save(ans, raw, format="quantem", compression="ans", backend="cpu")
    first = io.load(h5, backend="cuda", representation="ans", apply_mask=False)
    second = io.load(ans, backend="cuda")
    original_pointers = [a.data.ptr for a in second.data._arrays]
    assert first.representation is io.DataRepresentation.ANS
    reconstructed = np.concatenate(
        [first.data.decode_chunk(i).get() for i in range(len(first.data.chunks))]
    ).reshape(raw.shape)
    np.testing.assert_array_equal(reconstructed, raw)
    session = detector.prepare([first, second])
    assert session.backend_metadata["query_abi"] == "streamed-spatial-counts-v1"
    assert [a.data.ptr for a in second.data._arrays] == original_pointers
    rr, cc = np.indices(raw.shape[-2:])
    retained = None
    for row, col, inner, outer in [
        (5.5, 3.125, 1.9, 4.5),
        (5.6, 3.2, 1.8, 4.6),
        (-8.7, 7.1, 0.1, 14.75),
        (55, 20, 1, 2),
        (0, 0, 0, 100),
    ]:
        radius2 = (rr - row) ** 2 + (cc - col) ** 2
        mask = (radius2 >= inner**2) & (radius2 < outer**2)
        actual = session.masked_sum(mask, output="native")
        expected = (raw * mask).sum((-2, -1), dtype=np.uint64)
        np.testing.assert_array_equal(
            actual.get(), np.broadcast_to(expected, actual.shape)
        )
        if retained is not None:
            np.testing.assert_array_equal(*retained)
        retained = actual.get(), actual.get().copy()
        actual.fill(7)  # Caller-owned output must not become the next baseline.
    for index in (0, 255, 256, 511, 512, 13):
        output = session.frame(index, output="native").get()
        np.testing.assert_array_equal(
            output, np.broadcast_to(raw.reshape(513, 11, 7)[index], output.shape)
        )
        assert output.dtype == dtype
    single = detector.prepare(first)
    assert single.series_shape == ()
    np.testing.assert_array_equal(single.frame(512, output="native").get(), raw[0, 512])
    with pytest.raises(ValueError, match="overlap"):
        session.masked_sum(
            np.ones((11, 7), bool), output="native", out=session._backend.previous
        )
    first.close()
    # Active sessions borrow buffers and remain usable after the loading owner closes.
    np.testing.assert_array_equal(session.frame(0, output="native").get()[0], raw[0, 0])


def test_index_sum_preserves_uint64_bound(tmp_path):
    pytest.importorskip("cupy")
    raw = np.full((1, 1, 257, 257), 65535, np.uint16)
    path = tmp_path / "high-counts.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
    loaded = io.load(path, backend="cuda", representation="ans", apply_mask=False)
    session = detector.prepare(loaded)
    output = session.masked_sum(np.ones((257, 257), bool), output="native")
    assert output.dtype == np.uint64
    np.testing.assert_array_equal(output.get(), raw.sum((-2, -1), dtype=np.uint64))


def test_mixed_dense_and_streamed_counts_keep_native_shapes(tmp_path):
    cp = pytest.importorskip("cupy")
    raw = (np.arange(513 * 6, dtype=np.uint16) % 17).reshape(1, 513, 2, 3)
    path = tmp_path / "counts.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
    encoded = io.load(path, backend="cuda", representation="ans", apply_mask=False)
    dense = cp.asarray(raw)
    session = detector.prepare([encoded, dense])
    for index in (0, 511, 512):
        np.testing.assert_array_equal(
            session.frame(index, output="native").get(),
            np.broadcast_to(raw[0, index], (2, 2, 3)),
        )
    mask = np.array([[1, 0, 1], [0, 1, 0]], bool)
    np.testing.assert_array_equal(
        session.masked_sum(mask, output="native").get(),
        np.broadcast_to((raw * mask).sum((-2, -1), dtype=np.uint64), (2, 1, 513)),
    )


def test_prepared_packed_and_streamed_h5_share_a_joint_query(tmp_path):
    import hashlib
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location(
        "packed_fixture", Path(__file__).parents[2] / "contracts/io/test_compact_h5.py"
    )
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    raw = (np.arange(32 * 6, dtype=np.uint16) % 12).reshape(32, 6)
    raw[:, 2] = 65535
    packed_path = tmp_path / "packed.h5"
    fixture._write_v3_fixture(
        packed_path, raw, detector_shape=(2, 3), masked_pixels=(2,), scan_shape=(4, 8)
    )
    ordinary = tmp_path / "ordinary.h5"
    with h5py.File(ordinary, "w") as handle:
        handle["entry/data/data"] = raw.reshape(4, 8, 2, 3)
    packed = io.load(
        packed_path,
        backend="cuda",
        expected_source_sha256=hashlib.sha256(packed_path.read_bytes()).hexdigest(),
    )
    streamed = io.load(ordinary, backend="cuda", representation="ans", apply_mask=False)
    session = detector.prepare([packed, streamed])
    for index in (0, 7, 31):
        np.testing.assert_array_equal(
            session.frame(index, output="native").get(),
            np.broadcast_to(raw[index].reshape(2, 3), (2, 2, 3)),
        )
    np.testing.assert_array_equal(
        session.masked_sum(np.ones((2, 3), bool), output="native").get(),
        np.stack(
            [np.delete(raw, 2, axis=1).sum(-1).reshape(4, 8), raw.sum(-1).reshape(4, 8)]
        ),
    )


def test_one_wide_scan_uses_actual_length_encoding_scratch(tmp_path):
    raw = np.full((1, 1, 2048, 2048), 37, np.uint16)
    path = tmp_path / "wide.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
    loaded = io.load(path, backend="cuda", representation="ans", apply_mask=False)
    np.testing.assert_array_equal(
        loaded.data.decode_chunk(0).get(), raw.reshape(1, 2048, 2048)
    )


def test_sparse_events_and_rare_counts_reconstruct_every_scan(tmp_path):
    raw = np.zeros((1, 1025, 1, 8), np.uint16)
    raw[0, [0, 511, 512, 1023, 1024], 0, 0] = [1, 128, 32, 129, 65535]
    raw[0, [4, 767], 0, 1] = [65535, 40000]
    raw[0, :, 0, 2] = 65535
    raw[0, ::16, 0, 3] = 31
    raw[0, ::19, 0, 4] = 32
    raw[0, :, 0, 5] = np.arange(1025) * 61
    path = tmp_path / "sparse-rare.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
    loaded = io.load(path, backend="cuda", representation="ans", apply_mask=False)
    decoded = np.concatenate(
        [loaded.data.decode_chunk(i).get() for i in range(len(loaded.data.chunks))]
    )
    np.testing.assert_array_equal(decoded.reshape(raw.shape), raw)
    session = detector.prepare(loaded)
    for mask in [
        np.ones((1, 8), bool),
        np.arange(8).reshape(1, 8) % 2 == 0,
        np.zeros((1, 8), bool),
    ]:
        np.testing.assert_array_equal(
            session.masked_sum(mask, output="native").get(),
            (raw * mask).sum((-2, -1), dtype=np.uint64),
        )
    for i in [0, 511, 512, 767, 1023, 1024]:
        np.testing.assert_array_equal(
            session.frame(i, output="native").get(), raw[0, i]
        )
