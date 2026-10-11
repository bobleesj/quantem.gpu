"""Complete H5/ANS workflows, count integrity and translated detector products."""

import h5py
import numpy as np
import pytest

from quantem.gpu import detector, io


def _median_corrected(raw, pixel_mask):
    expected = raw.copy()
    height, width = pixel_mask.shape
    for row, column in np.argwhere(pixel_mask != 0):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, column + dc
                if (
                    (dr or dc)
                    and 0 <= rr < height
                    and 0 <= cc < width
                    and pixel_mask[rr, cc] == 0
                ):
                    neighbors.append(raw[..., rr, cc])
        expected[..., row, column] = np.median(
            np.stack(neighbors, axis=-1), axis=-1
        ).astype(raw.dtype)
    return expected


@pytest.fixture(autouse=True)
def cuda_device():
    """Run these scientific workflows only when a CUDA device is available."""
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("Requires an admitted CUDA device.")


def test_index_sum_preserves_uint64_bound(tmp_path):
    pytest.importorskip("cupy")
    raw = np.full((1, 1, 257, 257), 65535, np.uint16)
    path = tmp_path / "high-counts.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
    loaded = io.load(path, backend="cuda", representation="encoded", apply_mask=False)
    session = detector.prepare(loaded)
    output = session.masked_sum(np.ones((257, 257), bool), output="native")
    assert output.dtype == np.uint64
    np.testing.assert_array_equal(output.get(), raw.sum((-2, -1), dtype=np.uint64))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize(
    "representation", ["encoded", None], ids=["encoded", "default-encoded"]
)
def test_h5_defaults_to_gpu_median_hot_pixel_correction(
    tmp_path, dtype, representation
):
    raw = (np.arange(2 * 5 * 5 * 5).reshape(2, 5, 5, 5) * 7 % 251).astype(dtype)
    mask = np.zeros((5, 5), np.uint8)
    mask[0, 0] = 16
    mask[2, 3] = 20
    raw[..., mask != 0] = np.iinfo(dtype).max
    expected = _median_corrected(raw, mask)
    path = tmp_path / "hot-pixels.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask

    loaded = io.load(
        path,
        backend="cuda",
        representation=representation,
        apply_mask=False,
        verbose=False,
    )
    try:
        assert loaded.representation is io.DataRepresentation.ENCODED
        assert loaded.metadata["source_path"] == str(path)
        correction = loaded.metadata["hot_pixel_correction"]
        assert correction["method"] == "median"
        assert correction["pixel_count"] == 2
        assert correction["coordinates_row_column"] == [[0, 0], [2, 3]]
        assert correction["applied"] is True
        session = detector.prepare(loaded)
        output = "native"
        for index in (0, 9):
            frame = session.frame(index, output=output)
            np.testing.assert_array_equal(
                frame.get() if output == "native" else frame,
                expected.reshape(-1, 5, 5)[index],
            )
        mean_dp = session.mean_dp(output=output).get()
        np.testing.assert_allclose(
            mean_dp,
            expected.mean(axis=(0, 1)),
            rtol=0,
            atol=1e-5,
        )
    finally:
        loaded.close()


@pytest.mark.parametrize("method", ["zero", "none"])
def test_h5_ans_hot_pixel_correction_overrides(tmp_path, method):
    raw = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(1, 3, 4, 4)
    raw[..., 1, 2] = np.iinfo(np.uint16).max
    mask = np.zeros((4, 4), np.uint8)
    mask[1, 2] = 20
    path = tmp_path / f"hot-pixels-{method}.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask

    loaded = io.load(
        path,
        backend="cuda",
        representation="encoded",
        apply_mask=False,
        hot_pixel_correction=method,
        verbose=False,
    )
    try:
        decoded = loaded.data.decode_scan_range_device(0, 3).get().reshape(raw.shape)
        expected = raw.copy()
        if method == "zero":
            expected[..., 1, 2] = 0
        np.testing.assert_array_equal(decoded, expected)
        assert loaded.metadata["hot_pixel_correction"]["method"] == method
        assert loaded.metadata["hot_pixel_correction"]["applied"] is (
            method == "zero"
        )
    finally:
        loaded.close()


def test_two_streamed_h5_sources_share_a_joint_query(tmp_path):
    raw = (np.arange(32 * 6, dtype=np.uint16) % 12).reshape(4, 8, 2, 3)
    other = raw + 100
    loaded = []
    for name, values in (("first", raw), ("second", other)):
        path = tmp_path / f"{name}.h5"
        with h5py.File(path, "w") as handle:
            handle["entry/data/data"] = values
        loaded.append(io.load(path, backend="cuda", apply_mask=False))
    session = detector.prepare(loaded)
    try:
        for index in (0, 7, 31):
            np.testing.assert_array_equal(
                session.frame(index, output="native").get(),
                np.stack([raw.reshape(32, 2, 3)[index], other.reshape(32, 2, 3)[index]]),
            )
        np.testing.assert_array_equal(
            session.masked_sum(np.ones((2, 3), bool), output="native").get(),
            np.stack([raw.sum((-2, -1)), other.sum((-2, -1))]),
        )
    finally:
        session.close()
        for source in loaded:
            source.close()


def test_one_wide_scan_uses_actual_length_encoding_scratch(tmp_path):
    raw = np.full((1, 1, 2048, 2048), 37, np.uint16)
    path = tmp_path / "wide.h5"
    with h5py.File(path, "w") as handle:
        handle["entry/data/data"] = raw
    loaded = io.load(path, backend="cuda", representation="encoded", apply_mask=False)
    np.testing.assert_array_equal(
        loaded.data.decode_scan_range_device(0, 1).get(), raw.reshape(1, 2048, 2048)
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
    loaded = io.load(path, backend="cuda", representation="encoded", apply_mask=False)
    decoded = loaded.data.decode_scan_range_device(0, 1025).get()
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


@pytest.mark.parametrize("method", ["median", "zero"])
def test_uint32_master_sentinels_are_corrected_before_encoding(tmp_path, method):
    """Inspect corrected patterns without narrowing valid detector counts."""
    raw = (np.arange(10 * 8 * 8).reshape(10, 8, 8) * 7 % 251).astype(np.uint32)
    mask = np.zeros((8, 8), np.uint8)
    mask[0, 0] = 16
    mask[2, 3] = 20
    raw[:, mask != 0] = np.iinfo(np.uint32).max
    expected = _median_corrected(raw, mask) if method == "median" else raw.copy()
    if method == "zero":
        expected[:, mask != 0] = 0
    import hdf5plugin

    path = tmp_path / "sentinel_master.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/data/data_000001", data=raw,
                              chunks=(1, 8, 8), **hdf5plugin.Bitshuffle())
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask
    with io.load(
        path, backend="cuda", representation="encoded", scan_shape=(2, 5),
        apply_mask=False, hot_pixel_correction=method, verbose=False,
    ) as loaded:
        session = detector.prepare(loaded)
        try:
            for index in (0, 9):
                np.testing.assert_array_equal(session.frame(index), expected[index])
            np.testing.assert_array_equal(
                session.masked_sum(np.ones((8, 8), bool)),
                expected.sum((-2, -1), dtype=np.uint64).reshape(2, 5),
            )
        finally:
            session.close()
        assert loaded.metadata["hot_pixel_correction"]["applied"] is True
        assert loaded.metadata["file_counts_exact"] is False
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["entry/data/data_000001"][:], raw)

    # A genuine over-range count at a valid pixel must never be clipped.
    with h5py.File(path, "r+") as handle:
        handle["entry/data/data_000001"][0, 1, 1] = 70000
    with pytest.raises(ValueError, match="counts above 65535"):
        io.load(path, backend="cuda", representation="encoded", scan_shape=(2, 5),
                apply_mask=False, hot_pixel_correction=method, verbose=False)


@pytest.mark.slow
def test_one_decode_batch_beyond_4_gib_keeps_every_frame_in_place(tmp_path):
    """Frames past byte 2^32 of one decode launch land at their own position, not wrapped to the start."""
    import cupy as cp

    from quantem.gpu.formats.hdf5.reads import FrameReader
    from quantem.gpu.io.hdf5.cuda.decode import decompress_prepared

    frames = 8200  # 8200 * 512 * 512 * 2 bytes = 4.3 GB, past 2^32
    values = cp.broadcast_to((cp.arange(frames, dtype=cp.uint16) % 997)[:, None, None], (frames, 512, 512))
    master = tmp_path / "wide_master.h5"
    io.save(str(master), cp.ascontiguousarray(values).reshape(8200, 1, 512, 512), verbose=False)
    reader = FrameReader(str(master))
    try:
        decoded = decompress_prepared(reader.prepare(np.arange(frames)), batch_bytes_target=frames * 512 * 512 * 2)
    finally:
        reader.close()
    first_values = decoded[:, 0, 0]
    assert bool(cp.array_equal(first_values, values[:, 0, 0]))
    assert bool(cp.array_equal(decoded[-1], values[-1]))


def test_truncated_lz4_block_raises_instead_of_returning_garbage(tmp_path):
    """A chunk whose LZ4 block stops early (a file still being written) refuses to decode."""
    import cupy as cp

    from quantem.gpu.formats.hdf5.reads import FrameReader
    from quantem.gpu.io.hdf5.cuda.decode import decompress_prepared

    values = cp.asarray(np.random.default_rng(5).poisson(3.0, (4, 1, 64, 64)).astype(np.uint16))
    master = tmp_path / "partial_master.h5"
    io.save(str(master), values, verbose=False)
    reader = FrameReader(str(master))
    try:
        intact = decompress_prepared(reader.prepare(np.arange(4)))
        assert bool(cp.array_equal(intact, values.reshape(4, 64, 64)))
        prepared = reader.prepare(np.arange(4))
        header = int(prepared["chunk_offsets"][2]) + int(prepared["block_starts"][prepared["block_offsets"][2]])
        size = int.from_bytes(bytes(prepared["read_buffer"][header:header + 4]), "big")
        prepared["read_buffer"][header:header + 4] = np.frombuffer((size // 2).to_bytes(4, "big"), np.uint8)
        with pytest.raises(ValueError, match="truncated or corrupt"):
            decompress_prepared(prepared)
    finally:
        reader.close()
