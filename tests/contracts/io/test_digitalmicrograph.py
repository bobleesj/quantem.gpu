"""Open calibrated camera data without interpreting survey images as scans."""

import struct
import os

import numpy as np
import pytest

pytest.importorskip("ncempy")


def write_dm4(path, values):
    """Write a small standard DM4 image with a survey followed by diffraction."""
    def entry(name, kind, contents):
        label = name.encode()
        return struct.pack(">BH", kind, len(label)) + label + struct.pack(">Q", len(contents)) + contents

    def group(name, entries):
        body = b"\x01\x01" + struct.pack(">Q", len(entries)) + b"".join(entries)
        return entry(name, 20, body)

    def tag(name, types, data):
        return entry(name, 21, b"%%%%" + struct.pack(">Q", len(types))
                     + b"".join(struct.pack(">Q", value) for value in types) + data)

    def scalar(name, value, code, fmt):
        return tag(name, [code], struct.pack("<" + fmt, value))

    def image(name, data, units):
        calibrations = []
        for axis, unit in enumerate(units[::-1], 1):
            text = unit.encode("utf-16le")
            calibrations.append(group(str(axis), [
                scalar("Origin", 0, 6, "f"), scalar("Scale", 0.25, 6, "f"),
                tag("Units", [20, 4, len(text) // 2], text),
            ]))
        return group(name, [group("ImageData", [
            group("Calibrations", [group("Dimension", calibrations)]),
            scalar("DataType", 6 if data.dtype == np.uint8 else 10, 5, "I"),
            group("Dimensions", [scalar(str(i), size, 5, "I")
                                  for i, size in enumerate(data.shape[::-1], 1)]),
            tag("Data", [20, 10 if data.dtype == np.uint8 else 4, data.size], data.tobytes()),
        ])])

    images = [image("1", np.zeros((3, 4), np.uint8), ["nm", "nm"]),
              image("2", values, ["nm", "nm", "1/nm", "1/nm"])]
    body = b"\x01\x01" + struct.pack(">Q", 1) + group("ImageList", images)
    path.write_bytes(struct.pack(">IQI", 4, len(body), 1) + body)
    return path


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dm4_matches_independent_reader(tmp_path, dtype):
    """Read the calibrated 4D image, retaining rectangular axes and native counts."""
    from ncempy.io import dm
    from quantem.gpu import io

    values = np.arange(5 * 7 * 12 * 16).reshape(5, 7, 12, 16).astype(dtype)
    path = write_dm4(tmp_path / "camera.dm4", values)
    info = io.inspect(path)
    with io.load(path, backend="cpu", representation="dense") as loaded:
        with dm.fileDM(path) as reference:
            expected = reference.getDataset(0)["data"]
        np.testing.assert_array_equal(loaded.data, expected)
        np.testing.assert_array_equal(loaded.data, values)
        assert info.scan_shape == (5, 7)
        assert info.detector_shape == (12, 16)
        assert loaded.metadata["scan_sampling_A"] == [2.5, 2.5]
        assert loaded.metadata["source_metadata"]["dm4.ImageData.Calibrations.Dimension.1.Units"] == "1/nm"


@pytest.mark.skipif(not os.environ.get("QUANTEM_TEST_CUDA"), reason="CUDA opt-in")
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dm4_ans_reconstructs_tail_and_detector_products(tmp_path, dtype):
    """A non-512 scan tail preserves each diffraction count and translated masks."""
    from quantem.gpu import detector, io

    values = np.random.default_rng(20260915).poisson(0.5, (23, 25, 24, 32)).astype(dtype)
    values[0, 0, 3, 7] = np.iinfo(dtype).max
    path = write_dm4(tmp_path / "camera.dm4", values)
    with io.load(path, backend="cuda") as loaded:
        for first, stop in [(0, 13), (510, 516), (570, 575)]:
            actual = loaded.data.decode_scan_range_device(first, stop).get()
            np.testing.assert_array_equal(actual, values.reshape(-1, 24, 32)[first:stop])
        session = detector.prepare(loaded)
        try:
            row, col = np.ogrid[:24, :32]
            for center in [(12, 16), (8.3, 9.7)]:
                radius = np.hypot(row - center[0], col - center[1])
                for mask in [radius <= 5, (radius > 5) & (radius <= 10), radius > 5]:
                    actual = session.masked_sum_exact(mask)
                    expected = values[..., mask].sum(-1, dtype=np.uint64)
                    np.testing.assert_array_equal(actual, expected)
        finally:
            session.close()


@pytest.mark.skipif(not os.environ.get("QUANTEM_TEST_CUDA"), reason="CUDA opt-in")
def test_camera_mask_plan_matches_signed_reference():
    """Full and translated masks retain exact signed pixel and tile coefficients."""
    from quantem.gpu._compact.interaction import plan
    from quantem.gpu._compact.mask_plan import CudaMaskPlanner

    for shape in [(864, 864), (57, 83)]:
        planner = CudaMaskPlanner(shape)
        row, col = np.ogrid[:shape[0], :shape[1]]
        first = ((row - shape[0] / 2)**2 + (col - shape[1] / 2)**2 < (shape[0] / 4)**2).astype(np.int32)
        second = ((row - shape[0] / 2 + 1.7)**2 + (col - shape[1] / 2 - 3.1)**2 < (shape[0] / 4)**2).astype(np.int32)
        for mask in [first, second - first, np.random.default_rng(3).integers(-1, 2, shape, dtype=np.int32)]:
            actual, expected = planner(mask), plan(mask)
            for offset in (0, 2):
                order = np.argsort(actual[offset])
                np.testing.assert_array_equal(actual[offset][order], expected[offset])
                np.testing.assert_array_equal(actual[offset + 1][order], expected[offset + 1])


def test_dm4_rejects_geometry_changes_and_truncated_payload(tmp_path):
    """Opening never silently reshapes an acquisition or accepts a partial transfer."""
    from quantem.gpu import io

    values = np.zeros((5, 7, 12, 16), np.uint8)
    path = write_dm4(tmp_path / "camera.dm4", values)
    with pytest.raises(ValueError, match="disagrees with DM geometry"):
        io.load(path, backend="cpu", scan_shape=(7, 5))
    with pytest.raises(NotImplementedError, match="complete acquisition"):
        io.load(path, backend="cuda", detector_bin=2)
    path.write_bytes(path.read_bytes()[:-200])
    with pytest.raises((ValueError, OSError)):
        io.inspect(path)


@pytest.mark.skipif(not os.environ.get("QUANTEM_TEST_CUDA"), reason="CUDA opt-in")
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_saved_camera_ans_reopens_without_original_or_encoder(tmp_path, dtype, monkeypatch):
    """A standalone snapshot preserves every encoded byte and native detector sum."""
    from quantem.gpu import detector, io
    from quantem.gpu._compact.streamed import StreamedCounts

    values = np.random.default_rng(17).poisson(0.7, (23, 25, 24, 32)).astype(dtype)
    values[0, 0, 0, 0] = np.iinfo(dtype).max
    path = write_dm4(tmp_path / "camera.dm4", values)
    saved = tmp_path / "camera.qem"
    with io.load(path, backend="cuda") as original:
        expected_arrays = [[array.get() for array in chunk.arrays] for chunk in original.data.chunks]
        io.save(saved, original, format="quantem", backend="cuda")
        with pytest.raises(FileExistsError):
            io.save(saved, original, format="quantem", backend="cuda")
    path.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Reopening must never encode or rebuild an index")

    monkeypatch.setattr(StreamedCounts, "append", forbidden)
    monkeypatch.setattr(StreamedCounts, "_index", forbidden)
    assert io.inspect(saved).scan_shape == (23, 25)
    with io.load(saved, backend="cuda") as reopened:
        assert reopened.metadata["scan_sampling_A"] == [2.5, 2.5]
        for chunk, expected in zip(reopened.data.chunks, expected_arrays):
            for actual, reference in zip(chunk.arrays, expected):
                np.testing.assert_array_equal(actual.get(), reference)
        actual = reopened.data.decode_scan_range_device(0, 575).get()
        np.testing.assert_array_equal(actual, values.reshape(-1, 24, 32))
        session = detector.prepare(reopened)
        try:
            row, col = np.ogrid[:24, :32]
            radius = np.hypot(row - 8.3, col - 9.7)
            for mask in [radius <= 5, (radius > 5) & (radius <= 10), radius > 5]:
                np.testing.assert_array_equal(session.masked_sum_exact(mask),
                                              values[..., mask].sum(-1, dtype=np.uint64))
        finally:
            session.close()
    # Both metadata and encoded payload corruption fail closed before exposure.
    original_bytes = saved.read_bytes()
    corrupted = bytearray(original_bytes)
    corrupted[-1] ^= 1
    saved.write_bytes(corrupted)
    with pytest.raises(ValueError, match="checksum mismatch"):
        io.load(saved, backend="cuda")
    saved.write_bytes(original_bytes[:-1])
    with pytest.raises(ValueError, match="Incomplete ANS"):
        io.inspect(saved)
    corrupted = bytearray(original_bytes)
    corrupted[80] ^= 1
    saved.write_bytes(corrupted)
    with pytest.raises(ValueError, match="header checksum"):
        io.inspect(saved)


@pytest.mark.skipif(not os.environ.get("QUANTEM_TEST_CUDA"), reason="CUDA opt-in")
def test_snapshot_retains_multiblock_chunks_and_validity(tmp_path):
    """Native streamed H5-style chunks retain multiple coding blocks and masks."""
    import cupy as cp
    from quantem.gpu import io
    from quantem.gpu._compact.streamed import StreamedCounts

    shape = (7, 147, 8, 12)
    values = np.random.default_rng(18).integers(0, 500, shape, dtype=np.uint16)
    valid = np.ones(shape[2:], bool)
    valid[2, 3] = False
    resident = StreamedCounts(shape, values.dtype, valid)
    resident.append(cp.asarray(values.reshape(-1, 8, 12)))
    path = tmp_path / "multiblock.qem"
    io.save(path, resident, format="quantem", backend="cuda")
    resident.release()
    assert io.inspect(path).pixel_mask[2, 3] == 1
    with io.load(path, backend="cuda") as loaded:
        np.testing.assert_array_equal(loaded.data.valid_pixels, valid)
        np.testing.assert_array_equal(loaded.data.decode_scan_range_device(0, 1029).get(),
                                      values.reshape(-1, 8, 12))
