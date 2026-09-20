"""Open, inspect, save and reopen float measurements on the selected accelerator."""

import hashlib
import os
import base64

import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io._float_ans import FloatANSResident, MAX_DECODE_BYTES
from quantem.gpu.io.qem_validation import validate_qem
from quantem.gpu.io._streamed_file import read_header


def _host(output):
    if hasattr(output, "to_numpy"):
        try:
            return output.to_numpy()
        finally:
            output.release()
    return output.get()


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in ("mps", "cuda"):
        pytest.skip("Set QEM_TEST_BACKEND=mps or cuda on a physical accelerator.")
    return backend


def test_float_qem_preserves_bits_and_metadata_without_dense_residency(tmp_path):
    backend = _backend()
    words = np.random.default_rng(10).integers(
        0, 2**32, (2, 4, 128, 128), dtype=np.uint32
    )
    words.ravel()[512:519] = [
        0,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        1,
        0x3E800000,
    ]
    # Include zero, constant, sparse and entropy-eligible lanes, not just literals.
    words[:, :, 0, :16] = np.arange(16, dtype=np.uint32)
    words[:, :, 1, :16] = np.arange(8, dtype=np.uint32).reshape(2, 4, 1)
    content = '{"voltage_kV": 200, "unknown_vendor_field": "preserve me"}'
    document = dict(
        filename="synthetic.json",
        mediaType="application/json",
        content=content,
        sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    path = tmp_path / "original.qem"
    io.save(
        path,
        words.view(np.float32),
        backend="cpu",
        batch_size=3,
        metadata={"source_documents": [document], "voltage_kV": 200},
    )
    with io.load(path, backend=backend) as loaded:
        assert loaded.metadata["representation"] == "encoded"
        assert loaded.metadata["resident_profile"] == "float32-bit-lanes-rans-v1"
        assert loaded.dtype == np.dtype("float32")
        for first, stop in ((0, 1), (2, 5), (7, 8)):
            actual = _host(loaded.data.decode_scan_range_device(first, stop))
            np.testing.assert_array_equal(
                actual.view(np.uint32), words.reshape(-1, 128, 128)[first:stop]
            )
        with pytest.raises(TypeError, match="stays encoded"):
            np.asarray(loaded.data)
        with pytest.raises(NotImplementedError, match="encoded"):
            io.load(path, backend=backend, representation="dense")
        path.unlink()  # Saving must not depend on retaining or rereading the source.
        copied = tmp_path / "copied.qem"
        io.save(copied, loaded)
        assert loaded.data.peak_decode_bytes <= MAX_DECODE_BYTES
    assert validate_qem(copied)["integrity"] == "verified"
    with io.load(copied, backend=backend) as loaded:
        assert loaded.metadata["scientific_metadata"]["source_documents"] == [document]
        for index in range(8):
            np.testing.assert_array_equal(
                _host(loaded.data.extract_diffraction_device(*divmod(index, 4))).view(
                    np.uint32
                ),
                words.reshape(-1, 128, 128)[index],
            )


def test_float_products_and_selected_mean_stay_on_device(tmp_path):
    backend = _backend()
    # Tiny independent reference only; full acquisitions are compared to GPU results.
    raw = (
        np.arange(8 * 128 * 128, dtype=np.float32).reshape(2, 4, 128, 128) % 101 - 50
    ) / 8
    path = tmp_path / "measurements.qem"
    io.save(path, raw, backend="cpu", batch_size=3)
    mask = np.zeros((128, 128), bool)
    mask[19:40, 17:90] = True
    with io.load(path, backend=backend) as loaded:
        session = detector.prepare(loaded)
        np.testing.assert_array_equal(session.frame(4), raw[1, 0])
        np.testing.assert_allclose(
            session.masked_sum(mask), raw[:, :, mask].sum(-1), atol=1e-5, rtol=1e-6
        )
        np.testing.assert_allclose(
            session.mean_dp(), raw.mean((0, 1)), atol=1e-5, rtol=1e-6
        )
        indices = [1, 3, 6]
        np.testing.assert_allclose(
            session.reduce_frames(indices),
            raw.reshape(8, 128, 128)[indices].mean(0),
            atol=1e-5,
            rtol=1e-6,
        )
        row, column = session.center_of_mass(mask)
        weights = np.where(mask, raw.astype(np.float64), 0)
        rr, cc = np.indices((128, 128))
        total = weights.sum((2, 3))
        expected_row = np.divide(
            (weights * rr).sum((2, 3)),
            total,
            out=np.full_like(total, np.nan),
            where=total != 0,
        )
        expected_column = np.divide(
            (weights * cc).sum((2, 3)),
            total,
            out=np.full_like(total, np.nan),
            where=total != 0,
        )
        np.testing.assert_allclose(
            row, expected_row - np.nanmean(expected_row), atol=2e-3, rtol=2e-5
        )
        np.testing.assert_allclose(
            column, expected_column - np.nanmean(expected_column), atol=2e-3, rtol=2e-5
        )
        assert loaded.data.peak_decode_bytes <= MAX_DECODE_BYTES


def test_decode_limit_is_checked_before_allocating():
    source = FloatANSResident.__new__(FloatANSResident)
    source.shape = (1, 1024, 128, 128)
    source.is_released = False
    source.peak_decode_bytes = 0
    with pytest.raises(ValueError, match="32 MiB"):
        source.decode_scan_range_device(0, 513)
    assert source.peak_decode_bytes == 0


def test_float_background_is_applied_once_and_survives_export(tmp_path):
    backend = _backend()
    raw = np.full((1, 4, 128, 128), 7, dtype=np.float32)
    dark = np.full((128, 128), 2, dtype=np.float32)
    empad = {
        "format_identifier": "synthetic-float32",
        "background": {"values_float32_le": base64.b64encode(dark.tobytes()).decode()},
    }
    path = tmp_path / "background.qem"
    io.save(path, raw, backend="cpu", batch_size=2, metadata={"qem_empad": empad})
    copied = tmp_path / "copy.qem"
    for path in (path, copied):
        with io.load(path, backend=backend) as loaded:
            session = detector.prepare(loaded)
            np.testing.assert_array_equal(
                _host(loaded.data.extract_diffraction_device(0, 0)), raw[0, 0]
            )
            np.testing.assert_array_equal(session.frame(0), raw[0, 0] - dark)
            np.testing.assert_array_equal(
                session.masked_sum(np.ones((128, 128), bool)), [[81920] * 4]
            )
            np.testing.assert_array_equal(session.mean_dp(), raw[0, 0] - dark)
            if not copied.exists():
                io.save(copied, loaded)
        assert loaded.data.nbytes == 0
        with pytest.raises(RuntimeError, match="released"):
            loaded.data.mean_dp_device()


def test_float_reductions_keep_cancellation_and_nonfinite_semantics(tmp_path):
    backend = _backend()
    raw = np.zeros((1, 4, 128, 128), dtype=np.float32)
    raw[0, :, 0, :3] = [1e20, 1, -1e20]
    raw[0, 2, 1, 0] = np.nan
    path = tmp_path / "cancellation.qem"
    io.save(path, raw, backend="cpu", batch_size=2)
    mask = np.zeros((128, 128), bool)
    mask[0, :3] = True
    with io.load(path, backend=backend) as loaded:
        session = detector.prepare(loaded)
        np.testing.assert_array_equal(session.masked_sum(mask), [[1] * 4])
        mask[1, 0] = True
        result = session.masked_sum(mask)
        assert np.isnan(result[0, 2])
        mask[1, 0] = False
        np.testing.assert_array_equal(session.masked_sum(mask), [[1] * 4])
        assert np.isnan(session.reduce_frames([0, 2], mode="max")[1, 0])


def test_float_com_does_not_overflow_or_hide_invalid_frames(tmp_path):
    backend = _backend()
    raw = np.zeros((1, 4, 128, 128), dtype=np.float32)
    raw[0, 0, 10, 20] = 1e38
    raw[0, 1, 30, 40] = 1e38
    raw[0, 3, 2, 3] = np.inf
    path = tmp_path / "moments.qem"
    io.save(path, raw, backend="cpu", batch_size=2)
    with io.load(path, backend=backend) as loaded:
        row, column = detector.prepare(loaded).center_of_mass()
        np.testing.assert_array_equal(row[0, :2], [-10, 10])
        np.testing.assert_array_equal(column[0, :2], [-10, 10])
        assert np.isnan(row[0, 2:]).all() and np.isnan(column[0, 2:]).all()


def test_corrupt_upload_releases_partial_resident(tmp_path, monkeypatch):
    backend = _backend()
    path = tmp_path / "damaged.qem"
    io.save(path, np.ones((1, 4, 128, 128), np.float32), backend="cpu", batch_size=2)
    _, start = read_header(path)
    with path.open("r+b") as handle:
        handle.seek(start)
        value = handle.read(1)[0]
        handle.seek(start)
        handle.write(bytes([value ^ 1]))
    released = []
    original = FloatANSResident.release

    def release(source):
        original(source)
        released.append(source)

    monkeypatch.setattr(FloatANSResident, "release", release)
    with pytest.raises(ValueError, match="checksum"):
        io.load(path, backend=backend)
    assert len(released) == 1 and released[0].nbytes == 0


def test_cuda_resident_survives_client_device_switch(tmp_path):
    if _backend() != "cuda":
        pytest.skip("CUDA multi-device ownership check")
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Requires two CUDA devices")
    path = tmp_path / "device.qem"
    io.save(path, np.ones((1, 2, 128, 128), np.float32), backend="cpu")
    with io.load(path, backend="cuda", device=0) as loaded:
        with cp.cuda.Device(1):
            # No scientific work is submitted on device 1.
            session = detector.prepare(loaded)
            np.testing.assert_array_equal(session.frame(0), np.ones((128, 128)))
            np.testing.assert_array_equal(session.mean_dp(), np.ones((128, 128)))
            np.testing.assert_array_equal(
                session.reduce_frames([0, 1]), np.ones((128, 128))
            )
            np.testing.assert_array_equal(session.center_of_mass()[0], [[0, 0]])
