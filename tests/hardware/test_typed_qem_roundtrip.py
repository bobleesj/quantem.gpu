"""Native measurement types survive GPU encoding and portable QEM copies."""

import os

import h5py
import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.formats.qem.reference import load_array
from quantem.gpu.formats.qem.validation import validate_qem


@pytest.fixture
def backend():
    name = os.environ.get("QEM_TEST_BACKEND")
    if name not in ("cuda", "mps"):
        pytest.skip("Set QEM_TEST_BACKEND on the physical CUDA or MPS device.")
    return name


def original_values(dtype, shape=(33, 17, 3, 5)):
    """Include every bit position, extrema and noncanonical IEEE NaN payloads."""
    dtype = np.dtype(dtype)
    unsigned = np.dtype(f"u{dtype.itemsize}")
    bits = np.random.default_rng(61017).integers(
        0, np.iinfo(unsigned).max, shape, dtype=unsigned, endpoint=True
    )
    # Exercise zero, constant and sparse streams across two codec intervals.
    bits[..., 0, :3] = 0
    bits[..., 1, :3] = 1
    bits[0, 0, 0, 0] = np.iinfo(unsigned).max
    return bits.view(dtype)


def raw_read(source):
    """Read every original value in windows; release each Metal result promptly."""
    output = np.empty(source.shape, source.dtype).reshape(-1, *source.shape[2:])
    for first in range(0, len(output), 113):
        stop = min(first + 113, len(output))
        values = source.decode_scan_range_device(first, stop)
        try:
            output[first:stop] = values.get()
        finally:
            if hasattr(values, "release"):
                values.release()
    return output.reshape(source.shape)


@pytest.mark.parametrize(
    "kind",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    ],
)
@pytest.mark.parametrize("extension", ["npy", "h5"])
def test_original_dtype_and_bits_survive_move(tmp_path, backend, kind, extension):
    values = original_values(kind)
    original = tmp_path / f"original.{extension}"
    if extension == "npy":
        np.save(original, values)
    else:
        with h5py.File(original, "w") as handle:
            handle.create_dataset("entry/data/data", data=values)
    copy = tmp_path / "saved.qem"
    with io.load(
        original, backend=backend, hot_pixel_correction="none", verbose=False
    ) as loaded:
        assert loaded.dtype == values.dtype
        assert raw_read(loaded.data).tobytes() == values.tobytes()
        pattern = detector.prepare(loaded).frame(0, output="native")
        try:
            actual = pattern.get() if hasattr(pattern, "get") else pattern.cpu().numpy()
            assert actual.dtype == values.dtype
            assert actual.tobytes() == values[0, 0].tobytes()
        finally:
            if hasattr(pattern, "release"):
                pattern.release()
        assert loaded.metadata["file_counts_exact"]
        io.save(copy, loaded, verbose=False)
    assert validate_qem(copy)["dtype"] == kind
    original.unlink()
    moved = tmp_path / "moved" / copy.name
    moved.parent.mkdir()
    copy.rename(moved)
    with io.load(moved, backend=backend, verbose=False) as loaded:
        assert loaded.dtype == values.dtype
        assert raw_read(loaded.data).tobytes() == values.tobytes()
        if kind == "float64" and backend == "mps":
            with pytest.raises((TypeError, NotImplementedError), match="float64"):
                loaded.read(scan_region=(0, 1, 0, 1))
        else:
            assert (
                loaded.read(scan_region=(0, 1, 0, 1)).cpu().numpy().tobytes()
                == values[:1, :1].tobytes()
            )
        native = loaded.read(
            scan_region=(1, 3, 2, 5), detector_region=(1, 3, 1, 4), output="native"
        )
        try:
            assert native.dtype == values.dtype
            assert native.get().tobytes() == values[1:3, 2:5, 1:3, 1:4].tobytes()
        finally:
            if hasattr(native, "release"):
                native.release()
    # Independent scalar decoder only for this tiny fixture, never full real data.
    decoded, _ = load_array(moved)
    assert decoded.dtype == values.dtype
    assert decoded.tobytes() == values.tobytes()


@pytest.mark.parametrize("kind", ["uint32", "uint64", "int32", "int64"])
def test_integer_virtual_images_are_exact_and_checked(tmp_path, backend, kind):
    dtype = np.dtype(kind)
    values = np.zeros((2, 3, 2, 4), dtype)
    if dtype.kind == "u":
        values[..., 0, 0] = np.iinfo(dtype).max
    else:
        values[..., 0, :2] = [np.iinfo(dtype).max, np.iinfo(dtype).min]
    path = tmp_path / "counts.npy"
    np.save(path, values)
    with io.load(path, backend=backend, verbose=False) as loaded:
        session = detector.prepare(loaded)
        mask = np.ones((2, 4), bool)
        result = session.masked_sum_exact(mask)
        expected = np.iinfo(dtype).max if dtype.kind == "u" else -1
        np.testing.assert_array_equal(result, np.full((2, 3), expected, result.dtype))
        native = session.masked_sum_exact(mask, output="native")
        native = native.get() if backend == "cuda" else native.cpu().numpy()
        np.testing.assert_array_equal(native, result)
        np.testing.assert_array_equal(
            session.masked_sum_exact(np.zeros((2, 4), bool)), 0
        )
    if dtype.itemsize == 8:
        values[..., 0, 1] = 1
        np.save(path, values)
        with io.load(path, backend=backend, verbose=False) as loaded:
            with pytest.raises(OverflowError, match="64-bit"):
                detector.prepare(loaded).masked_sum_exact(mask)


def test_calibration_override_survives_save_move_and_resave(tmp_path, backend):
    original = tmp_path / "source.npy"
    np.save(original, np.arange(24, dtype=np.uint32).reshape(2, 3, 2, 2))
    saved = tmp_path / "calibrated.qem"
    with io.load(original, backend=backend, verbose=False) as data:
        data.set_calibration(
            voltage_kV=300,
            semiangle_mrad=30,
            scan_sampling_A=(0.4, 0.5),
            detector_sampling_mrad=(0.2, 0.3),
            evidence="Fixture calibration",
        )
        summary = data.calibration.copy()
        io.save(saved, data, verbose=False)
    original.unlink()
    moved = tmp_path / "moved.qem"
    saved.rename(moved)
    with io.load(moved, backend=backend, verbose=False) as data:
        assert data.calibration.equals(summary)
        assert data.sampling == (0.4, 0.5, 0.2, 0.3)
        data.set_calibration(semiangle_mrad=25, evidence="Updated calibration")
        io.save(tmp_path / "updated.qem", data, verbose=False)
    with io.load(tmp_path / "updated.qem", backend=backend, verbose=False) as data:
        assert data.metadata["semiangle_mrad"] == 25
        assert data.metadata["voltage_kV"] == 300


@pytest.mark.parametrize("kind", ["uint32", "uint64", "int32", "int64"])
def test_selected_patterns_retain_integer_precision(tmp_path, backend, kind):
    values = np.zeros((33, 17, 2, 3), dtype=kind)
    frames = values.reshape(-1, 2, 3)
    selected = [1, 510, 560]
    maximum = np.iinfo(kind).max
    frames[selected, 0, 0] = [maximum // 3, maximum // 3, maximum % 3]
    frames[selected, 0, 1] = [16777217, 16777219, 16777221]
    if values.dtype.kind == "i":
        frames[selected, 1, 0] = [-16777217, 16777219, -16777221]
        frames[selected, 1, 1] = [np.iinfo(kind).min, maximum, -1]
    original = tmp_path / "wide.npy"
    np.save(original, values)
    with io.load(original, backend=backend, verbose=False) as loaded:
        session = detector.prepare(loaded)
        expected = frames[selected].astype(object).sum(axis=0)
        total = session.reduce_frames_exact(selected)
        np.testing.assert_array_equal(total, expected.astype(total.dtype))
        np.testing.assert_array_equal(
            session.reduce_frames_max(selected), frames[selected].max(axis=0)
        )
        np.testing.assert_array_equal(
            session.reduce_frames(selected),
            (expected.astype(np.float64) / 3).astype(np.float32),
        )
        np.testing.assert_array_equal(
            session.mean_dp(),
            (expected.astype(np.float64) / len(frames)).astype(np.float32),
        )


@pytest.mark.parametrize("kind", [">u4", ">i8", ">f8"])
def test_foreign_byte_order_preserves_measurements(tmp_path, backend, kind):
    dtype = np.dtype(kind)
    values = original_values(dtype.newbyteorder("=")).byteswap().view(dtype)
    original = tmp_path / "foreign.npy"
    np.save(original, values)
    with io.load(original, backend=backend, verbose=False) as loaded:
        assert loaded.dtype == dtype.newbyteorder("=")
        expected = values.byteswap().view(dtype.newbyteorder("="))
        assert raw_read(loaded.data).tobytes() == expected.tobytes()


def test_float64_display_rounds_without_changing_storage(tmp_path, backend):
    # Storage accepts arbitrary IEEE words; display rounds finite values to
    # float32. Include underflow, subnormals, overflow and half-way ties.
    values = original_values("float64", (4, 3, 7, 19))
    special = [
        0,
        -0.0,
        np.inf,
        -np.inf,
        np.nan,
        2**-150,
        2**-149,
        2**-126,
        2**128,
        1 + 2**-24,
        1 + 3 * 2**-24,
    ]
    values.flat[: len(special)] = special
    path = tmp_path / "double.npy"
    np.save(path, values)
    with io.load(path, backend=backend, verbose=False) as loaded:
        if backend == "mps":
            from quantem.gpu.resident.mps.float64_display import float32_view

            raw = loaded.read(output="native")
            try:
                result = float32_view(raw).cpu().numpy()
                assert raw.get().tobytes() == values.tobytes()
            finally:
                raw.release()
        else:
            # CUDA exposes float64 calculation arrays directly. Metal's
            # explicit display conversion is the operation being checked here.
            result = loaded.data._tensor(0, 12).get().reshape(values.shape)
            assert result.tobytes() == values.tobytes()
            return
        with np.errstate(over="ignore", invalid="ignore"):
            expected = values.astype(np.float32)
        np.testing.assert_array_equal(result, expected)
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(
            result.view(np.uint32)[finite], expected.view(np.uint32)[finite]
        )


@pytest.mark.parametrize("kind", ["uint32", "uint64", "int32", "float64"])
def test_fixed_ssb_survives_typed_save_reopen(tmp_path, backend, kind):
    from quantem.gpu import SSB

    rows, cols = np.indices((16, 16))
    disk = (rows - 7.5) ** 2 + (cols - 7.5) ** 2 <= 5**2
    values = np.random.default_rng(31).integers(1, 40, (16, 16, 16, 16))
    values = (values * disk).astype(kind)
    if kind == "float64":
        values /= 7
    original = tmp_path / "source.npy"
    np.save(original, values)
    settings = dict(
        backend=backend,
        voltage_kV=300,
        semiangle_mrad=20,
        scan_sampling_A=(0.5, 0.5),
        detector_sampling_mrad=(4, 4),
        bright_field_center_px=(7.5, 7.5),
        bright_field_radius_px=5,
    )

    def phases(data):
        with SSB(data, **settings) as session:
            result = []
            for tilt in (None, (1.0, -2.0)):
                session.sample_tilt_mrad = tilt
                session.depth_spread_nm = 5 if tilt else None
                phase = session.reconstruct(
                    aberrations={"C10_nm": 2},
                    upsampling_factor=2,
                    compute_loss=False,
                ).phase
                result.append(phase.get() if backend == "cuda" else np.asarray(phase))
            return result

    saved = tmp_path / "copy.qem"
    with io.load(original, backend=backend, verbose=False) as loaded:
        before = phases(loaded)
        io.save(saved, loaded, verbose=False)
    original.unlink()
    with io.load(saved, backend=backend, verbose=False) as loaded:
        after = phases(loaded)
        for first, second in zip(before, after, strict=True):
            np.testing.assert_array_equal(first, second)
