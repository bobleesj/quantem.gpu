"""2026-11-24: io.load(source, detector_transpose=True) swaps each pattern's rows and columns on the GPU as it is encoded.

Some acquisitions record the detector transposed relative to the scan (an EMPAD-G1 WSe2 scan), a reflection that no
scan-detector rotation can undo, and a reconstruction may need the detector in the scan's frame. Every source the
loader handles takes the swap:
NumPy, EMPAD RAW, generic HDF5, ARINA masters (direct chunk decoding, with and without detector_bin) and .qem copies.
Each resident must hold exactly the GPU transpose of the frames a plain load holds, its record must swap every
detector axis, and a saved copy must reopen transposed without being asked again.
"""

import h5py
import hdf5plugin
import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _compiles():
    try:
        cp.ElementwiseKernel("T x", "T y", "y = x + 1", "transpose_probe")(cp.zeros(1)).item()
        return True
    except (cp.cuda.compiler.CompileException, cp.cuda.runtime.CUDARuntimeError, RuntimeError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _compiles(), reason="needs a CUDA GPU that compiles kernels")

# 23 x 29 = 667 positions: two ingestion blocks of at most 512, the second starting inside a scan row
SCAN, DETECTOR = (23, 29), (12, 10)


def transposed_on_gpu(values) -> "cp.ndarray":
    """The reference: the same frames moved to the GPU and transposed there."""
    return cp.ascontiguousarray(cp.transpose(cp.asarray(values), (0, 1, 3, 2)))


def assert_bits_equal(actual, expected) -> None:
    """Every stored bit equal (NaN payloads and signed zeros included), compared on the GPU."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    unsigned = np.dtype(f"u{expected.dtype.itemsize}")
    assert bool(cp.array_equal(cp.ascontiguousarray(actual).view(unsigned), expected.view(unsigned)))


def measured_values(dtype, shape=(*SCAN, *DETECTOR)):
    """Every bit pattern of the dtype (extrema, NaN payloads, negative zero), with constant streams."""
    dtype = np.dtype(dtype)
    unsigned = np.dtype(f"u{dtype.itemsize}")
    bits = np.random.default_rng(1124).integers(0, np.iinfo(unsigned).max, shape, dtype=unsigned, endpoint=True)
    bits[..., 0, :3] = 0
    bits[..., 1, :3] = 1
    if dtype.kind == "f":
        bits[0, 0, 2, 3] = np.array([-0.0], dtype).view(unsigned)[0]
    return bits.view(dtype)


def counts(dtype=np.uint16, shape=(*SCAN, *DETECTOR), mean=40.0):
    return np.random.default_rng(7).poisson(mean, size=shape).astype(dtype)


# --- the device transpose itself


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32", "float64"])
@pytest.mark.parametrize("frame", [(33, 70), (70, 33), (65, 31), (3, 40)])
def test_transposed_frames_across_partial_tiles(dtype, frame):
    """32 x 32 tiles: frames that are not square and not a multiple of 32, on several tiles each way."""
    from quantem.gpu.io.arrays import transposed_frames

    raw = cp.asarray(measured_values(dtype, shape=(1, 3, *frame)).reshape(3, *frame))
    assert_bits_equal(transposed_frames(raw), cp.ascontiguousarray(raw.transpose(0, 2, 1)))


# --- every source: the resident is the GPU transpose of the frames, bit for bit


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "int16", "float32", "float64"])
def test_numpy_source_is_the_gpu_transpose_of_its_frames(tmp_path, dtype):
    from quantem.gpu import io

    values = counts(dtype) if dtype in ("uint8", "uint16") else measured_values(dtype)
    path = tmp_path / "scan.npy"
    np.save(path, values)
    with io.load(path, detector_transpose=True, verbose=False) as loaded:
        assert loaded.shape == (*SCAN, DETECTOR[1], DETECTOR[0])
        assert loaded.metadata["detector_transpose"] is True
        assert loaded.metadata["detector_shape"] == (DETECTOR[1], DETECTOR[0])
        expected = transposed_on_gpu(values)
        assert_bits_equal(loaded.read(output="native"), expected)
        # one pattern through indexing, and a bounded region, read as a scientist reads them
        assert_bits_equal(cp.asarray(loaded[4, 7]), expected[4, 7])
        assert_bits_equal(
            loaded.read(scan_region=(2, 5, 20, 29), detector_region=(1, 9, 3, 11), output="native"),
            expected[2:5, 20:29, 1:9, 3:11],
        )


@pytest.mark.parametrize("swap", [False, True])
def test_float_checksum_covers_the_stored_values_in_order(tmp_path, swap):
    """Three ingestion blocks (512 + 512 + 49 frames): the hash worker keeps block order, and the transposed path
    reuses its first page-locked buffer only after that block was hashed."""
    import hashlib

    from quantem.gpu import io

    values = measured_values("float32", shape=(37, 29, *DETECTOR))
    np.save(tmp_path / "scan.npy", values)
    stored = np.transpose(values, (0, 1, 3, 2)) if swap else values
    with io.load(tmp_path / "scan.npy", detector_transpose=swap, verbose=False) as loaded:
        assert loaded.data.header["logical_sha256"] == hashlib.sha256(np.ascontiguousarray(stored).tobytes()).hexdigest()


def test_a_plain_load_is_unchanged_and_records_no_transpose(tmp_path):
    from quantem.gpu import io

    values = counts()
    path = tmp_path / "scan.npy"
    np.save(path, values)
    with io.load(path, verbose=False) as loaded:
        assert loaded.shape == (*SCAN, *DETECTOR)
        assert "detector_transpose" not in loaded.metadata
        assert_bits_equal(loaded.read(output="native"), cp.asarray(values))


def empad_raw(folder, scan=(5, 6)):
    """A headerless EMPAD-G1 RAW: 130 x 128 float32 records, the last two rows dropped by the reader."""
    records = np.random.default_rng(11).normal(0.0, 3.0, size=(*scan, 130, 128)).astype("<f4")
    records[..., :128, :] += np.arange(128, dtype="<f4")[:, None] * 0.01   # rows differ from columns
    path = folder / "scan_x6_y5.raw"
    records.tofile(path)
    return path, records[:, :, :128, :]


def test_empad_raw_is_the_gpu_transpose_of_its_frames(tmp_path):
    from quantem.gpu import io

    path, frames = empad_raw(tmp_path)
    with io.load(path, scan_shape=(5, 6), detector_transpose=True, verbose=False) as loaded:
        assert loaded.metadata["detector_transpose"] is True
        assert_bits_equal(loaded.read(output="native"), transposed_on_gpu(frames))


def test_generic_hdf5_is_the_gpu_transpose_of_its_frames(tmp_path):
    from quantem.gpu import io

    values = measured_values("float32")
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/data/data", data=values)
    with io.load(path, hot_pixel_correction="none", detector_transpose=True, verbose=False) as loaded:
        assert loaded.metadata["detector_transpose"] is True
        assert_bits_equal(loaded.read(output="native"), transposed_on_gpu(values))


FLAGGED = (3, 7)   # (row, col) in the stored detector


def arina_master(folder, dtype=np.uint16, detector=(16, 12), sampling=(0.5, 0.25)):
    """A bitshuffle+LZ4 ARINA master with one flagged pixel and a (row, col) detector sampling that differs by axis."""
    data = counts(dtype, shape=(SCAN[0] * SCAN[1], *detector), mean=4.0)
    mask = np.zeros(detector, np.uint32)
    data[:, FLAGGED[0], FLAGGED[1]] = np.iinfo(dtype).max
    mask[FLAGGED] = 1
    path = folder / "scan_master.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/instrument/detector")
        group["detectorSpecific/pixel_mask"] = mask
        group["detectorSpecific/ntrigger"] = data.shape[0]
        handle.attrs["detector_sampling_mrad"] = np.array(sampling)
        with h5py.File(folder / "scan_data_000001.h5", "w") as part:
            part.create_dataset("entry/data/data", data=data, chunks=(1, *detector),
                                **hdf5plugin.Bitshuffle(cname="lz4"))
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data_000001.h5", "entry/data/data")
    return path


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_arina_master_is_the_gpu_transpose_of_the_corrected_frames(tmp_path, dtype):
    """Flagged pixels are filled in the stored frame first; the record follows each detector axis."""
    from quantem.gpu import io

    path = arina_master(tmp_path, dtype)
    plain = io.load(path, scan_shape=SCAN, hot_pixel_correction="median", verbose=False)
    swapped = io.load(path, scan_shape=SCAN, hot_pixel_correction="median", detector_transpose=True, verbose=False)
    try:
        assert swapped.shape == (*SCAN, 12, 16)
        assert_bits_equal(swapped.read(output="native"), transposed_on_gpu(plain.read(output="native")))
        record, stored = swapped.metadata, plain.metadata
        assert record["detector_transpose"] is True
        assert record["detector_sampling_mrad"] == [0.25, 0.5]
        assert stored["detector_sampling_mrad"] == [0.5, 0.25]
        np.testing.assert_array_equal(record["pixel_mask"], np.asarray(stored["pixel_mask"]).T)
        assert record["hot_pixel_correction"]["coordinates_row_col"] == [[FLAGGED[1], FLAGGED[0]]]
        assert record["hot_pixel_correction"]["applied"] is True
    finally:
        plain.close()
        swapped.close()


def test_arina_master_binned_then_transposed(tmp_path):
    from quantem.gpu import io

    path = arina_master(tmp_path)
    binned = io.load(path, scan_shape=SCAN, hot_pixel_correction="median", detector_bin=2, verbose=False)
    swapped = io.load(path, scan_shape=SCAN, hot_pixel_correction="median", detector_bin=2,
                      detector_transpose=True, verbose=False)
    try:
        assert swapped.shape == (*SCAN, 6, 8)
        assert_bits_equal(swapped.read(output="native"), transposed_on_gpu(binned.read(output="native")))
        assert swapped.metadata["detector_bin"] == 2
        assert swapped.metadata["binned_from_detector_shape"] == [12, 16]
        assert swapped.metadata["detector_sampling_mrad"] == [0.5, 1.0]
    finally:
        binned.close()
        swapped.close()


def test_flagged_pixels_kept_as_stored_stay_excluded_at_their_swapped_place(tmp_path):
    """hot_pixel_correction="none": the marker is kept and the transposed validity mask excludes it."""
    from quantem.gpu import io

    path = arina_master(tmp_path)
    with io.load(path, scan_shape=SCAN, detector_transpose=True, verbose=False) as swapped:
        valid = swapped.data.valid_pixels
        assert valid.shape == (12, 16)
        assert not valid[FLAGGED[1], FLAGGED[0]] and valid.sum() == valid.size - 1


# --- .qem copies: reopened transposed, saved transposed


@pytest.mark.parametrize("dtype", ["uint16", "float32", "int16"])
def test_qem_copy_reopened_transposed_is_the_gpu_transpose(tmp_path, dtype):
    """A .qem reopens its encoded arrays without decoding; the swap decodes on the GPU and encodes again."""
    from quantem.gpu import io

    values = counts() if dtype == "uint16" else measured_values(dtype)
    np.save(tmp_path / "scan.npy", values)
    with io.load(tmp_path / "scan.npy", verbose=False) as loaded:
        io.save(tmp_path / "scan.qem", loaded, verbose=False)
    with io.load(tmp_path / "scan.qem", detector_transpose=True, verbose=False) as swapped:
        assert swapped.metadata["detector_transpose"] is True
        assert "qem_header_sha256" not in swapped.metadata
        assert_bits_equal(swapped.read(output="native"), transposed_on_gpu(values))


@pytest.mark.parametrize("dtype", ["uint16", "float32"])
def test_a_transposed_copy_reopens_transposed_and_says_so(tmp_path, dtype):
    from quantem.gpu import io
    from quantem.gpu.formats.qem.reference import load_array
    from quantem.gpu.formats.qem.snapshot import read_envelope

    values = counts() if dtype == "uint16" else measured_values(dtype)
    np.save(tmp_path / "scan.npy", values)
    copy = tmp_path / "swapped.qem"
    with io.load(tmp_path / "scan.npy", detector_transpose=True, verbose=False) as swapped:
        io.save(copy, swapped, verbose=False)
    header, _ = read_envelope(copy)
    assert header["shape"] == [*SCAN, DETECTOR[1], DETECTOR[0]]
    assert header["metadata"]["detector_transpose"] is True
    assert {"operation": "detector_transpose", "changes_measurements": False} in header["scientific_metadata"]["processing"]
    with io.load(copy, verbose=False) as reopened:
        assert reopened.metadata["detector_transpose"] is True
        assert_bits_equal(reopened.read(output="native"), transposed_on_gpu(values))
    # the independent reference decoder checks the stored checksum (float32: over the transposed values)
    decoded, _ = load_array(copy)
    assert decoded.tobytes() == np.ascontiguousarray(np.transpose(values, (0, 1, 3, 2))).tobytes()
    with pytest.raises(ValueError, match="already swapped"):
        io.load(copy, detector_transpose=True, verbose=False)


def test_a_transposed_master_saved_and_reopened_keeps_its_swapped_calibration(tmp_path):
    from quantem.gpu import io

    path = arina_master(tmp_path)
    copy = tmp_path / "swapped.qem"
    with io.load(path, scan_shape=SCAN, hot_pixel_correction="median", detector_transpose=True,
                 verbose=False) as swapped:
        expected = swapped.read(output="native").copy()
        io.save(copy, swapped, verbose=False)
    with io.load(copy, verbose=False) as reopened:
        assert reopened.metadata["detector_transpose"] is True
        assert reopened.metadata["detector_sampling_mrad"] == [0.25, 0.5]
        assert reopened.sampling[2:] == (0.25, 0.5)
        assert reopened.metadata["hot_pixel_correction"]["coordinates_row_col"] == [[FLAGGED[1], FLAGGED[0]]]
        assert_bits_equal(reopened.read(output="native"), expected)


def test_a_master_copy_reopened_transposed_keeps_its_correction_and_calibration(tmp_path):
    from quantem.gpu import io

    path = arina_master(tmp_path)
    copy = tmp_path / "plain.qem"
    with io.load(path, scan_shape=SCAN, hot_pixel_correction="median", verbose=False) as plain:
        expected = transposed_on_gpu(plain.read(output="native"))
        io.save(copy, plain, verbose=False)
    with io.load(copy, detector_transpose=True, verbose=False) as swapped:
        assert_bits_equal(swapped.read(output="native"), expected)
        assert swapped.metadata["detector_sampling_mrad"] == [0.25, 0.5]
        assert swapped.metadata["hot_pixel_correction"]["applied"] is True
        assert swapped.metadata["hot_pixel_correction"]["coordinates_row_col"] == [[FLAGGED[1], FLAGGED[0]]]
        swapped_copy = tmp_path / "swapped.qem"
        io.save(swapped_copy, swapped, verbose=False)
    with io.load(swapped_copy, verbose=False) as reopened:
        assert reopened.metadata["detector_transpose"] is True
        assert_bits_equal(reopened.read(output="native"), expected)


def test_a_calibrated_copy_reopened_transposed_swaps_each_value_once(tmp_path):
    """Set calibration and disk on a plain copy, reopen it transposed, save that and reopen: swapped exactly once, in
    the flat keys and in the saved schema-2 record."""
    from quantem.gpu import io

    np.save(tmp_path / "scan.npy", counts())
    with io.load(tmp_path / "scan.npy", verbose=False) as loaded:
        loaded.calibration.detector_sampling_mrad = (0.2, 0.3)
        loaded.metadata["bright_field_disk"] = dict(schema="quantem.bright-field-disk/1", method="half-plateau disk edge",
                                                    bright_field_center_px=[4.0, 7.0], bright_field_edge_radius_px=3.0)
        io.save(tmp_path / "plain.qem", loaded, verbose=False)
    with io.load(tmp_path / "plain.qem", detector_transpose=True, verbose=False) as swapped:
        assert swapped.sampling[2:] == (0.3, 0.2)
        assert swapped.metadata["bright_field_disk"]["bright_field_center_px"] == [7.0, 4.0]
        io.save(tmp_path / "swapped.qem", swapped, verbose=False)
    with io.load(tmp_path / "swapped.qem", verbose=False) as reopened:
        assert reopened.sampling[2:] == (0.3, 0.2)
        assert reopened.metadata["bright_field_disk"]["bright_field_center_px"] == [7.0, 4.0]


def test_a_recorded_rotation_flip_is_inverted_and_every_axis_value_moves():
    """Direct ptychography transposes the frames when scan_detector_rotation_flipped is set; these already are."""
    from quantem.gpu.io.arrays import transposed_record

    record = transposed_record(dict(
        scan_detector_rotation_deg=12.5, scan_detector_rotation_flipped=True,
        detector_sampling_mrad=np.array([0.5, 0.25]), sampling=(0.4, 0.4, 0.5, 0.25), detector_shape=(16, 12),
        valid_pixels=np.eye(16, 12, dtype=bool),
    ))
    assert record["scan_detector_rotation_flipped"] is False and record["scan_detector_rotation_deg"] == 12.5
    np.testing.assert_array_equal(record["detector_sampling_mrad"], [0.25, 0.5])
    assert record["sampling"] == (0.4, 0.4, 0.25, 0.5) and record["detector_shape"] == (12, 16)
    np.testing.assert_array_equal(record["valid_pixels"], np.eye(16, 12, dtype=bool).T)


# --- where the swap is refused


def test_a_transposed_master_copy_is_not_swapped_back(tmp_path):
    """An HDF5 copy carries the record as root attributes: asking it to swap again is refused before it is read."""
    from quantem.gpu import io

    np.save(tmp_path / "scan.npy", counts())
    copy = tmp_path / "swapped_master.h5"
    with io.load(tmp_path / "scan.npy", detector_transpose=True, verbose=False) as swapped:
        expected = swapped.read(output="native").copy()
        io.save(copy, swapped, verbose=False)
    with io.load(copy, verbose=False) as reopened:
        assert bool(reopened.metadata["detector_transpose"])
        assert_bits_equal(reopened.read(output="native"), expected)
    with pytest.raises(ValueError, match="already swapped"):
        io.load(copy, detector_transpose=True, verbose=False)



def test_detector_transpose_is_refused_where_it_cannot_apply(tmp_path):
    from quantem.gpu import io
    from tests.contracts.io.test_digitalmicrograph import write_dm4

    path = arina_master(tmp_path)
    with pytest.raises(NotImplementedError, match="paired layout"):
        io.load(path, scan_shape=SCAN, representation="paired", detector_transpose=True, verbose=False)
    dm = write_dm4(tmp_path / "scan.dm4", measured_values("float32", shape=(2, 3, 8, 6)))
    with pytest.raises(NotImplementedError, match="DigitalMicrograph"):
        io.load(dm, detector_transpose=True, verbose=False)
    np.save(tmp_path / "float.npy", measured_values("float32", shape=(2, 3, 8, 6)))
    with pytest.raises(NotImplementedError, match="scaled precision"):
        io.load(tmp_path / "float.npy", dtype="scaled_uint16", detector_transpose=True, verbose=False)
