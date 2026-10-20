"""Open original files, query native DPs, and retain measurements in QEM."""

import os
import hashlib

import h5py
import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io import arrays
from quantem.gpu.formats.qem import reference
from quantem.gpu.formats.qem.validation import validate_qem


@pytest.fixture
def backend():
    selected = os.environ.get("QEM_TEST_BACKEND")
    if selected not in ("mps", "cuda"):
        pytest.skip("Set QEM_TEST_BACKEND=mps or cuda on a physical accelerator.")
    return selected


def _host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value.get()


@pytest.mark.parametrize(
    "detector_shape",
    [
        (1, 1),
        (3, 5),
        (100, 100),
        (128, 128),
        (192, 192),
        (210, 210),
        (256, 384),
        (1024, 1024),
    ],
)
def test_float_detector_geometries(tmp_path, backend, detector_shape):
    """Inspect native fractional DPs and export exact rectangular detector data."""
    shape = (2, 3, *detector_shape)
    values = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 101) / 8
    original, saved = tmp_path / "original.npy", tmp_path / "saved.qem"
    np.save(original, values)
    mask = np.zeros(detector_shape, bool)
    mask[::2, ::2] = True
    with io.load(original, backend=backend, verbose=False) as loaded:
        session = detector.prepare(loaded)
        np.testing.assert_array_equal(session.frame(5), values[-1, -1])
        np.testing.assert_allclose(session.mean_dp(), values.mean((0, 1)), atol=1e-6)
        np.testing.assert_allclose(
            session.masked_sum(mask), values[:, :, mask].sum(-1), rtol=1e-6
        )
        native_image = session.masked_sum(mask, output="native")
        assert not isinstance(native_image, np.ndarray)
        np.testing.assert_array_equal(_host(native_image), session.masked_sum(mask))
        np.testing.assert_allclose(
            session.reduce_frames([0, 5]),
            values.reshape(-1, *detector_shape)[[0, 5]].mean(0),
            atol=1e-6,
        )
        io.save(saved, loaded)
        assert loaded.data.peak_decode_bytes <= 32 << 20
    assert validate_qem(saved)["codec_layout"] == "verified"
    with io.load(saved, backend=backend, verbose=False) as loaded:
        np.testing.assert_array_equal(
            detector.prepare(loaded).frame(5).view(np.uint32),
            values[-1, -1].view(np.uint32),
        )


def test_existing_float_products_preserve_gpu_baseline(tmp_path, backend):
    """Geometry generalization preserves the previously frozen MPS products."""
    if backend != "mps":
        pytest.skip("Frozen MPS baseline, not a cross-backend reduction claim.")
    original = tmp_path / "baseline.npy"
    values = (
        np.arange(32 * 128 * 128, dtype=np.float32).reshape(4, 8, 128, 128) % 101
    ) / 8
    np.save(original, values)
    with io.load(original, backend=backend, verbose=False) as loaded:
        session = detector.prepare(loaded)
        results = [
            session.frame(17),
            session.mean_dp(),
            session.masked_sum(np.ones((128, 128), bool)),
        ]
        hashes = [
            hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for value in results
        ]
        assert hashes == [
            "e75bd9741c8a1c47be11ba8aafe78ab6a926b669b2d75bad3572981726ba16b5",
            "aeffd932799fc7baa7bd19c71567167c7fe426b155a235a0fcb2857879e21038",
            "11a9a47ab6e1cdd156eba157376643f73e921264e31ed0e48c273dac8a08557d",
        ]


@pytest.mark.parametrize(
    "format_name",
    [
        "numpy",
        "empad-raw",
        "empad-xml",
        "empad2-xml",
        "hdf5-4d",
        "hdf5-3d",
        "hdf5-gzip",
        "dm4",
        "dm3",
    ],
)
def test_float_original_to_qem_and_native_patterns(
    tmp_path, monkeypatch, backend, format_name
):
    """Original fractional measurements stay exact across bounded load/save/reopen."""
    values = (np.arange(5 * 7 * 128 * 128).reshape(5, 7, 128, 128) % 101).astype(
        np.float32
    ) / 8
    # Force multiple windows even on this small workflow fixture. Mapping the
    # whole source to device would violate the limit asserted below.
    limit = 3 * 128 * 128 * 4
    monkeypatch.setattr(arrays, "MAX_INGEST_BYTES", limit)

    def no_reference_encoder(*args, **kwargs):
        raise AssertionError(
            "Original GPU loading must not use the CPU reference encoder."
        )

    monkeypatch.setattr(reference, "_encode_stream", no_reference_encoder)
    if format_name == "numpy":
        original = tmp_path / "scan.npy"
        np.save(original, values)
    elif format_name.startswith("empad"):
        original = tmp_path / "scan.raw"
        record = np.zeros(
            (5, 7, 128 if format_name == "empad2-xml" else 130, 128), np.float32
        )
        record[:, :, :128] = values
        record.tofile(original)
        if format_name == "empad-xml":
            original = tmp_path / "scan.xml"
            original.write_text(
                '<root><pix_y>5</pix_y><pix_x>7</pix_x><raw_file filename="scan.raw"/></root>'
            )
        elif format_name == "empad2-xml":
            original = tmp_path / "scan.xml"
            original.write_text(
                "<root><sensor><type>EMPAD2</type><shape>(128,128)</shape></sensor>"
                "<scan><type>scan</type><shape>(5,7)</shape></scan>"
                "<rawfile><filename>scan.raw</filename><datatype>float32</datatype></rawfile>"
                "<pdcu><SerialNumber>synthetic</SerialNumber></pdcu>"
                "<grabber><avg_scan_even_offset>8000</avg_scan_even_offset>"
                "<avg_scan_odd_offset>8000</avg_scan_odd_offset></grabber></root>"
            )
    elif format_name in {"dm3", "dm4"}:
        from tests.contracts.io.test_digitalmicrograph import write_dm

        original = write_dm(
            tmp_path / f"scan.{format_name}", values, version=int(format_name[-1])
        )
    else:
        original = tmp_path / "scan.h5"
        with h5py.File(original, "w") as handle:
            frames = (
                values.reshape(-1, 128, 128) if format_name == "hdf5-3d" else values
            )
            handle.create_dataset(
                "entry/data/data",
                data=frames,
                compression="gzip" if format_name == "hdf5-gzip" else None,
            )
    saved = tmp_path / "scan.qem"
    inspected = io.inspect(original, scan_shape=(5, 7))
    assert inspected.ready and inspected.detector_shape == (128, 128)
    assert inspected.scan_shape == (5, 7) and inspected.dtype == "float32"
    options = {} if format_name == "numpy" else {"backend": backend}
    with io.load(original, scan_shape=(5, 7), verbose=False, **options) as loaded:
        assert loaded.representation.value == "encoded"
        assert loaded.metadata["backend"] == backend
        assert loaded.metadata["load_timings"]["peak_ingest_bytes"] <= limit
        if format_name in ("dm3", "dm4"):
            correction = loaded.metadata["hot_pixel_correction"]
            # io.load's default policy is recorded; DM stores no mask, so nothing is flagged or changed
            assert correction["method"] == "median"
            assert correction["pixel_count"] == 0
            assert correction["applied"] is False
        session = detector.prepare(loaded)
        pattern = session.frame(34, output="native")
        assert not isinstance(pattern, np.ndarray)
        np.testing.assert_array_equal(_host(pattern), values[-1, -1])
        mean = session.mean_dp(output="native")
        assert not isinstance(mean, np.ndarray)
        np.testing.assert_allclose(_host(mean), values.mean((0, 1)), atol=1e-6)
        mean[...] = 0
        np.testing.assert_allclose(session.mean_dp(), values.mean((0, 1)), atol=1e-6)
        io.save(saved, loaded)
    assert validate_qem(saved)["integrity"] == "verified"
    with io.load(saved, backend=backend, verbose=False) as reopened:
        assert reopened.representation.value == "encoded"
        np.testing.assert_array_equal(
            reopened.read(scan_region=(4, 5, 6, 7)).cpu().numpy().view(np.uint32),
            values[4:5, 6:7].view(np.uint32),
        )
        if format_name in ("empad-xml", "empad2-xml"):
            assert (
                "empad_xml"
                in reopened.metadata["scientific_metadata"]["source_metadata"]
            )
    if format_name == "empad2-xml":
        # The same retained raw offsets must not admit encoded sensor words.
        record.view(np.uint32)[:] = 0x40001234
        record.tofile(tmp_path / "scan.raw")
        with pytest.raises(NotImplementedError, match="encoded detector words"):
            io.load(original, backend=backend, verbose=False)


@pytest.mark.parametrize("format_name", ["numpy", "hdf5"])
@pytest.mark.parametrize("count_dtype", ["uint8", "uint16"])
def test_integer_original_native_outputs_and_qem(
    tmp_path, backend, format_name, count_dtype
):
    """Native integer patterns retain dtype and independent output ownership."""
    values = (np.arange(5 * 7 * 8 * 12).reshape(5, 7, 8, 12) * 37).astype(count_dtype)
    original = tmp_path / ("scan.npy" if format_name == "numpy" else "scan.h5")
    if format_name == "numpy":
        np.save(original, values)
    else:
        with h5py.File(original, "w") as handle:
            handle["entry/data/data"] = values
    with io.load(original, backend=backend, verbose=False) as loaded:
        session = detector.prepare(loaded)
        pattern = session.frame(34, output="native")
        assert not isinstance(pattern, np.ndarray)
        assert _host(pattern).dtype == np.dtype(count_dtype)
        np.testing.assert_array_equal(_host(pattern), values[-1, -1])
        mean = session.mean_dp(output="native")
        assert not isinstance(mean, np.ndarray)
        np.testing.assert_allclose(_host(mean), values.mean((0, 1)), rtol=1e-6)
        saved = tmp_path / "scan.qem"
        io.save(saved, loaded)
    np.testing.assert_array_equal(_host(pattern), values[-1, -1])
    with io.load(saved, backend=backend, verbose=False) as reopened:
        np.testing.assert_array_equal(
            detector.prepare(reopened).frame(34), values[-1, -1]
        )


@pytest.mark.parametrize("detector_shape", [(3, 5), (128, 128), (129, 131)])
def test_float_ingestion_preserves_ieee_bits(tmp_path, backend, detector_shape):
    """Fractional, signed-zero, subnormal and nonfinite bits survive GPU encoding."""
    words = np.random.default_rng(42).integers(
        0, 2**32, (2, 3, *detector_shape), dtype=np.uint32
    )
    words.ravel()[:7] = [
        0,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        1,
        0x3E800000,
    ]
    original, saved = tmp_path / "bits.npy", tmp_path / "bits.qem"
    np.save(original, words.view(np.float32))
    with io.load(original, backend=backend, verbose=False) as loaded:
        io.save(saved, loaded)
    assert validate_qem(saved)["integrity"] == "verified"
    with io.load(saved, backend=backend, verbose=False) as loaded:
        for row, column in np.ndindex(words.shape[:2]):
            result = loaded.read(scan_region=(row, row + 1, column, column + 1))
            np.testing.assert_array_equal(
                result.cpu().numpy().view(np.uint32)[0, 0], words[row, column]
            )


@pytest.mark.parametrize("maximum", [27, 1024])
def test_simulated_integer_counts_to_qem(
    tmp_path, backend, monkeypatch, maximum
):
    """Simulated int32 counts retain every value and their signed dtype."""
    values = (np.arange(6 * 80 * 80).reshape(2, 3, 80, 80) % (maximum + 1)).astype(
        "int32"
    )
    original, saved = tmp_path / "counts.npy", tmp_path / "counts.qem"
    np.save(original, values)
    monkeypatch.setattr(arrays, "MAX_INGEST_BYTES", 2 * 80 * 80 * 4)
    with io.load(original, backend=backend, verbose=False) as loaded:
        assert loaded.metadata["source_dtype"] == "int32"
        assert loaded.metadata["dtype"] == "int32"
        assert "count_range" not in loaded.metadata
        assert loaded.metadata["load_timings"]["peak_ingest_bytes"] <= 2 * 80 * 80 * 4
        session = detector.prepare(loaded)
        np.testing.assert_array_equal(session.frame(5), values[-1, -1])
        np.testing.assert_allclose(session.mean_dp(), values.mean((0, 1)), rtol=1e-6)
        io.save(saved, loaded)
    assert validate_qem(saved)["integrity"] == "verified"
    with io.load(saved, backend=backend, verbose=False) as reopened:
        np.testing.assert_array_equal(
            detector.prepare(reopened).frame(5), values[-1, -1]
        )
        assert all(record["operation"] != "exact_integer_narrowing"
                   for record in reopened.metadata["scientific_metadata"]["processing"])
    values[-1, -1, -1, -1] = 65536
    np.save(original, values)
    with io.load(original, backend=backend, verbose=False) as loaded:
        assert loaded.dtype == np.int32
        assert int(loaded[-1, -1, -1, -1]) == 65536


@pytest.mark.parametrize("format_name", ["numpy", "hdf5", "hdf5-gzip"])
def test_exact_promoted_float_measurements(tmp_path, backend, monkeypatch, format_name):
    """Simulation floats promoted to double retain every bit through QEM."""
    values = (np.arange(6 * 8 * 12, dtype=np.float32) / 101).reshape(2, 3, 8, 12)
    values[0, 0, 0, 0] = -0.0
    promoted = values.astype(np.float64)
    original = tmp_path / (
        "simulation.npy" if format_name == "numpy" else "simulation.mat"
    )
    saved = tmp_path / "simulation.qem"

    def write_source():
        if format_name == "numpy":
            np.save(original, promoted)
        else:
            with h5py.File(original, "w") as handle:
                handle.create_dataset(
                    "cbed",
                    data=promoted,
                    chunks=(2, 3, 2, 3) if format_name == "hdf5-gzip" else None,
                    compression="gzip" if format_name == "hdf5-gzip" else None,
                )

    write_source()
    limit = 2 * 8 * 12 * 24
    monkeypatch.setattr(arrays, "MAX_INGEST_BYTES", limit)
    with io.load(original, backend=backend, verbose=False) as loaded:
        assert loaded.metadata["source_dtype"] == "float64"
        assert loaded.metadata["working_dtype"] == "float64"
        assert "exact_float_narrowing" not in loaded.metadata
        assert loaded.metadata["load_timings"]["peak_ingest_bytes"] <= limit
        decoded = loaded.data.extract_diffraction_device(0, 0)
        try:
            assert decoded.get().tobytes() == promoted[0, 0].tobytes()
        finally:
            if hasattr(decoded, "release"):
                decoded.release()
        io.save(saved, loaded)
    assert validate_qem(saved)["integrity"] == "verified"
    with io.load(saved, backend=backend, verbose=False) as reopened:
        for index, expected in enumerate(promoted.reshape(6, 8, 12)):
            decoded = reopened.data.extract_diffraction_device(*divmod(index, 3))
            try:
                assert decoded.get().tobytes() == expected.tobytes()
            finally:
                if hasattr(decoded, "release"):
                    decoded.release()
        assert all(record["operation"] != "exact_float_narrowing"
                   for record in reopened.metadata["scientific_metadata"]["processing"])
    promoted[-1, -1, -1, -1] = np.nextafter(promoted[-1, -1, -1, -1], np.inf)
    write_source()
    with io.load(original, backend=backend, verbose=False) as loaded:
        decoded = loaded.data.extract_diffraction_device(1, 2)
        try:
            assert decoded.get().tobytes() == promoted[1, 2].tobytes()
        finally:
            if hasattr(decoded, "release"):
                decoded.release()


def test_empad_multiple_detector_regions(tmp_path, backend):
    """Open an XML acquisition with several virtual detectors and preserve them."""
    record = np.arange(6 * 130 * 128, dtype=np.float32).reshape(2, 3, 130, 128) / 8
    record.tofile(tmp_path / "scan.raw")
    source = tmp_path / "acquisition.xml"
    source.write_text(
        '<root><raw_file filename="scan.raw"/>'
        '<scan_parameters mode="search"><scan_resolution_y>64</scan_resolution_y>'
        "<scan_resolution_x>64</scan_resolution_x></scan_parameters>"
        '<scan_parameters mode="acquire"><scan_resolution_y>2</scan_resolution_y>'
        "<scan_resolution_x>3</scan_resolution_x></scan_parameters>"
        '<reconstruction_parameters><roimask roi_idx="0"><radius_outer>40</radius_outer></roimask>'
        '<roimask roi_idx="1"><radius_outer>60</radius_outer></roimask></reconstruction_parameters>'
        "<exposure_time>1</exposure_time><iom_measurements><high_voltage>300000</high_voltage>"
        "</iom_measurements></root>"
    )
    saved = tmp_path / "copy.qem"
    assert io.inspect(source).ready
    assert io.inspect(tmp_path / "scan.raw").scan_shape == (2, 3)
    with io.load(tmp_path / "scan.raw", backend=backend, verbose=False) as loaded:
        np.testing.assert_array_equal(
            detector.prepare(loaded).frame(5), record[-1, -1, :128]
        )
        io.save(saved, loaded)
    with io.load(saved, backend=backend, verbose=False) as loaded:
        metadata = loaded.metadata["scientific_metadata"]
        assert metadata["source_metadata"]["empad_xml"] == source.read_text()
        assert (
            metadata["source_metadata"][
                "empad/reconstruction_parameters/roimask[0]/radius_outer"
            ]
            == "40"
        )
        assert (
            metadata["source_metadata"][
                "empad/reconstruction_parameters/roimask[1]/radius_outer"
            ]
            == "60"
        )
        quantities = metadata["electron_microscope"]
        assert quantities["electron_source/accelerating_voltage"]["value"] == 300
        assert quantities["scan_controller/regular_scan/dwell_time"]["value"] == 1000
    (tmp_path / "other.xml").write_text(source.read_text())
    with pytest.raises(ValueError, match="Multiple XML acquisitions"):
        io.load(tmp_path / "scan.raw", backend=backend, verbose=False)


def test_prepared_bitshuffle_stack_found_by_inspect_loads_through_the_dataset_reader(tmp_path, backend):
    """A /dp stack that inspect selects is read as that dataset, not as an Arina chunk layout.

    Metal raised "No detector data chunks found" and CUDA routed the stack
    through the Arina loader when the dataset came from inspect rather than
    from an explicit dataset_path.
    """
    import hdf5plugin
    from scipy.io import savemat

    counts = np.random.default_rng(9).integers(0, 900, (12, 16, 16), dtype=np.uint16)
    stack = tmp_path / "data_roi0_dp.hdf5"
    with h5py.File(stack, "w") as handle:
        handle.create_dataset("dp", data=counts, chunks=(1, 16, 16), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    savemat(tmp_path / "params_backup.mat", dict(
        Np_p=np.array([16, 16]), voltage=300.0, alpha=21.4, dk=0.02, dx=0.4, crop_idx0=np.array([1, 4, 1, 3]),
    ))
    assert io.inspect(stack).metadata["dataset_path"] == "/dp"
    with io.load(stack, backend=backend, verbose=False) as loaded:
        assert loaded.metadata["load_timings"]["storage_reader"] == "bounded-hdf5"
        np.testing.assert_array_equal(_host(loaded[:, :]).reshape(12, 16, 16), counts)


@pytest.mark.parametrize("stored", ["every frame", "one frame unfiltered", "last frame never written"])
def test_gzip_frame_stack_loads_every_stored_value(tmp_path, backend, stored):
    """A /dp stack of one gzip chunk per pattern (the TCMEP layout) inflates on several cores.

    The stored bytes are read chunk by chunk and inflated in parallel; a chunk
    whose filter the writer skipped is copied as stored, and a stack with an
    unwritten chunk keeps HDF5's own read, which fills it with zeros.
    """
    import zlib

    intensities = np.random.default_rng(13).gamma(2.0, 40.0, (3 * 300, 16, 16)).astype(np.float32)
    stack = tmp_path / "data_roi0_dp.hdf5"
    with h5py.File(stack, "w") as handle:
        data = handle.create_dataset("dp", shape=intensities.shape, dtype=np.float32, chunks=(1, 16, 16), compression="gzip")
        frames = len(intensities) - (stored == "last frame never written")
        for index in range(frames):
            unfiltered = stored == "one frame unfiltered" and index == 517
            payload = intensities[index].tobytes() if unfiltered else zlib.compress(intensities[index].tobytes(), 4)
            data.id.write_direct_chunk((index, 0, 0), payload, filter_mask=int(unfiltered))
    expected = intensities.copy()
    expected[frames:] = 0
    with io.load(stack, backend=backend, dataset_path="dp", scan_shape=(3, 300), verbose=False) as loaded:
        assert loaded.metadata["load_timings"]["storage_reader"] == "bounded-hdf5"
        restored = _host(loaded[:, :]).reshape(expected.shape)
        assert restored.dtype == np.float32 and restored.tobytes() == expected.tobytes()
