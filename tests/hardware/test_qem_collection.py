"""Convert an ARINA acquisition without changing counts or microscope metadata."""

import os

import h5py
import numpy as np
import pytest

from quantem.gpu.io import convert
from quantem.gpu.formats.qem.snapshot import read_header


def _master(tmp_path, dtype):
    import hdf5plugin

    values = np.random.default_rng(71).integers(0, 20, (1024, 16, 16), dtype=dtype)
    values[:, 2, 3] = np.arange(1024)  # flagged measurements must not be replaced
    mask = np.zeros((16, 16), np.uint8)
    mask[2, 3] = 1
    chunk = tmp_path / "detector.h5"
    with h5py.File(chunk, "w") as handle:
        handle.create_dataset("entry/data/data", data=values, chunks=(1, 16, 16),
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        # Simulate source-file overhead, ensuring conversion exercises publication.
        handle["unused_storage"] = np.zeros(1 << 20, np.uint8)
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(chunk.name, "/entry/data/data")
        root = "entry/instrument/detector/"
        handle[root + "description"] = b"Dectris ARINA Si"
        handle[root + "detectorSpecific/ntrigger"] = np.uint32(1024)
        handle[root + "detectorSpecific/pixel_mask"] = mask
        handle[root + "detectorSpecific/photon_energy"] = 200000.0
        handle[root + "count_time"] = 49.5e-6
        handle[root + "frame_time"] = 49.6e-6
    return master


@pytest.fixture
def backend():
    selected = os.environ.get("QEM_TEST_BACKEND", "mps")
    if selected == "cuda":
        cp = pytest.importorskip("cupy")
        assert cp.cuda.runtime.getDeviceCount() > 0
    else:
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("Requires Metal or QEM_TEST_BACKEND=cuda")
    return selected


def test_collection_preserves_flagged_counts_and_master(tmp_path, backend):
    master = _master(tmp_path, np.uint16)
    original_master = master.read_bytes()
    destination = tmp_path / "scan.qem"
    result = convert.convert(master, destination, backend=backend)
    assert not result.skipped and not result.larger and result.verified is True
    assert result.verification["compared_values"] == 1024 * 16 * 16
    assert result.verification["differing_values"] == 0
    assert result.verification["flagged_pixels"] == 1
    header, _ = read_header(destination)
    quantities = header["scientific_metadata"]["electron_microscope"]
    assert quantities["electron_source/accelerating_voltage"]["value"] == 200
    assert quantities["scan_controller/regular_scan/dwell_time"]["value"] == pytest.approx(49.5)
    restored = convert.restore_master(destination, tmp_path / "restored.h5")
    assert restored.read_bytes() == original_master == master.read_bytes()
    assert convert.convert(master, destination, backend=backend).skipped
    assert not list(tmp_path.glob(".qem-convert-*"))


def test_collection_warns_of_master_calibration_it_does_not_read(tmp_path, backend):
    """Converting a master that holds calibration this version does not read warns, and the copy is uncalibrated there.

    ``convert`` loads quietly. While the notice was a line of the verbose load report, such a master became a
    ``.qem`` without its detector calibration and without a word.
    """
    import warnings

    from quantem.gpu import io

    master = _master(tmp_path, np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert convert.convert(master, tmp_path / "current.qem", backend=backend).verified is True
    with h5py.File(master, "r+") as handle:
        handle.attrs["scan_sampling_A"] = [0.4, 0.6]
        handle.attrs["detector_sampling"] = [0.2, 0.3]
        handle.attrs["detector_sampling_unit"] = "mrad"
    with pytest.warns(UserWarning, match="Calibration not read: scan_master.h5 holds detector_sampling, detector_sampling_unit") as caught:
        result = convert.convert(master, tmp_path / "earlier.qem", backend=backend)
    assert len(caught) == 1 and result.verified is True
    with io.load(tmp_path / "earlier.qem", backend=backend, verbose=False) as copy:
        assert copy.sampling == (0.4, 0.6, None, None)
        assert copy.units == ("angstrom", "angstrom", None, None)


def test_uint32_flagged_markers_and_wide_counts_survive_conversion(tmp_path, backend):
    from quantem.gpu import io

    master = _master(tmp_path, np.uint32)
    result = convert.convert(master, tmp_path / "fits.qem", backend=backend)
    assert result.verified is True
    assert result.verification["flagged_markers_stored_as_zero"] == 0
    header, _ = read_header(tmp_path / "fits.qem")
    assert header["metadata"]["source_dtype"] == "uint32"
    assert header["metadata"]["file_counts_exact"] is True

    # Arina marks flagged pixels of a uint32 file with 0xFFFFFFFF in every frame.
    with h5py.File(tmp_path / "detector.h5", "r+") as handle:
        handle["entry/data/data"][:, 2, 3] = 0xFFFFFFFF
        source = handle["entry/data/data"][()]
    with h5py.File(master, "r") as handle:
        mask = handle["entry/instrument/detector/detectorSpecific/pixel_mask"][()]
    result = convert.convert(master, tmp_path / "markers.qem", backend=backend)
    assert result.verified is True
    assert result.verification["flagged_markers_stored_as_zero"] == 0
    header, _ = read_header(tmp_path / "markers.qem")
    assert header["metadata"]["flagged_markers_stored_as_zero"] == 0
    assert header["metadata"]["file_counts_exact"] is True
    operations = {record["operation"] for record in header["scientific_metadata"]["processing"]}
    assert "flagged_marker_zeroing" not in operations
    with io.load(tmp_path / "markers.qem", backend=backend, apply_mask=False, verbose=False) as loaded:
        counts = loaded.read().cpu().numpy().reshape(source.shape)
        np.testing.assert_array_equal(np.asarray(loaded.metadata["pixel_mask"]), mask)
    valid = mask == 0
    np.testing.assert_array_equal(counts[:, valid], source[:, valid])
    np.testing.assert_array_equal(counts[:, ~valid], source[:, ~valid])

    # A wide valid count must remain native uint32, just like a flagged marker.
    with h5py.File(tmp_path / "detector.h5", "r+") as handle:
        handle["entry/data/data"][0, 1, 1] = 70000
    wider = convert.convert(master, tmp_path / "wide.qem", backend=backend)
    assert wider.verified is True
    with io.load(tmp_path / "wide.qem", backend=backend, verbose=False) as loaded:
        assert loaded.dtype == np.uint32
        assert int(loaded[0, 0, 1, 1]) == 70000


@pytest.mark.parametrize("largest, stored", [(255, "uint8"), (1023, "uint16")])
def test_conversion_stores_the_narrowest_exact_type(tmp_path, backend, largest, stored):
    """A uint32 master whose counts fit 8 or 16 bits converts to uint8 or uint16, verifies by value and reloads exactly."""
    from quantem.gpu import io
    from quantem.gpu.formats.qem.snapshot import INTEGER_CODEC

    master = _master(tmp_path, np.uint32)
    with h5py.File(tmp_path / "detector.h5", "r+") as handle:
        handle["entry/data/data"][:, 2, 3] = np.arange(1024) % (largest + 1)
        source = handle["entry/data/data"][()]
    result = convert.convert(master, tmp_path / "archive.qem", backend=backend)
    assert result.verified is True and result.verification["differing_values"] == 0
    assert result.verification["compared_values"] == source.size
    header, _ = read_header(tmp_path / "archive.qem")
    narrow = backend == "mps"  # the CUDA loader keeps uint32 storage
    assert header["dtype"] == (stored if narrow else "uint32")
    assert header["metadata"]["source_dtype"] == "uint32"
    assert header["codec"] == (INTEGER_CODEC if narrow else header["codec"])
    with io.load(tmp_path / "archive.qem", backend=backend, apply_mask=False, verbose=False) as loaded:
        assert loaded.dtype == np.dtype(header["dtype"])
        np.testing.assert_array_equal(loaded.read().cpu().numpy().reshape(source.shape), source)
    with io.load(tmp_path / "archive.qem", backend=backend, apply_mask=False, dtype="native",
                 verbose=False) as native:
        assert native.dtype == np.uint32
        counts = native.read().cpu().numpy()
        assert counts.dtype == np.uint32
        np.testing.assert_array_equal(counts.reshape(source.shape), source)

    # io.save of the loaded acquisition writes the same narrow type and reloads equal to the file.
    with io.load(master, backend=backend, scan_shape=(32, 32), hot_pixel_correction="none",
                 verbose=False) as loaded:
        assert loaded.dtype == np.dtype(header["dtype"])
        io.save(tmp_path / "analysis.qem", loaded)
    saved, _ = read_header(tmp_path / "analysis.qem")
    assert (saved["dtype"], saved["metadata"]["source_dtype"]) == (header["dtype"], "uint32")
    with io.load(tmp_path / "analysis.qem", backend=backend, apply_mask=False, verbose=False) as reloaded:
        np.testing.assert_array_equal(reloaded.read().cpu().numpy().reshape(source.shape), source)


def test_failed_verification_does_not_publish_a_copy(tmp_path, backend, monkeypatch):
    master = _master(tmp_path, np.uint16)
    monkeypatch.setattr(convert, "verify_against_source",
                        lambda *args, **kwargs: {"identical": False})
    result = convert.convert(master, tmp_path / "rejected.qem", backend=backend)
    assert result.failed and result.verified is False
    assert not (tmp_path / "rejected.qem").exists()
    assert not list(tmp_path.glob(".qem-convert-*"))


def test_convert_command_writes_a_qem_copy_with_the_source_counts(tmp_path, backend, capsys):
    from quantem.gpu import io
    from quantem.gpu.cli import main

    master = _master(tmp_path, np.uint16)
    with h5py.File(tmp_path / "detector.h5", "r") as handle:
        expected = handle["entry/data/data"][()]
    out = tmp_path / "copies"
    assert main(["convert", str(master), "--out", str(out), "--backend", backend]) == 0
    printed = capsys.readouterr().out
    assert f"verified: {1024 * 16 * 16:,} values identical" in printed
    assert "1 flagged pixels also verified" in printed
    copy = out / "scan.qem"
    with io.load(copy, backend=backend, apply_mask=False, verbose=False) as loaded:
        counts = loaded.read().cpu().numpy()
    np.testing.assert_array_equal(counts.reshape(expected.shape), expected)


@pytest.mark.parametrize("sidecar", [None, "dataset.yaml", "dataset.yml"])
def test_command_calibrates_and_repairs_one_acquisition(tmp_path, backend, capsys, sidecar):
    from quantem.gpu import io
    from quantem.gpu.cli import main

    master = _master(tmp_path, np.uint32)
    with h5py.File(tmp_path / "detector.h5", "r+") as handle:
        handle["entry/data/data"][:, 2, 3] = 0xFFFFFFFF
        original = handle["entry/data/data"][()]
    if sidecar:
        (tmp_path / sidecar).write_text(
            "schema: quantem.dataset/2\nmicroscope:\n  voltage_kV: 80\n"
            "  semiangle_mrad: 10\n  detector_sampling_mrad: [0.2, 0.3]\n"
            "calibrations:\n  default:\n    scan_sampling_A: [0.1, 0.2]\n"
        )
    out = tmp_path / "copies"
    assert main([
        "convert", str(master), "--out", str(out), "--backend", backend,
        "--voltage-kv", "300", "--semiangle-mrad", "30",
        "--scan-sampling-a", "0.5", "--detector-sampling-mrad", "0.8", "0.9",
        "--hot-pixel-correction", "median",
    ]) == 0
    printed = capsys.readouterr().out
    assert "median correction: 1 detector pixels" in printed
    assert "1,024 replacements checked" in printed
    expected = original.copy()
    neighbors = [(r, c) for r in range(1, 4) for c in range(2, 5) if (r, c) != (2, 3)]
    # np.round rounds a .5 median half to even, as the fill does
    expected[:, 2, 3] = np.round(np.median(np.stack([original[:, r, c] for r, c in neighbors], axis=1), axis=1))
    with io.load(out / "scan.qem", backend=backend, verbose=False) as data:
        assert data.sampling == pytest.approx((0.5, 0.5, 0.8, 0.9))
        assert data.metadata["voltage_kV"] == 300 and data.metadata["semiangle_mrad"] == 30
        assert data.metadata["hot_pixel_correction"]["coordinates_row_col"] == [[2, 3]]
        np.testing.assert_array_equal(data.read().cpu().numpy().reshape(expected.shape), expected)
        records = data.metadata["scientific_metadata"]
        assert all("Explicit conversion parameter" in value["evidence"] for value in records["calibration_overrides"].values())
        assert any(record["changes_measurements"] for record in records["processing"])
    with h5py.File(tmp_path / "detector.h5", "r") as handle:
        np.testing.assert_array_equal(handle["entry/data/data"][()], original)


def test_conversion_retains_session_detector_sampling_without_cli_values(tmp_path, backend):
    from quantem.gpu import io

    master = _master(tmp_path, np.uint16)
    (tmp_path / "dataset.yml").write_text(
        "schema: quantem.dataset/2\nmicroscope:\n  semiangle_mrad: 30\n"
        "  detector_sampling_mrad: 0.8\ncalibrations:\n  default:\n    scan_sampling_A: [0.5, 0.6]\n"
    )
    destination = tmp_path / "calibrated.qem"
    result = convert.convert(master, destination, backend=backend)
    assert result.verified is True
    assert all("dataset.yml sha256:" in value["evidence"] for value in result.session_calibration.values())
    with io.load(destination, backend=backend, verbose=False) as data:
        assert data.sampling == pytest.approx((0.5, 0.6, 0.8, 0.8))
        assert data.metadata["voltage_kV"] == 200  # source value, no session or CLI override
        assert data.metadata["semiangle_mrad"] == 30


@pytest.mark.parametrize("method", ["median", "zero"])
def test_requested_mask_correction_keeps_wide_valid_counts(tmp_path, backend, method):
    from quantem.gpu import io

    master = _master(tmp_path, np.uint32)
    with h5py.File(master, "r+") as handle:
        mask_path = "entry/instrument/detector/detectorSpecific/pixel_mask"
        mask = handle[mask_path][()]
        mask[0, 0] = mask[0, 1] = 1
        handle[mask_path][...] = mask
    with h5py.File(tmp_path / "detector.h5", "r+") as handle:
        values = handle["entry/data/data"][()]
        values[:, :4, :5] += np.uint32(0xFFFFFF00)
        values[:, mask != 0] = np.uint32(0xFFFFFFFF)
        handle["entry/data/data"][...] = values
    expected = values.copy()
    for row, col in np.argwhere(mask):
        neighbors = [(r, c)
                     for r in range(max(0, row - 1), min(16, row + 2))
                     for c in range(max(0, col - 1), min(16, col + 2))
                     if mask[r, c] == 0]
        replacement = 0
        if method == "median" and neighbors:
            selected = np.sort(np.stack([values[:, r, c] for r, c in neighbors], axis=1), axis=1)
            n = selected.shape[1]
            total = selected[:, (n - 1) // 2].astype(np.uint64) + selected[:, n // 2].astype(np.uint64)
            lower = total // 2
            replacement = lower + (total & lower & np.uint64(1))  # a .5 median rounds half to even
        expected[:, row, col] = replacement
    destination = tmp_path / "corrected.qem"
    result = convert.convert(master, destination, backend=backend,
                             hot_pixel_correction=method)
    assert result.verified is True and not result.failed
    assert result.verification["corrected_values_compared"] == 1024 * int(np.count_nonzero(mask))
    with io.load(destination, backend=backend, apply_mask=False, verbose=False) as reopened:
        decoded = reopened.read().cpu().numpy().reshape(expected.shape)
    np.testing.assert_array_equal(decoded, expected)
    with h5py.File(tmp_path / "detector.h5", "r") as handle:
        np.testing.assert_array_equal(handle["entry/data/data"][()], values)
