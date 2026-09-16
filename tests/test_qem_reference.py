"""Portable scientific workflows, independent of accelerated codec implementations."""

from pathlib import Path
import os
import copy
import hashlib
import json

import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.io.qem_validation import validate_qem


def test_read_shared_native_files_without_a_gpu():
    root = Path(__file__).parent / "data/qem-v1"
    for name in ("uint8", "uint16"):
        with io.load(root / f"{name}.qem", backend="cpu") as acquisition:
            np.testing.assert_array_equal(
                acquisition.data, np.load(root / f"{name}.npy")
            )
            assert acquisition.metadata["backend"] == "cpu"


def test_frozen_portable_conformance_bundle():
    from quantem.gpu.io._qem_metadata import validate_header
    from quantem.gpu.io._qem_reference import read_envelope

    root = Path(__file__).parent / "data/qem-v2"
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["entries"]:
        path = root / (entry["name"] + ".qem")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        expected = np.load(root / (entry["name"] + ".npy"))
        with io.load(path, backend="cpu") as restored:
            assert restored.data.tobytes() == expected.tobytes()
            assert (
                hashlib.sha256(restored.data.tobytes()).hexdigest()
                == entry["counts_sha256"]
            )
    header, _ = read_envelope(root / "u16-multiple-chunks.qem")
    for case in json.loads((root / "invalid-metadata.json").read_text()):
        changed = copy.deepcopy(header)
        target = changed["scientific_metadata"]
        for key in case["path"][:-1]:
            target = target[key]
        if case.get("delete"):
            del target[case["path"][-1]]
        else:
            target[case["path"][-1]] = case["value"]
        with pytest.raises(ValueError):
            validate_header(changed)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_save_numpy_and_reopen_across_chunks(tmp_path, dtype):
    counts = np.random.default_rng(24).poisson(3, (3, 179, 17, 19)).astype(dtype)
    counts[:, :, 0, 0] = 0
    counts[:, :, 0, 1] = np.iinfo(dtype).max
    counts[:, :, 0, 2] = 0
    counts[0, 7, 0, 2] = 128
    counts[1, 3, 0, 2] = 1
    counts[2, 178, 16, 18] = np.iinfo(dtype).max
    path = tmp_path / "synthetic.qem"
    source = dict(
        scan_sampling_A=[0.4, 0.6],
        voltage_kV=300,
        source_metadata={"example": "synthetic, no acquisition data"},
    )
    io.save(path, counts, backend="cpu", metadata=source)
    inspection = io.inspect(path)
    assert inspection.metadata["scientific_metadata"]["schema"].endswith("/2")
    assert inspection.metadata["scan_sampling_A"] == pytest.approx([0.4, 0.6])
    assert validate_qem(path)["codec_layout"] == "verified"
    with io.load(path, backend="cpu") as restored:
        np.testing.assert_array_equal(restored.data, counts)
        assert restored.metadata["source_metadata"] == source["source_metadata"]
        assert restored.metadata["voltage_kV"] == 300
        np.testing.assert_array_equal(
            restored.data.sum(axis=(2, 3)), counts.sum(axis=(2, 3))
        )
    with pytest.raises(ValueError, match="non-existing"):
        io.save(path, counts, backend="cpu")


def test_float_measurement_bits_survive_without_quantization(tmp_path):
    words = np.random.default_rng(10).integers(
        0, 2**32, (1, 3, 128, 128), dtype=np.uint32
    )
    words.ravel()[:6] = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x3E800000]
    path = tmp_path / "float.qem"
    io.save(path, words.view(np.float32), backend="cpu")
    assert validate_qem(path)["codec_layout"] == "verified"
    with io.load(path, backend="cpu") as restored:
        np.testing.assert_array_equal(restored.data.view(np.uint32), words)
        assert restored.metadata["background_applied"] is False
        copied = tmp_path / "copied.qem"
        io.save(copied, restored, backend="cpu")
    with io.load(copied, backend="cpu") as restored:
        np.testing.assert_array_equal(restored.data.view(np.uint32), words)


def test_float_export_rejects_unrepresentable_validity_mask(tmp_path):
    valid = np.ones((128, 128), dtype=bool)
    valid[0, 0] = False
    path = tmp_path / "masked.qem"
    with pytest.raises(NotImplementedError, match="validity mask"):
        io.save(
            path,
            np.zeros((1, 1, 128, 128), np.float32),
            backend="cpu",
            metadata={"valid_pixels": valid},
        )
    assert not path.exists()


def test_float_export_does_not_mislabel_already_corrected_data(tmp_path):
    path = tmp_path / "corrected.qem"
    with pytest.raises(NotImplementedError, match="background-corrected"):
        io.save(
            path,
            np.zeros((1, 1, 128, 128), np.float32),
            backend="cpu",
            metadata={"background_applied": True},
        )
    assert not path.exists()


def test_empad_rectangular_modern_calibration_survives_conversion(tmp_path):
    raw = np.zeros((2, 3, 130, 128), dtype=np.float32)
    (tmp_path / "scan.raw").write_bytes(raw.tobytes())
    xml = tmp_path / "scan.xml"
    xml.write_text(
        '<root><raw_file filename="scan.raw"/><pix_y>2</pix_y>'
        "<pix_x>3</pix_x><type>scan</type><iom_measurements>"
        "<full_scan_field_of_view><x>6e-9</x><y>6e-9</y>"
        "<scale_factor>2</scale_factor></full_scan_field_of_view>"
        "<calibrated_pixelsize>1.826537060227288e-10</calibrated_pixelsize></iom_measurements></root>"
    )
    with io.load(xml, backend="cpu") as original:
        io.save(tmp_path / "calibrated.qem", original, backend="cpu")
    with io.load(tmp_path / "calibrated.qem", backend="cpu") as restored:
        assert restored.metadata["scan_sampling_A"] == pytest.approx([10, 10])
        assert restored.metadata["detector_sampling"] == pytest.approx(
            [0.01826537060227288] * 2
        )
        assert restored.metadata["detector_sampling_unit"] == "1/angstrom"
        assert restored.metadata["source_metadata"]["empad_xml"] == xml.read_text()


def test_empad_and_numpy_sources_convert_using_public_python_calls(tmp_path):
    raw = (
        np.arange(2 * 3 * 130 * 128, dtype=np.float32).reshape(2, 3, 130, 128) / 4 - 20
    )
    (tmp_path / "scan.raw").write_bytes(raw.tobytes())
    xml = tmp_path / "scan.xml"
    xml.write_text(
        '<root><raw_file filename="scan.raw"/><pix_y>2</pix_y>'
        "<pix_x>3</pix_x><type>scan</type><iom_measurements>"
        "<high_voltage>300000</high_voltage></iom_measurements></root>"
    )
    with io.load(xml, backend="cpu") as original:
        np.testing.assert_array_equal(original.data, raw[:, :, :128])
        io.save(tmp_path / "empad.qem", original, backend="cpu")
    with io.load(tmp_path / "empad.qem", backend="cpu") as restored:
        np.testing.assert_array_equal(restored.data, raw[:, :, :128])
        assert restored.metadata["voltage_kV"] == 300
        assert "empad_xml" in restored.metadata["source_metadata"]
    counts = np.arange(120, dtype=np.uint16).reshape(2, 3, 4, 5)
    np.save(tmp_path / "counts.npy", counts)
    with io.load(tmp_path / "counts.npy", backend="cpu") as original:
        io.save(tmp_path / "numpy.qem", original, backend="cpu")
    with io.load(tmp_path / "numpy.qem", backend="cpu") as restored:
        np.testing.assert_array_equal(restored.data, counts)


@pytest.mark.skipif(
    os.environ.get("QEM_TEST_BACKEND") != "mps", reason="Explicit Metal qualification"
)
def test_cpu_writer_opens_on_metal_with_exact_products(tmp_path):
    from quantem.gpu import detector

    counts = np.random.default_rng(91).poisson(2, (3, 179, 17, 19)).astype(np.uint16)
    counts[0, 0, 0, 0] = 65535
    valid = np.ones((17, 19), bool)
    valid[0, 0] = False
    path = tmp_path / "cpu-to-metal.qem"
    io.save(path, counts, backend="cpu", metadata={"valid_pixels": valid})
    with io.load(path, backend="mps") as restored:
        decoded = restored.data.decode_scan_range_device(0, 537)
        try:
            np.testing.assert_array_equal(
                decoded.to_numpy().reshape(counts.shape), counts
            )
        finally:
            decoded.release()
        session = detector.prepare(restored)
        try:
            mask = np.zeros((17, 19), bool)
            mask[2:15, 3:18] = True
            actual = session.masked_sum_exact(mask)
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(3, 179),
                (counts * (mask & valid)).sum(axis=(2, 3)),
            )
        finally:
            session.close()
