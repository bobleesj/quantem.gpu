"""Collection discovery, destinations and refusals of the ``quantem-gpu convert`` command."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from quantem.gpu.io import _qem_metadata, qem_conversion


def _acquisition(folder: Path, name: str, dtype: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    master = folder / f"{name}_master.h5"
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(f"{name}_data_000001.h5", "/entry/data/data")
    with h5py.File(folder / f"{name}_data_000001.h5", "w") as handle:
        handle["entry/data/data"] = np.zeros((4, 8, 8), dtype)
    return master


def test_find_masters_walks_a_collection_and_ignores_resource_forks(tmp_path):
    first = _acquisition(tmp_path / "day1", "gold_01", "uint16")
    second = _acquisition(tmp_path / "day2" / "session", "gold_02", "uint16")
    (tmp_path / "day1" / "._gold_01_master.h5").write_bytes(b"")
    assert qem_conversion.find_masters(tmp_path) == [first, second]
    assert qem_conversion.find_masters(first) == [first]
    with pytest.raises(FileNotFoundError):
        qem_conversion.find_masters(tmp_path / "missing")


def test_destination_sits_beside_the_master_or_mirrors_the_collection(tmp_path):
    master = _acquisition(tmp_path / "day2" / "session", "gold_02", "uint16")
    assert qem_conversion.destination_for(master, tmp_path, None) == master.with_name("gold_02.qem")
    out = tmp_path / "copies"
    assert qem_conversion.destination_for(master, tmp_path, out) == out / "day2" / "session" / "gold_02.qem"
    assert qem_conversion.destination_for(master, master, out) == out / "gold_02.qem"


def test_convert_refuses_without_writing(tmp_path):
    real = _acquisition(tmp_path, "real", "float32")
    result = qem_conversion.convert(real, tmp_path / "real.qem")
    assert "float32" in result.skipped and not (tmp_path / "real.qem").exists()

    alone = tmp_path / "alone_master.h5"
    with h5py.File(alone, "w"):
        pass
    assert "no detector files" in qem_conversion.convert(alone, tmp_path / "alone.qem").skipped

    present = _acquisition(tmp_path, "present", "uint16")
    (tmp_path / "present.qem").write_bytes(b"keep")
    assert "exists" in qem_conversion.convert(present, tmp_path / "present.qem").skipped
    assert (tmp_path / "present.qem").read_bytes() == b"keep"


def test_arina_master_fields_become_scientific_metadata():
    metadata = {
        "entry/instrument/detector/description": "Dectris ARINA Si",
        "entry/instrument/detector/frame_time": 4.96e-05,
        "entry/instrument/detector/detectorSpecific/photon_energy": 200000.0,
        "detector_name": "Dectris ARINA Si",
    }
    scientific = _qem_metadata.acquisition_metadata((2, 2, 8, 8), metadata)
    assert scientific["source_format"] == "dectris-arina-hdf5"
    assert scientific["source_metadata"]["entry/instrument/detector/frame_time"] == 4.96e-05
    assert "detector_name" not in scientific["source_metadata"]
    microscope = scientific["electron_microscope"]
    assert microscope["electron_source/accelerating_voltage"] == dict(
        value=200.0, unit="kV", provenance="source_metadata",
        evidence="entry/instrument/detector/detectorSpecific/photon_energy")
    assert microscope["scan_controller/regular_scan/dwell_time"]["unit"] == "us"
    assert microscope["scan_controller/regular_scan/dwell_time"]["value"] == pytest.approx(49.6)


def test_master_metadata_keeps_every_field_attribute_and_unit(tmp_path):
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle.attrs["default"] = "entry"
        detector = handle.create_group("entry/instrument/detector")
        detector["description"] = b"Dectris ARINA Si"
        detector["count_time"] = 4.95e-05
        detector["count_time"].attrs["units"] = "s"
        detector["detectorSpecific/data_collection_date"] = b"2026-04-15T13:11:43.329-07:00"
        detector["detectorSpecific/flatfield"] = np.ones((16, 16), np.float32)
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data_000001.h5", "/entry/data/data")
    fields = qem_conversion.master_metadata(master)
    detector = "entry/instrument/detector/"
    assert fields["@default"] == "entry"
    assert fields[detector + "count_time"] == 4.95e-05
    assert fields[detector + "count_time@units"] == "s"
    assert fields[detector + "detectorSpecific/data_collection_date"].startswith("2026-04-15")
    assert "embedded master" in fields[detector + "detectorSpecific/flatfield"]
    assert "scan_data_000001.h5" in fields["entry/data/data_000001"]
    assert fields["sourceFormat"] == "dectris-arina-hdf5"


def test_embedded_master_is_the_source_file_byte_for_byte(tmp_path):
    import base64
    import zlib

    master = _acquisition(tmp_path, "gold_01", "uint16")
    embedded = qem_conversion._embedded_master(master)
    assert embedded["name"] == "gold_01_master.h5"
    assert zlib.decompress(base64.b64decode(embedded["data"])) == master.read_bytes()


def test_master_links_select_data_not_orphan_files(tmp_path):
    master = _acquisition(tmp_path, "scan", "uint16")
    (tmp_path / "scan_data_999999.h5").write_bytes(b"old unrelated chunk")
    assert qem_conversion.detector_files(master) == [tmp_path / "scan_data_000001.h5"]


def test_arina_units_and_exposure_precede_frame_period():
    root = "entry/instrument/detector/"
    source = {root + "description": "ARINA", root + "count_time": 49.5,
              root + "count_time@units": "us", root + "frame_time": 49.6e-6,
              root + "detectorSpecific/photon_energy": 200,
              root + "detectorSpecific/photon_energy@units": "keV"}
    normalized = _qem_metadata.acquisition_metadata((1, 1, 8, 8), source)
    quantities = normalized["electron_microscope"]
    assert quantities["electron_source/accelerating_voltage"]["value"] == 200
    assert quantities["scan_controller/regular_scan/dwell_time"]["value"] == pytest.approx(49.5)
    source[root + "description"] = "X-ray detector"
    assert not _qem_metadata.acquisition_metadata((1, 1, 8, 8), source)["electron_microscope"]


def test_session_file_calibrates_its_own_acquisition(tmp_path):
    """The session's dataset.yaml gives the semi-angle, voltage and, through the
    file's own entry, the scan step; another series with the same number gets no
    scan step, and the attachment carries only the fields used, not the notes."""
    (tmp_path / "dataset.yaml").write_text(
        "schema_version: 1\nsession:\n  name: s1\n  notes: private\n"
        "calibrations:\n  mag_5p1:\n    scan_sampling_A: 0.373\n"
        "microscope:\n  voltage_kV: 300\n  semiangle_mrad: 30\n"
        "files:\n  16:\n    master: zoneB_16_master.h5\n    mag: mag_5p1\n"
    )
    overrides, attachment = qem_conversion.session_calibration(tmp_path / "zoneB_16_master.h5")
    assert overrides["illumination_system/semi_convergence_angle"]["value"] == 30.0
    assert overrides["electron_source/accelerating_voltage"]["value"] == 300e3
    assert overrides["scan_controller/regular_scan/pixel_size_row"]["value"] == pytest.approx(0.373e-10)
    assert overrides["scan_controller/regular_scan/pixel_size_column"]["value"] == pytest.approx(0.373e-10)
    assert all(q["evidence"].startswith("dataset.yaml sha256:") for q in overrides.values())
    assert "private" not in attachment["content"]
    other, _ = qem_conversion.session_calibration(tmp_path / "zoneA_16_master.h5")
    assert "scan_controller/regular_scan/pixel_size_row" not in other
    scientific = _qem_metadata.acquisition_metadata(
        (2, 2, 8, 8), {"calibration_overrides": overrides, "source_documents": [attachment]})
    effective = _qem_metadata.effective_metadata({}, scientific)
    assert effective["semiangle_mrad"] == 30.0 and effective["voltage_kV"] == 300.0
    assert effective["scan_sampling_A"] == pytest.approx([0.373, 0.373])
