"""Collection discovery, destinations and refusals of the ``quantem-gpu convert`` command."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from quantem.gpu.formats.qem import metadata as qem_metadata
from quantem.gpu.io import convert


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
    assert convert.find_masters(tmp_path) == [first, second]
    assert convert.find_masters(first) == [first]
    with pytest.raises(FileNotFoundError):
        convert.find_masters(tmp_path / "missing")


def test_destination_sits_beside_the_master_or_mirrors_the_collection(tmp_path):
    master = _acquisition(tmp_path / "day2" / "session", "gold_02", "uint16")
    assert convert.destination_for(master, tmp_path, None) == master.with_name("gold_02.qem")
    out = tmp_path / "copies"
    assert convert.destination_for(master, tmp_path, out) == out / "day2" / "session" / "gold_02.qem"
    assert convert.destination_for(master, master, out) == out / "gold_02.qem"


def test_convert_refuses_without_writing(tmp_path):
    real = _acquisition(tmp_path, "real", "complex64")
    result = convert.convert(real, tmp_path / "real.qem")
    assert "complex64" in result.skipped and not (tmp_path / "real.qem").exists()

    alone = tmp_path / "alone_master.h5"
    with h5py.File(alone, "w"):
        pass
    assert "no detector files" in convert.convert(alone, tmp_path / "alone.qem").skipped

    present = _acquisition(tmp_path, "present", "uint16")
    (tmp_path / "present.qem").write_bytes(b"keep")
    assert "exists" in convert.convert(present, tmp_path / "present.qem").skipped
    assert (tmp_path / "present.qem").read_bytes() == b"keep"


def test_arina_master_fields_become_scientific_metadata():
    metadata = {
        "entry/instrument/detector/description": "Dectris ARINA Si",
        "entry/instrument/detector/frame_time": 4.96e-05,
        "entry/instrument/detector/detectorSpecific/photon_energy": 200000.0,
        "detector_name": "Dectris ARINA Si",
    }
    scientific = qem_metadata.acquisition_metadata((2, 2, 8, 8), metadata)
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
    fields = convert.master_metadata(master)
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
    embedded = convert._embedded_master(master)
    assert embedded["name"] == "gold_01_master.h5"
    assert zlib.decompress(base64.b64decode(embedded["data"])) == master.read_bytes()


def test_master_links_select_data_not_orphan_files(tmp_path):
    master = _acquisition(tmp_path, "scan", "uint16")
    (tmp_path / "scan_data_999999.h5").write_bytes(b"old unrelated chunk")
    assert convert.detector_files(master) == [tmp_path / "scan_data_000001.h5"]


def test_arina_units_and_exposure_precede_frame_period():
    root = "entry/instrument/detector/"
    source = {root + "description": "ARINA", root + "count_time": 49.5,
              root + "count_time@units": "us", root + "frame_time": 49.6e-6,
              root + "detectorSpecific/photon_energy": 200,
              root + "detectorSpecific/photon_energy@units": "keV"}
    normalized = qem_metadata.acquisition_metadata((1, 1, 8, 8), source)
    quantities = normalized["electron_microscope"]
    assert quantities["electron_source/accelerating_voltage"]["value"] == 200
    assert quantities["scan_controller/regular_scan/dwell_time"]["value"] == pytest.approx(49.5)
    source[root + "description"] = "X-ray detector"
    assert not qem_metadata.acquisition_metadata((1, 1, 8, 8), source)["electron_microscope"]


@pytest.mark.parametrize("description,dtype", [
    (None, "f8"), (b"Dectris EIGER2 Si 4M", "f8"), (b"Dectris ARINA Si", "f4"), (b"Dectris ARINA Si", "f8"),
])
def test_loader_dwell_time_is_the_value_a_saved_copy_records(tmp_path, description, dtype):
    """A master's dwell time is its exposure, as the saved copy records it; the loader used to report the frame period.

    The rule is the same for every detector and for a time stored as float32:
    such masters reported no dwell time while the rule went through the ARINA record.
    """
    from quantem.gpu.formats.hdf5.master import get_metadata

    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        if description is not None:
            detector["description"] = description
        detector.create_dataset("count_time", data=np.dtype(dtype).type(4.95e-05))
        detector.create_dataset("frame_time", data=np.dtype(dtype).type(4.96e-05))
    loaded = get_metadata(master)
    saved = qem_metadata.recorded_metadata(qem_metadata.acquisition_metadata((1, 1, 8, 8), loaded))
    assert loaded["dwell_time_us"] == saved["dwell_time_us"] == pytest.approx(49.5)
    # a master without an exposure reports its frame period
    with h5py.File(master, "r+") as handle:
        del handle["entry/instrument/detector/count_time"]
    assert get_metadata(master)["dwell_time_us"] == pytest.approx(49.6)


def test_session_file_calibrates_its_own_acquisition(tmp_path):
    """The session's dataset.yaml gives the semi-angle, voltage and, through the
    file's own entry, the scan step; another series with the same number gets no
    scan step, and the attachment carries only the fields used, not the notes."""
    (tmp_path / "dataset.yaml").write_text(
        "schema: quantem.dataset/2\nsession:\n  name: s1\n  notes: private\n"
        "calibrations:\n  mag_5p1:\n    scan_sampling_A: [0.373, 0.41]\n"
        "microscope:\n  voltage_kV: 300\n  semiangle_mrad: 30\n"
        "files:\n  16:\n    master: scan_16_master.h5\n    mag: mag_5p1\n    notes: file note\n"
    )
    overrides, attachment = convert.session_calibration(tmp_path / "scan_16_master.h5")
    assert overrides["illumination_system/semi_convergence_angle"]["value"] == 30.0
    assert overrides["electron_source/accelerating_voltage"]["value"] == 300e3
    assert overrides["scan_controller/regular_scan/pixel_size_row"]["value"] == pytest.approx(0.373e-10)
    assert overrides["scan_controller/regular_scan/pixel_size_column"]["value"] == pytest.approx(0.41e-10)
    assert all(q["evidence"].startswith("dataset.yaml sha256:") for q in overrides.values())
    assert "private" not in attachment["content"] and "file note" not in attachment["content"]
    other, _ = convert.session_calibration(tmp_path / "other_16_master.h5")
    assert "scan_controller/regular_scan/pixel_size_row" not in other
    scientific = qem_metadata.acquisition_metadata(
        (2, 2, 8, 8), {"calibration_overrides": overrides, "source_documents": [attachment]})
    effective = qem_metadata.effective_metadata({}, scientific)
    assert effective["semiangle_mrad"] == 30.0 and effective["voltage_kV"] == 300.0
    assert effective["scan_sampling_A"] == pytest.approx([0.373, 0.41])


@pytest.mark.parametrize("text", [
    "schema: quantem.dataset/2\nmicroscope: {voltage_kV: 300",
    "schema: quantem.dataset/2\nfiles: [a, b]",
    "schema: quantem.dataset/2\nsession: s1",
    "- a\n- b",
    "schema: quantem.dataset/2\nmicroscope: {voltage_kV: 300, 1: a}\nfiles:\n  1:\n    master: x_master.h5\n    mag: [a]\n",
    "schema: quantem.dataset/2\ncalibrations:\n  default:\n    scan_sampling_A: 0.5\n",            # a step is [row, col]
    "schema: quantem.dataset/2\nfiles:\n  1:\n    master: x_master.h5\n    scan_sampling_A: [0.5, -1]\n",
])
def test_malformed_session_file_is_named_not_fatal(tmp_path, text):
    """A malformed dataset.yaml raises one ValueError naming the file (convert then copies without it), never a parser or
    attribute error that would stop a batch."""
    (tmp_path / "dataset.yaml").write_text(text)
    with pytest.raises(ValueError, match="dataset.yaml"):
        convert.session_calibration(tmp_path / "x_master.h5")


def test_session_file_of_another_schema_is_refused_naming_the_rewrite(tmp_path):
    """A dataset.yaml written before quantem.dataset/2 (one-number step, screen_rotation_deg) is refused by name; reading
    it would find no step and no rotation without saying so. convert then copies without it and records why."""
    (tmp_path / "dataset.yaml").write_text(
        "schema_version: 1\nmicroscope:\n  voltage_kV: 300\n  semiangle_mrad: 30\n  screen_rotation_deg: 12\n"
        "calibrations:\n  scan_sampling_A: 0.5\n  mag_5p1:\n    scan_sampling_A: 0.373\n"
        "files:\n  16:\n    master: scan_16_master.h5\n    mag: mag_5p1\n    rotation_deg: 0.0\n"
        "reference_structure:\n  cif: ./BaTiO3.cif\n")
    for read in (convert.session_calibration, convert.session_specimen):
        with pytest.raises(ValueError, match=r"schema None is not read.*quantem\.dataset/2.*tools/rewrite_dataset_yaml\.py"):
            read(tmp_path / "scan_16_master.h5")


def test_session_step_is_the_entry_pair_else_its_magnification_else_the_session_default(tmp_path):
    """quantem.live's order for the [row, col] step: the file entry's own, the calibrations row of its mag, then
    calibrations.default; a scan without an entry takes only the session default, never another entry's step."""
    (tmp_path / "dataset.yaml").write_text(
        "schema: quantem.dataset/2\nmicroscope:\n  semiangle_mrad: 30\n"
        "calibrations:\n  default:\n    scan_sampling_A: [0.5, 0.6]\n  mag_5p1:\n    scan_sampling_A: [0.373, 0.41]\n"
        "files:\n"
        "  16:\n    master: scan_16_master.h5\n    mag: mag_5p1\n    scan_sampling_A: [0.2, 0.3]\n"
        "  17:\n    master: scan_17_master.h5\n    mag: mag_5p1\n"
        "  18:\n    master: scan_18_master.h5\n    mag: mag_9p9\n")
    steps = {}
    for scan in ("scan_16", "scan_17", "scan_18", "other_16"):
        overrides, attachment = convert.session_calibration(tmp_path / f"{scan}_master.h5")
        steps[scan] = tuple(overrides[f"scan_controller/regular_scan/pixel_size_{axis}"]["value"] * 1e10 for axis in ("row", "column"))
        assert json.loads(attachment["content"])["calibration"]["scan_sampling_A"] == pytest.approx(list(steps[scan]))
    assert steps["scan_16"] == pytest.approx((0.2, 0.3))
    assert steps["scan_17"] == pytest.approx((0.373, 0.41))
    assert steps["scan_18"] == pytest.approx((0.5, 0.6))
    assert steps["other_16"] == pytest.approx((0.5, 0.6))



def test_session_listed_by_scan_number_calibrates_only_an_unambiguous_scan(tmp_path):
    """A session that lists scans by number alone gives the scan step to the one
    scan ending in that number, and says so in the evidence; two series ending
    in the same number get none."""
    (tmp_path / "dataset.yaml").write_text(
        "schema: quantem.dataset/2\ncalibrations:\n  mag_3p6:\n    scan_sampling_A: [0.525, 0.525]\nmicroscope:\n  semiangle_mrad: 30\n"
        "files:\n  0:\n    mag: mag_3p6\n    notes: sample_54___00\n"
    )
    (tmp_path / "sample_54___00_master.h5").write_bytes(b"")
    overrides, attachment = convert.session_calibration(tmp_path / "sample_54___00_master.h5")
    row = overrides["scan_controller/regular_scan/pixel_size_row"]
    assert row["value"] == pytest.approx(0.525e-10) and row["evidence"].endswith("files[0] by scan number")
    assert "sample_54___00" not in attachment["content"].replace("sample_54___00_master", "")
    (tmp_path / "sample_55___00_master.h5").write_bytes(b"")
    overrides, _ = convert.session_calibration(tmp_path / "sample_54___00_master.h5")
    assert "scan_controller/regular_scan/pixel_size_row" not in overrides


def test_number_entries_still_match_beside_named_entries(tmp_path):
    """Entries listed by number keep matching their scan when another entry names its master (quantem.live adds a named
    entry when it records a thickness); an entry that names a different master is never matched by its number."""
    files = {0: {"mag": "mag_3p6"}, "7": {"master": "sample_7_master.h5"}, 5: {"master": "other_5_master.h5"}}
    for name in ("sample_0_master.h5", "sample_5_master.h5", "sample_7_master.h5"):
        (tmp_path / name).write_bytes(b"")
    assert convert.session_file_entry(files, tmp_path / "sample_0_master.h5")[::2] == (0, "scan number")
    assert convert.session_file_entry(files, tmp_path / "sample_7_master.h5")[::2] == ("7", "name")
    assert convert.session_file_entry(files, tmp_path / "sample_5_master.h5") == (None, None, "")


def test_session_specimen_becomes_the_sample_group(tmp_path):
    """The declared specimen, its CIF and the file's own components and thickness estimates become the .qem sample
    group (thickness in angstrom), with the CIF text as a JSON document; another series gets no file fields; a header
    with the group validates and keeps it."""
    (tmp_path / "BaTiO3.cif").write_text("data_BaTiO3\n_cell_length_a 4.0\n")
    (tmp_path / "dataset.yaml").write_text(
        "schema: quantem.dataset/2\n"
        "specimen:\n  name: BaTiO3 film on SrTiO3\n  geometry: cross-section\n  components:\n"
        "    BTO:\n      role: film\n      chemical_formula: BaTiO3\n      cif: BaTiO3.cif\n      zone_axis: [0, 0, 1]\n"
        "    STO:\n      chemical_formula: SrTiO3\n      zone_axis: '[1-10]'\n"
        "files:\n  '38':\n    master: scan_38_master.h5\n    components_in_view: [BTO]\n    thickness:\n      BTO:\n"
        "        - method: ptychography_multislice\n          value_nm: 41\n          uncertainty_nm: 4\n"
        "          region:\n            rows: [320, 512]\n            cols: [128, 384]\n          preferred: true\n")
    sample, documents = convert.session_specimen(tmp_path / "scan_38_master.h5")
    estimate = sample["components"]["BTO"]["thickness_estimates"][0]
    assert (estimate["value"], estimate["unit"], estimate["uncertainty"]) == (410.0, "angstrom", 40.0)
    assert sample["components"]["STO"]["zone_axis"] == [1, -1, 0] and sample["components_in_view"] == ["BTO"]
    assert sample["provenance"] == "dataset.yaml" and sample["evidence"].endswith("files[38] by name")
    assert [d["filename"] for d in documents] == ["BaTiO3.cif.json"]
    assert json.loads(documents[0]["content"])["cif"].startswith("data_BaTiO3")
    assert sample["components"]["BTO"]["cif"]["sha256"] == documents[0]["sha256"]
    other, _ = convert.session_specimen(tmp_path / "other_38_master.h5")
    assert "components_in_view" not in other and "thickness_estimates" not in other["components"]["BTO"]
    scientific = qem_metadata.acquisition_metadata((2, 2, 8, 8), {"sample": sample, "source_documents": documents})
    qem_metadata.validate_header(dict(container="quantem.qem", container_version=1, codec="c", profile="c",
                                       shape=[2, 2, 8, 8], scientific_metadata=scientific))
    assert scientific["sample"] == sample


def test_session_specimen_reads_only_specimen_and_rejects_a_missing_cif(tmp_path):
    """quantem.dataset/2 declares the specimen under specimen; an older reference_structure block is not read."""
    (tmp_path / "BaTiO3.cif").write_text("data_BaTiO3\n")
    (tmp_path / "dataset.yaml").write_text("schema: quantem.dataset/2\nreference_structure:\n  cif: ./BaTiO3.cif\n")
    assert convert.session_specimen(tmp_path / "x_master.h5") == (None, [])
    (tmp_path / "dataset.yaml").write_text("schema: quantem.dataset/2\nspecimen:\n  components:\n    A:\n      cif: missing.cif\n")
    with pytest.raises(ValueError, match="missing.cif"):
        convert.session_specimen(tmp_path / "x_master.h5")


@pytest.mark.parametrize("specimen", [
    "specimen:\n  components: [BTO]\n",                                         # components not a mapping
    "specimen:\n  components:\n    BTO: BaTiO3\n",                             # a component as text
    "specimen:\n  components:\n    A:\n      zone_axis: [0.5, 0, 1]\n",        # not integers: refused, not truncated
    "specimen:\n  components:\n    A:\n      zone_axis: [null, 0, 1]\n",
    "specimen:\n  components:\n    A:\n      cif: ../../../etc/hostname\n",    # outside the session folder
    "specimen:\n  components:\n    A:\n      cif: /etc/hostname\n",
    "specimen:\n  components:\n    A: {}\nfiles:\n  '1':\n    master: x_1_master.h5\n    thickness: [40]\n",
    "specimen:\n  components:\n    A: {}\nfiles:\n  '1':\n    master: x_1_master.h5\n    thickness:\n      A:\n        method: nominal\n",
    "specimen:\n  components:\n    A: {}\nfiles:\n  '1':\n    master: x_1_master.h5\n    thickness:\n      A:\n        - method: nominal\n          value_nm: 4\n          range_nm: 5\n",
    "specimen:\n  components:\n    A: {}\nfiles:\n  '1':\n    master: x_1_master.h5\n    thickness:\n      A:\n        - method: nominal\n          value_nm: 4\n          preferred: 'no'\n",
    "specimen:\n  components:\n    A: {}\nfiles:\n  '1':\n    master: x_1_master.h5\n    thickness:\n      A:\n        - method: nominal\n          value_nm: 4\n          region: ZZZ\n",
    "- a\n- b\n",                                                                 # the top level is not a mapping
])
def test_a_malformed_specimen_is_one_named_error(tmp_path, specimen):
    """Every malformed specimen raises one ValueError naming dataset.yaml (convert then copies without it, with a note),
    never AttributeError or TypeError, and never embeds a file from outside the session folder."""
    (tmp_path / "dataset.yaml").write_text(specimen if specimen.startswith("-") else f"schema: quantem.dataset/2\n{specimen}")
    with pytest.raises(ValueError, match="dataset.yaml"):
        convert.session_specimen(tmp_path / "x_1_master.h5")


def test_specimen_directions_and_shared_cif(tmp_path):
    """'[10 0 1]' keeps its two-digit index; two components sharing one CIF embed it once."""
    (tmp_path / "X.cif").write_text("data_X\n")
    (tmp_path / "dataset.yaml").write_text("schema: quantem.dataset/2\nspecimen:\n  components:\n    A:\n      cif: X.cif\n      zone_axis: '[10 0 1]'\n"
                                           "    B:\n      cif: ./X.cif\n      zone_axis: '[1-10]'\n")
    sample, documents = convert.session_specimen(tmp_path / "x_1_master.h5")
    assert sample["components"]["A"]["zone_axis"] == [10, 0, 1] and sample["components"]["B"]["zone_axis"] == [1, -1, 0]
    assert len(documents) == 1 and sample["components"]["A"]["cif"] == sample["components"]["B"]["cif"]
