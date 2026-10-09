"""Portable reference integrity, moved-file use and damaged-copy detection."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from quantem.gpu.formats.qem.validation import validate_qem

REFERENCES = Path(__file__).parent / "data" / "qem-v1"


@pytest.mark.parametrize("name", ["uint8", "uint16"])
def test_shareable_reference_after_moving(name, tmp_path):
    manifest = json.loads((REFERENCES / "manifest.json").read_text())
    entry = next(item for item in manifest["entries"] if item["name"] == name)
    for filename, expected in entry["files"].items():
        assert (
            hashlib.sha256((REFERENCES / filename).read_bytes()).hexdigest() == expected
        )
    original = np.load(REFERENCES / f"{name}.npy")
    assert hashlib.sha256(original.tobytes()).hexdigest() == entry["counts_sha256"]
    relocated = tmp_path / "shared.data"
    shutil.copyfile(REFERENCES / f"{name}.qem", relocated)
    report = validate_qem(relocated)
    assert report["integrity"] == report["codec_layout"] == "verified"
    assert report["decoded_parity"] == "not_checked"
    assert report["shape"] == entry["shape"]
    assert report["dtype"] == name
    assert report["metadata_coverage"] == "reader-retained"


def test_damaged_downloads_are_not_accepted(tmp_path):
    complete = (REFERENCES / "uint8.qem").read_bytes()
    for name, content in (
        ("truncated", complete[:-1]),
        ("header", complete[:56] + bytes([complete[56] ^ 1]) + complete[57:]),
        ("body", complete[:-1] + bytes([complete[-1] ^ 1])),
    ):
        path = tmp_path / f"{name}.qem"
        path.write_bytes(content)
        with pytest.raises(ValueError):
            validate_qem(path)


def test_stored_changes_must_be_declared_in_processing():
    from quantem.gpu.formats.qem import metadata, snapshot

    narrowed = dict(source_dtype="uint32", dtype="uint16", working_counts_exact=True,
                    hot_pixel_correction=dict(applied=True, method="median", pixel_count=4))
    scientific = metadata.acquisition_metadata((2, 2, 8, 8), narrowed)
    operations = {record["operation"]: record for record in scientific["processing"]}
    assert operations["exact_integer_narrowing"]["changes_measurements"] is False
    assert operations["flagged_pixel_replacement"]["changes_measurements"] is True
    header = dict(dtype="uint16", metadata=narrowed, scientific_metadata=scientific)
    snapshot.validate_declared_processing(header)

    undeclared = metadata.acquisition_metadata((2, 2, 8, 8), {})
    header = dict(dtype="uint16", metadata=narrowed, scientific_metadata=undeclared)
    with pytest.raises(ValueError, match="flagged_pixel_replacement"):
        snapshot.validate_declared_processing(header)
    header["metadata"] = dict(source_dtype="uint32")
    with pytest.raises(ValueError, match="exact_integer_narrowing"):
        snapshot.validate_declared_processing(header)

    for source_dtype, stored_dtype in (("float32", "float16"), ("uint32", "uint16")):
        with pytest.raises(ValueError, match="processing provenance"):
            metadata.acquisition_metadata(
                (2, 2, 8, 8), dict(source_dtype=source_dtype, dtype=stored_dtype)
            )

    exact_float = {
        "source_dtype": "float64", "dtype": "float32", "file_counts_exact": True,
        "exact_float_narrowing": {"method": "float64-float32-float64-bitwise"},
    }
    scientific = metadata.acquisition_metadata((2, 2, 8, 8), exact_float)
    header = {"dtype": "float32", "metadata": exact_float, "scientific_metadata": scientific}
    snapshot.validate_declared_processing(header)
    scientific["processing"][1]["stored_dtype"] = "float16"
    with pytest.raises(ValueError, match="matching exact_float_narrowing"):
        snapshot.validate_declared_processing(header)

    # Uncorrected uint32 flagged-pixel markers stored as 0 are declared, never silent.
    markers = dict(source_dtype="uint32", dtype="uint16", working_counts_exact=True,
                   file_counts_exact=False, flagged_markers_stored_as_zero=4096)
    scientific = metadata.acquisition_metadata((2, 2, 8, 8), markers)
    operations = {record["operation"]: record for record in scientific["processing"]}
    assert operations["flagged_marker_zeroing"] == dict(
        operation="flagged_marker_zeroing", changes_measurements=False, value_count=4096)
    snapshot.validate_declared_processing(dict(dtype="uint16", metadata=markers, scientific_metadata=scientific))
    scientific["processing"] = [record for record in scientific["processing"]
                                if record["operation"] != "flagged_marker_zeroing"]
    with pytest.raises(ValueError, match="flagged_marker_zeroing"):
        snapshot.validate_declared_processing(dict(dtype="uint16", metadata=markers, scientific_metadata=scientific))


def test_validate_command_reports_shared_reference(capsys):
    """Validate a downloaded reference through the documented public command."""
    from quantem.gpu.cli import main

    assert main(["validate", str(REFERENCES / "uint16.qem")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["integrity"] == "verified"
    assert report["measurements_changed_by"] == []


def test_convert_command_names_a_missing_source_in_one_line(tmp_path):
    """A mistyped source path, expect one quantem-gpu line and exit status 1, not a traceback."""
    from quantem.gpu.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["convert", str(tmp_path / "scan_master.h5")])
    assert stop.value.code == f"quantem-gpu convert: {tmp_path / 'scan_master.h5'} is neither a master file nor a folder."


def test_serve_command_refuses_a_missing_data_folder(tmp_path):
    """A mistyped data folder, expect one quantem-gpu line before any server starts (it used to serve nothing)."""
    from quantem.gpu.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["serve", str(tmp_path / "missing"), "--port", "18799"])
    assert stop.value.code == f"quantem-gpu serve: {tmp_path / 'missing'} is not a folder"
