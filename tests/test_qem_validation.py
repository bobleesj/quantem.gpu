"""Portable reference integrity, moved-file use and damaged-copy detection."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from quantem.gpu.io.qem_validation import validate_qem

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
    from quantem.gpu.io import _qem_metadata, qem_validation

    narrowed = dict(source_dtype="uint32", dtype="uint16", working_counts_exact=True,
                    hot_pixel_correction=dict(applied=True, method="median", pixel_count=4))
    scientific = _qem_metadata.acquisition_metadata((2, 2, 8, 8), narrowed)
    operations = {record["operation"]: record for record in scientific["processing"]}
    assert operations["exact_integer_narrowing"]["changes_measurements"] is False
    assert operations["flagged_pixel_replacement"]["changes_measurements"] is True
    header = dict(dtype="uint16", metadata=narrowed, scientific_metadata=scientific)
    qem_validation._validate_declared_processing(header)

    undeclared = _qem_metadata.acquisition_metadata((2, 2, 8, 8), {})
    header = dict(dtype="uint16", metadata=narrowed, scientific_metadata=undeclared)
    with pytest.raises(ValueError, match="flagged_pixel_replacement"):
        qem_validation._validate_declared_processing(header)
    header["metadata"] = dict(source_dtype="uint32")
    with pytest.raises(ValueError, match="exact_integer_narrowing"):
        qem_validation._validate_declared_processing(header)

    for source_dtype, stored_dtype in (("float32", "float16"), ("uint32", "uint16")):
        with pytest.raises(ValueError, match="processing provenance"):
            _qem_metadata.acquisition_metadata(
                (2, 2, 8, 8), dict(source_dtype=source_dtype, dtype=stored_dtype)
            )


def test_validate_command_reports_shared_reference(capsys):
    """Validate a downloaded reference through the documented public command."""
    from quantem.gpu.cli import main

    assert main(["validate", str(REFERENCES / "uint16.qem")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["integrity"] == "verified"
    assert report["measurements_changed_by"] == []
