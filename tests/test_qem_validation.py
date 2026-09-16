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
