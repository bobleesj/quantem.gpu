"""Original metadata survives exact measurement export and source-independent reopen."""

import hashlib

import numpy as np
import pytest

from quantem.gpu.io._qem_reference import load_array, read_envelope, save_array
from quantem.gpu.io.qem_validation import validate_qem
from quantem.gpu.io._qem_metadata import _validate_source_documents


@pytest.mark.parametrize("extension,content", [
    ("xml", '<root><vendor_field units="µs">49.6</vendor_field><!-- retained --></root>\n'),
    ("json", '{ "unknown": {"scientist": "Å", "values": [1, 2, 3]} }\n'),
])
def test_metadata_document_survives_float_roundtrip(tmp_path, extension, content):
    source = np.arange(128 * 128, dtype=np.float32).reshape(1, 1, 128, 128) / 8
    document = dict(filename=f"acquisition.{extension}", mediaType=f"application/{extension}",
                    content=content, sha256=hashlib.sha256(content.encode()).hexdigest())
    path = tmp_path / "measurements.qem"
    save_array(path, source, metadata={"source_documents": [document]})
    assert validate_qem(path)["integrity"] == "verified"
    header, _ = read_envelope(path)
    assert header["scientific_metadata"]["source_documents"] == [document]
    restored, _ = load_array(path)
    np.testing.assert_array_equal(restored.view(np.uint32), source.view(np.uint32))
    copy = tmp_path / "copy.qem"
    save_array(copy, restored, metadata={"scientific_metadata": header["scientific_metadata"]})
    assert read_envelope(copy)[0]["scientific_metadata"]["source_documents"] == [document]
    document["content"] += "changed"
    with pytest.raises(ValueError, match="checksum"):
        save_array(tmp_path / "corrupt.qem", source, metadata={"source_documents": [document]})


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_attachment_rejects_nonstandard_json_constants(constant):
    content = '{"value": ' + constant + '}'
    document = dict(filename="metadata.json", mediaType="application/json",
                    content=content, sha256=hashlib.sha256(content.encode()).hexdigest())
    with pytest.raises(ValueError, match="non-standard constant"):
        _validate_source_documents([document])
