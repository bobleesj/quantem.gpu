"""Public loading with explicit, complete scientific source seals."""

import hashlib
from importlib import import_module

import pytest

from quantem.gpu import io
from quantem.gpu.io.integrity import _seal_source


def test_public_load_forwards_one_source_seal_and_rejects_conflicts(
    tmp_path, monkeypatch
):
    source = tmp_path / "packed.h5"
    source.write_bytes(b"QGPUH5\0\x01" + b"qualified source")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    integrity = _seal_source(source, expected)
    calls = []
    marker = object()

    def load_source(path, **options):
        calls.append((path, options))
        return marker

    monkeypatch.setattr(
        import_module("quantem.gpu.io.load"), "_load_lossless_packed", load_source
    )
    assert io.load(source, source_integrity=integrity) is marker
    assert calls[0][1]["source_integrity"] is integrity
    assert calls[0][1]["expected_source_sha256"] == expected
    with pytest.raises(ValueError, match="conflicts"):
        io.load(source, source_integrity=integrity, expected_source_sha256="b" * 64)
    assert len(calls) == 1


def test_sealed_ranges_cover_the_entire_source_and_detect_size_drift(tmp_path):
    source = tmp_path / "source.h5"
    source.write_bytes(b"some complete source")
    integrity = _seal_source(source, hashlib.sha256(source.read_bytes()).hexdigest())
    integrity.validate_source(source)
    with pytest.raises(ValueError, match="ranges"):
        io.SourceIntegrity(
            integrity.whole_file_sha256, integrity.file_bytes, 1, integrity.chunk_sha256
        )
    source.write_bytes(source.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="Source has"):
        integrity.validate_source(source)
