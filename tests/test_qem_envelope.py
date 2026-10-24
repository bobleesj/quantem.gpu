"""A ``.qem`` file written in one pass is its definition, byte for byte."""

import hashlib
import json
import struct

import pytest

from quantem.gpu.formats.qem import metadata, snapshot


@pytest.mark.parametrize("mode", ["generator", "resident", "short-write", "fallback"])
def test_envelope_is_header_then_body_with_zero_gaps_and_block_checksums(tmp_path, monkeypatch, mode):
    """Spans that cross checksum blocks and leave alignment gaps, produced from two reused buffers."""
    monkeypatch.setattr(snapshot, "BLOCK_BYTES", 64)
    pieces = [(0, bytes(range(100))), (104, bytes(range(200, 230))), (136, bytes(range(256)) * 2), (648, b"\x07" * 5)]
    body = bytearray(653)
    for offset, piece in pieces:
        body[offset : offset + len(piece)] = piece

    def spans():
        # Like the CUDA saver, reuse a buffer as soon as the writer allows: every other span.
        buffers = [bytearray(512), bytearray(512)]
        for number, (offset, piece) in enumerate(pieces):
            buffers[number % 2][:] = b"\xff" * 512
            buffers[number % 2][: len(piece)] = piece
            yield offset, memoryview(buffers[number % 2])[: len(piece)]

    header = {"codec": "envelope-test", "chunks": [], "metadata": {"note": "one pass"}}
    expected = dict(
        header, bytes=len(body),
        sha256=[hashlib.sha256(body[first : first + 64]).hexdigest() for first in range(0, len(body), 64)],
    )
    blob = json.dumps(expected, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    path = tmp_path / "envelope.qem"
    if mode == "short-write":
        if not hasattr(snapshot.os, "pwrite"):
            pytest.skip("Positional write fault injection is unavailable on this platform.")
        write = snapshot.os.pwrite
        monkeypatch.setattr(
            snapshot.os, "pwrite", lambda fd, view, offset: write(fd, view[:7], offset)
        )
    elif mode == "fallback":
        monkeypatch.delattr(snapshot.os, "pwrite", raising=False)
    snapshot.write_envelope(path, header, len(body), spans() if mode == "generator" else pieces)
    assert path.read_bytes() == (
        metadata.MAGIC + struct.pack("<QQ", len(blob), 56 + len(blob))
        + hashlib.sha256(blob).digest() + blob + bytes(body)
    )
    assert not list(tmp_path.glob(".qem-*"))


def test_failed_resident_save_cleans_temporary_file_and_allows_retry(tmp_path, monkeypatch):
    """A failed destination write leaves no partial copy and can be retried."""
    if not hasattr(snapshot.os, "pwrite"):
        pytest.skip("Positional write fault injection is unavailable on this platform.")
    monkeypatch.setattr(snapshot, "BLOCK_BYTES", 64)
    header = {"codec": "envelope-test", "chunks": [], "metadata": {}}
    path = tmp_path / "retry.qem"
    body = bytes(range(256)) * 2

    def interrupted_write(*args):
        raise OSError("destination is full")

    with monkeypatch.context() as failure:
        failure.setattr(snapshot.os, "pwrite", interrupted_write)
        with pytest.raises(OSError, match="destination is full"):
            snapshot.write_envelope(path, header, len(body), [(0, body)])
    assert not path.exists()
    assert not list(tmp_path.glob(".qem-*"))
    snapshot.write_envelope(path, header, len(body), [(0, body)])
    assert path.read_bytes().endswith(body)
    assert not list(tmp_path.glob(".qem-*"))
