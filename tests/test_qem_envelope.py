"""A ``.qem`` file written in one pass is its definition, byte for byte."""

import hashlib
import json
import struct

from quantem.gpu.formats.qem import metadata, snapshot


def test_envelope_is_header_then_body_with_zero_gaps_and_block_checksums(tmp_path, monkeypatch):
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
    snapshot.write_envelope(path, header, len(body), spans())
    assert path.read_bytes() == (
        metadata.MAGIC + struct.pack("<QQ", len(blob), 56 + len(blob))
        + hashlib.sha256(blob).digest() + blob + bytes(body)
    )
    assert not list(tmp_path.glob(".qem-*"))
