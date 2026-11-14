"""Detector files are read as raw bytes on every OS (Windows opens descriptors in text mode by default)."""

import os

from quantem.gpu.formats.hdf5.reads import _pread_into, open_for_reading


def test_bytes_text_mode_would_spoil_come_back_whole(tmp_path):
    # 0x1A ends a text-mode read on Windows, and CR LF becomes LF
    payload = bytes(range(256)) * 64 + bytes([0x1A, 0x0D, 0x0A, 0x1A]) * 1000
    path = tmp_path / "counts.bin"
    path.write_bytes(payload)
    fd = open_for_reading(path)
    try:
        view = memoryview(bytearray(len(payload)))
        got = 0
        while got < len(payload):
            read = _pread_into(fd, view[got:], got)
            assert read > 0, f"short read at byte {got}"
            got += read
        assert bytes(view) == payload
    finally:
        os.close(fd)
