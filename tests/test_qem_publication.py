"""Saved acquisitions remain exact on ordinary and removable filesystems."""

import ctypes
import errno
import sys
from types import SimpleNamespace

import pytest

from quantem.gpu.io import _publish


@pytest.mark.parametrize("removable", [False, True])
def test_publish_complete_acquisition_without_overwriting(tmp_path, monkeypatch, removable):
    source = tmp_path / "staged"
    destination = tmp_path / "acquisition.qem"
    contents = bytes(range(256)) * 1000
    source.write_bytes(contents)
    if removable:
        def unsupported_link(*args):
            raise OSError(errno.ENOTSUP, "filesystem has no hard links")

        def exclusive_rename(source, target, flags):
            assert flags == 4
            if destination.exists():
                ctypes.set_errno(errno.EEXIST)
                return -1
            source_path = _publish.os.fsdecode(source)
            destination_path = _publish.os.fsdecode(target)
            original_rename(source_path, destination_path)
            return 0

        original_rename = _publish.os.rename
        monkeypatch.setattr(_publish.sys, "platform", "darwin")
        monkeypatch.setattr(_publish.os, "link", unsupported_link)
        monkeypatch.setattr(
            _publish.ctypes, "CDLL",
            lambda *args, **kwargs: SimpleNamespace(renamex_np=exclusive_rename),
        )
    _publish.publish_file(source, destination)
    assert destination.read_bytes() == contents
    source.unlink(missing_ok=True)
    source.write_bytes(b"a different acquisition")
    with pytest.raises(FileExistsError):
        _publish.publish_file(source, destination)
    assert destination.read_bytes() == contents


def test_unsupported_volume_does_not_expose_partial_acquisition(tmp_path, monkeypatch):
    source = tmp_path / "staged"
    destination = tmp_path / "acquisition.qem"
    source.write_bytes(bytes(range(256)) * 1000)

    def unsupported_link(*args):
        raise OSError(errno.ENOTSUP, "filesystem has no hard links")

    def unsupported_rename(*args):
        ctypes.set_errno(errno.ENOTSUP)
        return -1

    monkeypatch.setattr(_publish.sys, "platform", "darwin")
    monkeypatch.setattr(_publish.os, "link", unsupported_link)
    monkeypatch.setattr(
        _publish.ctypes, "CDLL",
        lambda *args, **kwargs: SimpleNamespace(renamex_np=unsupported_rename),
    )
    with pytest.raises(OSError, match="without exposing partial data"):
        _publish.publish_file(source, destination)
    assert source.exists()
    assert not destination.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS exclusive rename")
def test_native_exclusive_rename_publishes_complete_file(tmp_path, monkeypatch):
    source = tmp_path / "staged"
    destination = tmp_path / "acquisition.qem"
    contents = bytes(range(256)) * 1000
    source.write_bytes(contents)

    def unsupported_link(*args):
        raise OSError(errno.ENOTSUP, "filesystem has no hard links")

    monkeypatch.setattr(_publish.os, "link", unsupported_link)
    _publish.publish_file(source, destination)
    assert not source.exists()
    assert destination.read_bytes() == contents

    source.write_bytes(b"a different acquisition")
    with pytest.raises(FileExistsError):
        _publish.publish_file(source, destination)
    assert destination.read_bytes() == contents
