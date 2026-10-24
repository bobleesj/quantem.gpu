"""Exercise per-file cache instrumentation without changing scientific inputs."""

import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_file_cache as cache


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="POSIX cache controls")
def test_cache_probe_and_eviction_preserve_file(tmp_path: Path) -> None:
    """A real eviction request leaves bytes and file metadata unchanged."""
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    content = bytes(range(256)) * (3 * page_bytes // 256) + b"tail"
    path = tmp_path / "disposable.bin"
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    before = path.stat()

    warm = cache.cache_state(path)
    control = cache.evict_file_cache(path)
    after = cache.cache_state(path)

    for state in (warm, after):
        assert state["available"] is True
        assert state["bytes"] == len(content)
        assert state["page_bytes"] == page_bytes
        assert state["pages"] == 4
        assert 0 <= state["resident_pages"] <= 4
        assert state["resident_percent"] == 25 * state["resident_pages"]
    # Eviction is advisory; a request alone cannot establish a cold trial.
    assert control
    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="POSIX cache controls")
def test_empty_file_probe_has_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.touch()
    with pytest.raises(ValueError, match="nonempty file"):
        cache.cache_state(path)


def test_unsupported_cache_platform_is_explicit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache.sys, "platform", "unsupported")
    assert cache.cache_state(tmp_path / "absent")["available"] is False
    with pytest.raises(NotImplementedError, match="Linux or macOS"):
        cache.evict_file_cache(tmp_path / "absent")
