"""Portable conversion orchestration checks; no accelerator parity is implied."""

import hashlib
import os
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from quantem.gpu.io import convert as converter


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    shard = tmp_path / "detector.h5"
    with h5py.File(shard, "w") as handle:
        handle["entry/data/data"] = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(shard.name, "/entry/data/data")
    state = SimpleNamespace(master=master, shard=shard, destination=tmp_path / "saved.qem",
                            closed=False, saved=False, verified=False, hash_finished=False,
                            load_hook=lambda: None, save_metadata=None)

    class Resident:
        def __init__(self):
            self.metadata = {}

        resident_bytes = 64
        shape = (2, 2, 4, 4)

        def close(self):
            state.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    resident = Resident()
    state.resident = resident

    def load(*args, **kwargs):
        state.load_hook()
        return resident

    def save(path, acquisition, *, metadata, **kwargs):
        state.saved = True
        state.save_metadata = metadata
        path.write_bytes(b"portable orchestration fixture")

    def verify(*args, **kwargs):
        state.verified = True
        return {"identical": True, "compared_values": 64, "differing_values": 0, "flagged_pixels": 0}

    fingerprint = converter._source_files

    def hashes(paths, **kwargs):
        result = fingerprint(paths, **kwargs)
        state.hash_finished = True
        return result

    monkeypatch.setattr(converter, "load", load)
    monkeypatch.setattr(converter, "save", save)
    monkeypatch.setattr(converter, "verify_against_source", verify)
    monkeypatch.setattr(converter, "_source_files", hashes)
    return state


@pytest.mark.parametrize("verify,source_hashes", [(False, False), (True, False), (False, True), (True, True)])
def test_conversion_checks_are_independent_opt_ins(acquisition, monkeypatch, verify, source_hashes):
    state = acquisition

    def forbidden(*args, **kwargs):
        pytest.fail("Conversion performed a check that was not requested.")

    if not source_hashes:
        monkeypatch.setattr(converter, "_source_files", forbidden)
    if not verify:
        monkeypatch.setattr(converter, "verify_against_source", forbidden)
    options = {"verify": verify, "source_hashes": source_hashes} if verify or source_hashes else {}
    result = converter.convert(state.master, state.destination, backend="mps", **options)
    assert result.verified is (True if verify else None)
    assert state.verified is verify and state.hash_finished is source_hashes
    assert state.saved and state.closed and state.destination.is_file()
    for record, path in zip(state.save_metadata["source_files"], (state.master, state.shard), strict=True):
        assert record["name"] == path.name and record["bytes"] == path.stat().st_size
        assert ("sha256" in record) is source_hashes
    assert not list(state.master.parent.glob(".qem-convert-*"))


@pytest.mark.parametrize("flags,verify,source_hashes", [
    ([], False, False), (["--no-verify"], False, False),
    (["--verify"], True, False), (["--source-hashes"], False, True),
    (["--verify", "--source-hashes"], True, True),
])
def test_command_uses_the_same_optional_checks(acquisition, capsys, flags, verify, source_hashes):
    from quantem.gpu.cli import main

    state = acquisition
    assert main(["convert", str(state.master), "--backend", "mps", *flags]) == 0
    assert state.verified is verify and state.hash_finished is source_hashes
    printed = capsys.readouterr().out
    assert ("values identical" in printed) is verify
    assert ("full source comparison not requested" in printed) is (not verify)


def test_complete_source_hashes_overlap_load_and_precede_save(acquisition, monkeypatch):
    state = acquisition
    started = threading.Event()
    loaded = threading.Event()
    hashes = converter._source_files

    def hash_while_loading(paths, **kwargs):
        started.set()
        assert loaded.wait(5)
        return hashes(paths, **kwargs)

    def load_hook():
        assert started.wait(5)
        loaded.set()

    state.load_hook = load_hook
    monkeypatch.setattr(converter, "_source_files", hash_while_loading)
    result = converter.convert(state.master, state.destination, backend="mps", verify=True, source_hashes=True)
    assert result.verified and state.closed and state.saved and state.hash_finished
    assert state.save_metadata["source_files"] == [
        {"name": path.name, "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (state.master, state.shard)
    ]
    assert not list(state.master.parent.glob(".qem-convert-*"))


def test_estimate_does_not_schedule_provenance_reads(acquisition, monkeypatch):
    state = acquisition

    def forbidden(*args, **kwargs):
        pytest.fail("Estimate unexpectedly read provenance files.")

    monkeypatch.setattr(converter, "_source_files", forbidden)
    result = converter.convert(state.master, state.destination, write=False, backend="mps", verify=True, source_hashes=True)
    assert result.qem_bytes == 64 and state.closed
    assert not state.saved and not state.destination.exists()


def test_hash_failure_closes_the_resident_without_saving(acquisition, monkeypatch):
    state = acquisition

    def unreadable(*args, **kwargs):
        raise OSError("source fingerprint read failed")

    monkeypatch.setattr(converter, "_source_files", unreadable)
    with pytest.raises(OSError, match="source fingerprint read failed"):
        converter.convert(state.master, state.destination, backend="mps", verify=True, source_hashes=True)
    assert state.closed and not state.saved and not state.destination.exists()


@pytest.mark.parametrize("outcome", ["load-failure", "larger-copy"])
def test_abandoned_conversion_cancels_and_drains_hashing(acquisition, monkeypatch, outcome):
    state = acquisition
    started = threading.Event()
    finished = threading.Event()

    def hash_until_abandoned(paths, *, stop=None, **kwargs):
        assert stop is not None
        started.set()
        try:
            assert stop.wait(5), "Abandoned conversion did not cancel provenance hashing."
            from concurrent.futures import CancelledError
            raise CancelledError
        finally:
            finished.set()

    def load_hook():
        assert started.wait(5)
        if outcome == "load-failure":
            raise OSError("load failed")
        state.resident.resident_bytes = 1 << 30

    state.load_hook = load_hook
    monkeypatch.setattr(converter, "_source_files", hash_until_abandoned)
    if outcome == "load-failure":
        with pytest.raises(OSError, match="load failed"):
            converter.convert(state.master, state.destination, backend="mps", verify=True, source_hashes=True)
    else:
        result = converter.convert(state.master, state.destination, backend="mps", verify=True, source_hashes=True)
        assert result.larger and state.closed
    assert finished.is_set() and not state.saved and not state.destination.exists()


def test_changed_source_still_prevents_publication(acquisition, monkeypatch):
    state = acquisition

    def verify_after_change(*args, **kwargs):
        stat = state.shard.stat()
        os.utime(state.shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return {"identical": True, "compared_values": 64, "differing_values": 0, "flagged_pixels": 0}

    monkeypatch.setattr(converter, "verify_against_source", verify_after_change)
    result = converter.convert(state.master, state.destination, backend="mps", verify=True, source_hashes=True)
    assert result.failed and "Source files changed" in result.skipped
    assert state.closed and state.saved and not state.destination.exists()
    assert not list(state.master.parent.glob(".qem-convert-*"))


def test_source_hash_cancellation_stops_before_another_block(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 4)
    stop = threading.Event()
    real_open = open
    reads = []

    class ControlledRead:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def readinto(self, buffer):
            count = self.handle.readinto(buffer)
            reads.append(count)
            stop.set()
            return count

    monkeypatch.setattr(converter, "_HASH_BLOCK", 64)
    monkeypatch.setattr(converter, "open", ControlledRead, raising=False)
    from concurrent.futures import CancelledError
    with pytest.raises(CancelledError):
        converter._source_files([source], stop=stop)
    assert reads == [64]
