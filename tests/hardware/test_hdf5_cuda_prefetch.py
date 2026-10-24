"""Bounded compressed read-ahead must recover from reader and encoder failures."""

import collections
import os
import threading

import h5py
import numpy as np
import pytest


@pytest.mark.parametrize("failure", ["read", "encode"])
def test_prefetch_failure_releases_all_staging_and_allows_retry(
    tmp_path, monkeypatch, failure
):
    if os.environ.get("QEM_TEST_BACKEND") != "cuda":
        pytest.skip("Set QEM_TEST_BACKEND=cuda on a physical CUDA device.")
    import hdf5plugin

    from quantem.gpu import io
    from quantem.gpu.formats.hdf5 import reads
    from quantem.gpu.io import encoded
    from quantem.gpu.io.hdf5.cuda import decode

    values = np.random.default_rng(71).integers(0, 100, (4096, 16, 16), dtype=np.uint16)
    chunk = tmp_path / "detector.h5"
    with h5py.File(chunk, "w") as handle:
        data = handle.create_dataset(
            "entry/data/data", data=values, chunks=(1, 16, 16),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        damaged_offset = data.id.get_chunk_info_by_coord((512, 0, 0)).byte_offset + 12
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(chunk.name, "/entry/data/data")
        root = "entry/instrument/detector/"
        handle[root + "description"] = b"Dectris ARINA Si"
        handle[root + "detectorSpecific/ntrigger"] = np.uint32(4096)
    monkeypatch.setattr(encoded, "_CUDA_MAX_STAGING_SCANS", 512)

    allocated, released = collections.Counter(), collections.Counter()
    lock = threading.Lock()
    original_allocate = reads._alloc_pinned_fast
    original_release = reads._release_pinned
    original_close = reads.FrameReader.close
    original_prepare = reads.FrameReader.prepare
    original_append = encoded.StreamedCounts.append
    active = [0]
    closed = []

    def allocate(size):
        buffer = original_allocate(size)
        with lock:
            allocated[buffer.ctypes.data] += 1
        return buffer

    def release(buffer):
        with lock:
            released[buffer.ctypes.data] += 1
        return original_release(buffer)

    def prepare(reader, indices):
        with lock:
            active[0] += 1
        try:
            return original_prepare(reader, indices)
        finally:
            with lock:
                active[0] -= 1

    def close(reader):
        assert active[0] == 0
        original_close(reader)
        closed.append(reader.closed)

    def fail_append(*_args):
        raise RuntimeError("injected encoder failure")

    monkeypatch.setattr(reads, "_alloc_pinned_fast", allocate)
    for module in (reads, decode, encoded):
        monkeypatch.setattr(module, "_release_pinned", release)
    monkeypatch.setattr(reads.FrameReader, "prepare", prepare)
    monkeypatch.setattr(reads.FrameReader, "close", close)
    if failure == "read":
        with chunk.open("r+b") as handle:
            handle.seek(damaged_offset)
            original = handle.read(4)
            handle.seek(damaged_offset)
            handle.write(b"\xff" * 4)
    else:
        monkeypatch.setattr(encoded.StreamedCounts, "append", fail_append)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            io.load(master, backend="cuda", hot_pixel_correction="none", verbose=False)
    finally:
        if failure == "read":
            with chunk.open("r+b") as handle:
                handle.seek(damaged_offset)
                handle.write(original)
        else:
            monkeypatch.setattr(encoded.StreamedCounts, "append", original_append)
    assert closed == [True]
    assert active == [0]
    assert allocated == released

    with io.load(master, backend="cuda", hot_pixel_correction="none", verbose=False) as loaded:
        actual = loaded.data.decode_scan_range_device(0, 4096).get()
        np.testing.assert_array_equal(actual, values)
    assert allocated == released
    assert closed == [True, True]
