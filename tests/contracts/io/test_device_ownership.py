"""Complete series and device ownership, using CPU arrays and fake contexts."""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest


class _Contexts:
    """Model thread-local context selection without importing a CUDA runtime."""

    def __init__(self):
        self.local = threading.local()
        self.compiled = []
        self.launches = []
        self.fences = []
        self.lock = threading.Lock()

    def key(self):
        return getattr(self.local, "key", (0, 100))

    def select(self, device, context=None):
        self.local.key = (device, 100 + device if context is None else context)

    def device(self, device=None):
        owner = self

        class Device:
            def __enter__(self):
                self.previous = owner.key()
                if device is not None and device != self.previous[0]:
                    owner.select(device)
                return self

            def __exit__(self, *_args):
                owner.local.key = self.previous

            def synchronize(self):
                owner.fences.append(owner.key())

        return Device()

    def raw_module(self, **_kwargs):
        key = self.key()
        with self.lock:
            self.compiled.append(key)

        def get_function(name):
            def launch(values):
                assert self.key() == key, "function reused in another context"
                with self.lock:
                    self.launches.append((key, name))
                return np.sum(values, dtype=np.uint64)

            return launch

        return SimpleNamespace(get_function=get_function)

    def cupy(self):
        return SimpleNamespace(
            cuda=SimpleNamespace(
                Device=self.device,
                runtime=SimpleNamespace(
                    getDevice=lambda: self.key()[0],
                    free=lambda _pointer: self.select(self.key()[0]),
                    getDeviceCount=lambda: 2,
                ),
                driver=SimpleNamespace(ctxGetCurrent=lambda: self.key()[1]),
                get_current_stream=lambda: self.device(),
                alloc_pinned_memory=bytearray,
            ),
            RawModule=self.raw_module,
            empty=np.empty,
            uint8=np.uint8,
            get_default_memory_pool=lambda: SimpleNamespace(
                free_all_blocks=lambda: None
            ),
        )


@pytest.fixture
def io_contexts(monkeypatch):
    # Import the package without initializing a device, then replace every
    # runtime operation with this CPU fixture, even on a GPU host.
    loader = import_module("quantem.gpu.io.load")
    decoder = import_module("quantem.gpu.io.backends.cuda.decoder")
    monkeypatch.setitem(sys.modules, "cupy", None)
    contexts = _Contexts()
    fake_cupy = contexts.cupy()
    monkeypatch.setattr(loader, "cp", fake_cupy)
    monkeypatch.setattr(decoder, "cp", fake_cupy)
    monkeypatch.setattr(decoder, "_cuda_modules", {})
    monkeypatch.setattr(decoder, "_cuda_functions", {})
    monkeypatch.setattr(loader, "_FAILED_DECOMPRESSIONS", [])
    monkeypatch.setattr(
        "quantem.gpu.io.backends.resolve_backend", lambda _backend: "cuda"
    )
    return loader, decoder, contexts


def test_imported_decoder_callable_follows_alternating_devices(io_contexts):
    """One imported callable remains correct in two devices and a new context."""
    loader, decoder, contexts = io_contexts
    kernel = decoder.bitshuffle_kernel_u16
    loader_kernel = loader._lazy_kernel("bitshuffle_kernel_u16")
    values = np.array([0, 255, 65535], dtype=np.uint16)
    for device, context in [(0, 100), (1, 101), (0, 100), (0, 200)]:
        contexts.select(device, context)
        assert kernel(values) == 65790
        assert loader_kernel(values) == 65790

    start = threading.Barrier(2)

    def run(device):
        contexts.select(device)
        start.wait(timeout=2)
        return [loader_kernel(values) for _ in range(8)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, [0, 1])) == [[65790] * 8] * 2
    assert sorted(contexts.compiled) == [(0, 100), (0, 200), (1, 101)]


def test_decompressors_keep_independent_device_scratch(io_contexts, monkeypatch):
    """A decompressor can be reused without another device replacing its buffers."""
    loader, _decoder, contexts = io_contexts
    owners = []
    for device in (0, 1):
        contexts.select(device)
        owners.append(loader.GPUDecompressor(32, 2, 8, 1))
    assert owners[0]._concat_gpu is not owners[1]._concat_gpu
    barrier = threading.Barrier(2)

    def decode(owner, path, dataset_path):
        assert contexts.key() == owner._context_key
        barrier.wait(timeout=2)
        owner._shuffled_output[:] = int(path)
        return owner._shuffled_output.copy()

    monkeypatch.setattr(loader.GPUDecompressor, "_load", decode)

    def run(device):
        contexts.select(1 - device)
        result = owners[device].load(str(device + 3))
        assert contexts.key()[0] == 1 - device
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [0, 1]))
    np.testing.assert_array_equal(results[0], np.full(16, 3, dtype=np.uint8))
    np.testing.assert_array_equal(results[1], np.full(16, 4, dtype=np.uint8))
    contexts.select(0, 999)
    with pytest.raises(RuntimeError, match="different CUDA context"):
        owners[0].load("7")


def _series(monkeypatch, loader, *, shapes=None, failure=None):
    paths = [f"tilt_{index}_master.h5" for index in range(6)]
    values = {
        path: np.full(
            (shapes or {}).get(index, (2, 2, 2, 2)),
            65535 - index,
            dtype=np.uint16,
        )
        for index, path in enumerate(paths)
    }
    released = []
    prepared_buffers = []
    original_load = loader.load

    def prepare(path, *_args):
        if path == failure:
            raise OSError("missing detector shard")
        buffer = np.empty(1, dtype=np.uint8)
        prepared_buffers.append(buffer)
        return {"read_buffer": buffer, "path": path}

    def decode(prepared, **_kwargs):
        released.append(prepared["read_buffer"])
        return values[prepared["path"]].copy()

    def load(source, **kwargs):
        if isinstance(source, list):
            return original_load(source, **kwargs)
        if source == failure:
            raise OSError("missing detector shard")
        return loader.LoadResult(values[source].copy(), {"scan_shape": (2, 2)})

    monkeypatch.setattr(loader, "load", load)
    monkeypatch.setattr(loader, "_discover_chunk_names", lambda _path: ["data"])
    monkeypatch.setattr(loader, "_prepare_master", prepare)
    monkeypatch.setattr(loader, "_decompress_prepared", decode)
    monkeypatch.setattr(loader, "_release_pinned", released.append)
    monkeypatch.setattr(loader, "get_metadata", lambda _path: {"scan_shape": (2, 2)})
    monkeypatch.setattr(loader, "_apply_scan_shape", lambda data, *_args: data)
    monkeypatch.setattr(loader, "_disk_interleaved_indices", lambda paths: list(range(len(paths))))
    return paths, values, prepared_buffers, released


@pytest.mark.parametrize("devices", [None, [0, 1]])
def test_load_returns_every_acquisition_in_source_order(io_contexts, monkeypatch, devices):
    """Complete unmasked uint16 series retains counts on one or two devices."""
    loader, _decoder, _contexts = io_contexts
    paths, values, prepared, released = _series(monkeypatch, loader)
    result = loader.load(
        paths, backend="cuda", dtype="u16", apply_mask=False,
        devices=devices, verbose=False,
    )
    if devices is None:
        np.testing.assert_array_equal(result.data, np.stack(list(values.values())))
        assert result.metadata["n_files"] == len(paths)
    else:
        assert set(result.metadata["device_map"]) == set(range(len(paths)))
        for device, indices in result.metadata["shard_order"].items():
            np.testing.assert_array_equal(
                result.data[device], np.stack([values[paths[i]] for i in indices])
            )
    assert sorted(map(id, prepared)) == sorted(map(id, released))


@pytest.mark.parametrize("devices", [None, [0, 1]])
@pytest.mark.parametrize("problem", ["missing", "shape", "cross_device_shape"])
def test_incomplete_series_never_returns_partial_data(io_contexts, monkeypatch, devices, problem):
    """A bad required tilt prevents a misleading smaller successful stack."""
    loader, _decoder, _contexts = io_contexts
    shapes = {2: (1, 2, 2, 2)} if problem == "shape" else None
    if problem == "cross_device_shape":
        shapes = {i: (1, 2, 2, 2) for i in (1, 3, 5)}
    failure = "tilt_2_master.h5" if problem == "missing" else None
    paths, _values, prepared, released = _series(
        monkeypatch, loader, shapes=shapes, failure=failure
    )
    with pytest.raises((OSError, ValueError), match="tilt_[12345]_master.h5"):
        loader.load(paths, backend="cuda", dtype="u16", devices=devices, verbose=False)
    assert sorted(map(id, prepared)) == sorted(map(id, released))


def test_failed_first_decode_drains_bounded_read_ahead(io_contexts, monkeypatch):
    """A failed load finishes its reader and releases every unused host buffer."""
    loader, _decoder, _contexts = io_contexts
    paths, _values, prepared, released = _series(monkeypatch, loader)
    original_prepare = loader._prepare_master
    ahead = threading.Event()

    def prepare(*args):
        result = original_prepare(*args)
        if len(prepared) == 4:
            ahead.set()
        return result

    def decode(current, **_kwargs):
        assert ahead.wait(timeout=2)
        released.append(current["read_buffer"])
        raise RuntimeError("decode failed")

    monkeypatch.setattr(loader, "_prepare_master", prepare)
    monkeypatch.setattr(loader, "_decompress_prepared", decode)
    with pytest.raises(RuntimeError, match="decode failed"):
        loader.load(paths, backend="cuda", dtype="u16", verbose=False)
    assert len(prepared) == 4
    assert sorted(map(id, prepared)) == sorted(map(id, released))


@pytest.mark.parametrize("unfinished", [False, True])
def test_decode_failure_keeps_host_lease_until_fence(io_contexts, monkeypatch, unfinished):
    """Failed work cannot make a still-borrowed pinned buffer available again."""
    loader, _decoder, _contexts = io_contexts
    prepared = {"read_buffer": np.zeros(32, dtype=np.uint8)}
    actions = []

    def decode(*_args, **_kwargs):
        actions.append("decode")
        raise RuntimeError("decode failed")

    def fence():
        actions.append("fence")
        if unfinished:
            raise RuntimeError("device completion unknown")

    monkeypatch.setattr(loader, "_decompress_prepared_impl", decode)
    monkeypatch.setattr(loader.cp.cuda, "Device", lambda: SimpleNamespace(synchronize=fence))
    monkeypatch.setattr(loader, "_release_pinned", lambda _buffer: actions.append("release"))
    with pytest.raises(RuntimeError, match="decode failed"):
        loader._decompress_prepared(prepared)
    assert actions == (["decode", "fence"] if unfinished else ["decode", "fence", "release"])
    if unfinished:
        assert loader._FAILED_DECOMPRESSIONS[0][0] is prepared


def test_ans_load_and_conversion_stay_on_selected_device(io_contexts, monkeypatch, tmp_path):
    """Load an encoded acquisition on another device and retain that ownership."""
    loader, _, contexts = io_contexts
    from quantem.gpu.io._ans import write_ans_reference
    ans_backend = import_module("quantem.gpu.io.backends.cuda._ans")
    monkeypatch.setitem(sys.modules, "cupy", contexts.cupy())
    calls = []

    class Counts:
        nbytes = 64

        def __init__(self, **_encoded):
            self.owner = contexts.key()
            calls.append(("load", self.owner))

        def to_packed(self):
            assert contexts.key() == self.owner
            calls.append(("pack", self.owner))
            return SimpleNamespace(nbytes=32, release=self.release)

        def release(self):
            assert contexts.key() == self.owner
            calls.append(("release", self.owner))

    monkeypatch.setattr(ans_backend, "CudaANSResidentCounts", Counts)
    path = write_ans_reference(tmp_path / "counts.ans", np.zeros((2, 3, 4, 5), np.uint16))
    contexts.select(0)
    loaded = loader.load(path, backend="cuda", representation="packed", device=1)
    assert contexts.key() == (0, 100)
    assert loaded.representation.value == "packed"
    assert calls == [("load", (1, 101)), ("pack", (1, 101)), ("release", (1, 101))]
    with contexts.device(1):
        loaded.close()
    assert calls[-1] == ("release", (1, 101))
