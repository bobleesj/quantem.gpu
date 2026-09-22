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
