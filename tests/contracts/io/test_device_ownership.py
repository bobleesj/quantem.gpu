"""Complete series and device ownership, using CPU arrays and fake contexts."""
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
    runtime = import_module("quantem.gpu.device.cuda_runtime")
    decoder = import_module("quantem.gpu.io.hdf5.cuda.decode")
    monkeypatch.setitem(sys.modules, "cupy", None)
    contexts = _Contexts()
    fake_cupy = contexts.cupy()
    monkeypatch.setattr(runtime, "cp", fake_cupy)
    monkeypatch.setattr(runtime, "_MODULES", {})
    monkeypatch.setattr(decoder, "cp", fake_cupy)
    monkeypatch.setattr(decoder, "_FAILED_DECOMPRESSIONS", [])
    monkeypatch.setattr(
        "quantem.gpu.device.select.resolve_backend", lambda _backend: "cuda"
    )
    return decoder, contexts


def test_imported_decoder_callable_follows_alternating_devices(io_contexts):
    """One imported callable remains correct in two devices and a new context."""
    decoder, contexts = io_contexts

    def kernel(values):
        return decoder.kernel("shuf_8192_16_batched")(values)

    values = np.array([0, 255, 65535], dtype=np.uint16)
    for device, context in [(0, 100), (1, 101), (0, 100), (0, 200)]:
        contexts.select(device, context)
        assert kernel(values) == 65790

    start = threading.Barrier(2)

    def run(device):
        contexts.select(device)
        start.wait(timeout=2)
        return [kernel(values) for _ in range(8)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, [0, 1])) == [[65790] * 8] * 2
    assert sorted(contexts.compiled) == [(0, 100), (0, 200), (1, 101)]


@pytest.mark.parametrize("unfinished", [False, True])
def test_decode_failure_keeps_host_lease_until_fence(io_contexts, monkeypatch, unfinished):
    """Failed work cannot make a still-borrowed pinned buffer available again."""
    decoder, _contexts = io_contexts
    prepared = {"read_buffer": np.zeros(32, dtype=np.uint8)}
    actions = []

    def decode(*_args, **_kwargs):
        actions.append("decode")
        raise RuntimeError("decode failed")

    def fence():
        actions.append("fence")
        if unfinished:
            raise RuntimeError("device completion unknown")

    monkeypatch.setattr(decoder, "_decode_prepared", decode)
    monkeypatch.setattr(decoder.cp.cuda, "Device", lambda: SimpleNamespace(synchronize=fence))
    monkeypatch.setattr(decoder, "_release_pinned", lambda _buffer: actions.append("release"))
    with pytest.raises(RuntimeError, match="decode failed"):
        decoder.decompress_prepared(prepared)
    assert actions == (["decode", "fence"] if unfinished else ["decode", "fence", "release"])
    if unfinished:
        assert decoder._FAILED_DECOMPRESSIONS[0][0] is prepared
