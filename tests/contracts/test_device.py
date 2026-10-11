import sys
from importlib import import_module
from types import SimpleNamespace

import pytest

from quantem.gpu import device
from quantem.gpu.device import select


def test_detect_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(select, "_cuda_probe", lambda: (True, None))
    monkeypatch.setattr(select, "_mps_probe", lambda: (True, None))

    assert device.detect() == "cuda"


def test_profile_is_non_printing_and_reports_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(select, "_cuda_probe", lambda: (False, "missing CUDA"))
    monkeypatch.setattr(select, "_mps_probe", lambda: (False, "missing MPS"))

    result = device.profile()

    assert set(result) == {
        "platform",
        "torch",
        "backend",
        "device",
        "available_devices",
    }
    assert result["platform"]
    assert result["torch"]
    assert result["backend"] in {"cuda", "mps", "cpu"}
    assert result["device"] in {"cuda:0", "mps", "cpu"}
    assert result["device"] in result["available_devices"]


def test_profile_selects_explicit_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    result = device.profile(device="cuda:1")

    assert result["backend"] == "cuda"
    assert result["device"] == "cuda:1"
    assert result["available_devices"] == ["cuda:0", "cuda:1"]


def test_profile_rejects_cuda_index_outside_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match=r"2 CUDA device\(s\) are visible"):
        device.profile(device="cuda:2")


def test_no_cpu_scientific_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(select, "_cuda_probe", lambda: (False, "missing CUDA"))
    monkeypatch.setattr(select, "_mps_probe", lambda: (False, "missing MPS"))

    with pytest.raises(RuntimeError, match="No QuantEM GPU backend"):
        device.detect()
    with pytest.raises(ValueError, match="Unknown GPU backend"):
        device.resolve("cpu")


def test_cupy_lookups_follow_the_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module installed as ``cupy`` serves CuPy lookups only while it is installed."""
    from quantem.gpu.device import cuda_runtime

    if cuda_runtime.cp is None:
        pytest.skip("CuPy is not installed.")
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "cupy", SimpleNamespace(ndarray="stand-in"))
        assert cuda_runtime.cp.ndarray == "stand-in"
    assert cuda_runtime.cp.ndarray is import_module("cupy").ndarray


def test_release_cached_memory_empties_every_opened_gpu_and_returns_what_the_allocators_gave_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Three simulated GPUs, of which the process has opened 0 and 2: GPU 1 must not be visited.

    Garbage is collected, then Torch's cache is emptied (8 GB on GPU 0) and CuPy's pool on each opened GPU:
    30 GB on GPU 0, and 2 GB on GPU 2 only when the cached FFT plan there is dropped first, because its 0.5 GB
    work area sits in the pool beside 1 GB of live arrays. The result is the sum, CuPy's current GPU is
    restored, and ``verbose`` prints one line per opened GPU with the driver's totals, which other processes
    change freely without changing the result.
    """
    import torch

    from quantem.gpu.device import cuda_runtime

    total_bytes = 96_000_000_000
    driver_free_bytes = {0: 48_000_000_000, 2: 20_000_000_000}
    torch_reserved_bytes = {0: 9_000_000_000, 2: 0}
    pool_bytes = {0: 30_000_000_000, 2: 3_000_000_000}
    pool_live_bytes = {0: 0, 2: 1_500_000_000}
    current_gpu = [0]
    calls = []

    class Device:
        def __init__(self, index: int) -> None:
            assert index != 1, "visiting GPU 1 would open it"
            self.index = index

        def __enter__(self) -> None:
            self.previous, current_gpu[0] = current_gpu[0], self.index

        def __exit__(self, *exception) -> None:
            current_gpu[0] = self.previous

    def clear_fft_plans() -> None:
        calls.append(f"cupy fft plans {current_gpu[0]}")
        pool_live_bytes[2] = 1_000_000_000

    def free_pool() -> None:
        calls.append(f"cupy pool {current_gpu[0]}")
        pool_bytes[current_gpu[0]] = pool_live_bytes[current_gpu[0]]

    def empty_torch_cache() -> None:
        calls.append("torch cache")
        torch_reserved_bytes[0] = 1_000_000_000

    pool = SimpleNamespace(free_all_blocks=free_pool, total_bytes=lambda: pool_bytes[current_gpu[0]])
    plans = SimpleNamespace(clear=clear_fft_plans, get_curr_size=lambda: 1 if current_gpu[0] == 2 else 0)
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(Device=Device),
        fft=SimpleNamespace(config=SimpleNamespace(get_plan_cache=lambda: plans)),
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: calls.append("cupy pinned pool")),
    )
    monkeypatch.setattr(cuda_runtime, "cp", fake_cupy)
    monkeypatch.setattr(select.gc, "collect", lambda: calls.append("garbage collection"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    # absent from a Torch built without CUDA, which the hosted test machines install
    monkeypatch.setattr(torch._C, "_cuda_hasPrimaryContext", lambda index: index != 1, raising=False)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: torch_reserved_bytes[index])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (driver_free_bytes[index], total_bytes))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_torch_cache)

    assert device.release_cached_memory(verbose=True) == 40.0
    assert calls == ["garbage collection", "torch cache", "cupy pool 0", "cupy fft plans 2", "cupy pool 2", "cupy pinned pool"]
    assert current_gpu == [0]
    assert capsys.readouterr().out == (
        "GPU 0: released 38.0 GB, 48.0 GB in use, 48.0 GB free\n"
        "GPU 2: released 2.0 GB, 76.0 GB in use, 20.0 GB free\n"
    )


def test_release_cached_memory_on_apple_silicon_returns_the_metal_driver_drop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Simulated Apple GPU holding 15 GB: every step's share is in the result, read from the Metal driver.

    Garbage collection hands a 10 GB Metal resident of a reference cycle straight back to the driver, emptying
    Torch's cache returns 2 GB and MLX's 1 GB, so the driver is read before the collection and after the last
    cache. Torch's tensor counter stays at 1 GB throughout: a result computed from it would be 0.0.
    """
    import torch

    driver_bytes = [15_000_000_000]
    calls = []

    def step(name: str, freed_bytes: int) -> None:
        calls.append(name)
        driver_bytes[0] -= freed_bytes

    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(clear_cache=lambda: step("mlx cache", 1_000_000_000)))
    monkeypatch.setattr(select.gc, "collect", lambda: step("garbage collection", 10_000_000_000))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: step("torch cache", 2_000_000_000))
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: driver_bytes[0])
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 1_000_000_000)

    assert device.release_cached_memory(verbose=True) == 13.0
    assert calls == ["garbage collection", "torch cache", "mlx cache"]
    assert capsys.readouterr().out == "MPS: released 13.0 GB, 2.0 GB in use\n"
