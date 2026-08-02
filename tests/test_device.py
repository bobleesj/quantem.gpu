from __future__ import annotations

import pytest

from quantem.gpu import device
from quantem.gpu.device import backend


def test_detect_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_cuda_probe", lambda: (True, None))
    monkeypatch.setattr(backend, "_mps_probe", lambda: (True, None))

    assert device.detect() == "cuda"


def test_resolve_keeps_explicit_webgpu() -> None:
    assert device.resolve("webgpu") == "webgpu"


def test_profile_is_non_printing_and_reports_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "_cuda_probe", lambda: (False, "missing CUDA"))
    monkeypatch.setattr(backend, "_mps_probe", lambda: (False, "missing MPS"))

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
    monkeypatch.setattr(backend, "_cuda_probe", lambda: (False, "missing CUDA"))
    monkeypatch.setattr(backend, "_mps_probe", lambda: (False, "missing MPS"))

    with pytest.raises(RuntimeError, match="No QuantEM GPU backend"):
        device.detect()
    with pytest.raises(ValueError, match="Unknown GPU backend"):
        device.resolve("cpu")
