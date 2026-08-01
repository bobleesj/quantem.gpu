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


def test_profile_is_non_printing_and_reports_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_cuda_probe", lambda: (False, "missing CUDA"))
    monkeypatch.setattr(backend, "_mps_probe", lambda: (False, "missing MPS"))

    result = device.profile()

    assert result["backend"] in {"cuda", "mps", "cpu"}
    assert result["device"] in {"cuda:0", "mps", "cpu"}
    assert result["host"]
    assert result["python"]


def test_no_cpu_scientific_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_cuda_probe", lambda: (False, "missing CUDA"))
    monkeypatch.setattr(backend, "_mps_probe", lambda: (False, "missing MPS"))

    with pytest.raises(RuntimeError, match="No QuantEM GPU backend"):
        device.detect()
    with pytest.raises(ValueError, match="Unknown GPU backend"):
        device.resolve("cpu")
