"""Resource-lifetime tests for the public MPS SSB backend."""
from __future__ import annotations

from types import SimpleNamespace


def test_close_releases_backend_state_and_mlx_cache(monkeypatch) -> None:
    """Closing an MPS workflow must return allocator-owned unified memory."""

    from quantem.gpu.ssb.compute.mps import backend as backend_module

    calls: list[str] = []
    monkeypatch.setattr(
        backend_module,
        "_require_mlx",
        lambda: SimpleNamespace(clear_cache=lambda: calls.append("clear")),
    )
    monkeypatch.setattr(
        backend_module,
        "_clear_mps_io_cache",
        lambda: calls.append("io-clear"),
    )
    backend = backend_module.MpsSSBBackend.__new__(backend_module.MpsSSBBackend)
    for name in (
        "_prepared",
        "_frames",
        "_mean_phase_buffer",
        "_sumsq_buffer",
        "_fit_preview_phase",
        "_fit_preview_loss",
        "_fit_preview_aberrations",
    ):
        setattr(backend, name, object())

    class _Source:
        def free(self) -> None:
            calls.append("free")

    backend._source_data = _Source()

    backend.close()

    assert calls == ["free", "io-clear", "clear"]
    assert backend._prepared is None
    assert backend._frames is None
    assert backend._mean_phase_buffer is None
    assert backend._sumsq_buffer is None
    assert backend._fit_preview_phase is None
    assert backend._fit_preview_loss is None
    assert backend._fit_preview_aberrations is None
    assert backend._source_data is None


def test_workflow_drops_shared_source_before_mps_allocator_flush() -> None:
    """The MPS close hook must run only after the workflow releases data."""

    from quantem.gpu.ssb.workflow import SSB

    workflow = SSB.__new__(SSB)
    workflow._cuda_session = None
    calls: list[str] = []

    class _Source:
        def free(self) -> None:
            calls.append("free")

    source = _Source()
    workflow._data = source
    observations: list[object | None] = []

    class _Backend:
        def close(self) -> None:
            observations.append(workflow._data)
            source.free()

    workflow._mps_backend = _Backend()

    workflow.close()

    assert observations == [None]
    assert calls == ["free"]
    assert workflow._mps_backend is None
    assert workflow._data is None


def test_workflow_releases_unprepared_mps_source() -> None:
    """Closing before backend preparation must still free the raw Metal load."""

    from quantem.gpu.ssb.workflow import SSB

    calls: list[str] = []

    class _Source:
        def free(self) -> None:
            calls.append("free")

    workflow = SSB.__new__(SSB)
    workflow._cuda_session = None
    workflow._mps_backend = None
    workflow._data = _Source()

    workflow.close()

    assert calls == ["free"]
    assert workflow._data is None
