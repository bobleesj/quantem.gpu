"""Resource-lifetime tests for the public MPS SSB backend."""

from types import SimpleNamespace

import numpy as np


def test_close_releases_backend_state_and_mlx_cache(monkeypatch) -> None:
    """Closing an MPS backend returns allocator-owned unified memory and leaves its source to the owner."""

    from quantem.gpu.ssb.mps import backend as backend_module

    calls: list[str] = []
    monkeypatch.setattr(
        backend_module,
        "require_mlx",
        lambda: SimpleNamespace(clear_cache=lambda: calls.append("clear")),
    )
    backend = backend_module.MpsSSBBackend.__new__(backend_module.MpsSSBBackend)
    for name in (
        "_prepared",
        "_frames",
        "_fit_preview_phase",
        "_fit_preview_loss",
        "_fit_preview_aberrations",
    ):
        setattr(backend, name, object())

    backend.close()

    assert calls == ["clear"]
    assert backend._prepared is None
    assert backend._frames is None
    assert backend._fit_preview_phase is None
    assert backend._fit_preview_loss is None
    assert backend._fit_preview_aberrations is None


def test_workflow_drops_shared_source_before_mps_allocator_flush() -> None:
    """The MPS close hook must run only after the workflow drops its reference to the data."""

    from quantem.gpu.ssb.workflow import SSB

    workflow = SSB.__new__(SSB)
    workflow._cuda_session = None
    workflow._data = np.ones((2, 2, 3, 3), dtype=np.uint16)
    observations: list[object | None] = []

    class _Backend:
        def close(self) -> None:
            observations.append(workflow._data)

    workflow._mps_backend = _Backend()

    workflow.close()

    assert observations == [None]
    assert workflow._mps_backend is None
    assert workflow._data is None
