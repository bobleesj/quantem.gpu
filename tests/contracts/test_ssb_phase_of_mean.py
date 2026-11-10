"""SSB reconstructs the phase of the mean wave and records it as phase_of_mean."""

import numpy as np

ABERRATIONS = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": 30.0}


class _Backend:
    """The one backend member a reconstruction calls, recording what it was asked for."""

    backend = "mps"
    num_bright_field_pixels = 5

    def __init__(self):
        self.calls = []

    def preview_upsampled(self, aberrations, *, upsample, compute_loss, **kwargs):
        assert "phase_estimator" not in kwargs
        self.calls.append((upsample, compute_loss))
        return np.zeros((2, 2), dtype=np.float32), 0.5 if compute_loss else None


def _session(monkeypatch):
    from quantem.gpu.direct_ptycho import workflow

    backend = _Backend()
    monkeypatch.setattr(workflow, "resolve", lambda _backend: "mps")
    session = workflow.DirectPtycho(np.zeros((2, 2, 3, 3), np.uint16), voltage_kV=300.0, semiangle_mrad=30.0,
                           scan_sampling_A=(0.2, 0.2), detector_sampling_mrad=(1.0, 1.0), aberrations=ABERRATIONS)
    monkeypatch.setattr(session, "_backend", lambda: backend)
    return session, backend


def test_phase_of_mean_is_what_ssb_records(monkeypatch):
    session, backend = _session(monkeypatch)
    phase, loss = session.preview(ABERRATIONS, compute_loss=False)
    assert phase.shape == (2, 2) and loss is None
    result = session.run(compute_loss=True)
    assert result.phase_estimator == "phase_of_mean"
    assert not result.amplitude_estimated
    assert result.loss == 0.5
    assert backend.calls == [(1, False), (1, True)]


def test_an_unrecorded_estimator_is_not_shown():
    from quantem.gpu.direct_ptycho.results import DirectPtychoResult

    assert "Estimator" not in repr(DirectPtychoResult(object_wave=np.ones((2, 2), np.complex64), backend="mps"))
