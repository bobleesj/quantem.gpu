"""SSB reconstructs the phase of the mean wave; the retired mean_phase and complex_wave estimators name their replacement."""

import json

import numpy as np
import pytest

ABERRATIONS = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": 30.0}


class _Backend:
    """The one backend member a reconstruction calls, recording what it was asked for."""

    backend = "mps"
    num_bright_field_pixels = 5

    def __init__(self):
        self.calls = []

    def preview_upsampled(self, aberrations, *, upsampling_factor, compute_loss, **kwargs):
        assert "phase_estimator" not in kwargs
        self.calls.append((upsampling_factor, compute_loss))
        return np.zeros((2, 2), dtype=np.float32), 0.5 if compute_loss else None


def _session(monkeypatch):
    from quantem.gpu.direct_ptycho import workflow

    backend = _Backend()
    monkeypatch.setattr(workflow, "resolve", lambda _backend: "mps")
    session = workflow.SSB(np.zeros((2, 2, 3, 3), np.uint16), voltage_kV=300.0, semiangle_mrad=30.0,
                           scan_sampling_A=(0.2, 0.2), detector_sampling_mrad=(1.0, 1.0), aberrations=ABERRATIONS)
    monkeypatch.setattr(session, "_backend", lambda: backend)
    return session, backend


@pytest.mark.parametrize(("estimator", "replacement"), [("mean_phase", "method='phase'"),
                                                        ("complex_wave", "method='complex'")])
def test_a_retired_estimator_raises_naming_its_replacement(monkeypatch, estimator, replacement):
    session, backend = _session(monkeypatch)
    with pytest.raises(ValueError, match=f"{estimator}' was retired.*{replacement}"):
        session.run(phase_estimator=estimator)
    with pytest.raises(ValueError, match=f"{estimator}' was retired.*{replacement}"):
        session.preview(ABERRATIONS, phase_estimator=estimator)
    with pytest.raises(ValueError, match=f"{estimator}' was retired"):
        session.run(method="phase", phase_estimator=estimator)
    assert backend.calls == []


def test_an_unknown_estimator_names_the_one_ssb_computes(monkeypatch):
    session, _ = _session(monkeypatch)
    with pytest.raises(ValueError, match="must be 'phase_of_mean'"):
        session.preview(ABERRATIONS, phase_estimator="phase_mean")


def test_phase_of_mean_is_what_ssb_records_and_still_accepted(monkeypatch):
    session, backend = _session(monkeypatch)
    phase, loss = session.preview(ABERRATIONS, phase_estimator="phase_of_mean", compute_loss=False)
    assert phase.shape == (2, 2) and loss is None
    result = session.run(compute_loss=True)
    assert result.phase_estimator == "phase_of_mean"
    assert not result.amplitude_estimated
    assert result.loss == 0.5
    assert backend.calls == [(1, False), (1, True)]


def test_a_result_saved_with_a_retired_estimator_loads_with_a_notice(tmp_path, capsys):
    from quantem.gpu.direct_ptycho.persistence import load_result, result_paths, save_result
    from quantem.gpu.direct_ptycho.results import DirectPtychoResult

    paths = result_paths(tmp_path, "find_aberrations")
    signature = {"operation": "find_aberrations"}
    saved = DirectPtychoResult(object_wave=np.exp(1j * np.ones((2, 2), np.float32)), backend="mps",
                      aberrations=dict(ABERRATIONS), phase_estimator="phase_of_mean", amplitude_estimated=False)
    save_result(saved, paths=paths, signature=signature, input_metadata={})
    record = json.loads(paths[1].read_text())
    record["result"]["phase_estimator"] = "mean_phase"
    paths[1].write_text(json.dumps(record))

    loaded = load_result(paths=paths, signature=signature, backend="mps")
    assert loaded is not None and loaded.phase_estimator == "mean_phase"
    assert loaded.aberrations == ABERRATIONS
    printed = capsys.readouterr().out
    assert "saved with the retired mean_phase estimator" in printed and "method='phase'" in printed


def test_an_unrecorded_estimator_is_not_shown():
    from quantem.gpu.direct_ptycho.results import DirectPtychoResult, retired_estimator_note

    assert "Estimator" not in repr(DirectPtychoResult(object_wave=np.ones((2, 2), np.complex64), backend="mps"))
    assert retired_estimator_note("phase_of_mean", "fit.json") is None
    assert retired_estimator_note(None, "fit.json") is None
    assert "method='complex'" in retired_estimator_note("complex_wave", "fit.json")
