"""Saved-result reuse for the public SSB workflow."""
from __future__ import annotations

import json

import numpy as np


class _Backend:
    """Small backend double that records expensive operations."""

    def __init__(self, backend: str = "mps") -> None:
        self.backend = backend
        self.fit_calls = 0
        self.reconstruct_calls = 0

    def fit(self, **settings):
        from quantem.gpu import SSBResult

        self.fit_calls += 1
        trials = int(settings["trials"])
        return SSBResult(
            object_wave=np.full((2, 2), 1 + 2j, dtype=np.complex64),
            backend=self.backend,
            aberrations={"C10": -101.0, "C12": 4.0, "phi12": 0.25},
            rotation_angle_deg=3.0,
            loss=0.125,
            elapsed=2.5,
            timings={"fit": 2.0, "reconstruct": 0.5},
            n_trials=trials,
            num_bf=17,
            refine_method=settings["refinement"],
            refine_nfev=23,
            refine_elapsed=0.4,
            voltage_kV=300.0,
            semiangle_mrad=30.0,
            scan_sampling_A=(0.2, 0.3),
            bf_center=(31.5, 32.5),
            bf_radius=14.0,
            detected_bf_radius=14.5,
            optuna_trials=[
                {
                    "params": {"C10_nm": -101.0, "C12_nm": 4.0},
                    "loss": 0.125,
                }
            ],
        )

    def reconstruct_result(self, aberrations, *, compute_loss=True):
        from quantem.gpu import SSBResult

        self.reconstruct_calls += 1
        return SSBResult(
            object_wave=np.full((2, 2), 3 + 4j, dtype=np.complex64),
            backend=self.backend,
            aberrations=dict(aberrations),
            loss=0.25 if compute_loss else None,
            timings={"reconstruct": 0.2},
        )


def _session(monkeypatch, source, backend, device="mps"):
    from quantem.gpu.ssb import workflow

    monkeypatch.setattr(workflow, "_resolve_backend", lambda _backend: device)
    monkeypatch.setattr(
        workflow.SSB,
        "_backend_protocol",
        property(lambda _self: backend),
    )
    return workflow.SSB.from_array(
        np.zeros((2, 2, 3, 3), dtype=np.uint16),
        backend=device,
        voltage_kV=300.0,
        semiangle_mrad=30.0,
        scan_sampling_A=(0.2, 0.3),
        det_sampling=(0.01, 0.01),
        source_path=str(source),
    )


def test_fit_reuses_exact_saved_result_with_complete_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    """A matching fit loads its object wave and metadata without fitting."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    first = _session(monkeypatch, source, backend).fit(
        trials=200,
        refinement="nelder-mead",
        save_to=saved,
        verbose=False,
    )
    second = _session(monkeypatch, source, backend).fit(
        trials=200,
        refinement="nelder-mead",
        save_to=saved,
        verbose=False,
    )

    assert backend.fit_calls == 1
    assert not first.reused
    assert second.reused
    np.testing.assert_array_equal(second.object_wave, first.object_wave)
    assert second.aberrations == first.aberrations
    assert second.timings == first.timings
    assert second.optuna_trials == first.optuna_trials
    assert second.saved_path == saved / "ssb-fit.npz"
    assert second.metadata["signature"]["settings"]["trials"] == 200
    assert second.metadata["signature"]["software"]["ssb_source_sha256"]
    assert second.metadata["input"]["source_path"] == str(source)
    assert second.metadata["result"]["loss"] == 0.125

    readable = json.loads((saved / "ssb-fit.json").read_text(encoding="utf-8"))
    assert readable["signature"]["instrument"]["voltage_kV"] == 300.0
    assert readable["signature"]["data"]["dtype"] == "uint16"
    assert readable["result"]["bf_center"] == [31.5, 32.5]


def test_changed_fit_settings_or_source_recompute(tmp_path, monkeypatch) -> None:
    """Scientific input changes invalidate a prior fit automatically."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"first detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    _session(monkeypatch, source, backend).fit(
        trials=200, save_to=saved, verbose=False
    )
    _session(monkeypatch, source, backend).fit(
        trials=201, save_to=saved, verbose=False
    )
    source.write_bytes(b"different detector counts")
    _session(monkeypatch, source, backend).fit(
        trials=201, save_to=saved, verbose=False
    )

    assert backend.fit_calls == 3


def test_changed_calibration_recomputes(tmp_path, monkeypatch) -> None:
    """A changed calibration artifact invalidates an otherwise matching fit."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"rotation": 1}', encoding="utf-8")
    saved = tmp_path / "result"
    backend = _Backend()

    first = _session(monkeypatch, source, backend)
    first.calibration_path = str(calibration)
    first.fit(save_to=saved, verbose=False)

    same = _session(monkeypatch, source, backend)
    same.calibration_path = str(calibration)
    reused = same.fit(save_to=saved, verbose=False)

    calibration.write_text('{"rotation": 2}', encoding="utf-8")
    changed = _session(monkeypatch, source, backend)
    changed.calibration_path = str(calibration)
    recomputed = changed.fit(save_to=saved, verbose=False)

    assert reused.reused
    assert not recomputed.reused
    assert backend.fit_calls == 2


def test_force_recomputes_matching_fit(tmp_path, monkeypatch) -> None:
    """force=True bypasses an otherwise exact saved result."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    _session(monkeypatch, source, backend).fit(save_to=saved, verbose=False)
    forced = _session(monkeypatch, source, backend).fit(
        save_to=saved, force=True, verbose=False
    )

    assert backend.fit_calls == 2
    assert not forced.reused


def test_fixed_reconstruction_reuses_only_matching_aberrations(
    tmp_path,
    monkeypatch,
) -> None:
    """Fixed SSB reconstructions include coefficients in their identity."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend()
    first_coefs = {"C10": -100.0, "C12": 5.0, "phi12": 0.2}
    changed_coefs = {"C10": -99.0, "C12": 5.0, "phi12": 0.2}

    first = _session(monkeypatch, source, backend).reconstruct(
        first_coefs, save_to=saved
    )
    second = _session(monkeypatch, source, backend).reconstruct(
        first_coefs, save_to=saved
    )
    changed = _session(monkeypatch, source, backend).reconstruct(
        changed_coefs, save_to=saved
    )

    assert backend.reconstruct_calls == 2
    assert not first.reused
    assert second.reused
    assert not changed.reused
    assert changed.aberrations == changed_coefs
    assert second.metadata["signature"]["settings"]["compute_loss"] is True


def test_saved_result_requires_source_identity(tmp_path, monkeypatch) -> None:
    """In-memory arrays cannot be reused from shape alone."""

    from quantem.gpu.ssb import workflow

    backend = _Backend()
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _backend: "mps")
    monkeypatch.setattr(
        workflow.SSB,
        "_backend_protocol",
        property(lambda _self: backend),
    )
    session = workflow.SSB.from_array(
        np.zeros((2, 2, 3, 3), dtype=np.uint16),
        backend="mps",
        voltage_kV=300.0,
        semiangle_mrad=30.0,
        scan_sampling_A=0.2,
    )

    import pytest

    with pytest.raises(ValueError, match="requires source_path"):
        session.fit(save_to=tmp_path / "result", verbose=False)


def test_cuda_reuse_restores_object_wave_to_cuda(tmp_path, monkeypatch) -> None:
    """A reused CUDA result returns to device memory rather than staying on CPU."""

    import pytest

    cp = pytest.importorskip("cupy")
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend("cuda")

    _session(monkeypatch, source, backend, device="cuda").fit(
        save_to=saved, verbose=False
    )
    reused = _session(monkeypatch, source, backend, device="cuda").fit(
        save_to=saved, verbose=False
    )

    assert reused.reused
    assert isinstance(reused.object_wave, cp.ndarray)
