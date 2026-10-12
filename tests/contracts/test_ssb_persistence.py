"""Saved-result reuse for the public SSB workflow."""

import json
import math

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
            scan_detector_rotation_deg=3.0,
            loss=0.125,
            elapsed=2.5,
            timings={"fit": 2.0, "reconstruct": 0.5},
            num_trials=trials,
            num_bright_field_pixels=17,
            refine_method=settings["refinement"],
            refine_nfev=23,
            refine_elapsed=0.4,
            voltage_kV=300.0,
            semiangle_mrad=30.0,
            object_sampling_A=(0.2, 0.3),
            bright_field_center_px=(31.5, 32.5),
            bright_field_radius_px=14.0,
            bright_field_edge_radius_px=14.5,
            trial_records=[
                {
                    "params": {"C10_nm": -101.0, "C12_nm": 4.0},
                    "loss": 0.125,
                }
            ],
        )

    num_bright_field_pixels = 17

    def preview(self, aberrations, *, compute_loss=True, **kwargs):
        self.reconstruct_calls += 1
        if self.backend == "cuda":
            import cupy as xp
        else:
            xp = np
        return xp.full((2, 2), 0.2, dtype=xp.float32), 0.25 if compute_loss else None

    def preview_upsampled(self, aberrations, **kwargs):
        return self.preview(aberrations, **kwargs)

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

    monkeypatch.setattr(workflow, "resolve", lambda _backend: device)
    monkeypatch.setattr(workflow.SSB, "_backend", lambda _self: backend)
    session = workflow.SSB(
        np.zeros((2, 2, 3, 3), dtype=np.uint16),
        backend=device,
        voltage_kV=300.0,
        semiangle_mrad=30.0,
        scan_sampling_A=(0.2, 0.3),
        detector_sampling_mrad=(0.01, 0.01),
        source_path=str(source),
    )
    # This double represents the immutable source opened by SSB.open. Direct
    # arrays have a separate no-reuse contract below.
    session.source_kind = "detector"
    return session


def test_fit_reuses_exact_saved_result_with_complete_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    """A matching fit loads its object wave and metadata without fitting."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    first = _session(monkeypatch, source, backend).find_aberrations(
        trials=200,
        refinement="nelder-mead",
        save_to=saved,
        verbose=False,
    )
    second = _session(monkeypatch, source, backend).find_aberrations(
        trials=200,
        refinement="nelder-mead",
        save_to=saved,
        verbose=False,
    )

    assert backend.fit_calls == 1
    assert not first.reused
    assert second.reused
    np.testing.assert_array_equal(second.object_wave, first.object_wave)
    assert dict(second) == dict(first)
    assert second["C10_nm"] == -10.1
    assert second["C12_nm"] == 0.4
    assert second["phi12_deg"] == math.degrees(0.25)
    copied = dict(second)
    copied["C10_nm"] = 0.0
    assert second["C10_nm"] == first["C10_nm"] == -10.1
    assert second.timings == first.timings
    assert second.trial_records == first.trial_records
    assert second.saved_path == saved / "ssb-find_aberrations.npz"
    assert second.metadata["signature"]["settings"]["trials"] == 200
    assert second.metadata["signature"]["software"]["ssb_source_sha256"]
    assert second.metadata["input"]["source_path"] == str(source)
    assert second.metadata["result"]["loss"] == 0.125

    readable = json.loads((saved / "ssb-find_aberrations.json").read_text(encoding="utf-8"))
    assert readable["signature"]["instrument"]["voltage_kV"] == 300.0
    assert readable["signature"]["data"]["dtype"] == "uint16"
    assert readable["result"]["bright_field_center_px"] == [31.5, 32.5]


def test_changed_fit_settings_or_source_recompute(tmp_path, monkeypatch) -> None:
    """Scientific input changes invalidate a prior fit automatically."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"first detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    _session(monkeypatch, source, backend).find_aberrations(
        trials=200, save_to=saved, verbose=False
    )
    _session(monkeypatch, source, backend).find_aberrations(
        trials=201, save_to=saved, verbose=False
    )
    source.write_bytes(b"different detector counts")
    _session(monkeypatch, source, backend).find_aberrations(
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
    first.find_aberrations(save_to=saved, verbose=False)

    same = _session(monkeypatch, source, backend)
    same.calibration_path = str(calibration)
    reused = same.find_aberrations(save_to=saved, verbose=False)

    calibration.write_text('{"rotation": 2}', encoding="utf-8")
    changed = _session(monkeypatch, source, backend)
    changed.calibration_path = str(calibration)
    recomputed = changed.find_aberrations(save_to=saved, verbose=False)

    assert reused.reused
    assert not recomputed.reused
    assert backend.fit_calls == 2


def test_force_recomputes_matching_fit(tmp_path, monkeypatch) -> None:
    """force=True bypasses an otherwise exact saved result."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    _session(monkeypatch, source, backend).find_aberrations(save_to=saved, verbose=False)
    forced = _session(monkeypatch, source, backend).find_aberrations(
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
    first_coefs = {"C10_nm": -100.0, "C12_nm": 5.0, "phi12_deg": math.degrees(0.2)}
    changed_coefs = {"C10_nm": -99.0, "C12_nm": 5.0, "phi12_deg": math.degrees(0.2)}

    first = _session(monkeypatch, source, backend).reconstruct(
        aberrations=first_coefs, save_to=saved
    )
    second = _session(monkeypatch, source, backend).reconstruct(
        aberrations=first_coefs, save_to=saved
    )
    changed = _session(monkeypatch, source, backend).reconstruct(
        aberrations=changed_coefs, save_to=saved
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
    monkeypatch.setattr(workflow, "resolve", lambda _backend: "mps")
    monkeypatch.setattr(workflow.SSB, "_backend", lambda _self: backend)
    session = workflow.SSB(
        np.zeros((2, 2, 3, 3), dtype=np.uint16),
        backend="mps",
        voltage_kV=300.0,
        semiangle_mrad=30.0,
        scan_sampling_A=(0.2, 0.2),
    )

    import pytest

    with pytest.raises(ValueError, match="requires source_path"):
        session.find_aberrations(save_to=tmp_path / "result", verbose=False)


def test_array_inputs_never_reuse_saved_results_by_source_path(tmp_path, monkeypatch):
    """Different crops and in-place edits cannot inherit another array's result."""
    source = tmp_path / "scan.bin"
    source.write_bytes(b"unchanged full acquisition")
    backend = _Backend()
    for offset in (0, 1, 2):
        session = _session(monkeypatch, source, backend)
        session.source_kind = "array"
        session._data[...] = offset
        search = session.find_aberrations(save_to=tmp_path / "search", verbose=False)
        result = session.reconstruct(save_to=tmp_path / "fixed")
        assert not search.reused
        assert not result.reused
        assert search.saved_path.is_file()
        assert result.saved_path.is_file()
    assert backend.fit_calls == 3
    assert backend.reconstruct_calls == 6


def test_changed_refinement_start_invalidates_saved_search(tmp_path, monkeypatch):
    """Current coefficients participate in the saved search identity."""
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    backend = _Backend()
    session = _session(monkeypatch, source, backend)
    settings = dict(trials=0, refinement=None, check_rotation=False,
                    save_to=tmp_path / "search", verbose=False)
    first = session.find_aberrations(**settings)
    session.reconstruct(aberrations={"C10_nm": 12.0})
    second = session.find_aberrations(**settings)
    assert not second.reused
    assert backend.fit_calls == 2
    assert first.metadata["signature"]["settings"]["starting_aberrations"]["C10_nm"] == 0.0
    assert second.metadata["signature"]["settings"]["starting_aberrations"]["C10_nm"] == 12.0


def test_cuda_reuse_restores_object_wave_to_cuda(tmp_path, monkeypatch) -> None:
    """A reused CUDA result returns to device memory rather than staying on CPU."""

    import pytest

    cp = pytest.importorskip("cupy")
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend("cuda")

    _session(monkeypatch, source, backend, device="cuda").find_aberrations(
        save_to=saved, verbose=False
    )
    reused = _session(monkeypatch, source, backend, device="cuda").find_aberrations(
        save_to=saved, verbose=False
    )

    assert reused.reused
    assert isinstance(reused.object_wave, cp.ndarray)


def test_stored_fit_without_the_current_schema_is_no_fit(tmp_path):
    """A stored fit counts only with the current schema marker; any other block reads as no fit, without an error."""
    from quantem.gpu.ssb.persistence import FIT_SCHEMA, RESULT_SCHEMA, SCHEMA
    from quantem.gpu.ssb.series import saved_fit, stored_fit

    aberrations = {"C10_nm": 10.0, "C12_nm": 2.0, "phi12_deg": math.degrees(0.3)}
    block = {"aberrations": aberrations, "scan_detector_rotation_deg": 210.0, "detector_transposed": False}
    config = tmp_path / "config.json"
    for marker in ({}, {"schema": f"quantem.ssb-fit/{SCHEMA - 1}"}, {"schema": SCHEMA}):
        config.write_text(json.dumps({"computed": {"ssb": {**block, **marker}}}))
        assert stored_fit(json.loads(config.read_text())) == {}
        assert saved_fit(tmp_path) is None
    config.write_text(json.dumps({"computed": {"ssb": {**block, "schema": FIT_SCHEMA}}}))
    assert saved_fit(tmp_path) == {**block, "schema": FIT_SCHEMA}
    config.unlink()
    record = tmp_path / "ssb-fit" / "ssb-find_aberrations.json"
    record.parent.mkdir()
    # the saved result carries a schema string too; the earlier bare number, old or current, is no longer a match
    for marker in (SCHEMA - 1, SCHEMA, f"quantem.ssb-result/{SCHEMA - 1}"):
        record.write_text(json.dumps({"schema": marker, "result": block}))
        assert saved_fit(tmp_path) is None
    record.write_text(json.dumps({"schema": RESULT_SCHEMA, "result": block}))
    assert saved_fit(tmp_path) == {"aberrations": aberrations, "scan_detector_rotation_deg": 210.0}


def test_engine_written_fit_reads_back_and_replaces_an_earlier_block(tmp_path):
    """write_fit_metadata marks its block and saved_fit returns the exact values; no field of an unmarked block survives."""
    from quantem.gpu import SSBResult
    from quantem.gpu.ssb.persistence import FIT_SCHEMA
    from quantem.gpu.ssb.series import saved_fit, write_fit_metadata

    config = tmp_path / "config.json"
    earlier = {"aberrations": {"C10": 5.0, "C12": -0.3, "phi12": 0.18}, "rotation_angle_deg": 0.0, "com_reversed": True,
               "aberration_unit": "nm", "picked_c10_nm": 4.0, "optuna_trials": [{"loss": 0.02}]}
    config.write_text(json.dumps({"human": {"notes": "kept"}, "computed": {"bright_field_radius_px": 55.0, "ssb": earlier}}))
    aberrations = {"C10_nm": -10.5, "C12_nm": 0.25, "phi12_deg": math.degrees(-0.5)}
    result = SSBResult(object_wave=np.ones((2, 2), dtype=np.complex64), backend="cuda", aberrations=aberrations,
                       scan_detector_rotation_deg=200.0, loss=0.125, bright_field_radius_px=48.0)
    settings = {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": (0.5, 0.5), "scan_detector_rotation_deg": 200.0, "bright_field_radius_px": None}
    write_fit_metadata(tmp_path, settings, result)
    written = json.loads(config.read_text())
    block = {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": [0.5, 0.5], "schema": FIT_SCHEMA, "aberrations": aberrations,
             "scan_detector_rotation_deg": 200.0, "detector_transposed": False, "loss": 0.125, "bright_field_radius_px": 48.0}
    assert written == {"human": {"notes": "kept"}, "computed": {"bright_field_radius_px": 55.0, "ssb": block}}
    assert saved_fit(tmp_path) == block
    # a field the dashboard adds to a current block stays through the next fit
    written["computed"]["ssb"]["ran_at"] = "2026-10-11T12:00:00"
    config.write_text(json.dumps(written))
    write_fit_metadata(tmp_path, settings, result)
    assert json.loads(config.read_text())["computed"]["ssb"] == {**block, "ran_at": "2026-10-11T12:00:00"}
