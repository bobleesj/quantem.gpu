"""Saved-result reuse for the public SSB workflow."""

import json
from pathlib import Path
import math

import numpy as np
import pytest


class _Backend:
    """Small backend double that records expensive operations."""

    def __init__(self, backend: str = "mps") -> None:
        self.backend = backend
        self.fit_calls = 0
        self.reconstruct_calls = 0

    def fit(self, **settings):
        from quantem.gpu import DirectPtychoResult

        self.fit_calls += 1
        trials = int(settings["trials"])
        return DirectPtychoResult(
            object_wave=np.full((2, 2), 1 + 2j, dtype=np.complex64),
            backend=self.backend,
            aberrations={"C10": -101.0, "C12": 4.0, "phi12": 0.25},
            scan_detector_rotation_deg=3.0,
            loss=0.125,
            elapsed_s=2.5,
            timings={"fit": 2.0, "reconstruct": 0.5},
            num_trials=trials,
            num_bright_field_pixels=17,
            refinement=settings["refinement"],
            refine_nfev=23,
            refine_elapsed_s=0.4,
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

    def band_power(self, band_inv_A):
        """G's power over the band: the fits' sample_agreement is their agreement over it."""
        return 10.0

    def cache_rotation(self, angle):
        """Accept the workflow's temporary diagnostic rotation."""

    def preview(self, aberrations, *, compute_loss=True, **kwargs):
        self.reconstruct_calls += 1
        if self.backend == "cuda":
            import cupy as xp
        else:
            xp = np
        return xp.full((2, 2), 0.2, dtype=xp.float32), 0.25 if compute_loss else None

    def preview_upsampled(self, aberrations, **kwargs):
        return self.preview(aberrations, **kwargs)


def _session(monkeypatch, source, backend, device="mps"):
    from quantem.gpu.direct_ptycho import workflow

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
        # a given rotation is fitted once; an estimated one would fit both 180-degree branches
        scan_detector_rotation_deg=3.0,
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

    first = _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False,
        num_trials=200,
        save_dir=saved,
        verbose=False,
    )
    second = _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False,
        num_trials=200,
        save_dir=saved,
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
    assert second.metadata["signature"]["settings"]["num_trials"] == 200
    assert second.metadata["signature"]["software"]["ssb_source_sha256"]
    assert second.metadata["input"]["source"] == {"file_name": "scan.bin", "relative_path": None}
    assert second.metadata["result"]["loss"] == 0.125

    readable = json.loads((saved / "ssb-find_aberrations.json").read_text(encoding="utf-8"))
    assert readable["signature"]["instrument"]["voltage_kV"] == 300.0
    assert readable["signature"]["data"]["dtype"] == "uint16"
    assert readable["result"]["bright_field_center_px"] == [31.5, 32.5]


def test_saved_fit_without_diagnostic_does_not_satisfy_a_loss_request(tmp_path, monkeypatch):
    """Reuse the fast saved fit, but compute the diagnostic when requested."""
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    backend = _Backend()
    options = dict(tilt=False, num_trials=200, save_dir=tmp_path / "result",
                   verbose=False)
    first = _session(monkeypatch, source, backend).find_aberrations(search="primary",
        **options, compute_loss=False,
    )
    reopened = _session(monkeypatch, source, backend).find_aberrations(search="primary",
        **options, compute_loss=False,
    )
    assert reopened.reused and reopened.loss is None and first.loss is None
    assert backend.fit_calls == 1
    np.testing.assert_array_equal(reopened.phase, first.phase)
    diagnostic = _session(monkeypatch, source, backend).find_aberrations(search="primary", **options)
    assert not diagnostic.reused and diagnostic.loss == 0.125
    assert backend.fit_calls == 2


def test_changed_fit_settings_or_source_recompute(tmp_path, monkeypatch) -> None:
    """Scientific input changes invalidate a prior fit automatically."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"first detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False,
        num_trials=200, save_dir=saved, verbose=False
    )
    _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False,
        num_trials=201, save_dir=saved, verbose=False
    )
    source.write_bytes(b"different detector counts")
    _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False,
        num_trials=201, save_dir=saved, verbose=False
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
    first.find_aberrations(search="primary", tilt=False, save_dir=saved, verbose=False)

    same = _session(monkeypatch, source, backend)
    same.calibration_path = str(calibration)
    reused = same.find_aberrations(search="primary", tilt=False, save_dir=saved, verbose=False)

    calibration.write_text('{"rotation": 2}', encoding="utf-8")
    changed = _session(monkeypatch, source, backend)
    changed.calibration_path = str(calibration)
    recomputed = changed.find_aberrations(search="primary", tilt=False, save_dir=saved, verbose=False)

    assert reused.reused
    assert not recomputed.reused
    assert backend.fit_calls == 2


def test_force_recomputes_matching_fit(tmp_path, monkeypatch) -> None:
    """force=True bypasses an otherwise exact saved result."""

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend()

    _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False, save_dir=saved, verbose=False)
    forced = _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False,
        save_dir=saved, force=True, verbose=False
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
        aberrations=first_coefs, save_dir=saved
    )
    second = _session(monkeypatch, source, backend).reconstruct(
        aberrations=first_coefs, save_dir=saved
    )
    changed = _session(monkeypatch, source, backend).reconstruct(
        aberrations=changed_coefs, save_dir=saved
    )

    assert backend.reconstruct_calls == 2
    assert not first.reused
    assert second.reused
    assert not changed.reused
    assert changed.aberrations == changed_coefs
    assert second.metadata["signature"]["settings"]["compute_loss"] is True


def test_saved_result_requires_source_identity(tmp_path, monkeypatch) -> None:
    """In-memory arrays cannot be reused from shape alone."""

    from quantem.gpu.direct_ptycho import workflow

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
        session.find_aberrations(search="primary", tilt=False, save_dir=tmp_path / "result", verbose=False)


def test_array_inputs_never_reuse_saved_results_by_source_path(tmp_path, monkeypatch):
    """Different crops and in-place edits cannot inherit another array's result."""
    source = tmp_path / "scan.bin"
    source.write_bytes(b"unchanged full acquisition")
    backend = _Backend()
    for offset in (0, 1, 2):
        session = _session(monkeypatch, source, backend)
        session.source_kind = "array"
        session._data[...] = offset
        search = session.find_aberrations(search="primary", tilt=False, save_dir=tmp_path / "search", verbose=False)
        result = session.reconstruct(save_dir=tmp_path / "fixed")
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
    settings = dict(tilt=False, num_trials=0, refinement=None, check_rotation=False,
                    save_dir=tmp_path / "search", verbose=False)
    first = session.find_aberrations(search="primary", **settings)
    session.reconstruct(aberrations={"C10_nm": 12.0})
    second = session.find_aberrations(search="primary", **settings)
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

    _session(monkeypatch, source, backend, device="cuda").find_aberrations(search="primary", tilt=False,
        save_dir=saved, verbose=False
    )
    reused = _session(monkeypatch, source, backend, device="cuda").find_aberrations(search="primary", tilt=False,
        save_dir=saved, verbose=False
    )

    assert reused.reused
    assert isinstance(reused.object_wave, cp.ndarray)


def test_stored_fit_without_the_current_schema_is_no_fit(tmp_path):
    """A stored fit counts only with the current schema marker; any other block reads as no fit, without an error."""
    from quantem.gpu.direct_ptycho.persistence import FIT_SCHEMA, RESULT_SCHEMA, RESULT_VERSION, SCHEMA
    from quantem.gpu.direct_ptycho.series import saved_fit, stored_fit

    aberrations = {"C10_nm": 10.0, "C12_nm": 2.0, "phi12_deg": math.degrees(0.3)}
    block = {"aberrations": aberrations, "scan_detector_rotation_deg": 210.0, "scan_detector_rotation_flipped": False}
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
    for marker in (SCHEMA - 1, SCHEMA, f"quantem.ssb-result/{RESULT_VERSION - 1}"):
        record.write_text(json.dumps({"schema": marker, "result": block}))
        assert saved_fit(tmp_path) is None
    record.write_text(json.dumps({"schema": RESULT_SCHEMA, "result": block}))
    assert saved_fit(tmp_path) == {"aberrations": aberrations, "scan_detector_rotation_deg": 210.0,
                                   "scan_detector_rotation_flipped": False}


def test_engine_written_fit_reads_back_and_replaces_an_earlier_block(tmp_path):
    """write_fit_metadata marks its block and saved_fit returns the exact values; no field of an unmarked block survives."""
    from quantem.gpu import DirectPtychoResult
    from quantem.gpu.direct_ptycho.persistence import FIT_SCHEMA, SCREEN_ACQUISITION_SCHEMA
    from quantem.gpu.direct_ptycho.series import saved_fit, write_fit_metadata

    config = tmp_path / "config.json"
    earlier = {"aberrations": {"C10": 5.0, "C12": -0.3, "phi12": 0.18}, "rotation_angle_deg": 0.0, "com_reversed": True,
               "aberration_unit": "nm", "picked_c10_nm": 4.0, "optuna_trials": [{"loss": 0.02}]}
    # a current screening record that still holds a fit block of an earlier fit schema
    config.write_text(json.dumps({"schema": SCREEN_ACQUISITION_SCHEMA, "human": {"notes": "kept"},
                                  "computed": {"bright_field_radius_px": 55.0, "ssb": earlier}}))
    aberrations = {"C10_nm": -10.5, "C12_nm": 0.25, "phi12_deg": math.degrees(-0.5)}
    result = DirectPtychoResult(object_wave=np.ones((2, 2), dtype=np.complex64), backend="cuda", aberrations=aberrations,
                       scan_detector_rotation_deg=200.0, loss=0.125, bright_field_radius_px=48.0)
    settings = {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": (0.5, 0.5), "scan_detector_rotation_deg": 200.0, "bright_field_radius_px": None}
    write_fit_metadata(tmp_path, settings, result, (0.55, 0.56))
    written = json.loads(config.read_text())
    block = {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": [0.5, 0.5], "schema": FIT_SCHEMA, "aberrations": aberrations,
             "scan_detector_rotation_deg": 200.0, "scan_detector_rotation_flipped": False, "loss": 0.125, "bright_field_radius_px": 48.0,
             "bright_field_center_px": None, "detector_sampling_mrad": [0.55, 0.56]}
    assert written == {"schema": SCREEN_ACQUISITION_SCHEMA, "human": {"notes": "kept"},
                       "computed": {"bright_field_radius_px": 55.0, "ssb": block}}
    assert saved_fit(tmp_path) == block
    # a field the dashboard adds to a current block stays through the next fit
    written["computed"]["ssb"]["ran_at"] = "2026-10-11T12:00:00"
    config.write_text(json.dumps(written))
    write_fit_metadata(tmp_path, settings, result, (0.55, 0.56))
    assert json.loads(config.read_text())["computed"]["ssb"] == {**block, "ran_at": "2026-10-11T12:00:00"}
    # quantem.live's 180-degree decision on the earlier fit does not survive a new one: beside a new rotation it said
    # this fit had been moved to the other branch
    written = json.loads(config.read_text())
    written["computed"]["ssb"].update(column_sign=-0.4, session_moved_180=True, rotation_disagrees=False,
                                      session_rotation={"scan_detector_rotation_deg": 20.0})
    config.write_text(json.dumps(written))
    write_fit_metadata(tmp_path, settings, result, (0.55, 0.56))
    assert json.loads(config.read_text())["computed"]["ssb"] == {**block, "ran_at": "2026-10-11T12:00:00"}


def test_session_opened_on_exact_columns_holds_the_detector_sampling_its_fit_block_needs(tmp_path, monkeypatch):
    """SSB.open on exact bright-field columns holds the detector sampling its backend uses, and the fit block exports.

    The session kept None there and the backend derived the sampling on its own, so reconstruct_series stored a fit
    with no detector sampling and export_result failed on it with a TypeError. The Apple backend needs MLX, so a
    double takes its place: this checks what the session holds and hands over, not a reconstruction on Apple hardware.
    """
    import h5py

    from quantem.gpu import DirectPtychoResult
    from quantem.gpu.formats.ssb_phase import export_result, read_result
    from quantem.gpu.direct_ptycho import workflow
    from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk
    from quantem.gpu.direct_ptycho.mps.frames import MpsBfColumnFrames
    from quantem.gpu.direct_ptycho.series import write_fit_metadata

    received = []

    class RecordingBackend:
        """Stands in for the Apple backend, which needs MLX: it keeps the sampling the session hands it."""

        def __init__(self, data, **settings):
            received.append(settings["detector_sampling_mrad"])

    selection = BrightfieldDisk(rows=np.asarray([1, 1]), cols=np.asarray([1, 2]), bright_field_center_px=(1.0, 1.5),
                                bright_field_radius_px=1.0, bright_field_edge_radius_px=48.0, detector_shape=(3, 4))
    columns = tmp_path / "columns.u16"
    np.zeros((2, 4), dtype=np.uint16).tofile(columns)
    monkeypatch.setattr(workflow, "resolve", lambda _backend: "mps")
    monkeypatch.setattr(workflow, "MpsDirectPtychoBackend", RecordingBackend)
    settings = {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": (0.25, 0.25)}
    # an export with its own calibration, then one that records only the disk edge
    for exported, expected in (((0.61, 0.62), (0.61, 0.62)), (None, (30.0 / 48.0, 30.0 / 48.0))):
        frames = MpsBfColumnFrames(columns, selection=selection, scan_shape=(2, 2), dtype="uint16", dc_value=1.0,
                                   detector_sampling_mrad=exported)
        monkeypatch.setattr(workflow, "find_bf_columns", lambda *_args, frames=frames, **_kwargs: frames)
        session = workflow.SSB.open(str(tmp_path / "scan_master.h5"), backend="mps", **settings)
        assert session.detector_sampling_mrad == expected
        session._backend()
        assert received[-1] == expected

    master = tmp_path / "scan_master.h5"
    with h5py.File(tmp_path / "scan_data.h5", "w") as handle:
        handle["data"] = np.ones((2, 2, 2), dtype=np.uint16)
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data.h5", "/data")
    np.save(tmp_path / "ssb_phase.npy", np.zeros((2, 2), np.float32))
    result = DirectPtychoResult(object_wave=np.ones((2, 2), dtype=np.complex64), backend="mps", scan_detector_rotation_deg=192.0,
                       aberrations={"C10_nm": 7.0, "C12_nm": 2.0, "phi12_deg": 5.0}, bright_field_center_px=(1.0, 1.5))
    write_fit_metadata(tmp_path, {**settings, "scan_detector_rotation_deg": 192.0}, result, session.detector_sampling_mrad)
    record, _ = read_result(export_result(tmp_path, master, tmp_path / "out.json"))
    assert record["calibration"]["detector_sampling_mrad"] == [30.0 / 48.0, 30.0 / 48.0]


def test_fit_is_stored_only_in_a_current_screening_record(tmp_path):
    """The fit block goes into quantem.live's screening record: created with its schema, or left as found.

    A record this function created held only ``computed.ssb``. quantem.live reads a screening record only with
    ``"schema": "quantem.screen-acquisition/2"`` at its top level, so it left such an acquisition out of its Screen
    list. A record of another schema, or none, is not written into: its other values are in an earlier layout.
    """
    import pytest

    from quantem.gpu import DirectPtychoResult
    from quantem.gpu.direct_ptycho.persistence import FIT_SCHEMA, SCREEN_ACQUISITION_SCHEMA
    from quantem.gpu.direct_ptycho.series import saved_fit, write_fit_metadata

    assert SCREEN_ACQUISITION_SCHEMA == "quantem.screen-acquisition/2"
    aberrations = {"C10_nm": -10.5, "C12_nm": 0.25, "phi12_deg": -0.5}
    result = DirectPtychoResult(object_wave=np.ones((2, 2), dtype=np.complex64), backend="cuda", aberrations=aberrations,
                       scan_detector_rotation_deg=200.0, loss=0.125, bright_field_center_px=(31.5, 32.5))
    settings = {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": (0.5, 0.5), "scan_detector_rotation_deg": 200.0}
    # created: the record carries the schema first, and the fit reads back
    created = tmp_path / "created"
    write_fit_metadata(created, settings, result, (0.6, 0.6))
    record = json.loads((created / "config.json").read_text())
    assert list(record) == ["schema", "computed"] and record["schema"] == SCREEN_ACQUISITION_SCHEMA
    assert saved_fit(created)["aberrations"] == aberrations and record["computed"]["ssb"]["schema"] == FIT_SCHEMA
    # existing and current: every other key stays byte for byte, in its place
    current = tmp_path / "current"
    current.mkdir()
    existing = {"schema": SCREEN_ACQUISITION_SCHEMA, "source_master": "scan_master.h5",
                "human": {"notes": "kept", "rating": 4, "magnification_MX": 5.1},
                "computed": {"bright_field_center_px": [95.25, 96.5], "scan_sampling_A": [0.264, 0.264],
                             "ssb_tilt": {"schema": FIT_SCHEMA, "loss": 0.0625}},
                "timestamp": "2026-10-11T12:00:00"}
    (current / "config.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")
    write_fit_metadata(current, settings, result, (0.6, 0.6))
    expected = json.loads(json.dumps(existing))
    expected["computed"]["ssb"] = json.loads((created / "config.json").read_text())["computed"]["ssb"]
    assert (current / "config.json").read_text(encoding="utf-8") == json.dumps(expected, indent=2)
    # existing with another schema, or none: refused before anything is written
    for number, earlier in enumerate(({"computed": {"bf_radius": 55.0}}, {"schema_version": 1, "computed": {}},
                                      {"schema": "quantem.screen-acquisition/1", "computed": {}})):
        folder = tmp_path / f"earlier-{number}"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(earlier), encoding="utf-8")
        before = (folder / "config.json").read_bytes()
        with pytest.raises(ValueError, match="quantem.screen-acquisition/2") as refused:
            write_fit_metadata(folder, settings, result, (0.6, 0.6))
        assert str(folder / "config.json") in str(refused.value) and repr(earlier.get("schema")) in str(refused.value)
        assert (folder / "config.json").read_bytes() == before and sorted(path.name for path in folder.iterdir()) == ["config.json"]


def _saved_joint_fit(tmp_path, monkeypatch):
    from quantem.gpu import DirectPtychoResult

    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    session = _session(monkeypatch, source, _Backend())
    result = DirectPtychoResult(
        object_wave=np.ones((2, 2), dtype=np.complex64), backend="mps",
        aberrations={"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": 3.0},
        sample_tilt_mrad=(1.0, 2.0), depth_spread_nm=10.0,
        source_path=str(source), scan_detector_rotation_deg=12.0,
    )
    paths = (tmp_path / "fit.npz", tmp_path / "fit.json")
    session._save_result(result, paths=paths, signature=session._result_signature("find_aberrations", {}))
    return session, result, paths


@pytest.mark.parametrize("change", ["source", "counts", "sampling", "detector", "shape", "signature"])
def test_parameter_slices_reject_mismatched_saved_fit_without_writes(tmp_path, monkeypatch, change):
    session, result, paths = _saved_joint_fit(tmp_path, monkeypatch)
    before = [path.read_bytes() for path in paths]
    if change == "source":
        session.source_path = str(tmp_path / "other.bin")
    elif change == "counts":
        (tmp_path / "scan.bin").write_bytes(b"changed detector counts")
    elif change == "sampling":
        session.scan_sampling_A = (0.4, 0.3)
    elif change == "detector":
        session.scan_detector_rotation_flipped = True
    elif change == "shape":
        session._data = np.zeros((4, 2, 3, 3), dtype=np.uint16)
    else:
        result.metadata.pop("signature")
    monkeypatch.setattr(session, "_backend", lambda: pytest.fail("Mismatched input must not evaluate the objective"))
    with pytest.raises(ValueError, match="session"):
        session.measure_parameters(result)
    assert result.parameter_slices is None
    assert [path.read_bytes() for path in paths] == before


def test_parameter_slices_require_live_ownership_for_saved_array_fit(tmp_path, monkeypatch):
    session, result, paths = _saved_joint_fit(tmp_path, monkeypatch)
    session.source_kind = "array"
    session._save_result(result, paths=paths, signature=session._result_signature("find_aberrations", {}))
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError, match="Array fits"):
        session.measure_parameters(result)
    assert [path.read_bytes() for path in paths] == before
    result.saved_path = None
    result.metadata = {}
    with pytest.raises(ValueError, match="Array fits"):
        session.measure_parameters(result)


def test_parameter_slices_preserve_rotation_and_compatible_saved_record(tmp_path, monkeypatch):
    from quantem.gpu import DirectPtychoResult
    from quantem.gpu.direct_ptycho import parameters
    from quantem.gpu.direct_ptycho.persistence import load_result

    session, result, paths = _saved_joint_fit(tmp_path, monkeypatch)
    signature = result.metadata["signature"]
    session.scan_detector_rotation_deg = 24.0
    measured = {"elapsed_s": 0.5, "slices": {}}

    def measure(*args, **kwargs):
        assert session.scan_detector_rotation_deg == result.scan_detector_rotation_deg
        return measured

    monkeypatch.setattr(parameters, "measure_slices", measure)
    assert session.measure_parameters(result) == measured
    assert session.scan_detector_rotation_deg == 24.0
    metadata = json.loads(paths[1].read_text())
    assert metadata["signature"] == signature
    assert metadata["parameter_slices"] == measured
    assert "parameter_slices" not in metadata["result"]
    # The existing scientific payload needs no new constructor keyword.
    assert DirectPtychoResult(object_wave=result.object_wave, **metadata["result"]).parameter_slices is None
    restored = load_result(paths=paths, signature=signature, backend="mps")
    assert restored.parameter_slices == measured
    assert session.measure_parameters(restored) == measured

    unsaved = DirectPtychoResult(
        object_wave=result.object_wave, backend="mps", aberrations=result.aberrations,
        sample_tilt_mrad=result.sample_tilt_mrad, depth_spread_nm=result.depth_spread_nm,
        scan_detector_rotation_deg=result.scan_detector_rotation_deg,
    )
    assert session.measure_parameters(unsaved) == measured
    unsaved.source_path = str(tmp_path / "other.bin")
    with pytest.raises(ValueError, match="original SSB source"):
        session.measure_parameters(unsaved)


@pytest.mark.parametrize("tilt", [True, False])
def test_a_retired_refinement_reuses_the_automatic_result(tmp_path, monkeypatch, capsys, tilt):
    """Every fit uses the automatic refinement: an explicit other value is announced and shares its cache."""
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    saved = tmp_path / "result"
    backend = _Backend()
    joint_calls = []

    def fit_sample(**settings):
        joint_calls.append(settings)
        return {"C10": -101.0, "C12": 4.0, "phi12": 0.25,
                "tilt_row_mrad": 3.0, "tilt_col_mrad": -4.0,
                "depth_spread_A": 150.0, "fit": 9.0, "gain": 1.5, "band_inv_A": [0.2, 0.9], "trial_records": []}

    backend.fit_sample = fit_sample
    settings = dict(tilt=tilt, save_dir=saved, check_rotation=False, verbose=False)
    first = _session(monkeypatch, source, backend).find_aberrations(search="primary", **settings)
    assert first.refinement == "newton"
    assert first.num_trials == 200
    capsys.readouterr()
    for method in ("newton", "nelder-mead", "bfgs"):
        explicit = _session(monkeypatch, source, backend).find_aberrations(
            search="primary", refinement=method, **settings)
        assert explicit.reused and explicit.refinement == "newton"
        assert explicit.metadata["signature"]["settings"]["refinement"] == "newton"
        np.testing.assert_array_equal(explicit.object_wave, first.object_wave)
        assert f"refinement={method!r} is retired" in capsys.readouterr().out
    if tilt:
        assert len(joint_calls) == 1
        assert joint_calls[0]["polish"] == "newton" and joint_calls[0]["polish_starts"] == 3
    else:
        assert backend.fit_calls == 1
    unrefined = _session(monkeypatch, source, backend).find_aberrations(search="primary",
        refinement=None, **settings,
    )
    assert not unrefined.reused
    assert unrefined.refinement is None
    if tilt:
        assert joint_calls[-1]["polish_starts"] == 0


def test_fixed_optics_cache_tracks_calibration_and_retains_sample_agreement(tmp_path, monkeypatch):
    """A regional fixed-optics result reloads only under the same calibration."""
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    backend = _Backend()
    calls = []

    def fit_sample(**settings):
        calls.append(settings)
        return {**settings["fixed_aberrations"], "tilt_row_mrad": 3.0,
                "tilt_col_mrad": -4.0, "depth_spread_A": 150.0, "fit": 9.0,
                "gain": None, "band_inv_A": [0.2, 0.9], "trial_records": []}

    backend.fit_sample = fit_sample
    optics = {"C10_nm": -10.1, "C12_nm": .4, "phi12_deg": 15.0}
    settings = dict(tilt=True, fixed_aberrations=optics, save_dir=tmp_path / "result",
                    check_rotation=False, compare_standard=False, verbose=False)
    first = _session(monkeypatch, source, backend).find_aberrations(search="primary", **settings)
    second = _session(monkeypatch, source, backend).find_aberrations(search="primary", **settings)
    assert first.sample_agreement == second.sample_agreement == 0.9
    assert second.reused and len(calls) == 1
    assert calls[0]["polish"] == "newton"
    assert calls[0]["fixed_aberrations"]["C10"] == -101.0
    settings["fixed_aberrations"] = {**optics, "C10_nm": -9.0}
    changed = _session(monkeypatch, source, backend).find_aberrations(search="primary", **settings)
    assert not changed.reused and len(calls) == 2


def test_save_dir_is_a_results_directory_and_refuses_a_file_name(tmp_path):
    """save_dir names a folder; a path ending in .npz or .json is a file, not a folder of that name."""
    from quantem.gpu.direct_ptycho.persistence import result_paths

    assert result_paths(tmp_path / "results", "reconstruct") == (
        (tmp_path / "results" / "ssb-reconstruct.npz").resolve(),
        (tmp_path / "results" / "ssb-reconstruct.json").resolve(),
    )
    assert result_paths(tmp_path / "scan.v2", "find_aberrations")[0].name == "ssb-find_aberrations.npz"
    for name in ("fit.npz", "fit.JSON"):
        with pytest.raises(ValueError, match="save_dir names a results directory.*save_dir='fit'"):
            result_paths(tmp_path / name, "find_aberrations")


def test_a_result_record_of_an_earlier_schema_is_refused_with_one_line(tmp_path, capsys):
    """quantem.ssb-result/9 named its source by absolute path; /10 reads only its own layout and says so."""
    from quantem.gpu.direct_ptycho.persistence import RESULT_SCHEMA, load_result, result_paths

    assert RESULT_SCHEMA == "quantem.ssb-result/10"
    paths = result_paths(tmp_path / "fit", "find_aberrations")
    paths[0].parent.mkdir()
    np.savez(paths[0], object_wave=np.ones((2, 2), np.complex64))
    paths[1].write_text(json.dumps({"schema": "quantem.ssb-result/9", "signature": {}, "result": {}}))
    assert load_result(paths=paths, signature={}, backend="mps") is None
    out = capsys.readouterr().out
    assert "'quantem.ssb-result/9' record" in out and "'quantem.ssb-result/10' only" in out


def _absolute_paths(value, found=None):
    """Every string in a JSON value that is an absolute path or names the home directory."""
    found = [] if found is None else found
    if isinstance(value, dict):
        for item in value.values():
            _absolute_paths(item, found)
    elif isinstance(value, list):
        for item in value:
            _absolute_paths(item, found)
    elif isinstance(value, str) and (value.startswith("/") or str(Path.home()) in value or "\\" in value):
        found.append(value)
    return found


def test_saved_result_is_reused_after_its_folder_moves_and_holds_no_absolute_path(tmp_path, monkeypatch):
    """The record names its source relative to the results folder and matches on content, so a moved folder reuses it.

    A different file of the same name (another size or header) is a different acquisition and is fitted again.
    """
    import shutil

    from quantem.gpu.direct_ptycho.persistence import acquisition_identity

    folder = tmp_path / "session"
    source = folder / "data" / "scan.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"detector counts")
    backend = _Backend()
    first = _session(monkeypatch, source, backend).find_aberrations(search="primary", tilt=False, save_dir=folder, verbose=False)
    record = json.loads((folder / "ssb-find_aberrations.json").read_text(encoding="utf-8"))
    assert record["input"]["source"] == {"file_name": "scan.bin", "relative_path": "data/scan.bin"}
    assert _absolute_paths(record) == []
    identity = record["signature"]["source"]["detector"]
    assert identity == acquisition_identity(source)
    assert set(identity) == {"schema", "file_name", "size", "header_sha256", "header_bytes", "linked_files",
                             "scan_shape", "detector_shape", "hot_pixel_correction"}

    moved = tmp_path / "elsewhere" / "session"
    moved.parent.mkdir()
    shutil.move(folder, moved)
    again = _session(monkeypatch, moved / "data" / "scan.bin", backend).find_aberrations(search="primary",
        tilt=False, save_dir=moved, verbose=False)
    assert again.reused and backend.fit_calls == 1
    np.testing.assert_array_equal(again.object_wave, first.object_wave)

    (moved / "data" / "scan.bin").write_bytes(b"other detector counts")
    other = _session(monkeypatch, moved / "data" / "scan.bin", backend).find_aberrations(search="primary",
        tilt=False, save_dir=moved, verbose=False)
    assert not other.reused and backend.fit_calls == 2
