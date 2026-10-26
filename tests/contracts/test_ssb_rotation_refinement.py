"""find_aberrations(refine_rotation=True): the 1-D search over an estimated rotation, its record and its persistence."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from quantem.gpu.direct_ptycho.rotation_refinement import refine_rotation, refinement_summary


def _agreement(true_deg, width_deg=0.45):
    """A peak like a real fit's: the agreement falls 0.1% at ``width_deg`` from the true angle."""
    def value(rotation_deg):
        offset = (rotation_deg - true_deg + 180.0) % 360.0 - 180.0
        return 1e14 * (1.0 - 1e-3 * (offset / width_deg) ** 2)
    return value


class _Fit:
    """The two fields the search reads, and the fit each polish started from."""

    def __init__(self, rotation_deg, agreement, *, loss=0.1, start=None):
        self.scan_detector_rotation_deg = rotation_deg % 360.0
        self.sample_agreement = agreement
        self.loss = loss
        self.start = start


def _search(estimate_deg, true_deg, **options):
    peak = _agreement(true_deg)
    polished = []

    def polish(rotation_deg, start):
        polished.append((rotation_deg, start.scan_detector_rotation_deg))
        return _Fit(rotation_deg, peak(rotation_deg), start=start)

    refined, record = refine_rotation(_Fit(estimate_deg, peak(estimate_deg)), polish, **options)
    return refined, record, polished


def _angle_between(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize("estimate_deg, true_deg", [(345.83, 347.9), (12.0, 10.3), (359.4, 1.3), (90.0, 90.0)])
def test_the_search_recovers_a_rotation_offset_within_a_tenth_of_a_degree(estimate_deg, true_deg):
    refined, record, polished = _search(estimate_deg, true_deg)
    assert _angle_between(refined.scan_detector_rotation_deg, true_deg) < 0.1
    assert record["estimate_deg"] == pytest.approx(estimate_deg % 360.0)
    assert record["refined_deg"] == refined.scan_detector_rotation_deg
    assert record["refined_agreement"] >= record["estimate_agreement"]
    assert record["evaluations"] == len(polished) <= 12
    # every angle polishes from the nearest angle already evaluated, so the search follows one basin of the fit
    evaluated = [estimate_deg]
    for rotation, start in polished:
        nearest = min(evaluated, key=lambda known: abs(known - rotation))
        assert start == pytest.approx(nearest % 360.0)
        evaluated.append(rotation)


def test_the_bracket_walks_to_a_far_optimum_and_stops_at_ten_degrees():
    refined, record, _ = _search(100.0, 106.4)
    assert _angle_between(refined.scan_detector_rotation_deg, 106.4) < 0.1
    refined, record, polished = _search(100.0, 115.0)
    assert refined.scan_detector_rotation_deg == pytest.approx(110.0, abs=0.06)
    assert max(rotation for rotation, _ in polished) <= 110.0 + 1e-9


def test_a_phase_variance_fit_is_refined_on_its_loss():
    """A fit without a model agreement (the phase-variance objective) has the lowest loss at the refined angle."""
    def polish(rotation_deg, start):
        return _Fit(rotation_deg, None, loss=0.1 + 1e-3 * (rotation_deg - 31.2) ** 2)

    refined, record = refine_rotation(_Fit(30.0, None, loss=0.1 + 1e-3 * 1.2**2), polish)
    assert refined.scan_detector_rotation_deg == pytest.approx(31.2, abs=0.1)
    assert record["refined_loss"] < record["estimate_loss"]
    assert "phase variance -" in refinement_summary(record)


def test_the_summary_states_both_angles_the_change_and_the_agreement_gain():
    record = {"estimate_deg": 345.83, "refined_deg": 346.85, "estimate_agreement": 2.0, "refined_agreement": 2.011,
              "estimate_loss": 0.1, "refined_loss": 0.1, "evaluations": 9, "seconds": 8.4}
    assert refinement_summary(record) == ("refined rotation 345.83° → 346.85° (+1.02°, agreement +0.55%; "
                                          "9 polishes, 8.4 s)")


# ---- the workflow: an estimated rotation is refined, a given one never; the record is saved and reused


class _Backend:
    """Standard-fit backend double: the agreement peaks at 5 degrees and each fit reports the session's rotation."""

    def __init__(self, true_deg=5.0):
        self.angle_deg = 3.0  # the sessions below are built at 3 degrees; set_rotation moves it
        self.peak = _agreement(true_deg)
        self.fits = []

    num_bright_field_pixels = 17

    def cache_rotation(self, angle_rad):
        self.angle_deg = math.degrees(angle_rad)

    def fit(self, **settings):
        from quantem.gpu import SSBResult

        self.fits.append((self.angle_deg, settings["trials"]))
        return SSBResult(object_wave=np.ones((2, 2), np.complex64), backend="mps",
                         aberrations={"C10": -101.0 + self.angle_deg, "C12": 4.0, "phi12": 0.25},
                         scan_detector_rotation_deg=self.angle_deg, loss=0.125, elapsed_s=1.0,
                         num_trials=settings["trials"], refinement=settings["refinement"],
                         sample_agreement=self.peak(self.angle_deg),
                         trial_records=[{"params": {"C10_nm": -10.1, "C12_nm": 0.4}, "loss": 0.125}] * settings["trials"])

    def preview(self, aberrations, *, compute_loss=True, **kwargs):
        return np.full((2, 2), 0.2, np.float32), 0.25 if compute_loss else None

    def preview_upsampled(self, aberrations, **kwargs):
        return self.preview(aberrations, **kwargs)


def _session(monkeypatch, tmp_path, backend, *, estimated):
    from quantem.gpu.direct_ptycho import workflow

    source = tmp_path / "scan.bin"
    if not source.exists():
        source.write_bytes(b"detector counts")
    monkeypatch.setattr(workflow, "resolve", lambda _backend: "mps")
    monkeypatch.setattr(workflow.SSB, "_backend", lambda _self: backend)
    session = workflow.SSB(np.zeros((2, 2, 3, 3), np.uint16), backend="mps", voltage_kV=300.0, semiangle_mrad=30.0,
                           scan_sampling_A=(0.2, 0.2), detector_sampling_mrad=(0.01, 0.01), source_path=str(source),
                           scan_detector_rotation_deg=3.0)
    session.source_kind = "detector"
    if estimated:
        session.scan_detector_rotation_source = "estimated"
    return session


def test_an_estimated_rotation_is_refined_recorded_reported_and_reused(tmp_path, monkeypatch, capsys):
    backend = _Backend()
    session = _session(monkeypatch, tmp_path, backend, estimated=True)
    fit = session.find_aberrations(search="primary", tilt=False, num_trials=4, check_rotation=False, refine_rotation=True,
                                   save_dir=tmp_path / "result")
    assert backend.fits[0] == (3.0, 4) and all(trials == 0 for _, trials in backend.fits[1:])
    assert 3 <= len(backend.fits) - 1 <= 12
    assert fit.rotation_refined and fit.rotation_estimate_deg == 3.0
    assert fit.scan_detector_rotation_deg == pytest.approx(5.0, abs=0.1)
    assert session.scan_detector_rotation_deg == fit.scan_detector_rotation_deg
    assert fit.aberrations["C10_nm"] == pytest.approx((-101.0 + fit.scan_detector_rotation_deg) / 10.0)
    record = fit.rotation_refinement
    assert record["estimate_agreement"] == backend.peak(3.0) < record["refined_agreement"] == fit.sample_agreement
    assert record["evaluations"] == len(backend.fits) - 1 and len(record["evaluated_deg"]) == record["evaluations"]
    # the search trials of the fit stay; a polish adds none
    assert fit.num_trials == 4 and len(fit.trial_records) == 4
    assert f"SSB: refined rotation 3.00° → {fit.scan_detector_rotation_deg:.2f}° (+" in capsys.readouterr().out
    report = fit.report().iloc[:, 0]
    assert report["Estimated rotation (deg)"] == 3.0
    assert report["SSB-selected rotation (deg)"] == pytest.approx(5.0, abs=0.1)
    assert "Refined from   3.00° (estimate)" in repr(fit)
    # reconstruct() applies the refined fit and carries its record
    applied = session.reconstruct(fit)
    assert applied.rotation_refined and applied.rotation_estimate_deg == 3.0

    calls = len(backend.fits)
    again = _session(monkeypatch, tmp_path, backend, estimated=True).find_aberrations(search="primary",
        tilt=False, num_trials=4, check_rotation=False, refine_rotation=True, save_dir=tmp_path / "result",
        verbose=False)
    assert again.reused and len(backend.fits) == calls
    assert again.metadata["signature"]["settings"]["refine_rotation"] is True
    assert (again.rotation_refined, again.rotation_estimate_deg) == (True, 3.0)
    assert again.scan_detector_rotation_deg == fit.scan_detector_rotation_deg
    assert again.rotation_refinement == record
    # without refine_rotation the saved refined fit is not this fit: one search again, no polishes
    unrefined = _Backend()
    plain = _session(monkeypatch, tmp_path, unrefined, estimated=True).find_aberrations(search="primary",
        tilt=False, num_trials=4, check_rotation=False, save_dir=tmp_path / "result", verbose=False)
    assert not plain.reused and unrefined.fits == [(3.0, 4)]
    assert "refine_rotation" not in plain.metadata["signature"]["settings"]


def test_a_given_rotation_is_never_refined(tmp_path, monkeypatch, capsys):
    backend = _Backend()
    fit = _session(monkeypatch, tmp_path, backend, estimated=False).find_aberrations(search="primary",
        tilt=False, num_trials=4, refine_rotation=True)
    assert backend.fits == [(3.0, 4)]
    assert fit.scan_detector_rotation_deg == 3.0
    assert (fit.rotation_refined, fit.rotation_estimate_deg, fit.rotation_refinement) == (False, None, None)
    assert "SSB: rotation 3.00° is given; refine_rotation leaves it as it is." in capsys.readouterr().out
    assert "Estimated rotation (deg)" not in fit.report().index


def test_the_default_fit_searches_once_and_records_no_refinement(tmp_path, monkeypatch):
    results = []
    for options in ({}, {"refine_rotation": False}):
        backend = _Backend()
        results.append(_session(monkeypatch, tmp_path, backend, estimated=True).find_aberrations(search="primary",
            tilt=False, num_trials=4, check_rotation=False, verbose=False, **options))
        assert backend.fits == [(3.0, 4)]
    for fit in results:
        assert (fit.rotation_refined, fit.rotation_estimate_deg, fit.rotation_refinement) == (False, None, None)
        assert fit.scan_detector_rotation_deg == 3.0
    assert dict(results[0]) == dict(results[1])


def test_refine_rotation_needs_local_refinement(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="refine_rotation polishes the fit"):
        _session(monkeypatch, tmp_path, _Backend(), estimated=True).find_aberrations(search="primary",
            tilt=False, refinement=None, refine_rotation=True, verbose=False)
