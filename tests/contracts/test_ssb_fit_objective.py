"""The three SSB fit methods of ``find_aberrations``: one ``objective`` vocabulary, routed, checked, saved, reported.

``find_aberrations()`` is the default model agreement with the sample tilt fitted, ``tilt=False`` holds the tilt at 0,
and ``tilt=False, objective="phase_variance"`` is the older phase-variance search. The backend double records what
reaches it; no GPU is used.
"""

import pytest

from tests.contracts.test_ssb_persistence import _Backend, _session


class _Recording(_Backend):
    """The persistence double, keeping the settings of every standard fit it ran."""

    def __init__(self):
        super().__init__()
        self.fits = []

    def fit(self, **settings):
        self.fits.append(settings)
        return super().fit(**settings)


def _fit_sample(**settings):
    return {"C10": -101.0, "C12": 4.0, "phi12": 0.25, "tilt_row_mrad": 3.0, "tilt_col_mrad": -4.0,
            "depth_spread_A": 150.0, "fit": 9.0, "gain": 1.5, "trial_records": []}


def _source(tmp_path):
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    return source


@pytest.mark.parametrize("objective, refinement", [("agreement", "newton"), ("phase_variance", "nelder-mead")])
def test_standard_fit_passes_the_objective_and_refines_by_it(tmp_path, monkeypatch, objective, refinement):
    """tilt=False reaches the backend with its objective; "auto" is Newton for the agreement, Nelder-Mead otherwise."""
    backend = _Recording()
    options = {} if objective == "agreement" else {"objective": objective}
    fit = _session(monkeypatch, _source(tmp_path), backend).find_aberrations(tilt=False, verbose=False, **options)
    assert backend.fits[-1]["objective"] == objective
    assert backend.fits[-1]["refinement"] == fit.refinement == refinement
    assert fit.objective == objective
    assert fit.report()["objective"].iloc[0] == objective
    assert f"Objective      {objective}" in repr(fit)


def test_default_fit_is_the_tilt_corrected_agreement(tmp_path, monkeypatch):
    backend = _Recording()
    backend.fit_sample = _fit_sample
    backend.preview_sample = lambda coefs, sample, **kwargs: backend.preview(coefs, **kwargs)
    session = _session(monkeypatch, _source(tmp_path), backend)
    fit = session.find_aberrations(check_rotation=False, verbose=False)
    assert fit.objective == "agreement" and fit.sample_tilt_mrad == (3.0, -4.0)
    assert not backend.fits  # the joint fit, not the standard one
    # a reconstruction of exactly that fit says which objective found its aberrations; changed ones do not
    assert session.reconstruct().objective == "agreement"
    assert session.reconstruct(aberrations={"C10_nm": 1.0}).objective is None


def test_phase_variance_cannot_fit_the_tilt_and_refines_with_nelder_mead(tmp_path, monkeypatch):
    backend = _Recording()
    session = _session(monkeypatch, _source(tmp_path), backend)
    with pytest.raises(ValueError, match="cannot fit the sample tilt"):
        session.find_aberrations(objective="phase_variance", verbose=False)
    with pytest.raises(ValueError, match="cannot fit the sample tilt"):
        session.compare_rotation_branches(objective="phase_variance", verbose=False)
    with pytest.raises(ValueError, match="refines with 'nelder-mead'"):
        session.find_aberrations(tilt=False, objective="phase_variance", refinement="newton", verbose=False)
    with pytest.raises(ValueError, match="objective is 'agreement'"):
        session.find_aberrations(tilt=False, objective="variance", verbose=False)
    assert not backend.fits


def test_phase_variance_holds_fixed_aberrations_on_mps(tmp_path, monkeypatch):
    """fixed_aberrations means a held search value and a refinement lock in either standard fit on MPS.

    CUDA's standard fits, both objectives, are checked on the device in tests/hardware/cuda/test_ssb_standard_agreement.py
    (this double returns NumPy arrays, which the CUDA session's post-processing does not take).
    """
    backend = _Recording()
    _session(monkeypatch, _source(tmp_path), backend).find_aberrations(
        tilt=False, objective="phase_variance", fixed_aberrations={"C12_nm": 0.0}, verbose=False)
    assert backend.fits[-1]["search_ranges"]["C12_A"] == 0.0
    assert backend.fits[-1]["refine_lock"] == ["C12"]


def test_saved_fit_is_reused_only_for_its_objective(tmp_path, monkeypatch):
    """The objective is a saved setting: a phase-variance fit never answers an agreement call, and the reverse."""
    backend = _Recording()
    source, saved = _source(tmp_path), tmp_path / "result"
    settings = dict(tilt=False, save_dir=saved, verbose=False)
    first = _session(monkeypatch, source, backend).find_aberrations(objective="phase_variance", **settings)
    again = _session(monkeypatch, source, backend).find_aberrations(objective="phase_variance", **settings)
    other = _session(monkeypatch, source, backend).find_aberrations(**settings)
    assert first.metadata["signature"]["settings"]["objective"] == "phase_variance"
    assert again.reused and again.objective == "phase_variance"
    assert not other.reused and other.objective == "agreement"
    assert other.metadata["signature"]["settings"]["objective"] == "agreement"
    assert [fit["objective"] for fit in backend.fits] == ["phase_variance", "agreement"]
