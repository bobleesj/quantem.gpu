"""The SSB fit of ``fit_aberrations`` maximizes the model agreement: routed, checked, saved, reported.

``fit_aberrations()`` fits the sample tilt and depth spread with the aberrations, ``tilt=False`` holds them at 0.
``objective`` is retired: ``"agreement"`` prints one line, ``"phase_variance"`` raises. The backend double records what
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
            "depth_spread_A": 150.0, "fit": 9.0, "gain": 1.5, "band_inv_A": [0.2, 0.9], "trial_records": []}


def _source(tmp_path):
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")
    return source


def test_standard_fit_is_the_agreement_refined_by_the_newton(tmp_path, monkeypatch):
    """tilt=False reaches the backend without an objective (there is one) and polishes with the Newton."""
    backend = _Recording()
    fit = _session(monkeypatch, _source(tmp_path), backend).fit_aberrations(search="primary", tilt=False, verbose=False)
    assert "objective" not in backend.fits[-1]
    assert backend.fits[-1]["refinement"] == fit.refinement == "newton"
    assert fit.objective == "agreement"
    assert fit.report().loc["objective"].iloc[0] == "agreement"
    assert "Objective      agreement" in repr(fit)


def test_standard_fit_is_the_tilt_corrected_agreement(tmp_path, monkeypatch):
    backend = _Recording()
    backend.fit_sample = _fit_sample
    session = _session(monkeypatch, _source(tmp_path), backend)
    fit = session.fit_aberrations(search="primary", check_rotation=False, verbose=False)
    assert fit.objective == "agreement" and fit.sample_tilt_mrad == (3.0, -4.0)
    assert not backend.fits  # the joint fit, not the standard one
    # a reconstruction of exactly that fit says which objective found its aberrations; changed ones do not
    assert session.reconstruct().objective == "agreement"
    assert session.reconstruct(aberrations={"C10_nm": 1.0}).objective is None


def test_the_objective_keyword_is_retired(tmp_path, monkeypatch, capsys):
    """objective='agreement' prints one line and fits; 'phase_variance' raises before any fit (a silent agreement fit
    would land elsewhere), as does any other value."""
    backend = _Recording()
    session = _session(monkeypatch, _source(tmp_path), backend)
    for call in (lambda: session.fit_aberrations(search="primary", objective="phase_variance", verbose=False),
                 lambda: session.fit_aberrations(search="primary", tilt=False, objective="phase_variance",
                                                  verbose=False),
                 lambda: session.scan_rotations(report=False, objective="phase_variance", verbose=False)):
        with pytest.raises(ValueError, match="objective='phase_variance' is retired"):
            call()
    with pytest.raises(ValueError, match="objective is retired"):
        session.fit_aberrations(search="primary", tilt=False, objective="variance", verbose=False)
    assert not backend.fits
    fit = session.fit_aberrations(search="primary", tilt=False, objective="agreement", verbose=False)
    assert fit.objective == "agreement" and len(backend.fits) == 1
    assert capsys.readouterr().out.splitlines() == ["objective is retired; every fit maximizes the model agreement."]


def test_an_explicit_agreement_objective_reuses_the_saved_fit(tmp_path, monkeypatch):
    backend = _Recording()
    source, saved = _source(tmp_path), tmp_path / "result"
    settings = dict(tilt=False, save_dir=saved, verbose=False)
    first = _session(monkeypatch, source, backend).fit_aberrations(search="primary", **settings)
    again = _session(monkeypatch, source, backend).fit_aberrations(search="primary", objective="agreement", **settings)
    assert first.metadata["signature"]["settings"]["objective"] == "agreement"
    assert again.reused and again.objective == "agreement"
    assert len(backend.fits) == 1


def test_implicit_reconstruction_uses_default_fifth_order_fit(tmp_path, monkeypatch):
    """An unfitted session remains reconstructable on CUDA and MPS."""
    backend = _Recording()
    backend.fit_sample = _fit_sample
    session = _session(monkeypatch, _source(tmp_path), backend)
    from quantem.gpu.direct_ptycho import fifth_order_fit

    def fitted(ssb, **options):
        assert options["search"] == "standard" and options["trials"] == 200
        fit = ssb._find_aberrations("off", compute_loss=False, verbose=False)
        fit.aberrations["C50_nm"] = 100.
        fit.objective = "broad_fifth_agreement"
        ssb._aberration_search = ssb._accept_result(fit)
        return fit

    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", fitted)
    result = session.reconstruct(verbose=False)
    assert result.objective == "broad_fifth_agreement"
    assert result.aberrations["C50_nm"] == 100.
    assert result.sample_tilt_mrad == (3.0, -4.0)
