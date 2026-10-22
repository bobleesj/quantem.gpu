"""The readable search-budget keyword preserves existing notebook fits and caches."""

import numpy as np
import pytest

from tests.contracts.test_ssb_fit_objective import _Recording, _source
from tests.contracts.test_ssb_persistence import _session


@pytest.mark.parametrize("count", [0, 12, 200])
def test_trials_and_legacy_keyword_produce_the_same_fit(tmp_path, monkeypatch, count):
    backend = _Recording()
    source = _source(tmp_path)
    session = _session(monkeypatch, source, backend)
    options = dict(tilt=False, check_rotation=False, verbose=False)
    current = session.find_aberrations(trials=count, **options)
    legacy = _session(monkeypatch, source, backend).find_aberrations(num_trials=count, **options)
    assert backend.fits[-1] == backend.fits[-2]
    assert current.num_trials == legacy.num_trials == count
    np.testing.assert_array_equal(current.phase, legacy.phase)
    with pytest.raises(TypeError, match="not both"):
        session.find_aberrations(trials=count, num_trials=count, **options)


def test_trials_reuses_the_fit_saved_with_the_legacy_keyword(tmp_path, monkeypatch):
    backend = _Recording()
    source = _source(tmp_path)
    options = dict(tilt=False, check_rotation=False, verbose=False, save_dir=tmp_path / "fit")
    old = _session(monkeypatch, source, backend).find_aberrations(num_trials=200, **options)
    new = _session(monkeypatch, source, backend).find_aberrations(trials=200, **options)
    assert new.reused
    assert backend.fit_calls == 1
    np.testing.assert_array_equal(old.phase, new.phase)


def test_rotation_reviews_accept_the_same_trial_budget(monkeypatch):
    from types import SimpleNamespace
    from quantem.gpu import SSB
    from quantem.gpu.ssb import rotation_defocus

    budgets = []
    def record(*args, **kwargs):
        budgets.append(kwargs["num_trials"])
        return SimpleNamespace(rotation_branches="branches")

    session = SimpleNamespace(_device=None, scan_detector_rotation_source="recorded", _find_aberrations=record)
    monkeypatch.setattr(rotation_defocus, "compare_rotation_defocus", record)
    for method in (SSB.compare_rotation_defocus, SSB.compare_rotation_branches):
        for options in ({}, {"trials": 12}, {"num_trials": 12}):
            method(session, **options)
        with pytest.raises(TypeError, match="not both"):
            method(session, trials=200, num_trials=200)
    assert budgets == [200, 12, 12, 200, 12, 12]
