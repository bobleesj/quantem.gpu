"""Search selection must preserve standard behavior and reject silent substitution."""
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu.ssb._broad_newton import Tracker, newton
from quantem.gpu.ssb.workflow import SSB


def call(session, **options):
    return inspect.unwrap(SSB.find_aberrations)(session, **options)


def test_standard_preserves_budget_seed_and_options():
    session = SimpleNamespace(_find_aberrations=Mock(return_value="fit"))
    assert call(session, search="standard") == "fit"
    options = session._find_aberrations.call_args.kwargs
    assert options["num_trials"] == 200
    assert options["seed"] == 42
    assert options["tilt"] is True


def test_broad_default_dispatch(monkeypatch):
    from quantem.gpu.ssb import broad_search
    run = Mock(return_value="broad fit")
    monkeypatch.setattr(broad_search, "find_broad", run)
    assert call(object()) == "broad fit"
    assert run.call_args.kwargs["trials"] == 400
    assert run.call_args.kwargs["seed"] == 17
    call(object(), search="broad", trials=200, seed=42)
    assert run.call_args.kwargs["seed"] == 42
    assert run.call_args.kwargs["trials"] == 200


@pytest.mark.parametrize("options", [
    {"search": "coarse"}, {"search": "broad", "tilt": False},
    {"search": "broad", "search_ranges": {"C10_nm": (-5, 5)}},
    {"search": "broad", "refinement": "invalid"},
])
def test_invalid_options_fail_before_computation(options):
    with pytest.raises(ValueError):
        call(object(), **options)


def test_mps_does_not_substitute_standard():
    with pytest.raises(NotImplementedError, match="search='standard'"):
        call(SimpleNamespace(backend="mps"))


def test_bounded_newton_quadratic():
    target = np.linspace(-0.3, 0.3, 28)
    start = np.zeros(28)
    bounds = np.tile([-2., 2.], (28, 1))
    tracker = Tracker(lambda points: 100 - np.sum((points - target)**2, axis=1), start, bounds, 3000)
    newton(tracker, start, bounds)
    np.testing.assert_allclose(tracker.best_z, target, atol=1e-5, rtol=0)


@pytest.mark.parametrize("refinement", ["newton", "nelder-mead"])
def test_broad_preserves_refinement_choice(monkeypatch, refinement):
    from quantem.gpu.ssb import broad_search
    run = Mock(return_value="fit")
    monkeypatch.setattr(broad_search, "find_broad", run)
    assert call(object(), search="broad", refinement=refinement) == "fit"
    assert run.call_args.kwargs["refinement"] == refinement
    assert run.call_args.kwargs["trials"] == 400


def test_nelder_mead_can_move_initially_zero_higher_terms():
    from quantem.gpu.ssb._broad_nelder_mead import nelder_mead
    target = np.array([.1, -.1, .2, -.2, .1, .5, .15, -.15])
    start = np.zeros(8)
    bounds = np.tile([-2., 2.], (8, 1))
    tracker = Tracker(lambda points: 1 - np.sum((points - target)**2, axis=1), start, bounds, 3000)
    tracker(start)
    nelder_mead(tracker, start, bounds)
    np.testing.assert_allclose(tracker.best_z, target, atol=.002, rtol=0)
    assert tracker.nfev <= 3000



@pytest.mark.parametrize("refinement, cap", [("newton", 5000), ("nelder-mead", 3000)])
def test_broad_records_refinement_budget(monkeypatch, tmp_path, refinement, cap):
    """The actual solver, persisted result and cache signature share one budget."""
    from quantem.gpu.ssb import broad_search
    from quantem.gpu.ssb.cuda import broad_batch, broad_objective

    objective = SimpleNamespace(evaluate=lambda points, order: np.ones(len(np.atleast_2d(points))))
    monkeypatch.setattr(broad_batch, "Objective", lambda engine: objective)
    monkeypatch.setattr(broad_objective, "vector", lambda fit, order: np.zeros(28))
    monkeypatch.setattr(broad_objective, "public_params", lambda *args: ({}, (0., 0.), 1.))
    study = SimpleNamespace(
        best_params={f"p{i}": (1. if i == 5 else 0.) for i in range(28)},
        trials=[], enqueue_trial=lambda value: None, optimize=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(broad_search.optuna, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(broad_search, "make_independent_sampler", lambda **kwargs: None)
    def refine(tracker, start, bounds):
        assert tracker.cap == cap
        return 0, "test stop"
    monkeypatch.setattr(broad_search, refinement.replace("-", "_"), refine)
    monkeypatch.setattr(broad_search, "result_paths", lambda *args: object())
    monkeypatch.setattr(broad_search, "load_result", lambda **kwargs: None)
    signature = Mock(return_value="signature")
    session = SimpleNamespace(
        backend="cuda", source_kind="hdf5", find_aberrations=Mock(return_value=SimpleNamespace()),
        _backend=lambda: SimpleNamespace(_get_accelerator=lambda: object()),
        _result_signature=signature, _accept_result=lambda result: result,
        reconstruct=lambda result, **kwargs: result, _save_result=lambda result, **kwargs: result,
    )
    result = broad_search.find_broad(
        session, trials=400, seed=17, save_dir=tmp_path,
        force=False, verbose=False, primary_options={}, refinement=refinement,
    )
    assert signature.call_args.args[1]["newton_cap"] == cap
    assert result.optimizer["refinement_cap"] == cap
    assert result.optimizer["stop"] == "test stop"
    assert result.optimizer["trials"] == 400
    assert session.find_aberrations.call_args.kwargs["trials"] == 200
    if refinement == "newton":
        assert result.optimizer["newton_cap"] == 5000
    else:
        assert "newton_cap" not in result.optimizer


def test_trial_budget_comparison_keeps_start_and_saved_results_independent(tmp_path):
    """A budget comparison must not seed its next fit from the previous winner."""
    from contextlib import contextmanager

    options = []
    closed = []
    @contextmanager
    def select_region(region):
        assert region == (0, 512, 0, 512)
        child = SimpleNamespace(aberrations={"C10_nm": 0.})
        def run(**kwargs):
            assert child.aberrations == {"C10_nm": 0.}
            child.aberrations["C10_nm"] = 99.
            options.append(kwargs)
            return SimpleNamespace(aberrations=dict(child.aberrations), num_trials=kwargs["trials"])
        child.find_aberrations = run
        try:
            yield child
        finally:
            closed.append(True)
    parent = SimpleNamespace(input_scan_shape=(512, 512), select_region=select_region,
                             aberrations={"C10_nm": 0.})
    fits = call(parent, trials=(200, 400, 600), seed=17, refinement="nelder-mead",
                save_dir=tmp_path, verbose=False)
    assert list(fits) == [200, 400, 600]
    assert parent.aberrations == {"C10_nm": 0.}
    assert len(closed) == 3
    assert [item["save_dir"] for item in options] == [tmp_path / f"trials-{n}" for n in fits]
    assert all(item["seed"] == 17 and item["refinement"] == "nelder-mead" for item in options)
