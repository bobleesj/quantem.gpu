"""Search selection must preserve explicit model choices and reject silent substitution."""
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu.ssb._broad_newton import Tracker, newton
from quantem.gpu.ssb.workflow import SSB


def call(session, **options):
    return inspect.unwrap(SSB.find_aberrations)(session, **options)


def test_primary_preserves_budget_seed_and_options():
    session = SimpleNamespace(_find_aberrations=Mock(return_value="fit"))
    assert call(session, search="primary") == "fit"
    options = session._find_aberrations.call_args.kwargs
    assert options["num_trials"] == 200
    assert options["seed"] == 42
    assert options["tilt"] is True


def test_standard_default_and_explicit_broad_dispatch(monkeypatch):
    from quantem.gpu.ssb import fifth_order_fit
    run = Mock(return_value="broad fit")
    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", run)
    assert call(object()) == "broad fit"
    assert run.call_args.kwargs["trials"] == 200
    assert run.call_args.kwargs["seed"] == 42
    assert run.call_args.kwargs["search"] == "standard"
    call(object(), search="broad")
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


def test_mps_uses_same_default_model(monkeypatch):
    from quantem.gpu.ssb import fifth_order_fit
    run = Mock(return_value="fifth-order fit")
    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", run)
    session = SimpleNamespace(backend="mps")
    assert call(session) == "fifth-order fit"
    assert run.call_args.kwargs["search"] == "standard"
    assert run.call_args.kwargs["trials"] == 200


def test_bounded_newton_quadratic():
    target = np.linspace(-0.3, 0.3, 28)
    start = np.zeros(28)
    bounds = np.tile([-2., 2.], (28, 1))
    tracker = Tracker(lambda points: 100 - np.sum((points - target)**2, axis=1), start, bounds, 3000)
    newton(tracker, start, bounds)
    np.testing.assert_allclose(tracker.best_z, target, atol=1e-5, rtol=0)


@pytest.mark.parametrize("refinement", ["newton", "nelder-mead"])
def test_broad_preserves_refinement_choice(monkeypatch, refinement):
    from quantem.gpu.ssb import fifth_order_fit
    run = Mock(return_value="fit")
    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", run)
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
    from quantem.gpu.ssb import fifth_order_fit
    import sys

    # Exercise orchestration without importing a CUDA runtime on CPU CI.
    broad_batch = SimpleNamespace(Objective=None)
    broad_objective = SimpleNamespace(vector=None, public_params=None)
    monkeypatch.setitem(sys.modules, "quantem.gpu.ssb.cuda.broad_batch", broad_batch)
    monkeypatch.setitem(sys.modules, "quantem.gpu.ssb.fifth_order_parameters", broad_objective)

    objective = SimpleNamespace(evaluate=lambda points, order: np.ones(len(np.atleast_2d(points))))
    monkeypatch.setattr(broad_batch, "Objective", lambda engine: objective)
    monkeypatch.setattr(broad_objective, "vector", lambda fit, order: np.zeros(28))
    monkeypatch.setattr(broad_objective, "public_params", lambda *args: ({}, (0., 0.), 1.))
    study = SimpleNamespace(
        best_params={f"p{i}": (1. if i == 5 else 0.) for i in range(28)},
        trials=[], enqueue_trial=lambda value: None, optimize=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(fifth_order_fit.optuna, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(fifth_order_fit, "make_independent_sampler", lambda **kwargs: None)
    def refine(tracker, start, bounds):
        assert tracker.cap == cap
        return 0, "test stop"
    monkeypatch.setattr(fifth_order_fit, refinement.replace("-", "_"), refine)
    monkeypatch.setattr(fifth_order_fit, "result_paths", lambda *args: object())
    monkeypatch.setattr(fifth_order_fit, "load_result", lambda **kwargs: None)
    signature = Mock(return_value="signature")
    session = SimpleNamespace(
        backend="cuda", source_kind="hdf5", _find_aberrations=Mock(
            return_value=SimpleNamespace(rotation_branches=None, trial_records=[])),
        _backend=lambda: SimpleNamespace(_get_accelerator=lambda: object()),
        _result_signature=signature, _accept_result=lambda result: result,
        run=lambda result, **kwargs: result, _save_result=lambda result, **kwargs: result,
    )
    result = fifth_order_fit.fit_fifth_order(
        session, search="broad", trials=400, seed=17, save_dir=tmp_path,
        force=False, verbose=False, primary_options={"check_rotation": True, "refine_rotation": False}, refinement=refinement,
        higher_order_limit=2,
    )
    assert signature.call_args.args[1]["newton_cap"] == cap
    assert result.optimizer["refinement_cap"] == cap
    assert result.optimizer["stop"] == "test stop"
    assert result.optimizer["trials"] == 400
    assert session._find_aberrations.call_args.kwargs["num_trials"] == 200
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
    fits = call(parent, search="broad", trials=(200, 400, 600), seed=17, refinement="nelder-mead",
                save_dir=tmp_path, verbose=False)
    assert list(fits) == [200, 400, 600]
    assert parent.aberrations == {"C10_nm": 0.}
    assert len(closed) == 3
    assert [item["save_dir"] for item in options] == [tmp_path / f"trials-{n}" for n in fits]
    assert all(item["seed"] == 17 and item["refinement"] == "nelder-mead" for item in options)


@pytest.mark.parametrize("refinement", ["newton", "nelder-mead"])
def test_standard_uses_initial_fit_without_extra_tpe(monkeypatch, refinement):
    """Default initialization reaches the same joint solver without an extra study."""
    from quantem.gpu.ssb import fifth_order_fit
    import sys

    start = np.zeros(28)
    start[5] = 1.
    objective = SimpleNamespace(evaluate=lambda points, order: np.ones(len(np.atleast_2d(points))))
    monkeypatch.setitem(sys.modules, "quantem.gpu.ssb.cuda.broad_batch",
                        SimpleNamespace(Objective=lambda engine: objective))
    monkeypatch.setitem(sys.modules, "quantem.gpu.ssb.fifth_order_parameters", SimpleNamespace(
        vector=lambda fit, order: start.copy(),
        public_params=lambda *args: ({"C50_nm": 1.}, (1., -2.), 10.),
    ))
    monkeypatch.setattr(fifth_order_fit.optuna, "create_study",
                        lambda **kwargs: pytest.fail("Extra TPE search in default route"))
    def refine(tracker, initial, bounds):
        np.testing.assert_array_equal(initial, start)
        return 0, "small bounded step"
    monkeypatch.setattr(fifth_order_fit, refinement.replace("-", "_"), refine)
    primary = SimpleNamespace(rotation_branches="initial branches", trial_records=[{"loss": -1.}])
    session = SimpleNamespace(
        backend="cuda", _find_aberrations=Mock(return_value=primary),
        _backend=lambda: SimpleNamespace(_get_accelerator=lambda: object()),
        run=lambda result, **kwargs: result, _accept_result=lambda result: result,
    )
    fit = fifth_order_fit.fit_fifth_order(
        session, search="standard", trials=200, seed=42, save_dir=None,
        force=False, verbose=False, refinement=refinement, higher_order_limit=2,
        primary_options={"check_rotation": False, "refine_rotation": False},
    )
    assert session._find_aberrations.call_args.args == ("off",)
    assert session._find_aberrations.call_args.kwargs["num_trials"] == 200
    assert session._find_aberrations.call_args.kwargs["refinement"] == "newton"
    assert fit.optimizer["primary_trials"] == 200
    assert fit.optimizer["exploration_trials"] == 0
    assert fit.optimizer["search"] == "standard"
    assert fit.optimizer["coarse_scores"] == []
    assert fit.rotation_branches == "initial branches"
    assert fit.sample_tilt_mrad == (1., -2.) and fit.depth_spread_nm == 10.
    assert fit.refinement == refinement
    assert fit.phase_clim is None
