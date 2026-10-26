"""Search selection must preserve explicit model choices and reject silent substitution."""
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.workflow import SSB


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
    from quantem.gpu.direct_ptycho import fifth_order_fit
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
])
def test_invalid_options_fail_before_computation(options):
    with pytest.raises(ValueError):
        call(object(), **options)


def test_mps_uses_same_default_model(monkeypatch):
    from quantem.gpu.direct_ptycho import fifth_order_fit
    run = Mock(return_value="fifth-order fit")
    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", run)
    session = SimpleNamespace(backend="mps")
    assert call(session) == "fifth-order fit"
    assert run.call_args.kwargs["search"] == "standard"
    assert run.call_args.kwargs["trials"] == 200


@pytest.mark.parametrize("refinement", ["newton", "nelder-mead", "bfgs"])
def test_a_retired_refinement_prints_one_line_and_runs_auto(monkeypatch, capsys, refinement):
    from quantem.gpu.direct_ptycho import fifth_order_fit
    run = Mock(return_value="fit")
    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", run)
    assert call(object(), search="broad", refinement=refinement) == "fit"
    assert "refinement" not in run.call_args.kwargs
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"refinement={refinement!r} is retired; using refinement='auto', the exact-derivative fit."]
    session = SimpleNamespace(_find_aberrations=Mock(return_value="primary fit"))
    assert call(session, search="primary", refinement=refinement) == "primary fit"
    assert session._find_aberrations.call_args.kwargs["refinement"] == "auto"
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_no_refinement_is_for_the_primary_search_only(capsys):
    session = SimpleNamespace(_find_aberrations=Mock(return_value="primary fit"))
    assert call(session, search="primary", refinement=None) == "primary fit"
    assert session._find_aberrations.call_args.kwargs["refinement"] is None
    with pytest.raises(ValueError, match="Fifth-order search uses its joint model"):
        call(object(), search="standard", refinement=None)
    assert capsys.readouterr().out == ""


def test_auto_prints_nothing(monkeypatch, capsys):
    from quantem.gpu.direct_ptycho import fifth_order_fit
    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", Mock(return_value="fit"))
    call(object(), refinement="auto")
    assert capsys.readouterr().out == ""


def _exact_session(monkeypatch, start, refine, *, study=None, **session):
    """A CUDA session stub whose fifth-order fit runs on a constant objective with ``refine`` as the refinement."""
    from quantem.gpu.direct_ptycho import fifth_order_fit, fifth_order_parameters

    objective = SimpleNamespace(evaluate=lambda points, order: np.ones(len(np.atleast_2d(points))),
                                p=SimpleNamespace(wavelength=0.019687489006849, semiangle_rad=0.03))
    monkeypatch.setattr(fifth_order_fit, "_objective", lambda ssb, gradients=False: objective)
    monkeypatch.setattr(fifth_order_parameters, "vector", lambda fit, order: start.copy())
    monkeypatch.setattr(fifth_order_parameters, "public_params",
                        lambda *args: ({"C50_nm": 1.}, (1., -2.), 10.))
    monkeypatch.setattr(fifth_order_fit, "_refine_with_gradients", refine)
    if study is not None:
        monkeypatch.setattr(fifth_order_fit.optuna, "create_study", lambda **kwargs: study)
        monkeypatch.setattr(fifth_order_fit, "make_independent_sampler", lambda **kwargs: None)
    return SimpleNamespace(
        backend="cuda", _backend=lambda: SimpleNamespace(_get_accelerator=lambda: object()),
        run=lambda result, **kwargs: result, _accept_result=lambda result: result, **session)


def _record(point, stop="test stop"):
    return {"coordinates": np.asarray(point, dtype=float), "agreement": 1.0, "objectives": 3, "gradients": 2,
            "subsampled": 0, "iterations": 1, "stop": stop, "trace": [[1, 1.0, 0.0]], "hessian_bf_stride": 64,
            "coarse_bf_stride": 8}


def test_broad_records_the_exact_gradient_budget(monkeypatch, tmp_path):
    """The refinement, the persisted result and the cache signature share one budget."""
    from quantem.gpu.direct_ptycho import fifth_order_fit

    start = np.zeros(28)
    start[5] = 1.
    caps = []

    def refine(objective, point, bounds, norm, cap, *, on_progress=None):
        caps.append(cap)
        return _record(point)

    study = SimpleNamespace(trials=[], enqueue_trial=lambda value: None, optimize=lambda *args, **kwargs: None)
    signature = Mock(return_value="signature")
    session = _exact_session(
        monkeypatch, start, refine, study=study, source_kind="hdf5", _result_signature=signature,
        _save_result=lambda result, **kwargs: result,
        _find_aberrations=Mock(return_value=SimpleNamespace(rotation_branches=None, trial_records=[])))
    monkeypatch.setattr(fifth_order_fit, "result_paths", lambda *args: object())
    monkeypatch.setattr(fifth_order_fit, "load_result", lambda **kwargs: None)
    result = fifth_order_fit.fit_fifth_order(
        session, search="broad", trials=400, seed=17, save_dir=tmp_path, force=False, verbose=False,
        primary_options={"check_rotation": True, "refine_rotation": False}, higher_order_limit=2)
    assert caps == [400]
    assert signature.call_args.args[1]["newton_cap"] == 400 and signature.call_args.args[1]["refinement"] == "bfgs"
    assert result.optimizer["refinement_cap"] == 400 and "newton_cap" not in result.optimizer
    assert result.optimizer["stop"] == "test stop" and result.refinement == "bfgs"
    assert result.optimizer["trials"] == 400
    assert session._find_aberrations.call_args.kwargs["num_trials"] == 200


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
    fits = call(parent, search="broad", trials=(200, 400, 600), seed=17, save_dir=tmp_path, verbose=False)
    assert list(fits) == [200, 400, 600]
    assert parent.aberrations == {"C10_nm": 0.}
    assert len(closed) == 3
    assert [item["save_dir"] for item in options] == [tmp_path / f"trials-{n}" for n in fits]
    assert all(item["seed"] == 17 and item["refinement"] == "auto" for item in options)


def test_standard_uses_initial_fit_without_extra_tpe(monkeypatch):
    """Default initialization reaches the same joint solver without an extra study."""
    from quantem.gpu.direct_ptycho import fifth_order_fit

    start = np.zeros(28)
    start[5] = 1.

    def refine(objective, initial, bounds, norm, cap, *, on_progress=None):
        np.testing.assert_array_equal(initial, start)
        return _record(initial, "converged")

    primary = SimpleNamespace(rotation_branches="initial branches", trial_records=[{"loss": -1.}])
    session = _exact_session(monkeypatch, start, refine, _find_aberrations=Mock(return_value=primary))
    monkeypatch.setattr(fifth_order_fit.optuna, "create_study",
                        lambda **kwargs: pytest.fail("Extra TPE search in default route"))
    fit = fifth_order_fit.fit_fifth_order(
        session, search="standard", trials=200, seed=42, save_dir=None, force=False, verbose=False,
        higher_order_limit=2, primary_options={"check_rotation": False, "refine_rotation": False})
    assert session._find_aberrations.call_args.args == ("off",)
    assert session._find_aberrations.call_args.kwargs["num_trials"] == 200
    assert "refinement" not in session._find_aberrations.call_args.kwargs
    assert fit.optimizer["primary_trials"] == 200
    assert fit.optimizer["exploration_trials"] == 0
    assert fit.optimizer["search"] == "standard"
    assert fit.optimizer["coarse_scores"] == []
    assert fit.rotation_branches == "initial branches"
    assert fit.sample_tilt_mrad == (1., -2.) and fit.depth_spread_nm == 10.
    assert fit.refinement == "bfgs"
    assert fit.phase_clim is None


def test_higher_order_subsets_are_retired(monkeypatch, capsys):
    """A plain primary call asking for higher orders runs the fifth-order fit; with a primary-only control the
    primary fit holds them at 0. Either way one line says so."""
    from quantem.gpu.direct_ptycho import fifth_order_fit
    run = Mock(return_value="fifth-order fit")
    monkeypatch.setattr(fifth_order_fit, "fit_fifth_order", run)
    assert call(object(), search="primary", higher_order=("C21", "C30")) == "fifth-order fit"
    assert run.call_args.kwargs["search"] == "standard"
    assert capsys.readouterr().out.splitlines() == [
        "higher_order is retired; running search='standard', the fifth-order fit that fits coma, trefoil, C30 and "
        "every term through fifth order."]
    session = SimpleNamespace(_find_aberrations=Mock(return_value="primary fit"))
    assert call(session, search="primary", tilt=False, higher_order=("C21", "C30")) == "primary fit"
    assert "higher_order" not in session._find_aberrations.call_args.kwargs
    assert capsys.readouterr().out.splitlines() == [
        "higher_order is retired; the default fit (search='standard') fits every term through fifth order, and this "
        "primary fit holds them at 0."]
