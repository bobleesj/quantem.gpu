"""Initial notebook handoff must not change scientific fitting or session state."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu.ssb import fifth_order_fit
from quantem.gpu.ssb.fit_display import FitDisplay
from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.ssb.workflow import SSB


def _fit():
    return SSBResult(
        object_wave=np.ones((8, 8), dtype=np.complex64), backend="mps",
        aberrations={"C10_nm": 1.0, "C12_nm": 0.2, "phi12_deg": 17.0},
        sample_tilt_mrad=(0.1, -0.2), depth_spread_nm=10.0,
        scan_detector_rotation_deg=12.0, loss=0.25,
        phase_estimator="phase_of_mean", amplitude_estimated=False,
        object_sampling_A=(0.5, 0.5), num_bright_field_pixels=9,
    )


@pytest.fixture
def handoff(monkeypatch):
    """Exercise real handoff code with a bounded, deterministic CPU control stub."""
    ipython = pytest.importorskip("IPython")
    import IPython.display
    from quantem.gpu.ssb import fifth_order_parameters

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(fifth_order_fit.time, "perf_counter", lambda: clock.now)
    events = []
    figures = []
    display_handle = Mock()

    def publish(figure, *, display_id):
        assert display_id is True
        events.append("publish")
        clock.now += 3.0
        return display_handle

    def replace(figure):
        events.append("replace")
        clock.now += 3.0

    display_handle.update.side_effect = replace
    monkeypatch.setattr(ipython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    monkeypatch.setattr(IPython.display, "display", Mock(side_effect=publish))

    def draw(result, *, show_parameters):
        assert show_parameters is False
        events.append("draw")
        clock.now += 4.0
        figure = Mock()
        figures.append(figure)
        return figure

    monkeypatch.setattr(SSBResult, "show", draw)

    class Session:
        backend = "mps"
        source_kind = "array"
        source_path = None
        scan_detector_rotation_flipped = False
        bright_field_center_px = (1.0, 1.0)
        scan_detector_rotation_deg = 12.0
        _compute = None

        def __init__(self):
            self.calls = []
            self.primary = _fit()
            self._accept_result(self.primary)
            self._aberration_search = self.primary

        def _accept_result(self, fit):
            return SSB._accept_result(self, fit)

        def _find_aberrations(self, *args, **kwargs):
            events.append("initial")
            clock.now += 2.0
            return self.primary

        def reconstruct(self, fit, **options):
            factor = options["upsampling_factor"]
            self.calls.append(factor)
            events.append(f"reconstruct:{factor}")
            clock.now += 1.0
            result = copy.deepcopy(fit)
            result.upsampling_factor = 2 if factor == "auto" else factor
            result.object_wave = np.full(
                (8 * result.upsampling_factor,) * 2,
                np.complex64(1.0 + fit.aberrations["C10_nm"] * 1j),
            )
            result.loss = 0.25 if options.get("compute_loss") else None
            return self._accept_result(result)

    class Objective:
        def __init__(self):
            self.points = []

        def evaluate(self, points, order):
            events.append("objective")
            clock.now += 0.5
            points = np.atleast_2d(points)
            self.points.extend(points.copy())
            return 10.0 - np.sum(points ** 2, axis=1)

    objectives = []

    def objective(session):
        events.append("setup")
        clock.now += 3.0
        instance = Objective()
        objectives.append(instance)
        return instance

    def refine(tracker, start, bounds):
        events.append("refine")
        candidate = start.copy()
        candidate[0] *= 0.5
        tracker(candidate)
        return 1, "small bounded Newton step"

    monkeypatch.setattr(fifth_order_fit, "_objective", objective)
    monkeypatch.setattr(fifth_order_fit, "newton", refine)
    monkeypatch.setattr(fifth_order_parameters, "public_params", lambda obj, z, order: (
        {"C10_nm": float(z[0] * 10), "C12_nm": 0.2, "phi12_deg": 17.0,
         "C50_nm": 2.0}, (0.1, -0.2), 10.0,
    ))

    def run(session=None, **options):
        return fifth_order_fit.fit_fifth_order(
            Session() if session is None else session,
            search="standard", trials=200, seed=42,
            save_dir=options.pop("save_dir", None), force=False, verbose=False,
            primary_options={"check_rotation": False, "refine_rotation": False},
            **options,
        )

    return SimpleNamespace(
        Session=Session, clock=clock, events=events, objectives=objectives,
        run=run, figures=figures, display=IPython.display.display,
        handle=display_handle,
    )


def test_initial_and_final_auto_views_preserve_native_fit(handoff):
    session = handoff.Session()
    result = handoff.run(session, show=True)
    assert session.calls == ["auto", 1, "auto"]
    assert handoff.display.call_count == 1
    handoff.handle.update.assert_called_once_with(handoff.figures[1])
    assert result.upsampling_factor == 1
    assert session._aberration_search is result
    assert session._reconstruction is result
    assert session.aberrations == result.aberrations
    assert session.sample_tilt_mrad == result.sample_tilt_mrad


def test_show_false_avoids_preview_and_preserves_trajectory(handoff):
    shown = handoff.run(show=True)
    shown_points = handoff.objectives[-1].points
    calls_before = handoff.display.call_count
    hidden_session = handoff.Session()
    hidden = handoff.run(hidden_session, show=False)
    assert hidden_session.calls == [1]
    assert handoff.display.call_count == calls_before
    assert len(handoff.objectives[-1].points) == len(shown_points)
    for hidden_point, shown_point in zip(handoff.objectives[-1].points, shown_points):
        np.testing.assert_array_equal(hidden_point, shown_point)
    np.testing.assert_array_equal(hidden.object_wave, shown.object_wave)
    assert hidden.optimizer["coordinates"] == shown.optimizer["coordinates"]
    assert hidden.optimizer["refinement_trace"] == shown.optimizer["refinement_trace"]
    assert hidden.elapsed_s == shown.elapsed_s
    assert hidden.timings["initial_handoff_s"] == 0
    assert hidden.timings["final_display_s"] == 0


def test_callback_runs_once_before_refinement_and_outside_fit_clock(handoff):
    session = handoff.Session()
    callback_results = []

    def receive(result):
        callback_results.append(result)
        handoff.events.append("callback")
        handoff.clock.now += 7.0

    result = handoff.run(session, show=True, on_initial_fit=receive)
    assert callback_results == [session.primary]
    assert session.primary.upsampling_factor == 1
    assert handoff.events.index("initial") < handoff.events.index("callback")
    assert handoff.events.index("callback") < handoff.events.index("setup")
    assert result.timings["initial_handoff_s"] == 15.0
    assert result.timings["final_display_s"] == 8.0
    assert result.timings["primary_fit_s"] == 2.0
    assert result.timings["fifth_order_setup_s"] == 3.5
    assert result.timings["refinement_s"] == 1.0
    assert result.elapsed_s == 7.5
    assert result.timings["fit_and_display_s"] == 30.5
    assert result.optimizer["refinement_start_s"] == 5.5
    np.testing.assert_array_equal(
        np.asarray(result.optimizer["refinement_trace"])[:, 2], [0.5, 1.0],
    )


def test_initial_callback_runs_when_notebook_display_is_disabled(handoff):
    receive = Mock()
    session = handoff.Session()
    handoff.run(session, show=False, on_initial_fit=receive)
    receive.assert_called_once_with(session.primary)
    assert session.calls == [1]
    handoff.display.assert_not_called()


def test_auto_display_does_not_replace_session_result_or_loss(handoff):
    session = handoff.Session()
    result, loss = session._reconstruction, session.best_loss
    FitDisplay(session, enabled=True).update(result, final=False)
    assert session._reconstruction is result
    assert session.best_loss == loss


def test_reused_fit_display_keeps_the_native_session_result(handoff, monkeypatch, tmp_path):
    session = handoff.Session()
    session.source_kind = "hdf5"
    session._result_signature = Mock(return_value={})
    reused = session.primary
    receive = Mock()
    monkeypatch.setattr(fifth_order_fit, "load_result", lambda **kwargs: reused)
    result = handoff.run(session, show=True, save_dir=tmp_path, on_initial_fit=receive)
    assert result is reused
    assert session.calls == ["auto"]
    receive.assert_not_called()
    assert session._reconstruction is reused
    assert session.best_loss == reused.loss


def test_display_failure_keeps_session_result_and_loss(handoff, monkeypatch):
    session = handoff.Session()
    result, loss = session._reconstruction, session.best_loss
    monkeypatch.setattr(SSBResult, "show", Mock(side_effect=RuntimeError("renderer failed")))
    with pytest.raises(RuntimeError, match="renderer failed"):
        FitDisplay(session, enabled=True).update(result, final=False)
    assert session._reconstruction is result
    assert session.best_loss == loss


def test_mps_objective_preserves_prepared_source_and_selected_rotation(monkeypatch):
    """MPS handoff must reuse physical-backend preparation without a CUDA fallback."""
    from quantem.gpu.ssb.mps import fifth_order

    prepared = object()
    constructor = Mock(return_value=object())
    monkeypatch.setattr(fifth_order, "Objective", constructor)
    backend = SimpleNamespace(_prepared=prepared, cache_rotation=Mock())
    session = SimpleNamespace(
        backend="mps", scan_detector_rotation_deg=12.0, _backend=lambda: backend,
    )
    result = fifth_order_fit._objective(session)
    assert result is constructor.return_value
    constructor.assert_called_once_with(prepared)
    backend.cache_rotation.assert_called_once_with(np.deg2rad(12.0))
