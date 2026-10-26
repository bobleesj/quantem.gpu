"""Initial and final notebook views share one display and never change fit data."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quantem.gpu.direct_ptycho.fit_display import FitDisplay


@pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge", 750, 900])
@pytest.mark.parametrize("columns", [1, 2, 3])
def test_preview_compares_three_final_methods_without_refitting(monkeypatch, size, columns):
    ipython = pytest.importorskip("IPython")
    import IPython.display
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from quantem.gpu.direct_ptycho.results import SSBResult

    plt.switch_backend("Agg")
    display = Mock(return_value=Mock())
    monkeypatch.setattr(ipython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    monkeypatch.setattr(IPython.display, "display", display)
    wave = np.exp(1j * np.arange(256).reshape(16, 16) / 100)
    image = SSBResult(object_wave=wave, backend="cuda", object_sampling_A=(.2, .2))
    complex_image = SSBResult(object_wave=wave, backend="cuda", object_sampling_A=(.2, .2),
                              reconstruction_method="complex",
                              reconstruction_options={"phase_regularization": .03, "amplitude_regularization": .03})
    phase_image = SSBResult(object_wave=wave, backend="cuda", object_sampling_A=(.2, .2),
                            reconstruction_method="phase", reconstruction_options={"phase_regularization": .03})
    ssb = SimpleNamespace(run=Mock(side_effect=[image, image, complex_image, phase_image]),
                          _accept_result=Mock(), backend="cuda", _streamed=None)
    initial, final = object(), object()
    view = FitDisplay(ssb, enabled=True, size=size, columns=columns)
    view.update(initial, final=False)
    view.update(final, final=True)
    assert display.call_count == 1
    display.return_value.update.assert_called_once()
    figure = display.return_value.update.call_args.args[0]
    axes = [axis for axis in figure.axes if axis.images]
    assert len(axes) == 3
    assert [axis.get_subplotspec().colspan.start for axis in axes] == [i % columns for i in range(3)]
    assert [axis.get_subplotspec().rowspan.start for axis in axes] == [i // columns for i in range(3)]
    assert [axis.get_title().splitlines()[0] for axis in axes] == [
        "Conventional SSB", "Regularized phase-only", "Amplitude + phase",
    ]
    assert len({axis.images[0].get_clim() for axis in axes}) == 1
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    titles = [axis.title.get_window_extent(renderer) for axis in axes]
    heading = figure._suptitle.get_window_extent(renderer)
    assert all(title.y1 < heading.y0 for title in titles)
    assert all(figure.bbox.contains(title.x0, title.y0)
               and figure.bbox.contains(title.x1, title.y1) for title in titles)
    assert not any(left.overlaps(right) for left, right in zip(titles, titles[1:]))
    assert [call.args[0] for call in ssb.run.call_args_list] == [initial, final, final, final]
    assert all(call.kwargs["upsampling_factor"] == "auto" and not call.kwargs["compute_loss"]
               for call in ssb.run.call_args_list)
    assert [call.kwargs.get("method", "ssb") for call in ssb.run.call_args_list] == [
        "ssb", "ssb", "complex", "phase",
    ]
    ssb._accept_result.assert_called_with(final)


@pytest.mark.parametrize("enabled, notebook", [(False, True), (True, False)])
def test_disabled_or_cli_fit_creates_no_display(monkeypatch, enabled, notebook):
    ipython = pytest.importorskip("IPython")
    monkeypatch.setattr(ipython, "get_ipython", lambda: SimpleNamespace(kernel=object()) if notebook else None)
    ssb = SimpleNamespace(run=Mock())
    FitDisplay(ssb, enabled=enabled).update(object(), final=True)
    ssb.run.assert_not_called()


@pytest.mark.parametrize("size,width", [("large", 6), (750, 7.5)])
def test_large_convergence_is_live_throttled_and_does_not_reconstruct(monkeypatch, size, width):
    import IPython
    import IPython.display
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    display = Mock(return_value=Mock())
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    monkeypatch.setattr(IPython.display, "display", display)
    clock = [1.0]
    monkeypatch.setattr("quantem.gpu.direct_ptycho.fit_display.time.perf_counter", lambda: clock[0])
    ssb = SimpleNamespace(run=Mock())
    view = FitDisplay(ssb, enabled=False, size=size, show_convergence=True)
    view.progress(0)
    view.progress(1, 1.1)
    display.assert_called_once()
    display.return_value.update.assert_not_called()
    clock[0] += 0.6
    view.progress(2, 1.2)
    display.return_value.update.assert_called_once()
    view.progress(3, 1.3, final=True)
    assert display.return_value.update.call_count == 2
    assert list(view.line.get_ydata()) == [1.1, 1.2, 1.3]
    assert view.figure.get_size_inches()[0] == width
    ssb.run.assert_not_called()


def test_default_does_not_collect_live_chart_history():
    view = FitDisplay(SimpleNamespace(), enabled=False)
    view.progress(1, 1.5)
    assert view.evaluations == []
    assert view.scores == []
    assert view.figure is None


@pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge", 750, 900])
def test_fit_heading_clears_the_phase_title(size):
    """An inline fit preview keeps its heading above the full image title."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from quantem.gpu.direct_ptycho.results import SSBResult

    image = SSBResult(
        object_wave=np.exp(1j * np.arange(256).reshape(16, 16) / 100),
        backend="mps", object_sampling_A=(0.2, 0.2), upsampling_factor=3,
    )
    ssb = SimpleNamespace(run=Mock(return_value=image), _accept_result=Mock(), backend="mps")
    view = FitDisplay(ssb, enabled=True, size=size)
    view.display = Mock(return_value=Mock())
    view.update(image, final=True)
    figure = view.display.call_args.args[0]
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    heading = figure._suptitle.get_window_extent(renderer)
    for axis in figure.axes:
        if axis.get_title():
            title = axis.title.get_window_extent(renderer)
            assert title.y1 + 2 < heading.y0


def test_progress_does_not_change_the_refinement_or_its_evaluation_count():
    import numpy as np
    from quantem.gpu.direct_ptycho._gradient_refinement import refine

    start = np.zeros(6)
    start[5] = 1.0
    bounds = np.tile([-2.0, 2.0], (6, 1))
    bounds[5] = (0.2, 6.0)

    def evaluate(points, kind, stride):
        points = np.atleast_2d(points)
        values = (10 - np.sum((points - 0.5) ** 2, axis=1)) / 10
        return values, None if kind == "value" else -2 * (points - 0.5) / 10

    observed = []
    plain = refine(evaluate, start, bounds)
    live = refine(evaluate, start, bounds, progress=lambda count, best: observed.append((count, best)))
    np.testing.assert_array_equal(plain["coordinates"], live["coordinates"])
    assert plain["agreement"] == live["agreement"]
    assert plain["objectives"] == live["objectives"]
    assert observed[-1] == (live["objectives"], live["agreement"])


def test_showptycho_rotation_handoff_preserves_prepared_equations():
    import inspect
    from quantem.gpu.direct_ptycho.workflow import DirectPtycho

    prepared = object()
    backend = Mock()
    session = SimpleNamespace(scan_detector_rotation_deg=176.3,
                              _prepared_direct=prepared, _streamed=None,
                              _backend=Mock(return_value=backend))
    set_rotation = inspect.unwrap(DirectPtycho.set_rotation)
    set_rotation(session, 176.3)
    assert session._prepared_direct is prepared
    session._backend.assert_not_called()
    set_rotation(session, 177.3)
    assert session._prepared_direct is None
    backend.cache_rotation.assert_called_once()


@pytest.mark.parametrize("backend", ["cuda", "mps"])
@pytest.mark.parametrize("columns", [1, 2, 3])
def test_regional_preview_boxes_measured_positions_at_auto_sampling(monkeypatch, backend, columns):
    import IPython
    import IPython.display
    import numpy as np
    from quantem.gpu.direct_ptycho.results import SSBResult

    display = Mock(return_value=Mock())
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    monkeypatch.setattr(IPython.display, "display", display)
    # A 126-position acquisition was padded to 128; the context omits padding.
    full = SSBResult(
        object_wave=np.exp(1j * np.arange(256 * 256).reshape(256, 256) / 1e5),
        backend="mps", object_sampling_A=(0.2, 0.2), upsampling_factor=2,
    )
    fit = SSBResult(object_wave=np.ones((64, 64)), backend="mps", fit_scan_region=(32, 96, 16, 80))
    session = SimpleNamespace(input_scan_shape=(126, 126), run=Mock(return_value=full),
                              _accept_result=Mock(), backend=backend, _streamed=None)
    view = FitDisplay(session, enabled=True, size=750, columns=columns,
                      scan_region=fit.fit_scan_region)
    view.update(fit, final=False)
    view.update(fit, final=True)
    assert display.call_count == 1
    assert display.return_value.update.call_count == 1
    figure = display.return_value.update.call_args.args[0]
    axes = [axis for axis in figure.axes if axis.images]
    count = 3 if backend == "cuda" else 1
    assert len(axes) == 2 * count
    overviews = [axis for axis in axes if "Full field" in axis.get_title()]
    crops = [axis for axis in axes if "Fit-region zoom" in axis.get_title()]
    for overview, crop in zip(overviews, crops, strict=True):
        assert overview.images[0].get_array().shape[:2] == (252, 252)
        assert overview.patches[0].get_xy() == (31.5, 63.5)
        assert overview.patches[0].get_width() == 128
        np.testing.assert_array_equal(crop.images[0].get_array(),
                                      overview.images[0].get_array()[64:192, 32:160])
        if columns == 1 or backend == "cuda":
            assert overview.get_position().y0 > crop.get_position().y1
            assert overview.get_position().x0 == crop.get_position().x0
    session._accept_result.assert_called_with(fit)
    assert fit.object_wave.shape == (64, 64)
