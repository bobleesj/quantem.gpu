"""Initial and final notebook views share one display and never change fit data."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quantem.gpu.ssb.fit_display import FitDisplay


def test_preview_updates_one_auto_view(monkeypatch):
    ipython = pytest.importorskip("IPython")
    import IPython.display

    display = Mock(return_value=Mock())
    monkeypatch.setattr(ipython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    monkeypatch.setattr(IPython.display, "display", display)
    figures = [Mock(), Mock()]
    image = SimpleNamespace(show=Mock(side_effect=figures))
    ssb = SimpleNamespace(run=Mock(return_value=image), _accept_result=Mock())
    initial, final = object(), object()
    view = FitDisplay(ssb, enabled=True)
    view.update(initial, final=False)
    view.update(final, final=True)
    assert display.call_count == 1
    display.return_value.update.assert_called_once_with(figures[1])
    assert [call.args[0] for call in ssb.run.call_args_list] == [initial, final]
    assert all(call.kwargs["upsampling_factor"] == "auto" and not call.kwargs["compute_loss"]
               for call in ssb.run.call_args_list)


@pytest.mark.parametrize("enabled, notebook", [(False, True), (True, False)])
def test_disabled_or_cli_fit_creates_no_display(monkeypatch, enabled, notebook):
    ipython = pytest.importorskip("IPython")
    monkeypatch.setattr(ipython, "get_ipython", lambda: SimpleNamespace(kernel=object()) if notebook else None)
    ssb = SimpleNamespace(run=Mock())
    FitDisplay(ssb, enabled=enabled).update(object(), final=False)
    ssb.run.assert_not_called()
