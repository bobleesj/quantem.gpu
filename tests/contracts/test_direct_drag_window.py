"""A failed preview optimization leaves exact Direct ptycho available."""

from types import SimpleNamespace

import numpy as np

from quantem.gpu.direct_ptycho import direct_workflow


def test_drag_stays_exact_after_idle_window_runs_out_of_memory(monkeypatch):
    steps = []
    previews = []

    def step():
        steps.append("build")
        # CuPy's OutOfMemoryError derives from MemoryError.
        raise MemoryError("No room for Chebyshev coefficients")

    def preview(aberrations, *, compute_loss, upsample, **settings):
        previews.append((aberrations["C10_nm"], compute_loss, upsample))
        return np.full((upsample, upsample), aberrations["C10_nm"], np.float32), None

    partial = SimpleNamespace(series=SimpleNamespace(ready=False), step=step)
    ready = SimpleNamespace(series=SimpleNamespace(ready=True))
    session = SimpleNamespace(
        backend="cuda", _device=None, _streamed=None, preview=preview,
        _direct_windows={"C10_nm": (None, [partial], 0), "C12_nm": (None, [ready], 0)},
    )
    policy = direct_workflow.settings("phase_of_mean", None, None, None, None)

    assert direct_workflow.advance(session)
    assert session._direct_windows == {}

    def unexpected_window(*args, **kwargs):
        raise AssertionError("Window allocations must stay off after a failure")

    monkeypatch.setattr(direct_workflow, "_window", unexpected_window)
    for value in (-3.5, 2.25):
        coefs = {"C10_nm": value}
        phase = direct_workflow.drag(session, "C10_nm", coefs, None, None, 3, policy)
        np.testing.assert_array_equal(phase, np.full((3, 3), value, np.float32))
        direct_workflow.prepare(session, ("C10_nm",), coefs, None, None, 3, policy)
        assert direct_workflow.advance(session)

    assert steps == ["build"]
    assert previews == [(-3.5, False, 3), (2.25, False, 3)]
