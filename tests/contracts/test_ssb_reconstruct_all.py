"""One scientist call reconstructs three methods with one shared optical fit."""
from unittest.mock import Mock

import numpy as np
import pytest

from quantem.gpu import DirectPtycho
from quantem.gpu.direct_ptycho import direct_workflow
from quantem.gpu.direct_ptycho.results import DirectPtychoResult


def test_all_reuses_one_fit_and_matches_separate_calls(monkeypatch):
    session = object.__new__(DirectPtycho)
    session._device = None
    session.backend = "cuda"
    session._aberrations_explicit = False
    fit = object()
    session.fit_aberrations = Mock(return_value=fit)
    session._resolve_upsample = lambda value: 3
    calls = []

    def conventional(parameters, **options):
        calls.append(("phase_of_mean", parameters, options))
        return "ssb-result"

    def direct(owner, parameters, **options):
        method = options["policy"]["method"]
        calls.append((method, parameters, options))
        return f"{method}-result"

    session._run_ssb = conventional
    monkeypatch.setattr(direct_workflow, "run", direct)
    images = session.run(method="all", upsample="auto", compute_loss=False,
                                 phase_damp=.05, amplitude_damp=.1)
    session.fit_aberrations.assert_called_once_with(compute_loss=False, verbose=False)
    assert list(images) == ["phase_of_mean", "phase", "complex"]
    assert all(parameters is fit for _, parameters, _ in calls)
    assert calls[1][2]["policy"]["phase_damp"] == .05
    assert "amplitude_damp" not in calls[1][2]["policy"]
    assert calls[2][2]["policy"]["amplitude_damp"] == .1
    assert all(not options["compute_loss"] for _, _, options in calls)
    for method in images:
        options = {} if method == "phase_of_mean" else {"phase_damp": .05}
        if method == "complex":
            options["amplitude_damp"] = .1
        assert session.run(fit, method=method, **options) == images[method]
    assert session.fit_aberrations.call_count == 1
    assert not hasattr(session, "find_aberrations")
    assert not hasattr(session, "compare_rotation_branches")
    assert not hasattr(session, "compare_rotation_defocus")


def test_all_rejects_invalid_options_before_fitting():
    session = object.__new__(DirectPtycho)
    session._device = None
    session.backend = "cuda"
    session.fit_aberrations = Mock()
    with pytest.raises(ValueError, match="positive"):
        session.run(method="all", phase_damp=-1)
    with pytest.raises(ValueError, match="trial_ids"):
        session.run(method="all", trial_ids=[1])
    session.backend = "other"
    with pytest.raises(NotImplementedError, match="CUDA"):
        session.run(method="all")
    session.fit_aberrations.assert_not_called()


def test_named_method_comparison_displays_each_result_once():
    import matplotlib
    matplotlib.use("Agg")
    wave = np.exp(1j * np.arange(64).reshape(8, 8) / 100)
    images = {name: DirectPtychoResult(object_wave=wave.copy(), backend="cuda", object_sampling_A=(.5, .5),
                             reconstruction_method=name)
              for name in ("phase_of_mean", "phase", "complex")}
    figure = images["phase_of_mean"].show(compare=images, columns=3, size="small")
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 6  # Full field and matched zoom, once per method.
    assert [axis.get_title().splitlines()[0] for axis in image_axes[:3]] == [
        "Conventional SSB", "Regularized phase-only", "Amplitude + phase",
    ]
