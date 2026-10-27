"""One notebook call compares rotations, retaining calibrated evidence and zooms."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from quantem.gpu import DirectPtycho
from quantem.gpu.direct_ptycho.branches import compare_branches
from quantem.gpu.direct_ptycho.results import SSBResult
from quantem.gpu.direct_ptycho.rotation_defocus import RotationDefocusComparison


def _review():
    rows, cols = np.mgrid[:48, :48]
    phase = .12 * np.cos(rows / 2) * np.cos(cols / 3)
    fits = tuple(SSBResult(
        object_wave=np.exp(1j * sign * phase), backend="cuda",
        aberrations={"C10_nm": sign * 8.0}, scan_detector_rotation_deg=angle,
        object_sampling_A=(.5, .5), loss=.2,
    ) for angle in (160., 340.) for sign in (-1, 1))
    return RotationDefocusComparison(fits, fits, ((-30, 0), (0, 30)))


def test_scan_rotation_report_and_zoom_preserve_four_case_evidence(monkeypatch):
    """The one-call report displays full fields, matching zooms and one table."""
    import IPython.display
    import matplotlib.pyplot as plt
    from quantem.gpu.direct_ptycho import rotation_defocus

    review = _review()
    calls, displayed = [], []
    def compare(session, **options):
        calls.append(options)
        return review
    monkeypatch.setattr(rotation_defocus, "compare_rotation_defocus", compare)
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    session = object.__new__(DirectPtycho)
    session._device = None
    session._data = SimpleNamespace(ndim=4, shape=(512, 512, 192, 192))
    session.scan_region = None
    session.scan_detector_rotation_deg = 340.
    original = [fit.phase.copy() for fit in review.fits]
    result = session.scan_rotations(defocus=True, region=(128, 384, 128, 384), zoom=3, size=400)
    assert result is review and session.scan_detector_rotation_deg == 340.
    assert calls[0] == {"region": (128, 384, 128, 384), "num_trials": 200}
    figure, table = displayed
    images = [ax.images[0] for ax in figure.axes if ax.images]
    assert [image.get_array().shape[:2] for image in images] == [(48, 48)] * 4 + [(16, 16)] * 4
    assert len({image.get_clim() for image in images}) == 1
    assert len(table) == 4 and "Unresolved" in table.index.name
    assert not plt.fignum_exists(figure.number)
    for fit, before in zip(review.fits, original):
        np.testing.assert_array_equal(fit.phase, before)
    displayed.clear()
    assert session.scan_rotations(defocus=True, report=False, size="large") is review
    assert not displayed
    with pytest.raises(ValueError, match="diagnostic"):
        session.scan_rotations(defocus=True, apply=True)
    assert len(calls) == 2


def test_two_angle_region_and_trial_alias_reuse_the_same_solver():
    """The new call and old notebook spelling use identical fits and selection rules."""
    calls = []
    result = SimpleNamespace(rotation_branches=object())
    session = object.__new__(DirectPtycho)
    session._device = None
    session._data = SimpleNamespace(ndim=4, shape=(512, 512, 192, 192))
    session.scan_region = None
    session.scan_detector_rotation_source = "estimated"
    def fit(choice, **options):
        calls.append((choice, options))
        return result
    session._find_aberrations = fit
    returned = session.scan_rotations(report=False, region="center", trials=20)
    legacy = session.scan_rotations(report=False, region=(128, 384, 128, 384), num_trials=20)
    assert returned is legacy is result.rotation_branches
    assert calls == [("apply", {"scan_region": (128, 384, 128, 384), "num_trials": 20})] * 2
    session.scan_detector_rotation_source = "recorded"
    session.scan_rotations(report=False, region=())
    assert calls[-1] == ("keep", {"scan_region": None, "num_trials": 200})


def test_two_angle_zoom_keeps_full_fields_and_column_markers():
    """Magnification crops a copy for display while the decision uses native pixels."""
    rows, cols = np.mgrid[:48, :48]
    phase = .2 * np.cos(rows / 2) * np.cos(cols / 3)
    fits = [SimpleNamespace(scan_detector_rotation_deg=angle, phase=sign * phase,
                            aberrations={"C10_nm": 8 * sign}, loss=.1,
                            sample_tilt_mrad=None, depth_spread_nm=None)
            for angle, sign in [(160., -1), (340., 1)]]
    review = compare_branches(fits, dark_field=None, scan_sampling_A=.5, keep_first=False)
    review._upsampled = ("auto", tuple(fit.phase.repeat(2, 0).repeat(2, 1) for fit in fits))
    review.upsampling = 2
    before = [item.phase.copy() for item in review.branches]
    figure = review.show(zoom=3, size=400)
    images = [ax.images[0] for ax in figure.axes if ax.images]
    assert [image.get_array().shape[:2] for image in images] == [(96, 96)] * 2 + [(32, 32)] * 2
    assert len({image.get_clim() for image in images}) == 1
    for branch, saved in zip(review.branches, before):
        np.testing.assert_array_equal(branch.phase, saved)
