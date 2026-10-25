"""detector.show puts a round, physically labelled scale bar on every calibrated panel.

Scan images get a real-space bar from ``scan_sampling_A``, the mean diffraction an
angular bar from ``detector_sampling_mrad``; a missing sampling leaves its panels bare.
The bar and its label grow on screen with the panel size, so a 3-inch and a
9-inch panel both read.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from quantem.gpu import detector


def _bars(figure):
    """Each image panel's (label, label points, bar thickness in points)."""
    found = []
    for axis in figure.axes:
        if not axis.images:
            continue
        rows = axis.images[0].get_array().shape[0]
        inches = axis.get_position().height * figure.get_size_inches()[1]
        for bar in axis.artists:
            if isinstance(bar, AnchoredSizeBar):
                label = bar.txt_label._text
                height = bar.size_bar.get_children()[0].get_height()
                found.append((label.get_text(), label.get_fontsize(), height / rows * inches * 72))
                assert label.get_path_effects() and label.get_color() == "white"
        if not any(isinstance(bar, AnchoredSizeBar) for bar in axis.artists):
            found.append(None)
    return found


@pytest.fixture
def scan(monkeypatch):
    """256 x 256 scan at 0.5 Å (128 Å field) and a 96 x 96 detector at 0.5 mrad (48 mrad)."""
    image = np.random.default_rng(0).random((256, 256), dtype=np.float32)
    for name in ("bf", "df", "adf"):
        monkeypatch.setattr(detector.workflow, name, lambda data: image)
    monkeypatch.setattr(detector.workflow, "mean",
                        lambda data: np.random.default_rng(1).random((96, 96), dtype=np.float32))
    return SimpleNamespace(calibration=SimpleNamespace(scan_sampling_A=(.5, .5), detector_sampling_mrad=(.5, .5)))


def test_every_panel_gets_a_round_bar_near_a_fifth_of_the_field(scan):
    labels = [bar[0] for bar in _bars(detector.show(scan, size="medium"))]
    # 20 % of 128 Å is 25.6 Å, nearest 20 Å; 20 % of 48 mrad is 9.6, nearest 10 mrad
    assert labels == ["2 nm", "2 nm", "2 nm", "10 mrad"]


def test_bar_and_label_grow_with_panel_size(scan):
    small = _bars(detector.show(scan, size="small"))
    xlarge = _bars(detector.show(scan, size="xlarge"))
    integer = _bars(detector.show(scan, size=1200))
    for small_bar, xlarge_bar, integer_bar in zip(small, xlarge, integer, strict=True):
        assert small_bar[0] == xlarge_bar[0] == integer_bar[0]
        assert small_bar[1] < xlarge_bar[1] <= integer_bar[1]
        assert small_bar[2] < xlarge_bar[2] < integer_bar[2]


def test_missing_sampling_leaves_its_panels_without_a_bar(scan):
    scan.calibration.detector_sampling_mrad = None
    assert _bars(detector.show(scan))[3] is None
    scan.calibration.scan_sampling_A = None
    assert _bars(detector.show(scan)) == [None] * 4
