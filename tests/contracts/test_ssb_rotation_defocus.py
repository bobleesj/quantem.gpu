"""Review rotation/focus evidence without altering the reconstructed phases."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from quantem.gpu.direct_ptycho.results import DirectPtychoResult
from quantem.gpu.direct_ptycho.rotation_defocus import RotationDefocusComparison


def test_review_shares_contrast_and_retains_unresolved_phase_evidence():
    """A four-case review keeps phase polarity and shows the same physical ROI."""
    import matplotlib.pyplot as plt

    rows, cols = np.mgrid[:32, :32]
    phase = 0.15 * np.cos(rows) * np.cos(cols)
    fits, reconstructions = [], []
    for angle in (160.0, 340.0):
        for focus in (-8.0, 8.0):
            wave = np.exp(1j * phase * np.sign(focus))
            fields = dict(
                backend="mps", aberrations={"C10_nm": focus},
                scan_detector_rotation_deg=angle, sample_tilt_mrad=(1.0, 2.0),
                depth_spread_nm=10.0,
            )
            fits.append(DirectPtychoResult(object_wave=wave, object_sampling_A=(0.5, 0.5), **fields))
            reconstructions.append(DirectPtychoResult(
                object_wave=wave.repeat(2, axis=0).repeat(2, axis=1),
                object_sampling_A=(0.25, 0.25), upsampling_factor=2, **fields,
            ))
    review = RotationDefocusComparison(tuple(fits), tuple(reconstructions), ((-30, 0), (0, 30)))
    originals = [result.phase.copy() for result in reconstructions]
    figure = review.show(view_fraction=0.5)
    panels = [axis.images[0] for axis in figure.axes if axis.images]
    assert len(panels) == 4
    assert len({panel.get_clim() for panel in panels}) == 1
    assert all(panel.get_array().shape == (32, 32) for panel in panels)
    for panel, original in zip(panels, originals, strict=True):
        region = original[16:48, 16:48]
        np.testing.assert_array_equal(panel.get_array(), region - np.median(region))
    for result, original in zip(reconstructions, originals, strict=True):
        np.testing.assert_array_equal(result.phase, original)
    assert review.status.startswith("Unresolved")
    assert not plt.fignum_exists(figure.number)  # Returned once, no implicit duplicate display.
    assert review.show(view_fraction=1).axes[0].images[0].get_array().shape == (64, 64)


def test_comparison_defaults_to_table_and_show_displays_one_figure(monkeypatch):
    """Use the notebook entry point with optional display and a reusable report."""
    import IPython.display
    from types import SimpleNamespace
    import pandas as pd
    from quantem.gpu.direct_ptycho import rotation_defocus
    from quantem.gpu.direct_ptycho.workflow import SSB

    fits = tuple(DirectPtychoResult(
        object_wave=np.exp(1j * np.arange(64).reshape(8, 8) / 100), backend="mps",
        aberrations={"C10_nm": focus}, scan_detector_rotation_deg=angle,
        object_sampling_A=(0.5, 0.5), loss=0.2,
    ) for angle in (160.0, 340.0) for focus in (-8.0, 0.0))
    review = RotationDefocusComparison(fits, fits, ((-30, 0), (0, 30)))
    # GPU fitting is tested on hardware; isolate the notebook rendering contract.
    monkeypatch.setattr(rotation_defocus, "compare_rotation_defocus", lambda *args, **kwargs: review)
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    returned = SSB.compare_rotation_defocus(SimpleNamespace(_device=None))
    assert returned is review
    assert not displayed
    table = returned.report()
    assert isinstance(table, pd.DataFrame)
    assert table["C10 (nm)"].tolist() == [-8.0, 0.0, -8.0, 0.0]
    assert table["f = -C10 (nm)"].tolist() == [8.0, 0.0, 8.0, 0.0]
    assert table["At zero boundary"].tolist() == [False, True, False, True]
    assert "Unresolved" in returned._repr_html_()
    SSB.compare_rotation_defocus(SimpleNamespace(_device=None), show=True)
    assert len(displayed) == 1



def test_comparison_reuses_evidence_without_warm_starting_any_case():
    """Four independent starts retain all trial budgets and leave the caller unchanged."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    from quantem.gpu.direct_ptycho.rotation_defocus import compare_rotation_defocus

    calls, selections, closed = [], [], []
    initial = {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0}

    class Candidate:
        def set_rotation(self, angle):
            self.scan_detector_rotation_deg = angle

        def find_aberrations(self, **options):
            calls.append((self.scan_detector_rotation_deg, dict(self.aberrations),
                          self.sample_tilt_mrad, self.depth_spread_nm, options))
            # A completed fit changes state; none of it may seed the next case.
            self.aberrations = {"C10_nm": 13.0, "C12_nm": 2.0, "phi12_deg": 37.0}
            self.sample_tilt_mrad, self.depth_spread_nm = (3.0, 4.0), 9.0
            self.latest = SimpleNamespace(case=len(calls))
            return self.latest

        def run(self, *, upsampling_factor):
            assert upsampling_factor == "auto"
            return self.latest

    @contextmanager
    def select_region(region, **options):
        selections.append((region, options))
        candidate = Candidate()
        candidate.set_rotation(options["scan_detector_rotation_deg"])
        try:
            yield candidate
        finally:
            closed.append(True)

    parent = SimpleNamespace(input_scan_shape=(128, 128), scan_detector_rotation_deg=340.0,
                             select_region=select_region, aberrations={"C10_nm": 4.0})
    comparison = compare_rotation_defocus(parent, num_trials=200, refinement="newton", verbose=False)
    assert len(selections) == len(closed) == 1
    assert selections[0][0] == (0, 128, 0, 128)
    assert [row[0] for row in calls] == [160.0, 160.0, 340.0, 340.0]
    assert all(row[1] == initial and row[2:4] == (None, None) for row in calls)
    assert [row[4]["search_ranges"]["C10_nm"] for row in calls] == [(-30, 0), (0, 30)] * 2
    assert all(row[4]["num_trials"] == 200 and row[4]["refinement"] == "newton" for row in calls)
    assert [result.case for result in comparison.fits] == [1, 2, 3, 4]
    assert comparison.status.startswith("Unresolved")
    assert parent.scan_detector_rotation_deg == 340.0 and parent.aberrations == {"C10_nm": 4.0}
