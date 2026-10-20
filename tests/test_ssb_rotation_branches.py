"""The 180-degree branch statistics: phase skewness, dark-field columns, the column z score and the decision."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from quantem.gpu.ssb.branches import (
    COLUMN_EFFECT_MIN,
    COLUMN_Z_MIN,
    RotationBranches,
    column_filter_px,
    column_phase_test,
    compare_branches,
    locate_atomic_columns,
)
from quantem.gpu.ssb.results import column_sign, phase_skewness


def _lattice(n=96, period_px=12, width_px=1.5):
    """Gaussian columns of unit height on a square lattice of ``period_px`` pixels, sites at period / 2."""
    from scipy.ndimage import gaussian_filter

    sites = np.zeros((n, n))
    sites[period_px // 2::period_px, period_px // 2::period_px] = 2 * np.pi * width_px**2
    return gaussian_filter(sites, width_px, mode="wrap")


def test_phase_skewness_is_the_mean_centred_third_moment_ratio():
    """S = mean(p^3) / mean(p^2)^1.5 with p = phase - mean(phase); odd under negation, blind to an offset."""
    phase = np.random.default_rng(4).exponential(size=(48, 48))
    p = phase - phase.mean()
    assert phase_skewness(phase) == pytest.approx(np.mean(p**3) / np.mean(p**2) ** 1.5, rel=1e-12)
    assert phase_skewness(-phase) == pytest.approx(-phase_skewness(phase), rel=1e-12)
    assert phase_skewness(phase + 5.0) == pytest.approx(phase_skewness(phase), rel=1e-9)
    assert phase_skewness(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(2 / np.sqrt(3))
    assert phase_skewness(np.ones((4, 4))) == 0.0
    # the median-centred column_sign measures a different moment on the same image
    assert column_sign(phase) != pytest.approx(phase_skewness(phase), rel=1e-3)


def test_filters_follow_the_scan_step():
    """2 Å peak separation and 2.1 Å high pass: 17 x 17 and 8 pixels at 0.264 Å, 9 x 9 and 4.2 pixels at 0.5 Å."""
    sigma, window = column_filter_px(0.264)
    assert window == (17, 17) and sigma[0] == pytest.approx(7.95, abs=0.01)
    sigma, window = column_filter_px((0.5, 0.25))
    assert window == (9, 17) and sigma == pytest.approx((4.2, 8.4))


def test_columns_are_the_most_prominent_lattice_maxima():
    """The top 20% of the maxima at least 2 Å apart: lattice sites only, and noise between them never counts."""
    dark_field = _lattice() + np.random.default_rng(0).normal(scale=0.02, size=(96, 96))
    columns = locate_atomic_columns(dark_field, scan_sampling_A=0.264)
    assert 0.15 * 64 <= len(columns) <= 0.3 * 64
    assert set(np.unique(columns % 12)) == {6}
    assert len(locate_atomic_columns(np.ones((32, 32)), scan_sampling_A=0.264)) == 0
    # noise alone at a coarse step gives tens of columns, not thousands
    assert len(locate_atomic_columns(np.random.default_rng(1).normal(size=(512, 512)), scan_sampling_A=0.5)) < 1000


def test_column_z_is_the_mean_over_its_standard_error():
    phase = 0.3 * _lattice() + np.random.default_rng(1).normal(scale=0.05, size=(96, 96))
    columns = locate_atomic_columns(_lattice(), scan_sampling_A=0.264)
    sampled, z, positive = column_phase_test(phase, columns, scan_sampling_A=0.264)
    assert z == pytest.approx(sampled.mean() / (sampled.std() / np.sqrt(sampled.size)))
    assert z > 10 and positive == 1.0
    _, negative_z, negative = column_phase_test(-phase, columns, scan_sampling_A=0.264)
    assert negative_z == pytest.approx(-z) and negative == 0.0
    assert column_phase_test(phase, columns[:2], scan_sampling_A=0.264)[1] is None


def _fit(rotation, phase):
    return SimpleNamespace(scan_detector_rotation_deg=rotation, phase=phase, aberrations={"C10_nm": 1.0},
                           loss=0.1, sample_tilt_mrad=None, depth_spread_nm=None)


def test_columns_decide_before_the_phase_tail():
    """Positive columns win even when the phase tail points the other way."""
    rng = np.random.default_rng(2)
    columns_image = _lattice()
    # bright columns on a background whose few deep holes give the image a negative tail
    phase = 0.2 * columns_image + rng.normal(scale=0.02, size=(96, 96))
    phase[rng.integers(0, 96, 40), rng.integers(0, 96, 40)] -= 3.0
    assert phase_skewness(phase) < 0
    branches = compare_branches([_fit(159.4, -phase), _fit(339.4, phase)], dark_field=columns_image,
                                scan_sampling_A=0.264, keep_first=False)
    assert branches.decided_by == "columns" and branches.kept == branches.chosen == 1
    assert branches.branches[1].column_z >= COLUMN_Z_MIN
    assert branches.summary.startswith("kept 339.4°: atomic columns positive (z +")
    assert branches.branches[1].column_effect >= COLUMN_EFFECT_MIN
    assert branches.summary.endswith("; 159.4° negative.")


def test_a_small_effect_over_many_columns_is_not_decisive():
    """z >= 3 from thousands of columns with a shift of a tenth of their spread leaves the choice to the phase tail."""
    rng = np.random.default_rng(7)
    columns_image = _lattice(n=1200)
    columns = locate_atomic_columns(columns_image, scan_sampling_A=0.264)
    noise = rng.normal(size=columns_image.shape)
    sampled, _, _ = column_phase_test(noise, columns, scan_sampling_A=0.264)
    shift = np.zeros_like(noise)
    shift[columns[:, 0], columns[:, 1]] = 0.1 * sampled.std() - sampled.mean()
    phase = noise + shift
    _, z, _ = column_phase_test(phase, columns, scan_sampling_A=0.264)
    assert z >= COLUMN_Z_MIN
    branches = compare_branches([_fit(0.0, phase), _fit(180.0, -phase)], dark_field=columns_image,
                                scan_sampling_A=0.264, keep_first=False)
    assert abs(branches.branches[0].column_effect) < COLUMN_EFFECT_MIN
    assert branches.decided_by == "phase_tail"
    assert "columns not decisive" in branches.summary


def test_unresolved_columns_fall_back_to_the_phase_tail():
    rng = np.random.default_rng(5)
    phase = rng.exponential(scale=0.1, size=(96, 96))
    dark_field = rng.normal(size=(96, 96))
    branches = compare_branches([_fit(170.7, -phase), _fit(350.7, phase)], dark_field=dark_field,
                                scan_sampling_A=0.5, keep_first=False)
    assert branches.decided_by == "phase_tail" and branches.chosen == 1
    assert "kept 350.7° by the phase tail (S +" in branches.summary
    without = compare_branches([_fit(170.7, -phase), _fit(350.7, phase)], dark_field=None,
                               scan_sampling_A=0.5, keep_first=False)
    assert without.decided_by == "phase_tail" and without.columns is None and without.chosen == 1
    kept = compare_branches([_fit(170.7, -phase), _fit(350.7, phase)], dark_field=None,
                            scan_sampling_A=0.5, keep_first=True)
    assert kept.kept == 0 and kept.chosen == 1
    assert kept.summary.startswith("kept the given 170.7°; the evidence selects 350.7°")


def test_record_round_trip_and_figure():
    import matplotlib

    matplotlib.use("Agg")
    phase = 0.2 * _lattice() + np.random.default_rng(6).normal(scale=0.02, size=(96, 96))
    branches = compare_branches([_fit(0.0, phase), _fit(180.0, -phase)], dark_field=_lattice(),
                                scan_sampling_A=0.264, keep_first=False)
    record, arrays = branches.to_record()
    restored = RotationBranches.from_record(record, arrays)
    assert restored.summary == branches.summary
    np.testing.assert_array_equal(restored.columns, branches.columns)
    np.testing.assert_array_equal(restored.branches[1].phase, branches.branches[1].phase)
    figure = branches.show()
    tail_axis, column_axis = figure.axes[:2]
    assert tail_axis.get_yscale() == "log"
    solid, dashed = tail_axis.get_lines()[:2]
    assert solid.get_color() == "#29243e" and solid.get_linestyle() == "-"
    assert dashed.get_color() == "#cc4b42" and dashed.get_linestyle() == "--"
    assert tail_axis.get_legend().get_texts()[0].get_text().startswith("0.0° · S = +")
    assert "z = +" in column_axis.get_legend().get_texts()[0].get_text()
    assert not np.any(solid.get_ydata() == 0)
