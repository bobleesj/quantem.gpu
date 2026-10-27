"""The 180-degree branch statistics: phase skewness, dark-field columns, the column z score and the decision."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from quantem.gpu.direct_ptycho.branches import (
    COLUMN_EFFECT_MIN,
    COLUMN_Z_MIN,
    RotationBranches,
    column_filter_px,
    column_phase_test,
    compare_branches,
    locate_atomic_columns,
)
from quantem.gpu.direct_ptycho.results import column_sign, phase_skewness


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


@pytest.mark.parametrize("weak_column_amplitude", [0.005, -0.03])
def test_negative_columns_do_not_certify_a_weak_alternative(weak_column_amplitude):
    """Independent fits need positive evidence on the branch selected by columns."""
    rng = np.random.default_rng(2)
    dark_field = _lattice()
    first = -0.2 * dark_field + rng.normal(scale=0.02, size=dark_field.shape)
    first[2::12, 2::12] += 3.0
    second = weak_column_amplitude * dark_field + rng.normal(scale=0.2, size=dark_field.shape)
    fits = [_fit(0.0, first), _fit(180.0, second)]
    result = compare_branches(fits, dark_field=dark_field, scan_sampling_A=0.264, keep_first=False)
    assert result.branches[0].column_z < -COLUMN_Z_MIN
    assert result.branches[0].column_effect < -COLUMN_EFFECT_MIN
    assert abs(result.branches[1].column_effect) < COLUMN_EFFECT_MIN
    assert result.branches[0].phase_skewness > result.branches[1].phase_skewness
    assert result.decided_by == "phase_tail" and result.chosen == 0
    assert "atomic columns positive" not in result.summary
    reversed_result = compare_branches(fits[::-1], dark_field=dark_field,
                                       scan_sampling_A=0.264, keep_first=False)
    assert reversed_result.chosen_rotation_deg == result.chosen_rotation_deg


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


class _Session:
    """Stands in for the fitting SSB session: reconstructs each branch at ``FACTOR`` (a 3x auto) by tiling its phase."""

    FACTOR = 3

    def __init__(self):
        self.calls = []

    def panels(self, branches, upsampling):
        self.calls.append(upsampling)
        factor = self.FACTOR if upsampling == "auto" else upsampling
        return factor, tuple(np.kron(branch.phase, np.ones((factor, factor), np.float32)) for branch in branches.branches)


def _branches():
    phase = 0.2 * _lattice() + np.random.default_rng(6).normal(scale=0.02, size=(96, 96))
    return compare_branches([_fit(0.0, phase), _fit(180.0, -phase)], dark_field=_lattice(),
                            scan_sampling_A=0.264, keep_first=False)


def test_record_round_trip_and_figure():
    import matplotlib

    matplotlib.use("Agg")
    branches = _branches()
    record, arrays = branches.to_record()
    assert "rotation_branch_upsampled_phases" not in arrays   # a full-scan session reconstructs them on reload
    restored = RotationBranches.from_record(record, arrays)
    assert restored.summary == branches.summary and restored.scan_sampling_A == (0.264, 0.264)
    np.testing.assert_array_equal(restored.columns, branches.columns)
    np.testing.assert_array_equal(restored.branches[1].phase, branches.branches[1].phase)
    figure = branches.show(panels=False)
    tail_axis, = figure.axes
    assert tail_axis.get_yscale() == "log"
    solid, dashed = tail_axis.get_lines()[:2]
    assert solid.get_color() == "#29243e" and solid.get_linestyle() == "-"
    assert dashed.get_color() == "#cc4b42" and dashed.get_linestyle() == "--"
    assert tail_axis.get_legend().get_texts()[0].get_text().startswith("0.0° · S = +")
    assert "high-passed" not in tail_axis.get_xlabel()
    assert not np.any(solid.get_ydata() == 0)


def test_figure_shows_both_fits_upsampled_with_columns_on_the_display_grid():
    """The panels are the session's auto reconstruction; markers sit at scan pixels x factor; the tests stay native."""
    import matplotlib

    matplotlib.use("Agg")
    branches, session = _branches(), _Session()
    native = [(branch.phase_skewness, branch.column_z) for branch in branches.branches]
    branches.bind_session(session.panels)
    figure = branches.show()
    assert session.calls == ["auto"] and branches.upsampling == 3
    assert branches.object_sampling_A == pytest.approx((0.264 / 3, 0.264 / 3))
    image_axes = [axis for axis in figure.axes if axis.images]
    assert [axis.get_title() for axis in image_axes] == ["0.0° kept · 3× auto (rad)", "180.0° · 3× auto (rad)"]
    for axis, branch in zip(image_axes, branches.branches, strict=True):
        assert axis.images[0].get_array().shape[:2] == (3 * 96, 3 * 96)
        markers = axis.collections[0].get_offsets()
        np.testing.assert_array_equal(markers, branches.columns[:, ::-1] * 3)
        assert axis.get_xlim() == pytest.approx((-0.5, 3 * 96 - 0.5))
    # the decision and the evidence panels still read the native phases
    assert [(branch.phase_skewness, branch.column_z) for branch in branches.branches] == native
    tail_axis = next(axis for axis in figure.axes if axis.get_yscale() == "log")
    assert tail_axis.get_legend().get_texts()[0].get_text() == f"0.0° · S = +{native[0][0]:.3f}"
    branches.show()
    assert session.calls == ["auto"]   # reconstructed once
    branches.show(upsampling=1)
    assert session.calls == ["auto", 1] and branches.upsampling == 1


def test_auto_panels_are_saved_with_the_record_once_reconstructed():
    branches, session = _branches(), _Session()
    branches.bind_session(session.panels)
    shown = branches.upsampled_phases()
    record, arrays = branches.to_record()
    restored = RotationBranches.from_record(record, arrays)
    assert restored.upsampling == 3
    np.testing.assert_array_equal(restored.upsampled_phases()[1], shown[1])


def test_panels_need_the_session_that_fitted_the_branches():
    """Without a live session there is nothing to reconstruct from; the evidence alone needs no GPU."""
    import matplotlib

    matplotlib.use("Agg")
    branches = _branches()
    with pytest.raises(ValueError, match="panels=False"):
        branches.show()
    session = _Session()
    branches.bind_session(session.panels)
    del session   # held weakly: a dropped session is never kept alive by its result
    with pytest.raises(ValueError, match="closed"):
        branches.upsampled_phases()
    assert len(branches.show(panels=False).axes) == 1


def test_display_helpers_default_to_the_auto_factor():
    """show_trials and branch panels resolve "auto" as reconstruct does: 2x at 0.264 Å, 3x at 0.373 Å, 4x at 0.5 Å."""
    import inspect

    from quantem.gpu.direct_ptycho.workflow import SSB

    assert inspect.signature(SSB.show_trials).parameters["upsampling"].default == "auto"
    assert inspect.signature(RotationBranches.show).parameters["upsampling"].default == "auto"
    assert inspect.signature(SSB.reconstruct).parameters["upsampling"].default == 1   # the fit grid
    for step, factor in ((0.264, 2), (0.373, 3), (0.5, 4)):
        calibration = SimpleNamespace(voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(step, step))
        assert SSB._resolve_upsampling(calibration, "auto") == factor
    assert SSB._resolve_upsampling(calibration, 2) == 2
    coarse = SimpleNamespace(voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(2.0, 2.0))
    assert SSB._resolve_upsampling(coarse, "auto") == 4   # capped; the notice is tested in test_ssb_auto_sampling


def test_a_native_phase_view_names_the_auto_factor(capsys):
    """result.show() draws the phase it holds; below the auto factor one line says how to get the auto one."""
    import matplotlib

    matplotlib.use("Agg")
    pytest.importorskip("quantem.core")
    from quantem.gpu.direct_ptycho.results import SSBResult

    wave = np.exp(1j * _lattice(n=32))
    native = SSBResult(object_wave=wave, backend="mps", voltage_kV=300.0, semiangle_mrad=30.0,
                       object_sampling_A=(0.5, 0.5))
    native.show(axsize=(2, 2))
    assert capsys.readouterr().out.strip() == (
        "SSB: showing this phase at 1x, the grid it was reconstructed on; "
        "ssb.reconstruct(result, upsampling='auto') shows it at 4x.")
    auto = SSBResult(object_wave=np.exp(1j * _lattice(n=128)), backend="mps", voltage_kV=300.0, semiangle_mrad=30.0,
                     object_sampling_A=(0.125, 0.125), upsampling=4)
    auto.show(axsize=(2, 2))
    assert capsys.readouterr().out == ""
