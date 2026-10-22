"""Presentation changes keep saved scientific evidence unchanged."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.ssb.workflow import SSB


def _saved_result():
    phase = np.sin(np.arange(256).reshape(16, 16)) / 10
    section = dict(column_values=[-1, 0, 1], row_values=[-1, 0, 1],
                   relative_agreement=[[0.3, 0.4, 0.3], [0.7, 1, 0.7], [0.3, 0.4, 0.3]],
                   fitted_point=[0, 0])
    return SSBResult(
        object_wave=np.exp(1j * phase), backend="mps", aberrations={"C10_nm": 0},
        object_sampling_A=(0.5, 0.5), sample_tilt_mrad=(0, 0), depth_spread_nm=0,
        parameter_slices={"scan_detector_rotation_deg": 17,
                          "slices": {"focus_depth": deepcopy(section), "tilt": deepcopy(section)}},
    )


def test_saved_landscape_layout_color_and_export(tmp_path):
    from PIL import Image

    result = _saved_result()
    original, evidence = result.phase.copy(), deepcopy(result.parameter_slices)
    figure = result.show("parameters", size="small", rows=2, cmap="viridis", save=tmp_path / "fit.png")
    assert tuple(figure.get_size_inches()) == pytest.approx((3, 6 * 4.5 / 5.2))
    for axis in figure.axes[:2]:
        assert axis.collections[0].get_cmap().name == "viridis"
    with Image.open(tmp_path / "fit.png") as image:
        image.verify()
    gallery = result.show(size="medium", columns=2)  # phase + two landscapes, partial last row
    assert len(gallery.axes) == 5  # three scientific panels and two colorbars
    assert gallery.axes[0].images[0].get_array().shape[:2] == (16, 16)
    np.testing.assert_array_equal(result.phase, original)
    assert result.parameter_slices == evidence
    for options in ({"columns": 0}, {"rows": True}, {"rows": 1, "columns": 1}):
        with pytest.raises(ValueError):
            result.show("parameters", **options)


def test_phase_save_and_comparison_colormap(tmp_path):
    result = _saved_result()
    original = result.phase.copy()
    figure = result.show(show_parameters=False, cmap="gray", columns=1, save=tmp_path / "phase.png")
    # Core show_2d supplies pre-colormapped RGBA, not a Matplotlib scalar image.
    pixels = figure.axes[0].images[0].get_array()
    np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
    np.testing.assert_array_equal(pixels[..., 1], pixels[..., 2])
    np.testing.assert_array_equal(result.phase, original)
    comparison = result.show(compare=result, cmap="viridis", save=tmp_path / "comparison.png")
    for axis in comparison.axes:
        if axis.images:
            pixels = axis.images[0].get_array()
            assert not np.array_equal(pixels[..., 0], pixels[..., 1])
    with pytest.raises(ValueError, match="fixed comparison layout"):
        result.show(compare=result, rows=2)


def test_invalid_interactive_save_fails_before_fitting():
    # No data/session methods: invalid presentation options must fail before GPU work.
    with pytest.raises(ValueError, match="mode='static'"):
        SSB.show(SimpleNamespace(), "focus", save="focus.png")
    with pytest.raises(ValueError, match="save_dir stores calibration"):
        SSB.show(SimpleNamespace(), save="phase.png")
