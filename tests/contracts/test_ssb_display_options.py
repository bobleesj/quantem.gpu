"""Presentation changes keep saved scientific evidence unchanged."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.results import SSBResult
from quantem.gpu.direct_ptycho.workflow import SSB


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
    for options in ({"columns": 0}, {"rows": True}, {"rows": 1, "columns": 1},
                    {"size": 0}, {"size": -1}, {"size": True}, {"size": 2.5}):
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


@pytest.mark.parametrize("region,start", [
    ("center", (6, 6)), ("top-left", (0, 0)), ("top-right", (0, 12)),
    ("bottom-left", (12, 0)), ("bottom-right", (12, 12)),
])
def test_method_comparison_marks_the_same_physical_zoom(region, start):
    conventional = _saved_result()
    phase = replace(
        conventional, reconstruction_method="phase",
        reconstruction_options={"phase_damping": 0.03},
    )
    fine = replace(
        conventional, object_wave=np.repeat(np.repeat(conventional.object_wave, 4, axis=0), 4, axis=1),
        upsampling=4, object_sampling_A=(0.125, 0.125),
    )
    original = phase.object_wave.copy()
    figure = phase.show(compare=fine, region=region, cmap="gray", size="large")
    axes = [axis for axis in figure.axes if axis.images]
    assert "Regularized phase-only" in axes[0].get_title()
    assert "Phase damping 0.03" in axes[0].get_title()
    assert "Conventional SSB" in axes[1].get_title()
    assert "4× upsampling" in axes[1].get_title()
    assert f"{region.capitalize()} zoom" in axes[2].get_title()
    for factor, full, zoom in zip((1, 4), axes[:2], axes[2:], strict=True):
        row, col = np.array(start) * factor
        rectangle = full.patches[0]
        assert rectangle.get_xy() == (col - .5, row - .5)
        assert rectangle.get_width() == 4 * factor
        assert rectangle.get_height() == 4 * factor
        np.testing.assert_array_equal(
            zoom.images[0].get_array(),
            full.images[0].get_array()[row:row + 4 * factor, col:col + 4 * factor],
        )
    np.testing.assert_array_equal(phase.object_wave, original)


def test_complex_phase_display_names_both_regularization_strengths():
    result = replace(
        _saved_result(), reconstruction_method="complex",
        reconstruction_options={"phase_damping": 0.03, "amplitude_damping": 0.1},
    )
    for show_parameters in (False, True):
        figure = result.show(show_parameters=show_parameters, size="large")
        title = figure.axes[0].get_title()
        assert "Amplitude + phase" in title
        assert "phase 0.03, amplitude 0.1" in title
        assert "phase (rad)" in title


@pytest.mark.parametrize("size", ["small", "medium", "large", "xlarge", 750, 900])
@pytest.mark.parametrize("columns", [1, 3])
def test_comparison_titles_stay_separate_and_inside_figure(size, columns):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    results = [replace(
        _saved_result(), reconstruction_method="phase",
        reconstruction_options={"phase_damping": value},
    ) for value in (0.01, 0.03, 0.05)]
    figure = results[0].show(
        compare=results[1:], region="bottom-left", show_fft=True, columns=columns, size=size,
    )
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    bounds = [axis.title.get_window_extent(renderer) for axis in figure.axes if axis.get_title()]
    bounds.append(figure._suptitle.get_window_extent(renderer))
    for index, box in enumerate(bounds):
        assert box.x0 >= 0 and box.x1 <= figure.bbox.x1
        assert box.y0 >= 0 and box.y1 <= figure.bbox.y1
        assert not any(box.overlaps(other) for other in bounds[index + 1:])


def test_compare_regularization_strengths_uses_one_contrast_and_region():
    baseline = _saved_result()
    original = baseline.object_wave.copy()
    results = [replace(
        baseline, object_wave=baseline.object_wave ** factor,
        reconstruction_method="phase", reconstruction_options={"phase_damping": strength},
        phase_clim=(-factor, factor),
    ) for strength, factor in ((0.01, 1), (0.03, 0.8), (0.05, 0.6))]
    figure = results[0].show(compare=results[1:], region="bottom-left", cmap="gray")
    axes = [axis for axis in figure.axes if axis.images]
    assert len(axes) == 6
    for result, full, zoom in zip(results, axes[:3], axes[3:], strict=True):
        assert f"Phase damping {result.reconstruction_options['phase_damping']:g}" in full.get_title()
        np.testing.assert_array_equal(zoom.images[0].get_array(), full.images[0].get_array()[12:16, :4])
        # The gallery must use the first result's limits even when another
        # result has its own contrast range saved.
        pair = results[0].show(compare=result, region="bottom-left", cmap="gray")
        pair_axes = [axis for axis in pair.axes if axis.images]
        np.testing.assert_array_equal(full.images[0].get_array(), pair_axes[1].images[0].get_array())
    np.testing.assert_array_equal(baseline.object_wave, original)


def test_crop_fft_uses_selected_region_and_preserves_relative_signal():
    """A lattice outside the crop cannot create peaks in the crop spectrum."""
    from quantem.gpu.direct_ptycho.results import _phase_spectra

    row, col = np.indices((128, 128))
    lattice = 0.2 * np.cos(2 * np.pi * col / 8)
    # The top-left lattice runs perpendicular to the rest of the image.
    lattice[:32, :32] = 0.2 * np.cos(2 * np.pi * row[:32, :32] / 8)
    result = replace(_saved_result(), object_wave=np.exp(1j * lattice), parameter_slices={})
    weaker = replace(result, object_wave=np.exp(0.5j * lattice))
    original = result.object_wave.copy()
    spectra, step = _phase_spectra([result.phase[:32, :32], weaker.phase[:32, :32]],
                                  [np.array([0.5, 0.5])] * 2)
    assert np.unravel_index(np.argmax(spectra[0]), spectra[0].shape) in ((12, 16), (20, 16))
    assert step == pytest.approx((1 / 16, 1 / 16))
    assert spectra[1].max() == pytest.approx(-6.0206, abs=1e-4)
    figure = result.show(compare=weaker, region="top-left", show_fft=True, columns=1, cmap="gray")
    images = [axis for axis in figure.axes if axis.images]
    assert len(images) == 6
    assert "Crop FFT" in images[2].get_title()
    assert "Crop FFT" in images[5].get_title()
    # Common limits preserve the weaker signal rather than stretching each FFT.
    assert images[2].images[0].get_array()[12, 16, 0] > images[5].images[0].get_array()[12, 16, 0]
    np.testing.assert_array_equal(result.object_wave, original)


@pytest.mark.parametrize("size,width", [("small", 300), ("medium", 450), ("large", 600), ("xlarge", 900), (750, 750), (900, 900)])
def test_interactive_phase_comparison_keeps_native_crops_and_shared_fft_scale(size, width):
    from ipywidgets import VBox
    from quantem.widget import Show2D

    result = _saved_result()
    widget = result.show(compare=[result, result], region="bottom-left",
                         interactive=True, show_fft=True, columns=2, size=size, cmap="gray")
    assert isinstance(widget, VBox)
    full, crop, fft = widget.children
    assert all(isinstance(viewer, Show2D) for viewer in widget.children)
    assert all(viewer.columns == 2 for viewer in widget.children)
    assert all(viewer.panel_width_px == width for viewer in widget.children)
    assert full.panel_overlays[0][0]["box"] == [11.5, -0.5, 15.5, 3.5]
    assert full.pixel_size == .5
    assert fft.pixel_size == .5  # 1 / (4 crop pixels × .5 Å per pixel)
    assert "Crop FFT" in fft.labels[0]
    # FFT and crops use separate galleries so Show2D never pads the crops
    # to the full-field shape before inspection or Fourier analysis.
    assert full._data.shape[-2:] == (16, 16)
    assert crop._data.shape[-2:] == fft._data.shape[-2:] == (4, 4)
    for viewer in widget.children:
        viewer.close()
    widget.close()
