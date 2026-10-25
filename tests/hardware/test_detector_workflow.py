"""Mean diffraction, disk fitting and virtual images on encoded data."""

import os

import numpy as np
import pytest

from quantem.gpu import detector, io


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_fit_once_for_encoded_detector_images(tmp_path, dtype, monkeypatch):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    rows, columns = np.indices((32, 40))
    disk = (rows - 11) ** 2 + (columns - 23) ** 2 <= 25
    pattern = np.where(disk, 100, 2)
    values = (np.arange(1, 21).reshape(4, 5, 1, 1) * pattern).astype(dtype)
    if dtype == np.float32:
        values = values / 8 - 3
    original = tmp_path / "input.npy"
    np.save(original, values)
    saved = tmp_path / "scan.qem"
    fits = []
    mean = detector.workflow.mean

    def record_mean(data):
        fits.append(data.shape)
        return mean(data)

    monkeypatch.setattr(detector.workflow, "mean", record_mean)
    for path in (original, saved):
        with io.load(path, backend=backend, verbose=False) as data:
            mean_dp = detector.mean(data)
            np.testing.assert_allclose(mean_dp, values.mean(axis=(0, 1)), rtol=1e-6)
            center, radius = detector.fit_probe(mean_dp)
            assert center == (11.0, 23.0)
            distance = np.hypot(rows - center[0], columns - center[1])
            fit_count = len(fits)
            np.testing.assert_array_equal(
                detector.bf(data, bright_field_center_px=center, bright_field_radius_px=3),
                values[..., distance <= 3].sum(axis=-1),
            )
            assert len(fits) == fit_count  # Fully specified geometry skips fitting.
            np.testing.assert_array_equal(
                detector.bf(data, bright_field_radius_px=3),
                values[..., distance <= 3].sum(axis=-1),
            )
            for image, selected in (
                (detector.bf(data), distance <= radius),
                (detector.adf(data),
                 (distance >= radius) & (distance <= 2 * radius)),
                (detector.df(data), distance >= radius),
                (detector.adf(data, inner_radius_px=6, outer_radius_px=9),
                 (distance >= 6) & (distance <= 9)),
            ):
                np.testing.assert_array_equal(image, values[..., selected].sum(axis=-1))
            shifted_center = (center[0] + 2, center[1] - 1)
            shifted_distance = np.hypot(
                rows - shifted_center[0], columns - shifted_center[1]
            )
            np.testing.assert_array_equal(
                detector.bf(data, bright_field_center_px=shifted_center),
                values[..., shifted_distance <= radius].sum(axis=-1),
            )
            np.testing.assert_array_equal(
                detector.bf(data), values[..., distance <= radius].sum(axis=-1)
            )
            assert len(fits) == fit_count + 1
            if path == original:
                io.save(saved, data)
    assert len(fits) == 2  # Reopening gets its own fit, not a serialized cache.


def test_show_detector_images_and_save_without_changing_intensities(tmp_path):
    """Display and save the same detector intensities as the explicit workflow."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    import matplotlib.pyplot as plt
    from quantem.core.visualization import show_2d

    rows, columns = np.indices((32, 40))
    pattern = np.where((rows - 11) ** 2 + (columns - 23) ** 2 <= 25, 100, 2)
    counts = (np.arange(1, 21).reshape(4, 5, 1, 1) * pattern).astype(np.uint16)
    source = tmp_path / "input.npy"
    np.save(source, counts)
    with io.load(source, backend=backend, verbose=False) as data:
        expected = [detector.bf(data), detector.df(data), detector.adf(data),
                    detector.mean(data)]
        for cmap in ("inferno", "gray"):
            path = tmp_path / f"detectors-{cmap}.png"
            reference_path = tmp_path / f"explicit-{cmap}.png"
            reference_figure, _ = show_2d(
                [[expected[0], expected[1]], [expected[2], expected[3]]],
                title=[["Bright field (BF)", "Dark field (DF)"],
                       ["Annular dark field (ADF)", "Mean diffraction"]],
                cmap=cmap,
                norm=[["power_sqrt", "power_sqrt"], ["power_sqrt", "log_auto"]],
                axsize=(4.5, 4.5), save=reference_path,
            )
            plt.close(reference_figure)
            reference_images = [axis.images[0] for axis in reference_figure.axes
                                if axis.images]
            figure = detector.show(data, size="medium", columns=2, cmap=cmap, save=path)
            images = [axis.images[0] for axis in figure.axes if axis.images]
            assert [image.axes.get_title() for image in images] == [
                "Bright field (BF)", "Dark field (DF)", "Annular dark field (ADF)",
                "Mean diffraction"
            ]
            for image, reference in zip(images, reference_images, strict=True):
                np.testing.assert_array_equal(image.get_array(), reference.get_array())
            assert not plt.fignum_exists(figure.number)
            np.testing.assert_array_equal(plt.imread(path), plt.imread(reference_path))
        for size, inches, requested_columns in (
            ("small", 3, 1), ("medium", 4.5, 2),
            ("large", 6, 3), ("xlarge", 9, 8),
        ):
            figure = detector.show(data, size=size, columns=requested_columns,
                                   norm="linear_auto", diffraction_norm="linear_auto")
            axes = [axis for axis in figure.axes if axis.images]
            rows, columns = axes[0].get_subplotspec().get_gridspec().get_geometry()
            assert (rows, columns) == (int(np.ceil(4 / min(requested_columns, 4))), min(requested_columns, 4))
            np.testing.assert_allclose(figure.get_size_inches(),
                                       [columns * inches, rows * inches])
            assert len(axes) == 4
            for axis, array in zip(axes, expected, strict=True):
                assert axis.images[0].get_array().shape[:2] == array.shape
            assert not plt.fignum_exists(figure.number)
        for actual, reference in zip(
            (detector.bf(data), detector.df(data), detector.adf(data), detector.mean(data)),
            expected, strict=True,
        ):
            np.testing.assert_array_equal(actual, reference)
        row = detector.show(data, rows=1)
        assert row.axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 4)
        with pytest.raises(ValueError, match="cannot hold 4 panels"):
            detector.show(data, rows=1, columns=2)


@pytest.mark.parametrize("options", [
    {"size": "huge"}, {"size": 0}, {"size": 450.5}, {"columns": 0},
    {"columns": -1}, {"columns": True}, {"columns": 2.5},
])
def test_show_rejects_invalid_layout_before_reading_data(options):
    with pytest.raises(ValueError, match="size=|positive integer"):
        detector.show(None, **options)
