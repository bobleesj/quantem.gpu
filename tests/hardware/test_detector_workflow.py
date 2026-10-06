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
                detector.bf(data, center=center, radius=3),
                values[..., distance <= 3].sum(axis=-1),
            )
            assert len(fits) == fit_count  # Fully specified geometry skips fitting.
            np.testing.assert_array_equal(
                detector.bf(data, radius=3),
                values[..., distance <= 3].sum(axis=-1),
            )
            for image, selected in (
                (detector.bf(data), distance <= radius),
                (detector.adf(data),
                 (distance >= radius) & (distance <= 2 * radius)),
                (detector.df(data), distance >= radius),
                (detector.adf(data, inner=6, outer=9, unit="px"),
                 (distance >= 6) & (distance <= 9)),
            ):
                np.testing.assert_array_equal(image, values[..., selected].sum(axis=-1))
            shifted_center = (center[0] + 2, center[1] - 1)
            shifted_distance = np.hypot(
                rows - shifted_center[0], columns - shifted_center[1]
            )
            np.testing.assert_array_equal(
                detector.bf(data, center=shifted_center),
                values[..., shifted_distance <= radius].sum(axis=-1),
            )
            np.testing.assert_array_equal(
                detector.bf(data), values[..., distance <= radius].sum(axis=-1)
            )
            assert len(fits) == fit_count + 1
            if path == original:
                io.save(saved, data)
    assert len(fits) == 2  # Reopening gets its own fit, not a serialized cache.
