"""Mean diffraction, disk fitting and virtual images on encoded data."""

import os

import numpy as np
import pytest

from quantem.gpu import detector, io


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_fit_once_for_encoded_detector_images(tmp_path, dtype):
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
    for path in (original, saved):
        with io.load(path, backend=backend, verbose=False) as data:
            mean_dp = detector.mean(data)
            np.testing.assert_allclose(mean_dp, values.mean(axis=(0, 1)), rtol=1e-6)
            center, radius = detector.fit_probe(mean_dp)
            assert center == (11.0, 23.0)
            distance = np.hypot(rows - center[0], columns - center[1])
            for image, selected in (
                (detector.bf(data, center=center, radius=radius), distance <= radius),
                (detector.adf(data, center=center, radius=radius),
                 (distance >= radius) & (distance <= 2 * radius)),
                (detector.df(data, center=center, radius=radius), distance >= radius),
            ):
                np.testing.assert_array_equal(image, values[..., selected].sum(axis=-1))
            if path == original:
                io.save(saved, data)
