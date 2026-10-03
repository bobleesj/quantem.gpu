"""Encoded native datasets feed SSB without transferring source ownership."""

import os

import numpy as np
import pytest

from quantem.gpu import SSB, io


def test_native_encoded_ssb_matches_dense_and_keeps_acquisition_open(tmp_path):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND on a physical GPU.")
    rows, cols = np.mgrid[:24, :24]
    disk = (rows - 11.5) ** 2 + (cols - 11.5) ** 2 < 16
    counts = np.random.default_rng(42).integers(
        1, 20, (128, 128, 24, 24), dtype=np.uint16
    ) * disk
    path = tmp_path / "counts.npy"
    np.save(path, counts)
    settings = dict(
        backend=backend, voltage_kV=300, semiangle_mrad=30,
        scan_sampling_A=0.99, det_sampling=6.0,
        bf_radius=4.0, bf_center=(11.5, 11.5),
    )
    aberrations = {"C10": -11.52, "C12": 4.88, "phi12": -0.57}
    with io.load(path, backend=backend, verbose=False) as data:
        with SSB(data, **settings) as native, SSB(counts, **settings) as dense:
            actual, actual_loss = native.preview(aberrations)
            expected, expected_loss = dense.preview(aberrations)
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
            assert actual_loss == pytest.approx(expected_loss, abs=1e-7, rel=1e-5)
            assert native._data.shape[-2:] == (12, 12)
        np.testing.assert_array_equal(data[0, 0].numpy(), counts[0, 0])
