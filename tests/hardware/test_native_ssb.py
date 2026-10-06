"""Encoded native datasets feed SSB without transferring source ownership.

``SSB(data)`` borrows the caller's dataset or array and never releases it;
``SSB.open`` owns the acquisition it loads and releases it once the
bright-field crop is read, and the crop when the session closes.
"""

import math
import os
import weakref

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import SSB, io

ABERRATIONS = {"C10": -11.52, "C12": 4.88, "phi12": -0.57}


@pytest.fixture
def backend() -> str:
    name = os.environ.get("QEM_TEST_BACKEND")
    if name not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND on a physical GPU.")
    return name


@pytest.fixture
def opened(monkeypatch) -> list:
    """Every dataset io.load returns during the test, so the test can check what SSB.open released."""
    loaded = []
    load = io.load

    def load_and_keep(*args, **kwargs):
        loaded.append(load(*args, **kwargs))
        return loaded[-1]

    monkeypatch.setattr(io, "load", load_and_keep)
    return loaded


def _counts(seed: int) -> np.ndarray:
    """128 x 128 scan of 12 x 12 detector counts inside a bright-field disk."""
    rows, cols = np.mgrid[:12, :12]
    disk = (rows - 5.5) ** 2 + (cols - 5.5) ** 2 < 16
    return (np.random.default_rng(seed).integers(1, 20, (128, 128, 12, 12), dtype=np.uint16) * disk).astype(np.uint16)


def _arina_master(path, counts: np.ndarray) -> str:
    """Write a synthetic Arina master: bitshuffle-LZ4 frames, which io.load keeps ANS encoded on the GPU."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data", data=counts.reshape(-1, *counts.shape[2:]),
            chunks=(1, *counts.shape[2:]), **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    return str(path)


def test_native_encoded_ssb_matches_dense_and_keeps_acquisition_open(tmp_path, backend):
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
    with io.load(path, backend=backend, verbose=False) as data:
        with SSB(data, **settings) as native, SSB(counts, **settings) as dense:
            actual, actual_loss = native.preview(ABERRATIONS)
            expected, expected_loss = dense.preview(ABERRATIONS)
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
            assert actual_loss == pytest.approx(expected_loss, abs=1e-7, rel=1e-5)
            assert native._data.shape[-2:] == (12, 12)
        np.testing.assert_array_equal(data[0, 0].cpu().numpy(), counts[0, 0])


@pytest.mark.slow
def test_detector_sampling_defaults_to_the_semiangle_over_the_disk_edge_radius(tmp_path, backend):
    """Without ``det_sampling`` every session calibrates the detector as semiangle / disk edge radius (mrad per pixel).

    The bright-field disk is the image of the aperture, so its edge sits at the convergence semiangle. Before 2026-10-05
    CUDA used twice the semiangle over an integer half-maximum radius, and MPS twice the semiangle over the equal-area
    radius, or over the half-maximum radius of the bright-field crop for a loaded dataset: about two times too coarse,
    and different per backend and input. An array, a loaded dataset and ``SSB.open`` now reconstruct exactly as a
    session given ``det_sampling = semiangle / radius``.
    """
    counts = _counts(5)
    rows, cols = np.mgrid[:12, :12]
    # the disk of _counts has a hard edge: every disk pixel is above half its plateau
    radius = math.sqrt(np.count_nonzero((rows - 5.5) ** 2 + (cols - 5.5) ** 2 < 16) / math.pi)
    settings = {"backend": backend, "voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": 0.99}
    master = _arina_master(tmp_path / "scan_master.h5", counts)
    with io.load(master, backend=backend, scan_shape=(128, 128), verbose=False) as loaded:
        pairs = [
            (SSB(counts, **settings), SSB(counts, det_sampling=30 / radius, **settings)),
            (SSB(loaded, **settings), SSB(loaded, det_sampling=30 / radius, **settings)),
            (SSB.open(master, scan_shape=(128, 128), **settings),
             SSB.open(master, scan_shape=(128, 128), det_sampling=30 / radius, **settings)),
        ]
        for automatic, calibrated in pairs:
            with automatic, calibrated:
                phase, loss = automatic.preview(ABERRATIONS)
                expected, expected_loss = calibrated.preview(ABERRATIONS)
                np.testing.assert_array_equal(phase, expected)
                assert loss == expected_loss
                result = automatic.reconstruct(aberrations=ABERRATIONS, phase_estimator="complex_wave")
                assert result.detected_bf_radius == pytest.approx(radius, rel=1e-12)


def test_encoded_storage_is_refused_and_the_callers_acquisition_survives_close(tmp_path, backend):
    """``SSB(data.data)`` fails at construction; closing ``SSB(data)``, before or after computing, leaves it readable."""
    counts = _counts(3)
    master = _arina_master(tmp_path / "scan_master.h5", counts)
    settings = {"backend": backend, "voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": 0.99, "det_sampling": 6.0}
    with io.load(master, backend=backend, scan_shape=(128, 128), verbose=False) as data:
        with pytest.raises(TypeError, match="not its encoded storage data.data"):
            SSB(data.data, **settings)
        SSB(data, **settings).close()
        with SSB(data, **settings) as session:
            session.preview(ABERRATIONS)
        assert not data.data.is_released
        np.testing.assert_array_equal(data[3, 5].cpu().numpy(), counts[3, 5])


def test_open_releases_the_acquisition_it_loads_and_its_crop_on_close(tmp_path, backend, opened):
    master = _arina_master(tmp_path / "scan_master.h5", _counts(4))
    with SSB.open(master, backend=backend, scan_shape=(128, 128), voltage_kV=300, semiangle_mrad=30,
                  scan_sampling_A=0.99, det_sampling=6.0, bf_radius=4) as session:
        assert opened[0].data.is_released
        session.preview(ABERRATIONS)
        crop = weakref.ref(session._data)
    assert crop() is None


def test_open_releases_the_acquisition_when_no_bright_field_disk_is_found(tmp_path, backend, opened):
    master = _arina_master(tmp_path / "dark_master.h5", np.zeros((128, 128, 12, 12), dtype=np.uint16))
    with pytest.raises(ValueError, match="No bright-field disk"):
        SSB.open(master, backend=backend, scan_shape=(128, 128), voltage_kV=300, semiangle_mrad=30,
                 scan_sampling_A=0.99, det_sampling=6.0)
    assert opened[0].data.is_released
