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

ABERRATIONS = {"C10_nm": -11.52, "C12_nm": 4.88, "phi12_deg": math.degrees(-0.57)}


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
        scan_sampling_A=(0.99, 0.99), detector_sampling_mrad=(6.0, 6.0),
        bright_field_radius_px=4.0, bright_field_center_px=(11.5, 11.5),
    )
    with io.load(path, backend=backend, verbose=False) as data:
        with SSB(data, **settings) as native, SSB(counts, **settings) as dense:
            actual, actual_loss = native.preview(ABERRATIONS)
            expected, expected_loss = dense.preview(ABERRATIONS)
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
            assert actual_loss == pytest.approx(expected_loss, abs=1e-7, rel=1e-5)
            assert native._data.shape[-2:] == (12, 12)
            # the centre counts from the first pixel of the full detector, also in the session that decoded only its
            # crop; that session used to report the centre inside the crop
            assert native.bright_field_center_px == dense.bright_field_center_px == (11.5, 11.5)
            cropped_result = native.reconstruct(aberrations=ABERRATIONS, phase_estimator="complex_wave")
            dense_result = dense.reconstruct(aberrations=ABERRATIONS, phase_estimator="complex_wave")
            assert cropped_result.bright_field_center_px == dense_result.bright_field_center_px == (11.5, 11.5)
        np.testing.assert_array_equal(data[0, 0].cpu().numpy(), counts[0, 0])


@pytest.mark.slow
def test_detector_sampling_defaults_to_the_semiangle_over_the_disk_edge_radius(tmp_path, backend):
    """Without ``detector_sampling_mrad`` every session calibrates the detector as semiangle / disk edge radius (mrad per pixel).

    The bright-field disk is the image of the aperture, so its edge sits at the convergence semiangle. Before 2026-10-05
    CUDA used twice the semiangle over an integer half-maximum radius, and MPS twice the semiangle over the equal-area
    radius, or over the half-maximum radius of the bright-field crop for a loaded dataset: about two times too coarse,
    and different per backend and input. An array, a loaded dataset and ``SSB.open`` now reconstruct exactly as a
    session given ``detector_sampling_mrad = semiangle / radius``.
    """
    counts = _counts(5)
    rows, cols = np.mgrid[:12, :12]
    # the disk of _counts has a hard edge: every disk pixel is above half its plateau
    radius = math.sqrt(np.count_nonzero((rows - 5.5) ** 2 + (cols - 5.5) ** 2 < 16) / math.pi)
    settings = {"backend": backend, "voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": (0.99, 0.99)}
    master = _arina_master(tmp_path / "scan_master.h5", counts)
    with io.load(master, backend=backend, scan_shape=(128, 128), verbose=False) as loaded:
        pairs = [
            (SSB(counts, **settings), SSB(counts, detector_sampling_mrad=(30 / radius, 30 / radius), **settings)),
            (SSB(loaded, **settings), SSB(loaded, detector_sampling_mrad=(30 / radius, 30 / radius), **settings)),
            (SSB.open(master, scan_shape=(128, 128), **settings),
             SSB.open(master, scan_shape=(128, 128), detector_sampling_mrad=(30 / radius, 30 / radius), **settings)),
        ]
        for automatic, calibrated in pairs:
            with automatic, calibrated:
                phase, loss = automatic.preview(ABERRATIONS)
                expected, expected_loss = calibrated.preview(ABERRATIONS)
                np.testing.assert_array_equal(phase, expected)
                assert loss == expected_loss
                result = automatic.reconstruct(aberrations=ABERRATIONS, phase_estimator="complex_wave")
                assert result.bright_field_edge_radius_px == pytest.approx(radius, rel=1e-12)


def test_encoded_storage_is_refused_and_the_callers_acquisition_survives_close(tmp_path, backend):
    """``SSB(data.data)`` fails at construction; closing ``SSB(data)``, before or after computing, leaves it readable."""
    counts = _counts(3)
    master = _arina_master(tmp_path / "scan_master.h5", counts)
    settings = {"backend": backend, "voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": (0.99, 0.99), "detector_sampling_mrad": (6.0, 6.0)}
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
                  scan_sampling_A=(0.99, 0.99), detector_sampling_mrad=(6.0, 6.0), bright_field_radius_px=4) as session:
        assert opened[0].data.is_released
        session.preview(ABERRATIONS)
        crop = weakref.ref(session._data)
    assert crop() is None


def test_open_releases_the_acquisition_when_no_bright_field_disk_is_found(tmp_path, backend, opened):
    master = _arina_master(tmp_path / "dark_master.h5", np.zeros((128, 128, 12, 12), dtype=np.uint16))
    with pytest.raises(ValueError, match="No bright-field disk"):
        SSB.open(master, backend=backend, scan_shape=(128, 128), voltage_kV=300, semiangle_mrad=30,
                 scan_sampling_A=(0.99, 0.99), detector_sampling_mrad=(6.0, 6.0))
    assert opened[0].data.is_released


@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_open_retains_detector_correction_and_matches_saved_counts(tmp_path, backend, depth_nm):
    """Reopening raw H5 and corrected QEM keeps the same SSB phase and loss."""
    counts = _counts(7).astype(np.uint32)
    counts[..., 5, 5] = np.iinfo(np.uint32).max
    master = _arina_master(tmp_path / "masked_master.h5", counts)
    mask = np.zeros((12, 12), dtype=np.uint32)
    mask[5, 5] = 1
    with h5py.File(master, "a") as handle:
        handle.create_dataset("entry/instrument/detector/pixel_mask", data=mask)
    settings = dict(
        backend=backend, voltage_kV=300, semiangle_mrad=30,
        scan_sampling_A=(0.99, 0.99), detector_sampling_mrad=(6.0, 6.0),
        bright_field_radius_px=4.0, bright_field_center_px=(5.5, 5.5),
    )
    preview = dict(
        upsampling_factor=3, sample_tilt_mrad=(-2.0, 3.0), depth_spread_nm=depth_nm,
    )
    # The acquisition loader remains lossless; the SSB file workflow owns its
    # explicit preprocessing choice and never changes the source file.
    with io.load(master, backend=backend, scan_shape=(128, 128), verbose=False) as raw:
        np.testing.assert_array_equal(raw[0, 0].cpu().numpy(), counts[0, 0])
    with io.load(
        master, backend=backend, scan_shape=(128, 128),
        hot_pixel_correction="median", verbose=False,
    ) as corrected:
        saved = tmp_path / "corrected.qem"
        io.save(saved, corrected)
        with SSB(corrected, **settings) as expected:
            expected_phase, expected_loss = expected.preview(ABERRATIONS, **preview)
    for source in (master, str(saved)):
        with SSB.open(source, scan_shape=(128, 128), **settings) as actual:
            phase, loss = actual.preview(ABERRATIONS, **preview)
            np.testing.assert_array_equal(phase, expected_phase)
            assert loss == expected_loss
            assert actual.source_provenance["hot_pixel_correction"]["method"] == "median"
    with h5py.File(master) as handle:
        np.testing.assert_array_equal(handle["entry/data/data"][0], counts[0, 0])


def test_one_number_is_the_scan_step_on_both_axes_at_every_entry_point(tmp_path, backend):
    """SSB(data), a loaded dataset and SSB.open take scan_sampling_A=0.99 as (0.99, 0.99) and report the pair."""
    counts = _counts(3)
    master = _arina_master(tmp_path / "scan_master.h5", counts)
    scalar = {"backend": backend, "voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": 0.99,
              "detector_sampling_mrad": 0.6}
    paired = {**scalar, "scan_sampling_A": (0.99, 0.99), "detector_sampling_mrad": (0.6, 0.6)}
    with io.load(master, backend=backend, scan_shape=(128, 128), verbose=False) as loaded:
        pairs = [
            (SSB(counts, **scalar), SSB(counts, **paired)),
            (SSB(loaded, **scalar), SSB(loaded, **paired)),
            (SSB.open(master, scan_shape=(128, 128), **scalar), SSB.open(master, scan_shape=(128, 128), **paired)),
        ]
        for one, two in pairs:
            with one, two:
                assert one.scan_sampling_A == two.scan_sampling_A == (0.99, 0.99)
                assert one.detector_sampling_mrad == two.detector_sampling_mrad == (0.6, 0.6)
                assert one.calibration_table().loc["Scan step (row, col)", "Value used"] == (0.99, 0.99)
                np.testing.assert_array_equal(one.preview(ABERRATIONS)[0], two.preview(ABERRATIONS)[0])
    with pytest.raises(ValueError, match="one number for both axes or a \\(row, col\\) pair"):
        SSB(counts, **{**scalar, "scan_sampling_A": (0.99, 0.99, 0.99)})
