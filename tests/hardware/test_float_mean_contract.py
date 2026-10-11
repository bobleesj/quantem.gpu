"""Float32 acquisitions give the float64 mean pattern, rounded once to float32, on CUDA and Metal.

A float32 running total drifts: over the 16384 patterns of a 128 x 128 EMPAD
scan it ended up to 8 units in the last place off the float64 mean on Metal and
4 on CUDA. CUDA accumulates in float64; Metal has none and keeps each pixel's
float32 sum with its rounding residuals, which add up to the same total. The
reference is NumPy float64 over the loaded values, so a pass on each machine
means both backends return the same bits.

Real acquisitions are checked when ``QEM_FLOAT_MEAN_SOURCES`` lists float32
files (``os.pathsep`` between them), for example an EMPAD RAW scan and a
background-subtracted EMPAD HDF5.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import detector, io

SCAN_SHAPE = (40, 37)
DETECTOR_SHAPE = (12, 10)


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return backend


def _intensities():
    """Background-subtracted intensities over six decades: a bright disk, a dim tail and negative noise."""
    rng = np.random.default_rng(23)
    rows, cols = np.indices(DETECTOR_SHAPE)
    disk = np.hypot(rows - 5.3, cols - 4.6) < 3.2
    level = np.where(disk, 4.0e6, 17.0)[None, None]
    return (level * rng.gamma(9.0, 1 / 9.0, (*SCAN_SHAPE, *DETECTOR_SHAPE)) + rng.normal(0.0, 3.0, (*SCAN_SHAPE, *DETECTOR_SHAPE))).astype(np.float32)


def test_mean_pattern_and_scan_region_means_equal_the_float64_reference(tmp_path):
    backend = _backend()
    intensities = _intensities()
    frames = intensities.reshape(-1, *DETECTOR_SHAPE).astype(np.float64)
    np.save(tmp_path / "intensities.npy", intensities)
    # 1480 scan positions cross two 512-scan chunk boundaries of the stored streams.
    region = np.flatnonzero(np.hypot(*(np.indices(SCAN_SHAPE) - np.array([19.0, 18.0])[:, None, None])) < 14.5)
    with io.load(tmp_path / "intensities.npy", backend=backend) as loaded:
        assert loaded.dtype == np.float32
        session = detector.prepare(loaded)
        mean_dp = session.mean_dp()
        expected = (frames.sum(axis=0) / len(frames)).astype(np.float32)
        # A float32 running total is what this rules out: it differs from the reference here.
        running = np.zeros(DETECTOR_SHAPE, np.float32)
        for frame in intensities.reshape(-1, *DETECTOR_SHAPE):
            running += frame
        assert np.any(running / np.float32(len(frames)) != expected)
        np.testing.assert_array_equal(mean_dp, expected)
        np.testing.assert_array_equal(session.mean_dp(output="native").get() if backend == "cuda" else session.mean_dp(output="native").cpu().numpy(), expected)
        np.testing.assert_array_equal(session.reduce_frames(np.arange(len(frames)), "mean"), expected)
        np.testing.assert_array_equal(session.reduce_frames(region, "mean"), (frames[region].sum(axis=0) / len(region)).astype(np.float32))
        np.testing.assert_array_equal(session.reduce_frames(region, "sum"), frames[region].sum(axis=0).astype(np.float32))
        np.testing.assert_array_equal(session.reduce_frames(region, "max"), intensities.reshape(-1, *DETECTOR_SHAPE)[region].max(axis=0))


def test_real_float_acquisitions_equal_the_float64_reference():
    backend = _backend()
    sources = [Path(source) for source in os.environ.get("QEM_FLOAT_MEAN_SOURCES", "").split(os.pathsep) if source]
    if not sources:
        pytest.skip("Set QEM_FLOAT_MEAN_SOURCES to float32 acquisitions (EMPAD RAW, background-subtracted EMPAD HDF5).")
    for source in sources:
        with io.load(source, backend=backend, verbose=False) as loaded:
            assert loaded.dtype == np.float32, source.name
            scan_rows, scan_cols = loaded.shape[:2]
            total = np.zeros(loaded.shape[2:], np.float64)
            region_total = np.zeros(loaded.shape[2:], np.float64)
            region_rows, region_cols = slice(scan_rows // 4, scan_rows // 2), slice(scan_cols // 3, scan_cols - 5)
            for row in range(scan_rows):
                block = loaded.read(scan_region=(row, row + 1, 0, scan_cols)).cpu().numpy().astype(np.float64)[0]
                total += block.sum(axis=0)
                if region_rows.start <= row < region_rows.stop:
                    region_total += block[region_cols].sum(axis=0)
            session = detector.prepare(loaded)
            np.testing.assert_array_equal(session.mean_dp(), (total / (scan_rows * scan_cols)).astype(np.float32), err_msg=source.name)
            indices = (np.arange(scan_rows)[region_rows, None] * scan_cols + np.arange(scan_cols)[None, region_cols]).ravel()
            np.testing.assert_array_equal(
                session.reduce_frames(indices, "mean"), (region_total / len(indices)).astype(np.float32), err_msg=source.name
            )


def test_scan_region_over_released_rows_raises(tmp_path):
    """A released row is gone: skipping its patterns gave a mean over fewer patterns than it was divided by."""
    backend = _backend()
    intensities = (np.random.default_rng(29).gamma(3.0, 50.0, (4, 512, 8, 8))).astype(np.float32)
    frames = intensities.reshape(-1, 8, 8).astype(np.float64)
    np.save(tmp_path / "intensities.npy", intensities)
    with io.load(tmp_path / "intensities.npy", backend=backend, verbose=False) as loaded:
        session = detector.prepare(loaded)
        loaded.release_rows_before(2)
        kept = np.arange(2 * 512 + 5, 3 * 512 + 300)
        np.testing.assert_array_equal(session.reduce_frames(kept, "mean"), (frames[kept].sum(axis=0) / len(kept)).astype(np.float32))
        for mode in ("mean", "sum", "max"):
            with pytest.raises(ValueError, match="released"):
                session.reduce_frames(np.arange(500, 1100), mode)
