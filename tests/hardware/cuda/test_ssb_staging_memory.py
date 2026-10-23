"""Coherent previews avoid per-detector image staging; phase means still reuse it."""

import math

import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def test_coherent_previews_do_not_retain_per_detector_images():
    """Repeated optics and tilt previews retain only their required scratch."""
    ssb, engine = _synthetic_session()
    try:
        # The fixture computed a phase-variance diagnostic. Start the next
        # coherent review without that diagnostic's per-detector staging.
        engine.release_reconstruction_buffers()
        engine.cache_rotation(engine._cached_scan_detector_rotation_rad, force=True)
        assert engine._result_buffer is None
        for tilt, depth in [((0.0, 0.0), 0.0), ((2.0, -1.0), 12.0)]:
            ssb.sample_tilt_mrad = tilt
            ssb.depth_spread_nm = depth
            options = {
                "aberrations": {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0},
                "upsampling_factor": "auto", "compute_loss": False,
            }
            first = ssb.reconstruct(**options)
            repeated = ssb.reconstruct(**options, force=True)
            np.testing.assert_array_equal(first.phase.get(), repeated.phase.get())
            assert engine._result_buffer is None
    finally:
        ssb.close()


def test_phase_mean_staging_survives_rotation_round_trip():
    """Changing scan rotation reuses staging without changing restored phase/loss."""
    ssb, engine = _synthetic_session()
    try:
        expected, expected_loss = engine.reconstruct_with_loss(-80.0, 40.0, -0.7)
        expected = expected.copy()
        pointer = engine._result_buffer.data.ptr
        angle = engine._cached_scan_detector_rotation_rad
        engine.cache_rotation(angle + math.pi)
        engine.reconstruct_with_loss(-80.0, 40.0, -0.7)
        assert engine._result_buffer.data.ptr == pointer
        engine.cache_rotation(angle)
        actual, actual_loss = engine.reconstruct_with_loss(-80.0, 40.0, -0.7)
        assert engine._result_buffer.data.ptr == pointer
        np.testing.assert_array_equal(actual.get(), expected.get())
        assert actual_loss == expected_loss
    finally:
        ssb.close()
