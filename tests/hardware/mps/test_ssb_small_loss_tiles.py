"""Bounded row FFTs retain the original phase and diagnostic loss."""

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps
from .test_ssb_upsampling import _disk_session


@pytest.mark.parametrize("size", [128, 256, 1024])
def test_tiled_diagnostic_preserves_every_phase_and_loss_bit(size, monkeypatch):
    """Review a fitted sample with the same loss while bounding its FFT stack."""
    _require_mps()
    from quantem.gpu.ssb.mps import reconstruct

    optics = {"C10_nm": 8.0, "C12_nm": 1.0, "phi12_deg": 30.0}
    # the same optics in the backend's units, and the sample as the phase-variance objective takes it
    aberrations = {"C10": 80.0, "C12": 10.0, "phi12": float(np.radians(30.0))}
    sample = {"tilt_row_mrad": 2.0, "tilt_col_mrad": -1.0, "depth_spread_A": 120.0}
    with _disk_session(n=size, detector_sampling=(5.0, 5.0)) as session:
        monkeypatch.setattr(reconstruct, "LOSS_ROW_IFFT_TILE_BF", None)
        _, preview_loss = session.preview(optics, sample_tilt_mrad=(2.0, -1.0), depth_spread_nm=12.0)
        backend = session._backend()
        phase, loss = backend.sample_phase_variance(aberrations, sample)
        assert preview_loss == loss
        # Two full BF groups followed by a partial group use the original
        # reduction shape, even when the memory tile is not a group multiple.
        for tile in (32, 65):
            monkeypatch.setattr(reconstruct, "LOSS_ROW_IFFT_TILE_BF", tile)
            actual_phase, actual_loss = backend.sample_phase_variance(aberrations, sample)
            np.testing.assert_array_equal(actual_phase, phase)
            assert actual_loss == loss
