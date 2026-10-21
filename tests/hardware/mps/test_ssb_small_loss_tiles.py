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
    with _disk_session(n=size, detector_sampling=(5.0, 5.0)) as session:
        options = dict(sample_tilt_mrad=(2.0, -1.0), depth_spread_nm=12.0,
                       phase_estimator="mean_phase", compute_loss=True)
        monkeypatch.setattr(reconstruct, "LOSS_ROW_IFFT_TILE_BF", None)
        phase, loss = session.preview(optics, **options)
        # Two full BF groups followed by a partial group use the original
        # reduction shape, even when the memory tile is not a group multiple.
        for tile in (32, 65):
            monkeypatch.setattr(reconstruct, "LOSS_ROW_IFFT_TILE_BF", tile)
            actual_phase, actual_loss = session.preview(optics, **options)
            np.testing.assert_array_equal(actual_phase, phase)
            assert actual_loss == loss
