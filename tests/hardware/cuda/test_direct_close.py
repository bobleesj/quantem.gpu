"""Dataset changes release the previous CUDA direct ptychography engine and its spectra."""

import weakref

import numpy as np
import pytest

from .test_direct_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def test_close_releases_thick_engine_before_loading_another_dataset():
    """A saved phase survives close, while the engine's cyclic state does not."""
    direct, engine = _synthetic_session()
    options = dict(sample_tilt_mrad=(-9., 5.), depth_spread_nm=18.,
                   upsample=4, compute_loss=False)
    aberrations = {"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.}
    try:
        phase, _ = direct.preview(aberrations, **options)
        expected = phase.copy()
        references = [weakref.ref(value) for value in
                      (engine, engine.thick, engine.G_qk)]
        del engine
        direct.close()
        assert all(reference() is None for reference in references)
        np.testing.assert_array_equal(phase, expected)
        # Exercise the same close/reopen boundary used when browsing datasets.
        direct, engine = _synthetic_session()
        repeated, _ = direct.preview(aberrations, **options)
        np.testing.assert_array_equal(repeated, expected)
        del engine
    finally:
        direct.close()
