"""Host previews preserve full GPU values and independent image ownership."""

import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_host_preview_matches_gpu_and_survives_later_updates(factor):
    """Earlier NumPy images remain valid across defocus and sampling changes."""
    import cupy as cp

    ssb, _ = _synthetic_session()
    optics = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0,
              "C21_nm": 50.0, "phi21_deg": 23.0, "C50_nm": 800000.0}
    options = {"sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": 18.0,
               "compute_loss": False}
    try:
        native, _ = ssb.preview(optics, upsampling=factor, output="native", **options)
        expected = cp.asnumpy(native)
        host, _ = ssb.preview(optics, upsampling=factor, **options)
        assert isinstance(host, np.ndarray)
        assert host.dtype == np.float32
        np.testing.assert_array_equal(host, expected)
        for next_factor in (factor, 1, 4, factor):
            later, _ = ssb.preview(optics | {"C10_nm": 3.0},
                                   upsampling=next_factor, **options)
            assert not np.shares_memory(host, later)
            np.testing.assert_array_equal(host, expected)
        host[0, 0] = 123.0
        again, _ = ssb.preview(optics, upsampling=factor, **options)
        np.testing.assert_array_equal(again, expected)
    finally:
        ssb.close()
