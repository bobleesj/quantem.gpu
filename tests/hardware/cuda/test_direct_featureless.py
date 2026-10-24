"""A featureless scan has a finite wave and no defined relative error."""

import json

import cupy as cp
import pytest

from quantem.gpu import DirectPtycho

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("method", ["phase", "complex"])
def test_featureless_acquisition_reports_undefined_prediction_error(method):
    # A calibrated, constant vacuum diffraction pattern has exactly zero
    # non-DC scan signal. It must not be reported as a perfect validation fit.
    row, col = cp.indices((16, 16))
    pattern = (100 * ((row - 7.5) ** 2 + (col - 7.5) ** 2 < 25)).astype(cp.float32)
    counts = cp.broadcast_to(pattern, (128, 128, 16, 16)).copy()
    with DirectPtycho(
        counts, backend="cuda", voltage_kV=300., semiangle_mrad=20.,
        scan_sampling_A=0.3, detector_sampling_mrad=4.,
        bright_field_center_px=(7.5, 7.5), bright_field_radius_px=5.,
        scan_detector_rotation_deg=0., aberrations={"C10_nm": 0., "C12_nm": 0., "phi12_deg": 0.},
    ) as ptycho:
        result = ptycho.run(method=method, compute_loss=True, verbose=False)
        cp.testing.assert_array_equal(result.object_wave, cp.ones((128, 128), cp.complex64))
        diagnostics = result.reconstruction_diagnostics
        assert diagnostics["training_relative_sse"] is None
        assert diagnostics["prediction_error_status"] == "undefined: no non-DC signal"
        assert result.loss is None
        json.dumps(diagnostics, allow_nan=False)
