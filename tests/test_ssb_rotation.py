"""The scan-to-detector rotation is one physical angle in [0, 360) degrees, with a detector transposition flag.

The CoM curl fixes the rotation only up to 180 degrees; a result holds the settled physical angle and, when a
polarity check ran, the angle before it. Transposition exchanges the detector axes before rotation.
"""

import numpy as np
import pytest

from quantem.gpu.ssb import SSB
from quantem.gpu.ssb.results import SSBResult

# ---


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (165.5, 165.5),
        (345.5, 345.5),
        (-14.5, 345.5),     # quantem's CoM search range starts at -89 degrees
        (0.0, 0.0),
        (180.0, 180.0),
        (360.0, 0.0),
        (525.5, 165.5),
        (-2e-15, 0.0),      # the plain modulo rounds this to 360.0, outside the range
    ],
)
def test_result_holds_one_physical_rotation(angle, expected):
    result = SSBResult(object_wave=np.ones((4, 4), np.complex64), backend="mps", scan_detector_rotation_deg=angle)
    assert result.scan_detector_rotation_deg == pytest.approx(expected)
    assert 0.0 <= result.scan_detector_rotation_deg < 360.0
    assert result.detector_transposed is False
    assert f"{expected:.2f}°" in repr(result)


@pytest.mark.parametrize("retired", [{"rotation_angle_deg": 165.5}, {"com_reversed": True}])
def test_angle_below_180_with_reversal_flag_is_not_a_result_field(retired):
    """A record of the earlier form is not read as the physical angle: its two names are refused."""
    with pytest.raises(TypeError):
        SSBResult(object_wave=np.ones((4, 4), np.complex64), backend="mps", **retired)


def test_transposition_preserves_original_detector_coordinates(monkeypatch):
    """A rectangular crop exchanges axes without changing the public calibration."""
    monkeypatch.setattr("quantem.gpu.ssb.workflow.resolve", lambda backend: "mps")
    patterns = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    session = SSB(patterns, backend="mps", voltage_kV=300, semiangle_mrad=20,
                  scan_sampling_A=(.5, .6), detector_sampling_mrad=(2., 3.),
                  bright_field_center_px=(11.5, 22.), detector_transposed=True)
    session._crop_origin = (10, 20)
    np.testing.assert_array_equal(session._data, patterns.swapaxes(-1, -2))
    assert np.shares_memory(session._data, patterns)
    assert session._center_in_data() == (2., 1.5)
    assert session._sampling_in_data() == (3., 2.)
    result = SSBResult(object_wave=np.ones((2, 3), np.complex64), backend="mps",
                       bright_field_center_px=(2., 1.5))
    session._in_detector_frame(result)
    assert result.bright_field_center_px == (11.5, 22.)
    assert result.detector_transposed is True
