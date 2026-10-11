"""The scan-to-detector rotation is one physical angle in [0, 360) degrees, with a detector transposition flag.

The CoM curl fixes the rotation only up to 180 degrees; a result holds the settled physical angle and, when a
polarity check ran, the angle before it. SSB reconstructs only an untransposed detector and says so.
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


def test_ssb_refuses_a_transposed_detector_instead_of_ignoring_it():
    """A DPC or screening fit that exchanged the detector axes cannot be handed to SSB as if it had not."""
    patterns = np.zeros((2, 2, 4, 4), np.uint16)
    with pytest.raises(NotImplementedError, match="detector_transposed=True"):
        SSB(patterns, voltage_kV=300, semiangle_mrad=20, scan_sampling_A=0.5, detector_transposed=True)
