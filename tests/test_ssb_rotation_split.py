"""The scan-detector rotation is reported below 180 degrees plus ``com_reversed``; the physical angle is unchanged.

The CoM curl fixes the rotation only up to 180 degrees, and rotating by 180 degrees is the same as reversing every CoM
vector. Old saves and callers that give an angle up to 360 must land on the same physical rotation.
"""

import numpy as np
import pytest

from quantem.gpu.ssb.results import SSBResult, physical_rotation_deg, split_rotation

# ---


@pytest.mark.parametrize(
    ("angle", "com_reversed", "expected"),
    [
        (165.5, False, (165.5, False)),
        (345.5, False, (165.5, True)),     # an older save that stored the physical angle
        (165.5, True, (165.5, True)),
        (345.5, True, (165.5, False)),     # reversing twice restores the CoM
        (-14.5, False, (165.5, True)),     # quantem's CoM search range starts at -89 degrees
        (0.0, False, (0.0, False)),
        (180.0, False, (0.0, True)),
        (360.0, False, (0.0, False)),
    ],
)
def test_split_rotation_keeps_the_physical_angle(angle, com_reversed, expected):
    rotation, reversed_ = split_rotation(angle, com_reversed)
    assert (rotation, reversed_) == pytest.approx(expected)
    assert 0.0 <= rotation < 180.0
    assert physical_rotation_deg(rotation, reversed_) == pytest.approx((angle + 180.0 * com_reversed) % 360.0)


def test_result_from_an_older_save_reads_as_angle_below_180_plus_reversed():
    """SSBResult(**saved) of a schema-4 save holding 345.5 degrees: same physical rotation, new public form."""
    result = SSBResult(object_wave=np.ones((4, 4), np.complex64), backend="mps", rotation_angle_deg=345.5)
    assert result.rotation_angle_deg == pytest.approx(165.5) and result.com_reversed
    assert result.physical_rotation_deg == pytest.approx(345.5)
    assert "CoM reversed" in repr(result)
