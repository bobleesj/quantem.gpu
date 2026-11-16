"""2026-11-16 on the microscope: six ARINA pixels flagged at 2**16 - 1 made the bright-field disk 1.4 px wide."""

import numpy as np

from quantem.gpu import detector


def test_flagged_pixels_are_not_the_bright_field_disk():
    rows, cols = np.mgrid[:192, :192]
    pattern = np.where((rows - 104) ** 2 + (cols - 103) ** 2 <= 43 ** 2, 50.0, 0.07)
    clean = detector.bright_field_disk(pattern, semiangle_mrad=30)
    for row, col in [(21, 73), (101, 88), (125, 52), (142, 70), (143, 150), (164, 191)]:
        pattern[row, col] = 65535.0
    flagged = detector.bright_field_disk(pattern, semiangle_mrad=30)
    assert abs(flagged.bright_field_edge_radius_px - 43) < 1.5
    assert abs(flagged.bright_field_edge_radius_px - clean.bright_field_edge_radius_px) < 0.1
    assert np.allclose(flagged.bright_field_center_px, (104, 103), atol=0.5)
    assert abs(flagged.detector_sampling_mrad[0] - 30 / 43) < 0.03
