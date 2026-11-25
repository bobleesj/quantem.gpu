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


def test_a_float_disk_brighter_than_the_marker_value_is_measured_whole():
    """2026-11-24: an EMPAD-G1 WSe2 scan's mean pattern has a bright-field plateau near 66000 per pixel; every pixel
    at or above 65535 was zeroed as a marker and the disk edge read 4.98 px for an 11.64 px disk."""
    rows, cols = np.mgrid[:128, :128]
    radius = np.hypot(rows - 63.2, cols - 64.0)
    pattern = 300.0 + 66000.0 / (1.0 + np.exp((radius - 11.6) / 0.25))     # a soft-edged aperture, float values
    clean = detector.bright_field_disk(pattern, semiangle_mrad=24.9).bright_field_edge_radius_px
    assert abs(clean - 11.6) < 0.1          # pixel discretization of the edge; the bug read 4.98 px for this disk
    # integer markers are still left out: kept, they set the plateau and the rule reads 0.80 px (the old rule 5.23)
    for index in range(30):
        pattern[2 + index, 3] = 65535.0
    pattern[70, 100] = pattern[71, 100] = 4294967295.0
    assert abs(detector.disk_edge_radius(pattern) - clean) < 1e-3
