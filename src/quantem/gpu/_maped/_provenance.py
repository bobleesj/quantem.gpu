"""Backend-independent provenance for MAPED summary images."""

from __future__ import annotations

import math


def summary_record(shape: tuple[int, int, int, int], source_count: int) -> dict:
    """Describe the summary reductions used by MAPED alignment."""
    scan_shape = [int(shape[0]), int(shape[1])]
    detector_shape = [int(shape[2]), int(shape[3])]
    return {
        "version": 1,
        "source_count": int(source_count),
        "mean_bright_field": {
            "operation": "arithmetic_mean",
            "reduction_axes": ["detector_row", "detector_column"],
            "divisor": math.prod(detector_shape),
            "output_shape": scan_shape,
            "detector_selection": "complete_detector",
            "invalid_pixel_policy": (
                "stored detector-mask exclusions contribute zero; the divisor "
                "remains the complete detector pixel count"
            ),
            "alignment_role": "real_space",
        },
        "mean_diffraction_pattern": {
            "operation": "arithmetic_mean",
            "reduction_axes": ["scan_row", "scan_column"],
            "divisor": math.prod(scan_shape),
            "output_shape": detector_shape,
            "invalid_pixel_policy": "stored detector-mask exclusions contribute zero",
            "alignment_role": "diffraction_origin_and_shift",
        },
        "intensity_normalization": "none",
    }
