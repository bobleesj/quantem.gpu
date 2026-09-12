"""Backend-independent provenance for MAPED summary images."""

from __future__ import annotations

import math


def summary_record(shape: tuple[int, int, int, int], sources) -> dict:
    """Describe the summary reductions used by MAPED alignment."""
    sources = list(sources)
    scan_shape = [int(shape[0]), int(shape[1])]
    detector_shape = [int(shape[2]), int(shape[3])]
    corrections = [
        source.metadata.get(
            "hot_pixel_correction",
            {"method": "zero", "applied": False, "pixel_count": 0},
        )
        for source in sources
    ]
    methods = sorted(
        {str(record.get("method", "zero")) for record in corrections}
    )
    corrected = all(record.get("applied") is True for record in corrections)
    if corrected and methods == ["median"]:
        invalid_policy = "stored detector-mask pixels use their local 3x3 median"
    elif corrected and methods == ["zero"]:
        invalid_policy = "stored detector-mask pixels are replaced with zero"
    else:
        invalid_policy = (
            "stored detector-mask exclusions contribute zero; the divisor "
            "remains the complete detector pixel count"
        )
    return {
        "version": 1,
        "source_count": len(sources),
        "hot_pixel_correction": {
            "methods": methods,
            "applied_to_every_source": corrected,
            "pixel_counts": [
                int(record.get("pixel_count", 0)) for record in corrections
            ],
        },
        "mean_bright_field": {
            "operation": "arithmetic_mean",
            "reduction_axes": ["detector_row", "detector_column"],
            "divisor": math.prod(detector_shape),
            "output_shape": scan_shape,
            "detector_selection": "complete_detector",
            "invalid_pixel_policy": invalid_policy,
            "alignment_role": "real_space",
        },
        "mean_diffraction_pattern": {
            "operation": "arithmetic_mean",
            "reduction_axes": ["scan_row", "scan_column"],
            "divisor": math.prod(scan_shape),
            "output_shape": detector_shape,
            "invalid_pixel_policy": invalid_policy,
            "alignment_role": "diffraction_origin_and_shift",
        },
        "intensity_normalization": "none",
    }
