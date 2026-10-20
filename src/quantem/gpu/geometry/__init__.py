"""GPU-resident geometry operations for scientific STEM data, and scan-detector frame arithmetic."""

from quantem.gpu.geometry.frames import (
    mean_rotation_deg,
    opposite_rotation_deg,
    propagator_to_sample_tilt,
    rotation_gap_deg,
    sample_to_propagator_tilt,
)
from quantem.gpu.geometry.rotation import rotate_scan

__all__ = [
    "mean_rotation_deg",
    "opposite_rotation_deg",
    "propagator_to_sample_tilt",
    "rotate_scan",
    "rotation_gap_deg",
    "sample_to_propagator_tilt",
]
