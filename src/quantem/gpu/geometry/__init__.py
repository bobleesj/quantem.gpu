"""Scan-detector frame arithmetic: rotations between the scan and detector frames and tilt conversions."""

from quantem.gpu.geometry.frames import (
    mean_rotation_deg,
    opposite_rotation_deg,
    propagator_to_sample_tilt,
    rotation_gap_deg,
    sample_to_propagator_tilt,
)

__all__ = [
    "mean_rotation_deg",
    "opposite_rotation_deg",
    "propagator_to_sample_tilt",
    "rotation_gap_deg",
    "sample_to_propagator_tilt",
]
