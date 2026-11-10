"""Virtual-detector products and prepared detector sessions."""

from quantem.gpu.detector.session import DetectorSession, prepare
from quantem.gpu.detector.workflow import (
    BrightFieldDisk,
    adf,
    bf,
    detector_mask,
    detector_sampling_mrad,
    df,
    disk_edge_radius,
    bright_field_disk,
    masked_sum,
    mean,
    show,
)

__all__ = [
    "BrightFieldDisk",
    "DetectorSession",
    "adf",
    "bf",
    "detector_mask",
    "detector_sampling_mrad",
    "df",
    "disk_edge_radius",
    "bright_field_disk",
    "masked_sum",
    "mean",
    "prepare",
    "show",
]
