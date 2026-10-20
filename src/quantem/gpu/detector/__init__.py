"""Virtual-detector products and prepared detector sessions."""

from quantem.gpu.detector.session import DetectorSession, prepare
from quantem.gpu.detector.workflow import (
    adf,
    bf,
    detector_mask,
    detector_sampling_mrad,
    df,
    disk_edge_radius,
    fit_probe,
    masked_sum,
    mean,
)

__all__ = [
    "DetectorSession",
    "adf",
    "bf",
    "detector_mask",
    "detector_sampling_mrad",
    "df",
    "disk_edge_radius",
    "fit_probe",
    "masked_sum",
    "mean",
    "prepare",
]
