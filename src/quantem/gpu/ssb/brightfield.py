"""The bright-field disk SSB reads: its typed pixel selection, its edge radius, and the crop decoded from an encoded acquisition.

The disk edge also calibrates the detector. The disk is the image of the probe-forming aperture, so its radius in
detector pixels corresponds to the convergence semiangle, and the detector sampling in mrad per pixel is
``semiangle_mrad / disk_edge_radius(mean_pattern)``.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from quantem.gpu import detector
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU


@dataclass(frozen=True)
class BrightfieldDisk:
    """Validated detector coordinates and geometry for one BF selection.

    Coordinates and centers always follow the public ``(row, col)`` convention.
    The coordinate arrays are copied, normalized to ``int32``, and made
    read-only so every SSB stage consumes the same immutable evidence.
    """

    rows: np.ndarray
    cols: np.ndarray
    bright_field_center_px: tuple[float, float]
    bright_field_radius_px: float
    # The disk radius the detector calibration uses, semiangle / detector_sampling_mrad: ``disk_edge_radius`` of the full-detector
    # mean pattern when the sampling is automatic.
    bright_field_edge_radius_px: float
    detector_shape: tuple[int, int]

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int32).reshape(-1).copy()
        cols = np.asarray(self.cols, dtype=np.int32).reshape(-1).copy()
        if rows.size == 0 or rows.shape != cols.shape:
            raise ValueError(
                "BF rows and columns must be non-empty matching vectors."
            )

        detector_shape = tuple(int(value) for value in self.detector_shape)
        if len(detector_shape) != 2 or min(detector_shape) < 1:
            raise ValueError(
                "detector_shape must contain two positive (row, col) sizes."
            )
        if (
            int(rows.min()) < 0
            or int(cols.min()) < 0
            or int(rows.max()) >= detector_shape[0]
            or int(cols.max()) >= detector_shape[1]
        ):
            raise ValueError(
                f"BF coordinates fall outside detector shape {detector_shape}."
            )
        linear = rows.astype(np.int64) * detector_shape[1] + cols
        if np.unique(linear).size != rows.size:
            raise ValueError("BF coordinates must not contain duplicates.")

        center = tuple(float(value) for value in self.bright_field_center_px)
        if len(center) != 2 or not all(math.isfinite(value) for value in center):
            raise ValueError("bright_field_center_px must contain two finite values.")
        radius = float(self.bright_field_radius_px)
        detected_radius = float(self.bright_field_edge_radius_px)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("bright_field_radius_px must be a positive finite value.")
        if not math.isfinite(detected_radius) or detected_radius <= 0:
            raise ValueError(
                "bright_field_edge_radius_px must be a positive finite value."
            )

        rows.flags.writeable = False
        cols.flags.writeable = False
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "bright_field_center_px", center)
        object.__setattr__(self, "bright_field_radius_px", radius)
        object.__setattr__(self, "bright_field_edge_radius_px", detected_radius)
        object.__setattr__(self, "detector_shape", detector_shape)

    @property
    def size(self) -> int:
        """Number of selected BF detector pixels."""

        return int(self.rows.size)


def crop_bright_field(
    loaded: Dataset4dstemGPU,
    backend: str,
    threshold: float,
    bright_field_radius_px: float | None,
    *,
    calibrate_detector: bool = False,
    bright_field_center_px: tuple[float, float] | None = None,
    scan_region: tuple[int, int, int, int] | None = None,
    mean_pattern=None,
):
    """Decode only the bright-field crop of an encoded acquisition, keeping the full-detector calibration.

    Returns the counts (CuPy on CUDA), the disk centre in full-detector pixels, the disk radius, the calibration radius
    (``disk_edge_radius`` of the full-detector mean pattern; None unless ``calibrate_detector``), the crop's first
    detector ``(row, col)``, which a consumer of the crop subtracts from the centre, and the float64 mean pattern the disk
    was found on.

    ``scan_region`` (row_start, row_stop, col_start, col_stop) decodes those scan positions only; the disk is still found
    on the mean pattern of the whole scan, which ``mean_pattern`` hands over when the caller has computed it already
    (``detector.mean(loaded)``), so it is not computed a second time.

    SSB reads nothing outside the bright-field disk, but decoding the whole 4D cube (19 GB for 512^2 x 192^2 uint16) is
    what the GPU loader forbids. The disk is found on the full-detector mean pattern with the backend's own rule
    (pixels above ``threshold`` x max, within the radius around the centroid of pixels above mean + std, or within
    ``bright_field_radius_px`` around the weighted centroid), so the crop plus the pinned centre selects exactly the pixels a
    full-detector session would (tests/hardware/cuda/test_ssb_open_encoded.py). This is the equal-area rule of
    ``detector.fit_probe`` evaluated in float64; ``fit_probe``'s float32 would move the centre in its last bits and
    with it the pixels the session reads.
    """
    mean_pattern = np.asarray(detector.mean(loaded) if mean_pattern is None else mean_pattern, dtype=np.float64)
    # The calibration reads the full detector: the crop below cuts the disk edge off.
    calibration_radius = disk_edge_radius(mean_pattern) if calibrate_detector else None
    if bright_field_radius_px is None:
        disk = mean_pattern > mean_pattern.mean() + mean_pattern.std()
        total = int(disk.sum())
        if total == 0:
            raise ValueError("No bright-field disk found in the mean diffraction pattern.")
        rows, cols = np.nonzero(disk)
        center = (float(rows.mean()), float(cols.mean()))
        radius = math.sqrt(total / math.pi)
    else:
        rows, cols = np.nonzero(mean_pattern > mean_pattern.max() * float(threshold))
        weights = mean_pattern[rows, cols]
        center = (float((rows * weights).sum() / weights.sum()), float((cols * weights).sum() / weights.sum()))
        radius = float(bright_field_radius_px)
    if bright_field_center_px is not None:
        center = tuple(float(value) for value in bright_field_center_px)
    det_rows, det_cols = mean_pattern.shape
    # one pixel of margin beyond the disk so float rounding of the centre never drops an edge pixel
    row0, col0 = max(0, math.floor(center[0] - radius) - 1), max(0, math.floor(center[1] - radius) - 1)
    row1, col1 = min(det_rows, math.ceil(center[0] + radius) + 2), min(det_cols, math.ceil(center[1] + radius) + 2)
    counts = loaded.read(scan_region=scan_region, detector_region=(row0, row1, col0, col1))
    if backend == "cuda":
        # torch exports a tensor through DLPack from its current device and CuPy imports it onto its own: both must be
        # the acquisition's GPU, which need not be the current one
        with torch.cuda.device(counts.device), cp.cuda.Device(counts.device.index):
            counts = cp.from_dlpack(counts)
            # read() returns the original counts, including invalid detector
            # pixels. Exclude those from scientific BF selection, keeping the
            # encoded source untouched and the fitted/full-scan masks equal.
            valid = getattr(loaded.data, "valid_pixels", None)
            if valid is not None:
                counts *= cp.asarray(valid[row0:row1, col0:col1], dtype=counts.dtype)
    return counts, center, radius, calibration_radius, (row0, col0), mean_pattern


def disk_edge_radius(mean_pattern: np.ndarray) -> float:
    """Return the radius in detector pixels at which the bright-field disk falls to half its plateau intensity.

    The disk is the image of the probe-forming aperture, so this radius corresponds to the convergence semiangle and
    calibrates the detector: ``detector_sampling_mrad = semiangle_mrad / disk_edge_radius(mean_pattern)`` mrad per pixel. A
    symmetric blur of the aperture edge (detector point spread, partial coherence) leaves the half-intensity contour
    at the geometric edge, while a higher threshold moves it inward. The plateau is the median of the pixels above
    mean + std (the disk ``detector.fit_probe`` finds), and the radius is that of a disk with the area of the pixels
    above half of it, which resolves the edge below one pixel. On abTEM 4D-STEM with a known detector sampling this is
    within 0.4 % of the truth; on a 300 kV, 30 mrad Arina acquisition it gives 0.552 mrad per pixel, the measured
    calibration of that camera length (docs/maintainer/2026-10-05-ssb-detector-sampling.md).

    Parameters
    ----------
    mean_pattern : numpy.ndarray
        Mean diffraction pattern of the full detector, (row, col).

    Raises
    ------
    ValueError
        When no pixel stands above mean + std, so the pattern has no disk to calibrate on.
    """
    pattern = np.asarray(mean_pattern, dtype=np.float64)
    disk = pattern > pattern.mean() + pattern.std()
    if not disk.any():
        raise ValueError(
            "The mean diffraction pattern has no bright-field disk to calibrate the detector on; "
            "pass detector_sampling_mrad (mrad per detector pixel)."
        )
    half_plateau = 0.5 * float(np.median(pattern[disk]))
    return math.sqrt(np.count_nonzero(pattern > half_plateau) / math.pi)
