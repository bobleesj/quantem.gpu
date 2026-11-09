"""Measured virtual detector on the native scan grid."""

from numbers import Real

import numpy as np

from quantem.gpu.detector import prepare


def annular_limits(bounds) -> tuple[float, float]:
    """Normalize collection angles, including the inner edge and excluding the outer."""
    try:
        inner, outer = bounds
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in (inner, outer)):
            raise TypeError
        inner, outer = float(inner), float(outer)
        if np.isfinite((inner, outer)).all() and 0 <= inner < outer:
            return inner, outer
    except (TypeError, ValueError):
        pass
    raise ValueError("Use a finite annulus_mrad=(inner, outer) with 0 <= inner < outer, in mrad.")


def annular_mask(session, shape, limits, valid):
    """Select valid pixels by physical angle on a possibly anisotropic detector."""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    cy, cx = session.bright_field_center_px
    dy, dx = np.broadcast_to(session.detector_sampling_mrad, (2,))
    radius = np.hypot((rows - cy) * dy, (cols - cx) * dx)
    mask = (radius >= limits[0]) & (radius < limits[1])
    if valid is not None:
        mask &= valid
    if not mask.any():
        raise ValueError("The annulus has no valid detector pixels. Check its angles and detector calibration.")
    return mask


def measured_validity(scan, valid):
    """Exclude interpolated detector replacements from measured-count guides."""
    flagged = scan.metadata.get("pixel_mask")
    if flagged is None:
        return valid
    measured = np.asarray(flagged) == 0
    return measured if valid is None else valid & measured


def annular_counts(session, scan, limits):
    """Sum measured detector values on the scan device without resampling the scan."""
    det = prepare(scan)
    try:
        valid = measured_validity(scan, det.detector_validity)
        mask = annular_mask(session, scan.shape[-2:], limits, valid)
        return det.masked_sum(mask, output="native")
    finally:
        det.close()
