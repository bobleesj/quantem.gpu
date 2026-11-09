"""Scan-detector frame arithmetic shared by SSB, ptychography and session tools.

SSB states the sample tilt in the scan frame (``sample_tilt_mrad``, (row, col) along the scan axes). A ptychographic
reconstruction places its object and propagator in the detector frame, so its tilt (``propagator_tilt_mrad``) is
the same vector expressed along the detector's (row, col) axes. The two frames differ by the scan-detector rotation
``theta`` (``scan_detector_rotation_deg``) and, when ``scan_detector_rotation_flipped`` is true, by a swap of the
detector's row and column axes, the convention SSB, DPC and screening apply to the centre of mass:

    scan = R(theta) . D . detector,  R(theta) = [[cos, -sin], [sin, cos]],  D = swap when flipped, else identity

so ``detector = D . R(-theta) . scan``. The scan-detector rotation itself is measured from the centre-of-mass curl
only modulo 180 degrees; the rotation helpers here state the other branch exactly and average angles on a circle of
the right period, so 179 and 1 degrees agree (mod 180) instead of averaging to 90.
"""

import math
from collections.abc import Iterable

import numpy as np


def _rotation(scan_detector_rotation_deg: float) -> tuple[float, float]:
    angle = math.radians(float(scan_detector_rotation_deg))
    return math.cos(angle), math.sin(angle)


def _pair(name: str, value) -> tuple[float, float]:
    if np.ndim(value) != 1 or len(value) != 2:
        raise ValueError(f"{name} is a (row, col) pair in mrad; got {value!r}.")
    return float(value[0]), float(value[1])


def sample_to_propagator_tilt(
    sample_tilt_mrad: tuple[float, float],
    *,
    scan_detector_rotation_deg: float,
    scan_detector_rotation_flipped: bool = False,
) -> tuple[float, float]:
    """Express an SSB sample tilt (scan frame) along the detector axes, the frame of the ptychographic propagator.

    ``propagator = D . R(-theta) . sample``: the scan-frame vector turned by minus the scan-detector rotation, then
    its components swapped when the detector axes are flipped. Unflipped, this is
    ``(t_r cos theta + t_c sin theta, -t_r sin theta + t_c cos theta)``.

    Parameters
    ----------
    sample_tilt_mrad : (float, float)
        Sample tilt in mrad, (row, col) along the scan axes, as ``DirectPtychoResult.sample_tilt_mrad`` states it.
    scan_detector_rotation_deg : float
        Physical scan-to-detector rotation in degrees, the session's ``scan_detector_rotation_deg``.
    scan_detector_rotation_flipped : bool
        True when the detector's row and column axes are swapped relative to the scan.

    Returns
    -------
    tuple[float, float]
        The tilt in mrad, (row, col) along the original detector axes (``propagator_tilt_mrad``).

    Examples
    --------
    >>> sample_to_propagator_tilt((2.0, 0.0), scan_detector_rotation_deg=90.0)
    (1.2246467991473532e-16, -2.0)
    """
    row, col = _pair("sample_tilt_mrad", sample_tilt_mrad)
    cosine, sine = _rotation(scan_detector_rotation_deg)
    detector = (row * cosine + col * sine, -row * sine + col * cosine)
    return detector[::-1] if scan_detector_rotation_flipped else detector


def propagator_to_sample_tilt(
    propagator_tilt_mrad: tuple[float, float],
    *,
    scan_detector_rotation_deg: float,
    scan_detector_rotation_flipped: bool = False,
) -> tuple[float, float]:
    """Express a ptychographic propagator tilt (detector frame) along the scan axes, as SSB's ``sample_tilt_mrad``.

    The inverse of :func:`sample_to_propagator_tilt`: ``sample = R(theta) . D . propagator``. Unflipped, this is
    ``(o_r cos theta - o_c sin theta, o_r sin theta + o_c cos theta)``.

    Parameters
    ----------
    propagator_tilt_mrad : (float, float)
        Tilt in mrad, (row, col) along the original detector axes.
    scan_detector_rotation_deg : float
        Physical scan-to-detector rotation in degrees.
    scan_detector_rotation_flipped : bool
        True when the detector's row and column axes are swapped relative to the scan.

    Returns
    -------
    tuple[float, float]
        The tilt in mrad, (row, col) along the scan axes (``sample_tilt_mrad``).

    Examples
    --------
    >>> propagator_to_sample_tilt((0.0, -2.0), scan_detector_rotation_deg=90.0)
    (2.0, -1.2246467991473532e-16)
    """
    row, col = _pair("propagator_tilt_mrad", propagator_tilt_mrad)
    if scan_detector_rotation_flipped:
        row, col = col, row
    cosine, sine = _rotation(scan_detector_rotation_deg)
    return (row * cosine - col * sine, row * sine + col * cosine)


def opposite_rotation_deg(scan_detector_rotation_deg: float) -> float:
    """The other 180-degree branch of a scan-detector rotation, in [0, 360).

    The centre-of-mass curl cannot tell ``theta`` from ``theta + 180``. An angle at or above 180 has 180 subtracted,
    which is exact (adding 180 and wrapping at 360 would round), so the branch of an angle in [180, 360) maps back to
    it bit for bit.

    Examples
    --------
    >>> opposite_rotation_deg(158.9), opposite_rotation_deg(338.9)
    (338.9, 158.89999999999998)
    """
    angle = float(scan_detector_rotation_deg) % 360.0
    if angle >= 180.0:
        return angle - 180.0
    # an angle a rounding below 180 lands on 360, which is 0
    other = angle + 180.0
    return 0.0 if other >= 360.0 else other


def rotation_gap_deg(first_deg: float, second_deg: float, *, period_deg: float = 360.0) -> float:
    """Smallest angular distance between two rotations on a circle of ``period_deg``, in [0, period_deg / 2].

    Use ``period_deg=180`` for estimates known modulo 180 degrees (the centre-of-mass curl), so 179 and 1 degrees are
    2 degrees apart, and ``period_deg=360`` for physical rotations whose branch is settled.

    Examples
    --------
    >>> rotation_gap_deg(179.0, 1.0, period_deg=180.0), rotation_gap_deg(179.0, 1.0)
    (2.0, 178.0)
    """
    period = float(period_deg)
    gap = abs(float(first_deg) - float(second_deg)) % period
    return min(gap, period - gap)


def mean_rotation_deg(angles_deg: Iterable[float], *, period_deg: float = 360.0) -> float:
    """Circular mean of rotations on a circle of ``period_deg``, in [0, period_deg).

    Each angle is a unit vector at ``2 pi angle / period_deg``; the mean is the direction of their sum. With
    ``period_deg=180`` the mod-180 estimates of several files (179, 1, 0.5 degrees) average to about 0.2 degrees,
    where a mean of the numbers gives 60.2. Fold estimates of one session onto one 180-degree branch before taking
    a 360-degree mean of physical rotations.

    Raises
    ------
    ValueError
        No angles, or angles spread so evenly that they have no mean direction.

    Examples
    --------
    >>> round(mean_rotation_deg([179.0, 1.0, 0.5], period_deg=180.0), 3)
    0.167
    """
    values = np.asarray(list(angles_deg), dtype=np.float64)
    if values.size == 0:
        raise ValueError("mean_rotation_deg needs at least one angle.")
    period = float(period_deg)
    resultant = np.mean(np.exp(2j * np.pi * values / period))
    if abs(resultant) < 1e-12:
        raise ValueError(f"The angles {values.tolist()} have no mean direction on a {period:g}-degree circle.")
    mean = float(np.angle(resultant)) * period / (2 * np.pi) % period
    return 0.0 if mean == period else mean


__all__ = [
    "mean_rotation_deg",
    "opposite_rotation_deg",
    "propagator_to_sample_tilt",
    "rotation_gap_deg",
    "sample_to_propagator_tilt",
]
