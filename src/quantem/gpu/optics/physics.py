"""Electron wavelength and SSB sampling from microscope calibration."""

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np

PLANCK_H = 6.62607015e-34  # Planck constant (J·s)
ELECTRON_MASS = 9.1093837015e-31  # Electron rest mass (kg)
ELECTRON_CHARGE = 1.602176634e-19  # Elementary charge (C)
SPEED_OF_LIGHT = 299792458  # Speed of light (m/s)

# The SSB output factors a reconstruction accepts, and the largest one upsampling="auto" chooses.
SSB_UPSAMPLING_FACTORS = (1, 2, 3, 4, 8)
SSB_AUTO_UPSAMPLING_MAX = 4

# Wavelengths in Angstroms at common microscope voltages in kV, at full precision.
COMMON_VOLTAGES = {
    80: 0.041757160772834,
    120: 0.033492152726724,
    200: 0.025079340450548,
    300: 0.019687489006849,
    400: 0.016439434169908,
}


def wavelength_A_from_kV(voltage_kV: float) -> float:
    """Relativistic electron wavelength from the accelerating voltage.

    The wavelength sets every reciprocal-space calibration (detector angle per
    pixel, aberration phase), so it must be the relativistic one. Common
    microscope voltages (80, 120, 200, 300, 400 kV) return the tabulated
    full-precision values; other voltages are computed from

    .. math::
        \\lambda = \\frac{h}{\\sqrt{2 m_e e V (1 + \\frac{eV}{2 m_e c^2})}}

    with Planck's constant ``h``, the electron rest mass ``m_e``, the
    elementary charge ``e``, the accelerating voltage ``V`` and the speed of
    light ``c``.

    Parameters
    ----------
    voltage_kV : float
        Accelerating voltage in kilovolts.

    Returns
    -------
    float
        Electron wavelength in Angstroms.

    Examples
    --------
    >>> wavelength_A_from_kV(200)
    0.025079340450548
    >>> wavelength_A_from_kV(300)
    0.019687489006849
    """
    voltage_int = int(voltage_kV)
    if voltage_int == voltage_kV and voltage_int in COMMON_VOLTAGES:
        return COMMON_VOLTAGES[voltage_int]
    # Python float: in float32 (NumPy metadata) the constants' products underflow to 0.
    voltage_V = float(voltage_kV) * 1000
    gamma_factor = 1 + (ELECTRON_CHARGE * voltage_V) / (2 * ELECTRON_MASS * SPEED_OF_LIGHT**2)
    wavelength_m = PLANCK_H / math.sqrt(
        2 * ELECTRON_MASS * ELECTRON_CHARGE * voltage_V * gamma_factor
    )
    return wavelength_m * 1e10


def interaction_parameter(voltage_kV: float) -> float:
    """Relativistic interaction parameter sigma, in rad / (V Angstrom): the phase a projected potential imparts.

    A thin specimen with projected potential ``V_p`` (V Angstrom) shifts the electron wave's phase by
    ``sigma * V_p`` radians. With the relativistic electron mass ``m = gamma m_e``,

    .. math::
        \\sigma = \\frac{2 \\pi m e \\lambda}{h^2}
                = \\frac{2 \\pi}{\\lambda V} \\frac{m_e c^2 + e V}{2 m_e c^2 + e V},

    with the relativistic wavelength ``lambda`` of :func:`wavelength_A_from_kV`, so both functions agree at every
    voltage.

    Parameters
    ----------
    voltage_kV : float
        Accelerating voltage in kilovolts; finite and positive.

    Returns
    -------
    float
        Interaction parameter in radians per volt-Angstrom.

    Examples
    --------
    >>> round(interaction_parameter(300), 8)
    0.00065262
    >>> round(interaction_parameter(200), 8)
    0.00072884
    """
    voltage = float(voltage_kV)
    if not (math.isfinite(voltage) and voltage > 0):
        raise ValueError(f"voltage_kV must be finite and positive; got {voltage_kV!r}.")
    voltage_V = voltage * 1000
    rest_energy_eV = ELECTRON_MASS * SPEED_OF_LIGHT**2 / ELECTRON_CHARGE
    return (2 * math.pi / (wavelength_A_from_kV(voltage) * voltage_V)
            * (rest_energy_eV + voltage_V) / (2 * rest_energy_eV + voltage_V))


def wrap_degrees(angle_deg: float, period_deg: float) -> float:
    """Return an angle in ``[0, period_deg)``, wrapped once in double precision.

    Every result and record states a scan-detector rotation that a caller or an engine gives
    in ``[0, 360)``; ``screening.prepare`` also states a given angle modulo 180. Those producers
    wrap here as their last step, after all arithmetic with the angle. ``%`` alone returns the
    period itself for a negative angle smaller than the rounding of the period
    (``-2e-15 % 360.0`` is ``360.0``); that case is 0. An angle already in range is returned
    bit for bit. The rotation fitted from the centre-of-mass curl is not wrapped: it is the
    angle the fit applied, in [0, 180], and its phase belongs to that angle.

    Examples
    --------
    >>> wrap_degrees(-20.0, 360.0), wrap_degrees(190.0, 180.0), wrap_degrees(-2e-15, 360.0)
    (340.0, 10.0, 0.0)
    """
    wrapped = float(angle_deg) % period_deg
    return 0.0 if wrapped == period_deg else wrapped


def ssb_upsampling_requirement(
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | Sequence[float],
) -> float:
    """Calculate the fractional output sampling needed for nominal SSB support.

    Parameters
    ----------
    voltage_kV : float
        Positive accelerating voltage in kilovolts.
    semiangle_mrad : float
        Positive probe convergence semi-angle in milliradians.
    scan_sampling_A : float or sequence of float
        Native scan step in Angstroms, ``(row, col)``; one number means both axes.

    Returns
    -------
    float
        Uncapped factor with output Nyquist frequency equal to
        ``2 * semiangle / wavelength`` on the more coarsely sampled axis.

    Notes
    -----
    This inexpensive metadata calculation does not load data, fit aberrations,
    or establish measured resolution. Detector truncation, dose, model validity,
    and scan aliases can reduce recoverable information below this ideal bound.
    It does not use the detector's maximum collection angle as SSB bandwidth.
    """
    try:
        voltage = float(voltage_kV)
        angle = float(semiangle_mrad)
        if isinstance(scan_sampling_A, bool) or scan_sampling_A is None:
            raise TypeError("scan_sampling_A is a number")
        spacing = ((float(scan_sampling_A),) * 2 if np.ndim(scan_sampling_A) == 0
                   else tuple(float(value) for value in scan_sampling_A))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Provide calibrated voltage, convergence semi-angle, and scan step."
        ) from exc
    if len(spacing) != 2 or not all(
        math.isfinite(value) and value > 0 for value in (voltage, angle, *spacing)
    ):
        raise ValueError(
            "Voltage, semi-angle, and scan step must be finite and positive; "
            "give the step as one number or a (row, col) pair."
        )
    target_spacing = wavelength_A_from_kV(voltage) / (4 * angle * 1e-3)
    return max(spacing) / target_spacing


def ssb_upsampling_factor(
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | Sequence[float],
    max_factor: int = SSB_AUTO_UPSAMPLING_MAX,
) -> int:
    """Choose automatic SSB output sampling from the scan sampling, at most 4× by default.

    Parameters
    ----------
    voltage_kV : float
        Positive accelerating voltage in kilovolts.
    semiangle_mrad : float
        Positive probe convergence semi-angle in milliradians.
    scan_sampling_A : float or sequence of float
        Native scan step in Angstroms, ``(row, col)``.
    max_factor : {1, 2, 3, 4, 8}, optional
        Largest factor to choose, 4 by default. ``max_factor=8`` gives the
        factor the scan sampling alone chooses.

    Returns
    -------
    int
        Smallest of 1, 2, 3, 4 and 8, up to ``max_factor``, whose output
        Nyquist frequency covers nominal support on both scan axes, or
        ``max_factor`` when none does. Use ``ssb_upsampling_requirement`` to
        inspect the fractional requirement.

    Notes
    -----
    A capped factor undersamples nominal support. ``upsampling="auto"``
    uses the default cap and prints one line, once per session, naming the
    factor the scan sampling supports; pass that factor explicitly (for
    example ``upsampling=8``) to use it. This choice does not change
    fitting.

    Examples
    --------
    At 300 kV and 30 mrad, a 1 Å step needs about 6.1×:

    >>> ssb_upsampling_factor(voltage_kV=300, semiangle_mrad=30, scan_sampling_A=1.0)
    4
    >>> ssb_upsampling_factor(voltage_kV=300, semiangle_mrad=30, scan_sampling_A=1.0, max_factor=8)
    8
    """
    if isinstance(max_factor, bool) or not isinstance(max_factor, Integral) or max_factor not in SSB_UPSAMPLING_FACTORS:
        raise ValueError("max_factor must be 1, 2, 3, 4, or 8.")
    required = ssb_upsampling_requirement(
        voltage_kV=voltage_kV, semiangle_mrad=semiangle_mrad,
        scan_sampling_A=scan_sampling_A,
    )
    return next((factor for factor in SSB_UPSAMPLING_FACTORS
                 if factor <= max_factor and factor >= required - 1e-12), int(max_factor))
