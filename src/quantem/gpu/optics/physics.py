"""Electron wavelength and SSB sampling from microscope calibration."""

import math
from collections.abc import Sequence
from numbers import Real

# Physical constants
PLANCK_H = 6.62607015e-34  # Planck constant (J·s)
ELECTRON_MASS = 9.1093837015e-31  # Electron rest mass (kg)
ELECTRON_CHARGE = 1.602176634e-19  # Elementary charge (C)
SPEED_OF_LIGHT = 299792458  # Speed of light (m/s)

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


def ssb_upsampling_factor(
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | Sequence[float],
) -> int:
    """Choose the smallest output grid covering nominal circular-aperture SSB support.

    Parameters
    ----------
    voltage_kV : float
        Positive accelerating voltage in kilovolts.
    semiangle_mrad : float
        Positive probe convergence semi-angle in milliradians.
    scan_sampling_A : float or sequence of float
        Native scan step in Angstroms; scalar or ``(row, col)``.

    Returns
    -------
    int
        Smallest supported factor (1, 2, 3, 4, or 8) with output Nyquist
        frequency at least ``2 * semiangle / wavelength`` on both axes.

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
        if isinstance(scan_sampling_A, Real):
            spacing = (float(scan_sampling_A),)
        else:
            spacing = tuple(float(value) for value in scan_sampling_A)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Provide calibrated voltage, convergence semi-angle, and scan step."
        ) from exc
    if len(spacing) not in (1, 2) or not all(
        math.isfinite(value) and value > 0 for value in (voltage, angle, *spacing)
    ):
        raise ValueError(
            "Voltage, semi-angle, and scan step must be finite and positive; "
            "use a scalar or (row, col) step."
        )
    target_spacing = wavelength_A_from_kV(voltage) / (4 * angle * 1e-3)
    required = max(spacing) / target_spacing
    for factor in (1, 2, 3, 4, 8):
        if factor >= required - 1e-12:
            return factor
    raise ValueError(
        "Nominal SSB support needs more than 8× output; "
        "choose an explicit factor for a limited output."
    )
