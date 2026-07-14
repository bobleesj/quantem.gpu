"""Optical models and electron optics physics (CuPy GPU implementation)."""

from .physics import (
    wavelength_A_from_kV,
    wavelength_m_from_kV,
    keV_to_wavelength_nm,
    convergence_angle_to_k_max,
)
from .aberration import (
    chi_polar,
    chi_cartesian,
    ABERRATION_INDICES,
    N_ABERRATIONS,
)
from .aberration_fitting import (
    fit_aberrations,
    compute_shifts_from_aberrations,
    AberrationFitter,
    COEFF_LABELS,
)

__all__ = [
    'chi_polar',
    'chi_cartesian',
    'ABERRATION_INDICES',
    'N_ABERRATIONS',
    'fit_aberrations',
    'compute_shifts_from_aberrations',
    'AberrationFitter',
    'COEFF_LABELS',
    'wavelength_A_from_kV',
    'wavelength_m_from_kV',
    'keV_to_wavelength_nm',
    'convergence_angle_to_k_max',
]
