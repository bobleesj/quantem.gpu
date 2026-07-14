"""
Electron microscopy physics calculations.

Provides fundamental constants and calculations for electron wavelength,
relativistic corrections, and convergence angles specific to electron microscopy.

For general unit conversions (rad/deg, Å/nm/μm/mm), see quantem.gpu.ssb.optics.units
"""

import math


# Physical constants
PLANCK_H = 6.62607015e-34  # Planck constant (J·s)
ELECTRON_MASS = 9.1093837015e-31  # Electron rest mass (kg)
ELECTRON_CHARGE = 1.602176634e-19  # Elementary charge (C)
SPEED_OF_LIGHT = 299792458  # Speed of light (m/s)

# Mathematical constants
PI = math.pi


# Common microscope voltages and pre-computed wavelengths (full precision)
COMMON_VOLTAGES = {
    80: 0.041757160772834,   # 80 keV
    120: 0.033492152726724,  # 120 keV
    200: 0.025079340450548,  # 200 keV
    300: 0.019687489006849,  # 300 keV
    400: 0.016439434169908,  # 400 keV
}


def wavelength_A_from_kV(voltage_kV: float) -> float:
    """
    Calculate relativistic electron wavelength from accelerating voltage.
    
    Uses pre-computed values for common voltages (80, 120, 200, 300, 400 keV)
    to avoid repeated computation. For other voltages, computes from first principles.
    
    Parameters
    ----------
    voltage_kV : float
        Accelerating voltage in kilovolts (kV)
        
    Returns
    -------
    wavelength_A : float
        Electron wavelength in Angstroms (Å)
        
    Notes
    -----
    Uses the relativistic de Broglie wavelength formula:
    
    .. math::
        \\lambda = \\frac{h}{\\sqrt{2 m_e e V (1 + \\frac{eV}{2 m_e c^2})}}
    
    where:
    - h is Planck's constant
    - m_e is electron rest mass
    - e is elementary charge
    - V is accelerating voltage
    - c is speed of light
    
    Examples
    --------
    >>> wavelength_A_from_kV(200)  # 200 keV (pre-computed)
    0.02508
    
    >>> wavelength_A_from_kV(300)  # 300 keV (pre-computed)
    0.01969
    """
    # Check if it's a common voltage (convert to int for lookup)
    voltage_int = int(voltage_kV)
    if voltage_int == voltage_kV and voltage_int in COMMON_VOLTAGES:
        return COMMON_VOLTAGES[voltage_int]
    # Otherwise compute from first principles
    voltage_V = voltage_kV * 1000  # Convert kV to V
    # Relativistic correction factor: 1 + eV/(2m_e c²)
    gamma_factor = 1 + (ELECTRON_CHARGE * voltage_V) / (2 * ELECTRON_MASS * SPEED_OF_LIGHT**2)
    # Wavelength in meters
    wavelength_m = PLANCK_H / math.sqrt(
        2 * ELECTRON_MASS * ELECTRON_CHARGE * voltage_V * gamma_factor
    )
    # Convert to Angstroms
    wavelength_angstrom = wavelength_m * 1e10
    return wavelength_angstrom


def keV_to_wavelength_nm(keV: float) -> float:
    """
    Calculate relativistic electron wavelength in nanometers.
    
    Convenience function that returns wavelength in nm (common for aberration work).
    
    Parameters
    ----------
    keV : float
        Accelerating voltage in keV (e.g., 300 for 300 keV)
        
    Returns
    -------
    wavelength_nm : float
        Electron wavelength in nanometers
        
    Examples
    --------
    >>> keV_to_wavelength_nm(300)  # 300 keV electron
    0.001969
    >>> keV_to_wavelength_nm(200)  # 200 keV electron  
    0.002508
    """
    return wavelength_A_from_kV(keV) / 10.0


def wavelength_m_from_kV(keV: float) -> float:
    """
    Calculate relativistic electron wavelength in meters.
    
    This is the natural unit for SI-based calculations where aberration
    coefficients are in meters.
    
    Parameters
    ----------
    keV : float
        Accelerating voltage in keV (e.g., 300 for 300 keV)
        
    Returns
    -------
    wavelength_m : float
        Electron wavelength in meters
        
    Examples
    --------
    >>> wavelength_m_from_kV(300)  # 300 keV electron
    1.969e-12
    >>> wavelength_m_from_kV(200)  # 200 keV electron  
    2.508e-12
    """
    return wavelength_A_from_kV(keV) * 1e-10


def convergence_angle_to_k_max(convergence_angle_mrad: float, keV: float = 300.0) -> float:
    """
    Convert convergence semi-angle to maximum spatial frequency.
    
    Parameters
    ----------
    convergence_angle_mrad : float
        Convergence semi-angle in milliradians (mrad)
    keV : float, optional
        Accelerating voltage in keV (default: 300)
        
    Returns
    -------
    k_max : float
        Maximum spatial frequency in inverse Angstroms (Å^-1)
        
    Notes
    -----
    The relationship is:
    
    .. math::
        k_{\\text{max}} = \\frac{\\alpha}{\\lambda}
    
    where α is the convergence semi-angle and λ is the electron wavelength.
    
    Examples
    --------
    >>> convergence_angle_to_k_max(25.0, keV=300)
    1.268
    """
    wavelength_angstrom = wavelength_A_from_kV(keV)
    alpha_rad = convergence_angle_mrad / 1000  # Convert mrad to rad
    return alpha_rad / wavelength_angstrom


def electron_wavelength_angstrom(energy_ev: float) -> float:
    """
    Compute relativistic electron wavelength in Angstroms.
    
    This is an alias for wavelength_A_from_kV that accepts energy in eV.
    
    Parameters
    ----------
    energy_ev : float
        Electron energy in eV (e.g., 300000 for 300 keV)
        
    Returns
    -------
    float
        Wavelength in Angstroms
        
    Examples
    --------
    >>> electron_wavelength_angstrom(300e3)  # 300 keV
    0.01969
    >>> electron_wavelength_angstrom(200e3)  # 200 keV
    0.02508
    """
    return wavelength_A_from_kV(energy_ev / 1000.0)
