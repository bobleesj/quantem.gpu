"""Optical models and electron optics physics."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "wavelength_A_from_kV": ("quantem.gpu.ssb.optics.physics", "wavelength_A_from_kV"),
    "wavelength_m_from_kV": ("quantem.gpu.ssb.optics.physics", "wavelength_m_from_kV"),
    "keV_to_wavelength_nm": ("quantem.gpu.ssb.optics.physics", "keV_to_wavelength_nm"),
    "convergence_angle_to_k_max": (
        "quantem.gpu.ssb.optics.physics",
        "convergence_angle_to_k_max",
    ),
    "chi_polar": ("quantem.gpu.ssb.optics.aberration", "chi_polar"),
    "chi_cartesian": ("quantem.gpu.ssb.optics.aberration", "chi_cartesian"),
    "ABERRATION_INDICES": ("quantem.gpu.ssb.optics.aberration", "ABERRATION_INDICES"),
    "N_ABERRATIONS": ("quantem.gpu.ssb.optics.aberration", "N_ABERRATIONS"),
    "fit_aberrations": ("quantem.gpu.ssb.optics.aberration_fitting", "fit_aberrations"),
    "compute_shifts_from_aberrations": (
        "quantem.gpu.ssb.optics.aberration_fitting",
        "compute_shifts_from_aberrations",
    ),
    "AberrationFitter": ("quantem.gpu.ssb.optics.aberration_fitting", "AberrationFitter"),
    "COEFF_LABELS": ("quantem.gpu.ssb.optics.aberration_fitting", "COEFF_LABELS"),
}

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


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
