r"""
Aberration functions derived systematically from polar form (CuPy GPU implementation).

This module implements aberration :math:`\chi(u,v)` and its gradient 
:math:`\nabla\chi` (beam shifts) starting from the well-known polar form 
and converting to Cartesian.

Coordinate system
-----------------

Aberrations are defined in **k-space** (reciprocal space), not real space.
The coordinates :math:`(u, v)` represent the beam tilt angle:

- :math:`u = \alpha \cos(\phi)` - x-component of beam tilt (radians)
- :math:`v = \alpha \sin(\phi)` - y-component of beam tilt (radians)
- :math:`\alpha = \sqrt{u^2 + v^2}` - convergence semi-angle (radians)

Polar form (ground truth)
-------------------------

The wavefront aberration in polar coordinates is:

.. math::

    \chi(\alpha, \phi) = \frac{2\pi}{\lambda} \sum_{n,m} \frac{C_{n,m}}{n+1} 
                         \alpha^{n+1} \cos(m(\phi - \phi_{n,m}))

============  =====  ==============  =======================  =======================
Krivanek     (n,m)  Name            P_cos(u,v)               P_sin(u,v)
============  =====  ==============  =======================  =======================
C10          (1,0)  Defocus         u²+v²                    (none)
C12          (1,2)  2-fold astig    u²-v²                    2uv
C21          (2,1)  Coma            u(u²+v²)                 v(u²+v²)
C23          (2,3)  3-fold astig    u³-3uv²                  3u²v-v³
C30          (3,0)  Spherical       (u²+v²)²                 (none)
C32          (3,2)  Star            (u²-v²)(u²+v²)           2uv(u²+v²)
C34          (3,4)  4-fold astig    u⁴-6u²v²+v⁴              4u³v-4uv³
C41          (4,1)  4th coma        u(u²+v²)²                v(u²+v²)²
C43          (4,3)  3-lobe          (u³-3uv²)(u²+v²)         (3u²v-v³)(u²+v²)
C45          (4,5)  5-fold astig    u⁵-10u³v²+5uv⁴           5u⁴v-10u²v³+v⁵
C50          (5,0)  5th spherical   (u²+v²)³                 (none)
C52          (5,2)  5th star        (u²-v²)(u²+v²)²          2uv(u²+v²)²
C54          (5,4)  Rosette         (u⁴-6u²v²+v⁴)(u²+v²)     (4u³v-4uv³)(u²+v²)
C56          (5,6)  6-fold astig    u⁶-15u⁴v²+15u²v⁴-v⁶      6u⁵v-20u³v³+6uv⁵
============  =====  ==============  =======================  =======================
"""

import math
import cupy as cp
from .physics import keV_to_wavelength_nm


# =============================================================================
# Aberration indices (n, m) for the 14 standard aberrations up to 5th order
# =============================================================================

N_ABERRATIONS = 14

# CuPy arrays for vectorized operations (GPU assumed available)
_N_VALUES = cp.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5], dtype=cp.int32)
_M_VALUES = cp.array([0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6], dtype=cp.int32)

# Legacy list form for backward compatibility
ABERRATION_INDICES = [
    # (n, m, name, has_angle)
    (1, 0, 'C10', False),   # Defocus
    (1, 2, 'C12', True),    # 2-fold astigmatism
    (2, 1, 'C21', True),    # Coma
    (2, 3, 'C23', True),    # 3-fold astigmatism
    (3, 0, 'C30', False),   # Spherical aberration
    (3, 2, 'C32', True),    # Star aberration
    (3, 4, 'C34', True),    # 4-fold astigmatism
    (4, 1, 'C41', True),    # 4th order coma
    (4, 3, 'C43', True),    # 3-lobe aberration
    (4, 5, 'C45', True),    # 5-fold astigmatism
    (5, 0, 'C50', False),   # 5th order spherical
    (5, 2, 'C52', True),    # 5th order star
    (5, 4, 'C54', True),    # Rosette
    (5, 6, 'C56', True),    # 6-fold astigmatism
]

# Conversion factor: mrad to radians
MRAD_TO_RAD = 1e-3

# =============================================================================
# Chi function (wavefront aberration)
# =============================================================================

def chi_polar(
    alpha_mrad: cp.ndarray,
    phi_rad: cp.ndarray,
    mags_m: cp.ndarray,
    angles_rad: cp.ndarray,
    wavelength_m: float,
) -> cp.ndarray:
    r"""
    Compute wavefront aberration :math:`\chi_0(\alpha, \phi)` in radians.
    
    This is the main user-facing function for computing the wavefront aberration.
    It takes aberration coefficients in SI units (meters) and returns χ₀ in radians.
    
    The aberration function follows the standard form:
    
    .. math::

        \chi_0(\alpha, \phi) = \frac{2\pi}{\lambda} \sum_{n,m} 
            \frac{C_{n,m}}{n+1} \alpha^{n+1} \cos(m(\phi - \phi_{n,m}))
    
    Parameters
    ----------
    alpha_mrad : cp.ndarray
        Convergence angle in milliradians, any shape
    phi_rad : cp.ndarray
        Azimuthal angle in radians, same shape as alpha_mrad
    mags_m : cp.ndarray
        Aberration magnitudes in meters, shape (14,). Order:
        C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50, C52, C54, C56.
        Example: -50e-9 for -50 nm defocus, 1e-3 for 1 mm Cs
    angles_rad : cp.ndarray
        Aberration orientation angles in radians, shape (14,).
        Only used for non-rotationally-symmetric aberrations (m ≠ 0).
    wavelength_m : float
        Electron wavelength in meters (e.g., 1.97e-12 for 300 keV)
        
    Returns
    -------
    chi0_rad : cp.ndarray
        Wavefront aberration in radians, same shape as alpha_mrad
        
    Examples
    --------
    >>> import cupy as cp
    >>> from quantem.gpu.ssb.optics.aberration import chi_polar
    >>> from quantem.gpu.ssb.optics.physics import wavelength_m_from_kV
    >>> 
    >>> # Define aberrations in SI units (meters)
    >>> mags_m = cp.zeros(14, dtype=cp.float32)
    >>> mags_m[0] = -50e-9   # C10: Defocus (-50 nm)
    >>> mags_m[4] = 1e-3     # C30: Spherical aberration (1 mm)
    >>> angles_rad = cp.zeros(14, dtype=cp.float32)
    >>> 
    >>> # Create coordinate grid (in mrad, typical range 0-20 mrad)
    >>> alpha_mrad = cp.linspace(0, 20, 100, dtype=cp.float32)
    >>> phi_rad = cp.zeros(100, dtype=cp.float32)
    >>> 
    >>> # Compute chi
    >>> wavelength_m = wavelength_m_from_kV(300)  # ~1.97 pm
    >>> chi0_rad = chi_polar(alpha_mrad, phi_rad, mags_m, angles_rad, wavelength_m)
    """
    # Validate input shapes
    if mags_m.shape != (N_ABERRATIONS,):
        raise ValueError(f"mags_m must have shape (14,), got {mags_m.shape}")
    if angles_rad.shape != (N_ABERRATIONS,):
        raise ValueError(f"angles_rad must have shape (14,), got {angles_rad.shape}")
    # Convert mrad to rad
    alpha_rad = alpha_mrad * MRAD_TO_RAD
    # Get n, m values as float for computation
    n_vals = _N_VALUES.astype(cp.float32)
    m_vals = _M_VALUES.astype(cp.float32)
    # Prefactors: (2π/λ) * C_nm / (n+1), shape (14,)
    prefactors = (2 * math.pi / wavelength_m) * mags_m / (n_vals + 1)
    # Precompute α^k for k = 2, 3, 4, 5, 6 (we need α^(n+1) for n = 1..5)
    alpha_powers = cp.stack([
        alpha_rad ** 2,  # n=1: α^2
        alpha_rad ** 3,  # n=2: α^3
        alpha_rad ** 4,  # n=3: α^4
        alpha_rad ** 5,  # n=4: α^5
        alpha_rad ** 6,  # n=5: α^6
    ], axis=0)  # (5, *shape)
    # Map n values to indices: n=1->0, n=2->1, etc.
    n_indices = (n_vals - 1).astype(cp.int32)
    # Gather α^(n+1) for each aberration: (14, *shape)
    alpha_n1 = alpha_powers[n_indices]
    # Compute cos(m(φ - φ_nm)) for all aberrations
    # Reshape for broadcasting
    ndim = phi_rad.ndim
    phi_diff = phi_rad[cp.newaxis, ...] - angles_rad.reshape(-1, *([1] * ndim))  # (14, *shape)
    m_vals_bc = m_vals.reshape(-1, *([1] * ndim))  # (14, 1, 1, ...)
    cos_terms = cp.cos(m_vals_bc * phi_diff)  # (14, *shape)
    # Combine: prefactors (14,) * alpha_n1 (14, *shape) * cos_terms (14, *shape)
    prefactors_bc = prefactors.reshape(-1, *([1] * ndim))  # (14, 1, 1, ...)
    terms = prefactors_bc * alpha_n1 * cos_terms  # (14, *shape)
    # Sum over all aberrations
    chi0 = terms.sum(axis=0)  # (*shape)
    return chi0


# =============================================================================
# Chebyshev polynomials for Cartesian form
# =============================================================================

def _chebyshev_polynomials(u: cp.ndarray, v: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    r"""
    Compute :math:`\alpha^m \cos(m\phi)` and :math:`\alpha^m \sin(m\phi)` as polynomials in (u, v).
    
    Uses the identity: (u + iv)^m = α^m [cos(mφ) + i·sin(mφ)]
    
    Parameters
    ----------
    u, v : cp.ndarray
        K-space coordinates (beam tilt), dimensionless. Any shape.
        
    Returns
    -------
    cos_terms : cp.ndarray
        Shape (7, *u.shape) where cos_terms[m] = α^m cos(mφ)
    sin_terms : cp.ndarray
        Shape (7, *u.shape) where sin_terms[m] = α^m sin(mφ)
    """
    z = u + 1j * v  # z = u + iv (complex)
    # Compute z^m for m = 0, 1, ..., 6
    z1 = z
    z2 = z1 * z
    z3 = z2 * z
    z4 = z3 * z
    z5 = z4 * z
    z6 = z5 * z
    # Stack into tensor
    powers = cp.stack([
        cp.ones_like(z),  # z^0 = 1
        z1,               # z^1
        z2,               # z^2
        z3,               # z^3
        z4,               # z^4
        z5,               # z^5
        z6,               # z^6
    ], axis=0)
    return powers.real, powers.imag


def _compute_P_cos_sin(u: cp.ndarray, v: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    r"""
    Compute P_cos and P_sin for all 14 aberrations in one vectorized call.
    
    The aberration polynomials require :math:`\alpha^{n+1} \cos(m\phi)` and 
    :math:`\alpha^{n+1} \sin(m\phi)`. 
    
    Parameters
    ----------
    u, v : cp.ndarray
        K-space coordinates (beam tilt), dimensionless. Any shape.
        
    Returns
    -------
    P_cos : cp.ndarray
        Shape (14, *u.shape) - P_cos for each aberration
    P_sin : cp.ndarray
        Shape (14, *u.shape) - P_sin for each aberration
    """
    # Get Chebyshev terms for m = 0..6
    cos_m, sin_m = _chebyshev_polynomials(u, v)  # (7, *shape)
    α2 = u**2 + v**2  # α² = u² + v²
    # Precompute α² powers
    α2_0 = cp.ones_like(α2)
    α2_1 = α2
    α2_2 = α2 * α2
    α2_3 = α2_2 * α2
    # Build P_cos and P_sin for each aberration
    P_cos = cp.stack([
        cos_m[0] * α2_1,   # C10: m=0, p=1
        cos_m[2] * α2_0,   # C12: m=2, p=0
        cos_m[1] * α2_1,   # C21: m=1, p=1
        cos_m[3] * α2_0,   # C23: m=3, p=0
        cos_m[0] * α2_2,   # C30: m=0, p=2
        cos_m[2] * α2_1,   # C32: m=2, p=1
        cos_m[4] * α2_0,   # C34: m=4, p=0
        cos_m[1] * α2_2,   # C41: m=1, p=2
        cos_m[3] * α2_1,   # C43: m=3, p=1
        cos_m[5] * α2_0,   # C45: m=5, p=0
        cos_m[0] * α2_3,   # C50: m=0, p=3
        cos_m[2] * α2_2,   # C52: m=2, p=2
        cos_m[4] * α2_1,   # C54: m=4, p=1
        cos_m[6] * α2_0,   # C56: m=6, p=0
    ], axis=0)
    P_sin = cp.stack([
        sin_m[0] * α2_1,   # C10: m=0, p=1
        sin_m[2] * α2_0,   # C12: m=2, p=0
        sin_m[1] * α2_1,   # C21: m=1, p=1
        sin_m[3] * α2_0,   # C23: m=3, p=0
        sin_m[0] * α2_2,   # C30: m=0, p=2
        sin_m[2] * α2_1,   # C32: m=2, p=1
        sin_m[4] * α2_0,   # C34: m=4, p=0
        sin_m[1] * α2_2,   # C41: m=1, p=2
        sin_m[3] * α2_1,   # C43: m=3, p=1
        sin_m[5] * α2_0,   # C45: m=5, p=0
        sin_m[0] * α2_3,   # C50: m=0, p=3
        sin_m[2] * α2_2,   # C52: m=2, p=2
        sin_m[4] * α2_1,   # C54: m=4, p=1
        sin_m[6] * α2_0,   # C56: m=6, p=0
    ], axis=0)
    return P_cos, P_sin


def chi_cartesian(
    u: cp.ndarray,
    v: cp.ndarray,
    C_a: cp.ndarray,
    C_b: cp.ndarray,
    wavelength: float,
) -> cp.ndarray:
    r"""
    Compute wavefront aberration χ(u, v) in Cartesian k-space coordinates.
    
    This function computes χ directly from Cartesian coordinates (u, v) and
    Cartesian coefficients (C_a, C_b).
    
    Parameters
    ----------
    u : cp.ndarray
        Cartesian x-component, any shape. Related to polar: u = α cos(φ)
    v : cp.ndarray
        Cartesian y-component, same shape as u. Related to polar: v = α sin(φ)
    C_a : cp.ndarray
        Cartesian "cosine" coefficients, shape (14,).
        C_a = C_mag * cos(m * phi_nm) for each aberration.
    C_b : cp.ndarray
        Cartesian "sine" coefficients, shape (14,).
        C_b = C_mag * sin(m * phi_nm) for each aberration.
    wavelength : float
        Electron wavelength (same units as C_a, C_b)
        
    Returns
    -------
    chi0 : cp.ndarray
        Wavefront aberration in radians, same shape as u
    """
    # Validate input shapes
    if C_a.shape != (N_ABERRATIONS,):
        raise ValueError(f"C_a must have shape (14,), got {C_a.shape}")
    if C_b.shape != (N_ABERRATIONS,):
        raise ValueError(f"C_b must have shape (14,), got {C_b.shape}")
    # Get all P polynomials
    P_cos, P_sin = _compute_P_cos_sin(u, v)  # (14, *shape)
    # Prefactors: (2π/λ) / (n+1), shape (14,)
    n_vals = _N_VALUES.astype(cp.float32)
    prefactors = (2 * math.pi / wavelength) / (n_vals + 1)
    # Vectorized computation: reshape for broadcasting
    ndim = u.ndim
    prefactors = prefactors.reshape(-1, *([1] * ndim))
    C_a_bc = C_a.reshape(-1, *([1] * ndim))
    C_b_bc = C_b.reshape(-1, *([1] * ndim))
    # chi = Σ prefactor[i] * (C_a[i] * P_cos[i] + C_b[i] * P_sin[i])
    terms = prefactors * (C_a_bc * P_cos + C_b_bc * P_sin)  # (14, *shape)
    chi0 = terms.sum(axis=0)  # (*shape)
    return chi0
