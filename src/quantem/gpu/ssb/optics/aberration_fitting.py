r"""
Aberration fitting from pixel shifts (CuPy GPU implementation).

Recovers aberration coefficients from measured real-space shifts by solving
a linear least-squares problem.

This module complements :mod:`quantem.gpu.ssb.optics.aberration`, which computes
the forward model (aberration phase χ from coefficients). This module solves
the inverse problem (coefficients from measured shifts).

Physical relationship
---------------------

When the probe is tilted by angle :math:`\alpha`, the image shifts by:

.. math::

    S = C_{10} \times \alpha

for defocus, and similarly for higher-order aberrations:

.. math::

    S_x = \sum_{n,m} \frac{C_{nm}}{n+1} \frac{\partial P_{nm}}{\partial u}

Numerical stability
-------------------

Internally, all calculations are performed in Ångströms rather than meters
for numerical stability when fitting high-order aberrations.

See Also
--------
quantem.gpu.ssb.optics.aberration : Forward model (χ from coefficients)
"""

import cupy as cp
import numpy as np

from .aberration import ABERRATION_INDICES
from .physics import wavelength_A_from_kV

# Alias for backward compatibility
ABERRATION_INFO = ABERRATION_INDICES

# Coefficient labels in order (25 total)
COEFF_LABELS = []
for n, m, name, has_angle in ABERRATION_INDICES:
    COEFF_LABELS.append(name if m == 0 else f"{name}_a")
    if has_angle:
        COEFF_LABELS.append(f"{name}_b")


def _svd_polar(m: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Return the polar decomposition ``m = u @ p`` using SVD."""
    U, S, Vh = cp.linalg.svd(m)
    u = U @ Vh
    p = cp.conj(Vh.T) @ cp.diag(S).astype(m.dtype) @ Vh
    return u, p


def fit_aberrations_svd_polar(
    shifts_ang: np.ndarray,
    bf_mask: np.ndarray,
    wavelength: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
) -> dict[str, float]:
    """Fit C10, C12, phi12, and rotation from parallax shifts.

    This mirrors ``quantem.diffractive_imaging.direct_ptycho_utils``. It fits a
    2x2 matrix from measured shifts, then uses a polar decomposition to separate
    scan/detector rotation from the symmetric aberration component.

    Parameters
    ----------
    shifts_ang
        Measured parallax shifts in angstroms, shape ``(n_bf, 2)``.
    bf_mask
        Boolean detector mask selecting the BF pixels used for the fit.
    wavelength
        Electron wavelength in angstroms.
    gpts
        Detector grid shape ``(row, col)``.
    sampling
        Detector reciprocal-space sampling in ``1 / angstrom``.

    Returns
    -------
    dict[str, float]
        ``C10`` and ``C12`` in angstroms, ``phi12`` and ``rotation_angle`` in
        radians.
    """
    kxa_1d = np.fft.fftfreq(gpts[0], sampling[0]).astype(np.float64)
    kya_1d = np.fft.fftfreq(gpts[1], sampling[1]).astype(np.float64)
    kxa_2d = np.broadcast_to(kxa_1d[:, None], gpts)
    kya_2d = np.broadcast_to(kya_1d[None, :], gpts)

    kx_bf = kxa_2d[bf_mask]
    ky_bf = kya_2d[bf_mask]
    basis = np.stack([kx_bf, ky_bf], axis=1) * float(wavelength)

    shifts_f64 = np.asarray(shifts_ang, dtype=np.float64)
    m_np, _, _, _ = np.linalg.lstsq(basis, shifts_f64, rcond=None)
    m_rotation, m_aberration = _svd_polar(cp.asarray(m_np, dtype=cp.float64))

    rotation_rad = float(-cp.arctan2(m_rotation[1, 0], m_rotation[0, 0]).get())
    wrapped = (rotation_rad + np.pi) % (2 * np.pi) - np.pi
    if 2 * abs(wrapped) > np.pi:
        rotation_rad = rotation_rad % (2 * np.pi) - np.pi
        m_aberration = -m_aberration

    a = float(m_aberration[0, 0].get())
    b = float(((m_aberration[1, 0] + m_aberration[0, 1]) / 2).get())
    c = float(m_aberration[1, 1].get())
    c10 = (a + c) / 2
    c12a = (a - c) / 2
    c12b = b
    c12 = np.sqrt(c12a**2 + c12b**2)
    phi12 = np.arctan2(c12b, c12a) / 2

    return {
        "C10": c10,
        "C12": float(c12),
        "phi12": float(phi12),
        "rotation_angle": rotation_rad,
    }


def _compute_gradient_polynomials(
    u: cp.ndarray, v: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    r"""
    Compute gradients of aberration polynomials analytically.
    
    Returns :math:`\partial P / \partial u` and :math:`\partial P / \partial v`
    for all 14 aberration polynomials (both cos and sin components).
    
    Parameters
    ----------
    u, v : cp.ndarray
        K-space coordinates (beam tilt) in radians, any shape.
        
    Returns
    -------
    dPc_du, dPc_dv, dPs_du, dPs_dv : cp.ndarray
        Gradients of cos and sin polynomials, shape (14, *u.shape).
    """
    u2 = u * u
    v2 = v * v
    u3 = u2 * u
    v3 = v2 * v
    u4 = u2 * u2
    v4 = v2 * v2
    u5 = u4 * u
    v5 = v4 * v
    r2 = u2 + v2
    r4 = r2 * r2
    # ∂P_cos/∂u for all 14 aberrations
    dPc_du = cp.stack([
        2*u,                                    # C10 (1,0)
        2*u,                                    # C12 (1,2)
        3*u2 + v2,                              # C21 (2,1)
        3*u2 - 3*v2,                            # C23 (2,3)
        4*u*r2,                                 # C30 (3,0)
        4*u3,                                   # C32 (3,2)
        4*u3 - 12*u*v2,                         # C34 (3,4)
        5*u4 + 6*u2*v2 + v4,                    # C41 (4,1)
        5*u4 - 6*u2*v2 - 3*v4,                  # C43 (4,3)
        5*u4 - 30*u2*v2 + 5*v4,                 # C45 (4,5)
        6*u*r4,                                 # C50 (5,0)
        6*u5 + 4*u3*v2 - 2*u*v4,                # C52 (5,2)
        6*u5 - 20*u3*v2 - 10*u*v4,              # C54 (5,4)
        6*u5 - 60*u3*v2 + 30*u*v4,              # C56 (5,6)
    ], axis=0)
    # ∂P_cos/∂v for all 14 aberrations
    dPc_dv = cp.stack([
        2*v,                                    # C10 (1,0)
        -2*v,                                   # C12 (1,2)
        2*u*v,                                  # C21 (2,1)
        -6*u*v,                                 # C23 (2,3)
        4*v*r2,                                 # C30 (3,0)
        -4*v3,                                  # C32 (3,2)
        -12*u2*v + 4*v3,                        # C34 (3,4)
        4*u*v*r2,                               # C41 (4,1)
        -4*u3*v - 12*u*v3,                      # C43 (4,3)
        -20*u3*v + 20*u*v3,                     # C45 (4,5)
        6*v*r4,                                 # C50 (5,0)
        2*u4*v - 4*u2*v3 - 6*v5,                # C52 (5,2)
        -10*u4*v - 20*u2*v3 + 6*v5,             # C54 (5,4)
        -30*u4*v + 60*u2*v3 - 6*v5,             # C56 (5,6)
    ], axis=0)
    # ∂P_sin/∂u for all 14 aberrations
    zeros = cp.zeros_like(u)
    dPs_du = cp.stack([
        zeros,                                  # C10 (1,0) - m=0
        2*v,                                    # C12 (1,2)
        2*u*v,                                  # C21 (2,1)
        6*u*v,                                  # C23 (2,3)
        zeros,                                  # C30 (3,0) - m=0
        6*u2*v + 2*v3,                          # C32 (3,2)
        12*u2*v - 4*v3,                         # C34 (3,4)
        4*u*v*r2,                               # C41 (4,1)
        12*u3*v + 4*u*v3,                       # C43 (4,3)
        20*u3*v - 20*u*v3,                      # C45 (4,5)
        zeros,                                  # C50 (5,0) - m=0
        10*u4*v + 12*u2*v3 + 2*v5,              # C52 (5,2)
        20*u4*v - 4*v5,                         # C54 (5,4)
        30*u4*v - 60*u2*v3 + 6*v5,              # C56 (5,6)
    ], axis=0)
    # ∂P_sin/∂v for all 14 aberrations
    dPs_dv = cp.stack([
        zeros,                                  # C10 (1,0) - m=0
        2*u,                                    # C12 (1,2)
        u2 + 3*v2,                              # C21 (2,1)
        3*u2 - 3*v2,                            # C23 (2,3)
        zeros,                                  # C30 (3,0) - m=0
        2*u3 + 6*u*v2,                          # C32 (3,2)
        4*u3 - 12*u*v2,                         # C34 (3,4)
        u4 + 6*u2*v2 + 5*v4,                    # C41 (4,1)
        3*u4 + 6*u2*v2 - 5*v4,                  # C43 (4,3)
        5*u4 - 30*u2*v2 + 5*v4,                 # C45 (4,5)
        zeros,                                  # C50 (5,0) - m=0
        2*u5 + 12*u3*v2 + 10*u*v4,              # C52 (5,2)
        4*u5 - 20*u*v4,                         # C54 (5,4)
        6*u5 - 60*u3*v2 + 30*u*v4,              # C56 (5,6)
    ], axis=0)
    return dPc_du, dPc_dv, dPs_du, dPs_dv


def fit_aberrations(
    kx_pix: cp.ndarray,
    ky_pix: cp.ndarray,
    Sx_pix: cp.ndarray,
    Sy_pix: cp.ndarray,
    delta_k_A: float,
    delta_r_A: float,
    voltage_kV: float,
) -> dict[str, float]:
    r"""
    Fit aberration coefficients from pixel-space measurements.

    Takes k-space positions (in pixels) and measured real-space shifts (in pixels),
    converts to physical units, and solves the linear system to recover aberration
    coefficients.

    All length units are in Ångströms for numerical stability.

    Parameters
    ----------
    kx_pix, ky_pix : cp.ndarray
        K-space positions in pixels (from diffraction pattern center)
    Sx_pix, Sy_pix : cp.ndarray
        Measured shifts in pixels (from cross-correlation of BF images)
    delta_k_A : float
        K-space pixel size in 1/Å per pixel.
        Computed as: :math:`1 / (N \cdot \Delta r)` where N is detector size.
    delta_r_A : float
        Real-space pixel size in Å per pixel
    voltage_kV : float
        Accelerating voltage in kilovolts (e.g., 300 for 300 keV).

    Returns
    -------
    coeffs : dict[str, float]
        Fitted coefficients in Ångströms. Keys are:

        - 'C10' : defocus
        - 'C12_a', 'C12_b' : 2-fold astigmatism
        - 'C21_a', 'C21_b' : coma
        - 'C23_a', 'C23_b' : 3-fold astigmatism
        - 'C30' : spherical aberration
        - etc.

    Examples
    --------
    >>> import cupy as cp
    >>> from quantem.gpu.ssb.optics.aberration_fitting import fit_aberrations
    >>> 
    >>> # Experimental parameters
    >>> voltage_kV = 300
    >>> delta_r_A = 0.5
    >>> N_dp = 256
    >>> delta_k_A = 1 / (N_dp * delta_r_A)
    >>> 
    >>> # From tcBF measurements
    >>> kx_pix = cp.array([...])  # BF disk positions
    >>> ky_pix = cp.array([...])
    >>> Sx_pix = cp.array([...])  # measured shifts
    >>> Sy_pix = cp.array([...])
    >>> 
    >>> coeffs = fit_aberrations(
    ...     kx_pix, ky_pix, Sx_pix, Sy_pix,
    ...     delta_k_A, delta_r_A, voltage_kV
    ... )
    >>> print(f"Defocus: {coeffs['C10'] / 10:.1f} nm")
    """
    fitter = AberrationFitter(kx_pix, ky_pix, delta_k_A, delta_r_A, voltage_kV, dtype='float64')
    return fitter.fit(Sx_pix, Sy_pix)


def compute_shifts_from_aberrations(
    kx_pix: cp.ndarray,
    ky_pix: cp.ndarray,
    coeffs: dict[str, float],
    delta_k_A: float,
    delta_r_A: float,
    voltage_kV: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    r"""
    Compute expected shifts from aberration coefficients (forward model).
    
    Given fitted aberration coefficients, compute the expected real-space
    shifts at each k-space position. Useful for validating fits or
    generating synthetic data.
    
    Parameters
    ----------
    kx_pix, ky_pix : cp.ndarray
        K-space positions in pixels
    coeffs : dict[str, float]
        Aberration coefficients in Ångströms (from fit_aberrations)
    delta_k_A : float
        K-space pixel size in 1/Å per pixel
    delta_r_A : float
        Real-space pixel size in Å per pixel
    voltage_kV : float
        Accelerating voltage in kilovolts
        
    Returns
    -------
    Sx_pix, Sy_pix : cp.ndarray
        Predicted shifts in pixels
    """
    wavelength_A = wavelength_A_from_kV(voltage_kV)
    # Beam tilt coordinates
    u = kx_pix.flatten() * delta_k_A * wavelength_A
    v = ky_pix.flatten() * delta_k_A * wavelength_A
    # Compute gradients
    dPc_du, dPc_dv, dPs_du, dPs_dv = _compute_gradient_polynomials(u, v)    
    # Accumulate shifts
    Sx_A = cp.zeros_like(u)
    Sy_A = cp.zeros_like(v)
    for i, (n, m, name, has_angle) in enumerate(ABERRATION_INFO):
        factor = 1.0 / (n + 1)
        # 'a' component
        key_a = name if m == 0 else f"{name}_a"
        if key_a in coeffs:
            Sx_A += factor * coeffs[key_a] * dPc_du[i]
            Sy_A += factor * coeffs[key_a] * dPc_dv[i]
        # 'b' component
        if has_angle:
            key_b = f"{name}_b"
            if key_b in coeffs:
                Sx_A += factor * coeffs[key_b] * dPs_du[i]
                Sy_A += factor * coeffs[key_b] * dPs_dv[i]
    # Convert back to pixels
    Sx_pix = Sx_A / delta_r_A
    Sy_pix = Sy_A / delta_r_A
    return Sx_pix.reshape(kx_pix.shape), Sy_pix.reshape(ky_pix.shape)


class AberrationFitter:
    """
    Fast batch aberration fitting with pre-computed pseudo-inverse.

    For 4D-STEM datasets where k-space positions are fixed across all scan
    positions, pre-computing the pseudo-inverse reduces each fit from an
    expensive lstsq (SVD) to a single matrix-vector multiply.

    ~16,000x faster than sequential lstsq for 256x256 scans.
    With FP32 and GPU-native arrays: <5ms for 256x256 scans.

    Parameters
    ----------
    kx_pix, ky_pix : array-like
        K-space positions in pixels (1D arrays, n_k elements).
        Can be CuPy or NumPy arrays.
    delta_k_A : float
        K-space pixel size in 1/Å per pixel
    delta_r_A : float
        Real-space pixel size in Å per pixel
    voltage_kV : float
        Accelerating voltage in kilovolts
    dtype : str, optional
        Data type: 'float32' for speed (~2ms) or 'float64' for precision (~22ms).
        Default is 'float32'. For most aberration fitting, FP32 is sufficient.

    Examples
    --------
    >>> # Setup fitter once (computes pseudo-inverse)
    >>> fitter = AberrationFitter(kx_pix, ky_pix, delta_k_A, delta_r_A, 300.0)
    >>>
    >>> # Fast batch fitting for entire 4D-STEM dataset
    >>> # Sx_all, Sy_all have shape (n_k, scan_y, scan_x)
    >>> coeffs = fitter.fit_batch(Sx_all, Sy_all)
    >>> # coeffs has shape (25, scan_y, scan_x)
    >>>
    >>> # Or fit single position
    >>> coeffs_dict = fitter.fit(Sx_pix, Sy_pix)
    """

    def __init__(
        self,
        kx_pix,
        ky_pix,
        delta_k_A: float,
        delta_r_A: float,
        voltage_kV: float,
        dtype: str = 'float32',
    ):
        self.delta_r_A = delta_r_A
        self.dtype = np.float32 if dtype == 'float32' else np.float64
        self.cp_dtype = cp.float32 if dtype == 'float32' else cp.float64

        # Convert to numpy for setup (avoids GPU library issues)
        # Handle both CuPy and NumPy arrays
        if hasattr(kx_pix, 'get'):  # CuPy array
            kx_np = kx_pix.get().flatten().astype(np.float64)
            ky_np = ky_pix.get().flatten().astype(np.float64)
        else:
            kx_np = np.asarray(kx_pix).flatten().astype(np.float64)
            ky_np = np.asarray(ky_pix).flatten().astype(np.float64)
        self.n_k = len(kx_np)
        self.n_coeffs = len(COEFF_LABELS)
        # Build design matrix using NumPy (FP64 for accuracy during setup)
        wavelength_A = wavelength_A_from_kV(voltage_kV)
        u = kx_np * delta_k_A * wavelength_A
        v = ky_np * delta_k_A * wavelength_A
        # Compute gradient polynomials (NumPy version)
        u2, v2 = u * u, v * v
        u3, v3 = u2 * u, v2 * v
        u4, v4 = u2 * u2, v2 * v2
        u5, v5 = u4 * u, v4 * v
        r2, r4 = u2 + v2, (u2 + v2) ** 2
        # dP_cos/du for all 14 aberrations
        dPc_du = np.array([
            2*u, 2*u, 3*u2 + v2, 3*u2 - 3*v2, 4*u*r2, 4*u3, 4*u3 - 12*u*v2,
            5*u4 + 6*u2*v2 + v4, 5*u4 - 6*u2*v2 - 3*v4, 5*u4 - 30*u2*v2 + 5*v4,
            6*u*r4, 6*u5 + 4*u3*v2 - 2*u*v4, 6*u5 - 20*u3*v2 - 10*u*v4,
            6*u5 - 60*u3*v2 + 30*u*v4,
        ])
        dPc_dv = np.array([
            2*v, -2*v, 2*u*v, -6*u*v, 4*v*r2, -4*v3, -12*u2*v + 4*v3,
            4*u*v*r2, -4*u3*v - 12*u*v3, -20*u3*v + 20*u*v3,
            6*v*r4, 2*u4*v - 4*u2*v3 - 6*v5, -10*u4*v - 20*u2*v3 + 6*v5,
            -30*u4*v + 60*u2*v3 - 6*v5,
        ])
        zeros = np.zeros_like(u)
        dPs_du = np.array([
            zeros, 2*v, 2*u*v, 6*u*v, zeros, 6*u2*v + 2*v3, 12*u2*v - 4*v3,
            4*u*v*r2, 12*u3*v + 4*u*v3, 20*u3*v - 20*u*v3,
            zeros, 10*u4*v + 12*u2*v3 + 2*v5, 20*u4*v - 4*v5,
            30*u4*v - 60*u2*v3 + 6*v5,
        ])
        dPs_dv = np.array([
            zeros, 2*u, u2 + 3*v2, 3*u2 - 3*v2, zeros, 2*u3 + 6*u*v2, 4*u3 - 12*u*v2,
            u4 + 6*u2*v2 + 5*v4, 3*u4 + 6*u2*v2 - 5*v4, 5*u4 - 30*u2*v2 + 5*v4,
            zeros, 2*u5 + 12*u3*v2 + 10*u*v4, 4*u5 - 20*u*v4,
            6*u5 - 60*u3*v2 + 30*u*v4,
        ])
        # Build design matrix columns
        columns_du, columns_dv = [], []
        for i, (n, m, name, has_angle) in enumerate(ABERRATION_INFO):
            factor = 1.0 / (n + 1)
            columns_du.append(factor * dPc_du[i])
            columns_dv.append(factor * dPc_dv[i])
            if has_angle:
                columns_du.append(factor * dPs_du[i])
                columns_dv.append(factor * dPs_dv[i])
        # A has shape (2*n_k, 25)
        A_du = np.array(columns_du)  # (25, n_k)
        A_dv = np.array(columns_dv)  # (25, n_k)
        A = np.zeros((2 * self.n_k, self.n_coeffs), dtype=np.float64)
        A[0::2, :] = A_du.T
        A[1::2, :] = A_dv.T
        # Compute pseudo-inverse using NumPy FP64 (one-time cost)
        A_pinv_f64 = np.linalg.pinv(A)
        # Split pseudo-inverse into Sx and Sy components for faster computation
        # A_pinv has shape (25, 2*n_k) where even cols are Sx, odd cols are Sy
        # Pre-scale by delta_r_A to avoid runtime multiplication
        A_pinv_x_f64 = A_pinv_f64[:, 0::2] * delta_r_A  # (25, n_k) - contributions from Sx
        A_pinv_y_f64 = A_pinv_f64[:, 1::2] * delta_r_A  # (25, n_k) - contributions from Sy
        # Store CPU versions in target dtype
        self._A_pinv_x_np = A_pinv_x_f64.astype(self.dtype)
        self._A_pinv_y_np = A_pinv_y_f64.astype(self.dtype)
        self._A_pinv_np = A_pinv_f64.astype(self.dtype)  # Keep for single-position fit
        # Create GPU versions in target dtype. quantem.gpu requires CUDA;
        # failures should be visible instead of falling back to CPU math.
        try:
            self.A_pinv_x = cp.asarray(self._A_pinv_x_np, dtype=self.cp_dtype)
            self.A_pinv_y = cp.asarray(self._A_pinv_y_np, dtype=self.cp_dtype)
        except (MemoryError, RuntimeError, ImportError) as exc:
            raise RuntimeError(
                "CUDA/CuPy is required for aberration fitting in quantem.gpu."
            ) from exc

    def fit(self, Sx_pix, Sy_pix) -> dict[str, float]:
        """
        Fit aberrations for a single scan position.

        Parameters
        ----------
        Sx_pix, Sy_pix : array-like
            Measured shifts in pixels, shape (n_k,)

        Returns
        -------
        coeffs : dict[str, float]
            Fitted aberration coefficients in Ångströms
        """
        coeffs = self.fit_batch(
            cp.asarray(Sx_pix, dtype=self.cp_dtype).reshape(self.n_k, 1),
            cp.asarray(Sy_pix, dtype=self.cp_dtype).reshape(self.n_k, 1),
        ).reshape(self.n_coeffs)
        coeffs_host = cp.asnumpy(coeffs)
        return {label: float(coeffs_host[i]) for i, label in enumerate(COEFF_LABELS)}

    def fit_batch(self, Sx_all, Sy_all):
        """
        Fit aberrations for all scan positions in a single operation.

        For maximum speed, pass CuPy arrays that are already on GPU.
        With FP32: ~2ms for 256x256 scan.
        With FP64: ~22ms for 256x256 scan.

        Parameters
        ----------
        Sx_all, Sy_all : array-like
            Measured shifts for all scan positions.
            Shape: (n_k, scan_y, scan_x) or (n_k, n_scan)
            For best performance, use CuPy arrays already on GPU.

        Returns
        -------
        coeffs : ndarray
            Fitted coefficients, shape (25, scan_y, scan_x) or (25, n_scan).
            Coefficients are in Ångströms, ordered as COEFF_LABELS.
            Returns CuPy array if GPU available, else NumPy.
        """
        if hasattr(Sx_all, '__cuda_array_interface__'):
            return self._fit_batch_gpu_native(Sx_all, Sy_all)
        return self._fit_batch_gpu(Sx_all, Sy_all)

    def _fit_batch_gpu_native(self, Sx_all, Sy_all):
        """GPU batch fitting with data already on GPU (fastest path).

        Uses two separate GEMMs instead of building interleaved B matrix,
        which is ~3x faster due to avoiding expensive memory interleaving.
        Scaling by delta_r_A is pre-computed in the pseudo-inverse.
        """
        original_shape = Sx_all.shape[1:]
        # Reshape without copy (zero-cost if contiguous)
        Sx_flat = Sx_all.reshape(self.n_k, -1)
        Sy_flat = Sy_all.reshape(self.n_k, -1)
        # Two GEMMs + add: coeffs = A_pinv_x @ Sx + A_pinv_y @ Sy
        # delta_r_A scaling is pre-computed in A_pinv_x/y
        coeffs_flat = self.A_pinv_x @ Sx_flat + self.A_pinv_y @ Sy_flat
        return coeffs_flat.reshape(self.n_coeffs, *original_shape)

    def _fit_batch_gpu(self, Sx_all, Sy_all):
        """GPU batch fitting with CPU->GPU transfer."""
        # Convert to GPU with target dtype
        Sx_gpu = cp.asarray(Sx_all, dtype=self.cp_dtype)
        Sy_gpu = cp.asarray(Sy_all, dtype=self.cp_dtype)
        return self._fit_batch_gpu_native(Sx_gpu, Sy_gpu)
