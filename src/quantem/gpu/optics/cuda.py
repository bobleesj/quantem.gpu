"""Defocus, two-fold astigmatism and scan rotation from parallax shifts.

Each bright-field pixel at detector frequency ``k`` sees the scan image shifted
by ``lambda * M @ k``. A least-squares fit of the 2x2 matrix ``M`` and its polar
decomposition separate the scan-detector rotation from the symmetric aberration
part, which holds C10 and C12.
"""

import cupy as cp
import numpy as np


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
    # The fit runs in float64 whatever the sampling's type.
    kxa_1d = np.fft.fftfreq(gpts[0], sampling[0]).astype(np.float64)
    kya_1d = np.fft.fftfreq(gpts[1], sampling[1]).astype(np.float64)
    kxa_2d = np.broadcast_to(kxa_1d[:, None], gpts)
    kya_2d = np.broadcast_to(kya_1d[None, :], gpts)

    kx_bf = kxa_2d[bf_mask]
    ky_bf = kya_2d[bf_mask]
    basis = np.stack([kx_bf, ky_bf], axis=1) * float(wavelength)

    shifts_f64 = np.asarray(shifts_ang, dtype=np.float64)
    m_np = np.linalg.lstsq(basis, shifts_f64, rcond=None)[0]
    # Polar decomposition m = rotation @ aberration from the SVD m = U S Vh.
    m = cp.asarray(m_np, dtype=cp.float64)
    U, S, Vh = cp.linalg.svd(m)
    m_rotation = U @ Vh
    m_aberration = cp.conj(Vh.T) @ cp.diag(S) @ Vh

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
