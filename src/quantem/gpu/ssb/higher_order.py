"""The 14-term aberration polynomial shared by Metal and CUDA reconstruction.

Coefficients are in Angstrom and azimuths in radians. The Cartesian polynomial
equals the polar Krivanek expansion, including at the origin, without a
per-frequency atan2. This does not change the native-grid fitting objective.
"""

import numpy as np

ORDERS = ((1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4),
          (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (5, 6))


def pack_coefficients(magnitudes_A: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """Pack the 14 Krivanek terms for either GPU shader.

    Parameters
    ----------
    magnitudes_A : numpy.ndarray
        Aberration magnitudes in Angstrom, in ``ORDERS`` order.
    angles_rad : numpy.ndarray
        Corresponding azimuths in radians.

    Returns
    -------
    numpy.ndarray
        Float32 cosine/sine coefficient pairs, shape (14, 2), including the
        radial prefactor 1 / (n + 1).
    """
    magnitudes = np.asarray(magnitudes_A, dtype=np.float32)
    angles = np.asarray(angles_rad, dtype=np.float32)
    if magnitudes.shape != (14,) or angles.shape != (14,):
        raise ValueError("Supply 14 aberration magnitudes and 14 azimuths.")
    if not np.isfinite(magnitudes).all() or not np.isfinite(angles).all():
        raise ValueError("Aberration magnitudes and azimuths must be finite.")
    n, m = np.asarray(ORDERS).T
    scaled = magnitudes.astype(np.float64) / (n + 1)
    return np.stack((scaled * np.cos(m * angles.astype(np.float64)),
                     scaled * np.sin(m * angles.astype(np.float64))), axis=1).astype(np.float32)


# SSB_DEVICE and SSB_COEFFICIENTS specify the backend's function/address space.
# Re[(alpha_row + i alpha_col)^m] and Im[...] provide the angular terms;
# (alpha_row^2 + alpha_col^2)^((n+1-m)/2) supplies the remaining radial power.
CHI_SOURCE = r"""
SSB_DEVICE float ssb_chi_full(float row, float col, float wl, SSB_COEFFICIENTS coefficients) {
    float x = row * wl, y = col * wl;
    float r2 = x * x + y * y;
    float radial[4] = {1.0f, r2, r2 * r2, r2 * r2 * r2};
    float re[7], im[7];
    re[0] = 1.0f; im[0] = 0.0f;
    for (int m = 1; m <= 6; ++m) {
        re[m] = re[m-1] * x - im[m-1] * y;
        im[m] = im[m-1] * x + re[m-1] * y;
    }
    const int azimuth[14] = {0,2,1,3,0,2,4,1,3,5,0,2,4,6};
    const int power[14] = {1,0,1,0,2,1,0,2,1,0,3,2,1,0};
    float chi = 0.0f;
    for (int i = 0; i < 14; ++i) {
        int m = azimuth[i];
        chi += radial[power[i]] * (re[m] * coefficients[2*i] + im[m] * coefficients[2*i+1]);
    }
    return (6.283185307179586f / wl) * chi;
}
"""
