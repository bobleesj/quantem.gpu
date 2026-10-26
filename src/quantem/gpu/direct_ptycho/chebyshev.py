"""Chebyshev series along one aberration control: the window rule every backend's drag windows share.

Serves: ShowPtycho drags of one control (C10, C12, phi12 typically) at 3x on
the full 512 x 512 scan. Whatever a window holds (regularized normal
equations, or the single-sideband part of the SSB spectrum) is a sum over BF
pixels of f_k exp(i (c - c0) u_k), an entire function of the control c, so on
[c0 - h, c0 + h] its Chebyshev series converges like 2 (z / 2)^(N + 1) / (N + 1)!,
z = h max|u|. The reach z sets the width: 10 for a magnitude (C10 +-6.8 nm,
C12 +-3.4 nm on the test field), 20 for an azimuth, whose slider spans
180 deg (phi12 +-28 deg at C12 = 6.9 nm); N = 12, z = 3 gave +-2 nm, which a
hand drag leaves within a few frames. N is the smallest order whose bound
meets `TRUNCATION`: a Gram entry is a sum with cancellation, so the image
reaches the float32 floor (4-7e-7 of its largest phase) only at a bound of
1e-10 for z = 10 (N = 28) and 1e-14 for z = 20 (N = 49), measured; a 1e-8
bound left 2-4e-6 at z = 20.

The series storage and arithmetic belong to each backend (``direct_ptycho.cuda.chebyshev``, ``direct_ptycho.mps.direct``).
"""

import math

import numpy as np

# z of a window along a magnitude C_nm and along an azimuth phi_nm, and the bound its order must meet
REACH = {"C": 10.0, "phi": 20.0}
TRUNCATION = {"C": 1e-10, "phi": 1e-14}


def _family(control: str) -> str:
    return "phi" if control.startswith("phi") else "C"


def order(control: str) -> int:
    """N: the smallest order from 12 whose bound 2 (z / 2)^(N + 1) / (N + 1)! meets the control's `TRUNCATION`."""
    family = _family(control)
    reach, value = REACH[family], 12
    while 2 * (reach / 2) ** (value + 1) / math.factorial(value + 1) > TRUNCATION[family]:
        value += 1
    return value


def half_width(control: str, aberrations_nm: dict, wavelength_A: float, outer_angle_rad: float) -> float:
    """h = z / max|u|, max|u| the range of d chi / d control over the aperture, in radians per unit.

    chi enters every term as a difference of two pupil points, so max|u| is
    the spread of the derivative: (2 pi / lambda) alpha^(n+1) / (n + 1) per A
    for a magnitude C_nm, doubled for m > 0 where it changes sign; for an
    azimuth phi_nm, m |C_nm| times that per radian, doubled. ``outer_angle_rad``
    is the soft aperture's outer edge, the semiangle plus half a detector pixel.
    A control that does not change the model (an azimuth whose magnitude is zero)
    has h = inf.
    """
    digits = control.split("_")[0].lstrip("Cphi")
    n, m = int(digits[0]), int(digits[1])
    per_A = 2 * np.pi / wavelength_A * outer_angle_rad ** (n + 1) / (n + 1)
    if control.startswith("phi"):
        spread = 2 * m * per_A * 10 * abs(aberrations_nm.get(f"C{n}{m}_nm", 0.0)) * np.pi / 180
    else:
        spread = (2 if m else 1) * per_A * 10
    return REACH[_family(control)] / spread if spread > 0 else math.inf


def nodes_and_weights(center: float, half_width_value: float, order_value: int) -> tuple[list[float], np.ndarray]:
    """The nodes c0 + h x_j, x_j = cos(pi (j + 1/2) / (N + 1)), and the weights (2 / (N + 1)) T_n(x_j), row 0 halved.

    The coefficients of F are c_n = sum_j weights[n, j] F(x_j); a window with h = inf puts every node at c0.
    """
    x = np.cos(np.pi * (np.arange(order_value + 1) + 0.5) / (order_value + 1))
    nodes = list(center + (0.0 if math.isinf(half_width_value) else half_width_value) * x)
    weights = (2.0 / (order_value + 1)) * np.cos(np.outer(np.arange(order_value + 1), np.arccos(x)))
    weights[0] *= 0.5
    return nodes, weights


def basis(center: float, half_width_value: float, order_value: int, value: float) -> np.ndarray:
    """T_n((value - c0) / h), n = 0 .. N, float64; the argument is clipped to [-1, 1] (0 for h = inf)."""
    x = 0.0 if math.isinf(half_width_value) else (value - center) / half_width_value
    return np.cos(np.arange(order_value + 1) * np.arccos(np.clip(x, -1.0, 1.0)))
