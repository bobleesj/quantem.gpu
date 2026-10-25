"""Chebyshev series along one aberration control, shared by the drag windows.

Serves: ShowPtycho drags of one control (C10, C12, phi12 typically) at 3x on
the full 512 x 512 scan. Whatever a window holds (regularized normal
equations, or the single-sideband part of the SSB spectrum) is a sum over BF
pixels of f_k exp(i (c - c0) u_k), an entire function of the control c, so on
[c0 - h, c0 + h] its Chebyshev series converges like 2 (z / 2)^(N + 1) / (N + 1)!,
z = h max|u|. N = 24 and z = 10 bound the truncation by 4e-8, below the
float32 model's own error, over windows wide enough for a fast drag (C10
+-6.8 nm, C12 +-3.4 nm, phi12 +-14 deg on the test field; N = 12, z = 3 gave
+-2 nm, which a hand drag leaves within a few frames).
"""

import math

import cupy as cp
import numpy as np

ORDER = 24
REACH = 10.0

# coefficient row n gains weight[n] * node values: one node of a series
_accumulate = cp.ElementwiseKernel(
    "raw complex128 values, raw float64 weight, int64 size",
    "complex128 coefficient",
    "coefficient += weight[i / size] * values[i % size]",
    "direct_window_accumulate",
)


class Series:
    """Coefficients c_n = (2 / (N + 1)) sum_j T_n(x_j) F(x_j), c_0 halved, from F at the nodes x_j = cos(pi (j + 1/2) / (N + 1)).

    ``add`` takes F at ``nodes[added]`` (one exact evaluation); once all N + 1
    are in, ``evaluate`` returns F(c) = sum_n c_n T_n((c - c0) / h) as one
    matrix-vector product. A control that does not change the model (an
    azimuth whose magnitude is zero) has h = inf and every node at c0.
    """

    def __init__(self, center: float, half_width: float):
        self.center, self.half_width = center, half_width
        nodes = np.cos(np.pi * (np.arange(ORDER + 1) + 0.5) / (ORDER + 1))
        self.nodes = list(center + (0.0 if math.isinf(half_width) else half_width) * nodes)
        weights = (2.0 / (ORDER + 1)) * np.cos(np.outer(np.arange(ORDER + 1), np.arccos(nodes)))
        weights[0] *= 0.5
        self.weights = cp.asarray(weights)
        self.coefficients = None
        self.added = 0

    @property
    def ready(self) -> bool:
        return self.added == ORDER + 1

    def covers(self, value: float) -> bool:
        return abs(value - self.center) <= self.half_width

    def add(self, values):
        """Fold F at the next node (any complex128 array, flattened) into the coefficients."""
        values = values.ravel()
        if self.coefficients is None:
            self.coefficients = cp.zeros((ORDER + 1, values.size), cp.complex128)
        _accumulate(values, self.weights[:, self.added], np.int64(values.size), self.coefficients)
        self.added += 1

    def evaluate(self, value: float):
        """F(value), flattened complex128."""
        x = 0.0 if math.isinf(self.half_width) else (value - self.center) / self.half_width
        basis = cp.asarray(np.cos(np.arange(ORDER + 1) * np.arccos(np.clip(x, -1.0, 1.0))))
        return basis.astype(cp.complex128) @ self.coefficients


def half_width(engine, aberrations_nm: dict, control: str) -> float:
    """h = REACH / max|u|, max|u| the range of d chi / d control over the aperture, in radians per unit.

    chi enters every term as a difference of two pupil points, so max|u| is
    the spread of the derivative: (2 pi / lambda) alpha^(n+1) / (n + 1) per A
    for a magnitude C_nm, doubled for m > 0 where it changes sign; for an
    azimuth phi_nm, m |C_nm| times that per radian, doubled.
    """
    digits = control.split("_")[0].lstrip("Cphi")
    n, m = int(digits[0]), int(digits[1])
    cache = engine._cache
    alpha = cache["semiangle_rad"] + 0.5 * max(cache["ang_y_rad"], cache["ang_x_rad"])
    per_A = 2 * np.pi / engine.wavelength * alpha ** (n + 1) / (n + 1)
    if control.startswith("phi"):
        spread = 2 * m * per_A * 10 * abs(aberrations_nm.get(f"C{n}{m}_nm", 0.0)) * np.pi / 180
    else:
        spread = (2 if m else 1) * per_A * 10
    return REACH / spread if spread > 0 else math.inf
