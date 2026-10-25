"""Chebyshev series along one aberration control, shared by the drag windows.

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
"""

import math

import cupy as cp
import numpy as np

# z of a window along a magnitude C_nm and along an azimuth phi_nm, and the bound its order must meet
REACH = {"C": 10.0, "phi": 20.0}
TRUNCATION = {"C": 1e-10, "phi": 1e-14}

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

    def __init__(self, center: float, half_width: float, order: int):
        self.center, self.half_width, self.order = center, half_width, order
        nodes = np.cos(np.pi * (np.arange(order + 1) + 0.5) / (order + 1))
        self.nodes = list(center + (0.0 if math.isinf(half_width) else half_width) * nodes)
        weights = (2.0 / (order + 1)) * np.cos(np.outer(np.arange(order + 1), np.arccos(nodes)))
        weights[0] *= 0.5
        self.weights = cp.asarray(weights)
        self.coefficients = None
        self.added = 0

    @property
    def ready(self) -> bool:
        return self.added == self.order + 1

    def covers(self, value: float) -> bool:
        return abs(value - self.center) <= self.half_width

    def add(self, values):
        """Fold F at the next node (any complex128 array, flattened) into the coefficients."""
        values = values.ravel()
        if self.coefficients is None:
            self.coefficients = cp.zeros((self.order + 1, values.size), cp.complex128)
        _accumulate(values, self.weights[:, self.added], np.int64(values.size), self.coefficients)
        self.added += 1

    def evaluate(self, value: float):
        """F(value), flattened complex128."""
        x = 0.0 if math.isinf(self.half_width) else (value - self.center) / self.half_width
        basis = cp.asarray(np.cos(np.arange(self.order + 1) * np.arccos(np.clip(x, -1.0, 1.0))))
        return basis.astype(cp.complex128) @ self.coefficients


def series(engine, aberrations_nm: dict, control: str) -> Series:
    """An empty series centred on the control's value, `REACH` wide, its order from `TRUNCATION`."""
    family = "phi" if control.startswith("phi") else "C"
    reach, order = REACH[family], 12
    while 2 * (reach / 2) ** (order + 1) / math.factorial(order + 1) > TRUNCATION[family]:
        order += 1
    return Series(float(aberrations_nm.get(control, 0.0)), half_width(engine, aberrations_nm, control, reach), order)


def half_width(engine, aberrations_nm: dict, control: str, reach: float) -> float:
    """h = reach / max|u|, max|u| the range of d chi / d control over the aperture, in radians per unit.

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
    return reach / spread if spread > 0 else math.inf
