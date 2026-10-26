"""Chebyshev series along one aberration control on the GPU, for the CUDA drag windows.

The window rule (reach, truncation, order, width, nodes) is ``direct_ptycho.chebyshev``'s, shared with the MPS windows; this
module holds the coefficients on the device.
"""

import cupy as cp
import numpy as np

from quantem.gpu.direct_ptycho import chebyshev as rule

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
        self.nodes, weights = rule.nodes_and_weights(center, half_width, order)
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
        basis = cp.asarray(rule.basis(self.center, self.half_width, self.order, value))
        return basis.astype(cp.complex128) @ self.coefficients


def series(engine, aberrations_nm: dict, control: str) -> Series:
    """An empty series centred on the control's value, `rule.REACH` wide, its order from `rule.TRUNCATION`."""
    return Series(float(aberrations_nm.get(control, 0.0)), half_width(engine, aberrations_nm, control),
                  rule.order(control))


def half_width(engine, aberrations_nm: dict, control: str) -> float:
    """`rule.half_width` at the engine's wavelength and outer aperture angle."""
    cache = engine._cache
    outer = cache["semiangle_rad"] + 0.5 * max(cache["ang_y_rad"], cache["ang_x_rad"])
    return rule.half_width(control, aberrations_nm, engine.wavelength, outer)
