"""Trial rows of the higher-order agreement kernel (``mps.higher_order_agreement``): (thickness_A, tilt_row_mrad,
tilt_col_mrad, then 2 N Cartesian Krivanek coefficients in Angstrom, ``higher_order.ORDERS`` order).

The ``find_aberrations(higher_order=...)`` subset polish that used them is retired: the fifth-order fit fits every term
through C56 on exact gradients, with the half-band check.
"""

import numpy as np

from quantem.gpu.direct_ptycho.higher_order import ORDERS


def num_terms(rows: np.ndarray) -> int:
    """Krivanek terms an agreement row carries: (width - 3) / 2."""
    width = np.shape(rows)[-1]
    if width < 5 or (width - 3) % 2 or (width - 3) // 2 > len(ORDERS):
        raise ValueError(f"A trial row is (thickness, tilt_row, tilt_col) and 2 N Cartesian coefficients; got {width}.")
    return (width - 3) // 2


def packed(rows: np.ndarray) -> np.ndarray:
    """Agreement rows -> the kernels' float32 rows: tilt in rad and each Cartesian coefficient divided by (n + 1)."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    terms = num_terms(rows)
    order = np.repeat(np.asarray([n for n, _ in ORDERS[:terms]], dtype=np.float64), 2)
    out = np.empty_like(rows)
    out[:, 0] = rows[:, 0]
    out[:, 1:3] = rows[:, 1:3] * 1e-3
    out[:, 3:] = rows[:, 3:] / (order + 1.0)
    return out.astype(np.float32)
