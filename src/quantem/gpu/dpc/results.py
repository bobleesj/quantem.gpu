"""Result schema for differential phase contrast."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DPCResult:
    """Float32 DPC products in public ``(row, col)`` order."""

    phase: np.ndarray
    com_row: np.ndarray
    com_col: np.ndarray
    com_row_aligned: np.ndarray
    com_col_aligned: np.ndarray
    rotation_deg: float
    use_transpose: bool
    elapsed: float
