"""Result schema for diffraction pattern stacks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    """Aligned stack plus the per-frame shifts that produced it."""

    aligned: np.ndarray
    shifts: np.ndarray
    used: np.ndarray
    elapsed: float
