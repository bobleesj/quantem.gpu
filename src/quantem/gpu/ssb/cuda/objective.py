"""The phase-variance objective that the CUDA aberration search minimises.

For candidate aberrations the loss is the mean over scan pixels of the variance, over bright-field pixels, of the
corrected phase: the better the correction, the more every bright-field pixel agrees on the object phase. Each candidate
is the full reconstruction with a fixed-order reduction, so seeded fits get the same loss bits on every evaluation and a
fit, a reconstruction and a preview report the same loss for the same aberrations.
"""

import cupy as cp
import numpy as np


class PhaseVarianceObjective:
    """The aberration-search loss on one ``SSBEngine``'s prepared ``G_qk`` and rotation geometry."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def loss(self, C10: float, C12: float, phi12: float) -> float:
        """Loss of one candidate (C10, C12 in Angstrom, phi12 in radians)."""
        _, loss = self.engine.reconstruct_with_loss(C10, C12, phi12)
        return loss

    def loss_batch(
        self,
        C10: np.ndarray,
        C12: np.ndarray,
        phi12: np.ndarray,
        out: cp.ndarray | None = None,
    ) -> cp.ndarray:
        """Losses of a batch of candidates as float32, written to ``out`` when given."""
        c10_vals = np.asarray(C10, dtype=np.float32)
        c12_vals = np.asarray(C12, dtype=np.float32)
        phi_vals = np.asarray(phi12, dtype=np.float32)
        losses = out if out is not None else cp.empty(int(c10_vals.size), dtype=cp.float32)
        for index in range(int(c10_vals.size)):
            losses[index] = np.float32(self.loss(float(c10_vals[index]), float(c12_vals[index]), float(phi_vals[index])))
        return losses
