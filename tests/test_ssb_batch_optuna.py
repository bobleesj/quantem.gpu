from __future__ import annotations

import numpy as np


def test_eval_single_exact_fallback_calls_scalar_loss_once() -> None:
    # C1: exact reconstruction fallback is active, expect no batch padding
    # because padding would run the same full-IFFT objective four times.
    from quantem.gpu.ssb.batch_optuna import _eval_single

    class FakeAccel:
        uses_optimizer_reconstruct_fallback = True

        def __init__(self) -> None:
            self.scalar_calls = 0
            self.batch_calls = 0

        def variance_loss(self, c10: float, c12: float, phi12: float) -> float:
            self.scalar_calls += 1
            assert c10 == 1.0
            assert c12 == 2.0
            assert phi12 == 3.0
            return 4.0

        def variance_loss_batch(self, *args: object, **kwargs: object) -> np.ndarray:
            self.batch_calls += 1
            raise AssertionError("exact fallback should not use padded batch loss")

    accel = FakeAccel()
    loss = _eval_single(accel, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))

    assert loss == 4.0
    assert accel.scalar_calls == 1
    assert accel.batch_calls == 0
