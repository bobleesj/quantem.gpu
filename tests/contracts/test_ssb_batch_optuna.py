import numpy as np
import pytest


def test_eval_single_calls_scalar_loss_once() -> None:
    # C1: one simplex point is one exact reconstruction, never a padded batch.
    pytest.importorskip("cupy")
    from quantem.gpu.ssb.cuda.optimizer import _eval_single

    class FakeObjective:
        def __init__(self) -> None:
            self.scalar_calls = 0
            self.batch_calls = 0

        def loss(self, c10: float, c12: float, phi12: float) -> float:
            self.scalar_calls += 1
            assert c10 == 1.0
            assert c12 == 2.0
            assert phi12 == 3.0
            return 4.0

        def loss_batch(self, *args: object, **kwargs: object) -> np.ndarray:
            self.batch_calls += 1
            raise AssertionError("a single point should not use the batch loss")

    objective = FakeObjective()
    loss = _eval_single(objective, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))

    assert loss == 4.0
    assert objective.scalar_calls == 1
    assert objective.batch_calls == 0


def test_nelder_mead_keeps_first_vertex_for_tied_losses() -> None:
    """Stable tie ordering keeps a flat seeded fit at its input point."""
    cp = pytest.importorskip("cupy")
    from quantem.gpu.ssb.cuda.optimizer import batch_nelder_mead

    class FlatObjective:
        def loss(self, c10: float, c12: float, phi12: float) -> float:
            return 1.0

        def loss_batch(
            self,
            c10: np.ndarray,
            c12: np.ndarray,
            phi12: np.ndarray,
        ):
            assert len(c10) == len(c12) == len(phi12)
            return cp.ones(len(c10), dtype=cp.float32)

    x0 = np.asarray([-20.0, 22.0, 0.46], dtype=np.float64)
    first = batch_nelder_mead(FlatObjective(), x0, max_iter=4)
    second = batch_nelder_mead(FlatObjective(), x0, max_iter=4)

    np.testing.assert_array_equal(first[0], x0)
    np.testing.assert_array_equal(second[0], first[0])
    assert first[1:] == second[1:]
