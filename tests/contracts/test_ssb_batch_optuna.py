import numpy as np
import pytest


def test_nelder_mead_keeps_first_vertex_for_tied_losses() -> None:
    """Stable tie ordering keeps a flat seeded fit at its input point."""
    pytest.importorskip("cupy")
    from quantem.gpu.ssb.cuda.optimizer import batch_nelder_mead

    class FlatEngine:
        def reconstruct_with_loss(self, c10: float, c12: float, phi12: float):
            return None, 1.0

    x0 = np.asarray([-20.0, 22.0, 0.46], dtype=np.float64)
    first = batch_nelder_mead(FlatEngine(), x0, max_iter=4)
    second = batch_nelder_mead(FlatEngine(), x0, max_iter=4)

    np.testing.assert_array_equal(first[0], x0)
    np.testing.assert_array_equal(second[0], first[0])
    assert first[1:] == second[1:]
