"""Exact-integer accumulation checks for detector products."""

from __future__ import annotations

import numpy as np
import pytest


def test_torch_masked_sum_preserves_integer_accumulation() -> None:
    """Large uint16 detector sums match an integer reference exactly."""

    torch = pytest.importorskip("torch")
    from quantem.gpu import detector

    rng = np.random.default_rng(7)
    data = rng.integers(0, 65536, size=(3, 4, 192, 192), dtype=np.uint16)
    mask = rng.random((192, 192)) > 0.3
    expected = data[..., mask].sum(axis=-1, dtype=np.uint64).astype(np.float32)

    result = detector.masked_sum(torch.from_numpy(data), mask)

    np.testing.assert_array_equal(result, expected)
