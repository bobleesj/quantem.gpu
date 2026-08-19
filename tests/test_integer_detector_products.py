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


def test_exact_masked_sum_preserves_counts_above_float32_limit() -> None:
    """The exact public path does not round large integer detector sums."""

    from quantem.gpu import detector

    data = np.full((2, 3, 20, 20), 100_000, dtype=np.uint32)
    mask = np.ones((20, 20), dtype=bool)

    result = detector.prepare(data).masked_sum_exact(mask)

    assert result.dtype == np.uint64
    np.testing.assert_array_equal(result, np.full((2, 3), 40_000_000, np.uint64))
