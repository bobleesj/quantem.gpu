"""Public detector dispatch without backend coercion or implicit source decode."""

import numpy as np
import pytest

from quantem.gpu import detector
from quantem.gpu.io.backends.mps._ans import MPSANSResidentCounts


def test_adapter_owns_only_small_returned_outputs(monkeypatch):
    source = object.__new__(MPSANSResidentCounts)
    source.shape = (2, 3, 4, 5)
    expected = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(source.shape)
    outputs, calls = [], []

    class Output:
        def __init__(self, values):
            self.values, self.released = values, False
            outputs.append(self)

        def to_numpy(self):
            return self.values.copy()

        def release(self):
            self.released = True

    def frame(row, column):
        calls.append((row, column))
        return Output(expected[row, column])

    monkeypatch.setattr(source, "extract_diffraction_device", frame)
    monkeypatch.setattr(
        source,
        "detector_sum_device",
        lambda mask: Output((expected * mask).sum(axis=(2, 3), dtype=np.uint64)),
    )
    session = detector.prepare(source)
    np.testing.assert_array_equal(session.frame(5), expected[1, 2])
    assert calls == [(1, 2)]
    mask = np.ones((4, 5), dtype=bool)
    np.testing.assert_array_equal(
        session.masked_sum_exact(mask), expected.sum(axis=(2, 3), dtype=np.uint64)
    )
    assert all(output.released for output in outputs)
    with pytest.raises(NotImplementedError, match="not qualified"):
        session.mean_dp()
    with pytest.raises(ValueError, match="binary"):
        session.masked_sum_exact(np.full((4, 5), 1.5))
