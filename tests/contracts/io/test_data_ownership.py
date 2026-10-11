"""A loaded acquisition summarizes, iterates and exports without materializing its storage."""

from quantem.gpu.io.dataset import Dataset4dstemGPU

import pytest



def test_acquisition_summary_does_not_materialize_storage():
    """Notebook inspection is cheap even for a huge logical acquisition."""
    import numpy as np

    class EncodedOwner:
        shape = (512, 256, 192, 192)
        dtype = np.dtype("uint16")

        def __array__(self, *args, **kwargs):
            raise AssertionError("Inspection must not decode detector values")

        def __repr__(self):
            raise AssertionError("Inspection must not format resident detector values")

    storage = EncodedOwner()
    metadata = {"representation": "encoded"}
    data = Dataset4dstemGPU(storage, metadata)
    assert data.data is storage
    assert data.metadata is metadata
    assert not isinstance(data, tuple)
    assert data.shape == storage.shape
    assert len(data) == 512
    assert data.ndim == 4
    assert data.size == 512 * 256 * 192 * 192
    assert data.logical_bytes == data.size * 2
    summary = repr(data)
    assert "shape=(512, 256, 192, 192)" in summary
    assert "dtype=uint16" in summary
    with pytest.raises(TypeError, match="Select a bounded region first"):
        np.asarray(data)


def test_iteration_selects_rows_lazily(monkeypatch):
    """Iteration follows the scan axis without returning payload or metadata."""
    import numpy as np

    calls = []
    values = np.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2)
    data = Dataset4dstemGPU(values, {})

    def read_row(self, row):
        calls.append(row)
        return values[row]

    monkeypatch.setattr(Dataset4dstemGPU, "__getitem__", read_row)
    iterator = iter(data)
    assert calls == []
    np.testing.assert_array_equal(next(iterator), values[0])
    assert calls == [0]
    np.testing.assert_array_equal(next(iterator), values[1])
    assert calls == [0, 1]
    np.testing.assert_array_equal(next(iterator), values[2])
    with pytest.raises(StopIteration):
        next(iterator)
    assert calls == [0, 1, 2]
