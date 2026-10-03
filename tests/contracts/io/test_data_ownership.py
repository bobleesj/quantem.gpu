"""Context cleanup preserves the error that interrupted scientific work."""

from quantem.gpu.io.models import create_dataset

import pytest

from quantem.gpu.io.models import Dataset4dstem


def test_context_cleanup_preserves_scientific_failure():
    class Owner:
        def release(self):
            raise RuntimeError("cleanup failed")

    error = ValueError("scientific operation failed")
    with pytest.raises(ValueError) as raised:
        with create_dataset(Owner(), {}):
            raise error
    assert raised.value is error
    assert error.__notes__ == ["Resident cleanup also failed: cleanup failed"]


def test_context_cleanup_surfaces_failure_after_success():
    class Owner:
        def release(self):
            raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        with create_dataset(Owner(), {}):
            pass


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
    data = create_dataset(storage, metadata)
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
    data = create_dataset(values, {})

    def read_row(self, row):
        calls.append(row)
        return values[row]

    monkeypatch.setattr(Dataset4dstem, "__getitem__", read_row)
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


def test_old_load_result_names_are_not_exported():
    """There is one public owner type, without compatibility aliases."""
    from quantem.gpu import io
    from quantem.gpu.io import models

    assert io.Dataset4dstem is Dataset4dstem
    for name in ("FourDSTEMData", "LoadResult"):
        assert not hasattr(io, name)
        assert not hasattr(models, name)
