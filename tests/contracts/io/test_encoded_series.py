"""Acquisition series retain independent ANS owners, never a stacked dense cube."""

from importlib import import_module

import pytest

from quantem.gpu import io


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_encoded_series_keeps_order_geometry_and_device(monkeypatch, tmp_path, backend):
    """Open differently shaped acquisitions in source order and close each owner."""
    streamed = import_module("quantem.gpu.io._streamed")
    backends = import_module("quantem.gpu.io.backends")
    monkeypatch.setattr(backends, "resolve_backend", lambda requested: backend)
    paths = [tmp_path / f"scan_{index}.h5" for index in range(3)]
    calls = []
    owners = []

    class Resident:
        def __init__(self, index):
            self.shape = (index + 2, 3, 5, 7)
            self.closed = False

        def close(self):
            self.closed = True

    def load(path, **options):
        calls.append((path, options["device"], options["backend"]))
        owner = Resident(paths.index(path))
        owners.append(owner)
        return owner

    monkeypatch.setattr(streamed, "load_h5_ans", load)
    device = 0 if backend == "cuda" else None
    loaded = io.load(paths, backend=backend, device=device, stack=False)
    assert loaded == owners
    assert calls == [(path, device, backend) for path in paths]
    assert [owner.shape[0] for owner in loaded] == [2, 3, 4]
    for owner in loaded:
        owner.close()
    assert all(owner.closed for owner in owners)


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_incomplete_encoded_series_closes_prior_acquisitions(monkeypatch, tmp_path, backend):
    """A missing required acquisition cannot return a misleading partial series."""
    streamed = import_module("quantem.gpu.io._streamed")
    backends = import_module("quantem.gpu.io.backends")
    monkeypatch.setattr(backends, "resolve_backend", lambda requested: backend)
    paths = [tmp_path / f"scan_{index}.h5" for index in range(3)]
    released = []

    class Resident:
        def close(self):
            released.append(paths[0])

    def load(path, **options):
        if path == paths[1]:
            raise OSError(f"Missing required acquisition: {path.name}")
        return Resident()

    monkeypatch.setattr(streamed, "load_h5_ans", load)
    with pytest.raises(OSError, match="scan_1.h5"):
        io.load(paths, backend=backend, stack=False)
    assert released == paths[:1]
