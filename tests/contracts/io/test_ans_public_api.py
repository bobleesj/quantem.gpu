"""Public exact file round trips and honest unsupported conversion directions."""

import hashlib

import numpy as np
import pytest

from quantem.gpu import io


@pytest.mark.parametrize("backend", ["auto", "cuda", "mps"])
def test_encoded_request_requires_an_encoded_source(tmp_path, monkeypatch, backend):
    from quantem.gpu.io import backends

    def unavailable_backend(_backend):
        raise AssertionError("Source policy must be checked before probing a device")

    monkeypatch.setattr(backends, "resolve_backend", unavailable_backend)
    prepared = tmp_path / "packed.h5"
    prepared.write_bytes(b"QGPUH5\0\x01")
    assert io.DataRepresentation.detect_source(prepared).value == "packed"
    with pytest.raises(NotImplementedError, match="requires an encoded source"):
        io.load(prepared, backend=backend, representation="encoded")


def test_removed_names_are_rejected_before_saving_or_loading(tmp_path):
    counts = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    for name in ("ans", "arina-h5", "arina_h5", "h5"):
        with pytest.raises(ValueError, match="use format='arina' or 'quantem'"):
            io.save(tmp_path / "unused", counts, format=name, backend="cpu")
    with pytest.raises(ValueError, match="representation must be one of"):
        io.load(tmp_path / "unused", representation="lossless_packed")
    assert not (tmp_path / "unused").exists()


def test_failed_conversion_publication_releases_output_not_source():
    class Output:
        released = False

        @property
        def nbytes(self):
            raise RuntimeError("injected publication failure")

        def release(self):
            self.released = True

    output = Output()

    class Source:
        def to_packed(self):
            return output

        def release(self):
            raise AssertionError("caller-owned input must not be released")

    loaded = io.FourDSTEMData(Source(), {"representation": "encoded"})
    with pytest.raises(RuntimeError, match="publication failure"):
        loaded.to_representation("packed")
    assert output.released
