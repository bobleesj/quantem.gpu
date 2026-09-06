"""Failed loads and unsupported SSB must not retain owned packed storage."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu import SSB, io
from quantem.gpu.io._compact_h5 import CompactH5Index
from quantem.gpu.ssb import workflow


class ReleaseOnlyResident:
    """Exercise the lifetime method owned by MPS packed residents."""

    dtype = np.dtype("uint16")
    shape = (128, 128, 2, 2)
    nbytes = 128 * 128 * 2 * 2 * 2

    def __init__(self, cleanup_fails=False):
        self.release_count = 0
        self.cleanup_fails = cleanup_fails

    def release(self):
        self.release_count += 1
        if self.cleanup_fails:
            raise RuntimeError("injected release failure")


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_mps_metadata_releases_storage_and_preserves_error(
    tmp_path, monkeypatch, cleanup_fails
):
    loading = import_module("quantem.gpu.io._packed")
    mps = import_module("quantem.gpu.io.backends.mps.packed")
    source = tmp_path / "packed.h5"
    source.write_bytes(b"QGPUH5\0\1")
    resident = ReleaseOnlyResident(cleanup_fails)
    index = SimpleNamespace(schema_version=3, require_raw_reconstruction=lambda: None)
    monkeypatch.setattr(CompactH5Index, "from_file", lambda _: index)
    monkeypatch.setattr(
        import_module("quantem.gpu.io.backends"), "resolve_backend", lambda _: "mps"
    )
    monkeypatch.setattr(mps, "load_compact_v3_mps", lambda *a, **kw: resident)
    failure = ValueError("injected scientific metadata failure")

    def unavailable_metadata(*args):
        raise failure

    monkeypatch.setattr(loading, "_packed_metadata", unavailable_metadata)
    with pytest.raises(ValueError) as caught:
        io.load(source, backend="mps")
    assert caught.value is failure
    assert resident.release_count == 1
    if cleanup_fails:
        assert "injected release failure" in failure.__notes__[0]


@pytest.mark.parametrize("expected_sha256", [None, "a" * 64])
def test_packed_mps_ssb_rejects_before_any_source_allocation(
    tmp_path, monkeypatch, expected_sha256
):
    source = tmp_path / "packed.h5"
    source.write_bytes(b"QGPUH5\0\1")
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: "mps")

    def must_not_allocate(*args, **kwargs):
        pytest.fail("Unsupported packed MPS SSB must not load any source")

    monkeypatch.setattr(io, "load", must_not_allocate)
    monkeypatch.setattr(workflow, "_mps_brightfield_sources", must_not_allocate)
    with pytest.raises(NotImplementedError, match="Packed MPS detector"):
        SSB.open(
            str(source),
            backend="mps",
            expected_source_sha256=expected_sha256,
            voltage_kV=300,
            semiangle_mrad=25,
            scan_sampling_A=0.5,
        )


def test_failed_ssb_setup_releases_loaded_source_and_keeps_original_error(monkeypatch):
    resident = ReleaseOnlyResident(cleanup_fails=True)
    loaded = io.FourDSTEMData(
        resident,
        {"representation": "packed", "working_dtype": "uint16"},
    )
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: "cuda")
    monkeypatch.setattr(io, "load", lambda *args, **kwargs: loaded)
    failure = ValueError("injected SSB calibration failure")

    def unavailable_calibration(*args, **kwargs):
        raise failure

    monkeypatch.setattr(SSB, "__init__", unavailable_calibration)
    with pytest.raises(ValueError) as caught:
        SSB.open(
            "qualified.h5",
            backend="cuda",
            voltage_kV=300,
            semiangle_mrad=25,
            scan_sampling_A=0.5,
        )
    assert caught.value is failure
    assert resident.release_count == 1
    assert "injected release failure" in failure.__notes__[0]


def test_unprepared_ssb_close_uses_the_shared_release_contract(monkeypatch):
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: "mps")
    resident = ReleaseOnlyResident()
    with SSB.from_array(
        resident,
        backend="mps",
        voltage_kV=300,
        semiangle_mrad=25,
        scan_sampling_A=0.5,
    ) as session:
        assert resident.release_count == 0
    session.close()
    assert resident.release_count == 1


def test_borrowed_packed_mps_array_is_rejected_without_taking_ownership(monkeypatch):
    from quantem.gpu.io.backends.mps.packed import MPSCompactV3Resident

    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: "mps")
    resident = MPSCompactV3Resident.__new__(MPSCompactV3Resident)
    with pytest.raises(NotImplementedError, match="Packed MPS detector"):
        SSB.from_array(
            resident,
            backend="mps",
            voltage_kV=300,
            semiangle_mrad=25,
            scan_sampling_A=0.5,
        )
