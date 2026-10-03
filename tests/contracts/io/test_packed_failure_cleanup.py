"""Failed loads and unsupported SSB must not retain owned packed storage."""

from quantem.gpu.io.models import create_dataset

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
        loading._load_packed(source, backend="mps", expected_source_sha256=None, device=None)
    assert caught.value is failure
    assert resident.release_count == 1
    if cleanup_fails:
        assert "injected release failure" in failure.__notes__[0]


@pytest.mark.parametrize("backend", ["cuda", "mps"])
@pytest.mark.parametrize("expected_sha256", [None, "a" * 64])
def test_packed_ssb_rejects_before_any_source_allocation(
    tmp_path, monkeypatch, expected_sha256, backend
):
    source = tmp_path / "packed.h5"
    source.write_bytes(b"QGPUH5\0\1")
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: backend)

    def must_not_allocate(*args, **kwargs):
        pytest.fail("Unsupported packed SSB must not load any source")

    monkeypatch.setattr(io, "load", must_not_allocate)
    monkeypatch.setattr(workflow, "_mps_brightfield_sources", must_not_allocate)
    with pytest.raises(NotImplementedError, match="Reopen the original acquisition"):
        SSB.open(
            str(source),
            backend=backend,
            expected_source_sha256=expected_sha256,
            voltage_kV=300,
            semiangle_mrad=25,
            scan_sampling_A=0.5,
        )


def test_unprepared_ssb_close_uses_the_shared_release_contract(monkeypatch):
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: "mps")
    resident = ReleaseOnlyResident()
    with SSB(
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
        SSB(
            resident,
            backend="mps",
            voltage_kV=300,
            semiangle_mrad=25,
            scan_sampling_A=0.5,
        )


def test_ssb_open_forwards_scan_shape_and_closes_crop_source(tmp_path, monkeypatch):
    """The encoded loader needs the explicit raster before reading the BF crop."""

    source = tmp_path / "rectangular.h5"
    source.touch()
    resident = ReleaseOnlyResident()
    loaded = create_dataset(resident, {})
    calls = []
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: "cuda")

    def load_source(path, **kwargs):
        calls.append(kwargs)
        return loaded

    monkeypatch.setattr(io, "load", load_source)
    counts = np.ones((3, 4, 2, 2), dtype=np.float32)
    monkeypatch.setattr(
        workflow,
        "_bright_field_crop",
        lambda *a, **kw: (counts, (0.5, 0.5), 1.0, None),
    )
    with SSB.open(
        str(source), backend="cuda", scan_shape=(3, 4),
        voltage_kV=300, semiangle_mrad=25, scan_sampling_A=0.5,
    ) as session:
        assert calls[0]["scan_shape"] == (3, 4)
        assert session._scan_shape == (3, 4)
        assert resident.release_count == 1


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_ssb_open_releases_source_when_bf_detection_fails(
    tmp_path, monkeypatch, cleanup_fails
):
    """Failed BF detection must not retain the acquisition or hide its error."""
    from quantem.gpu.detector import workflow as detector_workflow

    source = tmp_path / "missing-disk.h5"
    source.touch()
    resident = ReleaseOnlyResident(cleanup_fails)
    loaded = create_dataset(resident, {})
    monkeypatch.setattr(workflow, "_resolve_backend", lambda _: "cuda")
    monkeypatch.setattr(io, "load", lambda *a, **kw: loaded)
    failure = ValueError("injected mean diffraction failure")

    def fail_mean(*args):
        raise failure

    monkeypatch.setattr(detector_workflow, "mean", fail_mean)
    with pytest.raises(ValueError) as caught:
        SSB.open(
            str(source), backend="cuda", voltage_kV=300,
            semiangle_mrad=25, scan_sampling_A=0.5,
        )
    assert caught.value is failure
    assert resident.release_count == 1
    if cleanup_fails:
        assert "injected release failure" in failure.__notes__[0]
