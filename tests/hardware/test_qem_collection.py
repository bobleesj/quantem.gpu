"""Convert an ARINA acquisition without changing counts or microscope metadata."""

import os

import h5py
import numpy as np
import pytest

from quantem.gpu.io import qem_conversion
from quantem.gpu.io._streamed_file import read_header


def _master(tmp_path, dtype):
    import hdf5plugin

    values = np.random.default_rng(71).integers(0, 20, (1024, 16, 16), dtype=dtype)
    values[:, 2, 3] = np.arange(1024)  # flagged measurements must not be replaced
    mask = np.zeros((16, 16), np.uint8)
    mask[2, 3] = 1
    chunk = tmp_path / "detector.h5"
    with h5py.File(chunk, "w") as handle:
        handle.create_dataset("entry/data/data", data=values, chunks=(1, 16, 16),
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        # Simulate source-file overhead, ensuring conversion exercises publication.
        handle["unused_storage"] = np.zeros(1 << 20, np.uint8)
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(chunk.name, "/entry/data/data")
        root = "entry/instrument/detector/"
        handle[root + "description"] = b"Dectris ARINA Si"
        handle[root + "detectorSpecific/ntrigger"] = np.uint32(1024)
        handle[root + "detectorSpecific/pixel_mask"] = mask
        handle[root + "detectorSpecific/photon_energy"] = 200000.0
        handle[root + "count_time"] = 49.5e-6
        handle[root + "frame_time"] = 49.6e-6
    return master


@pytest.fixture
def backend():
    selected = os.environ.get("QEM_TEST_BACKEND", "mps")
    if selected == "cuda":
        cp = pytest.importorskip("cupy")
        assert cp.cuda.runtime.getDeviceCount() > 0
    else:
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("Requires Metal or QEM_TEST_BACKEND=cuda")
    return selected


def test_collection_preserves_flagged_counts_and_master(tmp_path, backend):
    master = _master(tmp_path, np.uint16)
    original_master = master.read_bytes()
    destination = tmp_path / "scan.qem"
    result = qem_conversion.convert(master, destination, backend=backend)
    assert not result.skipped and not result.larger and result.verified is True
    assert result.verification["compared_values"] == 1024 * 16 * 16
    assert result.verification["differing_values"] == 0
    assert result.verification["flagged_pixels"] == 1
    header, _ = read_header(destination)
    quantities = header["scientific_metadata"]["electron_microscope"]
    assert quantities["electron_source/accelerating_voltage"]["value"] == 200
    assert quantities["scan_controller/regular_scan/dwell_time"]["value"] == pytest.approx(49.5)
    restored = qem_conversion.restore_master(destination, tmp_path / "restored.h5")
    assert restored.read_bytes() == original_master == master.read_bytes()
    assert qem_conversion.convert(master, destination, backend=backend).skipped
    assert not list(tmp_path.glob(".qem-convert-*"))


def test_cuda_uint32_conversion_never_wraps_flagged_values(tmp_path, backend):
    if backend != "cuda":
        pytest.skip("This change adds uint32 validation to CUDA only")
    master = _master(tmp_path, np.uint32)
    result = qem_conversion.convert(master, tmp_path / "fits.qem", backend=backend)
    assert result.verified is True
    header, _ = read_header(tmp_path / "fits.qem")
    assert header["metadata"]["source_dtype"] == "uint32"
    with h5py.File(tmp_path / "detector.h5", "r+") as handle:
        handle["entry/data/data"][0, 2, 3] = 0xFFFFFFFF
    rejected = qem_conversion.convert(master, tmp_path / "wide.qem", backend=backend)
    assert rejected.failed and "above 65535" in rejected.skipped
    assert not (tmp_path / "wide.qem").exists()


def test_failed_verification_does_not_publish_a_copy(tmp_path, backend, monkeypatch):
    master = _master(tmp_path, np.uint16)
    monkeypatch.setattr(qem_conversion, "verify_against_source",
                        lambda *args, **kwargs: {"identical": False})
    result = qem_conversion.convert(master, tmp_path / "rejected.qem", backend=backend)
    assert result.failed and result.verified is False
    assert not (tmp_path / "rejected.qem").exists()
    assert not list(tmp_path.glob(".qem-convert-*"))
