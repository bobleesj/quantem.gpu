"""Scientist-facing dense and encoded loading workflows."""

from importlib import import_module

import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.io.dataset import Dataset4dstem
from tests.parity import cpu_reference


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_original_acquisition_defaults_to_ans(monkeypatch, tmp_path, backend) -> None:
    """Opening an original without storage flags reaches ANS, never dense IO."""
    import h5py

    backends = import_module("quantem.gpu.device.select")
    streamed = import_module("quantem.gpu.io.encoded")
    source = tmp_path / "acquisition.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data", data=np.zeros((2, 3, 4, 5), np.uint16))
    # what a loader returns: io.load reads its metadata before handing it on
    expected = Dataset4dstem(None, {})
    calls = []

    def encoded(path, **options):
        calls.append((path, options["backend"]))
        return expected

    monkeypatch.setattr(backends, "resolve", lambda requested: backend)
    monkeypatch.setattr(streamed, "load_h5_ans", encoded)
    assert io.load(source) is expected
    assert calls == [(source, backend)]


def test_dense_representation_reports_memory() -> None:
    """A dense reference array reports its representation and exact byte counts."""
    values = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    loaded = cpu_reference.record_dense_representation(
        Dataset4dstem(values, {"backend": "cpu", "source_dtype": "uint16"})
    )

    assert isinstance(loaded, io.Dataset4dstem)
    assert loaded.representation == io.DataRepresentation.DENSE
    assert loaded.shape == (2, 3, 4, 5)
    assert loaded.dtype == np.dtype("uint16")
    assert loaded.logical_bytes == values.nbytes
    assert loaded.resident_bytes == values.nbytes
    np.testing.assert_array_equal(loaded.data, values)


def test_representation_values_do_not_encode_dtype() -> None:
    """Representation names remain independent of detector-count dtype."""
    assert {item.value for item in io.DataRepresentation} == {
        "dense",
        "encoded",
        "paired",
    }


def test_narrowed_dense_result_does_not_claim_unproven_losslessness() -> None:
    loaded = cpu_reference.record_dense_representation(
        Dataset4dstem(
            np.asarray([255], dtype=np.uint8),
            {"source_dtype": "uint16", "backend": "cpu"},
        )
    )
    assert not loaded.lossless
    assert loaded.representation == io.DataRepresentation.DENSE


def test_float64_result_does_not_claim_exact_uint64_counts() -> None:
    loaded = cpu_reference.record_dense_representation(
        Dataset4dstem(
            np.asarray([2**60 + 1], dtype=np.float64),
            {"source_dtype": "uint64", "backend": "cpu"},
        )
    )
    assert not loaded.lossless


@pytest.mark.parametrize("backend", ["cuda", "mps"])
@pytest.mark.parametrize("representation", ["dense"])
@pytest.mark.parametrize("extension", ["h5", "dm4", "npy", "raw", "emd", "qem"])
def test_gpu_acquisitions_reject_expansion_before_opening(
    monkeypatch, backend, representation, extension,
):
    """Selecting GPU residency never expands an acquisition as a loading option."""
    backends = import_module("quantem.gpu.device.select")
    monkeypatch.setattr(backends, "resolve", lambda requested: backend)
    with pytest.raises(NotImplementedError, match="must remain ANS encoded"):
        io.load(f"not-opened.{extension}", device=backend, representation=representation)


def test_inspect_and_reference_decode_preserve_real_hdf5_counts(tmp_path) -> None:
    """A master whose frames sit in an external shard inspects as ready and decodes to the stored counts."""
    import h5py

    source = tmp_path / "native_master.h5"
    counts = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
    counts[0, 0, 0] = 65535
    shard = tmp_path / "native_data_000001.h5"
    with h5py.File(shard, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts)
    with h5py.File(source, "w") as handle:
        handle.require_group("entry/data")["data_000001"] = h5py.ExternalLink(
            shard.name, "/entry/data/data"
        )
    report = io.inspect(source, scan_shape=(2, 3))
    assert report.ready
    assert report.metadata["representation"] == "dense"
    loaded = cpu_reference.load(source, scan_shape=(2, 3))
    np.testing.assert_array_equal(loaded.data, counts.reshape(2, 3, 4, 5))
    assert loaded.dtype == np.dtype("uint16")
    assert loaded.lossless
    assert loaded.logical_bytes == loaded.resident_bytes == counts.nbytes


def test_cpu_reference_refuses_a_pixel_mask_of_another_shape(tmp_path) -> None:
    """A mismatched dead-pixel mask raises; skipping it kept 65535 sentinels as counts without notice."""
    import h5py

    source = tmp_path / "native_master.h5"
    counts = np.arange(6 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
    counts[:, 1, 2] = 65535
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts)
    mask = np.zeros((4, 5), np.uint32)
    mask[1, 2] = 1
    expected = counts.copy()
    expected[:, 1, 2] = 0
    np.testing.assert_array_equal(cpu_reference.load_master(str(source), pixel_mask=mask, method="zero", verbose=False), expected)
    with pytest.raises(ValueError, match="pixel mask is 5 x 4, but the frames are 4 x 5; load with apply_mask=False"):
        cpu_reference.load_master(str(source), pixel_mask=mask.T, verbose=False)
