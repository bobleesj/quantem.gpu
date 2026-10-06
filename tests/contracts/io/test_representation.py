"""Scientist-facing dense and encoded loading workflows."""

from quantem.gpu.io.dataset import Dataset4dstemGPU

from importlib import import_module

import numpy as np
import pytest

from quantem.gpu import io


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_original_acquisition_defaults_to_ans(monkeypatch, tmp_path, backend) -> None:
    """Opening an original without storage flags reaches ANS, never dense IO."""
    import h5py

    cpu = import_module("quantem.gpu.io.hdf5.cpu")
    backends = import_module("quantem.gpu.device.select")
    streamed = import_module("quantem.gpu.io.encoded")
    source = tmp_path / "acquisition.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data", data=np.zeros((2, 3, 4, 5), np.uint16))
    expected = object()
    calls = []

    def encoded(path, **options):
        calls.append((path, options["backend"]))
        return expected

    def dense(*args, **kwargs):
        pytest.fail("Default acquisition loading reached dense materialization")

    monkeypatch.setattr(backends, "resolve_backend", lambda requested: backend)
    monkeypatch.setattr(streamed, "load_h5_ans", encoded)
    monkeypatch.setattr(cpu, "load_reference", dense)
    assert io.load(source) is expected
    assert calls == [(source, backend)]


def test_dense_representation_is_explicit_and_reports_memory(monkeypatch) -> None:
    """An explicit dense load reports its representation and exact byte counts."""
    cpu = import_module("quantem.gpu.io.hdf5.cpu")
    values = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    monkeypatch.setattr(
        cpu,
        "load_reference",
        lambda *args, **kwargs: cpu.record_dense_representation(
            Dataset4dstemGPU(values, {"backend": "cpu", "source_dtype": "uint16"})
        ),
    )

    loaded = io.load(
        "ordinary-master.h5",
        backend="cpu",
        representation="dense",
        verbose=False,
    )

    assert isinstance(loaded, io.Dataset4dstemGPU)
    assert loaded.representation == io.DataRepresentation.DENSE
    assert loaded.residency == "host"
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


def test_removed_representation_names_have_no_aliases() -> None:
    """Removed names fail explicitly instead of choosing another decoder."""
    assert not hasattr(io.DataRepresentation, "LOSSLESS_PACKED")
    assert not hasattr(io.DataRepresentation, "PACKED")
    with pytest.raises(ValueError, match="representation must be"):
        io.DataRepresentation.parse("lossless_packed")
    with pytest.raises(ValueError, match="representation must be"):
        io.DataRepresentation.parse("packed")
    with pytest.raises(ValueError, match="representation must be"):
        io.DataRepresentation.parse("ans")


def test_retired_detector_bin_spelling_is_not_supported() -> None:
    """The public loader has no detector-binning keyword; bin encoded reads instead."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'det_bin'"):
        io.load("ordinary-master.h5", det_bin=4)


def test_narrowed_dense_result_does_not_claim_unproven_losslessness() -> None:
    cpu = import_module("quantem.gpu.io.hdf5.cpu")
    loaded = cpu.record_dense_representation(
        Dataset4dstemGPU(
            np.asarray([255], dtype=np.uint8),
            {"source_dtype": "uint16", "backend": "cpu"},
        )
    )
    assert not loaded.lossless
    assert loaded.representation == io.DataRepresentation.DENSE


def test_float64_result_does_not_claim_exact_uint64_counts() -> None:
    cpu = import_module("quantem.gpu.io.hdf5.cpu")
    loaded = cpu.record_dense_representation(
        Dataset4dstemGPU(
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
    monkeypatch.setattr(backends, "resolve_backend", lambda requested: backend)
    with pytest.raises(NotImplementedError, match="must remain ANS encoded"):
        io.load(f"not-opened.{extension}", backend=backend, representation=representation)


def test_dense_cpu_load_and_inspect_preserve_real_hdf5_counts(tmp_path) -> None:
    """Exercise the retained dense path without replacing the actual loader."""
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
    loaded = io.load(source, backend="cpu", representation="dense",
                     scan_shape=(2, 3), dtype="native", verbose=False)
    np.testing.assert_array_equal(loaded.data, counts.reshape(2, 3, 4, 5))
    assert loaded.dtype == np.dtype("uint16")
    assert loaded.lossless
    assert loaded.logical_bytes == loaded.resident_bytes == counts.nbytes


def test_io_exposes_only_the_current_data_type() -> None:
    from quantem.gpu.io import dataset

    assert io.Dataset4dstemGPU is dataset.Dataset4dstemGPU
    assert not hasattr(io, "LoadResult")
    assert not hasattr(dataset, "LoadResult")
    assert not hasattr(import_module("quantem.gpu.io.load"), "LoadResult")
