"""Scientist-facing dense and lossless-packed loading workflows."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import io


def test_dense_representation_is_explicit_and_reports_memory(monkeypatch) -> None:
    """An explicit dense load reports its representation and exact byte counts."""
    load_module = import_module("quantem.gpu.io.load")
    values = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    monkeypatch.setattr(
        load_module,
        "_load",
        lambda *args, **kwargs: load_module.LoadResult(
            values,
            {"backend": "cpu", "source_dtype": "uint16"},
        ),
    )

    loaded = io.load(
        "ordinary-master.h5",
        backend="cpu",
        representation="dense",
        verbose=False,
    )

    assert isinstance(loaded, io.FourDSTEMData)
    assert loaded.representation is io.DataRepresentation.DENSE
    assert loaded.residency == "host"
    assert loaded.shape == (2, 3, 4, 5)
    assert loaded.dtype == np.dtype("uint16")
    assert loaded.logical_bytes == values.nbytes
    assert loaded.resident_bytes == values.nbytes
    np.testing.assert_array_equal(loaded.data, values)


def test_existing_lossless_pack_source_selects_packed_loader(
    monkeypatch, tmp_path
) -> None:
    """A prepared lossless source reaches the packed backend without API flags."""
    load_module = import_module("quantem.gpu.io.load")
    source = tmp_path / "prepared.h5"
    source.write_bytes(b"QGPUH5\0\x01" + b"prepared source")
    expected = load_module.FourDSTEMData(
        object(),
        {
            "representation": "lossless_packed",
            "residency": "device",
            "working_shape": (2, 3, 4, 5),
            "working_dtype": "uint16",
            "working_logical_tensor_bytes": 240,
            "physical_resident_bytes": 80,
            "lossless_exact": True,
        },
    )
    calls = {}

    def fake_load(source, **kwargs):
        calls["source"] = source
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(load_module, "_load_lossless_packed", fake_load)

    loaded = io.load(source, backend="mps", verbose=False)

    assert loaded is expected
    assert loaded.representation is io.DataRepresentation.LOSSLESS_PACKED
    assert loaded.logical_bytes == 240
    assert loaded.resident_bytes == 80
    assert calls == {
        "source": source,
        "backend": "mps",
        "expected_source_sha256": None,
        "device": None,
    }


def test_lossless_packed_request_rejects_ordinary_hdf5(tmp_path) -> None:
    """Packing is never claimed when a source has not been prepared."""
    source = tmp_path / "ordinary.h5"
    source.write_bytes(b"\x89HDF\r\n\x1a\n")

    with pytest.raises(ValueError, match="Prepare an immutable lossless-packed"):
        io.load(source, representation="lossless_packed", verbose=False)


def test_dense_request_does_not_silently_expand_lossless_pack(tmp_path) -> None:
    """Dense materialization is explicit work, not a hidden load side effect."""
    source = tmp_path / "prepared.h5"
    source.write_bytes(b"QGPUH5\0\x01" + b"prepared source")

    with pytest.raises(ValueError, match="Dense expansion"):
        io.load(source, representation="dense", verbose=False)


def test_representation_values_do_not_encode_dtype() -> None:
    """Representation names remain independent of detector-count dtype."""
    assert {item.value for item in io.DataRepresentation} == {
        "dense",
        "lossless_packed",
    }


def test_representation_wire_values_match_swift_and_receipt_schema() -> None:
    """Python, Swift, and JSON receipts use one backend-neutral vocabulary."""
    expected = {"dense", "lossless_packed"}
    schema = json.loads(
        Path("src/quantem/gpu/io/resident_contract.schema.json").read_text()
    )
    swift = Path(
        "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/"
        "Metal4DSTEMConsumerContract.swift"
    ).read_text()

    assert set(schema["properties"]["representation"]["enum"]) == expected
    assert 'case dense' in swift
    assert 'case losslessPacked = "lossless_packed"' in swift


def test_detector_bin_is_the_canonical_public_spelling(monkeypatch) -> None:
    """The descriptive detector-bin name reaches the existing implementation."""
    load_module = import_module("quantem.gpu.io.load")
    calls = {}

    def fake_load(*args, **kwargs):
        calls.update(kwargs)
        return load_module.LoadResult(
            np.zeros((1, 1, 2, 2), dtype=np.uint16),
            {"backend": "cpu"},
        )

    monkeypatch.setattr(load_module, "_load", fake_load)

    io.load(
        "ordinary-master.h5",
        backend="cpu",
        representation="dense",
        detector_bin=4,
        verbose=False,
    )

    assert calls["det_bin"] == 4


def test_conflicting_detector_bin_spellings_fail_closed(monkeypatch) -> None:
    """A deprecated alias cannot silently override the canonical parameter."""
    load_module = import_module("quantem.gpu.io.load")
    monkeypatch.setattr(
        load_module,
        "_load",
        lambda *args, **kwargs: load_module.LoadResult(np.zeros((1, 1)), {}),
    )

    with pytest.raises(ValueError, match="cannot request different bin factors"):
        io.load(
            "ordinary-master.h5",
            backend="cpu",
            detector_bin=2,
            det_bin=4,
            verbose=False,
        )


def test_narrowed_dense_result_does_not_claim_unproven_losslessness() -> None:
    load_module = import_module("quantem.gpu.io.load")
    loaded = load_module._record_dense_representation(
        io.FourDSTEMData(
            np.asarray([255], dtype=np.uint8),
            {"source_dtype": "uint16", "backend": "cpu"},
        )
    )
    assert not loaded.lossless
    assert loaded.representation is io.DataRepresentation.DENSE


def test_float64_result_does_not_claim_exact_uint64_counts() -> None:
    load_module = import_module("quantem.gpu.io.load")
    loaded = load_module._record_dense_representation(
        io.FourDSTEMData(
            np.asarray([2**60 + 1], dtype=np.float64),
            {"source_dtype": "uint64", "backend": "cpu"},
        )
    )
    assert not loaded.lossless


def test_packed_scan_order_is_not_silently_ignored(tmp_path) -> None:
    source = tmp_path / "prepared.h5"
    source.write_bytes(b"QGPUH5\0\x01")
    with pytest.raises(ValueError, match="already declare row-major"):
        io.load(source, scan_order="serpentine")


def test_dense_hash_request_is_not_silently_ignored(tmp_path) -> None:
    source = tmp_path / "ordinary.h5"
    source.write_bytes(b"\x89HDF\r\n\x1a\n")
    with pytest.raises(ValueError, match="external shards"):
        io.load(source, representation="dense", expected_source_sha256="0" * 64)


def test_inspection_rejects_truncated_packed_index(tmp_path) -> None:
    source = tmp_path / "interrupted.h5"
    source.write_bytes(b"QGPUH5\0\x01")
    report = io.inspect(source)
    assert not report.ready
    assert report.reason.startswith("invalid_lossless_pack_index:")
    assert report.metadata["representation"] == "lossless_packed"
    assert report.pixel_mask is None


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
