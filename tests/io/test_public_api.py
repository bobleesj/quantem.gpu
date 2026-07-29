"""Public I/O boundary and backend-selection regressions."""
from __future__ import annotations

import inspect as python_inspect
from importlib.util import find_spec
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


def test_io_exports_only_scientist_workflows() -> None:
    import quantem.gpu as gpu

    assert gpu.io.__all__ == ["discover", "inspect", "load", "save"]
    assert all(callable(getattr(gpu.io, name)) for name in gpu.io.__all__)
    assert not hasattr(gpu, "load")
    assert not hasattr(gpu, "save")

    load_parameters = python_inspect.signature(gpu.io.load).parameters
    assert "dtype" in load_parameters
    assert "output_dtype" not in load_parameters
    assert "prep_workers" not in load_parameters
    assert all(
        parameter.kind
        not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        for parameter in load_parameters.values()
    )


def test_removed_io_paths_do_not_exist() -> None:
    assert find_spec("quantem.gpu.io.hdf5") is None
    assert find_spec("quantem.gpu.uint4") is None


def test_auto_backend_never_selects_cpu(monkeypatch) -> None:
    from quantem.gpu.io.backends import protocol

    monkeypatch.setattr(protocol, "_has_cuda", lambda: False)
    monkeypatch.setattr(protocol, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(protocol, "_has_mps", lambda: False)

    with pytest.raises(RuntimeError, match="never selected by backend='auto'"):
        protocol.detect_backend()
    assert protocol.resolve_backend("cpu") == "cpu"


def test_auto_save_rejects_host_array_instead_of_using_cpu(tmp_path) -> None:
    from quantem.gpu.io import save

    data = np.zeros((1, 1, 2, 2), dtype=np.uint16)
    with pytest.raises(RuntimeError, match="could not infer an accelerated writer"):
        save(tmp_path / "scan_master.h5", data, backend="auto")


def test_inspect_and_discover_are_header_only_public_workflows(tmp_path) -> None:
    from quantem.gpu.io import discover, inspect

    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.zeros((1, 2, 3), dtype=np.uint16),
        )
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask",
            data=np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.uint8),
        )

    assert discover(str(tmp_path), verbose=False) == [str(master)]
    report = inspect(master, scan_shape=(1, 1))
    assert report.ready is True
    assert report.actual_frames == 1
    assert report.scan_shape == (1, 1)
    assert report.detector_shape == (2, 3)
    assert report.dtype == "uint16"
    assert report.metadata["detector_shape"] == (2, 3)
    np.testing.assert_array_equal(
        report.pixel_mask,
        np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.uint8),
    )
    assert report.source_kind == "inline"
    assert report.source_signature == inspect(master, scan_shape=(1, 1)).source_signature


def test_inspect_external_master_and_missing_source(tmp_path) -> None:
    from quantem.gpu.io import inspect

    data_path = tmp_path / "scan_data_000001.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.zeros((4, 2, 3), dtype=np.uint16),
        )
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        group = handle.require_group("entry/data")
        group["data_000001"] = h5py.ExternalLink(
            data_path.name,
            "/entry/data/data",
        )

    report = inspect(master, scan_shape=(2, 2))
    assert report.ready is True
    assert report.source_kind == "external"
    assert report.actual_frames == report.expected_frames == 4
    assert report.scan_shape == (2, 2)
    assert report.detector_shape == (2, 3)
    assert report.dtype == "uint16"
    assert report.source_signature == inspect(master, scan_shape=(2, 2)).source_signature

    data_path.unlink()
    missing = inspect(master, scan_shape=(2, 2))
    assert missing.ready is False
    assert "missing" in missing.reason
    assert missing.source_kind == "external"
    assert missing.pixel_mask is None


def test_inspect_reports_incomplete_corrupt_and_scan_shape_mismatch(tmp_path) -> None:
    from quantem.gpu.io import inspect

    incomplete = tmp_path / "incomplete_master.h5"
    with h5py.File(incomplete, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.zeros((3, 2, 2), dtype=np.uint8),
        )

    mismatch = inspect(incomplete, scan_shape=(2, 2))
    assert mismatch.ready is False
    assert mismatch.actual_frames == 3
    assert mismatch.expected_frames == 4
    assert mismatch.scan_shape == (2, 2)
    assert mismatch.detector_shape == (2, 2)
    assert mismatch.dtype == "uint8"
    assert "expected 4" in mismatch.reason
    assert mismatch.action

    corrupt = tmp_path / "corrupt_master.h5"
    corrupt.write_bytes(b"not an HDF5 file")
    broken = inspect(corrupt, scan_shape=(1, 1))
    assert broken.ready is False
    assert broken.source_kind == "unavailable"
    assert broken.actual_frames is None
    assert broken.expected_frames == 1
    assert broken.detector_shape is None
    assert broken.dtype is None
    assert broken.pixel_mask is None
    assert broken.reason
    assert broken.action
    assert broken.source_signature["files"]


def test_mps_series_handle_exposes_data_without_widget_construction() -> None:
    from quantem.gpu.io.backends.mps import series

    multi = SimpleNamespace(
        shape=(2, 8, 9, 4, 5),
        dtype=np.dtype(np.uint16),
        chunks=[np.zeros((72, 4, 5), dtype=np.uint16)],
    )
    handle = series.LazyMPSDatasets(
        ["a_master.h5", "b_master.h5"],
        det_bin=1,
        names=["a", "b"],
        multi=multi,
        decode=lambda _path: None,
        verbose=False,
    )

    assert handle._is_gpu_frames is True
    assert handle.device == "mps"
    assert handle.data is multi
    assert handle.shape == multi.shape
    assert handle.dtype == np.dtype(np.uint16)
    assert handle.scan_shape == (8, 9)
    assert handle.chunks is multi.chunks
    assert not hasattr(handle, "build_viewer")
    assert not hasattr(series, "load_4dstem_macbook")
    assert not hasattr(series, "load_macbook_datasets")


def test_single_mps_load_handle_advertises_accelerated_frame_contract() -> None:
    pytest.importorskip("Metal")
    from quantem.gpu.io.backends.mps.decoder import MPSChunked4DSTEM

    data = MPSChunked4DSTEM(
        chunks=[np.zeros((6, 4, 5), dtype=np.uint16)],
        metadata={},
        master_path="scan_master.h5",
        scan_shape=(2, 3),
    )
    assert data._is_gpu_frames is True
    assert data.device == "mps"
    assert data.shape == (6, 4, 5)
    assert data.scan_shape == (2, 3)
    assert data.dtype == np.dtype(np.uint16)
