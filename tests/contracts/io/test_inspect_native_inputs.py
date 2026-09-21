"""Open complete native geometry without guessing rectangular scan dimensions."""

import hashlib
import struct

import h5py
import json
import numpy as np
import pytest
from quantem.gpu import io


def _write_synthetic_qem(path, *, shape, excluded, processing=True):
    """Write a minimal valid saved copy so header handling is tested without a GPU."""
    from quantem.gpu._compact.streamed import field_count
    from quantem.gpu.io import _qem_metadata

    scans, pixels = shape[0] * shape[1], shape[2] * shape[3]
    fields = field_count(shape[2:])
    counts = {1: pixels + 1, 2: pixels, 3: 2, 4: fields + 1, 5: fields}
    sizes = ("u1", "<u4", "u1", "<u4", "<u8", "u1")
    cursor, arrays = 0, []
    for index, dtype in enumerate(sizes):
        cursor = (cursor + 7) & ~7
        count = counts.get(index, 4)
        arrays.append({"offset": cursor, "count": count})
        cursor += count * np.dtype(dtype).itemsize
    valid = np.ones(pixels, np.uint8)
    valid[list(excluded)] = 0
    header = dict(
        version=1, profile="runtime-column-rans-spatial-v2", interval=512,
        shape=list(shape), dtype="uint16",
        valid=np.packbits(valid).tobytes().hex(),
        chunks=[{"first": 0, "scans": scans, "arrays": arrays}],
        bytes=cursor, sha256=[hashlib.sha256(bytes(cursor)).hexdigest()],
        metadata={}, container=_qem_metadata.CONTAINER,
        container_version=_qem_metadata.CONTAINER_VERSION,
        codec="runtime-column-rans-spatial-v2",
        scientific_metadata=dict(
            schema=_qem_metadata.SCHEMA,
            source_metadata={},
            source_metadata_coverage="unknown",
            processing=[dict(operation="lossless_storage", changes_measurements=False)],
            axes=[dict(name=name, size=size)
                  for name, size in zip(_qem_metadata.AXIS_NAMES, shape)],
        ),
    )
    if not processing:
        header["scientific_metadata"].pop("processing")
    blob = json.dumps(header).encode()
    path.write_bytes(
        _qem_metadata.MAGIC + struct.pack("<QQ", len(blob), 56 + len(blob))
        + hashlib.sha256(blob).digest() + blob + bytes(cursor)
    )


@pytest.mark.parametrize("shape", [(4, 9, 7, 11), (3, 5, 8, 10)])
def test_explicit_four_dimensions_override_square_scan_inference(tmp_path, shape):
    path = tmp_path / "native.h5"
    with h5py.File(path, "w") as source:
        source["entry/data/data"] = np.zeros(shape, np.uint16)
    info = io.inspect(path)
    assert info.ready
    assert info.scan_shape == shape[:2]
    assert info.detector_shape == shape[2:]
    assert info.metadata["dataset_path"] == "entry/data/data"
    mismatched = io.inspect(path, scan_shape=(1, 1))
    assert not mismatched.ready and mismatched.expected_frames == 1


@pytest.mark.parametrize("geometry", ["positions", "crop"])
def test_prepared_stack_companions_define_rectangular_scan(tmp_path, geometry):
    """Prepared diffraction stacks retain calibration without square inference."""
    from scipy.io import savemat

    path = tmp_path / "data_roi0_Ndp8_dp.hdf5"
    with h5py.File(path, "w") as handle:
        handle["dp"] = np.zeros((15, 8, 12), np.float32)
    parameters = dict(Np_p=[12, 8], voltage=300, alpha=25, dk=0.04, dx=[0.2, 0.3])
    if geometry == "positions":
        with h5py.File(tmp_path / "data_position.hdf5", "w") as handle:
            handle["probe_positions_0"] = np.stack((np.repeat(np.arange(3), 5), np.tile(np.arange(5), 3))) * 0.5
    else:
        parameters["crop_idx0"] = [2, 6, 10, 12]
    savemat(tmp_path / "params_backup.mat", parameters)
    info = io.inspect(path)
    assert info.ready and info.scan_shape == (3, 5)
    assert info.detector_shape == (8, 12)
    assert info.metadata["voltage_kV"] == 300
    assert info.metadata["detector_sampling"] == [0.04, 0.04]
    assert info.metadata.get("scan_sampling_A") is None
    assert info.metadata["source_metadata"]["prepared_stack/params_backup"]["dx"] == [0.2, 0.3]


def test_saved_qem_inspection_uses_header_and_preserves_validity(tmp_path):
    """A saved copy reports its declared geometry and excluded-pixel mask."""
    path = tmp_path / "saved-copy.qem"
    _write_synthetic_qem(path, shape=(3, 5, 7, 9), excluded=[3, 20])
    info = io.inspect(path)
    assert info.ready and info.source_kind == "resident"
    assert info.scan_shape == (3, 5) and info.detector_shape == (7, 9)
    assert info.metadata["resident_bytes"] > 0
    np.testing.assert_array_equal(np.flatnonzero(info.pixel_mask), [3, 20])
    assert "unverified" in info.reason
    mismatched = io.inspect(path, scan_shape=(1, 1))
    assert not mismatched.ready and mismatched.reason == "scan_shape_mismatch"
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError, match="Incomplete QEM"):
        io.inspect(path)


def test_retired_ans_snapshot_is_rejected_with_guidance(tmp_path):
    path = tmp_path / "legacy.compressed.ans"
    path.write_bytes(b"QGPUSTRM" + b"\0" * 64)
    assert not io._streamed_file.is_streamed_file(path)
    with pytest.raises(ValueError, match="no longer supported"):
        io.load(path, backend="mps")


def test_saved_qem_inspection_rejects_missing_processing_provenance(tmp_path):
    path = tmp_path / "incomplete-metadata.qem"
    _write_synthetic_qem(path, shape=(3, 5, 7, 9), excluded=[], processing=False)
    with pytest.raises(ValueError, match="processing must list"):
        io.inspect(path)


def test_prepared_header_checks_expected_scan_dimensions(tmp_path):
    (tmp_path / "checkpoint.json").write_text(json.dumps(dict(
        complete=True,
        layout=dict(complete=True, shape=[2, 4, 9, 7, 11], source_dtype="uint16"),
    )))
    info = io.inspect(tmp_path, scan_shape=(4, 9))
    assert info.ready and info.scan_shape == (4, 9)
    mismatched = io.inspect(tmp_path, scan_shape=(1, 1))
    assert not mismatched.ready and mismatched.reason == "scan_shape_mismatch"
    assert mismatched.expected_frames == 1 and mismatched.actual_frames == 36


def test_maped_json_provenance_has_public_metadata_keys(tmp_path):
    path = tmp_path / "maped.h5"
    merge = {"version": 1, "source_count": 7}
    summary = {
        "version": 1,
        "mean_bright_field": {"operation": "arithmetic_mean"},
    }
    with h5py.File(path, "w") as source:
        source["entry/data/data"] = np.zeros((2, 3, 4, 5), np.uint16)
        source.attrs["quantem_maped_merge_v1"] = json.dumps(merge)
        source.attrs["quantem_maped_summary_v1"] = json.dumps(summary)

    metadata = io.inspect(path).metadata
    assert metadata["maped_merge"] == merge
    assert metadata["maped_summary"] == summary


def test_incomplete_transfer_reports_every_declared_chunk(tmp_path):
    """Track all pending files while an acquisition arrives in stages."""
    master = tmp_path / "arrival_master.h5"
    chunks = [tmp_path / f"arrival_data_{i:06d}.h5" for i in range(1, 4)]
    with h5py.File(master, "w") as source:
        group = source.create_group("entry/data")
        for i, chunk in enumerate(chunks, start=1):
            group[f"data_{i:06d}"] = h5py.ExternalLink(
                chunk.name, "/entry/data/data"
            )
    for completed in range(4):
        info = io.inspect(master, scan_shape=(3, 4))
        assert info.ready is (completed == 3)
        assert {record["path"] for record in info.source_signature["files"]} == {
            str(path) for path in [master, *chunks]
        }
        if completed < 3:
            with h5py.File(chunks[completed], "w") as source:
                source["entry/data/data"] = np.zeros((4, 8, 8), np.uint16)
