"""Open complete native geometry without guessing rectangular scan dimensions."""

import hashlib
import struct

import h5py
import json
import numpy as np
import pytest
from quantem.gpu import io


def _write_synthetic_qem(path, *, shape, excluded):
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
            axes=[dict(name=name, size=size)
                  for name, size in zip(_qem_metadata.AXIS_NAMES, shape)],
        ),
    )
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


def test_retired_ans_snapshot_is_rejected_with_guidance(tmp_path):
    path = tmp_path / "legacy.compressed.ans"
    path.write_bytes(b"QGPUSTRM" + b"\0" * 64)
    assert not io._streamed_file.is_streamed_file(path)
    with pytest.raises(ValueError, match="no longer supported"):
        io.load(path, backend="mps")


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
