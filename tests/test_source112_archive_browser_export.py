"""Metadata-only preparation of an exact, complete browser source archive."""

import errno
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.io._source112_archive_browser import _export_source112_browser


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _archive(root: Path) -> dict:
    root.mkdir()
    mapping = np.full(36864, -1, dtype="<i4")
    mapping[17466:] = np.arange(19398, dtype="<i4")[::-1]
    hardware = np.zeros(36864, dtype=bool)
    hardware[0] = True
    valid = np.ones(36864, dtype="i1")
    valid[[0, 7, 19]] = 0
    models = np.zeros((264, 36864), dtype="u1")
    models[:, 0] = 255
    arrays = {
        "planner__cache_map": mapping,
        "planner__hardware": hardware,
        "planner__valid": valid,
        "model_ids": models,
        "codec__decoding": np.zeros((81, 1024), dtype="<u4"),
    }
    np.savez_compressed(root / "global-state.npz", **arrays)
    components = []
    end = 0
    for name, words in [("dense", 1), ("dense_offsets", 17466 * 9),
                        ("sparse", 1), ("sparse_offsets", 19398 * 9 + 1)]:
        offset = (end + 63) // 64 * 64
        components.append({"name": name, "shape": [words], "dtype": "<u4",
                           "offset": offset, "nbytes": words * 4})
        end = offset + words * 4
    record_bytes = (end + 4095) // 4096 * 4096
    chunks = [{
        "chunk": index, "acquisition": index // 16,
        "first_scan": index % 16 * 16384, "scan_count": 16384,
        "shard": index % 2, "file_offset": index // 2 * record_bytes,
        "record_bytes": record_bytes, "sha256": _sha(f"record-{index}".encode()),
        "components": components,
    } for index in range(1056)]
    files = [{"name": f"data-{index}.bin", "nbytes": 528 * record_bytes}
             for index in range(2)]
    for item in files:
        # Sparse files provide realistic record extents without any payload IO.
        with (root / item["name"]).open("wb") as stream:
            stream.truncate(item["nbytes"])
    checkpoint = {
        "format": "quantem-prepared-source254-v1", "complete": True,
        "layout": {
            "format": "quantem-resident-source112-index180-v1", "complete": True,
            "shape": [66, 512, 512, 192, 192], "source_dtype": "<u2",
            "byte_order": "little", "chunk_scans": 16384, "stream_scans": 512,
            "component_alignment": 64, "record_alignment": 4096,
            "files": files, "chunks": chunks,
        },
        "global_state_file": "global-state.npz",
        "global_state_file_sha256": _sha((root / "global-state.npz").read_bytes()),
        "global_state": {name: {
            "shape": list(values.shape), "dtype": values.dtype.str,
            "nbytes": values.nbytes, "sha256": _sha(values.tobytes()),
        } for name, values in arrays.items()},
        "index_rebuild": {"source_codec": "source112-tans1024-pair-v1"},
        "original_acquisitions": {"title": "Preserved test series", "acquisitions": [
            {"id": index, "source_identity_sha256": _sha(f"source-{index}".encode()),
             "original_path_provenance": f"original/{index}.h5"}
            for index in range(66)
        ]},
        "provenance": {"parent_manifest_sha256": "f" * 64},
    }
    (root / "checkpoint.json").write_text(json.dumps(checkpoint))
    return checkpoint


@pytest.mark.parametrize("archive_prefix", ["quantem", "retained-session"])
def test_export_preserves_records_and_identity_without_payload_reads(
    tmp_path, monkeypatch, archive_prefix
):
    root = tmp_path / "archive"
    checkpoint = _archive(root)
    checkpoint["format"] = f"{archive_prefix}-prepared-source254-v1"
    checkpoint["layout"]["format"] = f"{archive_prefix}-resident-source112-index180-v1"
    (root / "checkpoint.json").write_text(json.dumps(checkpoint))
    source_stats = [(root / name).stat() for name in ("data-0.bin", "data-1.bin")]
    original_open = Path.open

    def metadata_only_open(path, *args, **kwargs):
        if path.name in {"data-0.bin", "data-1.bin"}:
            raise AssertionError("Metadata export opened a scientific payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only_open)
    output = tmp_path / "browser"
    manifest_path = _export_source112_browser(root, output)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "source112-tans1024-pair-v1"
    assert manifest["source_checkpoint"] == checkpoint
    assert manifest["layout"] == checkpoint["layout"]
    assert len(manifest["layout"]["chunks"]) == 1056
    assert manifest["original_acquisitions"] == checkpoint["original_acquisitions"]
    assert manifest["provenance"]["payload_hashes_verified"] is False
    checkpoint_sha256 = _sha((root / "checkpoint.json").read_bytes())
    assert manifest["provenance"]["checkpoint_sha256"] == checkpoint_sha256
    for item in manifest["layout"]["files"]:
        link = output / item["name"]
        assert not link.is_symlink()
        assert os.path.samefile(link, root / item["name"])
        before = source_stats[manifest["layout"]["files"].index(item)]
        after = (root / item["name"]).stat()
        assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    for descriptor in manifest["globals"].values():
        raw = (output / descriptor["file"]).read_bytes()
        assert len(raw) == descriptor["nbytes"]
        assert _sha(raw) == descriptor["sha256"]
        values = np.frombuffer(raw, dtype=descriptor["dtype"])
        assert values.size == np.prod(descriptor["shape"])
    valid = np.fromfile(output / "valid.u8", dtype="u1")
    expected_valid = np.ones(36864, dtype="u1")
    expected_valid[[0, 7, 19]] = 0
    np.testing.assert_array_equal(valid, expected_valid)
    source_valid_hash = checkpoint["global_state"]["planner__valid"]["sha256"]
    assert manifest["globals"]["valid"]["sha256"] == source_valid_hash
    dense = np.fromfile(output / "dense-columns.u32", dtype="<u4")
    sparse = np.fromfile(output / "sparse-columns.u32", dtype="<u4")
    np.testing.assert_array_equal(dense, np.arange(17466))
    np.testing.assert_array_equal(sparse, np.arange(17466, 36864)[::-1])


@pytest.mark.parametrize("defect", [
    "incomplete", "shape", "npz_hash", "array_hash", "metadata_path",
    "shard_path", "missing_record", "component_extent", "source_identity",
])
def test_export_rejects_untrustworthy_or_incomplete_metadata(tmp_path, defect):
    root = tmp_path / "archive"
    checkpoint = _archive(root)
    if defect == "incomplete":
        checkpoint["complete"] = False
    elif defect == "shape":
        checkpoint["layout"]["shape"][0] = 65
    elif defect == "npz_hash":
        checkpoint["global_state_file_sha256"] = "0" * 64
    elif defect == "array_hash":
        checkpoint["global_state"]["model_ids"]["sha256"] = "0" * 64
    elif defect == "metadata_path":
        checkpoint["global_state_file"] = "../global-state.npz"
    elif defect == "shard_path":
        checkpoint["layout"]["files"][0]["name"] = "../data-0.bin"
    elif defect == "missing_record":
        checkpoint["layout"]["chunks"].pop()
    elif defect == "component_extent":
        checkpoint["layout"]["chunks"][0]["components"][0]["nbytes"] = 1 << 30
    elif defect == "source_identity":
        checkpoint["original_acquisitions"]["acquisitions"].pop()
    (root / "checkpoint.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):
        _export_source112_browser(root, tmp_path / "browser")
    assert not (tmp_path / "browser").exists()


def test_export_does_not_write_into_archive_or_replace_existing_export(tmp_path):
    root = tmp_path / "archive"
    _archive(root)
    with pytest.raises(ValueError, match="outside"):
        _export_source112_browser(root, root / "browser")
    output = tmp_path / "browser"
    output.mkdir()
    (output / "keep.txt").write_text("original output")
    with pytest.raises(FileExistsError):
        _export_source112_browser(root, output)
    assert (output / "keep.txt").read_text() == "original output"


def test_export_cross_filesystem_reports_regular_file_grant_requirement(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    _archive(root)

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", cross_device)
    with pytest.raises(ValueError, match="same filesystem"):
        _export_source112_browser(root, tmp_path / "browser")
    assert not (tmp_path / "browser").exists()
    assert not list(tmp_path.glob(".browser-*"))
