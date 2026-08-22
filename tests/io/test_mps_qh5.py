import json
import struct
from pathlib import Path

import pytest

from quantem.gpu.io.backends.mps.qh5 import (
    QH5PreparedSourceError,
    _can_use_row8_bin,
    open_prepared_source,
)


def _write_prepared_source(root: Path) -> tuple[Path, Path]:
    master = root / "scan_master.h5"
    data = root / "scan_data_000001.h5"
    index = root / "scan_data_000001.qh5idx"
    master.write_bytes(b"master")
    data.write_bytes(bytes(range(16)))

    source_stat = data.stat()
    index_metadata = {
        "sourcePath": str(data.resolve()),
        "sourceBytes": source_stat.st_size,
        "sourceMtimeNs": source_stat.st_mtime_ns,
        "detRows": 4,
        "detCols": 8,
        "nFrames": 1,
        "srcDtype": "uint16",
        "blockElems": 4096,
        "nBlocksPerFrame": 1,
        "chunks": [
            {
                "startFrame": 0,
                "nFrames": 1,
                "rangeStart": 0,
                "rangeEnd": source_stat.st_size,
                "metaOffsetWords": 0,
                "metaWords": 2,
            }
        ],
    }
    encoded_metadata = json.dumps(index_metadata, separators=(",", ":")).encode()
    word_start = (16 + len(encoded_metadata) + 3) & ~3
    index.write_bytes(
        b"QH5IDX01"
        + struct.pack("<II", len(encoded_metadata), 2)
        + encoded_metadata
        + bytes(word_start - 16 - len(encoded_metadata))
        + struct.pack("<II", 0, 4)
    )

    source_identity = "a" * 64
    dataset = {
        "masterPath": str(master.resolve()),
        "dataFiles": [str(data.resolve())],
        "indexFiles": [str(index.resolve())],
        "sourceIdentitySHA256": source_identity,
        "sourceDtype": "uint16",
        "badPixelIndices": [3],
        "scanRows": 1,
        "scanCols": 1,
        "detectorRows": 4,
        "detectorCols": 8,
    }
    audit = {
        "schema": "quantem.gpu.value-range-audit/v1",
        "sourceIdentitySHA256": source_identity,
        "sourceDtype": "uint16",
        "badPixelIndices": [3],
        "maximum": 12,
        "pixelsAbove255": 0,
    }
    snapshots = []
    for path in (master, data):
        stat = path.stat()
        snapshots.append(
            {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "modificationNanoseconds": stat.st_mtime_ns,
            }
        )
    source_hashes = {
        "schema": "quantem.gpu.source-hashes/v1",
        "aggregateHash": source_identity,
        "snapshots": snapshots,
    }
    (root / "dataset.json").write_text(json.dumps(dataset))
    (root / "value-range-audit.json").write_text(json.dumps(audit))
    (root / "source-hashes.json").write_text(json.dumps(source_hashes))
    return master, data


def test_open_prepared_source_preserves_scientific_contract(tmp_path: Path) -> None:
    master, _ = _write_prepared_source(tmp_path)

    prepared = open_prepared_source(tmp_path, master_path=master)

    assert prepared.scan_shape == (1, 1)
    assert prepared.detector_shape == (4, 8)
    assert prepared.source_dtype == "uint16"
    assert prepared.bad_pixel_indices == (3,)
    assert prepared.audited_maximum == 12
    assert prepared.audited_pixels_above_255 == 0
    assert prepared.total_frames == 1


def test_open_prepared_source_rejects_stale_payload(tmp_path: Path) -> None:
    _, data = _write_prepared_source(tmp_path)
    data.write_bytes(data.read_bytes() + b"changed")

    with pytest.raises(QH5PreparedSourceError, match="snapshot is stale"):
        open_prepared_source(tmp_path)


def test_open_prepared_source_rejects_out_of_range_block(tmp_path: Path) -> None:
    _write_prepared_source(tmp_path)
    index = tmp_path / "scan_data_000001.qh5idx"
    payload = bytearray(index.read_bytes())
    payload[-8:] = struct.pack("<II", 15, 4)
    index.write_bytes(payload)

    with pytest.raises(QH5PreparedSourceError, match="outside its indexed range"):
        open_prepared_source(tmp_path)


def test_row8_bin_requires_aligned_adjacent_output_pixels() -> None:
    assert _can_use_row8_bin((192, 192))
    assert _can_use_row8_bin((128, 96))
    assert not _can_use_row8_bin((192, 80))
