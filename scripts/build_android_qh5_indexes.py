#!/usr/bin/env python
"""Build exact Android QH5 indexes from an audited HDF5 shard set.

The generated ``QH5IDX01`` files contain only HDF5 byte ranges and compressed
block lengths. Detector values and compressed payloads remain in the original
HDF5 files.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import mmap
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  Registers the audited bitshuffle filter.

MAGIC = b"QH5IDX01"
UINT16_BLOCK_ELEMENTS = 4096
MAX_DECODED_CHUNK_BYTES = 1 << 30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="Rehash every original HDF5 shard before writing its index.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_sidecar(
    path: Path,
    metadata: dict[str, Any],
    words: array.array[int],
) -> None:
    payload = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    if len(payload) > 0xFFFFFFFF or len(words) > 0xFFFFFFFF:
        raise ValueError(f"{path.name} exceeds the QH5IDX01 format limits.")
    if words.typecode != "I" or words.itemsize != 4:
        raise ValueError("QH5IDX01 metadata words must be native uint32 values.")
    if sys.byteorder != "little":
        words = array.array("I", words)
        words.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(MAGIC)
            stream.write(struct.pack("<II", len(payload), len(words)))
            stream.write(payload)
            stream.write(b"\0" * ((-len(payload)) % 4))
            stream.write(words)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _filter_ids(dataset: h5py.Dataset) -> list[int]:
    properties = dataset.id.get_create_plist()
    return [
        int(properties.get_filter(index)[0])
        for index in range(properties.get_nfilters())
    ]


def _contiguous_chunk_extent(dataset: h5py.Dataset) -> tuple[int, int, int]:
    """Return the compressed extent without enumerating every HDF5 chunk."""
    frame_count = int(dataset.shape[0])
    first = dataset.id.get_chunk_info_by_coord((0, 0, 0))
    last = dataset.id.get_chunk_info_by_coord((frame_count - 1, 0, 0))
    if first.byte_offset is None or last.byte_offset is None:
        raise ValueError("HDF5 source has an unallocated detector frame.")
    start = int(first.byte_offset)
    stop = int(last.byte_offset) + int(last.size)
    if start < 0 or stop <= start:
        raise ValueError("HDF5 source has an invalid compressed chunk extent.")
    return start, stop, frame_count


def _build_one(
    source: Path,
    dataset_path: str,
    destination: Path,
    detector_shape: tuple[int, int],
) -> dict[str, Any]:
    with h5py.File(source, "r") as handle:
        dataset = handle[dataset_path]
        if dataset.ndim != 3 or tuple(dataset.shape[1:]) != detector_shape:
            raise ValueError(f"{source.name} has unsupported shape {dataset.shape}.")
        if str(dataset.dtype) != "uint16":
            raise ValueError(f"{source.name} has unsupported dtype {dataset.dtype}.")
        if tuple(dataset.chunks or ()) != (1, *detector_shape):
            raise ValueError(f"{source.name} has unsupported chunks {dataset.chunks}.")
        if 32008 not in _filter_ids(dataset):
            raise ValueError(f"{source.name} is not bitshuffle-compressed.")
        extent_start, extent_stop, frame_count = _contiguous_chunk_extent(dataset)

    detector_elements = math.prod(detector_shape)
    if detector_elements % UINT16_BLOCK_ELEMENTS != 0:
        raise ValueError(
            f"Detector shape {detector_shape} is not divisible into "
            f"{UINT16_BLOCK_ELEMENTS}-element bitshuffle blocks."
        )
    detector_bytes = detector_elements * 2
    frames_per_chunk = max(1, MAX_DECODED_CHUNK_BYTES // detector_bytes)
    blocks_per_frame = detector_elements // UINT16_BLOCK_ELEMENTS
    words = array.array("I")
    output_chunks: list[dict[str, int]] = []
    metadata_gap_count = 0
    metadata_gap_bytes = 0
    with (
        source.open("rb") as stream,
        h5py.File(source, "r") as handle,
        mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ) as source_bytes,
    ):
        dataset = handle[dataset_path]
        position = extent_start

        def align_frame(frame: int, candidate: int) -> int:
            nonlocal metadata_gap_bytes, metadata_gap_count
            if candidate + 12 <= extent_stop:
                header = struct.unpack_from(">QI", source_bytes, candidate)
                if header == (detector_bytes, 8192):
                    return candidate
            info = dataset.id.get_chunk_info_by_coord((frame, 0, 0))
            if info.byte_offset is None:
                raise ValueError(f"{source.name} frame {frame} is unallocated.")
            actual = int(info.byte_offset)
            if actual < candidate or actual + 12 > extent_stop:
                raise ValueError(
                    f"{source.name} frame {frame} has an invalid chunk offset."
                )
            header = struct.unpack_from(">QI", source_bytes, actual)
            if header != (detector_bytes, 8192):
                raise ValueError(
                    f"{source.name} frame {frame} has invalid bitshuffle geometry."
                )
            metadata_gap_count += 1
            metadata_gap_bytes += actual - candidate
            return actual

        for start_frame in range(0, frame_count, frames_per_chunk):
            stop_frame = min(frame_count, start_frame + frames_per_chunk)
            position = align_frame(start_frame, position)
            range_start = position
            metadata_start = len(words)
            for frame in range(start_frame, stop_frame):
                position = align_frame(frame, position)
                frame_start = position
                decoded_bytes, block_bytes = struct.unpack_from(">QI", source_bytes, position)
                if decoded_bytes != detector_bytes or block_bytes != 8192:
                    raise AssertionError("align_frame admitted invalid bitshuffle geometry")
                position += 12
                for _ in range(blocks_per_frame):
                    if position + 4 > extent_stop:
                        raise ValueError(
                            f"{source.name} frame {frame} block header exceeds its "
                            "chunk extent."
                        )
                    compressed_bytes = struct.unpack_from(">I", source_bytes, position)[0]
                    payload = position + 4
                    relative = payload - range_start
                    if relative > 0xFFFFFFFF or compressed_bytes > 0xFFFFFFFF:
                        raise ValueError(
                            f"{source.name} exceeds QH5IDX01 block limits."
                        )
                    if payload + compressed_bytes > extent_stop:
                        raise ValueError(
                            f"{source.name} frame {frame} compressed block exceeds its "
                            "chunk extent."
                        )
                    words.extend((relative, compressed_bytes))
                    position = payload + compressed_bytes
                if position <= frame_start:
                    raise ValueError(f"{source.name} frame {frame} made no progress.")
            range_end = position
            if range_end - range_start > 0xFFFFFFFF:
                raise ValueError(f"{source.name} exceeds the QH5IDX01 range limit.")
            output_chunks.append(
                {
                    "startFrame": start_frame,
                    "nFrames": stop_frame - start_frame,
                    "rangeStart": range_start,
                    "rangeEnd": range_end,
                    "metaOffsetWords": metadata_start,
                    "metaWords": len(words) - metadata_start,
                }
            )
        if position != extent_stop:
            raise ValueError(
                f"{source.name} compressed detector frames do not exactly fill the "
                "first-to-last HDF5 chunk extent."
            )

    source_stat = source.stat()
    metadata = {
        "sourcePath": str(source.resolve()),
        "sourceBytes": source_stat.st_size,
        "sourceMtimeNs": source_stat.st_mtime_ns,
        "detRows": detector_shape[0],
        "detCols": detector_shape[1],
        "nFrames": frame_count,
        "srcDtype": "uint16",
        "blockElems": UINT16_BLOCK_ELEMENTS,
        "nBlocksPerFrame": blocks_per_frame,
        "chunks": output_chunks,
    }
    _write_sidecar(destination, metadata, words)
    return {
        "source_path": str(source.resolve()),
        "source_bytes": source_stat.st_size,
        "index_path": str(destination.resolve()),
        "index_bytes": destination.stat().st_size,
        "index_sha256": _sha256(destination),
        "frames": frame_count,
        "blocks_per_frame": blocks_per_frame,
        "metadata_chunks": len(output_chunks),
        "interleaved_metadata_gaps": metadata_gap_count,
        "interleaved_metadata_bytes": metadata_gap_bytes,
    }


def build_indexes(
    audit_path: Path,
    output_directory: Path,
    *,
    verify_source_hashes: bool = False,
) -> dict[str, Any]:
    """Build indexes for every source shard in a real-fixture audit.

    Parameters
    ----------
    audit_path
        Audit JSON containing the immutable ordered source-shard list.
    output_directory
        Destination for metadata-only ``QH5IDX01`` files.
    verify_source_hashes
        Recompute each source SHA-256 before building its index.

    Returns
    -------
    dict
        Exact source geometry plus ordered source and index provenance.
    """
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    shape = audit.get("source_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(not isinstance(value, int) or value <= 0 for value in shape)
        or shape[2:] != [192, 192]
    ):
        raise ValueError(
            "The audit must describe a positive scan grid with a 192x192 detector."
        )
    if audit.get("source_dtype") != "uint16" or not audit.get("real_data"):
        raise ValueError("The audit is not an admitted real uint16 source.")
    expected_logical_bytes = math.prod(shape) * 2
    if audit.get("source_logical_bytes") != expected_logical_bytes:
        raise ValueError(
            "The audit logical byte count does not match its uint16 geometry."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source_record in audit["source_files"]:
        source = Path(source_record["path"])
        if source.stat().st_size != int(source_record["file_bytes"]):
            raise ValueError(f"{source.name} no longer matches its audited byte size.")
        if verify_source_hashes:
            actual_hash = _sha256(source)
            if actual_hash != source_record["sha256"]:
                raise ValueError(
                    f"{source.name} no longer matches its audited SHA-256."
                )
        destination = output_directory / f"{source.stem}.qh5idx"
        row = _build_one(
            source,
            source_record["dataset_path"],
            destination,
            (shape[2], shape[3]),
        )
        row["source_sha256"] = source_record["sha256"]
        row["ordinal"] = int(source_record["ordinal"])
        rows.append(row)

    if [row["ordinal"] for row in rows] != list(range(len(rows))):
        raise ValueError("The audited sources are not in a complete ordinal sequence.")
    if sum(row["frames"] for row in rows) != math.prod(audit["source_shape"][:2]):
        raise ValueError("The indexed sources do not cover the complete scan.")
    return {
        "schema": "quantem-gpu-android-qh5-index-manifest-v1",
        "audit_path": str(audit_path.resolve()),
        "source_shape": audit["source_shape"],
        "source_dtype": audit["source_dtype"],
        "source_logical_bytes": audit["source_logical_bytes"],
        "source_identity_sha256": audit["source_identity_sha256"],
        "logical_source_sha256": audit["range_audit"]["logical_source_sha256"],
        "values_above_255": audit["range_audit"]["values_above_255"],
        "files": rows,
        "total_source_bytes": sum(row["source_bytes"] for row in rows),
        "total_index_bytes": sum(row["index_bytes"] for row in rows),
    }


def main() -> None:
    args = _parse_args()
    manifest = build_indexes(
        args.audit_json,
        args.output_dir,
        verify_source_hashes=args.verify_source_hashes,
    )
    manifest_path = args.manifest_json or args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
