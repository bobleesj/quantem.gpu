#!/usr/bin/env python3
"""Build an exact QGIX v1 uint16 HDF5 file from packed detector shards.

The input packed-shard manifest is produced by the native detector packer.  A
separate source contract binds those shards to the original HDF5 identity,
logical hash, detector mask, and optional detector calibration.  The writer
does not crop, bin, narrow, or zero the packed payload.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

USER_BLOCK_BYTES = 64 * 1024
USER_BLOCK_MAGIC = b"QGPUH5\0\1"
INDEX_MAGIC = b"QGIX\0\0\0\1"
PAYLOAD_CHUNK_BYTES = 128
SCAN_TILE = 128


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256_value(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _load_lz4(path: str | None) -> ctypes.CDLL:
    candidates = (
        [path]
        if path
        else ["liblz4.so.1", "liblz4.so", "/opt/homebrew/lib/liblz4.dylib"]
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        library.LZ4_compressBound.argtypes = [ctypes.c_int]
        library.LZ4_compressBound.restype = ctypes.c_int
        library.LZ4_compress_default.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.LZ4_compress_default.restype = ctypes.c_int
        return library
    raise RuntimeError("Cannot load liblz4. Tried " + "; ".join(errors))


def _compress_block(library: ctypes.CDLL, source: memoryview) -> bytes:
    source_bytes = bytes(source)
    capacity = int(library.LZ4_compressBound(len(source_bytes)))
    destination = ctypes.create_string_buffer(capacity)
    source_buffer = ctypes.create_string_buffer(source_bytes)
    encoded = int(
        library.LZ4_compress_default(
            source_buffer,
            destination,
            len(source_bytes),
            capacity,
        )
    )
    if encoded <= 0:
        raise RuntimeError("LZ4 compression failed")
    if encoded > 256:
        raise ValueError(
            f"A {PAYLOAD_CHUNK_BYTES}-byte payload block encoded to {encoded} "
            "bytes, beyond the QGIX v1 one-byte length contract."
        )
    return destination.raw[:encoded]


def _compress_payload(library: ctypes.CDLL, payload: memoryview) -> tuple[bytes, bytes]:
    compressed = bytearray()
    lengths_minus_one = bytearray()
    for start in range(0, len(payload), PAYLOAD_CHUNK_BYTES):
        encoded = _compress_block(
            library,
            payload[start : start + PAYLOAD_CHUNK_BYTES],
        )
        compressed.extend(encoded)
        lengths_minus_one.append(len(encoded) - 1)
    return bytes(compressed), bytes(lengths_minus_one)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain one JSON object")
    return value


def _validate_source_contract(contract: dict[str, Any]) -> dict[str, Any]:
    shape = tuple(contract.get("source_shape", ()))
    if len(shape) != 4 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("source_shape must contain four positive integers")
    if contract.get("source_dtype") != "uint16":
        raise ValueError(
            "the exact packed source contract requires source_dtype uint16"
        )
    for field in (
        "source_identity_sha256",
        "source_raw_logical_sha256",
        "detector_mask_sha256",
    ):
        _sha256_value(contract.get(field), field)
    coordinates = contract.get("masked_detector_pixels")
    if not isinstance(coordinates, list):
        raise TypeError("masked_detector_pixels must be a row-column list")
    detector_rows, detector_columns = shape[2:]
    flat: list[int] = []
    for coordinate in coordinates:
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(type(value) is not int for value in coordinate)
            or not 0 <= coordinate[0] < detector_rows
            or not 0 <= coordinate[1] < detector_columns
        ):
            raise ValueError(f"invalid detector-mask coordinate {coordinate!r}")
        flat.append(coordinate[0] * detector_columns + coordinate[1])
    if flat != sorted(set(flat)):
        raise ValueError("masked_detector_pixels must be unique row-major coordinates")
    calibration = contract.get("detector_calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            raise ValueError("detector_calibration must be a JSON object")
        if (
            calibration.get("source_identity_sha256")
            != contract["source_identity_sha256"]
        ):
            raise ValueError("detector_calibration belongs to a different source")
    return {**contract, "source_shape": shape}


def _validate_packed_shard(
    descriptors: np.ndarray,
    payload_words: int,
    *,
    detector_pixels: int,
    scans_per_shard: int,
) -> np.ndarray:
    if scans_per_shard <= 0 or scans_per_shard % SCAN_TILE:
        raise ValueError("packed shards must contain complete 128-scan tiles")
    tile_count = scans_per_shard // SCAN_TILE
    if descriptors.size != detector_pixels * tile_count:
        raise ValueError(
            f"packed shard has {descriptors.size} descriptors, expected "
            f"{detector_pixels * tile_count}"
        )
    widths = descriptors & np.uint32(31)
    maximum = int(widths.max(initial=0))
    if maximum > 16:
        raise ValueError(f"packed shard requires width {maximum}, beyond exact uint16")
    offsets = descriptors >> np.uint32(5)
    expected = np.zeros(descriptors.size, dtype=np.uint64)
    word_counts = widths.astype(np.uint64) * np.uint64(SCAN_TILE // 32)
    if expected.size > 1:
        np.cumsum(word_counts[:-1], dtype=np.uint64, out=expected[1:])
    if not np.array_equal(offsets.astype(np.uint64), expected):
        raise ValueError("packed descriptor offsets do not exactly cover the payload")
    if int(expected[-1] + word_counts[-1]) != payload_words:
        raise ValueError("packed descriptors do not end at the payload boundary")
    return widths.astype(np.uint8)


def build_compact_h5_uint16(
    packed_manifest_path: Path,
    source_contract_path: Path,
    output_path: Path,
    *,
    lz4_path: str | None = None,
) -> dict[str, Any]:
    """Build one immutable exact QGIX v1 uint16 source."""
    packed_manifest_path = packed_manifest_path.resolve()
    source_contract_path = source_contract_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing output {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packed = _load_json(packed_manifest_path, "packed manifest")
    contract = _validate_source_contract(
        _load_json(source_contract_path, "source contract")
    )
    if packed.get("raw_logical_sha256") != contract["source_raw_logical_sha256"]:
        raise ValueError("packed and source-contract raw logical hashes disagree")
    shape = contract["source_shape"]
    detector_pixels = shape[2] * shape[3]
    shard_records = packed.get("shards")
    if not isinstance(shard_records, list) or not shard_records:
        raise ValueError("packed manifest contains no shards")
    scans_per_shard = int(shard_records[0].get("scan_count", 0))
    if any(
        int(record.get("scan_count", 0)) != scans_per_shard for record in shard_records
    ):
        raise ValueError("packed shards must use one scans_per_shard value")
    if len(shard_records) * scans_per_shard != shape[0] * shape[1]:
        raise ValueError("packed shards do not cover the complete scan shape")

    library = _load_lz4(lz4_path)
    manifest: dict[str, Any] = {
        "schema": "quantem.gpu.packed-detector-h5/v1",
        "status": "complete",
        "source_identity_sha256": contract["source_identity_sha256"],
        "source_raw_logical_sha256": contract["source_raw_logical_sha256"],
        "source_shape": list(shape),
        "source_dtype": "uint16",
        "working_dtype": "uint16",
        "working_value_definition": (
            "all source counts exactly; authenticated detector-mask pixels "
            "are excluded from scientific products"
        ),
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "shard_count": len(shard_records),
        "scans_per_shard": scans_per_shard,
        "payload_chunk_bytes": PAYLOAD_CHUNK_BYTES,
        "payload_chunk_codec": "independent raw LZ4 blocks",
        "payload_chunk_length_codec": "uint8 encoded_bytes_minus_one",
        "descriptor_codec": "uint8 five-bit widths",
        "masked_detector_pixels": contract["masked_detector_pixels"],
        "detector_mask_sha256": contract["detector_mask_sha256"],
        "masked_detector_payload_policy": "retained_exactly_in_payload",
        "shards": [],
    }
    if contract.get("detector_calibration") is not None:
        manifest["detector_calibration"] = contract["detector_calibration"]

    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".partial",
        dir=output_path.parent,
    )
    os.close(temporary_descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    dataset_paths: list[tuple[dict[str, Any], str, str]] = []
    packed_root = packed_manifest_path.parent
    try:
        with h5py.File(
            temporary,
            "w",
            userblock_size=USER_BLOCK_BYTES,
            libver="latest",
        ) as handle:
            root = handle.create_group("quantem_gpu")
            root.attrs["schema"] = manifest["schema"]
            root.attrs["source_identity_sha256"] = manifest["source_identity_sha256"]
            root.attrs["source_shape"] = shape
            root.attrs["source_dtype"] = "uint16"
            root.attrs["working_dtype"] = "uint16"
            shards_group = root.create_group("shards")
            for ordinal, source_record in enumerate(shard_records):
                if int(source_record.get("ordinal", ordinal)) != ordinal:
                    raise ValueError("packed shard ordinals are not contiguous")
                descriptors_record = source_record["descriptors"]
                payload_record = source_record["payload"]
                descriptors_path = packed_root / descriptors_record["path"]
                payload_path = packed_root / payload_record["path"]
                if _sha256_file(descriptors_path) != descriptors_record["sha256"]:
                    raise ValueError(f"descriptor SHA-256 changed for shard {ordinal}")
                if _sha256_file(payload_path) != payload_record["sha256"]:
                    raise ValueError(f"payload SHA-256 changed for shard {ordinal}")
                descriptors = np.fromfile(descriptors_path, dtype="<u4")
                payload = np.memmap(payload_path, dtype="u1", mode="r")
                if payload.size % 4:
                    raise ValueError(f"payload shard {ordinal} is not u32 aligned")
                widths = _validate_packed_shard(
                    descriptors,
                    payload.size // 4,
                    detector_pixels=detector_pixels,
                    scans_per_shard=scans_per_shard,
                )
                compressed, lengths = _compress_payload(library, memoryview(payload))
                group = shards_group.create_group(f"{ordinal:03d}")
                payload_dataset = f"/quantem_gpu/shards/{ordinal:03d}/payload_lz4"
                lengths_dataset = (
                    f"/quantem_gpu/shards/{ordinal:03d}/payload_lengths_minus_one"
                )
                widths_dataset = (
                    f"/quantem_gpu/shards/{ordinal:03d}/descriptor_widths_u8"
                )
                group.create_dataset(
                    "payload_lz4",
                    data=np.frombuffer(compressed, dtype="u1"),
                    chunks=None,
                )
                group.create_dataset(
                    "payload_lengths_minus_one",
                    data=np.frombuffer(lengths, dtype="u1"),
                    chunks=None,
                )
                group.create_dataset(
                    "descriptor_widths_u8",
                    data=widths,
                    chunks=None,
                )
                record: dict[str, Any] = {
                    "ordinal": ordinal,
                    "scan_count": scans_per_shard,
                    "payload_decoded_bytes": int(payload.size),
                    "payload_decoded_sha256": payload_record["sha256"],
                    "payload_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                    "descriptor_count": int(widths.size),
                    "descriptor_widths_sha256": hashlib.sha256(widths).hexdigest(),
                    "payload_chunk_count": len(lengths),
                    "maximum_width": int(widths.max(initial=0)),
                }
                manifest["shards"].append(record)
                dataset_paths.extend(
                    (
                        (record, "payload", payload_dataset),
                        (record, "lengths", lengths_dataset),
                        (record, "descriptor_widths", widths_dataset),
                    )
                )
                del payload
            handle.flush()
            for record, label, path in dataset_paths:
                dataset = handle[path]
                offset = dataset.id.get_offset()
                if (
                    offset is None
                    or offset < USER_BLOCK_BYTES
                    or dataset.chunks is not None
                ):
                    raise RuntimeError(
                        f"{path} is not a directly addressable HDF5 range"
                    )
                record[f"{label}_file_offset"] = int(offset)
                record[f"{label}_file_bytes"] = int(dataset.size)

        header = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        binary = bytearray(
            struct.pack(
                "<8sIIIIIII",
                INDEX_MAGIC,
                len(shard_records),
                PAYLOAD_CHUNK_BYTES,
                *shape,
                scans_per_shard,
            )
        )
        coordinates = manifest["masked_detector_pixels"]
        binary.extend(struct.pack("<I", len(coordinates)))
        for row, column in coordinates:
            binary.extend(struct.pack("<I", row * shape[3] + column))
        binary.extend(bytes.fromhex(manifest["source_identity_sha256"]))
        for record in manifest["shards"]:
            binary.extend(
                struct.pack(
                    "<QQQQQQQII32s",
                    record["payload_file_offset"],
                    record["payload_file_bytes"],
                    record["lengths_file_offset"],
                    record["lengths_file_bytes"],
                    record["descriptor_widths_file_offset"],
                    record["descriptor_widths_file_bytes"],
                    record["payload_decoded_bytes"],
                    record["descriptor_count"],
                    record["payload_chunk_count"],
                    bytes.fromhex(record["payload_decoded_sha256"]),
                )
            )
        binary_offset = (24 + len(header) + 7) & ~7
        if binary_offset + len(binary) > USER_BLOCK_BYTES:
            raise RuntimeError("compact manifest exceeds the HDF5 user block")
        with temporary.open("r+b") as stream:
            stream.write(
                struct.pack(
                    "<8sIIII",
                    USER_BLOCK_MAGIC,
                    len(header),
                    zlib.crc32(header),
                    binary_offset,
                    len(binary),
                )
            )
            stream.write(header)
            stream.seek(binary_offset)
            stream.write(binary)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "schema": "quantem.gpu.compact-h5-uint16-build/v1",
        "status": "complete",
        "path": str(output_path),
        "file_bytes": output_path.stat().st_size,
        "whole_file_sha256": _sha256_file(output_path),
        "source_identity_sha256": manifest["source_identity_sha256"],
        "source_raw_logical_sha256": manifest["source_raw_logical_sha256"],
        "source_shape": list(shape),
        "source_dtype": "uint16",
        "working_dtype": "uint16",
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "resident_bytes": sum(
            int(record["payload_decoded_bytes"]) + int(record["descriptor_count"]) * 4
            for record in manifest["shards"]
        ),
        "maximum_width": max(
            int(record["maximum_width"]) for record in manifest["shards"]
        ),
        "shard_count": len(shard_records),
        "packed_manifest_sha256": _sha256_file(packed_manifest_path),
        "source_contract_sha256": _sha256_file(source_contract_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-manifest", required=True, type=Path)
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--liblz4")
    arguments = parser.parse_args()
    receipt = build_compact_h5_uint16(
        arguments.packed_manifest,
        arguments.source_contract,
        arguments.output,
        lz4_path=arguments.liblz4,
    )
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        if arguments.receipt.exists():
            raise FileExistsError(
                f"Refusing to replace existing receipt {arguments.receipt}"
            )
        arguments.receipt.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
