#!/usr/bin/env python3
"""Fully decode and authenticate every raw-LZ4 QGIX v1 payload shard."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from quantem.gpu.io._compact_h5 import CompactH5Index, CompactH5ReferenceDecoder


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


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
        library.LZ4_decompress_safe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.LZ4_decompress_safe.restype = ctypes.c_int
        return library
    raise RuntimeError("Cannot load liblz4. Tried " + "; ".join(errors))


def _read_exact(stream, byte_count: int, label: str) -> bytes:
    value = stream.read(byte_count)
    if len(value) != byte_count:
        raise ValueError(f"{label} ended after {len(value)} of {byte_count} bytes")
    return value


def validate_compact_h5_v1_full(
    compact_path: Path,
    *,
    expected_whole_file_sha256: str,
    lz4_path: str | None = None,
) -> dict[str, Any]:
    """Return a receipt after decoding and hashing every QGIX v1 shard."""
    started = time.perf_counter()
    compact_path = compact_path.resolve()
    actual_whole_file_sha256 = _sha256_file(compact_path)
    if actual_whole_file_sha256 != expected_whole_file_sha256:
        raise ValueError(
            f"compact whole-file SHA-256 is {actual_whole_file_sha256}, "
            f"expected {expected_whole_file_sha256}"
        )
    index = CompactH5Index.from_file(compact_path)
    if index.schema_version != 1:
        raise ValueError("full raw-LZ4 validation requires QGIX v1")
    decoder = CompactH5ReferenceDecoder(index)
    library = _load_lz4(lz4_path)
    results: list[dict[str, Any]] = []
    total_compressed_bytes = 0
    total_decoded_bytes = 0
    total_chunks = 0
    with compact_path.open("rb") as stream:
        for ordinal, shard in enumerate(index.shards):
            decoder.validate_shard_metadata(ordinal)
            manifest_record = index.manifest["shards"][ordinal]
            stream.seek(shard.payload_offset)
            encoded = bytearray(
                _read_exact(stream, shard.payload_bytes, f"shard {ordinal} payload")
            )
            stream.seek(shard.lengths_offset)
            lengths_bytes = _read_exact(
                stream,
                shard.lengths_bytes,
                f"shard {ordinal} chunk lengths",
            )
            stream.seek(shard.widths_offset)
            widths_bytes = _read_exact(
                stream,
                shard.widths_bytes,
                f"shard {ordinal} descriptor widths",
            )
            encoded_sha256 = hashlib.sha256(encoded).hexdigest()
            widths_sha256 = hashlib.sha256(widths_bytes).hexdigest()
            if encoded_sha256 != manifest_record.get("payload_compressed_sha256"):
                raise ValueError(f"shard {ordinal} compressed payload SHA-256 failed")
            if widths_sha256 != manifest_record.get("descriptor_widths_sha256"):
                raise ValueError(f"shard {ordinal} descriptor-width SHA-256 failed")
            lengths = np.frombuffer(lengths_bytes, dtype=np.uint8).astype(np.uint32)
            lengths += 1
            offsets = np.empty(lengths.size + 1, dtype=np.uint64)
            offsets[0] = 0
            np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])
            if int(offsets[-1]) != len(encoded):
                raise ValueError(f"shard {ordinal} chunk lengths do not cover payload")

            source_buffer = (ctypes.c_ubyte * len(encoded)).from_buffer(encoded)
            destination = (ctypes.c_ubyte * index.payload_chunk_bytes)()
            digest = hashlib.sha256()
            for chunk, compressed_bytes in enumerate(lengths):
                decoded_bytes = min(
                    index.payload_chunk_bytes,
                    shard.decoded_bytes - chunk * index.payload_chunk_bytes,
                )
                observed = int(
                    library.LZ4_decompress_safe(
                        ctypes.byref(source_buffer, int(offsets[chunk])),
                        destination,
                        int(compressed_bytes),
                        int(decoded_bytes),
                    )
                )
                if observed != decoded_bytes:
                    raise ValueError(
                        f"shard {ordinal} chunk {chunk} decoded {observed}, "
                        f"expected {decoded_bytes} bytes"
                    )
                digest.update(bytes(destination[:decoded_bytes]))
            decoded_sha256 = digest.hexdigest()
            if decoded_sha256 != shard.decoded_sha256:
                raise ValueError(f"shard {ordinal} decoded payload SHA-256 failed")
            total_compressed_bytes += shard.payload_bytes
            total_decoded_bytes += shard.decoded_bytes
            total_chunks += shard.chunk_count
            results.append(
                {
                    "ordinal": ordinal,
                    "compressed_bytes": shard.payload_bytes,
                    "decoded_bytes": shard.decoded_bytes,
                    "chunk_count": shard.chunk_count,
                    "compressed_sha256": encoded_sha256,
                    "descriptor_widths_sha256": widths_sha256,
                    "decoded_sha256": decoded_sha256,
                }
            )
    return {
        "schema": "quantem.gpu.compact-h5-v1-full-validation/v1",
        "status": "pass",
        "compact_path": str(compact_path),
        "whole_file_sha256": actual_whole_file_sha256,
        "source_identity_sha256": index.source_identity_sha256,
        "source_raw_logical_sha256": index.manifest.get("source_raw_logical_sha256"),
        "source_shape": list(index.shape),
        "source_dtype": "uint16",
        "working_dtype": index.manifest.get("working_dtype"),
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "shard_count": len(results),
        "total_chunk_count": total_chunks,
        "total_compressed_bytes": total_compressed_bytes,
        "total_decoded_bytes": total_decoded_bytes,
        "decoded_sha256_checks": len(results),
        "compressed_sha256_checks": len(results),
        "descriptor_width_sha256_checks": len(results),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "gpu_executed": False,
        "shards": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", required=True, type=Path)
    parser.add_argument("--expected-whole-file-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--liblz4")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to replace existing {arguments.output}")
    result = validate_compact_h5_v1_full(
        arguments.compact,
        expected_whole_file_sha256=arguments.expected_whole_file_sha256,
        lz4_path=arguments.liblz4,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
