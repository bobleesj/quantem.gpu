"""Audit byte-shuffled lossless compression on every original float word.

Usage: python full_audit.py /path/to/merged.npy
Uses eight-frame independent blocks, including their Zstandard frame overhead.
Compressed blocks are verified and discarded; no alternate data file is saved.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import zstandard as zstd


def _main() -> None:
    path = Path(sys.argv[1])
    before = path.stat()
    compressor = zstd.ZstdCompressor(level=3)
    decompressor = zstd.ZstdDecompressor()
    digest = hashlib.sha256()
    compressed_bytes = words_verified = blocks = 0
    with path.open("rb") as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError("This audit requires a NumPy v1 source")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        offset = stream.tell()
        if fortran or dtype != np.dtype("<f4") or len(shape) != 4:
            raise ValueError(f"Expected C-order 4D float32; got {shape}, {dtype}")
        expected_words = int(np.prod(shape))
        if before.st_size != offset + expected_words * 4:
            raise ValueError("Incomplete original source")
        stream.seek(0)
        digest.update(stream.read(offset))
        while raw := stream.read(8 * shape[2] * shape[3] * 4):
            digest.update(raw)
            octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
            packed = compressor.compress(octets.T.copy().tobytes())
            restored = np.frombuffer(decompressor.decompress(packed), dtype=np.uint8)
            if restored.reshape(4, -1).T.copy().tobytes() != raw:
                raise AssertionError(f"Bit-exact decompression failed at block {blocks}")
            compressed_bytes += len(packed)
            words_verified += len(raw) // 4
            blocks += 1
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ) or words_verified != expected_words:
        raise RuntimeError("Source changed or complete coverage failed")
    print(json.dumps({"scope": "Complete CPU source capacity and bit-exact compression audit",
                      "source_sha256": digest.hexdigest(), "source_bytes": before.st_size,
                      "shape": shape, "dtype": str(dtype), "words_verified": words_verified,
                      "blocks": blocks, "frames_per_block": 8, "codec": "byte-shuffle-zstd3",
                      "compressed_bytes": compressed_bytes, "block_index_bytes": (blocks + 1) * 8,
                      "exact_roundtrip": True, "native_gpu_decoder_implemented": False,
                      "gpu_load_time_measured": False,
                      "excludes": "Metal alignment, staging, decoder state, app and OS memory"},
                     indent=2), flush=True)


if __name__ == "__main__":
    _main()
