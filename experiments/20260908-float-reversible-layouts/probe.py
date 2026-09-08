"""Screen exact float transforms for capacity, not native loading speed.

Usage: python probe.py /path/to/merged.npy
Samples 32 stratified eight-frame blocks across the original acquisition.
Every candidate is decoded and compared as integer bits, not float values.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import zstandard as zstd


def _shuffle(words: np.ndarray, layout: str) -> bytes:
    """Arrange bytes or bit planes without changing any word bits."""
    if layout == "raw":
        return words.tobytes()
    octets = words.reshape(-1).view(np.uint8).reshape(-1, 4)
    if layout == "bytes":
        return octets.T.copy().tobytes()
    bits = np.unpackbits(octets, axis=1, bitorder="little")
    return np.packbits(bits.T, axis=1, bitorder="little").tobytes()


def _unshuffle(raw: bytes, shape: tuple[int, ...], layout: str) -> np.ndarray:
    """Invert the exact byte or bit-plane arrangement."""
    count = int(np.prod(shape))
    octets = np.frombuffer(raw, dtype=np.uint8)
    if layout == "raw":
        return octets.view("<u4").reshape(shape)
    if layout == "bytes":
        return octets.reshape(4, count).T.copy().view("<u4").reshape(shape)
    bits = np.unpackbits(octets.reshape(32, count // 8),
                         axis=1, bitorder="little").T
    return np.packbits(bits, axis=1, bitorder="little").copy().view("<u4").reshape(shape)


def _main() -> None:
    path = Path(sys.argv[1])
    before = path.stat()
    compressor = zstd.ZstdCompressor(level=3)
    decompressor = zstd.ZstdDecompressor()
    candidates = [("identity", "raw"), ("identity", "bytes"),
                  ("identity", "bits"), ("scan-xor", "bits"),
                  ("scan-subtract", "bytes"), ("scan-subtract", "bits")]
    rng = np.random.default_rng(20260908)
    records = []
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError(f"Expected NumPy v1 source, got {version}")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        offset = stream.tell()
        if fortran or dtype != np.dtype("<f4") or len(shape) != 4:
            raise ValueError(f"Expected C-order 4D float32 source, got {shape}, {dtype}")
        frames = shape[0] * shape[1]
        pixels = shape[2] * shape[3]
        if before.st_size != offset + frames * pixels * 4:
            raise ValueError("Incomplete source: exact original length required")
        for stratum in range(32):
            low, high = stratum * frames // 32, (stratum + 1) * frames // 32
            start = int(rng.integers(low, high - 8 + 1))
            stream.seek(offset + start * pixels * 4)
            raw = stream.read(8 * pixels * 4)
            words = np.frombuffer(raw, dtype="<u4").reshape(8, pixels)
            record = {"start_frame": start, "frames": 8,
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "original_bytes": len(raw), "candidates": {}}
            for predictor, layout in candidates:
                changed = words.copy()
                if predictor == "scan-xor":
                    changed[1:] = words[1:] ^ words[:-1]
                elif predictor == "scan-subtract":
                    changed[1:] = words[1:] - words[:-1]
                packed = compressor.compress(_shuffle(changed, layout))
                restored = _unshuffle(decompressor.decompress(packed),
                                       words.shape, layout)
                if predictor == "scan-xor":
                    restored = np.bitwise_xor.accumulate(restored, axis=0)
                elif predictor == "scan-subtract":
                    restored = np.cumsum(restored, axis=0, dtype=np.uint32)
                if restored.tobytes() != raw:
                    raise AssertionError(f"Lossless round trip failed: {predictor}/{layout}")
                record["candidates"][f"{predictor}/{layout}"] = len(packed)
            records.append(record)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError("Source changed during sampling")
    original_bytes = sum(record["original_bytes"] for record in records)
    estimates = {}
    for name in records[0]["candidates"]:
        size = sum(record["candidates"][name] for record in records)
        estimates[name] = {"sample_compressed_bytes": size,
                           "projected_full_bytes": round(size / original_bytes * frames * pixels * 4),
                           "ratio": size / original_bytes}
    print(json.dumps({"scope": "Stratified CPU capacity screen, not full-source packing or GPU speed",
                      "shape": shape, "dtype": str(dtype), "source_bytes": before.st_size,
                      "sample_bytes": original_bytes, "seed": 20260908,
                      "codec": "zstd-level3", "zstandard_version": zstd.__version__,
                      "numpy_version": np.__version__, "exact_sample_roundtrips": True,
                      "estimates_exclude": "GPU decoder, index, staging, app and OS memory",
                      "candidates": estimates, "blocks": records}, indent=2), flush=True)


if __name__ == "__main__":
    _main()
