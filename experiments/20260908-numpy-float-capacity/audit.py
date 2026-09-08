"""Measure exact fixed-block float packing size without a dense volume.

This is a source-capacity audit, not a GPU loading benchmark. It mirrors only
the existing 128-word XOR descriptor's storage formula, not its Metal kernels.
Usage: python audit.py /path/to/merged.npy
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    """Audit every source word using bounded reads and publish a JSON receipt."""
    path = Path(sys.argv[1])
    before = path.stat()
    digest = hashlib.sha256()
    histogram = np.zeros(33, dtype=np.int64)
    count = zeros = fractional = nonfinite = 0
    minimum, maximum = float("inf"), float("-inf")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}[version]
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
        if fortran or dtype != np.dtype("<f4") or len(shape) != 4:
            raise ValueError("Audit requires C-order four-dimensional float32 data")
        expected_words = int(np.prod(shape))
        if expected_words % 128 or before.st_size != offset + expected_words * 4:
            raise ValueError("Unexpected source length or incomplete packing block")
        stream.seek(0)
        digest.update(stream.read(offset))
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
            words = np.frombuffer(block, dtype="<u4").reshape(-1, 128)
            changed = np.bitwise_or.reduce(words ^ words[:, :1], axis=1)
            low = changed & -changed
            trailing = np.where(changed == 0, 32, np.log2(np.maximum(low, 1))).astype(int)
            high = np.where(changed == 0, 0, np.floor(np.log2(np.maximum(changed, 1))) + 1).astype(int)
            width = np.maximum(0, high - trailing)
            histogram += np.bincount(width, minlength=33)
            values = words.view("<f4")
            finite = np.isfinite(values)
            nonfinite += int(np.count_nonzero(~finite))
            zeros += int(np.count_nonzero(values == 0))
            fractional += int(np.count_nonzero(finite & (values != np.floor(values))))
            finite_values = values[finite]
            if finite_values.size:
                minimum = min(minimum, float(finite_values.min()))
                maximum = max(maximum, float(finite_values.max()))
            count += values.size
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise RuntimeError("Source changed during audit")
    assert count == expected_words
    # Each descriptor is 16 bytes; each width bit requires 128 bits = 16 bytes.
    payload = int(np.dot(histogram, np.arange(33, dtype=np.int64)) * 16)
    descriptors = int(histogram.sum()) * 16
    print(json.dumps({
        "shape": shape, "dtype": str(dtype), "source_bytes": before.st_size,
        "source_sha256": digest.hexdigest(), "words_audited": count,
        "zero_words": zeros, "fractional_words": fractional, "nonfinite_words": nonfinite,
        "minimum": minimum, "maximum": maximum, "block_words": 128,
        "width_histogram": histogram.tolist(), "payload_bytes": payload,
        "descriptor_bytes": descriptors, "packed_bytes": payload + descriptors,
        "memory_comparison": "Excludes Metal page alignment, staging, products and app memory",
        "scope": "Complete CPU source-capacity audit, not native load or GPU timing",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
