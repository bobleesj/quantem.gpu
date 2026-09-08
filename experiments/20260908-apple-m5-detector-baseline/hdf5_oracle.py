"""Independently integrate complete original HDF5 counts for the detector journey.

Example: ``python hdf5_oracle.py --index /tmp/index --reference results.jsonl``.
Requires NumPy, h5py and hdf5plugin. Processes bounded frame batches and never
calls the Metal loader or its detector implementation. All source pixels,
including high-count pixels, are preserved.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers the original bitshuffle/LZ4 filter
import numpy as np


def main() -> None:
    """Hash exact full-scan integrations and fail on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--batch-frames", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    references = [json.loads(line) for line in args.reference.read_text().splitlines()]
    expected = {}
    for row in references:
        if row.get("phase") == "detector":
            key = row["source_identity"], row["case"], row["step"]
            if key in expected and expected[key]["sha256_u32_le"] != row["sha256_u32_le"]:
                raise ValueError("Reference contains inconsistent repeat hashes")
            expected[key] = row
    completed = 0
    for path in sorted(args.index.glob("*/dataset.json")):
        source = json.loads(path.read_text())
        identity = source["sourceIdentitySHA256"]
        cases = sorted((key, value) for key, value in expected.items() if key[0] == identity)
        if not cases:
            continue
        scan_count = source["scanRows"] * source["scanCols"]
        rows, columns = source["detectorRows"], source["detectorCols"]
        detector_row, detector_column = np.indices((rows, columns))
        masks = []
        for _, case in cases:
            distances = ((detector_row - case["center_row"]) ** 2
                         + (detector_column - case["center_column"]) ** 2)
            inner, outer = case["inner_radius"], case["outer_radius"]
            masks.append(((distances <= outer ** 2)
                          & ((inner == 0) | (distances > inner ** 2))).ravel())
        products = np.empty((len(cases), scan_count), dtype=np.uint64)
        digest = hashlib.sha256()
        started = time.monotonic()
        offset = 0
        maximum = 0
        for file_path in source["dataFiles"]:
            with h5py.File(file_path, "r") as handle:
                dataset = handle["/entry/data/data"]
                if dataset.shape[1:] != (rows, columns) or dataset.dtype != np.dtype("uint16"):
                    raise ValueError("Unexpected original detector shape or dtype")
                for first in range(0, dataset.shape[0], args.batch_frames):
                    values = dataset[first:first + args.batch_frames].reshape(-1, rows * columns)
                    end = offset + values.shape[0]
                    if end > scan_count:
                        raise ValueError("Original has more frames than the indexed full scan")
                    maximum = max(maximum, int(values.max()))
                    digest.update(values.astype("<u4").tobytes())
                    for index, mask in enumerate(masks):
                        products[index, offset:end] = values[:, mask].sum(axis=1, dtype=np.uint64)
                    offset = end
            print(json.dumps({"phase": "oracle_progress", "source_identity": identity,
                              "frames_complete": offset, "frames_required": scan_count}), flush=True)
        if offset != scan_count or products.max() > np.iinfo(np.uint32).max:
            raise ValueError("Incomplete full scan or detector sum outside uint32")
        for index, (key, case) in enumerate(cases):
            actual = hashlib.sha256(products[index].astype("<u4").tobytes()).hexdigest()
            passed = actual == case["sha256_u32_le"]
            print(json.dumps({"phase": "oracle_detector", "source_identity": identity,
                              "case": key[1], "step": key[2], "pass": passed,
                              "sha256_u32_le": actual, "expected": case["sha256_u32_le"]}), flush=True)
            if not passed:
                raise ValueError("Independent original-HDF5 detector parity failed")
        print(json.dumps({"phase": "oracle_source", "source_identity": identity,
                          "pass": True, "shape": [source["scanRows"], source["scanCols"], rows, columns],
                          "source_dtype": "uint16", "source_maximum": maximum,
                          "source_sha256_u32_le": digest.hexdigest(), "excluded_pixels": 0,
                          "scan_bin": 1, "detector_bin": 1, "crop": None,
                          "seconds": time.monotonic() - started}), flush=True)
        completed += 1
        if args.limit and completed >= args.limit:
            break
    required = len({key[0] for key in expected})
    if completed == 0 or (args.limit is None and completed != required):
        raise ValueError("Oracle did not cover all requested acquisitions")
    print(json.dumps({"phase": "complete", "sources": completed,
                      "required_sources": required, "full_coverage": completed == required}), flush=True)


if __name__ == "__main__":
    main()
