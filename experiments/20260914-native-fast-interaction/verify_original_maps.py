"""Independent bounded HDF5/NumPy oracle for all seven native ADF maps.

CPU computation is deliberately reference-only, not an application load path.
Each native hash covers every uint32 value of the full 512x512 map. This proves
the final custom aperture, not every intermediate drag geometry.
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import hdf5plugin  # noqa: F401: registers the original file's bitshuffle codec
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native_result.read_text())
    state = native["modes"]["Fast"]["final_state"]
    expected = native["modes"]["Fast"]["custom_endpoint"]
    geometry = {k: state[k] for k in ("detector_center_row", "detector_center_col",
                "detector_inner_radius", "detector_outer_radius")}
    rows, cols = np.indices((192, 192), dtype=np.float32)
    squared = ((rows - np.float32(geometry["detector_center_row"])) ** 2
               + (cols - np.float32(geometry["detector_center_col"])) ** 2)
    aperture = ((squared >= np.float32(geometry["detector_inner_radius"]) ** 2)
                & (squared < np.float32(geometry["detector_outer_radius"]) ** 2))
    masters = sorted(args.folder.glob("*_master.h5"))
    assert len(masters) == len(expected) == 7
    report = {"pass": False, "geometry": geometry, "sources": [],
              "definition": "independent original HDF5 uint64 reduction, then exact uint32 full-map FNV comparison; no clipping"}
    try:
        for master, (dataset_id, native_hash) in zip(masters, expected.items()):
            started = time.monotonic()
            with h5py.File(master) as file:
                bad = file["entry/instrument/detector/detectorSpecific/pixel_mask"][...] != 0
                mask = aperture & ~bad
                sums = []
                for key in sorted(file["entry/data"]):
                    data = file["entry/data"][key]
                    assert data.shape[1:] == (192, 192) and data.dtype == np.uint16
                    for first in range(0, data.shape[0], 256):
                        block = data[first:first + 256]
                        sums.extend(np.sum(block[:, mask], axis=1, dtype=np.uint64).tolist())
            assert len(sums) == 512 * 512 and max(sums) <= np.iinfo(np.uint32).max
            hashed = 14695981039346656037
            for value in sums:
                hashed = ((hashed ^ value) * 1099511628211) & 0xffffffffffffffff
            actual = f"{hashed:016x}"
            record = {"dataset_id": dataset_id, "original": str(master), "full_map_values": len(sums),
                      "excluded_pixels": int(bad.sum()), "expected_hash": actual, "native_hash": native_hash,
                      "pass": actual == native_hash, "oracle_seconds": time.monotonic() - started}
            report["sources"].append(record)
            print(json.dumps(record), flush=True)
            assert record["pass"], "Independent full-map mismatch"
        report["pass"] = True
    finally:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
