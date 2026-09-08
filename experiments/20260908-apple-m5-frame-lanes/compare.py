"""Summarize matched original-to-packed A/B/A loads, including all repeats."""

import json
import statistics
import sys
from pathlib import Path


def read(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert records[-1]["phase"] == "complete", f"Incomplete run: {path}"
    return [record for record in records if record["phase"] == "resident"]


arms = {name: read(path) for name, path in zip(("a", "candidate", "b"), sys.argv[1:])}
assert len(arms) == 3
key = lambda row: (row["source_identity"], row["cycle"])
matched = {name: {key(row): row for row in rows} for name, rows in arms.items()}
assert matched["a"].keys() == matched["candidate"].keys() == matched["b"].keys()
for identity in matched["a"]:
    for field in ("sample_hashes", "resident_bytes", "shape", "source_dtype"):
        assert len({json.dumps(rows[identity][field]) for rows in matched.values()}) == 1

result = {
    "matched_loads_per_arm": len(matched["a"]),
    "sample_frame_hashes_exact": True,
    "resident_bytes_identical": True,
    "independent_full_count_parity": False,
    "cold_io_claim": False,
    "ui_present_measured": False,
    "boundary": "source open through full packed resident; cached packing plan; DPC reuse after cycle 0",
    "arms": {},
}
for name, rows in arms.items():
    result["arms"][name] = {}
    for cycle in sorted({row["cycle"] for row in rows}):
        selected = [row for row in rows if row["cycle"] == cycle]
        result["arms"][name][str(cycle)] = {
            "loads": len(selected),
            "median_seconds": statistics.median(row["seconds"] for row in selected),
            "max_seconds": max(row["seconds"] for row in selected),
            "median_decode_gpu_seconds": statistics.median(row["gpu_decode_seconds"] for row in selected),
        }
print(json.dumps(result, indent=2))
