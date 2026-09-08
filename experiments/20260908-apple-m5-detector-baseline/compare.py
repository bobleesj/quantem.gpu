"""Compare exact detector fingerprints and costs across identical journeys.

Run ``python compare.py baseline.jsonl candidate.jsonl``. Excludes trial zero
from steady-state timing, but checks every output hash, including trial zero.
GPU milliseconds are not a presented-frame-rate measurement.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def records(path: str) -> list[dict]:
    """Read machine-generated benchmark records from a JSON-lines file."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def summarize(rows: list[dict]) -> dict:
    """Summarize synchronized reduction costs by detector and displacement."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("phase") == "detector" and row["trial"] > 0:
            groups[f'{row["case"]}-{row["step"]}'].append(row["gpu_ms"])
    return {
        key: {"n": len(values), "p50_gpu_ms": statistics.median(values),
              "p95_gpu_ms": sorted(values)[int((len(values) - 1) * 0.95)],
              "max_gpu_ms": max(values)}
        for key, values in sorted(groups.items())
    }


def main() -> None:
    """Check complete journey coverage before reporting any speed ratio."""
    baseline = records(sys.argv[1])
    candidate = records(sys.argv[2])
    def fingerprints(rows):
        return {(row["source_identity"], row["case"], row["step"], row["trial"]):
                row["sha256_u32_le"] for row in rows if row.get("phase") == "detector"}
    before, after = fingerprints(baseline), fingerprints(candidate)
    missing = len(before.keys() - after.keys())
    added = len(after.keys() - before.keys())
    different = sum(before[key] != after[key] for key in before.keys() & after.keys())
    complete = all(any(row.get("phase") == "complete" for row in run)
                   for run in (baseline, candidate))
    passed = complete and bool(before) and not (missing or added or different)
    timing_before, timing_after = summarize(baseline), summarize(candidate)
    report = {"exact_hash_parity": passed, "compared": len(before),
              "missing": missing, "added": added, "different": different,
              "baseline": timing_before, "candidate": timing_after,
              "speed_ratios": {key: timing_before[key]["p50_gpu_ms"] /
                               timing_after[key]["p50_gpu_ms"]
                               for key in timing_before.keys() & timing_after.keys()},
              "resident_bytes_before": [row["resident_bytes"] for row in baseline
                                        if row.get("phase") == "resident"],
              "resident_bytes_after": [row["resident_bytes"] for row in candidate
                                       if row.get("phase") == "resident"],
              "ui_fps_claim": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Incomplete journey or exact detector hash mismatch")


if __name__ == "__main__":
    main()
