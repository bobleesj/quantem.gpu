"""Summarize retained ANS trials without confusing kernel time with visible FPS."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    samples = defaultdict(list)
    comparisons = 0
    independent = []
    modes = {}
    for row in records:
        if row.get("event") == "ans_opt_sample":
            comparisons += int(row["exact_vs_a1"])
            if row["source"] == 0:
                samples[row["mask"], row["arm"]].append(row["all_seven_wall_ms"])
        elif row.get("event") == "ans_opt_independent_parity":
            if row.get("status") == "passed":
                independent.append([row["source"], row["mask"]])
        elif row.get("event") == "ans_opt_mode_counts":
            key = row["mask"]
            counts = modes.setdefault(key, [0] * 256)
            for index, count in enumerate(row["counts_by_mode"]):
                counts[index] += count
    results = {}
    for (mask, arm), values in samples.items():
        ordered = sorted(values)
        results.setdefault(mask, {})[arm] = {
            "n": len(values),
            "p50_wall_ms": statistics.median(values),
            "p95_wall_ms": ordered[math.ceil(len(values) * 0.95) - 1],
            "max_wall_ms": max(values),
        }
    return {
        "scope": "seven-source detector return, not native presentation",
        "complete": any(row.get("event") == "ans_opt_complete" for row in records),
        "exact_full_map_comparisons": comparisons,
        "independent_full_map_checks": independent,
        "wall_by_mask_and_arm": results,
        "mode_counts_by_mask": modes,
        "caveat": "Mask names describe absolute geometry; deltas are from the preceding named mask.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.input), indent=2, sort_keys=True))
