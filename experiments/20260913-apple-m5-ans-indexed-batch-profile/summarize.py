"""Summarize the profiled and unprofiled all-seven A/B/A update runs."""

import json
import statistics
import sys
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main():
    """Print compact medians and tail estimates without changing raw data."""
    path = Path(sys.argv[1])
    runs = json.loads(path.read_text(encoding="utf-8"))
    for run in runs:
        samples = run["target_samples"]
        wall = [sample["all_seven_wall_ms"] for sample in samples
                if sample["source"] == 0]
        row = {
            "phase": run["phase"], "arm": run["arm"], "batch": run["batch"],
            "parity": run["fullmap_parity"],
            "all_seven_wall_median_ms": statistics.median(wall),
            "all_seven_wall_p95_ms": percentile(wall, 0.95),
            "resident_bytes": run["resident_bytes"],
            "metal_current_allocated_bytes": run["metal_current_allocated_bytes"],
        }
        if run["profile_enabled"]:
            fields = (
                "cpu_planning_milliseconds",
                "cpu_preparation_milliseconds",
                "command_gpu_milliseconds",
                "host_readback_milliseconds",
            )
            for field in fields:
                values = [sample["update_profile"][field]
                          for sample in samples
                          if field in sample.get("update_profile", {})]
                if values:
                    row[field + "_median"] = statistics.median(values)
            waits = []
            for sample in samples:
                profile = sample.get("update_profile", {})
                start = profile.get("host_commit_seconds")
                end = profile.get("host_wait_completion_seconds")
                if start is not None and end is not None and end >= start:
                    waits.append((end - start) * 1000)
            if waits:
                row["commit_to_completion_median_ms"] = statistics.median(waits)
            valid = [sample["update_profile"].get("stage_profile_valid")
                     for sample in samples
                     if "stage_profile_valid" in sample.get("update_profile", {})]
            if valid:
                row["stage_profile_valid_fraction"] = sum(valid) / len(valid)
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
