"""Summarize measured repetitions; warmups remain in the raw record."""

import json
from pathlib import Path
import statistics
import sys


def distribution(values):
    """Return descriptive timings, not a claim of displayed frame rate."""
    ordered = sorted(values)
    return {"n": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999) - 1)],
            "min_ms": min(values), "max_ms": max(values)}


def main():
    records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    assert all(record["response"]["exact"] for record in records)
    records = [record for record in records if record["repetition"] >= 0]
    arms = sorted({record["arm"] for record in records})
    control = {record["repetition"]: record for record in records
               if record["arm"] == "reuse-control"}

    def wall(record, stage):
        return next(sample["all_seven_wall_ms"] for sample in record["response"]["samples"]
                    if sample["source"] == 0 and sample["stage"] == stage)

    for arm in arms:
        selected = [record for record in records if record["arm"] == arm]
        assert len(selected) == 20, (arm, len(selected))
        summary = {"arm": arm, "exact_source_transitions": 7 * len(selected),
                   "resident_bytes": sorted({r["response"]["resident_bytes"] for r in selected})}
        if arm != "allocation-control":
            assert all(sample["allocated_scratch_bytes"] == 0 and sample["reused_scratch"]
                       for r in selected for sample in r["response"]["samples"])
        summary["prepared_scratch_bytes_all_sources"] = sum(
            sample["prepared_scratch_bytes"] for sample in selected[0]["response"]["samples"]
            if sample["stage"] == "combined")
        for stage in ["index", "residual", "combined"]:
            ratios = [wall(record, stage) / wall(control[record["repetition"]], stage)
                      for record in selected]
            summary[stage] = distribution([wall(record, stage) for record in selected])
            summary[stage]["median_ratio_to_round_reuse_control"] = statistics.median(ratios)
            summary[stage]["first_half_ratio"] = statistics.median(ratios[:10])
            summary[stage]["second_half_ratio"] = statistics.median(ratios[10:])
            summary[stage]["per_source_gpu_ms"] = {
                str(source): distribution([
                    sample["gpu_ms"] for record in selected
                    for sample in record["response"]["samples"]
                    if sample["stage"] == stage and sample["source"] == source])
                for source in range(7)}
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
