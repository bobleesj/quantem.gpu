"""Summarize five alternating arms, excluding each arm's first repeat."""
import json
import statistics
import sys

records = [json.loads(line) for line in open(sys.argv[1])]
errors = [r for r in records if r.get("event") == "ans_resident_loop_error"]
assert not errors, errors
records = [r for r in records if r.get("event") == "stage_isolation"]
for first in range(0, len(records), 5):
    arm = records[first:first + 5]
    if len(arm) < 5:
        continue
    assert all(r["exact"] for r in arm)
    summary = {"arm": first // 5 + 1, "branchless_pop": arm[0]["branchless_pop"]}
    for stage in ["index", "residual", "combined"]:
        values = [s["all_seven_wall_ms"] for r in arm[1:] for s in r["samples"]
                  if s["stage"] == stage and s["source"] == 0]
        summary[stage] = {"median_ms": statistics.median(values),
                          "min_ms": min(values), "max_ms": max(values)}
    print(json.dumps(summary))
