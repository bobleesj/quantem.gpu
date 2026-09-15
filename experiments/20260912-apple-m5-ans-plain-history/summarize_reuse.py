"""Separate full seven-source reuse hits from freshly computed updates."""

import json
import math
import statistics
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        if record["event"] != "ans_resident_loop_result":
            continue
        groups = {"reused": [], "computed": [], "mixed": []}
        samples = record["samples"]
        assert len(samples) % 7 == 0
        for start in range(0, len(samples), 7):
            update = samples[start:start + 7]
            assert sorted(item["source"] for item in update) == list(range(7))
            assert len({item["mask"] for item in update}) == 1
            if update[0]["cycle"] == 0:
                continue
            hits = sum(item["history_hit"] for item in update)
            kind = "reused" if hits == 7 else "computed" if hits == 0 else "mixed"
            groups[kind].append(update[0]["all_seven_wall_ms"])
        result = {}
        for kind, times in groups.items():
            if times:
                ordered = sorted(times)
                result[kind] = {
                    "updates": len(times),
                    "median_ms": statistics.median(times),
                    "p95_ms": ordered[math.ceil(0.95 * len(times)) - 1],
                    "max_ms": max(times),
                }
        print(json.dumps({
            "sequence": record["sequence"],
            "configuration": record["configuration"],
            "warm_seven_source_updates": result,
        }, sort_keys=True))
