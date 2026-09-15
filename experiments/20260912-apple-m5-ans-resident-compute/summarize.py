"""Summarize resident-loop synchronized compute timings, never presented FPS."""
import json
import statistics
import sys
from collections import defaultdict

for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    if record["event"] != "ans_resident_loop_result":
        print(json.dumps(record))
        continue
    timings = defaultdict(list)
    for sample in record["samples"]:
        if sample["source"] == 0 and sample["cycle"] > 0:
            timings[sample["mask"]].append(sample["all_seven_wall_ms"])
    print(json.dumps({
        "sequence": record["sequence"], "arm": record["arm"],
        "configuration": record["configuration"],
        "parity": record["fullmap_parity"],
        "resident_bytes": record.get("series_resident_bytes"),
        "allocated_bytes": record.get("metal_current_allocated_bytes"),
        "warm_median_ms": {name: round(statistics.median(times), 3)
                           for name, times in timings.items()},
    }, sort_keys=True))
