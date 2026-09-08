"""Separate CPU encoding, driver scheduling, and GPU shader intervals."""

import json
import statistics
import sys
from pathlib import Path


records = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.startswith("{"):
        continue
    row = json.loads(line)
    if row.get("phase") == "detector_host_stages":
        records.append(row)
assert records, "No host-stage measurements"
steady = records[1:]
result = {
    "count": len(records),
    "first": records[0],
    "steady_median_wall_ms": statistics.median(r["wall_after_prepare_ms"] for r in steady),
    "steady_median_gpu_ms": statistics.median(r["gpu_interval_ms"] for r in steady),
    "first_driver_interval_ms": 1000 * (
        records[0]["command_host_times_seconds"]["kernel_end"]
        - records[0]["command_host_times_seconds"]["kernel_start"]),
    "interpretation": "kernel_start/end are CPU driver times, not Metal shader execution",
    "sampling_caveat": "One-second process sampling is diagnostic, not an uncontended performance run",
}
print(json.dumps(result, indent=2))
