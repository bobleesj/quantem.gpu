"""Summarize checksum ablation separately from full-image performance."""

import json
import statistics
import sys
from collections import defaultdict


def summarize(path: str) -> dict:
    """Exclude first command warmup and retain all repeated controls."""
    grouped = defaultdict(lambda: defaultdict(list))
    records = []
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if record.get("event") == "ans_resident_loop_error":
            raise RuntimeError(record["error"])
        if record.get("event") == "ans_resident_loop_decode_checksum":
            assert record["all_parity"], "Checksum or repeated image mismatch"
            records.append(record)
    for record in records[1:]:
        for sample in record["samples"]:
            if sample["source"] != 0:
                continue
            metrics = grouped[sample["mask"]]
            for key in ["reference_all_seven_wall_ms", "all_seven_wall_ms",
                        "repeated_reference_all_seven_wall_ms"]:
                metrics[key].append(sample[key])
    return {
        "diagnostic_not_image": True,
        "commands": len(records),
        "packet_checksums_checked": sum(
            s["packet_count"] for r in records for s in r["samples"]),
        "repeated_full_images_checked": sum(len(r["samples"]) for r in records),
        "masks": {
            mask: {
                key: {"median": statistics.median(values), "min": min(values),
                      "max": max(values), "samples": len(values)}
                for key, values in metrics.items()
            }
            for mask, metrics in grouped.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2))
