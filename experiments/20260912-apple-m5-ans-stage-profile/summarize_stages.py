"""Report overlapping seven-source CPU/GPU intervals without adding overlaps."""

import json
import re
import statistics
import sys
from collections import defaultdict


def _union_ms(spans: list[tuple[float, float]]) -> float:
    spans = sorted((start, end) for start, end in spans if 0 < start < end)
    if not spans:
        return 0.0
    start, end = spans[0]
    elapsed = 0.0
    for next_start, next_end in spans[1:]:
        if next_start > end:
            elapsed += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return (elapsed + end - start) * 1000


def _intervals(
    profiles: list[dict[str, float]], start_key: str, end_key: str
) -> list[tuple[float, float]]:
    return [(p.get(start_key, 0), p.get(end_key, 0)) for p in profiles]


def _stage_intervals(
    profiles: list[dict[str, float]], stage: str
) -> list[tuple[float, float]]:
    spans = []
    pattern = re.compile(rf"gpu_{stage}_\d+_start_seconds")
    for profile in profiles:
        for key, start in profile.items():
            if pattern.fullmatch(key):
                spans.append((start, profile[key.replace("_start_", "_end_")]))
    return spans


for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    if record.get("event") != "ans_resident_loop_result":
        continue
    grouped = defaultdict(lambda: defaultdict(list))
    samples = record["samples"]
    assert len(samples) % 7 == 0
    for offset in range(0, len(samples), 7):
        update = samples[offset:offset + 7]
        assert {s["source"] for s in update} == set(range(7))
        assert len({(s["cycle"], s["mask"]) for s in update}) == 1
        if update[0]["cycle"] == 0:
            continue
        metrics = grouped[update[0]["mask"]]
        metrics["wall_ms"].append(update[0]["all_seven_wall_ms"])
        profiles = [s.get("update_profile", {}) for s in update]
        if not all(profiles):
            continue
        metrics["valid_stage_sources"].append(sum(
            p.get("stage_profile_valid", 0) == 1 for p in profiles))
        metrics["cpu_planning_union_ms"].append(_union_ms(_intervals(
            profiles, "cpu_planning_start_seconds", "cpu_planning_end_seconds")))
        metrics["cpu_preparation_union_ms"].append(_union_ms(_intervals(
            profiles, "cpu_preparation_start_seconds", "cpu_preparation_end_seconds")))
        metrics["gpu_command_union_ms"].append(_union_ms(_intervals(
            profiles, "command_gpu_start_seconds", "command_gpu_end_seconds")))
        metrics["readback_sum_ms"].append(sum(
            p.get("host_readback_milliseconds", 0) for p in profiles))
        metrics["diagnostic_resolve_sum_ms"].append(sum(
            p.get("host_profile_resolve_milliseconds", 0) for p in profiles))
        commits = [p["host_commit_seconds"] for p in profiles
                   if p.get("host_commit_seconds", 0) > 0]
        if commits:
            metrics["submission_span_ms"].append((max(commits) - min(commits)) * 1000)
        if all(p.get("stage_profile_valid") == 1 for p in profiles):
            polar = _stage_intervals(profiles, "polar")
            residual = _stage_intervals(profiles, "residual(?:_partials|_finish)?")
            metrics["gpu_polar_union_ms"].append(_union_ms(polar))
            metrics["gpu_residual_union_ms"].append(_union_ms(residual))
            metrics["gpu_stages_union_ms"].append(_union_ms(polar + residual))
        cache = update[0].get("polar_plan_cache_profile", {})
        for key, value in cache.items():
            metrics[f"plan_{key}"].append(value)
    print(json.dumps({
        "sequence": record["sequence"],
        "configuration": record["configuration"],
        "parity": record["fullmap_parity"],
        "resident_bytes": record["series_resident_bytes"],
        "masks": {
            mask: {key: {"median": statistics.median(values), "samples": len(values),
                         "min": min(values), "max": max(values)}
                   for key, values in metrics.items()}
            for mask, metrics in grouped.items()
        },
        "note": "Intervals overlap; medians and per-source sums are not additive wall-time parts.",
    }, sort_keys=True))
