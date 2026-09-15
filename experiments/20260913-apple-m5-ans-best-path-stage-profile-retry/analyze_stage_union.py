"""Summarize calibrated stage intervals across all seven source updates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SOURCES = 7
CYCLES = 20
TARGET_MASK = "adf-center-20"


def interval_union_ms(intervals: list[tuple[float, float]]) -> float:
    """Return the total duration covered by the union of time intervals."""
    intervals = sorted((start, end) for start, end in intervals if end > start)
    if not intervals:
        return 0.0

    covered = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    covered += end - start
    return covered * 1000.0


def nearest_rank(values: list[float], quantile: float) -> float:
    """Return the nearest-rank quantile of a non-empty sample."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summarize(path: Path) -> dict:
    """Validate and summarize the target mask's seven-source GPU intervals."""
    document = json.loads(path.read_text())
    candidate = next(
        arm for arm in document["arm_samples"] if arm["arm"] == "candidate"
    )
    target = [
        sample for sample in candidate["samples"]
        if sample["mask"] == TARGET_MASK
    ]

    cycles = sorted({sample["cycle"] for sample in target})
    if len(cycles) != CYCLES:
        raise ValueError(f"expected {CYCLES} measured cycles, got {len(cycles)}")

    per_cycle = []
    for cycle in cycles:
        rows = [sample for sample in target if sample["cycle"] == cycle]
        sources = {sample["source"] for sample in rows}
        if len(rows) != SOURCES or sources != set(range(SOURCES)):
            raise ValueError(f"cycle {cycle}: expected one record for each of 7 sources")

        profiles = [sample.get("update_profile") for sample in rows]
        if any(not profile or profile.get("stage_profile_valid") != 1 for profile in profiles):
            raise ValueError(f"cycle {cycle}: one or more source stage profiles are invalid")

        polar = [
            (profile["gpu_polar_0_start_seconds"], profile["gpu_polar_0_end_seconds"])
            for profile in profiles
        ]
        residual = [
            (profile["gpu_residual_1_start_seconds"], profile["gpu_residual_1_end_seconds"])
            for profile in profiles
        ]
        per_cycle.append({
            "cycle": cycle,
            "polar_union_ms": interval_union_ms(polar),
            "residual_union_ms": interval_union_ms(residual),
            "combined_union_ms": interval_union_ms(polar + residual),
            "command_span_ms": (
                max(profile["command_gpu_end_seconds"] for profile in profiles)
                - min(profile["command_gpu_start_seconds"] for profile in profiles)
            ) * 1000.0,
        })

    result = {"target_mask": TARGET_MASK, "cycles": per_cycle}
    for key in (
        "polar_union_ms", "residual_union_ms", "combined_union_ms", "command_span_ms"
    ):
        values = [cycle[key] for cycle in per_cycle]
        result[key] = {
            "p50": statistics.median(values),
            "p95_nearest_rank": nearest_rank(values, 0.95),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "results/summary.json",
        help="runner summary.json",
    )
    arguments = parser.parse_args()
    print(json.dumps(summarize(arguments.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
