#!/usr/bin/env python3
"""Summarize paired physical-device quantem.gpu Vulkan benchmark trials."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for sorted scalar values."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _label(record: dict[str, Any]) -> str:
    if (
        record["shard_scan_rows"] == 4
        and record["staging_ring_depth"] == 1
        and record["wait_after_each_shard"]
    ):
        return "baseline_serial"
    if (
        record["shard_scan_rows"] == 2
        and record["staging_ring_depth"] == 2
        and not record["wait_after_each_shard"]
    ):
        return "candidate_overlapped"
    raise ValueError(f"unrecognized benchmark configuration: {record}")


def _statistics(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(record[key]) for record in records]
    return {
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
        "minimum": min(values),
    }


def summarize(input_path: Path) -> dict[str, Any]:
    """Validate and summarize exactly seven trials per A/B configuration."""
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict[str, Any]]] = {
        "baseline_serial": [],
        "candidate_overlapped": [],
    }
    sequence: list[str] = []
    for record in records:
        if record.get("schema") != "quantem.gpu.android-vulkan-benchmark-trial/v1":
            raise ValueError("input contains a non-benchmark schema")
        if record.get("status") != "PASS" or record.get("mismatch_count") != 0:
            raise ValueError("input contains a failing or mismatched trial")
        label = _label(record)
        groups[label].append(record)
        sequence.append(label)
    if any(len(group) != 7 for group in groups.values()):
        raise ValueError("expected exactly seven baseline and seven candidate trials")

    metric_keys = [
        "first_correct_product_ms",
        "full_exact_completion_ms",
        "dpc_and_idpc_ms",
        "package_ready_ms",
        "source_staging_ms",
        "gpu_scan_products_ms",
        "gpu_mean_diffraction_ms",
        "effective_source_gbps",
        "user_cpu_ms",
        "system_cpu_ms",
        "maximum_resident_set_kibibytes",
        "minor_page_fault_count",
        "major_page_fault_count",
    ]
    summary: dict[str, Any] = {
        "schema": "quantem.gpu.android-vulkan-benchmark-summary/v1",
        "status": "PASS",
        "trial_count": len(records),
        "sequence": sequence,
        "groups": {},
    }
    for label, group in groups.items():
        summary["groups"][label] = {
            "configuration": {
                "shard_scan_rows": group[0]["shard_scan_rows"],
                "staging_ring_depth": group[0]["staging_ring_depth"],
                "wait_after_each_shard": group[0]["wait_after_each_shard"],
            },
            "trial_count": len(group),
            "metrics": {key: _statistics(group, key) for key in metric_keys},
            "raw_trials": group,
        }
    baseline = summary["groups"]["baseline_serial"]["metrics"]
    candidate = summary["groups"]["candidate_overlapped"]["metrics"]
    summary["candidate_change"] = {
        "full_exact_p50_speedup": (
            baseline["full_exact_completion_ms"]["p50"]
            / candidate["full_exact_completion_ms"]["p50"]
        ),
        "first_correct_p50_speedup": (
            baseline["first_correct_product_ms"]["p50"]
            / candidate["first_correct_product_ms"]["p50"]
        ),
        "maximum_resident_set_p50_ratio": (
            candidate["maximum_resident_set_kibibytes"]["p50"]
            / baseline["maximum_resident_set_kibibytes"]["p50"]
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    summary = summarize(arguments.input)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
