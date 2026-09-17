#!/usr/bin/env python3
"""Summarize the Metal fit timeline captured by probe-timeline.swift.

Categories are computed from differences only, so the mixture of monotonic
wall seconds and MTLCommandBuffer GPU seconds never crosses epochs.
"""
import json
import sys
from pathlib import Path


def load(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def percentile(values, p):
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values):
    return {
        "count": len(values),
        "sum": sum(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else float("nan"),
    }


def main(path, out_path):
    evals = load(path)
    if not evals:
        raise SystemExit("no evaluations recorded")

    wall_start = evals[0]["wall_start"]
    wall_end = evals[-1]["wall_end"]
    span = wall_end - wall_start

    eval_wall = 0.0
    cb_commit_wait = 0.0
    gpu_busy = 0.0
    kernel_busy = 0.0
    sync_overhead = 0.0
    preamble = 0.0
    inter_cb_cpu = 0.0
    tail = 0.0
    inter_eval_gap = 0.0
    by_label = {}
    intra_gaps = []
    inter_gaps = []
    per_eval = []

    for position, record in enumerate(evals):
        buffers = record["command_buffers"]
        record_wall = record["wall_end"] - record["wall_start"]
        eval_wall += record_wall
        if position:
            inter_eval_gap += record["wall_start"] - evals[position - 1]["wall_end"]
        local_gpu = 0.0
        local_kernel = 0.0
        if buffers:
            preamble += buffers[0]["wall_start"] - record["wall_start"]
            tail += record["wall_end"] - buffers[-1]["wall_end"]
        for index, buffer in enumerate(buffers):
            cb_wall = buffer["wall_end"] - buffer["wall_start"]
            cb_gpu = buffer["gpu_end"] - buffer["gpu_start"]
            cb_kernel = buffer["kernel_end"] - buffer["kernel_start"]
            cb_commit_wait += cb_wall
            local_gpu += cb_gpu
            local_kernel += cb_kernel
            sync_overhead += max(cb_wall - cb_gpu, 0.0)
            entry = by_label.setdefault(
                buffer["label"], {"wall": 0.0, "gpu": 0.0, "kernel": 0.0, "count": 0}
            )
            entry["wall"] += cb_wall
            entry["gpu"] += cb_gpu
            entry["kernel"] += cb_kernel
            entry["count"] += 1
            if index:
                intra_gaps.append(buffer["gpu_start"] - buffers[index - 1]["gpu_end"])
                inter_cb_cpu += max(buffer["wall_start"] - buffers[index - 1]["wall_end"], 0.0)
        gpu_busy += local_gpu
        kernel_busy += local_kernel
        per_eval.append(
            {
                "index": record["index"],
                "wall": record_wall,
                "gpu": local_gpu,
                "kernel": local_kernel,
                "gpu_seconds": record["gpu_seconds"],
                "command_buffers": len(buffers),
            }
        )
        if position + 1 < len(evals):
            following = evals[position + 1]["command_buffers"]
            if buffers and following:
                inter_gaps.append(following[0]["gpu_start"] - buffers[-1]["gpu_end"])

    accounted = eval_wall + inter_eval_gap
    categories = {
        "gpu_busy_total": gpu_busy,
        "gpu_busy_cache_accumulate": by_label.get("cache", {}).get("gpu", 0.0),
        "gpu_busy_clear_tables": by_label.get("clear", {}).get("gpu", 0.0),
        "gpu_busy_streamed": by_label.get("streamed", {}).get("gpu", 0.0),
        "inside_eval_gpu_idle": max(eval_wall - gpu_busy, 0.0),
        "commit_wait_minus_gpu": sync_overhead,
        "cpu_encoding_inside_eval": preamble + inter_cb_cpu,
        "cpu_reduction_tail": tail,
        "optimizer_between_evals": inter_eval_gap,
        "unaccounted_fit_wall": max(span - accounted, 0.0),
    }
    kernel_gpu_ratio = kernel_busy / gpu_busy if gpu_busy else float("nan")

    report = {
        "evaluations": len(evals),
        "fit_span_seconds": span,
        "eval_wall_seconds": eval_wall,
        "inter_eval_gap_seconds": inter_eval_gap,
        "gpu_busy_seconds": gpu_busy,
        "gpu_busy_fraction_of_fit": gpu_busy / span if span else float("nan"),
        "kernel_busy_seconds": kernel_busy,
        "kernel_over_gpu_ratio": kernel_gpu_ratio,
        "command_buffers": {label: entry for label, entry in sorted(by_label.items())},
        "intra_eval_cb_gap": stats(intra_gaps),
        "inter_eval_gpu_idle": stats(inter_gaps),
        "categories_seconds": categories,
        "categories_fraction_of_fit": {
            key: (value / span if span else float("nan"))
            for key, value in categories.items()
        },
        "per_eval_wall": stats([row["wall"] for row in per_eval]),
        "per_eval_gpu": stats([row["gpu"] for row in per_eval]),
        "per_eval_command_buffers": stats([row["command_buffers"] for row in per_eval]),
        "evals": per_eval,
    }
    Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"evaluations: {report['evaluations']}")
    print(f"fit span:    {span:8.3f} s")
    print(f"gpu busy:    {gpu_busy:8.3f} s  ({report['gpu_busy_fraction_of_fit'] * 100:5.1f}% of fit span)")
    print(f"kernel busy: {kernel_busy:8.3f} s  (kernel/gpu {kernel_gpu_ratio:.3f})")
    print()
    print("category                                seconds    % of fit")
    for key, value in sorted(categories.items(), key=lambda item: -item[1]):
        print(f"{key:38s} {value:9.3f}   {value / span * 100:6.1f}%")
    print()
    print("command buffers by label:")
    for label, entry in sorted(by_label.items(), key=lambda item: -item[1]["gpu"]):
        print(
            f"  {label:10s} n={entry['count']:5d} gpu={entry['gpu']:8.3f} s "
            f"wall={entry['wall']:8.3f} s kernel={entry['kernel']:8.3f} s"
        )
    print()
    for name in ("intra_eval_cb_gap", "inter_eval_gpu_idle"):
        row = report[name]
        print(
            f"{name}: n={row['count']} p50={row['p50'] * 1000:8.3f} ms "
            f"p95={row['p95'] * 1000:8.3f} ms max={row['max'] * 1000:8.3f} ms"
        )
    for name in ("per_eval_wall", "per_eval_gpu", "per_eval_command_buffers"):
        row = report[name]
        print(
            f"{name}: p50={row['p50']:8.3f} p95={row['p95']:8.3f} max={row['max']:8.3f}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
