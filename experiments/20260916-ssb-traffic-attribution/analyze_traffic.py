#!/usr/bin/env python3
"""Per-pass cost attribution for one production SSB evaluation.

Reads probe_traffic report.json sessions and reports:
  1. arm-level p50/min/p95/max with host load range,
  2. within-rep paired deltas against the full arm (drift-cancelling),
  3. a least-squares slope per pass over the executed fraction (0, 0.5, 1),
  4. an additivity check of pass costs against the all-skipped floor,
  5. the pinned-loss gate result.

usage: analyze_traffic.py <session-dir> [<session-dir> ...]
"""
import json
import pathlib
import statistics
import sys

SAMPLE_GB = {
    "column_read": 9.421,
    "column_write": 9.421,
    "column_tables": 0.157,
    "row_read": 9.421,
    "nyquist": 0.146,
    "trig_write": 0.075,
}

ARM_FRACTION = {
    "full": {"column": 1.0, "nyquist": 1.0, "row": 1.0, "trig": 1.0},
    "col_half": {"column": 0.5, "nyquist": 1.0, "row": 1.0, "trig": 1.0},
    "col_none": {"column": 0.0, "nyquist": 1.0, "row": 1.0, "trig": 1.0},
    "nyq_half": {"column": 1.0, "nyquist": 0.5, "row": 1.0, "trig": 1.0},
    "nyq_none": {"column": 1.0, "nyquist": 0.0, "row": 1.0, "trig": 1.0},
    "row_half": {"column": 1.0, "nyquist": 1.0, "row": 0.5, "trig": 1.0},
    "row_none": {"column": 1.0, "nyquist": 1.0, "row": 0.0, "trig": 1.0},
    "trig_none": {"column": 1.0, "nyquist": 1.0, "row": 1.0, "trig": 0.0},
    "floor": {"column": 0.0, "nyquist": 1.0, "row": 0.0, "trig": 0.0},
}


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = q * (len(values) - 1)
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def load_sessions(dirs):
    samples = []
    gates = []
    for directory in dirs:
        found = sorted(pathlib.Path(directory).glob("report*.json"))
        if not found:
            print(f"missing report in {directory}")
            continue
        for path in found:
            report = json.loads(path.read_text())
            gates.append((path.stem, report.get("gate", {})))
            for sample in report["samples"]:
                sample["session"] = path.stem
                samples.append(sample)
    return samples, gates


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return 1
    samples, gates = load_sessions(dirs)
    ablation = [s for s in samples if s["phase"] == "ablation"]
    if not ablation:
        print("no ablation samples")
        return 1

    print("== loss gate (full path vs frozen pins) ==")
    for directory, gate in gates:
        failures = gate.get("failures", [])
        rows = gate.get("rows", [])
        status = "PASS" if not failures else f"FAIL {failures}"
        print(f"  {pathlib.Path(directory).name}: {status}")
        for row in rows:
            print("    c10=%-10s loss=%.17g expected=%.17g match=%s" % (
                row["c10"], row["loss"], row["expected_loss"], row["match"]))

    print("\n== arms (gpu_ms) ==")
    print("  %-10s %3s %8s %8s %8s %8s %7s %7s" % (
        "arm", "n", "p50", "min", "p95", "max", "ld1min", "ld1max"))
    by_arm = {}
    for arm in ARM_FRACTION:
        rows = [s for s in ablation if s["arm"] == arm]
        if not rows:
            continue
        gpu = [s["gpu_ms"] for s in rows]
        loads = [s["load1"] for s in rows]
        by_arm[arm] = gpu
        print("  %-10s %3d %8.1f %8.1f %8.1f %8.1f %7.2f %7.2f" % (
            arm, len(gpu), percentile(gpu, 0.5), min(gpu), percentile(gpu, 0.95),
            max(gpu), min(loads), max(loads)))

    print("\n== wall_ms and untracked_ms (p50) ==")
    for arm in ARM_FRACTION:
        rows = [s for s in ablation if s["arm"] == arm]
        if not rows:
            continue
        wall = [s["wall_ms"] for s in rows]
        untracked = [s["wall_ms"] - s["gpu_ms"] for s in rows]
        print("  %-10s wall %8.1f  untracked %7.1f" % (
            arm, percentile(wall, 0.5), percentile(untracked, 0.5)))

    print("\n== within-rep paired deltas vs full (drift-cancelling) ==")
    by_rep = {}
    for sample in ablation:
        by_rep.setdefault((sample["session"], sample["rep"]), {})[sample["arm"]] = sample["gpu_ms"]
    deltas = {}
    for key, arms in by_rep.items():
        if "full" not in arms:
            continue
        for arm, value in arms.items():
            deltas.setdefault(arm, []).append(value - arms["full"])
    print("  %-10s %3s %9s %9s %9s" % ("arm", "n", "mean", "median", "stdev"))
    for arm in ARM_FRACTION:
        values = deltas.get(arm)
        if not values:
            continue
        stdev = statistics.stdev(values) if len(values) > 1 else float("nan")
        print("  %-10s %3d %9.2f %9.2f %9.2f" % (
            arm, len(values), statistics.mean(values), statistics.median(values), stdev))

    print("\n== per-pass least-squares slope over executed fraction ==")
    print("  slope = cost of running the pass on every dispatch (one eval)")
    slopes = {}
    # Each pass is regressed only over arms where that pass varies and every
    # other pass is pinned at 1.0, so the slope is not contaminated.
    for pass_name in ("column", "nyquist", "row", "trig"):
        pairs = []
        for arm, spec in ARM_FRACTION.items():
            if any(spec[other] != 1.0 for other in spec if other != pass_name):
                continue
            rows = [s for s in ablation if s["arm"] == arm]
            if not rows:
                continue
            cost = percentile([s["gpu_ms"] for s in rows], 0.5)
            pairs.append((spec[pass_name], cost))
        if len(pairs) < 2:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            continue
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        slopes[pass_name] = slope
        print("  %-8s slope %8.2f ms   (n_arms=%d)" % (pass_name, slope, len(pairs)))

    print("\n== additivity check ==")
    full_rows = [s for s in ablation if s["arm"] == "full"]
    floor_rows = [s for s in ablation if s["arm"] == "floor"]
    if full_rows and floor_rows:
        full = percentile([s["gpu_ms"] for s in full_rows], 0.5)
        floor = percentile([s["gpu_ms"] for s in floor_rows], 0.5)
        total = full - floor
        summed = sum(slopes.get(k, 0.0) for k in ("column", "nyquist", "row", "trig"))
        print("  full %.1f ms, floor(all passes off) %.1f ms, executed work %.1f ms" % (
            full, floor, total))
        print("  sum of slopes %.1f ms, interaction %+.1f ms (%.0f%% of executed work)" % (
            summed, summed - total, 100 * (summed - total) / total if total else float("nan")))

    print("\n== implied bandwidth (slope vs measured ghost ceilings) ==")
    ceilings = {"stream_read": 143.4, "stream_write": 143.9}
    if "column" in slopes:
        column_bytes = SAMPLE_GB["column_read"] + SAMPLE_GB["column_write"]
        print("  column %.1f ms over %.3f GB = %.1f GB/s (stream read %.1f, write %.1f)" % (
            slopes["column"], column_bytes, column_bytes / (slopes["column"] / 1e3),
            ceilings["stream_read"], ceilings["stream_write"]))
    if "row" in slopes:
        print("  row    %.1f ms over %.3f GB = %.1f GB/s" % (
            slopes["row"], SAMPLE_GB["row_read"],
            SAMPLE_GB["row_read"] / (slopes["row"] / 1e3)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
