#!/usr/bin/env python3
"""Summarize candidate-batch sweep JSONs: per-arm p50/min/p95 and speedup."""
import json
import statistics
import sys


def arm_stats(samples, arm):
    rows = [s for s in samples if s["arm"] == arm]
    if not rows:
        return None
    per_eval = [v for s in rows for v in s["perEvalSeconds"]]
    gpu_per_eval = []
    for s in rows:
        planes = sum(s["chunkSizes"])
        if planes:
            gpu_per_eval.append(sum(s["gpuSeconds"]) / planes)
    vals = sorted(1000 * v for v in per_eval)
    n = len(vals)
    gpu = sorted(1000 * v for v in gpu_per_eval)
    return {
        "n": n,
        "p50": statistics.median(vals),
        "min": vals[0],
        "p95": vals[min(n - 1, int(round(0.95 * (n - 1))))],
        "max": vals[-1],
        "gpu_p50": statistics.median(gpu) if gpu else float("nan"),
        "loads": [round(sum(s["loadAverage"]) / 3, 2) for s in rows],
    }


def main(paths):
    print(
        f"{'file':<34} {'k':>2} {'order':<14} {'bits':<5} "
        f"{'seq p50':>8} {'bat p50':>8} {'spd':>5} {'seq gpu':>8} {'bat gpu':>8}"
    )
    for path in sorted(paths):
        d = json.load(open(path))
        seq = arm_stats(d["samples"], "sequential")
        bat = arm_stats(d["samples"], "batched")
        print(
            f"{path.split('/')[-1]:<34} {d['k']:>2} {d['order']:<14} "
            f"{str(d['bit_identical']):<5} {seq['p50']:>8.1f} {bat['p50']:>8.1f} "
            f"{seq['p50'] / bat['p50']:>5.3f} {seq['gpu_p50']:>8.1f} {bat['gpu_p50']:>8.1f}"
        )
        print(
            f"    seq min={seq['min']:.1f} p95={seq['p95']:.1f} max={seq['max']:.1f} "
            f"n={seq['n']} loads={seq['loads']}"
        )
        print(
            f"    bat min={bat['min']:.1f} p95={bat['p95']:.1f} max={bat['max']:.1f} "
            f"n={bat['n']} loads={bat['loads']} stats={d.get('batch_stats')}"
        )
        if not d["bit_identical"]:
            print(f"    MISMATCHES: {d['mismatches'][:4]}")


if __name__ == "__main__":
    main(sys.argv[1:])
