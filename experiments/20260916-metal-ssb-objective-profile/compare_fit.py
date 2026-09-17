#!/usr/bin/env python3
"""Compare fit arms: stage split, trajectory divergence and per-trial drift."""
import json
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main(paths):
    arms = {name: load(path) for name, path in paths}
    baseline = arms.get("base")
    print(f"{'arm':>6} {'fit_s':>8} {'trials_s':>9} {'refine_s':>9} {'ref_evals':>9} "
          f"{'loss':>20} {'c10':>10} {'c12':>9} {'phi12':>9}")
    for name, arm in arms.items():
        best = arm["fit_best"]
        print(f"{name:>6} {arm['fit_seconds']:8.2f} {arm['trial_stage_seconds']:9.2f} "
              f"{arm['refinement_stage_seconds']:9.2f} {arm['refinement_evaluations']:9d} "
              f"{arm['fit_loss']:20.17g} {best[0]:10.5f} {best[1]:9.5f} {best[2]:9.5f}")
    if not baseline:
        return
    base_trials = baseline["trials"]
    for name, arm in arms.items():
        if name == "base":
            continue
        trials = arm["trials"]
        first = next(
            (i for i, (a, b) in enumerate(zip(base_trials, trials))
             if a["point"] != b["point"]), None)
        shared = sum(1 for a, b in zip(base_trials, trials) if a["point"] == b["point"])
        drift = max(
            (abs(a["loss"] - b["loss"]) for a, b in zip(base_trials, trials)
             if a["point"] == b["point"]), default=0.0)
        print(f"\n{name}: shared trial points {shared}/{len(trials)}, "
              f"first divergence index {first}, max loss drift on shared points {drift:.3e}")
        print(f"{name}: fit time {arm['fit_seconds']:.2f} s vs baseline "
              f"{baseline['fit_seconds']:.2f} s "
              f"({baseline['fit_seconds'] / arm['fit_seconds']:.3f}x)")
        print(f"{name}: loss {arm['fit_loss']:.17g} vs {baseline['fit_loss']:.17g} "
              f"(delta {arm['fit_loss'] - baseline['fit_loss']:.3e})")
        if "reeval" in arm:
            r = arm["reeval"]
            print(f"{name}: single-path re-evaluation of {len(r['rows'])} trials: "
                  f"max abs {r['max_abs']:.3e}, max rel {r['max_rel']:.3e}, "
                  f"bit mismatches {r['bit_mismatches']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(list(zip(args[0::2], args[1::2])))
