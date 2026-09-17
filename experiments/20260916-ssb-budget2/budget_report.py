#!/usr/bin/env python3
"""Assemble the trial-budget cross-check tables (markdown) from the runs.

  budget_report.py <runs-dir> [<runs-dir> ...]

Reads `metal-budget-*/fit-budget.json` produced by ssb-fit-budget and prints:
  1. per-arm table with the NM gain in float32 ulp of the loss value,
  2. the 50-vs-200 comparison per basis and seed,
  3. the per-seed verdict lines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def ulp(value: float) -> float:
    return float(np.spacing(np.float32(value)))


def load_runs(paths: list[str]) -> list[dict]:
    runs = []
    for path in paths:
        runs.extend(json.loads(p.read_text()) for p in sorted(Path(path).glob("metal-budget-*/fit-budget.json")))
    return runs


def arm_of(report: dict, name: str, passing: str) -> dict | None:
    for arm in report["arms"]:
        if arm["arm"] == name and arm["pass"] == passing:
            return arm
    return None


def tpe_best(arm: dict) -> float | None:
    losses = [row["loss"] for row in arm["trials"] if row["stage"] == "tpe"]
    return min(losses) if losses else None


def main(argv: list[str]) -> int:
    runs = load_runs(argv[1:])
    if not runs:
        print("no runs found", file=sys.stderr)
        return 2
    basis_label = {
        "arina-512-full-disk": "A historical 2464-active",
        "arina-512-aperture-matched-8937": "B matched 8937-active (0.0x file)",
        "arina-512-17p0x-production-8937": "C production 8937-active (-17.0x file)",
    }
    order = sorted(runs, key=lambda r: (r["caseName"], r["seed"]))
    print("## Per-arm results (forward pass; reverse is bit-identical)\n")
    print(
        "| basis | seed | arm | TPE trials | evals | NM evals | loss | NM gain (ulp) | ms/eval | load |"
    )
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for report in order:
        basis = basis_label.get(report["caseName"], report["caseName"])
        for name in ("tpe25", "tpe50", "tpe100", "tpe200", "nmWarm"):
            arm = arm_of(report, name, "forward")
            if arm is None:
                continue
            best = tpe_best(arm)
            gain = "" if best is None else f"{(best - arm['bestLoss']) / ulp(best):.0f}"
            ms = 1000.0 * arm["elapsedSeconds"] / max(1, arm["totalEvaluations"])
            print(
                f"| {basis} | {report['seed']} | {name} | {arm['globalTrials']} | "
                f"{arm['totalEvaluations']} | {arm['refinementEvaluations']} | "
                f"{arm['bestLoss']:.17g} | {gain} | {ms:.1f} | "
                f"{arm['loadAverageBefore']:.2f}->{arm['loadAverageAfter']:.2f} |"
            )

    print("\n## 50 trials vs 200 trials\n")
    print(
        "| basis | seed | loss(50) | loss(200) | delta(200-50) | delta (ulp) | point identical | optima |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    verdicts = []
    for report in order:
        basis = basis_label.get(report["caseName"], report["caseName"])
        a50, a200 = arm_of(report, "tpe50", "forward"), arm_of(report, "tpe200", "forward")
        r50, r200 = arm_of(report, "tpe50", "reverse"), arm_of(report, "tpe200", "reverse")
        if not a50 or not a200:
            continue
        delta = a200["bestLoss"] - a50["bestLoss"]
        same_point = (
            a50["bestC10Nanometers"] == a200["bestC10Nanometers"]
            and a50["bestC12Nanometers"] == a200["bestC12Nanometers"]
            and a50["bestPhi12Radians"] == a200["bestPhi12Radians"]
        )
        same_rev = (
            r50["bestLoss"] == r200["bestLoss"]
            and r50["bestC10Nanometers"] == r200["bestC10Nanometers"]
            and r50["bestC12Nanometers"] == r200["bestC12Nanometers"]
            and r50["bestPhi12Radians"] == r200["bestPhi12Radians"]
        )
        optima = (
            f"50: ({a50['bestC10Nanometers']:.4g}, {a50['bestC12Nanometers']:.4g}, "
            f"{a50['bestPhi12Radians']:.4g}) / 200: ({a200['bestC10Nanometers']:.4g}, "
            f"{a200['bestC12Nanometers']:.4g}, {a200['bestPhi12Radians']:.4g})"
        )
        print(
            f"| {basis} | {report['seed']} | {a50['bestLoss']:.17g} | "
            f"{a200['bestLoss']:.17g} | {delta:.6g} | {delta / ulp(a50['bestLoss']):.1f} | "
            f"{'yes' if same_point else 'no'} (rev {'yes' if same_rev else 'no'}) | {optima} |"
        )
        verdicts.append(
            {
                "basis": basis,
                "seed": report["seed"],
                "loss_identical": delta == 0.0,
                "point_identical": same_point,
                "delta_ulp": delta / ulp(a50["bestLoss"]),
                "seconds50": a50["elapsedSeconds"],
                "seconds200": a200["elapsedSeconds"],
            }
        )

    print("\n## Verdict per basis/seed\n")
    for verdict in verdicts:
        print(
            f"- basis {verdict['basis']} seed {verdict['seed']}: "
            f"loss {'identical' if verdict['loss_identical'] else 'differs'} "
            f"({verdict['delta_ulp']:.1f} ulp), point "
            f"{'identical' if verdict['point_identical'] else 'differs'}, "
            f"50 trials {verdict['seconds50']:.1f}s vs 200 trials {verdict['seconds200']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
