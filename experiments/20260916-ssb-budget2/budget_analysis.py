#!/usr/bin/env python3
"""Compare and summarize ssb-fit-budget reports (trial-budget cross-check).

  budget_analysis.py compare <reference.json> <candidate.json> [...]
  budget_analysis.py summarize <report.json> [...]        # per-arm table
  budget_analysis.py budget <report.json> [...]           # 50 vs 200 verdict

Trial traces are compared row by row including the exact double values, so
"bit-identical" here means the same IEEE-754 doubles, not a rounded print.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARM_ORDER = ["tpe25", "tpe50", "tpe100", "tpe200", "nmWarm"]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def arm_key(arm: dict) -> tuple[str, str]:
    return (arm["arm"], arm["pass"])


def trace_equal(a: dict, b: dict) -> tuple[int, int]:
    """Return (rows compared, first differing row or -1)."""
    ta, tb = a["trials"], b["trials"]
    if len(ta) != len(tb):
        return (min(len(ta), len(tb)), 0)
    for index, (x, y) in enumerate(zip(ta, tb)):
        if (
            x["stage"] != y["stage"]
            or x["c10Nanometers"] != y["c10Nanometers"]
            or x["c12Nanometers"] != y["c12Nanometers"]
            or x["phi12Radians"] != y["phi12Radians"]
            or x["loss"] != y["loss"]
        ):
            return (len(ta), index)
    return (len(ta), -1)


def compare(reference: dict, candidate: dict) -> bool:
    print(
        f"reference {reference['caseName']} seed {reference['seed']} "
        f"bf {reference['logicalBrightfieldCount']}/{reference['activeBrightfieldCount']}"
    )
    print(
        f"candidate {candidate['caseName']} seed {candidate['seed']} "
        f"bf {candidate['logicalBrightfieldCount']}/{candidate['activeBrightfieldCount']}"
    )
    ref = {arm_key(arm): arm for arm in reference["arms"]}
    cand = {arm_key(arm): arm for arm in candidate["arms"]}
    ok = True
    print(
        f"{'arm':<8}{'pass':<9}{'rows':>6}{'first diff':>11}  "
        f"{'loss ref':>22}{'loss cand':>22}  {'point':>7}"
    )
    for (name, passing), arm_ref in ref.items():
        arm_cand = cand.get((name, passing))
        if arm_cand is None:
            print(f"{name:<8}{passing:<9}{'-':>6}{'missing':>11}")
            ok = False
            continue
        rows, first = trace_equal(arm_ref, arm_cand)
        point_same = (
            arm_ref["bestC10Nanometers"] == arm_cand["bestC10Nanometers"]
            and arm_ref["bestC12Nanometers"] == arm_cand["bestC12Nanometers"]
            and arm_ref["bestPhi12Radians"] == arm_cand["bestPhi12Radians"]
        )
        loss_same = arm_ref["bestLoss"] == arm_cand["bestLoss"]
        exact = first == -1 and point_same and loss_same and rows == len(arm_ref["trials"])
        ok = ok and exact
        print(
            f"{name:<8}{passing:<9}{rows:>6}{first:>11}  "
            f"{arm_ref['bestLoss']:>22.17g}{arm_cand['bestLoss']:>22.17g}  "
            f"{'same' if exact else 'DIFF':>7}"
        )
    print(f"VERDICT: {'bit-identical' if ok else 'DIFFERS'}")
    return ok


def per_eval_ms(arm: dict) -> float:
    evals = max(1, int(arm["totalEvaluations"]))
    return 1000.0 * float(arm["elapsedSeconds"]) / evals


def summarize(report: dict) -> None:
    print(
        f"# {report['caseName']}  seed {report['seed']}  "
        f"scan {report['scanSide']}  BF {report['logicalBrightfieldCount']} "
        f"(active {report['activeBrightfieldCount']})  "
        f"load {report['loadAverageAtStart']:.2f}->{report['loadAverageAtEnd']:.2f}"
    )
    print(
        f"{'arm':<8}{'pass':<9}{'trials':>7}{'evals':>7}{'NM evals':>9}"
        f"{'best loss':>22}{'ms/eval':>9}  "
        f"{'C10':>21}{'C12':>21}{'phi12':>21}"
    )
    for arm in report["arms"]:
        print(
            f"{arm['arm']:<8}{arm['pass']:<9}{arm['globalTrials']:>7}"
            f"{arm['totalEvaluations']:>7}{arm['refinementEvaluations']:>9}"
            f"{arm['bestLoss']:>22.17g}{per_eval_ms(arm):>9.2f}  "
            f"{arm['bestC10Nanometers']:>21.15g}{arm['bestC12Nanometers']:>21.15g}"
            f"{arm['bestPhi12Radians']:>21.15g}"
        )


def budget(report: dict) -> dict:
    """Report whether 50 trials and 200 trials reach the same optimum."""
    result: dict = {"caseName": report["caseName"], "seed": report["seed"], "arms": {}}
    for passing in ("forward", "reverse"):
        arms = {a["arm"]: a for a in report["arms"] if a["pass"] == passing}
        if not {"tpe50", "tpe200"} <= arms.keys():
            continue
        a50, a200 = arms["tpe50"], arms["tpe200"]
        same_loss = a50["bestLoss"] == a200["bestLoss"]
        same_point = (
            a50["bestC10Nanometers"] == a200["bestC10Nanometers"]
            and a50["bestC12Nanometers"] == a200["bestC12Nanometers"]
            and a50["bestPhi12Radians"] == a200["bestPhi12Radians"]
        )
        result["arms"][passing] = {
            "loss50": a50["bestLoss"],
            "loss200": a200["bestLoss"],
            "loss_delta_200_minus_50": a200["bestLoss"] - a50["bestLoss"],
            "loss_identical": same_loss,
            "point_identical": same_point,
            "evals50": a50["totalEvaluations"],
            "evals200": a200["totalEvaluations"],
            "seconds50": a50["elapsedSeconds"],
            "seconds200": a200["elapsedSeconds"],
            "point50": [
                a50["bestC10Nanometers"],
                a50["bestC12Nanometers"],
                a50["bestPhi12Radians"],
            ],
            "point200": [
                a200["bestC10Nanometers"],
                a200["bestC12Nanometers"],
                a200["bestPhi12Radians"],
            ],
        }
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    command, paths = argv[1], argv[2:]
    if command == "compare":
        reference = load(paths[0])
        ok = True
        for path in paths[1:]:
            candidate = load(path)
            print(f"--- {path}")
            ok = compare(reference, candidate) and ok
        return 0 if ok else 1
    if command == "summarize":
        for path in paths:
            summarize(load(path))
            print()
        return 0
    if command == "budget":
        for path in paths:
            print(json.dumps(budget(load(path)), indent=2, sort_keys=True))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
