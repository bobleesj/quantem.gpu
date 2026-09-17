#!/usr/bin/env python3
"""MPS/MLX trial-budget table from `mps-fit-arm.jsonl` records.

  mps_report.py <jsonl> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def ulp(value: float) -> float:
    return float(np.spacing(np.float32(value)))


def main(argv: list[str]) -> int:
    records = []
    for path in argv[1:]:
        records.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    seeds = sorted({int(r["seed"]) for r in records})
    print("| arm | TPE trials | objective evals | NM evals | loss | NM gain (ulp) | p50 ms | wall s | peak GB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seed in seeds:
        for arm in ("tpe50", "tpe200", "nmWarm"):
            matches = [r for r in records if int(r["seed"]) == seed and r["label"].endswith(arm)]
            if not matches:
                continue
            # A repeated label is a determinism control: report every repeat's
            # loss, and use the trace-bearing record for the NM-gain column.
            losses = {round(r["stages"]["fit"]["loss"], 12) for r in matches}
            if len(matches) > 1:
                print(
                    f"<!-- determinism control: {arm} seed {seed} recorded "
                    f"{len(matches)}x, distinct losses {sorted(losses)} -->"
                )
            record = next((r for r in matches if r["stages"]["fit"].get("trace")), matches[0])
            fit = record["stages"]["fit"]
            calls = fit["objective_calls"]
            trace = fit.get("trace") or []
            gain = ""
            if trace:
                tpe_count = int(record["trials"])
                tpe_losses = [row["loss"] for row in trace[1 : 1 + tpe_count]]
                if tpe_losses:
                    gain = f"{(min(tpe_losses) - fit['loss']) / ulp(fit['loss']):.0f}"
            print(
                f"| seed {seed} {arm} | {record['trials']} | {calls['n'] + 1} | "
                f"{fit['refine_nfev']} | {fit['loss']:.17g} | {gain} | "
                f"{(calls['p50_seconds'] or 0) * 1000:.1f} | {fit['wall_seconds']:.2f} | "
                f"{fit['peak_active_bytes'] / 1e9:.2f} |"
            )
    print("\n50 vs 200 trials (MPS):\n")
    for seed in seeds:
        a = next((r for r in records if int(r["seed"]) == seed and r["label"].endswith("tpe50")), None)
        b = next((r for r in records if int(r["seed"]) == seed and r["label"].endswith("tpe200")), None)
        if not a or not b:
            continue
        la, lb = a["stages"]["fit"]["loss"], b["stages"]["fit"]["loss"]
        print(
            f"- seed {seed}: loss(50)={la:.17g} loss(200)={lb:.17g} "
            f"delta={lb - la:.6g} ({(lb - la) / ulp(la):.0f} ulp, "
            f"relative {(lb - la) / lb:.3g}); "
            f"wall {a['stages']['fit']['wall_seconds']:.1f}s -> {b['stages']['fit']['wall_seconds']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
