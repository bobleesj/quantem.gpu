"""Report the native SSB fit trajectory against its pin and the Python reference.

Numbers first: the unbatched (production) trajectory, the batched-pair
trajectory, what moved, and whether a frozen pin still reproduces. A mismatch is
a finding: this script never rewrites the pin, it prints what changed. The only
way this file writes a pin is ``--record-pin`` on a tree that has no pin yet;
recapturing an existing pin needs ``--recapture`` and a ``--reason``.

Usage:
    python scripts/ssb_fit_trajectory_report.py \
        --native parity-runs/fit-128.json \
        [--repeat parity-runs/fit-128-repeat.json] \
        [--reference parity-runs/fit-reference-128.json] \
        [--pin tests/parity/fixtures/ssb_fit_trajectory_128.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def float32_key(point: dict) -> str:
    return "-".join(
        str(struct.unpack("<I", struct.pack("<f", float(point[key])))[0])
        for key in ("c10Nanometers", "c12Nanometers", "phi12Radians")
    )


def trial_digest(trials: list[dict]) -> str:
    digest = hashlib.sha256()
    for trial in trials:
        digest.update(float32_key(trial).encode())
        digest.update(struct.pack("<d", float(trial["loss"])))
        digest.update(str(trial["stage"]).encode())
    return digest.hexdigest()


def optimum(run: dict) -> tuple[float, float, float]:
    return (
        float(run["bestC10Nanometers"]),
        float(run["bestC12Nanometers"]),
        float(run["bestPhi12Radians"]),
    )


def production_checks_pass(native: dict, repeat: dict | None = None) -> bool:
    """Require deterministic production evaluation without CPU reconstruction."""
    purity_fields = (
        "repeatSameEngineBitwiseMismatches",
        "freshEngineBitwiseMismatches",
        "float32AliasBitwiseMismatches",
    )
    for report in (native,) if repeat is None else (native, repeat):
        if not report["productionOptimizeAlwaysMatchesSequential"]:
            return False
        if any(report["objectivePurity"][field] != 0 for field in purity_fields):
            return False
    if repeat is not None:
        first, second = native["sequential"], repeat["sequential"]
        return (
            trial_digest(first["trials"]) == trial_digest(second["trials"])
            and optimum(first) == optimum(second)
            and first["bestLoss"] == second["bestLoss"]
        )
    return True


def _format_optimum(run: dict) -> str:
    c10, c12, phi12 = optimum(run)
    return f"{c10:<19.12g} {c12:<19.12g} {phi12:<19.12g} {run['bestLoss']:.12g}"


def _trajectory_comparison(left: dict, right: dict) -> dict:
    """Compare two native runs index by index without a shared artifact report."""

    left_trials = left["trials"]
    right_trials = right["trials"]
    shared = min(len(left_trials), len(right_trials))
    different_points = 0
    different_losses = 0
    max_loss_delta = 0.0
    left_best = left_trials[0]["loss"]
    right_best = right_trials[0]["loss"]
    best_so_far_different = 0
    for index in range(shared):
        one = left_trials[index]
        two = right_trials[index]
        if float32_key(one) != float32_key(two):
            different_points += 1
        if one["loss"] != two["loss"]:
            different_losses += 1
            max_loss_delta = max(max_loss_delta, abs(one["loss"] - two["loss"]))
        left_best = min(left_best, one["loss"])
        right_best = min(right_best, two["loss"])
        if left_best != right_best:
            best_so_far_different += 1
    return {
        "trialsCompared": shared,
        "trialsWithDifferentFloat32Point": different_points,
        "trialsWithDifferentLossAtSameIndex": different_losses,
        "maxLossDifferenceAtSameIndex": max_loss_delta,
        "bestSoFarDifferentCount": best_so_far_different,
        "finalBestPointDeltaC10": optimum(right)[0] - optimum(left)[0],
        "finalBestPointDeltaC12": optimum(right)[1] - optimum(left)[1],
        "finalBestPointDeltaPhi12": optimum(right)[2] - optimum(left)[2],
        "finalBestLossDelta": float(right["bestLoss"]) - float(left["bestLoss"]),
    }


def _print_comparison(label: str, entry: dict) -> None:
    print(f"    {label}:")
    print(f"      trials compared                            {entry['trialsCompared']}")
    print(f"      trials whose float32 point differs         {entry['trialsWithDifferentFloat32Point']}")
    print(f"      trials whose loss differs at same index    {entry['trialsWithDifferentLossAtSameIndex']}")
    print(f"      max loss difference at the same index      {entry['maxLossDifferenceAtSameIndex']:.12g}")
    print(f"      best-so-far differs on N trials            {entry['bestSoFarDifferentCount']}")
    if "sharedCandidateLossMismatches" in entry:
        print(f"      shared candidate with a different loss     {entry['sharedCandidateLossMismatches']}")
    print(f"      final optimum delta C10/C12/phi12          "
          f"{entry['finalBestPointDeltaC10']:.6g} / {entry['finalBestPointDeltaC12']:.6g} / "
          f"{entry['finalBestPointDeltaPhi12']:.6g}")
    print(f"      final loss delta (batched - sequential)    {entry['finalBestLossDelta']:.12g}")


def _reference_candidates(entry: dict) -> tuple[str, ...]:
    params = entry["params"]
    if "C10_nm" not in params:
        return ()
    return (
        float32_key({
            "c10Nanometers": params["C10_nm"],
            "c12Nanometers": params["C12_nm"],
            "phi12Radians": float(params.get("phi12_deg", 0.0)) * 3.141592653589793 / 180.0,
        }),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--repeat", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--pin", default=None)
    parser.add_argument("--record-pin", default=None)
    parser.add_argument("--recapture", action="store_true")
    parser.add_argument("--reason", default=None)
    args = parser.parse_args(argv)

    native = json.loads(Path(args.native).read_text(encoding="utf-8"))
    sequential = native["sequential"]
    print(f"case {native['caseName']}  side {native['scanSide']}  "
          f"logical BF {native['logicalBrightfieldCount']}  active {native['activeBrightfieldCount']}")
    print(f"seed {native['seed']}  trials {native['globalTrials']}  start {native['start']}")
    if native.get("caseJsonSha256"):
        print(f"case.json sha256 {native['caseJsonSha256']}")

    print("\n  fit path                    C10 (nm)            C12 (nm)            phi12 (rad)         loss")
    rows = [("sequential (production)", sequential)]
    if native.get("batched"):
        rows.append(("batched pair draw", native["batched"]))
    if native.get("batchedEngine"):
        rows.append(("batched native objective", native["batchedEngine"]))
    for name, run in rows:
        print(f"  {name:<26} {_format_optimum(run)}")
    print(f"  production optimize == sequential draw: "
          f"{native['productionOptimizeAlwaysMatchesSequential']}")
    print(f"  evaluations: sequential {sequential['totalEvaluations']} "
          f"(trials {sequential['globalTrials']} + refinement {sequential['refinementEvaluations']})")
    for name, run in rows[1:]:
        print(f"               {name} {run['totalEvaluations']} "
              f"(trials {run['globalTrials']} + refinement {run['refinementEvaluations']})")

    if native.get("comparison") or native.get("comparisonEngine"):
        print("\n  trajectory divergence (index by index)")
        if native.get("comparison"):
            _print_comparison("closure-batched vs sequential", native["comparison"])
        if native.get("comparisonEngine"):
            _print_comparison("native-batched vs sequential", native["comparisonEngine"])

    if native.get("batchObjective"):
        batch = native["batchObjective"]
        print("\n  native batch objective")
        print(f"    compiled into this build                   {batch['compiledIn']}")
        print(f"    candidate evaluations compared             {batch['probes']}")
        print(f"    losses not bit-identical to single eval    {batch['bitwiseIdentityMismatches']}")
        print(f"    max loss ULP difference                    {batch['maxLossBitDifference']}")
        print(f"    single evals perturbed after a batch run   {batch['postBatchPurityMismatches']}")

    purity = native["objectivePurity"]
    print("\n  objective purity (loss depends only on the float32 triple)")
    print(f"    probes                                     {purity['probes']}")
    print(f"    repeat, same engine, bitwise mismatches    {purity['repeatSameEngineBitwiseMismatches']}")
    print(f"    fresh engine, bitwise mismatches           {purity['freshEngineBitwiseMismatches']}")
    print(f"    same float32 triple from another double    {purity['float32AliasBitwiseMismatches']}")

    repeat = None
    if args.repeat:
        repeat = json.loads(Path(args.repeat).read_text(encoding="utf-8"))
        print("\n  determinism (same command, second process)")
        first = trial_digest(sequential["trials"])
        second = trial_digest(repeat["sequential"]["trials"])
        print(f"    sequential trajectory digest, run 1        {first}")
        print(f"    sequential trajectory digest, run 2        {second}")
        print(f"    identical                                  {first == second}")
        if repeat.get("batched") and native.get("batched"):
            first_batch = trial_digest(native["batched"]["trials"])
            second_batch = trial_digest(repeat["batched"]["trials"])
            print(f"    batched trajectory identical               {first_batch == second_batch}")

    status = 0 if production_checks_pass(native, repeat) else 1
    if status:
        print("    FAIL: production consistency, objective purity, or repeat determinism changed")
    if args.reference:
        reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        print("\n  Python/QuantEM reference fit on the same artifact")
        for run in reference["runs"]:
            best = run["best_recorded"] or {"params": {}, "loss": float("nan")}
            params = best["params"]
            print(f"    optuna_batch_size={run['optuna_batch_size']} trials={run['recorded_trials']} "
                  f"loss={best['loss']:.12g} C10={params.get('C10_nm')} "
                  f"C12={params.get('C12_nm')} phi12_deg={params.get('phi12_deg')} "
                  f"result_loss={run['result_loss']}")
        native_runs = [("sequential", sequential)]
        if native.get("batched"):
            native_runs.append(("closure-batched", native["batched"]))
        if native.get("batchedEngine"):
            native_runs.append(("native-batched", native["batchedEngine"]))
        for run in reference["runs"]:
            reference_keys = []
            for entry in run["trajectory"]:
                reference_keys.extend(_reference_candidates(entry))
            for name, run_native in native_runs:
                native_keys = [float32_key(trial) for trial in run_native["trials"]]
                shared = len(set(reference_keys) & set(native_keys))
                same_index = sum(
                    1 for left, right in zip(reference_keys, native_keys) if left == right
                )
                print(f"    reference batch={run['optuna_batch_size']} vs native {name}: "
                      f"{same_index}/{min(len(reference_keys), len(native_keys))} identical at the same "
                      f"index, {shared} candidates shared anywhere")

    pin_path = args.record_pin or args.pin
    if pin_path and Path(pin_path).is_file():
        pin = json.loads(Path(pin_path).read_text(encoding="utf-8"))
        digest = trial_digest(sequential["trials"])
        pinned = pin["sequential"]
        ok = digest == pinned["trajectorySha256"]
        same_optimum = optimum(sequential) == tuple(
            float(value) for value in pinned["best"]
        ) and sequential["bestLoss"] == float(pinned["bestLoss"])
        print("\n  frozen fit pin")
        print(f"    pin: {pin_path}")
        print(f"    sequential trajectory digest {digest}")
        print(f"    pinned digest                {pinned['trajectorySha256']}")
        print(f"    trajectory matches pin: {ok}")
        print(f"    optimum and loss match pin: {same_optimum}")
        if not (ok and same_optimum):
            print("    FAIL: the production fit moved; this is a finding, not a pin to rewrite")
            status = 1

    if args.record_pin:
        target = Path(args.record_pin)
        if target.is_file() and not args.recapture:
            print(f"\n  {target} already exists; pass --recapture --reason <why> to overwrite")
            return 2
        if target.is_file() and not args.reason:
            print("\n  --recapture requires --reason so the record shows why the pin moved")
            return 2
        pin = {
            "schema_version": 1,
            "name": "SSB native fit trajectory (Metal unbatched production path)",
            "case": native["caseName"],
            "case_json_sha256": native.get("caseJsonSha256"),
            "seed": native["seed"],
            "global_trials": native["globalTrials"],
            "start": native["start"],
            "note": (
                "Frozen trajectory of the production fit on one exported exact "
                "artifact. A mismatch is a finding: investigate the sampler, the "
                "objective or the engine cache before touching this file."
            ),
            "sequential": {
                "trajectorySha256": trial_digest(sequential["trials"]),
                "best": list(optimum(sequential)),
                "bestLoss": sequential["bestLoss"],
                "totalEvaluations": sequential["totalEvaluations"],
                "refinementEvaluations": sequential["refinementEvaluations"],
            },
            "productionOptimizeMatchesSequential": native[
                "productionOptimizeAlwaysMatchesSequential"
            ],
            "objectivePurity": native["objectivePurity"],
        }
        for name, key in (("batched", "batched"), ("batchedEngine", "batchedEngine")):
            if native.get(key):
                entry = native.get(
                    "comparisonEngine" if key == "batchedEngine" else "comparison"
                )
                pin[name] = {
                    "trajectorySha256": trial_digest(native[key]["trials"]),
                    "best": list(optimum(native[key])),
                    "bestLoss": native[key]["bestLoss"],
                    "totalEvaluations": native[key]["totalEvaluations"],
                }
                if entry:
                    pin[name]["divergenceFromSequential"] = entry
        if args.recapture:
            pin["recapture"] = {"reason": args.reason}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n  recorded {target}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
