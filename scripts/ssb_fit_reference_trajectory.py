"""Record the Python/QuantEM SSB fit trajectory on an exported exact artifact.

The native ``SSBOptimizer`` is a reimplementation of the QuantEM TPE contract.
This script records what the *reference* (the public ``quantem.gpu`` MPS fit)
actually draws, so a native trajectory can be compared against the reference on
identical inputs and one seed instead of against prose.

It patches nothing scientific: the only change is ``optuna_batch_size``, which
the reference itself exposes (default 2, matching the native pair draw), and a
recorder on ``optuna.study.Study.tell`` that copies the trial parameters and
losses the reference study has already told.

Usage:
    GPU_RUN_LABEL=parity ~/perf-lab/ssb-audit/gpurun \
      PYTHONPATH=src python scripts/ssb_fit_reference_trajectory.py \
        --case arina-128-full-disk --trials 200 --seed 42 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests" / "parity"))

from ssb_parity_case import CASES, ensure_case_directory, load_case_declaration  # noqa: E402

DEFAULT_RUNS = Path.home() / "perf-lab/ssb-audit/parity-runs"


def record(session, *, trials: int, seed: int, batch: int) -> dict:
    """Run one reference fit and return its recorded trajectory."""

    import optuna
    from optuna.study import Study

    from quantem.gpu.ssb.backends.mps import backend as mps_backend
    from quantem.gpu.ssb.backends.mps import optimizer as mps_optimizer

    recorded: list[dict] = []
    original_tell = Study.tell

    def recording_tell(study, trial, values=None, *args, **kwargs):
        entry = {
            "number": int(trial.number),
            "params": dict(trial.params),
            "loss": None if values is None else float(np.asarray(values).reshape(-1)[0]),
        }
        recorded.append(entry)
        return original_tell(study, trial, values, *args, **kwargs)

    original_optimize = mps_backend.optimize_mps

    def optimize_with_batch(data, **kwargs):
        kwargs["optuna_batch_size"] = int(batch)
        return original_optimize(data, **kwargs)

    Study.tell = recording_tell
    mps_backend.optimize_mps = optimize_with_batch
    try:
        result = session.fit(trials=int(trials), seed=int(seed), verbose=False)
    finally:
        Study.tell = original_tell
        mps_backend.optimize_mps = original_optimize
        del optuna  # keep the import local to this call
    best = min(
        (entry for entry in recorded if entry["loss"] is not None),
        key=lambda entry: entry["loss"],
        default=None,
    )
    return {
        "optuna_batch_size": int(batch),
        "recorded_trials": len(recorded),
        "best_recorded": best,
        "result_loss": None if result.loss is None else float(result.loss),
        "trajectory": recorded,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="arina-128-full-disk")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, action="append", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    args = parser.parse_args(argv)

    case = CASES[args.case]
    root = ensure_case_directory(case)
    meta = load_case_declaration(case)

    from quantem.gpu.ssb import SSB

    batches = args.batch or [2, 1]
    report = {
        "case": case.name,
        "source": str(case.source),
        "scan_side": int(case.scan_side),
        "bf_count": int(len(meta["brightfield_kx"])),
        "trials": int(args.trials),
        "seed": int(args.seed),
        "runs": [],
    }
    for batch in batches:
        session = SSB.open(
            str(root / "source"),
            backend="mps",
            voltage_kV=case.voltage_kV,
            semiangle_mrad=case.semiangle_mrad,
            scan_sampling_A=case.scan_sampling_A,
            det_sampling=float(meta["det_sampling_mrad"]),
            rotation_angle_deg=case.rotation_angle_deg,
            verbose=False,
        )
        run = record(session, trials=args.trials, seed=args.seed, batch=batch)
        report["runs"].append(run)
        best = run["best_recorded"] or {}
        params = best.get("params", {})
        print(
            f"optuna_batch_size={batch} trials={run['recorded_trials']} "
            f"best loss={best.get('loss')} C10={params.get('C10_nm')} "
            f"C12={params.get('C12_nm')} phi12_deg={params.get('phi12_deg')} "
            f"result_loss={run['result_loss']}",
            flush=True,
        )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
