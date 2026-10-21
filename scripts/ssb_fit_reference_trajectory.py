"""Record the Python/QuantEM SSB fit trajectory on an exported exact artifact.

The native ``SSBOptimizer`` is a reimplementation of the QuantEM TPE contract.
This script records what the *reference* (the public ``quantem.gpu`` MPS
phase-variance fit, ``find_aberrations(tilt=False, objective="phase_variance")``) actually draws, so a native trajectory can be compared against the reference on
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

import argparse
from contextlib import contextmanager
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_RUNS = Path(os.environ.get(
    "QUANTEM_SSB_PARITY_RUNS", Path.home() / "perf-lab/ssb-audit/parity-runs"
))


@contextmanager
def _reference_input(root: Path):
    """Adapt archived declarations in temporary storage, never in the fixture."""
    meta = json.loads((root / "case.json").read_text())
    source = root / "source"
    calibration = json.loads((source / "snapshots/cal.json").read_text())
    detector_sampling = meta.get("detector_sampling_mrad")
    if detector_sampling is None:
        detector_sampling = [meta["det_sampling_mrad"]] * 2
    options = {
        "voltage_kV": meta["voltage_kV"],
        "semiangle_mrad": meta["semiangle_mrad"],
        "scan_sampling_A": tuple(meta["scan_sampling_A"]),
        "detector_sampling_mrad": tuple(detector_sampling),
        "scan_detector_rotation_deg": (
            meta["scan_detector_rotation_deg"] if "scan_detector_rotation_deg" in meta
            else meta["rotation_angle_deg"]
        ),
    }
    if calibration.get("schema") == "quantem.ssb-browser-calibration/3":
        yield source, meta, options
        return
    # Only the known frozen archive layout is mapped; no calibration is fitted.
    current = {
        "schema": "quantem.ssb-browser-calibration/3",
        "bright_field_rows": calibration["bf_rows"],
        "bright_field_cols": calibration["bf_cols"],
        "bright_field_center_px": calibration["bf_center"],
        "bright_field_radius_px": calibration["bf_radius_px"],
        "detector_shape": calibration["detector_shape"],
        "scan_region": meta["scan_region"],
        "scan_shape": meta["scan_shape"],
        "bf_column_companion_path": "bf_columns.u16",
        "bf_column_encoding": "u16",
        "excluded_detector_pixels": calibration["bf_excluded_detector_pixels"],
    }
    with tempfile.TemporaryDirectory(prefix="ssb-fit-input-") as temporary:
        view = Path(temporary)
        (view / "cal.json").write_text(json.dumps(current))
        (view / "bf_columns.u16").symlink_to((source / "bf_columns.u16").resolve())
        yield view, meta, options


def record(session, *, trials: int, seed: int, batch: int) -> dict:
    """Run one reference fit and return its recorded trajectory."""

    import optuna
    from optuna.study import Study

    from quantem.gpu.ssb.mps import backend as mps_backend

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

    original_optimize = mps_backend.optimize_phase_variance

    def optimize_with_batch(*args, **kwargs):
        kwargs["optuna_batch_size"] = int(batch)
        return original_optimize(*args, **kwargs)

    Study.tell = recording_tell
    mps_backend.optimize_phase_variance = optimize_with_batch
    try:
        result = session.find_aberrations(tilt=False, objective="phase_variance", num_trials=int(trials),
                                          seed=int(seed), verbose=False)
    finally:
        Study.tell = original_tell
        mps_backend.optimize_phase_variance = original_optimize
        del optuna  # keep the import local to this call
    best = min(
        (entry for entry in recorded if entry["loss"] is not None),
        key=lambda entry: entry["loss"],
        default=None,
    )
    if len(recorded) != trials or any(
        entry["loss"] is None or not np.isfinite(entry["loss"]) for entry in recorded
    ):
        raise RuntimeError(f"Expected {trials} finite reference trials; recorded {len(recorded)}.")
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

    root = Path(args.runs) / "strict-parity" / args.case
    meta = json.loads((root / "case.json").read_text())

    from quantem.gpu.ssb import SSB

    batches = args.batch or [2, 1]
    report = {
        "case": args.case,
        "source": str(root),
        "scan_side": int(meta["scan_shape"][0]),
        "bf_count": int(len(meta["brightfield_kx"])),
        "trials": int(args.trials),
        "seed": int(args.seed),
        "runs": [],
    }
    for batch in batches:
        with _reference_input(root) as (source, _, options):
            with SSB.open(str(source), backend="mps", verbose=False, **options) as session:
                run = record(session, trials=args.trials, seed=args.seed, batch=batch)
        report["runs"].append(run)
        best = run["best_recorded"] or {}
        params = best.get("params", {})
        print(
            f"optuna_batch_size={batch} trials={run['recorded_trials']} "
            f"best loss={best.get('loss')} C10_A={params.get('C10_A')} "
            f"C12_A={params.get('C12_A')} phi12_deg={params.get('phi12_deg')} "
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
