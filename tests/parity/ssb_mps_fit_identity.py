"""MPS-side SSB fit identity probe (evidence only, not a gate).

Mirrors `tests/metal/ssb_production_protocol.swift` mode ``identity`` on the
public MPS path. It opens the same exported exact BF-column artifact with the
detector sampling the app's ``matchApertureToBrightfieldDisk()`` produces
(``semiangle / brightfield radius``), so both backends see the same 8937-pixel
disk, and then compares

  25 trials + Nelder-Mead,
  200 trials + Nelder-Mead,
  Nelder-Mead alone warm started from the 200-trial optimum

by optimum triple, loss and wall time. Nothing is recaptured: the script only
reports what the production MPS path returns.

Usage::

    python tests/parity/ssb_mps_fit_identity.py <case-dir> <out-json> [--probe-only]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def load_case(case_directory: Path) -> dict:
    return json.loads((case_directory / "case.json").read_text())


def open_session(case_directory: Path, case: dict, aberrations: dict | None = None):
    from quantem.gpu.ssb import SSB

    return SSB.open(
        str(case_directory / "source"),
        backend="mps",
        voltage_kV=float(case["voltage_kV"]),
        semiangle_mrad=float(case["semiangle_mrad"]),
        scan_sampling_A=float(case["scan_sampling_A"][0]),
        det_sampling=float(case["semiangle_mrad"]) / float(case["bf_radius_px"]),
        rotation_angle_deg=float(case["rotation_angle_deg"]),
        aberrations=aberrations,
        verbose=False,
    )


def describe_session(session) -> dict:
    """Report whatever the session exposes about the selected BF geometry."""

    fields: dict[str, object] = {}
    for name in dir(session):
        if name.startswith("_"):
            continue
        if not any(token in name for token in ("bf", "sampling", "radius", "shape")):
            continue
        try:
            value = getattr(session, name)
        except Exception:  # pragma: no cover - introspection only
            continue
        if isinstance(value, (int, float, str, bool, tuple)):
            fields[name] = value
    return fields


def probe_loss(session, aberrations: dict) -> dict:
    started = time.perf_counter()
    _phase, loss = session.preview(aberrations)
    return {"loss": float(loss), "seconds": time.perf_counter() - started}


def run_fit(session, label: str, trials: int, refinement: str = "nelder-mead") -> dict:
    started = time.perf_counter()
    result = session.fit(trials=trials, refinement=refinement, seed=42, verbose=False)
    seconds = time.perf_counter() - started
    return {
        "label": label,
        "trials": trials,
        "refinement": refinement,
        "best": {key: float(value) for key, value in dict(result.aberrations).items()},
        "loss": None if result.loss is None else float(result.loss),
        "seconds": seconds,
        "reported_elapsed": None if result.elapsed is None else float(result.elapsed),
        "timings": {key: float(value) for key, value in dict(result.timings or {}).items()},
    }


def same(a: dict, b: dict) -> bool:
    if not a or not b:
        return False
    keys = sorted(set(a) | set(b))
    return all(a.get(key) == b.get(key) for key in keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    case = load_case(args.case_directory)
    report: dict[str, object] = {
        "case": case["name"],
        "detector_sampling_mrad": float(case["semiangle_mrad"]) / float(case["bf_radius_px"]),
        "documented_detector_sampling_mrad": float(case["det_sampling_mrad"]),
        "start": {"C10": 0.0, "C12": 50.0, "phi12": 0.0},
        "configurations": [],
    }
    session = open_session(args.case_directory, case)
    report["session"] = describe_session(session)

    start_point = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}
    report["probe_start"] = probe_loss(session, start_point)
    print(f"probe at start {report['probe_start']}", flush=True)

    # Geometry gate: the MPS objective must see the same disk-matched aperture
    # as the production Metal engine. The reference point and loss come from
    # the Metal harness (`ssb-production-protocol ... ab`), which fitted the
    # app configuration on the same exported artifact.
    reference_point = {
        "C10": float(os.environ.get("QUANTEM_SSB_APP_OPT_C10", "5.884552648120136")),
        "C12": float(os.environ.get("QUANTEM_SSB_APP_OPT_C12", "0.4614111044099156")),
        "phi12": float(os.environ.get("QUANTEM_SSB_APP_OPT_PHI12", "1.3530133901569708")),
    }
    reference_loss = float(os.environ.get("QUANTEM_SSB_APP_OPT_LOSS", "0.15314708650112152"))
    report["probe_metal_optimum"] = probe_loss(session, reference_point)
    report["probe_metal_optimum"]["metal_loss"] = reference_loss
    deviation = abs(report["probe_metal_optimum"]["loss"] - reference_loss) / reference_loss
    report["probe_metal_optimum"]["relative_deviation"] = deviation
    print(
        f"probe at the Metal app optimum: mps {report['probe_metal_optimum']['loss']!r} "
        f"vs metal {reference_loss!r} (rel {deviation:.3e})",
        flush=True,
    )
    tolerance = float(os.environ.get("QUANTEM_SSB_GEOMETRY_TOLERANCE", "1e-4"))
    if deviation > tolerance:
        report["geometry"] = "mismatch"
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"geometry mismatch (>{tolerance:g}); skipping the fit arms")
        return 0
    report["geometry"] = "match"

    if args.probe_only:
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"wrote {args.output}")
        return 0

    arms = [
        ("tpe25", 25, start_point),
        ("tpe200", 200, start_point),
    ]
    best_200: dict | None = None
    for label, trials, point in arms:
        try:
            session = open_session(args.case_directory, case, aberrations=point)
            arm = run_fit(session, label, trials)
        except Exception as error:  # evidence: record the failure, keep going
            arm = {"label": label, "trials": trials, "error": repr(error)}
        report["configurations"].append(arm)
        print(f"{label}: {json.dumps(arm, sort_keys=True)}", flush=True)
        if label == "tpe200" and "best" in arm:
            best_200 = arm["best"]

    if best_200 is not None:
        try:
            session = open_session(args.case_directory, case, aberrations=best_200)
            arm = run_fit(session, "nmWarm", 0)
        except Exception as error:
            arm = {"label": "nmWarm", "trials": 0, "error": repr(error)}
        report["configurations"].append(arm)
        print(f"nmWarm: {json.dumps(arm, sort_keys=True)}", flush=True)

    by_label = {arm["label"]: arm for arm in report["configurations"]}
    comparisons = {
        "tpe25_optimum_matches_tpe200": None,
        "tpe25_loss_matches_tpe200": None,
        "nmWarm_optimum_matches_tpe200": None,
        "nmWarm_loss_matches_tpe200": None,
    }
    if {"tpe25", "tpe200"} <= set(by_label) and all(
        "best" in by_label[label] for label in ("tpe25", "tpe200")
    ):
        comparisons["tpe25_optimum_matches_tpe200"] = same(
            by_label["tpe25"]["best"], by_label["tpe200"]["best"]
        )
        comparisons["tpe25_loss_matches_tpe200"] = (
            by_label["tpe25"]["loss"] == by_label["tpe200"]["loss"]
        )
    if {"nmWarm", "tpe200"} <= set(by_label) and all(
        "best" in by_label[label] for label in ("nmWarm", "tpe200")
    ):
        comparisons["nmWarm_optimum_matches_tpe200"] = same(
            by_label["nmWarm"]["best"], by_label["tpe200"]["best"]
        )
        comparisons["nmWarm_loss_matches_tpe200"] = (
            by_label["nmWarm"]["loss"] == by_label["tpe200"]["loss"]
        )
    report["comparisons"] = comparisons
    report["blocked"] = sorted(
        arm["label"] for arm in report["configurations"] if "error" in arm
    )
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(comparisons, indent=1))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
