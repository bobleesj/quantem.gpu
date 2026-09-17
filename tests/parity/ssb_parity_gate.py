"""Strict float32/complex64 SSB parity gate.

One command proves that a measured SSB path did not lose precision:

    quantem-ssb-parity --case arina-128-full-disk

The gate compares, at byte-identical inputs (one exported exact BF-column
artifact, one declared geometry, one aberration setting):

1. the public Python ``quantem.gpu.SSB`` MPS path,
2. the native Swift ``MetalSSBKernels`` path through its standalone harness, and
3. the independent double-precision oracle in :mod:`ssb_double_reference`,

and reports the exact error metrics the user asked for: max absolute error,
max relative error, relative L2 of the object, max phase error in radians, and
max absolute (and relative) loss error. A failure is a finding; the tolerances
in :data:`GATES` are physical float32 bounds derived from the measured
single-precision arithmetic floor of the same formula, never loosened to make a
backend pass.

Run ``python tests/parity/ssb_parity_gate.py --help`` for the case list.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssb_double_reference import ssb_reference  # noqa: E402
from ssb_parity_case import (  # noqa: E402
    CASES,
    SSBParityCase,
    case_root,
    compare_products,
    ensure_case_directory,
    export_case,
    load_case_declaration,
    reference_products,
)

__all__ = ["GATES", "main", "run_case"]

#: Strict parity gates. Each metric must clear BOTH bounds:
#:
#: ``absolute``
#:     The float32 level the user asked for, derived from first principles
#:     rather than fitted to a measurement. ``object_relative_l2`` is set at
#:     1e-5: a float32 FFT of a 128-point row carries a relative round-off of a
#:     few ``eps32 = 2**-24 = 5.96e-8``, and the object is the normalized sum of
#:     ``N`` partially cancelling transforms, so the summed relative error of a
#:     faithful float32 evaluation lands in the ``1e-6`` decade. The
#:     ``object_max_abs_error_relative`` bound is 1e-5 for the same reason.
#:     ``phase_max_error_radians`` is the worst case implied by the object
#:     bound at a conservative amplitude floor: a pixel carrying at least 0.1 of
#:     the object's peak magnitude can rotate by at most
#:     ``object_max_abs_error_relative / 0.1 = 1e-4`` radians. The phase metric
#:     itself is evaluated over the implemented core set (``core_ratio = 1e-3``
#:     of the peak), and every gated setting measures a minimum core amplitude
#:     of 0.876 or above (recorded per entry as ``object_magnitude_ratio_min``),
#:     so the 0.1 floor is conservative by 8.8x and the bound implied by the
#:     observations is 1.14e-5 radians. In practice both sizes are governed by
#:     the tighter ``2x`` measured floor term (6.5e-6 to 1.2e-5 radians), and
#:     this absolute value only acts as a backstop. ``loss_relative_error`` is set at 1e-5: the loss is a phase
#:     variance whose spread is ``sqrt(loss)``, so a per-sample phase error of
#:     ``eps32`` perturbs it by about ``2 * eps32`` relative; 1e-5 is two
#:     orders below the 1.06e-3 relative error of the retired half-plane
#:     endpoint defect and 5x below the 5e-5 gate that the cached/streamed
#:     disagreement broke.
#:
#: ``floor_multiple``
#:     The measured single-precision floor of the identical formula, computed
#:     by re-running this repository's independent reference in float32 with a
#:     different FFT library (SciPy pocketfft). No accelerated float32 backend
#:     can be required to beat a straightforward float32 evaluation of the same
#:     objective, and a backend that is much worse than that floor is losing
#:     precision. The multiple is 2.0.
GATES: dict[str, dict[str, float]] = {
    "object_relative_l2": {"absolute": 1.0e-5, "floor_multiple": 2.0},
    "object_max_abs_error_relative": {"absolute": 1.0e-5, "floor_multiple": 2.0},
    "phase_max_error_radians": {"absolute": 1.0e-4, "floor_multiple": 2.0},
    "loss_relative_error": {"absolute": 1.0e-5, "floor_multiple": 2.0},
    #: The objective is a variance of phases that live on the unit circle. A
    #: float32 phase carries at most ``eps32`` of absolute error for a
    #: well-conditioned argument, so a squared phase carries at most
    #: ``2*pi*eps32`` and the variance of such terms at most ``4*pi*eps32 =
    #: 7.5e-7``. The absolute bound is the next round decade above that, which
    #: is also ten times tighter than the 1e-5 relative bound at the observed
    #: loss magnitude of about 0.1.
    "loss_absolute_error": {"absolute": 1.0e-6, "floor_multiple": 2.0},
}

#: Where the measured floor is zero, only the absolute bound applies.
GATE_FLOOR_METRIC = {
    "object_relative_l2": "object_relative_l2",
    "object_max_abs_error_relative": "object_max_abs_error_relative",
    "phase_max_error_radians": "phase_max_error_radians_core",
    "loss_relative_error": "loss_relative_error",
    "loss_absolute_error": "loss_absolute_error",
}


def _aberration_dict(case: SSBParityCase, index: int) -> dict[str, float]:
    c10, c12, phi12 = case.aberrations[index]
    return {"C10": float(c10), "C12": float(c12), "phi12": float(phi12)}


def _case_meta(case: SSBParityCase) -> dict[str, object]:
    return load_case_declaration(case)


def _floor_metrics(oracle, floor) -> dict[str, float]:
    """Return the single-precision floor of the identical formula."""

    metrics = compare_products(
        oracle.object_wave,
        oracle.loss,
        oracle.mean_phase,
        floor.object_wave,
        floor.loss,
        floor.mean_phase,
    )
    return {
        "object_relative_l2": metrics["object_relative_l2"],
        "object_max_abs_error_relative": metrics["object_max_abs_error_relative"],
        "phase_max_error_radians_core": metrics["object_phase_max_error_radians_core"],
        "loss_relative_error": metrics["loss_relative_error"],
        "loss_absolute_error": metrics["loss_abs_error"],
    }


def run_reference(
    case: SSBParityCase,
    meta: dict[str, object],
    setting: tuple[float, float, float],
    *,
    geometry: str = "exact",
    dtype=np.float64,
) -> object:
    """Return the independent oracle for one explicit aberration setting."""

    c10, c12, phi12 = (float(value) for value in setting)
    result, _summary = reference_products(
        case,
        meta,
        c10,
        c12,
        phi12,
        geometry=geometry,
        dtype=dtype,
    )
    return result


def open_mps_session(case: SSBParityCase, meta: dict[str, object]):
    """Open the production public MPS session on the exported exact artifact."""

    from quantem.gpu.ssb import SSB

    root = case_root(case)
    return SSB.open(
        str(root / "source"),
        backend="mps",
        voltage_kV=case.voltage_kV,
        semiangle_mrad=case.semiangle_mrad,
        scan_sampling_A=case.scan_sampling_A,
        det_sampling=float(meta["det_sampling_mrad"]),
        rotation_angle_deg=case.rotation_angle_deg,
        verbose=False,
    )


def run_mps(case: SSBParityCase, meta: dict[str, object], session=None) -> list[dict]:
    """Return the production MPS object, mean phase and loss per aberration."""

    session = open_mps_session(case, meta) if session is None else session
    products = []
    for index in range(len(case.aberrations)):
        aberrations = _aberration_dict(case, index)
        started = time.perf_counter()
        result = session.reconstruct(aberrations, compute_loss=False)
        object_seconds = time.perf_counter() - started
        started = time.perf_counter()
        phase, loss = session.preview(aberrations)
        preview_seconds = time.perf_counter() - started
        products.append(
            {
                "index": index,
                "object_wave": np.asarray(result.object_wave, dtype=np.complex128),
                "mean_phase": None if phase is None else np.asarray(phase, dtype=np.float64),
                "object_phase": np.angle(
                    np.asarray(result.object_wave, dtype=np.complex128)
                ),
                "loss": None if loss is None else float(loss),
                "object_seconds": object_seconds,
                "preview_seconds": preview_seconds,
                "num_bf": int(getattr(result, "num_bf", 0) or 0),
            }
        )
    return products


def metal_harness_path() -> Path | None:
    """Return the compiled native parity harness, if it has been built."""

    import json as _json

    marker = REPO_ROOT / "build" / "ssb-parity-check.json"
    if not marker.is_file():
        return None
    payload = _json.loads(marker.read_text(encoding="utf-8"))
    binary = Path(str(payload["binary"]))
    return binary if binary.is_file() else None


def run_metal(case: SSBParityCase, meta: dict[str, object]) -> list[dict]:
    """Return the native Metal object, mean phase and loss per aberration."""

    binary = metal_harness_path()
    if binary is None:
        raise RuntimeError(
            "The native parity harness is not built. Run "
            "`scripts/check_ssb_parity.sh --build-metal` first."
        )
    root = case_root(case)
    out_dir = root / "metal"
    subprocess.run(
        [str(binary), str(root), str(out_dir)],
        check=True,
    )
    summary = json.loads((out_dir / "metal.json").read_text(encoding="utf-8"))
    side = case.scan_side
    products = []
    for index in range(len(case.aberrations)):
        variants = {}
        for variant in summary["aberrations"][index]["variants"]:
            object_wave = np.fromfile(
                out_dir / f"object-{index}-{variant['name']}.f32", dtype=np.float32
            )
            # The native engine's ``phase(of:)`` runs the ``ssb_object_phase``
            # kernel, so this artifact is the phase of the complex object, not
            # the objective's mean bright-field phase.
            object_phase = np.fromfile(
                out_dir / f"phase-{index}-{variant['name']}.f32", dtype=np.float32
            )
            variants[variant["name"]] = {
                "object_wave": (
                    object_wave.reshape(-1, 2)[:, 0]
                    + 1j * object_wave.reshape(-1, 2)[:, 1]
                ).astype(np.complex128).reshape(side, side),
                "mean_phase": None,
                "object_phase": object_phase.astype(np.float64).reshape(side, side),
                "loss": float(variant["loss"]),
                "provenance": {
                    key: variant[key]
                    for key in (
                        "cacheBudgetBytes",
                        "logicalBrightfieldCount",
                        "executedBrightfieldCount",
                        "loggedCachedBrightfieldCount",
                        "loggedStreamedBrightfieldCount",
                        "wallSeconds",
                        "gpuSeconds",
                    )
                    if key in variant
                },
            }
        products.append(
            {
                "index": index,
                "variants": variants,
                "primary": "cached",
            }
        )
    return products


def run_case(
    case: SSBParityCase,
    *,
    export: bool,
    use_metal: bool,
    use_mps: bool = True,
    geometry: str = "exact",
    reference_only: bool = False,
) -> dict[str, object]:
    """Run the full three-way comparison for one case and return the report."""

    if export:
        export_case(case)
    root = ensure_case_directory(case)
    if not (root / "source" / "bf_columns.u16").is_file():
        raise FileNotFoundError(
            f"Case {case.name!r} has no exported exact inputs at {root}. Run "
            f"`scripts/check_ssb_parity.sh --export` first."
        )
    meta = _case_meta(case)
    report: dict[str, object] = {
        "case": case.name,
        "description": case.description,
        "scan_side": case.scan_side,
        "bf_count": int(len(meta["brightfield_kx"])),
        "documented_bf_count": int(meta["documented_bf_count"]),
        "active_bf_count": int(meta["active_bf_count"]),
        "det_sampling_mrad": float(meta["det_sampling_mrad"]),
        "rotation_angle_deg": float(meta["rotation_angle_deg"]),
        "aberrations": [],
    }
    settings = list(case.aberrations)
    if reference_only:
        mps_products = None
        metal_products = None
    else:
        mps_products = None
        if use_mps:
            session = open_mps_session(case, meta)
            mps_products = run_mps(case, meta, session=session)
        metal_products = run_metal(case, meta) if use_metal else None
    for index, setting in enumerate(settings):
        c10, c12, phi12 = (float(value) for value in setting)
        oracle = run_reference(case, meta, setting, geometry=geometry)
        floor = run_reference(case, meta, setting, geometry=geometry, dtype=np.float32)
        entry: dict[str, object] = {
            "index": index,
            "aberrations": {"C10": c10, "C12": c12, "phi12": phi12},
            "gated": True,
            "reference": {
                "loss": float(oracle.loss),
                "object_scale": float(oracle.object_scale),
                "active_bf": int(oracle.active_bf),
                "inactive_bf": int(oracle.inactive_bf),
                "dc_real": float(np.real(oracle.dc_value)),
            },
            "float32_floor": {
                key: value
                for key, value in _floor_metrics(oracle, floor).items()
            },
        }
        if mps_products is not None:
            mps = mps_products[index]
            entry["mps"] = compare_products(
                oracle.object_wave,
                oracle.loss,
                oracle.mean_phase,
                mps["object_wave"],
                mps["loss"],
                mps["mean_phase"],
                object_phase=mps["object_phase"],
            ) | {
                "object_seconds": mps["object_seconds"],
                "preview_seconds": mps["preview_seconds"],
                "num_bf": mps["num_bf"],
            }
        if metal_products is not None:
            variants = metal_products[index]["variants"]
            entry["metal"] = {
                name: compare_products(
                    oracle.object_wave,
                    oracle.loss,
                    oracle.mean_phase,
                    variant["object_wave"],
                    variant["loss"],
                    variant["mean_phase"],
                    object_phase=variant["object_phase"],
                )
                | {"provenance": variant["provenance"]}
                for name, variant in variants.items()
            }
            # Direct topology comparison: the cached objective was the one
            # the retired half-plane endpoint defect corrupted, so cached
            # versus hybrid/streamed is stated on its own rather than through
            # either backend's agreement with the oracle.
            for name in sorted(variants):
                if name == "cached":
                    continue
                entry[f"metal_cached_vs_{name}"] = compare_products(
                    variants["cached"]["object_wave"],
                    variants["cached"]["loss"],
                    variants["cached"]["mean_phase"],
                    variants[name]["object_wave"],
                    variants[name]["loss"],
                    variants[name]["mean_phase"],
                    object_phase=variants[name]["object_phase"],
                )
            if mps_products is not None:
                for name in sorted(variants):
                    # Both backends are measured implementations here, so the
                    # oracle's phase roles are filled by the native Metal
                    # products; the object-phase metric compares Metal's own
                    # phase kernel against the phase of the MPS object.
                    entry[f"mps_vs_metal_{name}"] = compare_products(
                        mps["object_wave"],
                        mps["loss"],
                        mps["mean_phase"],
                        variants[name]["object_wave"],
                        variants[name]["loss"],
                        variants[name]["mean_phase"],
                        object_phase=variants[name]["object_phase"],
                    )
        report["aberrations"].append(entry)
    for offset, setting in enumerate(case.diagnostic_aberrations):
        c10, c12, phi12 = (float(value) for value in setting)
        index = len(settings) + offset
        oracle = run_reference(case, meta, setting, geometry=geometry)
        floor = run_reference(case, meta, setting, geometry=geometry, dtype=np.float32)
        report["aberrations"].append(
            {
                "index": index,
                "aberrations": {"C10": c10, "C12": c12, "phi12": phi12},
                "gated": False,
                "reference": {"loss": float(oracle.loss)},
                "float32_floor": _floor_metrics(oracle, floor),
            }
        )
    return report


def _format_float(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.6g}"


def _metric_line(label: str, metrics: dict) -> str:
    """Return one compact metric line for a measured backend."""

    return (
        f"    {label:<22} object rel L2={_format_float(metrics['object_relative_l2'])} "
        f"max abs={_format_float(metrics['object_max_abs_error'])} "
        f"max rel={_format_float(metrics['object_max_abs_error_relative'])} "
        f"phase max={_format_float(metrics['object_phase_max_error_radians_core'])} rad "
        f"phase rms={_format_float(metrics['object_phase_rms_error_radians_core'])} rad "
        f"loss={_format_float(metrics['loss'])} "
        f"loss rel={_format_float(metrics['loss_relative_error'])}"
    )


def _gate(
    metric: str,
    value: float,
    floor: dict[str, float],
) -> tuple[float, bool]:
    """Return the effective bound for one metric and whether the value passes."""

    bound = GATES[metric]["absolute"]
    floor_value = floor.get(GATE_FLOOR_METRIC[metric])
    if floor_value is not None and floor_value > 0:
        bound = min(bound, GATES[metric]["floor_multiple"] * float(floor_value))
    return bound, float(value) <= bound


def _gate_rows(report: dict[str, object]) -> list[tuple[str, str, float, float, bool]]:
    """Return ``(label, metric, value, bound, passed)`` rows for every gate."""

    rows: list[tuple[str, str, float, float, bool]] = []
    name = str(report["case"])
    for entry in report["aberrations"]:
        suffix = f"#{entry['index']}"
        floor = entry["float32_floor"]
        rows.append(
            (
                f"{name}{suffix} floor",
                "float32 floor object rel L2",
                float(floor["object_relative_l2"]),
                float("inf"),
                True,
            )
        )
        if not entry.get("gated", False):
            continue
        backends: list[tuple[str, dict, dict[str, float]]] = []
        if "mps" in entry:
            backends.append(("mps", entry["mps"], floor))
        for variant, metrics in sorted(entry.get("metal", {}).items()):
            backends.append((f"metal:{variant}", metrics, floor))
        for key, metrics in sorted(entry.items()):
            if key.startswith("mps_vs_metal_"):
                backends.append(
                    (f"mps-vs-metal:{key.removeprefix('mps_vs_metal_')}", metrics, floor)
                )
            elif key.startswith("metal_cached_vs_"):
                backends.append(
                    (
                        f"metal cached-vs-{key.removeprefix('metal_cached_vs_')}",
                        metrics,
                        floor,
                    )
                )
        for label, metrics, floor_metrics in backends:
            for metric, key in (
                ("object_relative_l2", "object_relative_l2"),
                ("object_max_abs_error_relative", "object_max_abs_error_relative"),
                ("phase_max_error_radians", "object_phase_max_error_radians_core"),
                ("loss_relative_error", "loss_relative_error"),
                ("loss_absolute_error", "loss_abs_error"),
            ):
                value = float(metrics[key])
                bound, passed = _gate(metric, value, floor_metrics)
                rows.append((f"{name}{suffix} {label}", metric, value, bound, passed))
        mean_phase = entry.get("mps", {}).get("mean_phase_max_error_radians_core")
        if mean_phase is not None:
            bound, passed = _gate("phase_max_error_radians", float(mean_phase), floor)
            rows.append(
                (
                    f"{name}{suffix} mps",
                    "objective mean BF phase (rad)",
                    float(mean_phase),
                    bound,
                    passed,
                )
            )
    return rows


def print_report(report: dict[str, object], use_metal: bool) -> bool:
    """Print one case report and return whether every gate passed."""

    print(
        f"\n=== {report['case']} (BF {report['bf_count']} / documented "
        f"{report['documented_bf_count']}, aperture-active {report['active_bf_count']}, "
        f"det sampling {_format_float(report['det_sampling_mrad'])} mrad/px, "
        f"rotation {_format_float(report['rotation_angle_deg'])} deg) ==="
    )
    for entry in report["aberrations"]:
        aberr = entry["aberrations"]
        reference = entry["reference"]
        tag = "gated" if entry.get("gated", False) else "DIAGNOSTIC (not gated)"
        print(
            f"  #{entry['index']} [{tag}] C10={_format_float(aberr['C10'])} nm "
            f"C12={_format_float(aberr['C12'])} nm phi12={_format_float(aberr['phi12'])} rad"
        )
        print(f"    reference loss={reference['loss']:.10f}")
        floor = entry["float32_floor"]
        print(
            f"    float32 arithmetic floor : "
            f"object rel L2={_format_float(floor['object_relative_l2'])} "
            f"max rel={_format_float(floor['object_max_abs_error_relative'])} "
            f"phase max={_format_float(floor['phase_max_error_radians_core'])} rad "
            f"loss rel={_format_float(floor['loss_relative_error'])}"
        )
        if "mps" in entry:
            mps = entry["mps"]
            print(_metric_line("mps", mps))
            if "mean_phase_max_error_radians_core" in mps:
                print(
                    f"      objective mean BF phase: max={_format_float(mps['mean_phase_max_error_radians_core'])} rad "
                    f"rms={_format_float(mps['mean_phase_rms_error_radians_core'])} rad "
                    f"(object phase max={_format_float(mps['object_phase_max_error_radians_core'])} rad)"
                )
            print(
                f"      timings: object {_format_float(mps['object_seconds'])} s, "
                f"objective {_format_float(mps['preview_seconds'])} s"
            )
        for variant, metrics in sorted(entry.get("metal", {}).items()):
            print(_metric_line(f"metal:{variant}", metrics))
            provenance = metrics.get("provenance", {})
            print(
                f"      logical={provenance.get('logicalBrightfieldCount')} "
                f"executed={provenance.get('executedBrightfieldCount')} "
                f"cached={provenance.get('loggedCachedBrightfieldCount')} "
                f"streamed={provenance.get('loggedStreamedBrightfieldCount')} "
                f"budget={provenance.get('cacheBudgetBytes')}"
            )
        for key, metrics in sorted(entry.items()):
            if key.startswith("mps_vs_metal_"):
                print(
                    _metric_line(
                        f"mps vs metal:{key.removeprefix('mps_vs_metal_')}", metrics
                    )
                )
            elif key.startswith("metal_cached_vs_"):
                print(
                    _metric_line(
                        f"metal cached vs {key.removeprefix('metal_cached_vs_')}",
                        metrics,
                    )
                )
        if "metal" in entry:
            print(
                "      note: the native engine exposes no mean bright-field "
                "phase product, so metal rows report the ssb_object_phase "
                "kernel against the oracle object phase."
            )
    rows = _gate_rows(report)
    failures = [row for row in rows if not row[4]]
    print("    gates:")
    for label, metric, value, bound, passed in rows:
        bound_text = "measured" if bound == float("inf") else f"<= {_format_float(bound)}"
        print(
            f"      [{'ok  ' if passed else 'FAIL'}] {label:<36} {metric:<30} "
            f"{_format_float(value):>12} {bound_text}"
        )
    return not failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="case name (repeatable). Default: every case except the 512 one.",
    )
    parser.add_argument("--all-cases", action="store_true", help="run every case")
    parser.add_argument(
        "--export",
        action="store_true",
        help="(re)export the exact BF-column artifact before measuring",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="rewrite the artifact even when it already exists",
    )
    parser.add_argument("--no-metal", action="store_true", help="skip the native Metal path")
    parser.add_argument(
        "--metal-only",
        action="store_true",
        help=(
            "measure and gate only the native Metal pairs; skips the MPS path "
            "so a Metal-only change gets an unambiguous exit status"
        ),
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="compute only the oracle and its float32 floor (no GPU work)",
    )
    parser.add_argument(
        "--geometry",
        choices=("exact", "declared"),
        default="exact",
        help="oracle geometry precision (default: double-precision coordinates)",
    )
    parser.add_argument("--json", default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)

    if args.all_cases:
        names = list(CASES)
    elif args.case:
        names = [name for value in args.case for name in value.split(",")]
    else:
        names = [name for name in CASES if CASES[name].scan_side <= 256]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        parser.error(f"unknown cases {unknown}; known: {list(CASES)}")

    if args.force_export and not args.export:
        raise SystemExit(
            "--force-export only rewrites an existing artifact when --export is "
            "also given; pass --export --force-export (or use "
            "`scripts/check_ssb_parity.sh --export --force-export`)."
        )
    if args.metal_only and args.no_metal:
        parser.error("--metal-only and --no-metal are mutually exclusive")
    if args.metal_only and args.reference_only:
        parser.error("--metal-only and --reference-only are mutually exclusive")

    use_metal = not args.no_metal
    use_mps = not args.metal_only
    if args.metal_only:
        print(
            "NOTE: --metal-only - the MPS implementation is NOT measured in "
            "this run. A PASS below is the native Metal acceptance signal "
            "only; it is not a pass of the full parity gate, which is red on "
            "this artifact because of the recorded MPS findings."
        )
    if args.reference_only:
        use_metal = False
    elif args.metal_only and metal_harness_path() is None:
        raise SystemExit(
            "--metal-only needs the native harness. Build it first with "
            "`scripts/check_ssb_parity.sh --build-metal`."
        )
    elif use_metal and metal_harness_path() is None:
        print(
            "note: native Metal harness not built; measuring MPS against the "
            "oracle only (run scripts/check_ssb_parity.sh --build-metal).",
            file=sys.stderr,
        )
        use_metal = False

    reports = []
    passed = True
    for name in names:
        case = CASES[name]
        if args.export:
            export_case(case, force=args.force_export, verbose=True)
        report = run_case(
            case,
            export=False,
            use_metal=use_metal,
            use_mps=use_mps,
            geometry=args.geometry,
            reference_only=args.reference_only,
        )
        reports.append(report)
        passed = print_report(report, use_metal) and passed
    if args.json:
        Path(args.json).write_text(
            json.dumps(reports, indent=2, sort_keys=True, default=float),
            encoding="utf-8",
        )
        print(f"\nreport: {args.json}")
    if passed and not use_mps:
        print(
            "\nPASS: strict float32 parity (Metal pairs only, MPS not "
            "measured - not a full-gate pass)"
        )
    else:
        print("\n" + ("PASS: strict float32 parity" if passed else "FAIL: parity gate"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
