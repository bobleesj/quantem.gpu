"""Join scientific capability, retained evidence, and benchmark coverage.

This is a generated view, not another registry. Run ``backend_status.py check``
in CI, ``summary`` for the current blockers, or ``json --backend vulkan`` for
the complete records and their separate performance comparison conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .benchmark_registry import resolved_gates, resolved_measurements
else:
    from benchmark_registry import resolved_gates, resolved_measurements

ROOT = Path(__file__).resolve().parents[1]
GENERATED = Path("docs/_generated/backend_readiness.md")
BACKENDS = {
    "cpu-reference": "CPU reference",
    "cuda": "CUDA",
    "mps": "Python MPS",
    "swift-metal": "Native Swift/Metal",
    "webgpu": "WebGPU",
    "direct3d": "Direct3D",
    "vulkan": "Vulkan",
}
IMPLEMENTATION = {
    "reference": "implemented",
    "reference-fixture": "implemented",
    "required": "implemented",
    "required-hardware": "implemented",
    "partial-hardware": "partial",
    "not-implemented": "not-implemented",
}
EVIDENCE_STATES = {"ready", "evidence-gap", "unsupported"}
EVIDENCE_PROTOCOL = "quantem-gpu-cell-evidence/v1"
REQUIRED_OUTCOMES = {"scientific-parity", "real-data-e2e"}
REQUIRED_CAPABILITIES = {
    "geometry.scan-quarter-turn",
    "io.decode-bin-provenance",
    "io.selective-scan-loading",
    "detector.integer-products",
    "screening.prepared-products",
    "dpc.com-rotation-idpc",
    "display.transform-histogram-color-fft",
    "ssb.object-phase-loss",
    "ssb.calibration-200-nelder-mead",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}; retain one definition.")
        result[key] = value
    return result


def _read(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )


def _artifact_path(
    record: dict, root: Path, label: str, errors: list[str]
) -> Path | None:
    """Verify the identity of a retained artifact before inspecting its claims."""
    if not isinstance(record, dict):
        errors.append(f"{label}: artifact reference must contain path and sha256")
        return None
    relative = record.get("path", "")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        errors.append(f"{label}: artifact path must be repository-relative")
        return None
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        errors.append(f"{label}: artifact path must be inside the repository")
    elif not path.is_file():
        errors.append(f"{label}: missing retained evidence {relative}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
        errors.append(f"{label}: retained evidence digest differs for {relative}")
    else:
        return path
    return None


def _retained_evidence(cell: dict, policy: dict, root: Path, errors: list[str]) -> None:
    """Reject an untraceable ready claim instead of treating a test as evidence."""
    records = cell.get("retained_evidence", [])
    if cell["state"] == "ready" and not records:
        errors.append(
            f"{cell['id']}: ready requires retained_evidence, not a test path"
        )
    seen = set()
    for record in records:
        label = cell["id"]
        path = _artifact_path(record, root, label, errors)
        if path is None:
            continue
        if path in seen:
            errors.append(f"{label}: duplicate retained evidence {record['path']}")
        seen.add(path)
        try:
            run = _read(path)
        except ValueError as error:
            errors.append(f"{label}: structured run-evidence JSON required: {error}")
            continue
        if not isinstance(run, dict):
            errors.append(f"{label}: structured run-evidence object required")
            continue
        required_values = {
            "schema_version": 1,
            "protocol_version": policy["protocol_version"],
            "cell_id": cell["id"],
            "backend": cell["backend"],
            "runner": cell["runner"],
            "status": "passed",
        }
        for field, expected in required_values.items():
            if run.get(field) != expected:
                errors.append(f"{label}: run evidence {field} must be {expected!r}")
        if type(run.get("schema_version")) is not int:
            errors.append(f"{label}: run evidence schema_version must be an integer")
        if not isinstance(run.get("source_revision"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", run["source_revision"]
        ):
            errors.append(f"{label}: run evidence needs a full source_revision")
        fixture = run.get("fixture", {})
        if (
            not isinstance(fixture, dict)
            or not isinstance(fixture.get("id"), str)
            or not fixture["id"].strip()
            or not isinstance(fixture.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", fixture["sha256"])
        ):
            errors.append(f"{label}: run evidence needs fixture id and SHA-256")
        _artifact_path(run.get("result", {}), root, f"{label} result", errors)
        outcomes = run.get("outcomes", {})
        required_gates = set(policy["required_outcomes"]) | {cell["pr_gate"]}
        for gate in sorted(required_gates):
            if not isinstance(outcomes, dict) or outcomes.get(gate) != "passed":
                errors.append(
                    f"{label}: required gate {gate!r} must have passed outcome"
                )


def build_status(
    capability_matrix: dict,
    profile_matrix: dict,
    benchmark_registry: dict,
    *,
    root: Path = ROOT,
) -> dict:
    """Validate and derive readiness without promoting implementation to evidence.

    Parameters
    ----------
    capability_matrix, profile_matrix, benchmark_registry
        The three canonical repository registries, parsed from JSON.
    root
        Repository root used to verify retained evidence artifacts.

    Returns
    -------
    dict
        Per-backend cells, exact signoff blockers, and unaggregated measurements.

    Raises
    ------
    ValueError
        The registries disagree or a ready claim lacks retained evidence.

    Examples
    --------
    >>> # Inspect one runtime without substituting an untested fallback:
    >>> # status = build_status(capabilities, profiles, benchmarks)
    >>> # status["backends"]["vulkan"]["blocking_cells"]
    """
    errors = []
    policy = profile_matrix.get("required_evidence", {})
    if (
        policy.get("protocol_version") != EVIDENCE_PROTOCOL
        or not isinstance(policy.get("required_outcomes"), list)
        or any(
            not isinstance(gate, str) for gate in policy.get("required_outcomes", [])
        )
        or not REQUIRED_OUTCOMES.issubset(policy.get("required_outcomes", []))
    ):
        raise ValueError(
            "profile required_evidence must retain the versioned parity and real-data gates"
        )
    backend_names = capability_matrix["backends"]
    if len(backend_names) != len(set(backend_names)):
        errors.append("capability matrix repeats a backend")
    if set(backend_names) != set(BACKENDS):
        errors.append(
            "capability backends must retain every supported contract backend"
        )
    if set(profile_matrix["platforms"]) != set(BACKENDS):
        errors.append("profile platforms must match the capability backend set")

    capabilities = {}
    expected = {}
    for capability in capability_matrix["capabilities"]:
        capability_id = capability["id"]
        if capability_id in capabilities:
            errors.append(f"duplicate capability {capability_id}")
        capabilities[capability_id] = capability
        if set(capability["coverage"]) != set(BACKENDS):
            errors.append(f"{capability_id}: coverage must retain every backend")
        for backend, coverage in capability["coverage"].items():
            expected[f"{capability_id}::{backend}"] = coverage
            if coverage["level"] not in IMPLEMENTATION:
                errors.append(f"{capability_id}::{backend}: unknown support level")
    if set(profile_matrix["capabilities"]) != set(capabilities):
        errors.append("profile capability definitions differ from capability matrix")
    missing_capabilities = REQUIRED_CAPABILITIES - set(capabilities)
    if missing_capabilities:
        errors.append(
            f"required capabilities disappeared: {sorted(missing_capabilities)}"
        )

    cells = {}
    for cell in profile_matrix["cells"]:
        cell_id = cell["id"]
        if cell_id in cells:
            errors.append(f"duplicate profile cell {cell_id}")
        cells[cell_id] = cell
        if cell_id != f"{cell['capability']}::{cell['backend']}":
            errors.append(f"{cell_id}: ID differs from capability and backend")
        coverage = expected.get(cell_id)
        if coverage is None:
            errors.append(f"unexpected profile cell {cell_id}")
        elif cell["support_level"] != coverage["level"]:
            errors.append(f"{cell_id}: support level disagrees with capability matrix")
        if cell["state"] not in EVIDENCE_STATES:
            errors.append(f"{cell_id}: unknown evidence state {cell['state']!r}")
        if not isinstance(cell["release_signoff"], bool):
            errors.append(f"{cell_id}: release_signoff must be boolean")
        if cell["support_level"] == "not-implemented":
            if cell["state"] != "unsupported" or cell["release_signoff"]:
                errors.append(f"{cell_id}: unsupported capability cannot enter signoff")
        elif cell["state"] == "unsupported":
            errors.append(f"{cell_id}: implemented capability cannot be unsupported")
        if cell["support_level"] != "not-implemented" and cell[
            "runner"
        ] != profile_matrix["platforms"].get(cell["backend"], {}).get("runner"):
            errors.append(f"{cell_id}: runner disagrees with the profile platform")
        _retained_evidence(cell, policy, root, errors)
    for missing in sorted(set(expected) - set(cells)):
        errors.append(f"missing profile cell {missing}")

    gates = resolved_gates(benchmark_registry)
    measurements = resolved_measurements(benchmark_registry)
    for kind, rows in (("gate", gates), ("measurement", measurements)):
        seen = set()
        for row in rows:
            if row["id"] in seen:
                errors.append(f"duplicate benchmark {kind} {row['id']}")
            seen.add(row["id"])
            if row["platform"] not in BACKENDS.values():
                errors.append(
                    f"benchmark {kind} {row['id']}: unknown platform {row['platform']!r}"
                )
    if errors:
        raise ValueError("\n".join(errors))

    result = {"schema_version": 1, "scope": "package-engineering", "backends": {}}
    for backend in backend_names:
        backend_cells = []
        for capability_id in capabilities:
            cell_id = f"{capability_id}::{backend}"
            cell = dict(cells[cell_id])
            coverage = expected[cell_id]
            cell["implementation"] = IMPLEMENTATION[coverage["level"]]
            cell["evidence"] = cell.pop("state")
            cell["qualification"] = coverage.get("qualification", "")
            cell["gates"] = coverage["gates"]
            cell["signoff"] = (
                "not-scheduled"
                if not cell["release_signoff"]
                else "ready"
                if cell["evidence"] == "ready"
                else "blocked"
            )
            backend_cells.append(cell)
        blocking = [
            cell["id"] for cell in backend_cells if cell["signoff"] == "blocked"
        ]
        scoped = [cell for cell in backend_cells if cell["release_signoff"]]
        retained = [row for row in measurements if row["platform"] == BACKENDS[backend]]
        result["backends"][backend] = {
            "signoff": "blocked"
            if blocking
            else "ready"
            if scoped
            else "not-scheduled",
            "blocking_cells": blocking,
            "cells": backend_cells,
            "performance": {
                "state": "recorded" if retained else "not-recorded",
                "measurements": retained,
                "gates": [row for row in gates if row["platform"] == BACKENDS[backend]],
            },
        }
    return result


def render_status(status: dict) -> str:
    """Render the generated readiness view without a backend speed or score.

    Parameters
    ----------
    status
        The validated result of :func:`build_status`.

    Returns
    -------
    str
        Markdown shared by the CLI and generated documentation.

    Examples
    --------
    >>> render_status({"backends": {}}).startswith("<!-- Generated")
    True
    """
    lines = [
        "<!-- Generated by scripts/backend_status.py; do not edit. -->",
        "# Backend readiness",
        "",
        "Generated from the capability matrix, profiling matrix, and benchmark registry.",
        "Package-engineering signoff is not consumer-application or distribution approval.",
        "Ready means required evidence is retained; it does not mean every workload is qualified.",
        "Performance records remain separate, with exact workload and comparison conditions.",
        "Use `python scripts/backend_status.py json --backend BACKEND` for all records.",
        "",
        "| Backend | Package signoff | Performance records |",
        "|---|---|---|",
    ]
    for backend, entry in status["backends"].items():
        lines.append(
            f"| {backend} | {entry['signoff']} | {entry['performance']['state']} |"
        )
    for backend, entry in status["backends"].items():
        lines.extend(
            [
                "",
                f"## {backend}",
                "",
                "| Capability | Implementation | Evidence | Signoff |",
                "|---|---|---|---|",
            ]
        )
        for cell in entry["cells"]:
            lines.append(
                f"| {cell['capability']} | {cell['implementation']} | {cell['evidence']} | {cell['signoff']} |"
            )
        lines.extend(["", "Blocking signoff cells:", ""])
        lines.extend(f"- `{cell_id}`" for cell_id in entry["blocking_cells"])
        if not entry["blocking_cells"]:
            lines.append(
                "None in the scheduled package scope."
                if entry["signoff"] == "ready"
                else "No release signoff is scheduled."
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the repository CLI, for example ``backend_status.py summary``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "summary", "json", "render"))
    parser.add_argument("--backend", choices=tuple(BACKENDS))
    args = parser.parse_args()
    if args.backend and args.command != "json":
        parser.error("--backend is supported only with json")
    try:
        status = build_status(
            _read(ROOT / "tests/parity/backend_matrix.json"),
            _read(ROOT / "benchmarks/profile_matrix.json"),
            _read(ROOT / "benchmarks/benchmark_registry.json"),
        )
        rendered = render_status(status)
        if args.command == "check":
            if (
                not (ROOT / GENERATED).is_file()
                or (ROOT / GENERATED).read_text() != rendered
            ):
                raise ValueError(
                    "Generated readiness is stale; run scripts/backend_status.py render."
                )
            print(
                f"backend_status: OK - {len(status['backends'])} backends, "
                f"{sum(len(entry['cells']) for entry in status['backends'].values())} cells"
            )
        elif args.command == "render":
            (ROOT / GENERATED).write_text(rendered, encoding="utf-8")
        elif args.command == "summary":
            print(rendered, end="")
        else:
            value = status["backends"][args.backend] if args.backend else status
            print(json.dumps(value, indent=2))
    except (ValueError, KeyError, OSError) as error:
        print(f"backend_status: ERROR - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
