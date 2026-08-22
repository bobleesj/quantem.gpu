#!/usr/bin/env python
"""Inspect, validate, and render the QuantEM.GPU benchmark coverage registry.

The registry separates required coverage gates from retained measurements.  A
gate says what must be exercised; a measurement says what was actually run;
and a runbook provides the repository-owned entry point for repeating the
work.  This distinction prevents an absent configuration from disappearing
from the documentation and prevents a portable parity test from being
mistaken for a physical-device timing result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "benchmarks" / "benchmark_registry.json"
GENERATED_PATH = ROOT / "docs" / "_generated" / "benchmark_coverage.md"

STATE_ORDER = (
    "measured",
    "partial",
    "pending",
    "blocked",
    "refuted",
    "unsupported",
    "superseded",
)
ALLOWED_STATES = set(STATE_ORDER)
COMPLETE_STATES = {"measured"}
OPEN_STATES = {"partial", "pending", "blocked"}
FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")
FULL_SHA256 = re.compile(r"[0-9a-f]{64}")

PLATFORM_LABELS = {
    "cpu-reference": "CPU reference",
    "cuda": "CUDA",
    "mps": "Python MPS",
    "python-mps": "Python MPS",
    "swift-metal": "Native Swift/Metal",
    "native-swift-metal": "Native Swift/Metal",
    "webgpu": "WebGPU",
}

PLATFORM_ORDER = {
    "CUDA": 0,
    "Python MPS": 1,
    "Native Swift/Metal": 2,
    "WebGPU": 3,
    "CPU reference": 4,
}

MODULE_LABELS = {
    "loading": "I/O and load",
    "loading-and-products": "I/O and load",
    "selective-loading": "Selective loading",
    "screening": "Screening",
    "screening-cache-reopen": "Screening",
    "dpc": "CoM, DPC, and iDPC",
    "idpc": "CoM, DPC, and iDPC",
    "dpc-idpc": "CoM, DPC, and iDPC",
}

STATE_LABELS = {
    "measured": "✓ Measured",
    "partial": "◐ Partial",
    "pending": "○ Pending",
    "blocked": "! Blocked",
    "refuted": "× Refuted",
    "unsupported": "Not supported",
    "superseded": "↺ Superseded",
}

COMPUTER_LABELS = frozenset(
    {
        "Linux CUDA workstation (dual 96 GB Blackwell GPUs)",
        "MacBook Air (M2, 8 GB)",
        "MacBook Pro (M5, 24 GB)",
        "MacBook Pro (M5 Max, 128 GB)",
        "Portable CI runner",
    }
)

COMPUTER_SLUGS = {
    "Linux CUDA workstation (dual 96 GB Blackwell GPUs)": "linux-dual-blackwell-96gb",
    "MacBook Air (M2, 8 GB)": "macbook-air-m2-8gb",
    "MacBook Pro (M5, 24 GB)": "macbook-pro-m5-24gb",
    "MacBook Pro (M5 Max, 128 GB)": "macbook-pro-m5-max-128gb",
    "Portable CI runner": "portable-ci",
}

PASSING_ZERO_VIOLATION_PHRASES = (
    "zero tolerance violations",
    "0 tolerance violations",
    "no tolerance violations",
)


def _computer_label(device: Any, current: Any) -> str:
    """Return a reproducible hardware label instead of a local host nickname."""

    current_label = str(current or "")
    if current_label in COMPUTER_LABELS:
        return current_label

    device_label = str(device or "")
    if "Apple M5 Max" in device_label:
        return "MacBook Pro (M5 Max, 128 GB)"
    if "Apple M5 10-core" in device_label:
        return "MacBook Pro (M5, 24 GB)"
    if "Apple M2" in device_label:
        return "MacBook Air (M2, 8 GB)"
    if "NVIDIA RTX PRO 6000 Blackwell" in device_label:
        return "Linux CUDA workstation (dual 96 GB Blackwell GPUs)"
    if current_label.lower() == "portable ci":
        return "Portable CI runner"
    return current_label


def _public_measurement_id(raw_id: Any, computer: Any) -> str:
    """Replace a source-machine prefix with a reproducible hardware prefix."""

    value = str(raw_id or "")
    computer_slug = COMPUTER_SLUGS.get(str(computer or ""), "unspecified-device")
    if value.startswith(f"{computer_slug}-"):
        return value
    _, separator, suffix = value.partition("-")
    return f"{computer_slug}-{suffix if separator else value}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Return the raw benchmark registry."""

    return _read_json(path)


def _configuration(registry: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    key = gate.get("configuration")
    base = registry.get("configurations", {}).get(key, {})
    resolved = dict(base)
    resolved.update(gate)
    return resolved


def _artifact_lookup(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in evidence.get("artifacts", [])}


def _parity_adjudication_text(value: Any) -> tuple[str, bool]:
    """Remove explicitly passing zero-violation phrases before failure matching."""

    text = str(value or "").lower()
    zero_violations = any(phrase in text for phrase in PASSING_ZERO_VIOLATION_PHRASES)
    for phrase in PASSING_ZERO_VIOLATION_PHRASES:
        text = text.replace(phrase, "")
    return text, zero_violations


def _measurement_state(measurement: dict[str, Any]) -> str:
    explicit = measurement.get("state")
    if explicit in ALLOWED_STATES and explicit != "measured":
        return str(explicit)
    timing = (
        measurement.get("wall_p50_seconds"),
        measurement.get("wall_p95_seconds"),
        measurement.get("wall_max_seconds"),
    )
    parity = str(measurement.get("parity") or "").lower()
    adjudication_text, zero_violations = _parity_adjudication_text(parity)
    if any(
        marker in adjudication_text
        for marker in (
            "mismatch",
            "did not match",
            "failed parity",
            "parity failed",
            "tolerance violation",
        )
    ):
        return "refuted"
    if any(
        marker in parity
        for marker in (
            "not performed",
            "incomplete",
            "selected frames",
            "qualified probes",
            "probe only",
        )
    ):
        return "partial"
    positive_parity = zero_violations or any(
        marker in parity
        for marker in (
            "exact",
            "passed",
            "within tolerance",
            "byte-identical",
        )
    )
    if all(value is not None for value in timing) and positive_parity:
        return "measured"
    return "partial"


def _measurement_row(
    measurement: dict[str, Any],
    *,
    evidence_path: str,
    artifacts: dict[str, dict[str, Any]],
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    override = override or {}
    artifact = artifacts.get(str(measurement.get("artifact_id")), {})
    row_start = measurement.get("selected_scan_row_start")
    row_stop = measurement.get("selected_scan_row_stop")
    column_start = measurement.get("selected_scan_column_start")
    column_stop = measurement.get("selected_scan_column_stop")
    selected_rows = (
        int(row_stop) - int(row_start)
        if row_start is not None and row_stop is not None
        else measurement.get("source_scan_rows")
    )
    selected_columns = (
        int(column_stop) - int(column_start)
        if column_start is not None and column_stop is not None
        else measurement.get("source_scan_columns")
    )
    platform = str(measurement.get("platform") or "")
    module = MODULE_LABELS.get(
        str(measurement.get("functional_area") or ""),
        str(measurement.get("functional_area") or "Unclassified"),
    )
    device_peak = measurement.get("accelerator_peak_bytes")
    total_device_peak = measurement.get("total_card_peak_bytes")
    process_peak = measurement.get("process_peak_rss_bytes")
    if process_peak is None:
        process_peak = measurement.get("host_peak_rss_p50_bytes")
    if process_peak is None:
        process_peak = measurement.get("process_rss_p50_bytes")
    browser_peak = measurement.get("browser_tree_peak_rss_bytes")
    if browser_peak is not None:
        process_peak = browser_peak
    revision = artifact.get("implementation_revision") or artifact.get("observed_head")
    state = _measurement_state(measurement)
    computer = _computer_label(measurement.get("device"), measurement.get("computer"))
    public_measurement_id = _public_measurement_id(measurement["id"], computer)
    row: dict[str, Any] = {
        "id": f"evidence::{public_measurement_id}",
        "measurement_id": public_measurement_id,
        "source_measurement_id": measurement["id"],
        "module": module,
        "operation": str(measurement.get("functional_area") or "measurement"),
        "platform": PLATFORM_LABELS.get(platform, platform),
        "computer": computer,
        "device": measurement.get("device"),
        "source_scan_rows": measurement.get("source_scan_rows"),
        "source_scan_columns": measurement.get("source_scan_columns"),
        "selected_scan_rows": selected_rows,
        "selected_scan_columns": selected_columns,
        "source_detector_rows": measurement.get("source_detector_rows"),
        "source_detector_columns": measurement.get("source_detector_columns"),
        "scan_bin": measurement.get("scan_bin"),
        "detector_bin": measurement.get("detector_bin"),
        "output_detector_rows": measurement.get("working_detector_rows"),
        "output_detector_columns": measurement.get("working_detector_columns"),
        "source_dtype": measurement.get("source_dtype"),
        "working_dtype": measurement.get("working_dtype"),
        "cache_state": measurement.get("cache_state"),
        "timing_boundary": measurement.get("timing_boundary"),
        "sample_count": measurement.get("sample_count"),
        "p50_seconds": measurement.get("wall_p50_seconds"),
        "p95_seconds": measurement.get("wall_p95_seconds"),
        "max_seconds": measurement.get("wall_max_seconds"),
        "logical_resident_bytes": measurement.get("logical_resident_bytes"),
        "driver_allocated_after_load_bytes": measurement.get(
            "driver_allocated_after_load_bytes"
        ),
        "driver_allocated_after_release_bytes": measurement.get(
            "driver_allocated_after_release_bytes"
        ),
        "accelerator_peak_bytes": device_peak,
        "total_device_peak_bytes": total_device_peak,
        "process_tree_peak_bytes": process_peak,
        "swap_delta_bytes": measurement.get("swap_delta_bytes"),
        "parity": measurement.get("parity"),
        "state": state,
        "tested_date": measurement.get("date"),
        "source_revision": revision,
        "fixture_id": measurement.get("fixture_id"),
        "fixture_sha256": measurement.get("fixture_sha256"),
        "source_identity_sha256": measurement.get("source_identity_sha256"),
        "evidence": f"{evidence_path}#{measurement['id']}",
        "artifact_id": measurement.get("artifact_id"),
    }
    row.update(override)
    return row


def resolved_measurements(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve imported and registry-local measurement rows."""

    rows: list[dict[str, Any]] = []
    for source in registry.get("evidence_imports", []):
        relative = source["path"]
        evidence = _read_json(ROOT / relative)
        artifacts = _artifact_lookup(evidence)
        overrides = source.get("measurement_overrides", {})
        for measurement in evidence.get("measurements", []):
            computer = _computer_label(
                measurement.get("device"), measurement.get("computer")
            )
            public_measurement_id = _public_measurement_id(measurement["id"], computer)
            rows.append(
                _measurement_row(
                    measurement,
                    evidence_path=relative,
                    artifacts=artifacts,
                    override=overrides.get(public_measurement_id),
                )
            )
    for source_row in registry.get("additional_measurements", []):
        row = dict(source_row)
        row["computer"] = _computer_label(row.get("device"), row.get("computer"))
        row["measurement_id"] = _public_measurement_id(
            row.get("measurement_id"), row["computer"]
        )
        row["id"] = f"evidence::{row['measurement_id']}"
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["id"]))


def resolved_gates(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return coverage gates with named configurations expanded."""

    return [_configuration(registry, gate) for gate in registry.get("gates", [])]


def _is_positive_int_or_none(value: Any) -> bool:
    return value is None or (isinstance(value, int) and value > 0)


def validate_registry(
    errors: list[str] | None = None,
    *,
    check_render: bool = True,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, int]:
    """Validate the registry and return summary counts.

    Parameters
    ----------
    errors
        Optional error collector. When omitted, a ``ValueError`` is raised on
        failure.
    check_render
        Require the generated documentation fragment to match the registry.
    registry_path
        Registry to validate.
    """

    own_errors = errors is None
    failures: list[str] = [] if errors is None else errors
    registry = load_registry(registry_path)
    if registry.get("schema_version") != 1:
        failures.append("benchmark registry schema_version must be 1")
    if registry.get("protocol_version") != "quantem-gpu-benchmark-coverage/v1":
        failures.append("benchmark registry protocol_version must be frozen at v1")

    runbooks = registry.get("runbooks", {})
    for runbook_id, runbook in runbooks.items():
        required = {
            "title",
            "owner",
            "tier",
            "evidence_level",
            "command",
            "preflight",
            "required_environment",
            "required_artifacts",
        }
        missing = required - set(runbook)
        if missing:
            failures.append(f"runbook {runbook_id} is missing {sorted(missing)}")
        command = runbook.get("command")
        if not isinstance(command, str) or not command.strip():
            failures.append(f"runbook {runbook_id} command must be a nonempty string")
        elif "\n" in command:
            failures.append(f"runbook {runbook_id} command must stay one line")
        if not isinstance(runbook.get("preflight"), list):
            failures.append(f"runbook {runbook_id} preflight must be a list")

    measurements = resolved_measurements(registry)
    measurement_ids = {row["measurement_id"] for row in measurements}
    measurement_states = {row["measurement_id"]: row["state"] for row in measurements}
    if len(measurement_ids) != len(measurements):
        failures.append("retained measurement IDs must be unique")
    fixture_masters: dict[str, set[str]] = {}
    for row in measurements:
        label = f"measurement {row['id']}"
        if row.get("computer") not in COMPUTER_LABELS:
            failures.append(
                f"{label} computer must identify a supported hardware configuration"
            )
        if row.get("state") not in ALLOWED_STATES:
            failures.append(f"{label} has invalid state {row.get('state')}")
        revision = row.get("source_revision")
        if revision is not None and not FULL_GIT_SHA.fullmatch(str(revision)):
            failures.append(f"{label} source revision must be a full Git SHA")
        fixture_hash = row.get("fixture_sha256")
        if fixture_hash is not None and not FULL_SHA256.fullmatch(str(fixture_hash)):
            failures.append(f"{label} fixture SHA-256 is invalid")
        source_identity_hash = row.get("source_identity_sha256")
        if source_identity_hash is not None and not FULL_SHA256.fullmatch(
            str(source_identity_hash)
        ):
            failures.append(f"{label} source-identity SHA-256 is invalid")
        fixture_id = row.get("fixture_id")
        if fixture_id and fixture_hash:
            fixture_masters.setdefault(str(fixture_id), set()).add(str(fixture_hash))
        if row.get("state") == "measured":
            for field in ("p50_seconds", "p95_seconds", "max_seconds"):
                value = row.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    failures.append(f"{label} measured row lacks finite {field}")
            if not row.get("parity"):
                failures.append(f"{label} measured row lacks parity")
            if not row.get("tested_date") or not row.get("device"):
                failures.append(f"{label} measured row lacks date or device")
    for fixture_id, master_hashes in fixture_masters.items():
        if len(master_hashes) > 1:
            failures.append(
                f"fixture {fixture_id} maps to multiple master SHA-256 values"
            )

    gates = resolved_gates(registry)
    gate_ids: set[str] = set()
    required_gate_fields = {
        "id",
        "module",
        "operation",
        "platform",
        "computer",
        "configuration",
        "state",
        "runbook",
        "priority",
        "next_gate",
        "source_scan_rows",
        "source_scan_columns",
        "selected_scan_rows",
        "selected_scan_columns",
        "source_detector_rows",
        "source_detector_columns",
        "scan_bin",
        "detector_bin",
        "output_detector_rows",
        "output_detector_columns",
        "source_dtype",
        "working_dtype",
        "cache_state",
        "timing_boundary",
    }
    for gate in gates:
        label = f"gate {gate.get('id', '<missing>')}"
        missing = required_gate_fields - set(gate)
        if missing:
            failures.append(f"{label} is missing {sorted(missing)}")
            continue
        if gate["id"] in gate_ids:
            failures.append(f"duplicate gate {gate['id']}")
        gate_ids.add(gate["id"])
        if gate.get("computer") not in COMPUTER_LABELS:
            failures.append(
                f"{label} computer must identify a supported hardware configuration"
            )
        if gate["state"] not in ALLOWED_STATES:
            failures.append(f"{label} has invalid state {gate['state']}")
        if gate["runbook"] not in runbooks:
            failures.append(f"{label} names unknown runbook {gate['runbook']}")
        if not isinstance(gate["priority"], int) or not 1 <= gate["priority"] <= 5:
            failures.append(f"{label} priority must be an integer from 1 to 5")
        for field in (
            "source_scan_rows",
            "source_scan_columns",
            "selected_scan_rows",
            "selected_scan_columns",
            "source_detector_rows",
            "source_detector_columns",
            "scan_bin",
            "detector_bin",
            "output_detector_rows",
            "output_detector_columns",
        ):
            if not _is_positive_int_or_none(gate[field]):
                failures.append(f"{label} {field} must be a positive integer or null")
        if (
            gate["source_detector_rows"] is not None
            and gate["detector_bin"] is not None
        ):
            expected_rows = gate["source_detector_rows"] // gate["detector_bin"]
            expected_columns = gate["source_detector_columns"] // gate["detector_bin"]
            if gate["output_detector_rows"] != expected_rows:
                failures.append(
                    f"{label} output detector rows do not match detector bin"
                )
            if gate["output_detector_columns"] != expected_columns:
                failures.append(
                    f"{label} output detector columns do not match detector bin"
                )
        satisfied = gate.get("satisfied_by", [])
        unknown = set(satisfied) - measurement_ids
        if unknown:
            failures.append(f"{label} names unknown measurements {sorted(unknown)}")
        if gate["state"] in COMPLETE_STATES and not satisfied:
            failures.append(f"{label} measured gate needs satisfied_by evidence")
        if gate["state"] in COMPLETE_STATES:
            incomplete = {
                measurement_id: measurement_states.get(measurement_id)
                for measurement_id in satisfied
                if measurement_states.get(measurement_id) not in COMPLETE_STATES
            }
            if incomplete:
                failures.append(
                    f"{label} measured gate names non-measured evidence {incomplete}"
                )
        if gate["state"] in OPEN_STATES and not str(gate["next_gate"]).strip():
            failures.append(f"{label} open gate needs a concrete next_gate")
        if gate["state"] == "unsupported":
            if not str(gate.get("reason") or "").strip():
                failures.append(f"{label} unsupported gate needs a reason")
            if satisfied:
                failures.append(f"{label} unsupported gate cannot name timing evidence")

    if check_render and GENERATED_PATH.is_file():
        expected = render_document(registry)
        actual = GENERATED_PATH.read_text(encoding="utf-8")
        if actual != expected:
            failures.append(
                "generated benchmark coverage is stale; run "
                "python scripts/benchmark_registry.py render"
            )
    elif check_render:
        failures.append(
            "generated benchmark coverage is missing; run "
            "python scripts/benchmark_registry.py render"
        )

    if own_errors and failures:
        raise ValueError("\n".join(failures))
    return {
        "gates": len(gates),
        "measurements": len(measurements),
        "runbooks": len(runbooks),
    }


def _escape(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _shape(rows: Any, columns: Any) -> str:
    if rows is None or columns is None:
        return "n/a"
    return f"{rows} × {columns}"


def _seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    seconds = float(value)
    if seconds < 0.1:
        return f"{seconds * 1000:.3f} ms"
    return f"{seconds:.6f} s"


def _bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    count = int(value)
    if count == 0:
        return "0 B"
    return f"{count / (1 << 30):.3f} GiB"


def _parity_label(value: Any) -> str:
    if not value:
        return "Pending"
    lower = str(value).lower()
    if "not performed" in lower or "incomplete" in lower:
        return "Qualified probes"
    adjudication_text, _ = _parity_adjudication_text(lower)
    if any(
        marker in adjudication_text
        for marker in ("failed", "mismatch", "tolerance violation")
    ):
        return "Failed"
    return "Pass"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        output.append("| " + " | ".join(_escape(value) for value in row) + " |")
    return "\n".join(output)


def _platform_sort_key(platform: Any) -> tuple[int, str]:
    """Return a stable accelerator-first ordering for benchmark platforms."""

    label = str(platform or "")
    return PLATFORM_ORDER.get(label, len(PLATFORM_ORDER)), label


def _platform_computer_rows(gates: list[dict[str, Any]]) -> list[list[Any]]:
    """Summarize required coverage without hiding unmeasured configurations."""

    counts: dict[tuple[str, str], Counter[str]] = {}
    for gate in gates:
        key = gate["platform"], gate["computer"]
        counts.setdefault(key, Counter())[gate["state"]] += 1

    rows: list[list[Any]] = []
    for (platform, computer), states in sorted(
        counts.items(),
        key=lambda item: (*_platform_sort_key(item[0][0]), item[0][1]),
    ):
        rows.append(
            [
                platform,
                computer,
                sum(states.values()),
                states["measured"],
                states["partial"],
                states["pending"],
                states["blocked"],
                states["refuted"],
                states["unsupported"],
            ]
        )
    return rows


def _gate_rows(registry: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for gate in sorted(
        resolved_gates(registry),
        key=lambda item: (
            *_platform_sort_key(item["platform"]),
            item["computer"],
            item["module"],
            item["priority"],
            item["id"],
        ),
    ):
        rows.append(
            [
                gate["platform"],
                gate["computer"],
                STATE_LABELS[gate["state"]],
                gate["module"],
                gate["operation"],
                _shape(gate["selected_scan_rows"], gate["selected_scan_columns"]),
                _shape(gate["source_detector_rows"], gate["source_detector_columns"]),
                gate["detector_bin"],
                _shape(gate["output_detector_rows"], gate["output_detector_columns"]),
                gate["source_dtype"],
                gate["working_dtype"],
                gate["cache_state"],
                gate["timing_boundary"],
                gate["priority"],
                f"`{gate['runbook']}`",
                gate["next_gate"] or gate.get("reason"),
            ]
        )
    return rows


def _measurement_rows(registry: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in sorted(
        resolved_measurements(registry),
        key=lambda row: (
            *_platform_sort_key(row["platform"]),
            str(row.get("computer")),
            row["module"],
            str(row["id"]),
        ),
    ):
        revision = item.get("source_revision")
        rows.append(
            [
                item["platform"],
                item["computer"],
                STATE_LABELS[item["state"]],
                item["module"],
                item["operation"],
                _shape(
                    item.get("selected_scan_rows"), item.get("selected_scan_columns")
                ),
                _shape(
                    item.get("source_detector_rows"),
                    item.get("source_detector_columns"),
                ),
                item.get("detector_bin"),
                _shape(
                    item.get("output_detector_rows"),
                    item.get("output_detector_columns"),
                ),
                item.get("source_dtype"),
                item.get("working_dtype"),
                item.get("cache_state"),
                item.get("timing_boundary"),
                item.get("sample_count"),
                _seconds(item.get("p50_seconds")),
                _seconds(item.get("p95_seconds")),
                _seconds(item.get("max_seconds")),
                _bytes(item.get("logical_resident_bytes")),
                _bytes(item.get("driver_allocated_after_load_bytes")),
                _bytes(item.get("driver_allocated_after_release_bytes")),
                _bytes(item.get("accelerator_peak_bytes")),
                _bytes(item.get("total_device_peak_bytes")),
                _bytes(item.get("process_tree_peak_bytes")),
                _bytes(item.get("swap_delta_bytes")),
                _parity_label(item.get("parity")),
                item.get("device"),
                item.get("tested_date"),
                f"`{revision}`" if revision else None,
                item.get("fixture_id"),
                (f"`{item['fixture_sha256']}`" if item.get("fixture_sha256") else None),
                (
                    f"`{item['source_identity_sha256']}`"
                    if item.get("source_identity_sha256")
                    else None
                ),
                f"`{item['measurement_id']}`",
            ]
        )
    return rows


def render_document(registry: dict[str, Any]) -> str:
    """Return the generated Markdown coverage fragment."""

    gates = resolved_gates(registry)
    gate_states = Counter(gate["state"] for gate in gates)
    platform_counts = Counter(gate["platform"] for gate in gates)
    lines = [
        "<!-- Generated by scripts/benchmark_registry.py; do not edit by hand. -->",
        "",
        "## Coverage summary",
        "",
        "<!-- benchmark-coverage-summary-start -->",
        "",
        _markdown_table(
            ["State", "Gate count"],
            [[STATE_LABELS[state], gate_states.get(state, 0)] for state in STATE_ORDER],
        ),
        "",
        _markdown_table(
            ["Platform", "Tracked gates"],
            [
                [platform, platform_counts[platform]]
                for platform in sorted(platform_counts, key=_platform_sort_key)
            ],
        ),
        "",
        "### Platform and computer coverage",
        "",
        "Each row identifies one reproducible hardware configuration. Counts describe tracked cells, including explicit unsupported contracts; a pending value remains a test to run.",
        "Load, admission, memory, and performance gates are multiplied across compatible computers because hardware changes the result. Platform-wide correctness or unsupported contracts are recorded once instead of creating misleading duplicate hardware rows.",
        "",
        _markdown_table(
            [
                "Platform",
                "Computer",
                "Tracked cells",
                "Measured",
                "Partial",
                "Pending",
                "Blocked",
                "Refuted",
                "Unsupported",
            ],
            _platform_computer_rows(gates),
        ),
        "",
        "<!-- benchmark-coverage-summary-end -->",
        "",
        "## Required coverage gates",
        "",
        "Every row is one exact scientific and device configuration. A pending row is work to do, not an implicit failure and not evidence that a backend is supported on that device.",
        "",
        _markdown_table(
            [
                "Platform",
                "Computer",
                "State",
                "Module",
                "Operation",
                "Selected scan",
                "Source detector",
                "Detector bin",
                "Output detector",
                "Source dtype",
                "Working dtype",
                "Cache/process state",
                "Wall boundary",
                "Priority",
                "Runbook",
                "Next gate",
            ],
            _gate_rows(registry),
        ),
        "",
        "## Retained atomic measurements",
        "",
        "These rows are imported from immutable evidence or explicitly registered follow-up evidence. A partial row is retained but does not satisfy a complete timing-and-parity gate.",
        "",
        _markdown_table(
            [
                "Platform",
                "Computer",
                "State",
                "Module",
                "Operation",
                "Selected scan",
                "Source detector",
                "Detector bin",
                "Output detector",
                "Source dtype",
                "Working dtype",
                "Cache/process state",
                "Wall boundary",
                "Samples",
                "p50",
                "p95",
                "Maximum",
                "Logical resident",
                "Driver allocated after load",
                "Driver allocated after release",
                "Accelerator peak",
                "Total-device peak",
                "Process/tree peak",
                "Swap delta",
                "Parity",
                "Device tested",
                "Date tested",
                "Revision",
                "Fixture ID",
                "Master SHA-256",
                "Source identity SHA-256",
                "Measurement ID",
            ],
            _measurement_rows(registry),
        ),
        "",
        "## Reproducible runbooks",
        "",
        _markdown_table(
            [
                "Runbook",
                "Owner",
                "Tier",
                "Evidence level",
                "Command",
                "Required artifacts",
            ],
            [
                [
                    f"`{runbook_id}`",
                    runbook["owner"],
                    runbook["tier"],
                    runbook["evidence_level"],
                    f"`{runbook['command']}`",
                    "; ".join(runbook["required_artifacts"]),
                ]
                for runbook_id, runbook in sorted(registry["runbooks"].items())
            ],
        ),
        "",
    ]
    return "\n".join(lines)


class _SafeFormat(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _find_gate(registry: dict[str, Any], gate_id: str) -> dict[str, Any]:
    for gate in resolved_gates(registry):
        if gate["id"] == gate_id:
            return gate
    raise SystemExit(f"Unknown benchmark gate: {gate_id}")


def _command_values(gate: dict[str, Any]) -> _SafeFormat:
    """Return gate fields plus lifecycle values required by command templates."""

    values = _SafeFormat(gate)
    cache_state = str(gate.get("cache_state") or "").lower()
    values["warmup_count"] = 0 if cache_state.startswith("cold ") else 1
    return values


def _filters_match(item: dict[str, Any], args: argparse.Namespace) -> bool:
    for argument, field in (
        (args.state, "state"),
        (args.platform, "platform"),
        (args.module, "module"),
        (args.computer, "computer"),
    ):
        if argument and str(item.get(field, "")).lower() != argument.lower():
            return False
    return True


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", choices=sorted(ALLOWED_STATES))
    parser.add_argument("--platform")
    parser.add_argument("--module")
    parser.add_argument("--computer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate registry and generated docs."
    )
    validate.add_argument("--no-render-check", action="store_true")

    list_parser = subparsers.add_parser("list", help="List required coverage gates.")
    _add_filters(list_parser)

    next_parser = subparsers.add_parser(
        "next", help="List the highest-priority open gates."
    )
    _add_filters(next_parser)
    next_parser.add_argument("--limit", type=int, default=10)
    next_parser.add_argument(
        "--performance-entrypoint-only",
        action="store_true",
        help="Hide parity-only runbooks.",
    )

    show = subparsers.add_parser("show", help="Show one resolved coverage gate.")
    show.add_argument("gate_id")

    command = subparsers.add_parser(
        "command", help="Show one gate's preflight and command."
    )
    command.add_argument("gate_id")

    render = subparsers.add_parser("render", help="Regenerate the documentation table.")
    render.add_argument("--output", type=Path, default=GENERATED_PATH)
    return parser


def main() -> int:
    args = _parser().parse_args()
    registry = load_registry()
    if args.action == "validate":
        errors: list[str] = []
        counts = validate_registry(errors, check_render=not args.no_render_check)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(
            "benchmark_registry: OK -- "
            f"{counts['gates']} gates, {counts['measurements']} measurements, "
            f"and {counts['runbooks']} runbooks"
        )
        return 0
    if args.action == "render":
        output = args.output
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_document(registry), encoding="utf-8")
        print(output)
        return 0
    if args.action == "show":
        print(json.dumps(_find_gate(registry, args.gate_id), indent=2, sort_keys=True))
        return 0
    if args.action == "command":
        gate = _find_gate(registry, args.gate_id)
        runbook = registry["runbooks"][gate["runbook"]]
        values = _command_values(gate)
        print(f"Gate: {gate['id']}")
        print(f"State: {STATE_LABELS[gate['state']]}")
        print(f"Runbook: {gate['runbook']} -- {runbook['title']}")
        print(f"Evidence level: {runbook['evidence_level']}")
        print("Preflight:")
        for step in runbook["preflight"]:
            print(f"  - {step}")
        if runbook["required_environment"]:
            print("Required environment:")
            for name in runbook["required_environment"]:
                print(f"  - {name}")
        print("Command:")
        print(runbook["command"].format_map(values))
        print("Required artifacts:")
        for artifact in runbook["required_artifacts"]:
            print(f"  - {artifact}")
        print("Promotion boundary:")
        print(
            f"  {runbook.get('promotion_boundary', 'Human evidence review required.')}"
        )
        return 0

    gates = [gate for gate in resolved_gates(registry) if _filters_match(gate, args)]
    if args.action == "next":
        gates = [gate for gate in gates if gate["state"] in OPEN_STATES]
        if args.performance_entrypoint_only:
            gates = [
                gate
                for gate in gates
                if registry["runbooks"][gate["runbook"]]["evidence_level"]
                == "physical performance"
            ]
        gates.sort(
            key=lambda gate: (
                gate["priority"],
                *_platform_sort_key(gate["platform"]),
                gate["computer"],
                gate["module"],
                gate["id"],
            )
        )
        gates = gates[: max(0, args.limit)]
    else:
        gates.sort(
            key=lambda gate: (
                *_platform_sort_key(gate["platform"]),
                gate["computer"],
                gate["module"],
                gate["id"],
            )
        )
    for gate in gates:
        print(
            f"{gate['id']}\t{STATE_LABELS[gate['state']]}\t"
            f"{gate['platform']}\t{gate['computer']}\t{gate['runbook']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
