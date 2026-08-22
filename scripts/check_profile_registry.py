"""Validate the cross-platform profiling plan and retained experiment records."""

import json
import re
from pathlib import Path

from benchmark_registry import validate_registry

ROOT = Path(__file__).resolve().parents[1]
PROFILE_MATRIX = ROOT / "benchmarks" / "profile_matrix.json"
PARITY_MATRIX = ROOT / "tests" / "parity" / "backend_matrix.json"
EXPERIMENT_ROOT = ROOT / "experiments"
RUNS_INDEX = EXPERIMENT_ROOT / "RUNS.md"

FULL_SHA256 = re.compile(r"[0-9a-f]{64}")
FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")
ALLOWED_SCHEDULES = {"weekly", "none"}
ALLOWED_STATES = {"ready", "evidence-gap", "unsupported"}
ALLOWED_STATUSES = {
    "planned",
    "running",
    "completed",
    "failed",
    "refuted",
    "superseded",
}
TERMINAL_STATUSES = {"completed", "failed", "refuted", "superseded"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def _validate_profile_matrix(errors: list[str]) -> tuple[int, set[str]]:
    profile = _read_json(PROFILE_MATRIX)
    parity = _read_json(PARITY_MATRIX)

    if profile.get("schema_version") != 1:
        errors.append("profile matrix schema_version must be 1")
    if profile.get("protocol_version") != "quantem-gpu-profile/v1":
        errors.append("profile matrix protocol_version is not frozen at v1")
    if profile.get("parity_matrix") != "tests/parity/backend_matrix.json":
        errors.append("profile matrix must name the canonical parity matrix")

    expected = {
        (capability["id"], backend): entry["level"]
        for capability in parity["capabilities"]
        for backend, entry in capability["coverage"].items()
    }
    cells = profile.get("cells", [])
    seen: dict[tuple[str, str], str] = {}

    for index, cell in enumerate(cells):
        label = f"profile cell {index}"
        required = {
            "id",
            "capability",
            "backend",
            "support_level",
            "pr_gate",
            "scheduled_profile",
            "release_signoff",
            "runner",
            "state",
        }
        missing = required - set(cell)
        if missing:
            errors.append(f"{label} is missing {sorted(missing)}")
            continue

        key = (cell["capability"], cell["backend"])
        expected_id = f"{key[0]}::{key[1]}"
        if cell["id"] != expected_id:
            errors.append(f"{label} id must be {expected_id}")
        if key in seen:
            errors.append(f"duplicate profile cell for {key}")
        seen[key] = cell["support_level"]

        if key not in expected:
            errors.append(f"profile cell {cell['id']} is not in the parity matrix")
        elif cell["support_level"] != expected[key]:
            errors.append(
                f"profile cell {cell['id']} support level "
                f"{cell['support_level']} != {expected[key]}"
            )
        if cell["scheduled_profile"] not in ALLOWED_SCHEDULES:
            errors.append(f"profile cell {cell['id']} has invalid schedule")
        if cell["state"] not in ALLOWED_STATES:
            errors.append(f"profile cell {cell['id']} has invalid state")
        if not isinstance(cell["release_signoff"], bool):
            errors.append(f"profile cell {cell['id']} release_signoff must be boolean")

        unsupported = cell["support_level"] == "not-implemented"
        if unsupported:
            if cell["state"] != "unsupported":
                errors.append(f"unsupported cell {cell['id']} must stay unsupported")
            if cell["scheduled_profile"] != "none" or cell["release_signoff"]:
                errors.append(f"unsupported cell {cell['id']} cannot schedule timing")
            if cell["pr_gate"] != "unsupported-contract":
                errors.append(
                    f"unsupported cell {cell['id']} needs a fail-closed PR gate"
                )
        elif cell["state"] == "unsupported":
            errors.append(f"implemented cell {cell['id']} cannot be unsupported")

    missing_cells = set(expected) - set(seen)
    extra_cells = set(seen) - set(expected)
    if missing_cells:
        errors.append(f"profile matrix is missing cells: {sorted(missing_cells)}")
    if extra_cells:
        errors.append(f"profile matrix has extra cells: {sorted(extra_cells)}")

    capability_ids = set(profile.get("capabilities", {}))
    parity_ids = {capability["id"] for capability in parity["capabilities"]}
    if capability_ids != parity_ids:
        errors.append(
            "profile capability definitions must equal the parity capabilities"
        )
    for capability_id, definition in profile.get("capabilities", {}).items():
        if not definition.get("required_metrics"):
            errors.append(f"profile capability {capability_id} has no required metrics")

    contract = profile.get("comparison_contract", {})
    if contract.get("cross_key_comparison") != "forbidden":
        errors.append("cross-key performance comparisons must remain forbidden")
    if contract.get("parity_failure") != "block":
        errors.append("scientific parity failures must block promotion")
    if contract.get("default_regression_mode") != "report-only":
        errors.append("new hardware baselines must begin in report-only mode")
    if contract.get("minimum_accepted_sessions_before_hard_threshold", 0) < 5:
        errors.append("hard timing thresholds need at least five accepted sessions")

    return len(cells), {cell["id"] for cell in cells if "id" in cell}


def _validate_experiments(errors: list[str]) -> tuple[int, set[str]]:
    manifest_paths = sorted(EXPERIMENT_ROOT.glob("*/manifest.json"))
    experiment_ids: set[str] = set()

    for path in manifest_paths:
        manifest = _read_json(path)
        experiment_id = manifest.get("experiment_id", "<missing>")
        experiment_ids.add(experiment_id)
        label = f"experiment {experiment_id}"

        required = {
            "schema_version",
            "experiment_id",
            "status",
            "question",
            "paper",
            "code",
            "inputs",
            "parameters",
            "execution",
            "outputs",
            "timestamps",
        }
        missing = required - set(manifest)
        if missing:
            errors.append(f"{label} is missing {sorted(missing)}")
            continue
        if manifest["schema_version"] != 1:
            errors.append(f"{label} schema_version must be 1")
        if experiment_id != path.parent.name:
            errors.append(f"{label} id must match its directory")
        if manifest["status"] not in ALLOWED_STATUSES:
            errors.append(f"{label} has invalid status {manifest['status']}")
        if not manifest["question"].strip():
            errors.append(f"{label} needs a falsifiable question")

        revision = manifest["code"].get("revision", "")
        if not FULL_GIT_SHA.fullmatch(revision):
            errors.append(f"{label} code revision must be a full Git SHA")
        dirty = manifest["code"].get("dirty")
        if not isinstance(dirty, bool):
            errors.append(f"{label} code.dirty must be boolean")
        if dirty and not FULL_SHA256.fullmatch(
            manifest["code"].get("diff_sha256") or ""
        ):
            errors.append(f"{label} dirty source needs a diff SHA-256")

        for item in manifest["inputs"]:
            if not FULL_SHA256.fullmatch(item.get("sha256", "")):
                errors.append(f"{label} input {item.get('dataset_id')} lacks SHA-256")
        if manifest["status"] in TERMINAL_STATUSES:
            if not manifest["timestamps"].get("finished"):
                errors.append(f"{label} terminal status needs a finished timestamp")
            if not manifest["outputs"]:
                errors.append(f"{label} terminal status needs retained outputs")
        for item in manifest["outputs"]:
            artifact_hash = item.get("sha256") or item.get("sha256_manifest", "")
            if not FULL_SHA256.fullmatch(artifact_hash):
                errors.append(f"{label} output {item.get('artifact')} lacks SHA-256")
            if not item.get("result", "").strip():
                errors.append(f"{label} output {item.get('artifact')} lacks a result")

        for value in _all_strings(manifest):
            lower = value.lower()
            if "/users/" in lower or "/home/" in lower:
                errors.append(f"{label} exposes an absolute private path")
                break

    return len(manifest_paths), experiment_ids


def _validate_runs_index(experiment_ids: set[str], errors: list[str]) -> None:
    if not RUNS_INDEX.is_file():
        errors.append("experiments/RUNS.md is missing")
        return
    text = RUNS_INDEX.read_text(encoding="utf-8")
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("| 20"):
            continue
        fields = [field.strip() for field in line.strip("|").split("|")]
        if len(fields) >= 4:
            rows.setdefault(fields[0], []).append(fields[3])

    status_map = {
        "planned": "planned",
        "running": "running",
        "completed": "ok",
        "failed": "failed",
        "refuted": "refuted",
        "superseded": "superseded",
    }
    for experiment_id in sorted(experiment_ids):
        statuses = rows.get(experiment_id, [])
        if not statuses:
            errors.append(f"experiments/RUNS.md is missing {experiment_id}")
            continue
        if len(statuses) != 1:
            errors.append(f"experiments/RUNS.md repeats {experiment_id}")
            continue
        manifest = _read_json(EXPERIMENT_ROOT / experiment_id / "manifest.json")
        expected = status_map[manifest["status"]]
        if not statuses[0].startswith(expected):
            errors.append(
                f"experiments/RUNS.md status for {experiment_id} must start with "
                f"{expected}"
            )


def main() -> int:
    errors: list[str] = []
    cell_count, _ = _validate_profile_matrix(errors)
    experiment_count, experiment_ids = _validate_experiments(errors)
    _validate_runs_index(experiment_ids, errors)
    coverage = validate_registry(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        "check_profile_registry: OK -- "
        f"{cell_count} platform/module cells and "
        f"{experiment_count} retained experiments; "
        f"{coverage['gates']} exact benchmark gates, "
        f"{coverage['measurements']} measurements, and "
        f"{coverage['runbooks']} runbooks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
