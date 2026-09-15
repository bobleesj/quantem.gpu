"""Run the exact seven-source field-parallel polar-query A/B/A test."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = ROOT / (
    "experiments/20260913-apple-m5-ans-scan512-striped-accumulation-retry2/run.py"
)
SPEC = importlib.util.spec_from_file_location("polar_field_parallel4_base", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load validated scan512 runner: {BASE_RUNNER}")
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)

base = retry.base
base.ARMS = (
    ("A1", "A1", "scan512"),
    ("B", "candidate", "scan512-field4"),
    ("A2", "A2", "scan512"),
)
fingerprint_code = retry._fingerprint_code


def fingerprint_experiment_runner(root: Path, executable: Path, manifest: dict) -> None:
    fingerprint_code(root, executable, manifest, runner=Path(__file__).resolve())


base._fingerprint_code = fingerprint_experiment_runner

_record_outputs = base._record_outputs


def retain_prior_attempt_outputs(root: Path, out: Path, manifest: dict) -> None:
    prior_outputs = list(manifest.get("outputs", []))
    _record_outputs(root, out, manifest)
    current_outputs = manifest.get("outputs", [])
    existing_paths = {item.get("path") for item in prior_outputs}
    manifest["outputs"] = prior_outputs + [
        item for item in current_outputs if item.get("path") not in existing_paths
    ]


base._record_outputs = retain_prior_attempt_outputs


if __name__ == "__main__":
    base.main()
