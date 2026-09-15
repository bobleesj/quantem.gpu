"""Run scan512 plus trusted-table as a memory-capped seven-source A/B/A test."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-scan512-trusted-table-retry/run.py"
SPEC = importlib.util.spec_from_file_location("trusted_table_retry_runner", PREVIOUS_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load corrected runner: {PREVIOUS_RUNNER}")
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
base = previous.base

_original_fingerprint = previous._original_fingerprint


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _original_fingerprint(root, executable, manifest, runner=Path(__file__).resolve())


base._fingerprint_code = fingerprint_code

if __name__ == "__main__":
    base.main()
