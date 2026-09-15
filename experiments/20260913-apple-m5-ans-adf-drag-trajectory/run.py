"""Measure a 20-column exact ADF drag path over seven full residents."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-update-concurrency/run.py"
SPEC = importlib.util.spec_from_file_location("ans_update_concurrency_runner", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load benchmark protocol from {BASE_RUNNER}")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
runner = wrapper.base
runner.MASKS = ("adf-base",) + tuple(f"adf-drag-column-{column}" for column in range(1, 21))
wrapper._concurrency = {"A1": 7, "candidate": 7, "A2": 7}
_fingerprint_code = runner._fingerprint_code


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _fingerprint_code(root, executable, manifest, runner=Path(__file__).resolve())


runner._fingerprint_code = fingerprint_code


if __name__ == "__main__":
    runner.main()
