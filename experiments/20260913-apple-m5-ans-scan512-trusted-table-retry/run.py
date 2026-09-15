"""Run the corrected scan512/trusted-table seven-source A/B/A experiment."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAILED_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-scan512-trusted-table/run.py"
SPEC = importlib.util.spec_from_file_location("trusted_table_failed_runner", FAILED_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load retained experiment runner: {FAILED_RUNNER}")
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
base = previous.base

_previous_validate_response = base._validate_response


def validate_complete_configuration(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = base.CYCLES,
    streams_per_lane: int = 2,
):
    actual_config = response.get("configuration", {})
    actual_requested = response.get("requested_configuration", {})
    base._require(
        actual_config.get("macro") is False and actual_requested.get("macro") is False,
        f"macro mode must remain off in {expected_arm}",
    )
    complete_response = copy.deepcopy(response)
    complete_response["configuration"].pop("macro")
    complete_response["requested_configuration"].pop("macro")
    record, hashes, reference = _previous_validate_response(
        complete_response, expected_arm, variant, ready, expected_map_hashes,
        cycles=cycles, streams_per_lane=streams_per_lane,
    )
    record["configuration"]["macro"] = False
    record["requested_configuration"]["macro"] = False
    return record, hashes, reference


base._validate_response = validate_complete_configuration

_original_fingerprint = previous._fingerprint_code


def fingerprint_retry(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _original_fingerprint(
        root, executable, manifest, runner=Path(__file__).resolve()
    )


base._fingerprint_code = fingerprint_retry

if __name__ == "__main__":
    base.main()
