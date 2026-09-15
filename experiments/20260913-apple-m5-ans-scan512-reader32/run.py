"""Run the scan512 plus reader32 memory-capped seven-source A/B/A test."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("scan512_runner", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load benchmark protocol from {BASE_RUNNER}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

base.ARMS = (
    ("A1", "A1", "scan512"),
    ("B", "candidate", "scan512"),
    ("A2", "A2", "scan512"),
)

_environment = base._environment


def reader32_environment() -> dict[str, str]:
    environment = _environment()
    environment["QGPU_PAIRED_RUNTIME_PREPARE_READER32"] = "1"
    return environment


base._environment = reader32_environment

_request = base._request


def reader32_request(process, raw, command: dict) -> dict:
    request = dict(command)
    request["reader32"] = request.get("arm") == "candidate"
    return _request(process, raw, request)


base._request = reader32_request

_validate_response = base._validate_response


def validate_reader32_response(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = base.CYCLES,
    streams_per_lane: int = 2,
):
    expected_reader32 = expected_arm == "candidate"
    for configuration_name in ("configuration", "requested_configuration"):
        configuration = response.get(configuration_name, {})
        base._require(
            configuration.get("reader32", False) is expected_reader32,
            f"reader32 arm mismatch in {expected_arm}/{configuration_name}",
        )
    compatible = copy.deepcopy(response)
    for configuration_name in ("configuration", "requested_configuration"):
        compatible[configuration_name].pop("reader32", None)
    record, cycle_hashes, hashes = _validate_response(
        compatible,
        expected_arm,
        variant,
        ready,
        expected_map_hashes,
        cycles=cycles,
        streams_per_lane=streams_per_lane,
    )
    record["configuration"]["reader32"] = expected_reader32
    record["requested_configuration"]["reader32"] = expected_reader32
    return record, cycle_hashes, hashes


base._validate_response = validate_reader32_response

_fingerprint_code = base._fingerprint_code


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _fingerprint_code(root, executable, manifest, runner=Path(__file__).resolve())


base._fingerprint_code = fingerprint_code

if __name__ == "__main__":
    base.main()
