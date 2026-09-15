"""Profile only the scan512 + trusted-table candidate arm."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/run.py"
SPEC = importlib.util.spec_from_file_location("best_path_compose_retry", COMPOSE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load validated composition runner: {COMPOSE_RUNNER}")
compose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compose)
base = compose.base


_environment = base._environment


def profile_environment() -> dict[str, str]:
    environment = _environment()
    # The resident profiler must exist before all seven sources are created.
    environment["QGPU_PAIRED_RUNTIME_PROFILE"] = "1"
    return environment


base._environment = profile_environment

_request = base._request


def profile_request(process, raw, command: dict) -> dict:
    request = dict(command)
    if request.get("op") == "run":
        # control() deliberately disables profiles for A1/A2; profile the
        # candidate only so the validated controls remain unchanged.
        request["profile"] = request.get("arm") == "candidate"
    return _request(process, raw, request)


base._request = profile_request

_validate_response = base._validate_response


def validate_profile_response(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = base.CYCLES,
    streams_per_lane: int = 2,
):
    expected_profile = expected_arm == "candidate"
    config = response.get("configuration", {})
    requested = response.get("requested_configuration", {})
    if config.get("profile") is not expected_profile or requested.get("profile") is not expected_profile:
        raise RuntimeError(f"profiling flag mismatch in {expected_arm}")
    compatible = copy.deepcopy(response)
    compatible["configuration"]["profile"] = False
    compatible["requested_configuration"]["profile"] = False
    record, cycle_hashes, map_hashes = _validate_response(
        compatible,
        expected_arm,
        variant,
        ready,
        expected_map_hashes,
        cycles=cycles,
        streams_per_lane=streams_per_lane,
    )
    record["configuration"]["profile"] = expected_profile
    record["requested_configuration"]["profile"] = expected_profile
    return record, cycle_hashes, map_hashes


base._validate_response = validate_profile_response

_fingerprint = base._fingerprint_code


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _fingerprint(root, executable, manifest)
    manifest["code"]["profile_runner_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest["code"]["profile_runner"] = str(Path(__file__).resolve().relative_to(root))


base._fingerprint_code = fingerprint_code


if __name__ == "__main__":
    base.main()
