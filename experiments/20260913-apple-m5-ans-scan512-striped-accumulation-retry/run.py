"""Retry striped scan512 after fixing fixed-feature preservation in A1."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("scan512_stripe4_retry_base", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load validated base runner: {BASE_RUNNER}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

base.ARMS = (
    ("A1", "A1", "scan512"),
    ("B", "candidate", "scan512-stripe4"),
    ("A2", "A2", "scan512"),
)
base.SOURCE_FILES.update({
    "trusted_table_factory": (
        "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSTables.swift"
    ),
    "trusted_table_tests": (
        "src/quantem/gpu/swift/Tests/Native4DSTEMIOTests/PairedRuntimeTANSTablesTests.swift"
    ),
})

_environment = base._environment


def fixed_composition_environment() -> dict[str, str]:
    environment = _environment()
    environment["QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE"] = "1"
    environment["QGPU_PAIRED_RUNTIME_PRESERVE_TRUSTED_TABLE_CONTROL"] = "1"
    return environment


base._environment = fixed_composition_environment

_request = base._request


def trusted_table_request(process, raw, command: dict) -> dict:
    request = dict(command)
    if request.get("op") == "run":
        request["trusted_table"] = True
    return _request(process, raw, request)


base._request = trusted_table_request

_validate_response = base._validate_response


def validate_trusted_response(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = base.CYCLES,
    streams_per_lane: int = 2,
):
    actual = response.get("configuration", {}).get("trusted_table")
    requested = response.get("requested_configuration", {}).get("trusted_table")
    base._require(
        actual is True and requested is True,
        f"trusted-table must remain enabled for every arm: {expected_arm}",
    )
    compatible = copy.deepcopy(response)
    compatible["configuration"]["trusted_table"] = False
    compatible["requested_configuration"]["trusted_table"] = False
    record, cycle_hashes, map_hashes = _validate_response(
        compatible,
        expected_arm,
        variant,
        ready,
        expected_map_hashes,
        cycles=cycles,
        streams_per_lane=streams_per_lane,
    )
    record["configuration"]["trusted_table"] = True
    record["requested_configuration"]["trusted_table"] = True
    return record, cycle_hashes, map_hashes


base._validate_response = validate_trusted_response

_validate_ready = base._validate_ready


def validate_memory_capped_ready(record: dict) -> list[str]:
    identities = _validate_ready(record)
    base._require(
        record.get("series_resident_bytes", 0) <= 11_877_814_048,
        "resident bytes exceeded the established seven-source ceiling",
    )
    base._require(
        record.get("metal_current_allocated_bytes", 0) <= 11_883_921_408,
        "Metal allocation exceeded the established seven-source ceiling",
    )
    return identities


base._validate_ready = validate_memory_capped_ready

_fingerprint_code = base._fingerprint_code


def fingerprint_code(root: Path, executable: Path, manifest: dict) -> None:
    _fingerprint_code(root, executable, manifest, runner=Path(__file__).resolve())


base._fingerprint_code = fingerprint_code


if __name__ == "__main__":
    base.main()
