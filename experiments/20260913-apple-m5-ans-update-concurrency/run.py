"""Benchmark bounded concurrency for the exact seven-source ADF update."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("ans_scan512_runner", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load benchmark protocol from {BASE_RUNNER}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

base.ARMS = (
    ("A1", "A1", "scan512"),
    ("B", "candidate", "scan512"),
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


def trusted_environment() -> dict[str, str]:
    environment = _environment()
    environment["QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE"] = "1"
    # The resident-loop A1 arm normally clears optional features to recreate
    # the control configuration. Preserve this common feature across A/B/A so
    # only the in-flight submission limit changes.
    environment["QGPU_PAIRED_RUNTIME_PRESERVE_TRUSTED_TABLE_CONTROL"] = "1"
    return environment


base._environment = trusted_environment

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

_request = base._request
_concurrency = {"A1": 7, "candidate": 4, "A2": 7}


def concurrency_request(process, raw, command: dict) -> dict:
    request = dict(command)
    request["batch"] = False
    request["trusted_table"] = True
    request["bounded_concurrency"] = _concurrency[request["arm"]]
    return _request(process, raw, request)


base._request = concurrency_request

_validate_response = base._validate_response


def validate_concurrency_response(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = base.CYCLES,
    streams_per_lane: int = 2,
):
    expected_concurrency = _concurrency[expected_arm]
    actual_config = response.get("configuration", {})
    actual_requested = response.get("requested_configuration", {})
    base._require(
        actual_config.get("batch") is False
        and actual_requested.get("batch") is False,
        f"ordinary unbatched path must remain fixed in {expected_arm}",
    )
    base._require(
        actual_config.get("bounded_concurrency") == expected_concurrency
        and actual_requested.get("bounded_concurrency") == expected_concurrency,
        f"effective bounded concurrency mismatch in {expected_arm}",
    )
    base._require(
        actual_config.get("trusted_table") is True
        and actual_requested.get("trusted_table") is True,
        f"trusted-table decode must remain enabled in {expected_arm}",
    )

    # Reuse the established strict map/source/memory validator while translating
    # only the three deliberate experiment fields back to its fixed baseline.
    compatible = copy.deepcopy(response)
    for configuration in (compatible["configuration"], compatible["requested_configuration"]):
        base._require(
            configuration.get("macro") is False,
            f"macro decoding must remain disabled in {expected_arm}",
        )
        configuration.pop("macro", None)
        configuration["batch"] = False
        configuration["bounded_concurrency"] = base.SOURCE_COUNT
        configuration["trusted_table"] = False
    record, cycle_hashes, map_hashes = _validate_response(
        compatible, expected_arm, variant, ready, expected_map_hashes,
        cycles=cycles, streams_per_lane=streams_per_lane,
    )
    record["configuration"].update({
        "batch": False,
        "bounded_concurrency": expected_concurrency,
        "trusted_table": True,
    })
    record["requested_configuration"].update({
        "batch": False,
        "bounded_concurrency": expected_concurrency,
        "trusted_table": True,
    })
    return record, cycle_hashes, map_hashes


base._validate_response = validate_concurrency_response

_fingerprint_code = base._fingerprint_code


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _fingerprint_code(root, executable, manifest, runner=Path(__file__).resolve())


base._fingerprint_code = fingerprint_code

if __name__ == "__main__":
    base.main()
