"""Run split-8 A/B/A with scan512 and trusted-table fixed on."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


EXPERIMENT_ID = "20260913-apple-m5-ans-scan512-trusted-split8"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("scan512_protocol_split8", PROTOCOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load scan512 protocol: {PROTOCOL_PATH}")
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)

protocol.ARMS = (
    ("A1", "A1", "scan512"),
    ("B", "candidate", "scan512"),
    ("A2", "A2", "scan512"),
)
protocol.SOURCE_FILES.update({
    "trusted_table_factory": (
        "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSTables.swift"
    ),
    "trusted_table_tests": (
        "src/quantem/gpu/swift/Tests/Native4DSTEMIOTests/PairedRuntimeTANSTablesTests.swift"
    ),
    "runner": "experiments/20260913-apple-m5-ans-scan512-trusted-split8/run.py",
})

_original_environment = protocol._environment
_original_request = protocol._request
_original_validate_ready = protocol._validate_ready
_original_validate_response = protocol._validate_response


def _environment() -> dict[str, str]:
    environment = _original_environment()
    environment.update({
        "QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE": "1",
        "QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS": "1",
        "QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE_PACKET_SPLIT8": "1",
        "QGPU_PAIRED_RUNTIME_PACKET_SPLITS": "1",
        "QGPU_PAIRED_RUNTIME_TRUSTED_TABLE": "1",
        "QGPU_PAIRED_RUNTIME_PRESERVE_TRUSTED_TABLE_CONTROL": "1",
    })
    return environment


def _request(process, raw, command: dict) -> dict:
    adjusted = dict(command)
    if adjusted.get("op") == "run":
        candidate = str(adjusted.get("arm", "")).lower() == "candidate"
        adjusted["packet_splits"] = 8 if candidate else 1
        adjusted["trusted_table"] = True
    return _original_request(process, raw, adjusted)


def _validate_ready(record: dict) -> list[str]:
    identities = _original_validate_ready(record)
    protocol._require(
        len(identities) == 7 and len(set(identities)) == 7,
        "expected seven distinct full acquisitions",
    )
    protocol._require(
        record.get("series_resident_bytes", 0) <= 11_877_814_048,
        "series residents exceeded the existing seven-source ceiling",
    )
    protocol._require(
        record.get("metal_current_allocated_bytes", 0) <= 11_883_921_408,
        "Metal allocation exceeded the existing seven-source ceiling",
    )
    return identities


def _validate_response(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = protocol.CYCLES,
    streams_per_lane: int = 2,
) -> tuple[dict, dict, dict]:
    expected_split = 8 if expected_arm == "candidate" else 1
    for key in ("configuration", "requested_configuration"):
        actual = response.get(key, {})
        protocol._require(
            actual.get("trusted_table") is True,
            f"trusted-table must remain enabled in {expected_arm}/{key}",
        )
        protocol._require(
            actual.get("packet_splits") == expected_split,
            f"expected packet_splits={expected_split} in {expected_arm}/{key}",
        )

    # Reuse the frozen scan512 response checks after normalizing only the two
    # tested controls and an unrelated schema addition (`macro=false`).
    normalized = copy.deepcopy(response)
    for key in ("configuration", "requested_configuration"):
        normalized[key]["trusted_table"] = False
        normalized[key]["packet_splits"] = 1
        normalized[key].pop("macro", None)
    record, cycle_hashes, map_hashes = _original_validate_response(
        normalized,
        expected_arm,
        variant,
        ready,
        expected_map_hashes,
        cycles=cycles,
        streams_per_lane=streams_per_lane,
    )
    for key in ("configuration", "requested_configuration"):
        record[key]["trusted_table"] = True
        record[key]["packet_splits"] = expected_split
        record[key]["macro"] = response[key].get("macro", False)
    return record, cycle_hashes, map_hashes


_original_fingerprint = protocol._fingerprint_code


def _fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _original_fingerprint(root, executable, manifest, runner=Path(__file__).resolve())


protocol._environment = _environment
protocol._request = _request
protocol._validate_ready = _validate_ready
protocol._validate_response = _validate_response
protocol._fingerprint_code = _fingerprint_code


if __name__ == "__main__":
    protocol.main()
