"""Run the uint2 SIMD pair-reduction seven-source A/B/A experiment."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_RUNNER = ROOT / (
    "experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/run.py"
)
FOUNDATION_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load benchmark protocol from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compose = _load(COMPOSE_RUNNER, "scan512_trusted_table_compose_runner")
foundation = _load(FOUNDATION_RUNNER, "scan512_protocol_foundation")
base = compose.base
_validate_response = base._validate_response
ANCHOR_MANIFEST = ROOT / (
    "experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/manifest.json"
)
_anchor = json.loads(ANCHOR_MANIFEST.read_text(encoding="utf-8"))
_anchor_a1 = next(
    arm for arm in _anchor["parameters"]["arm_samples"] if arm["label"] == "A1"
)
ANCHOR_IDENTITIES = _anchor["parameters"]["source_identity_sha256"]
ANCHOR_MAP_HASHES = _anchor_a1["sha256_u32_le"]

base.SOURCE_FILES.update({
    "vector_pair_oracle": (
        "experiments/20260913-apple-m5-ans-paired-vector-reduction/oracle_uint2.py"
    ),
})

_environment = base._environment


def vector_environment() -> dict[str, str]:
    environment = _environment()
    environment["QGPU_PAIRED_RUNTIME_PREPARE_VECTOR_PAIR_REDUCTION"] = "1"
    environment["QGPU_PAIRED_RUNTIME_PRESERVE_TRUSTED_TABLE_CONTROL"] = "1"
    return environment


base._environment = vector_environment

_validate_ready = base._validate_ready


def validate_anchor_ready(record: dict) -> list[str]:
    identities = _validate_ready(record)
    base._require(
        identities == ANCHOR_IDENTITIES,
        "seven source identities differ from the frozen scan512/trusted-table anchor",
    )
    return identities


base._validate_ready = validate_anchor_ready


def vector_request(process, raw, command: dict) -> dict:
    request = dict(command)
    request["trusted_table"] = True
    request["vector_pair_reduction"] = request.get("arm") == "candidate"
    return foundation._request(process, raw, request)


base._request = vector_request


def validate_vector_response(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = base.CYCLES,
    streams_per_lane: int = 2,
):
    expected_vector = expected_arm == "candidate"
    actual_configuration = response.get("configuration", {})
    actual_requested = response.get("requested_configuration", {})
    base._require(
        actual_configuration.get("trusted_table") is True
        and actual_requested.get("trusted_table") is True,
        f"trusted-table must remain enabled in every arm; got {expected_arm}",
    )
    base._require(
        actual_configuration.get("vector_pair_reduction", False) is expected_vector
        and actual_requested.get("vector_pair_reduction", False) is expected_vector,
        f"vector-pair reduction arm mismatch in {expected_arm}",
    )

    compatible = copy.deepcopy(response)
    for configuration in (
        compatible["configuration"], compatible["requested_configuration"]
    ):
        configuration.pop("vector_pair_reduction", None)
        configuration["trusted_table"] = expected_vector

    record, cycle_hashes, map_hashes = _validate_response(
        compatible,
        expected_arm,
        variant,
        ready,
        expected_map_hashes,
        cycles=cycles,
        streams_per_lane=streams_per_lane,
    )
    base._require(
        map_hashes == ANCHOR_MAP_HASHES,
        f"full-map hashes differ from the frozen anchor in {expected_arm}",
    )
    for name in ("configuration", "requested_configuration"):
        record[name]["trusted_table"] = True
        if expected_vector:
            record[name]["vector_pair_reduction"] = True
    return record, cycle_hashes, map_hashes


base._validate_response = validate_vector_response

_original_fingerprint = base._fingerprint_code


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _original_fingerprint(root, executable, manifest, runner=Path(__file__).resolve())
    manifest["code"]["pipeline_status"] = "uint2 SIMD reduction; opt-in only"
    manifest["code"]["runner_sha256"] = base._sha256(Path(__file__).resolve())


base._fingerprint_code = fingerprint_code

if __name__ == "__main__":
    base.main()
