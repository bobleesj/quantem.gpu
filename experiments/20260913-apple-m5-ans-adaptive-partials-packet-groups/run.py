"""Run adaptive residual partials with the compatible packet-groups query."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


EXPERIMENT_ID = "20260913-apple-m5-ans-adaptive-partials-packet-groups"
ROOT = Path(__file__).resolve().parents[2]
PRIOR_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-adaptive-partials/run.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("adaptive_partials_harness", PRIOR_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reusable A/B/A runner: {PRIOR_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = _load_runner()
HARNESS.__file__ = str(Path(__file__).resolve())
HARNESS.EXPERIMENT_ID = EXPERIMENT_ID
_original_environment = HARNESS._environment
_original_configuration = HARNESS._expected_configuration

# Fingerprint the exact imported harness implementation as a dependency of this
# corrected experiment, in addition to the currently tested wrapper itself.
HARNESS.HELPERS.SOURCE_FILES = {
    **HARNESS.HELPERS.SOURCE_FILES,
    "prior_adaptive_partials_runner":
        "experiments/20260913-apple-m5-ans-adaptive-partials/run.py",
}


def _expected_configuration(kernel: str) -> dict:
    configuration = _original_configuration(kernel)
    configuration["polar_query_variant"] = "packet-groups"
    return configuration


def _validate_ready(record: dict) -> list[str]:
    require = HARNESS._require
    count = HARNESS.SOURCE_COUNT
    require(record.get("resident_count") == count,
            "expected exactly seven resident acquisitions")
    require(record.get("shape") == [512, 512, 192, 192],
            "the full 512x512x192x192 workload is required")
    require(record.get("logical_dtype") == "uint16",
            "the full-uint16 workload is required")
    require(record.get("indexed_mode_available") is True,
            "the indexed resident mode was not prepared")
    require(record.get("compact_offsets_enabled") == [True] * count,
            "compact offsets must be enabled for all seven residents")
    require(record.get("compact_offsets_requested") is True,
            "compact offsets were not requested by the harness")
    require(record.get("polar_query_variant") == "packet-groups",
            "packet-groups query must be selected at resident startup")
    offsets = record.get("compact_offset_bytes")
    require(isinstance(offsets, list) and len(offsets) == count
            and all(type(value) is int and value > 0 for value in offsets),
            "ready response omitted positive per-source compact-offset savings")
    identities = record.get("source_identity_sha256", [])
    require(len(identities) == count and len(set(identities)) == count
            and all(isinstance(value, str) and len(value) == 64 for value in identities),
            "seven distinct source identity hashes are required")
    residents = record.get("resident_bytes_by_source")
    require(isinstance(residents, list) and len(residents) == count
            and all(type(value) is int and value > 0 for value in residents),
            "ready response omitted per-source resident bytes")
    require(sum(residents) == record.get("series_resident_bytes"),
            "resident byte total disagrees with per-source residents")
    allocation = record.get("metal_current_allocated_bytes")
    require(type(allocation) is int and allocation <= HARNESS.MAX_METAL_BYTES,
            "compact-offset ready allocation exceeds the absolute Metal cap")
    return identities


def _environment() -> dict[str, str]:
    environment = _original_environment()
    environment.update({
        "QGPU_ANS_OPT_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT": "packet-groups",
        "QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "1",
        "QGPU_PAIRED_RUNTIME_TRUSTED_TABLE": "0",
    })
    return environment


HARNESS._expected_configuration = _expected_configuration
HARNESS._validate_ready = _validate_ready
HARNESS._environment = _environment
_original_request = HARNESS.HELPERS._request


def _request(process, raw, command: dict) -> dict:
    adjusted = dict(command)
    if adjusted.get("op") == "run":
        adjusted["polar_query_variant"] = "packet-groups"
    return _original_request(process, raw, adjusted)


HARNESS.HELPERS._request = _request


def main() -> None:
    HARNESS.main()


if __name__ == "__main__":
    main()
