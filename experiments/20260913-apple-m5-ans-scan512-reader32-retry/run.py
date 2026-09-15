"""Retry scan512 + reader32 after adapting the prior validator schema."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_RUNNER = (
    ROOT / "experiments/20260913-apple-m5-ans-scan512-reader32/run.py"
)
SPEC = importlib.util.spec_from_file_location("reader32_first_attempt", PREVIOUS_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load earlier reader32 protocol: {PREVIOUS_RUNNER}")
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)

# The benchmark reports macro=false after the earlier manifest/validator was
# authored. Assert the field is off, then remove only that schema addition for
# the older base validator; reader32 remains separately checked by its wrapper.
reader_validator = previous.validate_reader32_response
old_validator = reader_validator.__globals__["_validate_response"]


def validate_with_macro_compatibility(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = previous.base.CYCLES,
    streams_per_lane: int = 2,
):
    compatible = copy.deepcopy(response)
    for name in ("configuration", "requested_configuration"):
        configuration = compatible.get(name, {})
        if "macro" in configuration:
            previous.base._require(
                configuration["macro"] is False,
                f"unexpected macro specialization in {expected_arm}/{name}",
            )
            configuration.pop("macro")
    record, cycle_hashes, hashes = old_validator(
        compatible,
        expected_arm,
        variant,
        ready,
        expected_map_hashes,
        cycles=cycles,
        streams_per_lane=streams_per_lane,
    )
    record["configuration"]["macro"] = False
    record["requested_configuration"]["macro"] = False
    return record, cycle_hashes, hashes


reader_validator.__globals__["_validate_response"] = validate_with_macro_compatibility

# Fingerprint this retry runner, not the first failed runner it imports.
old_fingerprint = previous.base._fingerprint_code


def fingerprint_retry(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    old_fingerprint(root, executable, manifest, runner=Path(__file__).resolve())


previous.base._fingerprint_code = fingerprint_retry


if __name__ == "__main__":
    previous.base.main()
