"""Retry current-source exact seven-source scan512/trusted-table baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_RUNNER = ROOT / (
    "experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/run.py"
)
SPEC = importlib.util.spec_from_file_location("scan512_trusted_compose", COMPOSE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load protocol from {COMPOSE_RUNNER}")
compose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compose)
base = compose.base
trusted_validator = compose.previous._previous_validate_response


def validate_current_configuration(
    response: dict,
    expected_arm: str,
    variant: str,
    ready: dict,
    expected_map_hashes: dict | None,
    cycles: int = base.CYCLES,
    streams_per_lane: int = 2,
):
    configuration = response.get("configuration", {})
    requested = response.get("requested_configuration", {})
    base._require(configuration.get("macro") is False,
                  f"macro mode must remain off in {expected_arm}")
    base._require(requested.get("macro") is False,
                  f"requested macro mode must remain off in {expected_arm}")
    return trusted_validator(
        response, expected_arm, variant, ready, expected_map_hashes,
        cycles=cycles, streams_per_lane=streams_per_lane,
    )


base._validate_response = validate_current_configuration


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    compose._original_fingerprint(
        root, executable, manifest, runner=Path(__file__).resolve()
    )


base._fingerprint_code = fingerprint_code


if __name__ == "__main__":
    base.main()
