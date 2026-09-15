"""Run an exact packet-split 1/4/1 A/B/A test on seven full sources."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


EXPERIMENT_ID = "20260913-apple-m5-ans-packet-split-adaptive"
ROOT = Path(__file__).resolve().parents[2]
PRIOR_RUNNER = ROOT / (
    "experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups/run.py"
)


def _load_packet_groups_harness():
    spec = importlib.util.spec_from_file_location(
        "ans_adaptive_packet_groups_harness", PRIOR_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load packet-groups A/B/A runner: {PRIOR_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PACKET_GROUPS = _load_packet_groups_harness()
PACKET_GROUPS.__file__ = str(Path(__file__).resolve())
PACKET_GROUPS.EXPERIMENT_ID = EXPERIMENT_ID
HARNESS = PACKET_GROUPS.HARNESS
HARNESS.__doc__ = __doc__
HARNESS.__file__ = str(Path(__file__).resolve())
HARNESS.EXPERIMENT_ID = EXPERIMENT_ID

HARNESS.HELPERS.SOURCE_FILES = {
    **HARNESS.HELPERS.SOURCE_FILES,
    "prior_packet_groups_runner":
        "experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups/run.py",
}

_current_packet_splits = 1
_original_expected_configuration = HARNESS._expected_configuration
_original_environment = HARNESS._environment
_original_request = HARNESS.HELPERS._request


def _expected_configuration(kernel: str) -> dict:
    configuration = _original_expected_configuration(kernel)
    configuration["packet_splits"] = _current_packet_splits
    return configuration


def _environment() -> dict[str, str]:
    environment = _original_environment()
    environment["QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS"] = "1"
    environment["QGPU_PAIRED_RUNTIME_PACKET_SPLITS"] = "1"
    return environment


def _request(process, raw, command: dict) -> dict:
    global _current_packet_splits
    adjusted = dict(command)
    if adjusted.get("op") == "run":
        arm = str(adjusted.get("arm", "")).lower()
        _current_packet_splits = 4 if arm == "candidate" else 1
        adjusted["packet_splits"] = _current_packet_splits
    return _original_request(process, raw, adjusted)


HARNESS._expected_configuration = _expected_configuration
HARNESS._environment = _environment
HARNESS.HELPERS._request = _request


def main() -> None:
    HARNESS.main()


if __name__ == "__main__":
    main()
