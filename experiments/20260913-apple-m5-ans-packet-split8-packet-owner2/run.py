"""Run an exact packet-owner2 packet-split 1/8/1 A/B/A test."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


EXPERIMENT_ID = "20260913-apple-m5-ans-packet-split8-packet-owner2"
ROOT = Path(__file__).resolve().parents[2]
PRIOR_RUNNER = ROOT / (
    "experiments/20260913-apple-m5-ans-packet-split-packet-owner2/run.py"
)


def _load_prior_packet_owner2_runner():
    spec = importlib.util.spec_from_file_location(
        "ans_packet_owner2_split_1_4_1_prior", PRIOR_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prior packet-owner2 harness: {PRIOR_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRIOR = _load_prior_packet_owner2_runner()
PRIOR.__file__ = str(Path(__file__).resolve())
PRIOR.EXPERIMENT_ID = EXPERIMENT_ID
HARNESS = PRIOR.HARNESS
HARNESS.__file__ = str(Path(__file__).resolve())
HARNESS.EXPERIMENT_ID = EXPERIMENT_ID
HARNESS.__doc__ = __doc__

HARNESS.ARMS = (
    ("A1", "A1", "packet-owner2"),
    ("B", "candidate", "packet-owner2"),
    ("A2", "A2", "packet-owner2"),
)
HARNESS.HELPERS.SOURCE_FILES = {
    **HARNESS.HELPERS.SOURCE_FILES,
    "prior_packet_owner2_split_1_4_1_runner": (
        "experiments/20260913-apple-m5-ans-packet-split-packet-owner2/run.py"
    ),
}

_original_expected_configuration = HARNESS._expected_configuration
_packet_groups_request = PRIOR.PRIOR.PACKET_GROUPS._request


def _expected_configuration(_kernel: str) -> dict:
    configuration = _original_expected_configuration("packet-owner2")
    configuration["kernel"] = "packet-owner2"
    return configuration


def _request(process, raw, command: dict) -> dict:
    adjusted = dict(command)
    if adjusted.get("op") == "run":
        arm = str(adjusted.get("arm", "")).lower()
        split = 8 if arm == "candidate" else 1
        # The inherited expected-config wrapper reads this exact value. Bypass
        # the older 1/4 wrapper's request function so only this 1/8/1 trial is
        # sent to the benchmark process.
        PRIOR.PRIOR._current_packet_splits = split
        adjusted["kernel"] = "packet-owner2"
        adjusted["packet_splits"] = split
    return _packet_groups_request(process, raw, adjusted)


HARNESS._expected_configuration = _expected_configuration
HARNESS.HELPERS._request = _request


def main() -> None:
    HARNESS.main()


if __name__ == "__main__":
    main()
