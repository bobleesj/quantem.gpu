"""Run reader32 A/B/A without the scan512 query path."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/20260913-apple-m5-ans-scan512-reader32-retry/run.py"
SPEC = importlib.util.spec_from_file_location("reader32_compat_runner", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load reader32 validation protocol from {SOURCE}")
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)

compat.previous.base.ARMS = (
    ("A1", "A1", "packet-groups"),
    ("B", "candidate", "packet-groups"),
    ("A2", "A2", "packet-groups"),
)

# The legacy shared ready validator was written for scan512 experiments. For
# this packet-groups run, explicitly require scan512 is not selected/prepared,
# then reuse only its shape, dtype, seven-identity, and allocation checks.
old_ready_validator = compat.previous.base._validate_ready


def validate_packet_groups_ready(record: dict) -> list[str]:
    compat.previous.base._require(
        record.get("polar_query_variant") == "packet-groups",
        "packet-groups must be the startup query path",
    )
    compat.previous.base._require(
        record.get("polar_query_scan512_ab_a1_b_a2") is False,
        "scan512 query experiment must be disabled",
    )
    compat.previous.base._require(
        record.get("polar_query_scan512_pipeline_prepared") == [False] * 7,
        "scan512 pipelines must not be prepared in this isolated run",
    )
    compatible = dict(record)
    compatible["polar_query_scan512_ab_a1_b_a2"] = True
    compatible["polar_query_scan512_pipeline_prepared"] = [True] * 7
    return old_ready_validator(compatible)


compat.previous.base._validate_ready = validate_packet_groups_ready

# Keep the reader32 prep but disable scan512 selection; its prepared pipeline
# may remain resident in every arm, matching the earlier controlled setup.
base_environment = compat.previous.base._environment


def packet_groups_environment() -> dict[str, str]:
    environment = base_environment()
    environment["QGPU_ANS_OPT_POLAR_QUERY_SCAN512"] = "0"
    environment["QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT"] = "packet-groups"
    return environment


compat.previous.base._environment = packet_groups_environment

# Fingerprint this runner rather than either compatibility shim it imports.
old_fingerprint = compat.previous.base._fingerprint_code


def fingerprint_runner(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    old_fingerprint(root, executable, manifest, runner=Path(__file__).resolve())


compat.previous.base._fingerprint_code = fingerprint_runner


if __name__ == "__main__":
    compat.previous.base.main()
