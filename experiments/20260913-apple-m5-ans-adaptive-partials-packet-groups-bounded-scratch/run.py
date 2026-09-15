"""Run packet-groups adaptive partials with lazy-scratch-aware validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


EXPERIMENT_ID = "20260913-apple-m5-ans-adaptive-partials-packet-groups-bounded-scratch"
ROOT = Path(__file__).resolve().parents[2]
PRIOR_RUNNER = ROOT / (
    "experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups/run.py"
)


def _load_packet_groups_harness():
    spec = importlib.util.spec_from_file_location(
        "adaptive_partials_packet_groups_harness", PRIOR_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load packet-groups harness: {PRIOR_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PACKET_GROUPS = _load_packet_groups_harness()
PACKET_GROUPS.__file__ = str(Path(__file__).resolve())
PACKET_GROUPS.EXPERIMENT_ID = EXPERIMENT_ID
HARNESS = PACKET_GROUPS.HARNESS
HARNESS.__file__ = str(Path(__file__).resolve())
HARNESS.EXPERIMENT_ID = EXPERIMENT_ID
_original_response_validator = HARNESS._validate_response
_require = HARNESS._require
_series_by_arm: dict[str, int] = {}
_series_warmup_by_arm: dict[str, int] = {}
_source_identities: list[str] | None = None

# The packet-groups wrapper already fingerprints the earlier adaptive harness.
# Add that wrapper directly too, so the complete imported lineage is explicit.
HARNESS.HELPERS.SOURCE_FILES = {
    **HARNESS.HELPERS.SOURCE_FILES,
    "prior_packet_groups_runner":
        "experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups/run.py",
}


def _validate_response(
    response: dict,
    label: str,
    kernel: str,
    ready: dict,
    expected_hashes: dict | None,
    cycles: int,
) -> tuple[dict, dict]:
    global _source_identities
    identities = ready.get("source_identity_sha256", [])
    if _source_identities is None:
        _source_identities = list(identities)
    _require(identities == _source_identities,
             f"source identities changed during {label}")

    ready_series = ready["series_resident_bytes"]
    observed_series = response.get("series_resident_bytes")
    _require(type(observed_series) is int,
             f"aggregate series resident bytes missing in {label}")
    if label == "A1":
        _require(observed_series == ready_series,
                 "A1 series resident bytes differ from compact-offset ready baseline")
    elif label == "B" and cycles == HARNESS.WARMUP_CYCLES:
        growth = observed_series - ready_series
        _require(0 <= growth <= HARNESS.MAX_CANDIDATE_GROWTH_BYTES,
                 f"B scratch growth {growth} exceeds the 122 MiB series-resident bound")
        _series_by_arm["B"] = observed_series
        _series_warmup_by_arm["B"] = observed_series
    else:
        expected_series = _series_by_arm.get("B")
        _require(expected_series is not None,
                 f"candidate warmup series residency was not recorded before {label}")
        _require(observed_series == expected_series,
                 f"series resident bytes changed after candidate warmup at {label}")
    if cycles == HARNESS.WARMUP_CYCLES:
        _series_warmup_by_arm[label] = observed_series
    _series_by_arm[label] = observed_series

    # The shared exact-map validator also checks the runtime's aggregate series
    # byte field. Substitute only the observed value for that one equality;
    # this wrapper enforces the correct A1/B/A2 residency state machine above.
    adjusted_ready = {**ready, "series_resident_bytes": observed_series}
    return _original_response_validator(
        response, label, kernel, adjusted_ready, expected_hashes, cycles
    )


HARNESS._validate_response = _validate_response
_original_main = HARNESS.main


def main() -> None:
    _original_main()
    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest["parameters"]
    parameters.pop("resident_bytes_unchanged", None)
    parameters.pop("source_identities_unchanged", None)
    # The ready JSON is the authoritative aggregate series-resident baseline.
    result_path = Path(sys.argv[sys.argv.index("--out") + 1]).expanduser().resolve()
    ready = json.loads((result_path / "ready.json").read_text(encoding="utf-8"))
    ready_series = ready["series_resident_bytes"]
    measured = {
        arm["label"]: arm["series_resident_bytes"]
        for arm in parameters["arm_samples"]
    }
    parameters["series_resident_bytes_by_arm"] = measured
    parameters["series_resident_warmup_bytes_by_arm"] = dict(_series_warmup_by_arm)
    parameters["adaptive_series_resident_growth_bytes"] = (
        measured["B"] - ready_series
    )
    parameters["adaptive_warmup_series_growth_bytes"] = (
        _series_warmup_by_arm["B"] - ready_series
    )
    parameters["source_identity_stability_verified"] = (
        _source_identities == parameters["source_identity_sha256"]
    )
    parameters["input_identity_stability_verified"] = parameters[
        "source_identity_stability_verified"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
