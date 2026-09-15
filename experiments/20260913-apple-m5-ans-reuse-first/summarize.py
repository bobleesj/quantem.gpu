"""Validate and summarize complete seven-source compute and memory evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from run import _reference_maps

ARMS = ("control-a1", "compact", "control-a2")
SOURCE_COUNT = 7
STAGES = ("index", "residual", "combined")
DP_SCANS = (
    0, 1, 30, 31, 32, 33, 510, 511, 512, 513,
    16_383, 16_384, 16_385, 262_142, 262_143,
    *((index * 7919 + 123) % 262_144 for index in range(20)),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sample distribution")
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "mean": statistics.mean(ordered),
            "p95": ordered[round((len(ordered) - 1) * 0.95)],
            "min": min(ordered), "max": max(ordered), "n": len(ordered)}


def _check_stage_response(response: dict, joint: bool, label: str) -> None:
    _require(response.get("exact") is True, f"{label}: stage parity is not exact")
    _require(response.get("joint_plan") is joint, f"{label}: planner mode echo mismatch")
    expected = {(stage, source) for stage in STAGES for source in range(SOURCE_COUNT)}
    observed = Counter((sample.get("stage"), sample.get("source"))
                       for sample in response.get("samples", []))
    _require(set(observed) == expected and all(count == 1 for count in observed.values()),
             f"{label}: expected exactly {len(expected)} stage/source samples")


def _check_matrix_samples(record: dict, cycles: int, masks: set[str], label: str) -> None:
    samples = record.get("samples", [])
    expected = {
        (cycle, mask, source)
        for cycle in range(cycles)
        for mask in masks
        for source in range(SOURCE_COUNT)
    }
    observed = Counter((sample.get("cycle"), sample.get("mask"), sample.get("source"))
                       for sample in samples)
    _require(set(observed) == expected and all(count == 1 for count in observed.values()),
             f"{label}: full-map sample grid incomplete or duplicated")
    hash_maps = record.get("sha256_u32_le", {})
    _require(set(hash_maps) == masks and all(len(values) == SOURCE_COUNT
                                             for values in hash_maps.values()),
             f"{label}: returned full-map hash table has incomplete coverage")
    for sample in samples:
        mask = sample["mask"]
        source = sample["source"]
        _require(sample["sha256_u32_le"] == hash_maps[mask][source],
                 f"{label}: cycle sample hash disagrees with returned map for {(mask, source)}")


def _validate_arm(directory: Path, arm: str, reference_masks: set[str], repeats: int,
                  matrix_cycles: int) -> tuple[dict, dict, list[dict]]:
    records = _read_jsonl(directory / "raw.jsonl")
    ready_rows = [row for row in records if row.get("event") == "ans_resident_loop_ready"]
    end_rows = [row for row in records if row.get("event") == "ans_resident_loop_end"]
    dp_rows = [row for row in records if row.get("event") == "dp_audit"]
    stage_rows = [row for row in records if row.get("event") == "stage_isolation"]
    matrix_rows = [row for row in records if row.get("event") == "ans_resident_loop_result"]
    errors = [row for row in records if row.get("event") == "ans_resident_loop_error"]
    _require(not errors, f"{arm}: raw log contains benchmark errors")
    _require(len(ready_rows) == len(end_rows) == len(dp_rows) == 1,
             f"{arm}: expected one ready, DP audit, and clean-end event")
    _require(len(matrix_rows) == 3, f"{arm}: expected exactly three matrix runs")
    ready = ready_rows[0]
    _require(ready.get("resident_count") == SOURCE_COUNT,
             f"{arm}: ready event does not contain seven sources")
    _require(ready.get("shape") == [512, 512, 192, 192]
             and ready.get("logical_dtype") == "uint16",
             f"{arm}: full native uint16 shape is required")
    identities = ready.get("source_identity_sha256", [])
    _require(len(identities) == len(set(identities)) == SOURCE_COUNT,
             f"{arm}: seven distinct source identities are required")
    is_compact = arm == "compact"
    _require(ready.get("compact_offsets_enabled") == [is_compact] * SOURCE_COUNT,
             f"{arm}: ready offset layout does not match its arm")
    if is_compact:
        compact_bytes = ready.get("compact_offset_bytes", [])
        _require(len(compact_bytes) == SOURCE_COUNT and all(value > 0 for value in compact_bytes),
                 f"{arm}: compact directory byte counts are missing")
    _require(end_rows[0].get("all_released") is True,
             f"{arm}: resident storage was not fully released")

    dp_expected = {(source, scan) for source in range(SOURCE_COUNT) for scan in DP_SCANS}
    dp_samples = dp_rows[0].get("samples", [])
    dp_counts = Counter((sample.get("source"), sample.get("scan")) for sample in dp_samples)
    _require(set(dp_counts) == dp_expected and all(count == 1 for count in dp_counts.values()),
             f"{arm}: expected exactly {len(dp_expected)} unique DP audit keys")
    dp_hashes = {(sample["source"], sample["scan"]): sample["sha256_u32_le"]
                 for sample in dp_samples}

    trial_records = _read_jsonl(directory / "trials.jsonl")
    metadata_rows = [row for row in trial_records if row.get("event") == "run_metadata"]
    trials = [row for row in trial_records if row.get("event") == "stage_trial"]
    _require(len(metadata_rows) == 1, f"{arm}: missing or duplicated trial metadata")
    metadata = metadata_rows[0]
    _require(metadata.get("repeats") == repeats
             and metadata.get("matrix_cycles") == matrix_cycles,
             f"{arm}: run metadata differs across arms")
    _require(set(metadata.get("reference_masks", [])) == reference_masks,
             f"{arm}: run metadata mask set differs from frozen reference set")
    _require(metadata.get("expected_sources") == SOURCE_COUNT
             and metadata.get("expected_dp_scans") == list(DP_SCANS),
             f"{arm}: run metadata has unexpected parity coverage")

    expected_trial_keys = {(joint, repetition) for joint in (False, True)
                           for repetition in [-1, *range(repeats)]}
    trial_counts = Counter((row.get("joint"), row.get("repetition")) for row in trials)
    _require(set(trial_counts) == expected_trial_keys
             and all(count == 1 for count in trial_counts.values())
             and len(trials) == len(expected_trial_keys),
             f"{arm}: warmup/measured stage trial schedule is incomplete or duplicated")
    raw_stage_by_sequence = {row.get("sequence"): row for row in stage_rows}
    _require(len(raw_stage_by_sequence) == len(stage_rows) == len(trials),
             f"{arm}: raw stage event count does not match stored trial count")
    for trial in trials:
        _check_stage_response(trial["response"], trial["joint"], f"{arm} stage trial")
        sequence = trial.get("sequence")
        _require(sequence == trial["response"].get("sequence")
                 and raw_stage_by_sequence.get(sequence) == trial["response"],
                 f"{arm}: stage trial log does not match its raw JSONL event")

    expected_joint_modes = (False, True, False)
    map_hashes = None
    for index, (record, expected_joint) in enumerate(zip(matrix_rows, expected_joint_modes)):
        label = f"{arm} matrix run {index}"
        _require(record.get("fullmap_parity") is True
                 and record.get("exact_a1_hashes") is True,
                 f"{label}: full-map parity gate failed")
        _require(record.get("cycles") == matrix_cycles
                 and set(record.get("masks", [])) == reference_masks
                 and len(record.get("masks", [])) == len(reference_masks),
                 f"{label}: mask or cycle coverage mismatch")
        _require(record.get("configuration", {}).get("joint_plan") is expected_joint,
                 f"{label}: requested planner mode mismatch")
        _check_matrix_samples(record, matrix_cycles, reference_masks, label)
        current_hashes = record["sha256_u32_le"]
        _require(map_hashes is None or current_hashes == map_hashes,
                 f"{label}: full-map hashes changed across mode runs")
        map_hashes = current_hashes

    return ready, dp_hashes, matrix_rows


def main() -> None:
    """Validate a complete control/compact/control run and print its summary.

    Matrix timings are only two-cycle-per-mask replay/smoke observations. The
    20-trial interleaved stage measurements are the timing evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path, required=True,
                        help="JSONL containing the 140 independently checked frozen maps")
    args = parser.parse_args()
    frozen_maps = _reference_maps(args.reference)
    _require(len(set(DP_SCANS)) == 35, "The expected diffraction scan fixture is not unique")
    _require(all((args.directory / arm / "raw.jsonl").is_file()
                 and (args.directory / arm / "trials.jsonl").is_file() for arm in ARMS),
             "Complete control-a1/compact/control-a2 output is required; partial mode is unsupported")

    metadata_rows = []
    for arm in ARMS:
        rows = _read_jsonl(args.directory / arm / "trials.jsonl")
        found = [row for row in rows if row.get("event") == "run_metadata"]
        _require(len(found) == 1, f"{arm}: missing or duplicated run metadata")
        metadata_rows.append(found[0])
    first_metadata = metadata_rows[0]
    repeats = first_metadata.get("repeats")
    matrix_cycles = first_metadata.get("matrix_cycles")
    reference_masks = set(first_metadata.get("reference_masks", []))
    _require(isinstance(repeats, int) and repeats > 0,
             "Run metadata must contain a positive measured repeat count")
    _require(isinstance(matrix_cycles, int) and matrix_cycles > 0,
             "Run metadata must contain a positive matrix cycle count")
    _require(len(reference_masks) == 20, "Run metadata must name all 20 frozen masks")
    _require(reference_masks == {mask for mask, _ in frozen_maps},
             "Run masks differ from the independently checked frozen reference")
    for metadata in metadata_rows[1:]:
        _require(metadata == first_metadata, "Run metadata differs between arms")

    results = {}
    reference_dp = None
    reference_fullmaps = None
    for arm in ARMS:
        directory = args.directory / arm
        ready, dp_hashes, matrix_rows = _validate_arm(
            directory, arm, reference_masks, repeats, matrix_cycles)
        _require(reference_dp is None or dp_hashes == reference_dp,
                 f"{arm}: diffraction hashes differ across offset arms")
        reference_dp = dp_hashes
        matrix_hashes = [row["sha256_u32_le"] for row in matrix_rows]
        _require(reference_fullmaps is None or all(hashes == reference_fullmaps
                                                   for hashes in matrix_hashes),
                 f"{arm}: full-map hashes differ across offset arms")
        reference_fullmaps = matrix_hashes[0]

        trials = [row for row in _read_jsonl(directory / "trials.jsonl")
                  if row.get("event") == "stage_trial"]
        result = {
            "ready": ready,
            "stages": {},
            "matrix_smoke_runs": [],
            "matrix_timing_classification": (
                f"replay/smoke only: {matrix_cycles} cycles per mask; not stable timing evidence"
            ),
            "full_map_observations": sum(len(row["samples"]) for row in matrix_rows),
            "exact_dp_hashes": len(dp_hashes),
            "exact_stage_trials": len(trials),
            "exact_source_transitions": len(trials) * SOURCE_COUNT,
            "ended_and_released": True,
        }
        for joint in (False, True):
            chosen = [trial for trial in trials
                      if trial["joint"] is joint and trial["repetition"] >= 0]
            _require(len(chosen) == repeats,
                     f"{arm}: expected {repeats} measured trials for joint={joint}")
            mode_key = "joint" if joint else "greedy"
            grouped = {}
            for stage in STAGES:
                samples = [sample for trial in chosen
                           for sample in trial["response"]["samples"]
                           if sample["stage"] == stage]
                latency = [sample["all_seven_wall_ms"] for sample in samples
                           if sample["source"] == 0]
                fields_by_source = {}
                residuals_by_source = {}
                for source in range(SOURCE_COUNT):
                    source_samples = [sample for sample in samples
                                      if sample["source"] == source]
                    _require(len(source_samples) == repeats,
                             f"{arm}: {stage} source {source} count coverage mismatch")
                    fields_by_source[str(source)] = _distribution(
                        [sample["fields"] for sample in source_samples])
                    residuals_by_source[str(source)] = _distribution(
                        [sample["residuals"] for sample in source_samples])
                grouped[stage] = {
                    "all_seven_wall_ms": _distribution(latency),
                    "latency_observation": (
                        "one all-seven wall time per trial; source 0 avoids duplicate rows"
                    ),
                    "fields_by_source": fields_by_source,
                    "residuals_by_source": residuals_by_source,
                }
            result["stages"][mode_key] = grouped

        for run_index, row in enumerate(matrix_rows):
            by_mask = {}
            for mask in sorted(reference_masks):
                by_mask[mask] = [
                    sample["all_seven_wall_ms"]
                    for sample in sorted(
                        (sample for sample in row["samples"]
                         if sample["mask"] == mask and sample["source"] == 0),
                        key=lambda sample: sample["cycle"],
                    )
                ]
            result["matrix_smoke_runs"].append({
                "run_index": run_index,
                "joint_plan": row["configuration"]["joint_plan"],
                "cycles_per_mask": matrix_cycles,
                "all_seven_wall_ms_by_mask": by_mask,
            })
        results[arm] = result

        for hashes in matrix_hashes:
            _require(all(hashes[mask][source] == digest
                         for (mask, source), digest in frozen_maps.items()),
                     f"{arm}: maps disagree with independently checked frozen references")
        if arm != ARMS[0]:
            for key in ("source_identity_sha256", "indexed_resident_index_bytes"):
                _require(ready.get(key) == results[ARMS[0]]["ready"].get(key),
                         f"{arm}: {key} differs from the first control")

    a1_bytes = results["control-a1"]["ready"]["series_resident_bytes"]
    compact_bytes = results["compact"]["ready"]["series_resident_bytes"]
    a2_bytes = results["control-a2"]["ready"]["series_resident_bytes"]
    _require(a1_bytes == a2_bytes,
             f"Control resident-byte readings disagree: A1={a1_bytes}, A2={a2_bytes}")
    results["memory_comparison"] = {
        "control_a1_resident_bytes": a1_bytes,
        "compact_resident_bytes": compact_bytes,
        "control_a2_resident_bytes": a2_bytes,
        "control_a1_a2_match": True,
        "resident_bytes_saved_vs_control": a1_bytes - compact_bytes,
        "scope": "ready-state retained bytes only; not construction peak",
    }
    results["cross_arm_exact_hashes"] = {
        "dp_audit_keys": len(reference_dp),
        "full_map_hashes_match": True,
        "independent_frozen_map_keys": len(frozen_maps),
        "source_identities_and_index_bytes_match": True,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
