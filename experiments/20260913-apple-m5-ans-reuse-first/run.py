"""Compare exact compact-offset residents with controls and frozen map hashes."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

SOURCE_COUNT = 7
REFERENCE_MASK_COUNT = 20
STAGES = ("index", "residual", "combined")
DP_SCANS = (
    0, 1, 30, 31, 32, 33, 510, 511, 512, 513,
    16_383, 16_384, 16_385, 262_142, 262_143,
    *((index * 7919 + 123) % 262_144 for index in range(20)),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _reference_maps(path: Path) -> dict[tuple[str, int], str]:
    result = {}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") == "ans_opt_independent_parity":
                key = record["mask"], record["source"]
                if key in result:
                    raise ValueError(f"Duplicate frozen map key {key} in {path}")
                result[key] = record["sha256_u32_le"]
    masks = {mask for mask, _ in result}
    expected = {(mask, source) for mask in masks for source in range(SOURCE_COUNT)}
    if len(masks) != REFERENCE_MASK_COUNT or set(result) != expected:
        raise ValueError(
            f"Expected {REFERENCE_MASK_COUNT} masks × {SOURCE_COUNT} sources "
            f"({len(expected)} frozen map keys); found {len(result)} keys in {path}")
    return result


def _check_sample_grid(samples, cycles: int, masks: set[str], label: str) -> None:
    expected = {
        (cycle, mask, source)
        for cycle in range(cycles)
        for mask in masks
        for source in range(SOURCE_COUNT)
    }
    observed = Counter(
        (sample.get("cycle"), sample.get("mask"), sample.get("source"))
        for sample in samples
    )
    duplicates = sorted((key for key, count in observed.items() if count != 1), key=repr)
    missing = expected - set(observed)
    unexpected = set(observed) - expected
    _require(
        not duplicates and not missing and not unexpected,
        f"{label}: sample grid mismatch; duplicates={duplicates[:3]}, "
        f"missing={sorted(missing, key=repr)[:3]}, "
        f"unexpected={sorted(unexpected, key=repr)[:3]}",
    )


def _run_arm(args, arm: str, compact: bool, references, diffraction_reference):
    output = args.out / arm
    output.mkdir()
    environment = os.environ | {
        "QGPU_ANS_RESIDENT_LOOP": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS": "1",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "1" if compact else "0",
        "QGPU_PAIRED_RUNTIME_JOINT_PLAN": "0",
    }
    for name in list(environment):
        if name.startswith("QGPU_PREPARE_"):
            environment[name] = "0"
    reference_masks = {mask for mask, _ in references}
    expected_dp = {(source, scan) for source in range(SOURCE_COUNT) for scan in DP_SCANS}
    observed_diffraction = {}
    with (output / "raw.jsonl").open("x") as raw, \
            (output / "trials.jsonl").open("x") as trials, \
            (output / "stderr.log").open("x") as stderr:
        trials.write(json.dumps({
            "event": "run_metadata", "repeats": args.repeats,
            "matrix_cycles": args.matrix_cycles,
            "reference_masks": sorted(reference_masks),
            "expected_sources": SOURCE_COUNT,
            "expected_dp_scans": list(DP_SCANS),
        }) + "\n")
        trials.flush()
        process = subprocess.Popen(
            [str(args.exe.resolve()), str(args.folder.resolve()), str(args.cache.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, bufsize=1, env=environment,
        )

        def receive(event):
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                record = json.loads(line)
                if record.get("event") == "ans_resident_loop_error":
                    raise RuntimeError(record)
                if record.get("event") == event:
                    return record
            raise RuntimeError(f"{arm} exited before {event}: {process.poll()}")

        def request(command, event):
            process.stdin.write(json.dumps(command) + "\n")
            process.stdin.flush()
            return receive(event)

        try:
            ready = receive("ans_resident_loop_ready")
            print(json.dumps({"arm": arm, "ready": ready}), flush=True)
            _require(ready["resident_count"] == SOURCE_COUNT,
                     "The complete seven-source resident is required")
            _require(ready["compact_offsets_enabled"] == [compact] * SOURCE_COUNT,
                     "The actual resident offset layout did not match the arm")
            if compact:
                _require(
                    len(ready["compact_offset_bytes"]) == SOURCE_COUNT
                    and all(value > 0 for value in ready["compact_offset_bytes"]),
                    "Compact-offset arm did not report seven non-empty compact directories",
                )
            diffraction = request({"op": "dp_audit"}, "dp_audit")
            for sample in diffraction["samples"]:
                key = sample["source"], sample["scan"]
                _require(key in expected_dp, f"Unexpected diffraction audit key {key}")
                _require(key not in observed_diffraction, f"Duplicate diffraction audit key {key}")
                observed_diffraction[key] = sample["sha256_u32_le"]
            _require(
                set(observed_diffraction) == expected_dp,
                f"Diffraction audit coverage mismatch: expected {len(expected_dp)} unique keys, "
                f"received {len(observed_diffraction)}",
            )
            if diffraction_reference is not None:
                _require(
                    observed_diffraction == diffraction_reference,
                    "Diffraction hashes differ from the preceding arm",
                )
            schedule = [(False, -1), (True, -1)]
            generator = random.Random(1713)
            for repetition in range(args.repeats):
                order = [False, True]
                generator.shuffle(order)
                schedule.extend((joint, repetition) for joint in order)
            for joint, repetition in schedule:
                result = request({"op": "stage_isolation", "reuse_scratch": True,
                                  "joint_plan": joint}, "stage_isolation")
                _require(result["exact"], f"Stage-isolation parity failed: {result}")
                _require(result["joint_plan"] == joint,
                         f"Stage-isolation mode echo mismatch: expected {joint}")
                expected_stage_keys = {
                    (stage, source) for stage in STAGES for source in range(SOURCE_COUNT)
                }
                stage_counts = Counter(
                    (sample.get("stage"), sample.get("source"))
                    for sample in result["samples"]
                )
                _require(
                    set(stage_counts) == expected_stage_keys
                    and all(count == 1 for count in stage_counts.values())
                    and len(result["samples"]) == len(expected_stage_keys),
                    f"Stage-isolation sample coverage mismatch: {result}",
                )
                trials.write(json.dumps({"event": "stage_trial", "sequence": result["sequence"],
                                         "joint": joint, "repetition": repetition,
                                         "response": result}) + "\n")
                trials.flush()
                if repetition % 5 == 0 and joint:
                    print(f"{arm}: repeat {repetition}, exact stage outputs", flush=True)
            for joint in [False, True, False]:
                result = request({"op": "run", "mode": "indexed", "batch": False,
                                  "joint_plan": joint, "cycles": args.matrix_cycles,
                                  "arm": "candidate" if joint else "A1"},
                                 "ans_resident_loop_result")
                _require(result["fullmap_parity"],
                         f"Full-map parity failed: {result['fullmap_mismatches']}")
                _require(result["configuration"]["joint_plan"] == joint,
                         f"Full-map run mode mismatch: expected joint_plan={joint}")
                _require(result["cycles"] == args.matrix_cycles,
                         "Benchmark returned a different matrix cycle count")
                _require(set(result["masks"]) == reference_masks
                         and len(result["masks"]) == len(reference_masks),
                         "Benchmark mask set does not match the frozen reference set")
                _check_sample_grid(
                    result["samples"], args.matrix_cycles, reference_masks,
                    f"{arm} joint={joint} full-map matrix",
                )
                for sample in result["samples"]:
                    key = sample["mask"], sample["source"]
                    _require(key in references, f"Unexpected full-map hash key {key}")
                    _require(sample["sha256_u32_le"] == references[key],
                             f"Frozen full-map hash mismatch for {key}")
                print(f"{arm}: joint={joint}, {len(result['samples'])} frozen maps exact",
                      flush=True)
            request({"op": "quit"}, "ans_resident_loop_end")
            if process.wait(timeout=30) != 0:
                raise RuntimeError(f"{arm} failed on teardown")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return observed_diffraction


def main() -> None:
    """Run a sequential control/candidate/control comparison.

    Examples
    --------
    Use the full local acceptance folder and previously authenticated maps::

        python run.py --exe PATH --folder PATH --cache PATH --reference PATH --out PATH
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["exe", "folder", "cache", "reference", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--matrix-cycles", type=int, default=2)
    args = parser.parse_args()
    _require(args.repeats > 0, "--repeats must be positive")
    _require(args.matrix_cycles > 0, "--matrix-cycles must be positive")
    _require(len(set(DP_SCANS)) == 35, "The expected diffraction scan fixture is not unique")
    args.out.mkdir(parents=True, exist_ok=True)
    references = _reference_maps(args.reference)
    diffraction_reference = None
    for arm, compact in [("control-a1", False), ("compact", True), ("control-a2", False)]:
        diffraction_reference = _run_arm(
            args, arm, compact, references, diffraction_reference)
    _require(diffraction_reference is not None and len(diffraction_reference) == 245,
             "Final diffraction reference coverage is incomplete")
    with args.exe.open("rb") as executable:
        digest = hashlib.sha256()
        for chunk in iter(lambda: executable.read(1024 * 1024), b""):
            digest.update(chunk)
        binary_sha256 = digest.hexdigest()
    print(json.dumps({"exact": True, "dp_hashes_per_arm": len(diffraction_reference),
                      "binary_sha256": binary_sha256}), flush=True)


if __name__ == "__main__":
    main()
