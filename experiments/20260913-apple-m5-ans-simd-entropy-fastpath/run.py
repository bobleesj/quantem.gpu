"""Run the exact indexed seven-source SIMD32 entropy-path A/B/A test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


SOURCE_COUNT = 7
CYCLES = 21  # One warmup and twenty measured cycles per arm.
MASKS = ("adf-center-8", "adf-center-20")
ARMS = (("A1", "A1", False), ("B", "candidate", True), ("A2", "A2", False))
SOURCE_FILES = {
    "benchmark": "src/quantem/gpu/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark/main.swift",
    "resident": "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/MetalPairedRuntimeTANSResidentSource.swift",
    "kernel_api": "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/MetalPairedRuntimeTANSKernels.swift",
    "shader": "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/paired_runtime_tans.metal",
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, cwd=root,
        text=not binary,
    )
    return result.stdout.strip() if not binary else result.stdout


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "cannot summarize an empty timing sample")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(values),
        "p95_ms_nearest_rank": _percentile(values, 0.95),
    }


def _fingerprint_code(root: Path, executable: Path, manifest: dict) -> None:
    diff = _git(root, "diff", "HEAD", "--binary", binary=True)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    manifest["code"].update({
        "revision": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tested_binary": {
            "path": "<provided by --exe>",
            "sha256": _sha256(executable),
        },
        "source_fingerprints": {
            name: {"path": path, "sha256": _sha256(root / path)}
            for name, path in SOURCE_FILES.items()
        },
        "runner_sha256": _sha256(Path(__file__).resolve()),
    })


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if (key.startswith("QGPU_PAIRED_RUNTIME_")
                or key.startswith("QGPU_PREPARE_")
                or key.startswith("QGPU_ANS_OPT_")):
            environment[key] = "0"
    environment.update({
        "QGPU_ANS_RESIDENT_LOOP": "1",
        "QGPU_ANS_OPT_EXPERIMENT": "0",
        "QGPU_ANS_OPT_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS": "1",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "0",
        "QGPU_PAIRED_RUNTIME_JOINT_PLAN": "0",
        "QGPU_PAIRED_RUNTIME_PREPARE_SIMD_ENTROPY_FAST_PATH": "1",
        "QGPU_PAIRED_RUNTIME_SIMD_ENTROPY_FAST_PATH": "0",
    })
    return environment


def _receive(process: subprocess.Popen, raw, event: str) -> dict:
    assert process.stdout is not None
    for line in process.stdout:
        raw.write(line)
        raw.flush()
        record = json.loads(line)
        if record.get("event") == "ans_resident_loop_error":
            raise RuntimeError(record.get("error", "resident benchmark reported an error"))
        if record.get("event") == event:
            return record
        if record.get("event") == "ans_resident_loop_end":
            raise RuntimeError(f"benchmark ended before {event}: {record}")
    raise RuntimeError(f"benchmark exited before {event}: {process.poll()}")


def _request(process: subprocess.Popen, raw, command: dict) -> dict:
    assert process.stdin is not None
    process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
    process.stdin.flush()
    return _receive(process, raw, "ans_resident_loop_result")


def _sample_record(sample: dict) -> dict:
    fields = (
        "cycle", "mask", "source", "all_seven_wall_ms", "wall_ms", "gpu_ms",
        "changed_pixels", "excluded_pixels", "polar_field_count",
        "polar_residual_count", "sha256_u32_le", "effective_kernel",
        "history_hit", "history_base",
    )
    return {field: sample[field] for field in fields}


def _validate_ready(record: dict) -> list[str]:
    _require(record.get("resident_count") == SOURCE_COUNT,
             "expected exactly seven resident acquisitions")
    _require(record.get("shape") == [512, 512, 192, 192],
             "the complete 512x512x192x192 workload is required")
    _require(record.get("logical_dtype") == "uint16",
             "the exact full-uint16 workload is required")
    _require(record.get("simd_entropy_fast_path_pipeline_prepared") is True,
             "the opt-in SIMD entropy pipeline was not prepared before resident loading")
    _require(record.get("indexed_mode_available") is True,
             "the full indexed resident mode was not prepared")
    _require(record.get("polar_query_variant") == "packet-groups",
             "the baseline polar query variant must be packet-groups")
    _require(record.get("polar_query_scan512_ab_a1_b_a2") is False,
             "the scan512 A/B/A experiment flag must be disabled for this run")
    scan512_prepared = record.get("polar_query_scan512_pipeline_prepared")
    _require(scan512_prepared == [False] * SOURCE_COUNT,
             "scan512 pipelines must remain unprepared in the SIMD entropy experiment")
    identities = record.get("source_identity_sha256", [])
    _require(len(identities) == SOURCE_COUNT and len(set(identities)) == SOURCE_COUNT,
             "the seven resident sources must have distinct identity hashes")
    _require(all(isinstance(value, str) and len(value) == 64
                 and all(char in "0123456789abcdef" for char in value)
                 for value in identities),
             "resident source identities must be lowercase SHA-256 values")
    _require(isinstance(record.get("series_resident_bytes"), int),
             "ready response omitted total resident bytes")
    _require(isinstance(record.get("resident_bytes_by_source"), list)
             and len(record["resident_bytes_by_source"]) == SOURCE_COUNT,
             "ready response omitted per-source resident bytes")
    _require(sum(record["resident_bytes_by_source"]) == record["series_resident_bytes"],
             "total resident bytes disagree with per-source residents")
    return identities


def _validate_response(
    response: dict, arm: str, enabled: bool, ready: dict,
    expected_a1_map_hashes: dict | None,
) -> tuple[dict, dict, dict]:
    _require(response.get("arm") == arm, f"unexpected response arm: {response.get('arm')}")
    _require(response.get("fullmap_parity") is True,
             f"full-map exact parity failed for {arm}")
    _require(response.get("exact_a1_hashes") is True,
             f"frozen A1 parity failed for {arm}")
    _require(response.get("series_resident_bytes") == ready["series_resident_bytes"],
             f"resident bytes changed in {arm}")
    _require(isinstance(response.get("metal_current_allocated_bytes"), int),
             f"Metal allocation sample missing in {arm}")
    config = response.get("configuration", {})
    requested = response.get("requested_configuration", {})
    _require(config.get("mode") == "indexed" and config.get("kernel") == "packet-owner2",
             f"unexpected decoder configuration in {arm}")
    _require(config.get("polar_query_variant") == "packet-groups",
             f"unexpected polar query variant in {arm}")
    _require(config.get("simd_entropy_fast_path") is enabled
             and requested.get("simd_entropy_fast_path") is enabled,
             f"SIMD entropy setting did not take effect in {arm}")
    _require(config.get("batch") is False and config.get("streams_per_lane") == 2
             and config.get("packet_splits") == 1,
             f"unexpected submission or decoder geometry in {arm}")
    _require(config.get("bounded_concurrency") == SOURCE_COUNT,
             f"bounded concurrency was not seven in {arm}")
    _require(config.get("profile") is False,
             f"timing must be unprofiled in {arm}")
    _require(response.get("cycles") == CYCLES and response.get("masks") == list(MASKS),
             f"cycle count or mask order changed in {arm}")
    response_hashes = response.get("sha256_u32_le")
    _require(isinstance(response_hashes, dict), f"missing full-map hashes in {arm}")
    actual_a1_map_hashes = response.get("a1_sha256_u32_le")
    _require(isinstance(actual_a1_map_hashes, dict),
             f"missing frozen A1 full-map hashes in {arm}")
    _require(response_hashes == actual_a1_map_hashes,
             f"response full-map hashes differ from frozen A1 maps in {arm}")
    if expected_a1_map_hashes is not None:
        _require(actual_a1_map_hashes == expected_a1_map_hashes,
                 f"A1 reference hashes changed in {arm}")

    observed = {}
    for sample in response.get("samples", []):
        key = (sample.get("cycle"), sample.get("mask"), sample.get("source"))
        _require(key not in observed, f"duplicate cycle/mask/source sample in {arm}: {key}")
        observed[key] = sample
    expected = {
        (cycle, mask, source)
        for cycle in range(CYCLES) for mask in MASKS for source in range(SOURCE_COUNT)
    }
    _require(set(observed) == expected, f"incomplete sample grid in {arm}")
    for (cycle, mask, source), sample in observed.items():
        _require(sample.get("sha256_u32_le") == response_hashes[mask][source],
                 f"per-cycle hash does not match full-map hash in {arm}: {(cycle, mask, source)}")

    cycle_hashes = {
        key: sample["sha256_u32_le"] for key, sample in observed.items()
    }
    arm_record = {
        "arm": arm,
        "simd_entropy_fast_path": enabled,
        "configuration": config,
        "requested_configuration": requested,
        "fullmap_parity": True,
        "exact_a1_hashes": True,
        "a1_sha256_u32_le": response["a1_sha256_u32_le"],
        "sha256_u32_le": response_hashes,
        "series_resident_bytes": response["series_resident_bytes"],
        "metal_current_allocated_bytes": response.get("metal_current_allocated_bytes"),
        "source_identity_sha256": ready["source_identity_sha256"],
        "source_identity_scope": "same seven immutable residents held by the one benchmark process",
        "samples": [_sample_record(observed[key]) for key in sorted(observed)],
    }
    wall_by_mask = {}
    all_wall = []
    for mask in MASKS:
        per_cycle = []
        for cycle in range(1, CYCLES):
            source_values = [observed[(cycle, mask, source)]["all_seven_wall_ms"]
                             for source in range(SOURCE_COUNT)]
            _require(all(value == source_values[0] for value in source_values),
                     f"inconsistent all-seven wall value across sources in {arm}/{mask}/{cycle}")
            per_cycle.append(source_values[0])
        wall_by_mask[mask] = {
            "samples_ms": per_cycle,
            **_timing_summary(per_cycle),
        }
        all_wall.extend(per_cycle)
    arm_record["all_seven_wall_ms"] = {
        "measured_cycles_per_mask": CYCLES - 1,
        "samples_by_mask": wall_by_mask,
        "combined_samples_ms": all_wall,
        **_timing_summary(all_wall),
    }
    return arm_record, cycle_hashes, actual_a1_map_hashes


def _record_outputs(root: Path, out: Path, manifest: dict) -> None:
    outputs = []
    descriptions = {
        "raw": "Raw JSON-lines resident responses for every A/B/A arm and release.",
        "ready": "Seven-source identities, shapes, resident bytes, and prepared-pipeline gate.",
        "summaries": "Exact arm samples, cross-arm hash checks, allocation, and wall summaries.",
        "stderr": "Benchmark process diagnostics.",
        "failure": "Structured failure status and release-cleanup evidence, when a run fails.",
    }
    for name, result in descriptions.items():
        path = out / ("stderr.log" if name == "stderr" else f"{name}.json" if name != "raw" else "raw.jsonl")
        if not path.is_file():
            continue
        outputs.append({
            "artifact_id": name,
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "retention": "durable",
            "consuming_figures": [],
            "result": result,
        })
    manifest["outputs"] = outputs


def _update_registry(root: Path, manifest: dict, out: Path) -> None:
    registry = root / "experiments/RUNS.md"
    lines = registry.read_text(encoding="utf-8").splitlines()
    experiment_id = manifest["experiment_id"]
    matches = [index for index, line in enumerate(lines)
               if line.startswith(f"| {experiment_id} |")]
    _require(len(matches) == 1, f"expected one RUNS.md row for {experiment_id}")
    fields = [field.strip() for field in lines[matches[0]].strip("|").split("|")]
    if manifest["status"] == "completed":
        fields[3] = "ok"
        latency = manifest["parameters"]["latency_summary"]
        fields[4] = "Exact parity; all-seven p50/p95 ms " + ", ".join(
            f"{arm} {latency[arm]['p50_ms']:.2f}/{latency[arm]['p95_ms_nearest_rank']:.2f}"
            for arm in ("A1", "B", "A2")
        ) + "; residents unchanged and released"
    else:
        fields[3] = "failed"
        failure = manifest["execution"].get("failure") or {}
        message = f"{failure.get('type', 'Failure')}: {failure.get('message', 'unknown')}"
        message = " ".join(message.replace("|", "/").split())
        fields[4] = f"Harness failed: {message[:180]}"
    fields[5] = (
        f"[manifest]({experiment_id}/manifest.json); "
        f"[raw]({out.joinpath('raw.jsonl').relative_to(root).as_posix()})"
    )
    lines[matches[0]] = "| " + " | ".join(fields) + " |"
    registry.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.exe = args.exe.resolve()
    args.folder = args.folder.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    args.out = args.out.resolve()
    root = Path(__file__).resolve().parents[2]
    manifest_path = args.out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _require(args.exe.is_file(), f"benchmark executable not found: {args.exe}")
    _require(args.folder.is_dir(), f"source folder not found: {args.folder}")
    _require(not args.out.exists(), f"output path already exists: {args.out}")
    _require(args.cache != args.out, "cache and result paths must be different")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True)
    _fingerprint_code(root, args.exe, manifest)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["status"] = "running"
    manifest["timestamps"] = {"started": started, "finished": None}
    manifest["execution"]["failure"] = None
    manifest["execution"]["release"] = None
    manifest["parameters"]["arm_samples"] = []
    _write_json(manifest_path, manifest)

    raw_path = args.out / "raw.jsonl"
    ready = None
    release = None
    failure = None
    arms = []
    previous_cycle_hashes = None
    a1_map_hashes = None
    process = None
    with raw_path.open("x", encoding="utf-8") as raw, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                [str(args.exe), str(args.folder), str(args.cache)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, bufsize=1, env=_environment(),
            )
            ready = _receive(process, raw, "ans_resident_loop_ready")
            identities = _validate_ready(ready)
            manifest["parameters"]["source_identity_sha256"] = identities
            manifest["parameters"]["resident_bytes_by_source"] = ready["resident_bytes_by_source"]
            manifest["parameters"]["series_resident_bytes"] = ready["series_resident_bytes"]
            manifest["parameters"]["ready_metal_current_allocated_bytes"] = ready.get(
                "metal_current_allocated_bytes")
            _write_json(args.out / "ready.json", ready)

            for name, requested_arm, enabled in ARMS:
                command = {
                    "op": "run", "arm": requested_arm, "mode": "indexed",
                    "kernel": "packet-owner2", "batch": False,
                    "bounded_concurrency": SOURCE_COUNT, "profile": False,
                    "cycles": CYCLES, "mask_names": list(MASKS),
                    "simd_entropy_fast_path": enabled,
                }
                response = _request(process, raw, command)
                expected_arm = "candidate" if name == "B" else name
                arm_record, cycle_hashes, returned_a1_map_hashes = _validate_response(
                    response, expected_arm, enabled, ready, a1_map_hashes)
                if a1_map_hashes is None:
                    a1_map_hashes = returned_a1_map_hashes
                if previous_cycle_hashes is None:
                    previous_cycle_hashes = cycle_hashes
                _require(cycle_hashes == previous_cycle_hashes,
                         f"per-cycle full-map hashes differ across A/B/A at {name}")
                arm_record["label"] = name
                arms.append(arm_record)
                manifest["parameters"]["arm_samples"] = arms
                manifest["parameters"]["latency_summary"] = {
                    item["label"]: item["all_seven_wall_ms"] for item in arms
                }
                manifest["parameters"]["allocation_bytes_by_arm"] = {
                    item["label"]: {
                        "series_resident_bytes": item["series_resident_bytes"],
                        "metal_current_allocated_bytes": item[
                            "metal_current_allocated_bytes"],
                    }
                    for item in arms
                }
                _write_json(args.out / "summaries.json", {"arm_samples": arms})
                _write_json(manifest_path, manifest)
                print(json.dumps({
                    "arm": name,
                    "simd_entropy_fast_path": enabled,
                    "parity": True,
                    "series_resident_bytes": arm_record["series_resident_bytes"],
                    "metal_current_allocated_bytes": arm_record[
                        "metal_current_allocated_bytes"],
                    "all_seven_wall_ms": {
                        "p50": arm_record["all_seven_wall_ms"]["p50_ms"],
                        "p95_nearest_rank": arm_record["all_seven_wall_ms"][
                            "p95_ms_nearest_rank"],
                    },
                }, sort_keys=True), flush=True)

            assert process.stdin is not None
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            release = _receive(process, raw, "ans_resident_loop_end")
            _require(release.get("all_released") is True,
                     "benchmark did not release all seven residents")
            if process.wait(timeout=30) != 0:
                raise RuntimeError("benchmark exited unsuccessfully after release")
            _require(len(arms) == len(ARMS), "not all three A/B/A arms completed")
            manifest["status"] = "completed"
            manifest["parameters"]["per_cycle_hashes_match_across_arms"] = True
            manifest["parameters"]["source_identities_unchanged"] = True
            manifest["parameters"]["resident_bytes_unchanged"] = True
            manifest["parameters"]["all_residents_released"] = True
            manifest["parameters"]["failure_status"] = "completed"
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = "failed"
            manifest["execution"]["failure"] = failure
            manifest["parameters"]["failure_status"] = "failed"
        finally:
            if process is not None and process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write('{"op":"quit"}\n')
                    process.stdin.flush()
                    release = _receive(process, raw, "ans_resident_loop_end")
                    process.wait(timeout=30)
                except Exception as cleanup_error:
                    if failure is None:
                        failure = {
                            "type": type(cleanup_error).__name__,
                            "message": f"release cleanup failed: {cleanup_error}",
                        }
                        manifest["status"] = "failed"
                        manifest["execution"]["failure"] = failure
                        manifest["parameters"]["failure_status"] = "failed"
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()

    manifest["execution"]["release"] = release
    if failure is None and release is not None:
        manifest["parameters"]["all_residents_released"] = release.get("all_released") is True
    if failure is not None:
        manifest["execution"]["failure"] = failure
        _write_json(args.out / "failure.json", {
            "status": "failed", "failure": failure, "release": release,
        })
    if ready is not None:
        _write_json(args.out / "ready.json", ready)
    _write_json(args.out / "summaries.json", {"arm_samples": arms})
    manifest["timestamps"]["finished"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _record_outputs(root, args.out, manifest)
    _write_json(manifest_path, manifest)
    _update_registry(root, manifest, args.out)
    if failure is not None:
        raise RuntimeError(f"experiment failed; manifest records details: {failure}")
    print(f"completed; outputs and manifest retained under {args.out}", flush=True)


if __name__ == "__main__":
    main()
