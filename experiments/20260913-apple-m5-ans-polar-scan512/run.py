"""Run the exact indexed seven-source resident-loop scan512 A/B/A test."""

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
CYCLES = 20
WARMUP_CYCLES = 1
MASKS = ("adf-center-8", "adf-center-20")
ARMS = (
    ("A1", "A1", "packet-groups"),
    ("B", "candidate", "scan512"),
    ("A2", "A2", "packet-groups"),
)
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
        "QGPU_ANS_OPT_POLAR_QUERY_SCAN512": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT": "packet-groups",
        "QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512": "1",
        "QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS": "1",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "0",
        "QGPU_PAIRED_RUNTIME_JOINT_PLAN": "0",
    })
    return environment


def _fingerprint_code(
    root: Path, executable: Path, manifest: dict, runner: Path | None = None,
) -> None:
    diff = _git(root, "diff", "HEAD", "--binary", binary=True)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    manifest["code"].update({
        "revision": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tested_binary": {"path": executable.name, "sha256": _sha256(executable)},
        "source_fingerprints": {
            name: {"path": relative, "sha256": _sha256(root / relative)}
            for name, relative in SOURCE_FILES.items()
        },
        "runner_sha256": _sha256(runner or Path(__file__).resolve()),
    })


def _receive(process: subprocess.Popen, raw, expected_event: str) -> dict:
    assert process.stdout is not None
    for line in process.stdout:
        raw.write(line)
        raw.flush()
        record = json.loads(line)
        if record.get("event") == "ans_resident_loop_error":
            raise RuntimeError(record.get("error", "resident benchmark reported an error"))
        if record.get("event") == expected_event:
            return record
        if record.get("event") == "ans_resident_loop_end":
            raise RuntimeError(f"benchmark ended before {expected_event}: {record}")
    raise RuntimeError(f"benchmark exited before {expected_event}: {process.poll()}")


def _request(process: subprocess.Popen, raw, command: dict) -> dict:
    assert process.stdin is not None
    process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
    process.stdin.flush()
    return _receive(process, raw, "ans_resident_loop_result")


def _validate_ready(record: dict) -> list[str]:
    _require(record.get("resident_count") == SOURCE_COUNT,
             "expected exactly seven resident acquisitions")
    _require(record.get("shape") == [512, 512, 192, 192],
             "the complete 512x512x192x192 workload is required")
    _require(record.get("logical_dtype") == "uint16",
             "the exact full-uint16 workload is required")
    _require(record.get("indexed_mode_available") is True,
             "the full indexed resident mode was not prepared")
    _require(record.get("polar_query_variant") == "packet-groups",
             "startup query variant must be packet-groups")
    _require(record.get("polar_query_scan512_ab_a1_b_a2") is True,
             "the scan512 A/B/A startup flag was not enabled")
    prepared = record.get("polar_query_scan512_pipeline_prepared")
    _require(prepared == [True] * SOURCE_COUNT,
             "all seven scan512 query pipelines must be prepared before measurement")
    identities = record.get("source_identity_sha256", [])
    _require(all(isinstance(value, str) and len(value) == 64
                 and all(char in "0123456789abcdef" for char in value)
                 for value in identities),
             "resident source identities must be lowercase SHA-256 values")
    _require(len(identities) == SOURCE_COUNT and len(set(identities)) == SOURCE_COUNT,
             "the seven resident sources must have distinct identity hashes")
    resident_bytes = record.get("resident_bytes_by_source")
    _require(isinstance(resident_bytes, list) and len(resident_bytes) == SOURCE_COUNT,
             "ready response omitted per-source resident bytes")
    _require(all(type(value) is int and value > 0 for value in resident_bytes),
             "each source must report a positive integer resident-byte count")
    _require(sum(resident_bytes) == record.get("series_resident_bytes"),
             "total resident bytes disagree with per-source residents")
    _require(type(record.get("metal_current_allocated_bytes")) is int,
             "ready response omitted the device-allocation baseline")
    return identities


def _sample_record(sample: dict) -> dict:
    return dict(sample)


def _validate_response(
    response: dict, expected_arm: str, variant: str, ready: dict,
    expected_map_hashes: dict | None, cycles: int = CYCLES,
    streams_per_lane: int = 2,
) -> tuple[dict, dict, dict]:
    _require(response.get("arm") == expected_arm,
             f"unexpected response arm: {response.get('arm')}")
    _require(response.get("fullmap_parity") is True
             and response.get("exact_a1_hashes") is True,
             f"full-map exact parity failed in {expected_arm}")
    _require(response.get("series_resident_bytes") == ready["series_resident_bytes"],
             f"resident bytes changed in {expected_arm}")
    _require(type(response.get("metal_current_allocated_bytes")) is int,
             f"Metal allocation sample missing in {expected_arm}")
    _require(response.get("cycles") == cycles and response.get("masks") == list(MASKS),
             f"cycle count or mask order changed in {expected_arm}")
    config = response.get("configuration", {})
    requested = response.get("requested_configuration", {})
    expected_config = {
        "mode": "indexed", "kernel": "packet-owner2",
        "polar_query_variant": variant, "partial_groups": 8,
        "choose_base": False, "partial_stores": False,
        "streams_per_lane": streams_per_lane, "packet_splits": 1,
        "batch": False, "bounded_concurrency": SOURCE_COUNT,
        "reuse_word": False, "register_sums": False,
        "history": False, "history_base": False,
        "plain_sums": False, "trusted_table": False, "macro": False,
        "profile": False, "lazy_refill": False,
        "joint_plan": False, "simd_entropy_fast_path": False,
    }
    _require(config == expected_config,
             f"effective configuration mismatch in {expected_arm}: {config}")
    _require(requested == expected_config,
             f"requested configuration mismatch in {expected_arm}: {requested}")

    _require(response.get("metal_current_allocated_bytes")
             == ready.get("metal_current_allocated_bytes"),
             f"device allocation changed in {expected_arm}")

    map_hashes = response.get("sha256_u32_le")
    a1_hashes = response.get("a1_sha256_u32_le")
    _require(isinstance(map_hashes, dict) and map_hashes == a1_hashes,
             f"full-map hashes differ from the frozen A1 maps in {expected_arm}")
    if expected_map_hashes is not None:
        _require(a1_hashes == expected_map_hashes,
                 f"frozen A1 reference hashes changed in {expected_arm}")

    samples = response.get("samples", [])
    observed = {}
    for sample in samples:
        key = (sample.get("cycle"), sample.get("mask"), sample.get("source"))
        _require(key not in observed, f"duplicate sample in {expected_arm}: {key}")
        observed[key] = sample
        _require(sample.get("sha256_u32_le") == map_hashes[key[1]][key[2]],
                 f"per-cycle hash differs from full-map hash in {expected_arm}: {key}")
    expected_keys = {
        (cycle, mask, source)
        for cycle in range(cycles) for mask in MASKS for source in range(SOURCE_COUNT)
    }
    _require(set(observed) == expected_keys, f"incomplete sample grid in {expected_arm}")
    mask_order = [sample["mask"] for sample in samples if sample["source"] == 0]
    _require(mask_order == list(MASKS) * cycles,
             f"each center-20 update must follow center-8 in {expected_arm}")
    for sample in samples:
        _require(sample.get("effective_kernel") == "packet-owner2",
                 f"sample used the wrong detector kernel in {expected_arm}")

    cycle_hashes = {key: sample["sha256_u32_le"] for key, sample in observed.items()}
    by_mask = {}
    all_wall = []
    for mask in MASKS:
        timings = []
        for cycle in range(cycles):
            values = [observed[(cycle, mask, source)]["all_seven_wall_ms"]
                      for source in range(SOURCE_COUNT)]
            _require(all(value == values[0] for value in values),
                     f"all-seven wall time differs by source in {expected_arm}/{mask}/{cycle}")
            timings.append(values[0])
        by_mask[mask] = {"samples_ms": timings, **_timing_summary(timings)}
        all_wall.extend(timings)

    arm_record = {
        "arm": expected_arm,
        "label": "B" if variant == "scan512" else expected_arm,
        "polar_query_variant": variant,
        "configuration": config,
        "requested_configuration": requested,
        "fullmap_parity": True,
        "a1_sha256_u32_le": a1_hashes,
        "sha256_u32_le": map_hashes,
        "series_resident_bytes": response["series_resident_bytes"],
        "metal_current_allocated_bytes": response["metal_current_allocated_bytes"],
        "source_identity_sha256": ready["source_identity_sha256"],
        "samples": [_sample_record(observed[key]) for key in sorted(observed)],
        "all_seven_wall_ms": {
            "measured_cycles_per_mask": cycles,
            "samples_by_mask": by_mask,
            "combined_samples_ms": all_wall,
            **_timing_summary(all_wall),
        },
    }
    return arm_record, cycle_hashes, a1_hashes


def _record_outputs(root: Path, out: Path, manifest: dict) -> None:
    descriptions = {
        "raw": "All benchmark stdout JSON-line records, including ready, A/B/A responses, and release.",
        "ready": "Seven-source identities, shape, allocation, and scan512 pipeline preparation evidence.",
        "summary": "Exact arm samples, per-cycle hash checks, allocation, and all-seven wall summaries.",
        "stderr": "Benchmark process diagnostics.",
        "failure": "Structured failure and release-cleanup evidence, when a run fails.",
    }
    outputs = []
    for name, description in descriptions.items():
        filename = "stderr.log" if name == "stderr" else (
            "raw.jsonl" if name == "raw" else f"{name}.json")
        path = out / filename
        if not path.is_file():
            continue
        outputs.append({
            "artifact_id": name,
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "retention": "durable",
            "consuming_figures": [],
            "result": description,
        })
    manifest["outputs"] = outputs


def _update_registry(root: Path, out: Path, manifest: dict) -> None:
    registry = root / "experiments/RUNS.md"
    lines = registry.read_text(encoding="utf-8").splitlines()
    experiment_id = manifest["experiment_id"]
    matches = [index for index, line in enumerate(lines)
               if line.startswith(f"| {experiment_id} |")]
    _require(len(matches) == 1, f"expected one RUNS.md row for {experiment_id}")
    fields = [field.strip() for field in lines[matches[0]].strip("|").split("|")]
    if manifest["status"] == "completed":
        fields[3] = "ok"
        target_mask = MASKS[-1]
        p50 = manifest["parameters"]["latency_summary"][target_mask]
        fields[4] = f"{target_mask} p50/p95 ms A1/B/A2 " + ", ".join(
            f"{arm} {p50[arm]['p50_ms']:.2f}/{p50[arm]['p95_ms_nearest_rank']:.2f}"
            for arm in ("A1", "B", "A2")
        ) + "; parity and resident gates recorded"
    else:
        fields[3] = "failed"
        failure = manifest["execution"].get("failure") or {}
        message = " ".join(
            f"{failure.get('type', 'Failure')}: {failure.get('message', 'unknown')}"
            .replace("|", "/").split())
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
    manifest["status"] = "running"
    manifest["timestamps"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished": None,
    }
    manifest["execution"]["failure"] = None
    manifest["execution"]["release"] = None
    manifest["parameters"]["arm_samples"] = []
    manifest["parameters"]["warmup_samples"] = []
    _write_json(manifest_path, manifest)

    ready = None
    release = None
    failure = None
    arms = []
    expected_cycle_hashes = None
    expected_map_hashes = None
    process = None
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw, \
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
            manifest["parameters"]["resident_bytes_by_source"] = ready[
                "resident_bytes_by_source"]
            manifest["parameters"]["series_resident_bytes"] = ready[
                "series_resident_bytes"]
            manifest["parameters"]["ready_metal_current_allocated_bytes"] = ready[
                "metal_current_allocated_bytes"]
            _write_json(args.out / "ready.json", ready)

            for label, request_arm, variant in ARMS:
                warmup_command = {
                    "op": "run", "arm": request_arm, "mode": "indexed",
                    "kernel": "packet-owner2", "batch": False,
                    "bounded_concurrency": SOURCE_COUNT, "profile": False,
                    "cycles": WARMUP_CYCLES, "mask_names": list(MASKS),
                    "polar_query_variant": variant,
                }
                expected_arm = "candidate" if label == "B" else label
                warmup_response = _request(process, raw, warmup_command)
                warmup_record, _, warmup_map_hashes = _validate_response(
                    warmup_response, expected_arm, variant, ready,
                    expected_map_hashes, cycles=WARMUP_CYCLES)
                if expected_map_hashes is None:
                    expected_map_hashes = warmup_map_hashes
                manifest["parameters"]["warmup_samples"].append({
                    "label": label,
                    "polar_query_variant": variant,
                    "samples": warmup_record["samples"],
                    "fullmap_parity": True,
                })
                _write_json(manifest_path, manifest)
                command = {
                    "op": "run", "arm": request_arm, "mode": "indexed",
                    "kernel": "packet-owner2", "batch": False,
                    "bounded_concurrency": SOURCE_COUNT, "profile": False,
                    "cycles": CYCLES, "mask_names": list(MASKS),
                    "polar_query_variant": variant,
                }
                response = _request(process, raw, command)
                arm_record, cycle_hashes, map_hashes = _validate_response(
                    response, expected_arm, variant, ready, expected_map_hashes)
                if expected_map_hashes is None:
                    expected_map_hashes = map_hashes
                if expected_cycle_hashes is None:
                    expected_cycle_hashes = cycle_hashes
                _require(cycle_hashes == expected_cycle_hashes,
                         f"per-cycle full-map hashes differ across A/B/A at {label}")
                arm_record["label"] = label
                arms.append(arm_record)
                manifest["parameters"]["arm_samples"] = arms
                manifest["parameters"]["latency_summary"] = {
                    mask: {
                        arm["label"]: arm["all_seven_wall_ms"]["samples_by_mask"][mask]
                        for arm in arms
                    }
                    for mask in MASKS
                }
                manifest["parameters"]["allocation_bytes_by_arm"] = {
                    arm["label"]: {
                        "series_resident_bytes": arm["series_resident_bytes"],
                        "metal_current_allocated_bytes": arm[
                            "metal_current_allocated_bytes"],
                    }
                    for arm in arms
                }
                _write_json(args.out / "summary.json", {"arm_samples": arms})
                _write_json(manifest_path, manifest)
                print(json.dumps({
                    "arm": label, "polar_query_variant": variant,
                    "parity": True,
                    "target_mask": MASKS[-1],
                    "target_p50_ms": arm_record["all_seven_wall_ms"][
                        "samples_by_mask"][MASKS[-1]]["p50_ms"],
                    "target_p95_ms": arm_record["all_seven_wall_ms"][
                        "samples_by_mask"][MASKS[-1]]["p95_ms_nearest_rank"],
                    "series_resident_bytes": arm_record["series_resident_bytes"],
                    "metal_current_allocated_bytes": arm_record[
                        "metal_current_allocated_bytes"],
                }, sort_keys=True), flush=True)

            assert process.stdin is not None
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            release = _receive(process, raw, "ans_resident_loop_end")
            _require(release.get("resident_count") == SOURCE_COUNT
                     and release.get("all_released") is True,
                     "benchmark did not release all seven residents")
            if process.wait(timeout=30) != 0:
                raise RuntimeError("benchmark exited unsuccessfully after release")
            _require(len(arms) == len(ARMS), "not all three A/B/A arms completed")
            manifest["status"] = "completed"
            manifest["parameters"]["per_cycle_hashes_match_across_arms"] = True
            manifest["parameters"]["resident_bytes_unchanged"] = True
            manifest["parameters"]["source_identities_unchanged"] = True
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
    if release is not None:
        manifest["parameters"]["all_residents_released"] = (
            release.get("all_released") is True)
    if failure is not None:
        _write_json(args.out / "failure.json", {
            "status": "failed", "failure": failure, "release": release,
        })
    if ready is not None:
        _write_json(args.out / "ready.json", ready)
    _write_json(args.out / "summary.json", {"arm_samples": arms})
    manifest["timestamps"]["finished"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _record_outputs(root, args.out, manifest)
    _write_json(manifest_path, manifest)
    _update_registry(root, args.out, manifest)
    if failure is not None:
        raise RuntimeError(f"experiment failed; manifest records details: {failure}")
    print(f"completed; outputs and manifest retained under {args.out}", flush=True)


if __name__ == "__main__":
    main()
