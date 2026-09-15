"""Run an exact seven-source A/B/A test of adaptive residual partials."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


EXPERIMENT_ID = "20260913-apple-m5-ans-adaptive-partials"
SOURCE_COUNT = 7
CYCLES = 20
WARMUP_CYCLES = 1
MASKS = ("adf-center-8", "adf-center-20")
MAX_METAL_BYTES = 11_883_921_408
MAX_CANDIDATE_GROWTH_BYTES = 122 * 1024 * 1024
ARMS = (
    ("A1", "A1", "packet-owner2"),
    ("B", "candidate", "adaptive-partials"),
    ("A2", "A2", "packet-owner2"),
)
SOURCE_FILES = {
    "benchmark": "src/quantem/gpu/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark/main.swift",
    "resident": "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/MetalPairedRuntimeTANSResidentSource.swift",
    "kernel_api": "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/MetalPairedRuntimeTANSKernels.swift",
    "shader": "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/paired_runtime_tans.metal",
}


def _load_scan512_helpers():
    root = Path(__file__).resolve().parents[2]
    helper_path = root / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
    spec = importlib.util.spec_from_file_location("ans_polar_scan512_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared resident-loop helpers: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SOURCE_FILES = SOURCE_FILES
    return module


HELPERS = _load_scan512_helpers()


def _require(condition: bool, message: str) -> None:
    HELPERS._require(condition, message)


def _write_json(path: Path, value: object) -> None:
    HELPERS._write_json(path, value)


def _environment() -> dict[str, str]:
    environment = HELPERS._environment()
    environment.update({
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "1",
        "QGPU_PAIRED_RUNTIME_TRUSTED_TABLE": "0",
    })
    return environment


def _expected_configuration(kernel: str) -> dict:
    return {
        "mode": "indexed",
        "kernel": kernel,
        "polar_query_variant": "scan512",
        "partial_groups": 32,
        "choose_base": False,
        "partial_stores": False,
        "streams_per_lane": 2,
        "packet_splits": 1,
        "batch": False,
        "bounded_concurrency": SOURCE_COUNT,
        "reuse_word": False,
        "register_sums": False,
        "history": False,
        "history_base": False,
        "plain_sums": False,
        "trusted_table": False,
        "profile": False,
        "lazy_refill": False,
        "joint_plan": False,
        "simd_entropy_fast_path": False,
        "macro": False,
    }


def _validate_ready(record: dict) -> list[str]:
    _require(record.get("resident_count") == SOURCE_COUNT,
             "expected exactly seven resident acquisitions")
    _require(record.get("shape") == [512, 512, 192, 192],
             "the full 512x512x192x192 workload is required")
    _require(record.get("logical_dtype") == "uint16",
             "the full-uint16 workload is required")
    _require(record.get("indexed_mode_available") is True,
             "the indexed resident mode was not prepared")
    _require(record.get("polar_query_variant") == "packet-groups",
             "resident startup must begin at packet-groups before scan512 requests")
    _require(record.get("polar_query_scan512_ab_a1_b_a2") is True,
             "the scan512 A/B/A startup flag was not enabled")
    _require(record.get("polar_query_scan512_pipeline_prepared") == [True] * SOURCE_COUNT,
             "all seven scan512 query pipelines must be prepared")
    _require(record.get("compact_offsets_enabled") == [True] * SOURCE_COUNT,
             "compact offsets must be enabled for all seven residents")
    _require(record.get("compact_offsets_requested") is True,
             "compact offsets were not requested by the harness")
    offset_bytes = record.get("compact_offset_bytes")
    _require(isinstance(offset_bytes, list) and len(offset_bytes) == SOURCE_COUNT
             and all(type(value) is int and value > 0 for value in offset_bytes),
             "ready response omitted positive per-source compact-offset savings")
    identities = record.get("source_identity_sha256", [])
    _require(len(identities) == SOURCE_COUNT
             and len(set(identities)) == SOURCE_COUNT
             and all(isinstance(value, str) and len(value) == 64 for value in identities),
             "seven distinct source identity hashes are required")
    resident_bytes = record.get("resident_bytes_by_source")
    _require(isinstance(resident_bytes, list) and len(resident_bytes) == SOURCE_COUNT
             and all(type(value) is int and value > 0 for value in resident_bytes),
             "ready response omitted per-source resident bytes")
    _require(sum(resident_bytes) == record.get("series_resident_bytes"),
             "resident byte total disagrees with per-source residents")
    allocation = record.get("metal_current_allocated_bytes")
    _require(type(allocation) is int and allocation <= MAX_METAL_BYTES,
             "compact-offset ready allocation exceeds the absolute Metal cap")
    return identities


def _validate_response(
    response: dict, label: str, kernel: str, ready: dict,
    expected_hashes: dict | None, cycles: int,
) -> tuple[dict, dict]:
    expected_arm = "candidate" if label == "B" else label
    _require(response.get("arm") == expected_arm,
             f"unexpected response arm in {label}: {response.get('arm')}")
    _require(response.get("fullmap_parity") is True
             and response.get("exact_a1_hashes") is True,
             f"full-map exact parity failed in {label}")
    _require(response.get("series_resident_bytes") == ready["series_resident_bytes"],
             f"resident bytes changed in {label}")
    _require(response.get("cycles") == cycles and response.get("masks") == list(MASKS),
             f"cycle count or mask order changed in {label}")
    expected_config = _expected_configuration(kernel)
    _require(response.get("configuration") == expected_config,
             f"effective configuration mismatch in {label}: {response.get('configuration')}")
    _require(response.get("requested_configuration") == expected_config,
             f"requested configuration mismatch in {label}: {response.get('requested_configuration')}")

    hashes = response.get("sha256_u32_le")
    frozen = response.get("a1_sha256_u32_le")
    _require(isinstance(hashes, dict) and hashes == frozen,
             f"full-map hashes differ from the frozen A1 maps in {label}")
    if expected_hashes is not None:
        _require(frozen == expected_hashes,
                 f"frozen A1 reference hashes changed in {label}")

    samples = response.get("samples", [])
    observed = {}
    for sample in samples:
        key = (sample.get("cycle"), sample.get("mask"), sample.get("source"))
        _require(key not in observed, f"duplicate sample in {label}: {key}")
        _require(sample.get("effective_kernel") == kernel,
                 f"sample used the wrong kernel in {label}: {sample.get('effective_kernel')}")
        _require(sample.get("sha256_u32_le") == hashes[key[1]][key[2]],
                 f"per-cycle full-map hash mismatch in {label}: {key}")
        observed[key] = sample
    expected_keys = {
        (cycle, mask, source)
        for cycle in range(cycles) for mask in MASKS for source in range(SOURCE_COUNT)
    }
    _require(set(observed) == expected_keys, f"incomplete sample grid in {label}")
    order = [sample["mask"] for sample in samples if sample["source"] == 0]
    _require(order == list(MASKS) * cycles,
             f"ADF 8→20 order changed in {label}")

    by_mask = {}
    combined = []
    for mask in MASKS:
        timings = []
        for cycle in range(cycles):
            values = [observed[(cycle, mask, source)]["all_seven_wall_ms"]
                      for source in range(SOURCE_COUNT)]
            _require(all(value == values[0] for value in values),
                     f"all-seven wall time differs per source in {label}/{mask}/{cycle}")
            timings.append(values[0])
        by_mask[mask] = {
            "samples_ms": timings,
            **HELPERS._timing_summary(timings),
        }
        combined.extend(timings)

    cycle_hashes = {key: sample["sha256_u32_le"] for key, sample in observed.items()}
    return {
        "label": label,
        "kernel": kernel,
        "configuration": expected_config,
        "requested_configuration": expected_config,
        "fullmap_parity": True,
        "sha256_u32_le": hashes,
        "a1_sha256_u32_le": frozen,
        "series_resident_bytes": response["series_resident_bytes"],
        "metal_current_allocated_bytes": response["metal_current_allocated_bytes"],
        "source_identity_sha256": ready["source_identity_sha256"],
        "samples": [dict(observed[key]) for key in sorted(observed)],
        "all_seven_wall_ms": {
            "measured_cycles_per_mask": cycles,
            "samples_by_mask": by_mask,
            "combined_samples_ms": combined,
            **HELPERS._timing_summary(combined),
        },
    }, cycle_hashes


def _record_outputs(root: Path, out: Path, manifest: dict) -> None:
    descriptions = {
        "raw": "All benchmark JSON-line records, including ready, warmup, A/B/A responses, and release.",
        "ready": "Seven-source identities, compact-offset evidence, shape, and ready allocation.",
        "summary": "Exact arm samples, parity checks, allocation checks, and all-seven timing summaries.",
        "stderr": "Benchmark process diagnostics.",
        "failure": "Structured failure and release-cleanup evidence, when a run fails.",
    }
    outputs = []
    for name, description in descriptions.items():
        filename = "stderr.log" if name == "stderr" else (
            "raw.jsonl" if name == "raw" else f"{name}.json")
        path = out / filename
        if path.is_file():
            outputs.append({
                "artifact_id": name,
                "path": path.relative_to(root).as_posix(),
                "sha256": HELPERS._sha256(path),
                "size_bytes": path.stat().st_size,
                "retention": "durable",
                "consuming_figures": [],
                "result": description,
            })
    manifest["outputs"] = outputs


def _update_registry(root: Path, out: Path, manifest: dict) -> None:
    path = root / "experiments/RUNS.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [i for i, line in enumerate(lines)
               if line.startswith(f"| {EXPERIMENT_ID} |")]
    _require(len(matches) == 1, f"expected one registry row for {EXPERIMENT_ID}")
    index = matches[0]
    fields = [field.strip() for field in lines[index].strip("|").split("|")]
    if manifest["status"] == "completed":
        fields[3] = "ok"
        values = manifest["parameters"]["latency_summary"][MASKS[-1]]
        fields[4] = "ADF 8→20 p50/p95 ms A1/B/A2 " + ", ".join(
            f"{label} {values[label]['p50_ms']:.2f}/{values[label]['p95_ms_nearest_rank']:.2f}"
            for label in ("A1", "B", "A2")
        ) + "; exact maps, allocation budget, and release gates recorded"
    else:
        fields[3] = "failed"
        failure = manifest["execution"].get("failure") or {}
        message = " ".join(
            f"{failure.get('type', 'Failure')}: {failure.get('message', 'unknown')}"
            .replace("|", "/").split())
        fields[4] = f"Harness failed: {message[:180]}"
    fields[5] = (
        f"[manifest]({EXPERIMENT_ID}/manifest.json); "
        f"[raw]({out.joinpath('raw.jsonl').relative_to(root).as_posix()})"
    )
    lines[index] = "| " + " | ".join(fields) + " |"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.exe = args.exe.expanduser().resolve()
    args.folder = args.folder.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(args.exe.is_file(), f"benchmark executable not found: {args.exe}")
    _require(args.folder.is_dir(), f"source folder not found: {args.folder}")
    _require(not args.out.exists(), f"output path already exists: {args.out}")
    _require(args.cache != args.out, "cache and result paths must differ")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True)
    HELPERS._fingerprint_code(root, args.exe, manifest, runner=Path(__file__).resolve())
    manifest["status"] = "running"
    manifest["timestamps"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished": None,
    }
    manifest["execution"]["failure"] = None
    manifest["execution"]["release"] = None
    manifest["parameters"].update({
        "warmup_samples": [],
        "arm_samples": [],
        "allocation_bytes_by_arm": {},
    })
    _write_json(manifest_path, manifest)

    ready = release = failure = None
    process = None
    arms = []
    expected_hashes = None
    expected_cycle_hashes = None
    allocation_after_candidate_warmup = None
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                [str(args.exe), str(args.folder), str(args.cache)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, bufsize=1, env=_environment(),
            )
            ready = HELPERS._receive(process, raw, "ans_resident_loop_ready")
            identities = _validate_ready(ready)
            ready_allocation = ready["metal_current_allocated_bytes"]
            manifest["parameters"].update({
                "source_identity_sha256": identities,
                "resident_bytes_by_source": ready["resident_bytes_by_source"],
                "series_resident_bytes": ready["series_resident_bytes"],
                "ready_metal_current_allocated_bytes": ready_allocation,
                "compact_offset_bytes": ready["compact_offset_bytes"],
            })
            _write_json(args.out / "ready.json", ready)

            for label, requested_arm, kernel in ARMS:
                command_base = {
                    "op": "run",
                    "arm": requested_arm,
                    "mode": "indexed",
                    "kernel": kernel,
                    "partial_groups": 32,
                    "polar_query_variant": "scan512",
                    "trusted_table": False,
                    "batch": False,
                    "bounded_concurrency": SOURCE_COUNT,
                    "profile": False,
                    "mask_names": list(MASKS),
                }
                warmup_command = {
                    **command_base,
                    "cycles": WARMUP_CYCLES,
                }
                warmup_response = HELPERS._request(process, raw, warmup_command)
                warmup, _ = _validate_response(
                    warmup_response, label, kernel, ready, expected_hashes,
                    WARMUP_CYCLES)
                if expected_hashes is None:
                    expected_hashes = warmup["a1_sha256_u32_le"]
                warmup_allocation = warmup_response.get("metal_current_allocated_bytes")
                _require(type(warmup_allocation) is int,
                         f"Metal warmup allocation missing in {label}")
                if label == "A1":
                    _require(warmup_allocation == ready_allocation,
                             "A1 warmup allocation differs from compact-offset ready baseline")
                elif label == "B":
                    allocation_after_candidate_warmup = warmup_allocation
                    _require(warmup_allocation <= MAX_METAL_BYTES,
                             "candidate warmup exceeds absolute Metal allocation cap")
                    _require(warmup_allocation - ready_allocation <= MAX_CANDIDATE_GROWTH_BYTES,
                             "candidate scratch exceeds 122 MiB over compact-offset baseline")
                else:
                    _require(warmup_allocation == allocation_after_candidate_warmup,
                             "allocation changed after candidate warmup")
                manifest["parameters"]["warmup_samples"].append({
                    "label": label,
                    "kernel": kernel,
                    "samples": warmup["samples"],
                    "metal_current_allocated_bytes": warmup_allocation,
                    "fullmap_parity": True,
                })
                _write_json(manifest_path, manifest)

                command = {**command_base, "cycles": CYCLES}
                response = HELPERS._request(process, raw, command)
                record, cycle_hashes = _validate_response(
                    response, label, kernel, ready, expected_hashes, CYCLES)
                allocation = response["metal_current_allocated_bytes"]
                if label == "A1":
                    _require(allocation == ready_allocation,
                             "A1 allocation differs from compact-offset ready baseline")
                else:
                    _require(allocation == allocation_after_candidate_warmup,
                             f"Metal allocation was not stable after candidate warmup at {label}")
                    _require(allocation <= MAX_METAL_BYTES,
                             f"Metal allocation exceeds cap at {label}")
                    _require(allocation - ready_allocation <= MAX_CANDIDATE_GROWTH_BYTES,
                             f"Metal allocation growth exceeds 122 MiB at {label}")
                if expected_cycle_hashes is None:
                    expected_cycle_hashes = cycle_hashes
                _require(cycle_hashes == expected_cycle_hashes,
                         f"per-cycle full-map hashes differ across A/B/A at {label}")
                arms.append(record)
                manifest["parameters"]["arm_samples"] = arms
                manifest["parameters"]["allocation_bytes_by_arm"][label] = {
                    "metal_current_allocated_bytes": allocation,
                    "delta_from_compact_ready_bytes": allocation - ready_allocation,
                }
                manifest["parameters"]["latency_summary"] = {
                    mask: {
                        arm["label"]: arm["all_seven_wall_ms"]["samples_by_mask"][mask]
                        for arm in arms
                    }
                    for mask in MASKS
                }
                _write_json(args.out / "summary.json", {
                    "arm_samples": arms,
                    "allocation_after_candidate_warmup": allocation_after_candidate_warmup,
                    "ready_metal_current_allocated_bytes": ready_allocation,
                })
                _write_json(manifest_path, manifest)
                print(json.dumps({
                    "arm": label,
                    "kernel": kernel,
                    "parity": True,
                    "target_mask": MASKS[-1],
                    "target_p50_ms": record["all_seven_wall_ms"][
                        "samples_by_mask"][MASKS[-1]]["p50_ms"],
                    "target_p95_ms": record["all_seven_wall_ms"][
                        "samples_by_mask"][MASKS[-1]]["p95_ms_nearest_rank"],
                    "metal_current_allocated_bytes": allocation,
                }, sort_keys=True), flush=True)

            assert process.stdin is not None
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            release = HELPERS._receive(process, raw, "ans_resident_loop_end")
            _require(release.get("resident_count") == SOURCE_COUNT
                     and release.get("all_released") is True,
                     "benchmark did not release all seven residents")
            _require(process.wait(timeout=30) == 0,
                     "benchmark exited unsuccessfully after resident release")
            _require(len(arms) == len(ARMS), "not all A/B/A arms completed")
            manifest["status"] = "completed"
            manifest["parameters"].update({
                "per_cycle_hashes_match_across_arms": True,
                "resident_bytes_unchanged": True,
                "source_identities_unchanged": True,
                "allocation_stable_after_candidate_warmup": True,
                "allocation_within_both_caps": True,
                "all_residents_released": True,
            })
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = "failed"
            manifest["execution"]["failure"] = failure
        finally:
            if process is not None and process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write('{"op":"quit"}\n')
                    process.stdin.flush()
                    release = HELPERS._receive(process, raw, "ans_resident_loop_end")
                    process.wait(timeout=30)
                except Exception as cleanup_error:
                    if failure is None:
                        failure = {
                            "type": type(cleanup_error).__name__,
                            "message": f"release cleanup failed: {cleanup_error}",
                        }
                        manifest["status"] = "failed"
                        manifest["execution"]["failure"] = failure
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
    _write_json(args.out / "summary.json", {
        "arm_samples": arms,
        "allocation_after_candidate_warmup": allocation_after_candidate_warmup,
        "ready_metal_current_allocated_bytes": manifest["parameters"].get(
            "ready_metal_current_allocated_bytes"),
    })
    manifest["timestamps"]["finished"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _record_outputs(root, args.out, manifest)
    _write_json(manifest_path, manifest)
    _update_registry(root, args.out, manifest)
    if failure is not None:
        raise RuntimeError(f"experiment failed; retained evidence at {args.out}: {failure}")
    print(f"completed; outputs and manifest retained under {args.out}", flush=True)


if __name__ == "__main__":
    main()
