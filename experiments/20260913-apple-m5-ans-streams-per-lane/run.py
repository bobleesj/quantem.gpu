"""Exact indexed 2-to-4-to-2 stream-count A/B/A benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


SOURCE_COUNT = 7
CYCLES = 20
WARMUP_CYCLES = 1
MASKS = ("adf-center-8", "adf-center-20")
BASE_RUNNER_PATH = Path(__file__).resolve().parents[1] / (
    "20260913-apple-m5-ans-polar-scan512/run.py"
)
SPEC = importlib.util.spec_from_file_location("scan512_runner", BASE_RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def _command(streams: int, cycles: int, arm: str) -> dict:
    return {
        "op": "run", "arm": arm, "mode": "indexed",
        "kernel": "packet-owner2", "batch": False,
        "bounded_concurrency": SOURCE_COUNT, "profile": False,
        "cycles": cycles, "mask_names": list(MASKS),
        "polar_query_variant": "packet-groups",
        "streams_per_lane": streams,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate-streams", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    args.exe = args.exe.expanduser().resolve()
    args.folder = args.folder.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    args.out = args.out.resolve()
    arm_sequence = (("A1", "A1", 2), ("B", "candidate", args.candidate_streams),
                    ("A2", "A2", 2))
    root = Path(__file__).resolve().parents[2]
    manifest_path = args.out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    BASE._require(args.exe.is_file(), f"benchmark executable not found: {args.exe}")
    BASE._require(args.folder.is_dir(), f"source folder not found: {args.folder}")
    BASE._require(not args.out.exists(), f"output path already exists: {args.out}")
    BASE._require(args.cache != args.out, "cache and result paths must be different")
    args.cache.mkdir(parents=True, exist_ok=True)

    BASE._fingerprint_code(root, args.exe, manifest, runner=Path(__file__).resolve())
    manifest["status"] = "running"
    manifest["timestamps"] = {"started": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "finished": None}
    manifest["execution"]["failure"] = None
    manifest["execution"]["release"] = None
    parameters = manifest["parameters"]
    parameters.update({
        "arm_samples": [], "warmup_samples": [], "allocation_bytes_by_arm": {},
        "latency_summary": None, "all_residents_released": False,
        "arm_sequence": [
            {"arm": label, "streams_per_lane": streams}
            for label, _, streams in arm_sequence
        ],
    })
    BASE._write_json(manifest_path, manifest)

    process = None
    ready = None
    release = None
    failure = None
    arm_records = []
    expected_cycle_hashes = None
    expected_map_hashes = None
    args.out.mkdir(parents=True)
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                [str(args.exe), str(args.folder), str(args.cache)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, bufsize=1, env=BASE._environment(),
            )
            ready = BASE._receive(process, raw, "ans_resident_loop_ready")
            identities = BASE._validate_ready(ready)
            parameters["source_identity_sha256"] = identities
            parameters["resident_bytes_by_source"] = ready["resident_bytes_by_source"]
            parameters["series_resident_bytes"] = ready["series_resident_bytes"]
            parameters["ready_metal_current_allocated_bytes"] = ready[
                "metal_current_allocated_bytes"]
            BASE._write_json(args.out / "ready.json", ready)

            for label, request_arm, streams in arm_sequence:
                expected_arm = "candidate" if label == "B" else label
                warmup = BASE._request(
                    process, raw, _command(streams, WARMUP_CYCLES, request_arm))
                warm_record, _, warm_hashes = BASE._validate_response(
                    warmup, expected_arm, "packet-groups", ready,
                    expected_map_hashes, cycles=WARMUP_CYCLES,
                    streams_per_lane=streams)
                if expected_map_hashes is None:
                    expected_map_hashes = warm_hashes
                parameters["warmup_samples"].append({
                    "label": label, "streams_per_lane": streams,
                    "samples": warm_record["samples"], "fullmap_parity": True,
                })

                response = BASE._request(
                    process, raw, _command(streams, CYCLES, request_arm))
                record, cycle_hashes, map_hashes = BASE._validate_response(
                    response, expected_arm, "packet-groups", ready,
                    expected_map_hashes, cycles=CYCLES,
                    streams_per_lane=streams)
                if expected_cycle_hashes is None:
                    expected_cycle_hashes = cycle_hashes
                BASE._require(cycle_hashes == expected_cycle_hashes,
                              f"per-cycle hashes differ across A/B/A at {label}")
                BASE._require(map_hashes == expected_map_hashes,
                              f"full-map hashes differ across A/B/A at {label}")
                record["label"] = label
                record["streams_per_lane"] = streams
                arm_records.append(record)
                parameters["arm_samples"] = arm_records
                parameters["latency_summary"] = {
                    mask: {
                        arm["label"]: arm["all_seven_wall_ms"][
                            "samples_by_mask"][mask]
                        for arm in arm_records
                    }
                    for mask in MASKS
                }
                parameters["allocation_bytes_by_arm"] = {
                    arm["label"]: {
                        "series_resident_bytes": arm["series_resident_bytes"],
                        "metal_current_allocated_bytes": arm[
                            "metal_current_allocated_bytes"],
                    }
                    for arm in arm_records
                }
                BASE._write_json(args.out / "summary.json", {"arm_samples": arm_records})
                BASE._write_json(manifest_path, manifest)
                timing = record["all_seven_wall_ms"]["samples_by_mask"][
                    "adf-center-20"]
                print(json.dumps({
                    "arm": label, "streams_per_lane": streams,
                    "center20_p50_ms": timing["p50_ms"],
                    "center20_p95_ms": timing["p95_ms_nearest_rank"],
                    "parity": record["fullmap_parity"],
                    "series_resident_bytes": record["series_resident_bytes"],
                    "metal_current_allocated_bytes": record[
                        "metal_current_allocated_bytes"],
                }, sort_keys=True), flush=True)

            assert process.stdin is not None
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            release = BASE._receive(process, raw, "ans_resident_loop_end")
            BASE._require(release.get("resident_count") == SOURCE_COUNT
                          and release.get("all_released") is True,
                          "benchmark did not release all seven residents")
            BASE._require(process.wait(timeout=30) == 0,
                          "benchmark exited unsuccessfully after release")
            BASE._require(len(arm_records) == 3, "not all A/B/A arms completed")
            manifest["status"] = "completed"
            parameters["per_cycle_hashes_match_across_arms"] = True
            parameters["resident_bytes_unchanged"] = True
            parameters["source_identities_unchanged"] = True
            parameters["all_residents_released"] = True
            parameters["failure_status"] = "completed"
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = "failed"
            manifest["execution"]["failure"] = failure
            parameters["failure_status"] = "failed"
        finally:
            if process is not None and process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write('{"op":"quit"}\n')
                    process.stdin.flush()
                    release = BASE._receive(process, raw, "ans_resident_loop_end")
                    process.wait(timeout=30)
                except Exception as cleanup_error:
                    if failure is None:
                        failure = {
                            "type": type(cleanup_error).__name__,
                            "message": f"release cleanup failed: {cleanup_error}",
                        }
                        manifest["status"] = "failed"
                        manifest["execution"]["failure"] = failure
                        parameters["failure_status"] = "failed"
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()

    manifest["execution"]["release"] = release
    if release is not None:
        parameters["all_residents_released"] = release.get("all_released") is True
    if failure is not None:
        BASE._write_json(args.out / "failure.json", {
            "status": "failed", "failure": failure, "release": release,
        })
    if ready is not None:
        BASE._write_json(args.out / "ready.json", ready)
    BASE._write_json(args.out / "summary.json", {"arm_samples": arm_records})
    manifest["timestamps"]["finished"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    BASE._record_outputs(root, args.out, manifest)
    BASE._write_json(manifest_path, manifest)
    BASE._update_registry(root, args.out, manifest)
    if failure is not None:
        raise RuntimeError(f"experiment failed; manifest records details: {failure}")
    print(f"completed; outputs and manifest retained under {args.out}", flush=True)


if __name__ == "__main__":
    main()
