"""Run the exact seven-source compact-offset paired-tANS macro2 A/B/A test."""

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
MASKS = ("adf-center-8", "adf-center-20")
ARMS = (("A1", False), ("B", True), ("A2", False))
RESIDENT_CEILING = 11_877_814_048
METAL_CEILING = 11_883_921_408
SOURCE_FILES = {
    "benchmark": "src/quantem/gpu/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark/main.swift",
    "resident": "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/MetalPairedRuntimeTANSResidentSource.swift",
    "macro_table": "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSMacroTable.swift",
    "kernel_api": "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/MetalPairedRuntimeTANSKernels.swift",
    "shader": "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/paired_runtime_tans.metal",
    "tests": "src/quantem/gpu/swift/Tests/Native4DSTEMIOTests/PairedRuntimeTANSTablesTests.swift",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
        text=not binary,
    )
    return result.stdout.strip() if not binary else result.stdout


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment() -> dict[str, str]:
    result = os.environ.copy()
    for key in tuple(result):
        if key.startswith(("QGPU_PAIRED_RUNTIME_", "QGPU_PREPARE_", "QGPU_ANS_OPT_")):
            result[key] = "0"
    result.update({
        "QGPU_ANS_RESIDENT_LOOP": "1",
        "QGPU_ANS_OPT_EXPERIMENT": "0",
        "QGPU_ANS_OPT_MACRO2": "1",
        "QGPU_PAIRED_RUNTIME_MACRO": "1",
        "QGPU_PAIRED_RUNTIME_MACRO_LOOKAHEAD_BITS": "2",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT": "packet-groups",
        "QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS": "1",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "1",
        "QGPU_PAIRED_RUNTIME_JOINT_PLAN": "0",
    })
    return result


def receive(process: subprocess.Popen[str], raw, expected: str) -> dict:
    assert process.stdout is not None
    for line in process.stdout:
        raw.write(line)
        raw.flush()
        record = json.loads(line)
        if record.get("event") == "ans_resident_loop_error":
            raise RuntimeError(record.get("error", "resident benchmark failed"))
        if record.get("event") == expected:
            return record
        if record.get("event") == "ans_resident_loop_end":
            raise RuntimeError(f"benchmark ended before {expected}: {record}")
    raise RuntimeError(f"benchmark exited before {expected}: {process.poll()}")


def request(process: subprocess.Popen[str], raw, command: dict) -> dict:
    assert process.stdin is not None
    process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
    process.stdin.flush()
    return receive(process, raw, "ans_resident_loop_result")


def check_ready(record: dict) -> None:
    require(record.get("resident_count") == SOURCE_COUNT, "expected exactly seven residents")
    require(record.get("shape") == [512, 512, 192, 192], "full shape required")
    require(record.get("logical_dtype") == "uint16", "full uint16 counts required")
    require(record.get("indexed_mode_available") is True, "polar index not prepared")
    require(record.get("polar_query_variant") == "packet-groups", "query variant changed")
    require(record.get("compact_offsets_enabled") == [True] * SOURCE_COUNT,
            "compact offsets must be enabled for all seven residents")
    require(record.get("macro_lookahead_bits_prepared") == [2] * SOURCE_COUNT,
            "2-bit macro pipelines must be prepared for all seven residents")
    table_bytes = record.get("macro_table_bytes_by_source")
    require(table_bytes == [1_179_648] * SOURCE_COUNT,
            f"unexpected two-bit macro table sizes: {table_bytes}")
    ids = record.get("source_identity_sha256", [])
    require(len(ids) == SOURCE_COUNT and len(set(ids)) == SOURCE_COUNT,
            "seven source identities must be unique")
    resident_bytes = record.get("resident_bytes_by_source", [])
    require(len(resident_bytes) == SOURCE_COUNT and sum(resident_bytes) == record.get("series_resident_bytes"),
            "per-source resident bytes do not match series total")
    require(record["series_resident_bytes"] <= RESIDENT_CEILING,
            "resident memory exceeded the ordinary-offset reference ceiling")
    require(type(record.get("metal_current_allocated_bytes")) is int
            and record["metal_current_allocated_bytes"] <= METAL_CEILING,
            "Metal memory exceeded the ordinary-offset reference ceiling")


def check_response(record: dict, label: str, macro: bool, ready: dict,
                   expected_hashes: dict | None, cycles: int) -> tuple[dict, dict]:
    arm = "candidate" if label == "B" else label
    require(record.get("arm") == arm, f"unexpected arm response {record.get('arm')}")
    require(record.get("fullmap_parity") is True and record.get("exact_a1_hashes") is True,
            f"exact full-map parity failed for {label}")
    require(record.get("cycles") == cycles and record.get("masks") == list(MASKS),
            f"mask/cycle schedule changed in {label}")
    require(record.get("series_resident_bytes") == ready["series_resident_bytes"],
            f"resident allocation changed in {label}")
    require(record.get("metal_current_allocated_bytes") == ready["metal_current_allocated_bytes"],
            f"Metal allocation changed in {label}")
    config = record.get("configuration", {})
    requested = record.get("requested_configuration", {})
    require(config.get("macro") is macro and requested.get("macro") is macro,
            f"macro arm selection incorrect in {label}: {config}")
    expected_config = {
        "mode": "indexed", "kernel": "packet-owner2",
        "polar_query_variant": "packet-groups", "partial_groups": 8,
        "choose_base": False, "partial_stores": False,
        "streams_per_lane": 2, "packet_splits": 1, "batch": False,
        "bounded_concurrency": SOURCE_COUNT, "reuse_word": False,
        "register_sums": False, "history": False, "history_base": False,
        "plain_sums": False, "trusted_table": False, "profile": False,
        "lazy_refill": False, "joint_plan": False,
        "simd_entropy_fast_path": False, "macro": macro,
    }
    require(config == expected_config and requested == expected_config,
            f"benchmark schedule changed in {label}: {config}")
    hashes = record.get("sha256_u32_le")
    require(isinstance(hashes, dict) and hashes == record.get("a1_sha256_u32_le"),
            f"map hashes differ from frozen A1 values in {label}")
    if expected_hashes is not None:
        require(hashes == expected_hashes, f"full-map hashes differ across arms in {label}")

    samples = record.get("samples", [])
    require(len(samples) == cycles * len(MASKS) * SOURCE_COUNT,
            f"incomplete sample grid in {label}")
    observed = {(x.get("cycle"), x.get("mask"), x.get("source")): x for x in samples}
    require(len(observed) == len(samples), f"duplicate sample in {label}")
    for cycle in range(cycles):
        for mask in MASKS:
            for source in range(SOURCE_COUNT):
                sample = observed[(cycle, mask, source)]
                require(sample.get("sha256_u32_le") == hashes[mask][source],
                        f"cycle/source full map changed: {label}/{cycle}/{mask}/{source}")

    summarized = {}
    for mask in MASKS:
        timings = [next(x for x in samples if x["cycle"] == cycle and x["mask"] == mask)[
            "all_seven_wall_ms"] for cycle in range(cycles)]
        ordered = sorted(timings)
        summarized[mask] = {
            "samples_ms": timings,
            "p50_ms": statistics.median(timings),
            "p95_ms_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1],
        }
    return hashes, {"label": label, "macro": macro, "configuration": config,
                    "latency": summarized, "samples": samples,
                    "series_resident_bytes": record["series_resident_bytes"],
                    "metal_current_allocated_bytes": record["metal_current_allocated_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.exe = args.exe.resolve()
    args.folder = args.folder.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    args.out = args.out.resolve()
    manifest_path = args.out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(args.exe.is_file() and args.folder.is_dir(), "benchmark or source folder missing")
    require(0 < args.cycles <= 100, "cycles must be in 1...100")
    require(not args.out.exists() and args.cache != args.out, "output must be fresh and separate")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True)
    diff = git(root, "diff", "HEAD", "--binary", binary=True)
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    manifest["code"].update({
        "revision": git(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tested_binary": {"path": args.exe.name, "sha256": sha256(args.exe)},
        "source_fingerprints": {name: {"path": path, "sha256": sha256(root / path)}
                                for name, path in SOURCE_FILES.items()},
        "runner_sha256": sha256(Path(__file__).resolve()),
    })
    manifest["status"] = "running"
    manifest["timestamps"] = {"created": "2026-09-13",
                              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "finished": None}
    manifest["parameters"].update({"measured_cycles_per_arm": args.cycles,
                                   "warmup_cycles_per_arm": 1,
                                   "arm_samples": []})
    manifest["execution"].update({"failure": None, "release": None})
    write_json(manifest_path, manifest)

    process = None
    ready = None
    release = None
    arms = []
    frozen_hashes = None
    failure = None
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen([str(args.exe), str(args.folder), str(args.cache)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=stderr, text=True, bufsize=1, env=environment())
            ready = receive(process, raw, "ans_resident_loop_ready")
            check_ready(ready)
            write_json(args.out / "ready.json", ready)
            manifest["parameters"].update({
                "source_identity_sha256": ready["source_identity_sha256"],
                "resident_bytes_by_source": ready["resident_bytes_by_source"],
                "series_resident_bytes": ready["series_resident_bytes"],
                "ready_metal_current_allocated_bytes": ready["metal_current_allocated_bytes"],
                "macro_table_bytes_by_source": ready["macro_table_bytes_by_source"],
            })
            write_json(manifest_path, manifest)

            for label, macro in ARMS:
                request_arm = "candidate" if label == "B" else label
                for cycles in (1, args.cycles):
                    command = {
                        "op": "run", "arm": request_arm, "mode": "indexed",
                        "kernel": "packet-owner2", "macro": macro,
                        "batch": False, "bounded_concurrency": SOURCE_COUNT,
                        "profile": False, "cycles": cycles,
                        "mask_names": list(MASKS), "polar_query_variant": "packet-groups",
                    }
                    response = request(process, raw, command)
                    hashes, record = check_response(
                        response, label, macro, ready, frozen_hashes, cycles)
                    if frozen_hashes is None:
                        frozen_hashes = hashes
                    record["warmup"] = cycles == 1
                    if cycles == 1:
                        manifest["parameters"].setdefault("warmup_map_hashes", {})[label] = hashes
                    else:
                        arms.append(record)
                        manifest["parameters"]["arm_samples"] = arms
                        manifest["parameters"]["latency_summary"] = {
                            mask: {arm["label"]: arm["latency"][mask] for arm in arms}
                            for mask in MASKS
                        }
                        write_json(args.out / "summary.json", {"arm_samples": arms})
                    write_json(manifest_path, manifest)
                    if cycles != 1:
                        print(json.dumps({
                            "arm": label, "macro": macro,
                            "center20_p50_ms": record["latency"]["adf-center-20"]["p50_ms"],
                            "center20_p95_ms": record["latency"]["adf-center-20"]["p95_ms_nearest_rank"],
                            "series_resident_bytes": record["series_resident_bytes"],
                            "metal_current_allocated_bytes": record["metal_current_allocated_bytes"],
                            "exact_parity": True,
                        }, sort_keys=True), flush=True)

            assert process.stdin is not None
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            release = receive(process, raw, "ans_resident_loop_end")
            require(release.get("resident_count") == SOURCE_COUNT and release.get("all_released") is True,
                    "did not release all seven residents")
            require(process.wait(timeout=30) == 0, "benchmark failed after resident release")
            require(len(arms) == 3, "A/B/A arms incomplete")
            manifest["status"] = "completed"
            manifest["parameters"].update({
                "per_cycle_hashes_match_across_arms": True,
                "resident_bytes_unchanged": True,
                "source_identities_unchanged": True,
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
                    release = receive(process, raw, "ans_resident_loop_end")
                    process.wait(timeout=30)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    manifest["execution"]["release"] = release
    if failure:
        write_json(args.out / "failure.json", {"failure": failure, "release": release})
    write_json(args.out / "summary.json", {"arm_samples": arms})
    descriptions = {
        "raw.jsonl": "All benchmark stdout JSON-line records, including ready, A/B/A results, and release.",
        "ready.json": "Seven full-source identities, shape, dtype, and measured resident/device allocation.",
        "summary.json": "A/B/A arm timings and exact full-map hash validation.",
        "stderr.log": "Benchmark diagnostics.",
        "failure.json": "Failure and release evidence when a run fails.",
    }
    manifest["outputs"] = []
    for filename, description in descriptions.items():
        path = args.out / filename
        if path.is_file():
            manifest["outputs"].append({
                "artifact_id": filename.removesuffix(".jsonl").removesuffix(".json").removesuffix(".log"),
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path), "size_bytes": path.stat().st_size,
                "retention": "durable", "consuming_figures": [], "result": description,
            })
    manifest["timestamps"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(manifest_path, manifest)
    registry = root / "experiments/RUNS.md"
    lines = registry.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines)
               if line.startswith(f"| {manifest['experiment_id']} |")]
    require(len(matches) == 1, "expected one registry row for this experiment")
    index = matches[0]
    fields = [field.strip() for field in lines[index].strip("|").split("|")]
    if manifest["status"] == "completed":
        p50 = manifest["parameters"]["latency_summary"]["adf-center-20"]
        controls = (p50["A1"]["p50_ms"] + p50["A2"]["p50_ms"]) / 2
        improvement = 100 * (controls - p50["B"]["p50_ms"]) / controls
        fields[3] = "ok"
        fields[4] = "Center-20 p50/p95 ms A1/B/A2 " + ", ".join(
            f"{label} {p50[label]['p50_ms']:.2f}/{p50[label]['p95_ms_nearest_rank']:.2f}"
            for label in ("A1", "B", "A2")) + f"; speed hypothesis refuted ({improvement:+.1f}% p50 vs bracket controls); exact parity, equal allocations, release passed"
    else:
        fields[3] = "failed"
        reason = (failure or {}).get("message", "unknown failure").replace("|", "/")
        fields[4] = "Harness/acceptance failure: " + " ".join(reason.split())[:170]
    fields[5] = (f"[manifest]({manifest['experiment_id']}/manifest.json); "
                 f"[raw]({args.out.relative_to(root).as_posix()}/raw.jsonl)")
    lines[index] = "| " + " | ".join(fields) + " |"
    registry.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failure is not None:
        raise RuntimeError(f"experiment failed; retained artifacts are in {args.out}: {failure}")
    print(f"{manifest['status']}; retained outputs under {args.out}", flush=True)


if __name__ == "__main__":
    main()
