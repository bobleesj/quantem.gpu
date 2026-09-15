"""Profile the exact indexed seven-source detector update API in one process."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


SOURCE_COUNT = 7
CYCLES = 21  # One warmup plus twenty measured cycles per setting.
MASKS = ("adf-center-8", "adf-center-20")
SETTINGS = (("A1", False), ("B", True), ("A2", False))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _read_reference(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    _require(args.exe.is_file(), f"benchmark executable not found: {args.exe}")
    _require(args.folder.is_dir(), f"source folder not found: {args.folder}")
    _require(not args.out.exists(), f"output path already exists: {args.out}")
    args.out.mkdir(parents=True)
    _require(args.cache != args.out, "cache and result paths must be different")
    args.cache.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]
    manifest_path = args.out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diff = subprocess.run(
        ["git", "diff", "HEAD"], check=True, capture_output=True, cwd=root
    ).stdout
    manifest["code"]["revision"] = _git("rev-parse", "HEAD")
    manifest["code"]["dirty"] = bool(diff)
    manifest["code"]["diff_sha256"] = hashlib.sha256(diff).hexdigest()
    manifest["tested_binary_sha256"] = _sha256(args.exe)
    manifest["tested_benchmark_source_sha256"] = _sha256(
        root / "src/quantem/gpu/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark/main.swift"
    )
    manifest["tested_resident_source_sha256"] = _sha256(
        root / "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/MetalPairedRuntimeTANSResidentSource.swift"
    )
    manifest["tested_shader_sha256"] = _sha256(
        root / "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/paired_runtime_tans.metal"
    )
    manifest["parameters"]["cache_path_is_metadata_only"] = True
    manifest["parameters"]["batch_sequence"] = [
        {"arm": name, "batch": batch} for name, batch in SETTINGS
    ]
    manifest["parameters"]["profiled_repeats_per_arm"] = CYCLES - 1
    manifest["parameters"]["unprofiled_repeats_per_arm"] = CYCLES - 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    environment = os.environ | {
        "QGPU_ANS_RESIDENT_LOOP": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS": "1",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "0",
        "QGPU_PAIRED_RUNTIME_JOINT_PLAN": "0",
    }
    for name in list(environment):
        if name.startswith("QGPU_PREPARE_"):
            environment[name] = "0"

    raw_path = args.out / "raw.jsonl"
    trials_path = args.out / "trials.jsonl"
    ready_record = None
    baseline_hashes = None
    summaries = []
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with raw_path.open("x", encoding="utf-8") as raw, \
            trials_path.open("x", encoding="utf-8") as trials, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [str(args.exe), str(args.folder), str(args.cache)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, bufsize=1, env=environment,
        )

        def receive(event: str) -> dict:
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                record = json.loads(line)
                if record.get("event") == "ans_resident_loop_error":
                    raise RuntimeError(record)
                if record.get("event") == event:
                    return record
            raise RuntimeError(
                f"benchmark exited before {event}: {process.poll()}"
            )

        def request(command: dict, event: str) -> dict:
            process.stdin.write(json.dumps(command) + "\n")
            process.stdin.flush()
            return receive(event)

        try:
            ready_record = receive("ans_resident_loop_ready")
            _require(ready_record.get("resident_count") == SOURCE_COUNT,
                     "expected exactly seven fully resident acquisitions")
            identities = ready_record.get("source_identity_sha256", [])
            _require(len(identities) == SOURCE_COUNT and len(set(identities)) == SOURCE_COUNT,
                     "the seven sources must have distinct identity hashes")
            _require(ready_record.get("shape") == [512, 512, 192, 192],
                     "unexpected resident dimensions")
            _require(ready_record.get("logical_dtype") == "uint16",
                     "the exact full-uint16 test is required")

            for phase, profile in (("profiled", True), ("unprofiled", False)):
                phase_hashes = None
                for arm, batch in SETTINGS:
                    command = {
                        "op": "run", "arm": "candidate", "mode": "indexed",
                        "batch": batch, "bounded_concurrency": SOURCE_COUNT,
                        "profile": profile, "cycles": CYCLES,
                        "mask_names": list(MASKS),
                    }
                    response = request(command, "ans_resident_loop_result")
                    _require(response.get("fullmap_parity") is True,
                             f"full-map exact parity failed in {phase}/{arm}")
                    _require(response.get("configuration", {}).get("mode") == "indexed",
                             f"indexed mode was not active in {phase}/{arm}")
                    _require(response.get("configuration", {}).get("batch") is batch,
                             f"batch setting did not take effect in {phase}/{arm}")
                    _require(response.get("series_resident_bytes") ==
                             ready_record.get("series_resident_bytes"),
                             f"resident bytes changed in {phase}/{arm}")
                    _require(response.get("cycles") == CYCLES,
                             f"incorrect cycle count in {phase}/{arm}")

                    observed = {}
                    for sample in response.get("samples", []):
                        key = (sample["cycle"], sample["mask"], sample["source"])
                        _require(key not in observed, f"duplicate sample {phase}/{arm}/{key}")
                        observed[key] = sample
                    expected_keys = {
                        (cycle, mask, source)
                        for cycle in range(CYCLES)
                        for mask in MASKS
                        for source in range(SOURCE_COUNT)
                    }
                    _require(set(observed) == expected_keys,
                             f"incomplete seven-source timing grid in {phase}/{arm}")
                    hashes = {
                        (mask, source): observed[(0, mask, source)]["sha256_u32_le"]
                        for mask in MASKS for source in range(SOURCE_COUNT)
                    }
                    if phase_hashes is None:
                        phase_hashes = hashes
                    _require(hashes == phase_hashes,
                             f"full-map hashes differ across A/B/A in {phase}/{arm}")
                    if baseline_hashes is None:
                        baseline_hashes = response.get("a1_sha256_u32_le")
                    _require(response.get("a1_sha256_u32_le") == baseline_hashes,
                             f"frozen A1 hashes changed in {phase}/{arm}")

                    target = [observed[(cycle, MASKS[-1], source)]
                              for cycle in range(1, CYCLES)
                              for source in range(SOURCE_COUNT)]
                    if profile:
                        _require(all(isinstance(row.get("update_profile"), dict)
                                     for row in target),
                                 f"per-source profile missing in {phase}/{arm}")
                    summary = {
                        "phase": phase, "arm": arm, "batch": batch,
                        "profile_enabled": profile,
                        "measured_cycles": CYCLES - 1,
                        "resident_bytes": response["series_resident_bytes"],
                        "metal_current_allocated_bytes": response[
                            "metal_current_allocated_bytes"],
                        "fullmap_parity": response["fullmap_parity"],
                        "a1_hashes": response["a1_sha256_u32_le"],
                        "sample_hashes": response["sha256_u32_le"],
                        "target_samples": target,
                    }
                    summaries.append(summary)
                    trials.write(json.dumps(summary, separators=(",", ":")) + "\n")
                    trials.flush()
                    print(json.dumps({
                        "phase": phase, "arm": arm, "batch": batch,
                        "resident_bytes": summary["resident_bytes"],
                        "fullmap_parity": True,
                        "target_trials": len(target),
                    }), flush=True)

            end_record = request({"op": "quit"}, "ans_resident_loop_end")
            _require(end_record.get("all_released") is True,
                     "benchmark did not release all seven residents")
            if process.wait(timeout=30) != 0:
                raise RuntimeError("benchmark exited unsuccessfully after release")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    _write_json(args.out / "ready.json", ready_record)
    _write_json(args.out / "summaries.json", summaries)
    manifest["status"] = "completed"
    manifest["timestamps"]["started"] = started_utc
    manifest["timestamps"]["finished"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["outputs"] = []
    for artifact, result in (
        (raw_path, "Raw benchmark JSON-lines responses for all resident updates"),
        (trials_path, "Profiled and unprofiled exact A/B/A transition records"),
        (args.out / "ready.json", "Seven distinct full uint16 residents and allocation"),
        (args.out / "summaries.json", "Validated per-setting timing, parity, and profile samples"),
        (args.out / "stderr.log", "Benchmark process diagnostics"),
    ):
        manifest["outputs"].append({
            "artifact_id": artifact.stem,
            "path": str(artifact.relative_to(root)),
            "sha256": _sha256(artifact),
            "size_bytes": artifact.stat().st_size,
            "retention": "durable",
            "consuming_figures": [],
            "result": result,
        })
    manifest["parameters"]["measured_source_transitions"] = len(summaries) * 20 * SOURCE_COUNT
    manifest["parameters"]["exact_parity"] = "full detector arrays against the frozen A1 maps"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"completed; outputs and manifest retained under {args.out}", flush=True)


if __name__ == "__main__":
    main()
