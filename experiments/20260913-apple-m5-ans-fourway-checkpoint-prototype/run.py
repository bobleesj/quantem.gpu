"""Run one exact selected-stream checkpoint parity probe on apple-m5-24gb."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "20260913-apple-m5-ans-fourway-checkpoint-prototype"
DATASET_ID = "tilt-series-seven-native-v1"
METAL_ALLOCATION_CAP = 11_883_921_408
SOURCE_FILE = "src/quantem/gpu/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark/main.swift"
SHADER_FILE = "experiments/20260913-apple-m5-ans-fourway-checkpoint-prototype/fourway_checkpoint.metal"

BASE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_runner_base", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load benchmark helpers from {BASE_RUNNER}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def environment() -> dict[str, str]:
    result = base._environment()
    result.update({
        "QGPU_ANS_RESIDENT_LOOP": "1",
        "QGPU_ANS_OPT_EXPERIMENT": "0",
        "QGPU_ANS_OPT_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT": "packet-groups",
        "QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_PREPARE_READER32": "0",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "1",
        "QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS": "1",
        "QGPU_FOURWAY_CHECKPOINT_SHADER": str(ROOT / SHADER_FILE),
    })
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def receive(process: subprocess.Popen[str], raw, accepted: set[str]) -> dict:
    assert process.stdout is not None
    for line in process.stdout:
        raw.write(line)
        raw.flush()
        record = json.loads(line)
        if record.get("event") == "ans_resident_loop_error":
            raise RuntimeError(record.get("error", "resident benchmark reported an error"))
        if record.get("event") in accepted:
            return record
        if record.get("event") == "ans_resident_loop_end":
            raise RuntimeError(f"benchmark ended before a diagnostic result: {record}")
    raise RuntimeError(f"benchmark exited before a diagnostic result: {process.poll()}")


def update_registry(manifest: dict, out: Path) -> None:
    registry = ROOT / "experiments/RUNS.md"
    lines = registry.read_text(encoding="utf-8").splitlines()
    rows = [i for i, line in enumerate(lines) if line.startswith(f"| {EXPERIMENT_ID} |")]
    base._require(len(rows) == 1, f"expected one registry row for {EXPERIMENT_ID}")
    fields = [part.strip() for part in lines[rows[0]].strip("|").split("|")]
    if manifest["status"] == "ok":
        fields[3] = "ok (single-stream parity only)"
        fields[4] = (
            "One packet-0 entropy stream: all 512 decoded uint16 counts equal original-HDF5 reads; "
            "diagnostic is double-pass and not an ADF speed result"
        )
    else:
        fields[3] = manifest["status"]
        message = (manifest.get("execution", {}).get("failure") or {}).get("message", "unknown")
        fields[4] = "Parity probe did not pass: " + " ".join(str(message).split())[:180]
    fields[5] = (
        f"[manifest]({EXPERIMENT_ID}/manifest.json); "
        f"[raw]({out.joinpath('raw.jsonl').relative_to(ROOT).as_posix()}); "
        f"[result]({out.joinpath('result.json').relative_to(ROOT).as_posix()})"
    )
    lines[rows[0]] = "| " + " | ".join(fields) + " |"
    registry.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    manifest_path = args.out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base._require(manifest.get("experiment_id") == EXPERIMENT_ID, "experiment ID mismatch")
    base._require(args.exe.is_file(), f"benchmark executable missing: {args.exe}")
    base._require(args.folder.is_dir(), f"source folder missing: {args.folder}")
    base._require(not args.out.exists(), f"output path already exists: {args.out}")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True)

    base._fingerprint_code(ROOT, args.exe, manifest, runner=Path(__file__).resolve())
    manifest["code"]["source_fingerprints"]["fourway_shader"] = {
        "path": SHADER_FILE,
        "sha256": base._sha256(ROOT / SHADER_FILE),
    }
    manifest["parameters"].update({
        "source_index": 0,
        "scan_packet": 0,
        "candidate_limit": 8,
        "selection": "first valid ADF center-8-to-center-20 residual detector pixel whose stream mode is entropy 64..95",
        "selected_stream_count": 1,
        "output_count_values": 512,
        "parity_oracle": "independent original indexed HDF5 one-frame decoder; reference timing excluded",
        "resident_bytes_ceiling": 11_877_814_048,
        "metal_current_allocated_bytes_ceiling": METAL_ALLOCATION_CAP,
        "performance_acceptance": "none; decode path includes a separate serial checkpoint-capture pass",
    })
    manifest["execution"].update({
        "backend": "Metal runtime MSL on apple-m5-24gb",
        "processes": 1,
        "timing_boundary": "seven-source full-resident readiness; one selected packet-0 entropy stream; full diagnostic call includes runtime compile, mode inspection, serial checkpoint capture, and four-way decode; original-HDF5 parity reads are excluded",
        "failure": None,
        "release": None,
    })
    manifest["status"] = "running"
    manifest["timestamps"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished": None,
    }
    write_json(manifest_path, manifest)

    process = None
    failure = None
    result = None
    ready = None
    release = None
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                [str(args.exe), str(args.folder), str(args.cache)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, bufsize=1, env=environment(), cwd=ROOT,
            )
            ready = base._receive(process, raw, "ans_resident_loop_ready")
            base._require(ready.get("resident_count") == 7, "expected seven residents")
            base._require(ready.get("shape") == [512, 512, 192, 192], "full shape mismatch")
            base._require(ready.get("logical_dtype") == "uint16", "uint16 counts are required")
            identities = ready.get("source_identity_sha256", [])
            base._require(len(identities) == 7 and len(set(identities)) == 7,
                          "seven distinct source identities are required")
            base._require(ready.get("series_resident_bytes", METAL_ALLOCATION_CAP + 1)
                          <= 11_877_814_048, "resident byte ceiling exceeded")
            base._require(ready.get("metal_current_allocated_bytes", METAL_ALLOCATION_CAP + 1)
                          <= METAL_ALLOCATION_CAP, "Metal allocation ceiling exceeded")
            base._require(ready.get("compact_offsets_enabled") == [True] * 7,
                          "compact offsets must match the measured baseline")

            assert process.stdin is not None
            process.stdin.write(json.dumps({
                "command": "fourway_checkpoint",
                "source": 0,
                "candidate_limit": 8,
            }, separators=(",", ":")) + "\n")
            process.stdin.flush()
            result = receive(
                process, raw,
                {"fourway_checkpoint_parity", "fourway_checkpoint_no_entropy_candidate"},
            )
            base._require(result.get("event") == "fourway_checkpoint_parity",
                          "no supported entropy stream was selected in the candidate window")
            base._require(result.get("exact_512_count_parity") is True,
                          "the four-way UInt16 values differ from original HDF5 counts")
            base._require(result.get("mismatch_count") == 0, "decoded count mismatches found")
            base._require(result.get("selected_stream_in_residual") is True,
                "selected stream pixel is not part of the effective ADF residual")
            manifest["status"] = "ok"
        except Exception as error:  # retain a terminal result and ensure residents release
            failure = {"type": type(error).__name__, "message": str(error)}
            manifest["status"] = "failed"
        finally:
            if process is not None and process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write('{"command":"quit"}\n')
                    process.stdin.flush()
                    release = base._receive(process, raw, "ans_resident_loop_end")
                except Exception as error:
                    failure = failure or {"type": type(error).__name__, "message": str(error)}
                    manifest["status"] = "failed"
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
            elif process is not None:
                manifest["status"] = "failed"
                failure = failure or {
                    "type": "RuntimeError", "message": f"benchmark exited with {process.returncode}"
                }
        if release is not None:
            manifest["execution"]["release"] = release
            if not release.get("all_released"):
                failure = failure or {"type": "RuntimeError", "message": "resident release failed"}
                manifest["status"] = "failed"

    if ready is not None:
        write_json(args.out / "ready.json", ready)
    if result is not None:
        write_json(args.out / "result.json", result)
    if failure is not None:
        write_json(args.out / "failure.json", {"failure": failure, "release": release})
    manifest["execution"]["failure"] = failure
    manifest["timestamps"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["outputs"] = []
    descriptions = {
        "raw.jsonl": "JSON-line protocol with readiness, diagnostic, and explicit release records.",
        "stderr.log": "Swift and Metal runtime diagnostics.",
        "ready.json": "Seven distinct full-resolution uint16 sources and starting allocation snapshot.",
        "result.json": "One-stream exact UInt16 parity, mode, timing and allocation report.",
        "failure.json": "Failure and release evidence, when the gate fails.",
    }
    for filename, description in descriptions.items():
        path = args.out / filename
        if path.is_file():
            manifest["outputs"].append({
                "artifact_id": filename.removesuffix(".jsonl").removesuffix(".json").replace(".", "_"),
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": base._sha256(path),
                "size_bytes": path.stat().st_size,
                "retention": "durable",
                "consuming_figures": [],
                "result": description,
            })
    write_json(manifest_path, manifest)
    update_registry(manifest, args.out)
    if failure is not None:
        raise SystemExit(f"Four-way checkpoint parity probe failed: {failure['message']}")
    print(json.dumps({"ready": ready, "result": result, "release": release}, indent=2))


if __name__ == "__main__":
    main()
