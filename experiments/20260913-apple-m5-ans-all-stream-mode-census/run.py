"""Count full-resident ANS modes without running detector updates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("mode_census_helpers", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load benchmark helpers from {BASE_RUNNER}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

MODE_MASKS = (
    "bf-base",
    "bf-center-1-delta",
    "adf-base",
    "adf-center-1-delta",
    "adf-center-8-to-20-delta",
    "all-valid-detector-pixels",
)
SOURCE_COUNT = 7
PACKETS_PER_SOURCE = 512
RESIDENT_CAP = 11_877_814_048
METAL_CAP = 11_883_921_408


def environment() -> dict[str, str]:
    result = base._environment()
    result.update({
        "QGPU_ANS_RESIDENT_LOOP": "1",
        "QGPU_ANS_OPT_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
        "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
        "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1",
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT": "packet-groups",
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "1",
        "QGPU_PAIRED_RUNTIME_FULL_MODE_CENSUS": "1",
        "QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_PREPARE_READER32": "0",
    })
    return result


def write_artifacts(root: Path, out: Path, manifest: dict) -> None:
    descriptions = {
        "raw": ("raw.jsonl", "Benchmark protocol records including source identities, mode counts, and release."),
        "ready": ("ready.json", "Seven-source identities, full shape/dtype, and compact resident allocation."),
        "mode_counts": ("mode_counts.json", "Per-source 256-bin exact Metal ANS mode census and checkpoint byte estimates."),
        "stderr": ("stderr.log", "Benchmark diagnostics."),
        "failure": ("failure.json", "Structured failure and release evidence, when a run fails."),
    }
    outputs = []
    for artifact_id, (filename, result) in descriptions.items():
        path = out / filename
        if path.is_file():
            outputs.append({
                "artifact_id": artifact_id,
                "path": path.relative_to(root).as_posix(),
                "sha256": base._sha256(path),
                "size_bytes": path.stat().st_size,
                "retention": "durable",
                "consuming_figures": [],
                "result": result,
            })
    manifest["outputs"] = outputs


def update_registry(root: Path, out: Path, manifest: dict) -> None:
    path = root / "experiments/RUNS.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    key = manifest["experiment_id"]
    positions = [i for i, line in enumerate(lines) if line.startswith(f"| {key} |")]
    base._require(len(positions) == 1, f"expected one registry row for {key}")
    fields = [value.strip() for value in lines[positions[0]].strip("|").split("|")]
    if manifest["status"] == "completed":
        fields[3] = "ok (census only)"
        summary = manifest["parameters"]["entropy_streams"]
        fields[4] = (
            f"{summary['all_sources']} entropy streams across seven sources; "
            f"one aligned midpoint checkpoint estimate {summary['one_checkpoint_u32_bytes']:,} B; "
            "not a speed result or peak-memory proof"
        )
    else:
        fields[3] = "failed"
        failure = manifest["execution"].get("failure") or {}
        fields[4] = "Census failed: " + " ".join(str(failure.get("message", "unknown")).split())[:180]
    fields[5] = (
        f"[manifest]({key}/manifest.json); "
        f"[modes]({out.joinpath('mode_counts.json').relative_to(root).as_posix()})"
    )
    lines[positions[0]] = "| " + " | ".join(fields) + " |"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    manifest_path = args.out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base._require(args.exe.is_file(), f"executable missing: {args.exe}")
    base._require(args.folder.is_dir(), f"dataset folder missing: {args.folder}")
    base._require(not args.out.exists(), f"output already exists: {args.out}")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True)
    base._fingerprint_code(ROOT, args.exe, manifest, runner=Path(__file__).resolve())
    manifest["status"] = "running"
    manifest["timestamps"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished": None,
    }
    manifest["execution"]["failure"] = None
    manifest["execution"]["release"] = None
    base._write_json(manifest_path, manifest)

    ready = None
    release = None
    counts: list[dict] = []
    process = None
    failure = None
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                [str(args.exe), str(args.folder), str(args.cache)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, bufsize=1, env=environment(),
            )
            ready = base._receive(process, raw, "ans_resident_loop_ready")
            base._require(ready.get("resident_count") == SOURCE_COUNT, "expected seven residents")
            base._require(ready.get("shape") == [512, 512, 192, 192], "full shape mismatch")
            base._require(ready.get("logical_dtype") == "uint16", "expected full uint16 sources")
            base._require(ready.get("indexed_mode_available") is True, "polar index not prepared")
            base._require(ready.get("compact_offsets_enabled") == [True] * SOURCE_COUNT,
                          "compact offsets were not active on all sources")
            identities = ready.get("source_identity_sha256", [])
            base._require(len(identities) == SOURCE_COUNT and len(set(identities)) == SOURCE_COUNT,
                          "source identities must be seven distinct SHA-256 values")
            base._require(sum(ready["resident_bytes_by_source"]) == ready["series_resident_bytes"],
                          "resident byte accounting mismatch")
            base._require(ready["series_resident_bytes"] <= RESIDENT_CAP,
                          "resident byte cap exceeded")
            base._require(ready["metal_current_allocated_bytes"] <= METAL_CAP,
                          "Metal allocation cap exceeded")
            for _ in range(len(MODE_MASKS) * SOURCE_COUNT):
                record = base._receive(process, raw, "ans_opt_mode_counts")
                base._require(record.get("mask") in MODE_MASKS, "unexpected mode-census mask")
                base._require(type(record.get("source")) is int
                              and 0 <= record["source"] < SOURCE_COUNT,
                              "invalid source index in mode census")
                base._require(len(record.get("counts_by_mode", [])) == 256,
                              "mode census must contain all 256 codes")
                expected = record["selected_pixels"] * PACKETS_PER_SOURCE
                base._require(sum(record["counts_by_mode"]) == expected,
                              "mode histogram did not account for every selected stream")
                counts.append(record)
            keys = {(item["mask"], item["source"]) for item in counts}
            base._require(len(keys) == len(MODE_MASKS) * SOURCE_COUNT,
                          "duplicate or missing mask/source census")
            if process.stdin is None:
                raise RuntimeError("benchmark stdin is unavailable")
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            release = base._receive(process, raw, "ans_resident_loop_end")
            base._require(release.get("all_released") is True, "resident release failed")
            if process.wait(timeout=30) != 0:
                raise RuntimeError("benchmark exited unsuccessfully after release")

            full = [item for item in counts if item["mask"] == "all-valid-detector-pixels"]
            entropy = sum(sum(item["counts_by_mode"][64:96]) for item in full)
            changed = [item for item in counts if item["mask"] == "adf-center-8-to-20-delta"]
            changed_entropy = sum(sum(item["counts_by_mode"][64:96]) for item in changed)
            by_source = {
                str(item["source"]): {
                    "selected_pixels": item["selected_pixels"],
                    "excluded_pixels": item["excluded_pixels"],
                    "counts_by_mode": item["counts_by_mode"],
                    "entropy_streams": sum(item["counts_by_mode"][64:96]),
                }
                for item in full
            }
            estimates = {
                "all_valid_detector_pixels": {
                    "entropy_streams": entropy,
                    "one_checkpoint_u24_bytes": 3 * entropy,
                    "one_checkpoint_u32_bytes": 4 * entropy,
                    "three_checkpoints_u24_bytes": 9 * entropy,
                    "three_checkpoints_u32_bytes": 12 * entropy,
                },
                "adf_center_8_to_20_delta": {
                    "entropy_streams": changed_entropy,
                    "one_checkpoint_u24_bytes": 3 * changed_entropy,
                    "one_checkpoint_u32_bytes": 4 * changed_entropy,
                    "three_checkpoints_u24_bytes": 9 * changed_entropy,
                    "three_checkpoints_u32_bytes": 12 * changed_entropy,
                },
            }
            summary = {
                "source_identity_sha256": identities,
                "resident_bytes_by_source": ready["resident_bytes_by_source"],
                "series_resident_bytes": ready["series_resident_bytes"],
                "metal_current_allocated_bytes": ready["metal_current_allocated_bytes"],
                "metal_allocation_headroom_bytes": METAL_CAP - ready["metal_current_allocated_bytes"],
                "entropy_streams": {
                    "all_sources": entropy,
                    "one_checkpoint_u24_bytes": 3 * entropy,
                    "one_checkpoint_u32_bytes": 4 * entropy,
                    "three_checkpoints_u24_bytes": 9 * entropy,
                    "three_checkpoints_u32_bytes": 12 * entropy,
                    "adf_center_8_to_20_delta_sources": changed_entropy,
                    "adf_delta_one_checkpoint_u24_bytes": 3 * changed_entropy,
                    "adf_delta_one_checkpoint_u32_bytes": 4 * changed_entropy,
                },
                "by_source_all_valid": by_source,
                "checkpoint_byte_estimates_by_mask": estimates,
                "mode_records": counts,
                "resident_release": release,
                "timing_claim": "none; this is a mode census only",
            }
            base._write_json(args.out / "mode_counts.json", summary)
            manifest["status"] = "completed"
            manifest["execution"]["release"] = release
            manifest["parameters"].update({
                "all_residents_released": True,
                "source_identity_sha256": identities,
                "series_resident_bytes": ready["series_resident_bytes"],
                "metal_current_allocated_bytes": ready["metal_current_allocated_bytes"],
                "metal_allocation_headroom_bytes": METAL_CAP - ready["metal_current_allocated_bytes"],
                "entropy_streams": summary["entropy_streams"],
                "mode_counts": counts,
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
                    release = base._receive(process, raw, "ans_resident_loop_end")
                    process.wait(timeout=30)
                except Exception:
                    process.terminate()
                    process.wait(timeout=10)
            manifest["execution"]["release"] = release
            if ready is not None:
                base._write_json(args.out / "ready.json", ready)
            if failure is not None:
                base._write_json(args.out / "failure.json", {"failure": failure, "release": release})
    manifest["timestamps"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_artifacts(ROOT, args.out, manifest)
    base._write_json(manifest_path, manifest)
    update_registry(ROOT, args.out, manifest)
    if failure is not None:
        raise RuntimeError(f"mode census failed; manifest retained: {failure}")
    print(json.dumps(manifest["parameters"]["entropy_streams"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
