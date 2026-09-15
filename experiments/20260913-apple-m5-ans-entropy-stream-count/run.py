"""Count entropy-coded resident streams with a Metal threadgroup reduction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
from pathlib import Path
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("entropy_census_helpers", HELPERS)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load benchmark helpers from {HELPERS}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

SOURCE_COUNT = 7
PACKETS = 512
MASKS = ("adf-center-8-to-20-delta", "all-valid-detector-pixels")
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
        "QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512": "0",
        "QGPU_PAIRED_RUNTIME_PREPARE_READER32": "0",
    })
    return result


class JSONEventReader:
    """Read child output asynchronously so each protocol wait can time out."""

    def __init__(self, stream, raw) -> None:
        self.stream = stream
        self.raw = raw
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        try:
            for line in self.stream:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def next(self, timeout_s: float) -> dict:
        try:
            line = self.lines.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(f"timed out waiting {timeout_s:.0f}s for benchmark output") from exc
        if line is None:
            raise RuntimeError("benchmark stdout closed before the expected protocol event")
        self.raw.write(line)
        self.raw.flush()
        return json.loads(line)


def receive(reader: JSONEventReader, event: str, timeout_s: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting {timeout_s:.0f}s for {event}")
        record = reader.next(remaining)
        if record.get("event") == "ans_resident_loop_error":
            raise RuntimeError(record.get("error", "resident benchmark error"))
        if record.get("event") == event:
            return record
        if record.get("event") == "ans_resident_loop_end":
            raise RuntimeError(f"benchmark ended before {event}: {record}")


def record_outputs(out: Path, manifest: dict) -> None:
    definitions = {
        "raw": ("raw.jsonl", "Benchmark protocol output and explicit release record."),
        "ready": ("ready.json", "Seven source identities and exact resident allocation."),
        "entropy_counts": ("entropy_counts.json", "Exact entropy stream totals and checkpoint byte estimates."),
        "stderr": ("stderr.log", "Benchmark diagnostic output."),
        "failure": ("failure.json", "Failure and release evidence, when a run fails."),
    }
    entries = []
    for artifact_id, (filename, result) in definitions.items():
        path = out / filename
        if path.is_file():
            entries.append({
                "artifact_id": artifact_id,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": base._sha256(path),
                "size_bytes": path.stat().st_size,
                "retention": "durable",
                "consuming_figures": [],
                "result": result,
            })
    manifest["outputs"] = entries


def update_registry(out: Path, manifest: dict) -> None:
    registry = ROOT / "experiments/RUNS.md"
    lines = registry.read_text(encoding="utf-8").splitlines()
    experiment_id = manifest["experiment_id"]
    matches = [i for i, line in enumerate(lines) if line.startswith(f"| {experiment_id} |")]
    base._require(len(matches) == 1, "expected one registry row for entropy census")
    fields = [value.strip() for value in lines[matches[0]].strip("|").split("|")]
    if manifest["status"] == "completed":
        fields[3] = "ok (census only)"
        summary = manifest["parameters"]["checkpoint_estimates"]["all-valid-detector-pixels"]
        fields[4] = (
            f"{summary['entropy_stream_count']:,} exact entropy streams; "
            f"one aligned checkpoint estimate {summary['one_u32_bytes']:,} B; "
            "no speed or peak-memory claim"
        )
    else:
        fields[3] = "failed"
        failure = manifest["execution"].get("failure") or {}
        fields[4] = "Census failed: " + " ".join(str(failure.get("message", "unknown")).split())[:180]
    result_path = out / "entropy_counts.json"
    if not result_path.is_file():
        result_path = out / "failure.json" if (out / "failure.json").is_file() else out / "raw.jsonl"
    fields[5] = f"[manifest]({experiment_id}/manifest.json); [result]({result_path.relative_to(ROOT).as_posix()})"
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
    manifest_path = args.out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base._require(args.exe.is_file(), f"benchmark executable missing: {args.exe}")
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

    process = None
    reader = None
    ready = None
    release = None
    records = []
    failure = None
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw, \
            (args.out / "stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                [str(args.exe), str(args.folder), str(args.cache)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
                env=environment(),
            )
            assert process.stdout is not None
            reader = JSONEventReader(process.stdout, raw)
            ready = receive(reader, "ans_resident_loop_ready")
            base._require(ready.get("resident_count") == SOURCE_COUNT, "expected seven resident sources")
            base._require(ready.get("shape") == [512, 512, 192, 192], "full acquisition shape required")
            base._require(ready.get("logical_dtype") == "uint16", "full uint16 inputs required")
            base._require(ready.get("indexed_mode_available") is True, "polar index was not prepared")
            base._require(ready.get("compact_offsets_enabled") == [True] * SOURCE_COUNT,
                          "compact offsets must be active for all seven sources")
            identities = ready.get("source_identity_sha256", [])
            base._require(len(identities) == SOURCE_COUNT and len(set(identities)) == SOURCE_COUNT,
                          "seven distinct source identity hashes required")
            resident_bytes = ready.get("series_resident_bytes")
            metal_bytes = ready.get("metal_current_allocated_bytes")
            base._require(type(resident_bytes) is int and resident_bytes <= RESIDENT_CAP,
                          "resident byte ceiling exceeded")
            base._require(type(metal_bytes) is int and metal_bytes <= METAL_CAP,
                          "Metal allocation ceiling exceeded")
            if process.stdin is None:
                raise RuntimeError("benchmark stdin is unavailable")
            process.stdin.write('{"op":"entropy_census"}\n')
            process.stdin.flush()
            for _ in range(len(MASKS) * SOURCE_COUNT):
                item = receive(reader, "ans_opt_entropy_mode_counts")
                base._require(item.get("mask") in MASKS, "unexpected census mask")
                base._require(type(item.get("source")) is int and 0 <= item["source"] < SOURCE_COUNT,
                              "invalid source index")
                base._require(type(item.get("requested_pixels")) is int
                              and 0 <= item["requested_pixels"] <= 192 * 192,
                              "invalid requested-mask size")
                base._require(type(item.get("selected_pixels")) is int
                              and type(item.get("excluded_pixels")) is int
                              and item["selected_pixels"] + item["excluded_pixels"]
                              == item["requested_pixels"],
                              "selected and excluded pixels do not cover the requested mask")
                base._require(type(item.get("requested_mask_sha256")) is str
                              and len(item["requested_mask_sha256"]) == 64,
                              "requested mask hash is missing")
                base._require(type(item.get("effective_mask_sha256")) is str
                              and len(item["effective_mask_sha256"]) == 64,
                              "effective mask hash is missing")
                base._require(item.get("validity_policy")
                              == "requested_mask_and_source_detectorValidityMask",
                              "source-specific validity policy was not applied")
                expected_total = item.get("selected_pixels", -1) * PACKETS
                base._require(item.get("total_stream_count") == expected_total,
                              "total stream count disagrees with selected detector pixels")
                entropy_count = item.get("entropy_stream_count")
                base._require(type(entropy_count) is int and 0 <= entropy_count <= expected_total,
                              "entropy stream count is out of bounds")
                diagnostic_bytes = item.get("diagnostic_metal_allocated_bytes")
                base._require(type(diagnostic_bytes) is int and diagnostic_bytes <= METAL_CAP,
                              "census diagnostic allocation exceeded the Metal ceiling")
                records.append(item)
            base._require(len({(x["mask"], x["source"]) for x in records}) == len(MASKS) * SOURCE_COUNT,
                          "duplicate/missing source mask counts")
            for mask_name in MASKS:
                mask_records = [x for x in records if x["mask"] == mask_name]
                base._require(len({x["requested_pixels"] for x in mask_records}) == 1,
                              f"requested mask size differs across sources for {mask_name}")
                base._require(len({x["requested_mask_sha256"] for x in mask_records}) == 1,
                              f"requested mask identity differs across sources for {mask_name}")
            all_mask = [x for x in records if x["mask"] == "all-valid-detector-pixels"]
            base._require(all(x["requested_pixels"] == 192 * 192 for x in all_mask),
                          "all-valid mask does not include every detector pixel")
            if process.stdin is None:
                raise RuntimeError("benchmark stdin is unavailable")
            process.stdin.write('{"op":"quit"}\n')
            process.stdin.flush()
            assert reader is not None
            release = receive(reader, "ans_resident_loop_end", timeout_s=60.0)
            base._require(release.get("all_released") is True, "seven residents were not released")
            if process.wait(timeout=30) != 0:
                raise RuntimeError("benchmark exited unsuccessfully after resident release")

            census_allocation_bytes = max(
                item["diagnostic_metal_allocated_bytes"] for item in records
            )
            checkpoint_estimates = {}
            for mask in MASKS:
                count = sum(item["entropy_stream_count"] for item in records if item["mask"] == mask)
                total = sum(item["total_stream_count"] for item in records if item["mask"] == mask)
                checkpoint_estimates[mask] = {
                    "total_stream_count": total,
                    "entropy_stream_count": count,
                    "one_u24_bytes": 3 * count,
                    "one_u32_bytes": 4 * count,
                    "three_u24_bytes": 9 * count,
                    "three_u32_bytes": 12 * count,
                }
            for estimate in checkpoint_estimates.values():
                estimate["sampled_metal_headroom_after_one_u32_estimate"] = (
                    (METAL_CAP - census_allocation_bytes) - estimate["one_u32_bytes"]
                )
                estimate["sampled_metal_headroom_after_three_u32_estimate"] = (
                    (METAL_CAP - census_allocation_bytes) - estimate["three_u32_bytes"]
                )
            summary = {
                "source_identity_sha256": identities,
                "series_resident_bytes": resident_bytes,
                "metal_current_allocated_bytes": metal_bytes,
                "resident_headroom_bytes": RESIDENT_CAP - resident_bytes,
                "metal_allocation_headroom_bytes": METAL_CAP - metal_bytes,
                "census_diagnostic_metal_allocation_max_bytes": census_allocation_bytes,
                "checkpoint_estimates": checkpoint_estimates,
                "per_source_mask_counts": records,
                "resident_release": release,
                "timing_claim": "none; this is a count/memory-feasibility diagnostic only",
            }
            base._write_json(args.out / "entropy_counts.json", summary)
            manifest["status"] = "completed"
            manifest["execution"]["release"] = release
            manifest["parameters"].update({
                "all_residents_released": True,
                "source_identity_sha256": identities,
                "series_resident_bytes": resident_bytes,
                "metal_current_allocated_bytes": metal_bytes,
                "census_diagnostic_metal_allocation_max_bytes": census_allocation_bytes,
                "checkpoint_estimates": checkpoint_estimates,
                "mask_counts": records,
            })
        except (Exception, KeyboardInterrupt) as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = "failed"
            manifest["execution"]["failure"] = failure
        finally:
            if process is not None and process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write('{"op":"quit"}\n')
                    process.stdin.flush()
                    if reader is not None:
                        release = receive(reader, "ans_resident_loop_end", timeout_s=60.0)
                    process.wait(timeout=30)
                except Exception:
                    process.terminate()
                    process.wait(timeout=10)
            manifest["execution"]["release"] = release
            manifest["parameters"]["all_residents_released"] = (
                release is not None and release.get("all_released") is True
            )
            if ready is not None:
                base._write_json(args.out / "ready.json", ready)
            if failure is not None:
                base._write_json(args.out / "failure.json", {"failure": failure, "release": release})
    manifest["timestamps"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record_outputs(args.out, manifest)
    base._write_json(manifest_path, manifest)
    update_registry(args.out, manifest)
    if failure is not None:
        raise RuntimeError(f"entropy census failed; manifest retained: {failure}")
    print(json.dumps(manifest["parameters"]["checkpoint_estimates"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
