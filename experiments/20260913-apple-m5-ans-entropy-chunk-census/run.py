"""Run the exact seven-source entropy-chunk diagnostic and retain its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "20260913-apple-m5-ans-entropy-chunk-census"
EXPERIMENT = ROOT / "experiments" / EXPERIMENT_ID
SOURCE_FILES = (
    "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/MetalPairedRuntimeTANSKernels.swift",
    "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/paired_runtime_tans.metal",
    "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/MetalPairedRuntimeTANSResidentSource.swift",
    "src/quantem/gpu/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark/main.swift",
    "experiments/20260913-apple-m5-ans-entropy-chunk-census/run.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, artifact_id: str, result: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "retention": "durable",
        "consuming_figures": [],
        "result": result,
    }


def read_event(process: subprocess.Popen[str], raw: Any, event: str) -> dict[str, Any]:
    assert process.stdout is not None
    for line in process.stdout:
        raw.write(line)
        raw.flush()
        record = json.loads(line)
        if record.get("event") == "ans_resident_loop_error":
            raise RuntimeError(f"resident benchmark error: {record}")
        if record.get("event") == event:
            return record
    raise RuntimeError(
        f"resident benchmark exited before {event}; status={process.poll()}"
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def record_failure(error: BaseException) -> None:
    """Persist a failed attempt so a partial run cannot look merely planned."""
    manifest_path = EXPERIMENT / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if manifest.get("status") == "completed":
        return
    message = str(error).replace(str(ROOT), "<repo>").replace(
        str(Path.home()), "<home>"
    )
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["status"] = "failed"
    manifest.setdefault("timestamps", {})["finished"] = finished
    manifest["failure"] = {
        "recorded_utc": finished,
        "error_type": type(error).__name__,
        "message": message,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_experiment() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.exe = args.exe.expanduser().resolve()
    args.folder = args.folder.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    require(args.exe.is_file(), f"benchmark executable not found: {args.exe}")
    require(args.folder.is_dir(), f"source folder not found: {args.folder}")
    require(not args.out.exists(), f"result directory already exists: {args.out}")
    require(args.cache != args.out, "cache and result directories must differ")
    require(not args.cache.exists(), f"fresh cache path already exists: {args.cache}")
    args.out.mkdir(parents=True)
    args.cache.mkdir(parents=True)

    manifest_path = EXPERIMENT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "running", "registry must be running before GPU launch")
    tracked_diff = subprocess.run(
        ["git", "diff", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    source_hashes = {relative: sha256(ROOT / relative) for relative in SOURCE_FILES}
    code_fingerprint = hashlib.sha256(
        tracked_diff + json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest["code"].update(
        {
            "revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "dirty": True,
            "diff_sha256": code_fingerprint,
            "working_files_sha256": source_hashes,
        }
    )
    manifest["parameters"]["cache_path_is_metadata_only"] = True
    manifest["parameters"]["fresh_cache_path"] = True
    manifest["tested_binary_sha256"] = sha256(args.exe)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["timestamps"]["started"] = started
    manifest["timestamps"]["finished"] = None
    manifest["status"] = "running"
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
    stderr_path = args.out / "stderr.log"
    result_path = args.out / "result.json"
    process = subprocess.Popen(
        [str(args.exe), str(args.folder), str(args.cache)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_path.open("x", encoding="utf-8"),
        text=True,
        bufsize=1,
        env=environment,
    )
    ready: dict[str, Any] | None = None
    census: dict[str, Any] | None = None
    ended: dict[str, Any] | None = None
    run_error: BaseException | None = None
    with raw_path.open("x", encoding="utf-8") as raw:
        try:
            ready = read_event(process, raw, "ans_resident_loop_ready")
            require(ready.get("resident_count") == 7, "expected seven residents")
            require(ready.get("shape") == [512, 512, 192, 192], "wrong resident shape")
            require(ready.get("logical_dtype") == "uint16", "expected full uint16 input")
            identities = ready.get("source_identity_sha256", [])
            resident_bytes = ready.get("resident_bytes_by_source", [])
            require(len(identities) == 7 and len(set(identities)) == 7,
                    "source identities are not seven-way distinct")
            require(len(resident_bytes) == 7, "missing per-source resident byte counts")

            assert process.stdin is not None
            started_census = time.perf_counter()
            process.stdin.write(json.dumps({"op": "entropy_chunk_census"}) + "\n")
            process.stdin.flush()
            census = read_event(process, raw, "ans_resident_loop_entropy_chunk_census")
            host_wall_ms = (time.perf_counter() - started_census) * 1000
            require(census.get("source_count") == 7, "census did not cover seven sources")
            require(
                census.get("decoder_schedule") == {
                    "kernel": "packet-owner2",
                    "streams_per_lane": 2,
                    "packet_splits": 1,
                    "joint_plan": False,
                },
                "census schedule differs from the baseline packet-owner2 plan",
            )
            require(census.get("source_identities_unique_and_unchanged") is True,
                    "source identity changed during census")
            require(census.get("resident_bytes_unchanged") is True,
                    "persistent resident bytes changed during census")
            require(census.get("eligible_full_chunk_denominator_all_sources") == 57_344,
                    "wrong full-chunk denominator")
            require(census.get("eligible_full_simd_chunk_denominator_all_sources") == 118_272,
                    "wrong SIMD-full-chunk denominator")
            require(census.get("full_chunks_per_packet") == 16,
                    "wrong complete 64-stream chunk count")
            require(census.get("eligible_full_chunk_denominator_per_source") == 8_192,
                    "wrong per-source complete 64-stream denominator")
            require(census.get("trailing_streams_per_packet_not_eligible") == 43,
                    "wrong partial-chunk tail")
            require(census.get("tail_partial_chunk_count_per_source") == 512,
                    "wrong per-source 64-stream tail denominator")
            require(census.get("simd_width") == 32,
                    "wrong SIMD width")
            require(census.get("full_simd_chunks_per_packet") == 33,
                    "wrong complete SIMD32 chunk count")
            require(census.get("eligible_full_simd_chunk_denominator_per_source") == 16_896,
                    "wrong per-source SIMD32 denominator")
            require(census.get("simd_tail_streams_per_packet_not_eligible") == 11,
                    "wrong SIMD32 partial-chunk tail")
            require(census.get("simd_tail_partial_chunk_count_per_source") == 512,
                    "wrong per-source SIMD32 tail denominator")
            samples = census.get("samples", [])
            require(len(samples) == 7, "missing per-source census results")
            for source, row in enumerate(samples):
                require(row.get("source") == source, "source ordering changed")
                require(row.get("source_identity_sha256") == identities[source],
                        f"source {source} identity mismatch")
                require(row.get("rows") == 1067 and row.get("packets") == 512,
                        f"source {source} workload mismatch")
                require(row.get("full_chunks_per_packet") == 16,
                        f"source {source} full-chunk count mismatch")
                require(row.get("eligible_full_chunk_denominator") == 8192,
                        f"source {source} denominator mismatch")
                require(
                    row.get("all_entropy_full_chunks", -1)
                    + row.get("mixed_full_chunks", -1) == 8192,
                    f"source {source} full-chunk counters do not sum",
                )
                require(
                    row.get("all_entropy_tail_chunks_not_eligible", -1)
                    + row.get("mixed_tail_chunks_not_eligible", -1) == 512,
                    f"source {source} tail counters do not sum",
                )
                require(row.get("simd_width") == 32,
                        f"source {source} SIMD width mismatch")
                require(row.get("full_simd_chunks_per_packet") == 33,
                        f"source {source} SIMD-full-chunk count mismatch")
                require(row.get("eligible_full_simd_chunk_denominator") == 16_896,
                        f"source {source} SIMD denominator mismatch")
                require(
                    row.get("all_entropy_full_simd_chunks", -1)
                    + row.get("mixed_full_simd_chunks", -1) == 16_896,
                    f"source {source} SIMD-full counters do not sum",
                )
                require(
                    row.get("all_entropy_simd_tail_chunks_not_eligible", -1)
                    + row.get("mixed_simd_tail_chunks_not_eligible", -1) == 512,
                    f"source {source} SIMD-tail counters do not sum",
                )
                require(row.get("resident_bytes_before") == resident_bytes[source],
                        f"source {source} resident bytes differ from ready")
                require(row.get("resident_bytes_after") == resident_bytes[source],
                        f"source {source} resident bytes changed")
            require(
                census.get("all_entropy_full_chunks_all_sources", -1)
                + census.get("mixed_full_chunks_all_sources", -1) == 57_344,
                "aggregate full-chunk counters do not sum",
            )
            require(
                census.get("all_entropy_tail_chunks_all_sources_not_eligible", -1)
                + census.get("mixed_tail_chunks_all_sources_not_eligible", -1) == 3_584,
                "aggregate partial-tail counters do not sum",
            )
            require(
                census.get("all_entropy_full_simd_chunks_all_sources", -1)
                + census.get("mixed_full_simd_chunks_all_sources", -1) == 118_272,
                "aggregate SIMD-full counters do not sum",
            )
            require(
                census.get("all_entropy_simd_tail_chunks_all_sources_not_eligible", -1)
                + census.get("mixed_simd_tail_chunks_all_sources_not_eligible", -1) == 3_584,
                "aggregate SIMD-tail counters do not sum",
            )
            aggregate_fields = (
                ("all_entropy_full_chunks", "all_entropy_full_chunks_all_sources"),
                ("mixed_full_chunks", "mixed_full_chunks_all_sources"),
                ("all_entropy_tail_chunks_not_eligible",
                 "all_entropy_tail_chunks_all_sources_not_eligible"),
                ("mixed_tail_chunks_not_eligible",
                 "mixed_tail_chunks_all_sources_not_eligible"),
                ("all_entropy_full_simd_chunks", "all_entropy_full_simd_chunks_all_sources"),
                ("mixed_full_simd_chunks", "mixed_full_simd_chunks_all_sources"),
                ("all_entropy_simd_tail_chunks_not_eligible",
                 "all_entropy_simd_tail_chunks_all_sources_not_eligible"),
                ("mixed_simd_tail_chunks_not_eligible",
                 "mixed_simd_tail_chunks_all_sources_not_eligible"),
            )
            for sample_key, aggregate_key in aggregate_fields:
                per_source_sum = sum(int(row[sample_key]) for row in samples)
                require(census.get(aggregate_key) == per_source_sum,
                        f"aggregate {aggregate_key} differs from per-source results")
            require(
                census.get("diagnostic_transient_requested_bytes_per_source") == 4_304,
                "unexpected diagnostic transient allocation size",
            )
            census["tail_partial_chunk_denominator_all_sources"] = 3_584
            census["simd_tail_partial_chunk_denominator_all_sources"] = 3_584
            census["host_wall_ms_including_wait"] = host_wall_ms
        except BaseException as error:
            run_error = error
        finally:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write('{"op":"quit"}\n')
                    process.stdin.flush()
                    ended = read_event(process, raw, "ans_resident_loop_end")
                except BaseException as release_error:
                    if run_error is None:
                        run_error = release_error
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
                if run_error is None:
                    run_error = RuntimeError("resident loop did not exit after release")
    if run_error is not None:
        raise run_error
    assert ready is not None and census is not None and ended is not None
    require(ended.get("all_released") is True, "not all resident storage was released")
    require(ended.get("metal_after_release_bytes", -1)
            <= ended.get("metal_before_release_bytes", -1),
            "Metal allocated size rose after release")

    result = {
        "experiment_id": EXPERIMENT_ID,
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ready": ready,
        "census": census,
        "release": ended,
        "diagnostic_is_not_image_latency": True,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest["timestamps"]["finished"] = result["finished_utc"]
    manifest.pop("failure", None)
    manifest["parameters"].update(
        {
            "eligible_full_chunk_denominator_all_sources": 57_344,
            "all_entropy_full_chunks_all_sources": census[
                "all_entropy_full_chunks_all_sources"
            ],
            "mixed_full_chunks_all_sources": census[
                "mixed_full_chunks_all_sources"
            ],
            "all_entropy_full_chunk_fraction": (
                census["all_entropy_full_chunks_all_sources"] / 57_344
            ),
            "eligible_full_simd_chunk_denominator_all_sources": 118_272,
            "all_entropy_full_simd_chunks_all_sources": census[
                "all_entropy_full_simd_chunks_all_sources"
            ],
            "mixed_full_simd_chunks_all_sources": census[
                "mixed_full_simd_chunks_all_sources"
            ],
            "all_entropy_full_simd_chunk_fraction": (
                census["all_entropy_full_simd_chunks_all_sources"] / 118_272
            ),
            "tail_partial_chunk_denominator_all_sources": 3_584,
            "simd_tail_partial_chunk_denominator_all_sources": 3_584,
            "resident_bytes_by_source": census["resident_bytes_before_by_source"],
            "metal_allocated_bytes_during_census_max_sampled": census[
                "metal_allocated_bytes_during_census_max_sampled"
            ],
            "release_all_sources": ended["all_released"],
        }
    )
    manifest["outputs"] = [
        file_record(result_path, "result-json", "Seven-source exact chunk-mode census and resident release evidence."),
        file_record(raw_path, "raw-jsonl", "Raw resident-loop ready, census, and release events."),
        file_record(stderr_path, "stderr-log", "Benchmark process diagnostics."),
    ]
    manifest["results"] = {
        "eligible_full_chunk_denominator_per_source": 8_192,
        "eligible_full_chunk_denominator_all_sources": 57_344,
        "all_entropy_full_chunks_all_sources": census[
            "all_entropy_full_chunks_all_sources"
        ],
        "mixed_full_chunks_all_sources": census["mixed_full_chunks_all_sources"],
        "all_entropy_full_chunk_fraction": (
            census["all_entropy_full_chunks_all_sources"] / 57_344
        ),
        "tail_partial_chunk_denominator_per_source": 512,
        "tail_partial_chunk_denominator_all_sources": 3_584,
        "all_entropy_tail_chunks_all_sources_not_eligible": census[
            "all_entropy_tail_chunks_all_sources_not_eligible"
        ],
        "mixed_tail_chunks_all_sources_not_eligible": census[
            "mixed_tail_chunks_all_sources_not_eligible"
        ],
        "eligible_full_simd_chunk_denominator_per_source": 16_896,
        "eligible_full_simd_chunk_denominator_all_sources": 118_272,
        "all_entropy_full_simd_chunks_all_sources": census[
            "all_entropy_full_simd_chunks_all_sources"
        ],
        "mixed_full_simd_chunks_all_sources": census[
            "mixed_full_simd_chunks_all_sources"
        ],
        "all_entropy_full_simd_chunk_fraction": (
            census["all_entropy_full_simd_chunks_all_sources"] / 118_272
        ),
        "simd_tail_partial_chunk_denominator_per_source": 512,
        "simd_tail_partial_chunk_denominator_all_sources": 3_584,
        "all_entropy_simd_tail_chunks_all_sources_not_eligible": census[
            "all_entropy_simd_tail_chunks_all_sources_not_eligible"
        ],
        "mixed_simd_tail_chunks_all_sources_not_eligible": census[
            "mixed_simd_tail_chunks_all_sources_not_eligible"
        ],
        "diagnostic_transient_requested_bytes_per_source": 4_304,
        "per_source_counts": [
            {
                key: row[key]
                for key in (
                    "source", "source_identity_sha256", "all_entropy_full_chunks",
                    "mixed_full_chunks", "all_entropy_tail_chunks_not_eligible",
                    "mixed_tail_chunks_not_eligible", "all_entropy_full_simd_chunks",
                    "mixed_full_simd_chunks", "all_entropy_simd_tail_chunks_not_eligible",
                    "mixed_simd_tail_chunks_not_eligible",
                )
            }
            for row in census["samples"]
        ],
    }
    manifest["tested_binary_sha256"] = sha256(args.exe)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": result_path.as_posix(), "census": census}, indent=2))


def main() -> None:
    try:
        run_experiment()
    except BaseException as error:
        record_failure(error)
        raise


if __name__ == "__main__":
    main()
