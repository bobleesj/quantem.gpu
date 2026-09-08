"""Exact all-resident series A/B/A, including divergent histories and empty/full masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ["exe", "input", "index", "plans", "reference", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve()
    resources = {str(f.relative_to(exe.parent)): sha(f) for f in exe.parent.glob("*.bundle/Resources/*.metal")}
    binary = sha(exe)
    reference = [json.loads(line) for line in a.reference.read_text().splitlines()]
    expected = {(r["source_identity"], r["case"], r["step"]): r["sha256_u32_le"]
                for r in reference if r.get("phase") == "detector"}
    expected_ids = {key[0] for key in expected}
    env = {**os.environ, "COMPACT_ORIGINAL_FUSED_PLANES": "1", "COMPACT_ORIGINAL_PLANES": "0",
           "QGPU_ORIGINAL_PROFILE": "1", "QGPU_ORIGINAL_STAGE_PROFILE": "0",
           "QGPU_ORIGINAL_PLANE_VECTOR4": "1", "QGPU_ORIGINAL_PLANE_VECTOR2": "0",
           "QGPU_ORIGINAL_STANDARD_PLANES": "1", "QGPU_ORIGINAL_DIRECT_DPC": "1",
           "QGPU_ORIGINAL_DECODE_THREADS": "32", "QGPU_ORIGINAL_PACK_THREADS": "32",
           "QGPU_ORIGINAL_ZERO_SCRATCH": "0", "QGPU_ORIGINAL_SORT_BLOCKS": "0",
           "QGPU_ORIGINAL_COOPERATIVE_BLOCK": "0", "QGPU_ORIGINAL_PLANE_DEFERRED_VERIFY": "0",
           "COMPACT_RESIDENCY_SETS": "1", "COMPACT_RAW_QUAD_VECTOR": "1",
           "COMPACT_RAW_VECTOR_THREADS": "128", "COMPACT_RAW_QUAD_SCAN": "1",
           "COMPACT_RAW_OCTO_SCAN": "0", "COMPACT_RAW_PLANE_ILP": "0",
           "COMPACT_RAW_SCAN_COOPERATIVE": "0", "COMPACT_DETECTOR_WIDTH_BUCKETS": "0",
           "COMPACT_UPDATE_HOST_PROFILE": "1", "COMPACT_UPDATE_PHASE_PROFILE": "0"}
    results = []
    for name, enabled in [("control-a", "0"), ("shared", "1"), ("control-b", "0")]:
        assert sha(exe) == binary
        assert all(sha(exe.parent / f) == h for f, h in resources.items())
        stdout, stderr = a.out / (name + ".jsonl"), a.out / (name + ".stderr")
        record = {"case": name, "started_unix": time.time(), "binary_sha256": binary,
                  "metal_resource_sha256": resources, "environment": {**env, "COMPACT_SHARED_MASK_PREPARATION": enabled}}
        # Do not retain arbitrary inherited environment or personal paths.
        record["environment"] = {k: v for k, v in record["environment"].items()
                                 if k.startswith(("COMPACT_", "QGPU_"))}
        with stdout.open("w") as out, stderr.open("w") as err:
            run = subprocess.run([str(exe), str(a.input), str(a.index), "--series", "--repeats", "1",
                "--detector-trials", "4", "--budget-bytes", "17000000000", "--plan-directory", str(a.plans)],
                env={**env, "COMPACT_SHARED_MASK_PREPARATION": enabled}, stdout=out, stderr=err, timeout=600)
        record.update(exit=run.returncode, finished_unix=time.time(), stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), accepted=False)
        try:
            rows = [json.loads(line) for line in stdout.read_text().splitlines()]
            assert run.returncode == 0 and rows[-1]["phase"] == "complete", "Incomplete series run"
            loads = [r for r in rows if r["phase"] == "series_load"]
            assert {r["source_identity"] for r in loads} == expected_ids, "Missing or extra source identity"
            assert loads[-1]["resident_bytes"] == 14752175312, "Resident data allocation changed"
            assert rows[-1]["released_device_allocated_bytes"] < 64 << 20, (
                f"Teardown retained {rows[-1]['released_device_allocated_bytes']} bytes; limit {64 << 20}")
            maps = [r for r in rows if r["phase"] == "series_detector"]
            assert len(maps) == 4 * len(expected), "Missing full maps"
            for trial in range(4):
                observed = {(r["source_identity"], r["case"], r["step"]): r["sha256_u32_le"]
                            for r in maps if r["trial"] == trial}
                assert observed == expected, "Frozen complete image differs"
            assert any(r["phase"] == "series_boundary_parity" and r["empty_full_unchanged_masks"] for r in rows)
            host = []
            for line in stderr.read_text().splitlines():
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                if r.get("phase") == "detector_host_stages" and r["source_count"] == 7: host.append(r)
            assert len(host) == 64, "Missing host/path records"
            shared = sum(r["shared_mask_preparation"] for r in host)
            assert shared == (49 if enabled == "1" else 0), "Wrong shared/fallback route"
            stable = {(r["trial"], r["case"], r["step"]): r for r in maps if r["trial"] in [1, 2]}
            record.update(accepted=True, exact_maps=len(maps), shared_calls=shared,
                          heterogeneous_fallback_calls=15, resident_bytes=loads[-1]["resident_bytes"],
                          total_load_seconds=sum(r["seconds"] for r in loads),
                          call_median_ms=statistics.median(r["call_ms"] for r in stable.values()),
                          gpu_median_ms=statistics.median(r["gpu_ms"] for r in stable.values()),
                          prepare_median_ms=statistics.median(r["stages_ms_from_call_entry"]["prepare_end_ms"] for r in host[15:45]),
                          released_device_allocated_bytes=rows[-1]["released_device_allocated_bytes"])
        except (AssertionError, KeyError, ValueError, IndexError) as error:
            record["failure"] = str(error)
        results.append(record)
        (a.out / "comparison.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
        assert record["accepted"], "Series gate failed; evidence retained"


if __name__ == "__main__":
    main()
