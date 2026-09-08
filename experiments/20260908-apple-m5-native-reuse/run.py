"""Run same-build native arms serially, retaining first-use and presentation evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-repo", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--leases", action="store_true")
    parser.add_argument("--vector-groups", action="store_true")
    parser.add_argument("--shared-preparation", action="store_true")
    parser.add_argument("--drawable-pool", action="store_true")
    parser.add_argument("--presentation-screen", action="store_true")
    parser.add_argument("--loading-integration", action="store_true")
    parser.add_argument("--zero-tail", action="store_true")
    args = parser.parse_args()
    if not (args.loading_integration or args.zero_tail):
        parser.error("Historical detector/pool arms require their recorded source snapshot; current source supports --loading-integration or --zero-tail")
    args.out.mkdir(parents=True, exist_ok=False)
    exe = args.app_repo / ".build/release/Live4DSTEM"
    resources = sorted(exe.resolve().parent.glob("*.bundle/Resources/*.metal"))
    fingerprints = {str(p.relative_to(exe.resolve().parent)): digest(p) for p in resources}
    binary = digest(exe)
    env = {**os.environ, "DIAG": "1", "COMPACT_UPDATE_HOST_PROFILE": "1",
           "QGPU_ORIGINAL_PROFILE": "1", "COMPACT_ORIGINAL_FUSED_PLANES": "1",
           "COMPACT_ORIGINAL_PLANES": "0", "QGPU_ORIGINAL_PLANE_VECTOR4": "1",
           "QGPU_ORIGINAL_PLANE_VECTOR2": "0", "QGPU_ORIGINAL_PLANE_DEFERRED_VERIFY": "0",
           "QGPU_ORIGINAL_STANDARD_PLANES": "1", "QGPU_ORIGINAL_DIRECT_DPC": "1",
           "QGPU_ORIGINAL_ZERO_SCRATCH": "0", "QGPU_ORIGINAL_DECODE_THREADS": "32",
           "QGPU_ORIGINAL_SORT_BLOCKS": "0", "QGPU_ORIGINAL_COOPERATIVE_BLOCK": "0",
           "QGPU_ORIGINAL_LAZY_PLANES": "0",
           "QGPU_ORIGINAL_ZERO_TAIL": "0",
           "QGPU_LIBRARY_REUSE": "0",
           "COMPACT_SHARED_MASK_PREPARATION": "0", "COMPACT_RAW_VECTOR_THREADS": "128",
           "COMPACT_RAW_QUAD_BATCH": "0", "LIVE4DSTEM_DRAWABLE_AUTORELEASE_POOL": "0",
           "QGPU_ORIGINAL_PACK_THREADS": "32", "COMPACT_RESIDENCY_SETS": "0",
           "COMPACT_RAW_QUAD_SCAN": "1", "COMPACT_RAW_OCTO_SCAN": "0",
           "COMPACT_RAW_QUAD_VECTOR": "0", "COMPACT_RAW_PLANE_ILP": "0",
           "COMPACT_RAW_SCAN_COOPERATIVE": "0", "COMPACT_DETECTOR_WIDTH_BUCKETS": "0"}
    cases = [("quad-a", {}), ("octo", {"COMPACT_RAW_OCTO_SCAN": "1"}),
             ("quad-vector", {"COMPACT_RAW_QUAD_VECTOR": "1"}), ("quad-b", {})]
    if args.leases:
        cases = [("control-a", {}), ("leases", {"COMPACT_RESIDENCY_SETS": "1"}), ("control-b", {})]
    if args.vector_groups:
        assert not args.leases
        env.update(COMPACT_RAW_QUAD_VECTOR="1", COMPACT_RESIDENCY_SETS="1")
        cases = [("vector128-a", {"COMPACT_RAW_VECTOR_THREADS": "128"}),
                 ("vector32", {"COMPACT_RAW_VECTOR_THREADS": "32"}),
                 ("vector64", {"COMPACT_RAW_VECTOR_THREADS": "64"}),
                 ("vector128-b", {"COMPACT_RAW_VECTOR_THREADS": "128"})]
    if args.shared_preparation:
        assert not args.leases and not args.vector_groups
        env.update(COMPACT_RAW_QUAD_VECTOR="1", COMPACT_RESIDENCY_SETS="1")
        cases = [("control-a", {}),
                 ("shared", {"COMPACT_SHARED_MASK_PREPARATION": "1"}),
                 ("control-b", {})]
    if args.drawable_pool:
        assert not args.leases and not args.vector_groups and not args.shared_preparation
        env.update(COMPACT_RAW_QUAD_VECTOR="1", COMPACT_RESIDENCY_SETS="1",
                   COMPACT_SHARED_MASK_PREPARATION="1")
        cases = [("control-a", {}),
                 ("pool", {"LIVE4DSTEM_DRAWABLE_AUTORELEASE_POOL": "1"}),
                 ("control-b", {})]
    if args.presentation_screen:
        assert not any((args.leases, args.vector_groups, args.shared_preparation, args.drawable_pool))
        env.pop("DIAG", None)
        env.update(COMPACT_UPDATE_HOST_PROFILE="0", COMPACT_RAW_QUAD_VECTOR="1",
                   COMPACT_RESIDENCY_SETS="1", COMPACT_SHARED_MASK_PREPARATION="1")
        cases = [("control-a", {}),
                 ("pool", {"LIVE4DSTEM_DRAWABLE_AUTORELEASE_POOL": "1"}),
                 ("batch", {"COMPACT_RAW_QUAD_BATCH": "1"}),
                 ("batch-pool", {"COMPACT_RAW_QUAD_BATCH": "1", "LIVE4DSTEM_DRAWABLE_AUTORELEASE_POOL": "1"}),
                 ("control-b", {})]
    if args.loading_integration:
        assert not any((args.leases, args.vector_groups, args.shared_preparation,
                        args.drawable_pool, args.presentation_screen))
        env.pop("DIAG", None)
        env.update(COMPACT_UPDATE_HOST_PROFILE="0", COMPACT_RAW_QUAD_VECTOR="1",
                   COMPACT_RESIDENCY_SETS="1", COMPACT_SHARED_MASK_PREPARATION="1")
        legacy = {"QGPU_ORIGINAL_DECODE_THREADS": "128", "QGPU_ORIGINAL_PACK_THREADS": "128",
                  "QGPU_ORIGINAL_PLANE_VECTOR4": "0", "QGPU_ORIGINAL_STANDARD_PLANES": "0",
                  "QGPU_ORIGINAL_DIRECT_DPC": "0"}
        cases = [("legacy-a", legacy), ("qualified-loading", {}), ("legacy-b", legacy)]
    if args.zero_tail:
        assert not any((args.leases, args.vector_groups, args.shared_preparation,
                        args.drawable_pool, args.presentation_screen, args.loading_integration))
        env.pop("DIAG", None)
        env.update(COMPACT_UPDATE_HOST_PROFILE="0", COMPACT_RAW_QUAD_VECTOR="1",
                   COMPACT_RESIDENCY_SETS="1", COMPACT_SHARED_MASK_PREPARATION="1")
        cases = [("control-a", {}), ("zero-tail", {"QGPU_ORIGINAL_ZERO_TAIL": "1"}), ("control-b", {})]
    results = []
    for name, flags in cases:
        assert digest(exe) == binary
        assert all(digest(exe.resolve().parent / path) == sha for path, sha in fingerprints.items())
        command = ["python3", str(args.app_repo / "Tests/NativeUI/profile_seven_tilts.py"),
                   "--exe", str(exe), "--folder", str(args.folder), "--out", str(args.out / name),
                   "--average", "--capture", "--repeats", "3"]
        print("starting", name, flush=True)
        with (args.out / (name + ".log")).open("w") as stream:
            result = subprocess.run(command, env={**env, **flags}, stdout=stream, stderr=subprocess.STDOUT)
        summary = json.loads((args.out / name / "summary.json").read_text())
        actual_env = {**env, **flags}
        console = (args.out / name / "console.log").read_text()
        profiles = [json.loads(line.split(" ", 1)[1]) for line in console.splitlines()
                    if line.startswith("ORIGINAL_PACK_PROFILE ")]
        record = {"case": name, "flags": flags, "binary_sha256": binary,
                  "metal_resource_sha256": fingerprints, "returncode": result.returncode,
                  "load_stage_profiles": profiles, "summary": summary}
        results.append(record)
        (args.out / "comparison.json").write_text(json.dumps(results, indent=2))
        assert result.returncode == 0 and summary["exit"] == 0
        assert len(profiles) == 7, "Missing original-HDF5 load-stage records"
        for profile in profiles:
            assert profile["packing_plan_fallbacks"] == 0
            assert profile["scalar_decode_threads"] == int(actual_env["QGPU_ORIGINAL_DECODE_THREADS"]), "Decode setting was not executed"
            assert profile["bitshuffle_packing_threads"] == int(actual_env["QGPU_ORIGINAL_PACK_THREADS"]), "Packing setting was not executed"
            expected_columns = 4 if actual_env["QGPU_ORIGINAL_PLANE_VECTOR4"] == "1" else 1
            assert profile["bitshuffle_pixels_per_thread"] == expected_columns, "Vector loading setting was not executed"
            if args.zero_tail:
                expected_tail = actual_env["QGPU_ORIGINAL_ZERO_TAIL"] == "1" and profile["prepared_dpc_reused"]
                assert (profile.get("zero_tail_slices", 0) > 0) == expected_tail, "Zero-tail setting was not executed"
        assert summary["compare_state"]["comparison_resident_bytes"] == 14752175312
        assert summary["final_state"]["resident_count"] == 7
        assert summary["average_diffraction"]["drawable_presented"]
        assert all(row["tiles_presenting"] == 7 for key, row in summary["analysis"].items()
                   if not key.startswith("scan-drag"))
        assert all(digest(exe.resolve().parent / path) == sha for path, sha in fingerprints.items())
        print("completed", name, summary["first_comparison_submission"], flush=True)


if __name__ == "__main__":
    main()
