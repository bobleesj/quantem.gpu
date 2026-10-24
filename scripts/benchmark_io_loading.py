"""Loading-only mode for benchmark_io_scale; raw reports belong outside the repo."""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_file_cache import cache_state, evict_file_cache
from benchmark_qem import _process_io
from benchmark_source import runtime_source
from benchmark_support import _git_state, _sync_backend
from check_ans_io import _device, _fingerprint

_BOUNDARY = "io.load(encoded, native, correction=none) through device synchronization"
_HEADER_FIELDS = ("codec", "version", "shape", "dtype", "interval", "bytes",
                  "chunks", "valid", "sha256")


def _header_identity(header: dict) -> str:
    """Identify the exact encoded body and its interpretation, excluding provenance."""
    blob = json.dumps({key: header[key] for key in _HEADER_FIELDS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_snapshot(paths: list[Path]) -> dict:
    """Measure every input file; unknown residency cannot qualify a cache state."""
    files = [cache_state(path) for path in paths]
    known = all(item.get("available", False) for item in files)
    total = sum(item.get("pages", 0) for item in files)
    resident = sum(item.get("resident_pages", 0) for item in files)
    return {
        "files": files, "available": known,
        "resident_percent": 100 * resident / total if known and total else None,
    }


def _cache_matches(snapshot: dict, mode: str) -> bool:
    """Require measured empty/full cache, allowing only rounding at the boundary."""
    percent = snapshot.get("resident_percent")
    return bool(snapshot.get("available") and percent is not None and (
        percent <= 0.01 if mode == "cold" else percent >= 99.99
    ))


def _summary(runs: list[dict]) -> dict:
    """Keep all trials, including tail latency and incomplete cache controls."""
    values = sorted(run["seconds"] for run in runs)
    return {
        "n": len(values), "median_s": statistics.median(values),
        "min_s": values[0], "max_s": values[-1],
        "p90_s": values[max(0, (9 * len(values) + 9) // 10 - 1)],
        "cache_qualified": all(_cache_matches(run["cache_before"], run["mode"])
                               for run in runs),
        "correctness_passed": all(run["correctness"]["passed"] for run in runs),
    }


def _compare_reports(current: dict, baseline: dict, threshold: float) -> dict:
    """Compare only matched workloads; a baseline is never silently refreshed."""
    fields = ("schema", "backend", "device_identity", "storage_id", "boundary",
              "index_policy", "allocator_policy", "product_definition")
    mismatches = [key for key in fields if current.get(key) != baseline.get(key)]
    if not current.get("passed"):
        mismatches.append("current checks did not pass")
    if not baseline.get("passed"):
        mismatches.append("baseline checks did not pass")
    old = {case["id"]: case for case in baseline["cases"]}
    if set(old) != {case["id"] for case in current["cases"]}:
        mismatches.append("fixture IDs")
    comparisons = []
    for case in current["cases"]:
        previous = old.get(case["id"])
        if previous is None:
            continue
        for key in ("reference_identity", "source_master_sha256", "shape", "dtype",
                    "source_device", "qem_device"):
            if case.get(key) != previous.get(key):
                mismatches.append(f"{case['id']}: {key}")
        for format_name in ("hdf5", "qem"):
            for mode in ("cold", "warm"):
                label = f"{case['id']}/{format_name}/{mode}"
                now = case[format_name][mode + "_summary"]
                before = previous[format_name][mode + "_summary"]
                if not all(item["cache_qualified"] and item["correctness_passed"]
                           and item["n"] >= 3 for item in (now, before)):
                    mismatches.append(f"{label}: cache, correctness or sample count")
                    continue
                ratio = now["median_s"] / before["median_s"]
                comparisons.append({"measurement": label, "ratio": ratio,
                                    "review": ratio > 1 + threshold / 100})
    return {
        "status": "incomparable" if mismatches else (
            "review" if any(row["review"] for row in comparisons) else "within_threshold"
        ),
        "threshold_percent": threshold, "mismatches": mismatches,
        "comparisons": comparisons,
        "meaning": "Review threshold, not a statistical significance test; rerun in a quiet slot.",
    }


def _product_check(data, expected: dict) -> dict:
    """Check a frozen detector product on the GPU without expanding all counts."""
    import numpy as np
    from quantem.gpu import detector

    started = time.perf_counter()
    mask = np.zeros(tuple(expected["shape"][-2:]), dtype=bool)
    mask[::2, ::2] = True
    product = detector.prepare(data).masked_sum(mask)
    if hasattr(product, "get"):
        product = product.get()
    elif hasattr(product, "detach"):
        product = product.detach().cpu().numpy()
    digest = hashlib.sha256(np.asarray(product).tobytes()).hexdigest()
    shape, dtype = list(data.shape), str(data.dtype)
    passed = (digest == expected["indexed_product_sha256"]
              and shape == expected["shape"] and dtype == expected["dtype"])
    return {"passed": passed, "product_sha256": digest, "shape": shape,
            "dtype": dtype, "seconds": time.perf_counter() - started,
            "full_original_count_verification": False}


def _native_trial(path: Path, files: list[Path], expected: dict, mode: str,
                  binary: Path) -> dict:
    """Use native internal load timers; process startup and frame checks are separate."""
    primed_bytes = 0
    if mode == "warm":
        # The native prepared index can leave the unused master uncached.
        # Fill missing file pages outside timing instead of relaxing the gate.
        buffer = bytearray(8 << 20)
        for source in files:
            if not _cache_matches(cache_state(source), "warm"):
                with source.open("rb") as stream:
                    while count := stream.readinto(buffer):
                        primed_bytes += count
    eviction = [evict_file_cache(item) for item in files] if mode == "cold" else []
    before = _cache_snapshot(files)
    command = [str(binary.resolve()), str(path)]
    qem = path.suffix.lower() == ".qem"
    command += (["--qem-load-only"] if qem else [
        os.environ["QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR"], "--load-only",
    ])
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True, check=True)
    process_seconds = time.perf_counter() - started
    output = json.loads(process.stdout)
    measured = output if qem else output["acquisitions"][0]
    elapsed = measured["load_seconds"] if qem else measured["ready_seconds_including_preparation"]
    expected_dtype = expected.get("native_qem_dtype" if qem else "native_hdf5_dtype",
                                  expected["dtype"])
    correctness = {
        "passed": (measured["shape"] == expected["shape"]
                   and measured["source_dtype"] == expected_dtype
                   and measured["sample_frames"] == expected["sample_frames"]
                   and measured["sample_hashes_u32_le"] == expected["sample_hashes_u32_le"]),
        "shape": measured["shape"], "dtype": measured["source_dtype"],
        "expected_file_dtype": expected_dtype,
        "sample_frames": measured["sample_frames"],
        "sample_hashes_u32_le": measured["sample_hashes_u32_le"],
        "seconds": measured["verification_seconds_excluded"],
        "full_original_count_verification": False,
    }
    print(f"  native {path.suffix} {mode}: {elapsed:.4f} s; parity={correctness['passed']}",
          flush=True)
    return {"mode": mode, "seconds": elapsed, "cache_before": before,
            "cache_after": _cache_snapshot(files), "eviction": eviction,
            "cache_priming_bytes_excluded": primed_bytes,
            "process_wall_seconds": process_seconds, "native_report": output,
            "physical_read_bytes": measured.get("physical_read_bytes_including_preparation",
                                                 measured["physical_read_bytes"]),
            "resident_bytes": measured["resident_bytes"], "correctness": correctness}


def _trial(path: Path, files: list[Path], backend: str, expected: dict,
           mode: str, native_binary: Path | None = None) -> dict:
    """Time public loading only; cache inspection and parity stay outside it."""
    from quantem.gpu import io

    if backend == "native-metal":
        return _native_trial(path, files, expected, mode, native_binary)
    eviction = [evict_file_cache(item) for item in files] if mode == "cold" else []
    cache_before = _cache_snapshot(files)
    _sync_backend(backend)
    io_before = _process_io()
    started = time.perf_counter()
    with io.load(path, backend=backend, representation="encoded",
                 hot_pixel_correction="none", verbose=False) as data:
        _sync_backend(backend)
        elapsed = time.perf_counter() - started
        io_after = _process_io()
        timings = dict(data.metadata.get("load_timings", {}))
        resident_bytes = data.resident_bytes
        correctness = _product_check(data, expected)
    result = {
        "mode": mode, "seconds": elapsed, "cache_before": cache_before,
        "cache_after": _cache_snapshot(files), "eviction": eviction,
        "process_io_before": io_before, "process_io_after": io_after,
        "load_timings": timings, "resident_bytes": resident_bytes,
        "correctness": correctness,
    }
    print(f"  {path.suffix} {mode}: {elapsed:.4f} s; parity={correctness['passed']}",
          flush=True)
    return result


def _measure(path: Path, files: list[Path], backend: str, expected: dict,
             repetitions: int, persist, native_binary: Path | None = None) -> dict:
    """Record first separately, then paired cold and warm observations."""
    options = {"native_binary": native_binary} if backend == "native-metal" else {}
    result = {"first": _trial(path, files, backend, expected, "first", **options),
              "cold": [], "warm": []}
    persist(result)
    if not result["first"]["correctness"]["passed"]:
        raise ValueError("Frozen product mismatch on first load; retain evidence and inspect the input/code.")
    for _ in range(repetitions):
        for mode in ("cold", "warm"):
            trial = _trial(path, files, backend, expected, mode, **options)
            result[mode].append(trial)
            persist(result)
            if not trial["correctness"]["passed"]:
                raise ValueError("Frozen product mismatch; do not update the baseline.")
    for mode in ("cold", "warm"):
        result[mode + "_summary"] = _summary(result[mode])
    return result


def _run_loading_check(args) -> int:
    """Run the loading mode with disposable QEM/index files and immutable references."""
    from quantem.gpu.io.convert import convert, detector_files
    from quantem.gpu.formats.qem.snapshot import read_header

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "loading.json"
    if destination.exists():
        raise ValueError("loading.json already exists; choose a new --output-dir to preserve evidence.")
    manifest = json.loads(args.selection_manifest.read_text())
    items = manifest["selected_acquisitions"]
    baseline_bytes = args.baseline.read_bytes() if args.baseline else None
    if not items or len({item["id"] for item in items}) != len(items):
        raise ValueError("Select one or more distinct acquisition IDs in the reference manifest.")
    native = args.backend == "native-metal"
    device = _device("metal" if native else args.backend)
    if args.backend == "cuda":
        import cupy as cp
        bus = cp.cuda.runtime.deviceGetPCIBusId(cp.cuda.Device().id)
        identity = subprocess.check_output([
            "nvidia-smi", "-i", bus, "--query-gpu=uuid", "--format=csv,noheader",
        ], text=True).strip()
    else:
        # Kept private: host identity is needed to reject cross-machine comparisons.
        identity = platform.node() + ":" + device["name"]
    report = {
        "schema": "quantem-gpu-io-loading/v1", "backend": args.backend,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device, "device_identity": identity, "storage_id": args.storage_id,
        "boundary": ("native catalog/index plus encoded HDF5 load, or QEM snapshot load; through synchronous residency"
                     if native else _BOUNDARY),
        "index_policy": "fresh for first HDF5 load; reused thereafter",
        "allocator_policy": ("fresh native process per load; process startup excluded"
                             if native else "release each Dataset owner; retain process GPU/pinned memory pools"),
        "product_definition": ("raw diffraction frames 0, middle, last; uint32 little-endian SHA256"
                               if native else "masked_sum with mask[::2,::2]=True; raw output bytes SHA256"),
        "preparation_backend": "mps" if native else args.backend,
        "native_binary_sha256": (hashlib.sha256(args.native_binary.read_bytes()).hexdigest()
                                 if native else None),
        "source_fingerprint": _fingerprint(), "runtime_source_before": runtime_source(),
        "git": _git_state(), "repetitions": args.reps, "cases": [], "passed": False,
        "manifest_sha256": hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest(),
        "baseline": str(args.baseline) if args.baseline else None,
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest() if baseline_bytes else None,
        "preparation_timed_as_load": False,
        "full_original_count_verification": False,
        "limitations": [
            "Cache eviction is advisory; inspect measured residency and process physical reads.",
            "Frozen encoded-body hashes and indexed product check do not rerun full count verification.",
            "No contention inference: review GPU/storage telemetry before accepting a baseline.",
            "Native uses separate processes and three frozen diffraction frames; Python uses an indexed detector product.",
        ],
    }

    def persist() -> None:
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)

    persist()
    previous_cache = os.environ.get("QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR")
    try:
        with tempfile.TemporaryDirectory(prefix="io-loading-", dir=output) as scratch:
            for item in items:
                master = (args.source / item["filename"]).resolve()
                digest = hashlib.sha256(master.read_bytes()).hexdigest()
                if digest != item["source_master_sha256"]:
                    raise ValueError(f"{item['id']}: master changed; use the frozen source.")
                reference_path = Path(item["qem_reference_header"])
                if not reference_path.is_absolute():
                    reference_path = args.selection_manifest.parent / reference_path
                reference = json.loads(reference_path.read_text())
                files = [master, *detector_files(master)]
                signatures = [(path.stat().st_size, path.stat().st_mtime_ns) for path in files]
                case = {"id": item["id"], "source_master_sha256": digest,
                        "reference_identity": _header_identity(reference),
                        "shape": item["shape"], "dtype": item["dtype"],
                        "native_hdf5_dtype": item.get("native_hdf5_dtype"),
                        "native_qem_dtype": item.get("native_qem_dtype"),
                        "source_device": master.stat().st_dev,
                        "qem_device": Path(scratch).stat().st_dev,
                        "source_files": [str(path) for path in files],
                        "source_signatures": signatures}
                report["cases"].append(case)
                print(f"{item['id']}: full acquisition loading check", flush=True)
                with tempfile.TemporaryDirectory(dir=scratch) as index:
                    os.environ["QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR"] = index

                    def retain(format_name, result):
                        case[format_name] = result
                        persist()

                    case["hdf5"] = _measure(master, files, args.backend, item, args.reps,
                                            lambda result: retain("hdf5", result), args.native_binary)
                    qem = Path(scratch) / "prepared.qem"
                    started = time.perf_counter()
                    prepared = convert(master, qem, backend="mps" if native else args.backend,
                                       verify=False, source_hashes=False,
                                       hot_pixel_correction="none")
                    case["preparation_seconds_excluded"] = time.perf_counter() - started
                    if not qem.exists() or prepared.failed:
                        raise RuntimeError(f"QEM preparation failed: {prepared.skipped}; inspect the acquisition.")
                    header, _ = read_header(qem)
                    case["encoded_reference_passed"] = (
                        _header_identity(header) == case["reference_identity"]
                    )
                    persist()
                    if not case["encoded_reference_passed"]:
                        raise ValueError("Encoded body/layout differs from frozen QEM; investigate before updating the reference.")
                    case["qem_bytes"] = qem.stat().st_size
                    case["qem"] = _measure(qem, [qem], args.backend, item, args.reps,
                                           lambda result: retain("qem", result), args.native_binary)
                    qem.unlink()
                after = [(path.stat().st_size, path.stat().st_mtime_ns) for path in files]
                if after != signatures:
                    raise ValueError("Source files changed during measurement; rerun from stable inputs.")
                persist()
        report["source_unchanged"] = report["source_fingerprint"] == _fingerprint()
        report["runtime_source_after"] = runtime_source()
        report["passed"] = report["source_unchanged"]
        report["performance"] = (_compare_reports(report, json.loads(baseline_bytes),
                                                  args.regression_percent)
                                 if args.baseline else {"status": "no_baseline"})
        report["status"] = "completed" if report["passed"] else "invalidated"
        if not report["source_unchanged"]:
            report["error"] = "Source changed during measurement; rerun from a frozen checkout."
    except Exception as error:
        report["passed"] = False
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    finally:
        if previous_cache is None:
            os.environ.pop("QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR", None)
        else:
            os.environ["QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR"] = previous_cache
        report["scratch_removed"] = True
        persist()
    if not report["passed"]:
        return 1
    return 2 if report["performance"]["status"] in ("review", "incomparable") else 0
