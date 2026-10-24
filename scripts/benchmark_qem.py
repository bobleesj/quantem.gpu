#!/usr/bin/env python3
"""Measure QEM conversion and optional source verification on physical GPUs.

Reports contain private source paths: keep them in the private evidence archive.
Native runs use the repository's qem-roundtrip executable and independently
verify every saved count with the Python MPS reader. That verification is
explicitly a separate backend, not a native Swift verification timing.
"""

import argparse
import ctypes
import dataclasses
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from benchmark_file_cache import cache_state, evict_file_cache
from benchmark_hdf5_load import _sync_backend
from benchmark_source import runtime_source
from check_ans_io import _fingerprint


def _layout_bytes(header: dict, file_bytes: int) -> dict:
    """Account for every byte, separating spatial indexes from ANS streams."""
    import numpy as np

    from quantem.gpu.formats.qem.snapshot import ARRAY_DTYPES, INTEGER_CODEC

    if header["codec"] != INTEGER_CODEC:
        return {"codec": header["codec"], "file_bytes": file_bytes}
    names = ("ans_payload", "stream_offsets", "models", "spatial_words",
             "spatial_starts", "spatial_widths")
    arrays = {
        name: sum(chunk["arrays"][index]["count"] for chunk in header["chunks"])
        * np.dtype(dtype).itemsize
        for index, (name, dtype) in enumerate(zip(names, ARRAY_DTYPES))
    }
    spatial = sum(arrays[name] for name in names[3:])
    return {
        "codec": header["codec"], "file_bytes": file_bytes,
        "arrays": arrays, "spatial_index_bytes": spatial,
        "ans_core_bytes": sum(arrays[name] for name in names[:3]),
        "body_padding_bytes": header["bytes"] - sum(arrays.values()),
        "envelope_bytes": file_bytes - header["bytes"],
        "file_minus_spatial_bytes": file_bytes - spatial,
        "without_spatial_semantics": "Byte subtraction only, not a supported QEM file or measured load",
    }


def _process_io() -> dict:
    """Read process IO and memory counters where the operating system exposes them."""
    result = {}
    for name in ("/proc/self/io", "/proc/meminfo"):
        path = Path(name)
        if path.exists():
            result[name] = {
                key: int(value.split()[0])
                for key, value in (line.split(":", 1) for line in path.read_text().splitlines())
            }
    if sys.platform == "darwin":
        # RUSAGE_INFO_V2 is defined in the platform's sys/resource.h. Its
        # counters distinguish storage traffic from reads served by file cache.
        class Usage(ctypes.Structure):
            _fields_ = [
                ("uuid", ctypes.c_uint8 * 16),
                ("preceding", ctypes.c_uint64 * 16),
                ("diskio_bytesread", ctypes.c_uint64),
                ("diskio_byteswritten", ctypes.c_uint64),
            ]

        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            library.proc_pid_rusage.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(Usage),
            ]
            library.proc_pid_rusage.restype = ctypes.c_int
            usage = Usage()
            if library.proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage)) == 0:
                result["darwin/proc_pid_rusage"] = {
                    "diskio_bytesread": usage.diskio_bytesread,
                    "diskio_byteswritten": usage.diskio_byteswritten,
                    "resident_size": usage.preceding[6],
                    "phys_footprint": usage.preceding[7],
                }
            else:
                result["darwin/proc_pid_rusage_error"] = os.strerror(ctypes.get_errno())
        except (OSError, AttributeError) as error:
            result["darwin/proc_pid_rusage_error"] = str(error)
    return result


def _profile_reload(path: Path, backend: str, load) -> dict:
    """Profile an additional public load without altering its checks or arrays."""
    events = []
    local = threading.local()
    selected = {"readinto", "preadv", "openssl_sha256", "update", "hexdigest", "synchronize",
                "set", "empty", "alloc_pinned_memory"}

    def trace(frame, event, function):
        name = getattr(function, "__name__", "")
        if event not in {"c_call", "c_return", "c_exception"} or name not in selected:
            return
        if event == "c_call":
            local.active = (name, time.perf_counter(), threading.get_ident())
        elif getattr(local, "active", (None,))[0] == name:
            label, started, worker = local.active
            events.append({"operation": label, "start": started,
                           "end": time.perf_counter(), "thread": worker})
            local.active = (None,)

    previous = threading.getprofile()
    previous_main = sys.getprofile()
    main_thread = threading.get_ident()
    before = _process_io()
    _sync_backend(backend)
    started = time.perf_counter()
    threading.setprofile(trace)
    try:
        sys.setprofile(trace)
        with load(path, backend=backend, representation="encoded", verbose=False):
            _sync_backend(backend)
            elapsed = time.perf_counter() - started
            sys.setprofile(previous_main)
    finally:
        sys.setprofile(previous_main)
        threading.setprofile(previous)
    after = _process_io()
    for item in events:
        item["start"] -= started
        item["end"] -= started
    return {
        "wall_seconds": elapsed, "process_before": before, "process_after": after,
        "main_thread": main_thread,
        "worker_events": events,
        "semantics": "Additional instrumented load; traced C calls may overlap across threads and are not additive wall time. Host call time is not GPU event time.",
    }


def main() -> None:
    """Run one complete conversion with disposable QEM storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--backend", choices=("cuda", "mps", "native-metal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--native-load-binary", type=Path)
    parser.add_argument("--reloads", type=int, default=3)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                        help="Compare every original count (default: enabled for benchmark qualification).")
    parser.add_argument("--source-hashes", action=argparse.BooleanOptionalAction, default=True,
                        help="Hash all source files (default: enabled for benchmark qualification).")
    parser.add_argument("--cold-reloads", type=int, default=0,
                        help="Additional reloads after per-file cache eviction of the disposable QEM.")
    parser.add_argument("--profile-reload", action="store_true",
                        help="Profile one extra Python reload after normal timing trials.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence already exists; select a new --output path.")
    if args.reloads < 0 or args.cold_reloads < 0:
        parser.error("--reloads and --cold-reloads must be nonnegative.")
    if args.cold_reloads and sys.platform not in ("darwin", "linux"):
        parser.error("--cold-reloads requires Linux or macOS per-file cache controls.")
    if args.backend == "native-metal" and args.native_binary is None:
        parser.error("Build scripts/build_qem_roundtrip.sh and supply --native-binary.")
    if args.backend == "native-metal" and args.profile_reload:
        parser.error("--profile-reload supports Python CUDA and MPS; omit it for native-metal.")
    if args.backend == "native-metal" and (not args.verify or not args.source_hashes):
        parser.error("Optional source checks currently apply to Python CUDA and MPS only.")

    from quantem.gpu.formats.qem.snapshot import read_header
    from quantem.gpu.io import load

    converter = importlib.import_module("quantem.gpu.io.convert")
    imported_source = runtime_source()
    verification_backend = "mps" if args.backend == "native-metal" else args.backend
    report = {
        "source": str(args.source.resolve()),
        "backend": args.backend,
        "source_fingerprint": _fingerprint(),
        "runtime_source_before": imported_source,
        "verification_backend": verification_backend,
        "verify": args.verify,
        "source_hashes": args.source_hashes,
        "cache_policy": "uncontrolled OS cache; first observation is not cold-disk",
        "stages": [],
        "reload_seconds": [],
        "reload_runs": [],
        "cold_reload_runs": [],
        "cold_reload_policy": "Per-file eviction requested; inspect residency and physical reads to confirm cold storage",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    def timed(name, function):
        def run(*positional, **keywords):
            _sync_backend(verification_backend)
            before = _process_io()
            started = time.perf_counter()
            result = function(*positional, **keywords)
            _sync_backend(verification_backend)
            stage = {
                "stage": name, "seconds": time.perf_counter() - started,
                "process_before": before, "process_after": _process_io(),
            }
            if name == "load":
                stage["load_timings"] = dict(result.metadata.get("load_timings", {}))
            report["stages"].append(stage)
            persist()
            return result
        return run

    scratch = None
    try:
        with tempfile.TemporaryDirectory(prefix="qem-benchmark-") as directory:
            scratch = Path(directory)
            destination = scratch / "acquisition.qem"
            _sync_backend(verification_backend)
            workflow_started = time.perf_counter()
            if args.backend == "native-metal":
                started = time.perf_counter()
                native = subprocess.run(
                    [str(args.native_binary.resolve()), str(args.source.resolve()), str(destination)],
                    capture_output=True, text=True, check=False,
                )
                report["native_wall_seconds"] = time.perf_counter() - started
                report["native_stdout"] = native.stdout
                report["native_stderr"] = native.stderr
                report["native_returncode"] = native.returncode
                persist()
                native.check_returncode()
                report["verification"] = timed("full_source_verification", converter.verify_against_source)(
                    destination, args.source, backend="mps",
                )
            else:
                with (
                    patch.object(converter, "load", timed("load", converter.load)),
                    patch.object(converter, "save", timed("save", converter.save)),
                    patch.object(converter, "verify_against_source", timed("full_source_verification", converter.verify_against_source)),
                ):
                    converted = converter.convert(
                        args.source, destination, backend=args.backend,
                        verify=args.verify, source_hashes=args.source_hashes,
                    )
                report["conversion"] = dataclasses.asdict(converted)
                report["verification"] = converted.verification
                persist()
                if converted.failed:
                    raise RuntimeError("QEM conversion failed; inspect retained verification evidence.")
                if converted.larger:
                    # The production converter declines expansion. Benchmark
                    # a disposable raw-preserving copy without changing that policy.
                    report["larger_copy_benchmark"] = True
                    with timed("larger_copy_source_load", converter.load)(
                        args.source, backend=args.backend, representation="encoded",
                        hot_pixel_correction="none", verbose=False,
                    ) as acquisition:
                        timed("larger_copy_save", converter.save)(destination, acquisition, format="quantem")
                    if args.verify:
                        report["verification"] = timed("full_source_verification", converter.verify_against_source)(
                            destination, args.source, backend=args.backend,
                        )
            header, _ = read_header(destination)
            report["shape"] = header["shape"]
            report["stored_dtype"] = header["dtype"]
            report["source_files"] = header.get("metadata", {}).get("source_files", [])
            report["qem_bytes"] = destination.stat().st_size
            verification = report["verification"]
            if args.verify and (
                not verification.get("identical")
                or verification.get("compared_values") != math.prod(header["shape"])
            ):
                raise RuntimeError("Full lossless verification did not cover every count exactly.")
            report["conversion_seconds"] = time.perf_counter() - workflow_started
            report["conversion_boundary"] = (
                "Native round trip, including product checks, then exhaustive MPS source comparison"
                if args.backend == "native-metal" else
                "Canonical conversion through save, fsync and publication, plus requested source checks; "
                "includes disposable retry when larger"
            )
            if args.verify:
                report["verified_conversion_seconds"] = report["conversion_seconds"]
                report["verified_conversion_boundary"] = report["conversion_boundary"]
            report["layout_bytes"] = _layout_bytes(header, report["qem_bytes"])
            if args.backend == "native-metal":
                binary = args.native_load_binary or args.native_binary.parent / "metal-runtime-ans-benchmark"
                report["native_reload_runs"] = []
                for trial in range(args.reloads + args.cold_reloads):
                    cold = trial >= args.reloads
                    control = evict_file_cache(destination) if cold else None
                    state = cache_state(destination)
                    reopened = subprocess.run(
                        [str(binary.resolve()), str(destination), "--qem-load-only"],
                        capture_output=True, text=True, check=True,
                    )
                    measured = json.loads(reopened.stdout)
                    if measured["shape"] != header["shape"]:
                        raise RuntimeError("Native QEM reload changed the acquisition shape.")
                    measured.update(cache_before=state, cache_control=control)
                    if cold:
                        report["cold_reload_runs"].append(measured)
                    else:
                        report["native_reload_runs"].append(measured)
                        report["reload_seconds"].append(measured["load_seconds"])
                    persist()
            else:
                for trial in range(args.reloads + args.cold_reloads):
                    _sync_backend(args.backend)
                    cold = trial >= args.reloads
                    control = evict_file_cache(destination) if cold else None
                    state = cache_state(destination)
                    before = _process_io()
                    started = time.perf_counter()
                    with load(destination, backend=args.backend, representation="encoded", verbose=False) as acquisition:
                        _sync_backend(args.backend)
                        elapsed = time.perf_counter() - started
                        after = _process_io()
                        measured = {
                            "seconds": elapsed,
                            "cache_before": state,
                            "cache_control": control,
                            "process_before": before,
                            "process_after": after,
                            "load_timings": dict(acquisition.metadata.get("load_timings", {})),
                        }
                        if cold:
                            report["cold_reload_runs"].append(measured)
                        else:
                            report["reload_seconds"].append(elapsed)
                            report["reload_runs"].append(measured)
                    persist()
                if args.profile_reload:
                    report["reload_profile"] = _profile_reload(destination, args.backend, load)
                    persist()
            report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["scratch_removed"] = scratch is None or not scratch.exists()
        report["source_unchanged"] = report["source_fingerprint"] == _fingerprint()
        try:
            report["runtime_source_after"] = runtime_source()
        except RuntimeError as error:
            report["passed"] = False
            report["runtime_source_error"] = str(error)
            persist()
            raise
        persist()


if __name__ == "__main__":
    main()
