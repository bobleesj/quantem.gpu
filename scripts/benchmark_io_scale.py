#!/usr/bin/env python3
"""Run matched IO scale cases on one physical backend; keep reports private.

Execute once per backend and keep the three reports separate. Existing per-case
QEM reports are retained for resumability and must belong to the same inputs and
implementation; the aggregate records their provenance without relabeling them.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark_source import runtime_source
from check_ans_io import _device, _fingerprint


def main() -> None:
    """Measure full HDF5 loading and verified QEM workflows for every master."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--backend", choices=("cuda", "mps", "native-metal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cold-reloads", type=int, default=0,
                        help="Additional QEM reloads after per-file cache eviction; record residency and physical reads.")
    parser.add_argument("--native-binary", type=Path,
                        help="qem-roundtrip for scale; metal-runtime-ans-benchmark for --load-only.")
    parser.add_argument("--selection-manifest", type=Path,
                        help="Private manifest with selected_acquisitions: id, filename and source_master_sha256.")
    parser.add_argument("--fresh-index", action="store_true",
                        help="Start Python HDF5 trials with an empty per-acquisition frame-index cache.")
    parser.add_argument("--load-only", action="store_true",
                        help="Check HDF5/QEM loading against frozen references; preparation is untimed.")
    parser.add_argument("--baseline", type=Path,
                        help="Prior loading.json from the same device, storage and fixtures; never replaced.")
    parser.add_argument("--history-dir", type=Path,
                        help="Append loading evidence to a private immutable history with CSV/Markdown summaries.")
    parser.add_argument("--storage-id", help="Stable description of the storage used for loading checks.")
    parser.add_argument("--regression-percent", type=float, default=20.0,
                        help="Flag median slowdowns above this percentage for review (default: 20).")
    args = parser.parse_args()
    if args.load_only:
        if args.backend == "native-metal" and not args.native_binary:
            parser.error("Native loading requires --native-binary pointing to metal-runtime-ans-benchmark.")
        if not args.selection_manifest or not args.storage_id or args.reps < 3:
            parser.error("--load-only requires --selection-manifest, --storage-id and --reps >= 3 (recommended: 5).")
        if not 0 <= args.regression_percent < 100:
            parser.error("--regression-percent must be in [0, 100).")
        from benchmark_io_loading import _run_loading_check

        result = _run_loading_check(args)
        if args.history_dir:
            from benchmark_io_history import _register

            _register([args.output_dir / "loading.json"], args.history_dir)
        raise SystemExit(result)
    if args.reps < 1:
        parser.error("--reps must be positive.")
    if args.cold_reloads < 0:
        parser.error("--cold-reloads must be nonnegative.")
    if args.backend == "native-metal" and args.native_binary is None:
        parser.error("Supply the release qem-roundtrip --native-binary.")

    from quantem.gpu.io.convert import find_masters

    imported_source = runtime_source()
    masters = find_masters(args.source)
    if not masters:
        parser.error("No acquisition masters found; select the source collection.")
    selected = [(f"acquisition-{number:02d}", master) for number, master in enumerate(masters, start=1)]
    if args.selection_manifest:
        manifest = json.loads(args.selection_manifest.read_text())
        available = {master.name: master for master in masters}
        selected = []
        for item in manifest["selected_acquisitions"]:
            if item["filename"] not in available:
                parser.error(f"Selected master {item['filename']} is absent; use its matching source collection.")
            master = available[item["filename"]]
            if hashlib.sha256(master.read_bytes()).hexdigest() != item["source_master_sha256"]:
                parser.error(f"Selected master {item['filename']} changed; verify the acquisition before benchmarking.")
            selected.append((item["id"], master))
        if not selected or len({label for label, _ in selected}) != len(selected):
            parser.error("The selection manifest must contain distinct acquisition IDs and at least one acquisition.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    report = {
        "backend": args.backend,
        "device": _device("metal" if args.backend == "native-metal" else args.backend),
        "source_fingerprint": _fingerprint(),
        "runtime_source": imported_source,
        "source": str(args.source.resolve()),
        "repetitions": args.reps,
        "cold_reload_repetitions": args.cold_reloads,
        "cache_policy": "uncontrolled OS/compiler caches; native trials use separate processes",
        "fresh_python_frame_index": args.fresh_index,
        "selection_manifest_sha256": hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest() if args.selection_manifest else None,
        "cases": [],
    }

    def persist() -> None:
        (output / "scale.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(command: list[str], log: Path, env: dict[str, str] | None = None) -> int:
        with log.open("w") as handle:
            return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False, env=env).returncode

    persist()
    for label, master in selected:
        case = {"id": label, "master": str(master.resolve())}
        report["cases"].append(case)
        print(f"{label}: {args.backend} HDF5 load", flush=True)
        hdf5 = output / f"{master.stem}-hdf5.json"
        if not hdf5.exists():
            if args.backend == "native-metal":
                binary = args.native_binary.parent / "metal-runtime-ans-benchmark"
                runs = []
                with tempfile.TemporaryDirectory(prefix="native-io-index-") as directory:
                    for _ in range(args.reps):
                        result = subprocess.run(
                            [str(binary.resolve()), str(master.resolve()), directory, "--load-only"],
                            capture_output=True, text=True, check=False,
                        )
                        runs.append(json.loads(result.stdout) if result.returncode == 0 else {
                            "error": result.stderr, "returncode": result.returncode,
                        })
                        if result.returncode:
                            break
                hdf5.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
            else:
                with tempfile.TemporaryDirectory(prefix="python-io-index-") as directory:
                    environment = dict(os.environ)
                    if args.fresh_index:
                        environment["QUANTEM_GPU_HDF5_FRAME_SOURCE_CACHE_DIR"] = directory
                    case["hdf5_returncode"] = run([
                        sys.executable, str(scripts / "benchmark_hdf5_load.py"), str(master.resolve()),
                        "--backend", args.backend, "--dtype", "native", "--reps", str(args.reps),
                        "--hot-pixel-correction", "none",
                        "--cache-state", "Empty frame index; OS/compiler caches uncontrolled" if args.fresh_index else "OS cache uncontrolled",
                        "--fixture-id", label, "--json-out", str(hdf5),
                    ], output / f"{master.stem}-hdf5.log", env=environment)
        case["hdf5_report"] = str(hdf5)
        qem = output / f"{master.stem}-qem.json"
        # A native HDF5 rejection cannot produce a QEM. Keep that cell visible.
        native_rejected = args.backend == "native-metal" and any(
            "error" in item for item in json.loads(hdf5.read_text())["runs"]
        )
        if native_rejected:
            case["qem_status"] = "not-run: native HDF5 load rejected"
        elif not qem.exists():
            print(f"{label}: {args.backend} QEM conversion and full verification", flush=True)
            command = [
                sys.executable, str(scripts / "benchmark_qem.py"), str(master.resolve()),
                "--backend", args.backend, "--output", str(qem), "--reloads", str(args.reps),
                "--cold-reloads", str(args.cold_reloads),
            ]
            if args.native_binary:
                command += ["--native-binary", str(args.native_binary.resolve())]
            case["qem_returncode"] = run(command, output / f"{master.stem}-qem.log")
        else:
            case["qem_status"] = "retained existing measurement; inspect its provenance"
        if qem.exists():
            case["qem_report"] = str(qem)
            case["verified"] = json.loads(qem.read_text()).get("passed", False)
        persist()
    report["source_unchanged"] = report["source_fingerprint"] == _fingerprint()
    report["completed"] = True
    persist()


if __name__ == "__main__":
    main()
