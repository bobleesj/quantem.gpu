"""Command-line entry points for QuantEM GPU services."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from quantem.gpu import __version__
from quantem.gpu.formats.qem import validation
from quantem.gpu.io import convert


def _parser() -> argparse.ArgumentParser:
    """Build the command parser apart from :func:`main` so arguments can be checked without running a command."""
    parser = argparse.ArgumentParser(
        prog="quantem-gpu",
        description="Accelerated 4D-STEM compute services.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    # benchmark-ssb, the name before the Direct Ptycho rename, stays the same command (release CI calls it)
    benchmark = commands.add_parser(
        "benchmark-direct-ptycho", aliases=["benchmark-ssb"],
        help="measure full-field and regional Direct Ptycho joint fits on CUDA or MPS",
    )
    benchmark.add_argument("source", type=Path, help="local H5 or QEM acquisition")
    benchmark.add_argument("--out", type=Path, required=True, help="write the timing report outside the repository")
    benchmark.add_argument("--backend", choices=("auto", "cuda", "mps"), default="auto")
    benchmark.add_argument("--runs", type=int, default=6, help="one first observation plus five warm measurements")
    benchmark.add_argument("--region-sizes", type=int, nargs="+", default=[256, 128])
    benchmark.add_argument("--trials", type=int, default=200)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--require-installed", action="store_true", help="reject editable source installs for release qualification")
    benchmark.add_argument("--reference-phase", type=Path, help="compare with a frozen Auto phase .npy from the same backend and acquisition")
    benchmark.add_argument("--voltage-kv", type=float)
    benchmark.add_argument("--semiangle-mrad", type=float)
    benchmark.add_argument("--scan-sampling-a", type=float, nargs="+")
    benchmark.add_argument("--detector-sampling-mrad", type=float, nargs="+")
    convert_command = commands.add_parser(
        "convert",
        help="convert Arina HDF5 acquisitions into .qem copies",
        description=(
            "Write a .qem copy of one acquisition, or of every *_master.h5 below a "
            "folder. Original values are encoded losslessly by default; explicitly "
            "requested detector-mask corrections are recorded. "
            "Calibration, QEM checksums and disk synchronization are retained. "
            "Full source comparison and source-file hashing are opt-in. "
            "Source files are never modified or removed."
        ),
    )
    convert_command.add_argument("source", type=Path, help="a *_master.h5 file, or a folder of acquisitions")
    convert_command.add_argument("--out", type=Path, help="write copies here, mirroring the folder layout (default: beside each master)")
    convert_command.add_argument("--dry", action="store_true", help="encode on GPU and estimate payload size without writing a copy")
    convert_command.add_argument("--backend", choices=("auto", "cuda", "mps"), default="auto")
    convert_command.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False,
                                 help="compare every saved value with its source (default: off)")
    convert_command.add_argument("--source-hashes", action="store_true",
                                 help="read all original files to record SHA-256 fingerprints (default: off)")
    convert_command.add_argument("--voltage-kv", type=float, help="accelerating voltage in kV; overrides session/source metadata")
    convert_command.add_argument("--semiangle-mrad", type=float, help="convergence semiangle in mrad; overrides session/source metadata")
    convert_command.add_argument("--scan-sampling-a", type=float, nargs="+", metavar="VALUE",
                                 help="scan spacing in angstrom: one value, or row column")
    convert_command.add_argument("--detector-sampling-mrad", type=float, nargs="+", metavar="VALUE",
                                 help="detector spacing in mrad/pixel: one value, or row column")
    convert_command.add_argument("--hot-pixel-correction", choices=("none", "median", "zero"), default="none",
                                 help="repair only stored-mask pixels before encoding (default: none)")
    validate_command = commands.add_parser(
        "validate",
        help="check .qem integrity and metadata without a GPU",
    )
    validate_command.add_argument("paths", nargs="+", type=Path, help=".qem files to check")
    serve_command = commands.add_parser(
        "serve",
        help="serve native 4D-STEM browsing over loopback",
        description=(
            "Serve one remote 4D-STEM folder to a native client. The server "
            "listens only on 127.0.0.1 and is intended to be reached through SSH."
        ),
    )
    serve_command.add_argument("data_folder", help="folder containing *_master.h5 sessions")
    gpu_selection = serve_command.add_mutually_exclusive_group()
    gpu_selection.add_argument(
        "--gpu",
        type=int,
        help="use one CUDA device index",
    )
    gpu_selection.add_argument(
        "--gpus",
        default="auto",
        help="CUDA device pool: auto or comma-separated indices (default: auto)",
    )
    serve_command.add_argument(
        "--port", type=int, default=8780,
        help="loopback port (default: 8780; another local service may already use it, then pass --port)",
    )
    serve_command.add_argument(
        "--implementation-revision",
        default=__version__,
        help=(
            "quantem.gpu revision reported to clients, for example an exact Git "
            "commit (default: the installed package version)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested QuantEM GPU command."""
    args = _parser().parse_args(argv)
    if args.command in ("benchmark-direct-ptycho", "benchmark-ssb"):
        import json
        from quantem.gpu.benchmark import run_direct_ptycho

        calibration = {
            "voltage_kV": args.voltage_kv, "semiangle_mrad": args.semiangle_mrad,
            "scan_sampling_A": args.scan_sampling_a,
            "detector_sampling_mrad": args.detector_sampling_mrad,
        }
        calibration = {name: value for name, value in calibration.items() if value is not None}
        for name in ("scan_sampling_A", "detector_sampling_mrad"):
            value = calibration.get(name)
            if value is not None and len(value) == 1:
                calibration[name] = value[0]
        report = run_direct_ptycho(
            args.source, backend=args.backend, runs=args.runs,
            region_sizes=tuple(args.region_sizes), num_trials=args.trials,
            seed=args.seed, calibration=calibration, progress=print,
            require_installed=args.require_installed,
            reference_phase=args.reference_phase,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("Fit                              first (s)   warm median (s)   warm p90 (s)")
        for name, result in report["measurements"].items():
            print(f"{name:32} {result['first_s']:9.3f} {result['warm_p50_s']:17.3f} {result['warm_p90_s']:14.3f}")
        print("Each measurement uses one seed, the full trial budget and Newton. Browser paint is excluded.")
        return 0
    if args.command == "convert":
        try:
            convert._explicit_calibration(args.voltage_kv, args.semiangle_mrad,
                                          args.scan_sampling_a, args.detector_sampling_mrad)
        except ValueError as error:
            raise SystemExit(f"quantem-gpu convert: {error}") from None
        # convert every acquisition under the source and print sizes before and after
        try:
            masters = convert.find_masters(args.source)
        except FileNotFoundError as error:
            raise SystemExit(f"quantem-gpu convert: {error}") from None
        if not masters:
            raise SystemExit(f"No *_master.h5 acquisitions under {args.source}")
        print(f"{len(masters)} acquisition(s). Source files are not modified.")
        if args.hot_pixel_correction == "none":
            print("No hot-pixel correction; original values and flagged markers are preserved exactly.")
        else:
            print(f"Stored-mask pixels receive {args.hot_pixel_correction} correction. QEM losslessly stores the corrected values.")
        if args.dry:
            print("Dry run: GPU encoding estimates payload size; metadata and file overhead are additional. No copies written.")
        before = after = failed = kept = 0
        for master in masters:
            destination = convert.destination_for(master, args.source, args.out)
            try:
                result = convert.convert(
                    master, destination, write=not args.dry, verify=args.verify,
                    source_hashes=args.source_hashes,
                    backend=args.backend,
                    voltage_kV=args.voltage_kv, semiangle_mrad=args.semiangle_mrad,
                    scan_sampling_A=args.scan_sampling_a, detector_sampling_mrad=args.detector_sampling_mrad,
                    hot_pixel_correction=args.hot_pixel_correction,
                )
            except (OSError, ValueError, RuntimeError) as error:
                failed += 1
                print(f"  failed    {master.name}: {error}")
                continue
            name = master.name[: -len("_master.h5")]
            if result.skipped:
                failed += int(result.failed)
                print(f"  skipped   {name}: {result.skipped}")
                continue
            if result.larger:
                kept += 1
                print(
                    f"  kept HDF5 {name}: the copy would be larger "
                    f"({result.source_bytes / 1e9:.2f} GB -> {result.qem_bytes / 1e9:.2f} GB); nothing written"
                )
                continue
            before += result.source_bytes
            after += result.qem_bytes
            line = (
                f"  {name}: {result.source_bytes / 1e9:.2f} GB -> {result.qem_bytes / 1e9:.2f} GB "
                f"({result.source_bytes / result.qem_bytes:.2f}x, {result.seconds:.0f} s)"
            )
            if not result.master_embedded:
                line += "  (master file too large to embed; its fields are kept, its long tables are not)"
            if result.session_calibration:
                for path, quantity in result.session_calibration.items():
                    line += f"\n    {path}: {quantity['value']:g} {quantity['unit']} ({quantity['evidence']})"
            correction = result.hot_pixel_correction
            if correction.get("applied"):
                line += (f"\n    {correction['method']} correction: {correction['pixel_count']} detector pixels; "
                         f"(row, column) {correction['coordinates_row_col']}; original flags {correction['source_flags']}")
            if result.verified is True:
                check = result.verification
                line += f"  verified: {check['compared_values']:,} values identical"
                if check.get("corrected_values_compared"):
                    line += f" after declared correction ({check['corrected_values_compared']:,} replacements checked)"
                if check["flagged_pixels"]:
                    line += f"; {check['flagged_pixels']} flagged pixels also verified"
                if check.get("flagged_markers_stored_as_zero"):
                    line += f" ({check['flagged_markers_stored_as_zero']:,} uint32 markers stored as 0)"
            elif result.verified is False:
                failed += 1
                line += f"  VERIFICATION FAILED, no copy published: {result.verification}"
            elif not args.dry:
                line += "  full source comparison not requested"
            for note in result.session_notes:
                line += f"\n    note: {note}"
            print(line)
        if after:
            print(f"Total: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB ({before / after:.2f}x)")
        if kept:
            print(f"{kept} acquisition(s) stay as HDF5 because their copy would be larger.")
        return 1 if failed else 0
    if args.command == "validate":
        return validation.main([str(path) for path in args.paths])
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.gpu is not None and args.gpu < 0:
        raise SystemExit("--gpu must be zero or greater")
    if args.gpu is not None:
        gpus: list[int] | str = [args.gpu]
    elif args.gpus == "auto":
        gpus = "auto"
    else:
        try:
            gpus = list(dict.fromkeys(int(value) for value in args.gpus.split(",")))
        except ValueError as exc:
            raise SystemExit("--gpus must be 'auto' or comma-separated CUDA indices") from exc
        if not gpus or any(gpu < 0 for gpu in gpus):
            raise SystemExit("--gpus must contain CUDA indices zero or greater")
    if not Path(args.data_folder).is_dir():
        raise SystemExit(f"quantem-gpu serve: {args.data_folder} is not a folder")
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Remote viewing requires FastAPI and Uvicorn. Install "
            "quantem.gpu with the remote extra: "
            "pip install 'quantem.gpu[cuda,remote]'"
        ) from exc

    from quantem.gpu.remote import create_app

    app = create_app(
        args.data_folder,
        gpus=gpus,
        implementation_revision=args.implementation_revision,
    )
    service = app.state.browse_service
    if not service.residency.gpus:
        raise SystemExit(f"CUDA unavailable: {service.device_error}")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
