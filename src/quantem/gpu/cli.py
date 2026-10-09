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
    convert_command = commands.add_parser(
        "convert",
        help="convert Arina HDF5 acquisitions into verified .qem copies",
        description=(
            "Write a .qem copy of one acquisition, or of every *_master.h5 below a "
            "folder. Counts are stored exactly in compressed form and acquisition "
            "metadata is retained. Each copy is compared with its source files. "
            "Source files are never modified or removed."
        ),
    )
    convert_command.add_argument("source", type=Path, help="a *_master.h5 file, or a folder of acquisitions")
    convert_command.add_argument("--out", type=Path, help="write copies here, mirroring the folder layout (default: beside each master)")
    convert_command.add_argument("--dry", action="store_true", help="encode on GPU and estimate payload size without writing a copy")
    convert_command.add_argument("--backend", choices=("auto", "cuda", "mps"), default="auto")
    convert_command.add_argument("--no-verify", action="store_true", help="skip comparing each copy with its source files")
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
    if args.command == "convert":
        return _convert(args)
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


def _convert(args: argparse.Namespace) -> int:
    """Convert every acquisition under the source and print sizes before and after."""
    try:
        masters = convert.find_masters(args.source)
    except FileNotFoundError as error:
        raise SystemExit(f"quantem-gpu convert: {error}") from None
    if not masters:
        raise SystemExit(f"No *_master.h5 acquisitions under {args.source}")
    print(
        f"{len(masters)} acquisition(s). Every detector count and the pixel mask are kept exactly; "
        "flagged pixels holding the uint32 marker (above 65535) are stored as 0. Source files are not modified."
    )
    if args.dry:
        print("Dry run: GPU encoding estimates payload size; metadata and file overhead are additional. No copies written.")
    before = after = failed = kept = 0
    for master in masters:
        destination = convert.destination_for(master, args.source, args.out)
        try:
            result = convert.convert(
                master, destination, write=not args.dry, verify=not args.no_verify,
                backend=args.backend,
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
            line += f"  (calibration from dataset.yaml: {', '.join(path.rsplit('/', 1)[-1] for path in result.session_calibration)})"
        if result.verified is True:
            check = result.verification
            line += f"  verified: {check['compared_values']:,} values identical"
            if check["flagged_pixels"]:
                line += f"; {check['flagged_pixels']} flagged pixels also verified"
            if check.get("flagged_markers_stored_as_zero"):
                line += f" ({check['flagged_markers_stored_as_zero']:,} uint32 markers stored as 0)"
        elif result.verified is False:
            failed += 1
            line += f"  VERIFICATION FAILED, no copy published: {result.verification}"
        for note in result.session_notes:
            line += f"\n    note: {note}"
        print(line)
    if after:
        print(f"Total: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB ({before / after:.2f}x)")
    if kept:
        print(f"{kept} acquisition(s) stay as HDF5 because their copy would be larger.")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
