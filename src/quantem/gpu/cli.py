"""Command-line entry points for QuantEM GPU services."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantem-gpu",
        description="Accelerated 4D-STEM compute services.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-browse",
        help="verify a qualified packed source and create a new sealed registry",
    )
    prepare.add_argument("master", type=Path)
    prepare.add_argument("source", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--expected-source-sha256", required=True)
    convert = commands.add_parser(
        "convert",
        help="convert Arina HDF5 acquisitions into verified .qem copies",
        description=(
            "Write a .qem copy of one acquisition, or of every *_master.h5 below a "
            "folder. Counts are stored exactly in compressed form and acquisition "
            "metadata is retained. Each copy is compared with its source files. "
            "Source files are never modified or removed."
        ),
    )
    convert.add_argument("source", type=Path, help="a *_master.h5 file, or a folder of acquisitions")
    convert.add_argument("--out", type=Path, help="write copies here, mirroring the folder layout (default: beside each master)")
    convert.add_argument("--dry", action="store_true", help="encode on GPU and estimate payload size without writing a copy")
    convert.add_argument("--backend", choices=("auto", "cuda", "mps"), default="auto")
    convert.add_argument("--no-verify", action="store_true", help="skip comparing each copy with its source files")
    validate = commands.add_parser(
        "validate",
        help="check .qem integrity and metadata without a GPU",
    )
    validate.add_argument("paths", nargs="+", type=Path, help=".qem files to check")
    serve = commands.add_parser(
        "serve",
        help="serve native 4D-STEM browsing over loopback",
        description=(
            "Serve one remote 4D-STEM folder to a native client. The server "
            "listens only on 127.0.0.1 and is intended to be reached through SSH."
        ),
    )
    serve.add_argument("data_folder", help="folder containing *_master.h5 sessions")
    gpu_selection = serve.add_mutually_exclusive_group()
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
    serve.add_argument("--port", type=int, default=8780, help="loopback port (default: 8780)")
    serve.add_argument(
        "--implementation-revision",
        required=True,
        help="exact immutable quantem.gpu Git revision served in provenance",
    )
    serve.add_argument(
        "--compact-sources",
        type=Path,
        help=(
            "trusted JSON registry binding catalogued masters to immutable "
            "compact artifacts"
        ),
    )
    mps = commands.add_parser(
        "serve-ssb-mps",
        help="serve explicit local MPS SSB over loopback",
        description=(
            "Serve one local SSB data folder on 127.0.0.1 using MLX/Metal. "
            "This endpoint never falls back to CUDA or CPU."
        ),
    )
    mps.add_argument("data_folder", help="folder containing *_master.h5 data")
    mps.add_argument("--port", type=int, default=8781, help="loopback port (default: 8781)")
    mps.add_argument(
        "--implementation-revision",
        required=True,
        help="exact immutable quantem.gpu Git revision served in provenance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested QuantEM GPU command."""
    args = _parser().parse_args(argv)
    if args.command == "prepare-browse":
        from quantem.gpu.remote import prepare_browse_source

        registry = prepare_browse_source(
            args.master,
            args.source,
            args.destination,
            expected_source_sha256=args.expected_source_sha256,
        )
        print(registry)
        return 0
    if args.command == "convert":
        return _convert(args)
    if args.command == "validate":
        from quantem.gpu.io import qem_validation

        return qem_validation.main([str(path) for path in args.paths])
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.command == "serve-ssb-mps":
        return _serve_ssb_mps(args)
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
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Remote viewing requires FastAPI and Uvicorn. Install "
            "quantem.gpu with the remote extra: "
            "pip install 'quantem.gpu[cuda,remote]'"
        ) from exc

    from quantem.gpu.remote import create_app, load_compact_browse_sources

    compact_sources = (
        load_compact_browse_sources(args.compact_sources, args.data_folder)
        if args.compact_sources is not None
        else None
    )

    app = create_app(
        args.data_folder,
        gpus=gpus,
        compact_sources=compact_sources,
        implementation_revision=args.implementation_revision,
    )
    service = app.state.browse_service
    if service.backend != "cuda":
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
    from quantem.gpu.io import qem_conversion

    masters = qem_conversion.find_masters(args.source)
    if not masters:
        raise SystemExit(f"No *_master.h5 acquisitions under {args.source}")
    print(
        f"{len(masters)} acquisition(s). Stored counts, including flagged pixels, are preserved. "
        "Source files are not modified."
    )
    if args.dry:
        print("Dry run: GPU encoding estimates payload size; metadata and file overhead are additional. No copies written.")
    before = after = failed = kept = 0
    for master in masters:
        destination = qem_conversion.destination_for(master, args.source, args.out)
        try:
            result = qem_conversion.convert(
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
                line += (
                    f"; {check['flagged_pixels']} flagged pixels also verified"
                )
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


def _serve_ssb_mps(args: argparse.Namespace) -> int:
    """Run the explicit local-MPS-only SSB loopback worker."""

    try:
        import mlx.core as mx
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Local MPS SSB requires MLX, FastAPI, and Uvicorn. Install "
            "quantem.gpu with the mps and remote extras."
        ) from exc

    from quantem.gpu.remote.server import BrowseService, create_app
    from quantem.gpu.remote.ssb_api import SSBProtocolService

    device_name = str(mx.device_info().get("device_name") or "").strip()
    if not device_name:
        raise SystemExit("Local MPS SSB unavailable: MLX did not report a Metal device.")
    browse = BrowseService(args.data_folder, initialize_cuda=False)
    ssb = SSBProtocolService(
        args.data_folder,
        available_gpus=list,
        device_name=lambda _gpu: f"{device_name}; MLX {version('mlx')}",
        backend_kind="local_mps",
        implementation_revision=args.implementation_revision,
    )
    app = create_app(args.data_folder, service=browse, ssb_service=ssb)
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
