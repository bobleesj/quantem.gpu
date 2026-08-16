"""Command-line entry points for QuantEM GPU services."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantem-gpu",
        description="Accelerated 4D-STEM compute services.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested QuantEM GPU command."""
    args = _parser().parse_args(argv)
    if args.command != "serve":  # pragma: no cover - argparse owns this guard
        raise RuntimeError(f"Unsupported command {args.command!r}.")
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
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Remote viewing requires FastAPI and Uvicorn. Install "
            "quantem.gpu with the remote extra: "
            "pip install 'quantem.gpu[cuda,remote]'"
        ) from exc

    from quantem.gpu.remote import create_app

    app = create_app(args.data_folder, gpus=gpus)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
