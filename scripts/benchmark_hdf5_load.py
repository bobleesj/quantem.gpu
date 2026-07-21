#!/usr/bin/env python
"""Benchmark QuantEM HDF5 load/decompress paths.

Maintainer tool for comparing CUDA and MPS load/decompression against browser
WebGPU profiles. Paths are anonymized in the output by default; pass
``--show-paths`` only for local provenance reports.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any


def _parse_region(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 4:
        raise SystemExit("--scan-region must be r0,r1,c0,c1")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("masters", nargs="+", help="HDF5 master files.")
    parser.add_argument("--backend", default="cuda", choices=("cuda", "mps", "cpu", "auto"))
    parser.add_argument("--dtype", default=None, help="Optional browse dtype, e.g. u8.")
    parser.add_argument("--det-bin", type=int, default=1)
    parser.add_argument("--scan-region", type=_parse_region)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--show-paths", action="store_true")
    parser.add_argument("--skip-mps-memory-check", action="store_true")
    return parser.parse_args()


def _sync_backend(backend: str) -> None:
    if backend == "cuda":
        try:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
    elif backend == "mps":
        try:
            import mlx.core as mx

            mx.eval(mx.array(0))
        except Exception:
            pass


def _clear_backend(backend: str) -> None:
    gc.collect()
    if backend == "cuda":
        try:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    elif backend == "mps":
        try:
            from quantem.gpu.io import clear_mps_cache

            clear_mps_cache()
        except Exception:
            pass


def _device_memory(backend: str) -> dict[str, float] | None:
    if backend == "cuda":
        try:
            import cupy as cp

            free, total = cp.cuda.runtime.memGetInfo()
            return {"free_gb": free / 1e9, "total_gb": total / 1e9}
        except Exception:
            return None
    return None


def _nbytes(data: Any) -> int | None:
    if hasattr(data, "nbytes"):
        return int(data.nbytes)
    chunks = getattr(data, "chunks", None)
    if chunks is not None:
        return int(sum(int(getattr(chunk, "nbytes", 0)) for chunk in chunks))
    return None


def _shape(data: Any) -> list[int] | None:
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    return [int(value) for value in shape]


def _load_once(master: Path, args: argparse.Namespace):
    from quantem.gpu.io import load

    kwargs: dict[str, Any] = {
        "backend": args.backend,
        "det_bin": args.det_bin,
        "verbose": False,
    }
    if args.dtype is not None:
        kwargs["dtype"] = args.dtype
    if args.skip_mps_memory_check and args.scan_region is None:
        kwargs["skip_mps_memory_check"] = True
    if args.scan_region is not None:
        kwargs["scan_region"] = args.scan_region
    return load(str(master), **kwargs)


def main() -> None:
    args = _parse_args()
    masters = [Path(path).expanduser() for path in args.masters]
    rows: list[dict[str, Any]] = []

    for index, master in enumerate(masters):
        if not master.exists():
            raise SystemExit(f"missing master #{index + 1}")
        label = str(master) if args.show_paths else f"master-{index + 1}"
        for _ in range(max(0, args.warmup)):
            result = _load_once(master, args)
            _sync_backend(args.backend)
            del result
            _clear_backend(args.backend)

        times: list[float] = []
        last = None
        for _ in range(max(1, args.reps)):
            _clear_backend(args.backend)
            mem_before = _device_memory(args.backend)
            t0 = time.perf_counter()
            result = _load_once(master, args)
            _sync_backend(args.backend)
            elapsed = time.perf_counter() - t0
            mem_after = _device_memory(args.backend)
            times.append(elapsed)
            last = {
                "label": label,
                "backend": args.backend,
                "dtype": args.dtype or "native",
                "det_bin": args.det_bin,
                "scan_region": list(args.scan_region) if args.scan_region else None,
                "shape": _shape(result.data),
                "resident_gb": None if _nbytes(result.data) is None else _nbytes(result.data) / 1e9,
                "memory_before": mem_before,
                "memory_after": mem_after,
            }
            del result

        assert last is not None
        last.update(
            {
                "reps": len(times),
                "median_s": statistics.median(times),
                "min_s": min(times),
                "max_s": max(times),
            }
        )
        rows.append(last)

    print(json.dumps(rows, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
