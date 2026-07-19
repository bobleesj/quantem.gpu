#!/usr/bin/env python
"""Benchmark CUDA virtual-image reductions on resident 4D-STEM data.

This is a maintainer tool for comparing the pre-kernel widget reduction paths
against the shared ``quantem.gpu`` CUDA RawKernel backend. Reports anonymize
dataset names by default; pass explicit labels if you need provenance in a
private report.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class TimingResult:
    median_ms: float
    min_ms: float
    temp_gib: float | None
    output: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("masters", nargs="*", help="HDF5 master files to benchmark.")
    parser.add_argument(
        "--maped-dir",
        type=Path,
        help="Directory containing *_master.h5 files; used when masters are omitted.",
    )
    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--det-bin", type=int, default=1)
    parser.add_argument("--bf-radius", type=float, default=30.0)
    parser.add_argument("--scan-region", nargs=4, type=int, metavar=("R0", "R1", "C0", "C1"))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/quantem_gpu_cuda_vi_kernel"),
    )
    parser.add_argument(
        "--label-prefix",
        default="tilt",
        help="Public-safe prefix used for per-dataset labels in reports.",
    )
    return parser.parse_args()


def _cupy():
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA device is not available.")
    return cp


def _format_gib(nbytes: float | None) -> str:
    if nbytes is None:
        return "n/a"
    return f"{nbytes / (1 << 30):.3f}"


def _discover_masters(args: argparse.Namespace) -> list[Path]:
    if args.masters:
        return [Path(p).expanduser() for p in args.masters]
    if args.maped_dir is None:
        raise SystemExit("Pass master files or --maped-dir.")
    masters = sorted(args.maped_dir.expanduser().glob("*_master.h5"))
    if not masters:
        raise SystemExit(f"No *_master.h5 files found under {args.maped_dir}.")
    return masters[: args.limit]


def _sync() -> None:
    cp = _cupy()
    cp.cuda.Stream.null.synchronize()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _clear_runtime_caches() -> None:
    cp = _cupy()
    _sync()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mask(det_shape: tuple[int, int], lo: float, hi: float) -> np.ndarray:
    row = np.arange(det_shape[0], dtype=np.float32)[:, None]
    col = np.arange(det_shape[1], dtype=np.float32)[None, :]
    center = ((det_shape[0] - 1) / 2.0, (det_shape[1] - 1) / 2.0)
    dist = np.sqrt((row - center[0]) ** 2 + (col - center[1]) ** 2)
    return (dist >= float(lo)) & (dist <= float(hi))


def _masks(det_shape: tuple[int, int], bf_radius: float) -> dict[str, np.ndarray]:
    max_r = math.hypot(det_shape[0], det_shape[1])
    return {
        "BF r30": _mask(det_shape, 0.0, bf_radius),
        "ADF r30-60": _mask(det_shape, bf_radius, bf_radius * 2.0),
        "DF >r30": _mask(det_shape, bf_radius, max_r),
    }


def _legacy_cupy_selected(data, det_mask: np.ndarray) -> np.ndarray:
    cp = _cupy()
    flat = data.reshape(-1, data.shape[-2] * data.shape[-1])
    selected = cp.where(cp.asarray(det_mask.reshape(-1), dtype=cp.bool_))[0]
    out = flat[:, selected].sum(axis=1, dtype=cp.uint64).astype(cp.float32)
    return cp.asnumpy(out.reshape(data.shape[0], data.shape[1]))


def _new_cuda_backend(data, det_mask: np.ndarray, backend=None) -> np.ndarray:
    from quantem.gpu.compute.backends import compute_backend

    if backend is None:
        backend = compute_backend(data)
    return backend.masked_sum(det_mask)


def _time_call(
    fn: Callable[[], np.ndarray],
    *,
    reps: int,
    warmup: int,
    memory_kind: str,
) -> TimingResult:
    cp = _cupy()
    _clear_runtime_caches()
    torch = None
    if memory_kind == "torch":
        import torch as _torch

        torch = _torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch_base = int(torch.cuda.memory_allocated())
        else:
            torch_base = 0
    else:
        torch_base = 0
    pool = cp.get_default_memory_pool()
    cupy_base = int(pool.total_bytes())

    for _ in range(warmup):
        out = fn()
        _sync()
        del out

    times: list[float] = []
    last = None
    for _ in range(reps):
        start = time.perf_counter()
        last = fn()
        _sync()
        times.append((time.perf_counter() - start) * 1000.0)

    if memory_kind == "torch" and torch is not None and torch.cuda.is_available():
        temp_bytes = max(0, int(torch.cuda.max_memory_allocated()) - torch_base)
    elif memory_kind == "cupy":
        temp_bytes = max(0, int(pool.total_bytes()) - cupy_base)
    else:
        temp_bytes = None
    return TimingResult(
        median_ms=statistics.median(times),
        min_ms=min(times),
        temp_gib=None if temp_bytes is None else temp_bytes / (1 << 30),
        output=np.asarray(last, dtype=np.float32),
    )


def _save_preview(output_dir: Path, label: str, mask_name: str, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    safe = f"{label}_{mask_name}".replace(" ", "_").replace(">", "gt").replace("-", "_")
    paths: dict[str, str] = {}
    for name, arr in arrays.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        im = ax.imshow(arr, cmap="magma" if name != "abs_diff" else "viridis")
        ax.set_title(name)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = output_dir / f"{safe}_{name}.png"
        fig.tight_layout(pad=0.1)
        fig.savefig(path)
        plt.close(fig)
        paths[name] = path.name
    return paths


def _render_html(output_dir: Path, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> Path:
    def table(data: list[dict[str, object]]) -> str:
        if not data:
            return ""
        headers = list(data[0])
        out = ["<table>", "<thead><tr>"]
        out.extend(f"<th>{html.escape(str(h))}</th>" for h in headers)
        out.append("</tr></thead><tbody>")
        for row in data:
            out.append("<tr>")
            out.extend(f"<td>{html.escape(str(row[h]))}</td>" for h in headers)
            out.append("</tr>")
        out.append("</tbody></table>")
        return "\n".join(out)

    previews = []
    for row in rows:
        preview = row.get("preview")
        if not isinstance(preview, dict):
            continue
        label = html.escape(str(row["dataset"]))
        mask = html.escape(str(row["mask"]))
        imgs = "\n".join(
            f"<figure><img src='{html.escape(src)}'><figcaption>{html.escape(name)}</figcaption></figure>"
            for name, src in preview.items()
        )
        previews.append(f"<section><h3>{label} - {mask}</h3><div class='gallery'>{imgs}</div></section>")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QuantEM GPU CUDA Virtual Image Kernel Benchmark</title>
<style>
body {{ font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #202124; }}
h1, h2, h3 {{ line-height: 1.15; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px 8px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
thead th {{ position: sticky; top: 0; background: #f6f7f8; }}
.note {{ max-width: 920px; color: #4a4f55; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
figcaption {{ color: #4a4f55; margin-top: 4px; }}
</style>
</head>
<body>
<h1>CUDA Virtual Image Kernel Benchmark</h1>
<p class="note">Local private-data benchmark. Dataset names are anonymized; timings include the production NumPy return used by Show4DSTEM, after the 4D data is resident on the GPU.</p>
<h2>Summary</h2>
{table(summary)}
<h2>Per Dataset</h2>
{table([{k: v for k, v in row.items() if k != "preview"} for row in rows])}
<h2>Preview Images</h2>
{''.join(previews)}
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main() -> None:
    args = _parse_args()
    cp = _cupy()
    import torch
    from quantem.gpu.compute.backends import TorchBackend, compute_backend
    from quantem.gpu.io.hdf5 import load

    masters = _discover_masters(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    raw_json: list[dict[str, object]] = []
    for idx, master in enumerate(masters, start=1):
        _clear_runtime_caches()
        t0 = time.perf_counter()
        if args.scan_region is None:
            loaded = load(master, backend="cuda", det_bin=args.det_bin, verbose=False)
        else:
            loaded = load(
                master,
                backend="cuda",
                scan_region=tuple(args.scan_region),
                scan_shape=(512, 512),
                det_bin=args.det_bin,
                verbose=False,
            )
        data = loaded.data
        cp.cuda.Stream.null.synchronize()
        load_s = time.perf_counter() - t0
        label = f"{args.label_prefix} {idx}"
        shape = tuple(int(x) for x in data.shape)
        dtype = str(data.dtype)
        backend = compute_backend(data)
        legacy_torch_backend = TorchBackend(torch.from_dlpack(data))
        masks = _masks((shape[-2], shape[-1]), args.bf_radius / max(1, args.det_bin))

        for mask_name, det_mask in masks.items():
            old_torch = _time_call(
                lambda m=det_mask: legacy_torch_backend.masked_sum(m),
                reps=args.reps,
                warmup=args.warmup,
                memory_kind="torch",
            )
            old_cupy = _time_call(
                lambda m=det_mask: _legacy_cupy_selected(data, m),
                reps=args.reps,
                warmup=args.warmup,
                memory_kind="cupy",
            )
            new_cuda = _time_call(
                lambda m=det_mask: _new_cuda_backend(data, m, backend),
                reps=args.reps,
                warmup=args.warmup,
                memory_kind="cupy",
            )

            diff = np.abs(new_cuda.output - old_cupy.output)
            max_abs = float(diff.max(initial=0.0))
            mean_abs = float(diff.mean()) if diff.size else 0.0
            preview = {}
            if idx == 1:
                preview = _save_preview(
                    args.output_dir,
                    label,
                    mask_name,
                    {
                        "old_cupy": old_cupy.output,
                        "new_cuda": new_cuda.output,
                        "abs_diff": diff,
                    },
                )
            row = {
                "dataset": label,
                "shape": "x".join(str(x) for x in shape),
                "dtype": dtype,
                "mask": mask_name,
                "pixels": int(det_mask.sum()),
                "load_s": f"{load_s:.3f}",
                "old_widget_torch_ms": f"{old_torch.median_ms:.3f}",
                "old_cupy_ms": f"{old_cupy.median_ms:.3f}",
                "new_cuda_ms": f"{new_cuda.median_ms:.3f}",
                "speedup_vs_widget": f"{old_torch.median_ms / new_cuda.median_ms:.2f}x",
                "speedup_vs_cupy": f"{old_cupy.median_ms / new_cuda.median_ms:.2f}x",
                "max_abs_err": f"{max_abs:.6g}",
                "mean_abs_err": f"{mean_abs:.6g}",
                "old_widget_temp_gib": "n/a" if old_torch.temp_gib is None else f"{old_torch.temp_gib:.3f}",
                "old_cupy_temp_gib": "n/a" if old_cupy.temp_gib is None else f"{old_cupy.temp_gib:.3f}",
                "new_cuda_temp_gib": "n/a" if new_cuda.temp_gib is None else f"{new_cuda.temp_gib:.3f}",
                "preview": preview,
            }
            rows.append(row)
            raw_json.append({k: v for k, v in row.items() if k != "preview"})
            print(
                f"{label:>7} {mask_name:>10} "
                f"old-widget={old_torch.median_ms:8.3f} ms "
                f"old-cupy={old_cupy.median_ms:8.3f} ms "
                f"new={new_cuda.median_ms:8.3f} ms "
                f"max_abs={max_abs:g}"
            )

        del data, loaded, backend, legacy_torch_backend
        _clear_runtime_caches()

    summary: list[dict[str, object]] = []
    for mask_name in sorted({str(row["mask"]) for row in rows}):
        subset = [row for row in rows if row["mask"] == mask_name]
        old_widget = statistics.median(float(row["old_widget_torch_ms"]) for row in subset)
        old_cupy = statistics.median(float(row["old_cupy_ms"]) for row in subset)
        new_cuda = statistics.median(float(row["new_cuda_ms"]) for row in subset)
        summary.append(
            {
                "mask": mask_name,
                "datasets": len(subset),
                "median_old_widget_ms": f"{old_widget:.3f}",
                "median_old_cupy_ms": f"{old_cupy:.3f}",
                "median_new_cuda_ms": f"{new_cuda:.3f}",
                "speedup_vs_widget": f"{old_widget / new_cuda:.2f}x",
                "speedup_vs_cupy": f"{old_cupy / new_cuda:.2f}x",
                "max_abs_err": max(row["max_abs_err"] for row in subset),
            }
        )

    (args.output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": raw_json}, indent=2),
        encoding="utf-8",
    )
    html_path = _render_html(args.output_dir, rows, summary)
    print(f"\nHTML report: {html_path}")


if __name__ == "__main__":
    main()
