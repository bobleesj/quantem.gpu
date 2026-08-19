#!/usr/bin/env python3
"""Time torch CPU/MPS display FFT: fftshift(log1p(abs(fft2(x))))."""

from __future__ import annotations

import argparse
import time

import torch


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def make_source(rows: int, columns: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.float32, device=device).unsqueeze(1)
    column = torch.arange(columns, dtype=torch.float32, device=device).unsqueeze(0)
    return torch.sin(row * 0.017) + torch.cos(column * 0.013)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def display_fft(source: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(source))))


def time_op(op, iterations: int, device: torch.device) -> tuple[float, list[float]]:
    synchronize(device)
    started = time.perf_counter()
    op()
    synchronize(device)
    first_ms = (time.perf_counter() - started) * 1_000
    warm: list[float] = []
    for _ in range(max(0, iterations - 1)):
        started = time.perf_counter()
        op()
        synchronize(device)
        warm.append((time.perf_counter() - started) * 1_000)
    return first_ms, warm


def report(label: str, rows: int, columns: int, first_ms: float, warm: list[float]) -> None:
    p50 = percentile(warm, 0.50) if warm else first_ms
    p95 = percentile(warm, 0.95) if warm else first_ms
    print(
        f"{label} shape={rows}x{columns} first_ms={first_ms:.3f} "
        f"warm_p50_ms={p50:.3f} warm_p50_fps={1000.0 / max(p50, 1e-6):.1f} "
        f"warm_p95_ms={p95:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=int)
    parser.add_argument("columns", type=int, nargs="?", default=None)
    parser.add_argument("iterations", type=int, nargs="?", default=12)
    args = parser.parse_args()
    columns = args.columns or args.rows
    print(f"torch={torch.__version__} mps={torch.backends.mps.is_available()}")
    cpu = torch.device("cpu")
    source = make_source(args.rows, columns, cpu)
    report("torch_cpu_display_fft", args.rows, columns, *time_op(lambda: display_fft(source), args.iterations, cpu))
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        gpu = source.to(device)
        synchronize(device)
        report("torch_mps_display_fft", args.rows, columns, *time_op(lambda: display_fft(gpu), args.iterations, device))


if __name__ == "__main__":
    main()
