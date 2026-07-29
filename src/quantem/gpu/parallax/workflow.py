"""Backend-neutral parallax workflow."""
from __future__ import annotations

from .results import ParallaxResult


def run(
    data,
    scan_shape: tuple[int, int],
    *,
    backend: str = "auto",
    center: tuple[int, int] | None = None,
    bf_radius: int | None = None,
    sampling_radius: int | None = None,
    voltage_kV: float = 300,
    scan_sampling: float | None = None,
    upsampling_factor: int = 2,
    fit_aberrations: bool = False,
    verbose: bool = False,
) -> ParallaxResult:
    """Run parallax reconstruction on the selected GPU backend.

    Parallax currently has an optimized CUDA implementation only. Requesting
    another backend fails explicitly instead of changing the scientific path.
    """

    from quantem.gpu import device

    selected = device.resolve(backend)
    if selected != "cuda":
        raise RuntimeError(
            "Parallax currently requires CUDA; "
            f"backend={selected!r} was selected."
        )
    from .compute.cuda.backend import run_cuda

    return run_cuda(
        data,
        scan_shape,
        center=center,
        bf_radius=bf_radius,
        sampling_radius=sampling_radius,
        voltage_kV=voltage_kV,
        scan_sampling=scan_sampling,
        upsampling_factor=upsampling_factor,
        fit_aberrations=fit_aberrations,
        verbose=verbose,
    )
