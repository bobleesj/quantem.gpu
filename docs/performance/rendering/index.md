# Rendering and transfer

This section records the path from a scientific result to a displayed image:
GPU/host transfers, FFT display work, contrast/histograms and browser paints.
It does not redefine reconstruction resolution or scientific precision.

## Evidence

- [C10 caching and FFT transfer](reports/2026-10-24-ssb-c10-cache-transfer.md): exact display-buffer copies, isolated transfer timing and actual 2×/4× notebook updates.
- [Rendering reports](reports/index.md): dated measurements.
- [Display/export kernel guide](../../kernels/display-export.md): display ownership and numerical contracts.
- [WebGPU runtime](../../platforms/webgpu.md): implementation and hardware acceptance.

Count new scientific image paints, not animation callbacks or requested sweep
rates. Keep full result shape, displayed shape, backend and physical adapter
explicit. Read [SSB](../ssb/index.md) for reconstruction and fitting costs;
read [methodology](../methodology.md) for timing boundaries.
