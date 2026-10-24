# SSB fitting and reconstruction

Keep preparation, joint fitting, reconstruction, host transfer and displayed
image updates separate. These stages have different workloads and cannot be
compared as interchangeable FPS or fit times.

## Fitting

- [Full-field and regional benchmark](joint-fit.md): the reusable benchmark command, search budget and timing boundaries.
- [Initial fitting and fifth-order refinement](reports/2026-10-22-ssb-initialization-and-refinement.md): dated workflow evidence.
- [Four-case rotation review](reports/2026-10-22-rotation-comparison-reuse.md): preparation reuse and its measured limits.

## Reconstruction and interaction

- [C10 caching and FFT transfer](../rendering/reports/2026-10-24-ssb-c10-cache-transfer.md): full higher-order CUDA experiments and actual notebook paints. It records an unmet 60 fps target and no demonstrated 2× speedup.
- [Staging memory](reports/2026-10-22-ssb-staging-memory.md): memory reduction with numerical checks.
- [All SSB reports](reports/index.md): dated CUDA and Metal fitting, correction and memory studies.

The [SSB kernel guide](../../kernels/ssb.md) and [SSB API](../../api/ssb.md)
own the scientific and public contracts. [Rendering](../rendering/index.md)
owns display measurements. Report publication does not imply that a measured
development implementation is in the latest release.
