# SSB fitting and reconstruction

Keep preparation, joint fitting, reconstruction, host transfer and displayed
image updates separate. These stages have different workloads and cannot be
compared as interchangeable FPS or fit times.

## Fitting

- [Full-field and regional benchmark](joint-fit.md): the reusable benchmark command, search budget and timing boundaries.
- [Initial fitting and fifth-order refinement](reports/2026-10-22-ssb-initialization-and-refinement.md): dated workflow evidence.
- [Four-case rotation review](reports/2026-10-22-rotation-comparison-reuse.md): preparation reuse and its measured limits.

## Reconstruction and interaction

- [Native Metal redraw](reports/2026-10-24-native-metal-ssb-redraw.md): fifth-order 3× at 512² in 29–34 ms per frame on M5 Max (30 FPS met for a drag, at the line after long load); 256² above 97 FPS; app frame rate not yet measured.
- [C10 caching and FFT transfer](../rendering/reports/2026-10-24-ssb-c10-cache-transfer.md): full higher-order CUDA experiments and actual notebook paints. It records an unmet 60 fps target and no demonstrated 2× speedup.
- [Redraw symmetry and drag window](reports/2026-10-24-redraw-symmetry-and-drag-window.md): SSB 14-term redraw 56 to 30 ms by the q, -q symmetry; regularized drags in 1-3 ms.
- [Staging memory](reports/2026-10-22-ssb-staging-memory.md): memory reduction with numerical checks.
- [All SSB reports](reports/index.md): dated CUDA and Metal fitting, correction and memory studies.

The [SSB kernel guide](../../kernels/ssb.md) and [SSB API](../../api/ssb.md)
own the scientific and public contracts. [Rendering](../rendering/index.md)
owns display measurements. Report publication does not imply that a measured
development implementation is in the latest release.
