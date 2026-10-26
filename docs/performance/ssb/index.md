# Direct Ptycho fitting and reconstruction

Keep preparation, joint fitting, reconstruction, host transfer and displayed
image updates separate. These stages have different workloads and cannot be
compared as interchangeable FPS or fit times.

## Fitting

- [Backend parity](reports/2026-10-26-direct-ptycho-backend-parity.md): CUDA, Python MPS and native Metal fit within 1.4σ of each other on five measured scans (180 fits), repeat bitwise, and give the same images; low-order fit 0.57–0.73 s CUDA, 2.2–2.4 s native, 2.5 s MPS.
- [Full-field and regional benchmark](joint-fit.md): the reusable benchmark command, search budget and timing boundaries.
- [Initial fitting and fifth-order refinement](reports/2026-10-22-ssb-initialization-and-refinement.md): dated workflow evidence.
- [Four-case rotation review](reports/2026-10-22-rotation-comparison-reuse.md): preparation reuse and its measured limits.

## Reconstruction and interaction

- [Native Metal redraw](reports/2026-10-24-native-metal-ssb-redraw.md): fifth-order 3× at 512² in 29–34 ms per frame on M5 Max (30 FPS met for a drag, at the line after long load); 256² above 97 FPS; app frame rate not yet measured.
- [Native redraw precision](reports/2026-10-25-ssb-redraw-precision.md): `ssb-redraw` v0.3 resolves nearly cancelling sidebands at the first scan frequencies; within 1.08× CUDA's own float32 error against a float64 definition on five acquisitions (v0.2: up to 2.45×), same drag speed at 512².
- [C10 caching and FFT transfer](../rendering/reports/2026-10-24-ssb-c10-cache-transfer.md): full higher-order CUDA experiments and actual notebook paints. It records an unmet 60 fps target and no demonstrated 2× speedup.
- [Redraw symmetry and drag window](reports/2026-10-24-redraw-symmetry-and-drag-window.md): SSB 14-term redraw 56 to 30 ms by the q, -q symmetry; regularized drags in 1-3 ms.
- [Staging memory](reports/2026-10-22-ssb-staging-memory.md): memory reduction with numerical checks.
- [All Direct Ptycho reports](reports/index.md): dated CUDA and Metal fitting, correction and memory studies.

The [Direct Ptycho kernel guide](../../kernels/direct-ptycho.md) and [Direct Ptycho API](../../api/direct_ptycho.md)
own the scientific and public contracts. [Rendering](../rendering/index.md)
owns display measurements. Report publication does not imply that a measured
development implementation is in the latest release.
