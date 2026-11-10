# SSB performance targets

Microscopists steer aberration controls while viewing the same full
bright-field reconstruction used for the final result. For native
full-resolution live display the targets are:

- `512x512`: at least 30 FPS (`<=33.3 ms/redraw`).
- `1024x1024`: at least 10 FPS (`<=100 ms/redraw`).

Do not claim these numbers from detector binning, scan cropping, fewer BF
pixels, or saved complex64 caches. Those are separate preview/export choices.

Current behaviour and measurements: [SSB fit methods](../api/ssb.md), the
[kernel registry](../performance/kernels.md), the
[verified results](../performance/results.md) and the dated
[SSB reports](../performance/ssb/reports/index.md). The measured log of the
CUDA, MPS, native Metal and WebGPU SSB work up to 2026-10-18, including the
rejected layouts and kernels, is archived in
[SSB performance history](history/ssb-performance-to-2026-10-18.md); its
numbers describe retired code and are not current.
