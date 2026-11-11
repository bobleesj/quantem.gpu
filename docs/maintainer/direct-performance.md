# Direct ptychography performance targets

Microscopists steer aberration controls while viewing the same full
bright-field reconstruction used for the final result. For native
full-resolution live display the targets are:

- `512x512`: at least 30 FPS (`<=33.3 ms/redraw`).
- `1024x1024`: at least 10 FPS (`<=100 ms/redraw`).

Do not claim these numbers from detector binning, scan cropping, fewer BF
pixels, or saved complex64 caches. Those are separate preview/export choices.

Current behaviour and measurements: [direct ptychography fit methods](../api/direct.md), the
[kernel registry](../performance/kernels.md), the
[verified results](../performance/results.md) and the dated
[direct ptychography reports](../performance/direct/reports/index.md). The measured log of the
CUDA, MPS, native Metal and WebGPU direct ptychography work up to 2026-10-18, including the
rejected layouts and kernels, is archived in
[direct ptychography performance history](history/direct-performance-to-2026-10-18.md); its
numbers describe retired code and are not current.
