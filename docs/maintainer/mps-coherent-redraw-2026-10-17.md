# MPS coherent SSB redraw — 2026-10-17

Experiment: `20261017-mps-coherent-redraw` (private evidence).

## Question and contract

Can full-data, tilt-aware SSB redraw approach a 16.7 ms frame budget on Apple
M5 Max? The measured acquisition has a 512×512 scan, 192×192 detector, 4,697
selected bright-field pixels, and a 1024×1024 reconstructed phase. Inputs,
calibration, BF selection, complex64 arithmetic, precise trigonometry, output
sampling, and the `phase_of_mean` estimator are unchanged.

## Change

The previous path materialized a corrected spectrum for every BF pixel, then
reduced hundreds of chunks. The coherent path now corrects and sums each BF
group in one Metal kernel, using bounded partial arrays and ordered group
accumulation. It skips only frequencies outside the maximum support of both
antialiased probe apertures. Native phase-variance diagnostics retain their
separate per-detector inverse FFTs. Source Hermitian storage is expanded when
reading the source; the corrected spectrum still uses a complex inverse FFT.

## Measurements and parity

- Initial warm reconstruction: about 236 ms before and 63 ms after.
- A later interleaved before/after run under variable desktop load: medians
  328 ms and 114 ms (2.9×). These are measurements on a shared desktop, not
  exclusive-device throughput guarantees.
- Live notebook drag: median reconstruction 68 ms; median image paint interval
  106 ms, with additional pauses at release for the unchanged loss diagnostic.
  Controls retained a roughly 16.7 ms animation interval. Images do not reach
  60 fps.
- Three frozen full-data tilt settings: maximum phase difference below 1e-7
  radians. The inner float32 reduction order changes; this is not bit-exact
  phase reconstruction. Native losses and the two fixed-optics regional fit
  results were bit-exact in the before/after comparison.
- Native sizes 128, 256, 512, and 1024, each at output factors 1 and 2, passed
  frozen same-backend phase comparisons. Native mean-phase and loss outputs
  were bit-exact. Small BF counts remain dominated by dispatch overhead.
- Higher-order, alias/DC, anisotropic aperture, and bounded-launch checks pass.
  The broader MPS SSB run had 170 passes, two missing-simulation skips, and
  three fixture/calibration failures also reproduced with the frozen backend.
  The native Metal workflow gate passes. The frozen fit-trajectory gate could
  not run because its external fixture was absent; it is not counted as a pass.

Larger BF groups gave only small or unstable gains and changed accumulation
rounding more. Per-detector bounding tests and additional aperture branches
were not retained. The remaining compute cost is the full transfer-function
correction; the final FFT and ordered partial reduction are much smaller.
