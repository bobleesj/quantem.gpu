# Original EMPAD loading

The full public 4.36 GB MoS₂–MoSe₂ acquisition now reaches a native presented
frame in 2.25–2.35 s, versus approximately 13.44 s before these changes. **The 1 s
target remains unmet.** This is not a cached 2D preview, resident-cache switch,
or cold-I/O claim. See [result.json](result.json) for provenance, raw native
measurements and limitations.

## What changed

- Consecutive RAW frames are read in bounded windows; footer removal copies
  original bytes directly into shared Metal input storage. No per-float append
  loop, integer conversion or second full 4D tensor is involved.
- A SIMD group describes and packs each 128-word row. Each output word has one
  writer; original float32 bits and the packed representation are unchanged.
- Up to 512 frames share a packing submission instead of 64, with automatic
  smaller windows under memory pressure and an actual allocation check.
- SHA-256 runs while GPU row analysis reads the same immutable staging window.
  Its approximately 1.3 s cost is still included, not cached away or hidden.
- Center-of-mass preparation uses cooperative compensated reduction rather
  than one serial thread per complete diffraction pattern. Initial products
  dropped from 8.15 s to approximately 0.15 s.

## Acceptance

All 1,073,741,824 detector samples match the original float32 words exactly.
BF/ABF/ADF, total intensity, mean DP and CoM satisfy unchanged 1e-6 relative and
absolute reference tolerances. Eleven synthetic tests include all packing
widths, source mutation, selection order, cancellation and small-budget loading.
Native tests verify actual viewer buffers, FFT, failure retention, forced
reload, folder replacement, and original EMPAD folder navigation.

Single ARINA retains approximately 120 Hz presentation; seven-source comparison
remains in its preceding measured range with unchanged memory. These samples
do not prove zero regression on every dataset or device. No ARINA kernel was
changed for this experiment.

Wide EMPAD apertures still update at 30–40 Hz, not 120 Hz. A separate
[incremental-aperture experiment](../20260908-empad-incremental-apertures/manifest.json)
investigates that bottleneck. Installed releases and dependency pins were not
updated, and no release-readiness claim is made.
