# C10 invariant caching and FFT transfer

Benchmark record: **anonymous acquisition A**, warm prepared CUDA interaction.
This is a report-only measurement, not a replacement release baseline. Private
source identity and screenshots are retained only in the evidence archive.

This pass keeps the full 512 × 512 × 192 × 192 acquisition, all 8,807
bright-field pixels, 14 aberration coefficients (12 active higher-order
terms), sample tilt and depth. No input binning, correction approximation,
precision reduction or FFT downsampling is introduced.

## What remains in the implementation

The widget copies the already-computed full-resolution WebGPU FFT magnitude
into an owned colormap slot, instead of uploading the same array back from the
CPU. The owned slot is reusable at the same shape. The copy is submitted before
the FFT workspace can be reused. Deferred paints or different GPU devices use
the existing CPU-array upload. The CPU FFT array remains available for existing
inspection/export behavior, so this does not remove FFT readback.

A real browser WebGPU test checked every float32 bit at 1024² and 2048². The
owned display copy also remained unchanged after the next FFT reused its source
workspace. Alternating 30 transfer observations, excluding five warmups, gave:

| Full FFT array | CPU upload median | GPU copy median |
| --- | ---: | ---: |
| 1024² | 0.4 ms | 0.4 ms |
| 2048² | 2.4 ms | 0.3 ms |

These are transfer-operation measurements, not notebook FPS.

## CUDA cache experiments

The physical CUDA device was an RTX PRO 6000 Blackwell Max-Q Workstation
Edition (300 W). Reference and candidate alternated for 24 changing-C10
observations per sampling factor, excluding the first three from the warm
medians. Full output arrays were compared on every observation.

| Experiment | 2× reference → candidate, ms | 4× reference → candidate, ms | Outcome |
| --- | ---: | ---: | --- |
| Fixed higher-order coefficients in generated kernel | 29.030 → 28.448 | 30.376 → 29.794 | Exact; small gain, additional compilation/cache lifecycle; not retained |
| Cached aperture support masks | 29.026 → 32.482 | 30.367 → 31.042 | Exact, slower; rejected |
| Iterate cached mask bits | 29.166 → 32.433 | 30.516 → 30.996 | Exact, slower; rejected |
| Cache BF probe terms | 29.792 → 29.711 | 31.221 → 31.117 | Exact, negligible gain; not retained |
| Specialize geometry plus fixed optics | 29.822 → 28.401 | 31.282 → 29.815 | Not bit-exact; rejected |

The first four variants passed 48 exact full-output checks each. The last
changed phase by up to about 9.1e-6 and was rejected rather than relaxing the
contract. No production CUDA kernel is changed by this pass. Existing compact
support and prepared-data caches remain in use.

## Actual notebook image updates

The C10 sweep requested 60 updates/s. The existing CUDA notebook ran in the
in-app browser with phase and FFT visible and WebGPU rendering on Apple Metal.
Counts below are new scientific canvas paints, not animation callbacks or
requested rates. Phase is 1024² at 2× and 2048² at 4×; FFT keeps the full grid.

| Implementation | 2× phase / FFT, fps | 4× phase / FFT, fps |
| --- | ---: | ---: |
| Baseline | 31.58 / 31.58 | 21.18 / 21.18 |
| Pinned asynchronous host-transfer prototype | 32.55 / 32.56 | 22.13 / 22.13 |
| Retained GPU FFT copy, first interval | 30.33 / 30.31 | 22.90 / 22.90 |
| Retained GPU FFT copy, repeated interval | 30.20 / 30.20 | 24.44 / 24.43 |

These sequential intervals include normal variation; the retained change does
not demonstrate a 2× notebook speedup. The isolated 4× transfer is faster, but
the notebook results do not support a universal percentage speedup. The async
prototype was removed: the small observed gain did not justify shipping its
extra callback/ownership lifecycle without further cancellation and concurrency
validation. Its initial scheduling failure is preserved with the raw evidence.

60 fps remains unmet. Warm reconstruction around 29–31 ms still exceeds the
16.7 ms frame budget before notebook delivery and rendering. The experiments
show that more caching alone does not halve that work; they do not establish a
hardware performance ceiling. The next substantial improvement must reduce
correction arithmetic or its scheduling cost while preserving the exact model.

## Verification and provenance

Direct C10 keyboard movement, FFT histogram changes, 2×/4× switching and saved
joint-fit restoration were exercised. The final full 2048² phase was identical
to the frozen saved-fit phase. The notebook was left paused at 4×. The saved
calibration file was not overwritten.

Frontend tests: 396 passed, 6 skipped. TypeScript checking and the complete
widget build passed. Production Python and CUDA code were unchanged; no MPS,
release or installed-package qualification is claimed.

Experiment: `20261024-ssb-c10-specialization`. Raw records, frozen sources,
failed variants and browser frame counters are preserved privately. The CUDA
kernel source SHA-256 remained
`d38d2eb354011403790e000dd4dca9c680da2cf11e55c312de77f248292295aa`.
