# MPS SSB sampling and search, 2026-10-15

The Python MPS backend now reconstructs C10/C12 SSB on finer grids with the
existing `upsampling_factor="auto"` API. It also fuses the depth correction
into the 512-point row inverse FFT and reuses probe terms within joint search.
Public call signatures, search defaults, trial order and the phase estimator
are unchanged. These are MLX/Metal kernels behind the Python API; they do not
add upsampling to the native Swift application.

```python
parameters = ssb.find_aberrations(
    tilt=True, num_trials=200, refinement="nelder-mead",
)
result = ssb.reconstruct(parameters, upsampling_factor="auto")
```

## Matched measurement

Baseline: `c36a54f221f31ee72f5e9695d52b25110d8140c8`. Physical device:
Apple M5 Max, 128 GiB unified memory. Both acquisitions retain the complete
512 × 512 × 192 × 192 uint16 input, all 8,807/8,930 selected bright-field
pixels, float32 phase and complex64 Fourier storage. No scan crop, detector
binning, BF reduction or precision change was used for these measurements.

| Operation | Before | After |
| --- | ---: | ---: |
| Prepared 512² tilt/depth phase and loss, A | 2.70–2.72 s | 0.176–0.177 s |
| Prepared 512² tilt/depth phase and loss, B | 2.71–2.72 s | 0.183–0.186 s |
| Joint search, 200 trials plus Nelder–Mead, A | 43.82–48.20 s | 32.53–33.88 s |
| Ordinary phase-variance search, same budget, A | 28.78–30.50 s | 28.76–28.85 s |
| Automatic 3× phase, 1536², A | Unsupported | 4.49 s |
| Automatic 4× phase, 2048², B | Unsupported | 8.14 s |

Search used the same seed, initial coefficients, rotation check and optimizer
settings in four fresh processes ordered baseline, candidate, candidate,
baseline. Ranges contain two runs each, not distribution percentiles. The
ordinary search remains effectively unchanged. Search covers C10, C12 and its
angle; the joint model also searches two tilt components and depth spread.
These measurements do not certify a higher-order aberration search.

Prepared search timings cover the complete `find_aberrations` call, including
its final reconstruction, rotation check and result save. Loading and BF
preparation occur before that timer. Prepared reconstruction timings exclude
loading, BF preparation, display and file writes.
Automatic output timings include a synchronized first fine-grid call and
host output, with native data already prepared. Native reconstruction ranges
exclude compilation warmup. Filesystem caches were not cleared. The full
200-trial search is not a real-time interaction.

Separate fresh load-to-saved-result runs took 51.16 s for A and 104.20 s for
B. These include source loading, preparation/compilation, the joint search,
automatic reconstruction, native diagnostic loss and saving; there was one
observation per acquisition. Prepared reconstruction speed alone does not
describe the complete acquisition workflow.

## Numerical checks

- Ordinary fit histories, fitted coefficients, final phase and loss are
  bit-exact against the frozen GPU baseline on the matched acquisition.
- Joint fit histories and fitted parameters are bit-exact. Final native phase
  differs by at most 6.23e-6 rad in that fit, and its diagnostic loss differs
  by one float32 ULP. Fixed-coefficient controls on both full acquisitions
  stay below 6.35e-6 rad.
- Fine-grid output agrees with independent GPU zero-insertion/FFT construction
  within the preselected 2e-5 rad bound: maxima 1.27e-5 and 1.05e-5 rad on the
  complete 3× and 4× outputs. The first full-size probe failed this bound;
  matching the reference aperture arithmetic fixed it without changing the
  bound.
- Hardware tests cover both phase estimators, zero/nonzero depth, factors
  1/2/3/4/8, native sizes 128/256/512/1024, rotation changes, automatic pixel
  sampling and result saving. Native-grid loss is preserved at finer output
  sampling.

The final MPS hardware run passed 30 tests; two tests requiring external
simulation fixtures were deselected. The native Metal reconstruction and
saved-result workflow gate passed, including its 200-trial round trip and
integer-count cases. The separate frozen fit-trajectory gate could not run
because its external acquisition fixture was absent. It remains unverified;
the matched full-acquisition histories above do not replace that pin.

## Implementation and limits

The fine-grid kernel reads the measured scan-frequency aliases directly from
the retained Fourier data and evaluates the probe correction at expanded
frequencies. It never constructs a repeated full BF Fourier stack. Working
chunks have a 512 MiB target. This is reconstruction on a finer grid, not
interpolation of a reconstructed image or proof of extra spatial resolution.

`mean_phase` remains the MPS default. It computes each BF inverse transform
before averaging phase. The explicit `phase_of_mean` estimator accumulates
corrected spectra before one inverse transform; it is a different estimator.
The diagnostic loss remains the native-grid search objective.

Joint search now computes geometry and phase at k once per BF pixel and trial,
instead of repeating them at every q. The reduction order and batch size stay
unchanged. A tested 12.5 GB q/k geometry cache slowed the objective and was
rejected. Preliminary search timings with unequal starting coefficients were
also excluded; only the matched four-process measurements appear above.

Evidence identifier: `20261015-full-search`. Arrays, raw logs, exact inputs and
timing harnesses remain in the private evidence archive. CUDA, another Mac,
browser rendering and application deployment need their own validation.

## Follow-up: batched refinement and fine-grid scheduling

The second comparison freezes the already optimized implementation above.
MPS now batches objective evaluations from three independent SciPy
Nelder-Mead searches. The calling thread owns all device work; worker threads
only advance their respective solvers. Bounds, starts, evaluation sequences,
stopping criteria, the 200 reference trials and 200 joint trials are retained.
Other backends keep their existing refinement schedule.

The fine-grid kernel also computes probe-only geometry and phase once per BF
pixel. Up to four chunks are submitted before waiting, retaining the original
512 MiB chunk target and accumulation order. This uses more temporary storage
than one-at-a-time submission. Larger chunk experiments were not retained.

| Prepared public operation | Before, mean | After, mean | Time reduction |
| --- | ---: | ---: | ---: |
| Joint search and save, A | 33.52 s | 25.00 s | 25.4% |
| Joint search and save, B | 99.22 s | 70.93 s | 28.5% |
| Automatic output, loss and save, A | 5.40 s | 5.04 s | 6.7% |
| Automatic output, loss and save, B | 10.31 s | 9.63 s | 6.6% |

There are two observations per version in before-after-after-before order.
Source load and preparation are excluded. Timings vary on the shared machine:
the A searches ranged from 26.97 to 40.07 s before and 23.64 to 26.36 s after.
Separate warm, phase-only tests reduced fine-grid latency by approximately
10%, with the same complete output arrays. These are measured observations,
not a guaranteed latency or browser interaction rate.

Every returned native/fine phase array, fitted parameter, native loss and
trial record matches the frozen implementation bit for bit in all eight
public-API runs. Separate instrumentation also found identical evaluation
vectors and objective values for all 1,336 and 1,847 refinement evaluations
on the two acquisitions. The relevant Python run passed 47 tests; two external
simulation cases were deselected. Native workflow and public-repository checks
passed. The legacy frozen trajectory gate still lacks its external fixture.

The third experiment reused the transformed joint fit after a rotation flip.
It reduced the additional branch stage from 31.24/60.24 s to 0.42/0.45 s,
but changed the fitted phases by up to 0.000691/0.000942 rad. Objective values
were close and branch choices agreed; the second independent search was not
reproduced. This shortcut remains experimental and MPS branch behavior is
unchanged. Evidence identifier: `20261015-three-experiments`.
