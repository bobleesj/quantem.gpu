# Later SSB performance checkpoints

These updates follow the [frozen SSB evidence](ssb-performance.md).

## 2026-09-11 full-aperture Apple M5 floor checkpoint

The current native Apple M5 24 GB workload is the complete `512×512` scan with
all `8,937` calibrated bright-field terms and the original packed `uint32`
counts. One exact phase-variance evaluation performs approximately
`2,342,780,928` BF×scan samples before counting FFT arithmetic. The retained
Metal schedule measures a best isolated p50 of `253.4 ms` (repeated runs
`253–310 ms`), while the object redraw path is `69.6 ms` p50. A full 200-trial
plus Nelder–Mead run is `85.1 s` with `34` refinement evaluations. These are
full-aperture measurements; no crop, bin, BF reduction, reduced precision, or
fewer trials is allowed.

This gives a useful hardware/algorithm floor check: `100 ms` would require
about `23.4 billion` exact BF×scan samples per second and `50 ms` about
`46.9 billion`, while retaining two exact `512` transforms and scalar phase
and variance accumulation. The dominant current cost is the second row
transform and its phase consumer, which rereads about `18.8 GB` of intermediate
complex data per evaluation. Batch sizes `4/6/10/12/16`, BF partial reductions,
fast transcendental spelling, larger cache chunks, and a packed-real-256 IFFT
prototype were measured and rejected (the latter was both slower at `360.2 ms`
and outside the `1e-6` loss tolerance). The next credible breakthrough is a
new exact tiled transform topology that removes the global intermediate
round-trip; tuning constants alone are not sufficient.

The first bounded follow-up tested staging each BF/column cross term in
threadgroup memory. It preserved exact object and loss parity (zero loss
error), but measured `356.8 ms` p50 versus the retained `253.4 ms` best
repeat. The extra barrier and group setup outweighed the repeated global
load, so the source was removed and the arm is retained only as rejected
evidence in the experiment manifest. A no-barrier register-load variant also
preserved exact parity but measured `345.3 ms` p50, so it was rejected for the
same reason. A real-Hermitian 256-point split was then prototyped to reduce
row-FFT work; it measured about `430 ms` p50 and differed by `1.08e-3` in
loss, so it was rejected under the exactness gate and removed. A CUDA-style
radix-8/radix-8/radix-4 version fixed the numerical mismatch exactly, but
measured `376.6 ms` p50, slower than the retained 512-point consumer, and was
also removed. A final restored-kernel confirmation passed all parity gates but
was a noisy `356.9 ms` p50 (`357.7 ms` p95), illustrating why the best repeat
and repeated-distribution measurements are kept separately rather than
claiming a single-run floor.

## 2026-09-11 fftA intermediate and M5 hardware floor

Experiment `20260911-ssb-fft-intermediate` (reference-512 original file, full
`512×512` scan, `192×192` detector, all `8,937` BF terms, float32/complex64,
exact packed `uint32` counts) first reproduced the retained schedule in the
same session: loss p50 `253.3 ms` (p95 `254.3 ms`), object p50 `70.0 ms`;
eleven production-path repeats spanned `249.3–253.3 ms` p50 before the laptop
drifted into its slower state (`258.3 ms` p50, `292.2 ms` p95 after about an
hour of sustained GPU load). Compare arms only against an adjacent baseline.

Diagnostic kernels (compile-time switches, since removed) split the objective:

| Variant | Pass 1 (correct + column IFFT + fftA store) | Pass 2 (fftA load + row IFFT + atan moments) |
| --- | ---: | ---: |
| Production | `164.4 ms` | `79.4 ms` |
| no fftA store | `80.9 ms` | — |
| no G load | `140.4 ms` | `61.9 ms` (no fftA load) |
| arithmetic only | `57.7 ms` | `45.4 ms` (no load, no atan) |
| coalesced column-major store | `117.8 ms` | `179.8 ms` (strided column-major load) |

The row-major fftA store (one column per 64-thread group, 2,056-byte stride)
costs about `84 ms`; a coalesced store removes `46.5 ms` from pass 1, but moving
the transpose into pass 2's load is much worse (`292.5 ms` end to end, rejected
with bit-identical loss). Single-encoder and concurrent ping-pong schedules at
batch `2/4/8` did not move the objective, and batch `1` is dispatch-bound
(`200–235 ms`), so fftA cannot simply be kept cache-resident by shrinking
batches.

A standalone roofline on the idle M5 measured: `9.39 GB` streaming read
`77.6 ms` (`121 GB/s`); contiguous `9.4 GB` write `66.7–70.4 ms`; the
production strided column-store pattern in isolation `435.8 ms` (`6.5×`
slower); FP32 FMA `3.97 TFLOPS`; the `8,937×769` production 512-point IFFTs
with no memory traffic `65.5 ms`; `2.34e9` `atan` `14.1 ms`. Reading the
complex64 Hermitian G cache once and the transform arithmetic alone each exceed
`50 ms`, so a `50 ms` exact objective is below this machine's floor for the
current formulation. A CPU support analysis found `26.2%` of half-plane entries
exactly zero (independent of aberrations and rotation), bounding exact
aperture pruning at about `2.5–3.0 GB` of the `28.2 GB` per-objective traffic;
moving the G load behind the aperture test measured no robust gain.

Also rejected with exact parity: packing two BF rows into one complex pass-2
IFFT (Nyquist/DC columns projected to their real parts so semantics match
`Re(IFFT)` exactly) — loss error `0.0` but `258.7 ms` p50, because the halved
pass-2 FFT is hidden behind memory traffic.

Accepted: the fftA intermediate is now stored in four-row blocks,
`[bf][row/4][col 0..256][row%4]`. Pass 1 writes contiguous 32-byte segments;
pass 2 runs one 256-thread group per four rows, loads each (BF, row block) as
one contiguous `257×4` run into a padded threadgroup tile (26,752 B), prefetches
the next BF into registers, and keeps two barriers per BF. Per-value arithmetic
and the per-pixel BF accumulation order are unchanged. In interleaved A/B/A
runs it won 8 of 9 pairs (median paired p50 `−21.6 ms`); the final-code pairs
measured `222.7/222.8 ms` against `250.0/252.5 ms`, best `221.6 ms` p50. Object
relative L2 and loss error were `0.0`, resident bytes are unchanged, and the
200-trial + Nelder–Mead fit has identical loss, best parameters, and 34
refinement evaluations (`55.2 s` versus `59.5 s` same-session). Per-call loss
now spans about `221–279 ms`, versus `250–256 ms` before, so quote
distributions rather than single runs. An 8-row block, which needs the full
32 KB with the tile aliased onto FFT scratch and four barriers per BF, was
slower (`270.8/297.3 ms`) and was removed.
